"""The submitted bot's single-seat ICM and bubble factor."""

import random

import pytest

from bot.icm import bubble_factor, seat_equity
from engine.icm import icm_equity

WTA = (100, 0, 0, 0, 0, 0, 0, 0)
TOP3 = (50, 30, 20, 0, 0, 0, 0, 0)
FLAT = (30, 20, 15, 12, 9, 7, 5, 2)


def test_matches_the_engines_full_icm_for_every_seat():
    rng = random.Random(7)
    for _ in range(2000):
        totals = [rng.choice((0, rng.randint(1, 16000))) for _ in range(8)]
        alive = [t for t in totals if t]
        if len(alive) < 2:
            continue
        payouts = rng.choice([TOP3, WTA, FLAT, (40, 25, 15, 10, 6, 4), (70, 30)])
        full = icm_equity(alive, tuple(payouts) + (0,) * max(0, len(alive) - len(payouts)))
        seats = [s for s, t in enumerate(totals) if t]
        for k, s in enumerate(seats):
            assert seat_equity(totals, s, payouts) == pytest.approx(full[k], abs=1e-9)


def test_busted_seats_have_no_equity_and_the_last_one_standing_has_first():
    assert seat_equity([0, 5000, 0], 0, TOP3) == 0
    assert seat_equity([0, 5000, 0], 1, TOP3) == 50


def test_winner_take_all_makes_chips_and_points_the_same():
    totals = [6000, 1000, 4000, 3000, 2000]
    for opp in range(1, 5):
        assert bubble_factor(totals, 0, opp, WTA) == pytest.approx(1.0)


def test_flatter_points_make_losing_costlier():
    totals = [2000] * 8
    assert 1.0 < bubble_factor(totals, 0, 1, TOP3) < bubble_factor(totals, 0, 1, FLAT)


def test_the_bubble_is_where_it_bites_hardest():
    # Four left with three paid: a mid stack calling off against the chip leader
    # risks the bubble; once it has burst, the same flip is cheaper.
    bubble = bubble_factor([4000, 6000, 3000, 3000], 0, 1, TOP3)
    after = bubble_factor([4000, 6000, 6000, 0], 0, 1, TOP3)
    assert bubble > after > 1.0


def test_the_covering_stack_risks_less_than_the_one_it_covers():
    totals = [9000, 2000, 2500, 2500]
    assert bubble_factor(totals, 0, 1, TOP3) < bubble_factor(totals, 1, 0, TOP3)


# -- properties over every payout shape ------------------------------------------------------

PARTIAL = (60, 40, 0, 0, 0, 0, 0, 0)  # only two places paid
SHAPES = {"wta": WTA, "top3": TOP3, "flat": FLAT, "partial": PARTIAL}


@pytest.mark.parametrize("payouts", SHAPES.values(), ids=SHAPES.keys())
def test_equities_are_symmetric_monotonic_and_sum_to_the_open_prizes(payouts):
    rng = random.Random(3)
    for alive in range(2, 9):
        equal = [2000] * alive + [0] * (8 - alive)
        eqs = [seat_equity(equal, s, payouts) for s in range(alive)]
        assert max(eqs) - min(eqs) < 1e-9  # equal stacks, equal equity
        assert sum(eqs) == pytest.approx(sum(payouts[:alive]))
    for _ in range(300):
        alive = rng.randint(2, 8)
        totals = [rng.randint(1, 9000) for _ in range(alive)] + [0] * (8 - alive)
        eqs = [seat_equity(totals, s, payouts) for s in range(alive)]
        assert sum(eqs) == pytest.approx(sum(payouts[:alive]))  # the open places' points, no more
        order = sorted(range(alive), key=lambda s: totals[s])
        assert all(eqs[a] <= eqs[b] + 1e-9 for a, b in zip(order, order[1:]))  # more chips, no less
        richer = list(totals)
        richer[0] += 500
        assert seat_equity(richer, 0, payouts) >= eqs[0] - 1e-9


def test_the_bubble_factor_stays_close_to_deciding_by_expected_points():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.icm_check import PAYOUTS, compare

    for name, payouts in PAYOUTS.items():
        r = compare(payouts, spots=1500, seed=2)
        if name == "wta":
            assert r["max"] < 1e-9 and r["flips"] == 0
        else:  # measured: mean 0.005-0.012, flips 0.4-1.2% (see bot/icm.py)
            assert r["mean_abs"] < 0.02 and r["flips"] < 0.02, (name, r)


# -- the bot's objective is the simulator's score ----------------------------------------------

def test_at_the_hand_cap_the_simulator_pays_exactly_the_bots_icm():
    from engine.field import Maniac, PushFold
    from engine.rules import TableConfig
    from engine.sng import play_sng

    checked = 0
    for payouts in (TOP3, FLAT, PARTIAL):
        for seed in range(12):
            config = TableConfig(payouts=payouts, max_hands=30, enforce_time=False)
            res = play_sng([PushFold() for _ in range(6)] + [Maniac(), Maniac()], config, seed=seed)
            if not res.reached_cap:
                continue
            final = [0.0] * 8
            res2 = play_sng([PushFold() for _ in range(6)] + [Maniac(), Maniac()], config, seed=seed,
                            on_hand=lambda h, before, out: final.__setitem__(slice(None), out.stacks))
            assert res2.points == res.points
            for s in range(8):
                if final[s] > 0:  # survivors: the bot's single-seat ICM is what the table pays
                    assert res.points[s] == pytest.approx(seat_equity(final, s, payouts), abs=1e-9)
            checked += 1
    assert checked >= 10


def test_same_hand_busts_rank_by_starting_stack_in_real_play():
    from engine.field import Maniac
    from engine.rules import TableConfig
    from engine.sng import play_sng

    seen = 0
    for seed in range(40):
        busts = []

        def watch(hand, before, out):
            gone = [s for s in range(8) if before[s] > 0 and out.stacks[s] == 0]
            if len(gone) > 1:
                busts.append({s: before[s] for s in gone})

        res = play_sng([Maniac() for _ in range(8)], TableConfig(payouts=TOP3, enforce_time=False),
                       seed=seed, on_hand=watch)
        assert sum(res.points) == pytest.approx(sum(TOP3))
        for group in busts:
            seen += 1
            ordered = sorted(group, key=lambda s: -group[s])
            for a, b in zip(ordered, ordered[1:]):  # bigger stack going in, better (or equal) place
                assert res.finish[a] <= res.finish[b]
                if group[a] == group[b]:
                    assert res.finish[a] == res.finish[b] and res.points[a] == res.points[b]
    assert seen >= 5
