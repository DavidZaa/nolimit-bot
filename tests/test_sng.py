import pytest

from bot.state import fallback_action
from engine.icm import icm_equity
from engine.rules import BlindSchedule, TableConfig
from engine.sng import assign_places, play_sng
from helpers import RandomLegal

PAYOUTS = (50.0, 30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class CheckFold:
    def act(self, state):
        return fallback_action(state.legal)


# --- ICM ---------------------------------------------------------------------

def test_icm_heads_up_closed_form():
    eq = icm_equity([3000, 1000], [70.0, 30.0])
    assert eq[0] == pytest.approx(0.75 * 70 + 0.25 * 30)
    assert eq[1] == pytest.approx(0.25 * 70 + 0.75 * 30)


def test_icm_equal_stacks_split_evenly_and_sum_to_prize_pool():
    eq = icm_equity([2000] * 8, PAYOUTS)
    assert all(e == pytest.approx(100 / 8) for e in eq)
    eq = icm_equity([500, 4000, 1500, 2500, 7500], PAYOUTS)
    assert sum(eq) == pytest.approx(sum(PAYOUTS[:5]))


def test_icm_bigger_stack_is_worth_more_but_less_than_proportionally():
    eq = icm_equity([8000, 4000, 2000, 2000], PAYOUTS)
    assert eq[0] > eq[1] > eq[2] == pytest.approx(eq[3])
    # the chip leader holds half the chips but much less than half the prize pool
    assert eq[0] < 0.5 * sum(PAYOUTS[:4])


def test_icm_matches_hand_computed_three_way():
    # P(A 1st)=.5; P(A 2nd)= P(B 1st)*A/(A+C) + P(C 1st)*A/(A+B)
    a, b, c = 2, 1, 1
    eq = icm_equity([a, b, c], [5.0, 3.0, 0.0])
    p_a2 = (b / 4) * (a / (a + c)) + (c / 4) * (a / (a + b))
    assert eq[0] == pytest.approx(0.5 * 5 + p_a2 * 3)


# --- places ------------------------------------------------------------------

def test_same_hand_busts_ranked_by_starting_stack():
    places = assign_places([3, 5, 6], {3: 400, 5: 900, 6: 100}, best_place=6, payouts=PAYOUTS)
    assert places[5] == (6, 0.0)
    assert places[3][0] == 7
    assert places[6][0] == 8


def test_exact_ties_split_the_points_of_their_places():
    places = assign_places([1, 2], {1: 300, 2: 300}, best_place=2, payouts=PAYOUTS)
    assert places[1][1] == places[2][1] == pytest.approx((30 + 20) / 2)


# --- full tables -------------------------------------------------------------

def test_random_sng_finishes_and_hands_out_every_point():
    for seed in range(15):
        bots = [RandomLegal(seed=seed * 8 + s) for s in range(8)]
        res = play_sng(bots, TableConfig(payouts=PAYOUTS), seed=seed)
        assert 1 <= res.hands_played <= 200
        assert sum(res.points) == pytest.approx(sum(PAYOUTS))
        if not res.reached_cap:
            assert sum(res.finish) == pytest.approx(36)  # places 1..8, ties averaged


def test_cap_splits_remaining_points_by_icm():
    config = TableConfig(payouts=PAYOUTS, max_hands=5)
    res = play_sng([CheckFold() for _ in range(8)], config, seed=3)
    assert res.reached_cap and res.hands_played == 5
    assert sum(res.points) == pytest.approx(sum(PAYOUTS))
    # nobody busts while checking and folding for five hands at 10/20
    assert all(p > 0 for p in res.points)


def test_sng_is_reproducible_from_its_seed():
    def run():
        bots = [RandomLegal(seed=s) for s in range(8)]
        return play_sng(bots, TableConfig(payouts=PAYOUTS), seed=42)

    a, b = run(), run()
    assert a.points == b.points and a.hands_played == b.hands_played


def test_new_table_hook_is_called_with_seat():
    class Recorder(CheckFold):
        def new_table(self, seat, num_seats):
            self.seat = (seat, num_seats)

    bots = [Recorder() for _ in range(8)]
    play_sng(bots, TableConfig(max_hands=1), seed=0)
    assert [b.seat for b in bots] == [(s, 8) for s in range(8)]


def test_blinds_follow_the_schedule():
    seen = []

    class Watcher(CheckFold):
        def act(self, state):
            seen.append((state.hand_number, state.big_blind))
            return super().act(state)

    schedule = BlindSchedule(((10, 20, 0), (20, 40, 0)), hands_per_level=3)
    play_sng([Watcher() for _ in range(8)], TableConfig(schedule=schedule, max_hands=5), seed=1)
    assert {bb for h, bb in seen if h <= 3} == {20}
    assert {bb for h, bb in seen if h > 3} == {40}
