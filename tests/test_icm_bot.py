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
