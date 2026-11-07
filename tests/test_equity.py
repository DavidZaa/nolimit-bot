"""bot/equity.py: exact where the answer is known, reproducible, no overlapping cards, converging."""

import math
import random

import pytest

from bot import equity as E
from bot.handeval import hand_value
from bot.preflop import CHEN
from bot.state import cards_from_str as C
from engine.rules import NLHE, PLO, SHORT_DECK, TableConfig, game_rules

HOLDEM = game_rules(TableConfig(variant=NLHE))
OMAHA = game_rules(TableConfig(variant=PLO))
SHORT = game_rules(TableConfig(variant=SHORT_DECK))


# -- exact answers ------------------------------------------------------------------------------

def test_the_nuts_on_the_river_always_win():
    est = E.equity(C("AsKs"), C("QsJsTs2d3c"), HOLDEM)  # royal flush
    assert est.exact and est.equity == 1.0 and est.samples == math.comb(45, 2)


def test_the_board_plays_for_everyone_on_the_river():
    # A royal flush on the board: every hand chops two ways.
    est = E.equity(C("2c3d"), C("AsKsQsJsTs"), HOLDEM)
    assert est.exact and est.equity == pytest.approx(0.5)


def test_turn_equity_is_counted_exactly():
    # 9h8h on Th7c2d4s (open-ender): enumerate it by hand and compare.
    hole, board = C("9h8h"), C("Th7c2d4s")
    deck = [c for c in HOLDEM.deck() if c not in hole + board]
    total = n = 0
    from itertools import combinations
    for opp in combinations(deck, 2):
        rest = [c for c in deck if c not in opp]
        for river in rest:
            mine, theirs = hand_value(hole, (*board, river), HOLDEM), hand_value(opp, (*board, river), HOLDEM)
            total += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
            n += 1
    est = E.equity(hole, board, HOLDEM, exact_limit=50_000)
    assert est.exact and est.samples == n and est.equity == pytest.approx(total / n)
    assert not E.equity(hole, board, HOLDEM).exact  # by default a turn samples: too many to count in time


def test_omaha_equity_uses_its_own_rules():
    # Four hearts on board and one in hand: no flush in Omaha, unlike hold'em-style rules.
    est = E.equity(C("Ah2c3d4s"), C("KhQh7h9h5c"), OMAHA, samples=400, seed=1)
    holdem_view = E.equity(C("Ah2c"), C("KhQh7h9h5c"), HOLDEM)
    assert holdem_view.exact and not est.exact
    assert holdem_view.equity > 0.9 > est.equity + 4 * est.stderr


@pytest.mark.slow
def test_omaha_river_counted_exactly_matches_sampling():
    exact = E.equity(C("Ah2c3d4s"), C("KhQh7h9h5c"), OMAHA, exact_limit=10**6)
    sampled = E.equity(C("Ah2c3d4s"), C("KhQh7h9h5c"), OMAHA, samples=2000, seed=4)
    assert exact.exact and exact.samples == 123_410
    assert abs(exact.equity - sampled.equity) < 4 * sampled.stderr + 0.005


# -- sampling ---------------------------------------------------------------------------------

def test_the_same_seed_gives_the_same_estimate_and_another_seed_does_not():
    a = E.equity(C("AsAd"), (), HOLDEM, samples=200, seed=5)
    assert a == E.equity(C("AsAd"), (), HOLDEM, samples=200, seed=5)
    assert a != E.equity(C("AsAd"), (), HOLDEM, samples=200, seed=6)


def test_sampled_cards_never_overlap(monkeypatch):
    seen = []
    real = E._showdown

    def spy(hole, others, board, rules):
        cards = list(hole) + [c for h in others for c in h] + list(board)
        seen.append(cards)
        return real(hole, others, board, rules)

    monkeypatch.setattr(E, "_showdown", spy)
    for rules, hole in ((HOLDEM, "AsAd"), (OMAHA, "AsAdKhKc"), (SHORT, "AsAd")):
        E.equity(C(hole), (), rules, opponents=5, samples=300, seed=1)
    deck_sizes = {len(rules.deck()) for rules in (HOLDEM, OMAHA, SHORT)}
    for cards in seen:
        assert len(cards) == len(set(cards)), cards
    assert seen and max(deck_sizes) == 52


@pytest.mark.parametrize("hole, other, truth", [
    ("AsAd", "KsKd", 0.8195),  # well-known preflop all-in
    ("AsKs", "2c2d", 0.4980),
])
def test_estimates_converge_within_their_error(hole, other, truth):
    known = set(C(other))
    est = E.equity(C(hole), (), HOLDEM, opponent_range=lambda h: set(h) == known, samples=3000, seed=2,
                   max_rejections=2000)
    # Within 4 standard errors of the published equity (the error shrinks like 1/sqrt(n)).
    assert abs(est.equity - truth) < 4 * est.stderr + 0.005, est
    assert est.stderr < 0.01


def test_a_range_shifts_equity_the_right_way():
    vs_any = E.equity(C("KhQh"), (), HOLDEM, samples=1500, seed=3)
    vs_tight = E.equity(C("KhQh"), (), HOLDEM, opponent_range=E.top_share(CHEN, 0.05), samples=1500, seed=3)
    assert vs_tight.equity < vs_any.equity - 0.1
    more = E.equity(C("KhQh"), (), HOLDEM, opponents=4, samples=1500, seed=3)
    assert more.equity < vs_any.equity


def test_a_deadline_stops_early_and_says_how_many_samples_it_got():
    from time import perf_counter_ns

    est = E.equity(C("AsAd"), (), HOLDEM, samples=100_000, deadline_ns=perf_counter_ns() + 2_000_000)
    assert 1 <= est.samples < 100_000 and not est.exact


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        E.equity(C("AsAs"), (), HOLDEM)
    with pytest.raises(ValueError):
        E.equity(C("2s3s"), (), SHORT)  # not in a short deck
    with pytest.raises(ValueError):
        E.equity(C("AsAd"), (), None)
