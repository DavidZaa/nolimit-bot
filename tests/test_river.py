"""river_equity: exact equity over opponents' hidden cards in Bay Area, under budget."""

from dataclasses import replace
from time import perf_counter_ns

import pytest

from bot import river
from bot.discard import default_rules
from bot.policy import PolicyBot
from bot.state import CALL, CHECK, FOLD, RAISE, Action, cards_from_str as C
from helpers import Scripted, decisions

RULES = replace(default_rules(), seats=8)


def eq(hole, board, exposed, dead=""):
    return river.river_equity(C(hole), C(board), [C(e)[0] for e in exposed], C(dead), RULES, 10**7)


def test_the_nuts_are_certain_and_a_dead_hand_hopeless():
    assert eq("AsKs", "QsJsTs2d3c", ["9h"]) == 1.0
    assert eq("2h3h", "AsKdQc9h7s", ["Jc"]) == 0.0  # their jack outkicks our 7-high every time


def test_counting_by_hand_matches():
    hole, board, exposed = C("KcQd"), C("Kh9s4c2d7h"), C("9d")[0]
    known = set(hole) | set(board) | {exposed}
    from bot.handeval import hand_value
    mine = hand_value(hole, board, RULES)
    wins = ties = n = 0
    for h in range(52):
        if h in known:
            continue
        theirs = hand_value((exposed, h), board, RULES)
        wins += mine > theirs
        ties += mine == theirs
        n += 1
    assert n == 44
    assert eq("KcQd", "Kh9s4c2d7h", ["9d"]) == pytest.approx((wins + ties / 2) / n)


def test_dead_cards_change_the_odds_and_unknown_exposures_give_nothing():
    # Our discard and a folded player's exposed card can't be the hidden card.
    board, exposed = "Kh9s4c2d7h", ["9d"]
    plain = eq("KcQd", board, exposed)
    without_nines = eq("KcQd", board, exposed, dead="9c9h")  # the other nines are dead: fewer trips for them
    assert without_nines > plain
    assert river.river_equity(C("KcQd"), C(board), [-1], (), RULES, 10**7) is None
    assert river.river_equity(C("KcQd"), C(board), [C("9d")[0]], (), RULES, 1) is None  # no budget


def test_multiway_takes_the_worst_case():
    # Each opponent's hidden card can't be the other's exposed one.
    assert eq("KcQd", "Kh9s4c2d7h", ["9d", "3h"]) == min(eq("KcQd", "Kh9s4c2d7h", ["9d"], dead="3h"),
                                                        eq("KcQd", "Kh9s4c2d7h", ["3h"], dead="9d"))


def test_fast_enough_for_three_opponents():
    hole, board = C("KcQd"), C("Kh9s4c2d7h")
    opp = [C(e)[0] for e in ("9d", "3h", "Ad")]
    times = []
    for _ in range(300):
        t = perf_counter_ns()
        river.river_equity(hole, board, opp, (), RULES, 10**7)
        times.append(perf_counter_ns() - t)
    times.sort()
    budget = 300_000 if river._evaluate is not None else 1_500_000
    assert times[len(times) // 2] < budget // 3 and times[int(0.99 * len(times))] < budget


class ChecksBefore:
    """Limps in, checks every street, and hands the river to `bot`."""

    def __init__(self, bot):
        self.bot = bot

    def act(self, state):
        if state.street == 3:
            return self.bot.act(state)
        return Action(CHECK) if state.legal.can_check else Action(CALL)


def river_spot(hole, board, exposed_at_2, bet=None, monkeypatch=None, flag_on=True):
    """We're on the button; the big blind (seat 2) checks to the river, then bets `bet` or checks."""
    monkeypatch.setenv("NLB_RIVER_EQUITY", "on" if flag_on else "off")
    bot = PolicyBot()
    script = [Action(CHECK)] * 3 + ([Action(RAISE, bet)] if bet else [Action(CHECK)])
    seen = decisions(ChecksBefore(bot), 0, hole, board, others={2: Scripted(*script)})
    state, _ = next((st, a) for st, a in seen if st.street == 3)
    assert state.legal.can_check == (bet is None)
    exposed = [-1] * 8
    exposed[2] = C(exposed_at_2)[0]
    return bot.act(replace(state, rules=RULES, exposed=tuple(exposed)))


def test_river_decisions_follow_the_exact_equity(monkeypatch):
    # Top two pair against a visible nine on K-9-4-2-7: a big favourite, bets when checked to.
    assert river_spot("Kc9d", "Kh9s4c2d7h", "9c", monkeypatch=monkeypatch).kind == RAISE
    # King-queen against a visible nine facing a bet: 70% ahead, short of a raise: calls.
    assert river_spot("KcQd", "Kh9s4c2d7h", "9c", bet=40, monkeypatch=monkeypatch) == Action(CALL)
    # ...and against a visible three nothing hidden beats it: raises.
    assert river_spot("KcQd", "Kh9s4c2d7h", "3c", bet=40, monkeypatch=monkeypatch).kind == RAISE
    # Queen-high against a visible king: beaten by nearly everything: folds to a bet, checks behind.
    assert river_spot("QcJd", "Kh9s4c2d7h", "Kd", bet=40, monkeypatch=monkeypatch) == Action(FOLD)
    assert river_spot("QcJd", "Kh9s4c2d7h", "Kd", monkeypatch=monkeypatch) == Action(CHECK)
    # Flag off: the bucket logic, which only calls with top pair, no matter what's visible.
    assert river_spot("KcQd", "Kh9s4c2d7h", "3c", bet=40, monkeypatch=monkeypatch, flag_on=False) == Action(CALL)
