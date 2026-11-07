"""exposed_reads: opponents' exposed cards with the board."""

from dataclasses import replace

import pytest

from bot.discard import default_rules
from bot.policy import PolicyBot
from bot.state import CALL, CHECK, FOLD, RAISE, Action, cards_from_str as C
from bot.visible import strong_visible
from helpers import Scripted, decisions

RULES = replace(default_rules(), seats=8)


@pytest.mark.parametrize("card, board, strong", [
    ("Kd", "Kh7s2c", True),  # pairs the board
    ("7d", "Kh7s2c", True),  # pairs it low: still a made pair
    ("7d", "Kh7s7c", True),  # trips
    ("Ad", "Kh7s2c", False),  # an overcard alone isn't a made hand
    ("3d", "Kh7s2c", False),  # low and missing says nothing
    ("Ad", "KhKs2c9d", False),  # the board's own pair isn't theirs
    ("9h", "KhKs2c9d", True),  # two pair with the paired board
    ("Qh", "KhJh2h9h", True),  # a flush once four board cards are out
])
def test_strong_visible(card, board, strong):
    assert strong_visible(C(card)[0], C(board), RULES) is strong


class ChecksPreflop:
    """Checks its big-blind option, then lets `bot` play the flop."""

    def __init__(self, bot):
        self.bot = bot

    def act(self, state):
        return Action(CHECK) if state.street == 0 else self.bot.act(state)


def bet_spot(hole, board, exposed_at_3, flag_on, monkeypatch):
    monkeypatch.setenv("NLB_EXPOSED_READS", "on" if flag_on else "off")
    bot = PolicyBot()
    seen = decisions(ChecksPreflop(bot), 2, hole, board, others={3: Scripted(Action(CALL), Action(RAISE, 60))})
    state, _ = next((st, a) for st, a in seen if st.street == 1 and not st.legal.can_check)
    exposed = [-1] * 8
    exposed[3] = C(exposed_at_3)[0] if exposed_at_3 else -1
    return bot.act(replace(state, rules=RULES, exposed=tuple(exposed)))


def test_top_pair_folds_to_a_bet_from_a_hand_that_pairs_the_board(monkeypatch):
    assert bet_spot("KcQd", "Kh9h4d2c5c", "9s", True, monkeypatch) == Action(FOLD)
    assert bet_spot("KcQd", "Kh9h4d2c5c", "9s", False, monkeypatch) == Action(CALL)  # flag off
    assert bet_spot("KcQd", "Kh9h4d2c5c", "3s", True, monkeypatch) == Action(CALL)  # low card: no read
    assert bet_spot("Kc9d", "Kh9h4d2c5c", "9s", True, monkeypatch).kind == RAISE  # two pair still raises


def test_no_bluff_into_a_strong_visible_hand(monkeypatch):
    from bot.tracker import SeatStats

    monkeypatch.setenv("NLB_EXPOSED_READS", "on")
    bot = PolicyBot()
    bot.tracker.new_table(0, 8)
    bot.tracker.seats[2] = SeatStats(hands=20, bets_faced=10, bet_folds=9)  # folds to bets: we'd c-bet air
    # We raise preflop, the big blind calls and checks the flop to us.
    seen = decisions(bot, 0, "AsQd", "Kh7s2dJc4h", others={2: Scripted(Action(CALL), Action(CHECK))})
    state, plain = next((st, a) for st, a in seen if st.street == 1)
    assert plain.kind == RAISE  # the bluff c-bet, no reads
    exposed = [-1] * 8
    exposed[2] = C("7c")[0]
    assert bot.act(replace(state, rules=RULES, exposed=tuple(exposed))) == Action(CHECK)
    exposed[2] = C("3c")[0]
    assert bot.act(replace(state, rules=RULES, exposed=tuple(exposed))).kind == RAISE
