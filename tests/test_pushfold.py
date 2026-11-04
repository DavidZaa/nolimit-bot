"""The push/fold bot we submit first: its hand reader, its play, and its port from the field."""

import random

import pytest

from bot.hands import MONSTER, NOTHING, TOP_PAIR, TWO_PAIR_PLUS, WEAK_PAIR, made_hand
from bot.pushfold import PushFoldBot
from bot.state import CALL, CHECK, FOLD, RAISE, Action, cards_from_str
from engine import field
from engine.field import CallingStation, ChenTAG, EquityCaller, Maniac, PushFold, RandomBot
from engine.rules import NLHE, PLO, SHORT_DECK, TableConfig
from engine.sng import play_sng
from helpers import Scripted, decisions, facing_shove, first_action


# --- the hand reader ---------------------------------------------------------------

@pytest.mark.parametrize("hole, board, bucket", [
    ("KcQd", "Kh7s2d", TOP_PAIR),
    ("AcAd", "Kh7s2d", TOP_PAIR),
    ("7c5d", "Kh7s2d", WEAK_PAIR),
    ("AcQd", "Kh7s7d", NOTHING),
    ("Kc7d", "Kh7s2d", TWO_PAIR_PLUS),
    ("7c7d", "Kh7s2d", MONSTER),
    ("Ah5d", "2c3s4dKh", MONSTER),  # the wheel
    ("AhKh", "Qh7h2h", MONSTER),
    ("AcKd", "Qh7h2h9h", NOTHING),  # four hearts on board and none in hand
    ("AcKd", "Qh7h2h9hJh", MONSTER),  # five on board plays in hold'em
    ("Ah2c5d6s", "KhQh7h", NOTHING),  # Omaha needs two hearts from the hand
    ("AhKh5d6s", "QhJh2h", MONSTER),
])
def test_hand_buckets(hole, board, bucket):
    assert made_hand(cards_from_str(hole), cards_from_str(board)) == bucket


@pytest.mark.parametrize("hole_size", [2, 4])
def test_hand_reader_agrees_with_the_engine_evaluator(hole_size):
    rng = random.Random(hole_size)
    for _ in range(15_000):
        deck = rng.sample(range(52), hole_size + 5)
        hole, board = deck[:hole_size], deck[hole_size : hole_size + rng.choice((3, 4, 5))]
        assert made_hand(hole, board) == field.made_hand(hole, board), (hole, board)


def test_hand_reader_survives_missing_cards():
    assert made_hand((), ()) == NOTHING
    assert made_hand(cards_from_str("As"), cards_from_str("Kh7s")) == NOTHING


# --- how it plays ------------------------------------------------------------------

def test_at_ten_big_blinds_it_shoves_wide_late_and_tight_early():
    ten_bb = [200] * 8
    assert first_action(PushFoldBot(), 1, "Kc7d", "Kh7s2dJc4h", stacks=ten_bb) == Action(RAISE, 200)
    assert first_action(PushFoldBot(), 3, "Kc7d", "Kh7s2dJc4h", stacks=ten_bb) == Action(FOLD)


def test_deep_it_opens_small_instead_of_shoving():
    assert first_action(PushFoldBot(), 3, "AsAd", "Kh7s2dJc4h") == Action(RAISE, 50)


def test_it_calls_a_short_shove_with_the_top_of_the_shoving_range():
    ten_bb = [200] * 8
    assert facing_shove(PushFoldBot(), 2, "Ac5d", shover=1, stacks=ten_bb) == Action(CALL)
    assert facing_shove(PushFoldBot(), 2, "8c3d", shover=1, stacks=ten_bb) == Action(FOLD)


def test_it_calls_a_deep_blind_versus_blind_shove_only_with_premiums():
    assert facing_shove(PushFoldBot(), 2, "AsKd", shover=1) == Action(CALL)
    assert facing_shove(PushFoldBot(), 2, "AcQd", shover=1) == Action(FOLD)


def test_it_reads_a_deep_early_shove_as_a_nash_range_and_calls_only_aces():
    # Right against a solid shover, far too tight against a maniac. Fixing that
    # needs the opponent tracker's shove-rate stat.
    assert facing_shove(PushFoldBot(), 5, "AsAd") == Action(CALL)
    assert facing_shove(PushFoldBot(), 5, "KsKd") == Action(FOLD)


def test_it_continuation_bets_heads_up_as_the_raiser_then_respects_a_raise():
    big_blind = {2: Scripted(Action(CALL), Action(CHECK), Action(RAISE, 300))}
    seen = decisions(PushFoldBot(), 0, "AsQd", "Kh7s2dJc4h", others=big_blind)
    assert [a for _, a in seen] == [Action(RAISE, 50), Action(RAISE, 55), Action(FOLD)]


def test_it_value_bets_top_pair_when_checked_to():
    small_blind = {1: Scripted(Action(CALL), Action(CHECK))}
    seen = decisions(PushFoldBot(), 2, "Kc5d", "Kh7s2dJc4h", others=small_blind)
    assert seen[0][1] == Action(CHECK)
    assert seen[1][1] == Action(RAISE, 20)  # half of a 40-chip pot


# --- the port ----------------------------------------------------------------------

class Recorder:
    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def act(self, state):
        self.sink.append(state)
        return self.inner.act(state)


@pytest.mark.parametrize("variant", [NLHE, SHORT_DECK, PLO], ids=lambda v: v.name)
def test_it_starts_as_an_exact_port_of_the_field_bot(variant):
    # Remove once ours is deliberately tuned away from the frozen field copy.
    states = []
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(15):
        bots = [Recorder(PushFoldBot(), states) for _ in range(3)] + [
            ChenTAG(), EquityCaller(), CallingStation(), Maniac(), RandomBot(t)]
        random.Random(t).shuffle(bots)
        play_sng(bots, config, seed=t)
    assert len(states) > 1000
    ours, theirs = PushFoldBot(), PushFold()
    for st in states:
        assert ours.act(st) == theirs.act(st), st


@pytest.mark.parametrize("variant", [NLHE, SHORT_DECK, PLO], ids=lambda v: v.name)
def test_unwrapped_it_always_plays_legally(variant):
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(12):
        bots = [PushFoldBot() for _ in range(4)] + [ChenTAG(), EquityCaller(), Maniac(), RandomBot(t)]
        random.Random(t).shuffle(bots)
        res = play_sng(bots, config, seed=t)
        assert sum(res.stats.errors) == sum(res.stats.illegal) == 0
