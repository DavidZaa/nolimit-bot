"""The push/fold bot we submit first: its hand reader, its play, and its port from the field."""

import random

import pytest

from bot.hands import MONSTER, NOTHING, TOP_PAIR, TWO_PAIR_PLUS, WEAK_PAIR, made_hand, strong_draw
from bot.pushfold import PushFoldBot
from bot.state import CALL, CHECK, FOLD, RAISE, Action, cards_from_str
from engine import field
from engine.field import CallingStation, ChenTAG, EquityCaller, Maniac, PushFold, RandomBot
from engine.rules import NLHE, PLO, SHORT_DECK, TableConfig, game_rules
from engine.sng import play_sng
from helpers import Scripted, decisions, facing_shove, first_action


# --- the hand reader ---------------------------------------------------------------

HOLDEM = game_rules(TableConfig(variant=NLHE))
OMAHA = game_rules(TableConfig(variant=PLO))


@pytest.mark.parametrize("hole, board, rules, bucket", [
    ("KcQd", "Kh7s2d", HOLDEM, TOP_PAIR),
    ("AcAd", "Kh7s2d", HOLDEM, TOP_PAIR),
    ("7c5d", "Kh7s2d", HOLDEM, WEAK_PAIR),
    ("AcQd", "Kh7s7d", HOLDEM, NOTHING),
    ("Kc7d", "Kh7s2d", HOLDEM, TWO_PAIR_PLUS),
    ("7c7d", "Kh7s2d", HOLDEM, MONSTER),
    ("Ah5d", "2c3s4dKh", HOLDEM, MONSTER),  # the wheel
    ("AhKh", "Qh7h2h", HOLDEM, MONSTER),
    ("AcKd", "Qh7h2h9h", HOLDEM, NOTHING),  # four hearts on board and none in hand
    ("AcKd", "Qh7h2h9hJh", HOLDEM, MONSTER),  # five on board plays in hold'em
    ("Ah2c5d6s", "KhQh7h", OMAHA, NOTHING),  # Omaha needs two hearts from the hand
    ("AhKh5d6s", "QhJh2h", OMAHA, MONSTER),
    ("Kd8h8sJs", "9c2sKc4d", OMAHA, TOP_PAIR),  # kings with the board beat the pocket eights
])
def test_hand_buckets(hole, board, rules, bucket):
    assert made_hand(cards_from_str(hole), cards_from_str(board), rules) == bucket


def test_hand_reader_survives_missing_cards():
    assert made_hand((), (), HOLDEM) == NOTHING
    assert made_hand(cards_from_str("As"), cards_from_str("Kh7s"), HOLDEM) == NOTHING
    assert made_hand(cards_from_str("AsAd"), cards_from_str("Kh"), OMAHA) == NOTHING


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
def test_it_is_still_the_field_bot_except_where_it_reads_hands_correctly(variant):
    # It started as an exact port of the frozen field copy. The only intended difference
    # since: our hand reader follows the variant's rules (short deck's A-6-7-8-9 straight,
    # Omaha's best two-card pair), where the field copy's reader doesn't.
    states = []
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(15):
        bots = [Recorder(PushFoldBot(), states) for _ in range(3)] + [
            ChenTAG(), EquityCaller(), CallingStation(), Maniac(), RandomBot(t)]
        random.Random(t).shuffle(bots)
        play_sng(bots, config, seed=t)
    assert len(states) > 1000
    ours, theirs = PushFoldBot(), PushFold()
    splits = 0
    for st in states:
        if ours.act(st) != theirs.act(st):
            splits += 1
            assert variant is not NLHE and st.street > 0, st
            assert made_hand(st.hole, st.board, st.rules) != field.made_hand(st.hole, st.board), st
    assert splits < 0.05 * len(states)


@pytest.mark.parametrize("variant", [NLHE, SHORT_DECK, PLO], ids=lambda v: v.name)
def test_unwrapped_it_always_plays_legally(variant):
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(12):
        bots = [PushFoldBot() for _ in range(4)] + [ChenTAG(), EquityCaller(), Maniac(), RandomBot(t)]
        random.Random(t).shuffle(bots)
        res = play_sng(bots, config, seed=t)
        assert sum(res.stats.errors) == sum(res.stats.illegal) == 0


@pytest.mark.parametrize("hole, board, draw", [
    ("Ah5h", "Kh9h4d", True),  # nut flush draw
    ("9h8c", "Kh7h4h", True),  # one card to a flush draw
    ("Ac5c", "Kh9h4h", False),  # the flush draw is all on the board
    ("9c8d", "Th7s2d", True),  # open-ender
    ("5c4d", "3h2s9d", True),  # 2-3-4-5: the ace fills below
    ("KcQd", "JhTs2d", True),  # T-J-Q-K: a nine or an ace
    ("AcKd", "QhJs2d", False),  # J-Q-K-A only fills with a ten
    ("Kc2d", "Th9s8d", False),  # 8-9-T-? with no hole card in the run
    ("9c8d", "Th7s2dKh", True),  # still drawing on the turn
    ("9c8d", "Th7s2dKh3c", False),  # nothing left to draw to on the river
])
def test_strong_draws(hole, board, draw):
    assert strong_draw(cards_from_str(hole), cards_from_str(board), HOLDEM) is draw


@pytest.mark.parametrize("hole, board, draw", [
    ("Ah5hKdQd", "Kh9h4d", True),  # two hearts in hand, two on board
    ("Ah5cKdQd", "Kh9h4h", False),  # one heart in hand can't use a heart flush
    ("9c8dKsKh", "Th7s2d", True),  # 9-8 with T-7: a six or a jack
    ("9c2dKsKh", "Th8s7d", False),  # one card of the run in hand: Omaha needs two
])
def test_omaha_draws_use_exactly_two_hole_cards(hole, board, draw):
    assert strong_draw(cards_from_str(hole), cards_from_str(board), OMAHA) is draw
