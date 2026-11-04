"""The tracker-driven policy adapts to each seat, and plays exactly like push/fold without features."""

import random

import pytest

from bot.policy import FEATURES, PolicyBot
from bot.pushfold import PushFoldBot
from bot.state import CALL, CHECK, FOLD, RAISE, Action, fallback_action
from engine.field import CallingStation, ChenTAG, EquityCaller, FolderBot, Maniac, RandomBot
from engine.rules import NLHE, PLO, SHORT_DECK, TableConfig
from engine.sng import play_sng
from engine.table import play_hand
from helpers import Scripted, decisions, facing_shove


def primed(seat, opponents, hands=30, features=FEATURES):
    """A PolicyBot that has watched `hands` hands from `seat` against `opponents` (seat -> bot)."""
    bot = PolicyBot(features)
    bots = [FolderBot() for _ in range(8)]
    for s, other in opponents.items():
        bots[s] = other
    bots[seat] = bot
    bot.new_table(seat, 8)
    rng = random.Random(seat)
    for h in range(1, hands + 1):
        play_hand(bots, [2000] * 8, 0, h, TableConfig(), rng)
    return bot


class Bettor:
    """Limps, then bets the minimum on every street it can."""

    def act(self, s):
        if s.street == 0:
            return Action(CALL) if s.legal.can_call else Action(CHECK)
        return Action(RAISE, s.legal.min_raise_to) if s.legal.can_raise else fallback_action(s.legal)


def test_against_a_maniac_it_calls_shoves_with_kings_where_push_fold_folds():
    bot = primed(5, {3: Maniac()})
    assert facing_shove(bot, 5, "KsKd") == Action(CALL)
    assert facing_shove(PushFoldBot(), 5, "KsKd") == Action(FOLD)


def test_with_broken_bots_in_the_blinds_it_steals_with_any_two():
    bot = primed(0, {})
    assert bot.tracker.inactive(1) and bot.tracker.inactive(2)
    assert decisions(bot, 0, "7c2d", "Kh7s2dJc4h")[0][1] == Action(RAISE, 50)
    assert decisions(PushFoldBot(), 0, "7c2d", "Kh7s2dJc4h")[0][1] == Action(FOLD)


def test_unknown_players_get_the_default_ranges():
    fresh = PolicyBot()
    assert decisions(fresh, 0, "7c2d", "Kh7s2dJc4h")[0][1] == Action(FOLD)


def test_against_a_station_it_stops_bluffing_and_bets_value_bigger():
    bot = primed(5, {0: Bettor(), 2: CallingStation()})
    assert bot.tracker.fold_to_bet(2) < 0.25
    bb_calls_then_checks = lambda: {2: Scripted(Action(CALL), Action(CHECK))}

    air = decisions(bot, 0, "AsQd", "Kh7s2dJc4h", others=bb_calls_then_checks())
    assert air[0][1] == Action(RAISE, 50) and air[1][1] == Action(CHECK)
    value = decisions(bot, 0, "KcQd", "Kh7s2dJc4h", others=bb_calls_then_checks())
    assert value[1][1] == Action(RAISE, 82)  # three quarters of a 110 pot, not half

    plain = decisions(PushFoldBot(), 0, "AsQd", "Kh7s2dJc4h", others=bb_calls_then_checks())
    assert plain[1][1] == Action(RAISE, 55)  # push/fold c-bets the station anyway


def test_features_are_validated():
    with pytest.raises(ValueError):
        PolicyBot(features=("telepathy",))


class Recorder:
    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def act(self, state):
        self.sink.append(state)
        return self.inner.act(state)


@pytest.mark.parametrize("variant", [NLHE, PLO], ids=lambda v: v.name)
def test_without_features_it_is_exactly_push_fold(variant):
    states = []
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(10):
        bots = [Recorder(PushFoldBot(), states) for _ in range(3)] + [
            ChenTAG(), EquityCaller(), CallingStation(), Maniac(), RandomBot(t)]
        play_sng(bots, config, seed=t)
    plain, featureless = PushFoldBot(), PolicyBot(features=())
    for st in states:
        assert featureless.act(st) == plain.act(st)


@pytest.mark.parametrize("variant", [NLHE, SHORT_DECK, PLO], ids=lambda v: v.name)
def test_unwrapped_it_always_plays_legally(variant):
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(12):
        bots = [PolicyBot() for _ in range(3)] + [
            ChenTAG(), EquityCaller(), Maniac(), FolderBot(), RandomBot(t)]
        random.Random(t).shuffle(bots)
        res = play_sng(bots, config, seed=t)
        assert sum(res.stats.errors) == sum(res.stats.illegal) == 0
