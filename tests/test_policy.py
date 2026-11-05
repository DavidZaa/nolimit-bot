"""The tracker-driven policy adapts to each seat, and plays exactly like push/fold without features."""

import random

import pytest

from bot.policy import FEATURES, V4_FEATURES, V5_FEATURES, PolicyBot
from bot.preflop import CHEN
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


BUBBLE = [400, 150, 200, 150, 0, 0, 0, 0]  # four left, three paid; we're the BB on 200
TOP3 = (50, 30, 20, 0, 0, 0, 0, 0)
WTA = (100, 0, 0, 0, 0, 0, 0, 0)


def test_on_the_bubble_icm_folds_a_hand_chip_ev_would_call_off_with():
    with_icm = PolicyBot(FEATURES, payouts=TOP3)
    chips_only = PolicyBot(tuple(f for f in FEATURES if f != "icm"), payouts=TOP3)
    assert facing_shove(chips_only, 2, "KcQh", shover=0, stacks=BUBBLE) == Action(CALL)
    assert facing_shove(with_icm, 2, "KcQh", shover=0, stacks=BUBBLE) == Action(FOLD)
    assert facing_shove(with_icm, 2, "AcQd", shover=0, stacks=BUBBLE) == Action(CALL)


def test_winner_take_all_leaves_the_icm_feature_with_nothing_to_do():
    assert facing_shove(PolicyBot(FEATURES, payouts=WTA), 2, "KcQh", shover=0, stacks=BUBBLE) == Action(CALL)


@pytest.mark.parametrize("variant", [NLHE, PLO], ids=lambda v: v.name)
def test_with_winner_take_all_points_icm_changes_no_decision(variant):
    states = []
    config = TableConfig(variant=variant, enforce_time=False, payouts=WTA)
    for t in range(10):
        bots = [Recorder(PolicyBot(FEATURES, payouts=WTA), states) for _ in range(3)] + [
            ChenTAG(), EquityCaller(), CallingStation(), Maniac(), RandomBot(t)]
        play_sng(bots, config, seed=t)
    without = tuple(f for f in FEATURES if f != "icm")
    for st in states:
        # Fresh bots each time: trackers start from the priors in both.
        assert PolicyBot(FEATURES, payouts=WTA).act(st) == PolicyBot(without, payouts=WTA).act(st)


class Twin:
    """Plays `primary`, feeds `shadow` the exact same calls, and records any decision they split on."""

    def __init__(self, primary, shadow, splits):
        self.primary, self.shadow, self.splits = primary, shadow, splits
        self.decisions = 0

    def new_table(self, seat, num_seats):
        self.primary.new_table(seat, num_seats)
        self.shadow.new_table(seat, num_seats)

    def hand_over(self, result):
        self.primary.hand_over(result)
        self.shadow.hand_over(result)

    def act(self, state):
        action, other = self.primary.act(state), self.shadow.act(state)
        self.decisions += 1
        if action != other:
            self.splits.append((state, action, other))
        return action


@pytest.mark.parametrize("variant, reference", [
    (NLHE, V5_FEATURES), (SHORT_DECK, V5_FEATURES), (PLO, V4_FEATURES),
], ids=["nlhe plays v5", "short deck plays v5", "plo plays v4"])
def test_the_submitted_bot_plays_v5_but_v4_with_omaha_hands(variant, reference):
    splits, twins = [], []
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(10):
        pair = [Twin(PolicyBot(), PolicyBot(reference), splits) for _ in range(3)]
        twins += pair
        bots = pair + [ChenTAG(), EquityCaller(), Maniac(), FolderBot(), RandomBot(t)]
        random.Random(t).shuffle(bots)
        play_sng(bots, config, seed=t)
    assert sum(t.decisions for t in twins) > 1000
    assert not splits


def test_explicit_features_apply_to_omaha_hands_too():
    bot = PolicyBot(("steals",))
    assert bot.omaha_features == bot.features == {"steals"}
    assert PolicyBot(("steals",), omaha_features=()).omaha_features == frozenset()


def test_features_are_validated():
    with pytest.raises(ValueError):
        PolicyBot(features=("telepathy",))


class Recorder:
    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def new_table(self, seat, num_seats):
        getattr(self.inner, "new_table", lambda *a: None)(seat, num_seats)

    def hand_over(self, result):
        getattr(self.inner, "hand_over", lambda r: None)(result)

    def act(self, state):
        self.sink.append(state)
        return self.inner.act(state)


@pytest.mark.parametrize("variant", [NLHE, PLO], ids=lambda v: v.name)
def test_without_features_and_on_the_same_ranking_it_is_exactly_push_fold(variant):
    states = []
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(10):
        bots = [Recorder(PushFoldBot(), states) for _ in range(3)] + [
            ChenTAG(), EquityCaller(), CallingStation(), Maniac(), RandomBot(t)]
        play_sng(bots, config, seed=t)
    plain, featureless = PushFoldBot(), PolicyBot(features=(), ranks=CHEN)
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
