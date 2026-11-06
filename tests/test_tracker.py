"""The opponent tracker counts what each seat does, from real hands."""

import random

import pytest

from bot.state import (
    CALL, CHECK, FOLD, POST_BB, POST_SB, RAISE, Action, ActionRecord, HandResult, fallback_action,
)
from bot.tracker import INACTIVE_AFTER, PRIORS, Tracker
from engine.field import CallingStation, FolderBot, Maniac
from engine.rules import TableConfig
from engine.table import play_hand
from helpers import Scripted


class Watcher:
    """Checks or folds every decision while feeding a tracker, like our bot would."""

    def __init__(self, results=True):
        self.tracker = Tracker()
        self.results = results  # whether to pass hand results on (a kit might not)

    def new_table(self, seat, num_seats):
        self.tracker.new_table(seat, num_seats)

    def act(self, state):
        self.tracker.observe(state)
        return fallback_action(state.legal)

    def hand_over(self, result):
        if self.results:
            self.tracker.hand_over(result)


def play_hands(bots, hands, watcher_seat):
    """Play `hands` independent hands at 2,000 chips each, button on seat 0."""
    bots[watcher_seat].new_table(watcher_seat, len(bots))
    rng = random.Random(0)
    for h in range(1, hands + 1):
        play_hand(bots, [2000] * len(bots), 0, h, TableConfig(), rng)


def table_with(seats, watcher_seat=5, results=True):
    bots = [FolderBot() for _ in range(8)]
    for s, bot in seats.items():
        bots[s] = bot
    bots[watcher_seat] = Watcher(results)
    return bots, bots[watcher_seat].tracker


def test_estimates_start_at_the_priors():
    t = Tracker()
    t.new_table(0, 8)
    assert t.vpip(3) == PRIORS["vpip"] and t.shove_rate(3) == PRIORS["shove"]
    assert t.fold_to_steal(3) == PRIORS["fold_to_steal"] and t.fold_to_bet(3) == PRIORS["fold_to_bet"]
    assert t.hands(3) == 0 and not t.inactive(3)


@pytest.mark.parametrize("results", [True, False], ids=["with hand results", "from decisions only"])
def test_a_maniac_shows_up_as_a_shover(results):
    bots, tracker = table_with({3: Maniac()}, results=results)
    play_hands(bots, 30, watcher_seat=5)
    assert tracker.hands(3) >= 29  # the last hand may still be uncounted without results
    assert tracker.shove_rate(3) > 0.7 and tracker.vpip(3) > 0.7
    assert tracker.shove_rate(4) < PRIORS["shove"]  # a folder never shoves


def test_broken_bots_are_flagged_after_enough_hands():
    bots, tracker = table_with({})
    play_hands(bots, INACTIVE_AFTER - 1, watcher_seat=5)
    assert not tracker.inactive(3)
    play_hands(bots, 2, watcher_seat=5)  # new_table again: counts restart
    assert tracker.hands(3) == 2
    bots, tracker = table_with({})
    play_hands(bots, INACTIVE_AFTER + 5, watcher_seat=5)
    assert tracker.inactive(3) and tracker.vpip(3) < 0.1


def test_fold_to_steal_separates_folders_from_defenders():
    # The button opens every hand; the small blind folds, the big blind always defends.
    class Opener:
        def act(self, s):
            return Action(RAISE, 3 * s.big_blind) if s.street == 0 else fallback_action(s.legal)

    opener = Opener()
    bots, tracker = table_with({0: opener, 2: CallingStation()})
    play_hands(bots, 30, watcher_seat=5)
    assert tracker.fold_to_steal(1) > 0.9
    assert tracker.fold_to_steal(2) < 0.3
    assert tracker.vpip(0) > 0.7 and tracker.vpip(2) > 0.7


def test_fold_to_bet_finds_the_player_who_never_folds():
    class Bettor:
        def act(self, s):
            if s.street == 0:
                return Action(CALL) if s.legal.can_call else Action(CHECK)
            return Action(RAISE, max(s.legal.min_raise_to, 20)) if s.legal.can_raise else fallback_action(s.legal)

    bots, tracker = table_with({1: Bettor(), 2: CallingStation()})
    play_hands(bots, 25, watcher_seat=5)
    assert tracker.fold_to_bet(2) < 0.2  # the station calls every bet
    assert tracker.fold_to_bet(1) == PRIORS["fold_to_bet"]  # nobody ever bet into the bettor


def test_a_seat_count_mismatch_or_junk_seat_never_crashes():
    t = Tracker()
    t.new_table(0, 2)
    log = (ActionRecord(0, 0, "sb", 10), ActionRecord(7, 0, RAISE, 100), ActionRecord(1, 0, "bb", 20))
    t.hand_over(HandResult(1, (1990, 1980), {}, (1,), log))
    assert t.hands(0) == 1 and t.hands(7) == 0
    assert t.vpip(99) == PRIORS["vpip"]


def _hand(*flop, callers=(3,)):
    """Seat 0 opens, `callers` call in the big blind's place, then `flop` happens."""
    acts = [ActionRecord(1, 0, POST_SB, 10), ActionRecord(2, 0, POST_BB, 20), ActionRecord(0, 0, RAISE, 50)]
    acts += [ActionRecord(s, 0, CALL, 50) for s in callers]
    acts += [ActionRecord(s, 0, FOLD, 0) for s in (1, 2) if s not in callers]
    return acts + [ActionRecord(s, 1, kind, amount) for s, kind, amount in flop]


def test_cbets_count_only_the_opener_on_a_heads_up_flop_nobody_bet():
    t = Tracker(external=True)
    t.new_table(0, 8)
    t.record_hand(_hand((3, CHECK, 0), (0, RAISE, 60), (3, FOLD, 0)), None)  # c-bet
    t.record_hand(_hand((3, CHECK, 0), (0, CHECK, 0)), None)  # gave up
    t.record_hand(_hand((3, RAISE, 40), (0, FOLD, 0)), None)  # donked into: no chance
    t.record_hand(_hand((3, CHECK, 0), (2, CHECK, 0), (0, RAISE, 60), callers=(2, 3)), None)  # multiway
    assert (t.seats[0].cbet_chances, t.seats[0].cbets) == (2, 1)
    assert t.seats[3].cbet_chances == 0  # not the opener
    assert t.cbet(0) == (1 + PRIORS["cbet"] * 10) / (2 + 10)


def test_memory_written_before_the_cbet_counter_still_loads():
    old_row = [30, 9, 1, 12, 10, 5, 3]  # the seven counts memory carried before
    t = Tracker.from_list([old_row] * 8)
    assert t.hands(4) == 30 and t.seats[4].bet_folds == 3
    assert t.cbet(4) == PRIORS["cbet"]
    assert Tracker.from_list(t.to_list()).to_list() == t.to_list()
