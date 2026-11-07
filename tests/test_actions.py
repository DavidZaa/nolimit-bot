"""Betting semantics: raise-to totals, all-in vs max raise, short all-ins, pot-limit caps, kit translation."""

import poker_agent_sdk as sdk
import pytest

from adapter import IN_HAND, to_kit_action
from bot.actions import all_in, check_or, commit, max_raise, raise_to
from bot.pushfold import PushFoldBot
from bot.state import CALL, CHECK, FOLD, RAISE, Action, LegalActions
from engine.kit_bridge import KitSeat
from engine.rules import NLHE, PLO, TableConfig
from engine.table import is_legal
from helpers import Scripted, decisions


def spot(stacks, others, config=None, seat=2, hole="AsAd", board="Kh7s2dJc4h"):
    """The engine's own state and legal actions for `seat`'s first decision."""
    return decisions(Scripted(), seat, hole, board, stacks=stacks, others=others, config=config)[0][0]


NL = TableConfig(variant=NLHE, enforce_time=False)
PL = TableConfig(variant=PLO, enforce_time=False)
EIGHT = [2000] * 8

# Real spots from the engine, keyed by what makes them interesting. Seat 2 is the big blind.
SPOTS = {
    "unopened, deep": lambda: spot(EIGHT, {}, NL, seat=3),
    "facing an open": lambda: spot(EIGHT, {3: Scripted(Action(RAISE, 60))}, NL),
    "free check (limped to the big blind)": lambda: spot(EIGHT, {3: Scripted(Action(CALL))}, NL),
    "short all-in raise only": lambda: spot([2000, 2000, 150, 2000, 2000, 2000, 2000, 2000],
                                            {3: Scripted(Action(RAISE, 100))}, NL),
    "short call: stack below the bet": lambda: spot([2000, 2000, 80, 2000, 2000, 2000, 2000, 2000],
                                                   {3: Scripted(Action(RAISE, 300))}, NL),
    "pot-limit cap below the stack": lambda: spot(EIGHT, {3: Scripted(Action(RAISE, 60))}, PL,
                                                  hole="AsAdKhKc"),
    "pot-limit, stack under the cap": lambda: spot([2000, 2000, 150, 2000, 2000, 2000, 2000, 2000],
                                                   {3: Scripted(Action(RAISE, 60))}, PL, hole="AsAdKhKc"),
    "raise not permitted (everyone else all-in)": lambda: spot(
        [2000, 2000, 2000, 300, 0, 0, 0, 0], {3: Scripted(Action(RAISE, 300))}, NL),
}


@pytest.fixture(params=SPOTS, ids=list(SPOTS))
def state(request):
    return SPOTS[request.param]()


def test_the_spots_are_what_they_say():
    s = {name: make().legal for name, make in SPOTS.items()}
    assert s["unopened, deep"].max_raise_to == s["unopened, deep"].all_in_to == 2000
    assert s["short all-in raise only"].min_raise_to == s["short all-in raise only"].max_raise_to == 150
    short = s["short call: stack below the bet"]
    assert not short.can_raise and short.call_amount == 60 and short.all_in_to == 80
    cap = s["pot-limit cap below the stack"]
    assert cap.can_raise and cap.max_raise_to < cap.all_in_to == 2000
    under = s["pot-limit, stack under the cap"]
    assert under.max_raise_to == under.all_in_to == 150
    assert not s["raise not permitted (everyone else all-in)"].can_raise
    assert s["free check (limped to the big blind)"].can_check


def test_all_in_is_every_chip_or_nothing():
    for name, make in SPOTS.items():
        legal = make().legal
        a = all_in(legal)
        if a is None:
            # Only when every chip can't legally go in now: pot-limit below the stack,
            # or no raise and a call that leaves chips behind.
            assert not (legal.can_raise and legal.min_raise_to <= legal.all_in_to <= legal.max_raise_to), name
        elif a.kind == RAISE:
            assert a.amount == legal.all_in_to, name
        else:
            assert a == Action(CALL) and legal.call_amount == legal.all_in_to - legal.bet, name
        if a is not None:
            assert is_legal(a, legal), name


def test_max_raise_is_not_all_in_under_pot_limit():
    legal = SPOTS["pot-limit cap below the stack"]().legal
    assert all_in(legal) is None
    assert max_raise(legal) == Action(RAISE, legal.max_raise_to) != Action(RAISE, legal.all_in_to)
    assert commit(legal) == max_raise(legal)  # the pot, named as such, with chips behind


def test_commit_shoves_where_it_can_and_never_breaks_the_rules(state):
    legal = state.legal
    action = commit(legal)
    assert is_legal(action, legal)
    if all_in(legal) is not None:
        assert action == all_in(legal)
    elif legal.can_raise:
        assert action == Action(RAISE, legal.max_raise_to)
    else:
        assert action.kind in (CALL, CHECK)


def test_raise_to_clamps_into_range_and_falls_back_to_chips_in(state):
    legal = state.legal
    for want in (0, legal.min_raise_to - 1, legal.min_raise_to, legal.max_raise_to, legal.max_raise_to + 1, 10**9):
        a = raise_to(legal, want)
        assert is_legal(a, legal)
        if legal.can_raise:
            assert a.amount == max(legal.min_raise_to, min(want, legal.max_raise_to))
        else:
            assert a.kind == (CALL if legal.can_call else CHECK)
    assert is_legal(check_or(legal, call=True), legal) and is_legal(check_or(legal, call=False), legal)


# -- the kit translation -------------------------------------------------------------------------

def kit_view(gs, config):
    """What the kit would send for this engine state (through the bridge)."""
    bridge = KitSeat(object, config)
    bridge._start_hand(gs.hand_number, gs.board)
    bridge._catch_up(gs.actions, gs.board, gs.street)
    return bridge._state(gs)


def back(kit_action) -> Action:
    """The kit's action as the engine sees it (what the tournament would play)."""
    kind = kit_action.kind
    return Action(RAISE, kit_action.amount) if kind == "raise" else Action(kind)


INTENTS = lambda legal: [  # noqa: E731
    Action(FOLD), Action(CHECK), Action(CALL), Action(RAISE, 0), Action(RAISE, legal.min_raise_to - 1),
    Action(RAISE, legal.min_raise_to), Action(RAISE, legal.max_raise_to), Action(RAISE, legal.max_raise_to + 1),
    Action(RAISE, legal.all_in_to), Action(RAISE, 10**12), Action("bogus"),
]


@pytest.mark.parametrize("name", SPOTS)
def test_every_translated_action_is_legal_and_keeps_its_intent(name):
    gs = SPOTS[name]()
    config = PL if "pot-limit" in name else NL
    kit_state = kit_view(gs, config)
    legal = gs.legal
    for intent in INTENTS(legal):
        out = to_kit_action(intent, kit_state)
        played = back(out)
        assert is_legal(played, legal), (name, intent, out)
        if intent.kind == RAISE and legal.can_raise:
            assert played == Action(RAISE, max(legal.min_raise_to, min(intent.amount, legal.max_raise_to)))
        elif intent.kind == RAISE:
            assert played.kind == (CALL if legal.can_call else CHECK)  # wanted chips in: never folds
        elif intent.kind == CALL:
            assert played.kind == (CALL if legal.can_call else CHECK)
        elif intent.kind == CHECK:
            assert played.kind == (CHECK if legal.can_check else FOLD)  # never calls
        elif intent.kind == FOLD:
            assert played.kind == (FOLD if legal.can_fold else CHECK)


def test_a_raise_that_is_no_longer_offered_calls_rather_than_folds():
    state = kit_view(SPOTS["raise not permitted (everyone else all-in)"](), NL)
    assert state.can("raise") is None and state.can("call") is not None
    assert to_kit_action(Action(RAISE, 5000), state) == sdk.Action.call()


def test_the_adapter_reads_all_in_from_the_seat_not_the_max_raise():
    import adapter
    from bot.events import TableLog

    gs = SPOTS["pot-limit cap below the stack"]()
    kit_state = kit_view(gs, PL)
    log = TableLog()
    index = adapter.seat_index(kit_state)
    log.feed(kit_state.events, index, adapter.Tracker(external=True), 2000, len(index))
    ours = adapter.to_game_state(kit_state, log, index, gs.rules)
    assert ours.legal == gs.legal  # every field, all_in_to and bet included
    assert ours.legal.all_in_to == 2000 > ours.legal.max_raise_to


def test_seat_statuses_map_to_in_hand():
    assert IN_HAND == ("active", "all_in")
    gs = SPOTS["short call: stack below the bet"]()
    statuses = {s.status for s in kit_view(gs, NL).seats}
    assert statuses <= {"active", "folded", "all_in", "busted"}


# -- the strategy never treats a max raise as all-in ----------------------------------------------

def test_push_fold_shoves_all_in_in_no_limit_and_raises_the_pot_in_pot_limit():
    nl = decisions(PushFoldBot(), 7, "AsAd", "Kh7s2dJc4h", stacks=[300] * 8)[0]
    assert nl[1] == Action(RAISE, nl[0].legal.all_in_to)  # 15 BB, first in: a real shove
    pl = decisions(PushFoldBot(), 7, "AsAdKhKc", "Kh7s2dJc4h", stacks=[300] * 8, config=PL)[0]
    legal = pl[0].legal
    assert legal.max_raise_to < legal.all_in_to
    assert pl[1] == Action(RAISE, legal.max_raise_to)  # the pot, chips behind: not an all-in


@pytest.mark.skip(reason="PENDING: official SDK ships at kickoff. Then compare against the stand-in: "
                         "Action constructors, LegalAction fields, raise_to amount meaning, short all-in "
                         "offers (min/max), event names/fields and seat statuses; and run poker-agent check/play")
def test_against_the_official_sdk():
    pass
