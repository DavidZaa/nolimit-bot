"""Whatever a strategy does, what reaches the table must be legal.

A crash or illegal action in the real tournament is an auto-fold, which can
cost a whole table. These tests throw hostile bots and odd states at the
safety wrapper and the baseline, both through the engine and through
randomly generated states the engine would never produce.
"""

import math
import os
import random

import pytest

from bot.baseline import BaselineBot, preflop_percentile
from bot.safety import SafeBot, make_legal
from bot.state import (
    CALL,
    CHECK,
    FOLD,
    RAISE,
    Action,
    GameState,
    LegalActions,
    cards_from_str,
)
from engine.field import CallingStation, FolderBot, Maniac, RandomBot
from engine.rules import NLHE, PLO, SHORT_DECK, TableConfig
from engine.sng import play_sng
from engine.table import is_legal


class Hostile:
    def __init__(self, mode, seed=0):
        self.mode = mode
        self.rng = random.Random(seed)

    def act(self, state):
        m = self.mode
        if m == "crash":
            raise ValueError("boom")
        if m == "none":
            return None
        if m == "string":
            return "raise"
        if m == "huge":
            return Action(RAISE, 10**12)
        if m == "negative":
            return Action(RAISE, -5)
        if m == "float":
            return Action(RAISE, 123.7)
        if m == "nan":
            return Action(RAISE, math.nan)
        if m == "inf":
            return Action(RAISE, math.inf)
        if m == "bool":
            return Action(RAISE, True)
        if m == "unknown_kind":
            return Action("bet", 100)
        if m == "tuple":
            return ("call", 0)
        if m == "always_check":
            return Action(CHECK)
        if m == "always_fold":
            return Action(FOLD)
        if m == "always_call":
            return Action(CALL)
        return Action(self.rng.choice((FOLD, CHECK, CALL, RAISE, "x")), self.rng.randint(-50, 5000))

    def new_table(self, seat, num_seats):
        if self.mode == "crash":
            raise RuntimeError("bad hook")

    def hand_over(self, result):
        if self.mode == "crash":
            raise RuntimeError("bad hook")


MODES = [
    "crash", "none", "string", "huge", "negative", "float", "nan", "inf", "bool",
    "unknown_kind", "tuple", "always_check", "always_fold", "always_call", "random",
]


# --- make_legal on hand-picked situations -------------------------------------

FACING = LegalActions(True, False, True, 40, True, 80, 1000)
FREE = LegalActions(False, True, False, 0, True, 20, 1000)
SHORT = LegalActions(True, False, True, 40, True, 70, 70)  # only an all-in raise
NO_RAISE = LegalActions(True, False, True, 40, False, 0, 0)


@pytest.mark.parametrize("legal", [FACING, FREE, SHORT, NO_RAISE])
@pytest.mark.parametrize(
    "action",
    [None, "x", 5, Action(FOLD), Action(CHECK), Action(CALL), Action(RAISE, 0),
     Action(RAISE, 10**9), Action(RAISE, 55.5), Action(RAISE, math.nan), Action("bet", 1)],
)
def test_make_legal_always_returns_a_legal_action(action, legal):
    assert is_legal(make_legal(action, legal), legal)


def test_make_legal_keeps_intent():
    assert make_legal(Action(RAISE, 10**9), FACING) == Action(RAISE, 1000)  # shove
    assert make_legal(Action(RAISE, 1), FACING) == Action(RAISE, 80)  # min-raise
    assert make_legal(Action(RAISE, 500), NO_RAISE) == Action(CALL)  # can't raise -> call
    assert make_legal(Action(FOLD), FREE) == Action(CHECK)  # never fold for free
    assert make_legal(Action(CALL), FREE) == Action(CHECK)
    assert make_legal(Action(CHECK), FACING) == Action(FOLD)


# --- through the engine ---------------------------------------------------------

def _assert_clean(res, seats):
    st = res.stats
    for s in seats:
        assert st.errors[s] == st.illegal[s] == 0, (s, st.errors[s], st.illegal[s])


@pytest.mark.parametrize("variant", [NLHE, SHORT_DECK, PLO], ids=lambda v: v.name)
def test_wrapped_hostile_bots_never_reach_the_table_illegally(variant):
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(len(MODES)):
        modes = [MODES[(t + k) % len(MODES)] for k in range(8)]
        bots = [SafeBot(Hostile(m, seed=t * 8 + k)) for k, m in enumerate(modes)]
        _assert_clean(play_sng(bots, config, seed=t), range(8))


def _field(seed):
    return [RandomBot(seed), CallingStation(), Maniac(), FolderBot(), RandomBot(seed + 1)]


def _baseline_tables(tables, variant=NLHE, schedule=None):
    kw = {"variant": variant, "enforce_time": False}
    if schedule is not None:
        kw["schedule"] = schedule
    config = TableConfig(**kw)
    decisions = 0
    for t in range(tables):
        rng = random.Random(t)
        n_base = rng.randint(1, 8)
        bots = [BaselineBot() for _ in range(n_base)] + _field(t)[: 8 - n_base]
        while len(bots) < 8:
            bots.append(BaselineBot())
        rng.shuffle(bots)
        res = play_sng(bots, config, seed=t)
        _assert_clean(res, [s for s, b in enumerate(bots) if isinstance(b, BaselineBot)])
        decisions += sum(res.stats.decisions[s] for s, b in enumerate(bots) if isinstance(b, BaselineBot))
    return decisions


@pytest.mark.parametrize("variant", [NLHE, SHORT_DECK, PLO], ids=lambda v: v.name)
def test_unwrapped_baseline_is_always_legal(variant):
    assert _baseline_tables(60, variant) > 500


# --- randomly generated states ----------------------------------------------------

def random_state(rng):
    """A plausible-but-odd GameState: covers corners the engine rarely produces."""
    n = rng.randint(2, 9)
    seat = rng.randrange(n)
    bb = rng.choice((2, 20, 100, 2560))
    stacks = tuple(rng.choice((0, 1, rng.randint(1, 50_000))) for _ in range(n))
    bets = tuple(rng.choice((0, bb, rng.randint(0, 3000))) for _ in range(n))
    to_call = rng.choice((0, 0, rng.randint(1, 5000)))
    can_raise = rng.random() < 0.6
    lo = rng.randint(1, 6000)
    hi = rng.choice((lo, lo + rng.randint(0, 40_000)))
    legal = LegalActions(to_call > 0, to_call == 0, to_call > 0, to_call, can_raise,
                         lo if can_raise else 0, hi if can_raise else 0)
    deck = list(range(52))
    rng.shuffle(deck)
    hole = tuple(deck[: rng.choice((2, 2, 2, 4, 0, 1))])
    board = tuple(deck[10 : 10 + rng.choice((0, 3, 4, 5))])
    return GameState(
        hand_number=rng.randint(1, 250), seat=seat, num_seats=n, button=rng.randrange(n),
        sb_seat=rng.randrange(n), bb_seat=rng.randrange(n), small_blind=bb // 2, big_blind=bb,
        ante=rng.choice((0, bb // 10)), street=rng.randint(0, 3), hole=hole, board=board,
        stacks=stacks, bets=bets, pot=rng.randint(0, 100_000),
        in_hand=tuple(rng.random() < 0.7 for _ in range(n)), players_left=rng.randint(1, n),
        legal=legal, actions=(),
    )


def _fuzz_states(count, seed):
    rng = random.Random(seed)
    baseline = BaselineBot()
    wrapped = [SafeBot(Hostile(m, seed=i)) for i, m in enumerate(MODES)]
    safe_baseline = SafeBot(BaselineBot())
    for _ in range(count):
        st = random_state(rng)
        # the raw baseline may misjudge junk states, but must never crash
        baseline.act(st)
        assert is_legal(safe_baseline.act(st), st.legal)
        bot = wrapped[rng.randrange(len(wrapped))]
        assert is_legal(bot.act(st), st.legal)


def test_fuzz_random_states():
    _fuzz_states(30_000, seed=1)


@pytest.mark.slow
def test_fuzz_at_scale():
    _fuzz_states(int(os.environ.get("FUZZ_STATES", "1000000")), seed=2)
    assert _baseline_tables(int(os.environ.get("FUZZ_TABLES", "2000"))) > 100_000


# --- baseline sanity ---------------------------------------------------------------

def test_preflop_percentiles_order_sensible_hands():
    p = lambda s: preflop_percentile(cards_from_str(s))  # noqa: E731
    assert p("AsAd") < p("KsKd") < p("7c2d")
    assert p("AsKs") < p("AsKd")
    assert p("7c2d") > 0.9
    assert 0 < p("AsAd") < 0.01
