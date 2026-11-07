"""The tournament entry point: the only file that knows the starter kit's protocol.

Shipped as agent.py (scripts/pack.sh), next to bot/. The kit calls
Agent().act(state, memory) for every decision and wants (Action, bytes) back.
A pool of forked copies serves the tables and any copy may answer any table,
so nothing here keeps state on self or in globals between calls: the
opponent tracker and the current hand's action log travel in `memory`,
rebuilt at the start of each call from compact JSON and written back at the
end. Bad or foreign memory starts the table fresh rather than failing.

Kit-day checklist (names below follow the organizers' published description;
each is a PLACEHOLDER(kit) until checked against the real SDK):
  1. Diff kitsim/poker_agent_sdk against the real SDK: Action constructors,
     state.can(), legal action kinds and min/max, event kinds and fields,
     seats/status names, state.blinds, state.config.payouts.
  2. Rules: register the announced variant in bot/rules.py VARIANTS under the
     exact ID state.config.variant reports, and generate its starting-hand
     table (gen/hand_ranks.py docstring). Payouts, stack, blinds, level length,
     hand cap and seats need no step: every table's config carries its own.
  3. pytest -q; then scripts/pack.sh --variant <ID>, which runs the kit's own
     `poker-agent check`, and a `poker-agent play` against its examples.

Rules are never guessed. If the config names a variant with no registered
rules, or a tournament setting is missing or malformed, the decision checks
or folds (a legal action, not a failed decision) and the process prints one
RULES ERROR line to stderr, which `poker-agent play` shows locally.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # bot/ sits next to agent.py

import poker_agent_sdk as sdk  # noqa: E402

from bot.events import TableLog  # noqa: E402
from bot.handeval import warm as warm_evaluator  # noqa: E402
from bot.policy import PolicyBot  # noqa: E402
from bot.rules import VARIANTS, RulesError, rules_from_config  # noqa: E402
from bot.runtime import tune_gc  # noqa: E402
from bot.safety import SafeBot  # noqa: E402
from bot.state import (  # noqa: E402
    CALL, CHECK, FOLD, POST_BB, POST_SB, RAISE, Action, GameRules, GameState, LegalActions, card_from_str,
)
from bot.tracker import Tracker  # noqa: E402

MEMORY_VERSION = 1
MEMORY_BUDGET = 16_000  # bytes; the kit allows 64 KB but anything past ~16 KB costs decision time
IN_HAND = ("active", "all_in")


# -- memory ------------------------------------------------------------------

def load_memory(memory) -> tuple[Tracker, TableLog]:
    """The table's tracker and log from last call's bytes; anything unreadable starts fresh."""
    try:
        doc = json.loads(memory)
        if doc[0] != MEMORY_VERSION:
            raise ValueError("other version")
        return Tracker.from_list(doc[1]), TableLog.from_list(doc[2])
    except Exception:
        return Tracker(external=True), TableLog()


def dump_memory(tracker: Tracker, log: TableLog) -> bytes:
    out = json.dumps([MEMORY_VERSION, tracker.to_list(), log.to_list()], separators=(",", ":")).encode()
    if len(out) > MEMORY_BUDGET:  # never seen in testing; drop the in-hand log, keep the counts
        log.records, log.counted = [], True
        out = json.dumps([MEMORY_VERSION, tracker.to_list(), log.to_list()], separators=(",", ":")).encode()
    return out


# -- state ---------------------------------------------------------------------

def seat_index(state) -> dict:
    """The kit's seat ids (anonymised per table) -> 0..n-1, in seat order."""
    return {s.seat: i for i, s in enumerate(sorted(state.seats, key=lambda s: s.seat))}


def to_card(kit_card) -> int:
    return card_from_str(str(kit_card))


def _blinds(state, log: TableLog) -> tuple[int, int, int]:
    """(small, big, ante) for this hand, from state.blinds, else from what was posted."""
    raw = getattr(state, "blinds", None)
    try:
        if hasattr(raw, "big"):
            return int(raw.small), int(raw.big), int(getattr(raw, "ante", 0) or 0)
        nums = [int(x) for x in raw]
        return nums[0], nums[1], nums[2] if len(nums) > 2 else 0
    except Exception:
        posts = {k: a for _, st, k, a in log.records if st == 0 and k in (POST_SB, POST_BB)}
        big = posts.get(POST_BB, 0)
        return posts.get(POST_SB, big // 2), big, 0


def legal_actions(state, stack: int) -> LegalActions:
    fold, check, call, raise_ = (state.can(k) for k in ("fold", "check", "call", "raise"))
    to_call = min(int(getattr(state, "to_call", 0) or 0), stack)
    if raise_ is not None:
        lo, hi = int(raise_.min_amount), int(raise_.max_amount)
    else:
        lo = hi = 0
    return LegalActions(fold is not None, check is not None, call is not None, to_call,
                        raise_ is not None and hi >= lo > 0, lo if raise_ else 0, hi if raise_ else 0)


def to_game_state(state, log: TableLog, index: dict, rules: GameRules) -> GameState:
    n = len(index)
    seats = sorted(state.seats, key=lambda s: s.seat)
    stacks = tuple(int(s.stack) for s in seats)
    bets = tuple(int(s.bet) for s in seats)
    in_hand = tuple(s.status in IN_HAND for s in seats)
    me = index[state.seat]
    button = index.get(state.button, 0)
    sb_seat = bb_seat = None
    for seat, street, kind, _ in log.records:
        if street == 0 and kind == POST_SB and sb_seat is None:
            sb_seat = seat
        elif street == 0 and kind == POST_BB and bb_seat is None:
            bb_seat = seat
    if bb_seat is None:  # nothing posted that we saw: the seats after the button
        alive = [i for i in range(1, n + 1) if stacks[(button + i) % n] or bets[(button + i) % n]]
        sb_seat = (button + alive[0]) % n if alive else button
        bb_seat = (button + alive[1]) % n if len(alive) > 1 else sb_seat
    sb, bb, ante = _blinds(state, log)
    return GameState(
        hand_number=int(state.hand_number), seat=me, num_seats=n, button=button,
        sb_seat=sb_seat if sb_seat is not None else button, bb_seat=bb_seat,
        small_blind=sb, big_blind=bb, ante=ante, street=log.street,
        hole=tuple(to_card(c) for c in state.cards), board=tuple(to_card(c) for c in state.board),
        stacks=stacks, bets=bets, pot=int(state.pot), in_hand=in_hand,
        players_left=int(getattr(state, "players_remaining", 0) or sum(1 for s in stacks if s)),
        legal=legal_actions(state, stacks[me]), actions=log.actions(),
        rules=rules,
    )


# -- actions ---------------------------------------------------------------------

def automatic(state):
    """What the kit plays itself when neither check nor fold is offered: the first legal kind."""
    first = state.legal_actions[0]
    return sdk.Action(first.kind, first.min_amount)


def to_kit_action(action: Action, state):
    can = state.can
    kind = action.kind
    if kind == RAISE and can("raise") is not None:
        r = can("raise")
        return sdk.Action.raise_to(max(int(r.min_amount), min(int(action.amount), int(r.max_amount))))
    if kind in (RAISE, CALL) and can("call") is not None:
        return sdk.Action.call()
    if kind != FOLD and can("check") is not None:
        return sdk.Action.check()
    if can("fold") is not None:
        return sdk.Action.fold()
    if can("check") is not None:
        return sdk.Action.check()
    return automatic(state)


def _fallback(state):
    if state.can("check") is not None:
        return sdk.Action.check()
    if state.can("fold") is not None:
        return sdk.Action.fold()
    return automatic(state)


# -- the agent --------------------------------------------------------------------

def decide(state, memory: bytes):
    """One decision: rebuild the table from memory, catch up on events, act, write memory."""
    tracker, log = load_memory(memory)
    try:
        index = seat_index(state)
        if not tracker.seats:
            tracker.seats = Tracker.from_list([[0] * 7 for _ in index]).seats
        cfg = state.config
        log.feed(state.events, index, tracker, int(getattr(cfg, "starting_stack", 0) or 0), len(index))
        new_memory = dump_memory(tracker, log)
    except Exception:
        return _fallback(state), memory
    try:
        if getattr(state, "phase", "betting") != "betting" or not any(
                state.can(k) for k in ("fold", "check", "call", "raise")):
            return automatic(state), new_memory
        rules = rules_from_config(cfg)
        policy = PolicyBot()
        policy.tracker = tracker
        action = SafeBot(policy).act(to_game_state(state, log, index, rules))
        return to_kit_action(action, state), new_memory
    except RulesError as exc:
        _report_once(f"RULES ERROR (checking/folding every decision): {exc}")
        return _fallback(state), new_memory
    except Exception:
        return _fallback(state), new_memory


_reported: set = set()


def _report_once(message: str) -> None:
    """One stderr line per problem per process. The tournament never shows it; local
    `poker-agent play` does, and pack.sh fails a release that prints it."""
    if message not in _reported:
        _reported.add(message)
        print(f"nolimit-bot: {message}", file=sys.stderr, flush=True)


class Agent(sdk.Agent):
    def act(self, state, memory):
        return decide(state, memory)


def build_bot():
    """The strategy behind the agent, for local engine runs: PolicyBot (v5) behind SafeBot.

    Fallbacks if it misbehaves against the real kit, in order:
    PolicyBot(V4_FEATURES) (no ICM, so no dependence on payouts),
    bot.pushfold.PushFoldBot (same strategy, no tracker), then
    bot.baseline.BaselineBot.
    """
    return SafeBot(PolicyBot())


def _warm_up() -> None:
    """Import, build every table and run one decision per registered variant now, so
    the first real call doesn't pay for cold caches or a hand-table import
    (importing has its own startup allowance)."""
    for variant in VARIANTS:
        _warm_up_one(variant)


def _warm_up_one(variant: str) -> None:
    from types import SimpleNamespace as NS

    hole = {2: ("As", "Ad"), 4: ("As", "Ad", "Kh", "Qc")}.get(VARIANTS[variant]["hole_cards"], ())
    legal = (NS(kind="fold", min_amount=0, max_amount=0), NS(kind="call", min_amount=20, max_amount=20),
             NS(kind="raise", min_amount=40, max_amount=2000))
    seats = tuple(NS(seat=i, stack=2000 - (10 if i == 1 else 20 if i == 2 else 0),
                     bet=10 if i == 1 else 20 if i == 2 else 0, status="active") for i in range(8))
    events = (NS(kind="hand", seat=None, amount=1, cards=()), NS(kind="blind", seat=1, amount=10, cards=()),
              NS(kind="blind", seat=2, amount=20, cards=()))
    config = NS(variant=variant, players=8, starting_stack=2000, blinds=((10, 20, 0),), hands_per_level=20,
                hand_cap=250, payouts=(50, 30, 20, 0, 0, 0, 0, 0))
    state = NS(seat=3, cards=hole, board=(), pot=30, to_call=20, legal_actions=legal, seats=seats,
               button=0, blinds=(10, 20), hand_number=1, players_remaining=8, events=events,
               phase="betting", config=config, can=lambda k: next((a for a in legal if a.kind == k), None))
    try:
        decide(state, b"")
        warm_evaluator(rules_from_config(config))  # the first postflop read builds its tables
    except Exception:
        pass


_warm_up()
# Freeze everything loaded so far so a full GC can't land inside a decision.
tune_gc()
