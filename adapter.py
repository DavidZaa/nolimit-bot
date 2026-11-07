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

Rules degrade gracefully (bot.rules.rules_from_state). A registered variant
with a complete config plays its exact rules. Anything else plays the best
rules the config and the state allow: an unknown variant ID PLO-style with 4+
hole cards (else hold'em-style, pot-limit if the max raise is below the
all-in), bad payouts as 50/30/20 over the seats, a bad blinds schedule, level
length or hand cap as the current blinds, 20 and 250, missing players as the
seats at the table; each guess prints one RULES WARNING line to stderr. Only
if even that is impossible does it check or fold (RULES ERROR). The
tournament never shows stderr, so a silent check/fold bot would look healthy
while scoring nothing; that's why guessing beats failing closed here.

NLB_STRICT=1 restores fail-closed (any guess is a RULES ERROR and a
check/fold), so local checks and scripts/pack.sh catch config problems.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # bot/ sits next to agent.py

import poker_agent_sdk as sdk  # noqa: E402

from bot.events import TableLog  # noqa: E402
from bot.flags import flag  # noqa: E402
from bot.handeval import warm as warm_evaluator  # noqa: E402
from bot.policy import PolicyBot  # noqa: E402
from bot.rules import VARIANTS, RulesError, blind_level, rules_from_config, rules_from_state  # noqa: E402
from bot.runtime import tune_gc  # noqa: E402
from bot.safety import SafeBot  # noqa: E402
from bot.select import DEFAULT_SELECT, choose_select  # noqa: E402
from bot.state import (  # noqa: E402
    CALL, CHECK, FOLD, POST_BB, POST_SB, RAISE, Action, GameRules, GameState, LegalActions, card_from_str,
)
from bot.tracker import Tracker  # noqa: E402

# -- The Bay Area variant's SDK names (kit RULES.md, poker_agent_protocol.game) ----------
# After the flop is dealt and before flop betting, state.phase is "select" and the only
# legal kind is "select": Action("select", cards=(discard_index, reveal_index)) with indexes
# into the three cards in dealing order and amount 0. A failed or empty selection plays
# (2, 0). Afterwards state.cards holds the two kept cards, and a "reveal" event per seat
# carries its exposed card.
VARIANT_ALIASES: dict = {}  # SDK variant ID -> bot.rules.VARIANTS, for an ID not registered there
SELECT_PHASE = "select"
SELECT_KIND = "select"

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
    try:
        return blind_level(getattr(state, "blinds", None))  # the kit's BlindLevel, or a sequence
    except Exception:
        posts = {k: a for _, st, k, a in log.records if st == 0 and k in (POST_SB, POST_BB)}
        big = posts.get(POST_BB, 0)
        return posts.get(POST_SB, big // 2), big, 0


def legal_actions(state, stack: int, bet: int) -> LegalActions:
    """The kit's offered kinds in our terms. Its raise min/max are raise-to totals for the
    round, like ours; all-in is our own bet plus stack, never assumed to be the max."""
    fold, check, call, raise_ = (state.can(k) for k in ("fold", "check", "call", "raise"))
    to_call = min(int(getattr(state, "to_call", 0) or 0), stack)
    if raise_ is not None:
        lo, hi = int(raise_.min_amount), int(raise_.max_amount)
    else:
        lo = hi = 0
    return LegalActions(fold is not None, check is not None, call is not None, to_call,
                        raise_ is not None and hi >= lo > 0, lo if raise_ else 0, hi if raise_ else 0,
                        bet + stack, bet)


def exposed_cards(state, log: TableLog, index: dict) -> tuple[int, ...]:
    """Each seat's exposed card this hand (-1 if none): from the reveal events, else from what
    the seat shows (shown_cards holds just the exposed card until a showdown)."""
    out = [-1] * len(index)
    for seat, card in log.exposed.items():
        if 0 <= seat < len(out):
            out[seat] = card_from_str(card)
    for s in state.seats:
        i = index.get(s.seat)
        shown = tuple(getattr(s, "shown_cards", ()) or ())
        if i is not None and out[i] < 0 and len(shown) == 1:
            out[i] = card_from_str(str(shown[0]))
    return tuple(out)


def dead_cards(state, log: TableLog) -> tuple[int, ...]:
    """Every card we know is out of play: all exposed cards this hand (folded seats' too) and
    our own discard (state.variant["discarded"], this bot's only)."""
    out = {card_from_str(c) for c in log.exposed.values()}
    try:
        for c in (getattr(state, "variant", None) or {}).get("discarded", ()) or ():
            out.add(card_from_str(str(c)))
    except Exception:
        pass
    return tuple(sorted(out))


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
        legal=legal_actions(state, stacks[me], bets[me]), actions=log.actions(),
        rules=rules, exposed=exposed_cards(state, log, index), dead=dead_cards(state, log),
    )


# -- actions ---------------------------------------------------------------------

def automatic(state):
    """What the kit plays itself when neither check nor fold is offered: the first legal kind."""
    first = state.legal_actions[0]
    return sdk.Action(first.kind, first.min_amount)


def to_kit_action(action: Action, state):
    """Our action as the kit's, legal by construction. Each intent maps to itself when the
    kit offers it; otherwise to its nearest legal neighbour, never across a fold/call line:

      raise   raise_to(total) clamped to the offered min..max (both raise-to totals, as
              ours are); without a raise on offer, call (it wanted chips in), else check
      call    call; else check (nothing to call)
      check   check; else fold (a check intention never puts chips in)
      fold    fold; else check (never fold when checking is free)

    The kit doesn't clamp raises and rounds floats down, so the total is an int inside
    min_amount..max_amount before it leaves here. Anything else gets the fallback.
    """
    can = state.can
    kind = getattr(action, "kind", None)
    if kind == RAISE:
        r = can("raise")
        if r is not None and int(r.max_amount) >= int(r.min_amount):
            try:
                total = int(action.amount)
            except (TypeError, ValueError, OverflowError):
                total = int(r.min_amount)
            return sdk.Action.raise_to(max(int(r.min_amount), min(total, int(r.max_amount))))
        if can("call") is not None:
            return sdk.Action.call()
        return _fallback(state)
    if kind == CALL:
        if can("call") is not None:
            return sdk.Action.call()
        return _fallback(state)
    if kind == CHECK:
        if can("check") is not None:
            return sdk.Action.check()
        return sdk.Action.fold() if can("fold") is not None else _fallback(state)
    if kind == FOLD:
        if can("fold") is not None:
            return sdk.Action.fold()
        return _fallback(state)
    return _fallback(state)


def select_action(state):
    """The Bay Area selection: our discard and reveal, or the kit's default (2, 0) on any error
    (or when the build flag select=default asks for it, for A/B tests)."""
    if flag("select", "smart") == "default":
        return sdk.Action(SELECT_KIND, 0, DEFAULT_SELECT)
    try:
        try:
            rules, warnings = rules_from_state(state, aliases=VARIANT_ALIASES)
            for warning in warnings:
                _report_once(f"RULES WARNING (playing on a guess): {warning}")
        except RulesError:
            rules = None  # the choosers' own rules for this variant
        hole = tuple(to_card(c) for c in state.cards)
        board = tuple(to_card(c) for c in state.board)
        discard, reveal = choose_select(hole, board, rules)
        return sdk.Action(SELECT_KIND, 0, (int(discard), int(reveal)))
    except Exception:
        return sdk.Action(SELECT_KIND, 0, DEFAULT_SELECT)


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
    if getattr(state, "phase", None) == SELECT_PHASE or state.can(SELECT_KIND) is not None:
        return select_action(state), new_memory
    try:
        if getattr(state, "phase", "betting") != "betting" or not any(
                state.can(k) for k in ("fold", "check", "call", "raise")):
            return automatic(state), new_memory
        rules, warnings = rules_from_state(state, aliases=VARIANT_ALIASES)
        for warning in warnings:
            _report_once(f"RULES WARNING (playing on a guess): {warning}")
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

    hole = {2: ("As", "Ad"), 3: ("As", "Ad", "Kh"), 4: ("As", "Ad", "Kh", "Qc")}.get(
        VARIANTS[variant]["hole_cards"], ())
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
