"""The only file that knows the starter kit's API.

Everything else is written against bot.state. On kit day:

  1. Read the kit's Agent template: method names, what an observation holds,
     how actions are returned, how cards are written.
  2. Rename/re-sign `Agent` below to match the template exactly.
  3. Fill in `to_game_state` and `to_kit_action` (the PLACEHOLDER(kit) spots).
     If the kit reports finished hands, fill in `to_hand_result` and call
     `Agent.hand_over` from its end-of-hand hook: the opponent tracker then
     sees what happened after we folded, not just the hands we played.
  4. Run `pytest tests/test_adapter.py` plus a few hands in the kit's own
     simulator, then submit. The bot behind it is already tested.

Anything the kit doesn't expose (antes, other seats' bets) can be left at a
neutral value. The push/fold bot needs stacks, bets, blinds, positions, the
board and the legal actions; the action log only sharpens two reads (who
raised, for shove ranges and continuation bets) and can be left empty.
"""

from __future__ import annotations

from bot.policy import PolicyBot
from bot.runtime import tune_gc
from bot.safety import SafeBot
from bot.state import (
    CALL, CHECK, FOLD, RAISE, Action, GameState, HandResult, LegalActions, card_from_str,
)


def build_bot():
    """The policy we play: push/fold adjusted per seat by the opponent tracker.

    Fallbacks if it misbehaves against the real kit, in order:
    bot.pushfold.PushFoldBot (same strategy, no tracker), then
    bot.baseline.BaselineBot.
    """
    return SafeBot(PolicyBot())


def legal_actions(to_call: int, stack: int, min_raise_to: int, max_raise_to: int,
                  can_raise: bool = True) -> LegalActions:
    """Build consistent LegalActions from the numbers most kits expose.

    `to_call` is the chips still needed to call, `stack` the chips behind,
    and the raise bounds are "raise to" totals for this street.
    """
    facing = to_call > 0
    raise_ok = can_raise and stack > to_call and max_raise_to > 0
    lo = min(min_raise_to, max_raise_to) if raise_ok else 0
    hi = max_raise_to if raise_ok else 0
    return LegalActions(facing, not facing, facing, min(to_call, stack), raise_ok, lo, hi)


def to_card(kit_card) -> int:
    """PLACEHOLDER(kit): assumes cards like "As"/"Td"; adjust to the kit's format."""
    return card_from_str(str(kit_card))


def to_game_state(observation) -> GameState:
    """PLACEHOLDER(kit): map the kit's observation into our GameState."""
    raise NotImplementedError("map the kit observation into bot.state.GameState")


def to_hand_result(kit_result) -> HandResult:
    """PLACEHOLDER(kit): map the kit's end-of-hand report (stacks, shown cards, the
    hand's actions) into a HandResult. Leave `actions` empty if it has none."""
    raise NotImplementedError("map the kit's hand report into bot.state.HandResult")


def to_kit_action(action: Action, observation):
    """PLACEHOLDER(kit): turn our Action into whatever the kit expects back."""
    kind = action.kind
    if kind in (FOLD, CHECK, CALL):
        raise NotImplementedError(f"map {kind} to the kit's action")
    if kind == RAISE:
        raise NotImplementedError("map a raise-to total to the kit's raise action")
    raise ValueError(kind)


class Agent:
    """PLACEHOLDER(kit): match the class name and method signatures of the kit's template.

    If anything in here raises, the tournament checks/folds for us, which is
    the same thing our own fallback would do, so no extra guard is needed.
    """

    def __init__(self):
        self.bot = build_bot()

    def new_table(self, seat: int, num_seats: int) -> None:
        self.bot.new_table(seat, num_seats)

    def hand_over(self, kit_result) -> None:
        """PLACEHOLDER(kit): wire to the kit's end-of-hand hook, if it has one."""
        self.bot.hand_over(to_hand_result(kit_result))

    def act(self, observation):
        return to_kit_action(self.bot.act(to_game_state(observation)), observation)


# Freeze everything loaded so far so a full GC can't land inside a decision.
tune_gc()
