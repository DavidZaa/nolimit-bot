"""The only file that knows the starter kit's API.

Everything else is written against bot.state. On kit day:

  1. Read the kit's Agent template: method names, what an observation holds,
     how actions are returned, how cards are written.
  2. Rename/re-sign `Agent` below to match the template exactly.
  3. Fill in `to_game_state` and `to_kit_action` (the PLACEHOLDER(kit) spots).
  4. Run `pytest tests/test_adapter.py` plus a few hands in the kit's own
     simulator, then submit. The baseline behind it is already tested.

Anything the kit doesn't expose (antes, the action log, other seats' bets)
can be left at a neutral value; the baseline only needs stacks, bets,
blinds, positions and the legal actions.
"""

from __future__ import annotations

from bot.baseline import BaselineBot
from bot.runtime import tune_gc
from bot.safety import SafeBot
from bot.state import CALL, CHECK, FOLD, RAISE, Action, GameState, LegalActions, card_from_str


def build_bot():
    """The policy we play. Swap BaselineBot for the full policy once it's ready."""
    return SafeBot(BaselineBot())


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

    def act(self, observation):
        return to_kit_action(self.bot.act(to_game_state(observation)), observation)


# Freeze everything loaded so far so a full GC can't land inside a decision.
tune_gc()
