"""Named betting intents, made legal: what a strategy means, separated from what the table offers.

Amounts follow one convention everywhere in bot/ and the kit alike: a RAISE
amount is the total this player's bet becomes for the current betting round
("raise to"), not the chips added. An opening bet is a raise from zero.

Three different things that are easy to confuse:

  all_in(legal)     every chip: a raise to legal.all_in_to when that total is
                    legal, or a call when calling already takes every chip
                    (a short call). None when going all-in isn't legal now,
                    e.g. pot-limit caps the raise below the stack.
  max_raise(legal)  the largest raise offered. In no-limit that's all-in; in
                    pot-limit it's the pot, with chips left behind.
  commit(legal)     what push/fold means by "shove": all-in where legal,
                    otherwise the max raise (in pot-limit, the pot), and a
                    call when no raise is allowed at all.

None of them guesses: all_in_to comes from the table (the seat's bet plus
its stack), never from max_raise_to.
"""

from __future__ import annotations

from bot.state import CALL, CHECK, FOLD, RAISE, Action, LegalActions


def can_raise_to(legal: LegalActions, total: int) -> bool:
    return legal.can_raise and legal.min_raise_to <= total <= legal.max_raise_to


def all_in(legal: LegalActions) -> Action | None:
    """Every chip, legally, or None if that isn't possible right now."""
    total = legal.all_in_to
    if total <= 0:
        return None  # the table didn't say what all-in is: don't assume max_raise_to
    if can_raise_to(legal, total):
        return Action(RAISE, total)
    # Calling takes everything when the call is at least the whole stack (the table
    # reports call_amount already capped at the stack).
    if legal.can_call and legal.call_amount > 0 and legal.call_amount >= legal.all_in_to - legal.bet:
        return Action(CALL)
    return None


def max_raise(legal: LegalActions) -> Action | None:
    """The largest raise offered (pot-limit: the pot), or None if raising isn't allowed."""
    return Action(RAISE, legal.max_raise_to) if legal.can_raise else None


def raise_to(legal: LegalActions, total) -> Action:
    """A raise as close to `total` as the rules allow; call, then check, when raising isn't."""
    if legal.can_raise:
        return Action(RAISE, max(legal.min_raise_to, min(int(total), legal.max_raise_to)))
    return Action(CALL) if legal.can_call else Action(CHECK)


def commit(legal: LegalActions) -> Action:
    """Get the chips in: all-in if legal, else the max raise, else call, else check."""
    return all_in(legal) or max_raise(legal) or (Action(CALL) if legal.can_call else Action(CHECK))


def check_or(legal: LegalActions, call: bool) -> Action:
    """Check when free; otherwise call if `call`, else fold."""
    if legal.can_check:
        return Action(CHECK)
    return Action(CALL) if call else Action(FOLD)
