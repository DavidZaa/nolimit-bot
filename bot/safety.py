"""Last line of defence between a strategy and the table.

In the tournament an exception or an illegal action is an automatic
check/fold, and over millions of decisions any rare bug will fire. SafeBot
catches everything and repairs the action instead, keeping its intent where
it can: an oversized raise becomes a shove, a raise that isn't allowed
becomes a call, and a fold is never thrown away when checking is free.
"""

from __future__ import annotations

from bot.state import CALL, CHECK, FOLD, RAISE, Action, LegalActions, fallback_action


def make_legal(action, legal: LegalActions) -> Action:
    try:
        kind = action.kind
    except Exception:
        return fallback_action(legal)

    if kind == RAISE:
        if not legal.can_raise:
            if legal.can_call:
                return Action(CALL)
            return Action(CHECK) if legal.can_check else Action(FOLD)
        lo, hi = legal.min_raise_to, legal.max_raise_to
        try:
            amount = int(action.amount)  # floats, numpy ints, bools...; nan/inf raise
        except Exception:
            amount = lo
        return Action(RAISE, lo if amount < lo else hi if amount > hi else amount)
    if kind == CALL:
        if legal.can_call:
            return Action(CALL)
        return Action(CHECK) if legal.can_check else Action(FOLD)
    if kind == CHECK:
        return Action(CHECK) if legal.can_check else Action(FOLD)
    if kind == FOLD:
        return Action(FOLD) if legal.can_fold else Action(CHECK)
    return fallback_action(legal)


class SafeBot:
    """Wraps any bot so that nothing it does can produce an illegal action.

    It counts what it had to absorb, so evaluations can report it: `errors` (the
    strategy raised; the fallback played) and `repairs` (it returned something
    illegal that was mended into a legal action).
    """

    def __init__(self, inner):
        self.inner = inner
        self.errors = 0
        self.repairs = 0

    def new_table(self, seat: int, num_seats: int) -> None:
        hook = getattr(self.inner, "new_table", None)
        if hook is not None:
            try:
                hook(seat, num_seats)
            except Exception:
                pass

    def hand_over(self, result) -> None:
        hook = getattr(self.inner, "hand_over", None)
        if hook is not None:
            try:
                hook(result)
            except Exception:
                pass

    def act(self, state) -> Action:
        legal = state.legal
        try:
            wanted = self.inner.act(state)
        except Exception:
            self.errors += 1
            return fallback_action(legal)
        action = make_legal(wanted, legal)
        if action != wanted:
            self.repairs += 1
        return action
