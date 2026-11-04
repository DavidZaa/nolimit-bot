"""Push/fold bot: what we submit first on kit day.

At short stacks it shoves or folds with Nash-shaped ranges that widen as the
stack shrinks and as fewer players are left behind; deeper, it opens small
and raises or folds. Postflop it bets made hands, continuation-bets heads up
as the raiser, and gets out otherwise.

It started as a port of the field's PushFold opponent (engine/field.py),
which outscored the old shove-everything baseline by about 7 points a
table. That copy stays frozen as a sparring partner while this one gets
tuned, so the two are kept separate on purpose.
"""

from __future__ import annotations

from bot.baseline import preflop_percentile
from bot.hands import TOP_PAIR, TWO_PAIR_PLUS, made_hand
from bot.state import CALL, CHECK, FOLD, RAISE, Action

SHOVE_DEPTH_BB = 15  # first-in, shove or fold at or below this many big blinds
OPEN_TO_BB = 2.5
SHOVE_CALL_DEPTH_BB = 20  # facing a raise at or below this, it's all-in or fold
PREMIUM = 0.025  # top share of hands that 3-bets when deep (QQ+, AK)
DEEP_CALL = 0.08  # top share that flats a cheap raise when deep


def shove_range(depth_bb: float, behind: int) -> float:
    """Share of hands to shove first-in. Shaped like Nash charts (SB ~58% at 10 BB,
    UTG ~12%), not solved."""
    return min(1.0, 5.8 / (max(depth_bb, 1.0) * max(behind, 1) ** 0.8))


def _raise_to(legal, amount) -> Action:
    """Raise as close to `amount` as the rules allow; call or check when raising isn't."""
    if legal.can_raise:
        return Action(RAISE, max(legal.min_raise_to, min(int(amount), legal.max_raise_to)))
    return Action(CALL) if legal.can_call else Action(CHECK)


def _check_or(legal, call: bool) -> Action:
    if legal.can_check:
        return Action(CHECK)
    return Action(CALL) if call else Action(FOLD)


def _behind(state, seat: int) -> int:
    """Seats dealt in that act after `seat` preflop (the blinds act last)."""
    n = len(state.stacks)
    first = (state.bb_seat + 1) % n
    pos = (seat - first) % n
    in_hand, stacks = state.in_hand, state.stacks
    return sum(1 for s in range(n)
               if s != seat and (in_hand[s] or stacks[s] > 0) and (s - first) % n > pos)


def _depth_bb(state) -> float:
    """Effective stack in big blinds against the deepest opponent still in."""
    seat, stacks, bets = state.seat, state.stacks, state.bets
    deepest = max((stacks[s] + bets[s] for s in range(len(stacks)) if s != seat and state.in_hand[s]),
                  default=0)
    return min(stacks[seat] + bets[seat], deepest) / max(1, state.big_blind)


def _last_raiser(state, street: int):
    for rec in reversed(state.actions):
        if rec.street == street and rec.kind == RAISE:
            return rec.seat
    return None


class PushFoldBot:
    def act(self, state) -> Action:
        if state.street != 0:
            return self._postflop(state)
        legal, seat = state.legal, state.seat
        bb = max(1, state.big_blind)
        pct = preflop_percentile(state.hole)
        depth = _depth_bb(state)
        highest = max(state.bets)

        if highest <= bb:
            behind = _behind(state, seat)
            if depth <= SHOVE_DEPTH_BB:
                if pct <= shove_range(depth, behind):
                    return _raise_to(legal, legal.max_raise_to)
                return _check_or(legal, call=False)
            if pct <= 0.45 / max(behind, 1) ** 0.8:
                return _raise_to(legal, OPEN_TO_BB * bb)
            return _check_or(legal, call=False)

        if legal.call_amount >= 0.4 * state.stacks[seat] or depth <= SHOVE_CALL_DEPTH_BB:
            # Effectively all-in: get it in with the top of the range they'd shove.
            raiser = _last_raiser(state, 0)
            if raiser is None:
                theirs = 0.10
            else:
                their_depth = (state.stacks[raiser] + state.bets[raiser]) / bb
                theirs = shove_range(their_depth, _behind(state, raiser))
            waiting = sum(1 for s in range(len(state.stacks))
                          if s not in (seat, raiser) and state.in_hand[s] and state.stacks[s] > 0)
            if pct <= theirs * 0.7 / (1 + 0.5 * waiting):
                return _raise_to(legal, legal.max_raise_to)
            return _check_or(legal, call=False)

        if pct <= PREMIUM:
            return _raise_to(legal, 3 * highest)
        cheap = legal.call_amount <= 0.1 * (state.stacks[seat] + state.bets[seat])
        return _check_or(legal, call=pct <= DEEP_CALL and cheap)

    def _postflop(self, state) -> Action:
        legal = state.legal
        strength = made_hand(state.hole, state.board)
        if legal.can_check:
            heads_up = sum(state.in_hand) == 2
            cbet = state.street == 1 and heads_up and _last_raiser(state, 0) == state.seat
            if strength >= TOP_PAIR or cbet:
                return _raise_to(legal, state.pot // 2)
            return Action(CHECK)
        if strength >= TWO_PAIR_PLUS:
            return _raise_to(legal, 3 * max(state.bets))
        pot_sized_or_less = legal.call_amount * 2 <= state.pot  # the pot already holds the bet
        return _check_or(legal, call=strength >= TOP_PAIR and pot_sized_or_less)
