"""Sparring partners modelled on what a 60-minute field tends to produce.

The first four are the broken and degenerate bots. The last three are what a
team gets by asking Codex for "a solid poker bot" in an hour, each with the
blind spot that kind of code usually has:

- ChenTAG: fixed Chen-formula cutoffs and a 3x open at every stack depth,
  bets top pair or better and gives up otherwise. It calls off its whole
  stack with any hand it would call a min-raise with.
- EquityCaller: calls whenever its equity against a *random* hand beats the
  pot odds. Bettors don't hold random hands, so it pays off big bets.
- PushFold: Nash-shaped shove/fold ranges by stack depth and position,
  raise/fold when deep. The stand-in for the few strong teams.

They live engine-side and never ship; they can be as slow as the engine's
evaluator (tens of µs), which is still well inside the 1 ms limit.
"""

from __future__ import annotations

import random

from bot.baseline import chen_score, preflop_percentile
from bot.state import CALL, CHECK, FOLD, RAISE, Action, fallback_action
from engine.handeval import best_hand, category
from engine.rules import NLHE, PAIR, PLO, TRIPS, TWO_PAIR


class FolderBot:
    """A bot that crashes or times out every hand: the tournament checks/folds for it."""

    def act(self, state):
        return fallback_action(state.legal)


class CallingStation:
    def act(self, state):
        legal = state.legal
        return Action(CALL) if legal.can_call else Action(CHECK)


class Maniac:
    """Shoves every decision it can."""

    def act(self, state):
        legal = state.legal
        if legal.can_raise:
            return Action(RAISE, legal.max_raise_to)
        return Action(CALL) if legal.can_call else Action(CHECK)


class RandomBot:
    """Uniformly random legal action; raises pick min, max, or something between.

    Folds are down-weighted so hands actually reach later streets and showdowns.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, state):
        legal = state.legal
        rng = self.rng
        options = []
        if legal.can_fold and rng.random() < 0.35:
            options.append(Action(FOLD))
        if legal.can_check:
            options.append(Action(CHECK))
        if legal.can_call:
            options.append(Action(CALL))
        if legal.can_raise and rng.random() < 0.5:
            lo, hi = legal.min_raise_to, legal.max_raise_to
            options.append(Action(RAISE, rng.choice((lo, hi, rng.randint(lo, hi)))))
        if not options:
            return fallback_action(legal)
        return rng.choice(options)


# --- shared reads of the table ------------------------------------------------------

# Made-hand strength buckets, weakest first.
NOTHING, WEAK_PAIR, TOP_PAIR, TWO_PAIR_PLUS, MONSTER = range(5)


def made_hand(hole, board) -> int:
    """Bucket a postflop hand. A pair that's only on the board counts as nothing."""
    variant = PLO if len(hole) == 4 else NLHE
    cat = category(best_hand(hole, board, variant), variant)
    if cat >= TRIPS:  # trips, straights, flushes and up (category ids follow strength)
        return MONSTER
    if cat == TWO_PAIR:
        return TWO_PAIR_PLUS
    if cat != PAIR:
        return NOTHING
    board_ranks = {c >> 2 for c in board}
    top = max(board_ranks)
    hole_ranks = [c >> 2 for c in hole]
    pockets = [r for r in set(hole_ranks) if hole_ranks.count(r) >= 2]
    if pockets:
        return TOP_PAIR if max(pockets) > top else WEAK_PAIR
    paired = [r for r in set(hole_ranks) if r in board_ranks]
    if not paired:
        return NOTHING
    return TOP_PAIR if max(paired) == top else WEAK_PAIR


def draw_outs(hole, board) -> int:
    """Outs to a flush draw (9) and an open-ended straight draw (8) that use a hole card."""
    if len(board) >= 5:
        return 0
    cards = (*hole, *board)
    outs = 0
    for suit in range(4):
        if sum(1 for c in cards if c & 3 == suit) == 4 and any(c & 3 == suit for c in hole):
            outs += 9
            break
    ranks = {c >> 2 for c in cards}
    hole_ranks = {c >> 2 for c in hole}
    # Four in a row with a card free at both ends (the ace closes the bottom of 2345).
    for low in range(0, 9):
        window = set(range(low, low + 4))
        if window <= ranks and window & hole_ranks:
            outs += 8
            break
    return outs


def equity_vs_random(hole) -> float:
    """Rough preflop all-in equity against one random hand: AA .85, 22 .50, 72o .35."""
    if len(hole) != 2:  # Omaha hands run close together
        return 0.40 + 0.25 * (1 - preflop_percentile(hole))
    a, b = hole
    hi, lo = sorted((a >> 2, b >> 2), reverse=True)
    if hi == lo:
        return 0.50 + 0.029 * hi
    eq = 0.346 + 0.029 * (hi - 5) + 0.0095 * lo
    if (a & 3) == (b & 3):
        eq += 0.03
    if hi - lo <= 2:
        eq += 0.01
    return max(0.30, eq)


_MADE_EQUITY = (0.35, 0.60, 0.75, 0.85, 0.92)  # by made_hand bucket


def postflop_equity(hole, board) -> float:
    """Equity against a random hand the way quick bot code estimates it: made hand plus
    the "rule of 4 and 2" for draws."""
    strength = made_hand(hole, board)
    eq = _MADE_EQUITY[strength]
    outs = draw_outs(hole, board)
    if outs and strength < TWO_PAIR_PLUS:
        eq = min(0.90, eq + outs * (0.04 if len(board) == 3 else 0.02))
    return eq


def _raise_to(legal, amount) -> Action:
    """Raise as close to `amount` as the rules allow; call or check when raising isn't."""
    if legal.can_raise:
        return Action(RAISE, max(legal.min_raise_to, min(int(amount), legal.max_raise_to)))
    return Action(CALL) if legal.can_call else Action(CHECK)


def _check_or(legal, call: bool) -> Action:
    if legal.can_check:
        return Action(CHECK)
    return Action(CALL) if call else Action(FOLD)


def _dealt_in(state, s) -> bool:
    return state.in_hand[s] or state.stacks[s] > 0  # busted seats have neither


def _behind(state, seat: int) -> int:
    """Seats dealt in that act after `seat` preflop (the blinds act last)."""
    n = len(state.stacks)
    first = (state.bb_seat + 1) % n
    pos = (seat - first) % n
    return sum(1 for s in range(n) if s != seat and _dealt_in(state, s) and (s - first) % n > pos)


def _depth_bb(state) -> float:
    """Effective stack in big blinds against the deepest opponent still in."""
    seat = state.seat
    mine = state.stacks[seat] + state.bets[seat]
    deepest = max(
        (state.stacks[s] + state.bets[s] for s in range(len(state.stacks)) if s != seat and state.in_hand[s]),
        default=0,
    )
    return min(mine, deepest) / max(1, state.big_blind)


def _last_raiser(state, street: int):
    for rec in reversed(state.actions):
        if rec.street == street and rec.kind == RAISE:
            return rec.seat
    return None


# --- LLM-default bots ---------------------------------------------------------------

class ChenTAG:
    """Fixed Chen cutoffs, 3x opens, top pair or better bets; stack depth never enters."""

    OPEN, CALL_RAISE, RERAISE = 8, 9, 12

    def act(self, state):
        legal = state.legal
        bb = max(1, state.big_blind)
        if state.street == 0:
            chen = chen_score(state.hole[:2])  # Omaha: judges the first two cards, naively
            highest = max(state.bets)
            if highest <= bb:
                if chen >= self.OPEN:
                    limpers = sum(1 for s, b in enumerate(state.bets) if b == bb and s != state.bb_seat)
                    return _raise_to(legal, (3 + limpers) * bb)
                return _check_or(legal, call=False)
            if chen >= self.RERAISE:
                return _raise_to(legal, 3 * highest)
            return _check_or(legal, call=chen >= self.CALL_RAISE)

        strength = made_hand(state.hole, state.board)
        if strength < TOP_PAIR:
            return _check_or(legal, call=False)
        if legal.can_check:
            return _raise_to(legal, state.pot * 2 // 3)
        if strength >= TWO_PAIR_PLUS:
            return _raise_to(legal, 3 * max(state.bets))
        return Action(CALL)


class EquityCaller:
    """Calls whenever equity against a random hand beats the price, at any bet size."""

    def act(self, state):
        legal = state.legal
        if state.street == 0:
            eq = equity_vs_random(state.hole)
            if max(state.bets) <= state.big_blind:
                if eq >= 0.62:
                    return _raise_to(legal, 3 * state.big_blind)
                return _check_or(legal, call=eq >= 0.50)
        else:
            eq = postflop_equity(state.hole, state.board)
            if legal.can_check:
                return _raise_to(legal, state.pot * 2 // 3) if eq >= 0.70 else Action(CHECK)

        if eq >= 0.80 and legal.can_raise:
            return _raise_to(legal, 3 * max(state.bets))
        to_call = legal.call_amount
        return Action(CALL) if eq > to_call / (state.pot + to_call) else Action(FOLD)


def shove_range(depth_bb: float, behind: int) -> float:
    """Share of hands to shove first-in. Shaped like Nash charts (SB ~58% at 10 BB,
    UTG ~12%), not solved."""
    return min(1.0, 5.8 / (max(depth_bb, 1.0) * max(behind, 1) ** 0.8))


class PushFold:
    """Shove/fold at short stacks, raise/fold when deep, plain postflop."""

    SHOVE_DEPTH = 15  # first-in, shove or fold at or below this many big blinds
    OPEN_TO = 2.5

    def act(self, state):
        if state.street != 0:
            return self._postflop(state)
        legal, seat = state.legal, state.seat
        bb = max(1, state.big_blind)
        pct = preflop_percentile(state.hole)
        depth = _depth_bb(state)
        highest = max(state.bets)

        if highest <= bb:
            behind = _behind(state, seat)
            if depth <= self.SHOVE_DEPTH:
                if pct <= shove_range(depth, behind):
                    return _raise_to(legal, legal.max_raise_to)
                return _check_or(legal, call=False)
            if pct <= 0.45 / max(behind, 1) ** 0.8:
                return _raise_to(legal, self.OPEN_TO * bb)
            return _check_or(legal, call=False)

        if legal.call_amount >= 0.4 * state.stacks[seat] or depth <= 20:
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

        if pct <= 0.025:
            return _raise_to(legal, 3 * highest)
        cheap = legal.call_amount <= 0.1 * (state.stacks[seat] + state.bets[seat])
        return _check_or(legal, call=pct <= 0.08 and cheap)

    def _postflop(self, state):
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
