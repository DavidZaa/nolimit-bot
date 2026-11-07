"""Baseline push/fold bot: the thing we submit in the first five minutes.

It only has to be legal, fast and not embarrassing. Preflop it either shoves
or gets out, with ranges that widen as stacks get shorter, as fewer players
are left to act behind it, and when nobody has raised yet. Postflop it checks
and only calls at a big price. The real policy replaces all of this; the
baseline stays as the always-safe fallback.

Hand strength is the Chen formula turned into a percentile ("top x% of
starting hands") once at import, so a decision is one table lookup.
"""

from __future__ import annotations

from itertools import combinations

from bot.actions import commit

from bot.state import CALL, CHECK, FOLD, Action

# Chen formula points for the high card, indexed by rank (2..A).
_HIGH = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10)
_QUEEN = 10


def _chen(hi: int, lo: int, suited: bool) -> float:
    if hi == lo:
        return max(5.0, _HIGH[hi] * 2)
    score = _HIGH[hi] + (2 if suited else 0)
    gap = hi - lo - 1
    score -= (0, 1, 2, 4)[gap] if gap < 4 else 5
    if gap <= 1 and hi < _QUEEN:
        score += 1
    return score


def chen_score(hole) -> float:
    """Raw Chen points for two hole cards: AA 20, AKs 12, 72o -1.5."""
    a, b = hole
    ra, rb = a >> 2, b >> 2
    hi, lo = (ra, rb) if ra >= rb else (rb, ra)
    return _chen(hi, lo, (a & 3) == (b & 3))


def _build_percentiles() -> list[float]:
    classes = []
    for hi in range(13):
        for lo in range(hi + 1):
            for suited in ((False,) if hi == lo else (False, True)):
                combos = 6 if hi == lo else 4 if suited else 12
                classes.append((_chen(hi, lo, suited), hi, lo, suited, combos))
    classes.sort(key=lambda c: (-c[0], -c[1], -c[2]))
    table = [1.0] * (13 * 13 * 2)
    better = 0
    for _, hi, lo, suited, combos in classes:
        # Midpoint of the class's slice of all 1326 starting hands.
        table[(hi * 13 + lo) * 2 + suited] = (better + combos / 2) / 1326
        better += combos
    return table


_PERCENTILE = _build_percentiles()


def preflop_percentile(hole) -> float:
    """Fraction of starting hands at least this good: AA ~0.002, 72o ~1.0."""
    if len(hole) == 2:
        a, b = hole
        ra, rb = a >> 2, b >> 2
        hi, lo = (ra, rb) if ra >= rb else (rb, ra)
        return _PERCENTILE[(hi * 13 + lo) * 2 + ((a & 3) == (b & 3))]
    if len(hole) > 2:  # Omaha-style hands: judge by the best two cards
        return min(preflop_percentile(pair) for pair in combinations(hole, 2))
    return 1.0


def _open_range(depth_bb: float) -> float:
    if depth_bb <= 5:
        return 0.60
    if depth_bb <= 8:
        return 0.40
    if depth_bb <= 12:
        return 0.25
    if depth_bb <= 20:
        return 0.12
    if depth_bb <= 40:
        return 0.06
    return 0.03


def _call_range(depth_bb: float) -> float:
    if depth_bb <= 8:
        return 0.20
    if depth_bb <= 15:
        return 0.10
    return 0.05


class BaselineBot:
    def act(self, state) -> Action:
        legal = state.legal
        if state.street != 0:
            if legal.can_check:
                return Action(CHECK)
            if legal.call_amount * 4 <= state.pot:
                return Action(CALL)
            return Action(FOLD)

        stacks, bets, in_hand = state.stacks, state.bets, state.in_hand
        seat, n = state.seat, len(stacks)
        bb = state.big_blind or 1
        mine = stacks[seat] + bets[seat]
        biggest_other = max(
            (stacks[s] + bets[s] for s in range(n) if s != seat and in_hand[s]), default=0
        )
        depth = min(mine, biggest_other) / bb

        if max(bets) > bb:
            in_range = preflop_percentile(state.hole) <= _call_range(depth)
        else:
            # Unopened: fewer players left to act behind us means a wider range.
            first = state.bb_seat + 1
            my_pos = (seat - first) % n
            behind = sum(
                1 for s in range(n)
                if s != seat and in_hand[s] and stacks[s] > 0 and (s - first) % n > my_pos
            )
            in_range = preflop_percentile(state.hole) <= min(1.0, _open_range(depth) * 2 / (1 + 0.5 * behind))

        if in_range:
            return commit(legal)  # all-in where legal (pot-limit: the max raise)
        return Action(CHECK) if legal.can_check else Action(FOLD)
