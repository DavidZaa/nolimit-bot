"""Tournament equity for the submitted bot: one seat's ICM, fast enough to call mid-decision.

Points go by finishing place, so chips aren't worth their face value: losing
an all-in usually costs more points than winning the same all-in gains. The
bubble factor measures that ratio from the real payouts and stacks, and the
policy tightens all-in calls and shoves by it. Winner-take-all makes chips
exactly proportional to points, so the factor is 1 and nothing changes.

Equity is Malmuth-Harville (the model the tournament itself uses to split
points at the hand cap), but only for one seat, and only as deep as the
payouts differ: everything is measured relative to the last open place, so
50/30/20 with eight players left only looks three places deep.

Payouts are always the table's own (GameRules.payouts, from state.config);
there is no default here, so nothing can fall back to 50/30/20 unnoticed.

The bubble factor is an approximation of deciding by expected points
directly: it prices an even flip with no dead money. scripts/icm_check.py
measures it against the exact expected-points threshold for an all-in call
with blinds in the pot: mean gap 0.5-1.2 equity points under top-2, top-3
and flat payouts, 0.4-1.2% of decisions flipped for hands of random equity,
exact under winner-take-all, with rare outliers (up to ~0.55) where the
factor hits its clamp. How the policy turns the factor into a narrower
range is a further heuristic, measured by A/B (the icm feature) instead.
"""

from __future__ import annotations

from functools import lru_cache

# Below 1 is real (a big stack calling a short one on the bubble gains from
# busting it), but never so far that a range divided by it blows up.
MIN_BUBBLE_FACTOR, MAX_BUBBLE_FACTOR = 0.25, 4.0
MAX_SEATS = 10


def _tables(k: int):
    """For k other players: subsets grouped by size, and for each subset who's not in it."""
    by_size = [[] for _ in range(k + 1)]
    free = []
    for mask in range(1 << k):
        by_size[bin(mask).count("1")].append(mask)
        free.append(tuple((1 << j, j) for j in range(k) if not mask >> j & 1))
    return by_size, free


_TABLES = [_tables(k) for k in range(MAX_SEATS)]


def seat_equity(totals, seat: int, payouts) -> float:
    """Expected points for `seat` given every seat's chips (0 for busted seats). A seat with
    no chips gets 0 here: the points of the place it busted in are scored separately."""
    me = totals[seat]
    if me <= 0:
        return 0.0
    others = [t for s, t in enumerate(totals) if s != seat and t > 0]
    n = len(others) + 1
    prizes = [payouts[k] if k < len(payouts) else 0.0 for k in range(n)]
    floor = prizes[-1]  # every survivor finishes somewhere in places 1..n
    gains = [p - floor for p in prizes]
    depth = n
    while depth and gains[depth - 1] == 0:
        depth -= 1
    total = me + sum(others)
    equity = floor + (gains[0] * me / total if depth else 0.0)
    if depth <= 1:
        return equity

    # prob[mask]: chance the players in `mask` (bits index `others`) took the
    # top places in some order; left[mask]: chips still in play after them.
    k = len(others)
    by_size, free = _TABLES[k]
    left = [total] * (1 << k)
    for size in range(1, depth):
        for mask in by_size[size]:
            low = mask & -mask
            left[mask] = left[mask ^ low] - others[low.bit_length() - 1]
    prob = [0.0] * (1 << k)
    prob[0] = 1.0
    for place in range(1, depth):
        for mask in by_size[place - 1]:
            f = prob[mask] / left[mask]
            for bit, j in free[mask]:
                prob[mask | bit] += f * others[j]
        g = gains[place]
        if g:
            equity += g * me * sum(prob[m] / left[m] for m in by_size[place])
    return equity


def bubble_factor(totals, me: int, opp: int, payouts) -> float:
    """Points lost by losing an all-in to `opp`, over points won by winning it.

    Both sides of the flip are the smaller of the two stacks; 1.0 means chips
    and points are interchangeable, 2.0 that a loss hurts twice as much as a
    win helps.
    """
    at_risk = min(totals[me], totals[opp])
    if at_risk <= 0:
        return 1.0
    base = seat_equity(totals, me, payouts)
    won = list(totals)
    won[me] += at_risk
    won[opp] -= at_risk
    lost = list(totals)
    lost[me] -= at_risk
    lost[opp] += at_risk
    if lost[me] > 0:
        after_loss = seat_equity(lost, me, payouts)
    else:  # busting now means the worst place still open
        alive = sum(1 for t in totals if t > 0)
        after_loss = payouts[alive - 1] if alive - 1 < len(payouts) else 0.0
    gain = seat_equity(won, me, payouts) - base
    loss = base - after_loss
    if gain <= 0:
        return MAX_BUBBLE_FACTOR
    return min(MAX_BUBBLE_FACTOR, max(MIN_BUBBLE_FACTOR, loss / gain))


@lru_cache(maxsize=4096)
def cached_bubble_factor(totals: tuple, me: int, opp: int, payouts: tuple) -> float:
    """bubble_factor memoised on (stacks, seats, payouts). Tables can draw their own
    payouts, so the payouts are part of the key; losing the cache only costs time."""
    return bubble_factor(totals, me, opp, payouts)
