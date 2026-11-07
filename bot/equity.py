"""Equity of a hand against opponents' ranges, under the table's rules. Standard library only.

    equity(hole, board, rules, opponents=1, opponent_range=None, samples=64, seed=0) -> Estimate

Showdown share (ties split) of `hole` against `opponents` hands drawn from
`opponent_range` (None: any hand; otherwise a predicate on a hand, e.g. "in
the top 20% by the starting-hand table", see top_share), on boards completed
from the remaining deck. Cards never overlap: every draw comes from one deck
minus everything known.

Reproducible: the same seed gives the same answer. Exact when cheap: heads-up
against any hand, it enumerates every opponent hand and runout instead of
sampling when that's at most `exact_limit` showdowns (a hold'em river is
990; a hold'em turn 43,560 and a PLO river 123,410, so those sample unless
the caller raises the limit).

Cost is the catch. One showdown costs two or more full evaluations (8 us
for hold'em, ~25 us for PLO in CPython), so 64 samples is already ~1 ms
heads-up: too slow to run on every decision inside the 1 ms limit. Callers
must pass a small `samples` and fall back to a cheaper rule when they can't
afford it; `deadline_ns` (a perf_counter_ns value) stops sampling early and
reports how many samples it got, so the estimate's error is known.
"""

from __future__ import annotations

import math
import random
from itertools import combinations
from time import perf_counter_ns
from typing import Callable, NamedTuple

from bot.handeval import hand_value


class Estimate(NamedTuple):
    equity: float  # expected share of the pot at showdown, ties split
    samples: int  # showdowns behind it (exact enumeration: all of them)
    stderr: float  # standard error of `equity` (0 when exact)
    exact: bool


def top_share(ranks, share: float) -> Callable:
    """A range: hands in the top `share` of starting hands by `ranks` (bot.preflop.HandRanks)."""
    return lambda hand: ranks.percentile(hand) <= share


def _showdown(hole, others, board, rules) -> float:
    mine = hand_value(hole, board, rules)
    best, ties = mine, 1
    for hand in others:
        v = hand_value(hand, board, rules)
        if v > best:
            return 0.0
        if v == best:
            ties += 1
    return 1.0 / ties


def _exact(hole, board, rules, deck) -> Estimate:
    """Heads-up against any hand, at most one board card to come: every combination."""
    need = rules.board_size - len(board)
    total = count = 0.0
    for opp in combinations(deck, rules.hole_cards):
        rest = [c for c in deck if c not in opp]
        for extra in (combinations(rest, need) if need else [()]):
            total += _showdown(hole, (opp,), (*board, *extra), rules)
            count += 1
    return Estimate(total / count, int(count), 0.0, True)


def equity(hole, board, rules, opponents: int = 1, opponent_range: Callable | None = None,
           samples: int = 64, seed: int = 0, deadline_ns: int | None = None,
           max_rejections: int = 50, exact_limit: int = 1000) -> Estimate:
    if rules is None:
        raise ValueError("explicit rules required")
    known = set(hole) | set(board)
    if len(known) != len(hole) + len(board):
        raise ValueError("the same card appears twice")
    deck = [c for c in rules.deck() if c not in known]
    if len(deck) != len(rules.deck()) - len(known):
        raise ValueError("a card that isn't in this deck")
    need = rules.board_size - len(board)
    if need < 0:
        raise ValueError("the board is already longer than the rules allow")
    if opponents == 1 and opponent_range is None:
        showdowns = math.comb(len(deck), rules.hole_cards) * math.comb(len(deck) - rules.hole_cards, need)
        if showdowns <= exact_limit:
            return _exact(hole, board, rules, deck)

    rng = random.Random(seed)
    k = rules.hole_cards
    total = total_sq = 0.0
    done = 0
    for _ in range(samples):
        if deadline_ns is not None and done and perf_counter_ns() > deadline_ns:
            break
        available = deck[:]
        others = []
        for _o in range(opponents):
            hand = tuple(rng.sample(available, k))
            if opponent_range is not None:
                for _try in range(max_rejections):  # a too-narrow range keeps the last draw
                    if opponent_range(hand):
                        break
                    hand = tuple(rng.sample(available, k))
            others.append(hand)
            for c in hand:
                available.remove(c)
        runout = tuple(rng.sample(available, need))
        share = _showdown(hole, others, (*board, *runout), rules)
        total += share
        total_sq += share * share
        done += 1
    mean = total / done
    var = max(0.0, total_sq / done - mean * mean)
    return Estimate(mean, done, math.sqrt(var / done) if done > 1 else float("inf"), False)
