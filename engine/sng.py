"""Plays one eight-handed sit-and-go and scores it the way the tournament does.

A table ends when one bot has every chip or after `max_hands`. Busted bots get
the points for their finishing place; at the hand cap the places still open
are split among the survivors by ICM on their stacks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from engine.icm import icm_equity
from engine.rules import TableConfig
from engine.table import TableStats, next_alive, play_hand


@dataclass
class SngResult:
    points: list[float]  # per seat
    finish: list[float]  # finishing place per seat (1 = winner); tied places are averaged
    hands_played: int
    reached_cap: bool
    stats: TableStats


def _prize(payouts, place: int) -> float:
    return payouts[place - 1] if place - 1 < len(payouts) else 0.0


def assign_places(seats, key_stacks, best_place: int, payouts) -> dict[int, tuple[float, float]]:
    """Rank seats that left in the same hand: bigger starting stack, better place.

    Returns seat -> (place, points). Seats with exactly equal stacks share the
    average place and the average points of the places they cover.
    PLACEHOLDER(kit): the published format says busted bots score their place and the
    cap splits the rest by ICM, but not how same-hand busts are ordered; confirm against
    the engine in the kit's sdk/ at kickoff.
    """
    ordered = sorted(seats, key=lambda s: -key_stacks[s])
    out: dict[int, tuple[float, float]] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and key_stacks[ordered[j + 1]] == key_stacks[ordered[i]]:
            j += 1
        places = range(best_place + i, best_place + j + 1)
        place = sum(places) / len(places)
        points = sum(_prize(payouts, p) for p in places) / len(places)
        for s in ordered[i : j + 1]:
            out[s] = (place, points)
        i = j + 1
    return out


def play_sng(bots, config: TableConfig, seed: int, on_hand=None) -> SngResult:
    """`on_hand(hand_number, stacks_before, outcome)` runs after every hand (for viewers)."""
    n = config.seats
    rng = random.Random(seed)
    stats = TableStats(n)
    for seat, bot in enumerate(bots):
        hook = getattr(bot, "new_table", None)
        if hook is not None:
            try:
                hook(seat, n)
            except Exception:
                pass

    stacks = [config.starting_stack] * n
    points = [0.0] * n
    finish = [0.0] * n
    worst_open_place = n
    button = rng.randrange(n)
    hands = 0

    while hands < config.max_hands and sum(1 for s in stacks if s > 0) > 1:
        hands += 1
        before = stacks
        out = play_hand(bots, stacks, button, hands, config, rng, stats=stats)
        stacks = out.stacks
        if on_hand is not None:
            on_hand(hands, before, out)
        busted =[s for s in range(n) if before[s] > 0 and stacks[s] == 0]
        if busted:
            best = worst_open_place - len(busted) + 1
            for s, (place, pts) in assign_places(busted, before, best, config.payouts).items():
                finish[s], points[s] = place, pts
            worst_open_place = best - 1
        if sum(1 for s in stacks if s > 0) > 1:
            button = next_alive(stacks, button)

    survivors = [s for s in range(n) if stacks[s] > 0]
    reached_cap = len(survivors) > 1
    if reached_cap:
        # ICM over the open places 1..len(survivors).
        equity = icm_equity([stacks[s] for s in survivors], config.payouts[: len(survivors)])
        for s, e in zip(survivors, equity):
            points[s] = e
        for s, (place, _) in assign_places(survivors, stacks, 1, config.payouts).items():
            finish[s] = place
    else:
        winner = survivors[0]
        finish[winner], points[winner] = 1.0, _prize(config.payouts, 1)

    return SngResult(points, finish, hands, reached_cap, stats)
