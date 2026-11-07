"""How close is the bubble factor to deciding all-in calls by expected tournament points directly?

    python scripts/icm_check.py [--spots 4000] [--seed 1]

The policy tightens all-in calls by the bubble factor (bot/icm.py): points
lost by losing a flip over points won by winning it, for an even flip with
no dead money. The exact way to decide a call is by expected points:

    call iff  equity * points(win) + (1 - equity) * points(lose)  >  points(fold)

which needs equity >= (points(fold) - points(lose)) / (points(win) - points(lose)).
The bubble factor's version of that threshold is BF * risk / (BF * risk + risk + dead).
This samples random stacks (2-8 players left, blinds in the pot), computes
both thresholds, and reports how far apart they are and how often the two
would decide differently for a hand of random equity. Busting pays the
worst place still open, as the tournament scores it.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.icm import bubble_factor, seat_equity  # noqa: E402

PAYOUTS = {
    "top3": (50, 30, 20, 0, 0, 0, 0, 0),
    "flat": (30, 20, 15, 12, 9, 7, 5, 2),
    "top2": (65, 35, 0, 0, 0, 0, 0, 0),
    "wta": (100, 0, 0, 0, 0, 0, 0, 0),
}


def _points(stacks, me, payouts, alive_before):
    if stacks[me] <= 0:
        return payouts[alive_before - 1] if alive_before - 1 < len(payouts) else 0.0
    return seat_equity(stacks, me, payouts)


def compare(payouts, spots: int = 4000, seed: int = 1) -> dict:
    rng = random.Random(seed)
    gaps, flips = [], 0
    while len(gaps) < spots:
        alive = rng.randint(2, 8)
        cuts = sorted(rng.sample(range(1, 16000), alive - 1))
        stacks = [b - a for a, b in zip([0, *cuts], [*cuts, 16000])] + [0] * (8 - alive)
        me, opp = 0, 1
        bb = rng.choice((100, 200, 400))
        dead = bb + bb // 2  # the blinds, posted by a third player
        donor = next((s for s in range(2, 8) if stacks[s] > dead), None)
        if donor is None:
            continue
        risk = min(stacks[me], stacks[opp])
        fold, win, lose = list(stacks), list(stacks), list(stacks)
        for st in (fold, win, lose):
            st[donor] -= dead
        fold[opp] += dead
        win[me] += risk + dead
        win[opp] -= risk
        lose[me] -= risk
        lose[opp] += risk + dead
        p_fold, p_win, p_lose = (_points(x, me, payouts, alive) for x in (fold, win, lose))
        if p_win == p_lose:
            continue
        direct = (p_fold - p_lose) / (p_win - p_lose)
        bf = bubble_factor(stacks, me, opp, payouts)
        approx = bf * risk / (bf * risk + risk + dead)
        gaps.append(approx - direct)
        e = rng.random()
        flips += (e > approx) != (e > direct)
    absolute = sorted(abs(g) for g in gaps)
    return {"spots": len(gaps), "mean_abs": statistics.mean(absolute), "p90": absolute[int(0.9 * len(absolute))],
            "max": absolute[-1], "bias": statistics.mean(gaps), "flips": flips / len(gaps)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--spots", type=int, default=4000)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)
    print(f"required call equity, bubble factor vs direct expected points ({args.spots} spots, seed {args.seed})")
    for name, payouts in PAYOUTS.items():
        r = compare(payouts, args.spots, args.seed)
        print(f"  {name:5} mean |gap| {r['mean_abs']:.3f}  p90 {r['p90']:.3f}  max {r['max']:.3f}  "
              f"bias {r['bias']:+.3f}  decisions flipped {r['flips']:.1%}")


if __name__ == "__main__":
    main(sys.argv[1:])
