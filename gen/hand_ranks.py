"""Rebuild the bot's starting-hand ranking for whatever variant is announced.

VARIANT DAY (kickoff), in order:

  1. Set the points:    PAYOUTS in bot/icm.py, 1st place first, one entry per seat.
  2. Rank the hands:    python -m gen.hand_ranks --variant <preset or spec>
                        (presets: nlhe, short_deck, plo; anything else as a spec,
                        e.g. "base=nlhe,hole=3" or "base=plo,hole=5", see --help).
                        Cached tables for the presets are in gen/tables/, so
                        `cp gen/tables/plo.py bot/hand_ranks_table.py` is instant.
                        Plain hold'em: skip this step, the bot already uses Chen.
  3. Check it:          .venv/bin/pytest -q
  4. See it play:       python -m engine.run --variant <same> --tables 300 --workers 8 \
                            --lineup ours:2,pushfold:2,tag:2,equity,maniac
  5. Submit build_bot() through adapter.py.

How it ranks: every starting-hand class (suits interchangeable, see
bot/preflop.py) plays against one random hand on random boards, scored with
the engine's evaluator, so any deck, hole-card count, Omaha rule or category
order works. Every class faces the same pool of opponents and boards, and
each opponent's hand is scored once for the whole run: that halves the work
and takes most of the noise out of the ranking, which is all the bot uses.
The number of trials is set to fit --seconds on this machine.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

from bot.preflop import hand_class
from engine.handeval import best_hand
from engine.rules import NLHE, VARIANT_SPEC_HELP, parse_variant

OUT = Path(__file__).resolve().parent.parent / "bot" / "hand_ranks_table.py"
MIN_TRIALS, MAX_TRIALS = 150, 20_000


def enumerate_classes(variant) -> dict:
    """class -> (number of concrete hands in it, one concrete hand)."""
    classes: dict = {}
    for hole in combinations(variant.deck(), variant.hole_cards):
        key = hand_class(hole)
        seen = classes.get(key)
        classes[key] = (1, hole) if seen is None else (seen[0] + 1, seen[1])
    return classes


def _scenarios(variant, count: int, seed: int):
    """(cards used, opponent's score, board) for `count` random opponents and boards."""
    rng = random.Random(seed)
    deck = variant.deck()
    k, b = variant.hole_cards, sum(variant.board_schedule)
    out = []
    for _ in range(count):
        cards = rng.sample(deck, k + b)
        opp, board = tuple(cards[:k]), tuple(cards[k:])
        out.append((frozenset(cards), best_hand(opp, board, variant), board))
    return out


def _equities(spec: str, hands, trials: int, pool: int, seed: int) -> list[float]:
    variant = parse_variant(spec)
    scenarios = _scenarios(variant, pool, seed)
    result = []
    for hole in hands:
        mine = frozenset(hole)
        won = used = 0.0
        for cards, opp, board in scenarios:
            if not mine.isdisjoint(cards):
                continue
            ours = best_hand(hole, board, variant)
            won += 1.0 if ours > opp else 0.5 if ours == opp else 0.0
            used += 1
            if used >= trials:
                break
        result.append(won / used if used else 0.5)
    return result


def _clean_share(variant) -> float:
    """Chance a random scenario shares no card with a given starting hand."""
    d, k, need = len(variant.deck()), variant.hole_cards, variant.hole_cards + sum(variant.board_schedule)
    return math.comb(d - k, need) / math.comb(d, need)


def rank_hands(variant, seconds: float = 240, workers: int | None = None, seed: int = 0,
               trials: int | None = None, log=print) -> tuple[dict, int]:
    """(class -> percentile, trials per class) for `variant`."""
    workers = workers or os.cpu_count() or 1
    started = time.perf_counter()
    classes = enumerate_classes(variant)
    log(f"{variant.name}: {len(classes):,} starting-hand classes "
        f"({time.perf_counter() - started:.1f}s to enumerate)")

    if trials is None:
        probe = _scenarios(variant, 60, seed + 1)
        sample = [hole for _, hole in list(classes.values())[:60]]
        t0 = time.perf_counter()
        for hole, (_, _, board) in zip(sample, probe):
            best_hand(hole, board, variant)
        per_eval = (time.perf_counter() - t0) / len(sample)
        budget = max(1.0, seconds - (time.perf_counter() - started)) * workers * 0.8
        trials = int(min(MAX_TRIALS, max(MIN_TRIALS, budget / (len(classes) * per_eval))))
    pool = int(trials / _clean_share(variant) * 1.25) + 50
    log(f"{trials:,} trials per class against a shared pool of {pool:,} opponents and boards, "
        f"{workers} workers")

    keys = list(classes)
    hands = [classes[k][1] for k in keys]
    bounds = [len(keys) * w // workers for w in range(workers + 1)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = [ex.submit(_equities, variant_spec(variant), hands[lo:hi], trials, pool, seed)
                 for lo, hi in zip(bounds, bounds[1:])]
        equities = [e for f in parts for e in f.result()]

    order = sorted(range(len(keys)), key=lambda i: -equities[i])
    total = sum(n for n, _ in classes.values())
    better = 0
    percentiles = {}
    for i in order:
        n = classes[keys[i]][0]
        percentiles[keys[i]] = (better + n / 2) / total
        better += n
    log(f"done in {time.perf_counter() - started:.0f}s; best class "
        f"{_show(keys[order[0]])} ({equities[order[0]]:.3f} vs a random hand), worst "
        f"{_show(keys[order[-1]])} ({equities[order[-1]]:.3f})")
    return percentiles, trials


def variant_spec(variant) -> str:
    """The string parse_variant turns back into this variant."""
    return variant.name


def _show(key) -> str:
    ranks = "23456789TJQKA"
    return " ".join("".join(ranks[r] for r in group) + ("s" if len(group) > 1 else "")
                    for group in key)


def write_table(path: Path, variant, percentiles: dict, trials: int) -> None:
    lines = [
        f'"""Starting-hand percentiles for {variant.name}, written by gen/hand_ranks.py.',
        "",
        f"{len(percentiles):,} classes, {trials:,} trials each against random hands. Regenerate,",
        'don\'t edit: python -m gen.hand_ranks --variant <variant>',
        '"""',
        "",
        f"VARIANT = {variant.name!r}",
        f"HOLE_CARDS = {variant.hole_cards}",
        "PERCENTILES = {",
    ]
    for key, p in sorted(percentiles.items(), key=lambda kv: kv[1]):
        lines.append(f"    {key!r}: {p:.5f},")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", type=parse_variant, default=NLHE, help=VARIANT_SPEC_HELP)
    p.add_argument("--seconds", type=float, default=240, help="time budget (default 4 minutes)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--trials", type=int, default=None, help="fix trials per class instead of a budget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)

    percentiles, trials = rank_hands(args.variant, args.seconds, args.workers, args.seed, args.trials)
    write_table(args.out, args.variant, percentiles, trials)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
