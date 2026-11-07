"""Rebuild the bot's starting-hand ranking for whatever variant is announced.

VARIANT DAY (kickoff), in order:

  1. Register the rules: add the variant to VARIANTS in bot/rules.py under the
                        exact ID the SDK reports in state.config.variant (payouts
                        need no step: every table's config carries its own).
  2. Rank the hands:    python -m gen.hand_ranks --variant <preset or spec> --id <SDK id>
                        (presets: nlhe, short_deck, plo; anything else as a spec,
                        e.g. "base=nlhe,hole=3" or "base=plo,hole=5", see --help).
                        It writes bot/hand_tables/<id>.py in about 4 minutes,
                        stamped with the card rules it was simulated under; the
                        bot uses it only for tables whose rules match the stamp.
                        PLO and short deck are committed. Plain hold'em has no
                        table on purpose (Chen measured no worse).
  3. Check it:          .venv/bin/pytest -q
  4. See it play:       python -m engine.run --variant <same> --tables 300 --workers 8 \
                            --lineup ours:2,pushfold:2,tag:2,equity,maniac
  5. Release:           scripts/pack.sh --variant <SDK id> (refuses unregistered IDs).

How it ranks: every starting-hand class (suits interchangeable, see
bot/preflop.py) plays against one random hand on random boards, scored with
the engine's evaluator, so any deck, hole-card count, Omaha rule or category
order works. Every class faces the same pool of opponents and boards, and
each opponent's hand is scored once for the whole run: that halves the work
and takes most of the noise out of the ranking, which is all the bot uses.
The number of trials is set to fit --seconds on this machine.

Five-card Omaha and bigger have too many classes to simulate in minutes, so
they're ranked through sub-hands instead (see bot/preflop.py): the
four-card table (cached in gen/tables/plo.py when the rules match, else
simulated first), plus the spread of best-sub-hand scores over sampled
hands to turn that into a share of five-card hands.
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
from engine.rules import NLHE, PLO, VARIANT_SPEC_HELP, parse_variant

MAX_SIMULATED_HANDS = 500_000  # PLO's 270,725 starting hands simulate fine; five-card Omaha doesn't

TABLES = Path(__file__).resolve().parent.parent / "bot" / "hand_tables"
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
    k, b = variant.hole_cards, variant.board_size
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
    d, k, need = len(variant.deck()), variant.hole_cards, variant.hole_cards + variant.board_size
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
        budget = max(1.0, seconds - (time.perf_counter() - started)) * workers * 0.6
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


def card_rules_line(variant) -> str:
    """The CARD_RULES stamp bot.preflop.ranks_for checks a table against."""
    rules = {"ranks": tuple(variant.ranks), "hole_cards": variant.hole_cards,
             "hole_cards_used": variant.hole_cards_used, "preflop_board": variant.preflop_board,
             "board_schedule": tuple(variant.board_schedule), "category_order": tuple(variant.category_order),
             "ace_low_straight": variant.ace_low_straight}
    return f"CARD_RULES = {rules!r}"


def write_table(path: Path, variant, percentiles: dict, trials: int, variant_id: str | None = None) -> None:
    lines = [
        f'"""Starting-hand percentiles for {variant.name}, written by gen/hand_ranks.py.',
        "",
        f"{len(percentiles):,} classes, {trials:,} trials each against random hands. Regenerate,",
        'don\'t edit: python -m gen.hand_ranks --variant <variant> --id <SDK id>',
        '"""',
        "",
        f"VARIANT = {(variant_id or variant.name)!r}",
        f"HOLE_CARDS = {variant.hole_cards}",
        card_rules_line(variant),
        "PERCENTILES = {",
    ]
    for key, p in sorted(percentiles.items(), key=lambda kv: kv[1]):
        lines.append(f"    {key!r}: {p:.5f},")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


def _same_rules(a, b) -> bool:
    return (a.ranks, a.category_order, a.hole_cards_used) == (b.ranks, b.category_order, b.hole_cards_used)


def rank_by_subsets(variant, seconds=240, workers=None, seed=0, samples=60_000, log=print):
    """(sub-hand percentiles, sub-hand size, quantiles) for a variant with too many classes."""
    import dataclasses

    sub = 4 if variant.hole_cards_used else variant.hole_cards - 1
    base = dataclasses.replace(variant, name=f"{variant.name} [{sub}-card sub-hands]", hole_cards=sub,
                               preflop_board=0, board_schedule=(variant.board_size,))
    cached = TABLES / "plo.py"
    if sub == 4 and _same_rules(base, PLO) and cached.exists():
        namespace: dict = {}
        exec(compile(cached.read_text(), str(cached), "exec"), namespace)
        percentiles = namespace["PERCENTILES"]
        log(f"{variant.name}: using the cached four-card table ({len(percentiles):,} classes)")
    else:
        percentiles, _ = rank_hands(base, seconds, workers, seed, log=log)
    rng = random.Random(seed)
    deck = variant.deck()
    scores = sorted(
        min(percentiles[hand_class(h)] for h in combinations(rng.sample(deck, variant.hole_cards), sub))
        for _ in range(samples)
    )
    quantiles = [scores[i * len(scores) // 1000] for i in range(1000)]
    log(f"rescaled against {samples:,} sampled {variant.hole_cards}-card hands")
    return percentiles, sub, quantiles


def write_subset_table(path: Path, variant, percentiles: dict, sub: int, quantiles,
                       variant_id: str | None = None) -> None:
    lines = [
        f'"""Starting-hand ranking for {variant.name}, written by gen/hand_ranks.py.',
        "",
        f"Too many classes to simulate, so hands are judged by their best {sub}-card",
        f"sub-hand, rescaled against sampled {variant.hole_cards}-card hands. Regenerate, don't edit.",
        '"""',
        "",
        f"VARIANT = {(variant_id or variant.name)!r}",
        f"HOLE_CARDS = {variant.hole_cards}",
        card_rules_line(variant),
        f"SUBSET = {sub}",
        f"QUANTILES = {[round(q, 5) for q in quantiles]!r}",
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
    p.add_argument("--id", default=None,
                   help="the SDK's variant ID (state.config.variant); names bot/hand_tables/<id>.py")
    p.add_argument("--out", type=Path, default=None, help="write here instead of bot/hand_tables/<id>.py")
    args = p.parse_args(argv)

    v = args.variant
    variant_id = args.id or v.name
    if args.out is None and not variant_id.isidentifier():
        p.error(f"{variant_id!r} can't name a table module: pass --id with the SDK's variant ID")
    out = args.out or TABLES / f"{variant_id}.py"
    if math.comb(len(v.deck()), v.hole_cards) > MAX_SIMULATED_HANDS:
        percentiles, sub, quantiles = rank_by_subsets(v, args.seconds, args.workers, args.seed)
        write_subset_table(out, v, percentiles, sub, quantiles, variant_id)
    else:
        percentiles, trials = rank_hands(v, args.seconds, args.workers, args.seed, args.trials)
        write_table(out, v, percentiles, trials, variant_id)
    args.out = out
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
