"""Paired comparison: does a candidate bot beat a reference bot from the same seats?

    python -m engine.compare --candidate policy --reference ours \\
        --lineup hero,pushfold:2,tag:2,equity,folder,maniac --tables 4000 --workers 8

Every table is played twice with the same seating, deals and opponents: once
with the candidate in the `hero` seats and once with the reference. Pairing
helps less than it sounds: one different decision changes every hand after
it, so two bots that play differently drift apart within a few hands
(measured: pairing removes about a fifth of the variance, ours vs the old
baseline). The report prints how much it removed; trust the interval, and
expect roughly 10,000 tables to resolve half a point.
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from engine.rules import TableConfig
from engine.run import BOTS, add_table_args, parse_lineup, seat_names, table_config
from engine.sng import play_sng
from engine.table import CLOCK_NAME

HERO = "hero"


@dataclass
class Comparison:
    tables: int = 0
    diff_sum: float = 0.0
    diff_sq: float = 0.0
    candidate_sum: float = 0.0
    reference_sum: float = 0.0
    candidate_sq: float = 0.0
    reference_sq: float = 0.0
    decisions: int = 0  # the candidate's, with its failures below
    errors: int = 0
    timeouts: int = 0
    illegal: int = 0

    def merge(self, other: "Comparison") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @property
    def mean_diff(self) -> float:
        return self.diff_sum / self.tables if self.tables else 0.0

    @property
    def ci95(self) -> float:
        n = self.tables
        if n < 2:
            return float("inf")
        var = max(0.0, (self.diff_sq - n * self.mean_diff**2) / (n - 1))
        return 1.96 * math.sqrt(var / n)

    @property
    def variance_removed(self) -> float:
        """Share of the unpaired difference's variance that pairing took out."""
        n = self.tables
        if n < 2:
            return 0.0
        def var(total, sq):
            return (sq - total * total / n) / (n - 1)
        unpaired = var(self.candidate_sum, self.candidate_sq) + var(self.reference_sum, self.reference_sq)
        paired = var(self.diff_sum, self.diff_sq)
        return 1 - paired / unpaired if unpaired > 0 else 1.0


def _hero_lineup(spec: str) -> list[str]:
    parts = [p.strip() for p in spec.split(",")]
    heroes = sum(int(p.partition(":")[2] or 1) for p in parts if p.partition(":")[0] == HERO)
    if not heroes:
        raise ValueError(f"lineup needs at least one {HERO!r} seat, e.g. {HERO},pushfold:7")
    rest = ",".join(p for p in parts if p.partition(":")[0] != HERO)
    return [HERO] * heroes + (parse_lineup(rest) if rest else [])


def _run_chunk(candidate, reference, lineup, start, count, seed, config) -> Comparison:
    out = Comparison()
    for t in range(start, start + count):
        table_seed, names = seat_names(lineup, seed, t)
        heroes = [s for s, name in enumerate(names) if name == HERO]
        points = []
        for arm, who in enumerate((candidate, reference)):
            bots = [BOTS[who if name == HERO else name](table_seed * 16 + s, config)
                    for s, name in enumerate(names)]
            res = play_sng(bots, config, seed=table_seed)
            points.append(sum(res.points[s] for s in heroes) / len(heroes))
            if arm == 0:
                st = res.stats
                out.decisions += sum(st.decisions[s] for s in heroes)
                out.errors += sum(st.errors[s] for s in heroes)
                out.timeouts += sum(st.timeouts[s] for s in heroes)
                out.illegal += sum(st.illegal[s] for s in heroes)
        diff = points[0] - points[1]
        out.tables += 1
        out.diff_sum += diff
        out.diff_sq += diff * diff
        out.candidate_sum += points[0]
        out.reference_sum += points[1]
        out.candidate_sq += points[0] ** 2
        out.reference_sq += points[1] ** 2
    return out


def compare(candidate: str, reference: str, lineup: list[str], tables: int, seed: int = 0,
            config: TableConfig = TableConfig(), workers: int = 1) -> Comparison:
    for name in (candidate, reference):
        if name not in BOTS:
            raise ValueError(f"unknown bot {name!r}; known: {', '.join(BOTS)}")
    if HERO not in lineup:
        raise ValueError(f"lineup needs at least one {HERO!r} seat")
    if len(lineup) != config.seats:
        raise ValueError(f"lineup has {len(lineup)} bots for {config.seats} seats")
    workers = max(1, min(workers, tables))
    if workers == 1:
        return _run_chunk(candidate, reference, lineup, 0, tables, seed, config)
    bounds = [tables * w // workers for w in range(workers + 1)]
    total = Comparison()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_chunk, candidate, reference, lineup, lo, hi - lo, seed, config)
                   for lo, hi in zip(bounds, bounds[1:])]
        for f in futures:
            total.merge(f.result())
    return total


def format_comparison(c: Comparison, candidate: str, reference: str, lineup, config) -> str:
    lo, hi = c.mean_diff - c.ci95, c.mean_diff + c.ci95
    verdict = ("candidate is better" if lo > 0 else "candidate is worse" if hi < 0
               else "no difference detectable at this sample size")
    width = max(len(candidate), len(reference), len("difference"))
    return "\n".join([
        f"{c.tables} paired tables | {config.variant.name} | payouts {list(config.payouts)}"
        f" | timed on {CLOCK_NAME} clock",
        f"lineup: {','.join(lineup)}",
        f"{candidate:<{width}}  {c.candidate_sum / max(1, c.tables):6.2f} pts/table",
        f"{reference:<{width}}  {c.reference_sum / max(1, c.tables):6.2f} pts/table",
        f"{'difference':<{width}}  {c.mean_diff:+6.2f} ± {c.ci95:.2f} (95%): {verdict}",
        f"pairing removed {100 * c.variance_removed:.0f}% of the variance",
        f"{candidate}: {c.decisions} decisions, {c.errors} errors, {c.timeouts} slow, {c.illegal} illegal",
    ])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--candidate", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--tables", type=int, default=2000)
    p.add_argument("--workers", type=int, default=1)
    add_table_args(p)
    p.set_defaults(lineup=f"{HERO},pushfold:2,tag:2,equity,folder,maniac")
    args = p.parse_args(argv)

    config = table_config(args)
    lineup = _hero_lineup(args.lineup)
    result = compare(args.candidate, args.reference, lineup, args.tables, args.seed, config, args.workers)
    print(format_comparison(result, args.candidate, args.reference, lineup, config))


if __name__ == "__main__":
    main(sys.argv[1:])
