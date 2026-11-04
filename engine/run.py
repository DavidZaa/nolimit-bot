"""Run many sit-and-gos and report how each bot type scores.

    python -m engine.run --tables 1000 --lineup baseline:4,random:2,station,maniac

Every table reseats the lineup at random and builds fresh bots (state resets
per table, as in the tournament). Results are average points per table per
bot, with a 95% confidence interval, plus decision time (CPU time, so a busy
machine doesn't inflate it) and how often the tournament would have
auto-folded each bot.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from bot.baseline import BaselineBot
from bot.pushfold import PushFoldBot
from bot.safety import SafeBot
from engine.field import CallingStation, ChenTAG, EquityCaller, FolderBot, Maniac, PushFold, RandomBot
from engine.latency import LatencyHistogram
from engine.rules import PLACEHOLDER_PAYOUTS, SCHEDULES, VARIANTS, TableConfig
from engine.sng import play_sng

BOTS = {
    "ours": lambda seed: SafeBot(PushFoldBot()),  # what adapter.build_bot() submits
    "baseline": lambda seed: SafeBot(BaselineBot()),
    "baseline_raw": lambda seed: BaselineBot(),
    "random": lambda seed: RandomBot(seed),
    "station": lambda seed: CallingStation(),
    "maniac": lambda seed: Maniac(),
    "folder": lambda seed: FolderBot(),
    "tag": lambda seed: ChenTAG(),
    "equity": lambda seed: EquityCaller(),
    "pushfold": lambda seed: PushFold(),
}


def parse_lineup(spec: str) -> list[str]:
    """'baseline:3,random' -> ['baseline', 'baseline', 'baseline', 'random']"""
    lineup = []
    for part in spec.split(","):
        name, _, count = part.strip().partition(":")
        if name not in BOTS:
            raise ValueError(f"unknown bot {name!r}; known: {', '.join(BOTS)}")
        lineup.extend([name] * (int(count) if count else 1))
    return lineup


@dataclass
class BotSummary:
    instances: int = 0  # bot-tables played
    points_sum: float = 0.0
    points_sq: float = 0.0
    wins: int = 0
    decisions: int = 0
    errors: int = 0
    timeouts: int = 0
    illegal: int = 0
    latency: LatencyHistogram = field(default_factory=LatencyHistogram)

    def merge(self, other: "BotSummary") -> None:
        self.instances += other.instances
        self.points_sum += other.points_sum
        self.points_sq += other.points_sq
        self.wins += other.wins
        self.decisions += other.decisions
        self.errors += other.errors
        self.timeouts += other.timeouts
        self.illegal += other.illegal
        self.latency.merge(other.latency)

    @property
    def mean(self) -> float:
        return self.points_sum / self.instances if self.instances else 0.0

    @property
    def ci95(self) -> float:
        n = self.instances
        if n < 2:
            return float("inf")
        var = max(0.0, (self.points_sq - n * self.mean**2) / (n - 1))
        return 1.96 * math.sqrt(var / n)


def seat_table(lineup, seed: int, table: int) -> tuple[int, list[str], list]:
    """(table seed, bot name per seat, fresh bots) for table number `table` of a run.

    The hand viewer seats tables through this too, so any table from a run
    can be replayed card for card.
    """
    table_seed = seed * 1_000_003 + table
    names = list(lineup)
    random.Random(table_seed).shuffle(names)
    return table_seed, names, [BOTS[name](table_seed * 16 + s) for s, name in enumerate(names)]


def _run_chunk(lineup, start: int, count: int, seed: int, config: TableConfig) -> dict[str, BotSummary]:
    out = {name: BotSummary() for name in lineup}
    for t in range(start, start + count):
        table_seed, names, bots = seat_table(lineup, seed, t)
        res = play_sng(bots, config, seed=table_seed)
        st = res.stats
        for s, name in enumerate(names):
            summary = out[name]
            summary.instances += 1
            summary.points_sum += res.points[s]
            summary.points_sq += res.points[s] ** 2
            summary.wins += res.finish[s] == 1
            summary.decisions += st.decisions[s]
            summary.errors += st.errors[s]
            summary.timeouts += st.timeouts[s]
            summary.illegal += st.illegal[s]
            summary.latency.merge(st.latency[s])
    return out


def run_tables(
    lineup: list[str],
    tables: int,
    seed: int = 0,
    config: TableConfig = TableConfig(),
    workers: int = 1,
) -> dict[str, BotSummary]:
    if len(lineup) != config.seats:
        raise ValueError(f"lineup has {len(lineup)} bots for {config.seats} seats")
    workers = max(1, min(workers, tables))
    if workers == 1:
        return _run_chunk(lineup, 0, tables, seed, config)

    # Contiguous chunks keep every table's seed identical to a serial run.
    bounds = [tables * w // workers for w in range(workers + 1)]
    totals = {name: BotSummary() for name in lineup}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_chunk, lineup, lo, hi - lo, seed, config)
            for lo, hi in zip(bounds, bounds[1:])
        ]
        for f in futures:
            for name, part in f.result().items():
                totals[name].merge(part)
    return totals


def format_report(summaries: dict[str, BotSummary], tables: int, config: TableConfig) -> str:
    lines = [
        f"{tables} tables | {config.variant.name} | payouts {list(config.payouts)}",
        f"{'bot':<14}{'pts/table':>10}{'±95%':>8}{'win%':>7}{'decisions':>11}"
        f"{'p50µs':>7}{'p99.9µs':>9}{'maxµs':>8}{'err':>6}{'slow':>6}{'illegal':>8}",
    ]
    ranked = sorted(summaries.items(), key=lambda kv: -kv[1].mean)
    for name, s in ranked:
        lines.append(
            f"{name:<14}{s.mean:>10.2f}{s.ci95:>8.2f}{100 * s.wins / max(1, s.instances):>7.1f}"
            f"{s.decisions:>11}{s.latency.percentile_us(0.5):>7}{s.latency.percentile_us(0.999):>9}"
            f"{s.latency.max_ns // 1000:>8}{s.errors:>6}{s.timeouts:>6}{s.illegal:>8}"
        )
    return "\n".join(lines)


def add_table_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by every command that plays tables."""
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lineup", default="baseline:4,random:2,station,maniac")
    p.add_argument("--variant", choices=sorted(VARIANTS), default="nlhe")
    p.add_argument("--schedule", choices=sorted(SCHEDULES), default="doubling")
    p.add_argument("--payouts", default=",".join(str(x) for x in PLACEHOLDER_PAYOUTS),
                   help="points by finishing place, 1st first")
    p.add_argument("--max-hands", type=int, default=200)
    p.add_argument("--no-enforce-time", action="store_true",
                   help="don't auto-fold decisions that take more than 1 ms of CPU")


def table_config(args: argparse.Namespace) -> TableConfig:
    return TableConfig(
        variant=VARIANTS[args.variant],
        schedule=SCHEDULES[args.schedule],
        payouts=tuple(float(x) for x in args.payouts.split(",")),
        max_hands=args.max_hands,
        enforce_time=not args.no_enforce_time,
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tables", type=int, default=1000)
    p.add_argument("--workers", type=int, default=1)
    add_table_args(p)
    args = p.parse_args(argv)

    config = table_config(args)
    summaries = run_tables(parse_lineup(args.lineup), args.tables, args.seed, config, args.workers)
    print(format_report(summaries, args.tables, config))


if __name__ == "__main__":
    main(sys.argv[1:])
