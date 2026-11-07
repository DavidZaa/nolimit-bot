"""The evaluation to trust: a candidate against a frozen baseline, paired, over several fields and seeds.

    python -m engine.evaluate --candidate v6 --baseline ours --tables 2000 --seeds 9101,9102 \\
        [--mixtures target,tight,loose,calling,aggressive,random,llm] [--variant plo] [--workers 8]

Every table is played twice from the same seats, deals and opponents (see
engine/compare.py): once with the candidate in the hero seat, once with the
baseline. Both arms always share the exact rules, payouts, blind schedule
and lineup. For each mixture and seed it reports points per table for both,
the paired difference with a 95% interval, the finishing distribution, and
every failure on either side: exceptions and timeouts the table saw,
illegal actions, exceptions SafeBot absorbed and illegal actions it mended.

A JSON record of the run (code revision, rules, payouts, schedule, seeds,
lineups, environment, results) goes to runs/ (or --out), enough to rerun it
exactly. Don't tune on the same seeds you confirm on; and a gain counts only
if it holds across mixtures, not in one lineup.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from engine.compare import HERO, Comparison, compare
from engine.rules import TableConfig, game_rules
from engine.run import add_table_args, parse_lineup, table_config

# Seven opponents each, so the hero takes the eighth seat. Styles, not copies of ours.
MIXTURES = {
    "target": "pushfold:2,tag:2,equity,folder,maniac",
    "tight": "tag:3,folder:2,pushfold:2",
    "loose": "maniac:2,station:2,equity:2,random",
    "calling": "station:3,equity:3,tag",
    "aggressive": "maniac:3,pushfold:3,llm4",
    "random": "random:5,tag,station",
    "llm": "llm1,llm2,llm3,llm4,pushfold,tag,equity",
}
ROOT = Path(__file__).resolve().parent.parent


def _git() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"revision": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def environment() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "cpus": os.cpu_count(), **_git()}


def summarize(c: Comparison) -> dict:
    n = c.tables
    lo, hi = c.mean_diff - c.ci95, c.mean_diff + c.ci95
    arms = {}
    for i, arm in enumerate(("candidate", "baseline")):
        total = (c.candidate_sum, c.reference_sum)[i]
        sq = (c.candidate_sq, c.reference_sq)[i]
        mean = total / n
        sd = math.sqrt(max(0.0, (sq - n * mean * mean) / (n - 1))) if n > 1 else float("inf")
        finishes = c.arm_finishes[i]
        heroes = sum(finishes) or 1
        arms[arm] = {
            "points_per_table": mean, "ci95": 1.96 * sd / math.sqrt(n) if n else float("inf"),
            "finish_share": [round(f / heroes, 4) for f in finishes][:8],
            "decisions": c.arm_decisions[i], "exceptions": c.arm_errors[i], "timeouts": c.arm_timeouts[i],
            "illegal": c.arm_illegal[i], "safebot_absorbed": c.arm_absorbed[i],
            "safebot_repaired": c.arm_repaired[i],
        }
    verdict = "better" if lo > 0 else "worse" if hi < 0 else "no detectable difference"
    return {"tables": n, "difference": c.mean_diff, "ci95": c.ci95, "verdict": verdict,
            "variance_removed": c.variance_removed, **arms}


def evaluate(candidate: str, baseline: str, mixtures: dict, seeds, tables: int, config: TableConfig,
             workers: int = 1, log=print) -> dict:
    results, pooled = [], Comparison()
    for name, opponents in mixtures.items():
        lineup = [HERO, *parse_lineup(opponents)]
        for seed in seeds:
            t0 = time.time()
            c = compare(candidate, baseline, lineup, tables, seed, config, workers)
            pooled.merge(c)
            row = {"mixture": name, "lineup": ",".join(lineup), "seed": seed, **summarize(c),
                   "seconds": round(time.time() - t0, 1)}
            results.append(row)
            log(f"{name:10} seed {seed:<6} {c.tables} tables  {candidate} "
                f"{row['candidate']['points_per_table']:6.2f}  {baseline} {row['baseline']['points_per_table']:6.2f}  "
                f"diff {c.mean_diff:+.2f} ± {c.ci95:.2f}  ({row['verdict']})  "
                f"failures {candidate}: {_failures(row['candidate'])}, {baseline}: {_failures(row['baseline'])}")
    return {"results": results, "pooled": summarize(pooled)}


def _failures(arm: dict) -> str:
    return (f"{arm['exceptions']} exc/{arm['timeouts']} slow/{arm['illegal']} illegal/"
            f"{arm['safebot_absorbed']} absorbed/{arm['safebot_repaired']} repaired")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline", default="ours", help="frozen reference (default: the submitted bot)")
    p.add_argument("--mixtures", default=",".join(MIXTURES), help=f"any of {', '.join(MIXTURES)}")
    p.add_argument("--seeds", default="9101,9102", help="comma-separated; fresh ones for confirmation")
    p.add_argument("--tables", type=int, default=2000, help="paired tables per mixture and seed")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", type=Path, default=None, help="JSON record (default runs/eval-<time>.json)")
    add_table_args(p)
    args = p.parse_args(argv)

    config = table_config(args)
    mixtures = {m: MIXTURES[m] for m in args.mixtures.split(",")}
    seeds = [int(s) for s in args.seeds.split(",")]
    rules = game_rules(config)
    print(f"{args.candidate} vs {args.baseline} | {config.variant.name} | payouts {list(rules.payouts)} | "
          f"blinds {list(rules.blind_schedule[:3])}... every {rules.hands_per_level} hands | cap {rules.hand_cap} | "
          f"{args.tables} paired tables x {len(mixtures)} mixtures x {len(seeds)} seeds")
    record = {
        "command": " ".join(["python -m engine.evaluate", *(argv if argv is not None else sys.argv[1:])]),
        "candidate": args.candidate, "baseline": args.baseline, "seeds": seeds, "tables": args.tables,
        "rules": {k: getattr(rules, k) for k in rules.__slots__},
        "enforce_time": config.enforce_time, "mixtures": mixtures, "environment": environment(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    record.update(evaluate(args.candidate, args.baseline, mixtures, seeds, args.tables, config, args.workers))
    pooled = record["pooled"]
    print(f"pooled over every mixture and seed: {pooled['tables']} tables, diff {pooled['difference']:+.2f} "
          f"± {pooled['ci95']:.2f} ({pooled['verdict']})")
    wins = sum(r["verdict"] == "better" for r in record["results"])
    losses = sum(r["verdict"] == "worse" for r in record["results"])
    print(f"cells: {wins} better, {losses} worse, {len(record['results']) - wins - losses} no detectable difference")
    out = args.out or ROOT / "runs" / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=list))
    print(f"record: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
