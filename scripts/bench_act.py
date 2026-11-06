"""Time Agent.act the way the tournament sees it: memory parsing, event feeding,
the decision and writing memory back, over every decision of real tables.

    python scripts/bench_act.py [--tables 300] [--variant plo] [--payouts 30,20,15,12,9,7,5,2]

Runs in one process (no workers), with a fresh Agent per decision as in the
forked pool. Reports p50/p99/p99.9/max per act call with perf_counter_ns
(wall) and thread CPU time, the calls over the 1 ms limit, and the largest
memory the bot handed back. Wall-clock numbers on a busy laptop include OS
pauses, which CPU time doesn't; run it on the build box for the wall-clock
numbers that count.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.kit_bridge import CPU_NS, LATENCIES_NS, KitSeat  # noqa: E402  (puts the SDK stand-in on the path)
import adapter  # noqa: E402
from engine.field import ChenTAG, EquityCaller, Maniac, PushFold, RandomBot  # noqa: E402
from engine.rules import NLHE, VARIANT_SPEC_HELP, TableConfig, parse_variant  # noqa: E402
from engine.sng import play_sng  # noqa: E402


def bench(tables: int, config: TableConfig) -> dict:
    LATENCIES_NS.clear()
    CPU_NS.clear()
    biggest = 0
    for t in range(tables):
        ours = [KitSeat(adapter.Agent, config) for _ in range(4)]
        play_sng(ours + [ChenTAG(), PushFold(), EquityCaller(), Maniac() if t % 2 else RandomBot(t)],
                 config, seed=t)
        biggest = max([biggest] + [len(s.memory) for s in ours])
    out = {"decisions": len(LATENCIES_NS), "memory_max": biggest}
    for clock, raw in (("wall", LATENCIES_NS), ("cpu", CPU_NS)):
        xs = sorted(raw)
        at = lambda q: xs[min(len(xs) - 1, int(len(xs) * q))] / 1000  # noqa: E731
        out[clock] = {"p50": at(0.5), "p99": at(0.99), "p99.9": at(0.999), "max": xs[-1] / 1000,
                      "over_1ms": sum(x > 1_000_000 for x in xs)}
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tables", type=int, default=300)
    p.add_argument("--variant", type=parse_variant, default=NLHE, help=VARIANT_SPEC_HELP)
    p.add_argument("--payouts", default="50,30,20,0,0,0,0,0")
    args = p.parse_args(argv)
    config = TableConfig(variant=args.variant, payouts=tuple(float(x) for x in args.payouts.split(",")),
                         enforce_time=False)
    r = bench(args.tables, config)
    print(f"{args.variant.name} | payouts {args.payouts} | {r['decisions']:,} act calls | "
          f"largest memory {r['memory_max']:,} bytes")
    for clock in ("wall", "cpu"):
        c = r[clock]
        print(f"  {clock:4} p50 {c['p50']:4.0f} us   p99 {c['p99']:4.0f} us   p99.9 {c['p99.9']:4.0f} us   "
              f"max {c['max']:6.0f} us   over 1 ms: {c['over_1ms']}")


if __name__ == "__main__":
    main(sys.argv[1:])
