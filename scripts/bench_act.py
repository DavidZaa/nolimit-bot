"""Time Agent.act the way the tournament sees it, and smoke-test the bot doing it.

    python scripts/bench_act.py [--tables 300] [--variant plo] [--payouts 30,20,15,12,9,7,5,2]
                                [--agent-dir dist/agent] [--strict]

Every call is the full submitted path: memory decoding, event processing,
the policy, memory encoding. Four seats run the agent at each table against
four field bots, with a fresh Agent per decision as in the forked pool.
With --agent-dir it imports that built package (agent.py + bot/) instead of
the source tree; its bot/ goes first on the path, so even the simulator runs
on the packaged copy.

Reported: the sample size; p50, p99, p99.9 and max of wall time
(perf_counter_ns) and thread CPU time, with distribution-free 95% intervals
for the tail percentiles; calls over the decision limit; tables where a seat
would have drained the per-table time bank; the largest memory returned;
and every exception, illegal action, SafeBot fallback and RULES ERROR.
--strict exits non-zero on any of those, or on a drained bank.

Wall time on a busy laptop includes OS pauses that CPU time doesn't; run it
on the build box for numbers that mean anything about production, and even
those aren't proof of production timing.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _quantile_ci(xs, q: float) -> tuple[float, float]:
    """95% distribution-free interval for the q-quantile of sorted `xs` (order statistics)."""
    n = len(xs)
    half = 1.96 * math.sqrt(n * q * (1 - q))
    lo, hi = max(0, int(math.floor(n * q - half))), min(n - 1, int(math.ceil(n * q + half)))
    return xs[lo], xs[hi]


def summarize(raw_ns, limit_ns: int) -> dict:
    xs = sorted(raw_ns)
    at = lambda q: xs[min(len(xs) - 1, int(len(xs) * q))]  # noqa: E731
    out = {"n": len(xs), "max": xs[-1], "over_limit": sum(x > limit_ns for x in xs)}
    for name, q in (("p50", 0.5), ("p99", 0.99), ("p99.9", 0.999)):
        out[name] = at(q)
        out[name + "_ci"] = _quantile_ci(xs, q)
    return out


def bench(tables: int, config, agent_cls, bank_ns: int) -> dict:
    from engine.field import ChenTAG, EquityCaller, Maniac, PushFold, RandomBot
    from engine.kit_bridge import CPU_NS, LATENCIES_NS, KitSeat
    from engine.sng import play_sng

    LATENCIES_NS.clear()
    CPU_NS.clear()
    biggest, drained, points, errors, illegal = 0, 0, [], 0, 0
    limit = config.time_limit_ns
    for t in range(tables):
        ours = [KitSeat(agent_cls, config) for _ in range(4)]
        res = play_sng(ours + [ChenTAG(), PushFold(), EquityCaller(), Maniac() if t % 2 else RandomBot(t)],
                       config, seed=t)
        biggest = max([biggest] + [len(s.memory) for s in ours])
        for seat, s in enumerate(ours):
            # The bank pays for every overrun at this table (CPU time; wall adds OS noise).
            drained += sum(max(0, c - limit) for c in s.cpu_ns) > bank_ns
            points.append(res.points[seat])
            errors += res.stats.errors[seat]
            illegal += res.stats.illegal[seat]
    return {"decisions": len(LATENCIES_NS), "memory_max": biggest, "tables": tables,
            "wall": summarize(LATENCIES_NS, limit), "cpu": summarize(CPU_NS, limit),
            "drained_banks": drained, "points_per_seat": sum(points) / len(points),
            "exceptions": errors, "illegal": illegal}


def load_agent(agent_dir: str | None):
    """(Agent class, module): the built package's agent.py, or the source tree's adapter.py."""
    if agent_dir:
        sys.path.insert(0, os.path.abspath(agent_dir))  # its bot/ shadows the source tree's
        import agent as module
    else:
        import adapter as module
    return module.Agent, module


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tables", type=int, default=300)
    p.add_argument("--variant", default="nlhe")
    p.add_argument("--payouts", default="50,30,20,0,0,0,0,0")
    p.add_argument("--agent-dir", default=None, help="a built package (scripts/pack.sh output) to import")
    p.add_argument("--decision-ms", type=float, default=1.0)
    p.add_argument("--bank-ms", type=float, default=100.0)
    p.add_argument("--strict", action="store_true", help="exit 1 on any failure or drained bank")
    args = p.parse_args(argv)

    if args.agent_dir:
        sys.path.insert(0, os.path.abspath(args.agent_dir))
    sys.path.insert(1 if args.agent_dir else 0, ROOT)
    import engine.kit_bridge  # noqa: F401  (puts the stand-in SDK on the path if the real one is absent)
    from engine.rules import TableConfig, parse_variant

    agent_cls, module = load_agent(args.agent_dir)
    import bot
    config = TableConfig(variant=parse_variant(args.variant), payouts=tuple(float(x) for x in args.payouts.split(",")),
                         enforce_time=False, time_limit_ns=int(args.decision_ms * 1e6))
    r = bench(args.tables, config, agent_cls, int(args.bank_ms * 1e6))
    rules_errors = sorted(getattr(module, "_reported", ()))
    source = args.agent_dir or "source tree (adapter.py)"
    print(f"agent: {module.__file__} (bot/ from {os.path.dirname(bot.__file__)})")
    print(f"{args.variant} | payouts {args.payouts} | {r['tables']} tables, {r['decisions']:,} act calls "
          f"| {platform.python_version()} on {platform.platform()} | {source}")
    for clock in ("wall", "cpu"):
        c = r[clock]
        ci = lambda k: f"[{c[k + '_ci'][0] / 1000:.0f}-{c[k + '_ci'][1] / 1000:.0f}]"  # noqa: E731
        print(f"  {clock:4} p50 {c['p50'] / 1000:5.0f} us  p99 {c['p99'] / 1000:5.0f} us {ci('p99'):>11}  "
              f"p99.9 {c['p99.9'] / 1000:5.0f} us {ci('p99.9'):>11}  max {c['max'] / 1000:7.0f} us  "
              f"over {args.decision_ms:g} ms: {c['over_limit']}")
    print(f"  seat-tables that would drain the {args.bank_ms:g} ms bank (CPU): {r['drained_banks']} of {4 * r['tables']}")
    print(f"  largest memory {r['memory_max']:,} bytes | points per seat {r['points_per_seat']:.2f} | "
          f"exceptions {r['exceptions']} | illegal {r['illegal']} | rules errors {len(rules_errors)}")
    for line in rules_errors:
        print(f"  {line}")
    failed = r["exceptions"] or r["illegal"] or rules_errors or r["drained_banks"]
    if args.strict and failed:
        print("bench: FAILED (see above)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
