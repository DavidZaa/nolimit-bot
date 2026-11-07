"""A/B two builds of the bot in the kit's own simulator (poker-agent play), paired by table.

    python scripts/kit_ab.py --base dist/agent --a smart: --b default:select=default \\
        --tables 1000 --seeds 1,2 [--seats 4] [--kit ../starter-kit]
    python scripts/kit_ab.py --a new=dist/new --b old=dist/old ...

Each side is NAME:FLAGS (a copy of --base with bot/flags.py set to FLAGS, comma-separated
key=value) or NAME=DIR (a built package as is). Every table seats --seats of each side
(8 seats by default, so 4 + 4); `play` randomizes seating per table. Per table the
difference is (mean points of A's seats) - (mean points of B's seats), read from the
hand history's final equities, and the report gives its mean with a 95% interval per
seed and pooled, plus failed decisions per seat from play's own summary.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(spec: str, base: Path, work: Path) -> tuple[str, Path]:
    if "=" in spec.split(":")[0]:
        name, _, path = spec.partition("=")
        target = work / name
        shutil.copytree(path, target, ignore=shutil.ignore_patterns("__pycache__", ".*"))
        return name, target
    name, _, flags = spec.partition(":")
    target = work / name
    shutil.copytree(base, target, ignore=shutil.ignore_patterns("__pycache__", ".*"))
    pairs = dict(item.split("=", 1) for item in flags.split(",") if item)
    (target / "bot" / "flags.py").write_text(
        (ROOT / "bot" / "flags.py").read_text().replace("FLAGS: dict = {}", f"FLAGS: dict = {pairs!r}"))
    return name, target


def table_points(history: Path, styles: dict | None = None) -> list[dict]:
    """Per table: {seat label: final points}. With `styles`, also counts per label prefix:
    hands dealt, VPIP (a preflop call or raise) and shoves (the kit's prize_the_shover events)."""
    out = []
    with gzip.open(history, "rt") as f:
        for line in f:
            rec = json.loads(line)
            if "hands" not in rec or not rec["hands"]:
                continue
            seats = rec["seats"]
            out.append(dict(zip(seats, rec["hands"][-1]["equity_after"])))
            if styles is None:
                continue
            for hand in rec["hands"]:
                preflop, vpip, shoves = True, set(), set()
                raised, first_in, faced, folded_to = False, set(), set(), set()
                for ev in hand["events"]:
                    kind, seat = ev.get("kind"), ev.get("seat")
                    if kind == "board":
                        preflop = False
                    elif preflop and kind in ("call", "raise") and seat is not None:
                        vpip.add(seat)
                        if kind == "raise":
                            if not raised and seat not in vpip - {seat}:
                                first_in.add(seat)
                            raised = True
                    elif preflop and kind == "fold" and seat is not None and raised:
                        faced.add(seat)
                        folded_to.add(seat)
                    elif kind == "prize_the_shover" and seat is not None:
                        shoves.add(seat)
                    if preflop and raised and kind in ("call", "raise") and seat is not None:
                        faced.add(seat)
                for i, label in enumerate(seats):
                    if hand["stacks_before"][i] > 0:
                        st = styles.setdefault(label.split("#")[0], [0, 0, 0, [0, 0, 0, 0, 0], 0, 0, 0])
                        st[0] += 1
                        st[1] += i in vpip
                        st[2] += i in shoves
                        st[4] += i in first_in
                        st[5] += i in faced
                        st[6] += i in folded_to
            # Places: survivors by final chips, busted seats by how long they lasted.
            hands = rec["hands"]
            bust_hand = {}
            for n, hand in enumerate(hands):
                for i in range(len(seats)):
                    if hand["stacks_before"][i] > 0 and hand["stacks_after"][i] == 0:
                        bust_hand[i] = (n, hand["stacks_before"][i])
            final = hands[-1]["stacks_after"]
            order = sorted(range(len(seats)), key=lambda i: (final[i] > 0, final[i] if final[i] else 0,
                                                             bust_hand.get(i, (10**9, 0))), reverse=True)
            paid = sum(1 for x in rec.get("payouts", (1, 1, 1)) if x)
            for place, i in enumerate(order, 1):
                st = styles.setdefault(seats[i].split("#")[0], [0, 0, 0, [0, 0, 0, 0, 0], 0, 0, 0])
                st[3][4] += 1
                if place <= 3:
                    st[3][place - 1] += 1
                if place <= paid:
                    st[3][3] += 1
    return out


def mean_ci(xs) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else float("inf")
    return m, 1.96 * math.sqrt(var / n)


def play(dirs, tables: int, seed: str, kit: Path, work: Path, styles=None) -> tuple[list[dict], dict]:
    history = work / f"history-{seed}.jsonl.gz"
    cmd = [shutil.which("poker-agent") or "poker-agent", "play", *map(str, dirs), "--tables", str(tables),
           "--seed", seed, "--history", str(history)]
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=kit)
    if run.returncode != 0:
        raise SystemExit(f"poker-agent play failed:\n{run.stderr[-3000:]}")
    if "RULES" in run.stderr:
        print("warning: the bot printed RULES lines:\n" + run.stderr[-1500:], file=sys.stderr)
    return table_points(history, styles), json.loads(run.stdout)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", type=Path, default=ROOT / "dist" / "agent")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--seats", type=int, default=4, help="seats per side")
    p.add_argument("--tables", type=int, default=1000)
    p.add_argument("--seeds", default="1,2")
    p.add_argument("--kit", type=Path, default=ROOT.parent / "starter-kit")
    p.add_argument("--extra", default="", help="other bot folders to seat too, comma-separated")
    p.add_argument("--realistic", action="store_true",
                   help="2 + 2 of ours plus smart_tag, reader, default_tag, pushfold from ~/Desktop/2026/opponents")
    args = p.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="kit-ab-"))
    (a, a_dir), (b, b_dir) = build(args.a, args.base, work), build(args.b, args.base, work)
    extra = [Path(d).resolve() for d in args.extra.split(",") if d]
    if args.realistic:
        args.seats = 2
        extra += [Path.home() / "Desktop" / "2026" / "opponents" / n for n in ("smart_tag", "reader", "default_tag", "pushfold")]
    dirs = [a_dir] * args.seats + [b_dir] * args.seats + extra  # play reseats every table
    pooled = []
    for seed in args.seeds.split(","):
        styles: dict = {}
        tables, summary = play(dirs, args.tables, seed, args.kit, work, styles)
        diffs, a_pts, b_pts = [], [], []
        for t in tables:
            pa = [v for k, v in t.items() if k.startswith(f"{a}#")]
            pb = [v for k, v in t.items() if k.startswith(f"{b}#")]
            a_pts.append(sum(pa) / len(pa))
            b_pts.append(sum(pb) / len(pb))
            diffs.append(a_pts[-1] - b_pts[-1])
        pooled += diffs
        m, ci = mean_ci(diffs)
        failed = {k: v["failed_decisions"] for k, v in summary["bots"].items() if v["failed_decisions"]}
        print(f"seed {seed}: {len(tables)} tables | {a} {sum(a_pts) / len(a_pts):6.2f}  {b} "
              f"{sum(b_pts) / len(b_pts):6.2f} | {a} - {b} = {m:+.2f} ± {ci:.2f} (95%) | "
              f"failed decisions: {failed or 0}")
        for name, (hands, vpip, shoves, (p1, p2, p3, cash, n), opens, faced, folds) in sorted(styles.items()):
            print(f"    {name:14} VPIP {100 * vpip / hands:5.1f}%  shove {100 * shoves / hands:4.1f}%  "
                  f"first-in raise {100 * opens / hands:4.1f}%  fold-to-raise {100 * folds / max(1, faced):4.1f}%  "
                  f"1st {100 * p1 / n:4.1f}%  2nd {100 * p2 / n:4.1f}%  3rd {100 * p3 / n:4.1f}%  "
                  f"cash {100 * cash / n:4.1f}%  HU win {100 * p1 / max(1, p1 + p2):4.1f}%  ({n} seat-tables)")
    m, ci = mean_ci(pooled)
    verdict = f"{a} better" if m - ci > 0 else f"{b} better" if m + ci < 0 else "no clear difference"
    print(f"pooled: {len(pooled)} tables | {a} - {b} = {m:+.2f} ± {ci:.2f} (95%): {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
