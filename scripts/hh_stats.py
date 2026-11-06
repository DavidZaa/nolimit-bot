"""Per-player tendencies from a downloaded round's hand history.

    python scripts/hh_stats.py ROUND.jsonl.gz [--min-hands 50]

Prints VPIP, preflop shove rate, fold to a steal and fold to a postflop bet
for every player, pooled over every table in the file, so we can see what the
real field does and retune the tracker's priors between rounds.

The events are fed through bot.events.TableLog and bot.tracker.Tracker, the
same code the bot runs, so the numbers here are exactly what our tracker
would have counted. The download is gzipped JSON lines. PLACEHOLDER(kit): its
exact layout is only described in prose, so this accepts one event per line
or a table/hand per line carrying an "events" or "hands" list, and takes
player names from a "players"/"seats" list or a "name" field when present.
Check it against the kit's hand-history/ sample and streaming reader.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.events import TableLog  # noqa: E402
from bot.tracker import SeatStats, Tracker  # noqa: E402

TABLE_KEYS = ("table_id", "table")


def _events(obj):
    if isinstance(obj, dict):
        if "kind" in obj:
            yield obj
        for key in ("events", "hands", "history"):
            for item in obj.get(key) or ():
                yield from _events(item)
    elif isinstance(obj, list):
        for item in obj:
            yield from _events(item)


def _names(obj) -> dict:
    found = {}
    for key in ("players", "seats", "names"):
        for p in (obj.get(key) or ()) if isinstance(obj, dict) else ():
            if isinstance(p, dict) and "seat" in p:
                found[p["seat"]] = p.get("name") or p.get("team") or p.get("player") or p["seat"]
    return found


def read_tables(path):
    """table id -> (events in order, {seat id: player name})."""
    tables: dict = defaultdict(lambda: ([], {}))
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = next((obj[k] for k in TABLE_KEYS if isinstance(obj, dict) and k in obj), None)
            key = key if key is not None else ("stream" if isinstance(obj, dict) and "kind" in obj else n)
            events, names = tables[key]
            names.update(_names(obj))
            for ev in _events(obj):
                events.append(SimpleNamespace(kind=ev.get("kind"), seat=ev.get("seat"),
                                              amount=ev.get("amount"), cards=ev.get("cards") or ()))
                if ev.get("seat") is not None and ev.get("name"):
                    names[ev["seat"]] = ev["name"]
    return tables


def player_stats(tables, default_stack: int = 2000) -> dict:
    """player name -> SeatStats summed over every table they sat at."""
    totals: dict = defaultdict(SeatStats)
    for events, names in tables.values():
        seats = sorted({e.seat for e in events if e.seat is not None}, key=str)
        index = {s: i for i, s in enumerate(seats)}
        tracker = Tracker(external=True)
        tracker.seats = [SeatStats() for _ in seats]
        log = TableLog()
        log.feed(events, index, tracker, default_stack, len(seats))
        log.feed([SimpleNamespace(kind="hand", seat=None, amount=-1, cards=())], index, tracker,
                 default_stack, len(seats))  # count the last hand
        for seat, st in zip(seats, tracker.seats):
            total = totals[str(names.get(seat, seat))]
            for f in SeatStats.__slots__:
                setattr(total, f, getattr(total, f) + getattr(st, f))
    return totals


def _rate(hits: int, chances: int) -> str:
    return f"{100 * hits / chances:5.1f}% ({chances:>5})" if chances else "     -        "


def report(totals, min_hands: int = 0) -> str:
    lines = [f"{'player':<24}{'hands':>7}  {'VPIP':>14}  {'shove':>14}  {'fold/steal':>14}  {'fold/bet':>14}"]
    for name, st in sorted(totals.items(), key=lambda kv: -kv[1].hands):
        if st.hands < min_hands:
            continue
        lines.append(f"{name[:24]:<24}{st.hands:>7}  {_rate(st.vpip, st.hands)}  {_rate(st.shoves, st.hands)}  "
                     f"{_rate(st.steal_folds, st.steals_faced)}  {_rate(st.bet_folds, st.bets_faced)}")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("path")
    p.add_argument("--min-hands", type=int, default=0)
    p.add_argument("--stack", type=int, default=2000, help="starting stack, for spotting shoves on hand 1")
    args = p.parse_args(argv)
    print(report(player_stats(read_tables(args.path), args.stack), args.min_hands))


if __name__ == "__main__":
    main(sys.argv[1:])
