"""Watch one sit-and-go hand by hand.

    python -m engine.watch --table 3 | less
    python -m engine.watch --involved --lineup baseline,maniac:7

Tables are seated exactly like engine.run, so `--seed S --table T` replays
table T of `engine.run --seed S` card for card. Every seat's hole cards are
shown; seats running the --hero bot are starred.
"""

from __future__ import annotations

import argparse
import os
import sys

from bot.state import CALL, CHECK, FOLD, POST_ANTE, POST_BB, POST_SB, RAISE, card_to_str
from engine.handeval import best_hand, category
from engine.rules import CATEGORY_NAMES, TableConfig
from engine.run import add_table_args, parse_lineup, seat_table, table_config
from engine.sng import SngResult, play_sng
from engine.table import HandOutcome

STREET_NAMES = ("preflop", "flop", "turn", "river")


def _cards(cards) -> str:
    return " ".join(card_to_str(c) for c in cards)


def _role(seat: int, out: HandOutcome) -> str:
    roles = []
    if seat == out.button:
        roles.append("BTN")
    if seat == out.sb_seat:
        roles.append("SB")
    if seat == out.bb_seat:
        roles.append("BB")
    return " ".join(roles)


def _actions_by_street(before, out: HandOutcome) -> dict[int, list[str]]:
    """Readable voluntary actions per street; forced posts are shown on the seat rows instead."""
    n = len(before)
    stacks = list(before)
    street_bets = [0] * n
    highest = 0
    street = 0
    lines: dict[int, list[str]] = {}
    for rec in out.actions:
        if rec.street != street:
            street, street_bets, highest = rec.street, [0] * n, 0
        if rec.kind == POST_ANTE:  # antes go straight to the pot, not into the street's bets
            stacks[rec.seat] -= rec.amount
            continue
        added = rec.amount - street_bets[rec.seat]  # records carry the seat's bet after acting
        stacks[rec.seat] -= added
        street_bets[rec.seat] = rec.amount
        if rec.kind in (POST_SB, POST_BB):
            highest = max(highest, rec.amount)
            continue
        if rec.kind == FOLD:
            text = "folds"
        elif rec.kind == CHECK:
            text = "checks"
        elif rec.kind == CALL:
            text = f"calls {added:,}"
        elif rec.kind == RAISE:
            text = f"{'raises to' if highest else 'bets'} {rec.amount:,}"
            highest = rec.amount
        else:
            text = rec.kind
        if rec.kind in (CALL, RAISE) and stacks[rec.seat] == 0:
            text += " (all-in)"
        lines.setdefault(street, []).append(f"s{rec.seat} {text}")
    return lines


def format_hand(number: int, before, out: HandOutcome, names, config: TableConfig,
                hero: str) -> str:
    sb, bb, ante = config.schedule.level_for(number)
    lines = [f"Hand {number} · blinds {sb}/{bb}" + (f" ante {ante}" if ante else "")]

    for s, stack in enumerate(before):
        if stack == 0:
            continue
        mark = "*" if names[s] == hero else " "
        row = f"{mark} s{s} {names[s]:<12}{stack:>7,}  {_cards(out.holes[s])}  {_role(s, out)}"
        lines.append(row.rstrip())

    actions = _actions_by_street(before, out)
    dealt = 0
    for street, size in enumerate((0, *config.variant.board_schedule)):
        dealt += size
        if dealt > len(out.board):
            break
        label = STREET_NAMES[street] if street < len(STREET_NAMES) else f"street {street}"
        board = f"[{_cards(out.board[:dealt])}]" if street else ""
        lines.append(f"  {label:<8}{board:<17}{', '.join(actions.get(street, ['—']))}".rstrip())

    variant = config.variant
    if out.shown:
        hands = ", ".join(
            f"s{s} {_cards(hole)} "
            f"({CATEGORY_NAMES[category(best_hand(hole, out.board, variant), variant)]})"
            for s, hole in sorted(out.shown.items())
        )
        lines.append(f"  showdown  {hands}")

    changes = sorted(((out.stacks[s] - before[s], s) for s in range(len(before))),
                     key=lambda change: -change[0])
    results = ", ".join(
        f"s{s} {delta:+,}" + (" (busts)" if out.stacks[s] == 0 else "")
        for delta, s in changes if delta
    )
    lines.append(f"  result    {results}")
    return "\n".join(lines)


def format_standings(res: SngResult, names, hero: str) -> str:
    if res.reached_cap:
        how =f"hand cap reached after {res.hands_played} hands; survivors split the open places by ICM"
    else:
        how = f"finished after {res.hands_played} hands; one bot won every chip"
    lines = [f"Table {how}.", "  place  seat  bot           points"]
    for s in sorted(range(len(names)), key=lambda s: (res.finish[s], s)):
        mark = "*" if names[s] == hero else " "
        lines.append(f"{mark} {res.finish[s]:>5g}  s{s:<3} {names[s]:<12}{res.points[s]:>8.2f}")
    return "\n".join(lines)


def watch_table(lineup, seed: int, table: int, config: TableConfig, hero: str = "baseline",
                involved: bool = False, limit: int | None = None,
                write=print) -> tuple[list[str], SngResult]:
    """Play table `table` of run `seed`, writing each hand as it finishes, then the standings."""
    table_seed, names, bots = seat_table(lineup, seed, table)
    heroes = [s for s, name in enumerate(names) if name == hero]
    printed = 0

    def on_hand(number, before, out):
        nonlocal printed
        if limit is not None and printed >= limit:
            return
        if involved and not any(
            s in out.shown or any(r.seat == s and r.kind in (CALL, RAISE) for r in out.actions)
            for s in heroes
        ):
            return
        write(format_hand(number, before, out, names, config, hero) + "\n")
        printed += 1

    sb, bb, _ = config.schedule.level_for(1)
    write(f"Table {table} of run seed {seed} · {config.variant.name} · {config.starting_stack:,} chips"
          f" · blinds from {sb}/{bb}, up every {config.schedule.hands_per_level} hands"
          f" · payouts {'/'.join(f'{p:g}' for p in config.payouts if p)}\n")
    res = play_sng(bots, config, seed=table_seed, on_hand=on_hand)
    write(format_standings(res, names, hero))
    return names, res


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_table_args(p)
    p.add_argument("--table", type=int, default=0, help="which table of the run to replay")
    p.add_argument("--hero", default="baseline", help="bot type to star (and to follow with --involved)")
    p.add_argument("--involved", action="store_true",
                   help="only print hands where a hero seat called, raised or reached showdown")
    p.add_argument("--limit", type=int, help="print at most this many hands")
    args = p.parse_args(argv)

    lineup = parse_lineup(args.lineup)
    if args.involved and args.hero not in lineup:
        p.error(f"--involved needs the hero ({args.hero!r}) in the lineup")
    try:
        watch_table(lineup, args.seed, args.table, table_config(args), args.hero,
                    args.involved, args.limit)
    except BrokenPipeError:
        # Piped into `head` or a pager that quit early: stop quietly.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


if __name__ == "__main__":
    main(sys.argv[1:])
