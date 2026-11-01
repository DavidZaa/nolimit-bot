import random
import re

import pytest

from bot.state import CALL, CHECK, RAISE, Action, cards_from_str
from engine.field import FolderBot
from engine.rules import TableConfig
from engine.run import parse_lineup, run_tables
from engine.table import play_hand
from engine.watch import format_hand, main, watch_table
from helpers import Scripted

LINEUP = parse_lineup("baseline:4,random:2,station,maniac")


def all_in_preflop():
    """Button 0; UTG (seat 3) shoves aces, the big blind calls with kings, everyone else folds."""
    bots = [FolderBot() for _ in range(8)]
    bots[3] = Scripted(Action(RAISE, 2000))
    bots[2] = Scripted(Action(CALL))
    holes = {s: cards_from_str(h) for s, h in enumerate(
        ["2c3d", "4c5d", "KcKh", "AsAd", "6c8d", "2h4s", "3h5s", "7h9s"])}
    before = [2000] * 8
    out = play_hand(bots, before, 0, 1, TableConfig(), random.Random(0),
                    holes=holes, board=cards_from_str("2d7c9hJsQc"))
    return before, out


def test_hand_shows_seats_actions_runout_showdown_and_result():
    before, out = all_in_preflop()
    names = ["folder"] * 8
    names[3] = "baseline"
    text = format_hand(1, before, out, names, TableConfig(), hero="baseline")

    assert text.splitlines()[0] == "Hand 1 · blinds 10/20"
    assert re.search(r"^\* s3 baseline .*2,000  As Ad\s*$", text, re.M)  # hero is starred
    assert re.search(r"^  s0 .*BTN$", text, re.M)
    assert re.search(r"^  s2 .*Kc Kh  BB$", text, re.M)
    assert "s3 raises to 2,000 (all-in), s4 folds" in text
    assert "s2 calls 1,980 (all-in)" in text  # chips added, not the total
    assert re.search(r"flop +\[2d 7c 9h\] +—", text)  # dealt with nobody left to act
    assert re.search(r"river +\[2d 7c 9h Js Qc\]", text)
    assert "showdown  s2 Kc Kh (pair), s3 As Ad (pair)" in text
    assert "result    s3 +2,010, s1 -10, s2 -2,000 (busts)" in text


def test_postflop_wagers_read_as_bets_then_raises():
    bots = [FolderBot() for _ in range(8)]
    bots[3] = Scripted(Action(CALL), Action(RAISE, 100), Action(CALL))
    bots[2] = Scripted(Action(CHECK), Action(RAISE, 40), Action(RAISE, 300))
    before = [2000] * 8
    out = play_hand(bots, before, 0, 1, TableConfig(), random.Random(0))
    text = format_hand(1, before, out, ["x"] * 8, TableConfig(), hero="baseline")
    assert "s2 bets 40, s3 raises to 100, s2 raises to 300, s3 calls 200" in text


def test_watch_replays_the_same_tables_as_the_runner():
    config = TableConfig(enforce_time=False)
    watched: dict[str, float] = {}
    for t in range(3):
        names, res = watch_table(LINEUP, seed=7, table=t, config=config, write=lambda text: None)
        for name, pts in zip(names, res.points):
            watched[name] = watched.get(name, 0.0) + pts
    ran = run_tables(LINEUP, tables=3, seed=7, config=config)
    assert watched == pytest.approx({name: s.points_sum for name, s in ran.items()})


def test_involved_only_prints_hands_a_hero_seat_played():
    config = TableConfig(enforce_time=False)
    full, involved = [], []
    watch_table(LINEUP, seed=1, table=0, config=config, write=full.append)
    names, _ = watch_table(LINEUP, seed=1, table=0, config=config, involved=True,
                           write=involved.append)
    hands = [block for block in involved if block.startswith("Hand ")]
    assert 0 < len(hands) < sum(block.startswith("Hand ") for block in full)
    hero_seats = [s for s, name in enumerate(names) if name == "baseline"]
    for block in hands:
        assert any(re.search(rf"s{s} (calls|bets|raises)|showdown.* s{s} ", block) for s in hero_seats)


def test_limit_caps_the_hands_printed_but_not_the_table():
    printed = []
    _, res = watch_table(LINEUP, seed=0, table=0, config=TableConfig(enforce_time=False),
                         limit=2, write=printed.append)
    assert sum(block.startswith("Hand ") for block in printed) == 2
    assert res.hands_played > 2


@pytest.mark.parametrize("variant", ["nlhe", "short_deck", "plo"])
def test_cli_prints_hands_then_standings(capsys, variant):
    main(["--variant", variant, "--limit", "3", "--no-enforce-time"])
    out = capsys.readouterr().out
    assert "Hand 1 · " in out and "Hand 4 · " not in out
    assert "place  seat  bot" in out
