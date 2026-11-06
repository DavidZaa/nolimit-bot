"""The hand-history script reads players' tendencies back out of a round's events."""

import gzip
import json

from adapter import Agent
from engine.field import ChenTAG, FolderBot, Maniac, PushFold
from engine.kit_bridge import SEAT_ID, KitSeat
from engine.rules import TableConfig
from engine.sng import play_sng
from scripts.hh_stats import player_stats, read_tables, report

NAMES = ["us", "maniac", "folder", "tag", "pushfold", "folder2", "tag2", "pushfold2"]


def write_history(path, tables=3):
    cfg = TableConfig(enforce_time=False)
    with gzip.open(path, "wt") as f:
        for t in range(tables):
            us = KitSeat(Agent, cfg)
            bots = [us, Maniac(), FolderBot(), ChenTAG(), PushFold(), FolderBot(), ChenTAG(), PushFold()]
            play_sng(bots, cfg, seed=t)
            players = [{"seat": SEAT_ID(i), "name": name} for i, name in enumerate(NAMES)]
            f.write(json.dumps({"table_id": t, "players": players}) + "\n")
            for ev in us.delivered + us.pending:
                f.write(json.dumps({"table_id": t, "kind": ev.kind, "seat": ev.seat,
                                    "amount": ev.amount, "cards": list(ev.cards)}) + "\n")


def test_styles_come_back_out_of_a_history(tmp_path):
    path = tmp_path / "round.jsonl.gz"
    write_history(path)
    stats = player_stats(read_tables(path))
    assert set(stats) == set(NAMES)
    assert stats["folder"].hands > 50 and stats["folder"].vpip == 0
    assert stats["maniac"].vpip >= 0.9 * stats["maniac"].hands
    assert stats["maniac"].shoves >= 0.9 * stats["maniac"].hands
    assert stats["folder"].steal_folds == stats["folder"].steals_faced > 0
    text = report(stats)
    assert "maniac" in text and "fold/steal" in text and "cbet" in text
