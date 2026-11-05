import pytest

from engine.rules import TableConfig
from engine.run import main, parse_lineup, run_tables

PAYOUTS = (50.0, 30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
LINEUP = parse_lineup("baseline:3,random:2,station,maniac,folder")


def test_parse_lineup_expands_counts():
    assert parse_lineup("baseline:2,folder") == ["baseline", "baseline", "folder"]
    with pytest.raises(ValueError):
        parse_lineup("nobody")


def test_run_tables_accounts_for_every_point():
    summaries = run_tables(LINEUP, tables=20, seed=1, config=TableConfig(payouts=PAYOUTS))
    assert set(summaries) == {"baseline", "random", "station", "maniac", "folder"}
    assert summaries["baseline"].instances == 60
    total = sum(s.points_sum for s in summaries.values())
    assert total == pytest.approx(20 * sum(PAYOUTS))
    base = summaries["baseline"]
    assert base.decisions > 0 and base.errors == base.illegal == base.timeouts == 0


def test_parallel_matches_serial():
    config = TableConfig(payouts=PAYOUTS)
    serial = run_tables(LINEUP, tables=8, seed=3, config=config, workers=1)
    parallel = run_tables(LINEUP, tables=8, seed=3, config=config, workers=2)
    for name in serial:
        assert serial[name].points_sum == pytest.approx(parallel[name].points_sum)
        assert serial[name].decisions == parallel[name].decisions


def test_lineup_must_fill_the_table():
    with pytest.raises(ValueError):
        run_tables(["baseline"] * 3, tables=1)


def test_cli_prints_a_report(capsys):
    main(["--tables", "4", "--lineup", "baseline:4,random:4", "--seed", "2"])
    out = capsys.readouterr().out
    assert "baseline" in out and "random" in out and "p99.9" in out
    assert "timed on cpu clock" in out or "timed on wall clock" in out


def test_schedules_come_as_presets_level_lists_or_growth_rules():
    from engine.rules import DOUBLING, parse_schedule

    assert parse_schedule("doubling") == DOUBLING
    assert parse_schedule("x1.5", 10).hands_per_level == 10
    assert parse_schedule("10/20,15/30,25/50/5").levels == ((10, 20, 0), (15, 30, 0), (25, 50, 5))
    grown = parse_schedule("bb=100,x=1.5,ante=0.1,levels=3").levels
    assert grown == ((50, 100, 10), (75, 150, 15), (112, 225, 22))
    with pytest.raises(ValueError):
        parse_schedule("bb=20,speed=fast")


def test_cli_plays_the_announced_structure(capsys):
    main(["--tables", "3", "--lineup", "ours:2,pushfold:2,tag:2,random,maniac", "--no-enforce-time",
          "--variant", "base=nlhe,hole=3", "--schedule", "25/50/5,50/100/10", "--hands-per-level", "10",
          "--stack", "3000", "--payouts", "40,25,15,10,5,3,2,0"])
    out = capsys.readouterr().out
    assert "base=nlhe,hole=3" in out and "[40.0, 25.0, 15.0" in out
