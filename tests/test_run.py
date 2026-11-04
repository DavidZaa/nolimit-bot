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
