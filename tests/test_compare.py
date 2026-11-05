import pytest

from engine.compare import _hero_lineup, compare, main
from engine.rules import PLO, TableConfig

LINEUP = _hero_lineup("hero,pushfold:2,tag:2,equity,folder,maniac")
CONFIG = TableConfig(enforce_time=False)


def test_hero_lineup_expands_and_requires_a_hero():
    assert _hero_lineup("hero:2,tag:6") == ["hero", "hero"] + ["tag"] * 6
    with pytest.raises(ValueError):
        _hero_lineup("tag:8")


def test_a_bot_against_itself_differs_by_exactly_nothing():
    c = compare("ours", "ours", LINEUP, tables=30, config=CONFIG)
    assert c.tables == 30
    assert c.diff_sum == 0 and c.diff_sq == 0  # same seats, cards and opponents: identical games
    assert c.candidate_sum == c.reference_sum > 0
    assert c.variance_removed == pytest.approx(1.0)


def test_pairing_finds_a_real_edge_quickly():
    # ours beats the old baseline by ~4-5 points a table; 300 paired tables is plenty.
    c = compare("ours", "baseline", LINEUP, tables=300, config=CONFIG)
    assert c.mean_diff - c.ci95 > 0
    assert c.errors == c.illegal == 0 and c.decisions > 0
    assert 0 < c.variance_removed < 1


@pytest.mark.slow
@pytest.mark.parametrize("previous", ["v3", "v4"])
def test_the_submitted_bot_still_beats_the_previous_ones(previous):
    # v4 beat push/fold (v3) by +3.7 ± 0.8; v5 beat v4 by +1.9 ± 0.7 on a fresh seed.
    c = compare("ours", previous, LINEUP, tables=3000, seed=2026, config=CONFIG, workers=8)
    assert c.mean_diff - c.ci95 > 0, (c.mean_diff, c.ci95)


def test_with_omaha_hands_the_submitted_bot_is_exactly_v4():
    c = compare("ours", "v4", LINEUP, tables=40, config=TableConfig(variant=PLO, enforce_time=False))
    assert c.diff_sum == 0 and c.diff_sq == 0


def test_parallel_matches_serial():
    serial = compare("ours", "baseline", LINEUP, tables=8, seed=3, config=CONFIG)
    parallel = compare("ours", "baseline", LINEUP, tables=8, seed=3, config=CONFIG, workers=2)
    assert serial.diff_sum == pytest.approx(parallel.diff_sum)
    assert serial.decisions == parallel.decisions


def test_unknown_bots_and_bad_lineups_are_rejected():
    with pytest.raises(ValueError):
        compare("nobody", "ours", LINEUP, tables=1)
    with pytest.raises(ValueError):
        compare("ours", "baseline", ["tag"] * 8, tables=1)


def test_cli_reports_the_difference_and_a_verdict(capsys):
    main(["--candidate", "ours", "--reference", "baseline", "--tables", "40", "--no-enforce-time"])
    out = capsys.readouterr().out
    assert "difference" in out and "(95%)" in out and "paired tables" in out
