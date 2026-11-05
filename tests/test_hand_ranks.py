"""Starting-hand classes, the ranking generator, and the bot's use of its table."""

import importlib.util
import random

import pytest

from bot.baseline import preflop_percentile
from bot.preflop import HandRanks, hand_class
from bot.state import cards_from_str
from engine.rules import NLHE, PLO, SHORT_DECK, parse_variant
from gen.hand_ranks import enumerate_classes, rank_hands, write_table

C = cards_from_str
FEW_TRIALS = 20  # enough to order the extremes, fast enough for the suite


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_a_hands_class_ignores_which_suits_it_holds(size):
    rng = random.Random(size)
    for _ in range(500):
        hole = rng.sample(range(52), size)
        perm = rng.sample(range(4), 4)
        relabelled = [(c >> 2) * 4 + perm[c & 3] for c in hole]
        assert hand_class(hole) == hand_class(relabelled)
        assert hand_class(hole) == hand_class(reversed(hole))


def test_classes_tell_apart_what_matters():
    assert hand_class(C("AsKs")) != hand_class(C("AsKd"))
    assert hand_class(C("AsAdKsKd")) != hand_class(C("AsAdKsKh"))  # double-suited vs not
    assert hand_class(C("AsKdQhJc")) == hand_class(C("AhKsQcJd"))


@pytest.mark.parametrize("variant, classes, hands", [
    (NLHE, 169, 1326), (SHORT_DECK, 81, 630), (PLO, 16432, 270725),
])
def test_every_starting_hand_lands_in_one_class(variant, classes, hands):
    found = enumerate_classes(variant)
    assert len(found) == classes
    assert sum(n for n, _ in found.values()) == hands


@pytest.fixture(scope="module")
def nlhe_ranks():
    percentiles, _ = rank_hands(NLHE, workers=2, trials=600, log=lambda *a: None)
    return percentiles


def test_generated_hold_em_ranking_puts_the_right_hands_on_top_and_bottom(nlhe_ranks):
    p = lambda h: nlhe_ranks[hand_class(C(h))]
    assert p("AsAd") < p("KsKd") < 0.01
    assert p("AsKs") < p("AsKd")
    assert p("7c2d") > 0.9 and p("3c2d") > 0.95
    assert 0 < min(nlhe_ranks.values()) and max(nlhe_ranks.values()) < 1


def test_a_written_table_is_what_the_bot_reads(tmp_path, nlhe_ranks):
    path = tmp_path / "table.py"
    write_table(path, NLHE, nlhe_ranks, 600)
    spec = importlib.util.spec_from_file_location("table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ranks = HandRanks(module.PERCENTILES, module.HOLE_CARDS, module.VARIANT)
    assert ranks.percentile(C("AsAd")) == pytest.approx(nlhe_ranks[hand_class(C("AsAd"))], abs=1e-5)
    # Other hole-card counts fall back to Chen.
    assert ranks.percentile(C("AsAdKsKd")) == preflop_percentile(C("AsAdKsKd"))


def test_without_a_table_everything_is_chen():
    assert HandRanks().percentile(C("Ts9s")) == preflop_percentile(C("Ts9s"))


def test_odd_variants_can_be_ranked_too():
    three_card = parse_variant("base=nlhe,hole=3")
    percentiles, _ = rank_hands(three_card, workers=2, trials=FEW_TRIALS, log=lambda *a: None)
    p = lambda h: percentiles[hand_class(C(h))]
    assert p("AsAdAh") < 0.01 and p("2c3d4h") > p("AsAdAh")

