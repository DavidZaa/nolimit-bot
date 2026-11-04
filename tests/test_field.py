"""The sparring bots play like the archetypes they model, flaws included, and always legally."""

import random

import pytest

from bot.state import CALL, CHECK, FOLD, RAISE, Action, cards_from_str
from engine.field import (
    MONSTER, NOTHING, TOP_PAIR, TWO_PAIR_PLUS, WEAK_PAIR,
    ChenTAG, EquityCaller, FolderBot, Maniac, PushFold, RandomBot,
    draw_outs, equity_vs_random, made_hand, shove_range,
)
from engine.rules import NLHE, PLO, SHORT_DECK, TableConfig
from engine.run import parse_lineup, run_tables
from engine.sng import play_sng
from engine.table import play_hand
from helpers import Scripted


def decisions(bot, seat, hole, board, stacks=None, others=None):
    """Play one hand (button 0, blinds 10/20) with `bot` at `seat`; return its (state, action)s.

    Everyone else folds unless given in `others`; they get low filler cards.
    """
    stacks = stacks or [2000] * 8
    seen = []

    class Spy:
        def act(self, state):
            action = bot.act(state)
            seen.append((state, action))
            return action

    bots = [FolderBot() for _ in range(8)]
    for s, other in (others or {}).items():
        bots[s] = other
    bots[seat] = Spy()
    mine, shared = cards_from_str(hole), cards_from_str(board)
    spare = iter(c for c in range(52) if c not in mine + shared)
    holes = {s: mine if s == seat else (next(spare), next(spare)) for s in range(8) if stacks[s] > 0}
    play_hand(bots, stacks, 0, 1, TableConfig(), random.Random(0), holes=holes, board=shared)
    return seen


def first_action(*args, **kw):
    return decisions(*args, **kw)[0][1]


def facing_shove(bot, seat, hole, shover=3, stacks=None):
    """`bot`'s answer when `shover` moves all-in first. Builds a fresh shover each call."""
    stacks = stacks or [2000] * 8
    state, action = decisions(bot, seat, hole, "Kh7s2dJc4h", stacks=stacks,
                              others={shover: Scripted(Action(RAISE, stacks[shover]))})[0]
    assert max(state.bets) == stacks[shover]  # it really was facing the shove
    return action


# --- hand reading ------------------------------------------------------------------

@pytest.mark.parametrize("hole, board, bucket", [
    ("KcQd", "Kh7s2d", TOP_PAIR),
    ("AcAd", "Kh7s2d", TOP_PAIR),  # overpair
    ("7c5d", "Kh7s2d", WEAK_PAIR),
    ("4c4d", "Kh7s2d", WEAK_PAIR),  # underpair
    ("AcQd", "Kh7s7d", NOTHING),  # the pair is the board's
    ("AcQd", "Jh9s2d", NOTHING),
    ("Kc7d", "Kh7s2d", TWO_PAIR_PLUS),
    ("7c7d", "Kh7s2d", MONSTER),
    ("KcKd5h4s", "Kh7s2d", MONSTER),  # Omaha set
    ("Kc8d5h4s", "Kh7s2d", TOP_PAIR),
])
def test_made_hand_buckets(hole, board, bucket):
    assert made_hand(cards_from_str(hole), cards_from_str(board)) == bucket


@pytest.mark.parametrize("hole, board, outs", [
    ("AhKh", "Qh7h2d", 9),  # flush draw
    ("9c8d", "7h6s2d", 8),  # open-ended
    ("9h8h", "7h6h2d", 17),
    ("5c4d", "3h2sKd", 8),  # 2345: the ace or a six
    ("AcKd", "QhJs2d", 0),  # AKQJ only has one end
    ("9c7d", "8h5s2d", 0),  # gutshot
    ("AhKh", "Qh7h2d5c4s", 0),  # river: nothing left to draw
])
def test_draw_outs(hole, board, outs):
    assert draw_outs(cards_from_str(hole), cards_from_str(board)) == outs


def test_equity_vs_random_lands_near_the_known_values():
    eq = lambda h: equity_vs_random(cards_from_str(h))
    assert eq("AsAd") == pytest.approx(0.85, abs=0.01)
    assert eq("2s2d") == pytest.approx(0.50, abs=0.01)
    assert eq("7c2d") == pytest.approx(0.35, abs=0.01)
    assert eq("AsKd") == pytest.approx(0.65, abs=0.02)
    assert eq("AsKs") > eq("AsKd") > eq("KsQd") > eq("Ts9d") > eq("7c2d")


def test_shove_range_widens_with_shorter_stacks_and_fewer_players_behind():
    assert shove_range(10, 1) == pytest.approx(0.58)
    assert shove_range(10, 7) < shove_range(10, 3) < shove_range(10, 1)
    assert shove_range(20, 1) < shove_range(10, 1) < shove_range(5, 1) == 1.0


# --- ChenTAG ---------------------------------------------------------------------

@pytest.mark.parametrize("stack", [2000, 200])
def test_tag_opens_three_big_blinds_whatever_the_stack(stack):
    assert first_action(ChenTAG(), 3, "AsAd", "Kh7s2dJc4h", stacks=[stack] * 8) == Action(RAISE, 60)


def test_tag_folds_trash_preflop():
    assert first_action(ChenTAG(), 3, "7c2d", "Kh7s2dJc4h") == Action(FOLD)


def test_tag_calls_off_its_stack_with_any_hand_it_would_call_a_raise_with():
    assert facing_shove(ChenTAG(), 5, "AcQd") == Action(CALL)
    assert facing_shove(ChenTAG(), 5, "9c4d") == Action(FOLD)


def test_tag_bets_top_pair_and_gives_up_without_it():
    small_blind_checks = {1: Scripted(Action(CALL), Action(CHECK))}
    seen = decisions(ChenTAG(), 2, "Kc5d", "Kh7s2dJc4h", others=small_blind_checks)
    assert seen[0][1] == Action(CHECK)  # big blind option with K5o
    assert seen[1][1] == Action(RAISE, 26)  # two-thirds of a 40-chip pot

    small_blind_bets = {1: Scripted(Action(CALL), Action(RAISE, 40))}
    seen = decisions(ChenTAG(), 2, "Qc4d", "Kh7s2dJc5h", others=small_blind_bets)
    assert seen[1][1] == Action(FOLD)


# --- EquityCaller ----------------------------------------------------------------

def test_equity_caller_calls_a_shove_with_anything_ahead_of_a_random_hand():
    assert facing_shove(EquityCaller(), 5, "Kc7d") == Action(CALL)
    assert facing_shove(EquityCaller(), 5, "7c2d") == Action(FOLD)


def test_equity_caller_pays_off_a_pot_sized_river_bet_with_middle_pair_where_the_tag_folds():
    def river_action(bot):
        river_bet = {1: Scripted(Action(CALL), Action(CHECK), Action(CHECK), Action(RAISE, 40))}
        seen = decisions(bot, 2, "9c3d", "Kh9s2dJc4h", others=river_bet)
        assert [a for _, a in seen[:-1]] == [Action(CHECK)] * 3
        return seen[-1][1]

    assert river_action(EquityCaller()) == Action(CALL)
    assert river_action(ChenTAG()) == Action(FOLD)


# --- PushFold ----------------------------------------------------------------------

def test_pushfold_at_ten_big_blinds_shoves_wide_late_and_tight_early():
    ten_bb = [200] * 8
    assert first_action(PushFold(), 1, "Kc7d", "Kh7s2dJc4h", stacks=ten_bb) == Action(RAISE, 200)
    assert first_action(PushFold(), 3, "Kc7d", "Kh7s2dJc4h", stacks=ten_bb) == Action(FOLD)


def test_pushfold_raises_small_when_deep():
    assert first_action(PushFold(), 3, "AsAd", "Kh7s2dJc4h") == Action(RAISE, 50)


def test_pushfold_calls_a_short_shove_with_the_top_of_the_shoving_range():
    ten_bb = [200] * 8
    assert facing_shove(PushFold(), 2, "Ac5d", shover=1, stacks=ten_bb) == Action(CALL)
    assert facing_shove(PushFold(), 2, "8c3d", shover=1, stacks=ten_bb) == Action(FOLD)


def test_pushfold_continuation_bets_heads_up_as_the_raiser():
    big_blind_calls_then_checks = {2: Scripted(Action(CALL), Action(CHECK))}
    seen = decisions(PushFold(), 0, "AsQd", "Kh7s2dJc4h", others=big_blind_calls_then_checks)
    assert seen[0][1] == Action(RAISE, 50)
    assert seen[1][1] == Action(RAISE, 55)  # half of 50 + 50 + the folded small blind


# --- every table -------------------------------------------------------------------

@pytest.mark.parametrize("variant", [NLHE, SHORT_DECK, PLO], ids=lambda v: v.name)
def test_llm_bots_always_play_legally(variant):
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(12):
        bots = [ChenTAG(), EquityCaller(), PushFold(), ChenTAG(), EquityCaller(), PushFold(),
                RandomBot(t), Maniac()]
        random.Random(t).shuffle(bots)
        res = play_sng(bots, config, seed=t)
        assert sum(res.stats.errors) == sum(res.stats.illegal) == 0
        assert sum(res.stats.decisions) > 0


def test_llm_bots_decide_well_inside_the_time_limit():
    lineup = parse_lineup("tag:2,equity:2,pushfold:2,random,station")
    summaries = run_tables(lineup, tables=40, seed=0, config=TableConfig(enforce_time=False))
    for name in ("tag", "equity", "pushfold"):
        assert summaries[name].decisions > 500
        assert summaries[name].latency.percentile_us(0.999) < 500, name
