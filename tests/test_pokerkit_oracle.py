"""Replay hands from our engine in PokerKit and demand identical results.

Our engine has to be right in the rare spots (short all-ins, side pots, odd
chips) because a tournament round is millions of hands. PokerKit is a mature,
heavily tested implementation, so any disagreement is a bug on our side or a
rules difference we need to know about.
"""

import os
import random

import pytest

pokerkit = pytest.importorskip("pokerkit")
from pokerkit import Automation, Mode, NoLimitTexasHoldem  # noqa: E402

from bot.state import CALL, CHECK, FOLD, RAISE, card_to_str  # noqa: E402
from engine.handeval import best_hand  # noqa: E402
from engine.rules import NLHE, BlindSchedule, TableConfig  # noqa: E402
from engine.table import TableStats, next_alive, play_hand  # noqa: E402
from helpers import RandomLegal  # noqa: E402

AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)
BETTING = (FOLD, CHECK, CALL, RAISE)


def pokerkit_order(stacks, out):
    """Our seats in PokerKit's player order: [SB, BB, ..., button], or [BB, button] heads-up."""
    alive = [s for s in range(len(stacks)) if stacks[s] > 0]
    if len(alive) == 2:
        return [out.bb_seat, out.sb_seat]
    order, s = [out.sb_seat], out.sb_seat
    for _ in range(len(alive) - 1):
        s = next_alive(stacks, s)
        order.append(s)
    return order


def cards(cs):
    return "".join(card_to_str(c) for c in cs)


def random_table(rng):
    seats = rng.randint(2, 8)
    stacks = []
    for _ in range(seats):
        kind = rng.random()
        if kind < 0.1:
            stacks.append(0)  # busted seat the engine must skip
        elif kind < 0.35:
            stacks.append(rng.randint(1, 80))  # can be all-in from blinds or antes
        else:
            stacks.append(rng.randint(100, 4000))
    if sum(1 for s in stacks if s > 0) < 2:
        stacks[0] = stacks[-1] = 500
    sb = rng.choice((5, 10, 25, 50))
    ante = rng.choice((0, 0, 0, sb // 5, sb))
    return stacks, (sb, 2 * sb, ante)


def replay_in_pokerkit(stacks, blinds, out):
    sb, bb, ante = blinds
    order = pokerkit_order(stacks, out)
    index = {seat: i for i, seat in enumerate(order)}
    state = NoLimitTexasHoldem.create_state(
        AUTOMATIONS, True, ante, (sb, bb), bb,
        tuple(stacks[s] for s in order), len(order), mode=Mode.TOURNAMENT,
    )
    for seat in order:
        state.deal_hole(cards(out.holes[seat]))

    board = list(out.board)
    deal_sizes = iter((3, 1, 1))

    def deal_pending():
        while state.status and (state.can_burn_card() or state.can_deal_board()):
            if state.can_burn_card():
                state.burn_card("??")
            n = next(deal_sizes)
            state.deal_board(cards(board[:n]))
            del board[:n]

    for rec in out.actions:
        if rec.kind not in BETTING:
            continue
        deal_pending()
        assert state.actor_index == index[rec.seat], f"turn order differs at {rec}"
        if rec.kind == FOLD:
            assert state.can_fold(), rec
            state.fold()
        elif rec.kind in (CHECK, CALL):
            assert state.can_check_or_call(), rec
            state.check_or_call()
        else:
            assert state.can_complete_bet_or_raise_to(rec.amount), (
                rec,
                state.min_completion_betting_or_raising_to_amount,
                state.max_completion_betting_or_raising_to_amount,
            )
            state.complete_bet_or_raise_to(rec.amount)
    deal_pending()
    assert not state.status, "PokerKit thinks the hand is still running"
    return [state.stacks[index[s]] if s in index else 0 for s in range(len(stacks))]


def assert_same_result(stacks, out, theirs):
    """Exact match, except for where odd chips land in multi-pot chops.

    PokerKit mucks losing hands before building pots and merges pots whose
    remaining players match, so a chop across several side pots can put an odd
    chip on a different winner than our per-pot split does. Both are valid
    conventions (the kit decides ours). We still require identical totals,
    and only seats that tied for a pot may differ, by at most one chip per pot.
    """
    if out.stacks == theirs:
        return
    assert sum(out.stacks) == sum(theirs)
    scores = {s: best_hand(h, out.board, NLHE) for s, h in out.shown.items()}
    values = list(scores.values())
    tied = {s for s, v in scores.items() if values.count(v) > 1}
    for s, (a, b) in enumerate(zip(out.stacks, theirs)):
        if a != b:
            assert s in tied, f"seat {s} differs but did not chop: {out.stacks} vs {theirs}"
            assert abs(a - b) < len(stacks)  # at most one odd chip per pot


def run_oracle(hands, seed):
    rng = random.Random(seed)
    for i in range(hands):
        stacks, blinds = random_table(rng)
        config = TableConfig(schedule=BlindSchedule((blinds,)))
        alive = [s for s in range(len(stacks)) if stacks[s] > 0]
        bots = [RandomLegal(seed=seed * 1_000_003 + i * 11 + s) for s in range(len(stacks))]
        stats = TableStats(len(stacks))
        out = play_hand(bots, stacks, rng.choice(alive), 1, config, random.Random(i), stats=stats)
        assert sum(stats.illegal) == 0
        theirs = replay_in_pokerkit(stacks, blinds, out)
        assert_same_result(stacks, out, theirs)


def test_engine_matches_pokerkit():
    run_oracle(400, seed=1)


@pytest.mark.slow
def test_engine_matches_pokerkit_at_scale():
    run_oracle(int(os.environ.get("ORACLE_HANDS", "20000")), seed=2)
