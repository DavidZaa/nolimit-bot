"""Adapter pieces that don't depend on the kit. Extend on kit day with real observations."""

import random

import pytest

from adapter import build_bot, legal_actions, to_card
from bot.state import RAISE, cards_from_str
from engine.field import FolderBot
from engine.rules import TableConfig
from engine.run import BOTS
from engine.sng import play_sng
from engine.table import is_legal, play_hand


@pytest.mark.parametrize(
    "to_call, stack, lo, hi, can_raise",
    [(0, 1000, 20, 1000, True), (40, 1000, 80, 1000, True), (40, 30, 80, 30, True),
     (40, 40, 80, 40, True), (40, 70, 80, 70, True), (0, 500, 0, 0, False)],
)
def test_legal_actions_are_self_consistent(to_call, stack, lo, hi, can_raise):
    legal = legal_actions(to_call, stack, lo, hi, can_raise)
    assert legal.can_check != legal.can_fold
    assert legal.call_amount <= stack
    if legal.can_raise:
        assert stack > to_call and legal.min_raise_to <= legal.max_raise_to
    else:
        assert legal.min_raise_to == legal.max_raise_to == 0


def test_short_stack_cannot_raise_and_short_all_in_raise_is_pinned():
    assert not legal_actions(40, 40, 80, 40).can_raise  # calling is already all-in
    legal = legal_actions(40, 70, 80, 70)
    assert legal.can_raise and legal.min_raise_to == legal.max_raise_to == 70


def test_to_card_parses_standard_strings():
    assert (to_card("As"), to_card("Td")) == cards_from_str("AsTd")


def test_built_bot_plays_legally_through_the_engine():
    bots = [build_bot() for _ in range(8)]
    res = play_sng(bots, TableConfig(enforce_time=False), seed=5)
    assert sum(res.stats.errors) == sum(res.stats.illegal) == 0
    assert sum(res.stats.decisions) > 0


def test_built_bot_is_the_one_the_runner_calls_ours():
    assert type(build_bot().inner) is type(BOTS["ours"](0).inner)


def test_built_bot_raises_aces_unopened():
    bot = build_bot()
    seen = []

    class Spy:
        def act(self, state):
            action = bot.act(state)
            seen.append((state, action))
            return action

    hands = ["AsAd", "2c7d", "3c8d", "4c9d", "5cTd", "2h7s", "3h8s", "4h9s"]
    play_hand([Spy()] + [FolderBot() for _ in range(7)], [2000] * 8, 0, 1, TableConfig(),
              random.Random(0), holes={s: cards_from_str(h) for s, h in enumerate(hands)},
              board=cards_from_str("KhQhJh2s3s"))
    state, action = seen[0]
    assert action.kind == RAISE and is_legal(action, state.legal)
