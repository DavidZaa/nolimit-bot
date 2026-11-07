"""The push/fold bot we submit first: its hand reader, its play, and its port from the field."""

import random

import pytest

from bot.baseline import preflop_percentile
from bot.hands import MONSTER, NOTHING, TOP_PAIR, TWO_PAIR_PLUS, WEAK_PAIR, made_hand, strong_draw
from bot.pushfold import PushFoldBot
from bot.state import CALL, CHECK, FOLD, RAISE, Action, card_to_str, cards_from_str
from engine import field
from engine.field import CallingStation, ChenTAG, EquityCaller, Maniac, PushFold, RandomBot
from engine.rules import NLHE, PLO, SHORT_DECK, TableConfig, game_rules
from engine.sng import play_sng
from helpers import Scripted, decisions, facing_shove, first_action


# --- the hand reader ---------------------------------------------------------------

HOLDEM = game_rules(TableConfig(variant=NLHE))
OMAHA = game_rules(TableConfig(variant=PLO))


@pytest.mark.parametrize("hole, board, rules, bucket", [
    ("KcQd", "Kh7s2d", HOLDEM, TOP_PAIR),
    ("AcAd", "Kh7s2d", HOLDEM, TOP_PAIR),
    ("7c5d", "Kh7s2d", HOLDEM, WEAK_PAIR),
    ("AcQd", "Kh7s7d", HOLDEM, NOTHING),
    ("Kc7d", "Kh7s2d", HOLDEM, TWO_PAIR_PLUS),
    ("7c7d", "Kh7s2d", HOLDEM, MONSTER),
    ("Ah5d", "2c3s4dKh", HOLDEM, MONSTER),  # the wheel
    ("AhKh", "Qh7h2h", HOLDEM, MONSTER),
    ("AcKd", "Qh7h2h9h", HOLDEM, NOTHING),  # four hearts on board and none in hand
    ("AcKd", "Qh7h2h9hJh", HOLDEM, MONSTER),  # five on board plays in hold'em
    ("Ah2c5d6s", "KhQh7h", OMAHA, NOTHING),  # Omaha needs two hearts from the hand
    ("AhKh5d6s", "QhJh2h", OMAHA, MONSTER),
    ("Kd8h8sJs", "9c2sKc4d", OMAHA, TOP_PAIR),  # kings with the board beat the pocket eights
])
def test_hand_buckets(hole, board, rules, bucket):
    assert made_hand(cards_from_str(hole), cards_from_str(board), rules) == bucket


def test_hand_reader_survives_missing_cards():
    assert made_hand((), (), HOLDEM) == NOTHING
    assert made_hand(cards_from_str("As"), cards_from_str("Kh7s"), HOLDEM) == NOTHING
    assert made_hand(cards_from_str("AsAd"), cards_from_str("Kh"), OMAHA) == NOTHING


# --- how it plays ------------------------------------------------------------------

def test_at_ten_big_blinds_it_shoves_wide_late_and_tight_early():
    ten_bb = [200] * 8
    assert first_action(PushFoldBot(), 1, "Kc7d", "Kh7s2dJc4h", stacks=ten_bb) == Action(RAISE, 200)
    assert first_action(PushFoldBot(), 3, "Kc7d", "Kh7s2dJc4h", stacks=ten_bb) == Action(FOLD)


def test_deep_it_opens_small_instead_of_shoving():
    assert first_action(PushFoldBot(), 3, "AsAd", "Kh7s2dJc4h") == Action(RAISE, 50)


def test_it_calls_a_short_shove_with_the_top_of_the_shoving_range():
    ten_bb = [200] * 8
    assert facing_shove(PushFoldBot(), 2, "Ac5d", shover=1, stacks=ten_bb) == Action(CALL)
    assert facing_shove(PushFoldBot(), 2, "8c3d", shover=1, stacks=ten_bb) == Action(FOLD)


def test_it_calls_a_deep_blind_versus_blind_shove_only_with_premiums():
    assert facing_shove(PushFoldBot(), 2, "AsKd", shover=1) == Action(CALL)
    assert facing_shove(PushFoldBot(), 2, "AcQd", shover=1) == Action(FOLD)


def test_it_reads_a_deep_early_shove_as_a_nash_range_and_calls_only_aces():
    # Right against a solid shover, far too tight against a maniac. Fixing that
    # needs the opponent tracker's shove-rate stat.
    assert facing_shove(PushFoldBot(), 5, "AsAd") == Action(CALL)
    assert facing_shove(PushFoldBot(), 5, "KsKd") == Action(FOLD)


def test_it_continuation_bets_heads_up_as_the_raiser_then_respects_a_raise():
    big_blind = {2: Scripted(Action(CALL), Action(CHECK), Action(RAISE, 300))}
    seen = decisions(PushFoldBot(), 0, "AsQd", "Kh7s2dJc4h", others=big_blind)
    assert [a for _, a in seen] == [Action(RAISE, 50), Action(RAISE, 55), Action(FOLD)]


def test_it_value_bets_top_pair_when_checked_to():
    small_blind = {1: Scripted(Action(CALL), Action(CHECK))}
    seen = decisions(PushFoldBot(), 2, "Kc5d", "Kh7s2dJc4h", others=small_blind)
    assert seen[0][1] == Action(CHECK)
    assert seen[1][1] == Action(RAISE, 20)  # half of a 40-chip pot


# --- the port ----------------------------------------------------------------------

class Recorder:
    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink

    def act(self, state):
        self.sink.append(state)
        return self.inner.act(state)


@pytest.mark.parametrize("variant", [NLHE, SHORT_DECK, PLO], ids=lambda v: v.name)
def test_it_is_still_the_field_bot_except_where_it_reads_hands_correctly(variant):
    # It started as an exact port of the frozen field copy. The only intended difference
    # since: our hand reader follows the variant's rules (short deck's A-6-7-8-9 straight,
    # Omaha's best two-card pair), where the field copy's reader doesn't.
    states = []
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(15):
        bots = [Recorder(PushFoldBot(), states) for _ in range(3)] + [
            ChenTAG(), EquityCaller(), CallingStation(), Maniac(), RandomBot(t)]
        random.Random(t).shuffle(bots)
        play_sng(bots, config, seed=t)
    assert len(states) > 1000
    ours, theirs = PushFoldBot(), PushFold()
    splits = 0
    for st in states:
        if ours.act(st) != theirs.act(st):
            splits += 1
            assert variant is not NLHE and st.street > 0, st
            assert made_hand(st.hole, st.board, st.rules) != field.made_hand(st.hole, st.board), st
    assert splits < 0.05 * len(states)


@pytest.mark.parametrize("variant", [NLHE, SHORT_DECK, PLO], ids=lambda v: v.name)
def test_unwrapped_it_always_plays_legally(variant):
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(12):
        bots = [PushFoldBot() for _ in range(4)] + [ChenTAG(), EquityCaller(), Maniac(), RandomBot(t)]
        random.Random(t).shuffle(bots)
        res = play_sng(bots, config, seed=t)
        assert sum(res.stats.errors) == sum(res.stats.illegal) == 0


@pytest.mark.parametrize("hole, board, draw", [
    ("Ah5h", "Kh9h4d", True),  # nut flush draw
    ("9h8c", "Kh7h4h", True),  # one card to a flush draw
    ("Ac5c", "Kh9h4h", False),  # the flush draw is all on the board
    ("9c8d", "Th7s2d", True),  # open-ender
    ("5c4d", "3h2s9d", True),  # 2-3-4-5: the ace fills below
    ("KcQd", "JhTs2d", True),  # T-J-Q-K: a nine or an ace
    ("AcKd", "QhJs2d", False),  # J-Q-K-A only fills with a ten
    ("Kc2d", "Th9s8d", False),  # 8-9-T-? with no hole card in the run
    ("9c8d", "Th7s2dKh", True),  # still drawing on the turn
    ("9c8d", "Th7s2dKh3c", False),  # nothing left to draw to on the river
])
def test_strong_draws(hole, board, draw):
    assert strong_draw(cards_from_str(hole), cards_from_str(board), HOLDEM) is draw


@pytest.mark.parametrize("hole, board, draw", [
    ("Ah5hKdQd", "Kh9h4d", True),  # two hearts in hand, two on board
    ("Ah5cKdQd", "Kh9h4h", False),  # one heart in hand can't use a heart flush
    ("9c8dKsKh", "Th7s2d", True),  # 9-8 with T-7: a six or a jack
    ("9c2dKsKh", "Th8s7d", False),  # one card of the run in hand: Omaha needs two
])
def test_omaha_draws_use_exactly_two_hole_cards(hole, board, draw):
    assert strong_draw(cards_from_str(hole), cards_from_str(board), OMAHA) is draw


# -- the three-card discard variant ---------------------------------------------------------

def test_three_card_strength_is_the_best_two_plus_a_small_bonus_ranked_among_three_card_hands():
    from bot.preflop import CHEN, THREE_CONNECTED_BONUS, THREE_SUITED_BONUS, three_card_score

    p = lambda s: CHEN.percentile(cards_from_str(s))  # noqa: E731
    score = lambda s: three_card_score(cards_from_str(s))  # noqa: E731
    assert score("AsAd7c") == preflop_percentile(cards_from_str("AsAd"))  # best two, no bonus
    assert score("9h8h3h") == pytest.approx(preflop_percentile(cards_from_str("9h8h")) * (1 - THREE_SUITED_BONUS))
    assert score("9c8dTh") == pytest.approx(min(preflop_percentile(cards_from_str(x)) for x in ("9c8d", "9cTh", "8dTh"))
                                            * (1 - THREE_CONNECTED_BONUS))
    # As a percentile, it's a share of three-card hands: about as many below 10% as there should be.
    assert p("AsAd7c") < p("KsKd7c") < p("9c5d2h")
    import random
    rng = random.Random(1)
    hands = [rng.sample(range(52), 3) for _ in range(4000)]
    below = sum(CHEN.percentile(h) <= 0.10 for h in hands) / len(hands)
    assert 0.08 < below < 0.12
    # ...whereas on the two-card scale far more than 10% of three-card hands look like top-10% hands.
    assert sum(three_card_score(h) <= 0.10 for h in hands) / len(hands) > 0.2


def test_discard_rules_shrink_the_all_in_calling_range():
    from dataclasses import replace

    from bot.discard import default_rules
    from bot.pushfold import DISCARD_CALL_SCALE

    assert DISCARD_CALL_SCALE == 0.8
    # The small blind shoves 15 BB into our big blind: a wide calling spot (~27% of hands).
    # With discard rules (same state otherwise) some marginal calls must become folds, and
    # no fold may become a call.
    stacks = [300] * 8
    calls = flipped = 0
    bot = PushFoldBot()
    rng = random.Random(9)
    for _ in range(150):
        hole = "".join(card_to_str(c) for c in rng.sample(range(52), 2))
        state, plain = decisions(bot, 2, hole, "Kh7s2dJc4h", stacks=stacks,
                                 others={1: Scripted(Action(RAISE, 300))})[0]
        assert max(state.bets) == 300
        discarding = bot.act(replace(state, rules=replace(default_rules(), seats=8)))
        calls += plain.kind != FOLD
        assert not (plain.kind == FOLD and discarding.kind != FOLD), hole
        flipped += plain.kind != FOLD and discarding.kind == FOLD
    assert calls > 20 and flipped > 0, (calls, flipped)


def test_deep_in_the_discard_variant_only_the_very_top_stacks_off_preflop():
    from dataclasses import replace

    from bot.discard import default_rules
    from bot.preflop import CHEN
    from bot.pushfold import DEEP_SHOVE_TOP

    discard_rules = replace(default_rules(), seats=8)
    stacks = [3000] * 8  # 150 BB deep: a raise to 1300 is 43% of our stack, so it's all-in or fold
    bot = PushFoldBot()
    for hole, top in (("AsAdAh", True), ("AsAdKh", True), ("KsKd2c", False), ("AsKs7d", False)):
        # The engine deals two-card hands here; swap in the three-card hand and the variant's rules.
        state, _ = decisions(bot, 2, hole[:4], "Kh7s2dJc4h", stacks=stacks,
                             others={1: Scripted(Action(RAISE, 1300))})[0]
        deep = replace(state, rules=discard_rules, hole=cards_from_str(hole))
        assert (CHEN.percentile(deep.hole) <= DEEP_SHOVE_TOP) is top, hole
        if not top:
            assert bot.act(deep).kind == FOLD, hole
    # Two-card variants keep their old behaviour: kings still get it in there.
    state, plain = decisions(bot, 2, "KsKd", "Kh7s2dJc4h", stacks=stacks, others={1: Scripted(Action(RAISE, 1300))})[0]
    assert plain.kind == RAISE


def test_short_handed_heads_up_ranges_in_the_discard_variant(monkeypatch):
    from dataclasses import replace

    from bot.discard import default_rules

    rules = replace(default_rules(), seats=8)
    monkeypatch.setenv("NLB_SHORT_HANDED", "on")
    bot = PushFoldBot()
    stacks = [300, 300, 0, 0, 0, 0, 0, 0]  # heads-up, 15 BB; button 0 is the small blind
    # Button first in: a hand well outside normal ranges but inside the top 70% shoves.
    state, plain = decisions(bot, 0, "9c6d", "Kh7s2dJc4h", stacks=stacks)[0]
    hu = replace(state, rules=rules, hole=cards_from_str("9c6d3h"), players_left=2)
    assert bot.act(hu).kind == RAISE and plain.kind == FOLD
    # Big blind facing the shove: top 40% calls, the rest folds.
    shove = {0: Scripted(Action(RAISE, 300))}
    state, _ = decisions(bot, 1, "Qc9d", "Kh7s2dJc4h", stacks=stacks, others=shove)[0]
    assert bot.act(replace(state, rules=rules, hole=cards_from_str("Qc9d4s"), players_left=2)).kind in (CALL, RAISE)
    assert bot.act(replace(state, rules=rules, hole=cards_from_str("7c2d4s"), players_left=2)).kind == FOLD
    # Three or more left, or the flag off: the normal ranges.
    assert bot.act(replace(hu, players_left=3)).kind == FOLD
    monkeypatch.setenv("NLB_SHORT_HANDED", "off")
    assert bot.act(hu).kind == FOLD


def test_late_position_widens_the_button_and_plays_back_at_late_raises(monkeypatch):
    from dataclasses import replace

    from bot.discard import default_rules
    from bot.pushfold import LATE_3BET, LATE_CALL

    rules = replace(default_rules(), seats=8)
    monkeypatch.setenv("NLB_LATE_POSITION", "on")
    bot = PushFoldBot()
    # Button first in, 100 BB: a hand just outside the normal button range opens with the flag.
    state, plain = decisions(bot, 0, "Jc8d", "Kh7s2dJc4h")[0]
    late = replace(state, rules=rules, hole=cards_from_str("KcJd4s"))
    pct = bot.ranks.percentile(late.hole)
    assert 0.45 / 2 ** 0.8 < pct <= 0.45 / 2 ** 0.8 * 1.5  # (two behind: SB and BB)
    assert bot.act(late).kind == RAISE and bot.act(replace(late, players_left=4)).kind == FOLD
    # Facing a cutoff open in the big blind at 20 BB: top 8% shoves, top 15% calls cheap, else folds.
    stacks = [400] * 8
    state, _ = decisions(bot, 2, "AsKd", "Kh7s2dJc4h", stacks=stacks, others={7: Scripted(Action(RAISE, 50))})[0]
    for hole, want in (("AsAd7c", RAISE), ("AsQd7c", CALL), ("9c4d2h", FOLD)):
        deep = replace(state, rules=rules, hole=cards_from_str(hole))
        p = bot.ranks.percentile(deep.hole)
        assert (p <= LATE_3BET, p <= LATE_CALL) == {RAISE: (True, True), CALL: (False, True), FOLD: (False, False)}[want], hole
        assert bot.act(deep).kind == want, hole
