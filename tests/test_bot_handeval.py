"""The bot's own evaluator (bot/handeval.py): hand-made cases, rule variants, oracles, symmetries."""

import random
from dataclasses import replace

import pytest

from bot import handeval as H
from bot import rules as R
from bot.hands import HandStrength, hand_strength
from bot.state import cards_from_str, card_to_str
from engine import handeval as reference
from engine.rules import NLHE, PLO, SHORT_DECK, TableConfig, game_rules, parse_variant

HOLDEM = game_rules(TableConfig(variant=NLHE))
SHORT = game_rules(TableConfig(variant=SHORT_DECK))
OMAHA = game_rules(TableConfig(variant=PLO))


def value(cards, rules=HOLDEM, hole=None):
    """Value of `cards` (a string); for exact-usage rules pass `hole` and `cards` is the board."""
    if hole is not None:
        return H.hand_value(cards_from_str(hole), cards_from_str(cards), rules)
    c = cards_from_str(cards)
    return H.hand_value(c[:2], c[2:], rules)


def cat(cards, rules=HOLDEM, hole=None):
    return H.category(value(cards, rules, hole), rules)


# -- every category, and the standard order -------------------------------------------------

EXAMPLES = {
    R.HIGH_CARD: "AsJd9c6h3s",
    R.PAIR: "AsAd9c6h3s",
    R.TWO_PAIR: "AsAd9c9h3s",
    R.TRIPS: "AsAdAc6h3s",
    R.STRAIGHT: "9s8d7c6h5s",
    R.FLUSH: "AsJs9s6s3s",
    R.FULL_HOUSE: "AsAdAc6h6s",
    R.QUADS: "AsAdAcAh3s",
    R.STRAIGHT_FLUSH: "9s8s7s6s5s",
}


@pytest.mark.parametrize("category, cards", EXAMPLES.items(), ids=[R.CATEGORY_NAMES[c] for c in EXAMPLES])
def test_every_category_is_recognised(category, cards):
    assert cat(cards) == category


def test_standard_order():
    values = [value(EXAMPLES[c]) for c in R.STANDARD_ORDER]
    assert values == sorted(values) and len(set(values)) == 9


@pytest.mark.parametrize("better, worse", [
    ("AsAdKc6h3s", "AsAdQc6h3s"),  # pair kicker
    ("AsAdKcQh3s", "AsAdKcJh3s"),  # second kicker
    ("AsAd9c9hKs", "AsAd9c9hQs"),  # two pair kicker
    ("KsKdQcQh2s", "KsKdJcJh2s"),  # second pair
    ("AsAd2c2h3s", "KsKdQcQhJs"),  # top pair decides first
    ("8s8d8c2h2s", "7s7d7cAhAs"),  # full house: trips first
    ("AsJs9s6s3s", "AsJs9s6s2s"),  # flush down to the fifth card
    ("TsTdTcTh2s", "9s9d9c9hAs"),  # quads rank first
    ("6s5d4c3h2s", "5s4d3c2hAs"),  # the wheel is the lowest straight
    ("AsKdQcJhTs", "KsQdJcTh9s"),  # broadway the highest
    ("6s5s4s3s2s", "5h4h3h2hAh"),  # steel wheel is the lowest straight flush
])
def test_kickers_and_tiebreaks(better, worse):
    assert value(better) > value(worse)


def test_ties_are_equal_and_suits_never_break_them():
    assert value("AsKdQc9h7s") == value("AhKcQd9s7c")
    assert value("AsAdKcQh2s") == value("AhAcKdQs2c")


def test_best_five_of_seven_plays_the_board():
    hole, board = cards_from_str("2c3d"), cards_from_str("AsKsQsJsTs")
    assert H.category(H.hand_value(hole, board, HOLDEM), HOLDEM) == R.STRAIGHT_FLUSH
    assert H.hand_value(hole, board, HOLDEM) == H.hand_value(cards_from_str("4h5h"), board, HOLDEM)


def test_values_decode_to_category_and_tiebreak():
    v = value("KsKdQcQh2s")
    assert H.category(v, HOLDEM) == R.TWO_PAIR and H.tiebreak(v) == (11, 10, 0, 0, 0)


# -- variant rules ------------------------------------------------------------------------

def test_short_deck_flush_beats_full_house_and_trips_beat_a_straight():
    flush, boat = "AsJs9s7s6s", "AsAdAc7h7s"
    trips, straight = "6s6d6c8hTs", "As Kd Qc Jh Ts".replace(" ", "")
    for cards, category in ((flush, R.FLUSH), (boat, R.FULL_HOUSE), (trips, R.TRIPS), (straight, R.STRAIGHT)):
        assert cat(cards, SHORT) == category
    assert value(flush, SHORT) > value(boat, SHORT) and value(trips, SHORT) > value(straight, SHORT)
    assert value(flush, HOLDEM) < value(boat, HOLDEM) and value(trips, HOLDEM) < value(straight, HOLDEM)


def test_short_deck_wheel_is_ace_six_through_nine_and_ranks_lowest():
    assert cat("As9d8c7h6s", SHORT) == R.STRAIGHT
    assert value("As9d8c7h6s", SHORT) < value("Ts9d8c7h6s", SHORT)
    assert cat("As9s8s7s6s", SHORT) == R.STRAIGHT_FLUSH


def test_a_custom_deck_builds_its_own_straights():
    # Seven to ace: the ace plays low as A-7-8-9-T; A-2-3-4-5 can't exist.
    deck = R.make_rules("nlhe", **{k: getattr(HOLDEM, k) for k in R.TABLE_FIELDS})
    seven_up = replace(deck, ranks=tuple(range(5, 13)))
    assert cat("AsTd9c8h7s", seven_up) == R.STRAIGHT
    assert value("AsTd9c8h7s", seven_up) < value("JsTd9c8h7s", seven_up)
    no_wheel = replace(HOLDEM, ace_low_straight=False)
    assert cat("As5d4c3h2s", no_wheel) == R.HIGH_CARD
    assert cat("As5d4c3h2s", HOLDEM) == R.STRAIGHT


def test_any_category_order_is_honoured():
    reversed_order = replace(HOLDEM, category_order=tuple(reversed(R.STANDARD_ORDER)))
    assert value(EXAMPLES[R.HIGH_CARD], reversed_order) > value(EXAMPLES[R.STRAIGHT_FLUSH], reversed_order)
    # With high card on top, seven cards holding quads still pick a no-pair five if they can.
    assert cat("AsAdAcAh9s7d5c", reversed_order) == R.PAIR  # only four other ranks: a pair is unavoidable


# -- Omaha: exactly two from the hand, three from the board -------------------------------------

@pytest.mark.parametrize("hole, board, expected", [
    ("Ah2c5d6s", "KhQh7h9h", R.HIGH_CARD),  # four hearts on board, one in hand: no flush
    ("AhKh5d6s", "QhJh2h", R.FLUSH),
    ("2c3d4h5s", "AsKsQsJsTs", R.HIGH_CARD),  # the board's royal flush doesn't play, and A-2-3-4-5 needs four
    ("AsAdAcAh", "Kh7s2d", R.PAIR),  # four aces in hand still use only two
    ("KsKd2c3h", "AhAcAd", R.FULL_HOUSE),
    ("9c8d2s2h", "Th7s6d", R.STRAIGHT),
    ("9c2d3s4h", "Th8s7d", R.HIGH_CARD),  # one card of the run in hand isn't enough
])
def test_omaha_uses_exactly_two_hole_cards_and_three_from_the_board(hole, board, expected):
    assert cat(board, OMAHA, hole) == expected


def test_omaha_matches_brute_force_over_exactly_two_and_three():
    rng = random.Random(7)
    deck = OMAHA.deck()
    from itertools import combinations
    for _ in range(3000):
        cards = rng.sample(deck, 4 + 5)
        hole, board = cards[:4], cards[4:]
        brute = max(H.score5((*h, *b), OMAHA) for h in combinations(hole, 2) for b in combinations(board, 3))
        assert H.hand_value(hole, board, OMAHA) == brute


def test_too_few_cards_or_cards_off_the_deck_are_errors():
    with pytest.raises(ValueError):
        H.hand_value(cards_from_str("AsKd"), cards_from_str("Qh"), HOLDEM)
    with pytest.raises(ValueError):
        H.hand_value(cards_from_str("AsKdQhJh"), cards_from_str("2c3c"), OMAHA)
    with pytest.raises(ValueError):
        H.hand_value(cards_from_str("2s3d"), cards_from_str("4h5h9c"), SHORT)  # no deuces in short deck
    assert hand_strength(cards_from_str("AsKd"), cards_from_str("Qh"), HOLDEM) is None
    assert isinstance(hand_strength(cards_from_str("AsKd"), cards_from_str("QhJhTh"), HOLDEM), HandStrength)


# -- the reference evaluator, and PokerKit where the rules match -----------------------------

CUSTOM = parse_variant("ranks=5-A,trips_over_straight=1")
CUSTOM_PLO = parse_variant("base=plo,hole=5,flush_over_boat=1")


@pytest.mark.parametrize("variant", [NLHE, SHORT_DECK, PLO, CUSTOM, CUSTOM_PLO], ids=lambda v: v.name)
def test_agrees_value_for_value_with_the_simulator_reference(variant):
    rules = game_rules(TableConfig(variant=variant, seats=6 if variant.hole_cards == 5 else 8))
    rng = random.Random(3)
    deck = variant.deck()
    for _ in range(6000):
        cards = rng.sample(deck, variant.hole_cards + 5)
        hole, board = cards[:variant.hole_cards], cards[variant.hole_cards:variant.hole_cards + rng.choice((3, 4, 5))]
        assert H.hand_value(hole, board, rules) == reference.best_hand(hole, board, variant), (hole, board)


pokerkit = pytest.importorskip("pokerkit")

# PokerKit's ShortDeckHoldemHand keeps straight above trips (its documented rule set);
# ours (Triton) puts trips above. The comparison uses PokerKit's order so the rules match.
PK_SHORT = replace(SHORT, category_order=(R.HIGH_CARD, R.PAIR, R.TWO_PAIR, R.TRIPS, R.STRAIGHT,
                                          R.FULL_HOUSE, R.FLUSH, R.QUADS, R.STRAIGHT_FLUSH))


def _s(cards):
    return "".join(card_to_str(c) for c in cards)


@pytest.mark.parametrize("rules, pk_hand", [
    (HOLDEM, "StandardHighHand"), (PK_SHORT, "ShortDeckHoldemHand"), (OMAHA, "OmahaHoldemHand"),
], ids=["nlhe", "short_deck", "plo"])
def test_orders_hands_exactly_like_pokerkit(rules, pk_hand):
    pk = getattr(pokerkit, pk_hand)
    rng = random.Random(11)
    deck = rules.deck()
    n = rules.hole_cards
    for _ in range(2000):
        cards = rng.sample(deck, 5 + 2 * n)
        board, h1, h2 = cards[:5], cards[5:5 + n], cards[5 + n:]
        ours = H.hand_value(h1, board, rules) - H.hand_value(h2, board, rules)
        a, b = pk.from_game(_s(h1), _s(board)), pk.from_game(_s(h2), _s(board))
        assert (ours > 0) - (ours < 0) == (a > b) - (a < b), (_s(h1), _s(h2), _s(board))


# -- properties -------------------------------------------------------------------------------

@pytest.mark.parametrize("rules", [HOLDEM, SHORT, OMAHA], ids=lambda r: r.variant)
def test_values_ignore_suit_names_and_card_order(rules):
    rng = random.Random(5)
    deck = rules.deck()
    n = rules.hole_cards
    for _ in range(2000):
        cards = rng.sample(deck, n + 5)
        hole, board = cards[:n], cards[n:]
        v = H.hand_value(hole, board, rules)
        perm = rng.sample(range(4), 4)
        relabel = lambda cs: [(c >> 2) * 4 + perm[c & 3] for c in cs]  # noqa: E731
        assert H.hand_value(relabel(hole), relabel(board), rules) == v
        assert H.hand_value(rng.sample(hole, n), rng.sample(board, 5), rules) == v
        if rules.hole_cards_used is None:  # any five: which cards are "hole" doesn't matter
            mixed = rng.sample(cards, len(cards))
            assert H.hand_value(mixed[:n], mixed[n:], rules) == v


def test_the_evaluator_imports_nothing_from_the_simulator():
    import bot.handeval
    import bot.hands
    for module in (bot.handeval, bot.hands):
        text = open(module.__file__).read()
        assert "engine" not in text.replace("engine/handeval.py", "")
