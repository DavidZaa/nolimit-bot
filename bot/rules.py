"""The rules a decision is made under, from the SDK's table config. Bot-only: no engine imports.

The kit's state.config names the variant by ID only (plus players,
starting_stack, the blinds schedule, hands_per_level, hand_cap and payouts);
the variant's card rules are published in RULES.md as prose. So the card
rules live here, one explicit entry per variant ID, and the tournament
settings come from each table's config:

    rules_from_config(state.config) -> GameRules

Nothing is guessed. An unknown variant ID or a missing or malformed setting
raises RulesError, and the adapter then takes a safe legal action rather
than playing hold'em or 50/30/20 by default. scripts/pack.sh refuses to build
for a variant ID that has no entry here, so that can't ship by accident.

Kit day: add the announced variant's entry to VARIANTS under the exact ID the
SDK reports in state.config.variant (RULES.md has the rules), then generate
its starting-hand table (gen/hand_ranks.py writes bot/hand_tables/<id>.py).
"""

from __future__ import annotations

from functools import lru_cache

from bot.state import GameRules

# Hand categories. The numbers are identities, not strengths: a variant's
# category_order (weakest first) decides which beats which.
HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = (
    "high card", "pair", "two pair", "trips", "straight",
    "flush", "full house", "quads", "straight flush",
)
STANDARD_ORDER = (HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH)
# Triton-style short deck: flush beats full house, trips beat straight.
SHORT_DECK_ORDER = (HIGH_CARD, PAIR, TWO_PAIR, STRAIGHT, TRIPS, FULL_HOUSE, FLUSH, QUADS, STRAIGHT_FLUSH)

SUITS = ("c", "d", "h", "s")
BETTING = ("no_limit", "pot_limit")
ANY_FIVE = None  # hole_cards_used for hold'em-style hands: best five of all cards

# Card rules per SDK variant ID. PLACEHOLDER(kit): the IDs are ours until the
# SDK reports its own; add the announced one under the exact ID it uses.
VARIANTS: dict[str, dict] = {
    "nlhe": dict(ranks=tuple(range(13)), suits=SUITS, hole_cards=2, hole_cards_used=ANY_FIVE,
                 preflop_board=0, board_schedule=(3, 1, 1), category_order=STANDARD_ORDER,
                 ace_low_straight=True, betting="no_limit"),
    "short_deck": dict(ranks=tuple(range(4, 13)), suits=SUITS, hole_cards=2, hole_cards_used=ANY_FIVE,
                       preflop_board=0, board_schedule=(3, 1, 1), category_order=SHORT_DECK_ORDER,
                       ace_low_straight=True, betting="no_limit"),
    "plo": dict(ranks=tuple(range(13)), suits=SUITS, hole_cards=4, hole_cards_used=2,
                preflop_board=0, board_schedule=(3, 1, 1), category_order=STANDARD_ORDER,
                ace_low_straight=True, betting="pot_limit"),
}

CARD_FIELDS = ("ranks", "suits", "hole_cards", "hole_cards_used", "preflop_board", "board_schedule",
               "category_order", "ace_low_straight", "betting")
TABLE_FIELDS = ("payouts", "starting_stack", "blind_schedule", "hands_per_level", "hand_cap", "seats")


class RulesError(ValueError):
    """The table's rules are unknown or malformed; don't play a strategy on a guess."""


def _int(value, name: str, low: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) and not (
            isinstance(value, float) and value.is_integer()):
        try:  # NumPy ints and the like
            if int(value) != value:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise RulesError(f"{name} must be a whole number, got {value!r}") from None
    value = int(value)
    if value < low:
        raise RulesError(f"{name} must be at least {low}, got {value}")
    return value


def _level(raw, i: int) -> tuple[int, int, int]:
    """One blind level: (small, big[, ante]) as a sequence, or an object/dict with small/big/ante."""
    if isinstance(raw, dict):
        parts = (raw.get("small"), raw.get("big"), raw.get("ante", 0))
    elif hasattr(raw, "big"):
        parts = (getattr(raw, "small", None), raw.big, getattr(raw, "ante", 0))
    else:
        try:
            parts = tuple(raw)
        except TypeError:
            raise RulesError(f"blind level {i} is not a level: {raw!r}") from None
        if len(parts) == 2:
            parts = (*parts, 0)
        if len(parts) != 3:
            raise RulesError(f"blind level {i} needs small, big and optionally ante: {raw!r}")
    small, big, ante = (_int(p, f"blind level {i}") for p in parts)
    if big <= 0 or small > big:
        raise RulesError(f"blind level {i} is not small <= big, big > 0: {raw!r}")
    return small, big, ante


def check_card_rules(fields: dict) -> dict:
    """Validated, normalised card rules for one variant (raises RulesError)."""
    missing = [k for k in CARD_FIELDS if k not in fields]
    if missing:
        raise RulesError(f"variant rules missing {missing}")
    ranks = tuple(_int(r, "rank") for r in fields["ranks"])
    if len(ranks) < 5 or len(set(ranks)) != len(ranks) or max(ranks) > 12 or ranks != tuple(sorted(ranks)):
        raise RulesError(f"ranks must be 5+ distinct ascending ranks from 0 (deuce) to 12 (ace): {ranks}")
    suits = tuple(str(s) for s in fields["suits"])
    if suits != SUITS:  # cards are rank * 4 + suit everywhere in bot/
        raise RulesError(f"only the four standard suits are supported, got {suits}")
    hole = _int(fields["hole_cards"], "hole_cards", 1)
    used = fields["hole_cards_used"]
    used = ANY_FIVE if used in (None, "any") else _int(used, "hole_cards_used", 1)
    preflop_board = _int(fields["preflop_board"], "preflop_board")
    schedule = tuple(_int(x, "board_schedule") for x in fields["board_schedule"])
    board = preflop_board + sum(schedule)
    if used is not None and (used > hole or used > 5 or 5 - used > board):
        raise RulesError(f"can't make a hand using exactly {used} of {hole} hole cards and a {board}-card board")
    if used is None and hole + board < 5:
        raise RulesError(f"{hole} hole cards and a {board}-card board can't make five")
    order = tuple(_int(c, "category") for c in fields["category_order"])
    if sorted(order) != list(range(9)):
        raise RulesError(f"category_order must list all nine categories once: {order}")
    ace_low = fields["ace_low_straight"]
    if not isinstance(ace_low, bool):
        raise RulesError(f"ace_low_straight must be True or False, got {ace_low!r}")
    betting = fields["betting"]
    if betting not in BETTING:
        raise RulesError(f"unsupported betting structure {betting!r}; supported: {BETTING}")
    return dict(ranks=ranks, suits=suits, hole_cards=hole, hole_cards_used=used, preflop_board=preflop_board,
                board_schedule=schedule, category_order=order, ace_low_straight=ace_low, betting=betting)


def check_table_rules(fields: dict) -> dict:
    """Validated tournament settings (raises RulesError). Never fills in a default."""
    missing = [k for k in TABLE_FIELDS if fields.get(k) is None]
    if missing:
        raise RulesError(f"table config missing {missing}")
    seats = _int(fields["seats"], "seats (players)", 2)
    try:
        payouts = tuple(float(p) for p in fields["payouts"])
    except (TypeError, ValueError):
        raise RulesError(f"payouts must be numbers: {fields['payouts']!r}") from None
    if not payouts or any(not p >= 0 or p == float("inf") for p in payouts) or not any(payouts):
        raise RulesError(f"payouts must be finite, non-negative and not all zero: {payouts}")
    if len(payouts) > seats and any(payouts[seats:]):
        raise RulesError(f"payouts pay {len(payouts)} places at a {seats}-seat table: {payouts}")
    payouts = (payouts + (0.0,) * seats)[:seats]  # one entry per place; places past the list pay nothing
    try:
        levels = tuple(_level(raw, i) for i, raw in enumerate(fields["blind_schedule"]))
    except TypeError:
        raise RulesError(f"blinds schedule is not a list of levels: {fields['blind_schedule']!r}") from None
    if not levels:
        raise RulesError("blinds schedule is empty")
    return dict(payouts=payouts, starting_stack=_int(fields["starting_stack"], "starting_stack", 1),
                blind_schedule=levels, hands_per_level=_int(fields["hands_per_level"], "hands_per_level", 1),
                hand_cap=_int(fields["hand_cap"], "hand_cap", 1), seats=seats)


def card_rules(variant_id) -> dict:
    """The registered card rules for an SDK variant ID (raises RulesError if there are none)."""
    if not isinstance(variant_id, str) or variant_id not in VARIANTS:
        raise RulesError(f"no rules registered for variant {variant_id!r}; known: {sorted(VARIANTS)}")
    return check_card_rules(VARIANTS[variant_id])


def make_rules(variant_id, **table) -> GameRules:
    return GameRules(variant=variant_id, **card_rules(variant_id), **check_table_rules(table))


def _frozen(value):
    """A hashable copy of a config value, for the cache key."""
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    return value


def _blinds_key(raw):
    """The schedule as validated (small, big, ante) levels: whatever shape the SDK uses for a
    level (sequence, dict or object), the cache key and the rules see plain tuples."""
    if raw is None:
        return None
    try:
        return tuple(_level(level, i) for i, level in enumerate(raw))
    except TypeError:
        raise RulesError(f"blinds schedule is not a list of levels: {raw!r}") from None


@lru_cache(maxsize=256)
def _cached_rules(key) -> GameRules:
    variant, players, stack, blinds, per_level, cap, payouts = key
    return make_rules(variant, payouts=payouts, starting_stack=stack, blind_schedule=blinds,
                      hands_per_level=per_level, hand_cap=cap, seats=players)


def rules_from_config(cfg) -> GameRules:
    """GameRules from the SDK's state.config. Raises RulesError for anything missing,
    malformed or unregistered; every table's own payouts and schedule are used."""
    if cfg is None:
        raise RulesError("state has no config")
    raw = tuple(getattr(cfg, name, None) for name in _CONFIG_NAMES)
    try:  # the common case: this exact config was seen before (a dict lookup, no validation)
        return _BY_RAW[raw]
    except (KeyError, TypeError):  # new, or holds lists/dicts: freeze it and look again
        raw = tuple(_frozen(v) for v in raw)
        try:
            return _BY_RAW[raw]
        except (KeyError, TypeError):
            pass
    key = tuple(_blinds_key(v) if name == "blinds" else v for name, v in zip(_CONFIG_NAMES, raw))
    try:
        rules = _cached_rules(key)
    except TypeError:  # something unhashable slipped through: validate without the cache
        return _cached_rules.__wrapped__(key)
    try:
        if len(_BY_RAW) < 256:
            _BY_RAW[raw] = rules
    except TypeError:
        pass
    return rules


_CONFIG_NAMES = ("variant", "players", "starting_stack", "blinds", "hands_per_level", "hand_cap", "payouts")
_BY_RAW: dict = {}  # raw config values -> validated rules; only valid configs get in


def to_dict(rules: GameRules) -> dict:
    """Plain JSON-able form (tuples become lists)."""
    return {name: getattr(rules, name) for name in GameRules.__slots__}


def from_dict(doc: dict) -> GameRules:
    """Inverse of to_dict, re-validated; the variant need not be registered."""
    try:
        card = check_card_rules(doc)
        table = check_table_rules(doc)
        variant = doc["variant"]
    except (KeyError, TypeError) as exc:
        raise RulesError(f"not a rules document: {exc}") from None
    if not isinstance(variant, str) or not variant:
        raise RulesError(f"variant must be a non-empty string, got {variant!r}")
    return GameRules(variant=variant, **card, **table)
