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

DISCARD_VARIANT = dict(ranks=tuple(range(13)), suits=SUITS, hole_cards=3, hole_cards_used=ANY_FIVE,
                       preflop_board=0, board_schedule=(3, 1, 1), category_order=STANDARD_ORDER,
                       ace_low_straight=True, betting="no_limit", discards=1, exposes=1)

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
    # The Bay Area variant (kit RULES.md): three hole cards; after the flop is dealt and
    # before flop betting, every player still in discards one face down and exposes one of
    # the two kept; then hold'em with the two kept cards, best five of board and hand. The
    # engine's name is flop_discard_reveal, its public ID bay_area: the server may send either.
    "flop_discard_reveal": DISCARD_VARIANT,
    "bay_area": DISCARD_VARIANT,
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


def blind_level(raw, i: int = 0) -> tuple[int, int, int]:
    """One blind level as (small, big, ante). The kit sends BlindLevel(small_blind, big_blind,
    ante) objects (poker_agent_protocol.game); our simulator sends (small, big[, ante])
    sequences; dicts and .small/.big objects are accepted too."""
    if hasattr(raw, "big_blind"):  # the kit's BlindLevel
        parts = (getattr(raw, "small_blind", None), raw.big_blind, getattr(raw, "ante", 0) or 0)
    elif isinstance(raw, dict):
        parts = (raw.get("small_blind", raw.get("small")), raw.get("big_blind", raw.get("big")),
                 raw.get("ante", 0) or 0)
    elif hasattr(raw, "big"):
        parts = (getattr(raw, "small", None), raw.big, getattr(raw, "ante", 0) or 0)
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
    discards = _int(fields.get("discards", 0), "discards")
    exposes = _int(fields.get("exposes", 0), "exposes")
    if discards >= hole or exposes > hole - discards:
        raise RulesError(f"can't discard {discards} and expose {exposes} of {hole} hole cards")
    if used is None and hole - discards + board < 5:
        raise RulesError(f"{hole - discards} kept hole cards and a {board}-card board can't make five")
    return dict(ranks=ranks, suits=suits, hole_cards=hole, hole_cards_used=used, preflop_board=preflop_board,
                board_schedule=schedule, category_order=order, ace_low_straight=ace_low, betting=betting,
                discards=discards, exposes=exposes)


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
        levels = tuple(blind_level(raw, i) for i, raw in enumerate(fields["blind_schedule"]))
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
        return tuple(blind_level(level, i) for i, level in enumerate(raw))
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


# -- degrading gracefully -------------------------------------------------------------------
#
# The tournament never shows stderr, and a bot that checks or folds every hand
# passes the upload match, so failing closed on a config detail we guessed wrong
# (an unannounced variant ID, the blinds schedule's name or shape, short payouts)
# would look like a healthy bot scoring nothing. So the agent uses
# rules_from_state: the strict rules when the config is as expected, otherwise
# the best rules it can build from the config plus the state, with a warning per
# guess. NLB_STRICT=1 restores fail-closed (pack.sh and local play set it, so
# problems surface before upload). RulesError only when even that is impossible.

DEFAULT_PAYOUTS = (50.0, 30.0, 20.0)  # the published default table payouts
DEFAULT_HANDS_PER_LEVEL = 20
DEFAULT_HAND_CAP = 250
DEFAULT_STACK = 2000


def strict_mode() -> bool:
    import os

    return os.environ.get("NLB_STRICT") == "1"


def _current_level(state) -> tuple[int, int, int] | None:
    raw = getattr(state, "blinds", None)
    try:
        return blind_level(raw, 0)
    except Exception:
        return None


@lru_cache(maxsize=256)
def _lenient(raw, hole: int, betting_hint: str, seats_seen: int, current_level):
    """(rules, warnings) for a config the strict path refused. Cached on everything it reads:
    the raw config, the hole-card count, the betting evidence ("pot_limit" when the max raise
    is below the all-in, "no_limit" when it reaches it, "" when no raise is offered), the
    number of seats and the current blinds."""
    variant, players, stack, blinds, per_level, cap, payouts = raw
    warnings = []
    try:
        card = card_rules(variant)
    except RulesError:
        # 3 hole cards: the announced discard variant; 4+: PLO-style (exactly two from the
        # hand); fewer: hold'em-style.
        base = "flop_discard_reveal" if hole == 3 else "plo" if hole >= 4 else "nlhe"
        fields = dict(VARIANTS[base])
        if hole and base != "flop_discard_reveal":
            fields["hole_cards"] = hole
        if betting_hint:
            fields["betting"] = betting_hint
        card = check_card_rules(fields)
        warnings.append(f"unknown variant {variant!r}: playing {base}-style rules "
                        f"({card['hole_cards']} hole cards, {card['betting']})")
    try:
        seats = _int(players, "players", 2)
    except RulesError:
        seats = max(2, seats_seen)
        warnings.append(f"players {players!r} unusable: using the {seats} seats at the table")
    table = {}
    for name, raw_value, check, default in (
        ("starting_stack", stack, lambda v: _int(v, "starting_stack", 1), DEFAULT_STACK),
        ("hands_per_level", per_level, lambda v: _int(v, "hands_per_level", 1), DEFAULT_HANDS_PER_LEVEL),
        ("hand_cap", cap, lambda v: _int(v, "hand_cap", 1), DEFAULT_HAND_CAP),
    ):
        try:
            table[name] = check(raw_value)
        except RulesError:
            table[name] = default
            warnings.append(f"{name} {raw_value!r} unusable: using {default}")
    try:
        table["blind_schedule"] = check_table_rules(dict(
            payouts=(1,), starting_stack=1, blind_schedule=blinds, hands_per_level=1, hand_cap=1,
            seats=2))["blind_schedule"]
    except RulesError:
        if current_level is None:
            raise RulesError("no usable blinds schedule and no current blinds on the state") from None
        table["blind_schedule"] = (current_level,)
        warnings.append(f"blinds schedule {_short(blinds)} unusable: using the current blinds {current_level}")
    try:
        table["payouts"] = check_table_rules(dict(
            payouts=payouts, starting_stack=1, blind_schedule=((1, 2, 0),), hands_per_level=1, hand_cap=1,
            seats=seats))["payouts"]
    except RulesError:
        table["payouts"] = (DEFAULT_PAYOUTS + (0.0,) * seats)[:seats]
        warnings.append(f"payouts {_short(payouts)} unusable: using 50/30/20 over {seats} seats")
    name = variant if isinstance(variant, str) and variant else "unknown"
    return GameRules(variant=name, **card, **check_table_rules(dict(table, seats=seats))), tuple(warnings)


def _short(value) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


class _Aliased:
    """The SDK config with its variant ID mapped onto a registered one."""

    def __init__(self, cfg, variant):
        self._cfg, self.variant = cfg, variant

    def __getattr__(self, name):
        return getattr(self._cfg, name)


def rules_from_state(state, strict: bool | None = None, aliases: dict | None = None
                     ) -> tuple[GameRules, tuple[str, ...]]:
    """(rules, warnings) for a decision. The strict rules when the config is complete and
    registered; otherwise (unless NLB_STRICT=1) the best rules the config and the state
    allow, with one warning per guess. Raises RulesError only when nothing usable is left.
    `aliases` maps SDK variant IDs onto registered ones (adapter.VARIANT_ALIASES)."""
    cfg = getattr(state, "config", None)
    if aliases and cfg is not None:
        sdk_id = getattr(cfg, "variant", None)
        if isinstance(sdk_id, str) and sdk_id in aliases:
            cfg = _Aliased(cfg, aliases[sdk_id])
    if strict is None:
        strict = strict_mode()
    if not strict:  # a config already known to need guessing skips the failing strict pass
        try:
            guessed = _GUESSED.get(_guess_key(cfg, state))
        except TypeError:
            guessed = None
        if guessed is not None:
            return guessed
    try:
        return rules_from_config(cfg), ()
    except RulesError:
        if strict:
            raise
    raw = tuple(_frozen(getattr(cfg, name, None)) for name in _CONFIG_NAMES)
    hole = len(getattr(state, "cards", ()) or ())
    try:
        r = state.can("raise")
        me = next(s for s in state.seats if s.seat == state.seat)
        all_in = int(me.stack) + int(me.bet)
        betting = "" if r is None else "pot_limit" if int(r.max_amount) < all_in else "no_limit"
    except Exception:
        betting = ""
    seats_seen = len(getattr(state, "seats", ()) or ())
    try:
        result = _lenient(raw, hole, betting, seats_seen, _current_level(state))
    except TypeError:  # unhashable leftovers: skip the cache
        return _lenient.__wrapped__(raw, hole, betting, seats_seen, _current_level(state))
    # Remember it for this config and hole count, but only once the betting structure was
    # actually seen (a raise on offer), so a first guess without evidence isn't frozen in.
    if betting and len(_GUESSED) < 256:
        try:
            _GUESSED[_guess_key(cfg, state)] = result
        except TypeError:
            pass
    return result


_GUESSED: dict = {}  # _guess_key -> (rules, warnings), once guessed with betting evidence


def _guess_key(cfg, state) -> tuple:
    """Everything a guess reads except the betting evidence, which is fixed once seen."""
    return (tuple(getattr(cfg, name, None) for name in _CONFIG_NAMES),
            len(getattr(state, "cards", ()) or ()), len(getattr(state, "seats", ()) or ()),
            _current_level(state))
