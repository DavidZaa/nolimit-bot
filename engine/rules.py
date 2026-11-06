"""Tournament rules as data, so a new variant or blind schedule is a config change.

Anything we are assuming rather than reading from the official rules is marked
PLACEHOLDER(kit): the starter kit on event day replaces those values.
"""

from __future__ import annotations

from dataclasses import dataclass

from bot.icm import PAYOUTS

# Hand categories. The numbers are identities, not strengths: a variant's
# category_order decides which beats which.
HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = (
    "high card", "pair", "two pair", "trips", "straight",
    "flush", "full house", "quads", "straight flush",
)
STANDARD_ORDER = (HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH)


@dataclass(frozen=True)
class Variant:
    name: str
    ranks: tuple[int, ...] = tuple(range(13))  # which ranks are in the deck (0 = deuce)
    hole_cards: int = 2
    board_schedule: tuple[int, ...] = (3, 1, 1)  # cards dealt on flop, turn, river
    category_order: tuple[int, ...] = STANDARD_ORDER  # weakest first
    hole_cards_used: int | None = None  # None: best five of all cards; 2: Omaha
    pot_limit: bool = False
    preflop_board: int = 0  # board cards dealt face up before preflop betting (Courchevel: 1)

    @property
    def board_size(self) -> int:
        return self.preflop_board + sum(self.board_schedule)

    def deck(self) -> list[int]:
        return [r * 4 + s for r in self.ranks for s in range(4)]


NLHE = Variant("nlhe")
# Triton-style short deck: 6-A, flush beats full house, trips beat straight.
# PLACEHOLDER(kit): some rule sets (PokerKit's included) keep straight > trips.
SHORT_DECK = Variant(
    "short_deck",
    ranks=tuple(range(4, 13)),
    category_order=(HIGH_CARD, PAIR, TWO_PAIR, STRAIGHT, TRIPS, FULL_HOUSE, FLUSH, QUADS, STRAIGHT_FLUSH),
)
PLO = Variant("plo", hole_cards=4, hole_cards_used=2, pot_limit=True)
VARIANTS = {v.name: v for v in (NLHE, SHORT_DECK, PLO)}

VARIANT_SPEC_HELP = (
    "a preset (nlhe, short_deck, plo) or key=value pairs on top of one, e.g. "
    "'base=nlhe,hole=3' or 'base=plo,hole=5' or 'ranks=6-A,flush_over_boat=1'. "
    "Keys: base, hole, use (cards that must come from the hand; 'any' for hold'em), "
    "ranks (low-high, e.g. 2-A), board (street sizes, e.g. 3+1+1), up (board cards face up "
    "before preflop betting, e.g. Courchevel is base=plo,hole=5,up=1,board=2+1+1), pot_limit (0/1), "
    "flush_over_boat (0/1), trips_over_straight (0/1)"
)


def parse_variant(spec: str) -> Variant:
    """A preset by name, or a preset adjusted by key=value pairs (see VARIANT_SPEC_HELP)."""
    spec = spec.strip()
    if spec in VARIANTS:
        return VARIANTS[spec]
    fields = {}
    for part in spec.split(","):
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"unknown variant {spec!r}: {VARIANT_SPEC_HELP}")
        fields[key.strip()] = value.strip()
    base = VARIANTS[fields.pop("base", "nlhe")]
    kw = {"name": spec}
    if "hole" in fields:
        kw["hole_cards"] = int(fields.pop("hole"))
    if "use" in fields:
        use = fields.pop("use")
        kw["hole_cards_used"] = None if use in ("any", "0") else int(use)
    if "ranks" in fields:
        lo, _, hi = fields.pop("ranks").upper().partition("-")
        chars = "23456789TJQKA"
        kw["ranks"] = tuple(range(chars.index(lo), chars.index(hi) + 1))
    if "board" in fields:
        kw["board_schedule"] = tuple(int(x) for x in fields.pop("board").split("+"))
    if "up" in fields:
        kw["preflop_board"] = int(fields.pop("up"))
    if "pot_limit" in fields:
        kw["pot_limit"] = fields.pop("pot_limit") == "1"
    order = list(base.category_order)
    for key, (weaker, stronger) in (("flush_over_boat", (FULL_HOUSE, FLUSH)),
                                    ("trips_over_straight", (STRAIGHT, TRIPS))):
        if key in fields:
            want = fields.pop(key) == "1"
            i, j = order.index(weaker), order.index(stronger)
            if want != (i < j):
                order[i], order[j] = order[j], order[i]
    kw["category_order"] = tuple(order)
    if fields:
        raise ValueError(f"unknown variant keys {sorted(fields)}: {VARIANT_SPEC_HELP}")
    return Variant(**{**base.__dict__, **kw})


@dataclass(frozen=True)
class BlindSchedule:
    levels: tuple[tuple[int, int, int], ...]  # (small blind, big blind, ante) per level
    hands_per_level: int = 20  # the published default

    def level_for(self, hand_number: int) -> tuple[int, int, int]:
        i = (hand_number - 1) // self.hands_per_level
        return self.levels[min(i, len(self.levels) - 1)]


# Published defaults: start 10/20, up every 20 hands, 250-hand cap. The level
# sizes aren't published (PLACEHOLDER(kit): state.config has them). These two
# bracket the plausible range so strategy gets tuned against both.
DOUBLING = BlindSchedule(tuple((10 * 2**i, 20 * 2**i, 0) for i in range(8)))
X1_5 = BlindSchedule(tuple((bb // 2, bb, 0) for bb in (20, 30, 50, 70, 100, 150, 230, 350)))
SCHEDULES = {"doubling": DOUBLING, "x1.5": X1_5}

SCHEDULE_SPEC_HELP = (
    "a preset (doubling, x1.5), the levels as sb/bb[/ante] separated by commas "
    "(e.g. '10/20,15/30,25/50/5'; the last level repeats), or a growth rule "
    "'bb=20,x=1.5[,ante=0.1][,levels=12]' (ante as a fraction of the big blind)"
)


def parse_schedule(spec: str, hands_per_level: int | None = None) -> BlindSchedule:
    """A preset, an explicit level list, or a growth rule (see SCHEDULE_SPEC_HELP)."""
    spec = spec.strip()
    if spec in SCHEDULES:
        levels = SCHEDULES[spec].levels
    elif "/" in spec:
        levels = []
        for part in spec.split(","):
            nums = [int(x) for x in part.strip().split("/")]
            if len(nums) not in (2, 3):
                raise ValueError(f"bad level {part!r}: {SCHEDULE_SPEC_HELP}")
            levels.append((nums[0], nums[1], nums[2] if len(nums) == 3 else 0))
        levels = tuple(levels)
    else:
        fields = dict(part.strip().split("=", 1) for part in spec.split(","))
        bb, growth = float(fields.pop("bb")), float(fields.pop("x", 2))
        ante, count = float(fields.pop("ante", 0)), int(fields.pop("levels", 12))
        if fields:
            raise ValueError(f"unknown schedule keys {sorted(fields)}: {SCHEDULE_SPEC_HELP}")
        levels = []
        for i in range(count):
            big = int(round(bb * growth**i))
            levels.append((big // 2, big, int(round(big * ante))))
        levels = tuple(levels)
    return BlindSchedule(levels, hands_per_level or 20)

# Points by finishing place. One setting, in bot/icm.py, so the simulator
# always scores by the same points the bot plays for.
PLACEHOLDER_PAYOUTS = PAYOUTS


@dataclass(frozen=True)
class TableConfig:
    variant: Variant = NLHE
    schedule: BlindSchedule = DOUBLING
    payouts: tuple[float, ...] = PLACEHOLDER_PAYOUTS
    seats: int = 8
    starting_stack: int = 2000
    max_hands: int = 250  # the published hand cap
    time_limit_ns: int = 1_000_000  # the official 1 ms per decision
    enforce_time: bool = True  # treat slow decisions exactly like the tournament does

    def __post_init__(self):
        v = self.variant
        needed = self.seats * v.hole_cards + v.board_size
        if needed > len(v.deck()):
            raise ValueError(f"{v.name}: {self.seats} seats need {needed} cards, the deck has {len(v.deck())}")
