"""Tight-aggressive bot for 8-handed NLHE sit-and-go tables.

Each decision has to finish well inside 1 ms, so the bot never simulates.
Preflop play looks up precomputed hand percentiles: Sklansky groups for normal
raising and calling, and heads-up all-in equity for push/fold spots. Postflop
play runs a bitmask classifier that turns the holding into an equity estimate
and compares it with the price.

  Preflop   Raise-or-fold opens that depend on position, value-heavy 3-bets,
            set-mining when stacks are deep, and push/fold once the effective
            stack is short.
  Postflop  Value-bet strong made hands, semi-bluff big draws, c-bet heads-up
            (mostly on dry boards), and call only when equity, discounted for
            the bettor's likely range, beats the pot odds. Big calls get an
            extra survival margin, because points come from finishing place.
  Reads     Per-table VPIP, PFR, shove and postflop-aggression stats. Maniacs
            and always-all-in bots get called lighter, and calling stations
            get bluffed less.
"""

from __future__ import annotations

import random

from bot.state import (
    CALL,
    CHECK,
    FLOP,
    FOLD,
    POST_ANTE,
    PREFLOP,
    RAISE,
    RIVER,
    TURN,
    Action,
    GameState,
    HandResult,
    LegalActions,
    fallback_action,
)

# ---------------------------------------------------------------------------
# Lookup tables. These are built once at import, never during a decision.
# ---------------------------------------------------------------------------

_RANKS = "23456789TJQKA"

# Approximate heads-up all-in equity (%) against a random hand.
_PAIR_EQ = (50.3, 53.7, 57.0, 60.3, 63.3, 66.2, 69.1, 72.1, 75.1, 77.5, 79.9, 82.4, 85.2)
_OFFSUIT_EQ = (  # _OFFSUIT_EQ[hi][lo], offsuit, hi > lo
    (),
    (31.2,),
    (32.5, 34.4),
    (33.9, 35.9, 37.9),
    (34.0, 36.0, 38.0, 40.1),
    (34.6, 36.8, 38.8, 40.8, 42.7),
    (36.8, 37.5, 39.6, 41.6, 43.6, 45.5),
    (39.1, 39.9, 40.6, 42.6, 44.6, 46.5, 48.4),
    (41.7, 42.5, 43.4, 44.2, 46.3, 48.2, 50.0, 51.7),
    (44.4, 45.2, 46.1, 47.1, 47.9, 49.9, 51.7, 53.4, 55.4),
    (47.5, 48.4, 49.3, 50.2, 51.1, 51.9, 53.8, 55.5, 57.4, 58.2),
    (50.6, 51.4, 52.4, 53.3, 54.3, 55.4, 56.3, 58.0, 59.9, 60.6, 61.4),
    (54.6, 55.6, 56.4, 57.7, 57.8, 59.1, 60.1, 60.9, 62.9, 63.6, 64.5, 65.3),
)

# Sklansky-Malmuth starting-hand groups, the classic tight-aggressive ordering.
_SKLANSKY_GROUPS = (
    "AA KK QQ JJ AKs",
    "TT AQs AJs KQs AKo",
    "99 ATs KJs QJs JTs AQo",
    "T9s KQo 88 QTs 98s J9s AJo KTs",
    "77 87s Q9s T8s KJo QJo JTo 76s 97s A9s A8s A7s A6s A5s A4s A3s A2s 65s",
    "66 ATo 55 86s KTo QTo 54s K9s J8s 75s",
    "44 J9o 64s T9o 53s 33 98o 43s 22 K8s K7s K6s K5s K4s K3s K2s T7s Q8s",
    "87o A9o Q9o 76o 42s 32s 96s 85s J8o J7s 65o 54o 74s K9o T8o",
)

# Opening and push widths (fraction of hands) by the number of players left to act behind.
_OPEN_PCT = (0.30, 0.45, 0.40, 0.28, 0.21, 0.17, 0.14, 0.12, 0.11, 0.10)
_PUSH_BASE = (0.60, 0.58, 0.37, 0.27, 0.21, 0.17, 0.145, 0.125, 0.11, 0.10)  # at 10 BB

# Equity against one opponent for a made-hand level of 0.0, 0.5, ..., 6.0.
_LEVEL_EQ = (0.06, 0.12, 0.20, 0.31, 0.42, 0.52, 0.62, 0.70, 0.77, 0.84, 0.89, 0.93, 0.97)


def _hand_key(hi: int, lo: int, suited: bool) -> int:
    return ((hi * 13 + lo) << 1) | (1 if suited and hi != lo else 0)


def _allin_equity(hi: int, lo: int, suited: bool) -> float:
    if hi == lo:
        return _PAIR_EQ[hi]
    eq = _OFFSUIT_EQ[hi][lo]
    if suited:
        eq += 2.0 + 0.15 * (12 - hi)
    return eq


def _playability(hi: int, lo: int, suited: bool) -> float:
    score = _allin_equity(hi, lo, suited)
    if hi != lo:
        if suited:
            score += 0.8
        gap = hi - lo - 1
        score += 1.2 if gap == 0 else 0.6 if gap == 1 else 0.2 if gap == 2 else 0.0
    return score


def _parse_hand(code: str) -> int:
    a, b = _RANKS.index(code[0]), _RANKS.index(code[1])
    return _hand_key(max(a, b), min(a, b), code.endswith("s"))


_GROUP = {}
for _g, _line in enumerate(_SKLANSKY_GROUPS):
    for _code in _line.split():
        _GROUP[_parse_hand(_code)] = _g


def _percentiles(sort_key) -> tuple:
    """Map each hand key to the fraction of the 1326 combos ranked at or above it."""
    entries = []
    for hi in range(13):
        for lo in range(hi + 1):
            if hi == lo:
                entries.append((sort_key(hi, lo, False), 6, _hand_key(hi, lo, False)))
            else:
                entries.append((sort_key(hi, lo, True), 4, _hand_key(hi, lo, True)))
                entries.append((sort_key(hi, lo, False), 12, _hand_key(hi, lo, False)))
    entries.sort(key=lambda e: e[0])
    table = [1.0] * 338
    seen = 0
    for _, combos, key in entries:
        seen += combos
        table[key] = seen / 1326.0
    return tuple(table)


_PCT_ALLIN = _percentiles(lambda h, l, s: -_allin_equity(h, l, s))
_PCT_PLAY = _percentiles(
    lambda h, l, s: (_GROUP.get(_hand_key(h, l, s), len(_SKLANSKY_GROUPS)), -_playability(h, l, s))
)

# Straight detection over 13-bit rank masks (bit r means rank r is present).
_WINDOWS = tuple(sum(1 << r for r in range(t - 4, t + 1)) for t in range(12, 3, -1)) + (
    (1 << 12) | 0b1111,
)
_WINDOW_TOPS = tuple(range(12, 3, -1)) + (3,)
_POP = tuple(bin(m).count("1") for m in range(1 << 13))


def _straight_table() -> tuple:
    table = [-1] * (1 << 13)
    for mask in range(1 << 13):
        if _POP[mask] < 5:
            continue
        for wm, top in zip(_WINDOWS, _WINDOW_TOPS):
            if (mask & wm) == wm:
                table[mask] = top
                break
    return tuple(table)


_STRAIGHT_HIGH = _straight_table()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _open_pct(behind: int, heads_up: bool) -> float:
    if heads_up and behind <= 1:
        return 0.65
    return _OPEN_PCT[behind if behind < len(_OPEN_PCT) else -1]


def _push_pct(behind: int, stack_bb: float) -> float:
    base = _PUSH_BASE[behind if behind < len(_PUSH_BASE) else -1]
    w = base * (10.0 / max(stack_bb, 1.0)) ** 0.7
    return w if w < 1.0 else 1.0


def _level_equity(level: float) -> float:
    x = level * 2.0
    if x <= 0.0:
        return _LEVEL_EQ[0]
    if x >= 12.0:
        return _LEVEL_EQ[12]
    i = int(x)
    return _LEVEL_EQ[i] + (_LEVEL_EQ[i + 1] - _LEVEL_EQ[i]) * (x - i)


def _check_or_fold(legal: LegalActions) -> Action:
    if legal.can_check:
        return Action(CHECK)
    if legal.can_fold:
        return Action(FOLD)
    return fallback_action(legal)


def _check_or_call(legal: LegalActions) -> Action:
    if legal.can_check:
        return Action(CHECK)
    if legal.can_call:
        return Action(CALL)
    return _check_or_fold(legal)


def _raise_to(legal: LegalActions, target: float) -> Action:
    if not legal.can_raise:
        return _check_or_call(legal)
    t = int(target)
    if t < legal.min_raise_to:
        t = legal.min_raise_to
    if t > legal.max_raise_to:
        t = legal.max_raise_to
    return Action(RAISE, t)


def _shove(legal: LegalActions) -> Action:
    if not legal.can_raise:
        return _check_or_call(legal)
    return Action(RAISE, legal.max_raise_to)


def _sized_raise(legal: LegalActions, target: float, stack_total: int, shove_frac: float) -> Action:
    """Raise to `target`, but go all in once that would commit too much of the stack."""
    if not legal.can_raise:
        return _check_or_call(legal)
    if max(target, legal.min_raise_to) >= shove_frac * stack_total:
        return Action(RAISE, legal.max_raise_to)
    return _raise_to(legal, target)


def _players_behind(st: GameState, seat: int) -> int:
    """Players who act after `seat` in the first preflop orbit, up to and including the big blind."""
    n, bb = st.num_seats, st.bb_seat
    if seat == bb:
        return 0
    count, s = 0, seat
    for _ in range(n - 1):
        s = (s + 1) % n
        if st.in_hand[s] or st.stacks[s] > 0:
            count += 1
        if s == bb:
            break
    return count


def _post_rank(st: GameState, seat: int) -> int:
    """Postflop acting order: 0 acts first, larger numbers act later."""
    return (seat - st.button - 1) % st.num_seats


def _last_to_act(st: GameState) -> int:
    n, last = st.num_seats, -1
    for i in range(1, n + 1):
        s = (st.button + i) % n
        if st.in_hand[s] and (st.stacks[s] > 0 or s == st.seat):
            last = s
    return last


# ---------------------------------------------------------------------------
# Postflop hand classifier
# ---------------------------------------------------------------------------

def _analyze(hole, board):
    """Classify a postflop holding.

    Returns (level, outs, board_max_suit, board_connectedness). `level` rates
    the made hand from 0 to 6, compared with what the board already gives
    everyone: 6 is near the nuts, 3 is top pair with a good kicker or an
    overpair, and 0 is air. `outs` is a discounted count of cards that should
    improve the hand to the best one.
    """
    c1, c2 = hole[0], hole[1]
    r1, s1, r2, s2 = c1 >> 2, c1 & 3, c2 >> 2, c2 & 3
    if r1 < r2:
        r1, s1, r2, s2 = r2, s2, r1, s1
    pocket = r1 == r2
    nb = len(board)

    bcnt = [0] * 13
    bsuit = [0, 0, 0, 0]
    bmask = 0
    for c in board:
        r = c >> 2
        bcnt[r] += 1
        bsuit[c & 3] += 1
        bmask |= 1 << r
    acnt = bcnt[:]
    acnt[r1] += 1
    acnt[r2] += 1
    asuit = bsuit[:]
    asuit[s1] += 1
    asuit[s2] += 1
    amask = bmask | (1 << r1) | (1 << r2)

    top = bmask.bit_length() - 1
    second = (bmask ^ (1 << top)).bit_length() - 1
    bmax_suit = max(bsuit)
    board_paired = max(bcnt) >= 2
    sconn = 0
    for wm in _WINDOWS:
        c = _POP[bmask & wm]
        if c > sconn:
            sconn = c
    texture = (0.6 if bmax_suit >= 4 else 0.3 if bmax_suit == 3 else 0.0) + (
        0.5 if sconn >= 4 else 0.25 if sconn == 3 else 0.0
    )

    quad = trips = trips2 = pair1 = bt = bp = -1
    for r in range(12, -1, -1):
        c = acnt[r]
        if c == 4:
            quad = r
        elif c == 3:
            if trips < 0:
                trips = r
            elif trips2 < 0:
                trips2 = r
        elif c == 2 and pair1 < 0:
            pair1 = r
        c = bcnt[r]
        if c == 3 and bt < 0:
            bt = r
        elif c == 2 and bp < 0:
            bp = r

    fsuit = -1
    for s in range(4):
        if asuit[s] >= 5:
            fsuit = s
            break
    sh = _STRAIGHT_HIGH[amask]
    bsh = _STRAIGHT_HIGH[bmask]

    level = -1.0
    mine = -1
    af = bf = 0
    if fsuit >= 0:
        for c in board:
            if (c & 3) == fsuit:
                bf |= 1 << (c >> 2)
        af = bf
        if s2 == fsuit:
            af |= 1 << r2
            mine = r2
        if s1 == fsuit:
            af |= 1 << r1
            mine = r1
        if _STRAIGHT_HIGH[af] > _STRAIGHT_HIGH[bf]:
            level = 6.0  # straight flush
    if level < 0.0 and quad >= 0:
        if bcnt[quad] == 4:  # quads on the board, so only the kicker matters
            level = 4.5 if (r1 == 12 or (quad == 12 and r1 == 11)) else 1.0
        else:
            level = 6.0
    if level < 0.0 and trips >= 0 and (trips2 >= 0 or pair1 >= 0):
        p = trips2 if trips2 > pair1 else pair1
        if bt >= 0 and bp >= 0:  # the board is already a full house
            level = 4.5 if (trips, p) > (bt, bp) else 1.2
        elif bt >= 0:  # trips on the board, and we supply the pair
            hb = (bmask ^ (1 << bt)).bit_length() - 1
            level = 4.6 if (p >= hb and p >= 8) else 3.9
        else:
            level = 5.7 if trips >= bp else 5.2
    if level < 0.0 and fsuit >= 0:
        if mine < 0 or (bsuit[fsuit] >= 5 and mine < (bf & -bf).bit_length() - 1):
            level = 1.0  # the flush is all on the board
        else:
            missing = 0
            for r in range(mine + 1, 13):
                if not (af >> r) & 1:
                    missing += 1
            level = 5.6 if missing == 0 else 5.0 if missing == 1 else 4.5 if missing <= 3 else 4.0
            if bsuit[fsuit] >= 4 and missing:
                level -= 0.4
            if board_paired:
                level -= 0.6
    if level < 0.0 and sh >= 0:
        if sh <= bsh:
            level = 1.0  # the straight is all on the board
        else:
            level = 4.6
            for i in range(len(_WINDOWS)):
                if _WINDOW_TOPS[i] > sh and _POP[bmask & _WINDOWS[i]] >= 3:
                    level -= 0.4  # a bigger straight is still possible
                    break
            if bmax_suit >= 4:
                level -= 1.4
            elif bmax_suit == 3:
                level -= 0.9
            if board_paired:
                level -= 0.5
    if level < 0.0 and trips >= 0:
        if bcnt[trips] == 3:  # trips on the board, so it comes down to kickers
            level = 2.4 if (r1 == 12 or (trips == 12 and r1 == 11)) else 1.4 if r1 >= 10 else 0.8
        elif pocket:  # set
            level = (5.3 if trips == top else 5.0) - 0.8 * texture
        else:  # trips using one hole card
            kicker = r2 if r1 == trips else r1
            level = (4.5 if kicker >= 10 else 4.1) - texture
    if level < 0.0:
        hit1 = bcnt[r1] > 0
        hit2 = bcnt[r2] > 0
        if not pocket and hit1 and hit2:  # two pair using both hole cards
            if bp > r2:
                level = 2.4 - 0.6 * texture  # a higher board pair counterfeits the lower pair
            else:
                level = (4.3 if r1 == top else 3.9) - texture - (0.3 if board_paired else 0.0)
        elif pocket and not hit1:
            if r1 > top:
                level = 3.4 + (0.3 if r1 >= 10 else 0.0)  # overpair
            elif r1 > second:
                level = 2.3
            else:
                level = 1.4
            level -= 0.6 * texture + (0.15 if board_paired else 0.0)
        elif hit1 or hit2:
            pr, kicker = (r1, r2) if hit1 else (r2, r1)
            if pr == top:
                level = 3.2 if kicker >= 11 else 2.95 if kicker >= 9 else 2.7 if kicker >= 6 else 2.45
                if pr == 12:
                    level += 0.1
            elif pr == second:
                level = 2.1 + (0.2 if kicker == 12 else 0.0)
            else:
                level = 1.5
            level -= 0.6 * texture + (0.15 if board_paired else 0.0)
        else:
            level = 1.0 if r1 == 12 else 0.6 if r1 > top else 0.2
    if level < 0.0:
        level = 0.0

    outs = 0.0
    if nb < 5 and level < 4.4:
        fd = 0.0
        if fsuit < 0:
            for s in range(4):
                if asuit[s] == 4 and bsuit[s] <= 3 and (s1 == s or s2 == s):
                    if bsuit[s] == 2:
                        fd = 9.0
                    else:  # one-card flush draw, only worth much if it is to the nuts
                        mine = r1 if s1 == s else r2
                        sm = 1 << mine
                        for c in board:
                            if (c & 3) == s:
                                sm |= 1 << (c >> 2)
                        higher = 0
                        for r in range(mine + 1, 13):
                            if not (sm >> r) & 1:
                                higher += 1
                        fd = 8.0 if higher == 0 else 5.5 if higher == 1 else 4.0
                    if board_paired:
                        fd -= 1.5
                    break
        n_sr = 0
        if sh <= bsh:
            for r in range(13):
                bit = 1 << r
                if not amask & bit and _STRAIGHT_HIGH[amask | bit] >= 0 and _STRAIGHT_HIGH[bmask | bit] < 0:
                    n_sr += 1
            if n_sr > 2:
                n_sr = 2
        per = 3.0 if (fd > 0.0 or bmax_suit >= 3) else 3.5 if bmax_suit == 2 else 4.0
        if board_paired:
            per *= 0.8
        if level <= 1.0:
            extra = ((r1 > top) + (r2 > top)) * (2.5 if nb == 3 else 2.0) * (0.6 if board_paired else 1.0)
        elif pocket:
            extra = 1.8
        elif level < 3.5:
            extra = 2.5
        else:
            extra = 0.0
        outs = fd + per * n_sr + extra
        if outs > 18.0:
            outs = 18.0
    return level, outs, bmax_suit, sconn


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------

class Bot:
    """Tight-aggressive NLHE sit-and-go bot (see the module docstring)."""

    def __init__(self, *args, **kwargs) -> None:
        self._rng = random.Random(0x5EED)
        self._hand = -1  # hand number last seen in act()
        self._start = None  # starting stacks for that hand
        self._last_actions = ()
        self._recorded = -1  # last hand added to the opponent stats
        self._prev_stacks = None
        self._reset_stats(8)

    # -- engine hooks -------------------------------------------------------

    def new_table(self, seat: int, num_seats: int) -> None:
        try:
            self._rng = random.Random(0x5EED + 7919 * int(seat))
            self._hand = -1
            self._start = None
            self._last_actions = ()
            self._recorded = -1
            self._prev_stacks = None
            self._reset_stats(int(num_seats))
        except Exception:
            pass

    def act(self, state: GameState) -> Action:
        try:
            if state.num_seats != self._n:
                self._reset_stats(state.num_seats)
            self._observe(state)
            if len(state.hole) < 2:
                return _check_or_fold(state.legal)
            if state.street == PREFLOP:
                return self._preflop(state)
            return self._postflop(state)
        except Exception:
            return fallback_action(state.legal)

    def hand_over(self, result: HandResult) -> None:
        try:
            hn = result.hand_number
            if hn != self._recorded:
                self._recorded = hn
                mine = hn == self._hand
                actions = result.actions or (self._last_actions if mine else ())
                start = self._start if mine else self._prev_stacks
                if start is not None and len(start) != self._n:
                    start = None
                if actions:
                    self._record(actions, start)
            self._prev_stacks = tuple(result.stacks_after)
        except Exception:
            pass

    # -- opponent stats -----------------------------------------------------

    def _reset_stats(self, n: int) -> None:
        self._n = n
        self._dealt = [0] * n
        self._vpip = [0] * n
        self._pfr = [0] * n
        self._shoves = [0] * n
        self._post_aggr = [0] * n
        self._post_acts = [0] * n

    def _observe(self, st: GameState) -> None:
        self._last_actions = st.actions
        if st.hand_number == self._hand:
            return
        self._hand = st.hand_number
        n = st.num_seats
        antes = [0] * n
        street_top = [[0] * n for _ in range(4)]
        for a in st.actions:
            s = a.seat
            if not 0 <= s < n:
                continue
            if a.kind == POST_ANTE:
                antes[s] += a.amount
            elif 0 <= a.street < 4 and a.amount > street_top[a.street][s]:
                street_top[a.street][s] = a.amount
        self._start = tuple(
            st.stacks[s] + antes[s] + street_top[0][s] + street_top[1][s]
            + street_top[2][s] + street_top[3][s]
            for s in range(n)
        )

    def _record(self, actions, start) -> None:
        n = self._n
        seen = [False] * n
        vpip = [False] * n
        pfr = [False] * n
        shove = [False] * n
        ante = [0] * n
        for a in actions:
            s = a.seat
            if not 0 <= s < n:
                continue
            seen[s] = True
            kind = a.kind
            if a.street == PREFLOP:
                if kind == RAISE:
                    vpip[s] = pfr[s] = True
                    if start is not None and 0 < start[s] <= ante[s] + a.amount:
                        shove[s] = True
                elif kind == CALL:
                    vpip[s] = True
                elif kind == POST_ANTE:
                    ante[s] += a.amount
            elif kind == RAISE:
                self._post_aggr[s] += 1
                self._post_acts[s] += 1
            elif kind == CALL or kind == CHECK or kind == FOLD:
                self._post_acts[s] += 1
        for s in range(n):
            if seen[s]:
                self._dealt[s] += 1
                self._vpip[s] += vpip[s]
                self._pfr[s] += pfr[s]
                self._shoves[s] += shove[s]

    def _rate(self, counts, seat: int, prior: float, weight: float) -> float:
        if 0 <= seat < self._n:
            return (counts[seat] + prior * weight) / (self._dealt[seat] + weight)
        return prior

    def _aggression(self, seat: int) -> float:
        if 0 <= seat < self._n:
            return (self._post_aggr[seat] + 1.8) / (self._post_acts[seat] + 6.0)
        return 0.3

    def _loose_table(self, seats) -> bool:
        total = 0.0
        for s in seats:
            total += self._rate(self._vpip, s, 0.25, 8.0)
        return total > 0.5 * len(seats)

    def _raise_range(self, st: GameState, seat: int, nr: int, allin: bool, total_bb: float, bb: int) -> float:
        """Estimate how wide (as a top fraction of hands) `seat` is raising here."""
        pfr = self._rate(self._pfr, seat, 0.15, 8.0)
        if nr == 1:
            behind = _players_behind(st, seat)
            if allin:
                w = _push_pct(behind, total_bb)
            else:
                w = _open_pct(behind, st.players_left <= 2)
                if st.bets[seat] >= 5 * bb:
                    w *= 0.7
            w = max(w, 0.9 * pfr)
        elif nr == 2:
            w = max(0.12 if (allin and total_bb <= 25) else 0.06, 0.3 * pfr)
        else:
            w = max(0.05 if allin else 0.035, 0.15 * pfr)
        if allin:
            w = max(w, 1.25 * self._rate(self._shoves, seat, 0.03, 10.0))
        return 0.015 if w < 0.015 else 1.0 if w > 1.0 else w

    # -- sizing --------------------------------------------------------------

    def _bet(self, st: GameState, frac: float, value: bool):
        """Bet or raise by `frac` of the pot. A bluff that would commit the stack returns None."""
        legal = st.legal
        if not legal.can_raise:
            return None
        me = st.seat
        my_total = st.stacks[me] + st.bets[me]
        t = int(max(st.bets) + frac * (st.pot + legal.call_amount))
        if t < legal.min_raise_to:
            t = legal.min_raise_to
        if t >= 0.45 * my_total:
            if not value:
                return None
            t = legal.max_raise_to
        if t > legal.max_raise_to:
            t = legal.max_raise_to
        return Action(RAISE, t)

    # -- preflop ---------------------------------------------------------------

    def _preflop(self, st: GameState) -> Action:
        legal = st.legal
        n, me = st.num_seats, st.seat
        stacks, bets, in_hand = st.stacks, st.bets, st.in_hand
        bb = st.big_blind if st.big_blind > 0 else 1
        c1, c2 = st.hole[0], st.hole[1]
        hi, lo = c1 >> 2, c2 >> 2
        if hi < lo:
            hi, lo = lo, hi
        suited = (c1 & 3) == (c2 & 3)
        key = _hand_key(hi, lo, suited)
        p_play = _PCT_PLAY[key]  # Sklansky rank, used for normal raising and calling
        p_eq = _PCT_ALLIN[key]  # all-in equity rank, used for push/fold and call-offs
        pocket = hi == lo

        my_stack = stacks[me]
        my_total = my_stack + bets[me]
        max_bet = max(bets)
        call = legal.call_amount if legal.can_call else 0
        pot = st.pot
        heads_up = st.players_left <= 2

        biggest = 0
        for s in range(n):
            if s != me and in_hand[s]:
                t = stacks[s] + bets[s]
                if t > biggest:
                    biggest = t
        eff_bb = (min(my_total, biggest) if biggest else my_total) / bb

        raisers = []
        limpers = callers = 0
        limper = -1
        i_limped = False
        for a in st.actions:
            if a.street != PREFLOP:
                continue
            if a.kind == RAISE:
                raisers.append(a.seat)
                callers = 0
            elif a.kind == CALL:
                if raisers:
                    callers += 1
                else:
                    limpers += 1
                    limper = a.seat
                    if a.seat == me:
                        i_limped = True

        # ---- unopened pot ----
        if not raisers:
            if legal.can_check:
                # Big blind in a limped pot: raise to isolate with a strong hand, otherwise see a free flop.
                if limpers == 1 and limper == st.sb_seat:
                    w = 0.35
                else:
                    w = 0.20 if limpers <= 1 else 0.12 if limpers == 2 else 0.08
                if eff_bb <= 13:
                    if p_eq <= min(0.6, w * (13.0 / max(eff_bb, 1.0)) ** 0.5):
                        return _shove(legal)
                    return Action(CHECK)
                if p_play <= w:
                    return _sized_raise(legal, bb * (3.5 + limpers), my_total, 0.35)
                return Action(CHECK)

            behind = _players_behind(st, me)
            # Blinds, antes and limps already in the pot make stealing more profitable.
            boost = min(1.4, 1.0 + 0.25 * max(0.0, pot / bb - 1.5 - limpers))
            if eff_bb <= 13 or (limpers and eff_bb <= 18):
                if p_eq <= _push_pct(behind, eff_bb) * boost * 0.6 ** limpers:
                    return _shove(legal)
                return _check_or_fold(legal)
            if p_play <= _open_pct(behind, heads_up) * boost * 0.55 ** limpers:
                if behind == 1:
                    size = 2.5 if heads_up else 3.0
                else:
                    size = 2.2 if eff_bb <= 40 else 2.5
                return _sized_raise(legal, bb * (size + limpers), my_total, 0.35)
            if limpers and pocket and eff_bb >= 30 and call <= bb:
                return _check_or_call(legal)  # cheap overlimp, hoping to flop a set
            return _check_or_fold(legal)

        # ---- facing a raise ----
        last = raisers[-1]
        if last == me or legal.can_check:
            return _check_or_call(legal)
        nr = len(raisers)
        raiser_total = stacks[last] + bets[last]
        raiser_allin = stacks[last] == 0
        w = self._raise_range(st, last, nr, raiser_allin, raiser_total / bb, bb)

        eff = min(my_total, raiser_total)
        risk = max(call, eff - bets[me])
        final_pot = pot + risk + max(0, eff - bets[last])
        needed = risk / final_pot if final_pot > 0 else 0.5

        if raiser_allin or call >= 0.33 * my_stack or eff / bb <= 18:
            # Shove-or-fold spot: keep playing only with the part of the opponent's
            # range we beat often enough for the price.
            f = 0.35 + (0.5 - needed) * 2.5
            f = 0.15 if f < 0.15 else 1.6 if f > 1.6 else f
            if not raiser_allin:
                f *= 1.25  # fold equity
            if callers:
                f *= 0.65 ** callers  # someone has already called
            if not heads_up and risk >= 0.5 * my_total:
                f *= 0.9  # points come from finishing place, so protect the stack
            if p_eq > max(0.012, w * f):
                return _check_or_fold(legal)
            if raiser_allin or not legal.can_raise:
                return _check_or_call(legal)
            target = raiser_total if my_total > 1.5 * raiser_total else legal.max_raise_to
            return _raise_to(legal, target)

        in_pos = _post_rank(st, me) > _post_rank(st, last)
        if me not in raisers:
            if nr == 1:
                three_bet = p_play <= max(0.025, w * 0.2)
                if (not three_bet and suited and hi == 12 and lo <= 3 and w >= 0.25
                        and not callers and self._rng.random() < 0.35):
                    three_bet = True  # occasional 3-bet bluff with a suited wheel ace
                if three_bet:
                    size = (3.0 if in_pos else 3.7) + callers
                    return _sized_raise(legal, max_bet * size, my_total, 0.35)
                call_th = w * (0.5 if in_pos else 0.35)
                if me == st.bb_seat and max_bet <= 3.5 * bb:
                    call_th = max(call_th, w * 0.65)  # we close the action at a good price
                if i_limped:
                    call_th *= 1.2
                call_th *= 0.85 ** callers
                if p_play <= call_th and call <= 0.1 * my_stack:
                    return _check_or_call(legal)
                if pocket and call * 15 <= min(my_stack, stacks[last]) and call <= 0.07 * my_stack:
                    return _check_or_call(legal)  # set-mining with deep stacks
                return _check_or_fold(legal)
            # Facing a 3-bet or bigger without having raised: continue with premiums only.
            if p_play <= max(0.018, w * 0.3):
                return _sized_raise(legal, max_bet * 2.3, my_total, 0.35)
            if p_play <= max(0.03, w * 0.5) and call <= 0.15 * my_stack:
                return _check_or_call(legal)
            return _check_or_fold(legal)

        # We raised earlier and someone re-raised us.
        if p_play <= max(0.022, w * 0.35):
            if eff / bb <= 60:
                return _shove(legal)
            return _sized_raise(legal, max_bet * 2.3, my_total, 0.35)
        if p_play <= max(0.04, w * 0.7) and call <= 0.25 * my_stack:
            return _check_or_call(legal)
        if pocket and call * 12 <= min(my_stack, stacks[last]) and call <= 0.1 * my_stack:
            return _check_or_call(legal)
        return _check_or_fold(legal)

    # -- postflop ------------------------------------------------------------

    def _postflop(self, st: GameState) -> Action:
        legal = st.legal
        n, me, street = st.num_seats, st.seat, st.street
        stacks, in_hand = st.stacks, st.in_hand
        pot = st.pot if st.pot > 0 else 1
        my_stack = stacks[me]

        opps = [s for s in range(n) if s != me and in_hand[s]]
        if not opps:
            return _check_or_call(legal)
        active = [s for s in opps if stacks[s] > 0]
        k = len(opps)

        level, outs, flushy, conn = _analyze(st.hole, st.board)
        e_made = _level_equity(level)
        if street >= RIVER:
            d1 = d2 = 0.0
        elif street == FLOP:
            d1 = outs / 47.0
            d2 = 1.0 - (1.0 - outs / 47.0) * (1.0 - outs / 46.0)
        else:
            d1 = d2 = outs / 46.0
        if k > 1:
            e_made **= 1.0 + 0.7 * (k - 1)
            shrink = 0.92 ** (k - 1)
            d1 *= shrink
            d2 *= shrink

        pf_raiser = bettor = -1
        street_raises = opp_prev_aggr = 0
        my_prev_bet = False
        for a in st.actions:
            if a.kind != RAISE:
                continue
            if a.street == PREFLOP:
                pf_raiser = a.seat
            elif a.street == street:
                street_raises += 1
                bettor = a.seat
            elif a.seat == me:
                if a.street == street - 1:
                    my_prev_bet = True
            else:
                opp_prev_aggr += 1

        stations = self._loose_table(active or opps)
        rand = self._rng.random

        # ---- no bet to us ----
        if legal.can_check:
            n_act = len(active)
            if n_act == 0:
                return Action(CHECK)
            value_th = (0.64 if street == RIVER else 0.60) - (0.06 if stations else 0.0)
            if e_made >= value_th:
                frac = 0.55 if street == FLOP else 0.65 if street == TURN else 0.7
                if level >= 4.5 and street != RIVER and (flushy >= 2 or conn >= 3):
                    frac += 0.1  # make the draws pay
                act = self._bet(st, frac, True)
                return act if act is not None else Action(CHECK)
            if street != RIVER and d2 >= 0.30 and n_act <= 2 and rand() < (0.7 if n_act == 1 else 0.45):
                act = self._bet(st, 0.6, d2 >= 0.40)  # semi-bluff
                if act is not None:
                    return act
            p, size = 0.0, 0.5
            if street == FLOP and pf_raiser == me and n_act <= 2:  # continuation bet
                p = 0.75 if n_act == 1 else 0.45
                if flushy >= 2:
                    p *= 0.8
                if conn >= 3:
                    p *= 0.7
                if e_made >= 0.40:
                    p = max(p, 0.6)
            elif street == TURN and my_prev_bet and n_act == 1:  # second barrel
                p, size = (0.55 if (e_made >= 0.42 or d1 >= 0.15) else 0.25), 0.6
            elif street == RIVER and my_prev_bet and n_act == 1 and level <= 1.0:
                p, size = 0.2, 0.65  # occasional river bluff
            elif street != RIVER and n_act == 1 and _last_to_act(st) == me:  # stab after a check
                p = 0.45 if e_made >= 0.35 else 0.3
            if stations:
                p *= 0.5
            if p > 0.0 and rand() < p:
                act = self._bet(st, size, False)
                if act is not None:
                    return act
            return Action(CHECK)

        # ---- facing a bet ----
        call = legal.call_amount
        needed = call / (pot + call)
        if bettor < 0 or bettor == me:
            bettor = active[0] if active else opps[0]
        af = self._aggression(bettor)
        frac = call / max(1.0, pot - call)
        # Bigger bets, raise wars and earlier barrels mean a stronger range,
        # unless the bettor has shown they fire with anything.
        d = 0.10 + 0.07 * min(frac, 2.0) + 0.18 * max(0, street_raises - 1) + 0.04 * min(opp_prev_aggr, 3)
        d *= min(1.3, max(0.35, 1.35 - 1.3 * af))
        e_m = e_made - 2.2 * d * e_made * (1.0 - e_made)
        if e_m < 0.01:
            e_m = 0.01
        all_in_call = call >= my_stack
        if street == FLOP:
            draw = d2 if all_in_call else min(d2, 1.4 * d1)
        elif street == TURN:
            draw = d1 if (all_in_call or my_stack - call < pot) else 1.25 * d1
        else:
            draw = 0.0
        e = 1.0 - (1.0 - e_m) * (1.0 - draw)

        if e_m >= (0.80 if street_raises <= 1 else 0.88) and level >= 4.0:
            act = self._bet(st, 0.7, True)  # raise for value
            if act is not None:
                return act
        if (street == FLOP and street_raises == 1 and len(active) == 1 and stacks[bettor] > 0
                and d2 >= 0.40 and rand() < 0.5):
            act = self._bet(st, 0.7, True)  # raise with a combo draw
            if act is not None:
                return act
        margin = 0.0 if street == RIVER else 0.02
        if call >= 0.4 * my_stack and st.players_left > 2:
            margin += 0.04  # survival margin on stack-threatening calls
        if e >= needed + margin:
            if legal.can_raise and active and e >= 0.45 and my_stack - call <= 0.25 * (pot + call):
                return _shove(legal)  # don't call and leave only a few chips behind
            return _check_or_call(legal)
        return _check_or_fold(legal)
