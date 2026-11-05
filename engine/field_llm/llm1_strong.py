"""Tournament bot for 8-max no-limit hold'em sit-and-gos (standard library only).

act() never simulates. Each decision is table lookups plus O(len(actions))
arithmetic, which typically takes 30-150 microseconds on CPython.

Preflop   A 169-hand table of heads-up equity against a random hand gives both a
          hand ranking and a quick "equity against a top-x% range" estimate.
          Short stacks play push/fold using an explicit chip-EV formula with a
          call probability for each opponent. Deeper stacks use opening ranges
          scaled by position, then 3-bet, call or fold based on equity against
          the raiser's estimated range.
Postflop  A 7-card evaluator, a hand classifier (pair rank and kicker, set vs.
          trips, how close flushes and straights are to the nuts, board
          texture) and an outs counter give a showdown-strength estimate. That
          estimate is reduced for extra opponents and for the size and number
          of bets faced, then compared with pot odds (plus implied odds for
          draws).
Opponents Per-seat VPIP, PFR, shove rate, all-in call rate, fold-to-raise,
          postflop aggression and fold-to-bet. Each stat is pulled toward a
          prior, so play is sensible from hand one and adapts as the table
          shows its tendencies.
Survival  Calling off a big share of the stack needs extra equity (an ICM-style
          premium). The premium goes away heads-up.
"""

import math
import random

from bot.state import (
    CALL, CHECK, FLOP, FOLD, POST_ANTE, POST_BB, POST_SB, PREFLOP, RAISE,
    RANK_CHARS, RIVER, TURN, Action, fallback_action,
)

# --------------------------------------------------------------------------
# Bit tables and a direct 7-card evaluator
# --------------------------------------------------------------------------

_POP = [bin(i).count("1") for i in range(1 << 13)]
_WINDOWS = [(0x1F << (h - 4), h) for h in range(12, 3, -1)]  # ace-high down
_WINDOWS.append(((1 << 12) | 0xF, 3))                          # the wheel


def _straight_high(mask):
    for w, h in _WINDOWS:
        if mask & w == w:
            return h
    return -1


def _max_window(mask):
    """Largest number of distinct ranks inside any single straight window."""
    best = 0
    for w, _ in _WINDOWS:
        c = _POP[mask & w]
        if c > best:
            best = c
    return best


def _evaluate(cards):
    """Best hand among 1-7 cards as a comparable tuple, category first (0-8)."""
    rc = [0] * 13
    sm = [0, 0, 0, 0]
    for c in cards:
        r = c >> 2
        rc[r] += 1
        sm[c & 3] |= 1 << r
    flush = 0
    for m in sm:
        if _POP[m] >= 5:
            h = _straight_high(m)
            if h >= 0:
                return (8, h)
            flush = m
            break
    quad = -1
    trips, pairs, singles = [], [], []
    for r in range(12, -1, -1):
        k = rc[r]
        if k == 1:
            singles.append(r)
        elif k == 2:
            pairs.append(r)
        elif k == 3:
            trips.append(r)
        elif k == 4:
            quad = r
    if quad >= 0:
        rest = [r for r in range(12, -1, -1) if 0 < rc[r] < 4]
        return (7, quad, rest[0] if rest else -1)
    if trips and (len(trips) > 1 or pairs):
        p = trips[1] if len(trips) > 1 else -1
        if pairs and pairs[0] > p:
            p = pairs[0]
        return (6, trips[0], p)
    if flush:
        return (5,) + tuple([r for r in range(12, -1, -1) if (flush >> r) & 1][:5])
    h = _straight_high(sm[0] | sm[1] | sm[2] | sm[3])
    if h >= 0:
        return (4, h)
    if trips:
        return (3, trips[0]) + tuple(singles[:2])
    if len(pairs) >= 2:
        k = pairs[2] if len(pairs) > 2 else -1
        if singles and singles[0] > k:
            k = singles[0]
        return (2, pairs[0], pairs[1], k)
    if pairs:
        return (1, pairs[0]) + tuple(singles[:3])
    return (0,) + tuple(singles[:5])


# --------------------------------------------------------------------------
# Preflop tables: heads-up all-in equity vs. a random hand (percent)
# --------------------------------------------------------------------------

_PF_TEXT = """
AA 85.3 KK 82.4 QQ 79.9 JJ 77.5 TT 75.1 99 72.1 88 69.1 77 66.2 66 63.3
55 60.3 44 57.0 33 53.7 22 50.3
AKs 67.0 AQs 66.1 AJs 65.4 ATs 64.7 A9s 63.0 A8s 62.1 A7s 61.1 A6s 60.0
A5s 59.9 A4s 58.9 A3s 58.0 A2s 57.0
AKo 65.4 AQo 64.5 AJo 63.6 ATo 62.9 A9o 60.9 A8o 60.1 A7o 59.1 A6o 57.8
A5o 57.7 A4o 56.4 A3o 55.6 A2o 54.6
KQs 63.4 KJs 62.6 KTs 61.9 K9s 60.0 K8s 58.5 K7s 57.8 K6s 56.8 K5s 55.8
K4s 54.7 K3s 53.8 K2s 52.9
KQo 61.4 KJo 60.6 KTo 59.9 K9o 58.0 K8o 56.3 K7o 55.4 K6o 54.3 K5o 53.3
K4o 52.1 K3o 51.2 K2o 50.2
QJs 60.3 QTs 59.5 Q9s 57.9 Q8s 56.2 Q7s 54.5 Q6s 53.8 Q5s 52.9 Q4s 51.7
Q3s 50.7 Q2s 49.9
QJo 58.2 QTo 57.4 Q9o 55.5 Q8o 53.8 Q7o 51.9 Q6o 51.1 Q5o 50.2 Q4o 49.0
Q3o 47.9 Q2o 47.0
JTs 57.5 J9s 55.8 J8s 54.2 J7s 52.4 J6s 50.8 J5s 50.0 J4s 49.0 J3s 47.9 J2s 47.1
JTo 55.4 J9o 53.4 J8o 51.7 J7o 49.9 J6o 48.2 J5o 47.3 J4o 46.1 J3o 45.0 J2o 44.0
T9s 54.3 T8s 52.6 T7s 51.0 T6s 49.2 T5s 47.2 T4s 46.4 T3s 45.5 T2s 44.7
T9o 52.0 T8o 50.2 T7o 48.4 T6o 46.5 T5o 44.4 T4o 43.5 T3o 42.6 T2o 41.7
98s 51.1 97s 49.5 96s 47.7 95s 45.9 94s 43.8 93s 43.2 92s 42.3
98o 48.8 97o 46.9 96o 44.9 95o 42.9 94o 40.7 93o 40.0 92o 39.1
87s 48.2 86s 46.5 85s 44.8 84s 42.7 83s 40.8 82s 40.3
87o 45.7 86o 43.9 85o 42.0 84o 39.7 83o 37.5 82o 37.0
76s 45.7 75s 43.8 74s 41.8 73s 40.0 72s 38.1
76o 42.9 75o 41.1 74o 38.9 73o 36.9 72o 34.6
65s 43.2 64s 41.4 63s 39.4 62s 37.5 65o 40.4 64o 38.5 63o 36.5 62o 34.4
54s 41.1 53s 39.3 52s 37.5 54o 38.2 53o 36.3 52o 34.4
43s 38.0 42s 36.3 43o 35.0 42o 33.2 32s 35.1 32o 32.0
"""


def _pf_index(r1, r2, suited):
    """13x13 grid index: pairs on the diagonal, suited when row > col."""
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return r1 * 14
    return r1 * 13 + r2 if suited else r2 * 13 + r1


def _build_equity():
    eq = [0.0] * 169
    tok = _PF_TEXT.split()
    for i in range(0, len(tok) - 1, 2):
        name = tok[i]
        a, b = RANK_CHARS.find(name[0]), RANK_CHARS.find(name[1])
        if a >= 0 and b >= 0:
            eq[_pf_index(a, b, name[2:3] == "s")] = float(tok[i + 1]) / 100.0
    for i in range(169):  # safety net only; the table above is complete
        if eq[i] == 0.0:
            r, c = divmod(i, 13)
            if r == c:
                eq[i] = 0.50 + 0.029 * r
            else:
                hi, lo = (r, c) if r > c else (c, r)
                eq[i] = 0.30 + 0.022 * hi + 0.009 * lo + (0.02 if r > c else 0.0)
    return eq


_EQ = _build_equity()
_COMBOS = [6 if i // 13 == i % 13 else (4 if i // 13 > i % 13 else 12) for i in range(169)]


def _deep_score(i):
    """Playability ranking for deep-stack opens: favours suited/connected hands."""
    r, c = divmod(i, 13)
    e = _EQ[i]
    if r == c:
        return e + 0.012
    suited = r > c
    hi, lo = (r, c) if suited else (c, r)
    gap = hi - lo - 1
    if suited:
        e += 0.012
        if lo >= 3 and gap == 0:
            e += 0.015
        elif lo >= 3 and gap == 1:
            e += 0.008
        if hi == 12:
            e += 0.005
    else:
        if gap == 0 and lo >= 6:
            e += 0.006
        if hi >= 10 and lo < 7:
            e -= 0.015
    return e


def _rank(score):
    order = sorted(range(169), key=lambda i: -score[i])
    pct, mid = [0.0] * 169, [0.0] * 169
    cum = 0
    for i in order:
        mid[i] = (cum + 0.5 * _COMBOS[i]) / 1326.0
        cum += _COMBOS[i]
        pct[i] = cum / 1326.0
    return order, pct, mid


_ORDER, _PCT, _PMID = _rank(_EQ)
_PCT_DEEP = _rank([_deep_score(i) for i in range(169)])[1]


def _build_mean_top():
    out = [0.5] * 101
    w, s, j = 0, 0.0, 0
    for p in range(1, 101):
        lim = p * 13.26
        while j < 169 and w < lim:
            i = _ORDER[j]
            w += _COMBOS[i]
            s += _COMBOS[i] * _EQ[i]
            j += 1
        out[p] = s / w
    out[0] = out[1]
    return out


_MEAN_TOP = _build_mean_top()


def _eq_vs_range(i, rng):
    """Estimated all-in equity of hand i against the top `rng` fraction of hands.

    Blends a linear model on mean range strength with a percentile model that
    captures domination, which keeps it decent from 1% ranges up to 100%.
    """
    if rng < 0.01:
        rng = 0.01
    elif rng > 1.0:
        rng = 1.0
    lin = 0.5 + 1.15 * (_EQ[i] - _MEAN_TOP[int(rng * 100 + 0.5)])
    z = _PMID[i] / rng
    pm = 0.85 - 0.45 * math.sqrt(z) if z <= 1.0 else 0.40 - 0.07 * math.log2(z)
    e = 0.5 * (lin + pm)
    if i % 14 == 0 and e < 0.30:  # pocket pairs are rarely worse than ~30%
        e = 0.30
    return 0.1 if e < 0.1 else (0.9 if e > 0.9 else e)


# --------------------------------------------------------------------------
# Postflop hand classification
# --------------------------------------------------------------------------

_BOARD_PLAY = (0.05, 0.08, 0.12, 0.18, 0.35, 0.35, 0.40, 0.45, 0.50)
_FLUSH_BY_HIGHER = (0.97, 0.92, 0.86, 0.80, 0.75)


def _high_card(h1, h2, top):
    v = 0.10
    if h1 > top:
        v += 0.06
    if h2 > top:
        v += 0.06
    if h1 == 12:
        v += 0.04
    return v


def _pocket_value(pr, ub):
    above = 0
    for r in ub:
        if r > pr:
            above += 1
    if above == 0:
        return 0.73 + 0.012 * (pr - 7)
    return 0.52 if above == 1 else (0.42 if above == 2 else 0.32)


def _pair_value(pr, kick, ub):
    idx = ub.index(pr)
    if idx == 0:
        v = 0.64 + (0.09, 0.07, 0.05, 0.03)[12 - kick] if kick >= 9 else 0.64
        if pr == 12:
            v += 0.02
        elif pr <= 5:
            v -= 0.04
        return v
    if idx == 1:
        return 0.53 if kick >= 10 else 0.50
    return 0.40


def _analyze(hole, board):
    """Return (made, outs, wet) where made is showdown strength in [0, 1]."""
    c1, c2 = hole[0], hole[1]
    h1, s1, h2, s2 = c1 >> 2, c1 & 3, c2 >> 2, c2 & 3
    if h1 < h2:
        h1, s1, h2, s2 = h2, s2, h1, s1
    nb = len(board)
    b_rc = [0] * 13
    b_sc = [0, 0, 0, 0]
    b_sm = [0, 0, 0, 0]
    bmask = 0
    for c in board:
        r, s = c >> 2, c & 3
        b_rc[r] += 1
        b_sc[s] += 1
        b_sm[s] |= 1 << r
        bmask |= 1 << r
    max_suit = max(b_sc)
    b_paired = max(b_rc) >= 2
    conn = _max_window(bmask)
    ub = [r for r in range(12, -1, -1) if b_rc[r]]
    top = ub[0]
    wet = conn >= 3 or max_suit >= 3 or (max_suit >= 2 and nb < 5)
    v = _evaluate(tuple(hole) + tuple(board))
    cat = v[0]
    pocket = h1 == h2
    kind = 0  # 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight+, -1 board plays

    if nb == 5 and v == _evaluate(board):
        made, kind = _BOARD_PLAY[cat], -1
    elif cat == 8:
        made, kind = 1.0, 4
    elif cat == 7:
        kind = 4
        made = (0.75 if v[2] >= 11 else 0.45) if b_rc[v[1]] == 4 else 0.995
    elif cat == 6:
        kind = 4
        t, p = v[1], v[2]
        if h1 == t or h2 == t:
            made = 0.85 if (p > t and b_rc[p] >= 2) else 0.95
        elif pocket and all(r < h1 for r in ub if r != t):
            made = 0.88
        else:
            made = 0.72
    elif cat == 5:
        kind = 4
        fs = 0
        for s in range(4):
            if b_sc[s] + (s1 == s) + (s2 == s) >= 5:
                fs = s
                break
        mine = [r for r, s in ((h1, s1), (h2, s2)) if s == fs]
        if not mine:
            made = 0.35
        else:
            fm = b_sm[fs]
            for r in mine:
                fm |= 1 << r
            higher = 0
            for r in range(max(mine) + 1, 13):
                if not (fm >> r) & 1:
                    higher += 1
            made = _FLUSH_BY_HIGHER[min(higher, 4)]
            if b_sc[fs] >= 4 and len(mine) == 1 and higher > 0:
                made -= 0.06
            if b_paired:
                made -= 0.08
    elif cat == 4:
        kind = 4
        made = 0.88
        for w, h in _WINDOWS:
            if h <= v[1]:
                break
            if _POP[bmask & w] >= 3:  # a higher straight is two cards away
                made -= 0.12
                break
        if conn >= 4:
            made -= 0.06
        if max_suit >= 4:
            made -= 0.30
        elif max_suit == 3:
            made -= 0.12
        if b_paired:
            made -= 0.08
    elif cat == 3:
        t = v[1]
        if pocket and h1 == t:
            made, kind = 0.93, 3
        elif h1 == t or h2 == t:
            kick = h2 if h1 == t else h1
            made = 0.80 + (0.05 if kick >= 11 else (0.02 if kick >= 9 else 0.0))
            kind = 3
        else:
            made = _high_card(h1, h2, top)
    elif cat == 2:
        p1, p2 = v[1], v[2]
        if pocket:
            if h1 == p1 or h1 == p2:
                made, kind = _pocket_value(h1, ub) - 0.03, 1
            else:
                made = _high_card(h1, h2, top)
        else:
            u1 = p1 == h1 or p1 == h2
            u2 = p2 == h1 or p2 == h2
            if u1 and u2:
                i1, i2 = ub.index(p1), ub.index(p2)
                made = 0.86 if (i1 == 0 and i2 == 1) else (0.83 if i1 == 0 else 0.78)
                if b_paired:
                    made -= 0.06
                kind = 2
            elif u1 or u2:
                pr = p1 if u1 else p2
                made, kind = _pair_value(pr, h2 if pr == h1 else h1, ub) - 0.03, 1
            else:
                made = _high_card(h1, h2, top)
    elif cat == 1:
        pr = v[1]
        if pocket:
            made, kind = _pocket_value(h1, ub), 1
        elif pr == h1 or pr == h2:
            made, kind = _pair_value(pr, h2 if pr == h1 else h1, ub), 1
        else:
            made = _high_card(h1, h2, top)
    else:
        made = _high_card(h1, h2, top)

    if 1 <= kind <= 3:  # one/two pair and trips lose value on scary boards
        if max_suit >= 4:
            made -= 0.15
        elif max_suit == 3:
            made -= 0.06
        if conn >= 4:
            made -= 0.10
        elif conn >= 3:
            made -= 0.04

    outs = 0.0
    if nb < 5 and 0 <= kind <= 3:
        fd = False
        for s in range(4):
            hc = (s1 == s) + (s2 == s)
            if hc and b_sc[s] + hc == 4:
                fm, hr = b_sm[s], -1
                if s1 == s:
                    fm |= 1 << h1
                    hr = h1
                if s2 == s:
                    fm |= 1 << h2
                    hr = max(hr, h2)
                higher = 0
                for r in range(hr + 1, 13):
                    if not (fm >> r) & 1:
                        higher += 1
                if hc == 2:
                    q = 1.0 if higher <= 1 else 0.85
                else:
                    q = 1.0 if higher == 0 else (0.7 if higher == 1 else 0.45)
                outs += 9.0 * q
                fd = True
                break
        allmask = bmask | (1 << h1) | (1 << h2)
        sd = 0
        for r in range(13):
            bit = 1 << r
            if allmask & bit:
                continue
            h = _straight_high(allmask | bit)
            if h >= 0 and h > _straight_high(bmask | bit):
                sd += 1
        if sd:
            k = min(sd, 2)
            so = 4.0 * k - (1.0 * k if fd else 0.0)
            if max_suit >= 3:
                so *= 0.6
            outs += so
        if kind == 0:
            outs += 1.5 * ((h1 > top) + (h2 > top))
        elif kind == 1 and made < 0.55:
            outs += 2.0
        if b_paired:
            outs *= 0.85

    return (0.02 if made < 0.02 else (1.0 if made > 1.0 else made)), outs, wet


# --------------------------------------------------------------------------
# The bot
# --------------------------------------------------------------------------

class Bot:
    def __init__(self):
        self.seat = 0
        self.rng = random.Random(0x5EED)
        self._start_no = -1
        self._start = None
        self._prev_after = None
        self._reset(8)

    # ---- engine hooks -----------------------------------------------------

    def new_table(self, seat, num_seats):
        self.seat = seat
        self.rng = random.Random(0x5EED * 31 + seat * 7 + num_seats)
        self._start_no = -1
        self._start = None
        self._prev_after = None
        self._reset(num_seats)

    def act(self, state):
        try:
            if state.num_seats != self.n:
                self._reset(state.num_seats)
            if state.hand_number != self._start_no:
                self._capture_start(state)
            if state.street == PREFLOP:
                action = self._preflop(state)
            else:
                action = self._postflop(state)
            return self._legalize(action, state.legal)
        except Exception:
            return fallback_action(state.legal)

    def hand_over(self, result):
        try:
            if len(result.stacks_after) != self.n:
                self._reset(len(result.stacks_after))
            self._learn(result)
        except Exception:
            pass
        try:
            self._prev_after = tuple(result.stacks_after)
        except Exception:
            pass

    # ---- opponent model ---------------------------------------------------

    def _reset(self, n):
        self.n = n
        self.hands = [0] * n
        self.vpip_n = [0] * n
        self.pfr_n = [0] * n
        self.shove_n = [0] * n
        self.raise_faced = [0] * n
        self.raise_folded = [0] * n
        self.rr_faced = [0] * n
        self.rr_folded = [0] * n
        self.allin_faced = [0] * n
        self.allin_called = [0] * n
        self.post_aggr = [0] * n
        self.post_passive = [0] * n
        self.bet_faced = [0] * n
        self.bet_folded = [0] * n

    # Rates are pulled toward priors: VPIP .28, PFR .14, shove .03,
    # fold-to-raise .6, fold-to-bet .4, postflop aggression .3.
    def _vpip(self, s):
        return (self.vpip_n[s] + 2.24) / (self.hands[s] + 8)

    def _pfr(self, s):
        return (self.pfr_n[s] + 1.12) / (self.hands[s] + 8)

    def _shove_rate(self, s):
        return (self.shove_n[s] + 0.24) / (self.hands[s] + 8)

    def _fold_raise(self, s):
        return (self.raise_folded[s] + 3.6) / (self.raise_faced[s] + 6)

    def _fold_post(self, s):
        return (self.bet_folded[s] + 2.4) / (self.bet_faced[s] + 6)

    def _aggr(self, s):
        return (self.post_aggr[s] + 2.4) / (self.post_aggr[s] + self.post_passive[s] + 8)

    def _aggr_mult(self, s):
        m = (0.3 / self._aggr(s)) ** 0.6
        return 1.5 if m > 1.5 else (0.55 if m < 0.55 else m)

    def _adj_call(self, s, c):
        """Estimated fraction of hands seat s calls an all-in with."""
        c *= min(1.8, max(0.7, (self._vpip(s) / 0.28) ** 0.5))
        f = self.allin_faced[s]
        if f:
            c = (3.0 * c + self.allin_called[s]) / (3.0 + f)
        return 0.95 if c > 0.95 else (0.03 if c < 0.03 else c)

    def _capture_start(self, st):
        n = st.num_seats
        paid, street_bet = [0] * n, [0] * n
        street = -1
        for a in st.actions:
            s = a.seat
            if not 0 <= s < n:
                continue
            if a.street != street:
                for i in range(n):
                    paid[i] += street_bet[i]
                    street_bet[i] = 0
                street = a.street
            if a.kind == POST_ANTE:
                paid[s] += a.amount
            elif a.kind != FOLD and a.kind != CHECK:
                street_bet[s] = a.amount
        self._start = tuple(st.stacks[i] + paid[i] + street_bet[i] for i in range(n))
        self._start_no = st.hand_number

    def _learn(self, res):
        acts = res.actions
        if not acts:
            return
        n = self.n
        start = None
        if self._start is not None and self._start_no == res.hand_number:
            start = self._start
        elif self._prev_after is not None and len(self._prev_after) == n:
            start = self._prev_after
        dealt, vp, pr, shoved, faced, raised = set(), set(), set(), set(), set(), set()
        paid, sbet = [0] * n, [0] * n
        street, maxb, bb_amt = -1, 0, 0
        allin_open = False
        for a in acts:
            s = a.seat
            if not 0 <= s < n:
                continue
            if a.street != street:
                for i in range(n):
                    paid[i] += sbet[i]
                    sbet[i] = 0
                street, maxb = a.street, 0
            k = a.kind
            if k == POST_ANTE:
                paid[s] += a.amount
                dealt.add(s)
                continue
            if k == POST_SB or k == POST_BB:
                sbet[s] = a.amount
                maxb = max(maxb, a.amount)
                if k == POST_BB:
                    bb_amt = max(bb_amt, a.amount)
                dealt.add(s)
                continue
            facing = maxb > sbet[s]
            if street == PREFLOP:
                dealt.add(s)
                if k == CALL or k == RAISE:
                    vp.add(s)
                if facing and maxb > bb_amt:
                    if s not in faced:
                        faced.add(s)
                        self.raise_faced[s] += 1
                        if k == FOLD:
                            self.raise_folded[s] += 1
                    if s in raised:
                        self.rr_faced[s] += 1
                        if k == FOLD:
                            self.rr_folded[s] += 1
                    if allin_open:
                        self.allin_faced[s] += 1
                        if k != FOLD:
                            self.allin_called[s] += 1
                if k == RAISE:
                    pr.add(s)
                    raised.add(s)
                    if start is not None and paid[s] + a.amount >= start[s]:
                        shoved.add(s)
                        allin_open = True
            else:
                if k == RAISE:
                    self.post_aggr[s] += 1
                elif k == CALL or k == CHECK:
                    self.post_passive[s] += 1
                if facing:
                    self.bet_faced[s] += 1
                    if k == FOLD:
                        self.bet_folded[s] += 1
            if k == RAISE or k == CALL:
                sbet[s] = a.amount
                maxb = max(maxb, a.amount)
        for s in dealt:
            self.hands[s] += 1
        for s in vp:
            self.vpip_n[s] += 1
        for s in pr:
            self.pfr_n[s] += 1
        for s in shoved:
            self.shove_n[s] += 1

    # ---- tournament context -----------------------------------------------

    def _premium(self, st):
        """Extra equity needed to risk the whole stack (ICM-style survival value)."""
        pl = st.players_left
        if pl <= 2:
            return 0.0
        p = 0.05 if pl >= 6 else 0.06
        if st.hand_number >= 180:  # protect a chip lead near the 200-hand cap
            me = st.seat
            mine = st.stacks[me] + st.bets[me]
            if all(st.stacks[s] + st.bets[s] <= mine for s in range(st.num_seats) if s != me):
                p += 0.04
        return p

    def _open_range(self, st, n_behind, behind):
        r = 0.52 / (max(1, n_behind) ** 0.78)
        if n_behind == 1 and st.players_left == 2:
            r *= 1.4
        if behind:
            fr = sum(self._fold_raise(s) for s in behind) / len(behind)
            r *= min(1.5, max(0.65, (fr / 0.6) ** 1.2))
        dead = st.pot / (1.5 * max(1, st.big_blind))
        if dead > 1.0:
            r *= dead ** 0.4
        return min(1.0, r)

    def _raise_range(self, s, n_after, n_raises, allin, size_bb):
        """Estimated width (fraction of all hands) of seat s's raising range."""
        pm = 0.75 if n_after >= 5 else (1.0 if n_after >= 3 else (1.3 if n_after == 2 else 2.0))
        r = self._pfr(s) * pm
        if n_raises >= 2:
            r *= 0.38 ** (n_raises - 1)
        if allin:
            if size_bb > 25:
                r *= 0.8
            elif size_bb <= 15:
                nash = 5.8 / (max(size_bb, 1.0) * max(1, n_after) ** 0.7)
                if n_raises >= 2:
                    nash *= 0.6
                r = max(r, nash)
            r = max(r, self._shove_rate(s))
        elif n_raises == 1 and size_bb >= 6:
            r *= 0.85
        return min(1.0, max(0.015, r))

    def _shove_ev(self, st, hi, cands, limpers):
        """Chip EV (vs. folding) of moving all-in, plus the expected chips risked."""
        me = st.seat
        stacks, bets = st.stacks, st.bets
        bb = max(1, st.big_blind)
        my_total = stacks[me] + bets[me]
        pot = st.pot
        p_fold, ev_acc, risk_acc, w = 1.0, 0.0, 0.0, 0.0
        for s in cands:
            if stacks[s] <= 0:
                continue
            S = min(my_total, stacks[s] + bets[s])
            c = 2.5 / max(S / bb, 1.0)
            if s == st.sb_seat:
                c *= 0.75
            elif s != st.bb_seat:
                c *= 0.5
            if s in limpers:
                c *= 1.3
            c = self._adj_call(s, c)
            p_fold *= 1.0 - c
            mine = S - bets[me]
            e = _eq_vs_range(hi, c)
            ev_acc += c * (e * (pot + mine + S - bets[s]) - mine)
            risk_acc += c * mine
            w += c
        if w <= 0:
            return float(pot), 0.0
        p_call = 1.0 - p_fold
        return p_fold * pot + p_call * ev_acc / w, p_call * risk_acc / w

    def _reshove_ev(self, st, hi, agg, rng, behind, extra):
        """Chip EV of re-shoving over a raise, with fold equity from the raiser."""
        me = st.seat
        stacks, bets = st.stacks, st.bets
        bb = max(1, st.big_blind)
        my_total = stacks[me] + bets[me]
        pot = st.pot
        agg_total = stacks[agg] + bets[agg]
        S = min(my_total, agg_total)
        mine = S - bets[me]
        theirs = S - bets[agg]
        final = pot + mine + max(0, theirs)
        if theirs <= 0:
            c_frac = 1.0
        else:
            c_frac = 1.25 - 2.0 * theirs / final
            if bets[agg] >= 0.33 * agg_total:
                c_frac = max(c_frac, 0.85)  # pot-committed raisers call
            c_frac = min(1.0, max(0.2, c_frac))
            f = self.rr_faced[agg]
            if f:
                c_frac = (3.0 * c_frac + f - self.rr_folded[agg]) / (3.0 + f)
        p_fold = 1.0 - c_frac
        for s in behind:
            if s != agg and stacks[s] > 0:
                s_bb = max(min(my_total, stacks[s] + bets[s]) / bb, 1.0)
                base = 2.5 / s_bb * (0.8 if s == st.bb_seat else 0.45)
                p_fold *= 1.0 - self._adj_call(s, base)
        for s in extra:
            if stacks[s] > 0:
                p_fold *= 1.0 - self._adj_call(s, 0.3)
        e = _eq_vs_range(hi, rng * c_frac)
        p_call = 1.0 - p_fold
        return p_fold * pot + p_call * (e * final - mine), p_call * mine

    @staticmethod
    def _in_position(st, a, b):
        n = st.num_seats
        return (a - st.button - 1) % n > (b - st.button - 1) % n

    # ---- preflop ------------------------------------------------------------

    def _preflop(self, st):
        lg = st.legal
        n, me = st.num_seats, st.seat
        stacks, inh = st.stacks, st.in_hand
        c1, c2 = st.hole[0], st.hole[1]
        hi = _pf_index(c1 >> 2, c2 >> 2, (c1 & 3) == (c2 & 3))
        prem = self._premium(st)
        order = [(st.bb_seat + 1 + i) % n for i in range(n)]  # BB acts last
        n_raises, agg = 0, -1
        limpers, callers = [], []
        acted, dealt = set(), set()
        for a in st.actions:
            if a.street != PREFLOP:
                continue
            s, k = a.seat, a.kind
            dealt.add(s)
            if k == RAISE:
                n_raises += 1
                agg = s
                callers = []
            elif k == CALL:
                (limpers if n_raises == 0 else callers).append(s)
            elif k != FOLD and k != CHECK:
                continue  # forced posts are not decisions
            acted.add(s)
        for s in range(n):
            if inh[s]:
                dealt.add(s)
        pos = order.index(me)
        behind = [s for s in order[pos + 1:] if inh[s] and stacks[s] > 0 and s not in acted]
        if n_raises == 0:
            limpers = [s for s in limpers if s != me and inh[s]]
            if not limpers and not lg.can_check:
                return self._pf_unopened(st, hi, behind, prem)
            return self._pf_limped(st, hi, behind, limpers, prem)
        if agg == me:
            return Action(CALL) if lg.can_call else Action(CHECK)
        return self._pf_vs_raise(st, hi, order, dealt, agg, n_raises, callers, limpers, behind, prem)

    def _pf_unopened(self, st, hi, behind, prem):
        lg = st.legal
        me = st.seat
        stacks, bets = st.stacks, st.bets
        bb = max(1, st.big_blind)
        my_total = stacks[me] + bets[me]
        if not behind or not lg.can_raise:
            return self._price(st, _eq_vs_range(hi, 1.0), prem)
        eff_bb = min(my_total, max(stacks[s] + bets[s] for s in behind)) / bb
        if eff_bb <= 14:
            ev, risk = self._shove_ev(st, hi, behind, ())
            return self._allin(lg) if ev > prem * risk else Action(FOLD)
        if _PCT_DEEP[hi] <= self._open_range(st, len(behind), behind):
            mult = 2.5 if eff_bb > 40 else (2.2 if eff_bb > 22 else 2.0)
            if me == st.sb_seat:
                mult += 0.5
            return self._raise_to(st, mult * bb)
        if eff_bb <= 22:
            ev, risk = self._shove_ev(st, hi, behind, ())
            if ev > prem * risk + 0.25 * bb:
                return self._allin(lg)
        return Action(FOLD)

    def _pf_limped(self, st, hi, behind, limpers, prem):
        lg = st.legal
        me = st.seat
        stacks, bets = st.stacks, st.bets
        bb = max(1, st.big_blind)
        my_total = stacks[me] + bets[me]
        to_call = lg.call_amount if lg.can_call else 0
        pdeep = _PCT_DEEP[hi]
        rivals = behind + [s for s in limpers if stacks[s] > 0]
        if not rivals or not lg.can_raise:
            return self._price(st, _eq_vs_range(hi, 0.6), prem)
        eff_bb = min(my_total, max(stacks[s] + bets[s] for s in rivals)) / bb
        nl = len(limpers)
        if eff_bb <= 15:
            ev, risk = self._shove_ev(st, hi, rivals, limpers)
            if ev > prem * risk + (0.5 * st.pot if lg.can_check else 0.0):
                return self._allin(lg)
        else:
            r_iso = self._open_range(st, len(behind) + nl, behind) * 0.6
            if lg.can_check:
                r_iso = min(r_iso, 0.12)
            if pdeep <= r_iso:
                target = (3.0 + nl) * bb
                if me == st.sb_seat or me == st.bb_seat:
                    target += 0.5 * bb
                return self._raise_to(st, target)
        if lg.can_check:
            return Action(CHECK)
        if 2 * to_call <= bb and pdeep <= 0.5:
            return Action(CALL)
        if eff_bb >= 25 and to_call <= bb and (pdeep <= 0.25 or (hi % 14 == 0 and eff_bb >= 35)):
            return Action(CALL)
        return Action(FOLD)

    def _pf_vs_raise(self, st, hi, order, dealt, agg, n_raises, callers, limpers, behind, prem):
        lg = st.legal
        me = st.seat
        stacks, bets, inh = st.stacks, st.bets, st.in_hand
        bb = max(1, st.big_blind)
        my_stack = stacks[me]
        my_total = my_stack + bets[me]
        to_call = lg.call_amount if lg.can_call else 0
        pot = st.pot
        raise_to = max(bets)
        agg_allin = stacks[agg] == 0
        n_after = sum(1 for s in order[order.index(agg) + 1:] if s in dealt)
        rng = self._raise_range(agg, n_after, n_raises, agg_allin, raise_to / bb)
        extra = [s for s in callers if s != me and inh[s]]
        pending = [s for s in limpers if s != me and s != agg and inh[s] and s not in extra]
        e = _eq_vs_range(hi, rng) * (0.86 ** len(extra)) * (0.93 ** len(pending)) * (0.97 ** len(behind))
        if hi == 168 or hi == 154:  # AA / KK never just flat a raise war
            e = max(e, 0.7)
        need = to_call / (pot + to_call) if to_call > 0 else 0.0
        opp_max = max(stacks[s] + bets[s] for s in range(st.num_seats) if s != me and inh[s])
        eff = min(my_total, opp_max)
        if agg_allin or not lg.can_raise or to_call >= my_stack:
            return self._price(st, e, prem)
        if eff / bb <= 22 or to_call >= 0.35 * my_stack:
            ev, risk = self._reshove_ev(st, hi, agg, rng, behind, extra)
            if ev > prem * risk:
                return self._allin(lg)
            if to_call <= 0.1 * my_total and e * 0.8 >= need:
                return Action(CALL)
            return Action(FOLD)
        ip = self._in_position(st, me, agg)
        vthr = 0.555 if n_raises == 1 else 0.60 + 0.04 * (n_raises - 2)
        if e >= vthr:
            if n_raises == 1:
                target = raise_to * (3.0 if ip else 3.6) + len(extra) * raise_to
            else:
                target = raise_to * 2.3
            return self._raise_to(st, target)
        realize = 0.9 if ip else (0.84 if me == st.bb_seat else 0.78)
        if e * realize >= need:
            return Action(CALL)
        if n_raises == 1 and to_call * 16 <= eff and (hi % 14 == 0 or (_PCT_DEEP[hi] <= 0.3 and ip)):
            return Action(CALL)  # implied odds: set-mining / suited playables
        return Action(FOLD)

    # ---- postflop -----------------------------------------------------------

    def _postflop(self, st):
        lg = st.legal
        n, me = st.num_seats, st.seat
        stacks, bets, inh = st.stacks, st.bets, st.in_hand
        street, pot = st.street, st.pot
        to_call = lg.call_amount if lg.can_call else 0
        my_stack = stacks[me]
        my_total = my_stack + bets[me]
        opps = [s for s in range(n) if s != me and inh[s]]
        live = [s for s in opps if stacks[s] > 0]
        n_opp = len(opps) or 1

        made, outs, wet = _analyze(st.hole, st.board)
        pf_agg = prev_agg = last_agg = -1
        n_bets = 0
        acted = set()
        for a in st.actions:
            if a.street == street:
                acted.add(a.seat)
                if a.kind == RAISE:
                    n_bets += 1
                    last_agg = a.seat
            elif a.kind == RAISE:
                if a.street == PREFLOP:
                    pf_agg = a.seat
                if a.street == street - 1:
                    prev_agg = a.seat

        prem = self._premium(st)
        made_m = made ** (1.0 + 0.6 * (n_opp - 1))
        if street == RIVER or outs <= 0:
            d1 = d2 = 0.0
        elif street == FLOP:
            d1 = outs / 47.0
            d2 = 1.0 - (1.0 - outs / 47.0) * (1.0 - outs / 46.0)
        else:
            d1 = d2 = outs / 46.0
        fold_f = sum(self._fold_post(s) for s in live) / len(live) if live else 0.4
        station = fold_f < 0.3
        bluff_f = min(1.5, max(0.2, fold_f / 0.4))
        rnd = self.rng.random()

        if to_call <= 0:
            if not live or not lg.can_raise:
                return Action(CHECK)
            vthr = 0.60 + (0.02 if street == RIVER else 0.0) - (0.04 if station else 0.0)
            if made_m >= vthr:
                frac = 0.7 if made_m >= 0.85 else 0.55
                if wet and street != RIVER:
                    frac += 0.1
                if station:
                    frac += 0.1
                if street == RIVER and made_m >= 0.9:
                    frac = 0.9
                return self._bet(st, frac)
            if n_opp > 2:
                return Action(CHECK)
            if street != RIVER and outs >= 8:
                if rnd < min(1.0, (0.55 if outs < 12 else 0.85) * bluff_f):
                    return self._bet(st, 0.6, bluff=True)
                return Action(CHECK)
            if street == FLOP and pf_agg == me:
                p = (0.45 if wet else 0.7) * bluff_f * (0.4 if n_opp == 2 else 1.0)
                if rnd < p:
                    return self._bet(st, 0.6 if wet else 0.45, bluff=True)
                return Action(CHECK)
            if n_opp == 1 and street == TURN and prev_agg == me:
                if rnd < (0.5 if outs >= 4 else 0.3) * bluff_f:
                    return self._bet(st, 0.55, bluff=True)
                return Action(CHECK)
            if n_opp == 1 and all(s in acted for s in live):
                if made_m < 0.3 or (street != RIVER and made_m < 0.45):
                    if rnd < 0.3 * bluff_f:
                        return self._bet(st, 0.5, bluff=True)
            return Action(CHECK)

        # Facing a bet: discount made strength by bet size, raises and bettor style.
        size = to_call / max(1.0, pot - to_call)
        am = self._aggr_mult(last_agg) if last_agg >= 0 else 1.0
        penalty = (0.25 + 0.45 * min(size, 2.0) + 0.5 * max(0, n_bets - 1)) * am
        if street == RIVER:
            penalty += 0.1
        made_f = made_m ** (1.0 + penalty)
        final = to_call >= my_stack or not live
        eq = made_f + (1.0 - made_f) * (d2 if final else d1) * 0.9
        implied = 0.0
        if not final and street != RIVER and outs >= 7:
            rem = min(my_stack - to_call, max(stacks[s] for s in live))
            implied = 0.2 * min(rem, pot)
        need = to_call / (pot + to_call + implied)
        thr = need + prem * min(1.0, to_call / max(1, my_total))
        can_r = lg.can_raise and bool(live)
        big_raise = max(bets) + 0.85 * (pot + to_call)
        if can_r and made_f >= (0.72 if n_bets <= 1 else 0.82):
            return self._raise_to(st, big_raise)
        if can_r and street == FLOP and n_bets == 1 and n_opp == 1 and outs >= 12 and rnd < 0.4 * bluff_f:
            return self._raise_to(st, big_raise)
        if eq >= thr:
            if can_r and my_stack - to_call < 0.3 * (pot + to_call):
                return self._allin(lg)
            return Action(CALL)
        return Action(FOLD)

    # ---- action helpers -----------------------------------------------------

    def _price(self, st, e, prem):
        """Pure pot-odds call/fold (or check when free)."""
        lg = st.legal
        to_call = lg.call_amount if lg.can_call else 0
        if to_call <= 0:
            return Action(CHECK)
        my_total = st.stacks[st.seat] + st.bets[st.seat]
        need = to_call / (st.pot + to_call)
        if e >= need + prem * min(1.0, to_call / max(1, my_total)):
            return Action(CALL)
        return Action(FOLD)

    def _bet(self, st, frac, bluff=False):
        a = self._raise_to(st, st.bets[st.seat] + max(st.pot * frac, st.big_blind))
        if bluff and a.kind == RAISE and a.amount >= st.legal.max_raise_to > 1.2 * st.pot:
            return Action(CHECK)  # never bluff off a stack much bigger than the pot
        return a

    def _raise_to(self, st, target, commit=0.5):
        lg = st.legal
        if not lg.can_raise:
            return Action(CALL) if lg.can_call else Action(CHECK)
        me = st.seat
        my_total = st.stacks[me] + st.bets[me]
        t = max(int(target + 0.5), lg.min_raise_to)
        if t >= lg.max_raise_to:
            return Action(RAISE, lg.max_raise_to)
        pot_after = st.pot + (t - st.bets[me])
        if t >= commit * my_total or my_total - t < 0.35 * pot_after:
            t = lg.max_raise_to  # committed anyway: shove rather than leave crumbs
        return Action(RAISE, t)

    @staticmethod
    def _allin(lg):
        if lg.can_raise:
            return Action(RAISE, lg.max_raise_to)
        return Action(CALL) if lg.can_call else Action(CHECK)

    @staticmethod
    def _legalize(a, lg):
        if a is not None:
            k = a.kind
            if k == RAISE:
                if lg.can_raise:
                    amt = min(max(int(a.amount), lg.min_raise_to), lg.max_raise_to)
                    return Action(RAISE, amt)
                k = CALL
            if k == CALL:
                if lg.can_call:
                    return Action(CALL)
                if lg.can_check:
                    return Action(CHECK)
            else:  # CHECK or FOLD: never fold when checking is free
                if lg.can_check:
                    return Action(CHECK)
                if lg.can_fold:
                    return Action(FOLD)
        return fallback_action(lg)
