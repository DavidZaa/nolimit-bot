"""Monte Carlo equity bot for 8-handed no-limit hold'em sit-and-gos.

How it decides
--------------
* Unopened preflop pots use a Chen-formula hand ranking with opening ranges by
  position, and switch to push/fold at 13 big blinds or less. A 1 ms budget
  can't produce stable multiway preflop equity, and opening is mostly about
  fold equity anyway.
* When it faces a preflop raise, and on every postflop street, it estimates
  Monte Carlo equity against the opponents still in the hand. Each opponent's
  hole cards come from a range narrowed by what they did preflop and by how
  often that seat raised or played in earlier hands. The bot compares equity
  with pot odds, plus a margin and a tournament risk premium, to choose fold,
  call or raise.

Speed
-----
Lookup tables are built once at import (about half a second). After that, a
7-card evaluation is one dict lookup, plus one list lookup when the board
allows a flush. Simulation stops at a wall-clock deadline well inside the 1 ms
limit, and any exception returns the engine's own fallback (check, else fold).
"""
from __future__ import annotations

import random
import time

from bot.state import (
    CALL,
    CHECK,
    FLOP,
    FOLD,
    PREFLOP,
    RAISE,
    Action,
    GameState,
    HandResult,
    LegalActions,
    fallback_action,
)

# ---------------------------------------------------------------------------
# 7-card hand evaluation tables
# ---------------------------------------------------------------------------
# Strength is an int: category << 20 followed by five 4-bit ranks, so a larger
# value is a better hand. Categories: 0 high card, 1 pair, 2 two pair,
# 3 trips, 4 straight, 5 flush, 6 full house, 7 quads, 8 straight flush.

_RANK_KEY = tuple(1 << (3 * (c >> 2)) for c in range(52))  # 3-bit counter per rank
_RANK_BIT = tuple(1 << (c >> 2) for c in range(52))
_SUIT_KEY = tuple(1 << (3 * (c & 3)) for c in range(52))  # 3-bit counter per suit


def _pack(category, ranks):
    value = category
    for i in range(5):
        value = (value << 4) | (ranks[i] if i < len(ranks) else 0)
    return value


def _straight_high(mask):
    for high in range(12, 3, -1):
        run = 0x1F << (high - 4)
        if (mask & run) == run:
            return high
    return 3 if (mask & 0x100F) == 0x100F else -1  # the wheel, A-2-3-4-5


_STRAIGHT_HIGH = tuple(_straight_high(m) for m in range(1 << 13))


def _flush_value(mask):
    if bin(mask).count("1") < 5:
        return 0
    high = _STRAIGHT_HIGH[mask]
    if high >= 0:
        return _pack(8, (high,))
    return _pack(5, [r for r in range(12, -1, -1) if (mask >> r) & 1][:5])


# Rank mask of one suit's cards -> flush / straight-flush strength (0 if < 5 cards).
_FLUSH = tuple(_flush_value(m) for m in range(1 << 13))


def _rank_value(counts):
    desc = range(12, -1, -1)
    present = [r for r in desc if counts[r]]
    quads = [r for r in desc if counts[r] == 4]
    trips = [r for r in desc if counts[r] == 3]
    pairs = [r for r in desc if counts[r] == 2]
    if quads:
        q = quads[0]
        return _pack(7, (q, next(r for r in present if r != q)))
    if trips and (len(trips) > 1 or pairs):
        second = max(trips[1] if len(trips) > 1 else -1, pairs[0] if pairs else -1)
        return _pack(6, (trips[0], second))
    high = _STRAIGHT_HIGH[sum(1 << r for r in present)]
    if high >= 0:
        return _pack(4, (high,))
    if trips:
        t = trips[0]
        return _pack(3, [t] + [r for r in present if r != t][:2])
    if len(pairs) >= 2:
        p, q = pairs[0], pairs[1]
        return _pack(2, (p, q, next(r for r in present if r != p and r != q)))
    if pairs:
        p = pairs[0]
        return _pack(1, [p] + [r for r in present if r != p][:3])
    return _pack(0, present[:5])


# Sum of _RANK_KEY over any 7 cards -> best non-flush strength (49,205 entries).
_SEVEN: dict[int, int] = {}


def _fill(rank, left, counts, key):
    if rank < 0:
        if left == 0:
            _SEVEN[key] = _rank_value(counts)
        return
    if left > 4 * (rank + 1):
        return
    for n in range(min(4, left) + 1):
        counts[rank] = n
        _fill(rank - 1, left - n, counts, key + (n << (3 * rank)))
    counts[rank] = 0


_fill(12, 7, [0] * 13, 0)

# Sum of _SUIT_KEY over a 5-card board -> the suit with 3+ cards on it, else -1.
# Only that suit can make a flush with two hole cards.
_FLUSH_SUIT = tuple(
    next((s for s in range(4) if ((v >> (3 * s)) & 7) >= 3), -1) for v in range(1 << 12)
)

# ---------------------------------------------------------------------------
# Preflop ranking (Chen formula) and opponent ranges
# ---------------------------------------------------------------------------


def _chen(high, low, suited):
    score = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}.get(high, (high + 2) / 2)
    if high == low:
        return max(5.0, 2 * score)
    gap = high - low - 1
    if suited:
        score += 2
    score -= (0, 1, 2, 4)[gap] if gap < 4 else 5
    if gap <= 1 and high < 10:  # connected-ish and both cards below a queen
        score += 1
    return score


def _build_ranking():
    classes = [
        (hi, lo, suited)
        for hi in range(13)
        for lo in range(hi + 1)
        for suited in ((False,) if hi == lo else (True, False))
    ]
    classes.sort(key=lambda c: (-_chen(*c), -c[0], -c[1], not c[2]))
    combos, class_end, percentile = [], [], [1.0] * 2704
    for hi, lo, suited in classes:
        start = len(combos)
        for s1 in range(4):
            for s2 in range(4):
                if (s1 < s2) if hi == lo else ((s1 == s2) == suited):
                    combos.append((hi * 4 + s1, lo * 4 + s2))
        share = len(combos) / 1326
        for a, b in combos[start:]:
            percentile[a * 52 + b] = percentile[b * 52 + a] = share
        class_end.append(len(combos))
    return tuple(combos), tuple(class_end), tuple(percentile)


# _COMBOS: all 1,326 hole-card combos, strongest class first.
# _PERCENTILE[a * 52 + b]: share of combos at least as strong as this hand.
_COMBOS, _CLASS_END, _PERCENTILE = _build_ranking()

# Ranges are "top x of hands", snapped to these sizes so the preflop equity
# cache keeps getting reused.
_RANGE_LEVELS = (0.03, 0.05, 0.08, 0.12, 0.17, 0.24, 0.33, 0.45, 0.6, 0.8, 1.0)
_RANGE_SIZES = tuple(
    max([_CLASS_END[0]] + [e for e in _CLASS_END if e <= f * 1326 + 1e-6])
    for f in _RANGE_LEVELS
)


def _range_size(frac):
    for level, size in zip(_RANGE_LEVELS, _RANGE_SIZES):
        if frac <= level:
            return size
    return len(_COMBOS)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

_DEAD = 1 << 62  # mark for cards we can see; always "used" in every simulation


def _simulate(hole, board, range_sizes, deadline):
    """Play out random showdowns until `deadline` (perf_counter time), at least once.

    Opponent i is dealt from the top range_sizes[i] combos of the preflop
    ranking (1326 means any two cards). Returns (hero's summed pot share, sims).
    """
    rnd = random.random
    clock = time.perf_counter
    seven, flush, rkey, rbit = _SEVEN, _FLUSH, _RANK_KEY, _RANK_BIT
    skey, fsuit, combos = _SUIT_KEY, _FLUSH_SUIT, _COMBOS

    h1, h2 = hole[0], hole[1]
    hero_key = rkey[h1] + rkey[h2]
    hs1, hs2, hb1, hb2 = h1 & 3, h2 & 3, rbit[h1], rbit[h2]

    mark = [0] * 52  # card -> number of the simulation that last used it
    mark[h1] = mark[h2] = _DEAD
    board_key0 = suit_key0 = 0
    for c in board:
        mark[c] = _DEAD
        board_key0 += rkey[c]
        suit_key0 += skey[c]
    cards = list(board) + [0] * (5 - len(board))
    first = len(board)
    n_opp = len(range_sizes)
    opp_cards = [0] * (2 * n_opp)
    opp_keys = [0] * n_opp

    share = 0.0
    sims = 0
    while True:
        sims += 1
        sid = sims

        j = 0
        for size in range_sizes:
            tries = 0
            while True:
                a, b = combos[int(rnd() * size)]
                if mark[a] < sid and mark[b] < sid:
                    break
                tries += 1
                if tries == 30:  # range blocked by card removal; widen to any two
                    size = 1326
            mark[a] = mark[b] = sid
            opp_cards[2 * j] = a
            opp_cards[2 * j + 1] = b
            opp_keys[j] = rkey[a] + rkey[b]
            j += 1

        board_key, suit_key = board_key0, suit_key0
        for k in range(first, 5):
            c = int(rnd() * 52)
            while mark[c] >= sid:
                c = int(rnd() * 52)
            mark[c] = sid
            cards[k] = c
            board_key += rkey[c]
            suit_key += skey[c]

        hero = seven[board_key + hero_key]
        best, ties = -1, 0
        fs = fsuit[suit_key]
        if fs < 0:
            for key in opp_keys:
                v = seven[board_key + key]
                if v > best:
                    best, ties = v, 1
                elif v == best:
                    ties += 1
        else:
            fmask = 0
            for c in cards:
                if (c & 3) == fs:
                    fmask |= rbit[c]
            m = fmask
            if hs1 == fs:
                m |= hb1
            if hs2 == fs:
                m |= hb2
            if flush[m] > hero:
                hero = flush[m]
            for j in range(n_opp):
                a, b = opp_cards[2 * j], opp_cards[2 * j + 1]
                m = fmask
                if (a & 3) == fs:
                    m |= rbit[a]
                if (b & 3) == fs:
                    m |= rbit[b]
                v = seven[board_key + opp_keys[j]]
                if flush[m] > v:
                    v = flush[m]
                if v > best:
                    best, ties = v, 1
                elif v == best:
                    ties += 1

        if hero > best:
            share += 1.0
        elif hero == best:
            share += 1.0 / (ties + 1)
        if clock() >= deadline:
            return share, sims


# Preflop equity depends only on the hand class and the opponents' ranges, so
# results are pooled across decisions (and tables) until the estimate is tight.
_PREFLOP_CACHE: dict[tuple, list] = {}
_PREFLOP_CACHE_SIMS = 6000


def _preflop_equity(hole, sizes, deadline):
    a, b = hole[0], hole[1]
    hi, lo = max(a >> 2, b >> 2), min(a >> 2, b >> 2)
    key = (hi, lo, hi != lo and (a & 3) == (b & 3), sizes)
    record = _PREFLOP_CACHE.get(key)
    if record is None:
        record = _PREFLOP_CACHE[key] = [0.0, 0]
    if record[1] < _PREFLOP_CACHE_SIMS:
        share, sims = _simulate(hole, (), sizes, deadline)
        record[0] += share
        record[1] += sims
    return record[0] / record[1]


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

_OPEN_RANGE = (0.5, 0.42, 0.32, 0.25, 0.20, 0.16, 0.13, 0.11)  # by players left to act
_PUSH_RANGE = (1.0, 0.55, 0.36, 0.27, 0.21, 0.17, 0.14, 0.12)  # by players who can call
_RERAISE_TIGHTEN = (1.0, 0.45, 0.28, 0.2)  # open, 3-bet, 4-bet, 5-bet+
HAND_CAP = 200


def _scaled(heads_up, n):
    """Equity against n opponents that is comparable to `heads_up` against one."""
    return heads_up ** (n ** 0.75) if n > 1 else heads_up


def _push_range(callers, eff_bb, pot_bb):
    base = _PUSH_RANGE[min(callers, 7)]
    return min(1.0, base * (10.0 / max(eff_bb, 1.0)) ** 0.6 * (max(pot_bb, 1.5) / 1.5) ** 0.35)


def _preflop_profile(actions):
    """Return (levels, callers, n_raises, last_raiser) for the preflop betting.

    levels maps seat -> the raise number it last made (1 = open, 2 = 3-bet, ...).
    callers maps seat -> True if it called a raise, False if it only limped.
    """
    levels: dict[int, int] = {}
    callers: dict[int, bool] = {}
    n_raises, last = 0, -1
    for a in actions:
        if a.street != PREFLOP:
            continue
        if a.kind == RAISE:
            n_raises += 1
            levels[a.seat] = n_raises
            last = a.seat
        elif a.kind == CALL:
            callers[a.seat] = callers.get(a.seat, False) or n_raises > 0
    return levels, callers, n_raises, last


def _players_behind(s):
    """Players with chips still to act after us preflop, up to the big blind."""
    if s.seat == s.bb_seat:
        return 0
    count, seat = 0, s.seat
    for _ in range(s.num_seats - 1):
        seat = (seat + 1) % s.num_seats
        if s.in_hand[seat] and s.stacks[seat] > 0:
            count += 1
        if seat == s.bb_seat:
            break
    return count


def _risk_premium(s, risk, stack, eff_bb):
    """Extra equity required before risking a large share of our stack.

    Points come from finishing place, so busting costs more than doubling up
    gains. Short stacks have to gamble, so the premium fades below 10 big
    blinds. Near the hand cap (assumed to rank survivors by chips), the chip
    leader tightens up and a bottom-half stack loosens up.
    """
    if risk <= 0 or stack <= 0:
        return 0.0
    share = min(1.0, risk / stack)
    if share < 0.25:
        return 0.0
    premium = 0.05 * share
    if eff_bb < 10:
        premium *= max(0.0, eff_bb - 3.0) / 7.0
    if HAND_CAP - s.hand_number <= 20:
        mine = stack + s.bets[s.seat]
        ahead = sum(
            1 for i in range(s.num_seats) if i != s.seat and s.stacks[i] + s.bets[i] > mine
        )
        if ahead == 0:
            premium += 0.03
        elif ahead >= max(1, s.players_left // 2):
            premium -= 0.03
    return max(0.0, premium)


def _check_or_call(legal: LegalActions) -> Action:
    if legal.can_check:
        return Action(CHECK)
    if legal.can_call:
        return Action(CALL)
    return fallback_action(legal)


def _check_or_fold(legal: LegalActions) -> Action:
    if legal.can_check:
        return Action(CHECK)
    if legal.can_fold:
        return Action(FOLD)
    return _check_or_call(legal)


def _raise(legal: LegalActions, target, my_bet, stack) -> Action:
    """Raise to `target`, clamped to legal sizes; shove if that commits 45%+ of our stack."""
    if not legal.can_raise:
        return _check_or_call(legal)
    amount = max(legal.min_raise_to, min(legal.max_raise_to, int(target)))
    if amount - my_bet >= 0.45 * stack:
        amount = legal.max_raise_to
    return Action(RAISE, amount)


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------


class Bot:
    #: Seconds of simulation per decision. The engine allows 1 ms for the whole
    #: call; the rest is headroom for bookkeeping and slower hardware.
    SIM_BUDGET = 0.0003

    def __init__(self) -> None:
        # Per seat: hands seen, VPIP hands, PFR hands, postflop actions, postflop raises.
        self.stats: list[list[int]] = []

    def new_table(self, seat: int, num_seats: int) -> None:
        self.stats = [[0] * 5 for _ in range(num_seats)]

    def act(self, state: GameState) -> Action:
        deadline = time.perf_counter() + self.SIM_BUDGET
        try:
            return self._decide(state, deadline)
        except Exception:
            return fallback_action(state.legal)

    def hand_over(self, result: HandResult) -> None:
        try:
            self._learn(result.actions)
        except Exception:
            pass

    # -- opponent model ------------------------------------------------------

    def _grow(self, n):
        while len(self.stats) < n:
            self.stats.append([0] * 5)

    def _learn(self, actions):
        seen, vpip, pfr = set(), set(), set()
        for a in actions:
            self._grow(a.seat + 1)
            seen.add(a.seat)
            if a.street == PREFLOP:
                if a.kind == RAISE:
                    pfr.add(a.seat)
                    vpip.add(a.seat)
                elif a.kind == CALL:
                    vpip.add(a.seat)
            elif a.kind in (CHECK, CALL, RAISE, FOLD):
                row = self.stats[a.seat]
                row[3] += 1
                if a.kind == RAISE:
                    row[4] += 1
        for seat in seen:
            row = self.stats[seat]
            row[0] += 1
            row[1] += int(seat in vpip)
            row[2] += int(seat in pfr)

    def _freq(self, seat, column, prior, weight=8.0):
        """Smoothed per-seat frequency: VPIP (1), PFR (2) or postflop aggression (4)."""
        row = self.stats[seat]
        seen = row[3] if column == 4 else row[0]
        return (row[column] + prior * weight) / (seen + weight)

    def _range(self, s, seat, levels, callers):
        """Estimated share of hands (strongest first) this opponent holds."""
        if seat in levels:
            frac = self._freq(seat, 2, 0.18) * _RERAISE_TIGHTEN[min(levels[seat], 4) - 1]
            if s.street == PREFLOP and s.stacks[seat] == 0 and s.bets[seat] <= 15 * s.big_blind:
                frac = max(frac, 0.25)  # short-stack shoves come from wide ranges
        elif seat in callers:
            frac = self._freq(seat, 1, 0.30) * (0.6 if callers[seat] else 1.0)
        else:
            frac = 1.0
        return min(1.0, max(0.03, frac))

    # -- decisions -----------------------------------------------------------

    def _decide(self, s: GameState, deadline: float) -> Action:
        legal, me = s.legal, s.seat
        self._grow(s.num_seats)
        opps = [i for i in range(s.num_seats) if i != me and s.in_hand[i]]
        if not opps or len(s.hole) < 2:
            return _check_or_call(legal)
        stack, my_bet = s.stacks[me], s.bets[me]
        to_call = legal.call_amount if legal.can_call else 0
        their_max = max(s.stacks[o] + s.bets[o] for o in opps)
        eff_bb = min(stack + my_bet, their_max) / max(1, s.big_blind)
        if s.street == PREFLOP:
            return self._preflop(s, deadline, opps, stack, my_bet, to_call, eff_bb)
        return self._postflop(s, deadline, opps, stack, my_bet, to_call, eff_bb)

    def _preflop(self, s, deadline, opps, stack, my_bet, to_call, eff_bb):
        legal, me = s.legal, s.seat
        bb = max(1, s.big_blind)
        pct = _PERCENTILE[s.hole[0] * 52 + s.hole[1]]
        levels, callers, n_raises, _ = _preflop_profile(s.actions)

        if n_raises == 0:
            limpers = sum(1 for o in callers if o != me and s.in_hand[o])
            if legal.can_check:  # big blind option in an unraised pot
                short = eff_bb <= 15
                if pct <= (0.20 if short else 0.10):
                    target = legal.max_raise_to if short else (3 + limpers) * bb
                    return _raise(legal, target, my_bet, stack)
                return _check_or_call(legal)
            behind = _players_behind(s)
            if eff_bb <= 13:
                if pct <= _push_range(behind + limpers, eff_bb, s.pot / bb):
                    return _raise(legal, legal.max_raise_to, my_bet, stack)
                return _check_or_fold(legal)
            if pct <= _OPEN_RANGE[min(behind, 7)] * 0.7 ** limpers:
                if me == s.sb_seat:
                    target = 3 * bb
                else:
                    target = (2.3 if eff_bb < 30 else 2.6) * bb + limpers * bb
                return _raise(legal, target, my_bet, stack)
            return _check_or_fold(legal)

        # Facing a raise: Monte Carlo equity against everyone who put money in.
        involved = [o for o in opps if o in levels or o in callers] or opps
        pending = sum(
            1 for o in opps if o not in levels and o not in callers and s.stacks[o] > 0
        )
        sizes = tuple(sorted(_range_size(self._range(s, o, levels, callers)) for o in involved))
        equity = _preflop_equity(s.hole, sizes, deadline)

        n = len(sizes)
        price = to_call / (s.pot + to_call) if to_call > 0 else 0.0
        opp_live = any(s.stacks[o] > 0 for o in opps)
        if legal.can_raise and opp_live and equity >= _scaled(0.60 if n_raises == 1 else 0.64, n):
            if eff_bb <= 25:
                return _raise(legal, legal.max_raise_to, my_bet, stack)
            factor = 3.0 if n_raises == 1 else 2.3
            return _raise(legal, factor * max(s.bets), my_bet, stack)

        all_in = to_call >= stack or not any(s.stacks[o] > 0 for o in involved)
        realized = equity if all_in else 0.85 * equity  # we won't realize it all postflop
        need = (
            price
            + 0.02
            + min(0.06, 0.015 * pending)
            + _risk_premium(s, to_call, stack, eff_bb)
        )
        if realized >= need:
            if eff_bb <= 15 and legal.can_raise and opp_live and to_call < stack:
                return _raise(legal, legal.max_raise_to, my_bet, stack)  # re-shove, don't flat
            return _check_or_call(legal)
        return _check_or_fold(legal)

    def _postflop(self, s, deadline, opps, stack, my_bet, to_call, eff_bb):
        legal, me = s.legal, s.seat
        levels, callers, _, preflop_raiser = _preflop_profile(s.actions)
        sizes = [_range_size(self._range(s, o, levels, callers)) for o in opps]
        share, sims = _simulate(s.hole, s.board, sizes, deadline)
        equity = share / sims
        n = len(opps)
        pot = s.pot
        top_bet = max(s.bets)
        opp_live = any(s.stacks[o] > 0 for o in opps)

        if to_call <= 0:
            if legal.can_raise and opp_live:
                if equity >= _scaled(0.80, n):
                    return _raise(legal, top_bet + 0.75 * pot, my_bet, stack)
                if equity >= _scaled(0.62, n):
                    return _raise(legal, top_bet + 0.55 * pot, my_bet, stack)
                if s.street == FLOP and n == 1 and preflop_raiser == me and equity >= 0.33:
                    return _raise(legal, top_bet + 0.5 * pot, my_bet, stack)  # continuation bet
            return _check_or_call(legal)

        bettors = [a.seat for a in s.actions if a.street == s.street and a.kind == RAISE]
        if bettors and bettors[-1] != me:
            # Equity against a preflop range overstates how often we beat someone
            # who is betting into us. Bets from passive seats mean more.
            size = to_call / max(1, pot - to_call)
            cut = (0.08 + 0.12 * min(size, 1.5)) * (1.4 - self._freq(bettors[-1], 4, 0.35))
            if len(bettors) >= 2:
                cut *= 1.4
            equity *= 1.0 - min(cut, 0.4)

        price = to_call / (pot + to_call)
        if legal.can_raise and opp_live and equity >= _scaled(0.72, n):
            return _raise(legal, top_bet + 0.8 * (pot + to_call), my_bet, stack)
        if equity >= price + 0.02 + _risk_premium(s, to_call, stack, eff_bb):
            return _check_or_call(legal)
        return _check_or_fold(legal)
