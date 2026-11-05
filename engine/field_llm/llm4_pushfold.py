"""Push/fold bot for 8-max short-stack sit-and-gos (50/30/20 payouts).

Preflop all-in spots are played as an ICM best response. Opponents get
Nash-style shoving ranges (heads-up Nash is about 5.8 / stack_bb, narrower with
more players left to act). Their calling ranges come from their own
ICM-adjusted price. We shove or call only when that beats folding in tournament
points. A 169x169 heuristic equity table is built once at import, so a decision
is a few table lookups plus O(n^2) Malmuth-Harville ICM evaluations. A time
guard switches to a chart decision well before the 1 ms limit. Deeper stacks
play tight-aggressive preflop and a simple made-hand strategy postflop.
"""
from bisect import bisect_left
from time import perf_counter

from bot.state import (
    CALL, CHECK, FLOP, FOLD, PREFLOP, RAISE, RIVER,
    Action, GameState, HandResult, LegalActions, fallback_action,
)

PAYOUTS = (50.0, 30.0, 20.0)
HAND_CAP = 200
CAP_BLEND_START = 150        # after this, chip rank at the hand cap starts to outweigh ICM
PUSH_BB = 15.0               # effective depth (big blinds) at or below which we only shove or fold
VILLAIN_ICM_AWARENESS = 0.6  # how far opponents are assumed to move from chip-EV toward ICM
BUDGET = 0.0007              # seconds; the engine's hard limit is 0.001
TOTAL = 1326.0               # two-card combos


# ---------------------------------------------------------------- equity model

def _pair_vs(p, c, d, s):
    """All-in equity of pocket pair p against unpaired c > d (s = 1 if suited)."""
    if c == p:                      # KK vs K7: villain nearly drawing dead
        return 0.90
    if d == p:                      # 77 vs A7: one live overcard
        return 0.66
    if p > c:                       # two undercards
        return 0.86 - 0.035 * s - 0.012 * max(0, 4 - (c - d))
    if p < d:                       # two overcards: the classic race
        return 0.555 - 0.03 * s - 0.008 * max(0, 3 - (c - d))
    return 0.70 - 0.02 * s          # one over, one under


def _unpaired(a, b, c, d):
    """All-in equity of a > b against c > d, neither paired, suits ignored."""
    if a == c:
        return 0.5 if b == d else (0.72 if b > d else 0.28)  # shared top card: kicker decides
    if b == d:
        return 0.70 if a > c else 0.30
    if a == d:
        return 0.27                 # they hold our top card with a bigger kicker
    if b == c:
        return 0.73
    if b > c:
        return 0.635                # both our cards higher
    if d > a:
        return 0.365
    if a > c and d > b:
        return 0.56                 # we sandwich them (A2 vs KQ)
    if c > a and b > d:
        return 0.44
    return 0.59 if a > c else 0.41  # interleaved (AJ vs KQ)


def _raw_equity(h1, h2):
    a, b, s1 = h1
    c, d, s2 = h2
    if a == b and c == d:
        return 0.5 if a == c else (0.815 if a > c else 0.185)
    if a == b:
        return _pair_vs(a, c, d, s2)
    if c == d:
        return 1.0 - _pair_vs(c, a, b, s1)
    return (_unpaired(a, b, c, d) + 0.025 * (s1 - s2)
            + 0.006 * (min(c - d, 5) - min(a - b, 5)))


def _build_tables():
    classes, combos = [], []
    for hi in range(12, -1, -1):
        for lo in range(hi, -1, -1):
            if hi == lo:
                classes.append((hi, lo, 0))
                combos.append(6.0)
            else:
                classes.append((hi, lo, 1))
                combos.append(4.0)
                classes.append((hi, lo, 0))
                combos.append(12.0)
    n = len(classes)
    eq = [[0.5] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = 0.5 * (_raw_equity(classes[i], classes[j]) + 1.0 - _raw_equity(classes[j], classes[i]))
            v = min(0.95, max(0.05, v))
            eq[i][j], eq[j][i] = v, 1.0 - v
    vs_any = [sum(combos[j] * eq[i][j] for j in range(n)) / TOTAL for i in range(n)]
    top, acc = [], 0.0
    for i in sorted(range(n), key=lambda k: -vs_any[k]):
        if acc >= 0.25 * TOTAL:
            break
        top.append(i)
        acc += combos[i]
    vs_top = [sum(combos[j] * eq[i][j] for j in top) / acc for i in range(n)]
    # Shove/call ranges reward high cards and pairs, so rank against a tight range too.
    order = sorted(range(n), key=lambda k: -(0.6 * vs_any[k] + 0.4 * vs_top[k]))
    cum = [0.0]
    for i in order:
        cum.append(cum[-1] + combos[i])
    prefix = []
    for i in range(n):
        row, run, ei = [0.0], 0.0, eq[i]
        for j in order:
            run += combos[j] * ei[j]
            row.append(run)
        prefix.append(row)
    pct = [0.0] * n
    for k, i in enumerate(order):
        pct[i] = cum[k + 1] / TOTAL
    return classes, {c: i for i, c in enumerate(classes)}, order, cum, prefix, pct


CLASSES, CLASS_INDEX, ORDER, CUM, EQ_PREFIX, PCT = _build_tables()


def hand_class(hole):
    r1, r2 = hole[0] >> 2, hole[1] >> 2
    suited = int(r1 != r2 and (hole[0] & 3) == (hole[1] & 3))
    return CLASS_INDEX[(max(r1, r2), min(r1, r2), suited)]


def push_prior(eff_bb, n_behind):
    """Approximate Nash first-in shoving range, as a fraction of all hands."""
    x = 5.8 / max(eff_bb, 1.0) / max(1, n_behind) ** 0.75
    return min(1.0, max(0.025, x))


def _range_pos(x):
    t = min(1.0, max(x, 6.0 / TOTAL)) * TOTAL
    k = min(max(bisect_left(CUM, t), 1), len(ORDER))
    lo = CUM[k - 1]
    return k, (t - lo) / (CUM[k] - lo), t


def _eq_at(h, pos):
    k, f, t = pos
    row = EQ_PREFIX[h]
    return (row[k - 1] + f * (row[k] - row[k - 1])) / t


def _eq_vs(h, x):
    """Equity of hand class h against the top x fraction of hands."""
    return _eq_at(h, _range_pos(x))


def _call_frac(req, pos):
    """Widest top-k range whose weakest hand still has `req` equity vs range `pos`."""
    lo, hi = 0, len(ORDER)
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if _eq_at(ORDER[mid - 1], pos) >= req:
            lo = mid
        else:
            hi = mid - 1
    return CUM[lo] / TOTAL


# ------------------------------------------------------------ tournament value

def _icm(st, i):
    """Malmuth-Harville expected points for seat i (only the top 3 places pay)."""
    s = st[i]
    others = [x for j, x in enumerate(st) if j != i and x > 0]
    if s <= 0:
        place = len(others) + 1
        return PAYOUTS[place - 1] if place <= 3 else 0.0
    if not others:
        return PAYOUTS[0]
    tot = s + sum(others)
    n, t2, t3 = len(others), 0.0, 0.0
    for x in range(n):
        a = others[x]
        r = tot - a
        q = a / tot * s / r
        t2 += q
        if n > 1:
            acc = 0.0
            for y in range(n):
                if y != x:
                    b = others[y]
                    acc += b / (r - b)
            t3 += q * acc
    return PAYOUTS[0] * s / tot + PAYOUTS[1] * t2 + PAYOUTS[2] * t3


def _rank_value(st, i):
    """Points if the hand cap ended play now and places went by chip count."""
    s = st[i]
    if s <= 0:
        return _icm(st, i)
    higher = sum(1 for j, x in enumerate(st) if j != i and x > s)
    ties = sum(1 for j, x in enumerate(st) if j != i and x == s)
    pays = [PAYOUTS[p] if p < 3 else 0.0 for p in range(higher, higher + ties + 1)]
    return sum(pays) / len(pays)


def _end_state(stacks, bets, pot, players, winner, cap):
    """Stacks after `winner` takes the all-in pot among `players`, capped at our stake."""
    end = list(stacks)
    tw = min(stacks[winner] + bets[winner], cap)
    dead, gain = pot, 0
    for p in players:
        dead -= bets[p]
        if p != winner:
            tp = stacks[p] + bets[p]
            c = min(tp, tw)
            end[p] = tp - c
            gain += c
    end[winner] = stacks[winner] + bets[winner] + gain + dead
    return end


def _pot_to(stacks, pot, w):
    end = list(stacks)
    end[w] += pot
    return end


# ------------------------------------------------------------- postflop helpers

def _bits(ranks):
    b = 0
    for r in ranks:
        b |= 2 << r
    if b & (2 << 12):
        b |= 1                      # the ace also plays low
    return b


def _straight(b):
    return any((b >> lo) & 31 == 31 for lo in range(10))


def _category(cards):
    """0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house, 7 quads, 8 SF."""
    cnt, suits = [0] * 13, [0] * 4
    for c in cards:
        cnt[c >> 2] += 1
        suits[c & 3] += 1
    fsuit = next((s for s in range(4) if suits[s] >= 5), -1)
    if fsuit >= 0 and _straight(_bits([c >> 2 for c in cards if c & 3 == fsuit])):
        return 8
    if 4 in cnt:
        return 7
    trips, pairs = cnt.count(3), cnt.count(2)
    if trips and (pairs or trips > 1):
        return 6
    if fsuit >= 0:
        return 5
    if _straight(_bits([r for r in range(13) if cnt[r]])):
        return 4
    return 3 if trips else 2 if pairs >= 2 else 1 if pairs else 0


def _draw(hole, board):
    """Flush draw or open-ended straight draw that uses a hole card."""
    suits = [0] * 4
    for c in hole + board:
        suits[c & 3] += 1
    if suits[hole[0] & 3] == 4 or suits[hole[1] & 3] == 4:
        return True
    allb, holeb = _bits([c >> 2 for c in hole + board]), _bits([c >> 2 for c in hole])
    return any((allb >> lo) & 15 == 15 and (holeb >> lo) & 15 for lo in range(1, 10))


def _tier(hole, board):
    """0 air, 1 strong draw, 2 weak pair, 3 top pair or overpair, 4 two pair or better."""
    cat, bcat = _category(hole + board), _category(board)
    if cat >= 3 and cat > bcat:
        return 4
    h1, h2 = max(hole[0] >> 2, hole[1] >> 2), min(hole[0] >> 2, hole[1] >> 2)
    ranks = {c >> 2 for c in board}
    top = max(ranks)
    if h1 != h2 and h1 in ranks and h2 in ranks and bcat == 0:
        return 4
    if h1 == h2:
        return 3 if h1 > top else 2
    if top in (h1, h2):
        kicker = h2 if h1 == top else h1
        return 3 if kicker >= 8 else 2
    if h1 in ranks or h2 in ranks:
        return 2
    return 1 if len(board) < 5 and _draw(hole, board) else 0


def _bet(legal, to):
    if not legal.can_raise:
        return Action(CALL)
    to = max(legal.min_raise_to, int(to))
    return Action(RAISE, legal.max_raise_to if to >= 0.6 * legal.max_raise_to else to)


def _raise_or_jam(legal, to, total):
    to = int(to)
    return Action(RAISE, legal.max_raise_to if to >= 0.4 * total else to)


def _sanitize(action: Action, legal: LegalActions) -> Action:
    kind = action.kind
    if kind == RAISE:
        if legal.can_raise:
            return Action(RAISE, min(legal.max_raise_to, max(legal.min_raise_to, int(action.amount))))
        kind = CALL
    if kind == CALL and legal.can_call:
        return Action(CALL)
    if legal.can_check:
        return Action(CHECK)
    if legal.can_fold:
        return Action(FOLD)
    return fallback_action(legal)


class _OutOfTime(Exception):
    pass


# ------------------------------------------------------------------------ bot

class Bot:
    def __init__(self):
        self.seat = 0
        self.num_seats = 8
        self._stats = {}            # seat -> [hands seen, voluntarily played, raised] preflop
        self._t0 = 0.0
        self._cap_w = 0.0

    def new_table(self, seat, num_seats):
        self.seat, self.num_seats = seat, num_seats
        self._stats = {}

    def hand_over(self, result: HandResult) -> None:
        seen, vol, aggr = set(), set(), set()
        for a in result.actions:
            if a.street != PREFLOP:
                continue
            seen.add(a.seat)
            if a.kind in (CALL, RAISE):
                vol.add(a.seat)
            if a.kind == RAISE:
                aggr.add(a.seat)
        for s in seen - {self.seat}:
            rec = self._stats.setdefault(s, [0, 0, 0])
            rec[0] += 1
            rec[1] += s in vol
            rec[2] += s in aggr

    def act(self, gs: GameState) -> Action:
        self._t0 = perf_counter()
        legal = gs.legal
        try:
            self.seat = gs.seat
            self._cap_w = min(1.0, max(0.0, (gs.hand_number - CAP_BLEND_START)
                                       / float(HAND_CAP - CAP_BLEND_START)))
            if gs.street == PREFLOP:
                action = self._preflop(gs, legal)
            else:
                action = self._postflop(gs, legal)
            return _sanitize(action, legal)
        except Exception:
            return fallback_action(legal)

    # -- value helpers

    def _util(self, stacks, i):
        v = _icm(stacks, i)
        if self._cap_w > 0.0:
            v += self._cap_w * (_rank_value(stacks, i) - v)
        return v

    def _check_time(self):
        if perf_counter() - self._t0 > BUDGET:
            raise _OutOfTime

    def _loose(self, s):
        hands, played, _ = self._stats.get(s, (0, 0, 0))
        vpip = (played + 3.0) / (hands + 12.0)
        return min(2.0, max(0.6, (vpip / 0.25) ** 0.6))

    def _raise_range(self, gs, r, order, n_raises):
        depth = (gs.stacks[r] + gs.bets[r]) / float(gs.big_blind)
        behind = len(order) - 1 - order.index(r)
        if gs.stacks[r] == 0 or depth <= 1.3 * PUSH_BB:
            prior = push_prior(depth, behind)
        else:
            prior = min(0.5, 0.9 / (behind + 1.2))
        hands, _, raised = self._stats.get(r, (0, 0, 0))
        obs = min(1.0, 1.2 * raised / hands) if hands else prior
        if n_raises >= 2:
            prior, obs = prior * 0.45, obs * 0.6
        w = hands / (hands + 20.0)
        return min(1.0, max(0.02, (1.0 - w) * prior + w * max(0.5 * prior, obs)))

    @staticmethod
    def _preflop_order(gs):
        dealt = {s for s in range(gs.num_seats) if gs.in_hand[s]}
        dealt.update(a.seat for a in gs.actions)
        seats, s = [], gs.bb_seat
        for _ in range(gs.num_seats):
            s = (s + 1) % gs.num_seats
            if s in dealt:
                seats.append(s)
        return seats

    def _opp_list(self, gs, order, ranges):
        """(seat, range, forced) for every live opponent, in the order they respond to us."""
        me, st, bt, bb = gs.seat, gs.stacks, gs.bets, gs.big_blind
        tme = st[me] + bt[me]
        i = order.index(me)
        out = []
        for s in order[i + 1:] + order[:i]:
            if gs.in_hand[s]:
                x = ranges.get(s, 0.5 if bt[s] >= bb and s != gs.bb_seat else 1.0)
                out.append((s, x, st[s] == 0 or bt[s] >= tme))
        return out

    def _pass_ev(self, gs, opps):
        """Points from folding (or checking the big blind option)."""
        me, st, bt, pot = gs.seat, gs.stacks, gs.bets, gs.pot
        top = max(bt[s] for s in opps)
        takers = [s for s in opps if bt[s] == top or (st[s] == 0 and bt[s] > 0)]
        keep = sum(self._util(_pot_to(st, pot, w), me) for w in takers) / len(takers)
        if not gs.legal.can_check:
            return keep
        share = 1.0 / (len(opps) + 1)
        return share * self._util(_pot_to(st, pot, me), me) + (1.0 - share) * keep

    def _showdown(self, gs, players, e, cap):
        me, st, bt, pot = gs.seat, gs.stacks, gs.bets, gs.pot
        win = self._util(_end_state(st, bt, pot, players, me, cap), me)
        opp = [p for p in players if p != me]
        if not opp:
            return win
        lose = sum(self._util(_end_state(st, bt, pot, players, w, cap), me) for w in opp) / len(opp)
        return e * win + (1.0 - e) * lose

    def _allin_ev(self, gs, h, opps, x_me):
        """Tournament-point EV of getting our stack in.

        Forced opponents are already all-in or can't fold. Everyone else calls
        with the part of their range that beats their own ICM-adjusted price,
        given they put us on the top `x_me` of hands. Only one voluntary caller
        is modelled, which covers almost all of the EV.
        """
        me, st, bt, pot = gs.seat, gs.stacks, gs.bets, gs.pot
        tme = st[me] + bt[me]
        my_pos = _range_pos(x_me)
        base = [me] + [s for s, _, f in opps if f]
        e_base = 1.0
        for s, x, f in opps:
            if f:
                e_base *= _eq_vs(h, x)
        steal = _pot_to(st, pot, me)
        ev, p_none = 0.0, 1.0
        for s, x, f in opps:
            if f:
                continue
            self._check_time()
            pair = [me, s]
            won = _end_state(st, bt, pot, pair, me, tme)
            lost = _end_state(st, bt, pot, pair, s, tme)
            m = min(tme, st[s] + bt[s])
            req = (m - bt[s]) / float(pot - bt[me] - bt[s] + 2 * m)
            u_fold, u_win, u_lose = self._util(steal, s), self._util(lost, s), self._util(won, s)
            if u_win > u_lose:
                req += VILLAIN_ICM_AWARENESS * ((u_fold - u_lose) / (u_win - u_lose) - req)
            y = min(x, _call_frac(req, my_pos) * self._loose(s))
            q = y / x
            if q > 0.0:
                e = e_base * _eq_vs(h, y)
                if len(base) == 1:
                    sd = e * self._util(won, me) + (1.0 - e) * self._util(lost, me)
                else:
                    sd = self._showdown(gs, base + [s], e, tme)
                ev += p_none * q * sd
                p_none *= 1.0 - q
        if len(base) == 1:
            return ev + p_none * self._util(steal, me)
        return ev + p_none * self._showdown(gs, base, e_base, tme)

    # -- preflop

    def _preflop(self, gs, legal):
        h = hand_class(gs.hole)
        me, st, bt = gs.seat, gs.stacks, gs.bets
        order = self._preflop_order(gs)
        opps = [s for s in order if s != me and gs.in_hand[s]]
        if not opps or me not in order:
            return Action(CHECK)
        tme = st[me] + bt[me]
        eff_bb = min(tme, max(st[s] + bt[s] for s in opps)) / float(gs.big_blind)
        raises = [a for a in gs.actions if a.street == PREFLOP and a.kind == RAISE]
        try:
            if raises:
                return self._facing_raise(gs, h, order, opps, raises, eff_bb, tme)
            return self._unopened(gs, h, order, opps, eff_bb, tme)
        except _OutOfTime:
            pct = PCT[h]
            if raises:
                return Action(CALL) if pct <= 0.05 else Action(FOLD)
            if eff_bb <= PUSH_BB and pct <= push_prior(eff_bb, len(opps)):
                return Action(RAISE, legal.max_raise_to)
            return Action(FOLD)

    def _unopened(self, gs, h, order, opps, eff_bb, tme):
        legal, bb, bt = gs.legal, gs.big_blind, gs.bets
        if eff_bb <= PUSH_BB:
            ev_shove = self._allin_ev(gs, h, self._opp_list(gs, order, {}),
                                      push_prior(eff_bb, len(opps)))
            if ev_shove > self._pass_ev(gs, opps):
                return Action(RAISE, legal.max_raise_to)
            return Action(FOLD)
        pct = PCT[h]
        limpers = sum(1 for s in opps if bt[s] >= bb and s != gs.bb_seat)
        if legal.can_check:
            return _raise_or_jam(legal, bb * (3 + limpers), tme) if pct <= 0.08 else Action(CHECK)
        i = order.index(gs.seat)
        behind = sum(1 for s in order[i + 1:] if gs.in_hand[s])
        x_open = min(0.55, 0.9 / (behind + 1.2)) * (0.6 if limpers else 1.0)
        if pct <= x_open:
            return _raise_or_jam(legal, bb * (2.3 + limpers), tme)
        return Action(FOLD)

    def _facing_raise(self, gs, h, order, opps, raises, eff_bb, tme):
        legal, me, bb, st, bt = gs.legal, gs.seat, gs.big_blind, gs.stacks, gs.bets
        r = raises[-1].seat
        if r == me or r not in opps:
            r = max(opps, key=lambda s: bt[s])
        contenders = [s for s in opps if s == r or bt[s] > bb or (st[s] == 0 and bt[s] > 0)]
        to_call = legal.call_amount
        if st[r] == 0 or to_call >= 0.35 * st[me] or eff_bb <= PUSH_BB:
            xr = self._raise_range(gs, r, order, len(raises))
            ranges = {s: (xr if s == r else (1.0 if bt[s] <= bb else max(0.02, 0.6 * xr)))
                      for s in contenders}
            opp_list = self._opp_list(gs, order, ranges)
            x_me = max(0.03, 0.5 * push_prior(eff_bb, len(opps)))
            if self._allin_ev(gs, h, opp_list, x_me) <= self._pass_ev(gs, opps):
                return Action(FOLD)
            if all(f for s, _, f in opp_list if s in ranges):
                return Action(CALL)             # everyone in is all-in: just call
            return Action(RAISE, legal.max_raise_to)
        pct, mx = PCT[h], max(bt)
        if len(raises) >= 2:
            if pct <= 0.02:
                return Action(RAISE, legal.max_raise_to)
            return Action(CALL) if pct <= 0.035 and to_call <= 0.25 * st[me] else Action(FOLD)
        if pct <= 0.03:
            return _raise_or_jam(legal, 3 * mx + bb * (len(contenders) - 1), tme)
        if pct <= 0.08 and to_call <= 0.12 * tme:
            return Action(CALL)
        hi, lo, _ = CLASSES[h]
        if hi == lo and to_call <= 0.07 * tme and st[r] + bt[r] >= 12 * to_call:
            return Action(CALL)                 # set-mining with implied odds
        if me == gs.bb_seat and to_call <= 1.5 * bb and pct <= 0.35:
            return Action(CALL)
        return Action(FOLD)

    # -- postflop

    def _postflop(self, gs, legal):
        tier = _tier(gs.hole, gs.board)
        pot, to_call, stack = gs.pot, legal.call_amount, gs.stacks[gs.seat]
        if to_call <= 0:
            if tier >= 3:
                return _bet(legal, gs.bets[gs.seat] + pot * (0.75 if tier == 4 else 0.55))
            if tier == 1 and gs.street == FLOP and sum(gs.in_hand) == 2:
                return _bet(legal, gs.bets[gs.seat] + pot * 0.5)
            return Action(CHECK)
        odds = to_call / float(pot + to_call)
        if tier == 4:
            return _bet(legal, 3 * max(gs.bets))
        if tier == 3:
            if stack <= pot:
                return _bet(legal, legal.max_raise_to)
            limit = 0.34 if gs.street == RIVER else 0.42
        elif tier == 2:
            limit = 0.25
        elif tier == 1:
            limit = 0.27
        else:
            limit = 0.0
        return Action(CALL) if odds <= limit else Action(FOLD)
