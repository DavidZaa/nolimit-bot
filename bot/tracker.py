"""Per-seat opponent stats for one table.

Opponents are anonymous and every table starts fresh, so everything is learned
within one table's 200 hands. Each stat is a plain counter blended with a
prior worth PRIOR_WEIGHT hands, so an estimate starts at a sensible default
and moves toward what the seat actually does:

    estimate = (observed + prior * PRIOR_WEIGHT) / (opportunities + PRIOR_WEIGHT)

A hand is counted once it's over, from HandResult.actions when the table
reports the whole hand (including everything after we folded). When it
doesn't, the tracker falls back to the last action log it saw in a
GameState, which covers the hand up to our final decision in it.

Stats:
  vpip           hands it put chips in voluntarily preflop
  shove          hands it moved all-in with a preflop raise
  fold_to_steal  times it folded facing a single preflop raise nobody had called
  fold_to_bet    times it folded facing a bet after the flop
  cbet           times it bet a heads-up flop nobody had bet yet, having made
                 the last preflop raise: tells a seat that bets every flop it
                 raised from one that only bets when it hits
"""

from __future__ import annotations

from bot.state import CALL, FOLD, POST_ANTE, POST_BB, POST_SB, RAISE

PRIOR_WEIGHT = 10
PRIORS = {"vpip": 0.25, "shove": 0.03, "fold_to_steal": 0.70, "fold_to_bet": 0.50, "cbet": 0.65}
INACTIVE_AFTER = 20  # hands without one voluntary chip before a seat counts as broken


class SeatStats:
    __slots__ = ("hands", "vpip", "shoves", "steals_faced", "steal_folds", "bets_faced", "bet_folds",
                 "cbet_chances", "cbets", "reraises_faced", "reraise_folds")

    def __init__(self, hands=0, vpip=0, shoves=0, steals_faced=0, steal_folds=0, bets_faced=0, bet_folds=0,
                 cbet_chances=0, cbets=0, reraises_faced=0, reraise_folds=0):
        self.hands, self.vpip, self.shoves = hands, vpip, shoves
        self.steals_faced, self.steal_folds = steals_faced, steal_folds
        self.bets_faced, self.bet_folds = bets_faced, bet_folds
        self.cbet_chances, self.cbets = cbet_chances, cbets
        self.reraises_faced, self.reraise_folds = reraises_faced, reraise_folds


def _estimate(observed: int, chances: int, prior: float) -> float:
    return (observed + prior * PRIOR_WEIGHT) / (chances + PRIOR_WEIGHT)


def start_stacks(state) -> list[int]:
    """Stacks at the start of the hand: chips behind now plus everything put in so far."""
    put_in = [0] * len(state.stacks)
    street_bet: dict[tuple[int, int], int] = {}
    for rec in state.actions:
        if rec.kind == POST_ANTE:
            put_in[rec.seat] += rec.amount
        else:
            street_bet[rec.street, rec.seat] = rec.amount
    for (_, seat), amount in street_bet.items():
        put_in[seat] += amount
    return [state.stacks[s] + put_in[s] for s in range(len(state.stacks))]


class Tracker:
    def __init__(self, external: bool = False) -> None:
        """`external`: hands arrive through record_hand (the kit adapter) instead of
        being inferred from decisions, so observe() does nothing."""
        self.external = external
        self.seats: list[SeatStats] = []
        self._hand = None  # the hand in progress that hasn't been counted yet
        self._start = None  # its starting stacks, when known
        self._seen = ()  # the latest action log we saw during it
        self._next_start = None  # starting stacks of the next hand, from the last result

    # -- feeding it ----------------------------------------------------------

    def new_table(self, seat: int, num_seats: int) -> None:
        self.__init__(self.external)
        self.seats = [SeatStats() for _ in range(num_seats)]

    def record_hand(self, actions, start) -> None:
        """Count one finished hand: its full action log and everyone's starting stacks."""
        if self.seats and actions:
            self._record(actions, start)

    def to_list(self) -> list[list[int]]:
        return [[getattr(st, f) for f in SeatStats.__slots__] for st in self.seats]

    @classmethod
    def from_list(cls, rows, external: bool = True) -> "Tracker":
        t = cls(external)
        t.seats = [SeatStats(*map(int, row)) for row in rows]  # shorter, older rows leave new counts at 0
        return t

    def observe(self, state) -> None:
        """Call at the start of every decision."""
        if self.external:
            return
        if not self.seats:
            self.seats = [SeatStats() for _ in range(len(state.stacks))]
        if state.hand_number != self._hand:
            if self._hand is not None and self._seen:  # that hand never got a result
                self._record(self._seen, self._start)
            self._hand = state.hand_number
            # Exact from the hand itself; the last result's stacks only cover
            # hands we never got to see.
            self._start = start_stacks(state)
            self._next_start = None
        self._seen = state.actions

    def hand_over(self, result) -> None:
        start = self._start if self._hand == result.hand_number else self._next_start
        actions = result.actions or (self._seen if self._hand == result.hand_number else ())
        if actions and self.seats:
            self._record(actions, start)
        self._hand, self._seen = None, ()
        self._next_start = list(result.stacks_after)

    def _record(self, actions, start) -> None:
        n = len(self.seats)
        dealt: set[int] = set()
        vpip: set[int] = set()
        shoved: set[int] = set()
        bets, put_in = [0] * n, [0] * n
        street, highest, raises, raise_called = 0, 0, 0, False
        folded: set[int] = set()
        opener = None  # the last preflop raiser
        for rec in actions:
            s, kind = rec.seat, rec.kind
            if not 0 <= s < n:
                continue
            dealt.add(s)
            if rec.street != street:
                street, bets, highest = rec.street, [0] * n, 0
            if kind == POST_ANTE:
                put_in[s] += rec.amount
                continue
            facing = highest > bets[s]
            if kind in (POST_SB, POST_BB, CALL, RAISE):
                put_in[s] += rec.amount - bets[s]
                bets[s] = rec.amount
            if kind in (POST_SB, POST_BB):
                highest = max(highest, rec.amount)
                continue
            if kind == FOLD:
                folded.add(s)
            seat = self.seats[s]
            if street == 0:
                if kind in (CALL, RAISE):
                    vpip.add(s)
                if facing and raises == 1 and not raise_called:
                    seat.steals_faced += 1
                    seat.steal_folds += kind == FOLD
                if facing and raises == 2 and s == opener:  # the opener facing a re-raise
                    seat.reraises_faced += 1
                    seat.reraise_folds += kind == FOLD
                if kind == RAISE:
                    raises += 1
                    opener = s
                    if start is not None and 0 < start[s] <= put_in[s]:
                        shoved.add(s)
                elif kind == CALL and raises:
                    raise_called = True
            elif facing:
                seat.bets_faced += 1
                seat.bet_folds += kind == FOLD
            elif street == 1 and s == opener and len(dealt - folded) == 2:
                seat.cbet_chances += 1
                seat.cbets += kind == RAISE
            if kind == RAISE:
                highest = rec.amount
        for s in dealt:
            seat = self.seats[s]
            seat.hands += 1
            seat.vpip += s in vpip
            seat.shoves += s in shoved

    # -- reading it ----------------------------------------------------------

    def _seat(self, s: int) -> SeatStats:
        return self.seats[s] if 0 <= s < len(self.seats) else SeatStats()

    def hands(self, s: int) -> int:
        return self._seat(s).hands

    def vpip(self, s: int) -> float:
        st = self._seat(s)
        return _estimate(st.vpip, st.hands, PRIORS["vpip"])

    def shove_rate(self, s: int) -> float:
        st = self._seat(s)
        return _estimate(st.shoves, st.hands, PRIORS["shove"])

    def fold_to_steal(self, s: int) -> float:
        st = self._seat(s)
        return _estimate(st.steal_folds, st.steals_faced, PRIORS["fold_to_steal"])

    def fold_to_bet(self, s: int) -> float:
        st = self._seat(s)
        return _estimate(st.bet_folds, st.bets_faced, PRIORS["fold_to_bet"])

    def fold_to_reraise(self, s: int) -> float:
        st = self._seat(s)
        return _estimate(st.reraise_folds, st.reraises_faced, 0.73)

    def cbet(self, s: int) -> float:
        st = self._seat(s)
        return _estimate(st.cbets, st.cbet_chances, PRIORS["cbet"])

    def inactive(self, s: int) -> bool:
        """Broken or timing out: it hasn't put in a voluntary chip in INACTIVE_AFTER hands."""
        st = self._seat(s)
        return st.hands >= INACTIVE_AFTER and st.vpip == 0
