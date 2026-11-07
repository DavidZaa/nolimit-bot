"""Turn the kit's table events into our action log, and finished hands into tracker counts.

The tournament runs a pool of forked copies, and any copy may answer any
table, so nothing survives between decisions except the memory bytes the bot
hands back. TableLog is the per-table part of that memory: the current
hand's action log (in bot.state.ActionRecord terms) plus what's needed to
keep building it from the next batch of events. Each hand is counted into
the tracker once, when its closing stack events arrive or the next hand
starts.

Event kinds (poker_agent_protocol.game.Event): hand, ante, blind, fold,
check, call (chips added), raise (the raiser's total for the round), board
(new cards only), show, muck, win, stack (the seat's stack when the hand
ends); the Bay Area variant adds select (an acknowledgement, no cards) and
reveal (the seat's exposed card, kept as `exposed` for the rest of the hand).
Kinds we don't know are ignored.
"""

from __future__ import annotations

from bot.state import CALL, CHECK, FOLD, POST_ANTE, POST_BB, POST_SB, RAISE, ActionRecord

VOLUNTARY = {"fold": FOLD, "check": CHECK, "call": CALL, "raise": RAISE}


class TableLog:
    __slots__ = ("hand", "street", "acted", "bets", "records", "start", "ends", "counted", "blinds",
                 "exposed")

    def __init__(self) -> None:
        self.hand = None  # current hand number
        self.street = 0
        self.acted = False  # a voluntary action this street (an early board card doesn't start one)
        self.bets: dict[int, int] = {}  # seat -> chips in this street
        self.records: list[list[int | str]] = []  # [seat, street, kind, amount]
        self.start: list[int] | None = None  # stacks when this hand began
        self.ends: dict[int, int] = {}  # seat -> stack when the last hand ended
        self.counted = False  # this hand is already in the tracker
        self.blinds = 0  # blinds posted this hand, to tell small from big
        self.exposed: dict[int, str] = {}  # seat -> its revealed card this hand (Bay Area)

    def feed(self, events, index: dict, tracker, default_stack: int, seats: int) -> None:
        """Consume a batch of events. `index` maps the kit's seat ids to 0..seats-1."""
        for ev in events:
            kind = getattr(ev, "kind", None)
            raw = getattr(ev, "seat", None)
            seat = index.get(raw) if raw is not None else None
            amount = getattr(ev, "amount", None) or 0
            if kind == "hand":
                self._close(tracker)
                self._open(int(amount), default_stack, seats)
            elif self.hand is None:
                continue  # joined mid-hand: wait for the next one
            elif kind == "board":
                if self.acted:
                    self.street += 1
                    self.bets = {}
                    self.acted = False
            elif kind == "stack":
                self._close(tracker)
                if seat is not None:
                    self.ends[seat] = int(amount)
            elif seat is None:
                continue
            elif kind == "reveal":
                cards = tuple(getattr(ev, "cards", ()) or ())
                if cards:
                    self.exposed[seat] = str(cards[0])
            elif kind == "ante":
                self.records.append([seat, self.street, POST_ANTE, int(amount)])
            elif kind == "blind":
                self.bets[seat] = self.bets.get(seat, 0) + int(amount)
                post = POST_SB if self.blinds == 0 else POST_BB
                self.blinds += 1
                self.records.append([seat, self.street, post, self.bets[seat]])
            elif kind in VOLUNTARY:
                if kind == "call":
                    self.bets[seat] = self.bets.get(seat, 0) + int(amount)
                elif kind == "raise":
                    self.bets[seat] = int(amount)
                self.acted = True
                self.records.append([seat, self.street, VOLUNTARY[kind], self.bets.get(seat, 0)])

    def _open(self, number: int, default_stack: int, seats: int) -> None:
        if self.ends:
            start = [self.ends.get(s, 0) for s in range(seats)]
        else:
            start = [default_stack] * seats
        self.hand, self.street, self.acted, self.bets = number, 0, False, {}
        self.records, self.start, self.ends, self.counted, self.blinds = [], start, {}, False, 0
        self.exposed = {}

    def _close(self, tracker) -> None:
        if self.hand is not None and not self.counted and self.records:
            tracker.record_hand(self.actions(), self.start)
        self.counted = True

    def actions(self) -> tuple[ActionRecord, ...]:
        return tuple(ActionRecord(int(s), int(st), str(k), int(a)) for s, st, k, a in self.records)

    def to_list(self) -> list:
        return [self.hand, self.street, int(self.acted), sorted(self.bets.items()), self.records,
                self.start, sorted(self.ends.items()), int(self.counted), self.blinds,
                sorted(self.exposed.items())]

    @classmethod
    def from_list(cls, row) -> "TableLog":
        log = cls()
        (log.hand, log.street, acted, bets, log.records, log.start, ends, counted, log.blinds) = row[:9]
        log.exposed = {int(k): str(v) for k, v in (row[9] if len(row) > 9 else ())}
        log.acted, log.counted = bool(acted), bool(counted)
        log.bets = {int(k): int(v) for k, v in bets}
        log.ends = {int(k): int(v) for k, v in ends}
        return log
