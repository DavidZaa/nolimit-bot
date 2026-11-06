"""Plays a single hand at a sit-and-go table.

Rules follow PokerKit (our test oracle) wherever the official rules are
silent: antes before blinds, short blinds post all-in and callers match what
was actually posted, an all-in raise short of a full raise does not reopen
raising for players who already acted, and odd chips go to the first winner
left of the button.

Bots are called the way the tournament calls them: each decision is timed,
and an exception, an illegal action or (if enforced) a slow answer becomes a
check when checking is free and a fold otherwise.

Decisions are timed in CPU time on the bot's thread: everything the bot's
own code costs, garbage collection included, but not the time a busy laptop
or a pool of parallel workers keeps the process off the CPU. Some platforms
(Windows) only update that clock every ~15.6 ms, which would read most
decisions as free and a few as timeouts, so on import the clock is checked
and wall time is used instead when it's that coarse. Wall-clock latency on a
quiet process is measured separately by tests/test_timing.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter_ns, thread_time_ns

from bot.state import (
    CALL,
    CHECK,
    FOLD,
    POST_ANTE,
    POST_BB,
    POST_SB,
    RAISE,
    Action,
    ActionRecord,
    GameState,
    HandResult,
    LegalActions,
    fallback_action,
)
from engine.handeval import best_hand
from engine.latency import LatencyHistogram
from engine.rules import TableConfig


def clock_step_ns(clock, spin_ns: int = 2_000_000) -> int | None:
    """Smallest step `clock` takes while spinning for up to `spin_ns` of wall time.

    None if it never moved. Asking the OS isn't enough: Windows reports its
    thread clock's units (100 ns), not how often it actually ticks.
    """
    end = perf_counter_ns() + spin_ns
    last = clock()
    step = None
    changes = 0
    while changes < 20 and perf_counter_ns() < end:
        now = clock()
        if now != last:
            step = now - last if step is None else min(step, now - last)
            last = now
            changes += 1
    return step


def pick_decision_clock(cpu=thread_time_ns, fine_ns: int = 10_000):
    """(clock, name): the thread CPU clock if it ticks finer than `fine_ns`, else wall time."""
    step = clock_step_ns(cpu)
    if step is not None and step <= fine_ns:
        return cpu, "cpu"
    return perf_counter_ns, "wall"


decision_clock, CLOCK_NAME = pick_decision_clock()


class TableStats:
    """Per-seat decision counters for one table."""

    def __init__(self, seats: int) -> None:
        self.decisions = [0] * seats
        self.errors = [0] * seats
        self.timeouts = [0] * seats
        self.illegal = [0] * seats
        self.latency = [LatencyHistogram() for _ in range(seats)]


@dataclass
class HandOutcome:
    stacks: list[int]
    button: int
    sb_seat: int
    bb_seat: int
    board: tuple[int, ...]
    holes: dict[int, tuple[int, ...]]
    shown: dict[int, tuple[int, ...]]  # hole cards of everyone who reached showdown
    winners: tuple[int, ...]  # seats that took chips from any pot
    actions: tuple[ActionRecord, ...]


def next_alive(stacks, seat: int) -> int:
    n = len(stacks)
    for i in range(1, n + 1):
        s = (seat + i) % n
        if stacks[s] > 0:
            return s
    raise ValueError("no seat has chips")


def blind_seats(stacks, button: int) -> tuple[int, int]:
    """(small blind, big blind). Heads-up, the button posts the small blind."""
    if sum(1 for s in stacks if s > 0) == 2:
        return button, next_alive(stacks, button)
    sb = next_alive(stacks, button)
    return sb, next_alive(stacks, sb)


def is_legal(action, legal: LegalActions) -> bool:
    if not isinstance(action, Action):
        return False
    kind = action.kind
    if kind == CHECK:
        return legal.can_check
    if kind == CALL:
        return legal.can_call
    if kind == FOLD:
        return legal.can_fold
    if kind == RAISE:
        amount = action.amount
        return (
            legal.can_raise
            and type(amount) is int
            and legal.min_raise_to <= amount <= legal.max_raise_to
        )
    return False


def play_hand(
    bots,
    stacks: list[int],
    button: int,
    hand_number: int,
    config: TableConfig,
    rng: random.Random,
    holes: dict[int, tuple[int, ...]] | None = None,
    board: tuple[int, ...] | None = None,
    stats: TableStats | None = None,
) -> HandOutcome:
    """Play one hand. `holes`/`board` fix the cards (tests); otherwise `rng` shuffles."""
    return _Hand(bots, stacks, button, hand_number, config, rng, holes, board, stats).play()


class _Hand:
    def __init__(self, bots, stacks, button, hand_number, config, rng, holes, board, stats):
        n = len(stacks)
        self.bots = bots
        self.n = n
        self.config = config
        self.stats = stats
        self.hand_number = hand_number
        self.button = button
        self.stacks = list(stacks)
        self.alive = [s for s in range(n) if stacks[s] > 0]
        self.sb_seat, self.bb_seat = blind_seats(stacks, button)
        self.sb, self.bb, self.ante = config.schedule.level_for(hand_number)
        self.bets = [0] * n
        self.contrib = [0] * n  # everything put in this hand, current bets included
        self.in_hand = [stacks[s] > 0 for s in range(n)]
        self.actions: list[ActionRecord] = []
        self.street = 0
        self.board: tuple[int, ...] = ()

        variant = config.variant
        board_len = variant.board_size
        if holes is None:
            deck = variant.deck()
            rng.shuffle(deck)
            k = variant.hole_cards
            dealing_order = self._seats_from(self.sb_seat)
            holes = {s: tuple(deck[i * k : (i + 1) * k]) for i, s in enumerate(dealing_order)}
            board = tuple(deck[len(dealing_order) * k : len(dealing_order) * k + board_len])
        elif board is None or len(board) != board_len:
            raise ValueError(f"fixed deals need a full {board_len}-card board")
        self.holes = holes
        self.full_board = tuple(board)

    # -- helpers -----------------------------------------------------------

    def _seats_from(self, start: int) -> list[int]:
        """Seats dealt into this hand, clockwise starting at `start`."""
        n = self.n
        return [s for s in ((start + i) % n for i in range(n)) if s in self.alive]

    def _record(self, seat: int, kind: str, amount: int) -> None:
        self.actions.append(ActionRecord(seat, self.street, kind, amount))

    def _pay(self, seat: int, amount: int) -> None:
        self.stacks[seat] -= amount
        self.bets[seat] += amount
        self.contrib[seat] += amount

    def _post_forced(self) -> None:
        if self.ante > 0:
            for s in self.alive:
                amount = min(self.ante, self.stacks[s])
                self.stacks[s] -= amount
                self.contrib[s] += amount
                self._record(s, POST_ANTE, amount)
        for seat, size, kind in ((self.sb_seat, self.sb, POST_SB), (self.bb_seat, self.bb, POST_BB)):
            amount = min(size, self.stacks[seat])
            if amount > 0:
                self._pay(seat, amount)
                self._record(seat, kind, self.bets[seat])

    # -- betting -----------------------------------------------------------

    def _betting_round(self, first: int) -> None:
        # Reopening rules mirror PokerKit; PLACEHOLDER(kit): check the kit's simulator
        # treats short all-in raises the same way.
        #  - A raise at least as big as the largest raise so far this street (from 0,
        #    so the first bet always counts) reopens raising for everyone.
        #  - Consecutive short all-in raises accumulate; once they add up to a full
        #    raise, players who already acted may raise again.
        #  - Nobody may raise unless some other live player could put in more.
        n = self.n
        bets, stacks, in_hand = self.bets, self.stacks, self.in_hand
        highest = max(bets)
        largest_raise = 0
        acted: set[int] = set()
        short_all_ins: list[int] = []
        # Who acts is settled once, here, like PokerKit: a seat acts if it has chips
        # and could still win more than it has already put in, i.e. the second-largest
        # total (bet + stack) among live players exceeds its bet. The round is skipped
        # only when a single such seat is left and it already covers the bet;
        # mid-street everyone still due to act is asked, even if only to check.
        totals = sorted(bets[s] + stacks[s] for s in range(n) if in_hand[s])
        second = totals[-2] if len(totals) > 1 else 0
        needs = {s for s in range(n) if in_hand[s] and stacks[s] > 0 and second > bets[s]}
        if len(needs) == 1 and bets[next(iter(needs))] >= highest:
            return
        seat = (first - 1) % n
        while needs:
            seat = (seat + 1) % n
            if seat not in needs:
                continue
            to_call = highest - bets[seat]
            raise_allowed = (
                to_call >= largest_raise
                and not (short_all_ins and seat in acted)
                and any(s != seat and in_hand[s] and stacks[s] + bets[s] > highest for s in range(n))
            )
            legal = self._legal(seat, highest, to_call, largest_raise, raise_allowed)
            action = self._decide(seat, legal)
            kind = action.kind
            if kind == FOLD:
                in_hand[seat] = False
                needs.discard(seat)
                self._record(seat, FOLD, bets[seat])
                if sum(in_hand) == 1:
                    return
            elif kind == CHECK:
                needs.discard(seat)
                self._record(seat, CHECK, bets[seat])
            elif kind == CALL:
                self._pay(seat, min(to_call, stacks[seat]))
                needs.discard(seat)
                self._record(seat, CALL, bets[seat])
            else:
                raise_to = action.amount
                size = raise_to - highest
                self._pay(seat, raise_to - bets[seat])
                if size >= largest_raise:
                    acted.clear()
                largest_raise = max(largest_raise, size)
                if stacks[seat] > 0:
                    short_all_ins.clear()
                else:
                    short_all_ins.append(size)
                if sum(short_all_ins) >= largest_raise:
                    short_all_ins.clear()
                highest = raise_to
                needs = {s for s in range(n) if s != seat and in_hand[s] and stacks[s] > 0}
                self._record(seat, RAISE, raise_to)
            acted.add(seat)
            if stacks[seat] == 0:
                needs.discard(seat)

    def _legal(self, seat, highest, to_call, largest_raise, raise_allowed) -> LegalActions:
        stack = self.stacks[seat]
        can_raise = raise_allowed and stack > to_call
        if can_raise:
            max_to = self.bets[seat] + stack
            if self.config.variant.pot_limit:
                max_to = min(max_to, highest + sum(self.contrib) + to_call)
            min_to = min(highest + max(self.bb, largest_raise), max_to)
        else:
            min_to = max_to = 0
        facing = to_call > 0
        return LegalActions(facing, not facing, facing, min(to_call, stack), can_raise, min_to, max_to)

    def _decide(self, seat: int, legal: LegalActions) -> Action:
        state = GameState(
            self.hand_number, seat, self.n, self.button, self.sb_seat, self.bb_seat,
            self.sb, self.bb, self.ante, self.street, self.holes[seat], self.board,
            tuple(self.stacks), tuple(self.bets), sum(self.contrib), tuple(self.in_hand),
            len(self.alive), legal, tuple(self.actions),
        )
        bot = self.bots[seat]
        failed = False
        clock = decision_clock
        t0 = clock()
        try:
            action = bot.act(state)
        except Exception:
            failed = True
            action = None
        elapsed = clock() - t0

        stats = self.stats
        config = self.config
        slow = config.enforce_time and elapsed > config.time_limit_ns
        if stats is not None:
            stats.decisions[seat] += 1
            stats.latency[seat].add(elapsed)
            if failed:
                stats.errors[seat] += 1
            elif slow:
                stats.timeouts[seat] += 1
            elif not is_legal(action, legal):
                stats.illegal[seat] += 1
        if failed or slow or not is_legal(action, legal):
            return fallback_action(legal)
        return action

    def _return_uncalled(self) -> None:
        bets = self.bets
        top = max(range(self.n), key=bets.__getitem__)
        second = max(bets[s] for s in range(self.n) if s != top)
        excess = bets[top] - second
        if excess > 0:
            bets[top] -= excess
            self.contrib[top] -= excess
            self.stacks[top] += excess

    # -- the hand ------------------------------------------------------------

    def play(self) -> HandOutcome:
        self._post_forced()
        variant = self.config.variant
        schedule = variant.board_schedule
        dealt = variant.preflop_board
        for street in range(len(schedule) + 1):
            self.street = street
            if street == 0:
                self.board = self.full_board[:dealt]  # empty unless cards come up early
                first = (self.bb_seat + 1) % self.n
            else:
                self.bets = [0] * self.n
                dealt += schedule[street - 1]
                self.board = self.full_board[:dealt]
                first = (self.button + 1) % self.n
            self._betting_round(first)
            self._return_uncalled()
            if sum(self.in_hand) == 1:
                break
        return self._settle()

    def _settle(self) -> HandOutcome:
        n, contrib, stacks = self.n, self.contrib, self.stacks
        contenders = [s for s in range(n) if self.in_hand[s]]
        won: set[int] = set()
        shown: dict[int, tuple[int, ...]] = {}
        if len(contenders) == 1:
            stacks[contenders[0]] += sum(contrib)
            won.add(contenders[0])
        else:
            variant = self.config.variant
            scores = {s: best_hand(self.holes[s], self.full_board, variant) for s in contenders}
            shown = {s: self.holes[s] for s in contenders}
            # The whole odd-chip remainder goes to the first winner left of the
            # button (PokerKit's convention; PLACEHOLDER(kit)).
            by_position = sorted(contenders, key=lambda s: (s - self.button - 1) % n)
            for amount, eligible in self._pots(contenders):
                best = max(scores[s] for s in eligible)
                winners = [s for s in by_position if s in eligible and scores[s] == best]
                share, odd = divmod(amount, len(winners))
                for s in winners:
                    stacks[s] += share
                stacks[winners[0]] += odd
                won.update(winners)

        result = HandResult(self.hand_number, tuple(stacks), shown, tuple(sorted(won)),
                            tuple(self.actions), self.board)
        for s in self.alive:
            hook = getattr(self.bots[s], "hand_over", None)
            if hook is not None:
                try:
                    hook(result)
                except Exception:
                    pass  # like the tournament, a crashing bot just loses its own state

        return HandOutcome(
            stacks, self.button, self.sb_seat, self.bb_seat, self.board, self.holes,
            shown, tuple(sorted(won)), tuple(self.actions),
        )

    def _pots(self, contenders: list[int]) -> list[tuple[int, list[int]]]:
        """Main pot and side pots as (amount, eligible seats), smallest level first."""
        contrib = self.contrib
        pots: list[tuple[int, list[int]]] = []
        prev = 0
        for level in sorted({contrib[s] for s in contenders}):
            amount = sum(min(c, level) - min(c, prev) for c in contrib)
            if amount:
                pots.append((amount, [s for s in contenders if contrib[s] >= level]))
            prev = level
        # Dead money above the top contender's level (only folded players can be there).
        leftover = sum(c - prev for c in contrib if c > prev)
        if leftover:
            amount, eligible = pots[-1]
            pots[-1] = (amount + leftover, eligible)
        return pots
