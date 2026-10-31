"""Plays a single hand at a sit-and-go table.

Rules follow PokerKit (our test oracle) wherever the official rules are
silent: antes before blinds, short blinds post all-in and callers match what
was actually posted, an all-in raise short of a full raise does not reopen
raising for players who already acted, and odd chips go to the first winner
left of the button.

Bots are called the way the tournament calls them: each decision is timed,
and an exception, an illegal action or (if enforced) a slow answer becomes a
check when checking is free and a fold otherwise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter_ns

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
        board_len = sum(variant.board_schedule)
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
        n = self.n
        bets, stacks, in_hand = self.bets, self.stacks, self.in_hand
        highest = max(bets)
        min_raise = self.bb  # size of the last full raise; a big blind to start
        full_raises = 0
        acted_at = [-1] * n  # full_raises when the seat last acted; -1 = not yet
        needs = {s for s in range(n) if in_hand[s] and stacks[s] > 0}
        seat = first
        while needs:
            if seat in needs:
                to_call = highest - bets[seat]
                others_have_chips = any(
                    s != seat and in_hand[s] and stacks[s] > 0 for s in range(n)
                )
                if to_call <= 0 and not others_have_chips:
                    needs.discard(seat)  # nobody left to bet into
                else:
                    # Raising is open if we haven't acted yet (-1) or a full raise came since.
                    reopened = acted_at[seat] < full_raises
                    legal = self._legal(seat, highest, to_call, min_raise, others_have_chips and reopened)
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
                        self._pay(seat, raise_to - bets[seat])
                        if raise_to - highest >= min_raise:
                            min_raise = raise_to - highest
                            full_raises += 1
                        highest = raise_to
                        needs = {s for s in range(n) if s != seat and in_hand[s] and stacks[s] > 0}
                        self._record(seat, RAISE, raise_to)
                    acted_at[seat] = full_raises
                    if stacks[seat] == 0:
                        needs.discard(seat)
            seat = (seat + 1) % n

    def _legal(self, seat, highest, to_call, min_raise, raise_allowed) -> LegalActions:
        stack = self.stacks[seat]
        can_raise = raise_allowed and stack > to_call
        if can_raise:
            max_to = self.bets[seat] + stack
            if self.config.variant.pot_limit:
                max_to = min(max_to, highest + sum(self.contrib) + to_call)
            min_to = min(highest + min_raise, max_to)
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
        t0 = perf_counter_ns()
        try:
            action = bot.act(state)
        except Exception:
            failed = True
            action = None
        elapsed = perf_counter_ns() - t0

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
        schedule = self.config.variant.board_schedule
        dealt = 0
        for street in range(len(schedule) + 1):
            self.street = street
            if street == 0:
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
            # Pay winners left of the button first so odd chips land there.
            by_position = sorted(contenders, key=lambda s: (s - self.button - 1) % n)
            for amount, eligible in self._pots(contenders):
                best = max(scores[s] for s in eligible)
                winners = [s for s in by_position if s in eligible and scores[s] == best]
                share, odd = divmod(amount, len(winners))
                for i, s in enumerate(winners):
                    stacks[s] += share + (1 if i < odd else 0)
                won.update(winners)

        result = HandResult(self.hand_number, tuple(stacks), shown, tuple(sorted(won)))
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
