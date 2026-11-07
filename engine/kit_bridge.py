"""Play adapter.Agent inside our engine through the kit's protocol.

Each decision becomes what the kit sends: the public events since this seat's
last turn, a state with string cards, anonymised seat ids and legal actions
with min/max amounts, and the memory bytes the agent returned last time. A
fresh Agent is built for every decision, as in the tournament's forked pool,
so anything kept on self or in globals simply wouldn't survive here either.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kitsim"))

import poker_agent_sdk as sdk  # noqa: E402

from bot.state import (  # noqa: E402
    CALL, CHECK, FOLD, POST_ANTE, POST_BB, POST_SB, RAISE, Action, card_to_str, fallback_action,
)

SEAT_ID = lambda i: 10 + 3 * i  # noqa: E731  (kit seat ids are anonymised, so never 0..n-1)
LATENCIES_NS: list[int] = []  # every agent.act, memory parsing included, for benchmarks (wall)
CPU_NS: list[int] = []  # the same calls in thread CPU time, which OS pauses can't inflate


class KitSeat:
    def __init__(self, agent_cls, config, fresh_agent: bool = True):
        self.agent_cls, self.config, self.fresh = agent_cls, config, fresh_agent
        self.agent = agent_cls()
        self.wall_ns: list[int] = []  # this seat's act() times at this table
        self.cpu_ns: list[int] = []
        self.new_table(0, config.seats)

    def new_table(self, seat: int, num_seats: int) -> None:
        self.memory = b""
        self.pending: list[sdk.Event] = []
        self.delivered: list[sdk.Event] = []  # every event this seat was sent, for histories
        self.hand = None
        self.n = num_seats

    # -- events ------------------------------------------------------------------

    def _board_len(self, street: int) -> int:
        v = self.config.variant
        return v.preflop_board + sum(v.board_schedule[:street])

    def _start_hand(self, number: int, board) -> None:
        self.hand, self.seen, self.street, self.bets = number, 0, 0, {}
        self.pending.append(sdk.Event("hand", None, number))
        early = self._board_len(0)
        if early:
            self.pending.append(sdk.Event("board", None, None, tuple(card_to_str(c) for c in board[:early])))

    def _catch_up(self, actions, board, street: int) -> None:
        """Events for actions[self.seen:], with board events as each street opens."""
        for rec in actions[self.seen:]:
            self._open_streets(rec.street, board)
            seat, prev = SEAT_ID(rec.seat), self.bets.get(rec.seat, 0)
            if rec.kind == POST_ANTE:
                self.pending.append(sdk.Event("ante", seat, rec.amount))
                continue
            self.bets[rec.seat] = rec.amount if rec.kind != FOLD else prev
            kind = {POST_SB: "blind", POST_BB: "blind", CALL: "call"}.get(rec.kind, rec.kind)
            amount = rec.amount - prev if kind in ("blind", "call") else rec.amount if kind == RAISE else None
            self.pending.append(sdk.Event(kind, seat, amount))
        self.seen = len(actions)
        self._open_streets(street, board)

    def _open_streets(self, street: int, board) -> None:
        while self.street < street and len(board) >= self._board_len(self.street + 1):
            lo, hi = self._board_len(self.street), self._board_len(self.street + 1)
            self.street += 1
            self.bets = {}
            self.pending.append(sdk.Event("board", None, None, tuple(card_to_str(c) for c in board[lo:hi])))

    def hand_over(self, result) -> None:
        if self.hand != result.hand_number:
            self._start_hand(result.hand_number, result.board)
        last = max((r.street for r in result.actions), default=0)
        self._catch_up(result.actions, result.board, last)
        for seat, cards in sorted(result.shown.items()):
            self.pending.append(sdk.Event("show", SEAT_ID(seat), None, tuple(card_to_str(c) for c in cards)))
        for seat in result.winners:
            self.pending.append(sdk.Event("win", SEAT_ID(seat), None))
        for seat, stack in enumerate(result.stacks_after):
            self.pending.append(sdk.Event("stack", SEAT_ID(seat), stack))

    # -- a decision ----------------------------------------------------------------

    def act(self, gs) -> Action:
        if gs.hand_number != self.hand:
            self._start_hand(gs.hand_number, gs.board)
        self._catch_up(gs.actions, gs.board, gs.street)
        state = self._state(gs)
        self.delivered.extend(self.pending)
        self.pending = []
        agent = self.agent_cls() if self.fresh else self.agent
        t0, c0 = time.perf_counter_ns(), time.thread_time_ns()
        kit_action, memory = agent.act(state, self.memory)
        wall, cpu = time.perf_counter_ns() - t0, time.thread_time_ns() - c0
        LATENCIES_NS.append(wall)
        CPU_NS.append(cpu)
        self.wall_ns.append(wall)
        self.cpu_ns.append(cpu)
        if isinstance(memory, (bytes, bytearray)):
            self.memory = bytes(memory[: state.config.memory_bytes])
        kind = getattr(kit_action, "kind", None)
        if kind in (FOLD, CHECK, CALL):
            return Action(kind)
        if kind == RAISE:
            return Action(RAISE, kit_action.amount)
        return fallback_action(gs.legal)

    def _state(self, gs):
        legal = gs.legal
        kinds = []
        if legal.can_fold:
            kinds.append(sdk.LegalAction("fold"))
        if legal.can_check:
            kinds.append(sdk.LegalAction("check"))
        if legal.can_call:
            kinds.append(sdk.LegalAction("call", legal.call_amount, legal.call_amount))
        if legal.can_raise:
            kinds.append(sdk.LegalAction("raise", legal.min_raise_to, legal.max_raise_to))
        seats = []
        for i in range(gs.num_seats):
            if gs.in_hand[i]:
                status = "all_in" if gs.stacks[i] == 0 else "active"
            else:
                status = "folded" if gs.stacks[i] or gs.bets[i] else "busted"
            seats.append(sdk.Seat(SEAT_ID(i), gs.stacks[i], gs.bets[i], status))
        return sdk.State(
            seat=SEAT_ID(gs.seat), cards=tuple(card_to_str(c) for c in gs.hole),
            board=tuple(card_to_str(c) for c in gs.board), pot=gs.pot, to_call=legal.call_amount,
            legal_actions=tuple(kinds), seats=tuple(seats), button=SEAT_ID(gs.button),
            blinds=sdk.BlindLevel(gs.small_blind, gs.big_blind, gs.ante), hand_number=gs.hand_number,
            players_remaining=gs.players_left, events=tuple(self.pending),
            config=sdk_config(self.config),
        )


def sdk_config(cfg) -> "sdk.Config":
    """The SDK's state.config for a table run under our TableConfig: the published fields only,
    with the variant as an ID (its card rules come from bot.rules.VARIANTS)."""
    return sdk.Config(players=cfg.seats, starting_stack=cfg.starting_stack,
                      blinds=tuple(sdk.BlindLevel(*level) for level in cfg.schedule.levels),
                      hands_per_level=cfg.schedule.hands_per_level, hand_cap=cfg.max_hands,
                      payouts=tuple(cfg.payouts), variant=cfg.variant.name)
