"""Push/fold bot: what we submit first on kit day.

At short stacks it shoves or folds with Nash-shaped ranges that widen as the
stack shrinks and as fewer players are left behind; deeper, it opens small
and raises or folds. Postflop it bets made hands, continuation-bets heads up
as the raiser, and gets out otherwise.

It started as a port of the field's PushFold opponent (engine/field.py),
which outscored the old shove-everything baseline by about 7 points a
table. That copy stays frozen as a sparring partner while this one gets
tuned, so the two are kept separate on purpose.
"""

from __future__ import annotations

from bot.actions import check_or, commit, raise_to
from bot.flags import flag
from bot.preflop import CHEN, three_card_score
from bot.hands import TOP_PAIR, TWO_PAIR_PLUS, made_hand
from bot.state import CALL, CHECK, RAISE, Action

SHOVE_DEPTH_BB = 15  # first-in, shove or fold at or below this many big blinds
OPEN_TO_BB = 2.5
SHOVE_CALL_DEPTH_BB = 20  # facing a raise at or below this, it's all-in or fold
PREMIUM = 0.025  # top share of hands that 3-bets when deep (QQ+, AK)
# Discard variants: everyone keeps their best two of three, so a shover's range is stronger
# than its two-card percentile says. Our all-in calling range shrinks by this factor.
DISCARD_CALL_SCALE = 0.8
# Discard variants, deep: risking the whole stack preflop with more than this many big blinds
# effective only with the very top of the range. A quarter of our practice-round busts came
# from 60+ BB stack-offs.
DEEP_SHOVE_BB = 40
DEEP_SHOVE_TOP = 0.015
# Survival (build flag survival=on): with four or more players left and more than
# SURVIVE_BB effective, a preflop shove needs a top-SURVIVE_SHOVE hand and an all-in call a
# top-SURVIVE_CALL one.
SURVIVE_BB = 25
SURVIVE_SHOVE, SURVIVE_CALL = 0.03, 0.04
# short_handed (build flag): in discard variants, heads-up only, rank hands on the wider
# best-two-cards scale for first-in opens, steals and shoves; at or under HU_DEPTH_BB the
# button opens or shoves the top HU_OPEN and the big blind calls a shove with the top
# HU_CALL. Three or more left: the normal ranges (measured: wider ranges there cost cashes).
SHORT_HANDED_PLAYERS = 2
# late_position (build flag): in discard variants with 5+ left, first-in opens widen on the
# button and small blind; facing one raise from the cutoff, button or small blind, the top
# LATE_3BET re-raises (a shove at or under LATE_SHOVE_BB) and the top LATE_CALL calls when
# the call is at most LATE_CALL_STACK of our stack.
LATE_FULL_PLAYERS = 5
LATE_BUTTON_WIDEN, LATE_SB_WIDEN = 1.5, 1.3
LATE_3BET, LATE_CALL, LATE_CALL_STACK, LATE_SHOVE_BB = 0.08, 0.15, 0.10, 25
HU_DEPTH_BB = 20
HU_OPEN, HU_CALL = 0.70, 0.40
DEEP_CALL = 0.08  # top share that flats a cheap raise when deep


def shove_range(depth_bb: float, behind: int) -> float:
    """Share of hands to shove first-in. Shaped like Nash charts (SB ~58% at 10 BB,
    UTG ~12%), not solved."""
    return min(1.0, 5.8 / (max(depth_bb, 1.0) * max(behind, 1) ** 0.8))


def seats_behind(state, seat: int) -> list[int]:
    """Seats dealt in that act after `seat` preflop (the blinds act last)."""
    n = len(state.stacks)
    first = (state.bb_seat + 1) % n
    pos = (seat - first) % n
    in_hand, stacks = state.in_hand, state.stacks
    return [s for s in range(n)
            if s != seat and (in_hand[s] or stacks[s] > 0) and (s - first) % n > pos]


def _depth_bb(state) -> float:
    """Effective stack in big blinds against the deepest opponent still in."""
    seat, stacks, bets = state.seat, state.stacks, state.bets
    deepest = max((stacks[s] + bets[s] for s in range(len(stacks)) if s != seat and state.in_hand[s]),
                  default=0)
    return min(stacks[seat] + bets[seat], deepest) / max(1, state.big_blind)


def _last_raiser(state, street: int):
    for rec in reversed(state.actions):
        if rec.street == street and rec.kind == RAISE:
            return rec.seat
    return None


class PushFoldBot:
    ranks = CHEN  # starting-hand ranking; v3 stays on Chen so it remains a fixed reference

    def act(self, state) -> Action:
        if state.street != 0:
            return self._postflop(state)
        legal, seat = state.legal, state.seat
        bb = max(1, state.big_blind)
        pct = self.ranks.percentile(state.hole)
        depth = _depth_bb(state)
        highest = max(state.bets)
        short = (getattr(state.rules, "discards", 0) and flag("short_handed", "off") == "on"
                 and state.players_left <= SHORT_HANDED_PLAYERS and len(state.hole) == 3)
        if short:
            pct = three_card_score(state.hole)  # the wider scale
        heads_up = short and state.players_left == 2 and depth <= HU_DEPTH_BB
        late = (getattr(state.rules, "discards", 0) and flag("late_position", "off") == "on"
                and state.players_left >= LATE_FULL_PLAYERS)

        if highest <= bb:
            if heads_up and seat == state.sb_seat:
                if pct <= HU_OPEN:
                    return commit(legal) if depth <= SHOVE_DEPTH_BB else raise_to(legal, OPEN_TO_BB * bb)
                return check_or(legal, call=False)
            behind = self._behind(state, seat)
            widen = self._steal_widening(state, behind)
            if late and seat == state.button:
                widen *= LATE_BUTTON_WIDEN
            elif late and seat == state.sb_seat:
                widen *= LATE_SB_WIDEN
            if depth <= SHOVE_DEPTH_BB:
                if pct <= shove_range(depth, len(behind)) * widen / self._shove_tightness(state, behind):
                    return commit(legal)
                return check_or(legal, call=False)
            if pct <= 0.45 / max(len(behind), 1) ** 0.8 * widen:
                return raise_to(legal, OPEN_TO_BB * bb)
            return check_or(legal, call=False)

        discard_variant = bool(getattr(state.rules, "discards", 0))
        too_deep = discard_variant and depth > DEEP_SHOVE_BB and pct > DEEP_SHOVE_TOP
        survive = (discard_variant and flag("survival", "off") == "on" and state.players_left >= 4
                   and depth > SURVIVE_BB)
        if survive and pct > SURVIVE_SHOVE:
            too_deep = True  # no re-shove either
        if pct <= self._resteal_range(state, depth) and not too_deep:
            return commit(legal)

        if late and self._single_late_raise(state):
            if pct <= LATE_3BET:
                return commit(legal) if depth <= LATE_SHOVE_BB else raise_to(legal, 3 * highest)
            if pct <= LATE_CALL and legal.call_amount <= LATE_CALL_STACK * (state.stacks[seat] + state.bets[seat]):
                return Action(CALL) if legal.can_call else check_or(legal, call=False)

        if legal.call_amount >= 0.4 * state.stacks[seat] or depth <= SHOVE_CALL_DEPTH_BB:
            if heads_up and seat == state.bb_seat:
                return commit(legal) if pct <= HU_CALL else check_or(legal, call=False)
            # Effectively all-in: get it in with the top of the range they'd shove.
            raiser = _last_raiser(state, 0)
            theirs = self._raiser_range(state, raiser, bb)
            waiting = self._still_to_act(state, seat, raiser)
            scale = DISCARD_CALL_SCALE if getattr(state.rules, "discards", 0) else 1.0
            if pct <= theirs * 0.7 * scale / (1 + 0.5 * waiting) / self._call_tightness(state, raiser):
                calling = not legal.can_raise or legal.call_amount >= state.stacks[seat]
                if survive and pct > (SURVIVE_CALL if calling else SURVIVE_SHOVE):
                    return check_or(legal, call=False)
                if not too_deep:
                    return commit(legal)
            return check_or(legal, call=False)

        if pct <= PREMIUM:
            return raise_to(legal, 3 * highest)
        cheap = legal.call_amount <= 0.1 * (state.stacks[seat] + state.bets[seat])
        return check_or(legal, call=pct <= DEEP_CALL and cheap)

    def _postflop(self, state) -> Action:
        legal = state.legal
        exact = self._river_decision(state)
        if exact is not None:
            return exact
        strength = made_hand(state.hole, state.board, state.rules)
        if legal.can_check:
            size = self._bet_when_checked_to(state, strength)
            return Action(CHECK) if size is None else raise_to(legal, size)
        if strength >= TWO_PAIR_PLUS:
            return raise_to(legal, 3 * max(state.bets))
        pot_sized_or_less = legal.call_amount * 2 <= state.pot  # the pot already holds the bet
        if self._fold_marginal(state):
            return check_or(legal, call=False)  # a visible hand beats a marginal one
        stays = strength >= TOP_PAIR or self._defends(state, strength)
        return check_or(legal, call=stays and pot_sized_or_less)

    # -- reads the tracker-driven policy refines ------------------------------

    def _behind(self, state, seat: int) -> list[int]:
        """Players left to act behind `seat` who could still wake up with a hand."""
        return seats_behind(state, seat)

    def _steal_widening(self, state, behind: list[int]) -> float:
        """Multiplier on first-in ranges for how likely the players behind are to fold."""
        return 1.0

    def _raiser_range(self, state, raiser, bb: int) -> float:
        """Share of hands we put the last preflop raiser on."""
        if raiser is None:
            return 0.10
        their_depth = (state.stacks[raiser] + state.bets[raiser]) / bb
        return shove_range(their_depth, len(self._behind(state, raiser)))

    def _shove_tightness(self, state, behind: list[int]) -> float:
        """Divisor on the first-in shove range for what getting called would cost."""
        return 1.0

    def _call_tightness(self, state, raiser) -> float:
        """Divisor on the range we get all-in with against a raise."""
        return 1.0

    def _still_to_act(self, state, seat: int, raiser) -> int:
        """Others still in with chips who could overcall behind us."""
        return sum(1 for s in range(len(state.stacks))
                   if s not in (seat, raiser) and state.in_hand[s] and state.stacks[s] > 0)

    def _resteal_range(self, state, depth_bb: float) -> float:
        """Share of hands to re-shove over a raise as a resteal, on top of the usual ranges."""
        return 0.0

    def _single_late_raise(self, state) -> bool:
        """Exactly one preflop raise so far, nobody in with it, from the cutoff, button or small blind."""
        raises = [r for r in state.actions if r.street == 0 and r.kind == RAISE]
        calls = [r for r in state.actions if r.street == 0 and r.kind == CALL]
        if len(raises) != 1 or calls:
            return False
        n = len(state.stacks)
        return raises[0].seat in (state.button, state.sb_seat, (state.button - 1) % n)

    def _fold_marginal(self, state) -> bool:
        """Facing a bet with less than two pair: give it up (an opponent's visible hand is strong)."""
        return False

    def _river_decision(self, state):
        """A river action from exact equity, or None to use the bucket logic."""
        return None

    def _defends(self, state, strength: int) -> bool:
        """Call a bet with less than top pair (the call must still be pot-sized or less)."""
        return False

    def _bet_when_checked_to(self, state, strength: int):
        """Bet size when nobody has bet this street, or None to check."""
        heads_up = sum(state.in_hand) == 2
        cbet = state.street == 1 and heads_up and _last_raiser(state, 0) == state.seat
        if strength >= TOP_PAIR or cbet:
            return state.pot // 2
        return None
