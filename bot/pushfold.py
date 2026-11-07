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
# calibrated (build flag), from the top teams' round data, discard variants with 5+ left:
# first-in opens by position (non-all-in only; shove depth keeps the tight ranges), and
# facing one non-all-in raise, re-raise/call shares by the raiser's position.
CAL_OPEN = {"button": 0.60, "cutoff": 0.45, "sb": 0.70}
CAL_LATE_3BET, CAL_LATE_CALL = 0.08, 0.25  # vs a cutoff/button/small-blind raise
CAL_EARLY_3BET, CAL_EARLY_CALL = 0.025, 0.10  # vs an earlier raise
CAL_CALL_STACK = 0.10
# cal_tuned (build flag): opens measured under target with CAL_OPEN, and folds to a raise
# well under it, so wider open shares and narrower call shares.
TUNED_OPEN = {"button": 0.65, "cutoff": 0.45, "sb": 0.70}
TUNED_LATE_CALL, TUNED_EARLY_CALL, TUNED_EARLY_3BET = 0.25, 0.10, 0.025  # early raisers: unchanged
TUNED_CALL_STACK, TUNED_CALL_BB, TUNED_SB_STEAL_BB = 0.15, 4, 40  # no any-two SB steal at 40+ BB
# resteal (build flag): late openers fold to re-raises 73% of the time (real data). Facing one
# non-all-in late raise: re-raise 3x with the top RESTEAL_TOP over 25 BB, re-shove it at
# 15-25 BB; only the top RESTEAL_STICKY vs a seat folding under 55% to re-raises (5+ seen);
# continue against a 4-bet only with the top RESTEAL_4BET.
RESTEAL_TOP, RESTEAL_STICKY, RESTEAL_4BET = 0.25, 0.10, 0.05
FIGHT_BB, FIGHT_SHARE = 30, 0.40
# deep (build flag): 40+ BB effective, see more flops: wider middle/late opens, and calls of
# small opens in position and from the big blind. closer (build flag): 3-5 left as the chip
# leader: open the top 70% (any two button/small blind), re-shove over shorter stacks' opens
# with the top 35%, call their all-ins with the top 30%.
DEEP_BB, DEEP_WIDEN, DEEP_OPEN_BB = 40, 1.5, 3
DEEP_CALL_IP, DEEP_CALL_BB = 0.30, 0.40
CLOSER_PLAYERS, CLOSER_OPEN, CLOSER_RESHOVE, CLOSER_CALL = (3, 4, 5), 0.70, 0.35, 0.30
# deep_lite (build flag): 40+ BB, cutoff/button opens 1.3x wider; button calls opens up to 3 BB with the top 20%.
LITE_WIDEN, LITE_CALL = 1.3, 0.20
# deep_fight (build flag): over 30 BB, after our open gets one non-all-in re-raise: 4-bet
# shove the top 12% of the opening range at <= 60 BB, 4-bet to 2.3x their raise with the top
# 8% above (fold to a 5-bet unless top 3%); call with the next 25% in position when the
# re-raise is <= 3.5x our open.
DF_SHOVE_BB, DF_SHOVE_TOP, DF_4BET_TOP, DF_4BET_X, DF_5BET_TOP, DF_CALL_NEXT, DF_CALL_X = 60, 0.30, 0.15, 2.3, 0.03, 0.35, 3.5  # re-raised after our open at <= 30 BB: shove the top 40% of the open range
# bubble (build flag): with 4-5 left and our stack covering everyone still to act, pressure.
BUBBLE_PLAYERS = (4, 5)
BUBBLE_EARLY_OPEN, BUBBLE_SHOVE = 0.50, 0.50
HU_DEPTH_BB = 20
HU_OPEN, HU_CALL = 0.70, 0.50
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
        calibrated = (getattr(state.rules, "discards", 0) and flag("calibrated", "off") == "on"
                      and state.players_left >= LATE_FULL_PLAYERS)
        n = len(state.stacks)
        cutoff = next((s for i in range(1, n) for s in [(state.button - i) % n]
                       if state.in_hand[s] or state.stacks[s] > 0), None)
        tuned = calibrated and flag("cal_tuned", "off") == "on"
        open_shares = TUNED_OPEN if tuned else CAL_OPEN
        bubble = (getattr(state.rules, "discards", 0) and flag("bubble", "off") == "on"
                  and state.players_left in BUBBLE_PLAYERS)
        deep_flag = getattr(state.rules, "discards", 0) and flag("deep", "off") == "on" and depth >= DEEP_BB
        lite = getattr(state.rules, "discards", 0) and flag("deep_lite", "off") == "on" and depth >= DEEP_BB
        mine_total = state.stacks[seat] + state.bets[seat]
        closer = (getattr(state.rules, "discards", 0) and flag("closer", "off") == "on"
                  and state.players_left in CLOSER_PLAYERS
                  and all(state.stacks[s] + state.bets[s] < mine_total for s in range(n) if s != seat and state.in_hand[s]))
        if bubble:
            behind_all = seats_behind(state, seat)
            mine = state.stacks[seat] + state.bets[seat]
            covers = all(state.stacks[s] + state.bets[s] <= mine for s in behind_all)
            bubble = covers and bool(behind_all)

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
                if bubble and pct <= BUBBLE_SHOVE:
                    return commit(legal)
                if pct <= shove_range(depth, len(behind)) * widen / self._shove_tightness(state, behind):
                    return commit(legal)
                return check_or(legal, call=False)
            if deep_flag and len(behind) <= 4:
                widen *= DEEP_WIDEN
            if lite and seat in (state.button, cutoff):
                widen *= LITE_WIDEN
            if closer and depth > SHOVE_DEPTH_BB:
                if seat in (state.button, state.sb_seat) or pct <= CLOSER_OPEN:
                    return raise_to(legal, OPEN_TO_BB * bb)
                return check_or(legal, call=False)
            if bubble and depth > SHOVE_DEPTH_BB:
                late_seat = seat in (state.button, state.sb_seat, cutoff)
                if late_seat or pct <= BUBBLE_EARLY_OPEN:
                    return raise_to(legal, OPEN_TO_BB * bb)
                return check_or(legal, call=False)
            if calibrated:
                position = "button" if seat == state.button else "sb" if seat == state.sb_seat else (
                    "cutoff" if seat == cutoff else None)
                if position is not None:
                    steal = widen == float("inf") and not (tuned and position == "sb" and depth >= TUNED_SB_STEAL_BB)
                    share = open_shares[position] * (LITE_WIDEN if lite and position != "sb" else 1.0)
                    if pct <= share or steal:
                        return raise_to(legal, OPEN_TO_BB * bb)
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

        pre_raises = [r.seat for r in state.actions if r.street == 0 and r.kind == RAISE]
        if (getattr(state.rules, "discards", 0) and flag("maniac_calls", "off") == "on" and pre_raises
                and highest >= state.stacks[pre_raises[-1]] + state.bets[pre_raises[-1]]  # their all-in
                and self._maniac(pre_raises[-1])):
            return commit(legal) if pct <= 0.45 else check_or(legal, call=False)  # a known shover: call wide
        if (getattr(state.rules, "discards", 0) and flag("bb_defense", "off") == "on" and seat == state.bb_seat
                and len(pre_raises) == 1 and pre_raises[0] in (state.button, state.sb_seat)
                and highest <= 3.5 * bb and highest < state.stacks[pre_raises[0]] + state.bets[pre_raises[0]]
                and not any(r.kind == CALL and r.street == 0 for r in state.actions) and depth >= 10):
            call_top = 0.25 if highest <= 2.5 * bb else 0.18  # bigger opens (2.5-3.5 BB): call tighter
            if depth > 20:
                if pct <= 0.05:
                    return raise_to(legal, 3 * highest)
                if pct <= call_top:
                    return Action(CALL) if legal.can_call else check_or(legal, call=False)
                return check_or(legal, call=False)
            return commit(legal) if pct <= call_top else check_or(legal, call=False)
        if closer and len(pre_raises) == 1 and seat not in pre_raises:
            raiser = pre_raises[0]
            their_all_in = highest >= state.stacks[raiser] + state.bets[raiser]
            if pct <= (CLOSER_CALL if their_all_in else CLOSER_RESHOVE):
                return commit(legal)  # they can't bust us
            return check_or(legal, call=False)
        if (deep_flag and len(pre_raises) == 1 and seat not in pre_raises and highest <= DEEP_OPEN_BB * bb
                and highest < state.stacks[pre_raises[0]] + state.bets[pre_raises[0]] and legal.can_call
                and not any(r.kind == CALL and r.street == 0 for r in state.actions)):
            in_position = seat in (state.button, cutoff)
            if (in_position and pct <= DEEP_CALL_IP) or (seat == state.bb_seat and pct <= DEEP_CALL_BB):
                return Action(CALL)
        if (lite and len(pre_raises) == 1 and seat not in pre_raises and seat == state.button
                and highest <= DEEP_OPEN_BB * bb and highest < state.stacks[pre_raises[0]] + state.bets[pre_raises[0]]
                and legal.can_call and pct <= LITE_CALL
                and not any(r.kind == CALL and r.street == 0 for r in state.actions)):
            return Action(CALL)
        if calibrated and flag("resteal", "off") == "on" and seat in pre_raises and depth >= 15:
            if len(pre_raises) == 2 and pre_raises[0] == seat:  # our open got re-raised
                position = "button" if seat == state.button else "sb" if seat == state.sb_seat else (
                    "cutoff" if seat == cutoff else None)
                open_share = open_shares[position] if position else 0.45 / 3 ** 0.8
                if depth <= FIGHT_BB:
                    return commit(legal) if pct <= FIGHT_SHARE * open_share else check_or(legal, call=False)
                if flag("deep_fight", "off") == "on":
                    raiser = pre_raises[1]
                    their_all_in = highest >= state.stacks[raiser] + state.bets[raiser]
                    my_open = next(r.amount for r in state.actions if r.street == 0 and r.kind == RAISE and r.seat == seat)
                    if not their_all_in:
                        if depth <= DF_SHOVE_BB and pct <= DF_SHOVE_TOP * open_share:
                            return commit(legal)
                        if depth > DF_SHOVE_BB and pct <= DF_4BET_TOP * open_share:
                            return raise_to(legal, DF_4BET_X * highest)
                        in_position = raiser in (state.sb_seat, state.bb_seat) or raiser not in seats_behind(state, seat)
                        top_part = (DF_SHOVE_TOP if depth <= DF_SHOVE_BB else DF_4BET_TOP) * open_share
                        if (in_position and highest <= DF_CALL_X * my_open and legal.can_call
                                and pct <= (top_part + DF_CALL_NEXT * open_share)):
                            return Action(CALL)
                        return check_or(legal, call=False)
            if len(pre_raises) >= 3 and pre_raises[-2] == seat:  # our 3-bet got 4-bet (or our 4-bet a 5-bet)
                top = DF_5BET_TOP if flag("deep_fight", "off") == "on" and depth > FIGHT_BB else RESTEAL_4BET
                return commit(legal) if pct <= top else check_or(legal, call=False)
            if state.players_left == 2:  # heads-up: never flat-call a re-raise of our own open
                return commit(legal) if pct <= RESTEAL_4BET else check_or(legal, call=False)
        if calibrated and self._single_raise(state) is not None:
            raiser = self._single_raise(state)
            from_late = raiser in (state.button, state.sb_seat, cutoff)
            if flag("resteal", "off") == "on" and from_late and depth >= 15 and highest < state.stacks[raiser] + state.bets[raiser]:
                top = RESTEAL_TOP if self._fold_to_reraise(raiser) >= 0.55 else RESTEAL_STICKY
                if pct <= top:
                    return commit(legal) if depth <= LATE_SHOVE_BB else raise_to(legal, 3 * highest)
            late_call, early_call = (TUNED_LATE_CALL, TUNED_EARLY_CALL) if tuned else (CAL_LATE_CALL, CAL_EARLY_CALL)
            early_3bet = TUNED_EARLY_3BET if tuned else CAL_EARLY_3BET
            three_bet, call_share = (CAL_LATE_3BET, late_call) if from_late else (early_3bet, early_call)
            if pct <= three_bet:
                return commit(legal) if depth <= LATE_SHOVE_BB else raise_to(legal, 3 * highest)
            affordable = (legal.call_amount <= TUNED_CALL_STACK * (state.stacks[seat] + state.bets[seat])
                          and highest <= TUNED_CALL_BB * bb) if tuned else (
                legal.call_amount <= CAL_CALL_STACK * (state.stacks[seat] + state.bets[seat]))
            if pct <= call_share and affordable:
                return Action(CALL) if legal.can_call else check_or(legal, call=False)
            return check_or(legal, call=False)

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
        late = self._late_streets(state)
        if late is not None:
            return late
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

    def _fold_to_reraise(self, seat: int) -> float:
        return 0.73  # the policy reads its tracker

    def _maniac(self, seat: int) -> bool:
        return False  # the policy reads its tracker

    def _single_raise(self, state):
        """The seat of the only preflop raise so far if nobody is in with it yet, else None."""
        raises = [r for r in state.actions if r.street == 0 and r.kind == RAISE]
        calls = [r for r in state.actions if r.street == 0 and r.kind == CALL]
        return raises[0].seat if len(raises) == 1 and not calls else None

    def _single_late_raise(self, state) -> bool:
        """Exactly one preflop raise so far, nobody in with it, from the cutoff, button or small blind."""
        raiser = self._single_raise(state)
        n = len(state.stacks)
        return raiser is not None and raiser in (state.button, state.sb_seat, (state.button - 1) % n)

    def _fold_marginal(self, state) -> bool:
        """Facing a bet with less than two pair: give it up (an opponent's visible hand is strong)."""
        return False

    def _river_decision(self, state):
        """A river action from exact equity, or None to use the bucket logic."""
        return None

    def _late_streets(self, state):
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
