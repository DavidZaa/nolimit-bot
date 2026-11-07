"""The tracker-driven policy: push/fold, adjusted to how each seat actually plays.

Every adjustment is a named feature so each can be measured on its own
(engine.compare, candidate `policy_<feature>` against `ours`):

  shoves  put a raiser on at least the share of hands it has been seen
          shoving, so a maniac's all-in gets called with kings, not just aces
  steals  widen or narrow first-in ranges by how often the players behind
          fold to a raise, against what the default ranges assume
  cbets   value-bet bigger and thinner into players who never fold, and stop
          continuation-betting air at them
  broken  leave seats that never put in a voluntary chip out of every count
          of players who could wake up with a hand
  icm     tighten all-in calls and first-in shoves by the bubble factor
          against the player who could bust us, from the real payouts
  defend  stop folding to seats that c-bet every flop: call one heads-up
          flop bet with any pair or a strong draw (two pair and up already
          raise), and re-shove 15-25 BB over late-position opens from seats
          that fold too much to steals

With no features it plays exactly like PushFoldBot.

Omaha-style rules (hands must use an exact number of hole cards) get their
own feature set. Both range-scaling features measured the other way round in
PLO: preflop equities run so close that calling a known shover wider pays
and tightening for ICM over-folds.

Every decision reads its rules from state.rules: the variant picks the
feature set and the starting-hand table, and the table's own payouts drive
the ICM. A state without rules raises (SafeBot then checks or folds) rather
than playing hold'em with 50/30/20 by default.
"""

from __future__ import annotations

from bot.flags import flag
from bot.hands import TOP_PAIR, TWO_PAIR_PLUS, WEAK_PAIR, flush_draw, hand_strength, straight_outs, strong_draw
from bot.handeval import hand_value
from bot.rules import STRAIGHT
from bot.icm import cached_bubble_factor
from bot.preflop import ranks_for
from bot.rules import RulesError
from bot.state import CALL, CHECK, RAISE, Action
from bot.actions import check_or, raise_to
from bot.pushfold import PushFoldBot, _last_raiser, seats_behind
from bot.river import river_equity
from bot.tracker import PRIORS, Tracker, start_stacks
from bot.visible import strongest_visible_seats

FEATURES = ("shoves", "steals", "cbets", "broken", "icm", "defend")
V4_FEATURES = ("shoves", "steals", "cbets", "broken")
V5_FEATURES = ("steals", "cbets", "broken", "icm")
V6_FEATURES = V5_FEATURES + ("defend",)  # candidate, not submitted until it clears both fields
# What PolicyBot() and the submission play, for hold'em-style and Omaha hands.
DEFAULT_FEATURES, DEFAULT_OMAHA_FEATURES = V5_FEATURES, V4_FEATURES

STEAL_ANY_TWO = 0.9  # P(everyone behind folds) at which any two cards steal
CAL_STEAL_ANY_TWO = 0.7  # ... from the button or small blind with the calibrated flag
ANY_TWO = float("inf")  # a range multiplier every hand clears
MAX_WIDEN, MIN_WIDEN = 3.0, 0.5
STICKY = 0.25  # fold-to-bet below this: value-bet thin and big, never bluff
CBET_FOLDS = 0.35  # fold-to-bet needed before continuation-betting air
ICM_CALL_POWER = 1.0  # calling range shrinks by bubble_factor ** this
ICM_SHOVE_POWER = 1.0  # first-in shove range likewise
CBET_DEFEND = 0.7  # the bettor's c-bet rate from which we stop folding pairs and draws
RESTEAL_DEPTH_BB = (15, 25)
RESTEAL_FOLDS = 0.8  # the opener's fold-to-steal from which it gets re-shoved on
RESTEAL = 0.15  # share of hands to re-shove with, before ICM and players left behind us
LATE = 3  # an open with at most this many players left behind it (CO, button, SB)
STEAL_SIZE_BB = 4  # opens bigger than this aren't steals
RIVER_BUDGET_NS = 300_000  # exact river equity may spend this much of the 1 ms
RIVER_VALUE_BET = 0.60  # equity needed to bet the river when checked to
RIVER_RAISE = 0.85  # ... to raise a river bet
RIVER_MARGIN = 0.03  # equity over pot odds before a river call


class PolicyBot(PushFoldBot):
    def __init__(self, features=None, icm_powers=(ICM_CALL_POWER, ICM_SHOVE_POWER), omaha_features=None,
                 ranks=None):
        """No `features` means the submitted configuration. `omaha_features`
        defaults to the submitted Omaha set with it, and to `features` otherwise.
        `ranks` pins one starting-hand ranking; by default each decision uses
        bot.preflop.ranks_for(state.rules)."""
        self.fixed_ranks = ranks
        self.ranks = ranks
        if features is None:
            features, default_omaha = DEFAULT_FEATURES, DEFAULT_OMAHA_FEATURES
        else:
            default_omaha = features
        omaha_features = default_omaha if omaha_features is None else omaha_features
        unknown = (set(features) | set(omaha_features)) - set(FEATURES)
        if unknown:
            raise ValueError(f"unknown features {sorted(unknown)}; known: {FEATURES}")
        self.features = frozenset(features)
        self.omaha_features = frozenset(omaha_features)
        self.active = self.features  # the set in play for the current decision
        self.payouts = ()  # the current decision's, from state.rules
        self.call_power, self.shove_power = icm_powers
        self.tracker = Tracker()

    def new_table(self, seat: int, num_seats: int) -> None:
        self.tracker.new_table(seat, num_seats)

    def hand_over(self, result) -> None:
        self.tracker.hand_over(result)

    def act(self, state):
        rules = state.rules
        if rules is None:
            raise RulesError("no rules on this state")
        self.tracker.observe(state)
        self.active = self.features if rules.hole_cards_used is None else self.omaha_features
        self.ranks = self.fixed_ranks or ranks_for(rules)
        self.payouts = rules.payouts
        return super().act(state)

    def _behind(self, state, seat):
        behind = super()._behind(state, seat)
        if "broken" in self.active:
            behind = [s for s in behind if not self.tracker.inactive(s)]
        return behind

    def _steal_widening(self, state, behind):
        if "steals" not in self.active:
            return 1.0
        # Everyone behind, broken seats included: they fold for certain.
        everyone = seats_behind(state, state.seat)
        if not everyone:
            return 1.0
        tracker = self.tracker
        p_all_fold = 1.0
        for s in everyone:
            p_all_fold *= 1.0 if tracker.inactive(s) else tracker.fold_to_steal(s)
        any_two = STEAL_ANY_TWO
        if (flag("calibrated", "off") == "on" and getattr(state.rules, "discards", 0)
                and state.seat in (state.button, state.sb_seat)):
            any_two = CAL_STEAL_ANY_TWO  # the field folds 80% to late opens: steal any two sooner
        if p_all_fold >= any_two:
            return ANY_TWO
        assumed = PRIORS["fold_to_steal"] ** len(everyone)
        return min(MAX_WIDEN, max(MIN_WIDEN, p_all_fold / assumed))

    def _raiser_range(self, state, raiser, bb):
        model = super()._raiser_range(state, raiser, bb)
        if "shoves" in self.active and raiser is not None:
            return max(model, self.tracker.shove_rate(raiser))
        return model

    def _bubble(self, state, opp: int) -> float:
        """Bubble factor for an all-in against `opp`. Everyone's chips are fixed for
        the length of a hand, so it's computed once per hand and opponent."""
        return cached_bubble_factor(tuple(start_stacks(state)), state.seat, opp, self.payouts)

    def _call_tightness(self, state, raiser):
        if "icm" not in self.active or raiser is None:
            return 1.0
        return self._bubble(state, raiser) ** self.call_power

    def _shove_tightness(self, state, behind):
        if "icm" not in self.active or not behind:
            return 1.0
        # The deepest stack behind is the one whose call can bust us.
        totals = start_stacks(state)
        caller = max(behind, key=lambda s: totals[s])
        return self._bubble(state, caller) ** self.shove_power

    def _still_to_act(self, state, seat, raiser):
        if "broken" not in self.active:
            return super()._still_to_act(state, seat, raiser)
        return sum(1 for s in range(len(state.stacks))
                   if s not in (seat, raiser) and state.in_hand[s] and state.stacks[s] > 0
                   and not self.tracker.inactive(s))

    def _bet_when_checked_to(self, state, strength):
        size = self._bet_size_when_checked_to(state, strength)
        if (size is not None and state.street >= 2 and flag("min_size", "off") == "on"
                and getattr(state.rules, "exposes", 0)):
            size = max(size, state.pot * 3 // 5)  # the field barely folds to small turn/river bets
        return size

    def _bet_size_when_checked_to(self, state, strength):
        if "cbets" not in self.active:
            return super()._bet_when_checked_to(state, strength)
        others = [s for s in range(len(state.stacks)) if s != state.seat and state.in_hand[s]]
        folds = min((self.tracker.fold_to_bet(s) for s in others), default=PRIORS["fold_to_bet"])
        if (flag("late_streets", "off") == "on" and state.street >= 3 and strength >= TWO_PAIR_PLUS
                and self._two_pair_with_hole(state)):
            return state.pot * 3 // 4  # river value bet
        heads_up = len(others) == 1
        if ((flag("turn_barrel", "off") == "on" or flag("turn_semibluff", "off") == "on") and state.street == 2
                and heads_up and _last_raiser(state, 1) == state.seat):
            if strength >= WEAK_PAIR or strong_draw(state.hole, state.board, state.rules):
                return state.pot * 3 // 4  # our flop bet got called: barrel with a pair or a draw
            return None
        if strength >= TOP_PAIR:
            return state.pot * 3 // 4 if folds < STICKY else state.pot // 2
        if strength == WEAK_PAIR and folds < STICKY:
            return state.pot // 2
        heads_up = len(others) == 1
        if (flag("expose_play", "off") == "on" and state.street == 1 and len(others) <= 2
                and self._my_exposed_pairs_board(state) and (strength >= WEAK_PAIR
                                                             or strong_draw(state.hole, state.board, state.rules))):
            return state.pot // 2  # our exposed card pairs the flop: the field folds 66% to this bet
        if (flag("river_bluff", "off") == "on" and state.street == 3 and heads_up and strength < WEAK_PAIR
                and self._their_exposed_misses(state, others[0])):
            return state.pot  # the field folds 57% to a river bet after checking to us
        if (flag("late_streets", "off") == "on" and state.street == 1 and strength < TOP_PAIR
                and not strong_draw(state.hole, state.board, state.rules)
                and (len(others) >= 2 or self._strong_visible(state))):
            return None  # no c-bet into 3+ players or a visible made hand without top pair or a draw
        cbet = state.street == 1 and heads_up and _last_raiser(state, 0) == state.seat
        if cbet and folds >= CBET_FOLDS and not self._strong_visible(state):
            return state.pot // 2  # a bluff, never into a hand we can see is strong
        return None

    def _strong_visible(self, state) -> bool:
        """exposed_reads: an opponent still in whose exposed card pairs the board or better."""
        if flag("exposed_reads", "off") != "on":
            return False
        return bool(strongest_visible_seats(state))

    def _fold_marginal(self, state) -> bool:
        return self._strong_visible(state)

    def _two_pair_with_hole(self, state) -> bool:
        board_ranks = {c >> 2 for c in state.board}
        hole_ranks = [c >> 2 for c in state.hole]
        return any(r in board_ranks for r in hole_ranks) or (len(hole_ranks) == 2 and hole_ranks[0] == hole_ranks[1])

    def _their_exposed_misses(self, state, seat) -> bool:
        exposed = getattr(state, "exposed", ())
        card = exposed[seat] if seat < len(exposed) else -1
        return card >= 0 and (card >> 2) not in {c >> 2 for c in state.board}

    def _my_exposed_pairs_board(self, state) -> bool:
        exposed = getattr(state, "exposed", ())
        mine = exposed[state.seat] if state.seat < len(exposed) else -1
        return mine >= 0 and (mine >> 2) in {c >> 2 for c in state.board}

    def _late_streets(self, state):
        """late_streets (Bay Area): everyone keeps the best two of three, so bettors on the turn
        and river mostly hold two pair or better; raise and call much tighter than hold'em."""
        rules = state.rules
        if flag("late_streets", "off") != "on" or not getattr(rules, "exposes", 0) or state.street == 0:
            return None
        legal = state.legal
        if legal.can_check:
            return None
        s = hand_strength(state.hole, state.board, rules)
        if s is None:
            return None
        order = rules.category_order
        cat = order.index(s.category)
        straight_plus = cat >= order.index(STRAIGHT)
        board_ranks = {c >> 2 for c in state.board}
        hole_ranks = [c >> 2 for c in state.hole]
        pocket = len(hole_ranks) == 2 and hole_ranks[0] == hole_ranks[1]
        a_set = pocket and hole_ranks[0] in board_ranks
        uses_hole = any(r in board_ranks for r in hole_ranks) or pocket
        two_pair_plus_hole = s.bucket >= TWO_PAIR_PLUS and uses_hole
        both_hole = all(r in board_ranks for r in hole_ranks) and len(set(hole_ranks)) == 2
        pot_odds = legal.call_amount * 2 <= state.pot
        draw = strong_draw(state.hole, state.board, rules)
        if state.street == 1:  # flop: raise only with two pair+ using both cards, a set, or a combo draw
            combo = draw and flush_draw(state.hole, state.board, rules) and len(straight_outs(state.hole, state.board, rules)) >= 2
            if (s.bucket >= TWO_PAIR_PLUS and (both_hole or a_set)) or straight_plus or combo:
                return raise_to(legal, 3 * max(state.bets))
            if s.bucket >= TWO_PAIR_PLUS:
                return check_or(legal, call=pot_odds)
            return None  # the usual flop logic for the rest
        if straight_plus and len(state.board) >= 5:
            try:  # a straight or flush the board makes on its own is a call-only hand
                board_alone = hand_value((), state.board, rules)
                if board_alone >= s.value:
                    return check_or(legal, call=pot_odds)
            except ValueError:
                pass
        if straight_plus or a_set:
            return raise_to(legal, 3 * max(state.bets))
        if state.street == 2:  # turn: the field's turn bets are mostly two pair plus
            bet = legal.call_amount
            before = state.pot - bet
            return check_or(legal, call=(two_pair_plus_hole or a_set) or (draw and bet * 2 <= before)
                            or (s.bucket == TOP_PAIR and bet * 3 <= before))
        # river
        if self._strong_visible(state):
            return check_or(legal, call=False)
        small = legal.call_amount * 3 <= state.pot - legal.call_amount
        return check_or(legal, call=two_pair_plus_hole or (s.bucket == TOP_PAIR and small))

    def _maniac(self, seat):
        st = self.tracker._seat(seat)
        return st.hands >= 10 and st.shoves / st.hands >= 0.40

    def _fold_to_reraise(self, seat):
        st = self.tracker._seat(seat)
        return self.tracker.fold_to_reraise(seat) if st.reraises_faced >= 5 else 0.73

    def _river_decision(self, state):
        """river_equity: on the river, with every opponent's exposed card known, act on our
        exact equity over their possible hidden cards (bot/river.py)."""
        rules = state.rules
        if (flag("river_equity", "off") != "on" or not getattr(rules, "exposes", 0)
                or len(state.board) < rules.board_size):
            return None
        exposed = state.exposed
        opponents = [exposed[s] if s < len(exposed) else -1
                     for s in range(len(state.stacks)) if s != state.seat and state.in_hand[s]]
        equity = river_equity(state.hole, state.board, opponents, state.dead, rules, RIVER_BUDGET_NS)
        if equity is None:
            return None
        legal = state.legal
        if legal.can_check:
            return raise_to(legal, state.pot // 2) if equity >= RIVER_VALUE_BET else Action(CHECK)
        if equity >= RIVER_RAISE:
            return raise_to(legal, 3 * max(state.bets))
        odds = legal.call_amount / max(1, state.pot + legal.call_amount)  # the pot holds their bet
        return check_or(legal, call=equity > odds + RIVER_MARGIN)

    def _defends(self, state, strength):
        if "defend" not in self.active or state.street != 1 or sum(state.in_hand) != 2:
            return False
        flop = [rec for rec in state.actions if rec.street == 1 and rec.kind == RAISE]
        if len(flop) != 1 or self.tracker.cbet(flop[0].seat) < CBET_DEFEND:
            return False  # a raise, or a bettor who doesn't bet every flop: believe it
        return strength >= WEAK_PAIR or strong_draw(state.hole, state.board, state.rules)

    def _resteal_range(self, state, depth_bb):
        if "defend" not in self.active or not RESTEAL_DEPTH_BB[0] <= depth_bb <= RESTEAL_DEPTH_BB[1]:
            return 0.0
        preflop = [rec for rec in state.actions if rec.street == 0 and rec.kind in (RAISE, CALL)]
        if len(preflop) != 1 or preflop[0].kind != RAISE:
            return 0.0  # only a lone open, nobody in with it yet
        opener, bb = preflop[0].seat, max(1, state.big_blind)
        if (state.stacks[opener] == 0 or preflop[0].amount > STEAL_SIZE_BB * bb
                or len(seats_behind(state, opener)) > LATE
                or self.tracker.fold_to_steal(opener) < RESTEAL_FOLDS):
            return 0.0
        waiting = self._still_to_act(state, state.seat, opener)
        return RESTEAL / (1 + 0.5 * waiting) / self._call_tightness(state, opener)
