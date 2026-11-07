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

from bot.hands import TOP_PAIR, WEAK_PAIR, strong_draw
from bot.icm import cached_bubble_factor
from bot.preflop import ranks_for
from bot.rules import RulesError
from bot.state import CALL, RAISE
from bot.pushfold import PushFoldBot, _last_raiser, seats_behind
from bot.tracker import PRIORS, Tracker, start_stacks

FEATURES = ("shoves", "steals", "cbets", "broken", "icm", "defend")
V4_FEATURES = ("shoves", "steals", "cbets", "broken")
V5_FEATURES = ("steals", "cbets", "broken", "icm")
V6_FEATURES = V5_FEATURES + ("defend",)  # candidate, not submitted until it clears both fields
# What PolicyBot() and the submission play, for hold'em-style and Omaha hands.
DEFAULT_FEATURES, DEFAULT_OMAHA_FEATURES = V5_FEATURES, V4_FEATURES

STEAL_ANY_TWO = 0.9  # P(everyone behind folds) at which any two cards steal
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
        if p_all_fold >= STEAL_ANY_TWO:
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
        if "cbets" not in self.active:
            return super()._bet_when_checked_to(state, strength)
        others = [s for s in range(len(state.stacks)) if s != state.seat and state.in_hand[s]]
        folds = min((self.tracker.fold_to_bet(s) for s in others), default=PRIORS["fold_to_bet"])
        if strength >= TOP_PAIR:
            return state.pot * 3 // 4 if folds < STICKY else state.pot // 2
        if strength == WEAK_PAIR and folds < STICKY:
            return state.pot // 2
        heads_up = len(others) == 1
        cbet = state.street == 1 and heads_up and _last_raiser(state, 0) == state.seat
        if cbet and folds >= CBET_FOLDS:
            return state.pot // 2
        return None

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
