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

With no features it plays exactly like PushFoldBot.

Omaha hands (four hole cards) get their own feature set. Both range-scaling
features measured the other way round there: preflop equities run so close
that calling a known shover wider pays and tightening for ICM over-folds,
and the preflop ranking only sees the best two of the four cards.
"""

from __future__ import annotations

from bot.hands import TOP_PAIR, WEAK_PAIR
from bot.icm import PAYOUTS, cached_bubble_factor
from bot.preflop import ACTIVE
from bot.pushfold import PushFoldBot, _last_raiser, seats_behind
from bot.tracker import PRIORS, Tracker, start_stacks

FEATURES = ("shoves", "steals", "cbets", "broken", "icm")
V4_FEATURES = ("shoves", "steals", "cbets", "broken")
V5_FEATURES = ("steals", "cbets", "broken", "icm")
# What PolicyBot() and the submission play, for hold'em-style and Omaha hands.
DEFAULT_FEATURES, DEFAULT_OMAHA_FEATURES = V5_FEATURES, V4_FEATURES
OMAHA_HOLE_CARDS = 4

STEAL_ANY_TWO = 0.9  # P(everyone behind folds) at which any two cards steal
ANY_TWO = float("inf")  # a range multiplier every hand clears
MAX_WIDEN, MIN_WIDEN = 3.0, 0.5
STICKY = 0.25  # fold-to-bet below this: value-bet thin and big, never bluff
CBET_FOLDS = 0.35  # fold-to-bet needed before continuation-betting air
ICM_CALL_POWER = 1.0  # calling range shrinks by bubble_factor ** this
ICM_SHOVE_POWER = 1.0  # first-in shove range likewise


class PolicyBot(PushFoldBot):
    def __init__(self, features=None, payouts=PAYOUTS,
                 icm_powers=(ICM_CALL_POWER, ICM_SHOVE_POWER), omaha_features=None,
                 ranks=None):
        """No `features` means the submitted configuration. `omaha_features`
        defaults to the submitted Omaha set with it, and to `features` otherwise.
        `ranks` defaults to the generated starting-hand table (Chen without one)."""
        self.ranks = ACTIVE if ranks is None else ranks
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
        self.payouts = tuple(payouts)
        self.call_power, self.shove_power = icm_powers
        self.tracker = Tracker()

    def new_table(self, seat: int, num_seats: int) -> None:
        self.tracker.new_table(seat, num_seats)

    def hand_over(self, result) -> None:
        self.tracker.hand_over(result)

    def act(self, state):
        self.tracker.observe(state)
        omaha = len(state.hole) >= OMAHA_HOLE_CARDS
        self.active = self.omaha_features if omaha else self.features
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
