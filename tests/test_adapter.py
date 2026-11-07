"""adapter.Agent under the kit's protocol: memory-only state, events, payouts, legal actions."""

import json
import random

import poker_agent_sdk as sdk
import pytest

import adapter
from adapter import Agent, decide, load_memory, to_kit_action
from bot.policy import PolicyBot
from bot.safety import SafeBot
from bot.state import RAISE, Action
from engine.field import ChenTAG, EquityCaller, FolderBot, Maniac, PushFold
from engine.kit_bridge import KitSeat
from engine.rules import NLHE, PLO, SHORT_DECK, TableConfig
from engine.run import BOTS
from engine.sng import play_sng
from test_policy import Twin


def kit_state(events=(), legal=None, phase="betting", payouts=(50, 30, 20), memory_seat=3):
    legal = legal or (sdk.LegalAction("fold"), sdk.LegalAction("call", 20, 20), sdk.LegalAction("raise", 40, 2000))
    seats = tuple(sdk.Seat(10 + i, 2000 - (10 if i == 1 else 20 if i == 2 else 0),
                           10 if i == 1 else 20 if i == 2 else 0, "active") for i in range(8))
    events = events or (sdk.Event("hand", None, 1), sdk.Event("blind", 11, 10), sdk.Event("blind", 12, 20))
    return sdk.State(seat=10 + memory_seat, cards=("As", "Ad"), board=(), pot=30, to_call=20,
                     legal_actions=legal, seats=seats, button=10, blinds=(10, 20, 0), hand_number=1,
                     players_remaining=8, events=events, phase=phase,
                     config=sdk.Config(payouts=tuple(payouts)))


class TrackerTwin(Twin):
    """A Twin that also checks, at every decision, that the tracker the agent rebuilt
    from events and memory holds exactly the counts of the one fed directly."""

    def act(self, state):
        action = super().act(state)
        direct = self.primary.inner.tracker.to_list()
        if load_memory(self.shadow.memory)[0].to_list() != direct:
            self.splits.append((state, "tracker", direct))
        return action


@pytest.mark.parametrize("variant", [NLHE, SHORT_DECK, PLO], ids=lambda v: v.name)
def test_through_the_protocol_it_plays_exactly_like_the_direct_bot(variant):
    # Fresh Agent every decision, state only in memory, seats anonymised: same decisions,
    # and the tracker rebuilt from events matches the one fed directly, count for count.
    cfg = TableConfig(variant=variant, enforce_time=False)
    splits, twins = [], []
    for t in range(6):
        pair = [TrackerTwin(SafeBot(PolicyBot()), KitSeat(Agent, cfg), splits)
                for _ in range(3)]
        twins += pair
        bots = pair + [ChenTAG(), EquityCaller(), Maniac(), FolderBot(), PushFold()]
        random.Random(t).shuffle(bots)
        play_sng(bots, cfg, seed=t)
    assert sum(t.decisions for t in twins) > 800
    assert not splits


def test_opponent_stats_live_in_memory_and_stay_small():
    cfg = TableConfig(enforce_time=False)
    seats = [KitSeat(Agent, cfg) for _ in range(2)]
    biggest = 0

    class Watch:
        def __init__(self, inner):
            self.inner = inner

        def new_table(self, *a):
            self.inner.new_table(*a)

        def hand_over(self, r):
            self.inner.hand_over(r)

        def act(self, gs):
            nonlocal biggest
            action = self.inner.act(gs)
            biggest = max(biggest, len(self.inner.memory))
            return action

    bots = [Watch(seats[0]), Maniac(), FolderBot(), Watch(seats[1]), Maniac(), FolderBot(), ChenTAG(), PushFold()]
    play_sng(bots, cfg, seed=3)
    tracker, _ = load_memory(seats[0].memory)
    assert biggest < adapter.MEMORY_BUDGET
    assert max(st.hands for st in tracker.seats) > 10
    seen_enough = [s for s in (1, 4) if tracker.seats[s].hands >= 10]  # maniacs often bust early
    assert seen_enough and all(tracker.vpip(s) > 0.6 for s in seen_enough)
    assert tracker.vpip(2) < 0.1 and tracker.vpip(5) < 0.1  # the folders


@pytest.mark.parametrize("memory", [b"", b"\x00\xffnot json", b"[1,[[1]]", json.dumps([99, [], []]).encode(),
                                    json.dumps([1, "x", None]).encode()])
def test_unreadable_memory_starts_the_table_fresh(memory):
    action, new_memory = decide(kit_state(), memory)
    assert action.kind in ("fold", "call", "raise")
    tracker, log = load_memory(new_memory)
    assert log.hand == 1 and len(tracker.seats) == 8


def test_unknown_event_kinds_are_ignored():
    plain = kit_state()
    odd = kit_state(events=plain.events + (sdk.Event("rabbit", 12, 7), sdk.Event("raise", 99, 500)))
    assert decide(plain, b"")[0] == decide(odd, b"")[0]


def test_an_unknown_phase_takes_the_automatic_choice():
    legal = (sdk.LegalAction("discard", 1, 2), sdk.LegalAction("pass"))
    action, _ = decide(kit_state(legal=legal, phase="discard"), b"")
    assert action == sdk.Action("discard", 1)


def test_raises_are_whole_chips_inside_the_offered_range():
    state = kit_state()
    assert to_kit_action(Action(RAISE, 10**9), state) == sdk.Action.raise_to(2000)
    assert to_kit_action(Action(RAISE, 3), state) == sdk.Action.raise_to(40)
    no_raise = kit_state(legal=(sdk.LegalAction("fold"), sdk.LegalAction("call", 20, 20)))
    assert to_kit_action(Action(RAISE, 500), no_raise) == sdk.Action.call()


def test_each_table_plays_by_its_own_payouts(monkeypatch):
    seen = []
    original = adapter.to_game_state

    def spy(state, log, index, rules):
        seen.append(rules.payouts)
        return original(state, log, index, rules)

    monkeypatch.setattr(adapter, "to_game_state", spy)
    decide(kit_state(payouts=(100,)), b"")
    decide(kit_state(payouts=(10, 9, 8, 7, 6, 5, 4, 3)), b"")
    assert seen == [(100.0,) + (0.0,) * 7, (10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0)]


def test_the_agent_keeps_nothing_on_itself():
    agent = Agent()
    action, memory = agent.act(kit_state(), b"")
    assert vars(agent) == {} and isinstance(memory, bytes)


def test_aces_raise_first_in():
    action, _ = decide(kit_state(), b"")
    assert action.kind == "raise" and 40 <= action.amount <= 2000


def test_the_runner_and_the_agent_play_the_same_strategy():
    assert type(adapter.build_bot().inner) is type(BOTS["ours"](0, TableConfig()).inner)


def test_the_bot_and_the_simulator_use_the_one_default_points_setting():
    from bot import icm
    from engine.rules import PLACEHOLDER_PAYOUTS

    assert PLACEHOLDER_PAYOUTS is icm.PAYOUTS and TableConfig().payouts is icm.PAYOUTS
