"""Worst-case decision latency on real game states.

The tournament auto-folds any decision over 1 ms, so the tail is what matters.
States are recorded from real engine tables the bot itself plays (hold'em and
PLO, whose four-card hands are the most expensive path) and replayed against
it in a tight loop, outside the engine, so only the bot's own time is
measured. This is wall-clock time on a quiet process; the engine's own
timing is CPU time.

Machine noise (other processes, CPU frequency) can push a single sample's
wall-clock max into milliseconds on a desktop, so the wall-clock assertions
are on p99.9. The at-scale tests also hold every decision's CPU time, which
noise can't inflate, under the 1 ms limit.
"""

import os

import pytest

from bot.baseline import BaselineBot
from bot.policy import FEATURES, PolicyBot
from bot.pushfold import PushFoldBot
from bot.runtime import tune_gc
from bot.safety import SafeBot
from engine.field import CallingStation, ChenTAG, Maniac, RandomBot
from engine.latency import LatencyHistogram
from engine.rules import NLHE, PLO, TableConfig
from engine.sng import play_sng
from time import perf_counter_ns, thread_time_ns

BUDGET_P999_US = 200  # target from the design: far under the 1 ms limit
FLAT = (30, 20, 15, 12, 9, 7, 5, 2)  # the slowest points for ICM: every place pays differently
BOTS = {
    "baseline": BaselineBot,
    "pushfold": PushFoldBot,
    "policy": PolicyBot,
    "policy_all_flat": lambda: PolicyBot(FEATURES, payouts=FLAT),
}


class Recorder:
    """Records every state the bot sees and passes the table hooks through, so a
    tracker-driven bot plays (and is recorded) with a real tracker."""

    def __init__(self, inner, sink):
        self.inner = inner
        self.sink = sink

    def new_table(self, seat, num_seats):
        hook = getattr(self.inner, "new_table", None)
        if hook:
            hook(seat, num_seats)

    def hand_over(self, result):
        hook = getattr(self.inner, "hand_over", None)
        if hook:
            hook(result)

    def act(self, state):
        self.sink.append(state)
        return self.inner.act(state)


def collect_states(tables, variant, make_bot):
    states = []
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(tables):
        bots = [Recorder(make_bot(), states) for _ in range(4)] + [
            RandomBot(t), ChenTAG(), CallingStation(), Maniac()]
        play_sng(bots, config, seed=t)
    return states


def time_decisions(bot, states, repeats=1, cpu=None):
    """Wall-clock histogram of every decision; `cpu`, if given, collects thread CPU times."""
    hist = LatencyHistogram()
    act = bot.act
    for _ in range(repeats):
        for st in states:
            t0, c0 = perf_counter_ns(), thread_time_ns()
            act(st)
            if cpu is not None:
                cpu.append(thread_time_ns() - c0)
            hist.add(perf_counter_ns() - t0)
    return hist


def _report(label, hist):
    print(
        f"\n{label}: n={hist.count} p50={hist.percentile_us(0.5)}us "
        f"p99.9={hist.percentile_us(0.999)}us max={hist.max_ns // 1000}us"
    )


@pytest.mark.parametrize("variant", [NLHE, PLO], ids=lambda v: v.name)
@pytest.mark.parametrize("name", BOTS)
def test_safe_bot_latency(name, variant):
    states = collect_states(40, variant, BOTS[name])
    hist = time_decisions(SafeBot(BOTS[name]()), states, repeats=3)
    _report(f"safe {name} [{variant.name}]", hist)
    assert hist.count > 2000
    assert hist.percentile_us(0.999) <= BUDGET_P999_US


@pytest.mark.slow
@pytest.mark.parametrize("gc_mode", ["default", "frozen"])
@pytest.mark.parametrize("name", BOTS)
def test_safe_bot_latency_at_scale(name, gc_mode):
    states = collect_states(int(os.environ.get("TIMING_TABLES", "400")), NLHE, BOTS[name])
    states += collect_states(100, PLO, BOTS[name])
    if gc_mode == "frozen":
        tune_gc()
    cpu = []
    hist = time_decisions(SafeBot(BOTS[name]()), states, repeats=5, cpu=cpu)
    _report(f"safe {name} at scale [gc {gc_mode}]", hist)
    print(f"  worst CPU time {max(cpu) // 1000}us")
    assert hist.count > 500_000
    assert hist.percentile_us(0.999) <= BUDGET_P999_US
    # The wall-clock max includes the OS pausing us on a busy machine; CPU time
    # is what the bot's own code costs, and no decision may ever cost the limit.
    assert max(cpu) < 1_000_000
