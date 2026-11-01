"""Worst-case decision latency on real game states.

The tournament auto-folds any decision over 1 ms, so the tail is what matters.
States are recorded from real engine tables (hold'em and PLO, whose four-card
hands are the baseline's most expensive path) and replayed against the bot in
a tight loop, outside the engine, so only the bot's own time is measured.

Machine noise (other processes, CPU frequency) can push a single sample's
max around, so the hard assertions are on p99.9; the max is printed.
"""

import os

import pytest

from bot.baseline import BaselineBot
from bot.runtime import tune_gc
from bot.safety import SafeBot
from engine.field import CallingStation, Maniac, RandomBot
from engine.latency import LatencyHistogram
from engine.rules import NLHE, PLO, TableConfig
from engine.sng import play_sng
from time import perf_counter_ns

BUDGET_P999_US = 200  # target from the design: far under the 1 ms limit


class Recorder:
    def __init__(self, sink):
        self.inner = BaselineBot()
        self.sink = sink

    def act(self, state):
        self.sink.append(state)
        return self.inner.act(state)


def collect_states(tables, variant):
    states = []
    config = TableConfig(variant=variant, enforce_time=False)
    for t in range(tables):
        bots = [Recorder(states) for _ in range(4)] + [RandomBot(t), RandomBot(t + 1), CallingStation(), Maniac()]
        play_sng(bots, config, seed=t)
    return states


def time_decisions(bot, states, repeats=1):
    hist = LatencyHistogram()
    act = bot.act
    for _ in range(repeats):
        for st in states:
            t0 = perf_counter_ns()
            act(st)
            hist.add(perf_counter_ns() - t0)
    return hist


def _report(label, hist):
    print(
        f"\n{label}: n={hist.count} p50={hist.percentile_us(0.5)}us "
        f"p99.9={hist.percentile_us(0.999)}us max={hist.max_ns // 1000}us"
    )


@pytest.mark.parametrize("variant", [NLHE, PLO], ids=lambda v: v.name)
def test_safe_baseline_latency(variant):
    states = collect_states(40, variant)
    hist = time_decisions(SafeBot(BaselineBot()), states, repeats=3)
    _report(f"safe baseline [{variant.name}]", hist)
    assert hist.count > 2000
    assert hist.percentile_us(0.999) <= BUDGET_P999_US


@pytest.mark.slow
@pytest.mark.parametrize("gc_mode", ["default", "frozen"])
def test_safe_baseline_latency_at_scale(gc_mode):
    states = collect_states(int(os.environ.get("TIMING_TABLES", "400")), NLHE)
    states += collect_states(100, PLO)
    if gc_mode == "frozen":
        tune_gc()
    hist = time_decisions(SafeBot(BaselineBot()), states, repeats=5)
    _report(f"safe baseline at scale [gc {gc_mode}]", hist)
    assert hist.count > 500_000
    assert hist.percentile_us(0.999) <= BUDGET_P999_US
