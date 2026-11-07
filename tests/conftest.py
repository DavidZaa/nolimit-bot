"""Tests assume every strategy flag is off unless they turn one on themselves.

bot/flags.py may ship with flags on for the current round; the environment
overrides it, so each test starts from the plain strategy.
"""

import pytest


@pytest.fixture(autouse=True)
def _flags_off(monkeypatch):
    for name in ("SELECT", "SHORT_HANDED", "LATE_POSITION", "CALIBRATED", "EXPOSED_READS", "RIVER_EQUITY", "CAL_TUNED", "RESTEAL", "BUBBLE", "DEEP", "CLOSER", "DEEP_LITE", "DEEP_FIGHT",
                 "SURVIVAL"):
        monkeypatch.setenv(f"NLB_{name}", "off" if name != "SELECT" else "smart")
