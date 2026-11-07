"""Build flags, for A/B tests in the kit's own simulator.

`poker-agent play` starts every bot with a clean environment, so an A/B build bakes
its flags into this file (scripts/kit_ab.py writes FLAGS into a copy of the package).
An NLB_<NAME> environment variable overrides a flag where the environment does reach
the bot. The committed file has no flags: every feature plays its default.

  select         "smart" (default): our discard and reveal; "default": the kit's (2, 0)
  short_handed   heads-up ranges (bot/pushfold.py)
  late_position  late-position widening and play-back (bot/pushfold.py)
  calibrated     the top teams' open and defend rates and lower-card exposure
  exposed_reads, river_equity, survival: measured, off
"""

from __future__ import annotations

import os

FLAGS: dict = {"short_handed": "on", "late_position": "on", "calibrated": "on", "exposed_reads": "on"}  # round 3


def flag(name: str, default: str) -> str:
    return os.environ.get(f"NLB_{name.upper()}", FLAGS.get(name, default))
