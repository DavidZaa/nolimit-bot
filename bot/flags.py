"""Build flags, for A/B tests in the kit's own simulator.

`poker-agent play` starts every bot with a clean environment, so an A/B build bakes
its flags into this file (scripts/kit_ab.py writes FLAGS into a copy of the package).
An NLB_<NAME> environment variable overrides a flag where the environment does reach
the bot. The committed file has no flags: every feature plays its default.

  select  "smart" (default): our discard and reveal; "default": the kit's (2, 0)
"""

from __future__ import annotations

import os

FLAGS: dict = {}


def flag(name: str, default: str) -> str:
    return os.environ.get(f"NLB_{name.upper()}", FLAGS.get(name, default))
