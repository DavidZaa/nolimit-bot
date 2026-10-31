"""Process-level tuning for worst-case decision latency.

The 1 ms budget is per decision, so the tail matters, not the average.
The usual tail culprit in CPython is a full garbage collection walking every
object, including the large tables we load at import. Freezing everything
alive after import moves it out of the collector's reach, so later
collections only scan the few objects created during play.
"""

from __future__ import annotations

import gc


def tune_gc(disable: bool = False) -> None:
    """Call once, after tables are loaded and before play starts."""
    gc.collect()
    gc.freeze()
    if disable:
        # Only safe if nothing in the process builds reference cycles while playing;
        # the timing test measures whether it's worth it.
        gc.disable()
