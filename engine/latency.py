"""Fixed-size latency histogram.

A tournament round is millions of decisions, so keeping every sample is out.
Buckets are 1 µs wide below 100 µs, 10 µs below 1 ms and 100 µs below 10 ms,
which keeps the tail around the 1 ms limit readable and merging cheap.
"""

from __future__ import annotations

_BUCKETS = 281


def _bucket(us: int) -> int:
    if us < 100:
        return us
    if us < 1000:
        return 100 + (us - 100) // 10
    if us < 10_000:
        return 190 + (us - 1000) // 100
    return 280


def _upper_us(b: int) -> int:
    if b < 100:
        return b + 1
    if b < 190:
        return 100 + (b - 99) * 10
    if b < 280:
        return 1000 + (b - 189) * 100
    return 10**9


class LatencyHistogram:
    __slots__ = ("counts", "count", "max_ns")

    def __init__(self) -> None:
        self.counts = [0] * _BUCKETS
        self.count = 0
        self.max_ns = 0

    def add(self, ns: int) -> None:
        self.counts[_bucket(ns // 1000)] += 1
        self.count += 1
        if ns > self.max_ns:
            self.max_ns = ns

    def merge(self, other: "LatencyHistogram") -> None:
        mine = self.counts
        for i, c in enumerate(other.counts):
            if c:
                mine[i] += c
        self.count += other.count
        self.max_ns = max(self.max_ns, other.max_ns)

    def percentile_us(self, q: float) -> int:
        """Upper bound (µs) of the bucket holding the q-quantile, 0 <= q <= 1."""
        if not self.count:
            return 0
        target = q * self.count
        seen = 0
        for b, c in enumerate(self.counts):
            seen += c
            if seen >= target and c:
                return _upper_us(b)
        return _upper_us(_BUCKETS - 1)
