"""Independent Chip Model (Malmuth-Harville), exact.

Each remaining player wins the next-best place with probability proportional
to their stack; recurse on who is left. Memoised over subsets, so 8 players is
2^8 states: instant for scoring a table, but still far too slow to run inside
a 1 ms bot decision (the bot will use precomputed tables instead).
"""

from __future__ import annotations

from functools import lru_cache


def icm_equity(stacks, payouts) -> list[float]:
    """Expected payout per player. `payouts[k]` is the prize for place k+1."""
    n = len(stacks)
    prizes = tuple(payouts[k] if k < len(payouts) else 0.0 for k in range(n))
    stacks = tuple(stacks)

    @lru_cache(maxsize=None)
    def equity(mask: int) -> tuple[float, ...]:
        players = [i for i in range(n) if mask >> i & 1]
        place = n - len(players)
        result = [0.0] * n
        if not players:
            return tuple(result)
        total = sum(stacks[i] for i in players)
        for i in players:
            # With nothing left to tell them apart, the remaining places are shared.
            p = stacks[i] / total if total else 1 / len(players)
            if p == 0:
                continue
            rest = equity(mask & ~(1 << i))
            for j in players:
                result[j] += p * (prizes[place] if j == i else rest[j])
        return tuple(result)

    return list(equity((1 << n) - 1))
