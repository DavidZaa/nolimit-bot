import random
import time

from bot.state import CALL, CHECK, FOLD, POST_ANTE, POST_BB, POST_SB, RAISE, Action
from bot.state import cards_from_str as C
from engine.rules import BlindSchedule, TableConfig
from engine.table import TableStats, blind_seats, play_hand
from helpers import Exploding, RandomLegal, ReturnsJunk, Scripted


def cfg(sb=10, bb=20, ante=0, **kw):
    return TableConfig(schedule=BlindSchedule(((sb, bb, ante),)), **kw)


def play(bots, stacks, holes, board, button=0, config=None, stats=None):
    return play_hand(
        bots, list(stacks), button, 1, config or cfg(), random.Random(0),
        holes={s: C(h) for s, h in holes.items()}, board=C(board), stats=stats,
    )


def test_blind_seats_full_ring_and_heads_up():
    assert blind_seats([100] * 8, button=7) == (0, 1)
    # busted seats are skipped
    assert blind_seats([100, 0, 100, 0, 100, 0, 0, 0], button=0) == (2, 4)
    # heads-up: the button posts the small blind
    assert blind_seats([0, 100, 0, 100, 0, 0, 0, 0], button=3) == (3, 1)


def test_everyone_folds_to_big_blind():
    bots = [Scripted(Action(FOLD)), Scripted(), Scripted(Action(FOLD))]
    # button 0 -> SB 1, BB 2; seat 0 acts first preflop, then SB folds by fallback
    out = play(bots, [1000] * 3, {0: "2c3d", 1: "4c5d", 2: "6c7d"}, "AsKsQsJsTs")
    assert out.stacks == [1000, 990, 1010]
    assert out.board == ()  # no board dealt for an uncontested pot
    assert out.winners == (2,)


def test_heads_up_button_acts_first_preflop_and_last_after():
    a, b = Scripted(Action(CALL)), Scripted(Action(CHECK), Action(CHECK))
    play([a, b], [1000, 1000], {0: "2c3d", 1: "4c5d"}, "AsKsQhJdTc")
    assert a.seen[0].street == 0 and a.seen[0].sb_seat == 0
    assert b.seen[0].street == 0  # big blind option
    assert b.seen[1].street == 1 and len(a.seen) >= 2 and a.seen[1].street == 1
    # on the flop the big blind (seat 1) is asked before the button
    assert b.seen[1].actions[-1].street == 0


def test_big_blind_gets_its_option_after_limps():
    bots = [Scripted(Action(CALL)), Scripted(Action(CALL)), Scripted()]
    play(bots, [1000] * 3, {0: "2c3d", 1: "4c5d", 2: "6c7d"}, "AsKsQhJdTc")
    bb_state = bots[2].seen[0]
    assert bb_state.legal.can_check and bb_state.legal.can_raise
    assert bb_state.legal.min_raise_to == 40


def test_big_blind_still_gets_asked_when_everyone_else_is_all_in():
    # Button calls all-in for 7, SB folds. Nothing is left to bet into, but the
    # BB still gets its turn (check only), as PokerKit does mid-street.
    bots = [Scripted(Action(CALL)), Scripted(Action(FOLD)), Scripted()]
    play(bots, [7, 420, 52], {0: "2c3d", 1: "4c5d", 2: "6c7d"}, "AsKsQhJdTc",
         config=cfg(sb=25, bb=50))
    assert len(bots[2].seen) == 1
    legal = bots[2].seen[0].legal
    assert legal.can_check and not legal.can_raise


def test_big_blind_not_asked_when_already_covering_every_opponent():
    # Heads-up, the button has 47 total and the BB already posted 50: the BB has
    # nothing left to decide, so it never gets a turn.
    bots = [Scripted(), Scripted(Action(CALL))]
    out = play(bots, [1358, 47], {0: "2c3d", 1: "4c5d"}, "9h5s4hJs6s", button=1, config=cfg(sb=25, bb=50))
    assert bots[0].seen == []
    assert sum(out.stacks) == 1405


def test_no_betting_rounds_after_everyone_but_one_is_all_in():
    bots = [Scripted(Action(RAISE, 500)), Scripted(Action(CALL)), Scripted(Action(FOLD))]
    out = play(bots, [500, 2000, 2000], {0: "AsAd", 1: "2c3d", 2: "6c7d"}, "Ks9h7d4c2s", button=2)
    # SB shoves, BB calls with chips behind; no flop/turn/river decisions follow.
    assert all(st.street == 0 for b in bots for st in b.seen)
    assert out.stacks[0] == 1000


def test_forced_bets_are_logged_and_short_blinds_post_all_in():
    bots = [Scripted(), Scripted(), Scripted()]
    out = play(bots, [1000, 1000, 15], {0: "2c3d", 1: "4c5d", 2: "6c7d"}, "AsKsQhJdTc")
    first = bots[0].seen[0]
    assert [(r.seat, r.kind, r.amount) for r in first.actions] == [(1, POST_SB, 10), (2, POST_BB, 15)]
    # callers match what was actually posted; the min raise is still a full big blind on top
    assert first.legal.call_amount == 15
    assert first.legal.min_raise_to == 35
    assert sum(out.stacks) == 2015


def test_antes_come_before_blinds():
    bots = [Scripted(), Scripted(), Scripted()]
    out = play(bots, [1000, 5, 1000], {0: "2c3d", 1: "4c5d", 2: "6c7d"}, "AsKsQhJdTc",
               config=cfg(ante=10))
    kinds = [(r.seat, r.kind, r.amount) for r in bots[0].seen[0].actions]
    assert kinds == [(0, POST_ANTE, 10), (1, POST_ANTE, 5), (2, POST_ANTE, 10), (2, POST_BB, 20)]
    assert sum(out.stacks) == 2005


def test_incomplete_all_in_raise_does_not_reopen_betting():
    # button 3 -> SB 0, BB 1, seat 2 first; seat 3 has only 130.
    s2 = Scripted(Action(RAISE, 100))
    bots = [Scripted(Action(CALL)), Scripted(Action(CALL)), s2, Scripted(Action(RAISE, 130))]
    play(bots, [1000, 1000, 1000, 130], {0: "2c3d", 1: "4c5d", 2: "6c7d", 3: "8c9d"},
         "AsKsQhJdTc", button=3)
    facing = s2.seen[1]
    assert facing.legal.can_call and not facing.legal.can_raise


def test_full_raise_does_reopen_betting():
    s2 = Scripted(Action(RAISE, 100))
    bots = [Scripted(Action(FOLD)), Scripted(Action(FOLD)), s2, Scripted(Action(RAISE, 180))]
    play(bots, [1000, 1000, 1000, 1000], {0: "2c3d", 1: "4c5d", 2: "6c7d", 3: "8c9d"},
         "AsKsQhJdTc", button=3)
    facing = s2.seen[1]
    assert facing.legal.can_raise and facing.legal.min_raise_to == 260


def test_no_raising_when_nobody_could_call_more():
    # Seat 3 shoves 1000. The blinds still have chips, but neither can put in
    # more than 1000, so seat 0 may only call or fold.
    bots = [Scripted(), Scripted(), Scripted(), Scripted(Action(RAISE, 1000))]
    play(bots, [2000, 600, 400, 1000], {0: "2c3d", 1: "4c5d", 2: "6c7d", 3: "8c9d"}, "AsKsQhJdTc")
    legal = bots[0].seen[0].legal
    assert legal.can_call and not legal.can_raise


def test_short_all_ins_that_add_up_to_a_full_raise_reopen_betting():
    # Seat 3 opens to 100 (an 80 raise). Two short all-ins (+50, +60) add up to
    # more than 80, so when it comes back seat 3 may raise again (PokerKit rule).
    s3 = Scripted(Action(RAISE, 100))
    bots = [Scripted(Action(RAISE, 150)), Scripted(Action(RAISE, 210)), Scripted(Action(CALL)), s3]
    play(bots, [150, 210, 2000, 2000], {0: "2c3d", 1: "4c5d", 2: "6c7d", 3: "8c9d"}, "AsKsQhJdTc")
    facing = s3.seen[1].legal
    assert facing.can_raise and facing.min_raise_to == 290


def test_short_stack_can_only_raise_all_in():
    bots = [Scripted(), Scripted(), Scripted()]
    # button 2 -> SB 0, BB 1, so the 30-chip button is first to act facing 20
    play(bots, [1000, 1000, 30], {0: "2c3d", 1: "4c5d", 2: "6c7d"}, "AsKsQhJdTc", button=2)
    short = bots[2].seen[0].legal
    assert short.can_raise and short.min_raise_to == short.max_raise_to == 30


def test_side_pots_pay_the_right_players():
    # A (seat 0, SB) is short with the best hand, B (BB) second best, C (button) worst.
    # C limps, A shoves 100, B shoves 500, C calls.
    bots = [Scripted(Action(RAISE, 100)), Scripted(Action(RAISE, 500)),
            Scripted(Action(CALL), Action(CALL))]
    out = play(bots, [100, 500, 500], {0: "AsAd", 1: "KsKd", 2: "QsQd"}, "2c7h9dJc3s", button=2)
    assert out.stacks == [300, 800, 0]
    assert sum(out.stacks) == 1100


def test_uncalled_part_of_a_bet_is_returned():
    # Button (seat 2, 300 chips) limps, SB shoves 1000, BB folds, button calls all-in.
    bots = [Scripted(Action(RAISE, 1000)), Scripted(Action(FOLD)), Scripted(Action(CALL), Action(CALL))]
    out = play(bots, [1000, 1000, 300], {0: "AsAd", 1: "2c3d", 2: "KsKd"}, "2h7h9dJc3s", button=2)
    # Only 300 of the shove could be called; the other 700 was never at risk.
    assert out.stacks == [1000 + 300 + 20, 980, 0]


def test_split_pot_odd_chip_goes_left_of_button():
    # Ante 1 makes a 23-chip pot; SB (seat 1) and BB (seat 2) chop with the same hand.
    bots = [Scripted(Action(FOLD)), Scripted(Action(CALL)), Scripted(Action(CHECK))]
    out = play(bots, [1000] * 3, {0: "2c3d", 1: "AsKd", 2: "AhKc"}, "QsJh7d2s3s",
               config=cfg(sb=5, bb=10, ante=1))
    assert out.stacks == [999, 1001, 1000]


def test_three_way_chop_gives_whole_remainder_to_first_left_of_button():
    # Antes of 2 make a 38-chip pot; three players chop a royal on board: 12 each,
    # and both leftover chips go to the SB (PokerKit's convention).
    bots = [Scripted(Action(FOLD)), Scripted(Action(CALL)), Scripted(Action(CHECK)), Scripted(Action(CALL))]
    out = play(bots, [1000] * 4, {0: "2c3d", 1: "4c5d", 2: "6c7d", 3: "8c9d"}, "AsKsQsJsTs",
               config=cfg(sb=5, bb=10, ante=2))
    assert out.stacks == [998, 1002, 1000, 1000]


def test_crashes_junk_and_timeouts_become_check_or_fold():
    stats = TableStats(3)
    bots = [Exploding(), ReturnsJunk("shove"), ReturnsJunk(Action(RAISE, 10**9))]
    out = play(bots, [1000] * 3, {0: "2c3d", 1: "4c5d", 2: "6c7d"}, "AsKsQhJdTc", stats=stats)
    # seat 0 folds (error), SB folds (junk), BB keeps the blinds
    assert out.stacks == [1000, 990, 1010]
    assert stats.errors[0] == 1 and stats.illegal[1] == 1
    assert sum(stats.decisions) == 2


class Thinks:
    """Answers `action` after spending `ms` either computing or asleep."""

    def __init__(self, action, ms, busy=True):
        self.action, self.ns, self.busy = action, int(ms * 1e6), busy

    def act(self, state):
        if self.busy:
            end = time.thread_time_ns() + self.ns
            while time.thread_time_ns() < end:
                pass
        else:
            time.sleep(self.ns / 1e9)
        return self.action


def test_slow_decisions_are_treated_as_timeouts():
    stats = TableStats(2)
    slow = Thinks(Action(RAISE, 1000), ms=2)
    out = play([slow, Scripted()], [1000, 1000], {0: "AsAd", 1: "2c3d"}, "KsQh7d4c9s", stats=stats)
    assert stats.timeouts[0] == 1
    assert out.stacks == [990, 1010]


def test_time_the_os_spends_elsewhere_does_not_count():
    # Sleeping stands in for being descheduled on a busy machine: it uses no
    # CPU on the bot's thread, so it must not cost the bot its decision.
    stats = TableStats(2)
    paused = Thinks(Action(RAISE, 1000), ms=3, busy=False)
    out = play([paused, Scripted()], [1000, 1000], {0: "AsAd", 1: "2c3d"}, "KsQh7d4c9s", stats=stats)
    assert stats.timeouts[0] == 0
    assert out.stacks == [1020, 980]  # the shove stands and wins the big blind


def test_chips_are_conserved_over_random_hands():
    rng = random.Random(5)
    stats_total = 0
    for i in range(2000):
        n = rng.randint(2, 8)
        stacks = [rng.choice((0, rng.randint(1, 60), rng.randint(1, 3000))) for _ in range(n)]
        alive = [s for s in range(n) if stacks[s] > 0]
        if len(alive) < 2:
            continue
        sb = rng.choice((5, 10, 25))
        config = cfg(sb=sb, bb=2 * sb, ante=rng.choice((0, 0, sb // 5)))
        stats = TableStats(n)
        bots = [RandomLegal(seed=i * 10 + s) for s in range(n)]
        out = play_hand(bots, stacks, rng.choice(alive), 1, config, random.Random(i), stats=stats)
        assert sum(out.stacks) == sum(stacks), (i, stacks, out.actions)
        assert min(out.stacks) >= 0
        assert all(out.stacks[s] == 0 for s in range(n) if stacks[s] == 0)
        assert sum(stats.illegal) == sum(stats.errors) == 0
        stats_total += sum(stats.decisions)
    assert stats_total > 5000
