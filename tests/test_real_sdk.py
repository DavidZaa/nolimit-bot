"""The adapter against the official SDK's own types (skipped where the kit isn't installed).

Builds real poker_agent_protocol.game objects (GameState, TableConfig, BlindLevel, Event,
Seat, LegalAction), runs adapter.decide on them, and checks every reply with the kit's
own legality check and its strict reply parser (action_from_dict).
"""

import pytest

game = pytest.importorskip("poker_agent_protocol.game")

import adapter  # noqa: E402
from bot import rules as R  # noqa: E402

LEVELS = (game.BlindLevel(10, 20, 0), game.BlindLevel(20, 40, 0), game.BlindLevel(30, 60, 5))


def config(variant="flop_discard_reveal", **changes):
    fields = dict(table_id="t", seat=3, players=8, variant=variant, starting_stack=2000, blinds=LEVELS,
                  hands_per_level=20, hand_cap=250, payouts=(50, 30, 20), decision_ms=1, memory_bytes=65536,
                  time_bank_ms=100)
    fields.update(changes)
    return game.TableConfig(**fields)


def seats(stacks=None, bets=None, shown=()):
    stacks = stacks or [2000, 1990, 1980, 2000, 2000, 2000, 2000, 2000]
    bets = bets or [0, 10, 20, 0, 0, 0, 0, 0]
    return tuple(game.Seat(i, stacks[i], bets[i], "active", shown[i] if i < len(shown) else ())
                 for i in range(8))


def preflop(cards=("As", "Ad", "Kh"), cfg=None):
    events = (game.Event("hand", None, 1), game.Event("blind", 1, 10), game.Event("blind", 2, 20))
    legal = (game.LegalAction("fold"), game.LegalAction("call"), game.LegalAction("raise", 40, 2000))
    return game.GameState(events=events, hand_number=1, seat=3, cards=cards, board=(), pot=30, to_call=20,
                          legal_actions=legal, history=events, seats=seats(), button=0, blind_level=1,
                          blinds=LEVELS[0], hands_until_next_level=19, players_remaining=8, bust_order=(),
                          config=cfg or config())


def select(cards=("Kh", "7c", "2d"), board=("Kd", "9s", "4c")):
    events = (game.Event("hand", None, 1), game.Event("blind", 1, 10), game.Event("blind", 2, 20),
              game.Event("call", 3, 20), game.Event("fold", 4), game.Event("fold", 5), game.Event("fold", 6),
              game.Event("fold", 7), game.Event("fold", 0), game.Event("call", 1, 10), game.Event("check", 2),
              game.Event("board", None, 0, board), game.Event("select", 1), game.Event("select", 2))
    return game.GameState(events=events, hand_number=1, seat=3, cards=cards, board=board, pot=60, to_call=0,
                          legal_actions=(game.LegalAction("select"),), history=events,
                          seats=seats([2000, 1980, 1980, 1980, 2000, 2000, 2000, 2000], [0] * 8),
                          button=0, blind_level=1, blinds=LEVELS[0], hands_until_next_level=19,
                          players_remaining=8, bust_order=(), config=config(), phase="select")


def valid(action, state) -> bool:
    reply = {"kind": action.kind, "amount": action.amount, "cards": list(getattr(action, "cards", ()))}
    parsed = game.action_from_dict(reply)  # raises on anything the server would reject
    return game.is_legal(parsed, state.legal_actions)


def test_real_blind_levels_parse_for_rules_and_the_current_blinds(monkeypatch):
    monkeypatch.setenv("NLB_STRICT", "1")  # no guessing allowed: the real shapes must just work
    state = preflop()
    rules, warnings = R.rules_from_state(state)
    assert warnings == ()
    assert rules.blind_schedule == ((10, 20, 0), (20, 40, 0), (30, 60, 5))
    assert rules.payouts == (50.0, 30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert (rules.hole_cards, rules.discards, rules.exposes) == (3, 1, 1)
    assert adapter._blinds(state, adapter.TableLog()) == (10, 20, 0)
    assert R.blind_level(game.BlindLevel(125, 250, 25)) == (125, 250, 25)


@pytest.mark.parametrize("variant", ["flop_discard_reveal", "bay_area"])
def test_both_variant_ids_play_the_strategy_strictly(variant, monkeypatch):
    monkeypatch.setenv("NLB_STRICT", "1")
    state = preflop(cfg=config(variant))
    action, memory = adapter.decide(state, b"")
    assert action.kind == "raise" and valid(action, state) and memory  # AAK facing a raise


def test_the_selection_keeps_the_pair_and_shows_the_other_card():
    state = select()
    action, memory = adapter.decide(state, b"")
    assert action.kind == "select" and action.amount == 0
    assert tuple(action.cards) == (2, 1)  # throw the 2d, keep K7, show the 7 (hide the king)
    assert valid(action, state) and memory


def test_a_broken_selection_falls_back_to_the_kits_default(monkeypatch):
    monkeypatch.setattr(adapter, "choose_select", lambda *a: 1 / 0)
    action, memory = adapter.decide(select(), b"")
    assert (action.kind, action.amount, tuple(action.cards)) == ("select", 0, (2, 0)) and memory


def test_reveal_events_are_remembered_and_reach_the_strategy():
    state = select()
    _, memory = adapter.decide(state, b"")
    later = game.Event("reveal", 1, 0, ("Qh",)), game.Event("reveal", 2, 0, ("5c",)), game.Event("muck", 4)
    tracker, log = adapter.load_memory(memory)
    log.feed(later, adapter.seat_index(state), tracker, 2000, 8)
    assert log.exposed == {1: "Qh", 2: "5c"}
    restored = adapter.TableLog.from_list(log.to_list())
    assert restored.exposed == log.exposed
    gs = adapter.to_game_state(state, restored, adapter.seat_index(state), R.rules_from_state(state)[0])
    assert gs.exposed[1] == adapter.card_from_str("Qh") and gs.exposed[3] == -1
