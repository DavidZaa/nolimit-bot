import pytest

from bot.state import (
    CALL,
    CHECK,
    FOLD,
    LegalActions,
    card_from_str,
    card_to_str,
    cards_from_str,
    fallback_action,
)


def test_every_card_round_trips():
    seen = set()
    for c in range(52):
        s = card_to_str(c)
        assert card_from_str(s) == c
        seen.add(s)
    assert len(seen) == 52


def test_card_encoding_is_rank_major():
    # rank * 4 + suit lets evaluators get rank with c >> 2 and suit with c & 3
    assert card_from_str("2c") == 0
    assert card_from_str("As") == 51
    assert card_from_str("Td") >> 2 == 8


def test_cards_from_str_parses_runs_and_spaces():
    assert cards_from_str("AsKd") == (51, card_from_str("Kd"))
    assert cards_from_str("As Kd 2c") == cards_from_str("AsKd2c")
    assert cards_from_str("") == ()


def test_bad_card_string_raises():
    with pytest.raises(ValueError):
        card_from_str("1x")


def test_fallback_checks_when_free_and_folds_otherwise():
    free = LegalActions(False, True, False, 0, True, 20, 1000)
    facing = LegalActions(True, False, True, 40, True, 80, 1000)
    assert fallback_action(free).kind == CHECK
    assert fallback_action(facing).kind == FOLD
    assert CALL != FOLD
