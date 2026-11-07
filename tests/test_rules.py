"""Explicit rules: SDK config -> GameRules, validated, never guessed, and the same in the simulator."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import poker_agent_sdk as sdk
import pytest

import adapter
from adapter import decide
from bot import rules as R
from bot.policy import DEFAULT_FEATURES, DEFAULT_OMAHA_FEATURES, PolicyBot
from bot.preflop import CHEN, card_fingerprint, ranks_for, table_status
from bot.state import GameRules
from engine.kit_bridge import sdk_config
from engine.rules import NLHE, PLO, SHORT_DECK, BlindSchedule, TableConfig, game_rules, parse_variant
from helpers import decisions

ROOT = Path(__file__).resolve().parent.parent
PRESETS = [NLHE, SHORT_DECK, PLO]
CONFIGS = [
    TableConfig(variant=v, **extra) for v in PRESETS for extra in (
        {},
        {"payouts": (65, 35), "seats": 6, "starting_stack": 1500, "max_hands": 180,
         "schedule": BlindSchedule(((25, 50, 5), (50, 100, 10), (75, 150, 15)), 12)},
        {"payouts": (30, 20, 15, 12, 9, 7, 5, 2)},
    )
]


def kit_state(config, cards=("As", "Ad"), legal=None):
    legal = legal or (sdk.LegalAction("fold"), sdk.LegalAction("call", 20, 20), sdk.LegalAction("raise", 40, 2000))
    seats = tuple(sdk.Seat(10 + i, 2000 - (10 if i == 1 else 20 if i == 2 else 0),
                           10 if i == 1 else 20 if i == 2 else 0, "active") for i in range(8))
    events = (sdk.Event("hand", None, 1), sdk.Event("blind", 11, 10), sdk.Event("blind", 12, 20))
    return sdk.State(seat=13, cards=cards, board=(), pot=30, to_call=20, legal_actions=legal, seats=seats,
                     button=10, blinds=(10, 20, 0), hand_number=1, players_remaining=8, events=events,
                     config=config)


# -- translation -----------------------------------------------------------------------

@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"{c.variant.name}-{c.seats}seats-{c.payouts[:2]}")
def test_every_config_field_survives_translation_exactly(config):
    rules = R.rules_from_config(sdk_config(config))
    assert rules == game_rules(config)  # the SDK path and the simulator build the same rules
    v = config.variant
    assert (rules.variant, rules.ranks, rules.hole_cards, rules.hole_cards_used, rules.preflop_board,
            rules.board_schedule, rules.category_order, rules.ace_low_straight) == (
        v.name, v.ranks, v.hole_cards, v.hole_cards_used, v.preflop_board, v.board_schedule,
        v.category_order, v.ace_low_straight)
    assert rules.betting == ("pot_limit" if v.pot_limit else "no_limit")
    assert rules.suits == ("c", "d", "h", "s")
    assert rules.payouts == tuple(float(p) for p in config.payouts) + (0.0,) * (config.seats - len(config.payouts))
    assert rules.starting_stack == config.starting_stack
    assert rules.blind_schedule == config.schedule.levels
    assert rules.hands_per_level == config.schedule.hands_per_level
    assert rules.hand_cap == config.max_hands and rules.seats == config.seats


def test_the_registry_and_the_simulator_presets_agree():
    for v in PRESETS:
        registered = R.card_rules(v.name)
        built = game_rules(TableConfig(variant=v))
        assert card_fingerprint(registered) == card_fingerprint(built)
        assert registered["betting"] == built.betting


def test_every_registered_variant_is_valid():
    for name, fields in R.VARIANTS.items():
        assert R.check_card_rules(fields)["hole_cards"] >= 1, name


def test_blind_levels_can_come_as_pairs_dicts_or_objects():
    class Level:
        small, big, ante = 15, 30, 3

    cfg = replace(sdk_config(TableConfig()), blinds=((10, 20), {"small": 10, "big": 20, "ante": 2}, Level()))
    assert R.rules_from_config(cfg).blind_schedule == ((10, 20, 0), (10, 20, 2), (15, 30, 3))


def test_rules_serialize_and_come_back_identical():
    for config in CONFIGS:
        rules = game_rules(config)
        doc = json.loads(json.dumps(R.to_dict(rules)))
        assert R.from_dict(doc) == rules


# -- nothing is guessed --------------------------------------------------------------------

GOOD = sdk_config(TableConfig())
BAD_CONFIGS = {
    "unknown variant": dict(variant="rabbit_holdem"),
    "variant missing": dict(variant=None),
    "variant not a string": dict(variant=7),
    "payouts missing": dict(payouts=None),
    "payouts empty": dict(payouts=()),
    "payouts negative": dict(payouts=(60, -10, 50)),
    "payouts not numbers": dict(payouts=("first", "second")),
    "payouts all zero": dict(payouts=(0, 0, 0)),
    "payouts nan": dict(payouts=(float("nan"), 30, 20)),
    "payouts past the seats": dict(payouts=(30,) * 9 + (5,)),
    "blinds missing": dict(blinds=None),
    "blinds empty": dict(blinds=()),
    "blinds garbage": dict(blinds=("ten/twenty",)),
    "big blind zero": dict(blinds=((0, 0, 0),)),
    "small over big": dict(blinds=((40, 20, 0),)),
    "level length zero": dict(hands_per_level=0),
    "hand cap missing": dict(hand_cap=None),
    "players a string": dict(players="8"),
    "one player": dict(players=1),
    "stack fractional": dict(starting_stack=1999.5),
}


@pytest.mark.parametrize("change", BAD_CONFIGS.values(), ids=BAD_CONFIGS.keys())
def test_missing_malformed_or_unknown_rules_are_refused(change):
    with pytest.raises(R.RulesError):
        R.rules_from_config(replace(GOOD, **change))


@pytest.mark.parametrize("change", BAD_CONFIGS.values(), ids=BAD_CONFIGS.keys())
def test_strict_mode_fails_closed_on_bad_rules(change, monkeypatch):
    # NLB_STRICT=1 (pack.sh, local checks): refuse, and take the safe legal action.
    monkeypatch.setenv("NLB_STRICT", "1")
    with pytest.raises(R.RulesError):
        R.rules_from_state(kit_state(replace(GOOD, **change)))
    action, memory = decide(kit_state(replace(GOOD, **change)), b"")
    assert action == sdk.Action.fold() and memory


@pytest.mark.parametrize("change", BAD_CONFIGS.values(), ids=BAD_CONFIGS.keys())
def test_by_default_bad_rules_still_play_the_strategy(change, monkeypatch):
    # In the tournament a check/fold bot looks healthy and scores nothing: play on a guess.
    monkeypatch.delenv("NLB_STRICT", raising=False)
    good = decide(kit_state(GOOD), b"")[0]
    assert good.kind == "raise"  # aces facing a raise
    assert decide(kit_state(replace(GOOD, **change)), b"")[0] == good


def test_strict_mode_reports_a_rules_error_once(capsys, monkeypatch):
    monkeypatch.setenv("NLB_STRICT", "1")
    adapter._reported.clear()
    bad = kit_state(replace(GOOD, variant="unregistered_variant_x"))
    decide(bad, b"")
    decide(bad, b"")
    err = capsys.readouterr().err
    assert err.count("RULES ERROR") == 1 and "unregistered_variant_x" in err


def test_each_guess_is_warned_about_once(capsys, monkeypatch):
    monkeypatch.delenv("NLB_STRICT", raising=False)
    adapter._reported.clear()
    bad = kit_state(replace(GOOD, variant="unregistered_variant_y", payouts=None))
    decide(bad, b"")
    decide(bad, b"")
    err = capsys.readouterr().err
    assert err.count("RULES WARNING") == 2 and "unregistered_variant_y" in err and "50/30/20" in err
    assert "RULES ERROR" not in err


# -- what each guess is ----------------------------------------------------------------------

class Bare:
    """A config with only the given fields: the rest are simply missing, not None."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


FULL = dict(variant="nlhe", players=8, starting_stack=2000, blinds=((10, 20, 0), (20, 40, 0)),
            hands_per_level=20, hand_cap=250, payouts=(50, 30, 20, 0, 0, 0, 0, 0))


@pytest.mark.parametrize("missing", list(FULL))
def test_each_missing_field_still_gives_a_normal_strategy_action(missing, monkeypatch):
    monkeypatch.delenv("NLB_STRICT", raising=False)
    config = Bare(**{k: v for k, v in FULL.items() if k != missing})
    rules, warnings = R.rules_from_state(kit_state(config))
    assert warnings and any(missing in w or (missing == "variant" and "variant" in w) for w in warnings) or (
        missing == "blinds" and any("blinds" in w for w in warnings))
    assert decide(kit_state(config), b"")[0].kind == "raise"


def test_the_guesses_are_the_documented_defaults(monkeypatch):
    monkeypatch.delenv("NLB_STRICT", raising=False)
    rules, _ = R.rules_from_state(kit_state(Bare(variant="nlhe")))
    assert rules.payouts == (50.0, 30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # 50/30/20 over the 8 seats
    assert rules.blind_schedule == ((10, 20, 0),)  # the state's current blinds
    assert (rules.hands_per_level, rules.hand_cap, rules.seats) == (20, 250, 8)
    six = Bare(**dict(FULL, players=6, payouts="bad"))
    assert R.rules_from_state(kit_state(six))[0].payouts == (50.0, 30.0, 20.0, 0.0, 0.0, 0.0)


def test_an_unknown_variant_is_inferred_from_the_state(monkeypatch):
    from bot.preflop import table_status

    monkeypatch.delenv("NLB_STRICT", raising=False)
    cfg = Bare(**dict(FULL, variant="mystery_omaha"))
    four = kit_state(cfg, cards=("As", "Ad", "Kh", "Qc"), legal=(
        sdk.LegalAction("fold"), sdk.LegalAction("call", 20, 20), sdk.LegalAction("raise", 40, 90)))
    rules, warnings = R.rules_from_state(four)
    assert rules.variant == "mystery_omaha" and rules.hole_cards == 4 and rules.hole_cards_used == 2
    assert rules.betting == "pot_limit"  # the max raise (90) is below the all-in (2000)
    assert "mystery_omaha" in warnings[0]
    assert "plo.py" in table_status(rules)  # same card rules as the PLO table: it's used
    two = R.rules_from_state(kit_state(Bare(**dict(FULL, variant="mystery_holdem"))))[0]
    assert two.hole_cards == 2 and two.hole_cards_used is None and two.betting == "no_limit"


def test_only_nothing_usable_at_all_checks_or_folds(monkeypatch):
    monkeypatch.delenv("NLB_STRICT", raising=False)
    hopeless = replace(kit_state(Bare(variant="nlhe")), blinds="??")  # no schedule, no current blinds
    with pytest.raises(R.RulesError):
        R.rules_from_state(hopeless)
    assert decide(hopeless, b"")[0] == sdk.Action.fold()


def test_invalid_variant_entries_are_refused():
    good = dict(R.VARIANTS["nlhe"])
    for change in ({"ranks": (0, 1, 2, 3)}, {"ranks": (12, 11, 10, 9, 8)}, {"suits": ("x", "y")},
                   {"hole_cards_used": 3, "hole_cards": 2}, {"category_order": (0, 1, 2)},
                   {"betting": "fixed_limit"}, {"ace_low_straight": "yes"}, {"board_schedule": (1,)}):
        with pytest.raises(R.RulesError):
            R.check_card_rules({**good, **change})


# -- the policy reads the rules, not the hole-card count ----------------------------------

def test_a_state_without_rules_is_refused_not_guessed():
    from bot.safety import SafeBot
    from bot.state import FOLD, Action

    state = decisions(PolicyBot(), 3, "AsAd", "Kh7s2dJc4h")[0][0]
    no_rules = replace(state, rules=None)
    with pytest.raises(R.RulesError):
        PolicyBot().act(no_rules)
    assert SafeBot(PolicyBot()).act(no_rules) == Action(FOLD)


def test_the_feature_set_follows_the_omaha_rule_not_the_hole_cards(monkeypatch):
    # Four hole cards with hold'em's best-five rule plays the hold'em features, and two
    # hole cards with an exactly-two rule plays the Omaha ones.
    four_any = parse_variant("base=nlhe,hole=4")
    two_exact = parse_variant("base=nlhe,use=2")
    for variant, expected in ((four_any, DEFAULT_FEATURES), (two_exact, DEFAULT_OMAHA_FEATURES)):
        bot = PolicyBot()
        cards = "AsAdKhKc" if variant.hole_cards == 4 else "AsAd"
        board = "Kh7s2dJc4h" if variant.hole_cards == 2 else "Qh7s2dJc4h"
        decisions(bot, 3, cards, board, config=TableConfig(variant=variant))
        assert bot.active == frozenset(expected), variant.name


# -- starting-hand tables, one per variant ---------------------------------------------------

def test_each_variant_gets_its_own_table_and_hold_em_gets_chen():
    assert table_status(game_rules(TableConfig(variant=PLO))) == "bot/hand_tables/plo.py"
    assert table_status(game_rules(TableConfig(variant=SHORT_DECK))) == "bot/hand_tables/short_deck.py"
    assert ranks_for(game_rules(TableConfig(variant=NLHE))) is CHEN
    assert ranks_for(game_rules(TableConfig(variant=PLO))).hole_cards == 4
    assert ranks_for(game_rules(TableConfig(variant=SHORT_DECK))).variant == "short_deck"


def test_a_table_made_for_other_rules_is_not_used():
    # Named plo, but trips beat straights here: the stamped table doesn't apply.
    odd = replace(game_rules(TableConfig(variant=PLO)), category_order=R.SHORT_DECK_ORDER)
    assert ranks_for(odd) is CHEN and "other card rules" in table_status(odd)
    # No table under this ID, but one simulated under exactly these card rules: it's used.
    unnamed = replace(game_rules(TableConfig(variant=PLO)), variant="not a module name")
    assert ranks_for(unnamed).percentiles is ranks_for(game_rules(TableConfig(variant=PLO))).percentiles
    assert "same card rules" in table_status(unnamed)
    six_card = replace(unnamed, hole_cards=6)
    assert ranks_for(six_card) is CHEN


# -- the shipped package stands alone ----------------------------------------------------------

def test_the_shipped_agent_imports_with_only_its_own_files_and_the_sdk(tmp_path):
    art, site = tmp_path / "artifact", tmp_path / "site"
    art.mkdir()
    (art / "agent.py").write_text((ROOT / "adapter.py").read_text())
    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree(ROOT / "bot", art / "bot", ignore=ignore)
    shutil.copytree(ROOT / "kitsim" / "poker_agent_sdk", site / "poker_agent_sdk", ignore=ignore)
    probe = (
        "import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]\n"
        "import agent\n"
        "repo = sys.argv[3]\n"
        "for name in ('engine', 'gen', 'tests', 'kitsim', 'pokerkit', 'poker_engine', 'adapter'):\n"
        "    assert name not in sys.modules, name\n"
        "stray = [m.__file__ for m in list(sys.modules.values())\n"
        "         if getattr(m, '__file__', None) and m.__file__.startswith(repo)]\n"
        "assert not stray, stray\n"
        "assert sys.modules['bot'].__file__.startswith(sys.argv[1]), sys.modules['bot'].__file__\n"
        "assert agent.__file__.startswith(sys.argv[1])\n"
        "print('isolated import ok')\n"
    )
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    # -I: no PYTHONPATH, no cwd, no user site; -S: no site-packages either (the venv holds an
    # editable install of this repo, which would otherwise resolve `bot` to the source tree).
    out = subprocess.run([sys.executable, "-I", "-S", "-c", probe, str(art), str(site), str(ROOT)],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "isolated import ok" in out.stdout


def test_nothing_in_bot_imports_simulator_code():
    for path in (ROOT / "bot").rglob("*.py"):
        text = path.read_text()
        for banned in ("engine", "gen", "tests", "kitsim", "pokerkit", "poker_engine"):
            assert f"from {banned}" not in text and f"import {banned}" not in text, (path, banned)


@pytest.mark.skip(reason="PENDING: the official poker_agent_sdk ships at kickoff; validate translation "
                         "against a real state.config (variant ID, blinds shape, payouts) then")
def test_translation_against_the_official_sdk():
    pass
