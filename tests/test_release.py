"""scripts/pack.sh -> scripts/release.py: the exact upload, checked in isolation, never clobbering."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def pack(*args, env=None):
    return subprocess.run([str(ROOT / "scripts" / "pack.sh"), *args], capture_output=True, text=True,
                          env={**os.environ, **(env or {})})


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("release") / "agent"
    run = pack("--variant", "plo", "--out", str(out), "--tables", "3", "--skip-tests")
    assert run.returncode == 0, run.stderr + run.stdout
    return out, run


def test_the_artifact_holds_only_the_agent_and_the_bot_sources(built):
    out, _ = built
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert ".release" in files
    shipped = [f for f in files if f != ".release"]
    assert "agent.py" in shipped and "bot/rules.py" in shipped and "bot/hand_tables/plo.py" in shipped
    assert all(f == "agent.py" or (f.startswith("bot/") and f.endswith(".py")) for f in shipped), shipped
    assert not list(out.rglob("__pycache__"))
    assert (out / "agent.py").read_text() == (ROOT / "adapter.py").read_text()


def test_the_manifest_records_what_was_built_and_what_is_still_pending(built):
    out, run = built
    manifest = json.loads((out.parent / "agent.manifest.json").read_text())
    assert manifest["variant"]["id"] == "plo" and manifest["variant"]["card_rules"]["hole_cards_used"] == 2
    assert manifest["variant"]["starting_hands"] == "bot/hand_tables/plo.py"
    assert manifest["code"]["revision"] and manifest["tests"] == {"status": "skipped"}
    assert any("act calls" in line for line in manifest["smoke_and_latency"]["output"])
    if shutil.which("poker-agent") is None:
        assert manifest["sdk"] == "stand-in" and manifest["official_checks"]["status"] == "pending"
        assert "NOT YET VALIDATED AGAINST THE OFFICIAL SDK" in run.stdout


def test_it_never_overwrites_a_folder_it_did_not_build(tmp_path):
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "notes.txt").write_text("keep me")
    for extra in ((), ("--replace",)):
        run = pack("--variant", "nlhe", "--out", str(mine), "--skip-tests", *extra)
        assert run.returncode == 1
    assert (mine / "notes.txt").read_text() == "keep me"


def test_it_replaces_its_own_build_only_when_asked(built):
    out, _ = built
    assert pack("--variant", "plo", "--out", str(out), "--skip-tests", "--tables", "1").returncode == 1
    run = pack("--variant", "plo", "--out", str(out), "--skip-tests", "--tables", "1", "--replace")
    assert run.returncode == 0, run.stderr


def test_unregistered_variants_are_refused(tmp_path):
    run = pack("--variant", "rabbit_holdem", "--out", str(tmp_path / "a"), "--skip-tests")
    assert run.returncode == 1 and "isn't registered" in run.stderr
    assert not (tmp_path / "a").exists()


def test_the_isolated_import_cannot_lean_on_the_repo_or_the_stand_in(built, tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.release import ReleaseError, isolated_import

    broken = tmp_path / "broken"
    shutil.copytree(built[0], broken)
    (broken / "bot" / "rules.py").unlink()  # the repo has it, the artifact doesn't
    # Even with the repo and the stand-in on PYTHONPATH, the import must fail.
    monkeypatch.setenv("PYTHONPATH", f"{ROOT}{os.pathsep}{ROOT / 'kitsim'}")
    with pytest.raises(ReleaseError):
        isolated_import(broken)
    assert isolated_import(built[0]) in ("official", "stand-in")


def test_imports_of_simulator_code_are_caught(built, tmp_path):
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.release import ReleaseError, check_artifact

    bad = tmp_path / "bad"
    shutil.copytree(built[0], bad)
    with open(bad / "bot" / "policy.py", "a") as f:
        f.write("\nfrom engine.rules import NLHE  # a slip that works locally\n")
    with pytest.raises(ReleaseError, match="engine"):
        check_artifact(bad)
