"""Build and check the exact upload. Run through scripts/pack.sh.

    scripts/pack.sh --variant <SDK variant ID> [--out DIR] [--replace] [--engine-variant SPEC]
                    [--tables 200] [--kit KIT_DIR] [--skip-tests] [--allow-chen]

Steps, each fatal on failure:
  1. The variant ID must be registered in bot/rules.py (the bot would check or
     fold every hand otherwise), and its starting-hand table is reported; a
     variant with more than two hole cards and no table needs --allow-chen.
  2. Build OUT with agent.py (from adapter.py) and bot/**.py only. OUT must not
     exist yet, unless --replace (which only ever replaces a folder this script
     built: it must hold a .release marker).
  3. Check the artifact: only the intended files, within the kit's limits
     (1,000 entries, 200 MB unpacked, 1 MB agent.py, 512-char paths, 16 levels),
     and no imports of simulator or local-only modules.
  4. Import it in an isolated interpreter (-I -S: no PYTHONPATH, no cwd, no
     site-packages) with only the artifact and the SDK on the path; the official
     SDK when installed, else the stand-in, and the manifest says which.
  5. Unless --skip-tests: the test suite (.venv/bin/pytest -q).
  6. A smoke tournament and latency check on the artifact itself
     (scripts/bench_act.py --agent-dir OUT --strict).
  7. With the kit installed (poker-agent on PATH): poker-agent check, pack, and
     play against the kit's examples, failing on any RULES ERROR or RULES WARNING it
     prints (the bot had to guess part of its rules from the real config).
     Without it those checks are reported as PENDING, never as passed.
  8. Write OUT.manifest.json beside OUT (not inside: it isn't uploaded):
     variant and rules, revision, test and bench results, environment, SDK.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMITS = {"entries": 1000, "unpacked_bytes": 200 * 1024 * 1024, "agent_bytes": 1024 * 1024,
          "path_chars": 512, "path_levels": 16}
FORBIDDEN = ("engine", "kitsim", "gen", "tests", "scripts", "pokerkit", "poker_engine")
MARKER = ".release"


class ReleaseError(RuntimeError):
    pass


def say(msg: str) -> None:
    print(f"release: {msg}", flush=True)


def check_variant(variant: str, allow_chen: bool) -> dict:
    sys.path.insert(0, str(ROOT))
    from bot.preflop import table_status
    from bot.rules import VARIANTS, RulesError, make_rules

    if variant not in VARIANTS:
        raise ReleaseError(f"variant {variant!r} isn't registered in bot/rules.py VARIANTS "
                           f"(known: {sorted(VARIANTS)}); the bot would check or fold every hand")
    try:
        rules = make_rules(variant, payouts=(50, 30, 20), starting_stack=2000, blind_schedule=((10, 20, 0),),
                           hands_per_level=20, hand_cap=250, seats=8)
    except RulesError as exc:
        raise ReleaseError(f"variant {variant!r} is registered but invalid: {exc}") from None
    status = table_status(rules)
    say(f"variant {variant}: {rules.hole_cards} hole cards, "
        f"{'any five' if rules.hole_cards_used is None else f'exactly {rules.hole_cards_used} from the hand'}, "
        f"{rules.betting}; starting hands: {status}")
    if "Chen" in status and rules.hole_cards != 2 and not allow_chen:
        raise ReleaseError(f"no starting-hand table for {variant} ({rules.hole_cards} hole cards): generate one "
                           f"(python -m gen.hand_ranks --id {variant} ...) or pass --allow-chen")
    card = {k: getattr(rules, k) for k in ("ranks", "hole_cards", "hole_cards_used", "preflop_board",
                                           "board_schedule", "category_order", "ace_low_straight", "betting")}
    return {"id": variant, "card_rules": card, "starting_hands": status}


def build(out: Path, replace: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not replace:
            raise ReleaseError(f"{out} already exists and isn't empty; pick a new --out or pass --replace")
        if not (out / MARKER).exists():
            raise ReleaseError(f"{out} wasn't built by this script (no {MARKER} marker); not deleting it")
        shutil.rmtree(out)
        zipped = out.parent / f"{out.name}.zip"
        if zipped.exists():  # poker-agent pack won't overwrite it; it's the old build's upload
            zipped.unlink()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "adapter.py", out / "agent.py")
    shutil.copytree(ROOT / "bot", out / "bot", ignore=lambda d, names: [
        n for n in names if n.startswith(".") or n == "__pycache__" or (
            (Path(d) / n).is_file() and not n.endswith(".py"))])
    (out / MARKER).write_text("built by scripts/release.py; safe for --replace to delete\n")


def artifact_files(out: Path) -> list[Path]:
    """What the upload contains (everything but the marker, which pack excludes as hidden)."""
    return sorted(p for p in out.rglob("*") if not any(part.startswith(".") for part in p.relative_to(out).parts))


def check_artifact(out: Path) -> dict:
    files = artifact_files(out)
    rel = [p.relative_to(out).as_posix() for p in files]
    unexpected = [r for r in rel if not (r == "agent.py" or r == "bot" or (r.startswith("bot/") and (
        r.endswith(".py") or (out / r).is_dir())))]
    if unexpected:
        raise ReleaseError(f"unexpected files in the artifact: {unexpected[:10]}")
    if "agent.py" not in rel or "bot/__init__.py" not in rel:
        raise ReleaseError("the artifact is missing agent.py or bot/__init__.py")
    size = sum(p.stat().st_size for p in files if p.is_file())
    agent = (out / "agent.py").stat().st_size
    longest = max(len(r) for r in rel)
    deepest = max(r.count("/") + 1 for r in rel)
    for name, value in (("entries", len(rel)), ("unpacked_bytes", size), ("agent_bytes", agent),
                        ("path_chars", longest), ("path_levels", deepest)):
        if value > LIMITS[name]:
            raise ReleaseError(f"artifact over the kit's limit: {name} {value} > {LIMITS[name]}")
    for p in files:
        if p.suffix == ".py":
            for n, line in enumerate(p.read_text().splitlines(), 1):
                words = line.strip().split()
                if len(words) >= 2 and words[0] in ("import", "from") and words[1].split(".")[0] in FORBIDDEN:
                    raise ReleaseError(f"{p.relative_to(out)}:{n} imports {words[1]}, which won't exist on the server")
    say(f"artifact {out}: {len(rel)} entries, {size / 1024:.0f} KB, agent.py {agent / 1024:.0f} KB "
        f"(limits: {LIMITS['entries']} entries, 200 MB, 1 MB)")
    return {"entries": len(rel), "bytes": size, "agent_bytes": agent, "files": rel}


def sdk_site() -> tuple[str, str]:
    """(directories holding the SDK packages, joined by os.pathsep; 'official' or 'stand-in').
    The official SDK is two packages (poker_agent_sdk and poker_agent_protocol), which an
    editable install keeps in separate source folders."""
    probe = subprocess.run([sys.executable, "-c", "import os, poker_agent_sdk, poker_agent_protocol as p; "
                            "d = lambda m: os.path.dirname(os.path.dirname(os.path.abspath(m.__file__))); "
                            "print(d(poker_agent_sdk)); print(d(p))"],
                           capture_output=True, text=True, cwd=tempfile.gettempdir(),
                           env={"PATH": os.environ.get("PATH", "")})
    found = [line for line in probe.stdout.split() if line]
    if probe.returncode == 0 and found and not any(f.startswith(str(ROOT / "kitsim")) for f in found):
        return os.pathsep.join(dict.fromkeys(found)), "official"
    site = Path(tempfile.mkdtemp(prefix="release-sdk-"))
    shutil.copytree(ROOT / "kitsim" / "poker_agent_sdk", site / "poker_agent_sdk",
                    ignore=shutil.ignore_patterns("__pycache__"))
    return str(site), "stand-in"


def isolated_import(out: Path) -> str:
    site, which = sdk_site()
    probe = (
        "import os, sys; sys.path[:0] = [sys.argv[1], *sys.argv[2].split(os.pathsep)]\n"
        "import agent, bot\n"
        "assert agent.__file__.startswith(sys.argv[1]) and bot.__file__.startswith(sys.argv[1])\n"
        "for name in %r:\n"
        "    assert name not in sys.modules, name\n"
        "assert not getattr(agent, '_reported', None), sorted(agent._reported)\n"
        "print('ok')\n" % (FORBIDDEN,)
    )
    # -I ignores PYTHONDONTWRITEBYTECODE, so -B keeps bytecode out of the artifact.
    run = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", probe, str(out.resolve()), site],
                         capture_output=True, text=True, cwd=tempfile.gettempdir(),
                         env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"})
    if run.returncode != 0 or "ok" not in run.stdout:
        raise ReleaseError(f"the artifact doesn't import on its own:\n{run.stderr[-2000:]}")
    say(f"isolated import ok (only the artifact and the {which} SDK on the path)")
    return which


def run_tests() -> dict:
    pytest = ROOT / ".venv" / "bin" / "pytest"
    run = subprocess.run([str(pytest), "-q"], cwd=ROOT, capture_output=True, text=True)
    last = (run.stdout.strip().splitlines() or ["(no output)"])[-1]
    if run.returncode != 0:
        raise ReleaseError(f"tests failed: {last}\n{run.stdout[-3000:]}")
    say(f"tests: {last}")
    return {"command": ".venv/bin/pytest -q", "result": last}


def smoke(out: Path, engine_variant: str, tables: int) -> dict:
    cmd = [sys.executable, "-B", str(ROOT / "scripts" / "bench_act.py"), "--agent-dir", str(out), "--variant",
           engine_variant, "--tables", str(tables), "--strict"]
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                         env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"})
    print(run.stdout, end="")
    if f"(bot/ from {out.resolve() / 'bot'})" not in run.stdout:
        raise ReleaseError("the smoke run didn't use the artifact's bot/ package")
    if run.returncode != 0:
        raise ReleaseError(f"smoke tournament / latency check failed:\n{run.stderr[-2000:]}")
    return {"command": " ".join(cmd[1:]), "output": run.stdout.strip().splitlines()}


def official(out: Path, kit: Path | None, tables: int) -> dict:
    tool = shutil.which("poker-agent")
    if tool is None:
        say("PENDING: poker-agent isn't installed; official check/pack/play not run (they are NOT passed)")
        return {"status": "pending", "reason": "poker-agent not installed"}
    results = {"status": "run"}
    for name, cmd in (("check", [tool, "check", str(out / "agent.py")]),
                      ("pack", [tool, "pack", str(out), f"{out}.zip"])):
        run = subprocess.run(cmd, capture_output=True, text=True)
        results[name] = {"returncode": run.returncode, "output": (run.stdout + run.stderr)[-2000:]}
        if run.returncode != 0:
            raise ReleaseError(f"poker-agent {name} failed:\n{run.stdout}{run.stderr}")
        say(f"poker-agent {name} ok")
    if kit is None:
        say("PENDING: --kit not given, so poker-agent play against the kit's examples wasn't run")
        results["play"] = {"status": "pending", "reason": "no --kit"}
        return results
    examples = [str(kit.resolve() / "examples" / e) for e in ("check_or_fold", "min_raiser")]
    out = out.resolve()  # play runs from the kit's folder
    run = subprocess.run([tool, "play", str(out), str(out), str(out), str(out), *examples, *examples,
                          "--tables", str(tables)], capture_output=True, text=True, cwd=kit)
    results["play"] = {"returncode": run.returncode, "stdout": run.stdout[-4000:], "stderr": run.stderr[-4000:]}
    # play starts bots with a clean environment (no NLB_STRICT), so guesses show up as warnings.
    if run.returncode != 0 or "RULES ERROR" in run.stderr or "RULES WARNING" in run.stderr:
        raise ReleaseError(f"poker-agent play failed or the bot had to guess its rules:\n{run.stderr[-3000:]}")
    say("poker-agent play ok (no RULES ERROR or RULES WARNING)")
    return results


def git_info() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"revision": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", required=True, help="the SDK's variant ID (state.config.variant)")
    p.add_argument("--out", type=Path, default=ROOT / "dist" / "agent")
    p.add_argument("--replace", action="store_true", help="replace an OUT this script built earlier")
    p.add_argument("--engine-variant", default=None,
                   help="the simulator's spec for this variant, for the smoke run (default: the ID)")
    p.add_argument("--tables", type=int, default=200, help="smoke tournament tables")
    p.add_argument("--kit", type=Path, default=None, help="the unpacked starter kit, for poker-agent play")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--allow-chen", action="store_true")
    args = p.parse_args(argv)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        variant = check_variant(args.variant, args.allow_chen)
        build(args.out, args.replace)
        artifact = check_artifact(args.out)
        sdk = isolated_import(args.out)
        tests = {"status": "skipped"} if args.skip_tests else run_tests()
        bench = smoke(args.out, args.engine_variant or args.variant, args.tables)
        kit = official(args.out, args.kit, args.tables)
        for cache in args.out.rglob("__pycache__"):  # nothing the checks ran may linger in the upload
            shutil.rmtree(cache)
        if check_artifact(args.out)["files"] != artifact["files"]:
            raise ReleaseError("the artifact changed while it was being checked")
    except ReleaseError as exc:
        print(f"release: FAILED: {exc}", file=sys.stderr)
        return 1
    manifest = {
        "built": started, "variant": variant, "artifact": artifact, "sdk": sdk, "code": git_info(),
        "tests": tests, "smoke_and_latency": bench, "official_checks": kit,
        "payouts_and_blinds": "read per table from state.config at runtime; the smoke run used the "
                              "published defaults (50/30/20, 10/20 doubling every 20 hands, cap 250)",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine()},
    }
    manifest_path = args.out.parent / f"{args.out.name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, default=list))
    say(f"manifest {manifest_path}")
    if sdk != "official" or kit.get("status") == "pending":
        say("NOT YET VALIDATED AGAINST THE OFFICIAL SDK: install the kit and rerun before uploading")
    else:
        say(f"upload: {args.out}.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
