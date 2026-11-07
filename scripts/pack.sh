#!/usr/bin/env bash
# Build and check the upload: agent.py and bot/ only, nothing from the engine, tests or kit stand-in.
#
#   scripts/pack.sh --variant <SDK variant ID> [--out DIR] [--replace] [--kit KIT_DIR] ...
#
# Everything (build, limits, isolated import, tests, smoke tournament and latency on
# the artifact, the kit's own poker-agent check/pack/play, manifest) is in
# scripts/release.py; see its docstring. It never deletes a folder it didn't build.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
exec "$PY" scripts/release.py "$@"
