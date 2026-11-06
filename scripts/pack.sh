#!/usr/bin/env bash
# Build the upload: agent.py and bot/ only, nothing from the engine, tests or kit stand-in.
#
#   scripts/pack.sh [OUT_DIR]        (default dist/agent)
#
# Then, with the real kit installed: poker-agent check OUT_DIR/agent.py and
# poker-agent pack OUT_DIR OUT.zip (this script runs both when poker-agent exists).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-dist/agent}"

rm -rf "$OUT"
mkdir -p "$OUT"
cp adapter.py "$OUT/agent.py"
cp -R bot "$OUT/bot"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$OUT" -name '.*' -mindepth 1 -prune -exec rm -rf {} +

# The upload may import only the standard library, the SDK and requirements.txt.
if grep -rnE '^\s*(from|import)\s+(engine|kitsim|gen|tests|pokerkit|poker_engine)\b' "$OUT"; then
  echo "pack: the upload imports something that won't exist on the server" >&2
  exit 1
fi

# Import it the way the server does, from its own folder (the stand-in SDK
# answers for poker_agent_sdk until the real one is installed).
SDK_PATH="$(pwd)/kitsim"
python3 -c "import poker_agent_sdk" 2>/dev/null && SDK_PATH=""
(cd "$OUT" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SDK_PATH" python3 -c "import agent; print('agent.py imports and warms up')")

files=$(find "$OUT" -type f | wc -l | tr -d ' ')
size=$(du -sk "$OUT" | cut -f1)
echo "packed $OUT: $files files, ${size} KB (limits: 1,000 entries, 200 MB unpacked, 1 MB agent.py)"

if command -v poker-agent >/dev/null; then
  poker-agent check "$OUT/agent.py"
  poker-agent pack "$OUT" "$OUT.zip"
  echo "upload: $OUT.zip"
fi
