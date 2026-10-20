#!/bin/bash
# Launch the three staged signet-battery experiments on Flash, sequentially.
# Run from the Muse VM: copy signet_battery.py to ~/qwen-test/, then:
#   ssh -F ~/.ssh/config flash "bash ~/qwen-test/run_signet_battery.sh"
# Or run directly on Flash from ~/qwen-test.
#
# Appends to battery_run.log with real exit codes (never pipe through tail).
# The signet-watch cron picks up the result JSONs when each finishes.
set -u
cd ~/qwen-test || { echo "FATAL: ~/qwen-test missing"; exit 1; }

PY=.venv/bin/python
[ -x "$PY" ] || { echo "FATAL: $PY not executable"; exit 1; }
[ -f signet_battery.py ] || { echo "FATAL: signet_battery.py not in ~/qwen-test"; exit 1; }

echo "=== data dir ==="
ls data/ | head -40

resolve() { # $1 = description, $2.. = candidate globs (first existing wins)
  local desc="$1"; shift
  if [ -n "${!desc:-}" ]; then echo "${!desc}"; return 0; fi
  for g in "$@"; do
    for f in $g; do
      if [ -f "$f" ]; then echo "$f"; return 0; fi
    done
  done
  echo "FATAL: no file for $desc (tried: $*)" >&2
  return 1
}

# Env overrides (full paths) if auto-resolution guesses wrong:
#   S135_BASE S135_INST S135_TOK  S2360_BASE S2360_INST S2360_TOK  Q3_BASE Q3_INST Q3_TOK
# Exact filenames (2026-10-14 fix): the old glob resolver picked the instruct
# file for --base and the tokenizer_config for --tok under en_US.UTF-8
# collation, voiding two runs and crashing the third. No globs anymore.
S135_BASE="data/smollm1-135m.safetensors"
S135_INST="data/smollm1-135m-instruct.safetensors"
S135_TOK="data/smollm1-tokenizer.json"
S2360_BASE="data/smollm2-360m.safetensors"
S2360_INST="data/smollm2-360m-instruct.safetensors"
S2360_TOK="data/smollm2-360m-tokenizer.json"
Q3_INST="data/qwen3-0.6b.safetensors"
Q3_BASE="$Q3_INST"   # thinking test runs ON vs OFF on the one Qwen3 file
Q3_TOK="data/qwen3-tokenizer.json"
for f in "$S135_BASE" "$S135_INST" "$S135_TOK" "$S2360_BASE" "$S2360_INST" "$S2360_TOK" "$Q3_INST" "$Q3_TOK"; do
  [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
[ "$S135_BASE" != "$S135_INST" ] || { echo "FATAL: smollm135 base==instruct" >&2; exit 1; }
[ "$S2360_BASE" != "$S2360_INST" ] || { echo "FATAL: smollm2360 base==instruct" >&2; exit 1; }

echo "resolved:"
echo "  smollm-135m : $S135_BASE | $S135_INST | $S135_TOK"
echo "  smollm2-360m: $S2360_BASE | $S2360_INST | $S2360_TOK"
echo "  qwen3-0.6b  : $Q3_BASE | $Q3_INST | $Q3_TOK"

LOG=battery_run.log
run_one() { # $1 label, rest = command
  local label="$1"; shift
  echo "=== [$label] start $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "=== [$label] exit=$rc $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $rc
}

FAILED=0
run_one smollm-135m "$PY" signet_battery.py --model smollm-135m \
  --base "$S135_BASE" --instruct "$S135_INST" --tok "$S135_TOK" || FAILED=1
run_one smollm2-360m "$PY" signet_battery.py --model smollm2-360m \
  --base "$S2360_BASE" --instruct "$S2360_INST" --tok "$S2360_TOK" || FAILED=1
run_one qwen3-0.6b "$PY" signet_battery.py --model qwen3-0.6b \
  --base "$Q3_BASE" --instruct "$Q3_INST" --tok "$Q3_TOK" || FAILED=1

echo "=== battery done, failed=$FAILED $(date -u +%FT%TZ) ===" | tee -a "$LOG"
ls -la *.json | tee -a "$LOG"
exit $FAILED
