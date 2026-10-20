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
#
# BASE resolution must skip instruct files. Glob expansion is alphabetical and
# the instruct filename (smollm-135m-instruct...) sorts before the base file
# (smollm1-135m...), so the plain glob below once picked the instruct weights
# as BOTH base and instruct (delta=0.0, Part A void on 2026-10-14).
resolve_base() { # $1 = env-var name, $2.. = candidate globs (first non-instruct file wins)
  local name="$1"; shift
  if [ -n "${!name:-}" ]; then echo "${!name}"; return 0; fi
  for g in "$@"; do
    for f in $g; do
      case "$f" in *nstruct*) continue ;; esac
      if [ -f "$f" ]; then echo "$f"; return 0; fi
    done
  done
  echo "FATAL: no file for $name (tried: $*)" >&2
  return 1
}
S135_BASE=$(resolve_base S135_BASE "data/*smollm*135m*.safetensors" "data/*135m*.safetensors") || exit 1
S135_INST=$(resolve S135_INST "data/*smollm*135m*instruct*.safetensors" "data/*135m*instruct*.safetensors") || exit 1
S2360_BASE=$(resolve_base S2360_BASE "data/*smollm2*360m*.safetensors" "data/*360m*.safetensors") || exit 1
S2360_INST=$(resolve S2360_INST "data/*smollm2*360m*instruct*.safetensors" "data/*360m*instruct*.safetensors") || exit 1
Q3_INST=$(resolve Q3_INST "data/*qwen3*0.6b*.safetensors") || exit 1
# Qwen3 thinking test runs on the single instruct model (thinking ON vs OFF);
# there is no separate base file in data/, so fall back to the same file.
Q3_BASE=$(resolve Q3_BASE "data/*qwen3*0.6b*base*.safetensors" "data/*qwen3*base*.safetensors" 2>/dev/null) || true
if [ -z "${Q3_BASE:-}" ]; then
  echo "note: no separate qwen3 base file; using instruct file for both (delta stats will be trivial)" >&2
  Q3_BASE="$Q3_INST"
fi
if [ "$Q3_INST" = "$Q3_BASE" ]; then
  echo "note: qwen3 base==instruct (thinking ON/OFF test on one model)" >&2
fi
S135_TOK=$(resolve S135_TOK "data/*smollm*135m*/tokenizer.json" "data/*smollm*135m*tokenizer*.json" "data/*smollm1*tokenizer*.json" "data/*135m*/tokenizer.json" "data/*135m*tokenizer*.json") || exit 1
S2360_TOK=$(resolve S2360_TOK "data/*smollm2*360m*/tokenizer.json" "data/*smollm2*360m*tokenizer*.json" "data/*360m*/tokenizer.json" "data/*360m*tokenizer*.json") || exit 1
Q3_TOK=$(resolve Q3_TOK "data/*qwen3*/tokenizer.json" "data/*qwen3*tokenizer*.json") || exit 1

echo "resolved:"
echo "  smollm-135m : $S135_BASE | $S135_INST | $S135_TOK"
echo "  smollm2-360m: $S2360_BASE | $S2360_INST | $S2360_TOK"
echo "  qwen3-0.6b  : $Q3_BASE | $Q3_INST | $Q3_TOK"

# Hard guard: for the two signet models base must differ from instruct.
# (qwen3-0.6b is a thinking ON/OFF test on one model, so it is exempt.)
if [ "$S135_BASE" = "$S135_INST" ] || [ "$S2360_BASE" = "$S2360_INST" ]; then
  echo "FATAL: base==instruct for a signet model; Part A would be void. Aborting." >&2
  exit 1
fi

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
