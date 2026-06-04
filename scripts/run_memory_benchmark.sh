#!/usr/bin/env bash
#
# Parallel personalization benchmark across memory types.
#
# Usage:
#   bash scripts/run_memory_benchmark.sh                  # all 6 memory types
#   bash scripts/run_memory_benchmark.sh null rewrite     # only selected types
#
# Logs and results are saved under data/simulations/memory_benchmark_<timestamp>/
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Configuration ─────────────────────────────────────────────────────────────
ALL_MEMORY_TYPES=("null" "full_context" "rewrite" "rag" "rag_cache" "groundtruth")

# Use CLI args if provided, otherwise run all
if [ $# -gt 0 ]; then
    MEMORY_TYPES=("$@")
else
    MEMORY_TYPES=("${ALL_MEMORY_TYPES[@]}")
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# vita run internally prepends data/simulations/ to --save-to, so use a relative subdir name
SAVE_SUBDIR="memory_benchmark_${TIMESTAMP}"
RUN_DIR="data/simulations/${SAVE_SUBDIR}"
mkdir -p "$RUN_DIR"

echo "============================================================"
echo "  Memory Benchmark - ${#MEMORY_TYPES[@]} types"
echo "  Types: ${MEMORY_TYPES[*]}"
echo "  Output: ${RUN_DIR}/"
echo "  Started: $(date)"
echo "============================================================"

# Save run config
cat > "$RUN_DIR/run_config.json" <<REOF
{
  "timestamp": "$TIMESTAMP",
  "memory_types": $(printf '%s\n' "${MEMORY_TYPES[@]}" | jq -R . | jq -s .),
  "domain": "personalization",
  "agent_llm": "gpt-4.1",
  "user_llm": "gpt-4.1",
  "evaluator_llm": "gpt-4.1"
}
REOF

# ── Launch parallel jobs ──────────────────────────────────────────────────────
PIDS=()
for mem_type in "${MEMORY_TYPES[@]}"; do
    LOG_FILE="${RUN_DIR}/${mem_type}.log"
    SAVE_FILE="${mem_type}_result.json"

    echo "[$(date +%H:%M:%S)] Launching: memory-type=${mem_type} → ${LOG_FILE}"

    vita run \
        --domain personalization \
        --memory-type "$mem_type" \
        --save-to "${SAVE_SUBDIR}/${SAVE_FILE}" \
        --log-level DEBUG \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
done

echo ""
echo "All ${#PIDS[@]} jobs launched. Waiting for completion..."
echo "Monitor: tail -f ${RUN_DIR}/*.log"
echo ""

# ── Wait and collect results ──────────────────────────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    mem_type=${MEMORY_TYPES[$i]}

    if wait "$pid"; then
        echo "[$(date +%H:%M:%S)] DONE: ${mem_type} (PID ${pid}) - SUCCESS"
    else
        echo "[$(date +%H:%M:%S)] DONE: ${mem_type} (PID ${pid}) - FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Benchmark Complete - $(date)"
echo "  Results: ${RUN_DIR}/"
echo "  Succeeded: $((${#MEMORY_TYPES[@]} - FAILED)) / ${#MEMORY_TYPES[@]}"
echo "============================================================"

# Extract rewards from logs
echo ""
echo "  Memory Type       | Reward  | Duration"
echo "  ------------------|---------|----------"
for mem_type in "${MEMORY_TYPES[@]}"; do
    LOG_FILE="${RUN_DIR}/${mem_type}.log"
    if [ -f "$LOG_FILE" ]; then
        reward=$(grep -oP 'Average Reward: \K[0-9.]+' "$LOG_FILE" 2>/dev/null || echo "N/A")
        duration=$(grep -oP 'Total Duration: \K[0-9.]+min' "$LOG_FILE" 2>/dev/null || echo "N/A")
        printf "  %-18s | %-7s | %s\n" "$mem_type" "$reward" "$duration"
    else
        printf "  %-18s | %-7s | %s\n" "$mem_type" "NO LOG" "-"
    fi
done

echo ""
echo "Logs saved in: ${RUN_DIR}/"
ls -lh "$RUN_DIR/"
