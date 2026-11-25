#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
    echo "Usage: $0 EXPERIMENT_DIR PREVIOUS_ADAPTER DIALOGUE_VALIDATION_JSONL" >&2
    exit 2
fi

experiment=$1
previous_adapter=$2
dialogues=$3
python=.venv/bin/python

for path in "$experiment/fragment/config.json" "$experiment/lead/config.json" \
    "$previous_adapter/adapter_config.json" "$dialogues"; do
    test -s "$path"
done

for variant in fragment lead; do
    config="$experiment/$variant/config.json"
    echo "Running QLoRA variant: $variant"
    "$python" -m novaya.cli --config "$config" evaluate
    "$python" -m novaya.cli --config "$config" train --smoke
    "$python" -m novaya.cli --config "$config" train
done

"$python" -m novaya.editorial_compare --paired-experiment "$experiment" \
    --dialogues "$dialogues" --output "$experiment/blind-comparison" \
    --adapter "$previous_adapter" --adapter "$experiment/fragment/run/adapter" \
    --adapter "$experiment/lead/run/adapter"

comparison="$experiment/blind-comparison"
judge_model=runs/editorial/models/Qwen3.6-27B-Q3_K_M.gguf
judge_port=8084
/tmp/novaya-llama-build/bin/llama-server -m "$judge_model" -c 16384 -np 2 -ngl 99 \
    --host 127.0.0.1 --port "$judge_port" -fa on -ctk q8_0 -ctv q8_0 -b 256 -ub 128 \
    --fit off --reasoning-budget 0 > "$comparison/judge-server.log" 2>&1 &
judge_pid=$!
stop_judge() {
    kill -TERM "$judge_pid" 2>/dev/null || true
    wait "$judge_pid" 2>/dev/null || true
}
trap stop_judge EXIT
for attempt in {1..60}; do
    kill -0 "$judge_pid"
    if curl -fsS --max-time 1 "http://127.0.0.1:$judge_port/health" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
curl -fsS --max-time 3 "http://127.0.0.1:$judge_port/health"
"$python" -m novaya.editorial_blind_score --blind "$comparison/review.jsonl" \
    --output "$comparison/scores.jsonl" --port "$judge_port"
"$python" -m novaya.editorial_blind_summary --blind "$comparison/review.jsonl" \
    --scores "$comparison/scores.jsonl" --key "$comparison/key.json" \
    --output "$comparison/summary.json"
