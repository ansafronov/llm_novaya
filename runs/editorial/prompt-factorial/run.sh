#!/usr/bin/env bash
set -euo pipefail
d=runs/editorial/prompt-factorial
export PYTHONPATH=.
server_pid=
stop() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" || true
        server_pid=
    fi
}
trap stop EXIT
start() {
    /tmp/novaya-llama-build/bin/llama-server -m "runs/editorial/models/$1" \
        -c 16384 -np 2 -ngl 99 --host 127.0.0.1 --port "$2" -fa on \
        -ctk q8_0 -ctv q8_0 -b 256 -ub 128 --fit off --reasoning-budget 0 > "$d/$3.log" 2>&1 &
    server_pid=$!
    for attempt in {1..60}; do
        kill -0 "$server_pid"
        if curl -fsS --max-time 1 "http://127.0.0.1:$2/health" >/dev/null 2>&1; then return; fi
        sleep 2
    done
    return 1
}
if [[ ! -s "$d/rewritten.jsonl" ]]; then
    start Qwen3.8-27B-UD-IQ4_XS.gguf 8083 rewrite-server
    .venv/bin/python "$d/pilot.py" rewrite
    stop
fi
.venv/bin/python "$d/pilot.py" generate
start Qwen3.6-27B-Q3_K_M.gguf 8084 judge-server
.venv/bin/python "$d/pilot.py" judge
