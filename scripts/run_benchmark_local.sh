#!/usr/bin/env bash
# Run the benchmark on the host with the answer API served in-process, so no
# admin login/TOTP is needed. The app container's environment is reused (it
# runs with network_mode: host, so DB/Qdrant/LLM addresses are the same) and
# container paths under /app are mapped to this repository. Secrets stay in
# this process's environment and are never written to disk.
#
#   scripts/run_benchmark_local.sh --only-stt 1,5,156 --no-resume \
#       --output-json reports/X.json --output-xlsx reports/X.xlsx
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER="${LAWCHAT_APP_CONTAINER:-lawchat-app-1}"

while IFS= read -r line; do
  case "$line" in
    PATH=*|HOME=*|HOSTNAME=*|PYTHONPATH=*|VIRTUAL_ENV=*|UV_*|LANG=*|GPG_KEY=*|PYTHON_*) continue ;;
  esac
  name="${line%%=*}"
  value="${line#*=}"
  export "$name=${value//\/app\//$ROOT/}"
done < <(docker exec "$CONTAINER" env)

cd "$ROOT"
exec "$ROOT/.venv/bin/python" scripts/benchmark_200_questions.py \
  --file "200_Cau_Hoi_Test_Chatbot_RAG_Phap_Luat (1).xlsx" --in-process "$@"
