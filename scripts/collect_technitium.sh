#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

TECHNITIUM_URL="${TECHNITIUM_URL:-http://127.0.0.1:5380}"

if [[ -z "${TECHNITIUM_TOKEN:-}" ]]; then
  read -r -s -p "Technitium API token: " TECHNITIUM_TOKEN
  printf '\n'
fi
export TECHNITIUM_TOKEN TECHNITIUM_URL

COLLECTOR="$PROJECT_DIR/.venv/bin/dns-detect"
if [[ ! -x "$COLLECTOR" ]]; then
  echo "Can't find $COLLECTOR. Check that the project's virtual environment is installed."
  exit 1
fi

ARGS=(
  collect-technitium
  --app-name "Query Logs (Sqlite)"
  --class-path "QueryLogsSqlite.App"
  --out "$PROJECT_DIR/data/raw/technitium.csv"
)

# Optional: provide both start and end timestamps when running the script.
if [[ $# -eq 2 ]]; then
  ARGS+=(--start "$1" --end "$2")
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [START_TIMESTAMP END_TIMESTAMP]"
  exit 1
fi

"$COLLECTOR" "${ARGS[@]}"
