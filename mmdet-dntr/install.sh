#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --runtime        Recreate only the runtime environment (.venv)
  --test           Recreate only the test environment (testvenv)
  --all            Recreate both environments (default)
  --python PATH    Python interpreter to use (default: python3)
  --force          Remove existing environments before recreating them
  -h, --help       Show this message
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TARGET="all"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime) TARGET="runtime" ;;
    --test) TARGET="test" ;;
    --all) TARGET="all" ;;
    --python) shift; PYTHON_BIN="$1" ;;
    --force) FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

create_env() {
  local name="$1"
  local req_file="$2"
  local env_path="$ROOT_DIR/$name"

  if [[ -d "$env_path" ]]; then
    if [[ "$FORCE" -ne 1 ]]; then
      echo "[skip] $name exists. Use --force to recreate."
      return
    fi
    echo "[info] Removing existing $name..."
    rm -rf "$env_path"
  fi

  echo "[info] Creating $name using $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$env_path"

  # Activate in subshell to avoid leaking environment
  (
    set -euo pipefail
    cd "$ROOT_DIR"
    source "$env_path/bin/activate"
    python -m pip install --upgrade pip
    pip install --no-cache-dir -r "$req_file"
    deactivate
  )

  echo "[done] $name ready."
}

case "$TARGET" in
  runtime)
    create_env ".venv" "requirements/runtime-lock.txt"
    ;;
  test)
    create_env "testvenv" "requirements/test-lock.txt"
    ;;
  all)
    create_env ".venv" "requirements/runtime-lock.txt"
    create_env "testvenv" "requirements/test-lock.txt"
    ;;
  *)
    echo "Internal error: unknown target '$TARGET'" >&2
    exit 1
    ;;
esac

echo
echo "Environments created."
echo "  Runtime: source $ROOT_DIR/.venv/bin/activate"
echo "  Test   : source $ROOT_DIR/testvenv/bin/activate"
