#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run `dvc push` through the repo-local virtualenv after verifying that the default
remote is configured and is not a plain local filesystem path.

Usage:
  scripts/dvc-push-safe.sh [dvc push args...]

Behavior:
  - activates `backend/.venv` so the repo-local `dvc` is used
  - reads the configured default DVC remote
  - refuses to push if the default remote is missing
  - refuses to push if the default remote is a local path instead of a cloud/object-store URL
  - then runs `dvc push`

Examples:
  scripts/dvc-push-safe.sh
  scripts/dvc-push-safe.sh backend/models/digit_classifier.pt.dvc
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ACTIVATE="$REPO_ROOT/backend/.venv/bin/activate"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Missing backend virtualenv: $VENV_ACTIVATE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc is not available in backend/.venv" >&2
  exit 1
fi

cd "$REPO_ROOT"

DEFAULT_REMOTE="$(dvc config core.remote 2>/dev/null || true)"
if [[ -z "$DEFAULT_REMOTE" ]]; then
  echo "No default DVC remote is configured." >&2
  echo "Configure one first, then retry." >&2
  exit 1
fi

REMOTE_URL="$(dvc remote list 2>/dev/null | awk -v name="$DEFAULT_REMOTE" '$1 == name {print $2; exit}')"
if [[ -z "$REMOTE_URL" ]]; then
  echo "Could not resolve DVC remote '$DEFAULT_REMOTE'." >&2
  exit 1
fi

if [[ "$REMOTE_URL" = /* || "$REMOTE_URL" = .* ]]; then
  echo "Refusing to push: DVC remote '$DEFAULT_REMOTE' is a local path: $REMOTE_URL" >&2
  echo "This repo now assumes cloud-only DVC storage. Configure a non-local remote such as Backblaze B2." >&2
  exit 1
fi

if [[ ! "$REMOTE_URL" =~ ^[A-Za-z][A-Za-z0-9+.-]*:// ]]; then
  echo "Refusing to push: DVC remote '$DEFAULT_REMOTE' is not a recognized URL: $REMOTE_URL" >&2
  echo "Configure a non-local remote such as Backblaze B2 and retry." >&2
  exit 1
fi

echo "Using DVC remote '$DEFAULT_REMOTE': $REMOTE_URL"

exec dvc push "$@"
