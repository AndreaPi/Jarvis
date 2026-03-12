#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Package Tier 1 Jarvis artifacts that should not be lost.

Usage:
  scripts/package-tier1-artifacts.sh [--output-dir DIR] [--archive-name NAME]

Outputs:
  - <archive>.tar.gz
  - <archive>.tar.gz.sha256
  - <archive>.manifest.txt
EOF
}

OUTPUT_DIR="output/artifacts"
ARCHIVE_NAME="jarvis-tier1-$(date -u +%Y%m%dT%H%M%SZ)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --archive-name)
      ARCHIVE_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_DIR"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MANIFEST_PATH="$TMP_DIR/${ARCHIVE_NAME}.manifest.txt"
ARCHIVE_PATH="$OUTPUT_ROOT/${ARCHIVE_NAME}.tar.gz"
CHECKSUM_PATH="${ARCHIVE_PATH}.sha256"

mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"

{
  if [[ -d assets ]]; then
    find assets -type f ! -name '*:Zone.Identifier' | sort
  fi
  if [[ -d backend/data/roi_dataset/images ]]; then
    find backend/data/roi_dataset/images -type f | sort
  fi
  if [[ -d backend/data/roi_dataset/labels ]]; then
    find backend/data/roi_dataset/labels -type f | sort
  fi
  if [[ -d backend/data/digit_dataset/manifests ]]; then
    find backend/data/digit_dataset/manifests -type f | sort
  fi
  if compgen -G 'backend/models/*.pt' > /dev/null; then
    printf '%s\n' backend/models/*.pt | sort
  fi
  if [[ -d backend/runs ]]; then
    find backend/runs -type f \
      \( -name 'args.yaml' -o -name 'results.csv' -o -name 'results.png' -o -name '*summary.json' \) | sort
  fi
} | awk 'NF' | sort -u > "$MANIFEST_PATH"

if [[ ! -s "$MANIFEST_PATH" ]]; then
  echo "No Tier 1 artifacts found to package." >&2
  exit 1
fi

tar -czf "$ARCHIVE_PATH" -T "$MANIFEST_PATH"
sha256sum "$ARCHIVE_PATH" > "$CHECKSUM_PATH"
cp "$MANIFEST_PATH" "${ARCHIVE_PATH%.tar.gz}.manifest.txt"

echo "Created archive: $ARCHIVE_PATH"
echo "Created checksum: $CHECKSUM_PATH"
echo "Created manifest: ${ARCHIVE_PATH%.tar.gz}.manifest.txt"
