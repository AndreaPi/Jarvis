#!/usr/bin/env bash
set -euo pipefail

readonly SAMPLE_INTERVAL_MS=30000
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DEFAULT_LOG_DIR="$REPO_ROOT/backend/runs/thermal"

usage() {
  cat <<'EOF'
Sample macOS thermal pressure every 30 seconds and log non-nominal events.

Usage:
  scripts/monitor-thermal.sh [log-file]

The default log is:
  backend/runs/thermal/thermal-events-YYYYMMDD-HHMMSS.log

All samples are shown in the terminal. Only complete samples whose current
pressure level is not Nominal are appended to the log. Pass a path to append
to a different log. Press Ctrl-C to stop monitoring.
EOF
}

filter_thermal_events() {
  local log_file="$1"

  awk -v log_file="$log_file" '
    function flush_sample() {
      if (sample_started && non_nominal) {
        printf "%s", sample >> log_file
        close(log_file)
      }
      sample = ""
      sample_started = 0
      non_nominal = 0
    }

    {
      print
      fflush()

      if ($0 ~ /^\*\*\* Sampled system activity/) {
        flush_sample()
        sample_started = 1
      }

      if (sample_started) {
        sample = sample $0 ORS
        if (index($0, "Current pressure level:") > 0) {
          pressure = $0
          sub(/^.*Current pressure level:[[:space:]]*/, "", pressure)
          sub(/[[:space:]]*$/, "", pressure)
          if (pressure != "Nominal") {
            non_nominal = 1
          }
        }
      }
    }

    END {
      flush_sample()
    }
  '
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    return 0
  fi

  if (( $# > 1 )); then
    usage >&2
    return 2
  fi

  if [[ "$(uname -s)" != "Darwin" || ! -x /usr/bin/powermetrics ]]; then
    echo "This monitor requires macOS and /usr/bin/powermetrics." >&2
    return 1
  fi

  local log_file
  local status
  local -a pipeline_status
  if (( $# == 1 )); then
    log_file="$1"
  else
    log_file="$DEFAULT_LOG_DIR/thermal-events-$(date '+%Y%m%d-%H%M%S').log"
  fi

  if [[ "$log_file" != /* ]]; then
    log_file="$PWD/$log_file"
  fi

  mkdir -p "$(dirname "$log_file")"
  touch "$log_file"

  echo "Authenticating once so powermetrics can read thermal events..."
  sudo -v

  printf '[%s] Thermal monitoring started; interval=30 seconds\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'Saving non-nominal sample blocks to %s\n' "$log_file"

  set +e
  sudo /usr/bin/powermetrics \
    --samplers thermal \
    --sample-rate "$SAMPLE_INTERVAL_MS" \
    --sample-count -1 \
    --buffer-size 1 \
    2>&1 | filter_thermal_events "$log_file"
  pipeline_status=("${PIPESTATUS[@]}")
  status=${pipeline_status[0]}
  if (( pipeline_status[1] != 0 )); then
    status=${pipeline_status[1]}
  fi
  set -e

  printf '[%s] Thermal monitoring stopped; exit_status=%d\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$status"

  return "$status"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
