#!/bin/zsh

set -u
repo_dir="${0:A:h}"

if /usr/bin/python3 "$repo_dir/scripts/jarvis_local.py" start; then
  exit 0
fi

print
read -k 1 "reply?Jarvis could not start. Press any key to close this window."
print
exit 1
