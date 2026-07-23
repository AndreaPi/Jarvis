#!/bin/zsh

set -u
repo_dir="${0:A:h}"

if /usr/bin/python3 "$repo_dir/scripts/jarvis_local.py" stop; then
  exit 0
fi

print
read -k 1 "reply?Jarvis could not stop cleanly. Press any key to close this window."
print
exit 1
