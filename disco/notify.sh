#!/bin/bash
# DISCO-TAB notifier: notify.sh "<subject>" "<body>"
cd "$(dirname "$0")"
python3 notify.py --env "$(dirname "$0")/notify.env" "$1" "${2:-}" 2>&1 | tail -3
