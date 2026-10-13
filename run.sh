#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 build.py
printf '教程地址：http://127.0.0.1:%s\n' "${PORT:-8650}"
exec python3 -m http.server "${PORT:-8650}" --bind 127.0.0.1 --directory webapp
