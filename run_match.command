#!/bin/sh
set -eu
BASE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ ! -x "$BASE/.venv/bin/python" ]; then
  echo "请先运行 安装依赖.command。" >&2
  exit 1
fi
cd "$BASE"
exec "$BASE/.venv/bin/python" "$BASE/tools/macos_platform.py" "$@"
