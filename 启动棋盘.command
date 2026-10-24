#!/bin/bash
set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE"
if [ ! -x "$BASE/.venv/bin/python" ]; then
  echo '请先运行“安装依赖.command”。'
  read -r -p '按回车退出…'
  exit 1
fi
if ! "$BASE/.venv/bin/python" -m tools.board_platform; then
  echo '启动失败，请保留本窗口中的错误信息。'
  read -r -p '按回车退出…'
  exit 1
fi
