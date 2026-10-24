#!/bin/bash
set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE"
pause_exit() { if [ -t 0 ]; then read -r -p '按回车退出…'; fi; }
trap 'echo "安装失败，请保留上面的报错。请参阅《Mac使用教程.md》。"; pause_exit' ERR
PY=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys, ensurepip; sys.exit(not ((3,11) <= sys.version_info[:2] <= (3,13)))' 2>/dev/null; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ]; then
  echo '请先从 https://www.python.org/downloads/macos/ 安装 Python 3.13 的 macOS 64-bit universal2 installer。'
  echo '安装后重新打开终端，再运行本脚本。支持 Python 3.11–3.13。'
  pause_exit
  exit 1
fi
if [ -d "$BASE/.venv" ]; then
  if ! "$BASE/.venv/bin/python" -c 'import sys, pip; sys.exit(not ((3,11) <= sys.version_info[:2] <= (3,13)))' 2>/dev/null; then
    echo '现有 .venv 已损坏或不兼容。请仅删除本平台目录里的 .venv 文件夹，再重新安装。'
    pause_exit
    exit 1
  fi
else
  "$PY" -m venv "$BASE/.venv"
fi
"$BASE/.venv/bin/python" -m pip install --upgrade pip
"$BASE/.venv/bin/python" -m pip install -r "$BASE/requirements-macos.txt"
chmod +x "$BASE/NoGoEngine" "$BASE/CheckEngine.command" "$BASE/run_match.command" "$BASE/启动棋盘.command"
"$BASE/.venv/bin/python" -m tools.portable_engine --self-test
echo '安装及模型自检完成。下一步：双击“启动棋盘.command”。'
pause_exit
