#!/bin/sh
# 启动可选的 F5-TTS 组件。先在 README 里建好本目录的 .venv。
set -eu
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "没有找到 .venv/bin/python。先按 README 安装。" >&2
  exit 1
fi
exec ".venv/bin/python" server.py \
  --voices "$HOME/Library/Application Support/axo/models/voices" \
  --port 8800 --device auto
