@echo off
chcp 65001 >nul
REM 启动可选的 F5-TTS 组件。先在 README 里建好 components\f5-tts\.venv。
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 没有找到 .venv\Scripts\python.exe。先按 README 安装。
  pause
  exit /b 1
)
if "%LOCALAPPDATA%"=="" (
  echo 没有 LOCALAPPDATA，无法定位默认音色目录。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" server.py --voices "%LOCALAPPDATA%\axo\models\voices" --port 8800 --device cuda
pause
