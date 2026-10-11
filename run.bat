@echo off
REM 双击启动。调试入口——它保留控制台窗口，因为后端出问题时错误详情会打印到
REM stderr，而"没有控制台"会让一个可诊断的问题变成一个沉默的闪退。
REM
REM 想要无控制台运行，把最后一行换成：
REM     start "" "%~dp0.venv\Scripts\pythonw.exe" -m biosnn_memterm.ui.app
REM 代价是再也看不到任何错误输出。开发期不建议。

setlocal
cd /d "%~dp0"

REM uv run 会自动按 pyproject.toml 同步依赖；没有 uv 时回退到已建好的 venv。
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    uv run python -m biosnn_memterm.ui.app
) else (
    if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" -m biosnn_memterm.ui.app
    ) else (
        echo 找不到 uv，也没有已建好的 .venv。
        echo 先跑一次：uv sync --extra app --extra llm
        pause
        exit /b 1
    )
)

if %ERRORLEVEL% neq 0 pause
endlocal
