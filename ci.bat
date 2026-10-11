@echo off
REM 本地跑一遍 CI 的四条腿（对应 .github/workflows/ 里的 job 名）。
REM
REM 为什么是 .bat 套一层 .py：命令只有一份，写在 tools/ci_local.py 里。两边各抄
REM 一份的话，CI 改了个参数而本地没跟上，症状是"本地全绿、推上去红"——那正是
REM 这个入口存在的理由。
REM
REM 用它而不是推上去等 CI：本地一轮约一分钟，CI 排队加同步依赖要几分钟。
REM 想要更快（只改了文档或界面）：uv run python tools/ci_local.py --fast

setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    uv run python tools/ci_local.py %*
) else (
    if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" tools\ci_local.py %*
    ) else (
        echo 找不到 uv，也没有已建好的 .venv。
        echo 先跑一次：uv sync
        pause
        exit /b 1
    )
)

REM 退出码要传出去：1/2/3 是金标准的判定结果（见 tools/bench_recall.py），把它
REM 抹成 0 的话，"跑过 ci.bat"就成了一句没有内容的话。`endlocal & exit /b` 写在
REM 同一行——分开写的话中间那条命令会把 ERRORLEVEL 覆盖掉。
set _ci_code=%ERRORLEVEL%
REM 只在"连 python 都没起来"（9009 = 找不到命令）时 pause：那种情况下窗口一闪
REM 而过会看不到原因，而 1/2/3 是人会主动去看的判定结果。
if %_ci_code% geq 9009 pause
endlocal & exit /b %_ci_code%
