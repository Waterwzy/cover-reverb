@echo off
rem ============================================================
rem  cover-reverb launcher (Windows)
rem
rem  ASCII-only on purpose: cmd.exe parses .bat files using the OEM
rem  codepage, so non-ASCII text here would corrupt the script.
rem
rem  Usage:
rem    scripts\launch.bat                  start the web UI
rem    scripts\launch.bat --port 8080      use another port
rem    scripts\launch.bat --check          environment check only
rem    scripts\launch.bat --selftest       run the pipeline on synthetic audio
rem    scripts\launch.bat --batch --roots "D:\my songs"
rem    scripts\launch.bat --setup          reinstall dependencies
rem ============================================================

setlocal EnableDelayedExpansion
set "REPO=%~dp0.."
pushd "%REPO%"

set "ARGS="
set "PY="

rem ---- optional overrides via environment ----
if defined COVER_REVERB_PYTHON set "PY=%COVER_REVERB_PYTHON%"
if defined COVER_REVERB_ENV   set "ENV_NAME=%COVER_REVERB_ENV%"
if not defined ENV_NAME set "ENV_NAME=cover-reverb"

rem ---- explicit interpreter wins ----
if defined PY if exist "%PY%" goto run

rem ---- reuse an existing conda env if present ----
if not defined CONDA_ROOT if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_ROOT=%USERPROFILE%\anaconda3"
if not defined CONDA_ROOT if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_ROOT=%USERPROFILE%\miniconda3"
if not defined CONDA_ROOT if exist "C:\ProgramData\anaconda3\Scripts\conda.exe" set "CONDA_ROOT=C:\ProgramData\anaconda3"
if not defined CONDA_ROOT if exist "C:\ProgramData\miniconda3\Scripts\conda.exe" set "CONDA_ROOT=C:\ProgramData\miniconda3"
if not defined CONDA_ROOT (
    for /f "delims=" %%i in ('where conda 2^>nul') do (
        if not defined CONDA_ROOT set "CONDA_ROOT=%%~dpi.."
    )
)
if defined CONDA_ROOT (
    if "%CONDA_ROOT:~-1%"=="\" set "CONDA_ROOT=%CONDA_ROOT:~0,-1%"
    if exist "%CONDA_ROOT%\envs\%ENV_NAME%\python.exe" (
        set "PY=%CONDA_ROOT%\envs\%ENV_NAME%\python.exe"
        goto run
    )
)

rem ---- reuse an existing venv if present ----
if exist "%REPO%\.venv\Scripts\python.exe" (
    set "PY=%REPO%\.venv\Scripts\python.exe"
    goto run
)

rem ---- nothing usable yet: bootstrap an environment ----
echo.
echo   No environment found. Running bootstrap_env.py ...
echo.
where python >nul 2>nul
if errorlevel 1 (
    echo   [FAIL] python is not on PATH. Install Python 3.10+ or Anaconda first.
    popd
    pause
    exit /b 1
)
python "%REPO%\bootstrap_env.py" --with-dev
if errorlevel 1 (
    echo   [FAIL] environment setup failed.
    popd
    pause
    exit /b 1
)

if exist "%REPO%\.venv\Scripts\python.exe" set "PY=%REPO%\.venv\Scripts\python.exe"
if defined CONDA_ROOT if exist "%CONDA_ROOT%\envs\%ENV_NAME%\python.exe" set "PY=%CONDA_ROOT%\envs\%ENV_NAME%\python.exe"
if not defined PY (
    echo   [FAIL] environment was created but the interpreter was not found.
    popd
    pause
    exit /b 1
)

:run
echo.
echo   python: %PY%
echo.
"%PY%" -u "%REPO%\launch.py" %ARGS%
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (
    echo.
    echo   exit code %RC%
    pause
)
endlocal
exit /b %RC%
