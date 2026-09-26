@echo off
rem ===========================================================================
rem  run_windows.bat - run the whole project on Windows
rem
rem  Settings: run_config.yaml (repository root). Every option is passed on to
rem  run_all.py, for instance:
rem      run_windows.bat                          run every step switched on
rem      run_windows.bat --dry-run                print the commands, run nothing
rem      run_windows.bat --only pose analysis     run these steps only
rem      run_windows.bat --config my_run.yaml     another settings file
rem  Run it from a terminal (cmd or PowerShell: .\run_windows.bat). A double
rem  click works too, but the window closes at the end: the full output is kept
rem  in results\run_logs\.
rem ===========================================================================
setlocal
cd /d "%~dp0"

rem UTF-8 console and Python input/output: the logs hold non-ASCII characters.
chcp 65001 >nul
set "PYTHONUTF8=1"

rem The virtual environment of the README, else the Python of the PATH.
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

"%PYTHON%" run_all.py --config run_config.yaml %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo The run stopped with exit code %EXITCODE%: see the messages above and results\run_logs\.
)

endlocal & exit /b %EXITCODE%
