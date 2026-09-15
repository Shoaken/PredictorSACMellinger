@echo off
REM Pass-through to scripts\paper_runs.py
REM Requires: conda activate drones
REM   scripts\run_paper.bat --one --seed 1 --lam 0.25
REM   scripts\run_paper.bat --grid comparison --list
setlocal EnableExtensions
cd /d "%~dp0\.."

if not defined CONDA_PREFIX (
  echo ERROR: conda is not active in this terminal ^(CONDA_PREFIX is empty^).
  echo This terminal does not have the drones env.
  echo.
  echo Do this:
  echo   1. Open Anaconda Prompt
  echo   2. conda activate drones
  echo   3. cd to this repository
  echo   4. scripts\run_paper.bat --one --seed 1
  echo.
  echo Or call drones Python directly:
  echo   python scripts\paper_runs.py --one --seed 1
  echo ^(that python must be the drones interpreter, not Windows Store / Python312^)
  exit /b 1
)

if exist "%CONDA_PREFIX%\python.exe" set "PYTHON=%CONDA_PREFIX%\python.exe"
if not defined PYTHON if exist "%CONDA_PREFIX%\python" set "PYTHON=%CONDA_PREFIX%\python"
if not defined PYTHON (
  echo ERROR: No python inside CONDA_PREFIX=%CONDA_PREFIX%
  exit /b 1
)

echo Using PYTHON=%PYTHON%
"%PYTHON%" scripts\paper_runs.py %*
exit /b %ERRORLEVEL%
