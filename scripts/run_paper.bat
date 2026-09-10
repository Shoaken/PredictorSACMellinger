@echo off
REM Pass-through to scripts\paper_runs.py
REM   scripts\run_paper.bat --one --seed 1 --lam 0.25
REM   scripts\run_paper.bat --grid comparison --list
REM   scripts\run_paper.bat --grid paper --wandb-offline
setlocal EnableExtensions
cd /d "%~dp0\.."
if "%PYTHON%"=="" set "PYTHON=python"
"%PYTHON%" scripts\paper_runs.py %*
exit /b %ERRORLEVEL%
