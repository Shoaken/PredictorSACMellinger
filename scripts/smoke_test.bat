@echo off
REM Smoke-test that simulation training starts. Not a paper-length run.
REM From the repository root, after: conda activate drones
REM   scripts\smoke_test.bat
REM   scripts\smoke_test.bat --all
REM Paper training: scripts\run_paper.bat --grid comparison --list
setlocal EnableExtensions
cd /d "%~dp0\.."

if /I "%~1"=="-h" goto :help
if /I "%~1"=="--help" goto :help

if not exist "env\gym_pybullet_drones" (
  echo Missing env\gym_pybullet_drones. Run: git submodule update --init --recursive
  exit /b 1
)

if not defined CONDA_PREFIX (
  echo ERROR: conda is not active ^(CONDA_PREFIX is empty^). Run: conda activate drones
  exit /b 1
)
if exist "%CONDA_PREFIX%\python.exe" set "PYTHON=%CONDA_PREFIX%\python.exe"
if not defined PYTHON set "PYTHON=python"
if "%SMOKE_STEPS%"=="" set "SMOKE_STEPS=8"
if "%SMOKE_IDX%"=="" set "SMOKE_IDX=9999"

set "RUN_ALL=0"
if /I "%~1"=="--all" set "RUN_ALL=1"

echo Using PYTHON=%PYTHON%
"%PYTHON%" -c "import torch; from train.utils.env import TunableRewardAviary; print('imports ok;', 'cuda' if torch.cuda.is_available() else 'cpu'); print('obs', TunableRewardAviary(gui=False, record=False).observation_space.shape)"
if errorlevel 1 exit /b 1

call :run_one sac_predictor_IB tunable-reward
if errorlevel 1 exit /b 1
if "%RUN_ALL%"=="1" (
  call :run_one spederv3 tunable-reward
  if errorlevel 1 exit /b 1
  call :run_one domain-randomization domain-randomization
  if errorlevel 1 exit /b 1
)

echo.
echo Smoke test passed.
exit /b 0

:run_one
echo.
echo === smoke: --alg %~1 --env %~2 (%SMOKE_STEPS% env steps) ===
"%PYTHON%" main_pyb_train.py --alg "%~1" --env "%~2" --seed 1 --max_timesteps %SMOKE_STEPS% --eval_freq 1000000 --exp_idx %SMOKE_IDX% --wandb_offline
exit /b %ERRORLEVEL%

:help
echo Activate drones, then from the repository root:
echo   scripts\smoke_test.bat
echo   scripts\smoke_test.bat --all
exit /b 0
