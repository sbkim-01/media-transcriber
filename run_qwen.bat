@echo off
chcp 65001 >nul
setlocal

rem ===================================================================
rem  Qwen3-ASR 전사 실행 (conda 환경 자동 활성화)
rem  사용법:  run_qwen.bat "D:\강의\*.mp4" [옵션...]
rem  예:      run_qwen.bat "D:\강의\*.mp4" "D:\강의2\*.m4a"
rem           run_qwen.bat "D:\강의\01.mp4" --minutes 3
rem  환경 이름이 다르면 실행 전에  set QWEN_ENV=qwen_asr2
rem  conda 설치 위치는 자동 탐색. 못 찾으면 실행 전에  set CONDA_ROOT=<conda 설치 폴더>
rem ===================================================================

rem conda 위치 탐색: CONDA_ROOT 지정값 > CONDA_EXE(Anaconda Prompt 등) > 일반 설치 경로
if not defined CONDA_ROOT if defined CONDA_EXE for %%I in ("%CONDA_EXE%\..\..") do set "CONDA_ROOT=%%~fI"
if not defined CONDA_ROOT for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3") do if not defined CONDA_ROOT if exist "%%~D\Scripts\activate.bat" set "CONDA_ROOT=%%~D"
if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
    echo [오류] conda를 찾지 못했습니다. 실행 전에 지정하세요:  set CONDA_ROOT=^<conda 설치 폴더^>
    exit /b 1
)
if not defined QWEN_ENV set "QWEN_ENV=qwen_asr"

if "%~1"=="" (
    echo 사용법: run_qwen.bat "파일 또는 와일드카드" [옵션...]
    echo 옵션 목록: run_qwen.bat --help
    exit /b 1
)

call "%CONDA_ROOT%\Scripts\activate.bat" || exit /b 1
call conda activate %QWEN_ENV% || (
    echo [오류] conda 환경 '%QWEN_ENV%' 활성화 실패. setup_qwen.bat 으로 먼저 설치하세요.
    exit /b 1
)
set PYTHONIOENCODING=utf-8
python "%~dp0run_qwen_forced_align.py" %*
exit /b %errorlevel%
