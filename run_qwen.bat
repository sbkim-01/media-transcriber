@echo off
chcp 65001 >nul
setlocal

rem ===================================================================
rem  Qwen3-ASR 전사 실행 (conda 환경 자동 활성화)
rem  사용법:  run_qwen.bat                     (더블클릭: config.toml 의 폴더 전사)
rem           run_qwen.bat --minutes 3         (config.toml 폴더 + 옵션)
rem           run_qwen.bat "D:\강의\*.mp4" [옵션...]
rem  예:      run_qwen.bat "D:\강의\*.mp4" "D:\강의2\*.m4a"
rem           run_qwen.bat "D:\강의\01.mp4" --minutes 3
rem  환경 이름이 다르면 실행 전에  set QWEN_ENV=qwen_asr2
rem  conda 설치 위치는 자동 탐색. 못 찾으면 실행 전에  set CONDA_ROOT=<conda 설치 폴더>
rem ===================================================================

rem 탐색기에서 실행(더블클릭, 파일 끌어놓기)하면 끝난 뒤 창을 닫지 않음
rem cmd 창에서 직접 친 경우는 창이 원래 남아 있으므로 대기하지 않음. 자동화에서는 set QWEN_NO_PAUSE=1
set "PAUSE_AT_END="
echo(%cmdcmdline%| "%SystemRoot%\System32\find.exe" /i "%~nx0" >nul && set "PAUSE_AT_END=1"
if "%QWEN_NO_PAUSE%"=="1" set "PAUSE_AT_END="
set "RC=1"

rem conda 위치 탐색: CONDA_ROOT 지정값 > CONDA_EXE(Anaconda Prompt 등) > 일반 설치 경로
if not defined CONDA_ROOT if defined CONDA_EXE for %%I in ("%CONDA_EXE%\..\..") do set "CONDA_ROOT=%%~fI"
if not defined CONDA_ROOT for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3") do if not defined CONDA_ROOT if exist "%%~D\Scripts\activate.bat" set "CONDA_ROOT=%%~D"
if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
    echo [오류] conda를 찾지 못했습니다. 실행 전에 지정하세요:  set CONDA_ROOT=^<conda 설치 폴더^>
    goto :end
)
if not defined QWEN_ENV set "QWEN_ENV=qwen_asr"

call "%CONDA_ROOT%\Scripts\activate.bat" || goto :end
call conda activate %QWEN_ENV% || (
    echo [오류] conda 환경 '%QWEN_ENV%' 활성화 실패. setup_qwen.bat 으로 설치하거나  set QWEN_ENV=^<환경 이름^>
    goto :end
)
set PYTHONIOENCODING=utf-8
python "%~dp0run_qwen_forced_align.py" %*
set "RC=%errorlevel%"

:end
echo.
if "%RC%"=="0" (echo [끝] 정상 완료) else if "%RC%"=="130" (echo [끝] 중단됨 - 다시 실행하면 이어서 처리) else (echo [끝] 실패한 파일이 있거나 오류 발생 - 위 메시지 확인, 종료 코드 %RC%)
echo 끝난 시각: %date% %time:~0,8%
if defined PAUSE_AT_END pause
exit /b %RC%
