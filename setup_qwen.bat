@echo off
chcp 65001 >nul
setlocal

rem ===================================================================
rem  Qwen3-ASR 전사 환경 자동 설치 (conda)
rem  사용법:  setup_qwen.bat [환경이름]      (기본 환경이름: qwen_asr)
rem  - 같은 이름의 conda 환경이 이미 있으면 아무것도 하지 않고 종료함
rem  - conda 설치 위치는 자동 탐색. 못 찾으면 실행 전에 set CONDA_ROOT=<conda 설치 폴더>
rem ===================================================================

rem conda 위치 탐색: CONDA_ROOT 지정값 > CONDA_EXE(Anaconda Prompt 등) > 일반 설치 경로
if not defined CONDA_ROOT if defined CONDA_EXE for %%I in ("%CONDA_EXE%\..\..") do set "CONDA_ROOT=%%~fI"
if not defined CONDA_ROOT for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3") do if not defined CONDA_ROOT if exist "%%~D\Scripts\activate.bat" set "CONDA_ROOT=%%~D"

set "ENV_NAME=qwen_asr"
if not "%~1"=="" set "ENV_NAME=%~1"

if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
    echo [오류] conda를 찾지 못했습니다.
    echo        Anaconda/Miniconda 설치 폴더를 지정한 뒤 다시 실행하세요:  set CONDA_ROOT=^<설치 폴더^>
    exit /b 1
)

call "%CONDA_ROOT%\Scripts\activate.bat"

call conda env list | findstr /R /C:"^%ENV_NAME% " >nul
if not errorlevel 1 (
    echo [중단] conda 환경 '%ENV_NAME%' 이 이미 있습니다. 기존 환경 보호를 위해 설치하지 않습니다.
    echo        새로 만들려면 다른 이름을 지정하세요:  setup_qwen.bat qwen_asr2
    exit /b 1
)

echo [1/4] conda 환경 생성: %ENV_NAME% (Python 3.12)
call conda create -y -n %ENV_NAME% python=3.12 || goto :fail
call conda activate %ENV_NAME% || goto :fail

echo [2/4] pip 업데이트
python -m pip install -U pip || goto :fail

echo [3/4] torch CUDA 12.8 빌드 설치 (RTX 50 시리즈는 cu128 필수)
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 || goto :fail

echo [4/4] Qwen3-ASR 의존 패키지 설치
pip install -r "%~dp0requirements-qwen.txt" || goto :fail

echo.
echo GPU 인식 확인 (True 가 나와야 정상):
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
echo.
echo [완료] 실행 예:  run_qwen.bat "D:\강의\*.mp4" --minutes 3
echo        모델(약 6GB)은 첫 실행 때 자동으로 다운로드됩니다.
exit /b 0

:fail
echo.
echo [실패] 위 오류 메시지를 확인하세요. 문제 해결은 README.md 참고.
exit /b 1
