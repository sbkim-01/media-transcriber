# media-transcriber — 영상/음성 → 텍스트 변환기

강의 녹화 등 영상·음성 파일(mp4, m4a 등)을 **타임스탬프가 붙은 문단 TXT**로 일괄 변환하는 개인 도구입니다.
스크립트와 배치파일은 모두 자기 폴더 기준으로 동작하므로 저장소를 어느 위치에 받아도 됩니다.
아래 설명의 `<저장소>`는 clone한 폴더를 뜻합니다.

| 엔진 | 스크립트 | 상태 | 모델 |
|---|---|---|---|
| **Qwen3-ASR** | `run_qwen_forced_align.py` | ✅ 현재 사용 | `Qwen/Qwen3-ASR-1.7B` + `Qwen/Qwen3-ForcedAligner-0.6B` |
| Whisper | `run_whisper.py` | 보관 (2026-10 이후 미사용) | faster-whisper `large-v3` |

> 새 PC에서 처음 세팅한다면 **[빠른 시작](#빠른-시작-qwen)** 부터 따라 하세요.

---

## 목차

1. [필요 사양](#필요-사양)
2. [폴더 구조](#폴더-구조)
3. [빠른 시작 (Qwen)](#빠른-시작-qwen)
4. [수동 설치 (Qwen)](#수동-설치-qwen)
5. [사용법](#사용법)
6. [출력 형식과 동작 방식](#출력-형식과-동작-방식)
7. [Whisper 환경 (보관용)](#whisper-환경-보관용)
8. [문제 해결](#문제-해결)
9. [평가 자료 (_eval)](#평가-자료-_eval)
10. [git에 없는 것 (재설치 대상)](#git에-없는-것-재설치-대상)

---

## 필요 사양

| 항목 | 내용 |
|---|---|
| OS | Windows 10/11 (스크립트·배치파일이 Windows 기준) |
| GPU | **NVIDIA 필수** (CPU 실행 불가). bfloat16 사용 → RTX 30 시리즈 이상 권장 |
| VRAM | 8GB 이상 권장. 기본 배치 8에서 최대 약 12.8GB 사용 (16GB 기준) |
| 드라이버 | 최신 NVIDIA 드라이버 (CUDA 12.8 지원 필요) |
| 디스크 | 약 15GB (Qwen 모델 6.1GB + conda 환경/torch) |
| 기타 | Anaconda 또는 Miniconda, FFmpeg |

작성자 환경: RTX 5060 Ti 16GB, 드라이버 610.88, Anaconda (모든 사용자용 설치)

> **RTX 50 시리즈(Blackwell)는 반드시 torch `cu128` 빌드**를 써야 합니다. cu121/cu124 빌드는 GPU를 인식하지 못합니다.

---

## 폴더 구조

```
<저장소>\
├─ run_qwen_forced_align.py   # Qwen3-ASR 전사기 (메인)
├─ run_qwen.bat               # conda 환경 활성화 + 전사 실행 단축
├─ setup_qwen.bat             # Qwen conda 환경 자동 설치
├─ requirements-qwen.txt      # Qwen 직접 의존 패키지 (torch 제외)
├─ run_whisper.py             # Whisper 전사기 (보관)
├─ requirements-whisper.txt   # Whisper 직접 의존 패키지 (torch 제외)
├─ env\
│   ├─ qwen_asr_freeze.txt    # 동작 확인된 Qwen 환경 전체 스냅샷
│   └─ whisper_env_freeze.txt # 동작 확인된 Whisper 환경 전체 스냅샷
├─ _eval\                     # v2 개선 시 평가 스크립트/요약
│
│  ↓ 아래는 git에 없음 (직접 준비)
├─ ffmpeg-<버전>_build\bin\ffmpeg.exe   # 스크립트가 자동 탐색
├─ whisper_env\               # Whisper용 venv
└─ .progress\                 # 실행 중 체크포인트 (완료 시 자동 삭제)
```

---

## 빠른 시작 (Qwen)

### 1) 저장소 받기

원하는 위치에서:

```bat
git clone https://github.com/sbkim-01/media-transcriber.git
cd media-transcriber
```

> private 레포라 clone 전에 GitHub 로그인이 필요합니다 (`gh auth login` 또는 브라우저 인증 창).
> 폴더 이름을 바꾸고 싶으면 clone 명령 끝에 이름을 붙이면 됩니다 (예: `... media-transcriber.git ai_sound_txt`).

### 2) Anaconda 설치

<https://www.anaconda.com/download> (또는 Miniconda).

배치파일은 conda 위치를 아래 순서로 자동으로 찾습니다.
1. `CONDA_ROOT` 환경변수
2. Anaconda Prompt에서 실행한 경우 그 conda
3. 일반 설치 경로: 사용자 폴더, `ProgramData`, `AppData\Local` 아래의 `anaconda3` / `miniconda3`

못 찾는다는 오류가 나면 실행 전에 직접 지정합니다.

```bat
set CONDA_ROOT=<conda 설치 폴더>
```

### 3) FFmpeg 넣기 (설치 없음, 압축만 해제)

1. <https://www.gyan.dev/ffmpeg/builds/> → `ffmpeg-release-full.zip`(또는 essentials) 다운로드
2. 압축 푼 폴더 **통째로** 이 저장소 폴더에 넣기

```
<저장소>\ffmpeg-8.0-full_build\bin\ffmpeg.exe   ← 이 구조면 자동 인식
```

- 스크립트는 `ffmpeg-*/bin/ffmpeg.exe` → `ffmpeg.exe`(같은 폴더) → PATH 순서로 찾습니다.
- 기존에 쓰던 빌드: `ffmpeg-2026-03-26-git-fd9f1e9c52-full_build`

### 4) 환경 자동 설치

**cmd**(PowerShell 말고)에서 실행합니다.

```bat
cd /d <저장소>
setup_qwen.bat
```

- conda 환경 `qwen_asr`(Python 3.12) 생성 → torch cu128 → `requirements-qwen.txt` 순서로 설치합니다.
- 마지막에 `True NVIDIA GeForce ...` 가 나오면 성공입니다.
- 같은 이름의 환경이 이미 있으면 **아무것도 건드리지 않고 중단**합니다. 다른 이름으로 만들려면 `setup_qwen.bat qwen_asr2` (이 경우 실행 전 `set QWEN_ENV=qwen_asr2`).

### 5) 시험 실행 (앞 3분만)

```bat
run_qwen.bat "D:\강의\01.mp4" --minutes 3
```

- 첫 실행 때 모델 약 6.1GB가 Hugging Face에서 자동 다운로드됩니다 (`%USERPROFILE%\.cache\huggingface\hub`).
- 결과: `D:\강의\01.test-3min.txt`

---

## 수동 설치 (Qwen)

`setup_qwen.bat`이 하는 일을 직접 하려면, 시작 메뉴의 **Anaconda Prompt**를 열고 저장소 폴더에서:

```bat
conda create -y -n qwen_asr python=3.12
conda activate qwen_asr

python -m pip install -U pip
rem torch 를 먼저! 순서가 바뀌면 CPU 버전 torch 가 깔림
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-qwen.txt

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

conda 대신 venv를 써도 됩니다 (Python 3.12 설치 필요).

```bat
py -3.12 -m venv qwen_env
qwen_env\Scripts\activate
rem 이후 pip 명령은 위와 동일
```

모델을 미리 받아 두려면 (환경 활성화 상태):

```bat
hf download Qwen/Qwen3-ASR-1.7B
hf download Qwen/Qwen3-ForcedAligner-0.6B
```

라이브러리 버전 문제로 동작이 이상하면 `env\qwen_asr_freeze.txt`의 버전과 비교하세요. 이 조합이 실제로 동작 확인된 상태입니다.

---

## 사용법

### 배치파일로 (권장)

```bat
run_qwen.bat "D:\강의\*.mp4"
run_qwen.bat "D:\강의A\*.mp4" "D:\강의B\*.m4a"
run_qwen.bat "D:\강의" --recursive
```

### 직접 실행

Anaconda Prompt에서:

```bat
conda activate qwen_asr
python <저장소>\run_qwen_forced_align.py "D:\강의\*.mp4"
```

- 입력: 파일, 와일드카드, 폴더 (여러 개 가능)
- 지원 형식: mp4, m4a, mp3, wav, flac, aac, mkv, mov, webm, ogg, opus, wma, avi, m4b
- **이미 결과 `.txt`가 있는 파일은 자동으로 건너뜁니다.** 여러 강의 폴더를 한 번에 넣고 주기적으로 돌려도 새 파일만 처리됩니다.

### 옵션

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--minutes N` | 0 (전체) | 앞 N분만 시험 변환. 결과는 `파일명.test-Nmin.txt` |
| `--overwrite` | 꺼짐 | 기존 `.txt`가 있어도 다시 변환 |
| `--recursive` | 꺼짐 | 폴더 입력 시 하위 폴더까지 |
| `--batch-size N` | 8 | GPU 동시 처리 청크 수. VRAM 8GB면 2~4. OOM 시 자동으로 절반씩 줄임 |
| `--context "용어, ..."` | 없음 | 전문용어 힌트 (예: `"귀무가설, 회귀모형, 잔차"`) |
| `--context-file 파일` | 없음 | 용어 힌트를 UTF-8 텍스트 파일로 전달 |
| `--language Korean` | 자동감지 | 언어 강제. 한/영 혼용 강의, 무음 환각 억제에는 **자동감지 권장** |
| `--save-json` | 꺼짐 | 단어 단위 타임스탬프를 `.json`으로 함께 저장 |
| `--chunk-seconds N` | 60 | 분할 목표 길이(10~150초). 평가 결과 60초가 균형이 가장 좋음 |
| `--flexibility N` | 15 | 쉼 지점 탐색 허용 폭(초) |
| `--max-tokens N` | 4096 | 구간별 출력 토큰 상한 |
| `--dedupe` | 꺼짐 | 30자 이상 동일 문장 연속 반복 제거 |

### 중단과 재개

- `Ctrl+C`로 끊거나 오류가 나도 **같은 명령을 다시 실행하면 이어서** 처리합니다.
- 진행 기록은 스크립트 옆 `.progress\` 폴더에 저장되고, 정상 완료되면 자동 삭제됩니다.
- 모델/옵션(청크 길이, 언어, context 등)이나 원본 파일이 바뀌면 체크포인트는 무시되고 처음부터 다시 합니다.

---

## 출력 형식과 동작 방식

결과는 원본 파일 옆에 `<원본이름>.txt`로 저장됩니다.

```
00:00
안녕하세요, 오늘은 지난 시간에 이어서 회귀모형의 가정에 대해 이야기하겠습니다. ...

02:14
그러면 귀무가설을 기각할 수 있는지 t 통계량으로 확인해 보면 ...
```

- 문단마다 시작 시각 `MM:SS` (1시간 넘는 파일은 `HH:MM:SS`)
- 문단은 150~350자, 2.5초 이상 침묵이나 문장 끝에서 나눔
- 시각은 Qwen3-ForcedAligner가 텍스트와 음성을 강제 정렬한 실제 발화 시각

처리 흐름:

1. FFmpeg로 파일 전체를 16kHz mono PCM으로 한 번에 디코딩 (다음 파일은 GPU 작업 중 미리 디코딩)
2. 녹음 레벨 기준 상대 무음을 찾아 약 60초 단위로, 긴 쉼 지점에서 분할
3. Qwen3-ASR로 배치 전사. 출력 토큰 한도에 걸린 구간은 쉼 지점에서 반으로 나눠 재전사 (누락 방지)
4. ForcedAligner로 단어 단위 시각 정렬. 정렬 실패 구간은 텍스트를 보존하고 구간 시각으로 대체
5. 문단 구성 후 원자적 저장, 체크포인트 삭제

실행 끝에 `[확인 필요]`가 표시되면 해당 구간은 토큰 한도 도달 또는 정렬 없음이니 결과를 눈으로 확인하세요.

---

## Whisper 환경 (보관용)

현재는 Qwen만 사용합니다. Whisper가 다시 필요할 때를 위한 기록입니다.

- 스크립트: `run_whisper.py` — **faster-whisper** `large-v3`, CUDA float16, beam 5, VAD on, 한국어 고정
- 모델: `Systran/faster-whisper-large-v3` (약 2.9GB, 첫 실행 시 자동 다운로드)
- 동작 확인 환경: Python 3.10.11 venv, `env\whisper_env_freeze.txt`
- 출력: `<원본이름>.txt`, 한 줄에 `[MM:SS] 세그먼트` 형식

> ⚠ **기존 `whisper_env`는 현재 실행되지 않습니다.** venv가 기반으로 쓰던 Python 3.10 설치가 PC에서 삭제되어 `No Python at ...` 오류가 납니다. 다시 쓰려면 아래 절차로 새로 만드세요.

```bat
rem Python 3.10 설치 후 (https://www.python.org/downloads/windows/)
cd /d <저장소>
rmdir /s /q whisper_env
py -3.10 -m venv whisper_env
whisper_env\Scripts\activate

python -m pip install -U pip
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-whisper.txt

python run_whisper.py "D:\강의\*.mp4"
```

- FFmpeg 별도 설정 불필요 (faster-whisper는 PyAV로 디코딩)
- `cublas64_12.dll` / `cudnn` 관련 오류가 나면 torch cu128이 같은 환경에 설치되어 있는지 확인하세요. 기존 환경은 torch와 함께 설치된 상태에서 동작했습니다.

> **주의: 두 엔진의 출력 파일명이 같습니다 (`<원본이름>.txt`).** 둘 다 기존 `.txt`가 있으면 건너뛰므로, 같은 폴더에서 두 엔진을 번갈아 쓰면 나중에 실행한 쪽은 처리하지 않습니다. 비교가 필요하면 결과 파일을 다른 곳으로 옮긴 뒤 실행하세요.

---

## 문제 해결

| 증상 | 원인 / 조치 |
|---|---|
| `CUDA GPU를 사용할 수 없습니다` | torch가 CPU 빌드로 설치됨. `pip uninstall -y torch torchaudio` 후 cu128 인덱스로 재설치. 드라이버 업데이트 |
| `FFmpeg를 못 찾았습니다` | 스크립트 폴더에 `ffmpeg-*\bin\ffmpeg.exe` 구조로 있는지 확인 |
| `[GPU 메모리 부족]`이 반복됨 | `--batch-size 2` 또는 `1`. 게임, 브라우저 영상 등 GPU를 쓰는 프로그램 종료 |
| `activate`가 PowerShell에서 보안 오류 | cmd에서 실행하거나 `run_qwen.bat` 사용 |
| 한글이 깨져 보임 | cmd에서 `chcp 65001` 후 실행. 배치파일은 자동으로 설정함 |
| 모델 다운로드가 느리거나 끊김 | 다시 실행하면 이어서 받음 |
| `출력 토큰 상태를 확인하지 못했습니다` | `qwen-asr` 내부 구조가 바뀐 것. `requirements-qwen.txt`의 고정 버전(0.0.6)으로 재설치 |
| conda 환경 이름이 다름 | `set QWEN_ENV=<이름>` 후 `run_qwen.bat` 실행 |

---

## 평가 자료 (_eval)

2026-10-16 v2 개선 시 기존 버전과 비교한 기록입니다. 요약은 [`_eval/평가요약.md`](_eval/평가요약.md)에 있습니다.

- 처리 속도: 기본값(60초, 배치 8)에서 실시간 대비 약 25~30배 (모델 로딩 제외)
- 정확도: 정답 원고가 없어 Whisper 결과 대비 CER과 전문용어 정오 표기 수로 비교
- git에는 스크립트(`compare.py`, `terms.py`, `diffview.py`, `run_new.sh`, `run_old_eval.py`), 설정(`cfg*.json`), 리포트(`report_*.json`), VRAM 측정값만 있습니다.
- 샘플 강의 미디어(`media/`, `edge/`), 전사 결과(`out_*/`, `whisper_ref/`), 체크포인트(`old_ckpt/`)는 용량과 강의 저작권 때문에 제외했습니다. 재현하려면 미디어를 직접 넣어야 합니다.
- `run_old_eval.py`는 git에 없는 `backup_20261016_223100/`의 이전 버전 스크립트를 불러옵니다. 이전 버전은 git 이력이 아니라 로컬 백업에만 있습니다.
- `run_new.sh`는 평가 당시 후보 스크립트(`run_qwen_forced_align.new.py`, 출력 `.qwen.txt`/`.qwen.json`) 기준입니다. 이 후보가 지금의 `run_qwen_forced_align.py`가 되었고, 현재는 출력 이름이 `.txt`/`.json`이라 그대로 다시 돌리려면 파일 이름을 맞춰야 합니다.
- `cfg*.json`의 경로는 cfg 파일 위치 기준 상대 경로입니다 (`python compare.py cfg.json`).

---

## git에 없는 것 (재설치 대상)

| 대상 | 다시 준비하는 방법 |
|---|---|
| `ffmpeg-*_build\` | [빠른 시작 3)](#3-ffmpeg-넣기-설치-없음-압축만-해제) |
| conda 환경 `qwen_asr` | `setup_qwen.bat` |
| `whisper_env\` | [Whisper 환경](#whisper-환경-보관용) |
| 모델 가중치 | 첫 실행 시 자동 다운로드 (`%USERPROFILE%\.cache\huggingface\hub`) |
| 강의 미디어, 전사 결과 | 각 원본 파일 위치에 있음 |
| `텍스트추출방법_영상_음성.txt` | 개인 메모 (실제 강의 폴더 경로 포함). 로컬에만 보관, 내용은 이 README로 대체 |
| `backup_*`, `*.bak`, `qwen_asr_share.zip` | 로컬 수동 백업. 이후 변경은 git 이력으로 관리 |
