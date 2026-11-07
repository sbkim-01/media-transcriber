import argparse
import glob
import os
import time
import traceback

from faster_whisper import WhisperModel
from tqdm import tqdm


# ==========================================
# 기본 설정: 정확도 우선
# ==========================================

MODEL_NAME = "large-v3"
COMPUTE_TYPE = "float16"
BEAM_SIZE = 5

# 기존 전사 설정 유지
VAD_FILTER = True
CONDITION_ON_PREVIOUS_TEXT = False

# ==========================================


def format_timestamp(seconds):
    """초를 MM:SS 또는 HH:MM:SS(1시간 이상)로 변환. 초 미만은 버림"""
    m, s = divmod(int(max(0.0, seconds)), 60)
    h, m = divmod(m, 60)

    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def collect_files(input_patterns):
    """파일 경로 및 와일드카드 처리, 중복 제거"""
    files = []
    seen = set()

    for pattern in input_patterns:
        if os.path.isfile(pattern):
            matches = [pattern]
        else:
            matches = glob.glob(pattern, recursive=True)

        if not matches:
            print(f"[경고] 일치하는 파일이 없습니다: {pattern}")
            continue

        for path in matches:
            if not os.path.isfile(path):
                continue

            normalized = os.path.normcase(os.path.abspath(path))

            if normalized not in seen:
                seen.add(normalized)
                files.append(path)

    return files


def resolve_output_paths(files):
    """
    입력 파일별 출력 TXT 경로 결정.

    기본은 '<파일명>.txt'. 같은 폴더에 이름이 같고 확장자만 다른 파일
    (예: lecture.mp4, lecture.mp3)이 있어 출력 경로가 겹치면
    서로 덮어쓰지 않도록 원래 확장자를 포함해 '<파일명.확장자>.txt'로 저장.
    """
    default_paths = {}
    counts = {}

    for path in files:
        base, _ = os.path.splitext(path)
        output = f"{base}.txt"
        default_paths[path] = output

        key = os.path.normcase(os.path.abspath(output))
        counts[key] = counts.get(key, 0) + 1

    resolved = {}

    for path in files:
        output = default_paths[path]
        key = os.path.normcase(os.path.abspath(output))

        if counts[key] > 1:
            output = f"{path}.txt"
            print(
                f"[알림] 출력 파일명 충돌 방지: "
                f"{os.path.basename(path)} -> {os.path.basename(output)}"
            )

        resolved[path] = output

    return resolved


def segments_to_lines(segments, total_duration):
    """
    Whisper 세그먼트를 1세그먼트 = 1줄로 변환.

    - 세그먼트를 병합/분할하지 않고 Whisper가 반환한 start를 기록
    - 텍스트는 앞뒤 공백만 제거하고 내용은 가공하지 않음
    """
    lines = []

    with tqdm(
        total=total_duration,
        unit="초",
        desc="변환 진행률",
        dynamic_ncols=True
    ) as pbar:

        for segment in segments:
            # 원본 오디오 기준 진행률
            progress = min(max(segment.end, 0), total_duration)
            pbar.update(max(0, progress - pbar.n))

            text = segment.text.strip()

            if not text:
                continue

            lines.append(f"[{format_timestamp(segment.start)}] {text}")

        # 정상 종료된 경우에만 100% 표시
        pbar.update(max(0, total_duration - pbar.n))

    return lines


def write_atomic(output_path, content):
    """임시 파일에 작성 후 교체. 실패 시 기존 TXT 보존"""
    temp_path = f"{output_path}.{os.getpid()}.tmp"

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)

        os.replace(temp_path, output_path)

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="한국어 강의 타임라인 TXT 일괄 추출기"
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help="변환할 음성/영상 파일 경로 또는 와일드카드"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 TXT 파일 덮어쓰기 허용"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="오류 발생 시 상세 트레이스백 출력"
    )

    args = parser.parse_args()

    files = collect_files(args.input_files)

    if not files:
        print("변환할 파일이 없습니다.")
        return

    output_paths = resolve_output_paths(files)
    total_files = len(files)

    print("=" * 55)
    print(f"총 {total_files}개의 파일이 입력되었습니다.")
    print(f"모델: {MODEL_NAME}")
    print(f"연산 방식: CUDA / {COMPUTE_TYPE}")
    print(f"기존 TXT 보호: {'덮어쓰기' if args.overwrite else '건너뛰기'}")
    print("=" * 55)

    # GPU 모델은 최초 1회만 로딩
    print("\nAI 모델을 GPU에 로딩하는 중...")

    try:
        model = WhisperModel(
            MODEL_NAME,
            device="cuda",
            compute_type=COMPUTE_TYPE
        )
    except Exception as e:
        print(f"\n[오류] 모델 로딩 실패: {type(e).__name__}: {e}")
        print("CUDA 드라이버, cuDNN/cuBLAS DLL, GPU 사용 가능 여부,")
        print("모델 다운로드(네트워크/캐시) 상태를 확인하세요.")

        if args.verbose:
            traceback.print_exc()

        raise SystemExit(1)

    completed = 0
    skipped = 0
    failures = []  # (파일명, 사유)

    for index, file_path in enumerate(files, start=1):
        print(f"\n[{index}/{total_files}] {'-' * 45}")

        output_path = output_paths[file_path]

        print(f"파일: {os.path.basename(file_path)}")

        # 기존 파일 덮어쓰기 방지
        if os.path.exists(output_path) and not args.overwrite:
            print("[건너뜀] 기존 TXT 파일이 존재합니다.")
            skipped += 1
            continue

        start_time = time.time()

        try:
            segments, info = model.transcribe(
                file_path,
                language="ko",
                beam_size=BEAM_SIZE,
                vad_filter=VAD_FILTER,
                condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            )

            lines = segments_to_lines(segments, max(0, info.duration))

            # 빈 결과는 저장하지 않음 (이후 실행에서 '기존 TXT'로 오인 방지)
            if not lines:
                print("\n[경고] 전사 결과가 비어 있어 TXT를 저장하지 않았습니다.")
                print("무음이거나 VAD가 음성을 감지하지 못했을 수 있습니다.")
                failures.append((os.path.basename(file_path), "빈 전사 결과"))
                continue

            write_atomic(output_path, "\n".join(lines) + "\n")

            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            print(f"\n[완료] {output_path}")
            print(f"추출 세그먼트 수: {len(lines)}개")
            print(f"소요 시간: {minutes}분 {seconds}초")

            completed += 1

        except KeyboardInterrupt:
            print("\n[중단] 사용자가 작업을 취소했습니다.")
            raise SystemExit(130)

        except Exception as e:
            print(f"\n[오류] 전사 실패: {type(e).__name__}: {e}")

            if args.verbose:
                traceback.print_exc()

            failures.append(
                (os.path.basename(file_path), f"{type(e).__name__}: {e}")
            )
            print("다음 파일로 넘어갑니다.")

    print("\n" + "=" * 55)
    print("전체 작업 결과")
    print(f"완료: {completed}개")
    print(f"건너뜀: {skipped}개")
    print(f"실패: {len(failures)}개")

    for name, reason in failures:
        print(f"  - {name}: {reason}")

    print("=" * 55)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
