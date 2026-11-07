# -*- coding: utf-8 -*-
"""Qwen3-ASR 1.7B: Windows 대학 강의 음성/영상 일괄 전사기 (v2).

- FFmpeg: 파일 전체를 16kHz mono float32 PCM으로 1회 디코딩 (청크별 재탐색 없음)
- 분할: 녹음 레벨에 맞춘 상대 무음 기준으로, 긴 쉼(문장 경계)을 우선해 절단
- GPU: 모델 1회 로딩, 여러 청크를 배치로 전사/정렬, OOM 시 배치 자동 축소
- 누락 방지: 출력 토큰 한도에 걸린 청크는 쉼 지점에서 반으로 나눠 재전사
- 정렬 실패/정렬 단위 없음(문장부호만 있는 출력 등)은 텍스트를 보존하고 청크 시각으로 대체
- TXT: Qwen3-ForcedAligner 실제 정렬 시각 + 2.5초 침묵/문장/길이 기준 150~350자 문단
- 실패/중단: 구간별 체크포인트로 재개, 정상 완료 시 체크포인트 정리

타임스탬프는 Qwen3-ForcedAligner-0.6B가 ASR 텍스트와 음성을 강제 정렬한 값입니다.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Transformers의 반복 경고/정보 메시지는 숨기고 전사 진행률만 표시합니다.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
FORCED_ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
PIPELINE_VERSION = 2
SAMPLE_RATE = 16000
EXTENSIONS = {
    ".m4a", ".mp4", ".mp3", ".wav", ".flac", ".aac",
    ".mkv", ".mov", ".webm", ".ogg", ".opus", ".wma", ".avi", ".m4b"
}
MIN_PARAGRAPH = 150
MAX_PARAGRAPH = 350
PAUSE_SECONDS = 2.5

# qwen_asr는 타임스탬프 사용 시 180초 초과 입력을 내부에서 다시 자릅니다. 그 아래로 유지합니다.
MAX_CHUNK_SECONDS = 170.0
FRAME = 160                 # 10ms @ 16kHz (무음 분석 단위)
MIN_PAUSE_SECONDS = 0.15    # 이보다 짧은 저에너지 구간은 쉼으로 보지 않음
MIN_SPLIT_SECONDS = 8.0     # 출력 한도 초과 청크를 나눌 때 최소 길이
TOKENS_PER_SECOND = 12.0    # 청크 길이에 비례한 출력 토큰 한도 (한도 초과 시 분할 재전사)
MIN_NEW_TOKENS = 256


@dataclass(frozen=True)
class Chunk:
    start: float
    end: float


# ---------------------------------------------------------------- 입력/FFmpeg

def available_inputs(patterns, recursive=False):
    result, seen = [], set()
    for item in patterns:
        p = Path(item)
        if p.is_dir():
            paths = p.rglob("*") if recursive else p.iterdir()
            matches = [str(q) for q in paths if q.is_file() and q.suffix.lower() in EXTENSIONS]
        elif p.is_file():
            matches = [str(p)]
        else:
            matches = glob.glob(item, recursive=recursive)
        for item_path in sorted(matches):
            q = Path(item_path)
            if not q.is_file() or q.suffix.lower() not in EXTENSIONS:
                continue
            key = os.path.normcase(str(q.resolve()))
            if key not in seen:
                seen.add(key)
                result.append(q.resolve())
    return result


def find_ffmpeg():
    root = Path(__file__).resolve().parent
    candidates = sorted(root.glob("ffmpeg-*/bin/ffmpeg.exe"), reverse=True)
    candidates += [root / "ffmpeg.exe"]
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))
    for exe in candidates:
        if exe.is_file():
            return str(exe)
    raise RuntimeError(f"FFmpeg를 못 찾았습니다. {root}\\ffmpeg-*\\bin\\ffmpeg.exe, "
                       f"{root}\\ffmpeg.exe 또는 PATH를 확인하세요.")


def run_command(args, **kwargs):
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.run(args, creationflags=creationflags, **kwargs)


def load_audio(ffmpeg, source, limit_seconds=None):
    """파일 전체(또는 앞 N초)를 16kHz mono float32 PCM으로 한 번에 디코딩합니다.

    길이는 실제 디코딩된 샘플 수로 정하므로 컨테이너 길이(영상 트랙)와 무관합니다.
    """
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE)]
    if limit_seconds:
        cmd += ["-t", f"{limit_seconds:.3f}"]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    cp = run_command(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode:
        raise RuntimeError("오디오 디코딩 실패: " + cp.stderr.decode("utf-8", errors="replace")[-800:])
    audio = np.frombuffer(cp.stdout, dtype="<f4")
    if audio.size < SAMPLE_RATE // 2:
        raise RuntimeError("디코딩된 오디오가 너무 짧거나 비어 있습니다.")
    return audio


# ---------------------------------------------------------------- 무음 분석/분할

@dataclass
class AudioAnalysis:
    db: np.ndarray          # 10ms 프레임별 에너지(dB), 50ms 평활
    pause_starts: np.ndarray
    pause_ends: np.ndarray  # 쉼 구간 [start, end) (초)
    threshold_db: float


def analyze_audio(audio):
    """녹음 레벨에 상대적인 기준으로 쉼(저에너지) 구간을 찾습니다."""
    n = audio.size // FRAME
    power = np.empty(n, dtype=np.float64)
    block = 100_000
    for i in range(0, n, block):
        x = audio[i * FRAME:min(n, i + block) * FRAME].reshape(-1, FRAME).astype(np.float64)
        power[i:i + x.shape[0]] = np.mean(x * x, axis=1)
    power = np.convolve(power, np.ones(5) / 5.0, mode="same")
    db = 10.0 * np.log10(power + 1e-12)

    floor, loud = np.percentile(db, 10), np.percentile(db, 90)
    spread = loud - floor
    # 배경 소음 바닥보다 (발화-바닥) 간격의 30% 이상, 최소 6dB 높은 지점까지를 쉼으로 봅니다.
    threshold = floor + max(6.0, 0.3 * spread) if spread >= 12.0 else floor + spread / 2.0
    quiet = db < threshold
    edges = np.diff(np.concatenate(([0], quiet.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    keep = (ends - starts) * FRAME / SAMPLE_RATE >= MIN_PAUSE_SECONDS
    sec = FRAME / SAMPLE_RATE
    return AudioAnalysis(db.astype(np.float32), starts[keep] * sec, ends[keep] * sec, float(threshold))


def choose_cut(analysis, lower, upper, desired):
    """[lower, upper] 안에서 절단점을 고릅니다. 긴 쉼을 우선하고, 없으면 최저 에너지 지점."""
    span = max(upper - lower, 1e-6)
    best, best_score = None, -math.inf
    i = bisect_right(analysis.pause_ends, lower)
    while i < len(analysis.pause_starts) and analysis.pause_starts[i] < upper:
        left, right = analysis.pause_starts[i], analysis.pause_ends[i]
        pos = min(max((left + right) / 2.0, lower), upper)
        score = min(right - left, 2.0) - 0.6 * abs(pos - desired) / span
        if score > best_score:
            best, best_score = pos, score
        i += 1
    if best is not None:
        return best
    sec = FRAME / SAMPLE_RATE
    lo = max(0, int(lower / sec))
    hi = min(len(analysis.db), int(upper / sec) + 1)
    if hi - lo < 3:
        return desired
    times = (np.arange(lo, hi) + 0.5) * sec
    score = analysis.db[lo:hi] + 3.0 * np.abs(times - desired) / span
    return float(times[int(np.argmin(score))])


def plan_chunks(duration, analysis, target, flexibility):
    """전체 구간을 빈틈/겹침 없이 커버합니다. 마지막 청크도 최소 길이를 보장합니다."""
    min_len = max(5.0, target - flexibility)
    max_len = min(MAX_CHUNK_SECONDS, target + flexibility)
    chunks, start = [], 0.0
    while duration - start > max_len:
        lower = start + min_len
        upper = min(start + max_len, duration - min_len)
        if upper < lower:
            lower = upper = start + (duration - start) / 2.0
        desired = min(max(start + target, lower), upper)
        cut = choose_cut(analysis, lower, upper, desired)
        cut = round(min(max(cut, lower), upper), 3)
        chunks.append(Chunk(round(start, 3), cut))
        start = cut
    chunks.append(Chunk(round(start, 3), round(duration, 3)))
    return chunks


def slice_audio(audio, start, end):
    return audio[int(round(start * SAMPLE_RATE)):int(round(end * SAMPLE_RATE))]


# ---------------------------------------------------------------- 모델

class GenerationMonitor:
    """qwen_asr 내부 generate 호출을 감싸 출력이 토큰 한도에서 잘렸는지 기록합니다."""

    def __init__(self, hf_model):
        self.generate = hf_model.generate
        eos = hf_model.generation_config.eos_token_id or [151643, 151645]
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.rows = []
        hf_model.generate = self

    def __call__(self, *args, **kwargs):
        out = self.generate(*args, **kwargs)
        limit = kwargs.get("max_new_tokens")
        generated = out.sequences[:, kwargs["input_ids"].shape[1]:].tolist()
        for row in generated:
            ended = any(t in self.eos for t in row)
            tokens = next((k for k, t in enumerate(row) if t in self.eos), len(row))
            self.rows.append({"tokens": tokens, "truncated": not ended and limit is not None and len(row) >= limit})
        return out


def load_model(args):
    import torch
    try:
        from transformers.utils import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except Exception:
        pass
    from qwen_asr import Qwen3ASRModel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다. qwen_asr Conda 환경을 확인하세요.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Qwen 모델 로딩: {MODEL_ID}")
    print(f"Forced Aligner 로딩: {FORCED_ALIGNER_ID}")
    model = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        forced_aligner=FORCED_ALIGNER_ID,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
        max_inference_batch_size=-1,  # 배치는 이 스크립트가 직접 나눕니다.
        max_new_tokens=args.max_tokens,
    )
    monitor = GenerationMonitor(model.model)
    return model, monitor


def is_oom(exc):
    import torch
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


class BatchRunner:
    """배치 실행. OOM이면 캐시를 비우고 배치를 반으로 줄여 같은 입력을 다시 처리합니다."""

    def __init__(self, batch_size):
        self.batch_size = max(1, batch_size)

    def run(self, fn, items):
        import torch
        out, i = [], 0
        while i < len(items):
            part = items[i:i + self.batch_size]
            try:
                out.extend(fn(part))
                i += len(part)
            except Exception as exc:
                if not is_oom(exc) or self.batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                self.batch_size = max(1, self.batch_size // 2)
                tqdm.write(f"  [GPU 메모리 부족] 배치 크기를 {self.batch_size}(으)로 줄여 다시 처리합니다.")
        return out


def asr_batch(model, monitor, audio, segments, args):
    """segments: [(start, end)] → [{"language", "text", "tokens", "truncated"}]"""
    import torch
    longest = max(e - s for s, e in segments)
    model.max_new_tokens = min(args.max_tokens, max(MIN_NEW_TOKENS, int(longest * TOKENS_PER_SECOND) + 64))
    monitor.rows = []
    with torch.inference_mode():
        results = model.transcribe(
            audio=[(slice_audio(audio, s, e), SAMPLE_RATE) for s, e in segments],
            context=args.context,
            language=args.language,
            return_time_stamps=False,
        )
    if len(monitor.rows) != len(results):
        raise RuntimeError("출력 토큰 상태를 확인하지 못했습니다 (qwen_asr 내부 구조 변경 가능성).")
    return [{"language": str(r.language or ""), "text": r.text.strip(), **row}
            for r, row in zip(results, monitor.rows)]


def recognize(model, monitor, runner, audio, analysis, segments, args, depth=0):
    """ASR 후, 토큰 한도에 걸린 구간은 쉼 지점에서 둘로 나눠 다시 전사합니다(전사 누락 방지)."""
    outs = runner.run(lambda part: asr_batch(model, monitor, audio, part, args), segments)
    pieces = []
    for (s, e), out in zip(segments, outs):
        if out["truncated"] and e - s >= 2 * MIN_SPLIT_SECONDS and depth < 4:
            mid = choose_cut(analysis, s + (e - s) * 0.35, e - (e - s) * 0.35, (s + e) / 2.0)
            tqdm.write(f"  [재분할] {timestamp(s)}~{timestamp(e)} 출력 한도 도달 → {timestamp(mid)}에서 나눠 재전사")
            pieces.extend(recognize(model, monitor, runner, audio, analysis,
                                    [(s, mid), (mid, e)], args, depth + 1))
        else:
            if out["truncated"]:
                tqdm.write(f"  [경고] {timestamp(s)}~{timestamp(e)} 출력이 토큰 한도에서 잘렸을 수 있습니다.")
            pieces.append({"start": s, "end": e, **out})
    return pieces


def _has_align_unit(text):
    return any(_is_align_char(ch) for ch in text)


def align_batch(model, audio, pieces, args):
    """ForcedAligner 정렬. 실패하거나 정렬 단위가 없으면 텍스트는 그대로 두고 정렬만 비웁니다."""
    import torch
    todo = [p for p in pieces if p["text"] and _has_align_unit(p["text"])]
    if not todo:
        return
    with torch.inference_mode():
        results = model.forced_aligner.align(
            audio=[(slice_audio(audio, p["start"], p["end"]), SAMPLE_RATE) for p in todo],
            text=[p["text"] for p in todo],
            language=[p["language"] or args.language or "" for p in todo],
        )
    for p, res in zip(todo, results):
        p["aligned"] = clamp_alignment(res, p["start"], p["end"])


def clamp_alignment(result, start, end):
    """청크 기준 시각을 파일 기준으로 바꾸고, 청크 범위 안 + 시간순으로 보정합니다."""
    items, floor = [], start
    for it in result.items:
        s = min(max(start + float(it.start_time), floor), end)
        e = min(max(start + float(it.end_time), s), end)
        items.append({"text": str(it.text), "start": round(s, 3), "end": round(e, 3)})
        floor = s
    return items


def align_pieces(model, runner, audio, pieces, args):
    for p in pieces:
        p["aligned"] = []
    try:
        runner.run(lambda part: align_batch(model, audio, part, args) or part, pieces)
    except Exception as exc:
        if is_oom(exc) or "CUDA error" in str(exc):
            raise
        # 배치 중 한 항목 문제로 전체가 실패하지 않도록 개별 재시도합니다.
        for p in pieces:
            if p["aligned"]:
                continue
            try:
                align_batch(model, audio, [p], args)
            except Exception as item_exc:
                if is_oom(item_exc) or "CUDA error" in str(item_exc):
                    raise
                tqdm.write(f"  [정렬 실패] {timestamp(p['start'])}: {type(item_exc).__name__}: {item_exc} "
                           "→ 텍스트 보존, 청크 시각 사용")


def build_record(index, chunk, pieces):
    texts = [p["text"] for p in pieces if p["text"]]
    return {
        "index": index,
        "start": chunk.start,
        "end": chunk.end,
        "text": " ".join(texts),
        "language": next((p["language"] for p in pieces if p["language"]), ""),
        "aligned": [a for p in pieces for a in p["aligned"]],
        "pieces": [{"start": p["start"], "end": p["end"], "tokens": p["tokens"],
                    "truncated": p["truncated"], "aligned": bool(p["aligned"]) or not p["text"]}
                   for p in pieces],
    }


# ---------------------------------------------------------------- 문단 구성

def timestamp(seconds, hours=False):
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if (h or hours) else f"{m:02d}:{s:02d}"


def suppress_obvious_loops(text):
    """긴 문장의 바로 연속된 완전 동일 반복만 제거 (--dedupe 지정 시에만 사용)."""
    rx = re.compile(r"(.{30,}?[.!?。！？])\s+\1(?=\s|$)")
    for _ in range(3):
        updated = rx.sub(r"\1", text)
        if updated == text:
            break
        text = updated
    return text


def split_long_piece(text, limit=MAX_PARAGRAPH):
    """의미 있는 단어 경계에서만 긴 문장 분할; 문자 누락 없음."""
    out = []
    remaining = text.strip()
    while len(remaining) > limit:
        split_at = remaining.rfind(" ", MIN_PARAGRAPH, limit + 1)
        if split_at < MIN_PARAGRAPH:
            split_at = remaining.find(" ", limit)
        if split_at < 1:  # 공백 없는 긴 문자열은 예외적으로 길게 보존
            break
        out.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        out.append(remaining)
    return out


def _is_align_char(ch):
    """ForcedAligner와 같은 기준의 정렬 문자: 문자/숫자/아포스트로피."""
    if ch == "'":
        return True
    category = unicodedata.category(ch)
    return category.startswith("L") or category.startswith("N")


def _normalized_with_positions(text):
    normalized, positions = [], []
    for index, ch in enumerate(text):
        if not _is_align_char(ch):
            continue
        for c in ch.lower():
            normalized.append(c)
            positions.append(index)
    return "".join(normalized), positions


def _normalize_token(text):
    return "".join(ch.lower() for ch in text if _is_align_char(ch))


def alignment_spans(text, aligned_items):
    """ForcedAligner 단위를 원문 문자 위치에 대응시킵니다.

    정렬 단위는 같은 텍스트에서 문장부호만 제거해 순서대로 만든 것이므로 앞으로만 탐색합니다.
    """
    normalized, positions = _normalized_with_positions(text)
    if not normalized:
        return []
    spans, cursor = [], 0
    for item in aligned_items:
        token = _normalize_token(str(item.get("text", "")))
        if not token:
            continue
        found = normalized.find(token, cursor)
        if found < 0:
            continue
        spans.append({
            "char_start": positions[found],
            "char_end": positions[found + len(token) - 1] + 1,
            "start": float(item["start"]),
            "end": float(item["end"]),
        })
        cursor = found + len(token)
    return spans


def piece_time_range(aligned_spans, piece_start, piece_end, fallback_start, fallback_end):
    """원문 문자 범위에 겹치는 ForcedAligner 단위의 실제 시작/끝 시각을 반환합니다."""
    hits = [item for item in aligned_spans
            if item["char_end"] > piece_start and item["char_start"] < piece_end]
    if hits:
        return hits[0]["start"], hits[-1]["end"]
    previous = [item for item in aligned_spans if item["char_start"] <= piece_start]
    if previous:
        return previous[-1]["start"], previous[-1]["end"]
    if aligned_spans:
        return aligned_spans[0]["start"], aligned_spans[0]["end"]
    return fallback_start, fallback_end


def pause_split_ranges(aligned_spans, span_start, span_end, pause_seconds=PAUSE_SECONDS):
    """한 문장 안에서도 실제 발화 공백이 길면 그 지점에서 문자 범위를 나눕니다."""
    inside = [item for item in aligned_spans
              if item["char_end"] > span_start and item["char_start"] < span_end]
    if len(inside) < 2:
        return [(span_start, span_end)]
    cuts = [span_start]
    previous = inside[0]
    for current in inside[1:]:
        if current["start"] - previous["end"] >= pause_seconds:
            cut = max(span_start, min(span_end, current["char_start"]))
            if cut > cuts[-1]:
                cuts.append(cut)
        previous = current
    cuts.append(span_end)
    return [(left, right) for left, right in zip(cuts, cuts[1:]) if right > left]


def timed_pieces(records):
    """ForcedAligner 기준 실제 발화 시작/끝 시각이 붙은 문장 조각을 생성합니다."""
    for rec in records:
        text = rec["text"].strip()
        if not text:
            continue
        aligned = alignment_spans(text, rec.get("aligned", []))
        fallback_start, fallback_end = float(rec["start"]), float(rec["end"])
        for match in re.finditer(r".+?(?:[.!?。！？]+(?=\s|$)|$)\s*", text, flags=re.S):
            for range_start, range_end in pause_split_ranges(aligned, match.start(), match.end()):
                raw = text[range_start:range_end].strip()
                if not raw:
                    continue
                search_from = range_start
                for part in split_long_piece(raw):
                    local_start = text.find(part, search_from, range_end)
                    if local_start < 0:
                        local_start = min(search_from, len(text))
                    local_end = min(len(text), local_start + len(part))
                    start_time, end_time = piece_time_range(
                        aligned, local_start, max(local_start + 1, local_end),
                        fallback_start, fallback_end)
                    yield start_time, end_time, part
                    search_from = max(search_from, local_end)


def format_paragraphs(records, min_len=MIN_PARAGRAPH, max_len=MAX_PARAGRAPH, dedupe=False, hours=False):
    """문장/길이/실제 침묵을 이용해 MM:SS(1시간 이상 파일은 HH:MM:SS) 문단을 구성합니다."""
    blocks, fragments = [], []
    start = previous_end = None

    def flush():
        nonlocal fragments, start
        if fragments:
            body = " ".join(fragments).strip()
            if dedupe:
                body = suppress_obvious_loops(body)
            if body:
                blocks.append({"start": start, "end": previous_end, "text": body})
        fragments = []
        start = None

    for piece_start, piece_end, part in timed_pieces(records):
        if fragments and previous_end is not None and piece_start - previous_end >= PAUSE_SECONDS:
            flush()
        existing = len(" ".join(fragments))
        if fragments and existing >= min_len and existing + len(part) + 1 > max_len:
            flush()
        if start is None:
            start = piece_start
        fragments.append(part)
        previous_end = piece_end
        combined_len = len(" ".join(fragments))
        if combined_len >= max_len:
            flush()
        elif combined_len >= min_len and re.search(r"[.!?。！？][\"'”’)]*$", part):
            flush()
    flush()
    return [f"{timestamp(b['start'], hours)}\n{b['text']}" for b in blocks], blocks


# ---------------------------------------------------------------- 체크포인트/출력

def output_for(source, args):
    if args.minutes:
        minutes_str = f"{args.minutes:g}".replace(".", "p")
        return source.with_name(source.stem + f".test-{minutes_str}min.txt")
    return source.with_name(source.stem + ".txt")


def checkpoint_path(source):
    stamp = hashlib.sha256(os.path.normcase(str(source)).encode("utf-8")).hexdigest()[:14]
    folder = Path(__file__).resolve().parent / ".progress"
    folder.mkdir(exist_ok=True)
    return folder / f"{stamp}.jsonl"


def signature_for(source, args):
    """전사 결과에 영향을 주는 값만 포함합니다 (문단/중복제거 같은 후처리 옵션은 제외)."""
    stat = source.stat()
    info = {"version": PIPELINE_VERSION, "path": str(source), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "model": MODEL_ID, "forced_aligner": FORCED_ALIGNER_ID,
            "minutes": args.minutes, "chunk": args.chunk_seconds, "flex": args.flexibility,
            "language": args.language, "context": args.context, "max_tokens": args.max_tokens}
    return hashlib.sha256(json.dumps(info, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def restore_progress(checkpoint, signature, chunks):
    if not checkpoint.exists():
        return {}
    try:
        with checkpoint.open("r", encoding="utf-8") as f:
            header = json.loads(f.readline())
            if header.get("signature") != signature or header.get("chunks") != len(chunks):
                return {}
            records = {}
            for line in f:
                if not line.endswith("\n"):  # 마지막 기록이 끊긴 경우 무시
                    break
                try:
                    item = json.loads(line)
                except ValueError:
                    break
                i = item["index"]
                if (0 <= i < len(chunks) and abs(item["start"] - chunks[i].start) < 0.01
                        and abs(item["end"] - chunks[i].end) < 0.01):
                    records[i] = item
            return records
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def reset_checkpoint(checkpoint, signature, n_chunks, old=None):
    """기존 정상 구간을 보존하고, 깨진 마지막 줄을 제거해 원자적으로 복구."""
    tmp = checkpoint.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"signature": signature, "chunks": n_chunks}) + "\n")
            for index in sorted(old or {}):
                f.write(json.dumps(old[index], ensure_ascii=False) + "\n")
        os.replace(tmp, checkpoint)
    finally:
        tmp.unlink(missing_ok=True)


def append_checkpoint(checkpoint, items):
    with checkpoint.open("a", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_atomic(output, content):
    tmp = output.with_name(output.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()


# ---------------------------------------------------------------- 파일 처리

def process_file(model, monitor, runner, source, audio, args):
    output = output_for(source, args)
    duration = audio.size / SAMPLE_RATE
    analysis = analyze_audio(audio)
    chunks = plan_chunks(duration, analysis, args.chunk_seconds, args.flexibility)
    checkpoint = checkpoint_path(source)
    signature = signature_for(source, args)
    old = restore_progress(checkpoint, signature, chunks)
    reset_checkpoint(checkpoint, signature, len(chunks), old)
    print(f"  총 음성 길이: {timestamp(duration)} | 처리 구간: {len(chunks)}개 | 이전 완료 구간: {len(old)}개")
    start_time = time.perf_counter()
    records = dict(old)
    pending = [i for i in range(len(chunks)) if i not in records]
    already_done = sum(chunks[k].end - chunks[k].start for k in records)

    with tqdm(total=round(duration, 1), initial=round(already_done, 1), unit="초",
              desc="변환 진행률", dynamic_ncols=True) as pbar:
        step = max(1, args.batch_size)
        for b in range(0, len(pending), step):
            group = pending[b:b + step]
            segments = [(chunks[i].start, chunks[i].end) for i in group]
            pieces = recognize(model, monitor, runner, audio, analysis, segments, args)
            align_pieces(model, runner, audio, pieces, args)
            new_records = []
            for i in group:
                c = chunks[i]
                mine = [p for p in pieces if c.start - 1e-6 <= p["start"] and p["end"] <= c.end + 1e-6]
                new_records.append(build_record(i, c, mine))
            append_checkpoint(checkpoint, new_records)
            for rec in new_records:
                records[rec["index"]] = rec
            pbar.update(round(sum(chunks[i].end - chunks[i].start for i in group), 1))

    arranged = [records[i] for i in range(len(chunks))]
    lines, blocks = format_paragraphs(arranged, dedupe=args.dedupe, hours=duration >= 3600)
    save_atomic(output, "\n\n".join(lines) + ("\n" if lines else ""))
    if args.save_json:
        payload = {
            "source": str(source), "model": MODEL_ID, "forced_aligner": FORCED_ALIGNER_ID,
            "pipeline_version": PIPELINE_VERSION, "duration": round(duration, 3),
            "paragraphs": blocks,
            "chunks": [{k: r[k] for k in ("start", "end", "text", "language", "pieces")} for r in arranged],
            "words": [w for r in arranged for w in r["aligned"]],
        }
        save_atomic(output.with_suffix(".json"), json.dumps(payload, ensure_ascii=False, indent=1))
    checkpoint.unlink(missing_ok=True)

    elapsed = time.perf_counter() - start_time
    issues = [(r["start"], p) for r in arranged for p in r.get("pieces", []) if p["truncated"] or not p["aligned"]]
    for at, p in issues:
        kind = "출력 한도 도달" if p["truncated"] else "정렬 없음"
        print(f"  [확인 필요] {timestamp(p['start'])}~{timestamp(p['end'])} {kind}")
    speed = (duration - already_done) / elapsed if elapsed > 0 else 0
    print(f"  [완료] {output} | 문단 {len(lines)}개 | 처리 {elapsed:.0f}초 (실시간 대비 {speed:.1f}배)")
    return "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR 대학 강의 일괄 전사 (Qwen3 ForcedAligner 정밀 타임스탬프)")
    parser.add_argument("input_files", nargs="+", help="영상/음성 파일 여러 개, 와일드카드, 또는 폴더")
    parser.add_argument("--recursive", action="store_true", help="폴더 하위까지 재귀 탐색")
    parser.add_argument("--overwrite", action="store_true", help="기존 .txt 결과 덮어쓰기")
    parser.add_argument("--minutes", type=float, default=0, help="각 파일 앞부분 N분만 시험 (0=전체)")
    parser.add_argument("--chunk-seconds", type=float, default=60, help="구간 목표 길이(초, 기본 60)")
    parser.add_argument("--flexibility", type=float, default=15, help="쉼 지점 탐색 허용 폭(초, 기본 15)")
    parser.add_argument("--batch-size", type=int, default=8, help="GPU 동시 처리 청크 수(기본 8, OOM 시 자동 축소)")
    parser.add_argument(
        "--language", default=None,
        help="강제 언어 (예: Korean, English). 기본 자동감지; 무음/소음 억제와 혼합 발화에는 기본값 권장")
    parser.add_argument("--context", default="", help="전문용어 등 ASR 문맥 힌트 (기본 없음)")
    parser.add_argument("--context-file", default=None, help="문맥 힌트를 담은 UTF-8 텍스트 파일 (--context 뒤에 추가)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="구간별 출력 토큰 상한 (기본 4096; 실제 한도는 청크 길이에 비례해 자동 설정)")
    parser.add_argument("--dedupe", action="store_true",
                        help="30자 이상 문장이 연속으로 완전히 같으면 하나만 남김 (기본 꺼짐: 실제 반복 발화 보존)")
    parser.add_argument("--save-json", action="store_true",
                        help="단어 단위 정렬 시각을 .json으로 함께 저장")
    args = parser.parse_args()
    if args.minutes < 0 or not 10 <= args.chunk_seconds <= 150 or not 0 <= args.flexibility < args.chunk_seconds / 2:
        parser.error("--minutes >= 0, --chunk-seconds 10~150, --flexibility는 청크 절반보다 작게 설정하세요.")
    if args.max_tokens < 256 or args.batch_size < 1:
        parser.error("--max-tokens >= 256, --batch-size >= 1 이어야 합니다.")
    if args.context_file:
        extra = Path(args.context_file).read_text(encoding="utf-8").strip()
        args.context = (args.context + "\n" + extra).strip()
    files = available_inputs(args.input_files, args.recursive)
    if not files:
        parser.error("입력한 경로에 지원되는 음성/영상 파일이 없습니다.")
    ffmpeg = find_ffmpeg()
    if args.language:
        print("[주의] --language를 강제하면 비음성 구간에서도 텍스트 출력을 유도할 수 있습니다.")
        print("       한국어+영어 혼용 강의와 무음/소음 환각 억제에는 기본 자동감지를 먼저 권장합니다.")
    print(f"대상 {len(files)}개 | 모델 {MODEL_ID} | CUDA / BF16 | 배치 {args.batch_size}")
    print(f"타임스탬프: {FORCED_ALIGNER_ID} 강제 정렬")
    print("FFmpeg:", ffmpeg)
    to_run = [f for f in files if args.overwrite or not output_for(f, args).exists()]
    if not to_run:
        print("모든 파일에 Qwen 결과가 이미 존재합니다. 종료합니다.")
        return 0
    model, monitor = load_model(args)  # 모든 파일에서 GPU 모델 재사용
    runner = BatchRunner(args.batch_size)
    limit = args.minutes * 60.0 if args.minutes else None
    summary = {"ok": 0, "skipped": len(files) - len(to_run), "failed": 0}

    # GPU가 현재 파일을 처리하는 동안 다음 파일을 미리 디코딩합니다.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(load_audio, ffmpeg, to_run[0], limit)
        for i, file_path in enumerate(to_run, start=1):
            print(f"\n[{i}/{len(to_run)}] {file_path.name}")
            try:
                audio = future.result()
            except Exception as exc:
                audio = None
                load_error = exc
            future = pool.submit(load_audio, ffmpeg, to_run[i], limit) if i < len(to_run) else None
            try:
                if audio is None:
                    raise load_error
                summary[process_file(model, monitor, runner, file_path, audio, args)] += 1
            except KeyboardInterrupt:
                print("\n[중단] 진행 상황을 저장했습니다. 같은 명령으로 실행하면 중단 지점부터 재개합니다.")
                return 130
            except Exception as exc:
                summary["failed"] += 1
                print(f"  [오류] {type(exc).__name__}: {exc}", file=sys.stderr)
                if "CUDA error" in str(exc):
                    print("  CUDA 오류는 이후 파일도 실패시키므로 중단합니다. 다시 실행하면 이어서 처리합니다.")
                    break
                import torch
                torch.cuda.empty_cache()
                print("  다음 파일로 계속합니다. 실패 파일은 다음 실행에서 재개할 수 있습니다.")
            finally:
                audio = None
    print("\n전체 결과:", ", ".join(f"{k} {v}개" for k, v in summary.items()))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
