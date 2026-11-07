# -*- coding: utf-8 -*-
"""전사 결과 비교 (정답 원고가 없으므로 대리 지표 사용).

- cer_vs_whisper: 기존 Whisper large-v3 결과와의 문자 편집거리 / Whisper 글자 수 (낮을수록 일치)
- uncovered_speech: 발화 에너지가 있는데 정렬 단어가 0.5초 이내에 하나도 없는 시간(초) → 누락 의심
- words_in_pause: 단어 구간 전체가 쉼(저에너지) 안에 있는 단어 비율 → 타임스탬프 품질/환각 의심
- 설정 간 CER, 같은 단어의 시작 시각 차이(중앙값/90%)
"""
import difflib
import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("new", HERE.parent / "run_qwen_forced_align.py")
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)


def norm(text):
    return "".join(ch.lower() for ch in text if unicodedata.category(ch)[0] in "LN")


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    bb = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    prev = np.arange(len(b) + 1, dtype=np.int32)
    ar = np.arange(len(b) + 1, dtype=np.int32)
    for i, ch in enumerate(a, start=1):
        cost = (bb != ord(ch)).astype(np.int32)
        tmp = np.empty_like(prev)
        tmp[0] = i
        tmp[1:] = np.minimum(prev[1:] + 1, prev[:-1] + cost)
        prev = np.minimum.accumulate(tmp - ar) + ar
    return int(prev[-1])


def load_run(path):
    """old_ckpt jsonl 또는 new .qwen.json → (text, words)"""
    path = Path(path)
    if path.suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()[1:]
        recs = sorted((json.loads(l) for l in lines if l.strip()), key=lambda r: r["index"])
        return " ".join(r["text"] for r in recs), [w for r in recs for w in r["aligned"]], recs
    data = json.loads(path.read_text(encoding="utf-8"))
    return " ".join(c["text"] for c in data["chunks"]), data["words"], data["chunks"]


def whisper_text(path):
    txt = Path(path).read_text(encoding="utf-8")
    return " ".join(l for l in txt.splitlines() if not re.fullmatch(r"\s*\d{1,2}:\d{2}(:\d{2})?\s*", l))


def coverage_metrics(words, analysis, duration):
    sec = new.FRAME / new.SAMPLE_RATE
    n = len(analysis.db)
    active = analysis.db >= analysis.threshold_db
    covered = np.zeros(n, dtype=bool)
    in_pause = 0
    for w in words:
        a, b = int(max(0, w["start"] - 0.5) / sec), int(min(duration, w["end"] + 0.5) / sec) + 1
        covered[a:b] = True
        s, e = int(w["start"] / sec), max(int(w["start"] / sec) + 1, int(w["end"] / sec))
        if e > s and not active[s:e].any():
            in_pause += 1
    # 0.3초 미만으로 튀는 에너지는 무시하고, 1초 이상 연속된 미커버 발화만 집계
    gap = active & ~covered
    total, run = 0, 0
    runs = []
    for i, g in enumerate(np.append(gap, False)):
        if g:
            run += 1
        else:
            if run * sec >= 1.0:
                total += run
                runs.append(((i - run) * sec, run * sec))
            run = 0
    return total * sec, in_pause / max(1, len(words)), runs


def word_time_diff(w1, w2):
    t1 = [norm(w["text"]) for w in w1]
    t2 = [norm(w["text"]) for w in w2]
    sm = difflib.SequenceMatcher(None, t1, t2, autojunk=False)
    d = []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            d.append(abs(w1[blk.a + k]["start"] - w2[blk.b + k]["start"]))
    d = np.array(d) if d else np.array([np.nan])
    return np.median(d), np.percentile(d, 90), np.mean(d > 0.5), len(d)


def main():
    cfg_path = Path(sys.argv[1]).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    at = lambda p: cfg_path.parent / p  # cfg 안의 상대 경로는 cfg 파일 위치 기준
    ff = new.find_ffmpeg()
    report = {}
    for stem, item in cfg["files"].items():
        audio = new.load_audio(ff, at(item["media"]))
        duration = audio.size / new.SAMPLE_RATE
        analysis = new.analyze_audio(audio)
        ref = norm(whisper_text(at(item["whisper"])))
        runs = {name: load_run(at(p)) for name, p in item["runs"].items() if at(p).exists()}
        print(f"\n=== {stem} ({duration/60:.1f}분, whisper {len(ref)}자)")
        print(f"{'run':14s} {'chars':>7s} {'CER_vs_W':>9s} {'uncov_s':>8s} {'inPause%':>9s} {'words':>6s}")
        res = {}
        for name, (text, words, _) in runs.items():
            t = norm(text)
            cer = levenshtein(t, ref) / max(1, len(ref))
            unc, inp, gaps = coverage_metrics(words, analysis, duration)
            res[name] = {"chars": len(t), "cer_vs_whisper": cer, "uncovered_s": unc,
                         "words_in_pause": inp, "words": len(words),
                         "gaps": [(round(a, 1), round(b, 1)) for a, b in gaps if b >= 3]}
            print(f"{name:14s} {len(t):7d} {cer*100:8.2f}% {unc:8.1f} {inp*100:8.2f}% {len(words):6d}")
        names = list(runs)
        base = names[0]
        for name in names[1:]:
            c = levenshtein(norm(runs[name][0]), norm(runs[base][0])) / max(1, len(norm(runs[base][0])))
            med, p90, frac, n = word_time_diff(runs[base][1], runs[name][1])
            res[name]["cer_vs_" + base] = c
            res[name]["word_dt"] = {"median": med, "p90": p90, "frac_gt_0.5s": frac, "matched": n}
            print(f"  {name} vs {base}: CER {c*100:.2f}% | 단어 시작시각 차 중앙 {med:.3f}s, 90% {p90:.3f}s, >0.5s {frac*100:.1f}% (n={n})")
        for name in names:
            if res[name]["gaps"]:
                print(f"  {name} 3초 이상 미커버 발화: {res[name]['gaps'][:12]}")
        report[stem] = res
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
