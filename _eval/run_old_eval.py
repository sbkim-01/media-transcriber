# -*- coding: utf-8 -*-
"""평가용: 백업된 기존 스크립트를 그대로 실행하되 체크포인트(정렬 데이터)를 보존한다."""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "backup_20261016_223100"))
import run_qwen_forced_align as old  # noqa: E402

KEEP = HERE / "old_ckpt"
KEEP.mkdir(exist_ok=True)


class KeepPath(type(Path())):
    def unlink(self, missing_ok=False):
        pass


old.checkpoint_path = lambda source: KeepPath(KEEP / (Path(source).stem + ".jsonl"))
_ff = sorted(ROOT.glob("ffmpeg-*/bin/ffmpeg.exe"), reverse=True)[0]
old.find_ffmpeg = lambda: (str(_ff), str(_ff.with_name("ffprobe.exe")))

if __name__ == "__main__":
    t0 = time.perf_counter()
    rc = old.main()
    print(f"[eval] total wall {time.perf_counter() - t0:.1f}s")
    sys.exit(rc)
