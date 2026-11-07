# 강의별 전문용어 정오 표기 횟수 비교 (정답 원고 대신 쓰는 정확도 탐침)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load_run, whisper_text
terms = {
 "자본시장과윤리_07주차_01": {"good": ["사모", "한앤컴퍼니", "MBK", "구조조정", "투자 회수", "투자회수"], "bad": ["사무펀드", "사무 펀드", "삼호", "상호 펀드", "상호펀드", "한행", "결혼권", "결론권", "3억 펀드"]},
 "금융계량경제_06주차_01": {"good": ["귀무가설", "귀무 가설", "회귀모형", "회귀 모형", "잔차", "통계량", "자유도", "결정계수"], "bad": ["규모 가설", "기무가설", "기무 가설", "산별량", "삼별량", "회귀모양", "회귀 모양", "헬스로", "했수로"]},
 "포스트모던_06주차_01차시": {"good": ["보드리야르", "시뮬라크르", "그물망", "핫플"], "bad": ["기보드르", "기 모드르", "금을만"]},
}
runs = sys.argv[1:] or ["old","n35_b1","n35_b8","n60_b8","n90_b8","n120_b8"]
tot = {}
for s, t in terms.items():
    print("==", s)
    rows = {}
    for r in runs:
        p = Path(f"old_ckpt/{s}.jsonl" if r == "old" else f"out_{r}/{s}.qwen.json")
        if p.exists(): rows[r] = load_run(p)[0]
    rows["whisper"] = whisper_text(f"whisper_ref/{s}.txt")
    for r, text in rows.items():
        g = {k: text.count(k) for k in t["good"] if text.count(k)}
        b = {k: text.count(k) for k in t["bad"] if text.count(k)}
        tot.setdefault(r, [0, 0]); tot[r][0] += sum(g.values()); tot[r][1] += sum(b.values())
        print(f"  {r:8s} good={sum(g.values()):3d} bad={sum(b.values()):3d}  bad:{b}")
print("TOTAL", tot)
