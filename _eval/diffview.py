# 두 실행의 단어열 차이를 시각과 함께 보여 주고, 청크 경계(±3초) 근처인지 표시
import sys, difflib, json
sys.path.insert(0, ".")
from compare import load_run, norm, whisper_text
a_path, b_path, wpath = sys.argv[1:4]
ta, wa, ra = load_run(a_path); tb, wb, rb = load_run(b_path)
ba = [r["start"] for r in ra][1:]; bb = [r["start"] for r in rb][1:]
near = lambda t, B: any(abs(t - x) < 3 for x in B)
A = [norm(w["text"]) for w in wa]; B = [norm(w["text"]) for w in wb]
sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
n = nb_a = nb_b = 0
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op == "equal": continue
    t = wa[i1]["start"] if i1 < len(wa) else wb[j1]["start"]
    n += 1; xa = near(t, ba); xb = near(t, bb); nb_a += xa; nb_b += xb
    if len(sys.argv) > 4:
        print(f"{t:7.1f}s {'A경계' if xa else '    '} {'B경계' if xb else '    '} | A: {' '.join(w['text'] for w in wa[i1:i2])[:80]} | B: {' '.join(w['text'] for w in wb[j1:j2])[:80]}")
print(f"diff blocks {n}, near A boundary {nb_a}, near B boundary {nb_b}")
