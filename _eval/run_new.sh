#!/bin/bash
# usage: run_new.sh <name> [extra args...]   (출력은 _eval/out_<name>/ 으로 이동)
# Python은 QWEN_PY 로 지정 (기본: 현재 PATH의 python = 활성화된 qwen_asr 환경)
#   예) QWEN_PY=~/.conda/envs/qwen_asr/python.exe ./run_new.sh n60_b8 --batch-size 8
set -u
cd "$(dirname "$0")"
name=$1; shift
PY=${QWEN_PY:-python}
mkdir -p out_$name
start=$(date +%s)
PYTHONIOENCODING=utf-8 $PY ../run_qwen_forced_align.new.py media/*.mp4 media/*.m4a --overwrite --save-json "$@" > out_$name/run.log 2>&1
rc=$?
echo "rc=$rc wall=$(( $(date +%s) - start ))s" >> out_$name/run.log
mv media/*.qwen.txt media/*.qwen.json out_$name/ 2>/dev/null
tail -c 300 out_$name/run.log | tr '\r' '\n' | tail -3
grep -E "완료|재분할|경고|확인 필요|오류|메모리" out_$name/run.log | tr '\r' '\n' | grep -E "완료|재분할|경고|확인 필요|오류|메모리" | head -40
