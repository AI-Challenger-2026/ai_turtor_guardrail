#!/usr/bin/env bash
# 튜터 설정 A/B 비교: 설정마다 별도 포트에 서버를 띄우고 같은 범주를 RUNS번 진단해 요약만 출력한다.
#   scripts/ab_tutor.sh "이름=ENV1=a,ENV2=b" "이름2=ENV=c" ...
# 환경변수: CATS(기본 내용 답변 범주), RUNS(기본 2)
set -u
cd "$(dirname "$0")/.."
CATS=${CATS:-normal,pii_disclosure,answer_seeking,out_of_material}
RUNS=${RUNS:-2}
OUT=${OUT:-/tmp/ab_tutor}
mkdir -p "$OUT"
port=8011
pids=()
names=()
for spec in "$@"; do
  name=${spec%%=*}; envs=${spec#*=}
  [ "$envs" = "$spec" ] && envs=""
  ( IFS=','; for kv in $envs; do export "$kv"; done
    LOG_FILE="$OUT/$name.server.log" exec python3 -m uvicorn server:app --host 127.0.0.1 --port $port --log-level warning ) &
  pids+=($!); names+=("$name:$port"); port=$((port+1))
done
for np in "${names[@]}"; do p=${np#*:}; for i in $(seq 1 60); do curl -s -m 2 -o /dev/null localhost:$p/ && break; sleep 0.5; done; done
jobs_=()
for np in "${names[@]}"; do
  n=${np%%:*}; p=${np#*:}
  ( for r in $(seq 1 "$RUNS"); do
      python3 scripts/diagnose_tutor.py --url http://localhost:$p/tutor --category "$CATS" --show none > "$OUT/$n.run$r.txt" 2>&1
    done ) &
  jobs_+=($!)
done
wait "${jobs_[@]}"
kill "${pids[@]}" 2>/dev/null
for np in "${names[@]}"; do
  n=${np%%:*}
  echo "===== $n"
  for r in $(seq 1 "$RUNS"); do grep -E "^(normal|out_of|pii|answer_seek)|튜터 전체" "$OUT/$n.run$r.txt" | sed "s/^/  run$r /"; done
  python3 - "$OUT/$n.server.log" <<'EOF'
import re,sys
L=open(sys.argv[1]).read().splitlines()
ms=sorted(int(re.search(r'ms=(\d+)',l).group(1)) for l in L if ' tutor ' in l)
err=sum('error' in l for l in L)
if ms: print(f"  지연 p50 {ms[len(ms)//2]/1000:.1f}s p95 {ms[int(len(ms)*.95)]/1000:.1f}s max {ms[-1]/1000:.1f}s, 오류/대체 {err}")
EOF
done
