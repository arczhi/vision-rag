#!/usr/bin/env bash
# 端到端入库验证: 删旧数据 → /ingest(use_llm_segments) → 轮询 done
# 一次性跑完 3 个幻觉视频, 输出每个的 kept / num_clips
set -uo pipefail
BASE="http://127.0.0.1:28765"
cd /Users/alex/coding/vision-rag

declare -a IDS=("c6becb280a334117" "db65abded44ab390" "6e3590cfbec35aed")
declare -a FILES=("1779175721591血染茉莉花ep1.mp4" "17701993739072月4日 (1)(1).mp4" "17701938684882月4日(7).mp4")

echo "===== 第1步: 删除旧数据 ====="
for id in "${IDS[@]}"; do
  echo "DELETE $id"
  curl -s -m 60 -X DELETE "$BASE/videos/$id" -w " [HTTP %{http_code}]\n"
done

echo ""
echo "===== 第2步: 重新入库 (use_llm_segments=true) ====="
declare -a TASKS=()
for f in "${FILES[@]}"; do
  echo "INGEST $f"
  resp=$(curl -s -m 900 -X POST "$BASE/ingest" \
    -F "file=@data/videos/$f" \
    -F "skip_existing=false" \
    -F "use_llm_segments=true")
  echo "  -> $resp"
  tid=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('task_id',''))" 2>/dev/null)
  TASKS+=("$tid")
done

echo ""
echo "===== 第3步: 轮询至完成 ====="
for i in "${!TASKS[@]}"; do
  tid="${TASKS[$i]}"
  f="${FILES[$i]}"
  [ -z "$tid" ] && { echo "✗ $f 无 task_id, 跳过"; continue; }
  echo "轮询 $f (task=$tid)"
  for n in $(seq 1 90); do
    j=$(curl -s -m 15 "$BASE/tasks/$tid")
    st=$(echo "$j" | python3 -c "import sys,json;print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
    if [ "$st" = "done" ] || [ "$st" = "failed" ]; then
      echo "$j" | python3 -c "
import sys,json
t=json.load(sys.stdin)
ex=t.get('extra',{}) or {}
ss=ex.get('llm_segment_stats',{}) or {}
rj=t.get('result',{}) or {}
print(f'  [{t.get(\"status\")}] kept={ss.get(\"kept\",\"-\")} oob={ss.get(\"dropped_oob\",\"-\")} max_ts={ss.get(\"max_timestamp\",\"-\")}s num_clips={rj.get(\"num_clips\",\"-\")} llm_err={(ex.get(\"llm_error\") or \"-\")[:40]}')"
      break
    fi
    sleep 10
  done
done
echo ""
echo "===== 完成 ====="
