#!/bin/bash
# x512_test.sh : on the 512K lane-2 boot: wait healthy, record boot facts, sanity, then a ~470K-token needle
# with KV usage sampled every 2 s (the memory watchdog logs MemAvailable separately).
cd ~/glm53big-bench; R=results; L=dcp4-nvfp4-512k
for i in $(seq 1 480); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 localhost:8000/health)" = 200 ] && break
  docker ps --filter name=vllm_glm53big --format '{{.Status}}' | grep -q Up || { echo "CONTAINER DOWN $(date +%T)"; docker logs vllm_glm53big 2>&1 | grep -E "Error|error|Traceback" | tail -8; exit 1; }
  sleep 5
done
echo "healthy $(date +%T)"
docker logs vllm_glm53big 2>&1 | grep -E "GPU KV cache size|Maximum concurrency|Model loading took|init engine" | cut -c40-220 | tee $R/$L-boot.txt
python3 guard.py 1 3 -- python3 sanity.py http://localhost:8000 $L > $R/sanity-$L.json; echo "sanity exit=$?"
( while true; do m=$(curl -s -m 5 localhost:8000/metrics); echo "$(date +%s) $(echo "$m" | awk '/^vllm:kv_cache_usage_perc\{/{print $2}') $(echo "$m" | awk '/^vllm:num_requests_running\{/{print $2}')"; sleep 2; done ) > $R/$L-kvsamples.txt &
SP=$!
python3 guard.py 1 3 -- python3 needle.py http://localhost:8000 480000 $L-needle > $R/$L-needle.json 2> $R/$L-needle.err; echo "needle exit=$? $(date +%T)"
kill $SP
cat $R/$L-needle.json
echo "== x512 DONE $(date +%T)"
