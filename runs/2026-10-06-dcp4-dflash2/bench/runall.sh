#!/bin/bash
# runall.sh LABEL : full test chain against :8000, interference-guarded; results in ~/glm53big-bench/results
L=$1; cd ~/glm53big-bench; R=results; mkdir -p $R
echo "== $L start $(date +%T)"
python3 bench_tp2_night.py --label $L --suite all --effort low --prefill-sizes 1750,8700,30720 --longctx-n 32768 --longctx-levels 1 --outdir $R > $R/$L-bench.log 2>&1; echo "bench exit=$? $(date +%T)"
python3 guard.py 1 3 -- python3 needle.py http://localhost:8000 250000 $L-needle250k > $R/$L-needle250k.json 2> $R/$L-needle250k.err; echo "needle exit=$? $(date +%T)"
python3 guard.py 1 3 -- python3 kvtest.py 1 200000 $L-kv1-200k 1.5 > $R/$L-kvtest.log 2>&1; mv ~/kvtest-$L-kv1-200k.json $R/ 2>/dev/null; echo "kvtest exit=$? $(date +%T)"
echo "== $L DONE $(date +%T)"
