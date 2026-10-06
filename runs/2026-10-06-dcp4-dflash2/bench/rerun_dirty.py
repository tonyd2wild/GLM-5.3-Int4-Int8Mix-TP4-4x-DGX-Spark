#!/usr/bin/env python3
"""rerun_dirty.py RESULTS.json [TRIES] [QUIET_S] -- re-measure the cells the harness marked dirty (other clients on
the server during the measurement) and patch them in place. Uses the harness's own functions."""
import json, statistics, sys, time
import bench_tp2_night as B

path = sys.argv[1]; tries = int(sys.argv[2]) if len(sys.argv) > 2 else 20
QUIET = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0   # agents send a turn every 5-15 s; 30 s quiet = a real pause
_wait = B.wait_idle
B.wait_idle = lambda quiet=QUIET, limit=7200: _wait(quiet, limit)
d = json.load(open(path))
B.URL = d["url"].rstrip("/"); B.EFFORT = d.get("effort")
reps, rounds = d.get("reps", 3), d.get("rounds", 3)

for name, v in list(d.get("single", {}).items()):
    if not v.get("dirty_reps"):
        continue
    prompt, mt = B.PROMPTS[name]
    runs, pairs = [], []
    for _ in range(reps):
        r, ba, clean = B.clean_run(lambda: B.post_stream(prompt, mt, name), 1, tries)
        if clean and r.get("ok"):
            runs.append(r); pairs.append(ba)
    if len(runs) < reps:
        print(f"  {name}: still dirty after {tries} tries per rep", flush=True); continue
    d["single"][name] = {"decode_tok_s": B.st([r["decode_tok_s"] for r in runs]),
                         "e2e_tok_s": B.st([r["e2e_tok_s"] for r in runs]),
                         "ttft_s": B.st([r["ttft_s"] for r in runs]),
                         "ttft_answer_s": B.st([r["ttft_answer_s"] for r in runs]),
                         "total_s": B.st([r["total_s"] for r in runs]),
                         "think_chars": runs[0]["think_chars"], "answer_chars": runs[0]["answer_chars"],
                         "completion_tokens": runs[0]["completion_tokens"], "prompt_tokens": runs[0]["prompt_tokens"],
                         "spec": B.spec_sum(pairs), "dirty_reps": 0, "rerun": time.strftime("%H:%M:%S")}
    print(f"  {name}: re-run clean, decode median {d['single'][name]['decode_tok_s']['median']}", flush=True)
    json.dump(d, open(path, "w"), indent=2)

for c, v in list(d.get("sweep", {}).items()):
    if not v.get("dirty_rounds"):
        continue
    n = int(c[1:]); aggs, per, ttfts, pairs, fails = [], [], [], [], 0
    for r in range(rounds):
        jobs = [(f"[s{i}] " + B.PROMPTS[B.REAL[(i + r * n) % len(B.REAL)]][0], B.PROMPTS[B.REAL[(i + r * n) % len(B.REAL)]][1],
                 B.REAL[(i + r * n) % len(B.REAL)]) for i in range(n)]
        (wall, out), ba, clean = B.clean_run(lambda: B.wave(jobs), n, tries)
        if not clean:
            break
        ok = [o for o in out if o and o.get("ok")]; fails += n - len(ok); pairs.append(ba)
        if ok:
            aggs.append(sum(o["completion_tokens"] for o in ok) / wall)
            per += [o["decode_tok_s"] for o in ok]; ttfts += [o["ttft_s"] for o in ok]
    if len(pairs) < rounds:
        print(f"  {c}: still dirty", flush=True); continue
    d["sweep"][c] = {"agg_tok_s": B.st(aggs), "per_stream_decode": B.st(per), "ttft_s": B.st(ttfts), "fails": fails,
                     "spec": B.spec_sum(pairs), "dirty_rounds": 0, "rerun": time.strftime("%H:%M:%S"),
                     "preemptions": sum(B.g(a, "vllm:num_preemptions_total") - B.g(b, "vllm:num_preemptions_total") for b, a in pairs)}
    print(f"  {c}: re-run clean, agg median {d['sweep'][c]['agg_tok_s'].get('median')}", flush=True)
    json.dump(d, open(path, "w"), indent=2)

for key in ("prefill", "longctx"):
    for k, v in list(d.get(key, {}).items()):
        if v.get("clean") is False:
            print(f"  {key} {k}: dirty; re-run that suite by hand", flush=True)
print("done", flush=True)
