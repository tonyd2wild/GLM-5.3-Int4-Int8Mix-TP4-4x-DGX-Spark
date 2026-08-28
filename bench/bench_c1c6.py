#!/usr/bin/env python3
"""C1-C6 concurrency sweep against the GLM-5.3 Int4-Int8Mix TP4 endpoint.

NOTE: this lane serves with thinking ON (no --reasoning-parser, and
chat_template_kwargs enable_thinking=false does NOT take on this build), so
completion tokens include reasoning traces. Numbers are NOT comparable to
thinking-off benchmarks.
"""
import json
import time
import threading
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "glm-5.3"
MAX_TOKENS = 256

PROMPTS = [
    "Explain in one paragraph how a bicycle stays upright when moving.",
    "Write a Python function that reverses a linked list. Code only.",
    "Summarize the causes of the 1929 stock market crash in one paragraph.",
    "What is the difference between TCP and UDP? Two sentences.",
    "Write a SQL query that finds the second highest salary in a table.",
    "Explain what a hash table is and its average time complexity.",
]


def one(prompt, out, idx):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        dt = time.time() - t0
        ct = d["usage"]["completion_tokens"]
        pt = d["usage"]["prompt_tokens"]
        out[idx] = (dt, ct, pt)
    except Exception as e:
        out[idx] = (time.time() - t0, 0, 0)
        print(f"    stream {idx} FAILED: {e}")


print(f"{'conc':>5} {'wall_s':>8} {'tot_tok':>8} {'agg_tok/s':>10} "
      f"{'per_stream':>11} {'mean_lat_s':>11}")
print("-" * 62)

results = {}
for conc in range(1, 7):
    out = [None] * conc
    threads = [threading.Thread(target=one, args=(PROMPTS[i % len(PROMPTS)], out, i))
               for i in range(conc)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    tot = sum(o[1] for o in out if o)
    lats = [o[0] for o in out if o]
    agg = tot / wall if wall else 0
    per = agg / conc if conc else 0
    mean_lat = sum(lats) / len(lats) if lats else 0
    results[conc] = dict(wall=round(wall, 2), tokens=tot,
                         agg_tok_s=round(agg, 2), per_stream=round(per, 2),
                         mean_latency_s=round(mean_lat, 2))
    print(f"{conc:>5} {wall:>8.2f} {tot:>8} {agg:>10.2f} {per:>11.2f} {mean_lat:>11.2f}")
    time.sleep(3)

print()
print(json.dumps(results, indent=2))
