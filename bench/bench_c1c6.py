#!/usr/bin/env python3
"""C1-C6 concurrency sweep for GLM-5.3-Int4-Int8Mix + DFlash2 on 4x DGX Spark.

Reports END-TO-END tok/s (wall clock from request send to full response), not the
engine's internal decode rate -- per Tony's instruction "tes tok/s on that not decode".

C1 uses the count-to-100 prompt he asked for as the first bench.
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "glm-5.3"
COUNT100 = "Count from 1 to 100. One number per line. Nothing else."


def metrics():
    try:
        with urllib.request.urlopen("http://localhost:8000/metrics", timeout=15) as r:
            txt = r.read().decode()
    except Exception:
        return None
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k, tag in (("spec_decode_num_draft_tokens_total", "draft"),
                       ("spec_decode_num_accepted_tokens_total", "acc")):
            if k in line:
                out[tag] = float(line.rsplit(" ", 1)[1])
    return out


def one(prompt, max_tokens):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    u = d["usage"]
    return {"dt": dt, "out": u["completion_tokens"], "inp": u["prompt_tokens"],
            "text": (d["choices"][0]["message"].get("content") or "")}


def run(conc, prompt, max_tokens, label):
    before = metrics()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda _: one(prompt, max_tokens), range(conc)))
    wall = time.time() - t0
    after = metrics()
    tot = sum(r["out"] for r in res)
    per = [r["out"] / r["dt"] for r in res]
    acc = ""
    if before and after and after.get("draft", 0) > before.get("draft", 0):
        d = after["draft"] - before["draft"]
        a = after["acc"] - before["acc"]
        acc = f"  accept={100*a/d:.1f}%"
    print(f"C{conc}  {label:>12}  wall={wall:6.2f}s  out_tok={tot:5d}  "
          f"AGG={tot/wall:7.2f} tok/s  per-stream={sum(per)/len(per):6.2f} tok/s{acc}",
          flush=True)
    return res, tot / wall


if __name__ == "__main__":
    print("=== C1: count to 100 (Tony's first bench) ===", flush=True)
    res, _ = run(1, COUNT100, 1000, "count100")
    txt = res[0]["text"]
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    ok = sum(1 for i, l in enumerate(lines[:100], 1) if l.rstrip(".") == str(i))
    print(f"     correctness: {ok}/100 lines correct, {len(lines)} lines emitted")
    print(f"     head: {txt[:60]!r}")
    print(f"     tail: {txt[-60:]!r}")
    print(flush=True)

    print("=== C1-C6 sweep (same prompt, 256 tok each) ===", flush=True)
    for c in range(1, 7):
        run(c, "Write a detailed technical explanation of how speculative "
               "decoding accelerates LLM inference.", 256, "sweep")
