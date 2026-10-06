#!/usr/bin/env python3
"""bench_tp2_night.py -- speed-night harness for GLM-5.3-Flash TP2 (2x DGX Spark).

Stdlib only, so it runs on the head node itself (no Mac/tailnet jitter, survives the
laptop sleeping). Temperature 0 everywhere: a config delta is a config delta, not a
content delta.

Suites
  single   every prompt, single stream, --reps repeats -> median / p90 / peak decode,
           TTFT, e2e; spec acceptance per prompt (ratio AND mean accepted length)
  sweep    C1..C6 aggregate. MIXED real prompts (no counting): stream i in round r gets
           REAL[(i + r) % len(REAL)], so every level sees the same mix. Aggregate =
           output tokens / wall. Also per-stream decode median and TTFT p90.
  prefill  COLD prefill: unique salt at the front of every prompt, sizes 2K..128K.
  longctx  issue #14: 32K prompts at C1 vs C2 (the collapse short sweeps can't see).
  meta     KV pool (num_gpu_blocks * block_size from /metrics), preemptions delta.

Usage (on the head node):
  python3 bench_tp2_night.py --url http://127.0.0.1:8000 --label A-00-baseline --suite all
Writes <outdir>/<label>.json and prints a compact table.
"""
import argparse, json, os, re, statistics, threading, time, urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "glm-5.3-flash"
EFFORT = "low"   # chat_template_kwargs.reasoning_effort; None = server default (knapcio stack defaults to high)

COUNT_NUMS = lambda n: f"Count from 1 to {n}. Output only the numbers, one per line, nothing else."

PROMPTS = {
    # draft-acceptance CEILING probes (label them as such, never the headline)
    "count100": (COUNT_NUMS(100), 400),
    "count300": (COUNT_NUMS(300), 1300),
    # real work
    "code": ("Write a complete Python implementation of an LRU cache with get and put in "
             "O(1), using a dict plus a doubly linked list. Include the class, full method "
             "bodies, and a short docstring for each method.", 700),
    "json": ("Return a JSON array of 12 fictional employees. Each object must have exactly "
             "these keys: id (int), name (string), department (one of Engineering, Sales, "
             "Support, Finance), salary (int), start_date (YYYY-MM-DD), skills (array of 3 "
             "strings). Output only the JSON, no prose, no code fences.", 900),
    "sql": ("Given tables orders(id, customer_id, total, created_at) and customers(id, name, "
            "region), write a PostgreSQL query that returns, for each region, the top 3 "
            "customers by total spend in 2025 with their rank, using a window function. Then "
            "explain each clause of the query in one sentence.", 600),
    "tooluse": ("What's the weather in Tokyo, Paris and New York right now? Use the tool for "
                "each city.", 300),
    "math": ("A train leaves city A at 9:00 traveling 80 km/h toward city B, 360 km away. A "
             "second train leaves B at 10:00 traveling 100 km/h toward A. Solve step by step: "
             "at what time and how far from A do they meet? Show all arithmetic.", 600),
    "prose": ("Explain, in flowing prose with no code, no lists and no headings, how a "
              "modern CPU's branch predictor works and why mispredictions are expensive on "
              "a deeply pipelined machine.", 700),
    "narrative": ("Write an original short story of about 500 words about a lighthouse keeper "
                  "who discovers the lamp has been signalling to something out at sea. Use "
                  "vivid sensory detail and dialogue.", 800),
    "summary": ("Summarize the causes, key events and consequences of the 1929 stock market "
                "crash in five paragraphs for a high-school audience.", 700),
}
REAL = ["code", "json", "sql", "tooluse", "math", "prose", "narrative", "summary"]

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def post_stream(prompt, max_tokens, name=None, timeout=1800):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    if EFFORT:
        body["chat_template_kwargs"] = {"reasoning_effort": EFFORT}
    if name == "tooluse":
        body["tools"] = TOOLS
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; tans = None; comp = ptoks = None; think_chars = ans_chars = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                ch = ev.get("choices") or []
                if ch:
                    d = ch[0].get("delta") or {}
                    # this vLLM streams thinking as "reasoning"; older builds used "reasoning_content"
                    rz = d.get("reasoning") or d.get("reasoning_content")
                    ans = d.get("content") or d.get("tool_calls")
                    if rz: think_chars += len(rz)
                    if d.get("content"): ans_chars += len(d["content"])
                    if ttft is None and (rz or ans):
                        ttft = time.perf_counter()
                    if tans is None and ans:
                        tans = time.perf_counter()
                u = ev.get("usage")
                if u:
                    comp = u.get("completion_tokens"); ptoks = u.get("prompt_tokens")
        t1 = time.perf_counter()
    except Exception as e:
        return {"ok": False, "err": str(e)[:160]}
    if not comp:
        return {"ok": False, "err": "no usage"}
    ttft = ttft or t1; tans = tans or t1
    return {"ok": True, "ttft_s": ttft - t0, "ttft_answer_s": tans - t0, "total_s": t1 - t0,
            "think_chars": think_chars, "answer_chars": ans_chars, "prompt_tokens": ptoks,
            "completion_tokens": comp,
            "prefill_tok_s": ptoks / (ttft - t0) if ptoks and ttft > t0 else None,
            "decode_tok_s": (comp - 1) / max(t1 - ttft, 1e-9) if comp > 1 else None,
            "e2e_tok_s": comp / (t1 - t0)}


def metrics():
    out = {}
    try:
        txt = urllib.request.urlopen(URL + "/metrics", timeout=15).read().decode()
    except Exception:
        return out
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        p = line.rsplit(" ", 1)
        if len(p) == 2:
            try:
                out[p[0]] = float(p[1])
            except ValueError:
                pass
    return out


def g(d, prefix):
    return sum(v for k, v in d.items() if k.startswith(prefix))


def spec_delta(b, a):
    dd = g(a, "vllm:spec_decode_num_draft_tokens_total") - g(b, "vllm:spec_decode_num_draft_tokens_total")
    da = g(a, "vllm:spec_decode_num_accepted_tokens_total") - g(b, "vllm:spec_decode_num_accepted_tokens_total")
    dn = g(a, "vllm:spec_decode_num_drafts_total") - g(b, "vllm:spec_decode_num_drafts_total")
    r = {}
    if dd > 0: r["accept_ratio"] = round(da / dd, 4)
    if dn > 0: r["mean_accept_len"] = round(1 + da / dn, 3)
    pos = {}
    for k, v in a.items():
        if "accepted_tokens_per_pos" in k:
            m = re.search(r'position="(\d+)"', k)
            dv = v - b.get(k, 0.0)
            if m and dn > 0:
                pos[int(m.group(1))] = round(dv / dn, 3)
    if pos: r["per_pos"] = [pos[i] for i in sorted(pos)]
    return r



# --- interference guard (2026-10-06): the endpoint is shared with live agents. Every measurement
# waits for an idle server, samples running+waiting during the run, and is retried if the server
# saw more requests than the measurement itself sent. Spec counters are summed over clean attempts.
_MON = {"max": 0, "lock": threading.Lock(), "on": False}


def _busy_now():
    d = metrics()
    if not d:
        return None
    return g(d, "vllm:num_requests_running{") + g(d, "vllm:num_requests_waiting{")


def _mon_loop():
    while True:
        b = _busy_now()
        if b is not None:
            with _MON["lock"]:
                _MON["max"] = max(_MON["max"], b)
        time.sleep(0.5)


def _mon_start():
    if not _MON["on"]:
        _MON["on"] = True
        threading.Thread(target=_mon_loop, daemon=True).start()


def wait_idle(quiet=5.0, limit=5400):
    t0 = time.time(); since = None
    while time.time() - t0 < limit:
        b = _busy_now()
        if b == 0:
            since = since or time.time()
            if time.time() - since >= quiet:
                return time.time() - t0
        else:
            since = None
        time.sleep(0.5)
    return None


INTERFERENCE = {"retries": 0, "waited_s": 0.0, "dirty_kept": 0}


def clean_run(fn, own, tries=5):
    """Run one measurement that sends `own` concurrent requests; retry while the server saw more."""
    _mon_start()
    for t in range(tries):
        w = wait_idle()
        INTERFERENCE["waited_s"] += w or 0
        with _MON["lock"]:
            _MON["max"] = 0
        b = metrics(); r = fn(); a = metrics()
        time.sleep(0.6)
        with _MON["lock"]:
            seen = _MON["max"]
        if seen <= own:
            return r, (b, a), True
        INTERFERENCE["retries"] += 1
        print(f"   (interference: server saw {seen} requests > {own}; retry {t + 1})", flush=True)
    INTERFERENCE["dirty_kept"] += 1
    return r, (b, a), False


def spec_sum(pairs):
    agg_b, agg_a = {}, {}
    for b, a in pairs:
        for k, v in a.items():
            if k.startswith("vllm:spec_decode"):
                agg_a[k] = agg_a.get(k, 0.0) + v - b.get(k, 0.0)
                agg_b.setdefault(k, 0.0)
    return spec_delta(agg_b, agg_a)


def st(vals):
    s = sorted(v for v in vals if v is not None)
    if not s: return {}
    return {"median": round(statistics.median(s), 2), "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 2),
            "peak": round(max(s), 2), "min": round(min(s), 2), "n": len(s)}


def suite_single(reps):
    res = {}
    for name, (prompt, mt) in PROMPTS.items():
        runs = []; pairs = []; dirty = 0
        for _ in range(reps):
            r, ba, clean = clean_run(lambda: post_stream(prompt, mt, name), 1)
            dirty += 0 if clean else 1
            if r.get("ok"): runs.append(r); pairs.append(ba)
        if not runs:
            res[name] = {"error": "all failed"}; print(f"  {name:10s} FAILED", flush=True); continue
        res[name] = {"decode_tok_s": st([r["decode_tok_s"] for r in runs]),
                     "e2e_tok_s": st([r["e2e_tok_s"] for r in runs]),
                     "ttft_s": st([r["ttft_s"] for r in runs]),
                     "ttft_answer_s": st([r["ttft_answer_s"] for r in runs]),
                     "total_s": st([r["total_s"] for r in runs]),
                     "think_chars": runs[0]["think_chars"], "answer_chars": runs[0]["answer_chars"],
                     "completion_tokens": runs[0]["completion_tokens"],
                     "prompt_tokens": runs[0]["prompt_tokens"], "spec": spec_sum(pairs), "dirty_reps": dirty}
        s = res[name]["spec"]
        print(f"  {name:10s} dec med {res[name]['decode_tok_s']['median']:7.2f} peak "
              f"{res[name]['decode_tok_s']['peak']:7.2f}  e2e {res[name]['e2e_tok_s']['median']:7.2f}  "
              f"ttft {res[name]['ttft_s']['median']:.3f}s  answer@ {res[name]['ttft_answer_s']['median']:.2f}s  "
              f"think {runs[0]['think_chars']}ch  toks {runs[0]['completion_tokens']}  "
              f"acc {s.get('accept_ratio')} len {s.get('mean_accept_len')}", flush=True)
    return res


def wave(jobs):
    out = [None] * len(jobs)
    def go(i, p, mt, n): out[i] = post_stream(p, mt, n)
    ths = [threading.Thread(target=go, args=(i, *j)) for i, j in enumerate(jobs)]
    t0 = time.perf_counter(); [t.start() for t in ths]; [t.join() for t in ths]
    return time.perf_counter() - t0, out


def suite_sweep(levels, rounds):
    res = {}
    for c in levels:
        # throwaway wave first: first-time shapes at each level JIT Triton/TileLang kernels
        # mid-inference (18 s TTFT stalls seen at C2 on 2026-09-18) and would poison the median
        wave([(f"[warm{c}-{i}] " + PROMPTS[REAL[i % len(REAL)]][0], 96, REAL[i % len(REAL)]) for i in range(c)])
        aggs, per, ttfts, fails, pairs, dirty = [], [], [], 0, [], 0
        for r in range(rounds):
            jobs = []
            for i in range(c):
                n = REAL[(i + r * c) % len(REAL)]
                p, mt = PROMPTS[n]
                jobs.append((f"[s{i}] " + p, mt, n))
            (wall, out), ba, clean = clean_run(lambda: wave(jobs), c)
            pairs.append(ba); dirty += 0 if clean else 1
            ok = [o for o in out if o and o.get("ok")]
            fails += c - len(ok)
            if ok:
                aggs.append(sum(o["completion_tokens"] for o in ok) / wall)
                per += [o["decode_tok_s"] for o in ok]; ttfts += [o["ttft_s"] for o in ok]
        res[f"c{c}"] = {"agg_tok_s": st(aggs), "per_stream_decode": st(per), "ttft_s": st(ttfts),
                        "fails": fails, "spec": spec_sum(pairs), "dirty_rounds": dirty,
                        "preemptions": sum(g(a, "vllm:num_preemptions_total") - g(b, "vllm:num_preemptions_total") for b, a in pairs)}
        x = res[f"c{c}"]
        print(f"  C{c} agg med {x['agg_tok_s'].get('median')} peak {x['agg_tok_s'].get('peak')}  "
              f"per-stream {x['per_stream_decode'].get('median')}  ttft p90 {x['ttft_s'].get('p90')}s  "
              f"fails {fails}  preempt {x['preemptions']}  acc {x['spec'].get('accept_ratio')}", flush=True)
    return res


def filler(n, salt):
    # ~1 token per item; salt FIRST so no prefix can be reused across runs
    return f"[{salt}] " + " ".join(f"w{i}" for i in range(n))


def suite_prefill(sizes):
    res = {}
    for n in sizes:
        salt = f"{time.time_ns()}"
        r, _, clean = clean_run(lambda: post_stream(f"{filler(n, f'{salt}-{time.time_ns()}')}\n\nReply with exactly: DONE", 8), 1)
        r["clean"] = clean
        if r.get("ok"):
            res[str(n)] = {"prompt_tokens": r["prompt_tokens"], "ttft_s": round(r["ttft_s"], 3),
                           "prefill_tok_s": round(r["prefill_tok_s"] or 0, 1), "clean": r["clean"]}
            print(f"  prefill ~{n:6d}: {r['prompt_tokens']} tok  ttft {r['ttft_s']:.2f}s  "
                  f"{r['prefill_tok_s']:.0f} tok/s", flush=True)
        else:
            res[str(n)] = {"error": r.get("err")}; print(f"  prefill ~{n}: FAILED {r.get('err')}", flush=True)
    return res


def suite_longctx(n=32768, levels=(1, 2)):
    res = {}
    for c in levels:
        mk = lambda: [(f"{filler(n, f'{time.time_ns()}-{i}')}\n\nSummarize what kind of data this is "
                       "in about 150 words.", 250, None) for i in range(c)]
        (wall, out), _, clean = clean_run(lambda: wave(mk()), c)
        ok = [o for o in out if o and o.get("ok")]
        res[f"c{c}"] = {"agg_tok_s": round(sum(o["completion_tokens"] for o in ok) / wall, 2) if ok else None,
                        "per_stream_decode": round(statistics.median([o["decode_tok_s"] for o in ok]), 2) if ok else None,
                        "ttft_s": round(max(o["ttft_s"] for o in ok), 2) if ok else None,
                        "wall_s": round(wall, 1), "fails": c - len(ok), "clean": clean}
        print(f"  longctx C{c}: {res[f'c{c}']}", flush=True)
    return res


def suite_meta():
    m = metrics(); r = {}
    for k in m:
        if k.startswith("vllm:cache_config_info"):
            nb = re.search(r'num_gpu_blocks="(\d+)"', k); bs = re.search(r'block_size="(\d+)"', k)
            if nb and bs:
                r = {"num_gpu_blocks": int(nb.group(1)), "block_size": int(bs.group(1)),
                     "kv_pool_tokens": int(nb.group(1)) * int(bs.group(1))}
    r["preemptions_total"] = g(m, "vllm:num_preemptions_total")
    print(f"  meta: {r}", flush=True)
    return r


def main():
    global URL, EFFORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", default="low", help="reasoning_effort sent to the server; 'default' = omit")
    ap.add_argument("--url", default=os.environ.get("BENCH_URL", URL))
    ap.add_argument("--label", required=True)
    ap.add_argument("--suite", default="single,sweep")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--levels", default="1,2,3,4,5,6")
    ap.add_argument("--prefill-sizes", default="2048,8192,32768,131072")
    ap.add_argument("--longctx-n", type=int, default=32768, help="filler items for the long-context decode test")
    ap.add_argument("--longctx-levels", default="1,2", help="concurrency levels for the long-context decode test")
    ap.add_argument("--outdir", default=os.path.expanduser("~/speednight/results"))
    a = ap.parse_args(); URL = a.url.rstrip("/"); EFFORT = None if a.effort == "default" else a.effort
    suites = {s.strip() for s in a.suite.split(",")}
    if "all" in suites: suites = {"single", "sweep", "prefill", "longctx"}
    os.makedirs(a.outdir, exist_ok=True)
    out = {"label": a.label, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "url": URL, "effort": EFFORT,
           "reps": a.reps, "rounds": a.rounds}
    print(f"== {a.label} @ {URL} ==", flush=True)
    # warm-up per issue #21: cold TileLang/CuTe JIT mid-burst -> RPC timeout. 4 concurrent
    # decodes + one 4K prefill + a count burst before anything is measured.
    tw = time.perf_counter()
    wave([(f"[warm{i}] " + PROMPTS["code"][0], 64, "code") for i in range(4)])
    post_stream(filler(4096, "warm") + "\n\nReply with exactly: DONE", 8)
    post_stream(COUNT_NUMS(40), 120)
    print(f"  warm-up {time.perf_counter()-tw:.1f}s", flush=True)
    out["meta"] = suite_meta()
    if "single" in suites: print(" [single]", flush=True); out["single"] = suite_single(a.reps)
    if "sweep" in suites:
        print(" [sweep]", flush=True); out["sweep"] = suite_sweep([int(x) for x in a.levels.split(",")], a.rounds)
    if "prefill" in suites:
        print(" [prefill]", flush=True); out["prefill"] = suite_prefill([int(x) for x in a.prefill_sizes.split(",")])
    if "longctx" in suites: print(" [longctx]", flush=True); out["longctx"] = suite_longctx(n=a.longctx_n, levels=tuple(int(x) for x in a.longctx_levels.split(",")))
    out["meta_end"] = suite_meta()
    out["interference"] = INTERFERENCE
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = os.path.join(a.outdir, f"{a.label}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"  -> {p}", flush=True)


if __name__ == "__main__":
    main()
