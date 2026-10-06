#!/usr/bin/env python3
"""build_readme.py TEMPLATE OUT  -- fill readme_new_section.md markers from the raw results in results/."""
import json, re, sys

R = "results/"
L1 = json.load(open(R + "dcp4-fp8.json"))
L2 = json.load(open(R + "dcp4-nvfp4.json"))
X = json.load(open(R + "x512.json"))           # written after the 512K push
N1 = json.load(open(R + "dcp4-fp8-needle250k.json"))
N2 = json.load(open(R + "dcp4-nvfp4-needle250k.json"))
K2 = json.load(open(R + "kvtest-dcp4-nvfp4-kv1-200k.json"))
# lane 1's correctly sized prefill and 115K decode come from its follow-up log (dcp4-fp8-supp.log)
L1_SUPP = {"prefill": [(5019, 395), (31897, 381)], "lc_decode": 16.96, "lc_ttft": 313.2}

ORDER = [("count100", "count to 100 *"), ("count300", "count to 300 *"), ("tooluse", "tool call"), ("code", "code"),
         ("json", "json"), ("math", "math"), ("sql", "sql"), ("summary", "summary"), ("prose", "prose"),
         ("narrative", "narrative")]


def s(d, k, f, stat="median"):
    return d["single"][k][f][stat] if isinstance(d["single"][k][f], dict) else d["single"][k][f]


def cap(kv):
    one = [x for x in kv["samples"] if x.get("num_requests_running") == 1]
    pk = max(x["kv_cache_usage_perc"] for x in one)
    toks = kv["results"]["0"]["usage"]["total_tokens"] if "0" in kv["results"] else list(kv["results"].values())[0]["usage"]["total_tokens"]
    return toks, pk, int(toks / pk)


rows = ["| prompt | **Lane 2** decode tok/s | Lane 2 end to end | Lane 1 decode | Lane 1 end to end | Lane 2 TTFT / time to answer | Lane 2 accept, mean len |",
        "|---|---|---|---|---|---|---|"]
for k, name in ORDER:
    ttft, ans = s(L2, k, "ttft_s"), s(L2, k, "ttft_answer_s")
    tt = f"{ttft:.2f} s" if abs(ans - ttft) < 0.05 else f"{ttft:.2f} / {ans:.2f} s"
    sp = L2["single"][k]["spec"]
    rows.append(f"| {name} | {s(L2, k, 'decode_tok_s'):.1f} | {s(L2, k, 'e2e_tok_s'):.1f} | {s(L1, k, 'decode_tok_s'):.1f} | "
                f"{s(L1, k, 'e2e_tok_s'):.1f} | {tt} | {100 * sp['accept_ratio']:.0f}%, {sp['mean_accept_len']:.2f} |")
single = "\n".join(rows) + ("\n\nSingle stream, temperature 0, `reasoning_effort: low`, median of 3. \\* Counting prompts show the "
                            "speculative-decoding ceiling (draft acceptance ~97%), not a typical rate; prose-like text accepts "
                            "~20% of drafted tokens. The tool call is 34 tokens, so its end-to-end rate is mostly time to first token. "
                            "**The two lanes decode at the same speed:** they run the same kernels, and per-prompt gaps of up to "
                            "~10% either way track the draft acceptance of that run, which varies run to run on this stack. "
                            "What separates them is the KV pool.")

rows = ["| streams | **Lane 2** aggregate, median / peak | Lane 1 aggregate, median / peak | Lane 2 per-stream decode | Lane 2 TTFT p90 |",
        "|---|---|---|---|---|"]
for c in sorted(L2["sweep"], key=lambda c: int(c[1:])):
    a2, a1 = L2["sweep"][c], L1["sweep"][c]
    rows.append(f"| {c.upper()} | {a2['agg_tok_s']['median']:.1f} / {a2['agg_tok_s']['peak']:.1f} | "
                f"{a1['agg_tok_s']['median']:.1f} / {a1['agg_tok_s']['peak']:.1f} | {a2['per_stream_decode']['median']:.1f} | "
                f"{a2['ttft_s']['p90']:.2f} s |")
sweep = "\n".join(rows) + ("\n\nMixed real prompts (code, json, sql, tool call, math, prose, narrative, summary rotated across streams), "
                           "3 rounds per level, 0 failures, 0 preemptions. A round's aggregate is bound by its slowest stream "
                           "(a 700-token prose answer at ~15 tok/s), so median and peak differ with the mix.")

pf2 = [(v["prompt_tokens"], v["prefill_tok_s"], v["ttft_s"]) for v in L2["prefill"].values() if v.get("prompt_tokens")]
lc2 = L2["longctx"]["c1"]
toks2, pk2, cap2 = cap(K2)
lines = ["| | **Lane 2** (NVFP4) | Lane 1 (fp8) |", "|---|---|---|"]
l1pf = dict(L1_SUPP["prefill"]); l1pf.update({21745: 397, 106130: 364})
for (t, r, tt) in pf2:
    near = min(l1pf, key=lambda x: abs(x - t))
    lines.append(f"| cold prefill, {t:,} tokens | {r:.0f} tok/s (TTFT {tt:.1f} s) | {l1pf[near]} tok/s at {near:,} tokens |")
lines.append(f"| decode at a ~115K-token context, 1 stream | {lc2['per_stream_decode']:.1f} tok/s (TTFT {lc2['ttft_s']:.0f} s) | "
             f"{L1_SUPP['lc_decode']:.1f} tok/s (TTFT {L1_SUPP['lc_ttft']:.0f} s) |")
lines.append(f"| needle test, 3 codes at 10/50/90% depth | **{N2['needles_found']}** at {N2['prompt_tokens']:,} tokens "
             f"(TTFT {N2['ttft_s']:.0f} s) | {N1['needles_found']} at {N1['prompt_tokens']:,} tokens (TTFT {N1['ttft_s']:.0f} s) |")
lines.append(f"| needle test at the 512K window | **{X['needle']['needles_found']}** at {X['needle']['prompt_tokens']:,} tokens "
             f"(TTFT {X['needle']['ttft_s']:.0f} s) | - |")
lines.append(f"| one {toks2:,}-token request holds | {100 * pk2:.1f}% of the pool | 47.2% of the pool (204,901 tokens) |")
lines.append(f"| lowest free memory, any node | {X['mem_min_gb_262k']} GB (262K) / {X['mem_min_gb']} GB (512K) | 3.1 GB |")
longctx = "\n".join(lines)

t = open(sys.argv[1]).read()
rep = {
    "<!--TABLE:single-->": single, "<!--TABLE:sweep-->": sweep, "<!--TABLE:longctx-->": longctx,
    "<!--ALT:suite-->": "; ".join(f"{n} {s(L2, k, 'decode_tok_s'):.1f} vs {s(L1, k, 'decode_tok_s'):.1f}" for k, n in ORDER),
    "<!--ALT:sweep-->": "; ".join(f"{c.upper()} {L2['sweep'][c]['agg_tok_s']['peak']:.1f} vs {L1['sweep'][c]['agg_tok_s']['peak']:.1f}"
                                  for c in sorted(L2['sweep'], key=lambda c: int(c[1:]))),
    "<!--L2:measured-->": f"~{round(cap2, -3):,}",
    "<!--L2:c100e2e-->": f"{s(L2, 'count100', 'e2e_tok_s'):.1f}",
    "<!--L2:kvfinding-->": f"On lane 2 a {toks2:,}-token request held {100 * pk2:.1f}%: ~{round(cap2, -3):,} against the reported 689,772 "
                           f"({100 * (689772 / cap2 - 1):+.1f}%).",
    "<!--L2X:ctx-->": f"{X['maxlen']:,}", "<!--L2X:pool-->": f"{X['pool']:,} ({X['pool'] / X['maxlen']:.2f}×)",
    "<!--L2X:measured-->": X.get("measured_str", "not measured"), "<!--L2X:kv-->": f"{X['kvbytes'] / 1e9:.1f} GB",
    "<!--L2X:maxlen-->": str(X["maxlen"]), "<!--L2X:kvbytes-->": str(X["kvbytes"]),
}
for k, v in rep.items():
    t = t.replace(k, v)
left = re.findall(r"<!--[A-Z0-9:a-z]+-->", t)
if left:
    sys.exit(f"unfilled markers: {left}")
open(sys.argv[2], "w").write(t)
print("wrote", sys.argv[2])
