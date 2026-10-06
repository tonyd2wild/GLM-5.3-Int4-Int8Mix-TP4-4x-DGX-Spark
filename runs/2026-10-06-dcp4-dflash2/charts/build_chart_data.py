#!/usr/bin/env python3
"""build_chart_data.py OUT.json LABEL1=results1.json [LABEL2=results2.json]
Turns speed-night harness results into make_charts.py input. One lane gives single-series charts;
two lanes give side-by-side charts (first = blue, second = orange). Cells marked dirty (agent traffic
during the measurement) are refused: re-run them before charting."""
import json, sys

ORDER = [("count100", "count to 100 *"), ("count300", "count to 300 *"), ("tooluse", "tool call"),
         ("math", "math"), ("code", "code"), ("json", "json"), ("sql", "sql"),
         ("summary", "summary"), ("prose", "prose"), ("narrative", "narrative")]

out_path = sys.argv[1]
lanes = []
for arg in sys.argv[2:]:
    label, path = arg.split("=", 1)
    lanes.append((label, json.load(open(path))))

for label, d in lanes:
    dirty = [k for k, v in d.get("single", {}).items() if v.get("dirty_reps")]
    dirty += [k for k, v in d.get("sweep", {}).items() if v.get("dirty_rounds")]
    dirty += [k for k, v in d.get("prefill", {}).items() if v.get("clean") is False]
    dirty += [k for k, v in d.get("longctx", {}).items() if v.get("clean") is False]
    if dirty:
        sys.exit(f"{label}: dirty cells {dirty}; re-run them first")

charts = {}
cats = [name for key, name in ORDER]
def single(d, key):
    v = d["single"].get(key, {})
    return (v.get("decode_tok_s") or {}).get("median")

if len(lanes) == 1:
    label, d = lanes[0]
    charts["suite"] = {"kind": "suite", "title": "GLM-5.3 743B on 4x DGX Spark: decode speed by prompt type (tok/s, single stream)",
                       "sub": f"{label}, temperature 0, reasoning_effort low, median of 3",
                       "cats": cats, "vals": [single(d, k) or 0 for k, _ in ORDER], "note": ""}
else:
    charts["suite"] = {"kind": "suite2", "title": "GLM-5.3 743B on 4x DGX Spark: decode speed by prompt type (tok/s, single stream)",
                       "sub": "temperature 0, reasoning_effort low, median of 3",
                       "cats": cats, "series": [[label, [single(d, k) for k, _ in ORDER]] for label, d in lanes], "note": ""}

levels = sorted(lanes[0][1].get("sweep", {}), key=lambda c: int(c[1:]))
if levels:
    charts["sweep"] = {"kind": "sweep", "title": "Aggregate throughput, mixed real prompts (tok/s)",
                       "sub": "8 prompt types rotated across streams (no counting), peak of 3 rounds (table has median and peak)",
                       "x": [c.upper() for c in levels],
                       "series": [[label, [d["sweep"][c]["agg_tok_s"]["peak"] for c in levels]] for label, d in lanes],
                       "note": ""}

sizes = list(lanes[0][1].get("prefill", {}))
if sizes and len(lanes) == 2 and all(k in lanes[1][1].get("prefill", {}) for k in sizes):
    (l1, d1), (l2, d2) = lanes
    new = [d1["prefill"][s]["prefill_tok_s"] for s in sizes]; old = [d2["prefill"][s]["prefill_tok_s"] for s in sizes]
    charts["prefill"] = {"kind": "prefill", "title": "Cold prefill (tok/s)",
                         "cats": [f"{d1['prefill'][s]['prompt_tokens']:,} tokens" for s in sizes],
                         "new_label": l1, "old_label": l2, "new": new, "old": old,
                         "delta": [f"{(n / o - 1) * 100:+.0f}%" for n, o in zip(new, old)], "note": ""}

json.dump(charts, open(out_path, "w"), indent=1)
print("wrote", out_path, list(charts))
