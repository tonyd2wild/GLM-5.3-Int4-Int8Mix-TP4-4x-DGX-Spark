#!/usr/bin/env python3
"""tf_score.py BASE REF_GATE.json LABEL OUT.json   -- teacher-forced scoring.
For every prompt in a gate.py reference file, feed [chat-templated prompt + the reference completion] as token
ids to /v1/completions with prompt_logprobs and record the log-probability of every actual token. Sampling,
speculative decoding and prefix caching play no part, so two lanes can be compared token by token.
tf_score.py --compare A.json B.json [C.json]   -- per-prompt |delta logprob| statistics of B (and C) against A."""
import json, sys, urllib.request

PROMPT_TEXT = None


def post(base, path, body):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def score(base, ref_path, label, out):
    sys.path.insert(0, ".")
    import gate
    ref = json.load(open(ref_path))
    res = {"label": label, "ref": ref["label"], "prompts": {}}
    for name, text in gate.PROMPTS.items():
        comp = ref["prompts"][name]["content"]
        p_ids = post(base, "/tokenize", {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": text}],
                                         "add_generation_prompt": True,
                                         "chat_template_kwargs": {"enable_thinking": False}})["tokens"]
        c_ids = post(base, "/tokenize", {"model": "glm-5.3-flash", "prompt": comp, "add_special_tokens": False})["tokens"]
        ids = p_ids + c_ids
        r = post(base, "/v1/completions", {"model": "glm-5.3-flash", "prompt": ids, "max_tokens": 1,
                                           "temperature": 0.0, "prompt_logprobs": 1})
        plp = r["choices"][0]["prompt_logprobs"]
        lps = []
        for pos, tid in enumerate(ids):
            if pos == 0 or plp[pos] is None:
                lps.append(None); continue
            e = plp[pos].get(str(tid)) or plp[pos].get(tid)
            lps.append(round(e["logprob"], 5) if e else None)
        res["prompts"][name] = {"n_prompt": len(p_ids), "n_comp": len(c_ids), "logprobs": lps}
        print(f"{name}: {len(p_ids)} + {len(c_ids)} tokens scored", flush=True)
    json.dump(res, open(out, "w"))


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    a = runs[0]
    print(f"reference A = {a['label']}")
    for b in runs[1:]:
        print(f"\nB = {b['label']} vs A   (|delta logprob| over every scored token; prompt part / completion part)")
        allp, allc = [], []
        for name, pa in a["prompts"].items():
            pb = b["prompts"].get(name)
            if not pb or len(pb["logprobs"]) != len(pa["logprobs"]):
                print(f"  {name}: token count differs"); continue
            n = pa["n_prompt"]
            d = [abs(x - y) for x, y in zip(pa["logprobs"], pb["logprobs"]) if x is not None and y is not None]
            dp = [abs(x - y) for x, y in zip(pa["logprobs"][:n], pb["logprobs"][:n]) if x is not None and y is not None]
            dc = [abs(x - y) for x, y in zip(pa["logprobs"][n:], pb["logprobs"][n:]) if x is not None and y is not None]
            allp += dp; allc += dc
            def st(v):
                if not v: return "-"
                s = sorted(v); return f"mean {sum(s)/len(s):.4f} p99 {s[min(len(s)-1, int(0.99*len(s)))]:.3f} max {s[-1]:.3f}"
            print(f"  {name:10s} prompt[{len(dp):5d}] {st(dp):38s} | completion[{len(dc):3d}] {st(dc)}")
        for lab, v in (("ALL prompt tokens", allp), ("ALL completion tokens", allc)):
            s = sorted(v)
            print(f"  {lab}: n {len(s)}  mean {sum(s)/len(s):.4f}  p99 {s[int(0.99*len(s))]:.3f}  max {s[-1]:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2:])
    else:
        score(*sys.argv[1:5])
