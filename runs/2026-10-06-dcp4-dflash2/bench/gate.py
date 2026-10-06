#!/usr/bin/env python3
"""gate.py BASE LABEL OUT.json  -- temperature-0 reference outputs for a token-for-token comparison.
gate.py --compare A.json B.json  -- report per prompt: identical, or first divergence with both sides' top-2 logprobs.
Requests go one at a time (no batching effects). Thinking off, so the whole output is the answer."""
import json, random, sys, time, urllib.request

W = ("system memory network cache latency throughput kernel thread process scheduler compiler garden river mountain "
     "village market harbour teacher student library museum theatre orchestra painter novel poem history economy").split()


def filler(n_words, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(W) for _ in range(n_words))


PROMPTS = {
    "count100": "Count from 1 to 100, one number per line, digits only, nothing else.",
    "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two doctests.",
    "json": "Return a JSON array of 5 fictional books, each with title, author, year and genre. JSON only.",
    "prose": "In about 150 words, explain why the sky is blue to a curious ten-year-old.",
    "math": "A tank fills at 12 litres per minute and drains at 5 litres per minute. Starting empty, how long until it holds 245 litres? Show the steps.",
    "translate": "Translate into French: 'The library opens at nine, but the reading room stays closed until noon on Sundays.'",
    "long8k": "Read the following notes, then answer the question at the end.\n\n" + filler(6000, 11)
              + "\n\nIMPORTANT FACT: the archive code is AURORA-7731.\n\n" + filler(400, 12)
              + "\n\nQuestion: what is the archive code? Then list the first five distinct words of the notes.",
    "long20k": "Read the following log, then answer.\n\n" + filler(15000, 21)
               + "\n\nThe vault password is QUARTZ-4482.\n\n" + filler(600, 22)
               + "\n\nWhat is the vault password, and how many times does the word 'river' appear in the last sentence before the password?",
}


def run(base, label, out):
    res = {"label": label, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "prompts": {}}
    for name, text in PROMPTS.items():
        body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": text}], "max_tokens": 300,
                "temperature": 0.0, "logprobs": True, "top_logprobs": 2,
                "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.monotonic()
        req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=1800))
            ch = r["choices"][0]
            lp = (ch.get("logprobs") or {}).get("content") or []
            res["prompts"][name] = {
                "content": ch["message"].get("content") or "",
                "tokens": [t["token"] for t in lp],
                "top2": [[(c["token"], round(c["logprob"], 4)) for c in t.get("top_logprobs", [])[:2]] for t in lp],
                "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"],
                "wall_s": round(time.monotonic() - t0, 2)}
        except Exception as e:  # noqa: BLE001
            res["prompts"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        p = res["prompts"][name]
        print(f"{name}: {p.get('prompt_tokens')} -> {p.get('completion_tokens')} tok, {p.get('wall_s')} s, "
              f"logprobs {len(p.get('tokens', []))} {p.get('error', '')}", flush=True)
    json.dump(res, open(out, "w"), indent=1)


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print(f"A = {a['label']}   B = {b['label']}")
    same = 0
    for name in PROMPTS:
        pa, pb = a["prompts"].get(name, {}), b["prompts"].get(name, {})
        if "error" in pa or "error" in pb:
            print(f"  {name}: ERROR  A={pa.get('error')}  B={pb.get('error')}"); continue
        ta, tb = pa["tokens"] or list(pa["content"]), pb["tokens"] or list(pb["content"])
        if pa["content"] == pb["content"]:
            same += 1; print(f"  {name}: IDENTICAL ({len(ta)} tokens)"); continue
        i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), min(len(ta), len(tb)))
        gap_a = (pa["top2"][i][0][1] - pa["top2"][i][1][1]) if i < len(pa["top2"]) and len(pa["top2"][i]) == 2 else None
        gap_b = (pb["top2"][i][0][1] - pb["top2"][i][1][1]) if i < len(pb["top2"]) and len(pb["top2"][i]) == 2 else None
        print(f"  {name}: DIVERGES at token {i}/{len(ta)}: A {ta[i:i+1]} top2 {pa['top2'][i] if i < len(pa['top2']) else None} "
              f"| B {tb[i:i+1]} top2 {pb['top2'][i] if i < len(pb['top2']) else None} | logprob gap A {gap_a} B {gap_b}")
        print(f"      shared prefix: {''.join(ta[max(0, i-12):i])!r}")
    print(f"identical: {same}/{len(PROMPTS)}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
