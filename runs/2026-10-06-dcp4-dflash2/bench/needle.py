#!/usr/bin/env python3
"""needle.py BASE TARGET_TOKENS LABEL -- 3 needles at 10/50/90% depth of a salted filler doc; cold prefill + retrieval.
Stdlib only. Sizes with the server's /tokenize. Prints one JSON line."""
import json, random, sys, time, urllib.request, uuid
base, target, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
MODEL = "glm-5.3-flash"
W = ("system memory network cache latency throughput kernel thread process scheduler compiler garden river mountain "
     "village market harbour teacher student library museum theatre orchestra painter novel poem history economy "
     "policy budget contract engineer doctor patient hospital battery engine turbine bridge tunnel railway airport "
     "weather forecast season harvest winter summer autumn spring ocean island forest desert canyon glacier").split()
def post(path, body, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
def ntok(t): r = post("/tokenize", {"model": MODEL, "prompt": t}); return r.get("count") or len(r["tokens"])
rng = random.Random(7)
def para(i): return f"{i}. " + " ".join((" ".join(rng.choice(W) for _ in range(rng.randint(8, 16)))).capitalize() + "." for _ in range(rng.randint(3, 6)))
sample = "\n".join(para(i) for i in range(200)); per = ntok(sample) / 200
n = int(target / per)
paras = [para(i) for i in range(n)]
codes = {"falcon": str(rng.randint(10000, 99999)), "orchid": str(rng.randint(10000, 99999)), "granite": str(rng.randint(10000, 99999))}
for frac, (name, code) in zip((0.1, 0.5, 0.9), codes.items()):
    paras.insert(int(len(paras) * frac), f"IMPORTANT: the secret code for {name} is {code}. Remember it.")
doc = f"[{uuid.uuid4()}]\n" + "\n".join(paras)
q = doc + "\n\nWhat are the secret codes for falcon, orchid and granite? Answer exactly in the form falcon=NNNNN orchid=NNNNN granite=NNNNN."
body = {"model": MODEL, "messages": [{"role": "user", "content": q}], "max_tokens": 400, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"reasoning_effort": "low"}}
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.monotonic(); first = None; text = []; usage = None; err = None
try:
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: "): continue
            p = line[6:].strip()
            if p == b"[DONE]": break
            e = json.loads(p)
            if e.get("usage"): usage = e["usage"]
            for c in e.get("choices", []):
                d = c.get("delta", {}); piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                if piece:
                    if first is None: first = time.monotonic()
                    if d.get("content"): text.append(d["content"])
except Exception as ex: err = str(ex)[:200]
ans = "".join(text); hits = sum(1 for k, v in codes.items() if v in ans)
pt = usage["prompt_tokens"] if usage else None
print(json.dumps({"label": label, "prompt_tokens": pt, "ttft_s": round(first - t0, 1) if first else None,
                  "prefill_tok_s": round(pt / (first - t0)) if pt and first else None, "needles_found": f"{hits}/3",
                  "answer": ans.strip()[:120], "error": err}), flush=True)
