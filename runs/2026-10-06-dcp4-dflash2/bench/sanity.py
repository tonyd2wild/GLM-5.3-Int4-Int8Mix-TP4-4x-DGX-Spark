#!/usr/bin/env python3
"""sanity.py BASE LABEL -- functional checks before any benchmark. Stdlib only. Prints JSON, exits 1 on a fail."""
import json, sys, time, urllib.request

BASE, LABEL = sys.argv[1].rstrip("/"), sys.argv[2]
MODEL = "glm-5.3-flash"


def chat(messages, max_tokens=400, kwargs=None, tools=None, temperature=0.0):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if kwargs is not None:
        body["chat_template_kwargs"] = kwargs
    if tools:
        body["tools"] = tools
    t0 = time.monotonic()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    m = r["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
            "tool_calls": m.get("tool_calls") or [], "finish": r["choices"][0]["finish_reason"],
            "usage": r.get("usage"), "wall_s": round(time.monotonic() - t0, 2)}


checks = {}
r = chat([{"role": "user", "content": "What is the capital of France? Answer in one sentence."}], 200, {"reasoning_effort": "low"})
checks["capital"] = {"pass": "Paris" in r["content"], "content": r["content"][:200], "wall_s": r["wall_s"]}

r = chat([{"role": "user", "content": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
           "How much does the ball cost? Give the final answer as a dollar amount."}], 1500, {"reasoning_effort": "low"})
checks["bat_ball"] = {"pass": ("0.05" in r["content"] or "5 cents" in r["content"]), "content": r["content"][-200:],
                      "wall_s": r["wall_s"]}

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = chat([{"role": "user", "content": "What's the weather in Tokyo right now? Use the tool."}], 400,
         {"reasoning_effort": "low"}, tools)
tc = r["tool_calls"]
ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "tokyo" in tc[0]["function"]["arguments"].lower()
checks["tool_call"] = {"pass": ok, "tool_calls": tc[:2], "finish": r["finish"]}

r_off = chat([{"role": "user", "content": "Name three primary colors."}], 300, {"enable_thinking": False})
# a question that needs working: with a trivial prompt the model may legitimately skip thinking even at "high"
r_hi = chat([{"role": "user", "content": "A clock shows 3:15. What is the exact angle in degrees between the hour and "
              "minute hands? Explain briefly."}], 4000, {"reasoning_effort": "high"})
checks["thinking_off"] = {"pass": len(r_off["reasoning"]) == 0 and len(r_off["content"]) > 0,
                          "reasoning_chars": len(r_off["reasoning"]), "content": r_off["content"][:120]}
checks["thinking_high"] = {"pass": len(r_hi["reasoning"]) > 0 and len(r_hi["content"]) > 0,
                           "reasoning_chars": len(r_hi["reasoning"]), "content": r_hi["content"][:120]}

prompt = ("Count from 1 to 100, one number per line, digits only, nothing else.")
r = chat([{"role": "user", "content": prompt}], 600, {"enable_thinking": False})
lines = [l.strip() for l in r["content"].strip().splitlines() if l.strip()]
correct = sum(1 for i, l in enumerate(lines[:100]) if l == str(i + 1))
checks["count100"] = {"pass": correct == 100 and len(lines) == 100, "correct_lines": correct, "lines": len(lines),
                      "completion_tokens": (r["usage"] or {}).get("completion_tokens"), "wall_s": r["wall_s"]}

out = {"label": LABEL, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "all_pass": all(c["pass"] for c in checks.values()),
       "checks": checks}
print(json.dumps(out, indent=1))
sys.exit(0 if out["all_pass"] else 1)
