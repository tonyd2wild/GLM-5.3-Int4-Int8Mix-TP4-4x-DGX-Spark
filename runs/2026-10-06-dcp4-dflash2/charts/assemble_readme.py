#!/usr/bin/env python3
"""assemble_readme.py README_IN FILLED_SECTION X512.json README_OUT -- new status block + new top section."""
import json, sys
src, sec, xp, out = sys.argv[1:5]
s = open(src).read(); new = open(sec).read(); X = json.load(open(xp))
i = s.index("> **Status: DFlash2 stage complete (2026-08-29).**")
j = s.index("\n---\n", i)
status = (f"> **Status (2026-10-06): two DCP4 lanes.** TP4 + decode context parallelism (DCP4) + DFlash2 k=7, with\n"
          f"> [@ajclark](https://github.com/ajclark)'s DCP patch set. **Lane 2 (NVFP4 KV)** holds a **689,772-token KV\n"
          f"> pool at 262K context**, 2.35× the previous best lane, at the same decode speed, and serves a\n"
          f"> **{X['maxlen']:,}-token window** ({X['pool']:,}-token pool, needle {X['needle']['needles_found']} at "
          f"{X['needle']['prompt_tokens']:,} tokens). Lane 1 (fp8 KV) holds 462,308. The trade-off is prefill\n"
          f"> (~400 tok/s). Earlier stages (quantization 2026-08-28, DFlash2 and NVFP4 KV 2026-08-29) are below,\n"
          f"> unchanged. The 69-scenario quality eval has **not** been run; no claim of quality parity with the BF16\n"
          f"> base is made here.")
s = s[:i] + status + "\n\n---\n\n" + new.rstrip("\n").rstrip("-").rstrip() + "\n" + s[j:]
open(out, "w").write(s); print("assembled", out)
