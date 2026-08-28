#!/usr/bin/env python3
"""
MANDATORY pre-quant gate: verify the layer map before touching a single tensor.

The three group regexes decide which modules get 4-bit, which get 8-bit, and which
stay full precision. A regex that silently matches nothing produces a quant that
looks fine, loads fine, and is wrong -- and you will not find out until eval.

THE TRAP THIS SCRIPT EXISTS TO AVOID: the group patterns are anchored (`...o_proj$`)
and describe MODULE names, while `model.safetensors.index.json` keys are TENSOR
names (`...o_proj.weight`). Matching the patterns directly against weight_map keys
makes every group report zero matches -- which reads as "the recipe is catastrophically
broken" when the recipe is fine and the harness is wrong. Derive module names first.

Usage:  python3 dryrun_layermap.py /path/to/GLM-5.3-BF16
"""
import collections
import json
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from glm53_quant_stream import GROUPS, IGNORE, module_of  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/models/GLM-5.3-BF16"

# Expected counts for big GLM-5.3 (78 layers, 256 experts, first_k_dense_replace 3).
EXPECT = {"w4a16_experts": 57600, "w8a16_linears": 616, "w8a16_mtp_channel": 776}

idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
names = sorted(idx["weight_map"])
mods = sorted({module_of(n) for n in names})

print(f"src     : {SRC}")
print(f"tensors : {len(names)}")
print(f"modules : {len(mods)}")
print()

cover, fail = {}, []
for gname, args, pat in GROUPS:
    hits = {m for m in mods if pat.search(m) and not any(i.search(m) for i in IGNORE)}
    cover[gname] = hits
    ex = sorted(hits)[:1]
    print(f"{gname:20} {len(hits):6}   e.g. {ex}")
    if gname in EXPECT and len(hits) != EXPECT[gname]:
        fail.append(f"{gname}: matched {len(hits)}, expected {EXPECT[gname]}")

ign = {m for m in mods if any(i.search(m) for i in IGNORE)}
print(f"{'ignore':20} {len(ign):6}")
print()

# No module may land in two groups at once.
gs = list(cover)
for i in range(len(gs)):
    for j in range(i + 1, len(gs)):
        dup = cover[gs[i]] & cover[gs[j]]
        if dup:
            fail.append(f"DOUBLE-MATCHED {gs[i]} & {gs[j]}: {sorted(dup)[:5]}")

allq = set().union(*cover.values()) if cover else set()

# Modules owning a 2-D .weight that are neither quantized nor explicitly ignored.
# NOTE: an orphan is NOT fatal. llm-compressor / this quantizer only touch modules
# matching a group target, so an unmatched module simply stays full precision.
# `ignore` exists to OVERRIDE a target match, not to enumerate everything skipped.
lin = {module_of(n) for n in names
       if n.endswith(".weight") and "norm" not in n
       and "embed" not in n and "lm_head" not in n}
orphans = sorted(lin - allq - ign)

print(f"linear-ish weight modules : {len(lin)}")
print(f"covered by a group        : {len(lin & allq)}")
print(f"explicitly ignored        : {len(lin & ign)}")
print(f"orphans (informational)   : {len(orphans)}")
if orphans:
    c = collections.Counter(re.sub(r"[0-9]+", "N", o) for o in orphans)
    for k, v in c.most_common(10):
        print(f"    {v:6}  {k}   -> stays full precision")
print()

if fail:
    print("*** DRY-RUN FAILED ***")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("DRY-RUN OK - proceed")
