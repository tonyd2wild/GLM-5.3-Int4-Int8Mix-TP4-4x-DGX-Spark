#!/usr/bin/env python3
"""Fail-closed verification of the GLM-5.3 Int4-Int8Mix output."""
import json, os, re, sys, glob
from safetensors import safe_open

SRC = "/var/tmp/models/GLM-5.3-BF16"
OUT = "/var/tmp/models/GLM-5.3-Int4-Int8Mix"

sys.path.insert(0, "/var/tmp")
from glm53_quant_stream import scheme_for, module_of, GROUPS

src_idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
out_idx_f = f"{OUT}/model.safetensors.index.json"
out_idx = json.load(open(out_idx_f))
owm = out_idx["weight_map"]

fail = []

# 1. every source tensor accounted for
q_mods, passthru = set(), set()
for t in src_idx:
    g, _ = scheme_for(module_of(t))
    if g and t.endswith(".weight"):
        q_mods.add(module_of(t))
    else:
        passthru.add(t)

exp_out = {m + s for m in q_mods for s in
           (".weight_packed", ".weight_scale", ".weight_shape")} | passthru
got_out = set(owm)

missing = exp_out - got_out
extra = got_out - exp_out
print(f"source tensors      : {len(src_idx)}")
print(f"quantized modules   : {len(q_mods)}")
print(f"passthrough tensors : {len(passthru)}")
print(f"expected out tensors: {len(exp_out)}")
print(f"actual   out tensors: {len(got_out)}")
if missing:
    fail.append(f"MISSING {len(missing)}: {sorted(missing)[:5]}")
if extra:
    fail.append(f"EXTRA {len(extra)}: {sorted(extra)[:5]}")

# 2. per-group module counts
counts = {}
for m in q_mods:
    g, _ = scheme_for(m)
    counts[g] = counts.get(g, 0) + 1
print("\nper-group module counts:")
for g in ("w4a16_experts", "w8a16_linears", "w8a16_mtp_channel"):
    print(f"  {g:20} {counts.get(g,0)}")
EXPECT = {"w4a16_experts": 57600, "w8a16_linears": 616, "w8a16_mtp_channel": 776}
for g, n in EXPECT.items():
    if counts.get(g) != n:
        fail.append(f"group {g}: got {counts.get(g)} expected {n}")

# 3. sacred modules must remain full precision, un-packed
SACRED = ["model.layers.78.eh_proj.weight",
          "model.layers.10.mlp.gate.weight",
          "model.layers.0.self_attn.o_proj.weight",
          "model.layers.0.mlp.down_proj.weight",
          "lm_head.weight"]
print("\nsacred (must be plain .weight, BF16):")
for s in SACRED:
    if s not in src_idx:
        print(f"  {s:52} (absent in source - skip)")
        continue
    ok = s in owm
    packed_leak = module_of(s) + ".weight_packed" in owm
    print(f"  {s:52} present={ok} packed_leak={packed_leak}")
    if not ok:
        fail.append(f"sacred missing from output: {s}")
    if packed_leak:
        fail.append(f"sacred was QUANTIZED: {s}")

# 4. indexer never quantized
leaks = [k for k in owm if "indexer" in k and k.endswith(".weight_packed")]
if leaks:
    fail.append(f"indexer quantized: {leaks[:5]}")
print(f"\nindexer weight_packed leaks: {len(leaks)}")

# 5. every referenced shard exists; declared size sane
shards = sorted(set(owm.values()))
miss_sh = [s for s in shards if not os.path.exists(os.path.join(OUT, s))]
if miss_sh:
    fail.append(f"missing shards: {miss_sh[:5]}")
real = sum(os.path.getsize(os.path.join(OUT, s)) for s in shards if os.path.exists(os.path.join(OUT, s)))
print(f"shards={len(shards)} missing={len(miss_sh)} "
      f"declared={out_idx['metadata']['total_size']/2**30:.1f} GiB "
      f"on_disk={real/2**30:.1f} GiB")

# 6. spot-check dtypes on disk
print("\ndtype spot-check:")
probe = ["model.layers.10.mlp.experts.0.down_proj.weight_packed",
         "model.layers.10.mlp.experts.0.down_proj.weight_scale",
         "model.layers.78.eh_proj.weight",
         "model.layers.10.mlp.gate.weight"]
for p in probe:
    if p not in owm:
        print(f"  {p:60} <absent>")
        continue
    with safe_open(os.path.join(OUT, owm[p]), framework="pt") as h:
        sl = h.get_slice(p)
        print(f"  {p:60} {sl.get_dtype():8} {list(sl.get_shape())}")

# 7. config
cfg = json.load(open(f"{OUT}/config.json"))
qc = cfg.get("quantization_config")
print("\nconfig.json:")
print(f"  quant_method={qc and qc.get('quant_method')}  format={qc and qc.get('format')}")
print(f"  groups={list((qc or {}).get('config_groups', {}).keys())}")
print(f"  arch={cfg.get('architectures')}  layers={cfg.get('num_hidden_layers')}")
if not qc or qc.get("format") != "pack-quantized":
    fail.append("config.json quantization_config missing/wrong")

print()
if fail:
    print("*** VERIFY FAILED ***")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("VERIFY OK")
