#!/usr/bin/env python3
"""
Shard-streaming Int4-Int8Mix quantizer for GLM-5.3 (743B), QuantTrio-compatible.

Reads the BF16 base one safetensors shard at a time, quantizes the targeted
modules with compressed-tensors' own qparam/quantize/pack routines, and writes
a 1:1 output shard. Peak RAM is one input shard + one output shard, so it never
needs the accelerate disk-offload path (which would want ~1.4TB of scratch).

1:1 shard mapping makes the job resumable: completed output shards are skipped.

Modes:
  --selftest   round-trip a few real tensors and report dequant error, write nothing
  --limit N    process only the first N shards (smoke test)
  (default)    full run
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.quantization.lifecycle.forward import quantize as ct_quantize
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)

# ---------------------------------------------------------------- recipe ----
# Regexes verbatim from QuantTrio/GLM-5.2-Int4-Int8Mix/config.json.

def _alt(a, b):
    # longest-first so the alternation cannot short-match (77 before 7)
    return "(?:" + "|".join(str(i) for i in range(b, a - 1, -1)) + ")"


L3_77 = _alt(3, 77)
L1_77 = _alt(1, 77)

ATTN = (r"self_attn[.](?:fused_qkv_a_proj_with_mqa|q_a_proj|q_b_proj"
        r"|kv_a_proj_with_mqa|kv_b_proj|o_proj)")
MLP = (r"mlp[.](?:gate_up_proj|gate_proj|up_proj|down_proj"
       r"|shared_experts[.](?:gate_up_proj|gate_proj|up_proj|down_proj))")
MTP_MLP = (r"mlp[.](?:experts[.][0-9]+[.](?:gate_proj|up_proj|down_proj)"
           r"|gate_up_proj|gate_proj|up_proj|down_proj"
           r"|shared_experts[.](?:gate_up_proj|gate_proj|up_proj|down_proj))")

W4_GROUP = QuantizationArgs(num_bits=4, type="int", symmetric=True,
                            strategy="group", group_size=128, dynamic=False)
W8_GROUP = QuantizationArgs(num_bits=8, type="int", symmetric=True,
                            strategy="group", group_size=128, dynamic=False)
W8_CHANNEL = QuantizationArgs(num_bits=8, type="int", symmetric=True,
                              strategy="channel", group_size=-1, dynamic=False)

GROUPS = [
    ("w4a16_experts", W4_GROUP,
     re.compile(rf"model[.]layers[.]{L3_77}[.]mlp[.]experts[.][0-9]+"
                rf"[.](?:gate_proj|up_proj|down_proj)$")),
    ("w8a16_linears", W8_GROUP,
     re.compile(rf"model[.]layers[.]{L1_77}[.](?:{ATTN}|{MLP})$")),
    ("w8a16_mtp_channel", W8_CHANNEL,
     re.compile(rf"model[.]layers[.](?:78)[.](?:mtp_block[.])?"
                rf"(?:{ATTN}|{MTP_MLP})$")),
]

IGNORE = [re.compile(p) for p in [
    r"model[.]layers[.]0[.].*",
    r"model[.]layers[.][1-9][0-9]*[.](?:mtp_block[.])?mlp[.]gate(?:$|[.].*)",
    r"model[.]layers[.][1-9][0-9]*[.](?:mtp_block[.])?self_attn[.]indexer(?:$|[.].*)",
    r"model[.]layers[.][1-9][0-9]*[.](?:mtp_block[.])?self_attn[.]indexers_proj(?:$|[.].*)",
    r"model[.]layers[.][1-9][0-9]*[.](?:eh_proj|enorm|hnorm)[.].*",
    r"model[.]layers[.][1-9][0-9]*[.]shared_head[.]norm[.].*",
    r"model[.]layers[.][1-9][0-9]*[.]shared_head[.]head(?:$|[.].*)",
    r"lm_head",
]]

QUANT_CONFIG_JSON = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "ignore": [
        "re:model[.]layers[.]0[.].*",
        "re:model[.]layers[.][1-9][0-9]*[.](?:mtp_block[.])?mlp[.]gate(?:$|[.].*)",
        "re:model[.]layers[.][1-9][0-9]*[.](?:mtp_block[.])?self_attn[.]indexer(?:$|[.].*)",
        "re:model[.]layers[.][1-9][0-9]*[.](?:mtp_block[.])?self_attn[.]indexers_proj(?:$|[.].*)",
        "re:model[.]layers[.][1-9][0-9]*[.](?:eh_proj|enorm|hnorm)[.].*",
        "re:model[.]layers[.][1-9][0-9]*[.]shared_head[.]norm[.].*",
        "re:model[.]layers[.][1-9][0-9]*[.]shared_head[.]head(?:$|[.].*)",
    ],
    "config_groups": {},          # filled in below
    "packed_modules_mapping": {
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    },
}

_TARGET_SRC = {
    "w4a16_experts": [
        f"re:model[.]layers[.]{L3_77}[.]mlp[.]experts[.][0-9]+"
        f"[.](?:gate_proj|up_proj|down_proj)$"],
    "w8a16_linears": [f"re:model[.]layers[.]{L1_77}[.](?:{ATTN}|{MLP})$"],
    "w8a16_mtp_channel": [
        f"re:model[.]layers[.](?:78)[.](?:mtp_block[.])?(?:{ATTN}|{MTP_MLP})$"],
}
for _name, _args, _ in GROUPS:
    QUANT_CONFIG_JSON["config_groups"][_name] = {
        "targets": _TARGET_SRC[_name],
        "weights": {
            "num_bits": _args.num_bits,
            "type": "int",
            "symmetric": True,
            "strategy": _args.strategy.value if hasattr(_args.strategy, "value")
                        else str(_args.strategy),
            "group_size": _args.group_size if _args.group_size else -1,
            "dynamic": False,
        },
    }

SUFFIXES = (".weight_scale_inv", ".weight_zero_point", ".weight_scale",
            ".e_score_correction_bias", ".weight", ".bias")


def module_of(name):
    for s in SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def scheme_for(module):
    for pat in IGNORE:
        if pat.search(module):
            return None, None
    for gname, args, pat in GROUPS:
        if pat.search(module):
            return gname, args
    return None, None


# ------------------------------------------------------------ quantizing ----
def quantize_weight(W, args):
    """bf16 [out, in] -> (weight_packed int32, weight_scale bf16, shape int64)."""
    out_f, in_f = W.shape
    Wf = W.to(torch.float32)

    if args.strategy == "group" or str(args.strategy) == "QuantizationStrategy.GROUP":
        gs = args.group_size
        if in_f % gs:
            raise ValueError(f"in_features {in_f} not divisible by group_size {gs}")
        Wg = Wf.reshape(out_f, in_f // gs, gs)
        mn, mx = Wg.amin(-1), Wg.amax(-1)
    else:                                              # channel
        mn = Wf.amin(-1, keepdim=True)
        mx = Wf.amax(-1, keepdim=True)

    scale, zp = calculate_qparams(mn, mx, args)

    # Round the scale to the dtype we will actually persist, then quantize with
    # that rounded value -- otherwise dequant at serve time uses a slightly
    # different scale than the one quantization assumed.
    scale = scale.to(torch.bfloat16).to(torch.float32)

    q = ct_quantize(Wf, scale, zp, args, dtype=torch.int8)
    packed = pack_to_int32(q, args.num_bits, packed_dim=1)

    # Keep the scale 2-D ([out, n_groups]; channel -> [out, 1]) to match
    # QuantTrio's on-disk layout exactly -- do NOT squeeze.
    return (packed.contiguous(),
            scale.to(torch.bfloat16).contiguous(),
            torch.tensor([out_f, in_f], dtype=torch.int64))


def dequantize(packed, scale, shape, args):
    """Inverse, for the self-test only."""
    q = unpack_from_int32(packed, args.num_bits,
                          torch.Size(tuple(shape.tolist())), packed_dim=1)
    out_f, in_f = shape.tolist()
    s = scale.to(torch.float32)
    if s.dim() == 1:
        s = s.unsqueeze(-1)
    if s.shape[-1] == 1:
        return q.to(torch.float32) * s
    gs = in_f // s.shape[-1]
    return (q.to(torch.float32).reshape(out_f, s.shape[-1], gs)
            * s.unsqueeze(-1)).reshape(out_f, in_f)


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/var/tmp/models/GLM-5.3-BF16")
    ap.add_argument("--out", default="/var/tmp/models/GLM-5.3-Int4-Int8Mix")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--drop-cache", action="store_true",
                    help="fadvise DONTNEED each input shard after reading")
    args_cli = ap.parse_args()

    src, out = args_cli.src, args_cli.out
    dev = torch.device(args_cli.device)

    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wmap = index["weight_map"]
    shards = sorted(set(wmap.values()))
    by_shard = {}
    for tname, sh in wmap.items():
        by_shard.setdefault(sh, []).append(tname)

    print(f"src={src}\nout={out}\nshards={len(shards)} tensors={len(wmap)} "
          f"device={dev}", flush=True)

    # ---------------- self-test ----------------
    if args_cli.selftest:
        print("\n=== SELF-TEST: quantize -> pack -> unpack -> dequantize ===")
        # take up to PER_GROUP samples from EACH scheme so int4 + channel are
        # covered, not just whichever scheme happens to live in shard 1
        PER_GROUP = 2
        want = {g: PER_GROUP for g, _, _ in GROUPS}
        for sh in shards:
            if not any(want.values()):
                break
            names = [t for t in sorted(by_shard[sh]) if t.endswith(".weight")]
            hits = [(t, *scheme_for(module_of(t))) for t in names]
            hits = [(t, g, a) for t, g, a in hits if g and want.get(g, 0) > 0]
            if not hits:
                continue
            with safe_open(os.path.join(src, sh), framework="pt", device="cpu") as f:
                for tname, gname, qargs in hits:
                    if want[gname] <= 0:
                        continue
                    W = f.get_tensor(tname).to(dev)
                    if W.dim() != 2:
                        continue
                    p, s, shp = quantize_weight(W, qargs)
                    D = dequantize(p.cpu(), s.cpu(), shp, qargs)
                    Wf = W.to(torch.float32).cpu()
                    err = (D - Wf).abs()
                    denom = Wf.abs().mean().clamp_min(1e-12)
                    nuniq = int(unpack_from_int32(
                        p.cpu(), qargs.num_bits,
                        torch.Size(tuple(shp.tolist())), packed_dim=1).unique().numel())
                    print(f"  {gname:18} {tname}")
                    print(f"      shape={tuple(W.shape)} packed={tuple(p.shape)} "
                          f"scale={tuple(s.shape)} distinct_levels={nuniq}")
                    print(f"      mean|err|={err.mean():.6g}  max|err|={err.max():.6g}"
                          f"  rel={err.mean()/denom:.4%}")
                    want[gname] -= 1
        missed = [g for g, n in want.items() if n == PER_GROUP]
        if missed:
            print(f"\n*** WARNING: no sample found for {missed} ***")
        print("\nself-test done (wrote nothing)")
        return

    # ---------------- full / limited run ----------------
    os.makedirs(out, exist_ok=True)
    todo = shards[: args_cli.limit] if args_cli.limit else shards

    new_wmap, total_bytes = {}, 0
    t0 = time.time()
    stats = {"w4a16_experts": 0, "w8a16_linears": 0, "w8a16_mtp_channel": 0,
             "passthrough": 0}

    for i, sh in enumerate(todo, 1):
        dst = os.path.join(out, sh)
        meta_path = dst + ".meta.json"

        if os.path.exists(dst) and os.path.exists(meta_path):      # resume
            m = json.load(open(meta_path))
            for k in m["keys"]:
                new_wmap[k] = sh
            total_bytes += m["bytes"]
            for k, v in m.get("stats", {}).items():
                stats[k] = stats.get(k, 0) + v
            print(f"[{i}/{len(todo)}] {sh} SKIP (done)", flush=True)
            continue

        tensors = {}
        spath = os.path.join(src, sh)
        with safe_open(spath, framework="pt", device="cpu") as f:
            for tname in sorted(by_shard[sh]):
                mod = module_of(tname)
                gname, qargs = scheme_for(mod)
                W = f.get_tensor(tname)
                if gname is not None and tname.endswith(".weight") and W.dim() == 2:
                    p, s, shp = quantize_weight(W.to(dev), qargs)
                    tensors[mod + ".weight_packed"] = p.cpu()
                    tensors[mod + ".weight_scale"] = s.cpu()
                    tensors[mod + ".weight_shape"] = shp
                    stats[gname] += 1
                else:
                    tensors[tname] = W
                    stats["passthrough"] += 1

        save_file(tensors, dst, metadata={"format": "pt"})
        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        total_bytes += nbytes
        json.dump({"keys": list(tensors.keys()), "bytes": nbytes,
                   "stats": {k: stats[k] for k in stats}},
                  open(meta_path, "w"))
        for k in tensors:
            new_wmap[k] = sh

        if args_cli.drop_cache:
            fd = os.open(spath, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)

        el = time.time() - t0
        rate = i / el
        print(f"[{i}/{len(todo)}] {sh} -> {nbytes/2**30:.2f} GiB "
              f"| out {total_bytes/2**30:.1f} GiB "
              f"| {el/60:.1f} min elapsed, ETA {(len(todo)-i)/rate/60:.1f} min",
              flush=True)

    if args_cli.limit:
        print("\n--limit run: not writing index/config")
        return

    # index
    json.dump({"metadata": {"total_size": total_bytes}, "weight_map": new_wmap},
              open(os.path.join(out, "model.safetensors.index.json"), "w"),
              indent=2)

    # config.json + quantization_config
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["quantization_config"] = QUANT_CONFIG_JSON
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

    for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
               "chat_template.jinja", "LICENSE", "README.md"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, fn))

    print(f"\nDONE  {total_bytes/2**30:.1f} GiB  in {(time.time()-t0)/60:.1f} min")
    print("stats:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
