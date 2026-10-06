# glm-dcp-nv: ajclark's DCP overlay set + the NVFP4 KV port (2026-10-06)

Base: ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark stage/glm-dcp @ f0b64af (Apache-2.0).
Four files are three-way merges (git merge-file; base = the dflash2-port2 / ~/glm-triton original,
"ours" = the NVFP4 lane's version from vllm-glm52-b12x:nvfp4-dflash2-p2 and /var/tmp/glm-triton-nvfp4,
"theirs" = ajclark's DCP version):
- flashmla_sparse.py: 5 conflicts, resolved additively (nvfp4_ds_mla treated as fp8-family for the
  mixed-batch LSE path; DCP guards accept it; workspace skip applies to both; helper gets both
  kv_layout and topk_length)
- b12x_sparse_helpers.py: 2 conflicts (both kwargs kept; length_flat + variable record stride)
- mla_attention.py, kv_cache_interface.py: merged clean (NVFP4's 400 B/token page and the
  fp8-to-ds_mla conversion exemption carried into the DCP versions)
All other files are ajclark's, unmodified. Run on the nvfp4-dflash2-p2 image only.

Fifth edit, found at the first DCP4 boot (2026-10-06): `mla_attention.py`'s DCP decode guard allowed only
`fp8_ds_mla` ("DCP does not support this fp8 kv-cache configuration"). Its rationale (inline scales,
kernel-side dequantization, bf16 query) holds for `nvfp4_ds_mla` too, so the whitelist now names both.
