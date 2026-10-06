# DCP + DFlash2: the context-parallel overlay set

**This directory is Allan Clark's work ([@ajclark](https://github.com/ajclark)), vendored under his
Apache-2.0 license after he offered it as a lane for this repo.** He made decode context parallelism (DCP) run alongside DFlash2 on this
recipe, which this repo's README used to call impossible, and offered it back in
[issue #4](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/4).
Source of truth:
[ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark](https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark)
at the commit in `AJCLARK-COMMIT`, Apache-2.0 (`LICENSE-ajclark`, `NOTICE-ajclark`). Everything
here is copied unmodified from its `stage/` directory; `SHA256SUMS` lets you check that.

| path | what | from |
|---|---|---|
| `glm-dcp/` (27 files) | modified vLLM files: the sparse-MLA target's KV cache is sharded across the DCP ranks while the DFlash2 drafter's sliding-window group stays replicated; includes the DSA indexer bounds fix for [issue #6](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/6) | ajclark `stage/glm-dcp` |
| `glm-triton-aj/sm12x_mqa.py` | the paged-MQA kernel without per-request shape specialization (no late Triton compiles hours into serving), plus launch-failure diagnostics; the other nine sm12x overlays are unchanged from `~/glm-triton` | ajclark `stage/glm-triton` |
| `glm-dcp-nv/` | the same set with NVFP4 KV merged in (lane 2, see the README's results) | this repo, merge of the two |

The sm12x Triton kernels underneath (`~/glm-triton`) are
[CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10) (Apache-2.0, also
crediting ciprianveg). The base vLLM is commit `ab666069` with DFlash2 from
[vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816) back-ported
([`dflash2-port/`](../dflash2-port/)).

## Stage it

On every node:

```bash
cp -r dcp/glm-dcp ~/glm-dcp            # lane 1 (fp8 KV)
cp -r dcp/glm-dcp-nv ~/glm-dcp-nv      # lane 2 (NVFP4 KV)
cp -a ~/glm-triton ~/glm-triton-aj && cp dcp/glm-triton-aj/sm12x_mqa.py ~/glm-triton-aj/
```

Then launch ranks 1, 2, 3 first and 0 last with
[`launch/launch-glm53big-dcp4-nvfp4.sh`](../launch/launch-glm53big-dcp4-nvfp4.sh) (lane 2, image
`vllm-glm52-b12x:nvfp4-dflash2-p2`) or [`launch/launch-glm53big-dcp4.sh`](../launch/launch-glm53big-dcp4.sh)
(lane 1, image `vllm-glm52-b12x:dflash2-port2`). Both are ajclark's legacy launcher with this fleet's values
(switched fabric instead of his switchless ring, NFS weights on ranks 1-3, DCP 4, 262K, 7 GB KV per rank);
`MAXLEN=524288 KVBYTES=6000000000` gives lane 2 a 512K window.

## Not used here (yet)

His repo also has a multi-node NVMe KV tier ([issue #5](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/5)),
a concurrent checkpoint loader, a vLLM 0.29.0 runtime and adaptive-speculation experiments. Those are
his to document; see his repository.
