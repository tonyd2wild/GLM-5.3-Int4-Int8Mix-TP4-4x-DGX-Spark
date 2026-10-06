# Lane 2: TP4 + DCP4 + NVFP4 KV + DFlash2 k=7 (2026-10-06)

Launcher `launch/launch-glm53big-dcp4-nvfp4.sh` (ajclark's DCP overlay set merged with the NVFP4 KV port,
`dcp/glm-dcp-nv/`), image `vllm-glm52-b12x:nvfp4-dflash2-p2`, max-num-seqs 6, mnbt 2048,
`reasoning_effort: low`. Same session and GPUs as lane 1 (85.7-90.1 TFLOPS on all four).

## Capacity

| | 262K window, 7 GB KV per rank | 512K window, 6 GB KV per rank |
|---|---|---|
| KV pool, vLLM boot log | 689,772 tokens (2.63x 262,144) | 594,532 tokens (1.13x 524,288) |
| KV pool, measured | ~650,000: one 204,902-token request holds 31.5% (log 6.1% optimistic) | ~580,000: a 490,425-token request peaks at 84.6% (log 2.5% optimistic) |
| lowest MemAvailable | Reddie 2.3 GB during the 204K request, others 3.8-3.9 GB | Reddie 3.8 GB, others 5.3-5.5 GB, through a 490K-token prefill |
| needle (3 codes at 10/50/90% depth) | 3/3 at 254,389 tokens | **3/3 at 490,425 tokens** (TTFT 1,423 s, 345 tok/s) |

## Correctness gate (before any benchmark)

Reference: the same stack at DCP1 (`DCP_SIZE=1 MAXLEN=131072`, pool 173,447). Teacher-forced per-token log-probabilities
on 8 prompts (1,418 completion tokens, 22,336 prompt tokens):

| | completion mean / p99 / max | prompt mean / p99 / max |
|---|---|---|
| reference run 2 vs run 1 (noise floor) | 0.0220 / 0.487 / 1.991 | 0.2530 / 1.127 / 8.207 |
| DCP4 vs reference run 1 | 0.0232 / 0.551 / 2.465 | 0.2564 / 1.138 / 9.841 |

Greedy text: 4/8 identical (reference vs itself 3/8); both long-context prompts identical; AURORA-7731 and
QUARTZ-4482 retrieved; count-to-100 100/100. Files: `gate-*.json`, `tf-*.json`.

## Speed

Single stream, C1-C6 and prefill: `dcp4-nvfp4.json` (all cells clean; 3 interference retries, 6 min waited).
Long context: 115K-token prompt, 16.2 tok/s decode, TTFT 290 s. Needle: 3/3 at 254,389 tokens (TTFT 677 s, 376 tok/s).
Prefill: 415 tok/s at 5,020 and 31,900 tokens, 395 at 106,130.
