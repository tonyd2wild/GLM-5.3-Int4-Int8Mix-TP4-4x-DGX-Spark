# Lane 1: TP4 + DCP4 + DFlash2 k=7, fp8 KV, 262K (2026-10-06)

Launcher `launch/launch-glm53big-dcp4.sh` (ajclark's DCP overlay set + his sm12x_mqa fix), image
`vllm-glm52-b12x:dflash2-port2`, 7 GB KV per rank, max-num-seqs 6, mnbt 2048, `reasoning_effort: low`.
GPUs verified first: 85.7 / 86.6 / 89.5 / 90.1 TFLOPS (gputest.sh, all four nodes).

## Capacity and memory

| | |
|---|---|
| KV pool, vLLM boot log | 462,308 tokens (1.76 x 262,144) |
| KV pool, measured | ~433,900 tokens: one 204,901-token request holds 47.2% of the blocks (log overstates by 6.5%) |
| `cache_config_info.kv_cache_size_tokens` | 312,035 (a different, under-counting calculation in this build; not used) |
| lowest MemAvailable, whole run | Reddie 3.1 GB, others 4.3-4.7 GB; 0 preemptions |
| boot | weights 405 s (local) / 605 s (NFS ranks), engine init 127 s |

## Sanity: all pass
capital, bat-and-ball ($0.05), tool call `get_weather({"city":"Tokyo"})`, thinking off / high, count to 100 (100/100).

## Single stream (median of 3, temperature 0)

| prompt | decode tok/s | e2e tok/s | TTFT / time to answer | accept, mean len |
|---|---|---|---|---|
| count to 100 | 53.9 | 50.6 | 0.28 / 0.57 s | 96.5%, 7.76 |
| count to 300 | 52.9 | 51.6 | 0.28 / 0.70 s | 97.4%, 7.82 |
| tool call | 57.5 | 24.3 (34 tokens) | 0.82 s | 85.7%, 7.00 |
| code | 36.2 | 35.3 | 0.53 s | 69.0%, 5.83 |
| json | 36.0 | 35.2 | 0.47 / 0.90 s | 65.8%, 5.60 |
| math | 36.2 | 35.2 | 0.44 / 3.80 s | 67.3%, 5.71 |
| sql | 32.3 | 31.2 | 0.60 s | 56.4%, 4.95 |
| summary | 19.3 | 19.1 | 0.46 s | 27.0%, 2.89 |
| prose | 15.8 | 15.7 | 0.37 / 0.80 s | 20.2%, 2.41 |
| narrative | 14.7 | 14.6 | 0.49 s | 16.9%, 2.19 |

## C1-C6, mixed real prompts (aggregate tok/s, median / peak of 3 rounds)

| C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|
| 33.9 / 36.1 | 32.8 / 48.1 | 28.6 / 53.3 | 54.9 / 55.0 | 32.7 / 36.6 | 41.7 / 46.2 |

0 failures, 0 preemptions. Median and peak differ because each round sends a different mix; a round with a
700-token prose or narrative stream is bound by that stream.

## Cold prefill and long context

| | |
|---|---|
| 5,019-token prompt | 395 tok/s (TTFT 12.7 s) |
| 21,745-token prompt | 397 tok/s (TTFT 54.8 s) |
| 31,897-token prompt | 381 tok/s (TTFT 83.6 s) |
| 106,130-token prompt | 364 tok/s (TTFT 291.8 s) |
| 204,389-token prompt (kvtest) | ~347 tok/s (TTFT 589 s) |
| 254,388-token needle, 3 codes at 10/50/90% | **3/3** (TTFT 742.7 s, 343 tok/s) |
| decode at a ~115K-token context, 1 stream | 17.0 tok/s (TTFT 313 s; 250-token summary of filler) |
| decode at ~115K, 2 streams at once | not measured: an agent request arrived during it and was queued behind two long prefills, so it was stopped |

Interference: the endpoint served live agents during the run. Every measurement waited for an idle server and
was retried when other requests appeared (7 retries, 22.6 min waited); the one dirty cell (sql) was re-run clean.
