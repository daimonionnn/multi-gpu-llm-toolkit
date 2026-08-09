# Benchmarks

Results are grouped by **rig**, because the two test machines are not
comparable hardware — see [systems.md](systems.md). A number from `halo-win`
says nothing about what `dual-linux` will do, and vice versa. Treat any
cross-rig table as "two different machines", never as "Windows vs Linux".

Always record alongside a result: rig, backend mode, model and quant, context
size, `--tensor-split`, and any relevant BIOS/driver setting.

## Understanding the metrics

When looking at results (especially inside `benchmark.log`), you will see the following key metrics:

* **`PromptTokens` (Prefill Tokens):** The number of tokens in the initial input prompt sent to the model. The model processing these tokens all at once is known as the "prefill" phase.
* **`PredictTokens` (Generated Tokens):** The number of new tokens the model generated in response to the prompt. Generating these one by one is known as the "decoding" phase.
* **`GenTokPerSec` (Generation Speed):** This is the raw generation speed reported by the backend server. It calculates how fast tokens are generated *only during the decoding phase*. It completely excludes the time spent processing the initial prompt (prefill) and any network latency.
* **`TotalTokPerSec` (Total Throughput):** This represents the *end-to-end* throughput from the client's perspective (`Total Generated Tokens / Total Wall Time`). Because the wall time includes the HTTP overhead, the potentially long prompt processing (prefill) phase, and the actual generation phase, this number will always be lower than `GenTokPerSec`. Even when running with `--parallel 1` (a single request), `TotalTokPerSec` will be lower because the prompt prefill time drags down the overall average speed of the request lifecycle.

## Benchmarking tools

Two tools, measuring different things. Each exists per platform with the same
name and behaviour — `.ps1` under [../windows/scripts/](../windows/scripts/),
`.sh` under [../linux/scripts/](../linux/scripts/).

### 1. `run-llama-bench` — precise hardware testing

Uses the internal `llama-bench` tool. Best for measuring absolute maximum hardware capability, ideal parallel layer splits, and raw memory bandwidth. It executes the model locally in C++, completely bypassing HTTP, network latency, and the server queuing system. You can test parallel processing perfectly using batch sizes (`-b 1,2,4`).

* **Usage:** run the script directly. If `llama-server` is already running, it will automatically detect the parameters used (model path, layers, tensor splits, etc.).
* **Important:** If your model fills your VRAM, you should close `llama-server` before actually running the parsed command to avoid Out-Of-Memory (OOM) errors, as both programs will fight for the same GPU memory.

### 2. `benchmark-loaded-model` — real-world HTTP testing

Tests the true end-to-end experience of a client interacting with `llama-server` over the REST API. It accounts for web-server threading, connection handling, HTTP overhead, and the continuous KV-cache context switching that happens when multiple real clients hit the server simultaneously (`--parallel X`).

* **Usage:** Start your model server, then run the script. It will automatically detect the running `llama-server` to extract the mode and extra args, and log the results into `benchmark.log`.

---

## Results: `halo-win` (Strix Halo iGPU + RTX 5090, Windows)

Pending. To be measured across:

1. `rocm-cuda`
2. `vulkan-vulkan`
3. `vulkan-cuda`

| Model | Quant | Backend | Context | Tensor split | UMA | GenTok/s | TotalTok/s |
|-------|-------|---------|---------|--------------|-----|---------:|-----------:|
| —     | —     | —       | —       | —            | —   | —        | —          |

## Results: `dual-linux` (Radeon AI PRO R9700 + RTX PRO 6000, Linux)

Pending. The `rocm` and `vulkan` backends build; `rocm-cuda` and `vulkan-cuda`
are blocked by [cuda-glibc-243.md](cuda-glibc-243.md), so the first numbers here
will be `vulkan-vulkan` and single-GPU `rocm`.

Note the 3:1 VRAM asymmetry (96 GB NVIDIA vs 32 GB AMD): an even
`--tensor-split 1,1` leaves most of the RTX PRO 6000 idle, so the split column
matters more here than on `halo-win`.

| Model | Quant | Backend | Context | Tensor split | GenTok/s | TotalTok/s |
|-------|-------|---------|---------|--------------|---------:|-----------:|
| —     | —     | —       | —       | —            | —        | —          |
