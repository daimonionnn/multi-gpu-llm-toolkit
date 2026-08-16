# Qwen3.5-122B-A10B Q6_K (~93 GiB) — Dual GPU profile
#
# Near the ~98 GiB combined GPU ceiling.
# Split 2:1 keeps CUDA0 under 32 GB dedicated VRAM (~30 GiB).
param(
    [ValidateSet("rocm", "cuda", "vulkan", "rocm-cuda", "vulkan-vulkan", "vulkan-cuda")]
    [string]$Mode = "rocm-cuda"
)

# GGUF root; override with $env:LLM_MODELS_DIR. Default is LM Studio's download dir.
$ModelsDir = if ($env:LLM_MODELS_DIR) { $env:LLM_MODELS_DIR } else { "$env:USERPROFILE\.lmstudio\models" }

& "$PSScriptRoot\start-llama-server.ps1" `
    -Mode $Mode `
    -ExtraArgs @(
        "-m", "$ModelsDir\mradermacher\Qwen3.5-122B-A10B-GGUF\Qwen3.5-122B-A10B-heretic-v2.Q6_K.gguf",
        "-c", "4096",
        "-ngl", "99",
        "--split-mode", "layer",
        "--tensor-split", "2,1",
        "--no-mmap",
        "--fit", "off",
        "--flash-attn", "auto",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0"
    )
