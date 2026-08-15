# MiniMax-M2.5 MXFP4 (~102 GiB) — Dual GPU profile
#
# Model is 101.76 GiB. AMD has ~95 GiB usable, NVIDIA ~31 GiB.
# Ratio ~3:1 puts ~76 GiB on AMD, ~25 GiB on NVIDIA.
param(
    [ValidateSet("rocm", "cuda", "vulkan", "rocm-cuda", "vulkan-vulkan", "vulkan-cuda")]
    [string]$Mode = "vulkan-vulkan"
)

# GGUF root; override with $env:LLM_MODELS_DIR. Default is LM Studio's download dir.
$ModelsDir = if ($env:LLM_MODELS_DIR) { $env:LLM_MODELS_DIR } else { "$env:USERPROFILE\.lmstudio\models" }

& "$PSScriptRoot\start-llama-server.ps1" `
    -Mode $Mode `
    -ExtraArgs @(
        "-m", "$ModelsDir\unsloth\MiniMax-M2.5\MiniMax-M2.5-MXFP4_MOE-00001-of-00004.gguf",
        "-c", "100000",
        "-ngl", "99",
        "--split-mode", "layer",
        "--tensor-split", "3,1",
        "--no-mmap",
        "--fit", "on",
        # "--fit-target", "500,500",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--main-gpu", "1",
        "--parallel", "1"
    )
