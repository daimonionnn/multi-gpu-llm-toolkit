# Qwen3.5-27B Claude 4.6 Opus Distilled Q4_K_M (~16 GiB)
param(
    [ValidateSet("rocm", "cuda", "vulkan", "rocm-cuda", "vulkan-vulkan", "vulkan-cuda")]
    [string]$Mode = "vulkan-vulkan"
)

# GGUF root; override with $env:LLM_MODELS_DIR. Default is LM Studio's download dir.
$ModelsDir = if ($env:LLM_MODELS_DIR) { $env:LLM_MODELS_DIR } else { "$env:USERPROFILE\.lmstudio\models" }

& "$PSScriptRoot\start-llama-server.ps1" `
    -Mode $Mode `
    -ExtraArgs @(
        "-m", "$ModelsDir\Jackrong\Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF\Qwen3.5-27B.Q4_K_M.gguf",
        "-c", "20000",
        "-ngl", "99",
        "--split-mode", "layer",
        "--tensor-split", "1,1",
        "--no-mmap",
        "--fit", "off",
        "--flash-attn", "auto",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--cache-ram", "4096",
        "--main-gpu", "1"
    )
