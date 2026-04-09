$ErrorActionPreference = "Stop"

function Install-WithWinget {
    param(
        [string]$Id,
        [string]$Name
    )
    Write-Host "Installing $Name..." -ForegroundColor Cyan
    winget install --id $Id --silent --accept-package-agreements --accept-source-agreements
}

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "winget is required for this script."
}

Install-WithWinget -Id "Git.Git" -Name "Git"
Install-WithWinget -Id "Kitware.CMake" -Name "CMake"
Install-WithWinget -Id "Ninja-build.Ninja" -Name "Ninja"
Install-WithWinget -Id "LunarG.VulkanSDK" -Name "Vulkan SDK"
Install-WithWinget -Id "Docker.DockerDesktop" -Name "Docker Desktop"

Write-Host "Attempting Visual Studio Build Tools install (C++ workload)..." -ForegroundColor Cyan
winget install --id Microsoft.VisualStudio.2022.BuildTools --accept-package-agreements --accept-source-agreements --override "--wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

Write-Host "If VS Build Tools fails, open Visual Studio Installer and install: Desktop development with C++" -ForegroundColor Yellow
Write-Host "After install, open 'Developer PowerShell for VS 2022', cd here, and run scripts/setup-llama.ps1" -ForegroundColor Yellow
