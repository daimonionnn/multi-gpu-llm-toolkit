param(
    [string]$ContainerName = "open-webui",
    [int]$Port = 3000,
    [string]$ApiBase = "http://host.docker.internal:8080/v1"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is not installed or not in PATH."
}

$exists = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $ContainerName }
if ($exists) {
    docker rm -f $ContainerName | Out-Null
}

docker run -d `
    -p "${Port}:8080" `
    --add-host=host.docker.internal:host-gateway `
    -e OPENAI_API_KEY="sk-local" `
    -e OPENAI_API_BASE_URL="$ApiBase" `
    -v open-webui:/app/backend/data `
    --name $ContainerName `
    --restart unless-stopped `
    ghcr.io/open-webui/open-webui:main | Out-Null

Write-Host "Open WebUI is starting on http://localhost:$Port" -ForegroundColor Green
Write-Host "Preconfigured API base: $ApiBase" -ForegroundColor Green
Write-Host "If models do not appear, set OpenAI endpoint manually in Open WebUI Admin settings." -ForegroundColor Yellow
