#!/usr/bin/env bash
# Open WebUI frontend pointed at a local llama-server.
# Linux port of start-open-webui.ps1.
#
# Usage: ./start-open-webui.sh [--port 3000] [--name open-webui] [--api-base URL]

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONTAINER_NAME="open-webui"
PORT=3000
# host.docker.internal needs --add-host on Linux; Docker Desktop provides it
# automatically on Windows, plain Docker Engine does not.
API_BASE="http://host.docker.internal:8080/v1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)     PORT="${2:?}"; shift 2 ;;
        --name)     CONTAINER_NAME="${2:?}"; shift 2 ;;
        --api-base) API_BASE="${2:?}"; shift 2 ;;
        -h|--help)  sed -n '2,6p' "$0"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

require_cmd docker

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    info "Removing existing container '$CONTAINER_NAME' ..."
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run -d \
    -p "${PORT}:8080" \
    --add-host=host.docker.internal:host-gateway \
    -e OPENAI_API_KEY="sk-local" \
    -e OPENAI_API_BASE_URL="$API_BASE" \
    -v open-webui:/app/backend/data \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    ghcr.io/open-webui/open-webui:main >/dev/null

ok "Open WebUI is starting on http://localhost:$PORT"
ok "Preconfigured API base: $API_BASE"
warn "If models do not appear, set the OpenAI endpoint manually in Open WebUI admin settings."
warn "Note: llama-server must listen on 0.0.0.0 (it does by default here) for the container to reach it."
