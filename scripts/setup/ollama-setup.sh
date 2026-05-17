#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  ollama-setup.sh — Idempotent Ollama Docker setup
#  Works on: WSL2, Linux, macOS
#  Usage:
#    ./ollama-setup.sh                        # default setup
#    ./ollama-setup.sh --force-recreate       # tear down + rebuild
#    ./ollama-setup.sh --pull-latest          # always re-pull image
#    ./ollama-setup.sh --port 12000           # custom port
#    ./ollama-setup.sh --name my-ollama       # custom container name
#    ./ollama-setup.sh --volume my-vol        # custom volume name
# ─────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ─────────────────────────────────
CONTAINER_NAME="ollama"
VOLUME_NAME="ollama_data"
PORT=11434
FORCE_RECREATE=false
PULL_LATEST=false

# ── Parse args ───────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)            CONTAINER_NAME="$2"; shift 2 ;;
        --volume)          VOLUME_NAME="$2";    shift 2 ;;
        --port)            PORT="$2";           shift 2 ;;
        --force-recreate)  FORCE_RECREATE=true; shift   ;;
        --pull-latest)     PULL_LATEST=true;    shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Colors ───────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; GRAY='\033[0;37m'; MAGENTA='\033[0;35m'
NC='\033[0m'

step() { echo -e "\n${CYAN}► $1${NC}"; }
ok()   { echo -e "  ${GREEN}✓ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "  ${RED}✗ $1${NC}"; exit 1; }
info() { echo -e "  ${GRAY}· $1${NC}"; }

# ── Banner ───────────────────────────────────
echo ""
echo -e "${MAGENTA}  ╔═══════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}  ║     🦙  Ollama Docker Setup           ║${NC}"
echo -e "${MAGENTA}  ║     Volume-backed · GPU-aware         ║${NC}"
echo -e "${MAGENTA}  ╚═══════════════════════════════════════╝${NC}"
echo ""
info "Container : $CONTAINER_NAME"
info "Volume    : $VOLUME_NAME"
info "Port      : $PORT → 11434"
echo ""

# ─────────────────────────────────────────────
#  Step 1 — Docker check
# ─────────────────────────────────────────────
step "Checking Docker..."

if ! command -v docker &>/dev/null; then
    fail "Docker is not installed. Visit https://docs.docker.com/get-docker/"
fi

if ! docker info &>/dev/null; then
    fail "Docker daemon is not running. Start Docker and try again."
fi

ok "Docker is running"

# ─────────────────────────────────────────────
#  Step 2 — GPU detection
# ─────────────────────────────────────────────
step "Detecting GPU..."

GPU_ARGS=()
GPU_LABEL="CPU only"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    if [[ -n "$GPU_NAME" ]]; then
        RUNTIME_CHECK=$(docker info --format "{{json .Runtimes}}" 2>/dev/null || echo "")
        if echo "$RUNTIME_CHECK" | grep -q "nvidia"; then
            GPU_ARGS=("--gpus" "all")
            GPU_LABEL="NVIDIA GPU ($GPU_NAME) — passthrough enabled"
            ok "$GPU_LABEL"
        else
            warn "NVIDIA GPU found ($GPU_NAME) but Docker NVIDIA runtime not detected."
            info "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
            info "Continuing with CPU mode."
        fi
    else
        info "No NVIDIA GPU detected — CPU mode."
    fi
else
    info "nvidia-smi not found — CPU mode."
fi

# ─────────────────────────────────────────────
#  Step 3 — Pull image
# ─────────────────────────────────────────────
step "Checking Ollama image..."

IMAGE_EXISTS=$(docker images ollama/ollama:latest --format "{{.Repository}}" 2>/dev/null || echo "")

if [[ -z "$IMAGE_EXISTS" ]] || [[ "$PULL_LATEST" == "true" ]]; then
    info "Pulling ollama/ollama:latest..."
    docker pull ollama/ollama:latest || fail "Image pull failed. Check your internet connection."
    ok "Image pulled"
else
    ok "Image ollama/ollama:latest already present (use --pull-latest to update)"
fi

# ─────────────────────────────────────────────
#  Step 4 — Volume
# ─────────────────────────────────────────────
step "Setting up Docker volume..."

if docker volume ls --filter "name=^${VOLUME_NAME}$" --format "{{.Name}}" | grep -q "^${VOLUME_NAME}$"; then
    ok "Volume '$VOLUME_NAME' already exists — models will be reused"
else
    docker volume create "$VOLUME_NAME" >/dev/null
    ok "Volume '$VOLUME_NAME' created"
fi

VOLUME_PATH=$(docker volume inspect "$VOLUME_NAME" --format "{{.Mountpoint}}" 2>/dev/null)
info "Host path: $VOLUME_PATH"

# ─────────────────────────────────────────────
#  Step 5 — Handle existing container
# ─────────────────────────────────────────────
step "Checking for existing container..."

SKIP_CREATE=false
CONTAINER_STATE=$(docker inspect "$CONTAINER_NAME" --format "{{.State.Status}}" 2>/dev/null || echo "gone")

if [[ "$CONTAINER_STATE" != "gone" ]]; then
    if [[ "$FORCE_RECREATE" == "true" ]]; then
        warn "ForceRecreate: removing container '$CONTAINER_NAME'..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm   "$CONTAINER_NAME" 2>/dev/null || true
        ok "Old container removed (volume '$VOLUME_NAME' preserved)"
        CONTAINER_STATE="gone"
    else
        info "Container '$CONTAINER_NAME' exists (state: $CONTAINER_STATE)"
        if [[ "$CONTAINER_STATE" == "running" ]]; then
            ok "Container is already running — nothing to do"
            SKIP_CREATE=true
        elif [[ "$CONTAINER_STATE" =~ ^(exited|created|paused)$ ]]; then
            info "Starting stopped container..."
            docker start "$CONTAINER_NAME" >/dev/null
            ok "Container started"
            SKIP_CREATE=true
        fi
    fi
fi

# ─────────────────────────────────────────────
#  Step 6 — Create container
# ─────────────────────────────────────────────
if [[ "$SKIP_CREATE" == "false" ]]; then
    step "Creating Ollama container..."

    docker run -d \
        --name    "$CONTAINER_NAME" \
        --restart unless-stopped \
        -p        "${PORT}:11434" \
        -v        "${VOLUME_NAME}:/root/.ollama" \
        -e        "OLLAMA_HOST=0.0.0.0" \
        -e        "OLLAMA_MODELS=/root/.ollama/models" \
        "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" \
        ollama/ollama:latest >/dev/null

    ok "Container '$CONTAINER_NAME' created and started"
    info "GPU mode   : $GPU_LABEL"
    info "Volume     : $VOLUME_NAME → /root/.ollama"
    info "API port   : $PORT"
    info "Restart    : unless-stopped"
fi

# ─────────────────────────────────────────────
#  Step 7 — Health check
# ─────────────────────────────────────────────
step "Waiting for Ollama API to be ready..."

MAX=20; i=0; READY=false
while [[ $i -lt $MAX ]]; do
    ((i++))
    sleep 2
    if curl -sf "http://localhost:${PORT}" >/dev/null 2>&1; then
        READY=true
        break
    fi
    printf "\r  · Waiting... (%s/%s)" "$i" "$MAX"
done
echo ""

if [[ "$READY" == "true" ]]; then
    ok "Ollama API is healthy at http://localhost:${PORT}"
else
    warn "API did not respond within timeout — may still be starting."
    info "Check with: docker logs $CONTAINER_NAME"
fi

# ─────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────
echo ""
echo -e "${MAGENTA}  ════════════════════════════════════════${NC}"
echo -e "${GREEN}  🦙  Ollama is ready!${NC}"
echo -e "${MAGENTA}  ════════════════════════════════════════${NC}"
echo ""
echo -e "  API endpoint   : ${CYAN}http://localhost:${PORT}${NC}"
echo -e "  Models volume  : ${CYAN}${VOLUME_NAME}${NC}"
echo -e "  GPU mode       : ${CYAN}${GPU_LABEL}${NC}"
echo ""
echo -e "${GRAY}  ─── Quick commands ──────────────────────${NC}"
echo -e "  Pull a model   : ${YELLOW}docker exec -it ${CONTAINER_NAME} ollama pull llama3.2${NC}"
echo -e "  List models    : ${YELLOW}docker exec -it ${CONTAINER_NAME} ollama list${NC}"
echo -e "  Chat (REPL)    : ${YELLOW}docker exec -it ${CONTAINER_NAME} ollama run llama3.2${NC}"
echo -e "  View logs      : ${YELLOW}docker logs -f ${CONTAINER_NAME}${NC}"
echo -e "  Stop container : ${YELLOW}docker stop ${CONTAINER_NAME}${NC}"
echo -e "  Backup models  : ${YELLOW}docker run --rm -v ${VOLUME_NAME}:/data -v \$(pwd):/out alpine tar czf /out/ollama_backup.tar.gz -C /data .${NC}"
echo ""
