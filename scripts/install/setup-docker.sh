#!/usr/bin/env bash
# EquiLens Docker Setup Script
# One-command installation: curl -fsSL https://raw.githubusercontent.com/Life-Experimentalist/EquiLens/main/setup-docker.sh | bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== EquiLens Docker Setup ===${NC}"
echo ""

# Check if Docker is installed
echo -e "${YELLOW}[1/8] Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED} Docker is not installed!${NC}"
    echo -e "${YELLOW}Please install Docker from: https://docs.docker.com/get-docker/${NC}"
    echo -e "${YELLOW}After installation, restart this script.${NC}"
    exit 1
fi

# Check if Docker is running
echo -e "${YELLOW}[2/8] Checking if Docker is running...${NC}"
if ! docker ps &> /dev/null; then
    echo -e "${RED} Docker is not running!${NC}"
    echo -e "${YELLOW}Please start Docker and try again.${NC}"
    exit 1
fi
echo -e "${GREEN} Docker is running${NC}"

# Check if docker-compose is available
echo -e "${YELLOW}[3/8] Checking docker-compose...${NC}"
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED} docker-compose is not available!${NC}"
    echo -e "${YELLOW}Please ensure Docker Compose is installed.${NC}"
    exit 1
fi
echo -e "${GREEN} docker-compose is available${NC}"

# Clone or download EquiLens repository
echo -e "${YELLOW}[4/8] Setting up EquiLens...${NC}"
REPO_URL="https://github.com/Life-Experimentalist/EquiLens"
TARGET_DIR="EquiLens"

if [ -d "$TARGET_DIR" ]; then
    echo -e "${YELLOW}Directory already exists. Pulling latest changes...${NC}"
    cd "$TARGET_DIR"
    git pull 2>/dev/null || true
    cd ..
else
    echo -e "${YELLOW}Cloning EquiLens repository...${NC}"
    git clone "$REPO_URL" "$TARGET_DIR" 2>/dev/null || true
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${YELLOW}Creating directory structure...${NC}"
    mkdir -p "$TARGET_DIR"
    cd "$TARGET_DIR"

    # Download docker-compose.yml
    COMPOSE_URL="$REPO_URL/raw/main/docker-compose.yml"
    curl -fsSL "$COMPOSE_URL" -o "docker-compose.yml" 2>/dev/null || true

    # Download Dockerfile
    DOCKERFILE_URL="$REPO_URL/raw/main/Dockerfile"
    curl -fsSL "$DOCKERFILE_URL" -o "Dockerfile" 2>/dev/null || true
else
    cd "$TARGET_DIR"
fi

echo -e "${GREEN} EquiLens setup complete${NC}"

# Create persistent volumes
echo -e "${YELLOW}[5/8] Creating Docker volumes...${NC}"

# Check if user already has an Ollama volume
echo -e "${GRAY}  Checking for existing Ollama volumes...${NC}"
EXISTING_OLLAMA=$(docker volume ls --format "{{.Name}}" | grep -i "ollama" || true)

if [ -n "$EXISTING_OLLAMA" ]; then
    echo -e "${CYAN}  Found existing Ollama volume(s):${NC}"
    echo "$EXISTING_OLLAMA" | while read vol; do
        echo -e "${NC}    - $vol${NC}"
    done
    echo ""
    echo -e "${YELLOW}  You can reuse an existing Ollama volume to keep your models.${NC}"
    echo -e "${YELLOW}  Edit docker-compose.yml and set 'external: true' for ollama_models volume.${NC}"
    echo -e "${YELLOW}  See docs/DOCKER_SETUP.md for details.${NC}"
    echo ""
fi

# Create EquiLens-specific volumes and new Ollama volume if needed
VOLUMES=(
    "ollama-models"
    "equilens-data"
    "equilens-results"
    "equilens-logs"
)

for vol in "${VOLUMES[@]}"; do
    if docker volume ls --format "{{.Name}}" | grep -q "^${vol}$"; then
        echo -e "${GRAY}  Volume already exists: $vol${NC}"
    else
        docker volume create "$vol" &> /dev/null
        echo -e "${GRAY}  Created volume: $vol${NC}"
    fi
done
echo -e "${GREEN} Volumes created${NC}"

# Pull Docker images
echo -e "${YELLOW}[6/8] Pulling Docker images (this may take a while)...${NC}"
echo -e "${GRAY}  Pulling Ollama...${NC}"
docker pull ollama/ollama:latest

echo -e "${GREEN} Images pulled${NC}"

# Start services
echo -e "${YELLOW}[7/8] Starting EquiLens services...${NC}"
if command -v docker-compose &> /dev/null; then
    docker-compose up -d
else
    docker compose up -d
fi

# Wait for services to be healthy
echo -e "${GRAY}  Waiting for Ollama to be ready...${NC}"
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -f http://localhost:11434/api/tags -s -o /dev/null 2>&1; then
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

echo -e "${GREEN} Services started${NC}"

# Pull default Ollama model
echo -e "${YELLOW}[8/8] Downloading default model (llama3.2:latest)...${NC}"
docker exec equilens-ollama ollama pull llama3.2:latest

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo -e "${CYAN}Services are running at:${NC}"
echo -e "  ${NC} Gradio UI:  http://localhost:7860${NC}"
echo -e "  ${NC} Web API:    http://localhost:8000${NC}"
echo -e "  ${NC} Ollama API: http://localhost:11434${NC}"
echo ""
echo -e "${CYAN}Useful commands:${NC}"
echo -e "${GRAY}  docker-compose logs -f          # View logs${NC}"
echo -e "${GRAY}  docker-compose ps               # Check status${NC}"
echo -e "${GRAY}  docker-compose down             # Stop services${NC}"
echo -e "${GRAY}  docker-compose up -d            # Start services${NC}"
echo -e "${GRAY}  docker exec -it equilens-app bash  # Access container${NC}"
echo ""
echo -e "${CYAN}To add more models:${NC}"
echo -e "${GRAY}  docker exec equilens-ollama ollama pull <model-name>${NC}"
echo ""

cd ..
