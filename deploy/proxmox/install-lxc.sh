#!/usr/bin/env bash
# ============================================================================
# Proxmox LXC / Debian / Ubuntu 24/7 Installer for Antigravity Google Home Bridge
# ============================================================================
set -euo pipefail

INSTALL_DIR="/opt/antigravity-google-home-bridge"
WORKSPACE_DIR="/opt/antigravity-workspace"
DATA_DIR="/var/lib/agy-home-bridge"

echo "==> [1/5] Installing system dependencies (Python 3, Avahi mDNS, PortAudio/ALSA)..."
apt-get update
apt-get install -y python3 python3-venv python3-pip git curl avahi-daemon avahi-utils libportaudio2 libsndfile1

echo "==> [2/5] Preparing directories..."
mkdir -p "$INSTALL_DIR" "$WORKSPACE_DIR" "$DATA_DIR/conversations" "$DATA_DIR/brain"

if [ ! -f "$INSTALL_DIR/pyproject.toml" ]; then
  if [ -f "./pyproject.toml" ]; then
    cp -R . "$INSTALL_DIR/"
  else
    git clone https://github.com/mavrikjc/antigravity-google-home-bridge.git "$INSTALL_DIR"
  fi
fi

cd "$INSTALL_DIR"

echo "==> [3/5] Creating Python virtual environment and installing Google Antigravity SDK & Bridge..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[antigravity,hardware-mic]" || pip install -e ".[hardware-mic]"

echo "==> [4/5] Configuring .env..."
if [ ! -f "$INSTALL_DIR/.env" ]; then
  cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
  LAN_IP=$(hostname -I | awk '{print $1}')
  sed -i "s|^HOST_LAN_IP=.*|HOST_LAN_IP=${LAN_IP}|" "$INSTALL_DIR/.env" || true
  echo "Created $INSTALL_DIR/.env (Detected LAN IP: ${LAN_IP})"
  echo "IMPORTANT: Edit $INSTALL_DIR/.env and set your GEMINI_API_KEY."
fi

echo "==> [5/5] Installing and starting systemd service (agy-home-bridge.service)..."
cp "$INSTALL_DIR/deploy/proxmox/agy-home-bridge.service" /etc/systemd/system/agy-home-bridge.service
systemctl daemon-reload
systemctl enable --now agy-home-bridge.service

LAN_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "============================================================================"
echo " Antigravity Google Home Bridge is now running 24/7!"
echo " - Live Voice PWA:             http://${LAN_IP}:8765"
echo " - MCP Server Endpoint:        http://${LAN_IP}:8765/mcp"
echo " - Google Home Fulfillment:    http://${LAN_IP}:8765/api/google-home/fulfillment"
echo " - Check service logs:         journalctl -u agy-home-bridge -f"
echo "============================================================================"
