#!/usr/bin/env bash
# =============================================================================
#  Personal Color Cloud — 업데이트/재배포 스크립트
#  setup.sh 이후 코드 변경사항을 반영할 때 사용
#
#  사용법: sudo bash update.sh
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${GREEN}[✔]${RESET} $*"; }
info() { echo -e "${CYAN}[•]${RESET} $*"; }
warn() { echo -e "${YELLOW}[!]${RESET} $*"; }

[[ "$EUID" -eq 0 ]] || { echo "root 필요: sudo bash $0"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
SERVE_DIR="/var/www/personal-color"
SERVICE_NAME="personal-color-backend"

# ── 1. git pull ───────────────────────────────────────────────────────────────
info "최신 코드 pull..."
cd "$SCRIPT_DIR"
git pull
log "코드 업데이트 완료"

# ── 2. backend 패키지 갱신 ────────────────────────────────────────────────────
info "backend 패키지 갱신..."
cd "$BACKEND_DIR"
npm ci --omit=dev --prefer-offline 2>/dev/null || npm install --omit=dev
log "backend 패키지 완료"

# ── 3. frontend 재빌드 ────────────────────────────────────────────────────────
info "frontend 빌드 중..."
cd "$FRONTEND_DIR"
npm ci --prefer-offline 2>/dev/null || npm install
npm run build
cp -r build/. "$SERVE_DIR/"
chown -R nginx:nginx "$SERVE_DIR"
restorecon -Rv "$SERVE_DIR" &>/dev/null || true
log "frontend 빌드 및 배포 완료"

# ── 4. 서비스 재시작 ──────────────────────────────────────────────────────────
systemctl restart "$SERVICE_NAME"
systemctl reload nginx
log "서비스 재시작 완료"

echo -e "\n${BOLD}${GREEN}업데이트 완료!${RESET}"
systemctl status "$SERVICE_NAME" --no-pager -l | head -5
