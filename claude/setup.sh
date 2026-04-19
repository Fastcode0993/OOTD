#!/usr/bin/env bash
# =============================================================================
#  Personal Color Cloud — Oracle Linux 원클릭 설치 스크립트
#  지원: Oracle Linux 8 / 9  (RHEL 계열)
#
#  사용법:
#    sudo bash setup.sh
#    sudo bash setup.sh --domain example.com
#    sudo bash setup.sh --domain example.com --port 5000 --kiosk-key MY_SECRET
#
#  설치 항목:
#    - Node.js 22.x LTS
#    - MongoDB 8.0
#    - nginx (리버스 프록시)
#    - backend npm 패키지
#    - frontend React 프로덕션 빌드
#    - systemd 서비스 (personal-color-backend)
#    - firewalld 포트 개방 (80, 443, 5000)
# =============================================================================
set -euo pipefail

# ── 색상 출력 ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${GREEN}[✔]${RESET} $*"; }
info() { echo -e "${BLUE}[•]${RESET} $*"; }
warn() { echo -e "${YELLOW}[!]${RESET} $*"; }
err()  { echo -e "${RED}[✘]${RESET} $*" >&2; exit 1; }
step() { echo -e "\n${BOLD}${CYAN}━━━ $* ━━━${RESET}"; }

# ── 스크립트 위치 기준 경로 ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
FRONTEND_BUILD_DIR="$FRONTEND_DIR/build"
SERVE_DIR="/var/www/personal-color"

# ── 기본값 ────────────────────────────────────────────────────────────────────
DOMAIN=""
BACKEND_PORT=5000
KIOSK_API_KEY="kiosk-$(openssl rand -hex 8)"
MONGODB_URI="mongodb://localhost:27017/personal_color"
NODE_VERSION="22"
MONGODB_VERSION="8.0"
SERVICE_NAME="personal-color-backend"
APP_USER="personal-color"

# ── 인수 파싱 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain)     DOMAIN="$2";        shift 2 ;;
    --port)       BACKEND_PORT="$2";  shift 2 ;;
    --kiosk-key)  KIOSK_API_KEY="$2"; shift 2 ;;
    --mongo-uri)  MONGODB_URI="$2";   shift 2 ;;
    --help|-h)
      grep '^#  ' "$0" | sed 's/^#  //'
      exit 0 ;;
    *) err "알 수 없는 옵션: $1  (--help 로 도움말 확인)" ;;
  esac
done

# =============================================================================
#  0. 사전 확인
# =============================================================================
step "사전 확인"

# root 확인
[[ "$EUID" -eq 0 ]] || err "root 권한이 필요합니다: sudo bash $0"

# Oracle Linux 버전 확인
if [[ -f /etc/oracle-release ]]; then
  OL_VERSION=$(grep -oP '\d+' /etc/oracle-release | head -1)
  log "Oracle Linux $OL_VERSION 감지됨"
elif [[ -f /etc/redhat-release ]]; then
  OL_VERSION=$(grep -oP '\d+' /etc/redhat-release | head -1)
  warn "RHEL 계열 $OL_VERSION — Oracle Linux로 취급합니다"
else
  err "Oracle Linux 8/9 이외의 OS는 지원하지 않습니다"
fi

[[ "$OL_VERSION" =~ ^(8|9)$ ]] || err "Oracle Linux 8 또는 9만 지원합니다 (감지: $OL_VERSION)"

# 디렉터리 확인
[[ -d "$BACKEND_DIR" ]]  || err "backend 폴더를 찾을 수 없습니다: $BACKEND_DIR"
[[ -d "$FRONTEND_DIR" ]] || err "frontend 폴더를 찾을 수 없습니다: $FRONTEND_DIR"

# 공인 IP 자동 감지 (domain 미지정 시)
if [[ -z "$DOMAIN" ]]; then
  PUBLIC_IP=$(curl -4 -sf --max-time 5 https://icanhazip.com \
           || curl -4 -sf --max-time 5 https://api.ipify.org \
           || echo "")
  if [[ -n "$PUBLIC_IP" ]]; then
    DOMAIN="$PUBLIC_IP"
    warn "도메인 미지정 — 공인 IP 사용: $DOMAIN"
    warn "도메인 사용 시: sudo bash setup.sh --domain example.com"
  else
    DOMAIN="localhost"
    warn "공인 IP 감지 실패 — localhost 사용"
  fi
fi

BASE_URL="http://$DOMAIN"
log "서비스 URL: $BASE_URL"
log "백엔드 포트: $BACKEND_PORT"

# =============================================================================
#  1. 시스템 패키지 업데이트 및 기본 도구 설치
# =============================================================================
step "시스템 패키지 업데이트"

dnf update -y -q
dnf install -y -q curl wget git tar openssl ca-certificates gnupg2 \
  policycoreutils-python-utils firewalld
log "기본 도구 설치 완료"

# =============================================================================
#  2. Node.js 22.x LTS 설치 (NodeSource)
# =============================================================================
step "Node.js $NODE_VERSION.x LTS 설치"

if command -v node &>/dev/null && node -e "process.exit(parseInt(process.version.slice(1)) >= $NODE_VERSION ? 0 : 1)" 2>/dev/null; then
  log "Node.js $(node -v) 이미 설치됨 — 건너뜀"
else
  info "NodeSource 저장소 등록 중..."
  curl -fsSL "https://rpm.nodesource.com/setup_${NODE_VERSION}.x" | bash - >/dev/null 2>&1
  dnf install -y -q nodejs
  log "Node.js $(node -v) / npm $(npm -v) 설치 완료"
fi

# =============================================================================
#  3. MongoDB 8.0 설치
# =============================================================================
step "MongoDB $MONGODB_VERSION 설치"

if systemctl is-active --quiet mongod 2>/dev/null; then
  log "MongoDB 이미 실행 중 — 건너뜀"
else
  # MongoDB 저장소 등록
  cat > /etc/yum.repos.d/mongodb-org-${MONGODB_VERSION}.repo << EOF
[mongodb-org-${MONGODB_VERSION}]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/redhat/${OL_VERSION}/mongodb-org/${MONGODB_VERSION}/x86_64/
gpgcheck=1
enabled=1
gpgkey=https://pgp.mongodb.com/server-${MONGODB_VERSION}.asc
EOF

  dnf install -y -q mongodb-org
  log "MongoDB $MONGODB_VERSION 설치 완료"

  # SELinux 허용 (OL 기본값)
  if command -v semanage &>/dev/null && getenforce | grep -qi enforcing; then
    semanage port -a -t mongod_port_t -p tcp 27017 2>/dev/null || true
  fi

  systemctl enable --now mongod
  # 기동 대기
  for i in $(seq 1 15); do
    mongosh --eval "db.runCommand({ping:1})" --quiet &>/dev/null && break
    sleep 1
  done
  log "MongoDB 서비스 시작 완료"
fi

# =============================================================================
#  4. nginx 설치
# =============================================================================
step "nginx 설치"

if ! command -v nginx &>/dev/null; then
  dnf install -y -q nginx
  log "nginx 설치 완료"
else
  log "nginx $(nginx -v 2>&1 | grep -oP '[\d.]+') 이미 설치됨"
fi

# =============================================================================
#  5. 전용 시스템 유저 생성
# =============================================================================
step "앱 전용 유저 생성"

if ! id "$APP_USER" &>/dev/null; then
  useradd --system --no-create-home --shell /sbin/nologin "$APP_USER"
  log "유저 '$APP_USER' 생성 완료"
else
  log "유저 '$APP_USER' 이미 존재"
fi

# =============================================================================
#  6. backend .env 생성
# =============================================================================
step "backend 환경설정 (.env)"

BACKEND_ENV="$BACKEND_DIR/.env"
if [[ ! -f "$BACKEND_ENV" ]]; then
  cat > "$BACKEND_ENV" << EOF
# Personal Color Cloud — Backend 환경설정
# 자동 생성: $(date '+%Y-%m-%d %H:%M:%S')

MONGODB_URI=$MONGODB_URI
PORT=$BACKEND_PORT
NODE_ENV=production
CLIENT_URL=$BASE_URL
KIOSK_API_KEY=$KIOSK_API_KEY
EOF
  log "backend .env 생성 완료"
else
  warn "backend .env 이미 존재 — 덮어쓰지 않음"
fi

# =============================================================================
#  7. backend npm 패키지 설치
# =============================================================================
step "backend npm 패키지 설치"

cd "$BACKEND_DIR"
npm ci --omit=dev --prefer-offline 2>/dev/null || npm install --omit=dev
log "backend npm 패키지 설치 완료"

# =============================================================================
#  8. frontend .env 생성 및 프로덕션 빌드
# =============================================================================
step "frontend 빌드"

FRONTEND_ENV="$FRONTEND_DIR/.env"
if [[ ! -f "$FRONTEND_ENV" ]]; then
  cat > "$FRONTEND_ENV" << EOF
# Personal Color Cloud — Frontend 환경설정
# 자동 생성: $(date '+%Y-%m-%d %H:%M:%S')

REACT_APP_API_URL=$BASE_URL/api
REACT_APP_BASE_URL=$BASE_URL
EOF
  log "frontend .env 생성 완료"
else
  warn "frontend .env 이미 존재 — 덮어쓰지 않음"
fi

cd "$FRONTEND_DIR"
npm ci --prefer-offline 2>/dev/null || npm install
info "React 프로덕션 빌드 시작 (2~5분 소요)..."
npm run build
log "React 빌드 완료: $FRONTEND_BUILD_DIR"

# 정적 파일 배포 디렉터리로 복사
mkdir -p "$SERVE_DIR"
cp -r "$FRONTEND_BUILD_DIR/." "$SERVE_DIR/"
chown -R nginx:nginx "$SERVE_DIR"
chmod -R 755 "$SERVE_DIR"
log "정적 파일 배포 완료: $SERVE_DIR"

# SELinux 컨텍스트 설정 (nginx가 파일 읽기 가능하도록)
if command -v restorecon &>/dev/null; then
  restorecon -Rv "$SERVE_DIR" &>/dev/null || true
fi

# =============================================================================
#  9. nginx 설정
# =============================================================================
step "nginx 리버스 프록시 설정"

NGINX_CONF="/etc/nginx/conf.d/personal-color.conf"

cat > "$NGINX_CONF" << EOF
# Personal Color Cloud — nginx 설정
# 생성: $(date '+%Y-%m-%d %H:%M:%S')

server {
    listen       80;
    server_name  $DOMAIN;
    charset      utf-8;

    # 정적 파일 (React 빌드)
    root  $SERVE_DIR;
    index index.html;

    # gzip 압축
    gzip on;
    gzip_types text/plain text/css application/json application/javascript
               text/xml application/xml image/svg+xml;
    gzip_min_length 1024;

    # 보안 헤더
    add_header X-Frame-Options       "SAMEORIGIN"  always;
    add_header X-Content-Type-Options "nosniff"    always;
    add_header X-XSS-Protection      "1; mode=block" always;
    add_header Referrer-Policy       "strict-origin-when-cross-origin" always;

    # API 리버스 프록시
    location /api/ {
        proxy_pass         http://127.0.0.1:$BACKEND_PORT;
        proxy_http_version 1.1;
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_set_header   X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;
        proxy_read_timeout 30s;
        proxy_connect_timeout 10s;
        client_max_body_size 10m;
    }

    # React SPA — 모든 경로를 index.html로 fallback
    location / {
        try_files \$uri \$uri/ /index.html;
    }

    # 헬스체크 (로드밸런서용)
    location /health {
        proxy_pass http://127.0.0.1:$BACKEND_PORT/api/health;
    }
}
EOF

# 기본 welcome 페이지 비활성화
if [[ -f /etc/nginx/conf.d/default.conf ]]; then
  mv /etc/nginx/conf.d/default.conf /etc/nginx/conf.d/default.conf.disabled 2>/dev/null || true
fi

nginx -t
log "nginx 설정 완료"

# SELinux: nginx → localhost 프록시 허용
setsebool -P httpd_can_network_connect 1 2>/dev/null || true

# =============================================================================
#  10. systemd 서비스 생성 (backend)
# =============================================================================
step "systemd 서비스 등록 ($SERVICE_NAME)"

cat > "/etc/systemd/system/$SERVICE_NAME.service" << EOF
[Unit]
Description=Personal Color Cloud Backend (Node.js)
After=network.target mongod.service
Requires=mongod.service

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$BACKEND_DIR
ExecStart=$(command -v node) server.js
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# 보안 강화
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$BACKEND_DIR
PrivateTmp=true

# 환경변수
EnvironmentFile=$BACKEND_ENV
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
EOF

# 파일 권한 설정
chown -R "$APP_USER":root "$BACKEND_DIR"
chmod 640 "$BACKEND_ENV"

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
log "systemd 서비스 등록 완료"

# =============================================================================
#  11. 방화벽 설정
# =============================================================================
step "방화벽 설정 (firewalld)"

systemctl enable --now firewalld

# HTTP/HTTPS 개방
firewall-cmd --permanent --add-service=http  --quiet
firewall-cmd --permanent --add-service=https --quiet

# 백엔드 포트는 외부 직접 노출 안 함 (nginx 경유만 허용)
# 내부 접근만 필요하므로 별도 개방 불필요

firewall-cmd --reload --quiet
log "방화벽 설정 완료 (80/443 개방, $BACKEND_PORT 내부 전용)"

# =============================================================================
#  12. 서비스 시작
# =============================================================================
step "서비스 시작"

systemctl start "$SERVICE_NAME"
systemctl reload nginx || systemctl start nginx

# 기동 대기 (최대 15초)
info "백엔드 기동 확인 중..."
STARTED=false
for i in $(seq 1 15); do
  if curl -sf "http://127.0.0.1:$BACKEND_PORT/api/health" &>/dev/null; then
    STARTED=true
    break
  fi
  sleep 1
done

if $STARTED; then
  log "백엔드 서비스 정상 기동 확인"
else
  warn "백엔드 헬스체크 응답 없음 — 로그 확인: journalctl -u $SERVICE_NAME -n 30"
fi

# =============================================================================
#  13. 설치 완료 요약
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  Personal Color Cloud 설치 완료!${RESET}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${CYAN}웹 앱${RESET}        : ${BOLD}$BASE_URL${RESET}"
echo -e "  ${CYAN}API 서버${RESET}     : ${BOLD}$BASE_URL/api${RESET}"
echo -e "  ${CYAN}헬스체크${RESET}     : ${BOLD}$BASE_URL/health${RESET}"
echo ""
echo -e "  ${CYAN}키오스크 API 키${RESET}: ${BOLD}$KIOSK_API_KEY${RESET}"
echo -e "  ${YELLOW}  → RaspberryPi .env의 KIOSK_API_KEY 에 동일하게 설정하세요${RESET}"
echo ""
echo -e "  ${CYAN}서비스 관리${RESET}:"
echo -e "    systemctl status  $SERVICE_NAME"
echo -e "    systemctl restart $SERVICE_NAME"
echo -e "    journalctl -u $SERVICE_NAME -f"
echo ""
echo -e "  ${CYAN}재배포${RESET}:"
echo -e "    git pull && sudo bash $SCRIPT_DIR/setup.sh"
echo ""
echo -e "  ${YELLOW}[HTTPS 설정]${RESET} 도메인 연결 후 Let's Encrypt 인증서 발급:"
echo -e "    sudo dnf install -y certbot python3-certbot-nginx"
echo -e "    sudo certbot --nginx -d $DOMAIN"
echo ""
