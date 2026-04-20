#!/usr/bin/env bash
# =============================================================================
#  Personal Color Cloud — Oracle Linux 원클릭 설치 스크립트
#  지원: Oracle Linux 8 / 9 / 10  (RHEL 계열)
#
#  사용법:
#    sudo bash setup.sh
#    sudo bash setup.sh --domain example.com --email admin@example.com
#    sudo bash setup.sh --domain example.com --email admin@example.com \
#                       --port 5000 --kiosk-key MY_SECRET
#    sudo bash setup.sh --domain example.com --skip-https   # HTTP 전용
#
#  설치 항목:
#    - Node.js 22.x LTS
#      · OL 8/9: NodeSource RPM 저장소
#      · OL 10 : AppStream 모듈 (nodejs:22) → NodeSource 폴백
#    - MongoDB 8.0  (el8 / el9 / el10 자동 분기)
#    - nginx (리버스 프록시)
#    - backend npm 패키지
#    - frontend React 프로덕션 빌드
#    - systemd 서비스 (personal-color-backend)
#    - firewalld 포트 개방 (80, 443)
#    - Let's Encrypt HTTPS 인증서 (실제 도메인 + --email 지정 시 자동 발급)
#      · certbot --nginx 로 nginx 설정 자동 수정
#      · systemd 타이머로 90일마다 자동 갱신
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
DOMAIN="personalootd.kro.kr"           # 기본 도메인 (--domain 으로 덮어쓰기 가능)
CERT_EMAIL="kimkimwoo1234@gmail.com"   # Let's Encrypt 이메일 (--email 으로 덮어쓰기 가능)
BACKEND_PORT=5000
KIOSK_API_KEY="kiosk-$(openssl rand -hex 8)"
MONGODB_URI="mongodb://localhost:27017/personal_color"
NODE_VERSION="22"
MONGODB_VERSION="8.0"
SERVICE_NAME="personal-color-backend"
APP_USER="personal-color"
SKIP_HTTPS=false    # --skip-https 시 HTTPS 발급 생략

# ── 인수 파싱 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain)      DOMAIN="$2";        shift 2 ;;
    --port)        BACKEND_PORT="$2";  shift 2 ;;
    --kiosk-key)   KIOSK_API_KEY="$2"; shift 2 ;;
    --mongo-uri)   MONGODB_URI="$2";   shift 2 ;;
    --email)       CERT_EMAIL="$2";    shift 2 ;;
    --skip-https)  SKIP_HTTPS=true;    shift ;;
    --help|-h)
      grep '^#  ' "$0" | sed 's/^#  //'
      exit 0 ;;
    *) err "알 수 없는 옵션: $1  (--help 로 도움말 확인)" ;;
  esac
done

# ── 도메인 유형 판별 (IP·localhost → HTTPS 불가) ─────────────────────────────
_is_real_domain() {
  local d="$1"
  [[ "$d" == "localhost" ]]                         && return 1
  [[ "$d" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]  && return 1  # IPv4
  [[ "$d" =~ : ]]                                   && return 1  # IPv6
  return 0
}

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
  err "Oracle Linux 8/9/10 이외의 OS는 지원하지 않습니다"
fi

[[ "$OL_VERSION" =~ ^(8|9|10)$ ]] || err "Oracle Linux 8, 9 또는 10만 지원합니다 (감지: $OL_VERSION)"

# 디렉터리 확인
[[ -d "$BACKEND_DIR" ]]  || err "backend 폴더를 찾을 수 없습니다: $BACKEND_DIR"
[[ -d "$FRONTEND_DIR" ]] || err "frontend 폴더를 찾을 수 없습니다: $FRONTEND_DIR"

# HTTPS 스킵이 아닌 경우 → https:// 로 빌드
if _is_real_domain "$DOMAIN" && ! $SKIP_HTTPS; then
  BASE_URL="https://$DOMAIN"
else
  BASE_URL="http://$DOMAIN"
fi

log "서비스 URL: $BASE_URL"
log "백엔드 포트: $BACKEND_PORT"

# =============================================================================
#  1. 시스템 패키지 업데이트 및 기본 도구 설치
# =============================================================================
step "시스템 패키지 업데이트"

info "패키지 목록 업데이트 중... (수분 소요될 수 있습니다)"
dnf update -y
log "패키지 업데이트 완료"

# OL 10은 python3-policycoreutils 패키지명으로 변경됨
if [[ "$OL_VERSION" -ge 10 ]]; then
  SEMANAGE_PKG="python3-policycoreutils"
else
  SEMANAGE_PKG="policycoreutils-python-utils"
fi

info "기본 도구 설치 중..."
dnf install -y curl wget git tar openssl ca-certificates gnupg2 \
  "$SEMANAGE_PKG" firewalld
log "기본 도구 설치 완료"

# =============================================================================
#  2. Node.js 22.x LTS 설치
#     OL 8/9 : NodeSource RPM 저장소
#     OL 10  : AppStream 모듈 (nodejs:22) → NodeSource 폴백
# =============================================================================
step "Node.js $NODE_VERSION.x LTS 설치"

_install_node_nodesource() {
  info "NodeSource 저장소 등록 중..."
  if curl -fsSL "https://rpm.nodesource.com/setup_${NODE_VERSION}.x" | bash - >/dev/null 2>&1; then
    dnf install -y nodejs
    return 0
  fi
  return 1
}

_install_node_appstream() {
  info "AppStream 모듈로 Node.js 설치 중 (nodejs:$NODE_VERSION)..."
  dnf module reset nodejs -y 2>/dev/null || true
  if dnf module enable "nodejs:${NODE_VERSION}" -y 2>/dev/null; then
    dnf install -y nodejs npm
    return 0
  fi
  return 1
}

if command -v node &>/dev/null && node -e "process.exit(parseInt(process.version.slice(1)) >= $NODE_VERSION ? 0 : 1)" 2>/dev/null; then
  log "Node.js $(node -v) 이미 설치됨 — 건너뜀"
elif [[ "$OL_VERSION" -ge 10 ]]; then
  # OL 10: AppStream 우선, 실패 시 NodeSource 폴백
  _install_node_appstream || _install_node_nodesource || err "Node.js 설치 실패"
  log "Node.js $(node -v) / npm $(npm -v) 설치 완료"
else
  _install_node_nodesource || err "Node.js 설치 실패"
  log "Node.js $(node -v) / npm $(npm -v) 설치 완료"
fi

# =============================================================================
#  3. MongoDB 8.0 설치
# =============================================================================
step "MongoDB $MONGODB_VERSION 설치"

if systemctl is-active --quiet mongod 2>/dev/null; then
  log "MongoDB 이미 실행 중 — 건너뜀"
else
  # OL 10 이상은 el10 저장소 사용, 그 이하는 버전 그대로
  MONGO_EL_VER="$OL_VERSION"
  # CPU 아키텍처 자동 감지 (x86_64 / aarch64)
  MONGO_ARCH=$(uname -m)
  log "서버 아키텍처: $MONGO_ARCH"

  # MongoDB 저장소 등록
  cat > /etc/yum.repos.d/mongodb-org-${MONGODB_VERSION}.repo << EOF
[mongodb-org-${MONGODB_VERSION}]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/redhat/${MONGO_EL_VER}/mongodb-org/${MONGODB_VERSION}/${MONGO_ARCH}/
gpgcheck=1
enabled=1
gpgkey=https://pgp.mongodb.com/server-${MONGODB_VERSION}.asc
EOF

  info "MongoDB $MONGODB_VERSION 설치 중..."
  dnf install -y mongodb-org
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
  dnf install -y nginx
  log "nginx 설치 완료"
else
  log "nginx $(nginx -v 2>&1 | grep -oP '[\d.]+') 이미 설치됨"
fi

# 부팅 자동 시작 — 재부팅 후에도 nginx가 항상 기동되도록
systemctl enable nginx
log "nginx 부팅 자동 시작 활성화"

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

# EnvironmentFile은 /etc/ 에 두어야 SELinux(init_t)가 읽을 수 있음
BACKEND_ENV="/etc/personal-color-backend.env"
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
  chmod 640 "$BACKEND_ENV"
  chown root:root "$BACKEND_ENV"
  log "backend .env 생성 완료: $BACKEND_ENV"
else
  warn "backend .env 이미 존재 — 덮어쓰지 않음: $BACKEND_ENV"
fi

# =============================================================================
#  7. backend npm 패키지 설치
# =============================================================================
step "backend npm 패키지 설치"

cd "$BACKEND_DIR"
info "backend npm 패키지 설치 중..."
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
info "frontend npm 패키지 설치 중..."
npm ci --prefer-offline 2>/dev/null || npm install
info "React 프로덕션 빌드 시작 중... (2~5분 소요, 기다려 주세요)"
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
ProtectHome=false
ReadWritePaths=$BACKEND_DIR
PrivateTmp=true

# 환경변수
EnvironmentFile=$BACKEND_ENV
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
EOF

# 파일 권한 설정 — .env는 이미 /etc/ 에 생성됨, 앱 디렉토리만 소유권 변경
# /home/ 하위 설치 시 중간 경로 실행 권한 부여
_dir="$BACKEND_DIR"
while [[ "$_dir" != "/" ]]; do
  _parent=$(dirname "$_dir")
  [[ "$_parent" == /home/* || "$_parent" == /home ]] && chmod o+x "$_parent" 2>/dev/null || true
  _dir="$_parent"
done
chown -R "$APP_USER":root "$BACKEND_DIR"

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
log "systemd 서비스 등록 완료"

# =============================================================================
#  11. 방화벽 설정
# =============================================================================
step "방화벽 설정 (firewalld + OCI iptables)"

systemctl enable --now firewalld

# firewalld — HTTP/HTTPS 영구 개방
firewall-cmd --permanent --add-service=http  --quiet
firewall-cmd --permanent --add-service=https --quiet
# 백엔드 포트는 외부 직접 노출 안 함 (nginx 경유만 허용)
firewall-cmd --reload --quiet
log "firewalld: 80/443 개방 완료"

# ── OCI 인스턴스 전용 iptables 처리 ────────────────────────────────────────
# Oracle Cloud 기본 이미지는 firewalld 와 별도로 iptables 에
# "-A INPUT -j REJECT --reject-with icmp-host-prohibited" 규칙이 존재해
# 포트 80/443 을 OS 수준에서 차단함. 아래에서 ACCEPT 규칙을 삽입한다.
if iptables -L INPUT -n 2>/dev/null | grep -q "REJECT\|DROP"; then
  info "OCI iptables REJECT 규칙 감지 — 80/443 ACCEPT 삽입 중..."
  # 중복 삽입 방지
  iptables -C INPUT -p tcp --dport 80  -j ACCEPT 2>/dev/null || \
    iptables -I INPUT -p tcp --dport 80  -j ACCEPT
  iptables -C INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || \
    iptables -I INPUT -p tcp --dport 443 -j ACCEPT
  # 영구 저장 (재부팅 후에도 유지)
  if [[ -f /etc/sysconfig/iptables ]]; then
    iptables-save > /etc/sysconfig/iptables
    systemctl enable iptables 2>/dev/null || true
    log "OCI iptables 규칙 저장 완료 (/etc/sysconfig/iptables)"
  elif command -v netfilter-persistent &>/dev/null; then
    netfilter-persistent save
    log "OCI iptables 규칙 저장 완료 (netfilter-persistent)"
  else
    warn "iptables 영구 저장 실패 — 재부팅 후 수동 확인 필요"
  fi
  log "OCI iptables: 80/443 ACCEPT 삽입 완료"
else
  log "iptables REJECT 규칙 없음 — 추가 처리 불필요"
fi

log "방화벽 설정 완료 (80/443 개방, $BACKEND_PORT 내부 전용)"

# =============================================================================
#  12. 서비스 시작 (HTTP 먼저 — certbot ACME 검증에 필요)
# =============================================================================
step "서비스 시작"

# 부팅 자동시작 재확인 (nginx · 백엔드 둘 다)
systemctl enable nginx          2>/dev/null || true
systemctl enable "$SERVICE_NAME" 2>/dev/null || true

systemctl start "$SERVICE_NAME"
systemctl reload nginx || systemctl start nginx

# ── 백엔드 내부 헬스체크 (최대 20초) ────────────────────────────────────────
info "백엔드 내부 헬스체크 중..."
STARTED=false
for i in $(seq 1 20); do
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

# ── 외부 HTTP 도메인 연결 확인 ───────────────────────────────────────────────
# (IP/localhost는 외부 연결 검증 생략)
if _is_real_domain "$DOMAIN"; then
  info "외부 HTTP 접속 확인 중 (http://$DOMAIN)..."
  HTTP_OK=false
  for i in $(seq 1 10); do
    HTTP_CODE=$(curl -o /dev/null -sf -w "%{http_code}" --max-time 6 \
                  "http://$DOMAIN/" 2>/dev/null || echo "000")
    # 200(OK) / 301·302(리디렉트) / 304 모두 정상 판단
    if [[ "$HTTP_CODE" =~ ^(200|301|302|304)$ ]]; then
      HTTP_OK=true
      break
    fi
    sleep 3
  done

  if $HTTP_OK; then
    log "외부 HTTP 접속 확인 완료 (HTTP $HTTP_CODE) → http://$DOMAIN"
  else
    warn "외부 HTTP 응답 없음 (코드: ${HTTP_CODE:-timeout})"
    warn "━━━ OCI Security List 설정 필요 ━━━"
    warn "  OCI 콘솔 → Networking → Virtual Cloud Networks"
    warn "  → Security Lists → Default Security List"
    warn "  → Add Ingress Rules:"
    warn "    Source CIDR: 0.0.0.0/0  Protocol: TCP  Port: 80"
    warn "    Source CIDR: 0.0.0.0/0  Protocol: TCP  Port: 443"
    warn "  설정 후: sudo systemctl reload nginx"
  fi
fi

# =============================================================================
#  13. Let's Encrypt HTTPS 인증서 발급
# =============================================================================
step "Let's Encrypt HTTPS 인증서"

HTTPS_ENABLED=false

if $SKIP_HTTPS; then
  warn "--skip-https 지정됨 — HTTPS 설정 건너뜀"
elif ! _is_real_domain "$DOMAIN"; then
  warn "IP 주소 또는 localhost는 Let's Encrypt 발급 불가 — HTTPS 건너뜀"
  warn "도메인 사용 시: sudo bash setup.sh --domain example.com --email admin@example.com"
else
  # ── certbot 설치 ─────────────────────────────────────────────────────────
  if ! command -v certbot &>/dev/null; then
    info "EPEL 저장소 및 certbot 설치 중..."

    # EPEL 활성화 (OL 버전별 분기)
    if ! dnf repolist enabled 2>/dev/null | grep -qi epel; then
      if [[ "$OL_VERSION" -ge 10 ]]; then
        dnf install -y -q epel-release 2>/dev/null || \
          dnf install -y -q \
            "https://dl.fedoraproject.org/pub/epel/epel-release-latest-10.noarch.rpm" \
            2>/dev/null || warn "EPEL 10 설치 실패 — certbot 설치를 시도합니다"
      else
        dnf install -y -q epel-release 2>/dev/null || \
          dnf install -y -q \
            "https://dl.fedoraproject.org/pub/epel/epel-release-latest-${OL_VERSION}.noarch.rpm" \
            2>/dev/null || warn "EPEL 설치 실패 — certbot 설치를 시도합니다"
      fi
    fi

    dnf install -y certbot python3-certbot-nginx || \
      err "certbot 설치 실패. EPEL이 활성화되어 있는지 확인하세요."
    log "certbot 설치 완료"
  else
    log "certbot $(certbot --version 2>&1 | grep -oP '[\d.]+' | head -1) 이미 설치됨"
  fi

  # ── 인증서 이메일 처리 ──────────────────────────────────────────────────
  if [[ -n "$CERT_EMAIL" ]]; then
    CERTBOT_EMAIL_ARGS="--email $CERT_EMAIL"
  else
    warn "--email 미지정 — 이메일 없이 발급합니다 (만료 알림 수신 불가)"
    CERTBOT_EMAIL_ARGS="--register-unsafely-without-email"
  fi

  # ── 인증서 발급 ─────────────────────────────────────────────────────────
  info "Let's Encrypt 인증서 발급 중 (도메인: $DOMAIN)..."
  info "  ※ 포트 80이 외부에서 접근 가능해야 합니다 (방화벽/보안그룹 확인)"

  if certbot --nginx \
       --non-interactive \
       --agree-tos \
       $CERTBOT_EMAIL_ARGS \
       -d "$DOMAIN" \
       --redirect \
       --staple-ocsp \
       --hsts; then

    HTTPS_ENABLED=true
    log "HTTPS 인증서 발급 및 nginx 설정 완료"

    # backend .env CLIENT_URL https 로 갱신
    sed -i "s|^CLIENT_URL=.*|CLIENT_URL=https://$DOMAIN|" "$BACKEND_ENV" 2>/dev/null || true

    # ── 자동 갱신 타이머 활성화 ────────────────────────────────────────
    # certbot 패키지가 설치한 타이머 이름은 OL 버전마다 다를 수 있음
    if systemctl list-unit-files certbot-renew.timer &>/dev/null; then
      systemctl enable --now certbot-renew.timer
      log "자동 갱신 타이머 활성화: certbot-renew.timer"
    elif systemctl list-unit-files certbot.timer &>/dev/null; then
      systemctl enable --now certbot.timer
      log "자동 갱신 타이머 활성화: certbot.timer"
    else
      # 타이머가 없으면 cron으로 폴백
      CRON_LINE="0 3 * * * root certbot renew --quiet --post-hook 'systemctl reload nginx'"
      if ! grep -qF "certbot renew" /etc/cron.d/certbot-renew 2>/dev/null; then
        echo "$CRON_LINE" > /etc/cron.d/certbot-renew
        log "자동 갱신 cron 등록: /etc/cron.d/certbot-renew (매일 03:00)"
      fi
    fi

    # nginx 재로드 (certbot이 설정 수정했으므로)
    systemctl reload nginx
    log "nginx HTTPS 설정 적용 완료"

    # 서비스 재시작 (backend .env CLIENT_URL 반영)
    systemctl restart "$SERVICE_NAME"

    # HTTPS 외부 연결 최종 확인
    info "HTTPS 외부 연결 최종 확인 중 (https://$DOMAIN)..."
    HTTPS_OK=false
    for i in $(seq 1 10); do
      HTTPS_CODE=$(curl -o /dev/null -sf -w "%{http_code}" --max-time 8 \
                     "https://$DOMAIN/" 2>/dev/null || echo "000")
      if [[ "$HTTPS_CODE" =~ ^(200|301|302|304)$ ]]; then
        HTTPS_OK=true
        break
      fi
      sleep 3
    done
    if $HTTPS_OK; then
      log "HTTPS 외부 접속 확인 완료 (HTTP $HTTPS_CODE) → https://$DOMAIN"
    else
      warn "HTTPS 외부 응답 없음 — 인증서는 발급됐으나 네트워크 확인 필요"
    fi

  else
    warn "인증서 발급 실패 — HTTP로 계속 진행합니다"
    warn "수동 발급: sudo certbot --nginx -d $DOMAIN --email your@email.com"
    # BASE_URL을 http로 되돌림
    BASE_URL="http://$DOMAIN"
  fi
fi

# =============================================================================
#  14. 설치 완료 요약
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  Personal Color Cloud 설치 완료!${RESET}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${RESET}"
echo ""
if $HTTPS_ENABLED; then
  echo -e "  ${CYAN}웹 앱${RESET}        : ${BOLD}https://$DOMAIN${RESET}  ${GREEN}[HTTPS ✔]${RESET}"
  echo -e "  ${CYAN}API 서버${RESET}     : ${BOLD}https://$DOMAIN/api${RESET}"
  echo -e "  ${CYAN}헬스체크${RESET}     : ${BOLD}https://$DOMAIN/health${RESET}"
  echo -e "  ${CYAN}인증서 경로${RESET}  : /etc/letsencrypt/live/$DOMAIN/"
  echo -e "  ${CYAN}인증서 갱신${RESET}  : 자동 (systemd 타이머 또는 cron)"
  echo -e "    수동 갱신: sudo certbot renew --post-hook 'systemctl reload nginx'"
else
  echo -e "  ${CYAN}웹 앱${RESET}        : ${BOLD}$BASE_URL${RESET}"
  echo -e "  ${CYAN}API 서버${RESET}     : ${BOLD}$BASE_URL/api${RESET}"
  echo -e "  ${CYAN}헬스체크${RESET}     : ${BOLD}$BASE_URL/health${RESET}"
  if _is_real_domain "$DOMAIN" && ! $SKIP_HTTPS; then
    echo ""
    echo -e "  ${YELLOW}[HTTPS 재시도]${RESET} 인증서 발급 실패 시 수동 발급:"
    echo -e "    sudo certbot --nginx -d $DOMAIN --email your@email.com"
  fi
fi
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
