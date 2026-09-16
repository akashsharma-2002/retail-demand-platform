#!/usr/bin/env bash
# One-command setup for a fresh Ubuntu 24.04 EC2 instance (run as root via AWS Session Manager):
#   curl -fsSL https://raw.githubusercontent.com/akashsharma-2002/retail-demand-platform/main/infra/aws/bootstrap.sh | sudo bash
#
# Safe to run again: it pulls the latest code, keeps existing passwords and data, and restarts services.
#
# Needs (see infra/aws/README.md):
#   - instance role with infra/aws/iam-instance-policy.json + AmazonSSMManagedInstanceCore
#   - Parameter Store SecureStrings: /retail/prod/OLLAMA_API_KEY (required),
#     /retail/prod/LANGFUSE_PUBLIC_KEY and /retail/prod/LANGFUSE_SECRET_KEY (optional)
#   - Parameter /retail/prod/BACKUP_BUCKET (String) naming the S3 bucket that holds retail.dump (first run only)
set -euo pipefail

REPO_URL="https://github.com/akashsharma-2002/retail-demand-platform.git"
APP_DIR="/opt/retail"
PREFIX="/retail/prod"
log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || fail "Run with sudo."

# ---------------------------------------------------------------------------------------------------- packages
log "Installing Docker, Git and the AWS CLI"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl git jq openssl unzip >/dev/null
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh >/dev/null
fi
if ! command -v aws >/dev/null; then
  snap install aws-cli --classic >/dev/null
fi
systemctl enable --now docker >/dev/null

if ! swapon --show | grep -q /swapfile; then
  log "Adding 4 GB swap as a memory safety margin"
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q /swapfile /etc/fstab || echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi

# ---------------------------------------------------------------------------------------------------- instance facts
TOKEN=$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
imds() { curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
export AWS_DEFAULT_REGION
AWS_DEFAULT_REGION=$(imds placement/region)
PUBLIC_IP=$(imds public-ipv4) || fail "This instance has no public IP. Attach an Elastic IP first."
DOMAIN="${PUBLIC_IP//./-}.sslip.io"
log "Region ${AWS_DEFAULT_REGION}, public address https://${DOMAIN}"

# ---------------------------------------------------------------------------------------------------- code
if [ -d "$APP_DIR/.git" ]; then
  log "Updating code"
  git -C "$APP_DIR" fetch --quiet origin main && git -C "$APP_DIR" reset --quiet --hard origin/main
else
  log "Cloning code"
  git clone --quiet "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

# ---------------------------------------------------------------------------------------------------- secrets
param_get() {  # name -> value or empty
  aws ssm get-parameter --name "$PREFIX/$1" --with-decryption --query Parameter.Value --output text 2>/dev/null || true
}
param_put_secret() {
  aws ssm put-parameter --name "$PREFIX/$1" --type SecureString --value "$2" --overwrite >/dev/null
}
generated_secret() {  # name -> existing value, or a new random one stored in Parameter Store
  local value
  value=$(param_get "$1")
  if [ -z "$value" ] || [ "$value" = "None" ]; then
    value=$(openssl rand -base64 48 | tr -dc 'A-Za-z0-9' | head -c 32)
    param_put_secret "$1" "$value"
  fi
  printf '%s' "$value"
}

log "Loading secrets from Parameter Store (${PREFIX})"
POSTGRES_PASSWORD=$(generated_secret POSTGRES_PASSWORD)
APP_WRITE_PASSWORD=$(generated_secret APP_WRITE_PASSWORD)
APP_READ_PASSWORD=$(generated_secret APP_READ_PASSWORD)
KEYCLOAK_DB_PASSWORD=$(generated_secret KEYCLOAK_DB_PASSWORD)
KEYCLOAK_ADMIN_PASSWORD=$(generated_secret KEYCLOAK_ADMIN_PASSWORD)
OLLAMA_API_KEY=$(param_get OLLAMA_API_KEY)
[ -n "$OLLAMA_API_KEY" ] && [ "$OLLAMA_API_KEY" != "None" ] || fail "Create SecureString ${PREFIX}/OLLAMA_API_KEY first."
LANGFUSE_PUBLIC_KEY=$(param_get LANGFUSE_PUBLIC_KEY); [ "$LANGFUSE_PUBLIC_KEY" = "None" ] && LANGFUSE_PUBLIC_KEY=""
LANGFUSE_SECRET_KEY=$(param_get LANGFUSE_SECRET_KEY); [ "$LANGFUSE_SECRET_KEY" = "None" ] && LANGFUSE_SECRET_KEY=""
RP_LLM_MODEL=$(param_get RP_LLM_MODEL); { [ -z "$RP_LLM_MODEL" ] || [ "$RP_LLM_MODEL" = "None" ]; } && RP_LLM_MODEL="gemma4:31b"

umask 077
cat > .env <<ENV
RP_ENV=production
DOMAIN=${DOMAIN}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
APP_WRITE_PASSWORD=${APP_WRITE_PASSWORD}
APP_READ_PASSWORD=${APP_READ_PASSWORD}
KEYCLOAK_DB_PASSWORD=${KEYCLOAK_DB_PASSWORD}
KEYCLOAK_ADMIN_PASSWORD=${KEYCLOAK_ADMIN_PASSWORD}
OIDC_PUBLIC_URL=https://${DOMAIN}/auth
WEB_ORIGIN=https://${DOMAIN}
RP_RAG_MODE=hybrid_rerank
RP_LLM_PROVIDER=ollama
RP_LLM_MODEL=${RP_LLM_MODEL}
OLLAMA_API_KEY=${OLLAMA_API_KEY}
LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
LANGFUSE_BASE_URL=https://cloud.langfuse.com
ENV
umask 022
sed "s/__DOMAIN__/${DOMAIN}/g" infra/keycloak/prod/retail-realm.template.json > infra/keycloak/prod/retail-realm.json

COMPOSE=(docker compose -f docker-compose.yml -f infra/aws/docker-compose.prod.yml)

# ---------------------------------------------------------------------------------------------------- images
log "Building images (first run takes about 10 minutes)"
docker build -q -t retail-platform-api . >/dev/null
docker build -q --build-arg "VITE_OIDC_URL=https://${DOMAIN}/auth" -t retail-platform-web web >/dev/null
docker image prune -f >/dev/null

# ---------------------------------------------------------------------------------------------------- database
log "Starting PostgreSQL and Redis"
"${COMPOSE[@]}" up -d postgres redis
# Over TCP: during first-time initialisation Postgres runs a temporary socket-only server, then restarts.
for _ in $(seq 1 90); do
  "${COMPOSE[@]}" exec -T postgres pg_isready -h 127.0.0.1 -U postgres -d retail >/dev/null 2>&1 && break
  sleep 2
done

psql_scalar() { "${COMPOSE[@]}" exec -T postgres psql -U postgres -d retail -tAc "$1" 2>/dev/null | tr -d '[:space:]'; }
if [ "$(psql_scalar "select to_regclass('public.items') is not null")" != "t" ]; then
  BUCKET=$(param_get BACKUP_BUCKET)
  [ -n "$BUCKET" ] && [ "$BUCKET" != "None" ] || fail "Database is empty. Set ${PREFIX}/BACKUP_BUCKET and upload retail.dump."
  log "Restoring database from s3://${BUCKET}/retail.dump"
  aws s3 cp --quiet "s3://${BUCKET}/retail.dump" /tmp/retail.dump
  # Extensions already exist (created by infra/postgres/init.sh); pg_restore reports them and continues.
  "${COMPOSE[@]}" exec -T postgres pg_restore -U postgres -d retail --no-owner --role=app_write < /tmp/retail.dump || true
  rm -f /tmp/retail.dump
fi
ITEMS=$(psql_scalar "select count(*) from items")
FORECASTS=$(psql_scalar "select count(*) from forecasts")
[ "${ITEMS:-0}" -gt 0 ] || fail "Database restore did not load items."
log "Database ready: ${ITEMS} items, ${FORECASTS} forecast rows"

# ---------------------------------------------------------------------------------------------------- services
log "Starting Keycloak, API, web and Caddy"
"${COMPOSE[@]}" up -d --remove-orphans keycloak api web caddy

log "Waiting for Keycloak"
for _ in $(seq 1 90); do
  "${COMPOSE[@]}" exec -T keycloak bash -c 'exec 3<>/dev/tcp/127.0.0.1/9000 && printf "GET /auth/health/ready HTTP/1.0\r\n\r\n" >&3 && grep -q UP <&3' >/dev/null 2>&1 && break
  sleep 5
done

log "Creating demo users (passwords stored in Parameter Store)"
KCADM=("${COMPOSE[@]}" exec -T keycloak /opt/keycloak/bin/kcadm.sh)
"${KCADM[@]}" config credentials --server http://localhost:8080/auth --realm master --user admin --password "$KEYCLOAK_ADMIN_PASSWORD" >/dev/null
for pair in viewer:viewer planner:planner approver:approver; do
  user="demo-${pair%%:*}"; role="${pair##*:}"
  if [ -z "$("${KCADM[@]}" get users -r retail -q "username=${user}" -q exact=true --fields id --format csv --noquotes 2>/dev/null)" ]; then
    password=$(generated_secret "demo/${user}")
    "${KCADM[@]}" create users -r retail -s "username=${user}" -s enabled=true -s emailVerified=true \
      -s "email=${user}@example.test" -s "firstName=Demo" -s "lastName=${role^}" >/dev/null
    "${KCADM[@]}" set-password -r retail --username "$user" --new-password "$password" >/dev/null
    "${KCADM[@]}" add-roles -r retail --uusername "$user" --rolename "$role" >/dev/null
  fi
done

log "Waiting for the API to load its models (first start downloads about 4.5 GB)"
for _ in $(seq 1 120); do
  "${COMPOSE[@]}" exec -T api curl -fsS http://localhost:8000/readyz >/dev/null 2>&1 && break
  sleep 5
done

log "Checking the public site"
for _ in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "https://${DOMAIN}/v1/stores" || true)
  [ "$code" = "401" ] && break
  sleep 10
done
[ "$code" = "401" ] || fail "Public API check returned ${code}. See: ${COMPOSE[*]} logs caddy api"

cat <<DONE

$(printf '\033[1;32m')Deployment complete.$(printf '\033[0m')

  Site:            https://${DOMAIN}
  Demo logins:     demo-viewer, demo-planner, demo-approver
  Their passwords: AWS console -> Systems Manager -> Parameter Store -> ${PREFIX}/demo/...

DONE
