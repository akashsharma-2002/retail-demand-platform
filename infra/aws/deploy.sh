#!/usr/bin/env bash
# Runs on the EC2 host via SSM. Pulls secrets from AWS Secrets Manager, pulls images, restarts, checks health.
set -euo pipefail
TAG="$1"
cd /opt/retail
git fetch --quiet origin main && git checkout --quiet "origin/main" -- infra docker-compose.yml

aws secretsmanager get-secret-value --secret-id retail/prod --query SecretString --output text \
  | python3 -c 'import json,sys; [print(f"{k}={v}") for k,v in json.load(sys.stdin).items()]' > .env
chmod 600 .env
export IMAGE_TAG="$TAG"

COMPOSE="docker compose -f docker-compose.yml -f infra/aws/docker-compose.prod.yml"
$COMPOSE pull api web
$COMPOSE up -d postgres redis
$COMPOSE run --rm --no-deps api python -m retail_platform.storage.migrate
$COMPOSE up -d --remove-orphans keycloak api web caddy

for i in $(seq 1 30); do
  if $COMPOSE exec -T api curl -fsS http://localhost:8000/readyz >/dev/null; then
    echo "deployed $TAG"; exit 0
  fi
  sleep 5
done
echo "API did not become ready" >&2
$COMPOSE logs --tail 100 api >&2
exit 1
