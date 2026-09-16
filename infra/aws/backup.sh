#!/usr/bin/env bash
# Nightly cron on the EC2 host: compressed pg_dump to S3, 7-day lifecycle rule on the bucket.
set -euo pipefail
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
cd /opt/retail
docker compose exec -T postgres pg_dump -U postgres -Fc retail \
  | aws s3 cp - "s3://${BACKUP_BUCKET:?}/postgres/retail-${STAMP}.dump" --sse aws:kms
