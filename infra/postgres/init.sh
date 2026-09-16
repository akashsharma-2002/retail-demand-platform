#!/bin/bash
# Least-privilege roles: app_write for the API and pipeline, app_read for reporting. Keycloak gets its own DB.
set -euo pipefail
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE EXTENSION IF NOT EXISTS pg_search;
  CREATE ROLE app_write LOGIN PASSWORD '${APP_WRITE_PASSWORD}';
  CREATE ROLE app_read LOGIN PASSWORD '${APP_READ_PASSWORD}';
  GRANT CONNECT ON DATABASE retail TO app_write, app_read;
  GRANT USAGE, CREATE ON SCHEMA public TO app_write;
  GRANT USAGE ON SCHEMA public TO app_read;
  ALTER DEFAULT PRIVILEGES FOR ROLE app_write IN SCHEMA public GRANT SELECT ON TABLES TO app_read;
  REVOKE ALL ON DATABASE retail FROM PUBLIC;
  CREATE ROLE keycloak LOGIN PASSWORD '${KEYCLOAK_DB_PASSWORD}';
  CREATE DATABASE keycloak OWNER keycloak;
SQL
