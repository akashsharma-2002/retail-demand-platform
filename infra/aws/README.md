# AWS deployment

One EC2 host running Docker Compose, fronted by Caddy with automatic TLS.

| Resource | Setting |
|---|---|
| EC2 | t3.xlarge (16 GB, BGE-M3 and reranker in memory), Ubuntu 24.04, SSM agent, no SSH key pair |
| Security group | inbound 443 and 80 (ACME redirect) only |
| IAM instance role | `AmazonSSMManagedInstanceCore`, read `retail/prod` secret, write backup bucket |
| Secrets Manager | `retail/prod`: POSTGRES_PASSWORD, APP_WRITE_PASSWORD, APP_READ_PASSWORD, KEYCLOAK_DB_PASSWORD, KEYCLOAK_ADMIN_PASSWORD, DOMAIN, OPENAI_API_KEY, LANGFUSE_* |
| S3 | backup bucket, SSE-KMS, public access blocked, 7-day lifecycle |
| GitHub OIDC role | trust `repo:<owner>/<repo>:environment:production`; allows `ssm:SendCommand` on the instance only |

Deploy flow: CI passes on `main` -> `deploy.yml` builds and pushes images to GHCR -> assumes the OIDC role ->
SSM runs `infra/aws/deploy.sh <tag>` on the host -> the script fetches secrets, pulls images, restarts and waits
for `/readyz`.

The production realm import must not contain the local demo users; create users in Keycloak admin instead.
