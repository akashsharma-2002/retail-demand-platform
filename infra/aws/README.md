# AWS deployment (Free plan)

One EC2 instance runs the whole stack with Docker Compose. Caddy is the only public entry point and obtains a
TLS certificate automatically for a free `sslip.io` hostname derived from the instance's Elastic IP.

```
Internet ──443──► Caddy ─┬─ /        → web (React)
                         ├─ /v1/*    → api (FastAPI) ─► PostgreSQL, Redis, Ollama cloud, Langfuse
                         └─ /auth/*  → Keycloak (admin console and master realm blocked)
```

| Resource | Setting |
|---|---|
| Region | Asia Pacific (Mumbai) `ap-south-1` |
| EC2 | `m7i-flex.large` (2 vCPU, 8 GB), Ubuntu 24.04, 30 GB gp3, IMDSv2 required, no key pair |
| Network | Security group inbound 80 and 443 only; Elastic IP |
| Access | AWS Systems Manager Session Manager (no SSH) |
| Instance role | `AmazonSSMManagedInstanceCore` + [`iam-instance-policy.json`](iam-instance-policy.json) |
| Secrets | Parameter Store SecureStrings under `/retail/prod/` (database and Keycloak passwords are generated on the instance) |
| Data | `retail.dump` restored once from a private S3 bucket `retail-backups-*` |
| Updates | GitHub Actions assumes a role through OIDC and runs `bootstrap.sh` via `ssm:SendCommand` |

## Setup

### 1. Secrets and backup
1. Switch the console region to **Asia Pacific (Mumbai)**.
2. **Systems Manager → Parameter Store → Create parameter**:
   - `/retail/prod/OLLAMA_API_KEY`: SecureString (key from ollama.com → Settings → Keys)
   - `/retail/prod/LANGFUSE_PUBLIC_KEY` and `/retail/prod/LANGFUSE_SECRET_KEY`: SecureString (optional)
3. **S3 → Create bucket** `retail-backups-<something-unique>` (keep "Block all public access" on) and upload
   `retail.dump`.
4. Parameter `/retail/prod/BACKUP_BUCKET`: String, value = the bucket name.

### 2. Instance role
**IAM → Roles → Create role →** trusted entity *AWS service*, use case *EC2* → attach
`AmazonSSMManagedInstanceCore` → name `retail-ec2-role`. Open the role → **Add permissions → Create inline
policy → JSON** → paste `iam-instance-policy.json` → name `retail-app-access`.

### 3. Instance
**EC2 → Launch instance**: name `retail-platform`, Ubuntu Server 24.04 LTS, `m7i-flex.large` (check the "Free
tier eligible" label), key pair *Proceed without a key pair*, new security group allowing **HTTP** and **HTTPS**
from anywhere (untick SSH), storage 30 GiB gp3, **Advanced details → IAM instance profile** `retail-ec2-role`,
**Metadata version** V2 only. Then **Elastic IPs → Allocate → Associate** with the instance.

### 4. Deploy
**EC2 → Instances → retail-platform → Connect → Session Manager → Connect**, then:

```bash
curl -fsSL https://raw.githubusercontent.com/akashsharma-2002/retail-demand-platform/main/infra/aws/bootstrap.sh | sudo bash
```

The script prints the site URL. Demo user passwords are in Parameter Store under `/retail/prod/demo/`.

### 5. Automatic deploys (optional)
1. **IAM → Identity providers → Add provider**: OpenID Connect, URL `https://token.actions.githubusercontent.com`,
   audience `sts.amazonaws.com`.
2. Role `retail-github-deploy` trusted by that provider, restricted to
   `repo:akashsharma-2002/retail-demand-platform:environment:production`, allowed only
   `ssm:SendCommand` (this instance and `AWS-RunShellScript`) and `ssm:GetCommandInvocation`.
3. GitHub → Settings → Environments → `production`; secrets `AWS_DEPLOY_ROLE_ARN`, `EC2_INSTANCE_ID`;
   variable `DEPLOY_ENABLED=true`.

## Costs and stopping
Everything runs on Free plan credits. Running 24/7 uses roughly $75 of credit a month (instance, public IPv4 and
disk). **Stop the instance when not demoing it** (EC2 → Instance state → Stop): only the 30 GB disk (about $3 a
month) and the Elastic IP keep billing. Start it again and the site returns on the same address.
