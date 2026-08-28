# Legacy: AWS EKS + Jenkins deployment

The project's first production deployment. It ran the full application on a managed
Kubernetes cluster with a self-hosted CI/CD server. It was **retired for cost** — roughly
**$150–200/month** (EKS control plane, NAT gateway, load balancer, worker nodes) versus
**~$15–30/month** for the single-EC2 setup that replaced it.

Kept here as a reference; nothing in this folder is deployed.

## What it contained

| Path | Purpose |
|------|---------|
| `terraform/` | VPC across 2 AZs, EKS (SPOT managed node group), ECR, EBS CSI driver, IAM, Jenkins EC2 |
| `ansible/` | Provisioned the Jenkins host: Java 21, Docker, kubectl, AWS CLI v2, nightly image prune |
| `Kubernetes/` | Deployments, Services, nginx Ingress, ConfigMap/Secret, Postgres StatefulSet, gp3 StorageClass, cert-manager ClusterIssuer, evaluation-sandbox RBAC + NetworkPolicy |
| `Jenkinsfile.*` | Per-repo pipelines: test → build → integration test → push to ECR → `kubectl` rollout |

## Notable pieces

- **`Kubernetes/postgres.Dockerfile`** — Postgres 15 with the `pgvector` extension compiled
  from source, verified by SHA-256, then flattened to drop the build toolchain from lower layers.
- **`Jenkinsfile.backend`** — ran a full Sprint-5 lifecycle integration test on every build:
  spun up Postgres + Redis on a throwaway Docker network, ran migrations and seeds, waited for
  `/api/health`, then executed the end-to-end script.
- **Migrations as an initContainer** so the schema was always current before the app started.
- **cert-manager + Let's Encrypt** for automatic, auto-renewing TLS on the ingress.

## Lessons carried into the current setup

- Migrations must run as a gated step *before* the app starts, never alongside it.
- `NEXT_PUBLIC_*` is baked into the Next.js bundle at build time, so the API base must be
  correct when the image is built — not at runtime.
- Docker build cache fills a small disk quickly; prune after every build and nightly.
- Concurrent Docker builds on one host corrupt each other's layers; serialize them.
- Health endpoints (`/api/health`, `/api/health/live`) gate rollouts safely.
