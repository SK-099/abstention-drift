# Deploying on an AWS virtual machine (EC2, free-plan credits)

This guide puts the notebook app (web UI, API, worker, Postgres, Redis, Qdrant and a language model) on one
EC2 virtual machine, so the team can use it from a browser at `http://<server-ip>:3000`.

**Time:** about 1 hour, most of it waiting for the first build.
**Cost:** paid from the AWS sign-up credits (see [Costs](#costs)). Stop the machine when nobody uses it.

## 1. Choose the machine

| Choice | Value | Why |
|---|---|---|
| Region | the one closest to the team (e.g. Asia Pacific (Mumbai) `ap-south-1`) | lower latency |
| Image (AMI) | **Ubuntu Server 24.04 LTS**, 64-bit (x86) | Docker and Ollama install with one command |
| Instance type | **`m7i-flex.large`** (2 vCPU, 8 GB RAM) | the largest free-plan-eligible type; the stack needs about 5 GB plus the model |
| Storage | **40 GB gp3** | Docker images (about 10 GB), models, database |
| Key pair | create one (`.pem`), keep it safe | SSH login |

There is no GPU on free-plan instance types, so a local model runs on the CPU and is slow (a 1.5B model
takes about 20-60 s per answer, a 3B model longer). For faster answers use a hosted model (step 5, option B).

## 2. Network: open only what is needed

In **Security group → Inbound rules** allow:

| Port | Source | Purpose |
|---|---|---|
| 22 (SSH) | **My IP** | administration |
| 3000 | the team's IP addresses (or My IP) | web UI |
| 8000 | the same IP addresses | API (the browser calls it directly) |

Do **not** open 5432 (Postgres), 6333 (Qdrant), 6379 (Redis) or 11434 (Ollama). The app runs in
`AUTH_MODE=dev`, which has **no login**: anyone who can reach ports 3000/8000 can use it and see every
notebook. Keep those ports limited to known IP addresses (an address that changes must be updated in the rule).

**Elastic IP:** allocate one and associate it with the instance (EC2 → Elastic IPs). Without it the public IP
changes every time the machine is stopped, and the web UI (which has the API address built in) would have to
be rebuilt.

## 3. Install Docker and Ollama

Connect (Windows PowerShell works too):

```bash
ssh -i abstain-key.pem ubuntu@<server-ip>
```

On the server:

```bash
sudo apt-get update && sudo apt-get -y upgrade
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu          # then log out and in again (exit, ssh ...) so docker works without sudo

# 4 GB swap: protects the 8 GB machine during builds and model loading
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Ollama (only for option A in step 5):

```bash
curl -fsSL https://ollama.com/install.sh | sh
# let the Docker containers reach Ollama (port 11434 stays closed in the security group)
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0"\nEnvironment="OLLAMA_KEEP_ALIVE=10m"\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
ollama pull qwen2.5:1.5b               # small and fast enough on a CPU; qwen2.5:3b is the calibrated research model
```

## 4. Get the code

```bash
git clone https://github.com/SK-099/abstention-drift.git
cd abstention-drift
cp .env.example .env
```

**Optional, the answerability gate:** the fine-tuned checker weights (about 1 GB) are not in the repository.
Without them the app runs without the gate (it still refuses through the model and the self-check). To use
it, copy the folder from the laptop that has it, from Windows PowerShell:

```powershell
scp -i abstain-key.pem -r F:\NLP\abstention-drift\checkpoints\checker\best ubuntu@<server-ip>:~/abstention-drift/checkpoints/checker/
```

## 5. Configure `.env`

Edit with `nano .env` and set (replace `<server-ip>` with the Elastic IP):

```ini
PUBLIC_API_URL=http://<server-ip>:8000
CORS_ORIGINS=["http://<server-ip>:3000"]
TORCH_THREADS=2
WORKER_CONCURRENCY=1
```

Then choose the language model:

- **Option A, local model with Ollama (no account needed, slow):**
  ```ini
  LLM_BASE_URL=http://host.docker.internal:11434/v1
  LLM_MODEL=qwen2.5:1.5b
  ```
- **Option B, hosted model (fast; needs an API key, usage may cost money):** for example OpenRouter
  ```ini
  OPENROUTER_API_KEY=sk-or-...
  DEFAULT_MODEL=openrouter:qwen/qwen-2.5-7b-instruct
  ```
  Per-model refusal thresholds are calibrated only for the ten local models; hosted models use the
  research defaults (shown without the shield in the model menu).

Never commit `.env`; it stays on the server.

## 6. Build and start

```bash
docker compose build api        # 15-30 min the first time on 2 vCPUs
docker compose build web        # bakes PUBLIC_API_URL into the web UI
docker compose up -d
docker compose ps               # api, worker, web, postgres, redis, qdrant: all "Up"
curl -s localhost:8000/healthz  # {"status":"ok",...}
```

Open **http://\<server-ip\>:3000** in a browser on a computer whose IP is allowed in the security group.

Audio overviews (Kokoro, CPU-heavy, about 1 GB RAM): `docker compose --profile tts up -d kokoro`, and
`docker compose stop kokoro` afterwards.

## 7. Day-to-day

| Task | Command |
|---|---|
| Update to the latest code | `git pull && docker compose build api web && docker compose up -d` |
| Logs | `docker compose logs --tail 50 api worker` |
| Restart | `docker compose restart api worker` |
| Stop the app (keeps data) | `docker compose down` |
| Stop paying for compute | stop the instance in the EC2 console (storage and the Elastic IP are still charged) |

The containers restart automatically when the instance starts again (`restart: unless-stopped`); Ollama is a
system service and starts too.

If `PUBLIC_API_URL` changes (a new IP address), run `docker compose build web && docker compose up -d`.

## Costs

The free plan gives USD 100 credit at sign-up (up to 100 more for completing activities) and lasts 6 months or
until the credit is used up. Rough on-demand prices in us-east-1 (other regions differ a little):

| Item | Price | Per month running 24/7 |
|---|---|---|
| `m7i-flex.large` | about USD 0.096 / hour | about USD 70 |
| 40 GB gp3 storage | about USD 0.08 / GB-month | about USD 3 |
| Public IPv4 / Elastic IP | USD 0.005 / hour | about USD 3.60 |

So USD 100 lasts roughly 5-6 weeks running non-stop, much longer when the instance is stopped outside demo
and testing hours. In **Billing → Budgets**, create a budget alert (e.g. at USD 20 and USD 50) so nobody is
surprised; creating one also counts towards the extra credit activities.

## Known limits on a cloud server

- **YouTube sources** may fail: YouTube often refuses transcript requests from cloud-provider IP addresses.
  Web pages, PDFs and pasted text are not affected.
- **No HTTPS and no login.** Fine for a closed demo limited by IP; for a public site, put it behind HTTPS
  (e.g. Caddy) and switch `AUTH_MODE` to `hs256` or `jwks` (see [service.md](service.md)).
- **CPU only:** answers from local models take tens of seconds; source guides and Studio outputs take minutes.
