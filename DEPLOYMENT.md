# Free Cloud Deployment

This guide covers multiple deployment options for the Portfolio Optimizer application.

---

## Quick Start: Choose Your Platform

| Platform | Difficulty | Best For |
|----------|-----------|----------|
| **[Koyeb](#option-1-koyeb)** | Easy | EU-based, Docker-native |
| **[Zeabur](#option-2-zeabur)** | Easy | Asia-optimized, pay-as-you-go |
| **[Docker](#docker-runtime)** | Advanced | Local dev, self-hosted |

---

## Prerequisites (All Platforms)

### Step 1: Neon Database (1 min)

All deployment options require a PostgreSQL database. We recommend Neon (free tier):

1. Go to **[neon.tech](https://neon.tech)** → Sign up with GitHub
2. Click **"Create Project"** → name: `portfolio-optimizer` → Create
3. Click **"Connect"** → copy the full connection string  
   (looks like: `postgresql://user:pass@ep-xxx.us-east-1.aws.neon.tech/portfolio-optimizer?sslmode=require`)

**Save this connection string - you'll need it for all deployment options.**

---

## Option 1: Koyeb Deployment

Koyeb is a Docker-native platform with a generous free tier.

### Deploy Steps

1. Go to **[koyeb.com](https://koyeb.com)** → Sign up with GitHub
2. Click **"Create App"** → select your GitHub repo
3. Fill in settings:

| Setting | Value |
|---------|-------|
| Name | `portfolio-optimizer` |
| Region | `Frankfurt` (or closest) |
| Branch | `main` |
| Builder | `Docker` |
| Dockerfile | `Dockerfile` |

4. Click **"Variables"** → Add environment variables (see [Common Variables](#common-environment-variables) below)
5. Click **"Deploy"** → wait 2-3 minutes

> **Note:** Koyeb uses port 8000 by default. The Dockerfile is already configured.

### Update BASE_URL After Deploy

After first deployment:
1. Copy your Koyeb URL (e.g., `https://portfolio-optimizer-xxx.koyeb.app`)
2. Go to **Variables** in Koyeb dashboard
3. Update `BASE_URL` to match your actual URL
4. Redeploy

---

## Option 2: Zeabur Deployment

Zeabur is optimized for Asian regions and offers pay-as-you-go pricing with a free tier.

### Deploy Steps

1. Go to **[zeabur.com](https://zeabur.com)** → Sign up with GitHub
2. Click **"Create Project"** → name: `portfolio-optimizer` → Create
3. Click **"Deploy New Service"** → select **"Git"** → choose your repository
4. Zeabur will auto-detect the Dockerfile and build

### Configure Environment Variables

Once deployed (or during deployment):

1. Click on your service → **"Variables"** tab
2. Add all required variables (see [Common Variables](#common-environment-variables) below)
3. Click **"Redeploy"** after adding variables

### Update BASE_URL After Deploy

1. Go to your service → copy the generated domain (e.g., `https://portfolio-optimizer-xxx.zeabur.app`)
2. Add/update the `BASE_URL` variable with your actual URL
3. Redeploy

### Zeabur-Specific Notes

- **Port:** Zeabur automatically detects the `PORT` environment variable from the Dockerfile/Procfile
- **Region:** Choose `Singapore` or `Taiwan` for best performance in Asia
- **Logs:** View real-time logs in the **"Logs"** tab
- **Custom Domain:** Add a custom domain in the **"Domain"** tab

---

## Common Environment Variables

These variables are required for both Koyeb and Zeabur deployments.

### Required Variables (MUST ADD)

| Variable | Value | Description |
|----------|-------|-------------|
| `SECRET_KEY` | `change-this-to-32-random-chars-now!` | Random string, min 32 chars |
| `DATABASE_URL` | `postgresql://...` | Paste your Neon connection string |
| `BASE_URL` | `https://your-app-name.koyeb.app` or `https://your-app.zeabur.app` | Your deployment URL |

### For AI Insights (Optional but Recommended)

Get a free API key from **[DeepSeek](https://platform.deepseek.com)** (has free credits) or use OpenAI.

| Variable | Value |
|----------|-------|
| `LLM_API_KEY` | `sk-...` |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` |
| `LLM_MODEL` | `deepseek-chat` |

### For Email (Registration & Password Reset)

**Without this, users cannot reset passwords.**

Use Gmail or any SMTP provider:

| Variable | Value |
|----------|-------|
| `SMTP_USER` | `your-email@gmail.com` |
| `SMTP_PASSWORD` | `your-app-password` |
| `ADMIN_EMAIL` | `your-email@gmail.com` |

**For Gmail:** Create an [App Password](https://myaccount.google.com/apppasswords) (requires 2FA enabled).

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AI_INSIGHTS` | `True` | Enable AI portfolio analysis |
| `MAX_PORTFOLIO_ASSETS` | `50` | Max assets per portfolio |
| `CACHE_TTL` | `3600` | Cache duration in seconds |

---

## Docker Runtime

Run the application locally or self-host using Docker.

### Quick Start with Docker

```bash
# Clone the repository
git clone <your-repo-url>
cd portfolio-optimizer

# Create environment file
cp .env.example .env
# Edit .env with your settings (see Environment Variables above)

# Build and run
docker build -t portfolio-optimizer .
docker run -p 8000:8000 --env-file .env portfolio-optimizer
```

The app will be available at `http://localhost:8000`

### Docker Compose (Recommended for Development)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=${DATABASE_URL}
      - BASE_URL=http://localhost:8000
      - LLM_API_KEY=${LLM_API_KEY:-}
      - LLM_BASE_URL=${LLM_BASE_URL:-}
      - LLM_MODEL=${LLM_MODEL:-}
      - SMTP_USER=${SMTP_USER:-}
      - SMTP_PASSWORD=${SMTP_PASSWORD:-}
      - ADMIN_EMAIL=${ADMIN_EMAIL:-}
    volumes:
      - ./:/app
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

### Production Docker Deployment

For production self-hosting:

```bash
# Build with no cache for clean image
docker build --no-cache -t portfolio-optimizer:latest .

# Run in detached mode with restart policy
docker run -d \
  --name portfolio-optimizer \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  portfolio-optimizer:latest

# View logs
docker logs -f portfolio-optimizer
```

### Dockerfile Details

The included `Dockerfile`:
- Uses Python 3.11 slim image
- Installs dependencies from `requirements.txt`
- Exposes port 8000
- Runs with Gunicorn WSGI server
- Uses the `PORT` environment variable (defaults to 8000)

### Docker Health Check

Add to your `docker-compose.yml` for health monitoring:

```yaml
services:
  app:
    build: .
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

---

## Full Production .env Example

```bash
# REQUIRED
SECRET_KEY=your-random-secret-key-32-chars-min
DATABASE_URL=postgresql://user:pass@host.neon.tech/db?sslmode=require
BASE_URL=https://your-app.koyeb.app  # or https://your-app.zeabur.app

# AI (optional)
LLM_API_KEY=sk-your-deepseek-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# Email (optional - needed for password reset)
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ADMIN_EMAIL=your-email@gmail.com
```

---

## Platform Comparison

| Feature | Koyeb | Zeabur |
|---------|-------|--------|
| Free Tier | Yes | Yes |
| Docker Support | Native | Auto-detect |
| Best Region | EU/US | Asia |
| Custom Domains | Yes | Yes |
| Auto Deploy from Git | Yes | Yes |
| Logs | Web UI | Web UI |
| Pricing | Eco→Starter→... | Pay-as-you-go |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Email service not configured" | Add SMTP_USER, SMTP_PASSWORD, ADMIN_EMAIL |
| "Worker timeout" | Platform has limited RAM - upgrade tier or reduce workers |
| "AI insights not working" | Add LLM_API_KEY from DeepSeek or OpenAI |
| "Password reset fails" | Check SMTP settings and BASE_URL is correct |
| "Database connection failed" | Verify DATABASE_URL and Neon is active |
| "Port already in use" (Docker) | Change host port: `-p 8080:8000` |

### Docker-Specific Issues

| Issue | Solution |
|-------|----------|
| "Cannot connect to database" | Ensure DATABASE_URL is set correctly in .env |
| "Permission denied" | Check Docker is running: `docker info` |
| "Build fails" | Try: `docker build --no-cache` |
| "Container exits immediately" | Check logs: `docker logs <container-id>` |

### Getting Help

- Check logs in your platform's dashboard
- Verify all required environment variables are set
- Test locally with Docker first
- Ensure DATABASE_URL is accessible from your deployment region
