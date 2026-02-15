# Free Cloud Deployment (3 minutes)

---

## Step 1: Neon Database (1 min)

1. Go to **[neon.tech](https://neon.tech)** → Sign up with GitHub
2. Click **"Create Project"** → name: `portfolio-optimizer` → Create
3. Click **"Connect"** → copy the full connection string (looks like: `postgresql://user:pass@ep-xxx.us-east-1.aws.neon.tech/portfolio-optimizer?sslmode=require`)

---

## Step 2: Koyeb Deployment (2 min)

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

4. Click **"Variables"** → Add these **EXACTLY**:

---

## Required Variables (MUST ADD)

| Variable | Value | Description |
|----------|-------|-------------|
| `SECRET_KEY` | `change-this-to-32-random-chars-now!` | Random string, min 32 chars |
| `DATABASE_URL` | `postgresql://...` | Paste your Neon connection string |
| `BASE_URL` | `https://your-app-name.koyeb.app` | Your Koyeb app URL (get this after first deploy, then update) |

---

## For AI Insights (Optional but Recommended)

Get a free API key from **[DeepSeek](https://platform.deepseek.com)** (has free credits) or use OpenAI.

| Variable | Value |
|----------|-------|
| `LLM_API_KEY` | `sk-...` |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` |
| `LLM_MODEL` | `deepseek-chat` |

---

## For Email (Registration & Password Reset)

**Without this, users cannot reset passwords.**

Use Gmail or any SMTP provider:

| Variable | Value |
|----------|-------|
| `SMTP_USER` | `your-email@gmail.com` |
| `SMTP_PASSWORD` | `your-app-password` |
| `ADMIN_EMAIL` | `your-email@gmail.com` |

**For Gmail:** Create an [App Password](https://myaccount.google.com/apppasswords) (requires 2FA enabled).

---

## Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AI_INSIGHTS` | `True` | Enable AI portfolio analysis |
| `MAX_PORTFOLIO_ASSETS` | `50` | Max assets per portfolio |
| `CACHE_TTL` | `3600` | Cache duration in seconds |

---

5. Click **"Deploy"** → wait 2-3 minutes

> **Note:** Koyeb uses port 8000 by default. The Dockerfile is already configured.

---

## Step 3: Get Your URL & Update BASE_URL

After first deploy:
1. Copy your Koyeb URL (e.g., `https://portfolio-optimizer-xxx.koyeb.app`)
2. Go to **Variables** in Koyeb dashboard
3. Update `BASE_URL` to match your actual URL
4. Redeploy

---

## Full Production .env Example

```bash
# REQUIRED
SECRET_KEY=your-random-secret-key-32-chars-min
DATABASE_URL=postgresql://user:pass@host.neon.tech/db?sslmode=require
BASE_URL=https://your-app.koyeb.app

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

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Email service not configured" | Add SMTP_USER, SMTP_PASSWORD, ADMIN_EMAIL |
| "Worker timeout" | Koyeb Eco has limited RAM - consider upgrading to Starter |
| "AI insights not working" | Add LLM_API_KEY from DeepSeek or OpenAI |
| "Password reset fails" | Check SMTP settings and BASE_URL is correct |
