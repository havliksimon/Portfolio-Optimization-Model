# Free Cloud Deployment Guide

Deploy your Portfolio Optimizer for **free** using Koyeb/Render + Neon DB with smart caching to minimize costs.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Koyeb/Render  │────▶│   Redis Cache    │────▶│   Neon PostgreSQL│
│   (Free Tier)   │     │   (Upstash)      │     │   (Free Tier)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐
│  Yahoo Finance  │
│  (Rate Limited) │
└─────────────────┘
```

## Why This Stack?

| Service | Free Tier | Purpose |
|---------|-----------|---------|
| **Koyeb** | 512MB RAM, 0.1 vCPU, sleeps after 30min inactivity | Application hosting |
| **Render** | 512MB RAM, sleeps after 15min inactivity | Alternative to Koyeb |
| **Neon** | 500MB storage, 190 compute hours/month | PostgreSQL database |
| **Upstash** | 10,000 commands/day | Redis caching |

**Estimated Monthly Cost: $0**

---

## Step 1: Neon Database Setup

### 1.1 Create Neon Account
1. Go to [neon.tech](https://neon.tech)
2. Sign up with GitHub or email
3. Create a new project

### 1.2 Get Connection String
1. In your Neon dashboard, click "Connect"
2. Select "PostgreSQL" as the client
3. Copy the connection string (it looks like):
   ```
   postgresql://username:password@host.neon.tech/dbname?sslmode=require
   ```

### 1.3 Configure for Serverless
Neon works great with serverless but needs connection pooling:

1. In Neon dashboard, go to "Connection Pooling"
2. Enable PgBouncer
3. Use the pooled connection string (port 5432 instead of direct)

---

## Step 2: Upstash Redis Setup (Caching)

### 2.1 Create Upstash Account
1. Go to [upstash.com](https://upstash.com)
2. Sign up with GitHub
3. Create a new Redis database

### 2.2 Get Connection Details
1. Select your database
2. Copy the `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN`

---

## Step 3: Koyeb Deployment

### 3.1 Prepare Your Repository

Create a `koyeb.yaml` file:

```yaml
name: portfolio-optimizer

# Build configuration
build:
  dockerfile: Dockerfile
  
# Environment variables (set in Koyeb dashboard)
env:
  - name: DATABASE_URL
    value: ${DATABASE_URL}
  - name: REDIS_URL
    value: ${REDIS_URL}
  - name: SECRET_KEY
    value: ${SECRET_KEY}
  - name: LLM_API_KEY
    value: ${LLM_API_KEY}

# Port configuration
ports:
  - port: 5000
    protocol: http

# Health check
healthcheck:
  http:
    path: /health
    port: 5000
  initial_delay: 30s
  interval: 10s
  timeout: 5s
  unhealthy_threshold: 3
```

### 3.2 Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p data

# Environment
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Expose port
EXPOSE 5000

# Run with gunicorn
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 0 app:app
```

### 3.3 Deploy to Koyeb

1. Go to [app.koyeb.com](https://app.koyeb.com)
2. Click "Create App"
3. Connect your GitHub repository
4. Select "Docker" as the builder
5. Add environment variables:
   - `DATABASE_URL`: Your Neon connection string
   - `REDIS_URL`: Your Upstash Redis URL
   - `SECRET_KEY`: A random secret (generate with `openssl rand -hex 32`)
   - `LLM_API_KEY`: Your DeepSeek/OpenAI key (optional)
6. Click "Deploy"

---

## Step 4: Render Deployment (Alternative)

### 4.1 Create `render.yaml`

```yaml
services:
  - type: web
    name: portfolio-optimizer
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: DATABASE_URL
        sync: false
      - key: REDIS_URL
        sync: false
      - key: SECRET_KEY
        generateValue: true
      - key: PYTHON_VERSION
        value: 3.11.0
```

### 4.2 Deploy to Render

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Select Python runtime
5. Add environment variables:
   - `DATABASE_URL`
   - `REDIS_URL`
   - `SECRET_KEY`
6. Click "Create Web Service"

---

## Step 5: Caching Configuration

### 5.1 Why Caching Matters

Without caching:
- Every user request hits the database
- Yahoo Finance API calls happen repeatedly
- Free tier quotas exhausted quickly

With caching:
- Market data cached for 1 hour
- Analysis results cached for 24 hours
- Database queries minimized

### 5.2 Environment Variables

Add to your `.env` file:

```env
# Database
DATABASE_URL=postgresql://...neon.tech/...?sslmode=require

# Redis Cache (Upstash)
REDIS_URL=rediss://default:...@...upstash.io:6379
UPSTASH_REDIS_REST_URL=https://...upstash.io
UPSTASH_REDIS_REST_TOKEN=...

# Cache Configuration
CACHE_TTL=3600  # 1 hour for market data
ANALYSIS_CACHE_TTL=86400  # 24 hours for analysis results
ENABLE_CACHE=true

# Neon-specific
NEON_POOLER_MODE=true
```

---

## Step 6: Keep-Alive (Prevent Sleeping)

### Option A: UptimeRobot (Free)

1. Go to [uptimerobot.com](https://uptimerobot.com)
2. Create account
3. Add new monitor:
   - Type: HTTP(s)
   - URL: Your Koyeb/Render URL
   - Interval: Every 5 minutes (free tier)

### Option B: GitHub Actions

Create `.github/workflows/keepalive.yml`:

```yaml
name: Keep Alive

on:
  schedule:
    - cron: '*/10 * * * *'  # Every 10 minutes

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Ping website
        run: curl -s https://your-app-url.com/health > /dev/null
```

---

## Cost Optimization Tips

### 1. Smart Cache Keys
```python
# Cache market data by ticker and date
cache_key = f"market_data:{ticker}:{datetime.now().strftime('%Y-%m-%d')}"
```

### 2. Batch Operations
```python
# Fetch multiple tickers in one request
# instead of individual requests
```

### 3. Lazy Loading
```python
# Only fetch data when needed
# Cache the result immediately
```

### 4. Database Connection Pooling
Already configured in the app for Neon compatibility.

---

## Monitoring

### Free Monitoring Tools

1. **UptimeRobot**: Uptime monitoring
2. **Sentry** (free tier): Error tracking
3. **Neon Dashboard**: Database metrics
4. **Upstash Dashboard**: Cache hit rates

### Health Check Endpoint

The app includes a `/health` endpoint that returns:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "database": "connected",
  "cache": "connected"
}
```

---

## Troubleshooting

### Database Connection Issues
```
Error: Connection terminated unexpectedly
```
**Fix**: Enable connection pooling in Neon dashboard

### Cache Connection Issues
```
Error: Connection refused
```
**Fix**: Use the `rediss://` (SSL) URL from Upstash

### Out of Memory
```
Error: Container killed due to memory
```
**Fix**: Reduce gunicorn workers from 2 to 1

### Cold Start Slow
**Fix**: Use UptimeRobot to keep the instance warm

---

## Summary

| Component | Service | Cost |
|-----------|---------|------|
| App Hosting | Koyeb/Render | Free |
| Database | Neon | Free (500MB) |
| Cache | Upstash | Free (10k/day) |
| Monitoring | UptimeRobot | Free |
| **Total** | | **$0/month** |

---

## Next Steps

1. Set up custom domain (free with Cloudflare)
2. Enable HTTPS (automatic on Koyeb/Render)
3. Add Sentry for error tracking
4. Configure backup schedule in Neon

For questions, open an issue on GitHub.
