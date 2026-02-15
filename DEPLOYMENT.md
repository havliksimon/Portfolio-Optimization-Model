# Free Cloud Deployment (5 minutes)

Deploy for **$0/month** using just your browser - no code, no YAML, no terminal.

---

## Step 1: Neon Database (2 min)

1. Go to **[neon.tech](https://neon.tech)** → Sign up with GitHub
2. Click **"Create Project"** → name it "portfolio-optimizer"
3. Click **"Connect"** → copy the connection string (save it for later)

---

## Step 2: Koyeb Deployment (3 min)

1. Go to **[koyeb.com](https://koyeb.com)** → Sign up with GitHub
2. Click **"Create App"**
3. Select your GitHub repo
4. Fill in the form:

| Setting | Value |
|---------|-------|
| **Name** | portfolio-optimizer |
| **Region** | choose closest to you |
| **Branch** | main |
| **Builder** | Docker |
| **Dockerfile** | `Dockerfile` |

5. Click **"Variables"** → Add these:

```
SECRET_KEY=random-string-here-min-32-chars
DATABASE_URL=postgresql://user:pass@host.neon.tech/dbname?sslmode=require
```

6. Click **"Deploy"** → wait 2-3 minutes

---

## Step 3: Open Your App

Once deployed, Koyeb gives you a URL like:
```
https://portfolio-optimizer-yourname.koyeb.app
```

---

## Optional: Redis Cache (for faster loading)

1. Go to **[upstash.com](https://upstash.com)** → Sign up with GitHub
2. Create Redis database → copy URL
3. In Koyeb, add variable: `REDIS_URL=your-upstash-url`

---

## Quick Reference

| Service | URL | What you need |
|---------|-----|---------------|
| **Neon** | neon.tech | Connection string |
| **Koyeb** | koyeb.com | Your GitHub repo |
| **Upstash** | upstash.com | Redis URL (optional) |

---

## Troubleshooting

**App not loading?**
- Check logs in Koyeb dashboard
- Verify DATABASE_URL is correct

**Database errors?**
- Make sure Neon project is not paused (check Neon dashboard)

**Want custom domain?**
- Koyeb → Settings → Domains → Add your domain
