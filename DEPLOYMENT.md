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

```
SECRET_KEY=change-this-to-random-string-at-least-32-chars
DATABASE_URL=<paste-your-neon-connection-string-here>
```

5. Click **"Deploy"** → wait 2 minutes

---

## Step 3: Open Your App

Koyeb gives you a URL like:
`https://portfolio-optimizer-xxx.koyeb.app`

---

## Variables to Add (COPY THIS)

| Variable | Value |
|----------|-------|
| `SECRET_KEY` | `your-secret-key-change-this-32-chars-minimum` |
| `DATABASE_URL` | `postgresql://user:password@host.neon.tech/dbname?sslmode=require` |

---

## Optional: Faster Loading with Redis

1. Go to **[upstash.com](https://upstash.com)** → Sign up with GitHub
2. Click **"Create Database"** → name: `cache`
3. Copy the **REST URL**
4. In Koyeb, add variable: `REDIS_URL=<your-upstash-url>`
