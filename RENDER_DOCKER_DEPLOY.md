# Render.com Docker Deployment Guide

## Overview

Your chess bot is now configured for Docker deployment on Render.com. The Docker container includes:
- React frontend (built and bundled)
- Flask backend API
- PyTorch neural network
- All dependencies

## Files Created

1. **`Dockerfile`** - Multi-stage Docker build
   - Stage 1: Builds React frontend
   - Stage 2: Sets up Python backend and serves frontend

2. **`render.yaml`** - Render deployment configuration
   - Single web service using Docker
   - Auto-detected by Render

3. **`.dockerignore`** - Excludes unnecessary files from Docker build

4. **Updated `backend_improved.py`**
   - Now serves frontend static files
   - Single application serving both frontend and backend
   - Uses PORT environment variable

5. **Updated `App.tsx`**
   - Uses relative URLs in production
   - Automatically works with same-domain deployment

## Deployment Steps

### 1. Push to GitHub

Your code is ready to push:

```bash
git add -A
git commit -m "Configure Docker deployment for Render.com"
git push origin main
```

### 2. Create Render Account

1. Go to [render.com](https://render.com)
2. Sign up with your GitHub account

### 3. Deploy with Blueprint (Recommended)

1. Click "New +" → "Blueprint"
2. Connect your GitHub repository
3. Select `Chess-Bot` repository
4. Render auto-detects `render.yaml`
5. Click "Apply"

### 4. Manual Deployment (Alternative)

If Blueprint doesn't work:

1. Click "New +" → "Web Service"
2. Connect GitHub repository
3. Configure:
   - **Name**: `chess-bot`
   - **Environment**: `Docker`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Dockerfile Path**: `./Dockerfile`
   - **Docker Build Context**: `.`
4. Set Environment Variables:
   - `PORT`: `5001` (auto-set by Render)
   - `FLASK_APP`: `backend_improved.py`
   - `PYTHONUNBUFFERED`: `1`
5. Click "Create Web Service"

### 5. Wait for Build

- First build takes 5-10 minutes
- Docker builds both frontend and backend
- You'll see build logs in real-time

### 6. Get Your URL

Once deployed, you'll get a URL like:
```
https://chess-bot-xxxx.onrender.com
```

### 7. Update README

Update line 3 in README.md:
```markdown
🎮 **[Live Demo on Render](https://chess-bot-xxxx.onrender.com)**
```

## Important Notes

### ⚠️ Model File

The trained model (`.pth`) is excluded from Docker build due to size. Options:

**Option A: Environment Variable (Recommended)**
1. Upload model to cloud storage (S3, Google Cloud Storage, Cloudflare R2)
2. Get public URL
3. Add environment variable in Render: `MODEL_URL`
4. Update backend to download on startup

**Option B: Render Disk**
1. Add persistent disk to Render service
2. Upload model file manually via SSH
3. Update backend to load from disk path

**Option C: No Model (Testing)**
- Deploy without model
- AI will use untrained network (random-ish moves)
- Good for testing deployment

### 🆓 Free Tier Info

**Render Free Tier:**
- 750 hours/month per service
- Services spin down after 15 minutes of inactivity
- Cold start: 30-60 seconds for first request
- Perfect for portfolio projects

**Paid Tier ($7/month):**
- No spindown
- Faster performance
- More memory

### 🐳 Docker Benefits

1. **Consistency**: Same environment everywhere
2. **Single Service**: Frontend + Backend in one container
3. **Easy Deployment**: Just push to GitHub
4. **Portable**: Can deploy anywhere (AWS, GCP, Azure, etc.)

## Architecture

```
┌─────────────────────────────────────┐
│     Render.com (Docker)             │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   Flask Application          │  │
│  │   (Port 5001)                │  │
│  │                              │  │
│  │  /api/*  → Backend API       │  │
│  │  /*      → React Frontend    │  │
│  │                              │  │
│  │  ├─ PyTorch Model            │  │
│  │  ├─ Chess Logic              │  │
│  │  └─ Static Files (built)    │  │
│  └──────────────────────────────┘  │
│                                     │
└─────────────────────────────────────┘
```

## Testing Locally with Docker

Before deploying, test locally:

```bash
# Build Docker image
docker build -t chess-bot .

# Run container
docker run -p 5001:5001 chess-bot

# Open browser
open http://localhost:5001
```

## Troubleshooting

### Build Fails

**"npm install failed"**
- Check `package.json` dependencies
- Ensure `package-lock.json` is committed

**"pip install failed"**
- Check `requirements.txt` syntax
- Some packages need system dependencies

**"Dockerfile not found"**
- Verify file is in root directory
- Check `render.yaml` path

### Runtime Errors

**"Module not found"**
- Check all imports in `backend_improved.py`
- Ensure all packages in `requirements.txt`

**"Frontend shows blank page"**
- Check browser console for errors
- Verify build completed successfully
- Check Flask is serving static files

**"API calls fail"**
- Check `/api/health` endpoint
- Verify CORS settings
- Check Render logs for errors

### Performance Issues

**Slow cold starts**
- Normal on free tier (30-60s)
- Upgrade to paid tier for instant response
- Or keep service warm with ping service

**AI moves are slow**
- PyTorch on CPU is slower
- Consider reducing search depth
- Or upgrade to instance with more CPU

## Monitoring

**Render Dashboard provides:**
- Real-time logs
- Metrics (CPU, Memory, Response Time)
- Deploy history
- Custom domains
- Environment variables

**Access logs:**
1. Go to your service dashboard
2. Click "Logs" tab
3. See real-time application output

## Updating Your App

Render auto-deploys on push:

```bash
# Make changes
git add .
git commit -m "Update chess bot"
git push origin main

# Render automatically:
# 1. Detects push
# 2. Builds new Docker image
# 3. Deploys updated version
# 4. Zero-downtime deployment
```

## Custom Domain (Optional)

1. Go to service settings
2. Click "Custom Domain"
3. Add your domain (e.g., `chess.yourdomain.com`)
4. Update DNS records as instructed
5. SSL certificate auto-generated

## Environment Variables

Available environment variables:

- `PORT` - Set by Render (usually 10000)
- `FLASK_APP` - Your Flask application file
- `PYTHONUNBUFFERED` - Python logging
- `MODEL_URL` - (Optional) URL to model file

Add more in Render dashboard → Environment tab.

## Health Checks

Render pings `/api/health` every 30 seconds:
- Response 200 = Healthy
- No response/Error = Unhealthy (restart)

## Cost Estimate

**Free Tier:**
- $0/month
- Perfect for portfolio
- Some limitations

**Starter ($7/month):**
- Always-on service
- No cold starts
- Better for real use

## Alternative: Docker Compose Locally

For local development with Docker:

```yaml
# docker-compose.yml
version: '3.8'
services:
  chess-bot:
    build: .
    ports:
      - "5001:5001"
    environment:
      - PORT=5001
      - FLASK_APP=backend_improved.py
    volumes:
      - ./models:/app/models
```

Run with: `docker-compose up`

## Support

**Render Support:**
- [Documentation](https://render.com/docs)
- [Community Forum](https://community.render.com)
- Email support (paid tiers)

**Common Issues:**
1. Check Render logs first
2. Test locally with Docker
3. Verify all files committed to Git
4. Check environment variables

## Success Checklist

- [ ] Code pushed to GitHub
- [ ] Render service created
- [ ] Build completed successfully
- [ ] Service is live at Render URL
- [ ] Frontend loads in browser
- [ ] Can make moves on board
- [ ] AI responds (even if randomly)
- [ ] No console errors
- [ ] README updated with live URL

## Next Steps

1. Deploy to Render
2. Test the live application
3. Add custom domain (optional)
4. Upload model file for better AI
5. Share URL on resume/portfolio!

Good luck with your deployment! 🚀
