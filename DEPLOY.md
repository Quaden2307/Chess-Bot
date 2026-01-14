# Deployment Guide for Render.com

## Prerequisites
1. A GitHub account with your chess bot repository
2. A Render.com account (free tier available)

## Step-by-Step Deployment

### 1. Create Render Account
- Go to [render.com](https://render.com)
- Sign up with your GitHub account

### 2. Deploy the Application

#### Option A: Using Blueprint (Recommended)
1. Click "New +" → "Blueprint"
2. Connect your GitHub repository
3. Select the `chessbot` repository
4. Render will automatically detect `render.yaml`
5. Click "Apply" to deploy both services

#### Option B: Manual Deployment

**Deploy Backend:**
1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Configure:
   - **Name**: `chessbot-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn backend_improved:app`
   - **Instance Type**: Free
4. Add Environment Variable:
   - Key: `PORT`
   - Value: `5001`
5. Click "Create Web Service"

**Deploy Frontend:**
1. Click "New +" → "Static Site"
2. Connect your GitHub repository
3. Configure:
   - **Name**: `chessbot-frontend`
   - **Build Command**: `cd chess-frontend && npm install && npm run build`
   - **Publish Directory**: `chess-frontend/build`
4. Add Environment Variable:
   - Key: `REACT_APP_API_URL`
   - Value: `https://chessbot-backend.onrender.com/api` (replace with your backend URL)
5. Click "Create Static Site"

### 3. Important Notes

⚠️ **Model File Issue:**
- The `.pth` model files are too large for Git (excluded in `.gitignore`)
- You have two options:

**Option A: Upload Model Manually (Recommended)**
1. After backend deployment, go to your service dashboard
2. Navigate to "Shell" tab
3. Upload `chess_model_improved.pth` or `chess_model_best.pth`
4. Place it in the root directory

**Option B: Train on Render (Slow)**
1. SSH into your backend service
2. Run: `python improved_chess_ai.py`
3. This will generate a new model (takes time)

⚠️ **Free Tier Limitations:**
- Services spin down after 15 minutes of inactivity
- First request after spindown will be slow (cold start ~30-60 seconds)
- Limited to 750 hours/month

### 4. Update README with Live Link

Once deployed, update your README.md:
```markdown
🎮 **[Live Demo on Render.com](https://your-actual-app-name.onrender.com)**
```

Replace `your-actual-app-name` with your frontend service URL.

### 5. Verify Deployment

1. Open your frontend URL (e.g., `https://chessbot-frontend.onrender.com`)
2. Check that the board loads
3. Make a move and verify the AI responds
4. Check browser console for any API errors

### 6. Troubleshooting

**Backend won't start:**
- Check logs in Render dashboard
- Verify model file exists
- Ensure all dependencies installed

**Frontend can't connect to backend:**
- Verify `REACT_APP_API_URL` environment variable is correct
- Check CORS settings in backend
- Ensure backend service is running

**AI is very slow:**
- This is expected on free tier (limited CPU)
- Consider upgrading to paid tier for better performance
- Reduce search depth in `choose_best_move_improved` function

**Model file missing:**
- Upload manually or train on server
- Verify file path in backend code

## Alternative: Deploy Frontend to GitHub Pages

If you prefer, you can deploy the frontend to GitHub Pages (free):

1. Add homepage to `chess-frontend/package.json`:
```json
"homepage": "https://Quaden2307.github.io/Chess-Bot"
```

2. Install gh-pages:
```bash
cd chess-frontend
npm install gh-pages --save-dev
```

3. Add scripts to `package.json`:
```json
"scripts": {
  "predeploy": "npm run build",
  "deploy": "gh-pages -d build"
}
```

4. Deploy:
```bash
npm run deploy
```

5. Keep backend on Render, update `REACT_APP_API_URL` to point to Render backend.

## Continuous Deployment

Render automatically deploys when you push to GitHub:
1. Make changes locally
2. Commit and push to GitHub
3. Render automatically rebuilds and deploys

## Cost Considerations

**Free Tier:**
- Backend: Free (with spindown)
- Frontend: Free
- Database: Not needed

**Paid Tier ($7/month per service):**
- No spindown
- Better performance
- More memory

## Support

For issues:
1. Check Render logs in dashboard
2. Review browser console for frontend errors
3. Test API endpoints directly using curl/Postman
