# Vercel Deployment Guide

## Prerequisites
1. A GitHub account with your chess bot repository
2. A Vercel account (free tier available)

## Step-by-Step Deployment

### 1. Create Vercel Account
- Go to [vercel.com](https://vercel.com)
- Sign up with your GitHub account

### 2. Deploy the Application

#### Import Project
1. Click "Add New..." → "Project"
2. Import your GitHub repository (`Chess-Bot`)
3. Vercel will automatically detect the `vercel.json` configuration

#### Configure Project
1. **Framework Preset**: Select "Create React App"
2. **Root Directory**: Leave as `./` (root)
3. **Build Command**: Will use the configuration from `vercel.json`
4. **Output Directory**: `chess-frontend/build`

#### Environment Variables
Add the following environment variable:
- Key: `REACT_APP_API_URL`
- Value: Leave empty initially (will be auto-set to your Vercel domain)

After first deployment, update it to:
- Value: `https://your-project-name.vercel.app/api`

### 3. Deploy
1. Click "Deploy"
2. Wait for build to complete (2-3 minutes)
3. You'll get a URL like `https://your-project-name.vercel.app`

### 4. Important: Model File

⚠️ **Critical Issue**: The trained model file (`.pth`) is too large for Git and Vercel's deployment.

**Solutions:**

**Option A: Use Vercel Blob Storage (Recommended)**
1. Go to your Vercel project dashboard
2. Navigate to "Storage" → "Create Database" → "Blob"
3. Upload your `chess_model_improved.pth` or `chess_model_best.pth`
4. Update `api/index.py` to load from Vercel Blob
5. Redeploy

**Option B: Use External Storage**
1. Upload model to S3, Google Cloud Storage, or Cloudflare R2
2. Update `api/index.py` to download model on cold start
3. Add URL as environment variable

**Option C: Use Smaller Model (Quick Fix)**
- The app will work without the model file but with random evaluations
- Train a smaller model with fewer positions for testing

### 5. Update Environment Variable

After deployment:
1. Go to Project Settings → Environment Variables
2. Update `REACT_APP_API_URL` to: `https://your-actual-domain.vercel.app/api`
3. Redeploy (trigger by pushing a new commit or manual redeploy)

### 6. Test Deployment

1. Open your Vercel URL
2. Check browser console (F12) for errors
3. Make a move - verify AI responds
4. Check Network tab to see API calls

### 7. Update README

Update your `README.md` with the live link:
```markdown
🎮 **[Live Demo on Vercel](https://your-project-name.vercel.app)**
```

## Vercel Configuration Files

The deployment uses these files:

### `vercel.json`
- Configures both frontend and backend (Flask API)
- Routes `/api/*` to Python serverless functions
- Routes everything else to React frontend

### `api/index.py`
- Flask backend adapted for Vercel serverless
- All endpoints work as serverless functions
- Automatically handles CORS

### `chess-frontend/package.json`
- Includes `vercel-build` script
- Builds React app for production

## Troubleshooting

### "Module not found" errors
- Check that all dependencies are in `requirements.txt`
- Redeploy after adding missing packages

### API calls fail with 404
- Verify `REACT_APP_API_URL` is set correctly
- Check that routes in `vercel.json` match your API paths
- Look at Function logs in Vercel dashboard

### Model file not found
- This is expected if you haven't uploaded the model
- See "Model File" section above for solutions
- App will use untrained model (random moves)

### Frontend builds but shows blank page
- Check browser console for errors
- Verify `outputDirectory` in `vercel.json`
- Check that `homepage` in `package.json` is not set incorrectly

### Cold starts are slow (15-30 seconds)
- This is normal for serverless functions on first request
- Free tier has cold starts after inactivity
- Upgrade to Pro for faster cold starts
- Consider keeping a ping endpoint warm

## Performance Considerations

### Free Tier Limitations
- **Bandwidth**: 100GB/month
- **Execution**: 100GB-hours/month
- **Cold Starts**: Functions sleep after inactivity
- **Build Time**: Limited concurrent builds

### Optimization Tips
1. **Reduce model size**: Train with fewer layers
2. **Add caching**: Cache evaluations for common positions
3. **Reduce search depth**: Use depth=1 or depth=2 for faster moves
4. **Use edge functions**: Consider Vercel Edge Functions for faster responses

## Continuous Deployment

Vercel automatically deploys when you push to GitHub:
1. Make changes locally
2. `git commit` and `git push`
3. Vercel automatically rebuilds and deploys
4. Preview deployments for each commit
5. Production deployment on main branch

## Custom Domain (Optional)

To use your own domain:
1. Go to Project Settings → Domains
2. Add your domain
3. Update DNS records as instructed
4. Vercel handles SSL automatically

## Monitoring

Vercel provides:
- **Analytics**: Page views, performance metrics
- **Logs**: Function execution logs
- **Errors**: Real-time error tracking
- **Performance**: Core Web Vitals monitoring

Access these in your project dashboard.

## Cost Considerations

**Free Tier (Hobby)**:
- Unlimited projects
- 100GB bandwidth/month
- Serverless functions included
- No credit card required

**Pro Tier ($20/month)**:
- Faster builds
- No cold starts
- More bandwidth
- Priority support
- Team collaboration

## Alternative: Split Deployment

If serverless functions don't work well for your model:

1. **Frontend on Vercel**: Deploy only the React app
2. **Backend on Render/Railway**: Deploy Flask backend separately
3. Update `REACT_APP_API_URL` to point to external backend

## Support Resources

- [Vercel Docs](https://vercel.com/docs)
- [Python on Vercel](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [Static Files](https://vercel.com/docs/deployments/static-files)

For issues, check:
1. Vercel deployment logs
2. Function logs (Runtime Logs)
3. Browser console (F12)
4. Network tab for API calls
