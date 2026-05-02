# Deployment

## Backend on Render

Create a Web Service from this repository and use these settings:

```text
Root Directory: stress_detector
Build Command: pip install -r requirements.txt
Start Command: gunicorn api.app:app
Health Check Path: /health
```

If Render is already connected to this repo, redeploy after pushing these files.

Backend URL used by the frontend:

```text
https://stress-detector-api-yhx4.onrender.com
```

## Frontend on Vercel

Use these settings if Vercel points directly at the React folder:

```text
Root Directory: stress_detector/frontend
Build Command: npm run build
Output Directory: dist
```

Use these settings if Vercel points at the project folder instead:

```text
Root Directory: stress_detector
Build Command: npm run build
Output Directory: frontend/dist
```

After changing settings, redeploy with:

```text
Redeploy without build cache
```

## How to verify

The deployed Vercel JavaScript bundle must contain:

```text
stress-detector-api-yhx4.onrender.com
```

If it still calls `/predict`, the deployed frontend is stale or using the wrong root directory.
