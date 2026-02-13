# Deployment Guide

This guide provides step-by-step instructions for deploying the Face Detection Dashboard to various platforms.

## Prerequisites

- Python 3.10 or higher
- Git (for version control)
- GitHub account (for Streamlit Cloud)
- Docker (optional, for containerized deployment)

## Quick Deploy: Streamlit Cloud

### Step 1: Prepare Your Repository

1. Initialize Git (if not already done):
   ```bash
   git init
   ```

2. Create a `.gitignore` file (already included in the project)

3. Add and commit your files:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   ```

### Step 2: Push to GitHub

1. Create a new repository on GitHub (if you haven't already)

2. Add the remote and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in the details:
   - **Repository**: Select your repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `app.py`
5. Click **"Deploy"**

**Deployment Time**: First deployment takes 5-10 minutes as it:
- Installs all Python dependencies
- Downloads YOLOv8 model weights (~6MB)
- Sets up the environment

**Note**: Streamlit Cloud free tier has resource limits. For GPU support, consider paid tiers or other platforms.

## Docker Deployment

### Build and Run Locally

```bash
# Build the image
docker build -t face-detection-dashboard .

# Run the container
docker run -p 8501:8501 face-detection-dashboard
```

### Deploy to Cloud Container Services

#### AWS ECS / Fargate
1. Build and push to ECR:
   ```bash
   aws ecr create-repository --repository-name face-detection-dashboard
   docker tag face-detection-dashboard:latest YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/face-detection-dashboard:latest
   docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/face-detection-dashboard:latest
   ```
2. Create ECS task definition and service

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/face-detection-dashboard
gcloud run deploy --image gcr.io/PROJECT_ID/face-detection-dashboard --platform managed
```

#### Azure Container Instances
```bash
# Build and push to Azure Container Registry
az acr build --registry YOUR_REGISTRY --image face-detection-dashboard:latest .
az container create --resource-group YOUR_GROUP --name face-detection --image YOUR_REGISTRY.azurecr.io/face-detection-dashboard:latest --dns-name-label YOUR_APP_NAME --ports 8501
```

## Environment Variables

For production deployments, you may want to set:

- `STREAMLIT_SERVER_PORT`: Port number (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)

## Performance Considerations

### CPU vs GPU

- **CPU**: Works fine for Haar cascade and small videos
- **GPU**: Recommended for YOLOv8 and real-time processing
  - NVIDIA GPU with CUDA support
  - Install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Resource Requirements

**Minimum (CPU)**:
- 2 CPU cores
- 2GB RAM
- Suitable for: Small videos, Haar cascade

**Recommended (GPU)**:
- 4+ CPU cores
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- Suitable for: Real-time webcam, large videos, YOLOv8

## Troubleshooting

### Common Issues

1. **Model download fails**
   - Check internet connection
   - YOLOv8 models download automatically on first use (~6MB)

2. **Out of memory errors**
   - Reduce `max_frames` in sidebar
   - Use Haar cascade instead of YOLO
   - Process smaller video chunks

3. **Webcam not accessible**
   - Check camera permissions
   - On cloud platforms, webcam access may be limited
   - Use uploaded video instead

4. **Slow performance**
   - Enable GPU if available
   - Reduce video resolution
   - Lower confidence threshold for faster processing

## Security Considerations

- Don't commit sensitive data or API keys
- Use environment variables for configuration
- Enable HTTPS in production
- Set up authentication if needed (Streamlit Cloud Pro or custom)

## Monitoring

For production deployments, consider:

- Application monitoring (e.g., Sentry)
- Log aggregation (e.g., CloudWatch, Datadog)
- Performance metrics (CPU, memory, GPU usage)

## Support

For issues or questions:
- Check the [README.md](README.md) for usage instructions
- Review Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
- Check YOLOv8 documentation: [docs.ultralytics.com](https://docs.ultralytics.com)
