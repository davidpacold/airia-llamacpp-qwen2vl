# Airia llama.cpp + Qwen2-VL 72B

High-performance vision model deployment using llama.cpp with Qwen2-VL 72B for Azure AKS with A10 GPUs.

## Features

- **llama.cpp** with CUDA 12.2 support
- **Qwen2-VL 72B** (Q4_K_M quantization)
- Optimized for NVIDIA A10 GPU (24GB VRAM)
- Full GPU offloading (all layers)
- Flash attention enabled
- Continuous batching for concurrent requests

## Performance

- **Inference Speed**: ~1-1.5 seconds per request
- **VRAM Usage**: ~20-22GB
- **Quality**: Superior OCR and document understanding
- **Concurrent Requests**: Up to 8 parallel

## Quick Start

### 1. Build via GitHub Actions

Simply push to `main` branch and GitHub Actions will automatically build and push to GitHub Container Registry (ghcr.io).

```bash
git push origin main
```

The image will be available at:
```
ghcr.io/YOUR_USERNAME/airia-llamacpp-qwen2vl:latest
```

### 2. Pull the Image

```bash
# Public repo (no auth needed)
docker pull ghcr.io/YOUR_USERNAME/airia-llamacpp-qwen2vl:latest

# Private repo (create GitHub token with read:packages scope)
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
docker pull ghcr.io/YOUR_USERNAME/airia-llamacpp-qwen2vl:latest
```

### 3. Deploy to Kubernetes

Update `deployment.yaml` with your GitHub username, then:

```bash
kubectl apply -f deployment.yaml
```

For private repos, create an image pull secret:

```bash
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=YOUR_USERNAME \
  --docker-password=YOUR_GITHUB_TOKEN \
  -n airia
```

## Manual Build (Optional)

```bash
docker build -t ghcr.io/YOUR_USERNAME/airia-llamacpp-qwen2vl:latest .
docker push ghcr.io/YOUR_USERNAME/airia-llamacpp-qwen2vl:latest
```

## Configuration

### Environment Variables

- `LLAMA_CONTEXT_SIZE`: Context window size (default: 8192)
- `LLAMA_BATCH_SIZE`: Batch size for processing (default: 2048)
- `LLAMA_THREADS`: CPU threads (default: 24)
- `LLAMA_GPU_LAYERS`: GPU layers to offload (default: 99 - all)
- `LLAMA_PARALLEL`: Max parallel requests (default: 8)

### Resource Allocations

**Requests:**
- CPU: 8 cores
- Memory: 32Gi
- GPU: 1x NVIDIA A10

**Limits:**
- CPU: 32 cores
- Memory: 128Gi
- GPU: 1x NVIDIA A10

## API

The server exposes an OpenAI-compatible API on port 8008:

```bash
curl http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ]
  }'
```

## Model Details

- **Model**: Qwen2-VL 72B Instruct
- **Quantization**: Q4_K_M (optimal quality/performance balance)
- **Architecture**: Vision-Language Model
- **Capabilities**:
  - Image understanding
  - OCR (Optical Character Recognition)
  - Document analysis
  - Visual question answering
  - Multilingual support

## Requirements

- Kubernetes cluster with NVIDIA GPU support
- Node with NVIDIA A10 GPU (24GB VRAM minimum)
- CUDA 12.2+ compatible drivers
- Azure Container Registry access
