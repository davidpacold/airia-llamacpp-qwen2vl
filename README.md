# Airia llama.cpp + Qwen2-VL

High-performance vision model deployment using llama.cpp with Qwen2-VL for Azure AKS with A10 GPUs.

## Features

- **llama.cpp** with CUDA 12.2 support and Flash Attention
- **Qwen2-VL 7B** (Q4_K_M quantization) - optimized for 24GB VRAM
- Full GPU offloading (all 28 layers)
- Continuous batching for concurrent requests
- OpenAI-compatible API endpoint
- Persistent model storage with PVC
- Production-ready with security best practices

## Performance

**Tested on NVIDIA A10-24Q GPU:**
- **VRAM Usage**: 8.4 GB / 24.5 GB (35% utilization, 65% headroom)
- **Prompt Processing**: 263-1,092 tokens/sec (varies by complexity)
- **Token Generation**: 6-44 tokens/sec for responses
- **Image Processing**: 399ms - 6.1 seconds per image
- **Concurrent Requests**: Up to 8 parallel slots
- **Context Window**: 32,768 tokens (4,096 per slot)

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

### 4. Hugging Face Token (Optional)

If you need to download gated models from Hugging Face, create a secret with your HF token:

```bash
kubectl create secret generic hf-token \
  --from-literal=token=YOUR_HUGGINGFACE_TOKEN \
  -n airia
```

To get your token:
1. Go to https://huggingface.co/settings/tokens
2. Create a token with "Read access to contents of all public gated repos you can access"
3. Use this token in the command above

**Note:** The Qwen2-VL 72B model used by default does not require authentication.

## Manual Build (Optional)

```bash
docker build -t ghcr.io/YOUR_USERNAME/airia-llamacpp-qwen2vl:latest .
docker push ghcr.io/YOUR_USERNAME/airia-llamacpp-qwen2vl:latest
```

## Configuration

### Critical Context Window Settings

**IMPORTANT**: For image processing with Qwen2-VL, proper context window sizing is critical:

- **Minimum per image**: 1024 tokens required by Qwen2-VL
- **Current configuration**: 32,768 total tokens / 8 parallel slots = 4,096 tokens per slot
- **Available for text**: 3,072 tokens per slot (after 1,024 for image)

**Context Window Sizing Formula:**
```
tokens_per_slot = LLAMA_CONTEXT_SIZE / LLAMA_PARALLEL
available_for_text = tokens_per_slot - 1024 (image requirement)
```

If you see **400 Bad Request** errors, increase `LLAMA_CONTEXT_SIZE`:
- 8,192 → 1,024 per slot → ❌ No room for text
- 16,384 → 2,048 per slot → ✅ Minimal (1,024 for text)
- **32,768 → 4,096 per slot → ✅ Recommended (3,072 for text)**

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CONTEXT_SIZE` | 32768 | Total context window (model max: 32,768) |
| `LLAMA_BATCH_SIZE` | 2048 | Batch size for prompt processing |
| `LLAMA_THREADS` | 24 | CPU threads for computation |
| `LLAMA_GPU_LAYERS` | 99 | GPU layers to offload (99 = all 28 layers) |
| `LLAMA_PARALLEL` | 8 | Max concurrent request slots |
| `CUDA_VISIBLE_DEVICES` | 0 | GPU device ID |
| `HF_TOKEN` | - | Hugging Face token (optional for gated models) |

### Resource Allocations

**Requests (minimum):**
- CPU: 8 cores
- Memory: 32Gi
- GPU: 1x NVIDIA A10 (24GB VRAM)

**Limits (maximum):**
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

- **Model**: Qwen2-VL 7B Instruct
- **Quantization**: Q4_K_M (optimal quality/performance balance for 24GB VRAM)
- **Architecture**: Vision-Language Model (28 layers)
- **Context Window**: 32,768 tokens maximum
- **Capabilities**:
  - Image understanding and analysis
  - OCR (Optical Character Recognition)
  - Document analysis (PDFs, screenshots, images)
  - Visual question answering
  - Multilingual support
  - Multi-image reasoning

## Monitoring and Metrics

### Check Pod Status
```bash
kubectl get pods -n airia -l app=airia-model-hosting-qwen2-vl-72b-llamacpp
```

### View Logs
```bash
# Real-time logs
kubectl logs -n airia -l app=airia-model-hosting-qwen2-vl-72b-llamacpp -f

# Check for errors
kubectl logs -n airia -l app=airia-model-hosting-qwen2-vl-72b-llamacpp --tail=100 | grep -i error

# Check request status codes
kubectl logs -n airia -l app=airia-model-hosting-qwen2-vl-72b-llamacpp --tail=100 | grep -E "200|400|500"
```

### GPU Metrics
```bash
# Check GPU utilization
POD=$(kubectl get pod -n airia -l app=airia-model-hosting-qwen2-vl-72b-llamacpp -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n airia $POD -- nvidia-smi
```

### llama.cpp Metrics
```bash
# Check server metrics (KV cache, tokens, requests)
POD=$(kubectl get pod -n airia -l app=airia-model-hosting-qwen2-vl-72b-llamacpp -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n airia $POD -- curl -s http://localhost:8008/metrics
```

### Health Check
```bash
# Check if server is responding
kubectl exec -n airia $POD -- curl -s http://localhost:8008/health
```

## Troubleshooting

### 400 Bad Request Errors

**Symptom**: Vision API requests return HTTP 400 with "insufficient context" errors

**Root Cause**: Context window too small for image processing

**Solution**:
1. Check current context size in logs:
   ```bash
   kubectl logs -n airia -l app=airia-model-hosting-qwen2-vl-72b-llamacpp | grep "n_ctx_seq"
   ```

2. If `n_ctx_seq < 2048`, increase `LLAMA_CONTEXT_SIZE`:
   - Edit `deployment.yaml` and `Dockerfile`
   - Set `LLAMA_CONTEXT_SIZE: "32768"` for optimal performance
   - Rebuild and redeploy

3. Verify fix:
   ```bash
   # Should show: n_ctx_seq = 4096
   kubectl logs -n airia -l app=airia-model-hosting-qwen2-vl-72b-llamacpp | grep "slot init"
   ```

### Pod CrashLoopBackOff

**Common Causes**:
1. Missing CUDA libraries
2. Insufficient VRAM
3. Model download failures

**Debug Steps**:
```bash
# Check pod events
kubectl describe pod -n airia <pod-name>

# Check init container logs (model download)
kubectl logs -n airia <pod-name> -c model-downloader

# Check main container logs
kubectl logs -n airia <pod-name> -c qwen2-vl-llamacpp
```

### Slow Performance

**Check**:
1. GPU utilization (should spike during inference)
2. Context window size (larger = slower)
3. Concurrent requests (check `llama_requests_processing` metric)
4. VRAM usage (should be ~8-10GB, not at limit)

**Optimize**:
- Reduce `LLAMA_PARALLEL` if memory constrained
- Reduce `LLAMA_CONTEXT_SIZE` if latency is critical
- Increase `LLAMA_BATCH_SIZE` for throughput

## Optimization Tips

### Maximize Throughput
- **Increase parallel slots**: `LLAMA_PARALLEL: "8"` (current)
- **Enable continuous batching**: Already enabled (`--cont-batching`)
- **Flash attention**: Already enabled (`--flash-attn on`)
- **Full GPU offload**: `LLAMA_GPU_LAYERS: "99"` (all layers)

### Minimize Latency
- **Reduce context size**: Use 16384 if 32768 is too slow
- **Reduce parallel slots**: `LLAMA_PARALLEL: "4"` for faster single requests
- **Increase batch size**: `LLAMA_BATCH_SIZE: "4096"` for faster prompt processing

### Balance Quality and Speed
- Current Q4_K_M quantization is optimal for A10 GPU
- For higher quality, use Q5_K_M (requires more VRAM)
- For faster inference, use Q3_K_M (lower quality)

### Multi-GPU Setup
To scale across multiple GPUs:
1. Set `nvidia.com/gpu: "2"` in deployment.yaml
2. Set `CUDA_VISIBLE_DEVICES: "0,1"`
3. Increase `LLAMA_PARALLEL` proportionally

## Requirements

- Kubernetes cluster with NVIDIA GPU support
- Node with NVIDIA A10 GPU (24GB VRAM minimum)
- CUDA 12.2+ compatible drivers
- 60Gi persistent storage for model weights
- GitHub Container Registry access
