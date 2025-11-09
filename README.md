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

## Build

```bash
docker build -t airiareleaseregistry.azurecr.io/baremetal/docker/airia-model-hosting-qwen2-vl-72b-llamacpp:0.1.0 .
docker push airiareleaseregistry.azurecr.io/baremetal/docker/airia-model-hosting-qwen2-vl-72b-llamacpp:0.1.0
```

## Deploy

```bash
kubectl apply -f deployment.yaml
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
