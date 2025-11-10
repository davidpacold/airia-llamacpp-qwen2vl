# Configuration Guide

Comprehensive configuration documentation for Qwen2-VL deployment with llama.cpp.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Context Window Configuration](#context-window-configuration)
- [Performance Tuning](#performance-tuning)
- [Memory Management](#memory-management)
- [CUDA Optimization](#cuda-optimization)
- [Kubernetes Configuration](#kubernetes-configuration)

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────┐
│                  Kubernetes Pod                      │
├─────────────────────────────────────────────────────┤
│  Init Container: model-downloader                   │
│  - Downloads Qwen2-VL 7B Q4_K_M from Hugging Face  │
│  - Downloads mmproj projector weights               │
│  - Stores in persistent volume (60Gi PVC)           │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Main Container: qwen2-vl-llamacpp                  │
├─────────────────────────────────────────────────────┤
│  llama-server (llama.cpp)                           │
│  - Port 8008: OpenAI-compatible API                 │
│  - CUDA 12.2 with Flash Attention                   │
│  - 8 parallel request slots                         │
│  - Continuous batching enabled                      │
│  - Metrics endpoint: /metrics                       │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│            NVIDIA A10-24Q GPU                        │
├─────────────────────────────────────────────────────┤
│  - 24GB VRAM (8.4GB used, 16GB free)                │
│  - All 28 model layers offloaded                    │
│  - Flash Attention kernels                          │
│  - KV cache in f16 precision                        │
└─────────────────────────────────────────────────────┘
```

### Request Flow

1. **Client Request** → API Gateway (Bearer token auth)
2. **API Gateway** → llama-server:8008 `/v1/chat/completions`
3. **llama-server** → Assigns to available slot (1 of 8)
4. **Image Processing** → mmproj encodes image (1024 tokens)
5. **Text Processing** → Model processes prompt + image tokens
6. **Generation** → Streaming or complete response
7. **Response** → Back to client with usage statistics

## Context Window Configuration

### Understanding Context Allocation

The total context window is divided among parallel slots:

```
LLAMA_CONTEXT_SIZE (total tokens)
    ÷
LLAMA_PARALLEL (number of slots)
    =
tokens per slot (n_ctx_seq)
```

### Example Configurations

#### Configuration 1: Minimal (Not Recommended)
```yaml
LLAMA_CONTEXT_SIZE: "8192"
LLAMA_PARALLEL: "8"
# Result: 1024 tokens per slot
# Problem: No room for text after 1024 image tokens
# Status: ❌ Will fail with 400 Bad Request
```

#### Configuration 2: Basic
```yaml
LLAMA_CONTEXT_SIZE: "16384"
LLAMA_PARALLEL: "8"
# Result: 2048 tokens per slot
# Available for text: 1024 tokens (after image)
# Status: ✅ Works but limited
# Use case: Simple image Q&A only
```

#### Configuration 3: Recommended (Current)
```yaml
LLAMA_CONTEXT_SIZE: "32768"
LLAMA_PARALLEL: "8"
# Result: 4096 tokens per slot
# Available for text: 3072 tokens (after image)
# Status: ✅ Optimal for complex tasks
# Use case: Document analysis, multi-turn conversations
```

#### Configuration 4: Low Latency
```yaml
LLAMA_CONTEXT_SIZE: "16384"
LLAMA_PARALLEL: "4"
# Result: 4096 tokens per slot
# Available for text: 3072 tokens (after image)
# Status: ✅ Faster single requests
# Use case: Real-time applications, fewer concurrent users
```

#### Configuration 5: High Throughput
```yaml
LLAMA_CONTEXT_SIZE: "32768"
LLAMA_PARALLEL: "16"
# Result: 2048 tokens per slot
# Available for text: 1024 tokens (after image)
# Status: ✅ More concurrent requests
# Use case: High-traffic APIs with simple queries
# Note: Requires more VRAM
```

### Image Token Requirements

Qwen2-VL processes images into fixed-size token sequences:

| Image Resolution | Approximate Tokens | Context Required |
|------------------|-------------------|------------------|
| 224x224 (min)    | 256-512          | 1024 minimum     |
| 512x512          | 512-1024         | 1024 minimum     |
| 1024x1024        | 1024-2048        | 2048+ recommended|
| Multiple images  | 1024 per image   | Scale accordingly|

**Important**: Always allocate at least 1024 tokens per slot for image processing, even for small images.

## Performance Tuning

### Batch Size Configuration

`LLAMA_BATCH_SIZE` controls how many tokens are processed in parallel during prompt evaluation:

```yaml
# Conservative (lower latency, lower throughput)
LLAMA_BATCH_SIZE: "512"

# Balanced (current configuration)
LLAMA_BATCH_SIZE: "2048"

# Aggressive (higher throughput, higher VRAM)
LLAMA_BATCH_SIZE: "4096"
```

**Observed Performance**:
- Small prompts (< 500 tokens): 1,000+ tokens/sec with batch size 2048
- Large prompts (> 2000 tokens): 250-500 tokens/sec
- Image encoding: 1,000-1,500 tokens/sec

### Thread Configuration

`LLAMA_THREADS` sets CPU threads for computation:

```yaml
# Minimum (GPU-focused)
LLAMA_THREADS: "8"

# Balanced (current)
LLAMA_THREADS: "24"

# Maximum (CPU-intensive tasks)
LLAMA_THREADS: "48"
```

**Rule of thumb**: Use 2-4x the number of CPU cores for hybrid CPU/GPU workloads.

### GPU Layer Offloading

`LLAMA_GPU_LAYERS` controls how many model layers run on GPU:

```yaml
# Partial offload (if VRAM limited)
LLAMA_GPU_LAYERS: "20"  # 20 of 28 layers on GPU

# Full offload (current, recommended)
LLAMA_GPU_LAYERS: "99"  # All layers on GPU

# CPU only (not recommended for this model)
LLAMA_GPU_LAYERS: "0"
```

**VRAM Usage by Layer Count** (Qwen2-VL 7B Q4_K_M):
- 0 layers (CPU): ~0.5GB VRAM (model weights in RAM)
- 14 layers: ~4GB VRAM
- 28 layers (all): ~8.4GB VRAM

## Memory Management

### KV Cache Configuration

The KV (Key-Value) cache stores attention states for generated tokens:

```dockerfile
# Current configuration (highest quality)
--cache-type-k f16    # 16-bit float for keys
--cache-type-v f16    # 16-bit float for values

# Memory-optimized (slightly lower quality)
--cache-type-k q8_0   # 8-bit quantized keys
--cache-type-v q8_0   # 8-bit quantized values

# Maximum memory savings (noticeable quality loss)
--cache-type-k q4_0   # 4-bit quantized keys
--cache-type-v q4_0   # 4-bit quantized values
```

**VRAM Usage Impact**:
- f16 KV cache: ~2-4GB for 32K context across 8 slots
- q8_0 KV cache: ~1-2GB for 32K context across 8 slots
- q4_0 KV cache: ~0.5-1GB for 32K context across 8 slots

### Memory Locking

```dockerfile
--mlock    # Lock model weights in RAM (prevents swapping)
```

**When to use**:
- ✅ Production deployments (current configuration)
- ✅ When RAM is plentiful (> 32GB)
- ❌ Development environments with limited RAM
- ❌ Shared hosting environments

## CUDA Optimization

### Flash Attention

```dockerfile
--flash-attn on    # Enable Flash Attention v2
```

**Benefits**:
- 2-4x faster attention computation
- 50% lower VRAM usage for attention
- Scales better with longer contexts

**Requirements**:
- CUDA compute capability >= 7.5 (A10 is 8.6 ✅)
- CUDA 11.8+ (using 12.2 ✅)

### Continuous Batching

```dockerfile
--cont-batching    # Enable continuous batching
```

**How it works**:
- Processes multiple requests simultaneously
- Dynamically adds/removes requests from batch
- Better GPU utilization under varying load

**Performance Impact**:
- Without: 1 request at a time, GPU idle between requests
- With: 4-8 concurrent requests, 80-95% GPU utilization

### CUDA Environment Variables

```yaml
# Current configuration
CUDA_VISIBLE_DEVICES: "0"         # Use first GPU only
CUDA_LAUNCH_BLOCKING: "0"         # Async kernel launches (faster)

# Debug configuration
CUDA_LAUNCH_BLOCKING: "1"         # Sync kernel launches (easier debugging)
CUDA_DEVICE_ORDER: "PCI_BUS_ID"   # Consistent GPU ordering
```

## Kubernetes Configuration

### Node Selector and Tolerations

```yaml
nodeSelector:
  agentpool: a10fullnodes    # Target nodes with A10 GPUs

tolerations:
- key: nvidia.com/gpu
  operator: Equal
  value: "true"
  effect: NoSchedule
```

**Purpose**: Ensures pod schedules only on GPU-enabled nodes.

### Resource Requests vs Limits

```yaml
resources:
  requests:
    cpu: "8000m"           # Minimum guaranteed CPU
    memory: "32Gi"         # Minimum guaranteed memory
    nvidia.com/gpu: "1"    # Request 1 GPU
  limits:
    cpu: "32000m"          # Maximum CPU (burst)
    memory: "128Gi"        # Maximum memory (OOM if exceeded)
    nvidia.com/gpu: "1"    # Maximum GPUs
```

**Best Practices**:
- Set requests = typical usage for scheduling
- Set limits = maximum burst capacity
- GPU requests = GPU limits (no fractional GPUs)
- Memory limit > model size + KV cache + overhead

### Security Context

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1654
  allowPrivilegeEscalation: false
  capabilities:
    drop: [ALL]
  seccompProfile:
    type: RuntimeDefault
```

**Rationale**:
- Non-root user for security
- No privilege escalation prevents container breakout
- Drop all capabilities (model serving doesn't need special caps)
- Seccomp profile restricts syscalls

### Persistent Volume

```yaml
volumeMounts:
- name: model-storage
  mountPath: /app/models

volumes:
- name: model-storage
  persistentVolumeClaim:
    claimName: qwen2-vl-72b-llamacpp-pvc
```

**Benefits**:
- Model downloads persist across pod restarts
- Faster pod startup (no re-download)
- Shared storage for multiple replicas (ReadOnlyMany)

**Storage Requirements**:
- Qwen2-VL 7B Q4_K_M: ~4.5GB
- mmproj file: ~1.5GB
- Total with overhead: 10-20GB (60GB PVC allows for future models)

## Advanced Configurations

### Multi-GPU Deployment

To use 2 GPUs:

```yaml
# deployment.yaml
resources:
  requests:
    nvidia.com/gpu: "2"
  limits:
    nvidia.com/gpu: "2"

env:
- name: CUDA_VISIBLE_DEVICES
  value: "0,1"
- name: LLAMA_PARALLEL
  value: "16"  # Scale slots with GPU count
```

### Model Selection

To use different model quantizations:

```dockerfile
# In Dockerfile, change MODEL_URL:
ENV MODEL_URL="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GGUF/resolve/main/qwen2-vl-7b-instruct-q5_k_m.gguf"
```

**Quantization Comparison**:
| Quantization | Size   | VRAM    | Quality | Speed    |
|--------------|--------|---------|---------|----------|
| Q3_K_M       | 3.5GB  | 6-7GB   | Lower   | Fastest  |
| Q4_K_M       | 4.5GB  | 8-9GB   | Good    | Fast     |
| Q5_K_M       | 5.5GB  | 10-11GB | Better  | Moderate |
| Q6_K         | 6.5GB  | 12-13GB | Best    | Slower   |
| Q8_0         | 8.5GB  | 14-15GB | Highest | Slowest  |

### API Authentication

Current configuration uses bearer token auth:

```bash
curl -H "Authorization: Bearer changeme" \
  http://localhost:8008/v1/chat/completions
```

To change the API key:

```dockerfile
# In Dockerfile
--api-key YOUR_SECURE_KEY
```

Or via Kubernetes secret:

```yaml
env:
- name: API_KEY
  valueFrom:
    secretKeyRef:
      name: llama-api-key
      key: key
```

Then update Dockerfile to use it:
```dockerfile
--api-key ${API_KEY}
```

## Configuration Validation

### Startup Checks

When the pod starts, verify configuration in logs:

```bash
kubectl logs -n airia <pod-name> | grep -A 10 "system_info"
```

Expected output:
```
system_info: n_threads = 24
system_info: n_ctx = 32768
system_info: n_batch = 2048
system_info: n_parallel = 8
system_info: n_gpu_layers = 99
```

### Slot Initialization

Check per-slot context allocation:

```bash
kubectl logs -n airia <pod-name> | grep "slot init"
```

Expected output:
```
slot init: id 0 | task -1 | new slot, n_ctx = 4096
slot init: id 1 | task -1 | new slot, n_ctx = 4096
...
slot init: id 7 | task -1 | new slot, n_ctx = 4096
```

If `n_ctx < 2048`, increase `LLAMA_CONTEXT_SIZE`.

## Troubleshooting Configuration Issues

### Issue: "Failed to allocate memory for KV cache"

**Cause**: Not enough VRAM for current configuration

**Solutions**:
1. Reduce `LLAMA_CONTEXT_SIZE`
2. Reduce `LLAMA_PARALLEL`
3. Use quantized KV cache (q8_0 instead of f16)
4. Offload fewer layers with `LLAMA_GPU_LAYERS`

### Issue: "Slot context too small for image"

**Cause**: `tokens_per_slot < 1024`

**Solution**: Ensure `LLAMA_CONTEXT_SIZE / LLAMA_PARALLEL >= 2048`

### Issue: Slow inference

**Causes and solutions**:
1. **CPU bottleneck**: Increase `LLAMA_THREADS`
2. **Small batch size**: Increase `LLAMA_BATCH_SIZE`
3. **Too many parallel slots**: Reduce `LLAMA_PARALLEL` for lower latency
4. **Flash attention disabled**: Verify `--flash-attn on` in logs

### Issue: Out of memory (OOM) kills

**Causes and solutions**:
1. **System memory**: Increase `memory` limit in deployment.yaml
2. **VRAM**: Reduce context size or parallel slots
3. **Memory leak**: Restart pod periodically (consider readiness probe)

## References

- [llama.cpp documentation](https://github.com/ggerganov/llama.cpp)
- [Qwen2-VL model card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [CUDA optimization guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Flash Attention paper](https://arxiv.org/abs/2205.14135)
