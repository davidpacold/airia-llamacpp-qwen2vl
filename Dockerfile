# Multi-stage build for smaller image size
# Stage 1: Build llama.cpp
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV LLAMA_CUDA=1

# Install only build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    libcurl4-openssl-dev \
    pkg-config \
    ca-certificates \
    cuda-nvcc-12-2 \
    && rm -rf /var/lib/apt/lists/*

# Verify CUDA is available
RUN nvcc --version

# Clone llama.cpp
WORKDIR /build
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git

WORKDIR /build/llama.cpp

# Create symlink for libcuda.so stub (linker needs this)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Build with CMake and CUDA support (static libraries to avoid runtime dependencies)
# Set library path to include stubs during linking
RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" \
    && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    cmake --build build --config Release --target llama-server -j$(nproc)

# Strip binary to reduce size
RUN strip build/bin/llama-server || true

# Stage 2: Runtime image
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the statically-linked binary from builder
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# Copy only CUDA runtime libraries (required even with static linking)
# Note: libcuda.so comes from NVIDIA driver at runtime, not copied here
RUN mkdir -p /usr/local/cuda/lib64
COPY --from=builder /usr/local/cuda-12.2/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda-12.2/lib64/libcublas.so* /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda-12.2/lib64/libcublasLt.so* /usr/local/cuda/lib64/

# Update library cache and add to LD_LIBRARY_PATH
RUN ldconfig /usr/local/cuda/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Create models directory with correct permissions for non-root user (1654)
RUN mkdir -p /app/models && chown -R 1654:1654 /app

WORKDIR /app

# Create optimized entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
MODEL_FILE="/app/models/Qwen2-VL-7B-Instruct-Q4_K_M.gguf"\n\
MMPROJ_FILE="/app/models/mmproj-Qwen2-VL-7B-Instruct-f16.gguf"\n\
\n\
# Download model if not exists (with progress and resume support)\n\
if [ ! -f "$MODEL_FILE" ]; then\n\
  echo "Downloading Qwen2-VL 7B model (~4.7GB)..."\n\
  wget -c --progress=dot:giga -O "$MODEL_FILE" \\\n\
    https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/Qwen2-VL-7B-Instruct-Q4_K_M.gguf\n\
fi\n\
\n\
if [ ! -f "$MMPROJ_FILE" ]; then\n\
  echo "Downloading MMProj model (~675MB)..."\n\
  wget -c --progress=dot:giga -O "$MMPROJ_FILE" \\\n\
    https://huggingface.co/bartowski/Qwen2-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-7B-Instruct-f16.gguf\n\
fi\n\
\n\
echo "Starting llama.cpp server with Qwen2-VL 7B..."\n\
exec llama-server \\\n\
  -m "$MODEL_FILE" \\\n\
  --mmproj "$MMPROJ_FILE" \\\n\
  --host 0.0.0.0 \\\n\
  --port 8008 \\\n\
  -c ${LLAMA_CONTEXT_SIZE:-8192} \\\n\
  -b ${LLAMA_BATCH_SIZE:-2048} \\\n\
  --threads ${LLAMA_THREADS:-24} \\\n\
  --n-gpu-layers ${LLAMA_GPU_LAYERS:-99} \\\n\
  --parallel ${LLAMA_PARALLEL:-8} \\\n\
  --cont-batching \\\n\
  --flash-attn on \\\n\
  --mlock \\\n\
  --cache-type-k f16 \\\n\
  --cache-type-v f16 \\\n\
  --metrics\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set default environment variables
ENV LLAMA_CONTEXT_SIZE=8192 \
    LLAMA_BATCH_SIZE=2048 \
    LLAMA_THREADS=24 \
    LLAMA_GPU_LAYERS=99 \
    LLAMA_PARALLEL=8 \
    CUDA_VISIBLE_DEVICES=0

EXPOSE 8008

ENTRYPOINT ["/app/entrypoint.sh"]
