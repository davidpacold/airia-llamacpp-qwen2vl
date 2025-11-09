FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    curl \
    wget \
    libcurl4-openssl-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp with CUDA support
WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    git checkout master && \
    cmake -B build -DLLAMA_CUBLAS=ON -DLLAMA_CURL=ON && \
    cmake --build build --config Release -j$(nproc)

WORKDIR /app/llama.cpp

# Create models directory
RUN mkdir -p /app/models

# Download Qwen2-VL 72B GGUF model
# Using Q4_K_M quantization for optimal quality/performance on 24GB VRAM
RUN wget -O /app/models/qwen2-vl-72b-instruct-q4_k_m.gguf \
    https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GGUF/resolve/main/qwen2-vl-72b-instruct-q4_k_m.gguf || \
    echo "Model will be downloaded on first run"

# Download mmproj file for vision capabilities
RUN wget -O /app/models/qwen2-vl-mmproj-model.gguf \
    https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GGUF/resolve/main/mmproj-model-f16.gguf || \
    echo "MMProj will be downloaded on first run"

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Download model if not exists\n\
if [ ! -f /app/models/qwen2-vl-72b-instruct-q4_k_m.gguf ]; then\n\
  echo "Downloading Qwen2-VL 72B model..."\n\
  wget -O /app/models/qwen2-vl-72b-instruct-q4_k_m.gguf \\\n\
    https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GGUF/resolve/main/qwen2-vl-72b-instruct-q4_k_m.gguf\n\
fi\n\
\n\
if [ ! -f /app/models/qwen2-vl-mmproj-model.gguf ]; then\n\
  echo "Downloading MMProj model..."\n\
  wget -O /app/models/qwen2-vl-mmproj-model.gguf \\\n\
    https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GGUF/resolve/main/mmproj-model-f16.gguf\n\
fi\n\
\n\
echo "Starting llama.cpp server with Qwen2-VL 72B..."\n\
exec /app/llama.cpp/build/bin/server \\\n\
  -m /app/models/qwen2-vl-72b-instruct-q4_k_m.gguf \\\n\
  --mmproj /app/models/qwen2-vl-mmproj-model.gguf \\\n\
  -ngl 99 \\\n\
  --host 0.0.0.0 \\\n\
  --port 8008 \\\n\
  -c ${LLAMA_CONTEXT_SIZE:-8192} \\\n\
  -b ${LLAMA_BATCH_SIZE:-2048} \\\n\
  --threads ${LLAMA_THREADS:-24} \\\n\
  --n-gpu-layers ${LLAMA_GPU_LAYERS:-99} \\\n\
  --parallel ${LLAMA_PARALLEL:-8} \\\n\
  --cont-batching \\\n\
  --flash-attn \\\n\
  --mlock \\\n\
  --cache-type-k f16 \\\n\
  --cache-type-v f16 \\\n\
  --metrics \\\n\
  --log-format json\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set default environment variables
ENV LLAMA_CONTEXT_SIZE=8192
ENV LLAMA_BATCH_SIZE=2048
ENV LLAMA_THREADS=24
ENV LLAMA_GPU_LAYERS=99
ENV LLAMA_PARALLEL=8
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8008

ENTRYPOINT ["/app/entrypoint.sh"]
