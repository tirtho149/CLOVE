#!/usr/bin/env bash
# Start one vLLM server per Titan RTX for configs/qwen3b_titan.yaml and wait until both
# are healthy. Stop them with:  kill $(cat artifacts/vllm/*.pid)
#
# Turing (sm_75) notes: no bf16 -> --dtype half; FlashAttention-2 is unavailable, vLLM
# picks a supported attention backend automatically. --limit-mm-per-prompt image=1 makes
# the server itself reject a second image (k_img = 0 enforced server-side too).
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
REVISION="${REVISION:-}"                     # pin an HF commit for the real run
GPU_UTIL="${GPU_UTIL:-0.85}"                 # leaves ~3.5 GB on GPU0 for SigLIP retrieval
MAXLEN="${MAXLEN:-8192}"                     # CLOVE prompts ~2k tokens, sage0 ~3-4k
MAX_PIXELS="${MAX_PIXELS:-802816}"           # must match model.max_pixels in the config
PORTS=(${PORTS:-8000 8001})
GPUS=(${GPUS:-0 1})
OUT="${OUT:-artifacts/vllm}"
mkdir -p "$OUT"

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
for i in "${!GPUS[@]}"; do
  g=${GPUS[$i]}; p=${PORTS[$i]}
  rev=(); [ -n "$REVISION" ] && rev=(--revision "$REVISION")
  CUDA_VISIBLE_DEVICES=$g nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" "${rev[@]}" --served-model-name "$MODEL" \
    --dtype half --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAXLEN" \
    --limit-mm-per-prompt '{"image": 1}' \
    --mm-processor-kwargs "{\"max_pixels\": $MAX_PIXELS}" \
    --enable-prefix-caching --max-logprobs 20 \
    --host 127.0.0.1 --port "$p" > "$OUT/vllm_gpu$g.log" 2>&1 &
  echo $! > "$OUT/vllm_gpu$g.pid"
  echo "GPU $g -> :$p (pid $(cat "$OUT/vllm_gpu$g.pid"), log $OUT/vllm_gpu$g.log)"
done
for p in "${PORTS[@]}"; do
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 && { echo ":$p healthy"; break; }
    sleep 5
  done
  curl -sf "http://127.0.0.1:$p/health" >/dev/null || { echo ":$p failed to start; see $OUT/"; exit 1; }
done
