#!/bin/bash
# Launch Qwen3.6-27B-GPTQ-4bit — PP=4 across all 4x Titan V (SM70)
# --hf-overrides forces text-only (Qwen3_5ForCausalLM) without touching model files,
#   skipping the vision encoder and saving ~2.4 GiB on PP0.
# Layer split: 12,20,20,12 (PP0/PP3 lighter — they carry embed/lm_head overhead)
# KV cache: turboquant_k8v4 (FP8 keys + 4-bit values, 2.6x compression)
#   Uses Triton float8e4b15 software emulation on SM70 — no FP8 hardware required.

MODEL=/home/ice/llm/models/Qwen3.6-27B-GPTQ-4bit
PORT=${1:-8021}
LOG=/home/ice/llm/tmp/qwen36_27b_pp4.log

echo "Starting Qwen3.6-27B-GPTQ-4bit (PP=4) on port $PORT ..."
echo "Log: $LOG"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \   # prevent fragmentation OOM on large allocs
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \            # 30 min timeout for slow PP steps on SM70
VLLM_PP_LAYER_PARTITION=13,19,20,12 \                # uneven split: PP0/PP3 lighter (carry embed+lm_head)
CUDA_VISIBLE_DEVICES=0,1,2,3 \                       # all 4x Titan V
  /home/ice/llm/vllm-build-env/bin/python3 -m vllm.entrypoints.openai.api_server \
  --port "$PORT" \                          # default 8021, overridable via $1
  --model "$MODEL" \                        # Qwen3.6-27B GPTQ-4bit local path
  --dtype half \                            # FP16 for non-GPTQ parts (embeddings, norms, lm_head, activations); SSM states stay FP32 per model config; SM70 has no BF16
  --quantization gptq \                    # 4-bit GPTQ weight quantization
  --max-model-len 65536 \                  # 64k context window
  --pipeline-parallel-size 4 \            # PP across 4 GPUs (no NVLink needed)
  --gpu-memory-utilization 0.90 \         # leave 10% headroom per GPU
  --kv-cache-dtype turboquant_k8v4 \      # FP8 keys + 4-bit values; 2.6x KV compression via Triton
  --max-num-seqs 2 \                       # limit concurrent sequences to fit KV cache
  --enable-prefix-caching \               # reuse KV cache for shared prefixes (system prompt etc.)
  --enable-chunked-prefill \              # interleave prefill chunks with decode to reduce stalls
  --max-num-batched-tokens 4096 \         # chunk budget per step; keeps PP bubbles small
  --disable-log-stats \                   # suppress per-request throughput logging
  --no-async-scheduling \                 # synchronous scheduler; async has PP deadlock risk
  --enable-auto-tool-choice \            # allow model to autonomously invoke tools
  --tool-call-parser qwen3_xml \          # parse <tool_call>/<function=...>/<parameter=...> XML format
  --hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}' \  # force text-only arch, skip vision encoder
  2>&1 | tee "$LOG"
