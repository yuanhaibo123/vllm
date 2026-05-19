"""Triton Flash Attention for SM70+ with native GQA support.

Implements the flash-attention algorithm (Dao et al.) in Triton,
avoiding O(N²) memory by tiling Q×K→softmax→×V in SRAM.
Supports grouped-query attention (GQA) natively.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qn, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_on, stride_oh, stride_od,
    seq_len,
    scale: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Flash attention forward kernel with causal masking and GQA.

    Grid: (cdiv(seq_len, BLOCK_M), num_q_heads)
    """
    block_m_idx = tl.program_id(0)
    head_q_idx = tl.program_id(1)
    head_kv_idx = head_q_idx // GQA_RATIO

    # Offsets for this Q block
    offs_m = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Pointers to Q for this head
    q_ptrs = (Q
              + head_q_idx * stride_qh
              + offs_m[:, None] * stride_qn
              + offs_d[None, :] * stride_qd)
    # Load Q block (BLOCK_M, D) in fp16 for tensor core dot
    q_mask = offs_m[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    # Pre-scale Q to avoid per-block multiply
    q = (q * scale).to(q.dtype)

    # Initialize accumulators in fp32
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Causal: only attend to positions <= max position in this Q block
    causal_bound = (block_m_idx + 1) * BLOCK_M

    # Iterate over K/V blocks
    for block_n_start in range(0, causal_bound, BLOCK_N):
        offs_n = block_n_start + tl.arange(0, BLOCK_N)

        # Load K block (BLOCK_N, D) — keep in fp16 for tensor core
        k_ptrs = (K
                  + head_kv_idx * stride_kh
                  + offs_n[:, None] * stride_kn
                  + offs_d[None, :] * stride_kd)
        k_mask = offs_n[:, None] < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # S = Q @ K^T  → (BLOCK_M, BLOCK_N) — uses tensor cores on SM70
        s = tl.dot(q, tl.trans(k)).to(tl.float32)

        # Causal mask: mask out positions where key > query
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        s = tl.where(causal_mask, s, float("-inf"))

        # Also mask out-of-bounds keys
        s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        # Update running sum
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Update accumulator: rescale old acc, add new contribution
        acc = acc * alpha[:, None]

        # Load V block (BLOCK_N, D)
        v_ptrs = (V
                  + head_kv_idx * stride_vh
                  + offs_n[:, None] * stride_vn
                  + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # acc += P @ V — cast P to fp16 for tensor core dot
        acc += tl.dot(p.to(v.dtype), v).to(tl.float32)

        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Write output
    out_ptrs = (Out
                + head_q_idx * stride_oh
                + offs_m[:, None] * stride_on
                + offs_d[None, :] * stride_od)
    out_mask = offs_m[:, None] < seq_len
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=out_mask)


def triton_flash_attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    seq_len: int,
) -> torch.Tensor:
    """Flash attention prefill for a single request.

    Args:
        q: (seq_len, Hq, D) - query
        k: (seq_len, Hk, D) - key
        v: (seq_len, Hk, D) - value
        scale: attention scale factor
        seq_len: actual sequence length (for masking)

    Returns:
        output: (seq_len, Hq, D)
    """
    N, Hq, D = q.shape
    Hk = k.shape[1]
    gqa_ratio = Hq // Hk

    out = torch.empty_like(q)

    # Block sizes tuned for SM70 with D=256
    # SM70: 96KB shared mem, tensor cores for fp16
    # With num_stages=1: Q(M×D×2) + K(N×D×2) + V(N×D×2) must fit
    if D <= 128:
        BLOCK_M = 128
        BLOCK_N = 64
    else:
        # D=256: Q(64×256×2)=32KB + K(32×256×2)=16KB + V(32×256×2)=16KB = 64KB
        BLOCK_M = 64
        BLOCK_N = 32

    grid = (triton.cdiv(N, BLOCK_M), Hq)

    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        seq_len,
        scale=scale,
        GQA_RATIO=gqa_ratio,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=D,
        num_stages=1,
        num_warps=4,
    )
    return out
