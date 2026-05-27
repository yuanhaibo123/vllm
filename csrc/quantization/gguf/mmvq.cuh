// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;
    const auto vec = blockIdx.y;

    if (row >= nrows || vec >= nvecs) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;


    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = vec*(nrows_y/QK8_1) + i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs  = vdr * (threadIdx.x % (qi/vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    }

    if (threadIdx.x == 0) {
        dst[vec*nrows + row] = tmp;
    }
}

template<typename scalar_t>
static void mul_mat_vec_q4_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q4_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q5_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q5_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q8_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q2_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q3_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q4_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q5_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq3_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq1_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq1_m_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq4_nl_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

// SM70-optimised IQ4_XS MMVQ: activation Q8_1 blocks are loaded once per
// warp-step into shared memory and reused across all IQ4_XS_MMVY output rows,
// exploiting SM70's 96 KB per-SM shared memory.
//
// Layout:  block = (WARP_SIZE=32, IQ4_XS_MMVY=8) = 256 threads
//          threadIdx.x  — selects which of the 4 weight blocks in this step
//                         AND the sub-block index within that block (iqs 0..7)
//          threadIdx.y  — selects which output row in [row_base, row_base+MMVY)
//
// Activation reuse: threadIdx.y==0 loads 32 Q8_1 blocks (1152 B) into smem;
//                   all MMVY rows then compute from smem → 8× BW saving.
#define IQ4_XS_MMVY 8

template <typename scalar_t>
static __global__ void __launch_bounds__(WARP_SIZE_GGUF * IQ4_XS_MMVY)
mul_mat_vec_iq4_xs_smem(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    scalar_t   * __restrict__ dst,
    const int ncols, const int nrows, const int nvecs)
{
    // compile-time derived constants (all macros → constexpr-safe)
    constexpr int qi    = QI4_XS;               // 8 : int32s per weight block
    constexpr int bpw   = WARP_SIZE_GGUF / qi;  // 4 : weight blocks per warp step
    constexpr int yperw = QK_K / QK8_1;         // 8 : Q8_1 blocks per weight block

    const int row = blockIdx.x * IQ4_XS_MMVY + threadIdx.y;
    const int vec = blockIdx.y;
    if (vec >= nvecs) return;

    const int blocks_per_row = ncols / QK_K;
    const int nrows_y  = (ncols + 512 - 1) / 512 * 512;
    const int y_stride = nrows_y / QK8_1;

    // Shared activation tile: bpw * yperw = 32 Q8_1 blocks = 1152 bytes.
    // All IQ4_XS_MMVY rows in the block reuse this tile each step.
    __shared__ block_q8_1 tile_y[bpw * yperw];

    const block_iq4_xs * x = (const block_iq4_xs *)vx;
    const block_q8_1   * y = (const block_q8_1   *)vy;

    float tmp = 0.0f;

    for (int i_outer = 0; i_outer < blocks_per_row; i_outer += bpw) {
        // ── Load: first warp-row (threadIdx.y==0) fills 32 tile_y slots.
        //    threadIdx.x=0..31 → one Q8_1 block each; skip slots whose
        //    corresponding weight block is out-of-range.
        if (threadIdx.y == 0) {
            const int wblk = i_outer + threadIdx.x / yperw;
            if (wblk < blocks_per_row) {
                tile_y[threadIdx.x] = y[vec * y_stride + i_outer * yperw + threadIdx.x];
            }
        }
        __syncthreads();

        // ── Compute: each (row, threadIdx.x) pair owns one weight block ──
        // threadIdx.x/qi  → which of the 4 weight blocks in this step (0..3)
        // threadIdx.x%qi  → iqs sub-block index (0..7) within that block
        const int i         = i_outer + threadIdx.x / qi;
        const int iby_local = (threadIdx.x / qi) * yperw;  // tile_y offset
        const int iqs       = threadIdx.x % qi;

        if (i < blocks_per_row && row < nrows) {
            tmp += vec_dot_iq4_xs_q8_1(
                &x[row * blocks_per_row + i], &tile_y[iby_local], iqs);
        }
        __syncthreads();
    }

    // Warp-level reduction across threadIdx.x (within each row)
#pragma unroll
    for (int mask = WARP_SIZE_GGUF / 2; mask > 0; mask >>= 1) {
        tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    }

    if (threadIdx.x == 0 && row < nrows) {
        dst[vec * nrows + row] = tmp;
    }
}

template<typename scalar_t>
static void mul_mat_vec_iq4_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + IQ4_XS_MMVY - 1) / IQ4_XS_MMVY;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, IQ4_XS_MMVY, 1);
    mul_mat_vec_iq4_xs_smem<scalar_t>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}

template<typename scalar_t>
static void mul_mat_vec_iq3_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(block_num_y, nvecs, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
}
