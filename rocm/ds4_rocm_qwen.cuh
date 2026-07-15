// Qwen3.6-35B-A3B decode/prefill primitives for DS4's ROCm backend.
//
// Direct HIP translation of metal/qwen35.metal.  Buffers contain F32 values and
// every stride is expressed in bytes, so the host binds packed rows or larger
// workspaces exactly as the Metal host does.  The numeric contract is identical
// to ds4_qwen_ref.c / ds4_qwen.c: arithmetic order is preserved so the ROCm
// kernels reproduce the scalar reference (and the Metal kernels) bit-comparably
// within the tolerances used by tests/qwen.
//
// simdgroup mapping: RDNA3 runs wave32 (via HSA_OVERRIDE_GFX_VERSION=11.0.0),
// so one Metal simdgroup == one 32-lane wavefront.  lane == threadIdx.x & 31,
// wave index == threadIdx.x >> 5, threads_per_simdgroup == 32.  Metal
// simd_sum/simd_max/simd_min become butterfly (__shfl_xor) reductions that
// broadcast the result to every lane, matching Metal's SIMD-wide semantics.

#ifndef DS4_ROCM_QWEN_CUH
#define DS4_ROCM_QWEN_CUH

// ---- args structs (byte-stride layout mirrors metal/qwen35.metal) ----------

struct qrocm_split_q_gate {
    uint32_t n_token, n_query_head, head_dim, reserved;
    uint64_t projection_token_stride, projection_head_stride, projection_dim_stride;
    uint64_t query_token_stride, query_head_stride, query_dim_stride;
    uint64_t gate_token_stride, gate_head_stride, gate_dim_stride;
};

struct qrocm_sigmoid_mul {
    uint64_t n_value, input_stride, gate_stride, output_stride;
};

struct qrocm_sigmoid_mul_rows {
    uint32_t n_row, row_width;
    uint64_t input_row_stride, input_dim_stride, gate_row_stride;
    uint64_t output_row_stride, output_dim_stride;
};

struct qrocm_rope {
    uint32_t n_token, n_head, head_dim, n_rot;
    float theta; uint32_t reserved;
    uint64_t source_token_stride, source_head_stride, source_dim_stride;
    uint64_t output_token_stride, output_head_stride, output_dim_stride;
    uint64_t position_stride;
};

struct qrocm_conv_step {
    uint32_t n_channel, kernel_size;
    uint64_t input_channel_stride, weight_channel_stride, weight_tap_stride;
    uint64_t state_channel_stride, state_tap_stride, output_channel_stride;
};

struct qrocm_conv_sequence {
    uint32_t n_token, n_channel, kernel_size, reserved;
    uint64_t input_token_stride, input_channel_stride;
    uint64_t weight_channel_stride, weight_tap_stride;
    uint64_t state_channel_stride, state_tap_stride;
    uint64_t output_token_stride, output_channel_stride;
};

struct qrocm_gated_delta_step {
    uint32_t n_key_head, n_value_head, key_dim, value_dim;
    uint64_t query_head_stride, query_dim_stride;
    uint64_t key_head_stride, key_dim_stride;
    uint64_t value_head_stride, value_dim_stride;
    uint64_t log_decay_head_stride, beta_head_stride;
    uint64_t state_head_stride, state_value_stride, state_key_stride;
    uint64_t output_head_stride, output_dim_stride;
};

struct qrocm_gated_delta_sequence {
    uint32_t n_token, n_key_head, n_value_head, key_dim, value_dim, reserved;
    uint64_t projection_token_stride, query_offset, key_offset, value_offset;
    uint64_t query_head_stride, query_dim_stride;
    uint64_t key_head_stride, key_dim_stride;
    uint64_t value_head_stride, value_dim_stride;
    uint64_t log_decay_token_stride, log_decay_head_stride;
    uint64_t beta_token_stride, beta_head_stride;
    uint64_t state_head_stride, state_value_stride, state_key_stride;
    uint64_t output_token_stride, output_head_stride, output_dim_stride;
};

struct qrocm_rmsnorm_gated {
    uint32_t n_vector, dim; float epsilon; uint32_t reserved;
    uint64_t input_vector_stride, input_dim_stride;
    uint64_t gate_vector_stride, gate_dim_stride, weight_dim_stride;
    uint64_t output_vector_stride, output_dim_stride;
};

struct qrocm_embedding_q8_0 {
    uint32_t row_index, n_embd, block_size, reserved;
    uint64_t source_row_stride, source_block_stride;
    uint64_t source_scale_offset, source_quant_offset, source_quant_stride;
    uint64_t output_dim_stride;
};

struct qrocm_embedding_q8_0_batch {
    uint32_t n_token, n_row, n_embd, block_size;
    uint64_t source_row_stride, source_block_stride;
    uint64_t source_scale_offset, source_quant_offset, source_quant_stride;
    uint64_t token_id_stride, output_token_stride, output_dim_stride;
};

struct qrocm_gated_delta_controls {
    uint32_t n_token, n_value_head;
    uint64_t alpha_logit_token_stride, alpha_logit_head_stride;
    uint64_t beta_logit_token_stride, beta_logit_head_stride;
    uint64_t ssm_a_head_stride, dt_bias_head_stride;
    uint64_t log_decay_token_stride, log_decay_head_stride;
    uint64_t beta_token_stride, beta_head_stride;
};

struct qrocm_gqa_decode {
    uint32_t n_kv, n_query_head, n_kv_head, head_dim;
    uint64_t query_head_stride, query_dim_stride;
    uint64_t key_token_stride, key_head_stride, key_dim_stride;
    uint64_t value_token_stride, value_head_stride, value_dim_stride;
    uint64_t output_head_stride, output_dim_stride;
};

struct qrocm_gqa_prefill {
    uint32_t position0, n_token, n_query_head, n_kv_head, head_dim, reserved;
    uint64_t query_token_stride, query_head_stride, query_dim_stride;
    uint64_t key_token_stride, key_head_stride, key_dim_stride;
    uint64_t value_token_stride, value_head_stride, value_dim_stride;
    uint64_t output_token_stride, output_head_stride, output_dim_stride;
};

struct qrocm_router_top8 {
    uint32_t n_token, reserved;
    uint64_t logits_token_stride, logits_stride;
    uint64_t selected_token_stride, selected_stride;
    uint64_t selected_weight_token_stride, selected_weight_stride;
};

// ---- device helpers --------------------------------------------------------

__device__ __forceinline__ float qrocm_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}
__device__ __forceinline__ float qrocm_silu(float x) { return x * qrocm_sigmoid(x); }
__device__ __forceinline__ float qrocm_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    if (x < -10.0f) {
        const float e = expf(x);
        return e - 0.5f * e * e + (e * e * e) / 3.0f;
    }
    if (x > 10.0f) return x + logf(1.0f + expf(-x));
    return logf(1.0f + expf(x));
}
__device__ __forceinline__ float qrocm_ld(const char *b, uint64_t o) {
    return *((const float *)(b + o));
}
__device__ __forceinline__ void qrocm_st(char *b, uint64_t o, float v) {
    *((float *)(b + o)) = v;
}
// All-lane (broadcast) 32-wide reductions == Metal simd_sum/max/min.
__device__ __forceinline__ float qrocm_wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
__device__ __forceinline__ float qrocm_wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    return v;
}
__device__ __forceinline__ uint32_t qrocm_wmin_u(uint32_t v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        uint32_t t = (uint32_t)__shfl_xor((int)v, o, 32);
        if (t < v) v = t;
    }
    return v;
}

// ---- router: stable softmax + deterministic top-8 (serial reference) -------

__global__ void qrocm_k_router_serial(
        qrocm_router_top8 args, const char *logits,
        char *selected, char *selected_weight) {
    constexpr uint32_t n_expert = 256u, n_selected = 8u;
    __shared__ float probability[256];
    const uint32_t tid = threadIdx.x, token = blockIdx.x;
    if (token >= args.n_token || blockDim.x < n_expert ||
        args.logits_stride < sizeof(float) ||
        args.selected_stride < sizeof(int32_t) ||
        args.selected_weight_stride < sizeof(float)) return;

    const uint64_t logits_base = (uint64_t)token * args.logits_token_stride;
    const uint64_t selected_base = (uint64_t)token * args.selected_token_stride;
    const uint64_t sw_base = (uint64_t)token * args.selected_weight_token_stride;

    __shared__ float ctl[2];
    if (tid == 0u) {
        float maximum = qrocm_ld(logits, logits_base);
        bool finite = isfinite(maximum);
        for (uint32_t e = 1u; e < n_expert; e++) {
            const float v = qrocm_ld(logits, logits_base + (uint64_t)e * args.logits_stride);
            finite = finite && isfinite(v);
            if (v > maximum) maximum = v;
        }
        ctl[0] = maximum;
        ctl[1] = finite ? 1.0f : 0.0f;
    }
    __syncthreads();

    const float maximum = ctl[0];
    const bool finite = ctl[1] != 0.0f;
    if (!finite) {
        if (tid < n_selected) {
            *((int32_t *)(selected + selected_base + (uint64_t)tid * args.selected_stride)) = -1;
            qrocm_st(selected_weight, sw_base + (uint64_t)tid * args.selected_weight_stride, 0.0f);
        }
        return;
    }
    if (tid < n_expert)
        probability[tid] = expf(qrocm_ld(logits, logits_base + (uint64_t)tid * args.logits_stride) - maximum);
    __syncthreads();
    if (tid != 0u) return;

    float total = 0.0f;
    for (uint32_t e = 0u; e < n_expert; e++) total += probability[e];
    for (uint32_t e = 0u; e < n_expert; e++) probability[e] /= total;

    int32_t chosen[n_selected];
    float chosen_weight[n_selected];
    for (uint32_t slot = 0u; slot < n_selected; slot++) {
        uint32_t best = n_expert;
        for (uint32_t e = 0u; e < n_expert; e++) {
            bool used = false;
            for (uint32_t p = 0u; p < slot; p++) if (chosen[p] == (int32_t)e) { used = true; break; }
            if (used) continue;
            if (best == n_expert || probability[e] > probability[best] ||
                (probability[e] == probability[best] && e < best)) best = e;
        }
        chosen[slot] = (int32_t)best;
        chosen_weight[slot] = probability[best];
    }
    float selected_total = 0.0f;
    for (uint32_t slot = 0u; slot < n_selected; slot++) selected_total += chosen_weight[slot];
    for (uint32_t slot = 0u; slot < n_selected; slot++) {
        *((int32_t *)(selected + selected_base + (uint64_t)slot * args.selected_stride)) = chosen[slot];
        qrocm_st(selected_weight, sw_base + (uint64_t)slot * args.selected_weight_stride,
                 chosen_weight[slot] / selected_total);
    }
}

// ---- embedding dequant (Q8_0) ---------------------------------------------

__global__ void qrocm_k_embed_q8_0(
        qrocm_embedding_q8_0 args, const char *embedding, char *output) {
    const uint32_t dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= args.n_embd || args.block_size != 32u) return;
    const uint32_t block = dim / args.block_size;
    const uint32_t within = dim - block * args.block_size;
    const uint64_t bo = (uint64_t)args.row_index * args.source_row_stride +
                        (uint64_t)block * args.source_block_stride;
    const __half scale = *((const __half *)(embedding + bo + args.source_scale_offset));
    const int8_t q = *((const int8_t *)(embedding + bo + args.source_quant_offset +
                       (uint64_t)within * args.source_quant_stride));
    qrocm_st(output, (uint64_t)dim * args.output_dim_stride, (float)scale * (float)q);
}

__global__ void qrocm_k_embed_q8_0_batch(
        qrocm_embedding_q8_0_batch args, const char *embedding,
        const char *token_ids, char *output) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)args.n_token * args.n_embd;
    if (gid >= total || args.n_embd == 0u || args.block_size != 32u) return;
    const uint64_t token = gid / args.n_embd;
    const uint32_t dim = (uint32_t)(gid - token * args.n_embd);
    const int32_t row = *((const int32_t *)(token_ids + token * args.token_id_stride));
    if (row < 0 || (uint32_t)row >= args.n_row) return;
    const uint32_t block = dim / args.block_size;
    const uint32_t within = dim - block * args.block_size;
    const uint64_t bo = (uint64_t)(uint32_t)row * args.source_row_stride +
                        (uint64_t)block * args.source_block_stride;
    const __half scale = *((const __half *)(embedding + bo + args.source_scale_offset));
    const int8_t q = *((const int8_t *)(embedding + bo + args.source_quant_offset +
                       (uint64_t)within * args.source_quant_stride));
    qrocm_st(output, token * args.output_token_stride + (uint64_t)dim * args.output_dim_stride,
             (float)scale * (float)q);
}

// ---- gated-delta controls --------------------------------------------------

__global__ void qrocm_k_gd_controls(
        qrocm_gated_delta_controls args, const char *alpha_logit,
        const char *beta_logit, const char *ssm_a, const char *dt_bias,
        char *log_decay, char *beta) {
    const uint64_t total = (uint64_t)args.n_token * args.n_value_head;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total || args.n_value_head == 0u) return;
    const uint32_t token = (uint32_t)(gid / args.n_value_head);
    const uint32_t head = (uint32_t)(gid - (uint64_t)token * args.n_value_head);
    const float alpha = qrocm_ld(alpha_logit,
        (uint64_t)token * args.alpha_logit_token_stride + (uint64_t)head * args.alpha_logit_head_stride);
    const float beta_v = qrocm_sigmoid(qrocm_ld(beta_logit,
        (uint64_t)token * args.beta_logit_token_stride + (uint64_t)head * args.beta_logit_head_stride));
    const float a = qrocm_ld(ssm_a, (uint64_t)head * args.ssm_a_head_stride);
    const float bias = qrocm_ld(dt_bias, (uint64_t)head * args.dt_bias_head_stride);
    qrocm_st(log_decay,
        (uint64_t)token * args.log_decay_token_stride + (uint64_t)head * args.log_decay_head_stride,
        a * qrocm_softplus(alpha + bias));
    qrocm_st(beta,
        (uint64_t)token * args.beta_token_stride + (uint64_t)head * args.beta_head_stride, beta_v);
}

// ---- split q/gate ----------------------------------------------------------

__global__ void qrocm_k_split_q_gate(
        qrocm_split_q_gate args, const char *projection, char *query, char *gate) {
    const uint64_t head_values = (uint64_t)args.n_query_head * args.head_dim;
    const uint64_t total = (uint64_t)args.n_token * head_values;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total || args.head_dim == 0 || args.n_query_head == 0) return;
    const uint64_t token = gid / head_values;
    const uint64_t wt = gid - token * head_values;
    const uint64_t head = wt / args.head_dim;
    const uint64_t dim = wt - head * args.head_dim;
    const uint64_t pbase = token * args.projection_token_stride + head * args.projection_head_stride;
    const float q = qrocm_ld(projection, pbase + dim * args.projection_dim_stride);
    const float g = qrocm_ld(projection, pbase + ((uint64_t)args.head_dim + dim) * args.projection_dim_stride);
    qrocm_st(query, token * args.query_token_stride + head * args.query_head_stride + dim * args.query_dim_stride, q);
    qrocm_st(gate, token * args.gate_token_stride + head * args.gate_head_stride + dim * args.gate_dim_stride, g);
}

// ---- sigmoid-mul (elementwise + row-wise) ----------------------------------

__global__ void qrocm_k_sigmoid_mul(
        qrocm_sigmoid_mul args, const char *input, const char *gate_logit, char *output) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= args.n_value) return;
    const float x = qrocm_ld(input, gid * args.input_stride);
    const float z = qrocm_ld(gate_logit, gid * args.gate_stride);
    qrocm_st(output, gid * args.output_stride, x * qrocm_sigmoid(z));
}

__global__ void qrocm_k_sigmoid_mul_rows(
        qrocm_sigmoid_mul_rows args, const char *input, const char *gate_logit, char *output) {
    const uint64_t total = (uint64_t)args.n_row * args.row_width;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total || args.row_width == 0u) return;
    const uint64_t row = gid / args.row_width;
    const uint64_t dim = gid - row * args.row_width;
    const float x = qrocm_ld(input, row * args.input_row_stride + dim * args.input_dim_stride);
    const float z = qrocm_ld(gate_logit, row * args.gate_row_stride);
    qrocm_st(output, row * args.output_row_stride + dim * args.output_dim_stride, x * qrocm_sigmoid(z));
}

// ---- RoPE (split-half NeoX over first n_rot dims) --------------------------

__global__ void qrocm_k_rope(
        qrocm_rope args, const char *source, const uint8_t *position, char *output) {
    const uint64_t head_values = (uint64_t)args.n_head * args.head_dim;
    const uint64_t total = (uint64_t)args.n_token * head_values;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total || args.n_head == 0 || args.head_dim == 0 ||
        args.n_rot == 0 || args.n_rot > args.head_dim ||
        (args.n_rot & 1u) != 0 || !(args.theta > 0.0f)) return;
    const uint64_t token = gid / head_values;
    const uint64_t wt = gid - token * head_values;
    const uint64_t head = wt / args.head_dim;
    const uint64_t dim = wt - head * args.head_dim;
    const uint64_t sbase = token * args.source_token_stride + head * args.source_head_stride;
    const uint64_t obase = token * args.output_token_stride + head * args.output_head_stride;
    if (dim >= (uint64_t)args.n_rot) {
        qrocm_st(output, obase + dim * args.output_dim_stride,
                 qrocm_ld(source, sbase + dim * args.source_dim_stride));
        return;
    }
    const uint64_t half_rot = (uint64_t)args.n_rot / 2u;
    if (dim >= half_rot) return;
    const float exponent = (2.0f * (float)dim) / (float)args.n_rot;
    const uint32_t pos = *((const uint32_t *)(position + token * args.position_stride));
    const float angle = (float)pos / powf(args.theta, exponent);
    const float c = cosf(angle), s = sinf(angle);
    const float a = qrocm_ld(source, sbase + dim * args.source_dim_stride);
    const float b = qrocm_ld(source, sbase + (dim + half_rot) * args.source_dim_stride);
    qrocm_st(output, obase + dim * args.output_dim_stride, a * c - b * s);
    qrocm_st(output, obase + (dim + half_rot) * args.output_dim_stride, b * c + a * s);
}

// ---- causal depthwise conv1d (k=4) + SiLU ----------------------------------

__global__ void qrocm_k_conv_step(
        qrocm_conv_step args, const char *input, const char *weight,
        char *state, char *output) {
    const uint64_t channel = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= (uint64_t)args.n_channel || args.kernel_size < 2u) return;
    const uint64_t sbase = channel * args.state_channel_stride;
    const uint64_t wbase = channel * args.weight_channel_stride;
    const float current = qrocm_ld(input, channel * args.input_channel_stride);
    float total = current * qrocm_ld(weight, wbase + ((uint64_t)args.kernel_size - 1u) * args.weight_tap_stride);
    const uint64_t hist = (uint64_t)args.kernel_size - 1u;
    for (uint64_t tap = 0; tap < hist; tap++)
        total += qrocm_ld(state, sbase + tap * args.state_tap_stride) *
                 qrocm_ld(weight, wbase + tap * args.weight_tap_stride);
    qrocm_st(output, channel * args.output_channel_stride, qrocm_silu(total));
    for (uint64_t tap = 0; tap + 1u < hist; tap++)
        qrocm_st(state, sbase + tap * args.state_tap_stride,
                 qrocm_ld(state, sbase + (tap + 1u) * args.state_tap_stride));
    qrocm_st(state, sbase + (hist - 1u) * args.state_tap_stride, current);
}

__global__ void qrocm_k_conv_sequence(
        qrocm_conv_sequence args, const char *input, const char *weight,
        char *state, char *output) {
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= args.n_channel || args.n_token == 0u || args.kernel_size != 4u) return;
    const uint64_t sbase = (uint64_t)channel * args.state_channel_stride;
    const uint64_t wbase = (uint64_t)channel * args.weight_channel_stride;
    float history[3];
    for (uint32_t tap = 0u; tap < 3u; tap++)
        history[tap] = qrocm_ld(state, sbase + (uint64_t)tap * args.state_tap_stride);
    for (uint32_t token = 0u; token < args.n_token; token++) {
        const uint64_t io = (uint64_t)token * args.input_token_stride +
                            (uint64_t)channel * args.input_channel_stride;
        const float current = qrocm_ld(input, io);
        float total = current * qrocm_ld(weight, wbase + 3u * args.weight_tap_stride);
        for (uint32_t tap = 0u; tap < 3u; tap++)
            total += history[tap] * qrocm_ld(weight, wbase + (uint64_t)tap * args.weight_tap_stride);
        qrocm_st(output, (uint64_t)token * args.output_token_stride +
                 (uint64_t)channel * args.output_channel_stride, qrocm_silu(total));
        history[0] = history[1]; history[1] = history[2]; history[2] = current;
    }
    for (uint32_t tap = 0u; tap < 3u; tap++)
        qrocm_st(state, sbase + (uint64_t)tap * args.state_tap_stride, history[tap]);
}

// ---- gated DeltaNet recurrent step (generic) -------------------------------
// One block owns one value head for one decode token.  blockDim.x is a multiple
// of 32; the block cooperatively reduces query/key norms then each lane strides
// over value rows.  scratch holds 2*n_wave floats.
__global__ void qrocm_k_gd_step(
        qrocm_gated_delta_step args, const char *query, const char *key,
        const char *value, const char *log_decay, const char *beta,
        char *state, char *output) {
    extern __shared__ float s_gd[];
    const uint32_t value_head = blockIdx.x, tid = threadIdx.x;
    const uint32_t lane = tid & 31u, wave = tid >> 5;
    const uint32_t n_wave = (blockDim.x + 31u) / 32u;
    if (value_head >= args.n_value_head || args.n_key_head == 0 ||
        args.key_dim == 0 || args.value_dim == 0 ||
        args.n_value_head % args.n_key_head != 0) return;
    float *query_partial = s_gd;
    float *key_partial = s_gd + n_wave;
    const uint32_t key_head = value_head % args.n_key_head;
    const uint64_t query_base = (uint64_t)key_head * args.query_head_stride;
    const uint64_t key_base = (uint64_t)key_head * args.key_head_stride;

    float qsq = 0.0f, ksq = 0.0f;
    for (uint32_t dim = tid; dim < args.key_dim; dim += blockDim.x) {
        const float q = qrocm_ld(query, query_base + (uint64_t)dim * args.query_dim_stride);
        const float k = qrocm_ld(key, key_base + (uint64_t)dim * args.key_dim_stride);
        qsq += q * q; ksq += k * k;
    }
    qsq = qrocm_wsum(qsq); ksq = qrocm_wsum(ksq);
    if (lane == 0u) { query_partial[wave] = qsq; key_partial[wave] = ksq; }
    __syncthreads();
    if (wave == 0u) {
        float q = lane < n_wave ? query_partial[lane] : 0.0f;
        float k = lane < n_wave ? key_partial[lane] : 0.0f;
        q = qrocm_wsum(q); k = qrocm_wsum(k);
        if (lane == 0u) {
            query_partial[0] = (1.0f / sqrtf((float)args.key_dim)) / sqrtf(q + 1.0e-6f);
            key_partial[0] = 1.0f / sqrtf(k + 1.0e-6f);
        }
    }
    __syncthreads();
    const float query_inverse = query_partial[0];
    const float key_inverse = key_partial[0];
    const float decay = expf(qrocm_ld(log_decay, (uint64_t)value_head * args.log_decay_head_stride));
    const float stepv = qrocm_ld(beta, (uint64_t)value_head * args.beta_head_stride);
    const uint64_t value_base = (uint64_t)value_head * args.value_head_stride;
    const uint64_t state_head_base = (uint64_t)value_head * args.state_head_stride;
    const uint64_t output_base = (uint64_t)value_head * args.output_head_stride;

    for (uint32_t vd = tid; vd < args.value_dim; vd += blockDim.x) {
        const uint64_t state_row = state_head_base + (uint64_t)vd * args.state_value_stride;
        float memory = 0.0f;
        for (uint32_t kd = 0; kd < args.key_dim; kd++) {
            const uint64_t so = state_row + (uint64_t)kd * args.state_key_stride;
            const float decayed = qrocm_ld(state, so) * decay;
            qrocm_st(state, so, decayed);
            const float nk = qrocm_ld(key, key_base + (uint64_t)kd * args.key_dim_stride) * key_inverse;
            memory += decayed * nk;
        }
        const float target = qrocm_ld(value, value_base + (uint64_t)vd * args.value_dim_stride);
        const float delta = (target - memory) * stepv;
        float result = 0.0f;
        for (uint32_t kd = 0; kd < args.key_dim; kd++) {
            const uint64_t so = state_row + (uint64_t)kd * args.state_key_stride;
            const float nk = qrocm_ld(key, key_base + (uint64_t)kd * args.key_dim_stride) * key_inverse;
            const float updated = qrocm_ld(state, so) + nk * delta;
            qrocm_st(state, so, updated);
            const float nq = qrocm_ld(query, query_base + (uint64_t)kd * args.query_dim_stride) * query_inverse;
            result += updated * nq;
        }
        qrocm_st(output, output_base + (uint64_t)vd * args.output_dim_stride, result);
    }
}

// gated DeltaNet decode specialized for key_dim=128: one wave per value row,
// four state cells per lane in registers, 32x4 block advances four rows.
// grid = (ceil(value_dim/4), n_value_head); block = (32, 4).
__global__ void qrocm_k_gd_step_128(
        qrocm_gated_delta_step args, const char *query, const char *key,
        const char *value, const char *log_decay, const char *beta,
        char *state, char *output) {
    constexpr uint32_t dims_per_lane = 4u, rows_per_group = 4u;
    __shared__ float norm[2];
    const uint32_t value_head = blockIdx.y, row_in_group = threadIdx.y;
    const uint32_t lane = threadIdx.x;
    if (value_head >= args.n_value_head || args.n_key_head == 0u ||
        args.key_dim != 128u || args.value_dim == 0u ||
        args.n_value_head % args.n_key_head != 0u) return;
    const uint32_t key_head = value_head % args.n_key_head;
    const uint64_t query_base = (uint64_t)key_head * args.query_head_stride;
    const uint64_t key_base = (uint64_t)key_head * args.key_head_stride;
    if (row_in_group == 0u) {
        float qsq = 0.0f, ksq = 0.0f;
        for (uint32_t j = 0u; j < dims_per_lane; j++) {
            const uint32_t dim = lane * dims_per_lane + j;
            const float q = qrocm_ld(query, query_base + (uint64_t)dim * args.query_dim_stride);
            const float k = qrocm_ld(key, key_base + (uint64_t)dim * args.key_dim_stride);
            qsq += q * q; ksq += k * k;
        }
        qsq = qrocm_wsum(qsq); ksq = qrocm_wsum(ksq);
        if (lane == 0u) {
            norm[0] = (1.0f / sqrtf((float)args.key_dim)) / sqrtf(qsq + 1.0e-6f);
            norm[1] = 1.0f / sqrtf(ksq + 1.0e-6f);
        }
    }
    __syncthreads();
    const uint32_t value_dim = blockIdx.x * rows_per_group + row_in_group;
    if (value_dim >= args.value_dim) return;
    const float query_inverse = norm[0], key_inverse = norm[1];
    const float decay = expf(qrocm_ld(log_decay, (uint64_t)value_head * args.log_decay_head_stride));
    const float stepv = qrocm_ld(beta, (uint64_t)value_head * args.beta_head_stride);
    const uint64_t state_row = (uint64_t)value_head * args.state_head_stride +
                               (uint64_t)value_dim * args.state_value_stride;
    float sv[dims_per_lane], nk[dims_per_lane], nq[dims_per_lane];
    float memory_partial = 0.0f;
    for (uint32_t j = 0u; j < dims_per_lane; j++) {
        const uint32_t dim = lane * dims_per_lane + j;
        const uint64_t so = state_row + (uint64_t)dim * args.state_key_stride;
        nk[j] = qrocm_ld(key, key_base + (uint64_t)dim * args.key_dim_stride) * key_inverse;
        nq[j] = qrocm_ld(query, query_base + (uint64_t)dim * args.query_dim_stride) * query_inverse;
        sv[j] = qrocm_ld(state, so) * decay;
        memory_partial += sv[j] * nk[j];
    }
    const float memory = qrocm_wsum(memory_partial);
    const uint64_t value_offset = (uint64_t)value_head * args.value_head_stride +
                                  (uint64_t)value_dim * args.value_dim_stride;
    const float target = qrocm_ld(value, value_offset);
    const float delta = (target - memory) * stepv;
    float result_partial = 0.0f;
    for (uint32_t j = 0u; j < dims_per_lane; j++) {
        const uint32_t dim = lane * dims_per_lane + j;
        const uint64_t so = state_row + (uint64_t)dim * args.state_key_stride;
        sv[j] += nk[j] * delta;
        qrocm_st(state, so, sv[j]);
        result_partial += sv[j] * nq[j];
    }
    const float result = qrocm_wsum(result_partial);
    if (lane == 0u) {
        qrocm_st(output, (uint64_t)value_head * args.output_head_stride +
                 (uint64_t)value_dim * args.output_dim_stride, result);
    }
}

// gated DeltaNet prefill, key_dim=value_dim=128; one wave per value row keeps
// its 128 state cells in registers across the whole token chunk (serial tokens,
// parallel rows/heads).  grid = (ceil(value_dim/4), n_value_head); block=(32,4).
__global__ void qrocm_k_gd_sequence_128(
        qrocm_gated_delta_sequence args, const char *projection,
        const char *log_decay, const char *beta, char *state, char *output) {
    constexpr uint32_t dims_per_lane = 4u, rows_per_group = 4u;
    __shared__ float norm[2];
    const uint32_t value_head = blockIdx.y, row_in_group = threadIdx.y;
    const uint32_t lane = threadIdx.x;
    if (value_head >= args.n_value_head || args.n_token == 0u ||
        args.n_key_head == 0u || args.key_dim != 128u ||
        args.value_dim != 128u || args.n_value_head % args.n_key_head != 0u) return;
    const uint32_t value_dim = blockIdx.x * rows_per_group + row_in_group;
    const bool active_row = value_dim < args.value_dim;
    const uint32_t key_head = value_head % args.n_key_head;
    const uint64_t state_row = (uint64_t)value_head * args.state_head_stride +
                               (uint64_t)value_dim * args.state_value_stride;
    float sv[dims_per_lane] = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (active_row) {
        for (uint32_t j = 0u; j < dims_per_lane; j++) {
            const uint32_t dim = lane * dims_per_lane + j;
            sv[j] = qrocm_ld(state, state_row + (uint64_t)dim * args.state_key_stride);
        }
    }
    for (uint32_t token = 0u; token < args.n_token; token++) {
        const uint64_t pbase = (uint64_t)token * args.projection_token_stride;
        const uint64_t query_base = pbase + args.query_offset + (uint64_t)key_head * args.query_head_stride;
        const uint64_t key_base = pbase + args.key_offset + (uint64_t)key_head * args.key_head_stride;
        if (row_in_group == 0u) {
            float qsq = 0.0f, ksq = 0.0f;
            for (uint32_t j = 0u; j < dims_per_lane; j++) {
                const uint32_t dim = lane * dims_per_lane + j;
                const float q = qrocm_ld(projection, query_base + (uint64_t)dim * args.query_dim_stride);
                const float k = qrocm_ld(projection, key_base + (uint64_t)dim * args.key_dim_stride);
                qsq += q * q; ksq += k * k;
            }
            qsq = qrocm_wsum(qsq); ksq = qrocm_wsum(ksq);
            if (lane == 0u) {
                norm[0] = (1.0f / sqrtf((float)args.key_dim)) / sqrtf(qsq + 1.0e-6f);
                norm[1] = 1.0f / sqrtf(ksq + 1.0e-6f);
            }
        }
        __syncthreads();
        if (active_row) {
            const float query_inverse = norm[0], key_inverse = norm[1];
            const float decay = expf(qrocm_ld(log_decay,
                (uint64_t)token * args.log_decay_token_stride +
                (uint64_t)value_head * args.log_decay_head_stride));
            const float stepv = qrocm_ld(beta,
                (uint64_t)token * args.beta_token_stride +
                (uint64_t)value_head * args.beta_head_stride);
            float nk[dims_per_lane], nq[dims_per_lane];
            float memory_partial = 0.0f;
            for (uint32_t j = 0u; j < dims_per_lane; j++) {
                const uint32_t dim = lane * dims_per_lane + j;
                nk[j] = qrocm_ld(projection, key_base + (uint64_t)dim * args.key_dim_stride) * key_inverse;
                nq[j] = qrocm_ld(projection, query_base + (uint64_t)dim * args.query_dim_stride) * query_inverse;
                sv[j] *= decay;
                memory_partial += sv[j] * nk[j];
            }
            const float memory = qrocm_wsum(memory_partial);
            const float target = qrocm_ld(projection, pbase + args.value_offset +
                (uint64_t)value_head * args.value_head_stride +
                (uint64_t)value_dim * args.value_dim_stride);
            const float delta = (target - memory) * stepv;
            float result_partial = 0.0f;
            for (uint32_t j = 0u; j < dims_per_lane; j++) {
                sv[j] += nk[j] * delta;
                result_partial += sv[j] * nq[j];
            }
            const float result = qrocm_wsum(result_partial);
            if (lane == 0u) {
                qrocm_st(output, (uint64_t)token * args.output_token_stride +
                    (uint64_t)value_head * args.output_head_stride +
                    (uint64_t)value_dim * args.output_dim_stride, result);
            }
        }
        __syncthreads();
    }
    if (active_row) {
        for (uint32_t j = 0u; j < dims_per_lane; j++) {
            const uint32_t dim = lane * dims_per_lane + j;
            qrocm_st(state, state_row + (uint64_t)dim * args.state_key_stride, sv[j]);
        }
    }
}

// ---- GQA decode / prefill (online softmax) ---------------------------------
// One block owns one query head; blockDim.x >= head_dim (multiple of 32).
// scratch = n_wave + 4 floats.
__global__ void qrocm_k_gqa_decode(
        qrocm_gqa_decode args, const char *query, const char *key_cache,
        const char *value_cache, char *output) {
    extern __shared__ float s_gqa[];
    const uint32_t query_head = blockIdx.x, tid = threadIdx.x;
    const uint32_t lane = tid & 31u, wave = tid >> 5;
    const uint32_t n_wave = (blockDim.x + 31u) / 32u;
    if (query_head >= args.n_query_head || args.n_kv == 0u ||
        args.n_kv_head == 0u || args.head_dim == 0u ||
        args.n_query_head % args.n_kv_head != 0u || blockDim.x < args.head_dim) return;
    if (n_wave > 32u) return;
    const uint32_t control = n_wave;
    const uint32_t query_per_kv = args.n_query_head / args.n_kv_head;
    const uint32_t kv_head = query_head / query_per_kv;
    const uint64_t query_base = (uint64_t)query_head * args.query_head_stride;
    const uint64_t output_base = (uint64_t)query_head * args.output_head_stride;
    const float scale = 1.0f / sqrtf((float)args.head_dim);
    float accumulator = 0.0f;
    if (tid == 0u) {
        s_gqa[control + 0u] = -INFINITY; s_gqa[control + 1u] = 0.0f;
        s_gqa[control + 2u] = 0.0f; s_gqa[control + 3u] = 0.0f;
    }
    __syncthreads();
    for (uint32_t token = 0; token < args.n_kv; token++) {
        const uint64_t key_base = (uint64_t)token * args.key_token_stride +
                                  (uint64_t)kv_head * args.key_head_stride;
        float dot = 0.0f;
        for (uint32_t dim = tid; dim < args.head_dim; dim += blockDim.x)
            dot += qrocm_ld(query, query_base + (uint64_t)dim * args.query_dim_stride) *
                   qrocm_ld(key_cache, key_base + (uint64_t)dim * args.key_dim_stride);
        dot = qrocm_wsum(dot);
        if (lane == 0u) s_gqa[wave] = dot;
        __syncthreads();
        if (wave == 0u) {
            float d = lane < n_wave ? s_gqa[lane] : 0.0f;
            d = qrocm_wsum(d);
            if (lane == 0u) {
                const float score = d * scale;
                const float pm = s_gqa[control + 0u];
                const float nm = fmaxf(pm, score);
                const float pf = expf(pm - nm), cf = expf(score - nm);
                s_gqa[control + 0u] = nm;
                s_gqa[control + 1u] = s_gqa[control + 1u] * pf + cf;
                s_gqa[control + 2u] = pf; s_gqa[control + 3u] = cf;
            }
        }
        __syncthreads();
        if (tid < args.head_dim) {
            const uint64_t vo = (uint64_t)token * args.value_token_stride +
                                (uint64_t)kv_head * args.value_head_stride +
                                (uint64_t)tid * args.value_dim_stride;
            accumulator = accumulator * s_gqa[control + 2u] +
                          qrocm_ld(value_cache, vo) * s_gqa[control + 3u];
        }
        __syncthreads();
    }
    if (tid < args.head_dim)
        qrocm_st(output, output_base + (uint64_t)tid * args.output_dim_stride,
                 accumulator / s_gqa[control + 1u]);
}

__global__ void qrocm_k_gqa_prefill(
        qrocm_gqa_prefill args, const char *query, const char *key_cache,
        const char *value_cache, char *output) {
    extern __shared__ float s_gqp[];
    const uint32_t query_head = blockIdx.x, query_token = blockIdx.y, tid = threadIdx.x;
    const uint32_t lane = tid & 31u, wave = tid >> 5;
    const uint32_t n_wave = (blockDim.x + 31u) / 32u;
    if (query_head >= args.n_query_head || query_token >= args.n_token ||
        args.n_kv_head == 0u || args.head_dim == 0u ||
        args.n_query_head % args.n_kv_head != 0u || blockDim.x < args.head_dim) return;
    if (n_wave > 32u) return;
    const uint32_t control = n_wave;
    const uint32_t query_per_kv = args.n_query_head / args.n_kv_head;
    const uint32_t kv_head = query_head / query_per_kv;
    const uint32_t n_kv = args.position0 + query_token + 1u;
    const uint64_t query_base = (uint64_t)query_token * args.query_token_stride +
                                (uint64_t)query_head * args.query_head_stride;
    const uint64_t output_base = (uint64_t)query_token * args.output_token_stride +
                                 (uint64_t)query_head * args.output_head_stride;
    const float scale = 1.0f / sqrtf((float)args.head_dim);
    float accumulator = 0.0f;
    if (tid == 0u) {
        s_gqp[control + 0u] = -INFINITY; s_gqp[control + 1u] = 0.0f;
        s_gqp[control + 2u] = 0.0f; s_gqp[control + 3u] = 0.0f;
    }
    __syncthreads();
    for (uint32_t token = 0u; token < n_kv; token++) {
        const uint64_t key_base = (uint64_t)token * args.key_token_stride +
                                  (uint64_t)kv_head * args.key_head_stride;
        float dot = 0.0f;
        for (uint32_t dim = tid; dim < args.head_dim; dim += blockDim.x)
            dot += qrocm_ld(query, query_base + (uint64_t)dim * args.query_dim_stride) *
                   qrocm_ld(key_cache, key_base + (uint64_t)dim * args.key_dim_stride);
        dot = qrocm_wsum(dot);
        if (lane == 0u) s_gqp[wave] = dot;
        __syncthreads();
        if (wave == 0u) {
            float d = lane < n_wave ? s_gqp[lane] : 0.0f;
            d = qrocm_wsum(d);
            if (lane == 0u) {
                const float score = d * scale;
                const float pm = s_gqp[control + 0u];
                const float nm = fmaxf(pm, score);
                const float pf = expf(pm - nm), cf = expf(score - nm);
                s_gqp[control + 0u] = nm;
                s_gqp[control + 1u] = s_gqp[control + 1u] * pf + cf;
                s_gqp[control + 2u] = pf; s_gqp[control + 3u] = cf;
            }
        }
        __syncthreads();
        if (tid < args.head_dim) {
            const uint64_t vo = (uint64_t)token * args.value_token_stride +
                                (uint64_t)kv_head * args.value_head_stride +
                                (uint64_t)tid * args.value_dim_stride;
            accumulator = accumulator * s_gqp[control + 2u] +
                          qrocm_ld(value_cache, vo) * s_gqp[control + 3u];
        }
        __syncthreads();
    }
    if (tid < args.head_dim)
        qrocm_st(output, output_base + (uint64_t)tid * args.output_dim_stride,
                 accumulator / s_gqp[control + 1u]);
}

// ---- gated RMSNorm ---------------------------------------------------------
// One block per row; scratch = n_wave floats.
__global__ void qrocm_k_rmsnorm_gated(
        qrocm_rmsnorm_gated args, const char *input, const char *gate,
        const char *weight, char *output) {
    extern __shared__ float s_rms[];
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const uint32_t lane = tid & 31u, wave = tid >> 5;
    const uint32_t n_wave = (blockDim.x + 31u) / 32u;
    if (row >= args.n_vector || args.dim == 0 || !(args.epsilon > 0.0f)) return;
    const uint64_t input_base = (uint64_t)row * args.input_vector_stride;
    float square = 0.0f;
    for (uint32_t dim = tid; dim < args.dim; dim += blockDim.x) {
        const float x = qrocm_ld(input, input_base + (uint64_t)dim * args.input_dim_stride);
        square += x * x;
    }
    square = qrocm_wsum(square);
    if (lane == 0u) s_rms[wave] = square;
    __syncthreads();
    if (wave == 0u) {
        float total = lane < n_wave ? s_rms[lane] : 0.0f;
        total = qrocm_wsum(total);
        if (lane == 0u) s_rms[0] = 1.0f / sqrtf(total / (float)args.dim + args.epsilon);
    }
    __syncthreads();
    const float inverse_rms = s_rms[0];
    const uint64_t gate_base = (uint64_t)row * args.gate_vector_stride;
    const uint64_t output_base = (uint64_t)row * args.output_vector_stride;
    for (uint32_t dim = tid; dim < args.dim; dim += blockDim.x) {
        const float x = qrocm_ld(input, input_base + (uint64_t)dim * args.input_dim_stride);
        const float z = qrocm_ld(gate, gate_base + (uint64_t)dim * args.gate_dim_stride);
        const float w = qrocm_ld(weight, (uint64_t)dim * args.weight_dim_stride);
        qrocm_st(output, output_base + (uint64_t)dim * args.output_dim_stride,
                 x * inverse_rms * w * qrocm_silu(z));
    }
}

// ===========================================================================
// Host entry points.  Stride expressions mirror ds4_metal.m exactly (the
// porting contract); geometry maps Metal threadgroups/simdgroups to HIP
// blocks/waves.  Resident-only: weights are resolved through the model range
// registry (the whole payload is registered in resident mode).
// ===========================================================================

static uint32_t qrocm_elem_threads(uint64_t count) {
    uint32_t n = count < 256u ? (uint32_t)count : 256u;
    return n == 0u ? 1u : n;
}
// clamp(preferred,32,1024) floored to a multiple of 32; *nsg = nth/32.
static uint32_t qrocm_reduction_threads(uint32_t preferred, uint32_t *nsg) {
    uint32_t nth = preferred < 32u ? 32u : (preferred > 1024u ? 1024u : preferred);
    nth -= nth % 32u;
    if (nth == 0u) nth = 32u;
    *nsg = nth / 32u;
    return nth;
}

extern "C" int ds4_gpu_qwen35_split_q_gate_batch_tensor(
        ds4_gpu_tensor *query, ds4_gpu_tensor *gate, const ds4_gpu_tensor *projection,
        uint32_t n_token, uint32_t n_query_head, uint32_t head_dim) {
    if (!query || !gate || !projection || !n_token || !n_query_head || !head_dim) return 0;
    const uint64_t row_values = (uint64_t)n_query_head * head_dim;
    const uint64_t values = row_values * n_token;
    if (values > UINT32_MAX) return 0;
    const uint64_t out_bytes = values * 4u;
    if (!cuda_tensor_has_bytes(projection, out_bytes * 2u) ||
        !cuda_tensor_has_bytes(query, out_bytes) || !cuda_tensor_has_bytes(gate, out_bytes)) return 0;
    qrocm_split_q_gate a{};
    a.n_token = n_token; a.n_query_head = n_query_head; a.head_dim = head_dim;
    a.projection_token_stride = 2u * row_values * 4u;
    a.projection_head_stride = 2ull * head_dim * 4u;
    a.projection_dim_stride = 4u;
    a.query_token_stride = row_values * 4u; a.query_head_stride = (uint64_t)head_dim * 4u; a.query_dim_stride = 4u;
    a.gate_token_stride = row_values * 4u; a.gate_head_stride = (uint64_t)head_dim * 4u; a.gate_dim_stride = 4u;
    const uint32_t nth = qrocm_elem_threads(values);
    qrocm_k_split_q_gate<<<(unsigned)((values + nth - 1) / nth), nth>>>(
        a, (const char *)projection->ptr, (char *)query->ptr, (char *)gate->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 split_q_gate");
}
extern "C" int ds4_gpu_qwen35_split_q_gate_tensor(
        ds4_gpu_tensor *query, ds4_gpu_tensor *gate, const ds4_gpu_tensor *projection,
        uint32_t n_query_head, uint32_t head_dim) {
    return ds4_gpu_qwen35_split_q_gate_batch_tensor(query, gate, projection, 1u, n_query_head, head_dim);
}

extern "C" int ds4_gpu_qwen35_sigmoid_mul_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *input, const ds4_gpu_tensor *gate,
        uint32_t n_value, bool broadcast_gate) {
    if (!out || !input || !gate || !n_value) return 0;
    const uint64_t bytes = (uint64_t)n_value * 4u;
    if (!cuda_tensor_has_bytes(input, bytes) || !cuda_tensor_has_bytes(out, bytes) ||
        !cuda_tensor_has_bytes(gate, broadcast_gate ? 4u : bytes)) return 0;
    qrocm_sigmoid_mul a{};
    a.n_value = n_value; a.input_stride = 4u; a.gate_stride = broadcast_gate ? 0u : 4u; a.output_stride = 4u;
    const uint32_t nth = qrocm_elem_threads(n_value);
    qrocm_k_sigmoid_mul<<<(unsigned)(((uint64_t)n_value + nth - 1) / nth), nth>>>(
        a, (const char *)input->ptr, (const char *)gate->ptr, (char *)out->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 sigmoid_mul");
}

extern "C" int ds4_gpu_qwen35_sigmoid_mul_rows_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *input, const ds4_gpu_tensor *gate,
        uint32_t n_row, uint32_t row_width) {
    if (!out || !input || !gate || !n_row || !row_width) return 0;
    const uint64_t values = (uint64_t)n_row * row_width;
    if (values > UINT32_MAX) return 0;
    const uint64_t bytes = values * 4u;
    if (!cuda_tensor_has_bytes(input, bytes) || !cuda_tensor_has_bytes(out, bytes) ||
        !cuda_tensor_has_bytes(gate, (uint64_t)n_row * 4u)) return 0;
    qrocm_sigmoid_mul_rows a{};
    a.n_row = n_row; a.row_width = row_width;
    a.input_row_stride = (uint64_t)row_width * 4u; a.input_dim_stride = 4u; a.gate_row_stride = 4u;
    a.output_row_stride = (uint64_t)row_width * 4u; a.output_dim_stride = 4u;
    const uint32_t nth = qrocm_elem_threads(values);
    qrocm_k_sigmoid_mul_rows<<<(unsigned)((values + nth - 1) / nth), nth>>>(
        a, (const char *)input->ptr, (const char *)gate->ptr, (char *)out->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 sigmoid_mul_rows");
}

static int qrocm_rope_impl(
        ds4_gpu_tensor *values, const ds4_gpu_tensor *positions, const uint32_t *single_position,
        uint32_t n_token, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, float theta) {
    if (!values || (positions == NULL) == (single_position == NULL)) return 0;
    if (!n_token || !n_head || !head_dim) return 0;
    if (n_rot == 0u) return 1;
    if (n_rot > head_dim || (n_rot & 1u) != 0u || !(theta > 0.0f) || !isfinite(theta)) return 0;
    const uint64_t row_values = (uint64_t)n_head * head_dim;
    const uint64_t count = row_values * n_token;
    if (count > UINT32_MAX) return 0;
    if (!cuda_tensor_has_bytes(values, count * 4u)) return 0;
    if (positions && !cuda_tensor_has_bytes(positions, (uint64_t)n_token * 4u)) return 0;
    qrocm_rope a{};
    a.n_token = n_token; a.n_head = n_head; a.head_dim = head_dim; a.n_rot = n_rot; a.theta = theta;
    a.source_token_stride = row_values * 4u; a.source_head_stride = (uint64_t)head_dim * 4u; a.source_dim_stride = 4u;
    a.output_token_stride = row_values * 4u; a.output_head_stride = (uint64_t)head_dim * 4u; a.output_dim_stride = 4u;
    a.position_stride = 4u;
    // Positions live in device memory; for the scalar form stage the single
    // value into a tiny device buffer so the kernel reads it uniformly.
    const uint8_t *pos_ptr = NULL;
    uint32_t *staged = NULL;
    if (positions) {
        pos_ptr = (const uint8_t *)positions->ptr;
    } else {
        if (cudaMalloc((void **)&staged, sizeof(uint32_t)) != cudaSuccess) return 0;
        if (cudaMemcpy(staged, single_position, sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(staged); return 0;
        }
        a.position_stride = 0u;
        pos_ptr = (const uint8_t *)staged;
    }
    const uint32_t nth = qrocm_elem_threads(count);
    qrocm_k_rope<<<(unsigned)((count + nth - 1) / nth), nth>>>(
        a, (const char *)values->ptr, pos_ptr, (char *)values->ptr);
    const int ok = cuda_ok(cudaGetLastError(), "qwen35 rope");
    if (staged) { cudaDeviceSynchronize(); cudaFree(staged); }
    return ok;
}
extern "C" int ds4_gpu_qwen35_rope_prefix_tensor(
        ds4_gpu_tensor *values, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
        uint32_t position, float theta) {
    return qrocm_rope_impl(values, NULL, &position, 1u, n_head, head_dim, n_rot, theta);
}
extern "C" int ds4_gpu_qwen35_rope_prefix_batch_tensor(
        ds4_gpu_tensor *values, const ds4_gpu_tensor *positions, uint32_t n_token,
        uint32_t n_head, uint32_t head_dim, uint32_t n_rot, float theta) {
    return qrocm_rope_impl(values, positions, NULL, n_token, n_head, head_dim, n_rot, theta);
}

extern "C" int ds4_gpu_qwen35_causal_conv_step_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *state, const ds4_gpu_tensor *input,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_channel, uint32_t kernel_size) {
    if (!out || !state || !input || !model_map || !n_channel || kernel_size < 2u) return 0;
    const uint64_t weight_bytes = (uint64_t)n_channel * kernel_size * 4u;
    const uint64_t state_bytes = (uint64_t)n_channel * (kernel_size - 1u) * 4u;
    if (!cuda_model_range_fits(model_size, weight_offset, weight_bytes)) return 0;
    if (!cuda_tensor_has_bytes(input, (uint64_t)n_channel * 4u) ||
        !cuda_tensor_has_bytes(state, state_bytes) ||
        !cuda_tensor_has_bytes(out, (uint64_t)n_channel * 4u)) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "qwen35_conv_step_w");
    if (!w) return 0;
    qrocm_conv_step a{};
    a.n_channel = n_channel; a.kernel_size = kernel_size;
    a.input_channel_stride = 4u; a.weight_channel_stride = (uint64_t)kernel_size * 4u; a.weight_tap_stride = 4u;
    a.state_channel_stride = (uint64_t)(kernel_size - 1u) * 4u; a.state_tap_stride = 4u; a.output_channel_stride = 4u;
    const uint32_t nth = qrocm_elem_threads(n_channel);
    qrocm_k_conv_step<<<(unsigned)(((uint64_t)n_channel + nth - 1) / nth), nth>>>(
        a, (const char *)input->ptr, w, (char *)state->ptr, (char *)out->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 conv_step");
}

extern "C" int ds4_gpu_qwen35_causal_conv_sequence_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *state, const ds4_gpu_tensor *input,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_token, uint32_t n_channel, uint32_t kernel_size) {
    if (!out || !state || !input || !model_map || !n_token || !n_channel || kernel_size != 4u) return 0;
    const uint64_t activation_bytes = (uint64_t)n_channel * n_token * 4u;
    const uint64_t state_bytes = (uint64_t)n_channel * 3u * 4u;
    const uint64_t weight_bytes = (uint64_t)n_channel * 4u * 4u;
    if (!cuda_model_range_fits(model_size, weight_offset, weight_bytes)) return 0;
    if (!cuda_tensor_has_bytes(input, activation_bytes) ||
        !cuda_tensor_has_bytes(state, state_bytes) ||
        !cuda_tensor_has_bytes(out, activation_bytes)) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "qwen35_conv_seq_w");
    if (!w) return 0;
    qrocm_conv_sequence a{};
    a.n_token = n_token; a.n_channel = n_channel; a.kernel_size = 4u;
    a.input_token_stride = (uint64_t)n_channel * 4u; a.input_channel_stride = 4u;
    a.weight_channel_stride = 16u; a.weight_tap_stride = 4u;
    a.state_channel_stride = 12u; a.state_tap_stride = 4u;
    a.output_token_stride = (uint64_t)n_channel * 4u; a.output_channel_stride = 4u;
    const uint32_t nth = qrocm_elem_threads(n_channel);
    qrocm_k_conv_sequence<<<(unsigned)(((uint64_t)n_channel + nth - 1) / nth), nth>>>(
        a, (const char *)input->ptr, w, (char *)state->ptr, (char *)out->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 conv_sequence");
}

extern "C" int ds4_gpu_qwen35_gated_delta_step_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *state, const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *key, const ds4_gpu_tensor *value, const ds4_gpu_tensor *log_decay,
        const ds4_gpu_tensor *beta, uint32_t n_key_head, uint32_t n_value_head,
        uint32_t key_dim, uint32_t value_dim) {
    if (!out || !state || !query || !key || !value || !log_decay || !beta) return 0;
    if (!n_key_head || !n_value_head || !key_dim || !value_dim ||
        n_value_head % n_key_head != 0u) return 0;
    const uint64_t key_bytes = (uint64_t)n_key_head * key_dim * 4u;
    const uint64_t value_bytes = (uint64_t)n_value_head * value_dim * 4u;
    const uint64_t control_bytes = (uint64_t)n_value_head * 4u;
    const uint64_t state_bytes = (uint64_t)n_value_head * value_dim * key_dim * 4u;
    if (!cuda_tensor_has_bytes(query, key_bytes) || !cuda_tensor_has_bytes(key, key_bytes) ||
        !cuda_tensor_has_bytes(value, value_bytes) || !cuda_tensor_has_bytes(log_decay, control_bytes) ||
        !cuda_tensor_has_bytes(beta, control_bytes) || !cuda_tensor_has_bytes(state, state_bytes) ||
        !cuda_tensor_has_bytes(out, value_bytes)) return 0;
    qrocm_gated_delta_step a{};
    a.n_key_head = n_key_head; a.n_value_head = n_value_head; a.key_dim = key_dim; a.value_dim = value_dim;
    a.query_head_stride = (uint64_t)key_dim * 4u; a.query_dim_stride = 4u;
    a.key_head_stride = (uint64_t)key_dim * 4u; a.key_dim_stride = 4u;
    a.value_head_stride = (uint64_t)value_dim * 4u; a.value_dim_stride = 4u;
    a.log_decay_head_stride = 4u; a.beta_head_stride = 4u;
    a.state_head_stride = (uint64_t)value_dim * key_dim * 4u; a.state_value_stride = (uint64_t)key_dim * 4u; a.state_key_stride = 4u;
    a.output_head_stride = (uint64_t)value_dim * 4u; a.output_dim_stride = 4u;
    if (key_dim == 128u) {
        dim3 grid((value_dim + 3u) / 4u, n_value_head, 1u);
        dim3 block(32u, 4u, 1u);
        qrocm_k_gd_step_128<<<grid, block>>>(
            a, (const char *)query->ptr, (const char *)key->ptr, (const char *)value->ptr,
            (const char *)log_decay->ptr, (const char *)beta->ptr, (char *)state->ptr, (char *)out->ptr);
        return cuda_ok(cudaGetLastError(), "qwen35 gd_step_128");
    }
    uint32_t nsg = 0; const uint32_t nth = qrocm_reduction_threads(128u, &nsg);
    qrocm_k_gd_step<<<n_value_head, nth, 2u * nsg * sizeof(float)>>>(
        a, (const char *)query->ptr, (const char *)key->ptr, (const char *)value->ptr,
        (const char *)log_decay->ptr, (const char *)beta->ptr, (char *)state->ptr, (char *)out->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 gd_step");
}

extern "C" int ds4_gpu_qwen35_gated_delta_sequence_128_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *state, const ds4_gpu_tensor *projection,
        const ds4_gpu_tensor *log_decay, const ds4_gpu_tensor *beta,
        uint32_t n_token, uint32_t n_key_head, uint32_t n_value_head) {
    if (!out || !state || !projection || !log_decay || !beta) return 0;
    if (!n_token || !n_key_head || !n_value_head || n_value_head % n_key_head != 0u) return 0;
    const uint64_t query_values = (uint64_t)n_key_head * 128u;
    const uint64_t value_values = (uint64_t)n_value_head * 128u;
    const uint64_t proj_row = query_values + query_values + value_values;
    const uint64_t proj_bytes = proj_row * n_token * 4u;
    const uint64_t out_bytes = value_values * n_token * 4u;
    const uint64_t control_bytes = (uint64_t)n_value_head * n_token * 4u;
    const uint64_t state_bytes = (uint64_t)n_value_head * 128u * 128u * 4u;
    if (!cuda_tensor_has_bytes(projection, proj_bytes) || !cuda_tensor_has_bytes(log_decay, control_bytes) ||
        !cuda_tensor_has_bytes(beta, control_bytes) || !cuda_tensor_has_bytes(state, state_bytes) ||
        !cuda_tensor_has_bytes(out, out_bytes)) return 0;
    qrocm_gated_delta_sequence a{};
    a.n_token = n_token; a.n_key_head = n_key_head; a.n_value_head = n_value_head;
    a.key_dim = 128u; a.value_dim = 128u;
    a.projection_token_stride = proj_row * 4u;
    a.query_offset = 0u; a.key_offset = query_values * 4u; a.value_offset = (query_values + query_values) * 4u;
    a.query_head_stride = 512u; a.query_dim_stride = 4u;
    a.key_head_stride = 512u; a.key_dim_stride = 4u;
    a.value_head_stride = 512u; a.value_dim_stride = 4u;
    a.log_decay_token_stride = (uint64_t)n_value_head * 4u; a.log_decay_head_stride = 4u;
    a.beta_token_stride = (uint64_t)n_value_head * 4u; a.beta_head_stride = 4u;
    a.state_head_stride = 65536u; a.state_value_stride = 512u; a.state_key_stride = 4u;
    a.output_token_stride = value_values * 4u; a.output_head_stride = 512u; a.output_dim_stride = 4u;
    dim3 grid(32u, n_value_head, 1u);
    dim3 block(32u, 4u, 1u);
    qrocm_k_gd_sequence_128<<<grid, block>>>(
        a, (const char *)projection->ptr, (const char *)log_decay->ptr,
        (const char *)beta->ptr, (char *)state->ptr, (char *)out->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 gd_sequence_128");
}

static int qrocm_gd_controls_impl(
        ds4_gpu_tensor *log_decay, ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *alpha_logit, const ds4_gpu_tensor *beta_logit,
        const void *model_map, uint64_t model_size, uint64_t ssm_a_offset, uint64_t dt_bias_offset,
        uint32_t n_token, uint32_t n_value_head) {
    if (!log_decay || !beta || !alpha_logit || !beta_logit || !model_map || !n_token || !n_value_head) return 0;
    const uint64_t bytes = (uint64_t)n_token * n_value_head * 4u;
    const uint64_t weight_bytes = (uint64_t)n_value_head * 4u;
    if (!cuda_model_range_fits(model_size, ssm_a_offset, weight_bytes) ||
        !cuda_model_range_fits(model_size, dt_bias_offset, weight_bytes)) return 0;
    if (!cuda_tensor_has_bytes(alpha_logit, bytes) || !cuda_tensor_has_bytes(beta_logit, bytes) ||
        !cuda_tensor_has_bytes(log_decay, bytes) || !cuda_tensor_has_bytes(beta, bytes)) return 0;
    const char *ssm_a = cuda_model_range_ptr(model_map, ssm_a_offset, weight_bytes, "qwen35_ssm_a");
    const char *dt_bias = cuda_model_range_ptr(model_map, dt_bias_offset, weight_bytes, "qwen35_dt_bias");
    if (!ssm_a || !dt_bias) return 0;
    qrocm_gated_delta_controls a{};
    a.n_token = n_token; a.n_value_head = n_value_head;
    a.alpha_logit_token_stride = (uint64_t)n_value_head * 4u; a.alpha_logit_head_stride = 4u;
    a.beta_logit_token_stride = (uint64_t)n_value_head * 4u; a.beta_logit_head_stride = 4u;
    a.ssm_a_head_stride = 4u; a.dt_bias_head_stride = 4u;
    a.log_decay_token_stride = (uint64_t)n_value_head * 4u; a.log_decay_head_stride = 4u;
    a.beta_token_stride = (uint64_t)n_value_head * 4u; a.beta_head_stride = 4u;
    const uint64_t count = (uint64_t)n_token * n_value_head;
    const uint32_t nth = qrocm_elem_threads(count);
    qrocm_k_gd_controls<<<(unsigned)((count + nth - 1) / nth), nth>>>(
        a, (const char *)alpha_logit->ptr, (const char *)beta_logit->ptr, ssm_a, dt_bias,
        (char *)log_decay->ptr, (char *)beta->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 gd_controls");
}
extern "C" int ds4_gpu_qwen35_gated_delta_controls_tensor(
        ds4_gpu_tensor *log_decay, ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *alpha_logit, const ds4_gpu_tensor *beta_logit,
        const void *model_map, uint64_t model_size, uint64_t ssm_a_offset, uint64_t dt_bias_offset,
        uint32_t n_value_head) {
    return qrocm_gd_controls_impl(log_decay, beta, alpha_logit, beta_logit, model_map, model_size,
                                  ssm_a_offset, dt_bias_offset, 1u, n_value_head);
}
extern "C" int ds4_gpu_qwen35_gated_delta_controls_batch_tensor(
        ds4_gpu_tensor *log_decay, ds4_gpu_tensor *beta,
        const ds4_gpu_tensor *alpha_logit, const ds4_gpu_tensor *beta_logit,
        const void *model_map, uint64_t model_size, uint64_t ssm_a_offset, uint64_t dt_bias_offset,
        uint32_t n_token, uint32_t n_value_head) {
    return qrocm_gd_controls_impl(log_decay, beta, alpha_logit, beta_logit, model_map, model_size,
                                  ssm_a_offset, dt_bias_offset, n_token, n_value_head);
}

extern "C" int ds4_gpu_qwen35_rmsnorm_gated_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *input, const ds4_gpu_tensor *gate,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_vector, uint32_t dim, float epsilon) {
    if (!out || !input || !gate || !model_map || !n_vector || !dim ||
        !(epsilon > 0.0f) || !isfinite(epsilon)) return 0;
    const uint64_t values = (uint64_t)n_vector * dim;
    const uint64_t bytes = values * 4u;
    const uint64_t weight_bytes = (uint64_t)dim * 4u;
    if (!cuda_model_range_fits(model_size, weight_offset, weight_bytes)) return 0;
    if (!cuda_tensor_has_bytes(input, bytes) || !cuda_tensor_has_bytes(gate, bytes) ||
        !cuda_tensor_has_bytes(out, bytes)) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "qwen35_rmsnorm_w");
    if (!w) return 0;
    qrocm_rmsnorm_gated a{};
    a.n_vector = n_vector; a.dim = dim; a.epsilon = epsilon;
    a.input_vector_stride = (uint64_t)dim * 4u; a.input_dim_stride = 4u;
    a.gate_vector_stride = (uint64_t)dim * 4u; a.gate_dim_stride = 4u; a.weight_dim_stride = 4u;
    a.output_vector_stride = (uint64_t)dim * 4u; a.output_dim_stride = 4u;
    uint32_t nsg = 0; const uint32_t nth = qrocm_reduction_threads(128u, &nsg);
    qrocm_k_rmsnorm_gated<<<n_vector, nth, nsg * sizeof(float)>>>(
        a, (const char *)input->ptr, (const char *)gate->ptr, w, (char *)out->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 rmsnorm_gated");
}

extern "C" int ds4_gpu_qwen35_dequant_embedding_q8_0_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t embedding_offset, uint32_t row_index, uint32_t n_embd) {
    if (!out || !model_map || !n_embd || (n_embd % 32u) != 0u) return 0;
    const uint64_t row_bytes = (uint64_t)(n_embd / 32u) * 34u;
    const uint64_t row_offset = embedding_offset + (uint64_t)row_index * row_bytes;
    if (!cuda_model_range_fits(model_size, row_offset, row_bytes)) return 0;
    if (!cuda_tensor_has_bytes(out, (uint64_t)n_embd * 4u)) return 0;
    const char *row = cuda_model_range_ptr(model_map, row_offset, row_bytes, "qwen35_embed_row");
    if (!row) return 0;
    qrocm_embedding_q8_0 a{};
    a.row_index = 0u; a.n_embd = n_embd; a.block_size = 32u;
    a.source_row_stride = row_bytes; a.source_block_stride = 34u;
    a.source_scale_offset = 0u; a.source_quant_offset = 2u; a.source_quant_stride = 1u; a.output_dim_stride = 4u;
    const uint32_t nth = qrocm_elem_threads(n_embd);
    qrocm_k_embed_q8_0<<<(unsigned)(((uint64_t)n_embd + nth - 1) / nth), nth>>>(
        a, row, (char *)out->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 embed_q8_0");
}

extern "C" int ds4_gpu_qwen35_dequant_embedding_q8_0_batch_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *token_ids, const void *model_map,
        uint64_t model_size, uint64_t embedding_offset, uint32_t n_token, uint32_t n_row, uint32_t n_embd) {
    if (!out || !token_ids || !model_map || !n_token || !n_row || !n_embd || (n_embd % 32u) != 0u) return 0;
    const uint64_t row_bytes = (uint64_t)(n_embd / 32u) * 34u;
    const uint64_t embedding_bytes = (uint64_t)n_row * row_bytes;
    const uint64_t values = (uint64_t)n_token * n_embd;
    if (values > UINT32_MAX) return 0;
    if (!cuda_model_range_fits(model_size, embedding_offset, embedding_bytes)) return 0;
    if (!cuda_tensor_has_bytes(token_ids, (uint64_t)n_token * 4u) ||
        !cuda_tensor_has_bytes(out, values * 4u)) return 0;
    const char *table = cuda_model_range_ptr(model_map, embedding_offset, embedding_bytes, "qwen35_embed_table");
    if (!table) return 0;
    qrocm_embedding_q8_0_batch a{};
    a.n_token = n_token; a.n_row = n_row; a.n_embd = n_embd; a.block_size = 32u;
    a.source_row_stride = row_bytes; a.source_block_stride = 34u;
    a.source_scale_offset = 0u; a.source_quant_offset = 2u; a.source_quant_stride = 1u;
    a.token_id_stride = 4u; a.output_token_stride = (uint64_t)n_embd * 4u; a.output_dim_stride = 4u;
    const uint32_t nth = qrocm_elem_threads(values);
    qrocm_k_embed_q8_0_batch<<<(unsigned)((values + nth - 1) / nth), nth>>>(
        a, table, (const char *)token_ids->ptr, (char *)out->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 embed_q8_0_batch");
}

extern "C" int ds4_gpu_qwen35_gqa_decode_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *query, const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache, uint32_t n_kv, uint32_t n_query_head,
        uint32_t n_kv_head, uint32_t head_dim) {
    if (!out || !query || !key_cache || !value_cache) return 0;
    if (!n_kv || !n_query_head || !n_kv_head || !head_dim || n_query_head % n_kv_head != 0u) return 0;
    const uint64_t query_bytes = (uint64_t)n_query_head * head_dim * 4u;
    const uint64_t cache_bytes = (uint64_t)n_kv * n_kv_head * head_dim * 4u;
    if (!cuda_tensor_has_bytes(query, query_bytes) || !cuda_tensor_has_bytes(key_cache, cache_bytes) ||
        !cuda_tensor_has_bytes(value_cache, cache_bytes) || !cuda_tensor_has_bytes(out, query_bytes)) return 0;
    qrocm_gqa_decode a{};
    a.n_kv = n_kv; a.n_query_head = n_query_head; a.n_kv_head = n_kv_head; a.head_dim = head_dim;
    a.query_head_stride = (uint64_t)head_dim * 4u; a.query_dim_stride = 4u;
    a.key_token_stride = (uint64_t)n_kv_head * head_dim * 4u; a.key_head_stride = (uint64_t)head_dim * 4u; a.key_dim_stride = 4u;
    a.value_token_stride = (uint64_t)n_kv_head * head_dim * 4u; a.value_head_stride = (uint64_t)head_dim * 4u; a.value_dim_stride = 4u;
    a.output_head_stride = (uint64_t)head_dim * 4u; a.output_dim_stride = 4u;
    const uint32_t preferred = ((head_dim + 31u) / 32u) * 32u;
    uint32_t nsg = 0; const uint32_t nth = qrocm_reduction_threads(preferred, &nsg);
    if (nth < head_dim || nth != preferred) return 0;
    qrocm_k_gqa_decode<<<n_query_head, nth, (nsg + 4u) * sizeof(float)>>>(
        a, (const char *)query->ptr, (const char *)key_cache->ptr,
        (const char *)value_cache->ptr, (char *)out->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 gqa_decode");
}

extern "C" int ds4_gpu_qwen35_gqa_prefill_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *query, const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache, uint32_t position0, uint32_t n_token,
        uint32_t n_query_head, uint32_t n_kv_head, uint32_t head_dim) {
    if (!out || !query || !key_cache || !value_cache) return 0;
    if (!n_token || !n_query_head || !n_kv_head || !head_dim || n_query_head % n_kv_head != 0u) return 0;
    if (position0 > UINT32_MAX - n_token) return 0;
    const uint32_t n_kv = position0 + n_token;
    const uint64_t query_bytes = (uint64_t)n_query_head * head_dim * n_token * 4u;
    const uint64_t cache_bytes = (uint64_t)n_kv_head * head_dim * n_kv * 4u;
    if (!cuda_tensor_has_bytes(query, query_bytes) || !cuda_tensor_has_bytes(key_cache, cache_bytes) ||
        !cuda_tensor_has_bytes(value_cache, cache_bytes) || !cuda_tensor_has_bytes(out, query_bytes)) return 0;
    qrocm_gqa_prefill a{};
    a.position0 = position0; a.n_token = n_token; a.n_query_head = n_query_head;
    a.n_kv_head = n_kv_head; a.head_dim = head_dim;
    a.query_token_stride = (uint64_t)n_query_head * head_dim * 4u; a.query_head_stride = (uint64_t)head_dim * 4u; a.query_dim_stride = 4u;
    a.key_token_stride = (uint64_t)n_kv_head * head_dim * 4u; a.key_head_stride = (uint64_t)head_dim * 4u; a.key_dim_stride = 4u;
    a.value_token_stride = (uint64_t)n_kv_head * head_dim * 4u; a.value_head_stride = (uint64_t)head_dim * 4u; a.value_dim_stride = 4u;
    a.output_token_stride = (uint64_t)n_query_head * head_dim * 4u; a.output_head_stride = (uint64_t)head_dim * 4u; a.output_dim_stride = 4u;
    const uint32_t preferred = ((head_dim + 31u) / 32u) * 32u;
    uint32_t nsg = 0; const uint32_t nth = qrocm_reduction_threads(preferred, &nsg);
    if (nth < head_dim || nth != preferred) return 0;
    dim3 grid(n_query_head, n_token, 1u);
    qrocm_k_gqa_prefill<<<grid, nth, (nsg + 4u) * sizeof(float)>>>(
        a, (const char *)query->ptr, (const char *)key_cache->ptr,
        (const char *)value_cache->ptr, (char *)out->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 gqa_prefill");
}

extern "C" int ds4_gpu_qwen35_router_softmax_top8_batch_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *selected_weight,
        const ds4_gpu_tensor *logits, uint32_t n_token) {
    if (!selected || !selected_weight || !logits || !n_token) return 0;
    const uint64_t logits_bytes = (uint64_t)n_token * 256u * 4u;
    const uint64_t selected_bytes = (uint64_t)n_token * 8u * 4u;
    if (!cuda_tensor_has_bytes(logits, logits_bytes) || !cuda_tensor_has_bytes(selected, selected_bytes) ||
        !cuda_tensor_has_bytes(selected_weight, selected_bytes)) return 0;
    qrocm_router_top8 a{};
    a.n_token = n_token;
    a.logits_token_stride = 1024u; a.logits_stride = 4u;
    a.selected_token_stride = 32u; a.selected_stride = 4u;
    a.selected_weight_token_stride = 32u; a.selected_weight_stride = 4u;
    qrocm_k_router_serial<<<n_token, 256u>>>(
        a, (const char *)logits->ptr, (char *)selected->ptr, (char *)selected_weight->ptr);
    return cuda_ok(cudaGetLastError(), "qwen35 router_top8");
}
extern "C" int ds4_gpu_qwen35_router_softmax_top8_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *selected_weight, const ds4_gpu_tensor *logits) {
    return ds4_gpu_qwen35_router_softmax_top8_batch_tensor(selected, selected_weight, logits, 1u);
}

// Resident Qwen top-8 routed MoE.  The shared ROCm Q4_K MoE core is specialized
// for DeepSeek's top-6 (fused sum6 kernels); Qwen routes top-8.  Run it as two
// independent 4-expert halves and add the partial outputs — the same split the
// Metal resident/streaming paths use.  selected_half0/half1 are offset-0/16
// views into selected_top8, so they already point at experts [0..3] and [4..7].
extern "C" int ds4_gpu_qwen35_routed_moe_top8_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *partial0, ds4_gpu_tensor *partial1,
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *experts,
        const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint64_t down_offset, uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes,
        uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected_top8, const ds4_gpu_tensor *weights_top8,
        const ds4_gpu_tensor *selected_half0, const ds4_gpu_tensor *selected_half1,
        const ds4_gpu_tensor *weights_half0, const ds4_gpu_tensor *weights_half1,
        uint32_t n_total_expert, float clamp, const ds4_gpu_tensor *x,
        uint32_t layer_index, bool trusted_gpu_route) {
    (void)selected_top8; (void)weights_top8; (void)trusted_gpu_route;
    if (!partial0 || !partial1 || !selected_half0 || !selected_half1 ||
        !weights_half0 || !weights_half1) return 0;
    int ok = ds4_gpu_routed_moe_one_tensor(
        partial0, gate, up, mid, experts, model_map, model_size, gate_offset, up_offset, down_offset,
        gate_type, down_type, gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
        expert_in_dim, expert_mid_dim, out_dim, selected_half0, weights_half0,
        n_total_expert, 4u, clamp, x, layer_index);
    if (!ok) return 0;
    ok = ds4_gpu_routed_moe_one_tensor(
        partial1, gate, up, mid, experts, model_map, model_size, gate_offset, up_offset, down_offset,
        gate_type, down_type, gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
        expert_in_dim, expert_mid_dim, out_dim, selected_half1, weights_half1,
        n_total_expert, 4u, clamp, x, layer_index);
    if (!ok) return 0;
    return ds4_gpu_add_tensor(out, partial0, partial1, out_dim);
}

// Profiling counters (stubs — resident route always takes the GPU path).
static uint64_t g_qwen35_resident_gpu_route_calls;
static uint64_t g_qwen35_gdn128_parallel_calls;
extern "C" void ds4_gpu_internal_qwen35_resident_route_stats_reset(void) { g_qwen35_resident_gpu_route_calls = 0; }
extern "C" void ds4_gpu_internal_qwen35_resident_route_stats_add(uint64_t calls) { g_qwen35_resident_gpu_route_calls += calls; }
extern "C" uint64_t ds4_gpu_internal_qwen35_resident_gpu_route_calls(void) { return g_qwen35_resident_gpu_route_calls; }
extern "C" uint64_t ds4_gpu_internal_qwen35_resident_host_readbacks(void) { return 0; }
extern "C" void ds4_gpu_internal_qwen35_gdn128_stats_reset(void) { g_qwen35_gdn128_parallel_calls = 0; }
extern "C" uint64_t ds4_gpu_internal_qwen35_gdn128_parallel_calls(void) { return g_qwen35_gdn128_parallel_calls; }

#endif // DS4_ROCM_QWEN_CUH
