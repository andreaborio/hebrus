// Per-op parity harness: ROCm Qwen3.6 kernels vs the scalar reference
// (ds4_qwen_ref.c).  Random inputs, host reference, device kernel, compare.
// Covers the ops that do not read model-map weights (the highest-risk GDN and
// GQA kernels among them).  Weight-based ops (conv, rmsnorm_gated, controls,
// embedding) are validated separately once a model map is registered.

#include "ds4_gpu.h"
#include "ds4_qwen_ref.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g_seed = 0x1234567u;
static float frand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return ((float)(g_seed >> 8) / (float)(1u << 24)) * 2.0f - 1.0f; // [-1,1)
}

static int compare(const char *name, const float *got, const float *want,
                   size_t n, float tol) {
    double worst = 0.0; size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        const float scale = fmaxf(1.0f, fabsf(want[i]));
        const double e = fabs((double)got[i] - (double)want[i]) / scale;
        if (!isfinite(got[i])) { fprintf(stderr, "%s[%zu] not finite\n", name, i); return 1; }
        if (e > worst) { worst = e; at = i; }
    }
    if (worst > tol) {
        fprintf(stderr, "FAIL %s: worst rel %.3g at %zu (got %.9g want %.9g)\n",
                name, worst, at, (double)got[at], (double)want[at]);
        return 1;
    }
    fprintf(stderr, "ok   %s (worst rel %.3g over %zu)\n", name, worst, n);
    return 0;
}

// Helper: alloc a device tensor, upload host floats.
static ds4_gpu_tensor *up_f32(const float *h, size_t n) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    if (!t) return NULL;
    ds4_gpu_tensor_write(t, 0, h, n * sizeof(float));
    return t;
}
static void down_f32(const ds4_gpu_tensor *t, float *h, size_t n) {
    ds4_gpu_tensor_read(t, 0, h, n * sizeof(float));
}

static int fails = 0;

static void test_split_q_gate(void) {
    enum { NT = 3, NH = 4, HD = 8 };
    float proj[NT*NH*2*HD], q_ref[NT*NH*HD], g_ref[NT*NH*HD];
    for (int i = 0; i < NT*NH*2*HD; i++) proj[i] = frand();
    ds4_qwen_ref_split_q_gate_f32(q_ref, g_ref, proj, NT, NH, HD);
    ds4_gpu_tensor *dp = up_f32(proj, NT*NH*2*HD);
    ds4_gpu_tensor *dq = ds4_gpu_tensor_alloc(NT*NH*HD*sizeof(float));
    ds4_gpu_tensor *dg = ds4_gpu_tensor_alloc(NT*NH*HD*sizeof(float));
    if (!ds4_gpu_qwen35_split_q_gate_batch_tensor(dq, dg, dp, NT, NH, HD)) { fprintf(stderr,"split dispatch failed\n"); fails++; return; }
    ds4_gpu_synchronize();
    float q_got[NT*NH*HD], g_got[NT*NH*HD];
    down_f32(dq, q_got, NT*NH*HD); down_f32(dg, g_got, NT*NH*HD);
    fails += compare("split_q_gate.query", q_got, q_ref, NT*NH*HD, 1e-6f);
    fails += compare("split_q_gate.gate", g_got, g_ref, NT*NH*HD, 1e-6f);
    ds4_gpu_tensor_free(dp); ds4_gpu_tensor_free(dq); ds4_gpu_tensor_free(dg);
}

static void test_sigmoid_mul_elements(void) {
    enum { N = 257 };
    float in[N], gate[N], ref[N];
    for (int i = 0; i < N; i++) { in[i] = frand(); gate[i] = frand()*4.0f; }
    ds4_qwen_ref_sigmoid_gate_elements_f32(ref, in, gate, N);
    ds4_gpu_tensor *di = up_f32(in, N), *dg = up_f32(gate, N);
    ds4_gpu_tensor *dout = ds4_gpu_tensor_alloc(N*sizeof(float));
    if (!ds4_gpu_qwen35_sigmoid_mul_tensor(dout, di, dg, N, false)) { fprintf(stderr,"sigmoid_mul dispatch failed\n"); fails++; return; }
    ds4_gpu_synchronize();
    float got[N]; down_f32(dout, got, N);
    fails += compare("sigmoid_mul.elements", got, ref, N, 2e-6f);
    ds4_gpu_tensor_free(di); ds4_gpu_tensor_free(dg); ds4_gpu_tensor_free(dout);
}

static void test_sigmoid_mul_rows(void) {
    enum { R = 5, W = 40 };
    float in[R*W], gate[R], ref[R*W];
    for (int i = 0; i < R*W; i++) in[i] = frand();
    for (int i = 0; i < R; i++) gate[i] = frand()*4.0f;
    // reference: broadcast scalar gate over each row
    for (int r = 0; r < R; r++) {
        const float g = 1.0f / (1.0f + expf(-gate[r]));
        for (int c = 0; c < W; c++) ref[r*W+c] = in[r*W+c] * g;
    }
    ds4_gpu_tensor *di = up_f32(in, R*W), *dg = up_f32(gate, R);
    ds4_gpu_tensor *dout = ds4_gpu_tensor_alloc(R*W*sizeof(float));
    if (!ds4_gpu_qwen35_sigmoid_mul_rows_tensor(dout, di, dg, R, W)) { fprintf(stderr,"sigmoid_rows dispatch failed\n"); fails++; return; }
    ds4_gpu_synchronize();
    float got[R*W]; down_f32(dout, got, R*W);
    fails += compare("sigmoid_mul.rows", got, ref, R*W, 2e-6f);
    ds4_gpu_tensor_free(di); ds4_gpu_tensor_free(dg); ds4_gpu_tensor_free(dout);
}

static void test_rope(void) {
    enum { NT = 2, NH = 3, HD = 16, NROT = 8 };
    float vals[NT*NH*HD], ref[NT*NH*HD];
    uint32_t pos[NT] = { 5, 37 };
    for (int i = 0; i < NT*NH*HD; i++) { vals[i] = frand(); ref[i] = vals[i]; }
    ds4_qwen_ref_text_rope_f32(ref, pos, NT, NH, HD, NROT, 1000000.0f);
    ds4_gpu_tensor *dv = up_f32(vals, NT*NH*HD);
    ds4_gpu_tensor *dpos = ds4_gpu_tensor_alloc(NT*sizeof(uint32_t));
    ds4_gpu_tensor_write(dpos, 0, pos, NT*sizeof(uint32_t));
    if (!ds4_gpu_qwen35_rope_prefix_batch_tensor(dv, dpos, NT, NH, HD, NROT, 1000000.0f)) { fprintf(stderr,"rope dispatch failed\n"); fails++; return; }
    ds4_gpu_synchronize();
    float got[NT*NH*HD]; down_f32(dv, got, NT*NH*HD);
    // 5e-6: RoPE differs only in transcendental rounding (powf/cosf/sinf vs the
    // reference libm), never in structure — the same tolerance the GQA online
    // softmax uses for its exp() chain.
    fails += compare("rope.batch", got, ref, NT*NH*HD, 5e-6f);
    ds4_gpu_tensor_free(dv); ds4_gpu_tensor_free(dpos);
}

static void test_gqa_prefill(void) {
    // Reference GQA is over a whole sequence; prefill kernel writes per-token
    // causal outputs into the same [token][head][dim] layout.
    enum { NT = 6, QH = 8, KVH = 2, HD = 16 };
    float q[NT*QH*HD], k[NT*KVH*HD], v[NT*KVH*HD], ref[NT*QH*HD];
    for (int i = 0; i < NT*QH*HD; i++) q[i] = frand();
    for (int i = 0; i < NT*KVH*HD; i++) { k[i] = frand(); v[i] = frand(); }
    ds4_qwen_ref_causal_gqa_f32(ref, q, k, v, NT, QH, KVH, HD);
    ds4_gpu_tensor *dq = up_f32(q, NT*QH*HD), *dk = up_f32(k, NT*KVH*HD), *dv = up_f32(v, NT*KVH*HD);
    ds4_gpu_tensor *dout = ds4_gpu_tensor_alloc(NT*QH*HD*sizeof(float));
    if (!ds4_gpu_qwen35_gqa_prefill_tensor(dout, dq, dk, dv, 0, NT, QH, KVH, HD)) { fprintf(stderr,"gqa_prefill dispatch failed\n"); fails++; return; }
    ds4_gpu_synchronize();
    float got[NT*QH*HD]; down_f32(dout, got, NT*QH*HD);
    fails += compare("gqa.prefill", got, ref, NT*QH*HD, 5e-6f);
    ds4_gpu_tensor_free(dq); ds4_gpu_tensor_free(dk); ds4_gpu_tensor_free(dv); ds4_gpu_tensor_free(dout);
}

static void test_gqa_decode(void) {
    // Decode = the last query row of a causal sequence vs the full K/V cache.
    enum { NKV = 20, QH = 8, KVH = 2, HD = 16 };
    float q[QH*HD], k[NKV*KVH*HD], v[NKV*KVH*HD], ref[QH*HD];
    for (int i = 0; i < QH*HD; i++) q[i] = frand();
    for (int i = 0; i < NKV*KVH*HD; i++) { k[i] = frand(); v[i] = frand(); }
    // reference: single-query causal attention over NKV cached tokens
    const float scale = 1.0f / sqrtf((float)HD);
    const int qpk = QH / KVH;
    for (int h = 0; h < QH; h++) {
        const int kvh = h / qpk;
        float mx = -INFINITY;
        for (int t = 0; t < NKV; t++) {
            float d = 0; for (int c = 0; c < HD; c++) d += q[h*HD+c]*k[(t*KVH+kvh)*HD+c];
            d *= scale; if (d > mx) mx = d;
        }
        float sum = 0, acc[HD]; for (int c=0;c<HD;c++) acc[c]=0;
        for (int t = 0; t < NKV; t++) {
            float d = 0; for (int c = 0; c < HD; c++) d += q[h*HD+c]*k[(t*KVH+kvh)*HD+c];
            const float w = expf(d*scale - mx); sum += w;
            for (int c = 0; c < HD; c++) acc[c] += w * v[(t*KVH+kvh)*HD+c];
        }
        for (int c = 0; c < HD; c++) ref[h*HD+c] = acc[c] / sum;
    }
    ds4_gpu_tensor *dq = up_f32(q, QH*HD), *dk = up_f32(k, NKV*KVH*HD), *dv = up_f32(v, NKV*KVH*HD);
    ds4_gpu_tensor *dout = ds4_gpu_tensor_alloc(QH*HD*sizeof(float));
    if (!ds4_gpu_qwen35_gqa_decode_tensor(dout, dq, dk, dv, NKV, QH, KVH, HD)) { fprintf(stderr,"gqa_decode dispatch failed\n"); fails++; return; }
    ds4_gpu_synchronize();
    float got[QH*HD]; down_f32(dout, got, QH*HD);
    fails += compare("gqa.decode", got, ref, QH*HD, 5e-6f);
    ds4_gpu_tensor_free(dq); ds4_gpu_tensor_free(dk); ds4_gpu_tensor_free(dv); ds4_gpu_tensor_free(dout);
}

static void test_router(void) {
    enum { NE = 256, NS = 8 };
    float logits[NE]; int32_t sel_ref[NS]; float w_ref[NS];
    for (int i = 0; i < NE; i++) logits[i] = frand()*6.0f;
    ds4_qwen_ref_softmax_topk_f32(sel_ref, w_ref, logits, NE, NS);
    ds4_gpu_tensor *dl = up_f32(logits, NE);
    ds4_gpu_tensor *dsel = ds4_gpu_tensor_alloc(NS*sizeof(int32_t));
    ds4_gpu_tensor *dw = ds4_gpu_tensor_alloc(NS*sizeof(float));
    if (!ds4_gpu_qwen35_router_softmax_top8_tensor(dsel, dw, dl)) { fprintf(stderr,"router dispatch failed\n"); fails++; return; }
    ds4_gpu_synchronize();
    int32_t sel_got[NS]; float w_got[NS];
    ds4_gpu_tensor_read(dsel, 0, sel_got, NS*sizeof(int32_t));
    ds4_gpu_tensor_read(dw, 0, w_got, NS*sizeof(float));
    int bad = 0;
    for (int i = 0; i < NS; i++) if (sel_got[i] != sel_ref[i]) {
        fprintf(stderr, "FAIL router.selected[%d] got %d want %d\n", i, sel_got[i], sel_ref[i]); bad = 1;
    }
    fails += bad;
    if (!bad) { fails += compare("router.weights", w_got, w_ref, NS, 2e-6f); fprintf(stderr,"ok   router.selected (exact top-8 match)\n"); }
    ds4_gpu_tensor_free(dl); ds4_gpu_tensor_free(dsel); ds4_gpu_tensor_free(dw);
}

static void test_gd_step(int key_dim) {
    // One decode token; reference over n_token=1.  V-head tiling: value head h
    // uses key head h % n_key_head.  Dims: SSM_VALUE_HEAD=32, group=16.
    const int nvh = 32, nkh = 16, vd = key_dim;
    const int kv = nkh*key_dim, vv = nvh*vd, sv = nvh*vd*key_dim;
    float *q = malloc(kv*4), *k = malloc(kv*4), *v = malloc(vv*4);
    float *ld = malloc(nvh*4), *bt = malloc(nvh*4);
    float *st_ref = calloc(sv,4), *st_got0 = calloc(sv,4);
    float *out_ref = calloc(vv,4), *out_got = calloc(vv,4);
    for (int i=0;i<kv;i++){q[i]=frand();k[i]=frand();}
    for (int i=0;i<vv;i++) v[i]=frand();
    for (int i=0;i<nvh;i++){ ld[i]=-fabsf(frand())*0.3f; bt[i]=0.3f+0.4f*(frand()*0.5f+0.5f);}
    // reference (n_token=1). q/k laid [token][key_head][key_dim]; v/out [token][value_head][value_dim]
    ds4_qwen_ref_gated_delta_rule_f32(out_ref, st_ref, q, k, v, ld, bt, 1, nkh, nvh, key_dim, vd);
    ds4_gpu_tensor *dq=up_f32(q,kv),*dk=up_f32(k,kv),*dv=up_f32(v,vv);
    ds4_gpu_tensor *dld=up_f32(ld,nvh),*dbt=up_f32(bt,nvh);
    ds4_gpu_tensor *dst=up_f32(st_got0,sv);
    ds4_gpu_tensor *dout=ds4_gpu_tensor_alloc(vv*4);
    char name[64]; snprintf(name,sizeof(name),"gd_step(kd=%d)",key_dim);
    if (!ds4_gpu_qwen35_gated_delta_step_tensor(dout,dst,dq,dk,dv,dld,dbt,nkh,nvh,key_dim,vd)) { fprintf(stderr,"%s dispatch failed\n",name); fails++; goto done; }
    ds4_gpu_synchronize();
    down_f32(dout,out_got,vv); down_f32(dst,st_got0,sv);
    { char n2[80]; snprintf(n2,sizeof(n2),"%s.out",name); fails+=compare(n2,out_got,out_ref,vv,5e-5f);
      snprintf(n2,sizeof(n2),"%s.state",name); fails+=compare(n2,st_got0,st_ref,sv,5e-5f); }
done:
    ds4_gpu_tensor_free(dq);ds4_gpu_tensor_free(dk);ds4_gpu_tensor_free(dv);
    ds4_gpu_tensor_free(dld);ds4_gpu_tensor_free(dbt);ds4_gpu_tensor_free(dst);ds4_gpu_tensor_free(dout);
    free(q);free(k);free(v);free(ld);free(bt);free(st_ref);free(st_got0);free(out_ref);free(out_got);
}

static void test_gd_sequence_128(void) {
    // Prefill form: packed [Q|K|V] projection per token, key_dim=value_dim=128.
    const int nt = 5, nvh = 32, nkh = 16, kd = 128, vd = 128;
    const int qv = nkh*kd, vv = nvh*vd, prow = qv+qv+vv;
    const int sv = nvh*vd*kd;
    float *proj = malloc((size_t)prow*nt*4);
    float *ld = malloc((size_t)nvh*nt*4), *bt = malloc((size_t)nvh*nt*4);
    float *st_ref = calloc(sv,4), *st_got = calloc(sv,4);
    float *out_ref = calloc((size_t)vv*nt,4), *out_got = calloc((size_t)vv*nt,4);
    for (int i=0;i<prow*nt;i++) proj[i]=frand();
    for (int i=0;i<nvh*nt;i++){ ld[i]=-fabsf(frand())*0.3f; bt[i]=0.3f+0.4f*(frand()*0.5f+0.5f);}
    // Build unpacked q/k/v [token][head][dim] for the reference from the packed rows.
    float *q = malloc((size_t)qv*nt*4), *k = malloc((size_t)qv*nt*4), *v = malloc((size_t)vv*nt*4);
    for (int t=0;t<nt;t++){
        memcpy(q+(size_t)t*qv, proj+(size_t)t*prow, qv*4);
        memcpy(k+(size_t)t*qv, proj+(size_t)t*prow+qv, qv*4);
        memcpy(v+(size_t)t*vv, proj+(size_t)t*prow+2*qv, vv*4);
    }
    ds4_qwen_ref_gated_delta_rule_f32(out_ref, st_ref, q, k, v, ld, bt, nt, nkh, nvh, kd, vd);
    ds4_gpu_tensor *dp=up_f32(proj,(size_t)prow*nt);
    ds4_gpu_tensor *dld=up_f32(ld,(size_t)nvh*nt),*dbt=up_f32(bt,(size_t)nvh*nt);
    ds4_gpu_tensor *dst=up_f32(st_got,sv);
    ds4_gpu_tensor *dout=ds4_gpu_tensor_alloc((size_t)vv*nt*4);
    if (!ds4_gpu_qwen35_gated_delta_sequence_128_tensor(dout,dst,dp,dld,dbt,nt,nkh,nvh)) { fprintf(stderr,"gd_sequence_128 dispatch failed\n"); fails++; goto done; }
    ds4_gpu_synchronize();
    down_f32(dout,out_got,(size_t)vv*nt); down_f32(dst,st_got,sv);
    fails+=compare("gd_sequence_128.out",out_got,out_ref,(size_t)vv*nt,5e-5f);
    fails+=compare("gd_sequence_128.state",st_got,st_ref,sv,5e-5f);
done:
    ds4_gpu_tensor_free(dp);ds4_gpu_tensor_free(dld);ds4_gpu_tensor_free(dbt);ds4_gpu_tensor_free(dst);ds4_gpu_tensor_free(dout);
    free(proj);free(ld);free(bt);free(st_ref);free(st_got);free(out_ref);free(out_got);free(q);free(k);free(v);
}

// Weight-based ops: the weight lives in the "model map".  Register a host
// buffer as the map (UMA maps it device-side) and pass offset 0.
static void test_conv_step(void) {
    enum { NC = 200, K = 4 };
    float in[NC], w[NC*K], st_ref[NC*(K-1)], st_got[NC*(K-1)], out_ref[NC], out_got[NC];
    for (int i=0;i<NC;i++) in[i]=frand();
    for (int i=0;i<NC*K;i++) w[i]=frand();
    for (int i=0;i<NC*(K-1);i++) { st_ref[i]=frand(); st_got[i]=st_ref[i]; }
    ds4_qwen_ref_causal_conv1d_silu_f32(out_ref, st_ref, in, w, 1, NC, K);
    ds4_gpu_set_model_map(w, sizeof(w));
    ds4_gpu_tensor *di=up_f32(in,NC), *dst=up_f32(st_got,NC*(K-1)), *dout=ds4_gpu_tensor_alloc(NC*4);
    if (!ds4_gpu_qwen35_causal_conv_step_tensor(dout,dst,di,w,sizeof(w),0,NC,K)) { fprintf(stderr,"conv_step dispatch failed\n"); fails++; return; }
    ds4_gpu_synchronize();
    down_f32(dout,out_got,NC); down_f32(dst,st_got,NC*(K-1));
    fails+=compare("conv_step.out",out_got,out_ref,NC,2e-6f);
    fails+=compare("conv_step.state",st_got,st_ref,NC*(K-1),2e-6f);
    ds4_gpu_tensor_free(di);ds4_gpu_tensor_free(dst);ds4_gpu_tensor_free(dout);
}

static void test_conv_sequence(void) {
    enum { NT = 7, NC = 128, K = 4 };
    float in[NT*NC], w[NC*K], st_ref[NC*(K-1)], st_got[NC*(K-1)], out_ref[NT*NC], out_got[NT*NC];
    for (int i=0;i<NT*NC;i++) in[i]=frand();
    for (int i=0;i<NC*K;i++) w[i]=frand();
    for (int i=0;i<NC*(K-1);i++) { st_ref[i]=frand(); st_got[i]=st_ref[i]; }
    ds4_qwen_ref_causal_conv1d_silu_f32(out_ref, st_ref, in, w, NT, NC, K);
    ds4_gpu_set_model_map(w, sizeof(w));
    ds4_gpu_tensor *di=up_f32(in,NT*NC), *dst=up_f32(st_got,NC*(K-1)), *dout=ds4_gpu_tensor_alloc(NT*NC*4);
    if (!ds4_gpu_qwen35_causal_conv_sequence_tensor(dout,dst,di,w,sizeof(w),0,NT,NC,K)) { fprintf(stderr,"conv_seq dispatch failed\n"); fails++; return; }
    ds4_gpu_synchronize();
    down_f32(dout,out_got,NT*NC); down_f32(dst,st_got,NC*(K-1));
    fails+=compare("conv_sequence.out",out_got,out_ref,NT*NC,2e-6f);
    fails+=compare("conv_sequence.state",st_got,st_ref,NC*(K-1),2e-6f);
    ds4_gpu_tensor_free(di);ds4_gpu_tensor_free(dst);ds4_gpu_tensor_free(dout);
}

static void test_rmsnorm_gated(void) {
    enum { NV = 6, D = 200 };
    float in[NV*D], gate[NV*D], w[D], ref[NV*D], got[NV*D];
    for (int i=0;i<NV*D;i++){ in[i]=frand(); gate[i]=frand()*3.0f; }
    for (int i=0;i<D;i++) w[i]=frand()*0.5f+1.0f;
    ds4_qwen_ref_rmsnorm_gated_f32(ref, in, gate, w, NV, D, 1e-6f);
    ds4_gpu_set_model_map(w, sizeof(w));
    ds4_gpu_tensor *di=up_f32(in,NV*D), *dg=up_f32(gate,NV*D), *dout=ds4_gpu_tensor_alloc(NV*D*4);
    if (!ds4_gpu_qwen35_rmsnorm_gated_tensor(dout,di,dg,w,sizeof(w),0,NV,D,1e-6f)) { fprintf(stderr,"rmsnorm_gated dispatch failed\n"); fails++; return; }
    ds4_gpu_synchronize();
    down_f32(dout,got,NV*D);
    fails+=compare("rmsnorm_gated",got,ref,NV*D,5e-6f);
    ds4_gpu_tensor_free(di);ds4_gpu_tensor_free(dg);ds4_gpu_tensor_free(dout);
}

static void test_gd_controls(void) {
    enum { NT = 4, NVH = 32 };
    float alpha[NT*NVH], betal[NT*NVH], ssm_a[NVH], dt_bias[NVH];
    float ld_ref[NT*NVH], bt_ref[NT*NVH], ld_got[NT*NVH], bt_got[NT*NVH];
    for (int i=0;i<NT*NVH;i++){ alpha[i]=frand()*2.0f; betal[i]=frand()*2.0f; }
    for (int i=0;i<NVH;i++){ ssm_a[i]=-fabsf(frand())-0.1f; dt_bias[i]=frand()*0.5f; }
    ds4_qwen_ref_gated_delta_controls_f32(ld_ref, bt_ref, alpha, betal, ssm_a, dt_bias, NT, NVH);
    // Pack ssm_a and dt_bias into one model map.
    float wbuf[2*NVH]; memcpy(wbuf, ssm_a, sizeof(ssm_a)); memcpy(wbuf+NVH, dt_bias, sizeof(dt_bias));
    ds4_gpu_set_model_map(wbuf, sizeof(wbuf));
    ds4_gpu_tensor *da=up_f32(alpha,NT*NVH), *db=up_f32(betal,NT*NVH);
    ds4_gpu_tensor *dld=ds4_gpu_tensor_alloc(NT*NVH*4), *dbt=ds4_gpu_tensor_alloc(NT*NVH*4);
    if (!ds4_gpu_qwen35_gated_delta_controls_batch_tensor(dld,dbt,da,db,wbuf,sizeof(wbuf),0,(uint64_t)NVH*4,NT,NVH)) { fprintf(stderr,"gd_controls dispatch failed\n"); fails++; return; }
    ds4_gpu_synchronize();
    down_f32(dld,ld_got,NT*NVH); down_f32(dbt,bt_got,NT*NVH);
    fails+=compare("gd_controls.log_decay",ld_got,ld_ref,NT*NVH,5e-6f);
    fails+=compare("gd_controls.beta",bt_got,bt_ref,NT*NVH,2e-6f);
    ds4_gpu_tensor_free(da);ds4_gpu_tensor_free(db);ds4_gpu_tensor_free(dld);ds4_gpu_tensor_free(dbt);
}

static void test_embedding_q8_0(void) {
    enum { NROW = 4, NEMBD = 64, NBLK = NEMBD/32 };
    // Build a Q8_0 table: per 32-elem block = 2-byte f16 scale (=1.0) + 32 int8.
    const uint16_t f16_one = 0x3C00u; // 1.0 in half
    const size_t row_bytes = NBLK*34;
    unsigned char table[NROW*NBLK*34];
    float ref[NEMBD];
    int8_t qv[NROW][NEMBD];
    for (int r=0;r<NROW;r++)
        for (int i=0;i<NEMBD;i++) qv[r][i] = (int8_t)((int)(frand()*100.0f));
    for (int r=0;r<NROW;r++)
        for (int b=0;b<NBLK;b++) {
            unsigned char *blk = table + (r*NBLK + b)*34;
            memcpy(blk, &f16_one, 2);
            for (int i=0;i<32;i++) blk[2+i] = (unsigned char)qv[r][b*32+i];
        }
    const int row = 2;
    for (int i=0;i<NEMBD;i++) ref[i] = 1.0f * (float)qv[row][i];
    ds4_gpu_set_model_map(table, sizeof(table));
    ds4_gpu_tensor *dout=ds4_gpu_tensor_alloc(NEMBD*4);
    if (!ds4_gpu_qwen35_dequant_embedding_q8_0_tensor(dout,table,sizeof(table),0,row,NEMBD)) { fprintf(stderr,"embed dispatch failed\n"); fails++; return; }
    ds4_gpu_synchronize();
    float got[NEMBD]; down_f32(dout,got,NEMBD);
    fails+=compare("embedding_q8_0",got,ref,NEMBD,2e-3f); // f16 scale exact here; int8->float exact
    ds4_gpu_tensor_free(dout);
    (void)row_bytes;
}

int main(void) {
    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 2; }
    test_split_q_gate();
    test_sigmoid_mul_elements();
    test_sigmoid_mul_rows();
    test_rope();
    test_router();
    test_gqa_prefill();
    test_gqa_decode();
    test_gd_step(128);   // step_128 fast path
    test_gd_step(64);    // generic reduction path
    test_gd_sequence_128();
    test_conv_step();
    test_conv_sequence();
    test_rmsnorm_gated();
    test_gd_controls();
    test_embedding_q8_0();
    ds4_gpu_cleanup();
    if (fails) { fprintf(stderr, "\n%d qwen35 ROCm parity check(s) FAILED\n", fails); return 1; }
    fprintf(stderr, "\nAll qwen35 ROCm parity checks passed\n");
    return 0;
}
