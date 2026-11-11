// Batch-1 Whisper decode kernels for sm_90. Every launch uses the current stream so the
// whole decode step can be captured into a CUDA graph.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#include <optional>
#include <torch/extension.h>

namespace {

constexpr int kHidden = 1280;  // whisper-large-v3-turbo d_model
constexpr int kHeadDim = 64;
constexpr int kSplitKeys = 64;  // keys per attention block

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// Orders (logit, index) so a larger u64 means a larger logit, ties going to the lower index,
// matching torch.argmax. Zero is reserved for "no candidate".
__device__ __forceinline__ unsigned long long pack_score(float v, int idx) {
  unsigned int u = __float_as_uint(v);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return (static_cast<unsigned long long>(u) << 32) | (0xFFFFFFFFu - static_cast<unsigned int>(idx));
}

// Fused LM head GEMV + argmax. Each warp streams whole weight rows with 128-bit loads, so the
// [vocab] logits vector is never written to memory. One atomicMax per block.
__global__ void __launch_bounds__(256) lm_head_argmax_kernel(
    const half* __restrict__ h, const half* __restrict__ w, const bool* __restrict__ suppress,
    int vocab, unsigned long long* __restrict__ best) {
  constexpr int kChunks = kHidden / 8 / 32;
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int num_warps = (gridDim.x * blockDim.x) >> 5;

  float hf[kChunks * 8];
  const uint4* h4 = reinterpret_cast<const uint4*>(h);
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    uint4 v = h4[lane + 32 * c];
    const half2* p = reinterpret_cast<const half2*>(&v);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float2 f = __half22float2(p[k]);
      hf[c * 8 + 2 * k] = f.x;
      hf[c * 8 + 2 * k + 1] = f.y;
    }
  }

  unsigned long long mine = 0;
  for (int row = warp; row < vocab; row += num_warps) {
    const uint4* w4 = reinterpret_cast<const uint4*>(w + static_cast<size_t>(row) * kHidden);
    uint4 wv[kChunks];
#pragma unroll
    for (int c = 0; c < kChunks; ++c) wv[c] = __ldcs(&w4[lane + 32 * c]);
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      const half2* p = reinterpret_cast<const half2*>(&wv[c]);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 f = __half22float2(p[k]);
        acc = fmaf(f.x, hf[c * 8 + 2 * k], acc);
        acc = fmaf(f.y, hf[c * 8 + 2 * k + 1], acc);
      }
    }
    acc = warp_sum(acc);
    // Round to fp16 like the reference fp16 logits so near-ties resolve the same way.
    acc = __half2float(__float2half(acc));
    if (lane == 0 && !suppress[row]) {
      unsigned long long s = pack_score(acc, row);
      mine = s > mine ? s : mine;
    }
  }

  __shared__ unsigned long long block_best[8];
  if (lane == 0) block_best[threadIdx.x >> 5] = mine;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long m = 0;
    for (int i = 0; i < (blockDim.x >> 5); ++i) m = block_best[i] > m ? block_best[i] : m;
    if (m) atomicMax(best, m);
  }
}

// Decodes the winning token, appends it, feeds it back as the next input, advances the
// position and resets the argmax slot. state = [position, generated count].
__global__ void argmax_finalize_kernel(
    unsigned long long* best, int64_t* tokens, int64_t* state, int64_t* cur, int max_pos) {
  const unsigned long long b = *best;
  *best = 0;
  const int64_t tok = static_cast<int64_t>(0xFFFFFFFFu - static_cast<unsigned int>(b & 0xFFFFFFFFull));
  const int64_t n = state[1];
  if (n < max_pos) tokens[n] = tok;
  state[1] = n < max_pos ? n + 1 : n;
  state[0] = state[0] + 1 < max_pos ? state[0] + 1 : max_pos - 1;
  *cur = tok;
}

// Split-K single-query attention (flash-decoding). Block (head, split) covers 64 keys and writes
// its local max, exp-sum and unnormalized value sum in fp32; the combine kernel merges the splits.
// Self attention reads its length from device memory (state[0] + 1) so the captured graph stays
// valid as the cache grows, and fuses the cache write: key pos comes from k_new/v_new, and the one
// block that owns pos stores the new rows into the cache. Cross attention passes len_state ==
// nullptr and attends over the full encoder length.
__global__ void __launch_bounds__(128) decode_attention_split_kernel(
    const half* __restrict__ q, half* __restrict__ k, half* __restrict__ v, const half* __restrict__ k_new,
    const half* __restrict__ v_new, const int64_t* __restrict__ len_state, int max_len, float scale,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc) {
  __shared__ float scores[kSplitKeys];
  __shared__ float partial[4][kHeadDim];
  __shared__ float red[4];

  const int head = blockIdx.x;
  const int split = blockIdx.y;
  const int slot = head * gridDim.y + split;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int pos = len_state ? static_cast<int>(len_state[0]) : -1;
  const int len = len_state ? pos + 1 : max_len;
  const int begin = split * kSplitKeys;
  const int n = min(kSplitKeys, len - begin);
  if (n <= 0) {
    if (threadIdx.x == 0) {
      part_m[slot] = -INFINITY;
      part_l[slot] = 0.f;
    }
    return;
  }

  half* kh = k + static_cast<size_t>(head) * max_len * kHeadDim;
  half* vh = v + static_cast<size_t>(head) * max_len * kHeadDim;
  const half2* kn = k_new ? reinterpret_cast<const half2*>(k_new + head * kHeadDim) : nullptr;
  const half2* vn = v_new ? reinterpret_cast<const half2*>(v_new + head * kHeadDim) : nullptr;
  if (kn && warp == 0 && pos / kSplitKeys == split) {
    reinterpret_cast<half2*>(kh + pos * kHeadDim)[lane] = kn[lane];
    reinterpret_cast<half2*>(vh + pos * kHeadDim)[lane] = vn[lane];
  }
  const float2 qf = __half22float2(reinterpret_cast<const half2*>(q + head * kHeadDim)[lane]);

  for (int i = warp; i < n; i += 4) {
    const int j = begin + i;
    const half2* kp = (kn && j == pos) ? kn : reinterpret_cast<const half2*>(kh + j * kHeadDim);
    float2 kf = __half22float2(kp[lane]);
    float s = warp_sum(qf.x * kf.x + qf.y * kf.y) * scale;
    if (lane == 0) scores[i] = s;
  }
  __syncthreads();

  float s = threadIdx.x < n ? scores[threadIdx.x] : -INFINITY;
  float wmax = s;
  for (int o = 16; o > 0; o >>= 1) wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffff, wmax, o));
  if (lane == 0) red[warp] = wmax;
  __syncthreads();
  const float m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  __syncthreads();

  float e = 0.f;
  if (threadIdx.x < n) {
    e = __expf(s - m);
    scores[threadIdx.x] = e;
  }
  e = warp_sum(e);
  if (lane == 0) red[warp] = e;
  __syncthreads();

  float2 acc = make_float2(0.f, 0.f);
  for (int i = warp; i < n; i += 4) {
    const int j = begin + i;
    const half2* vp = (vn && j == pos) ? vn : reinterpret_cast<const half2*>(vh + j * kHeadDim);
    float2 vf = __half22float2(vp[lane]);
    float p = scores[i];
    acc.x = fmaf(p, vf.x, acc.x);
    acc.y = fmaf(p, vf.y, acc.y);
  }
  partial[warp][2 * lane] = acc.x;
  partial[warp][2 * lane + 1] = acc.y;
  __syncthreads();
  if (threadIdx.x < kHeadDim) {
    part_acc[static_cast<size_t>(slot) * kHeadDim + threadIdx.x] =
        partial[0][threadIdx.x] + partial[1][threadIdx.x] + partial[2][threadIdx.x] + partial[3][threadIdx.x];
  }
  if (threadIdx.x == 0) {
    part_m[slot] = m;
    part_l[slot] = red[0] + red[1] + red[2] + red[3];
  }
}

// Merges the split results of one head. Splits with l == 0 are empty and are skipped, so their
// -inf max and unwritten accumulator never reach the sum.
__global__ void __launch_bounds__(64) decode_attention_combine_kernel(
    const float* __restrict__ part_m, const float* __restrict__ part_l, const float* __restrict__ part_acc,
    half* __restrict__ out, int splits) {
  const int head = blockIdx.x;
  const int d = threadIdx.x;
  const float* pm = part_m + head * splits;
  const float* pl = part_l + head * splits;
  const float* pa = part_acc + static_cast<size_t>(head) * splits * kHeadDim;

  float m = -INFINITY;
  for (int s = 0; s < splits; ++s) m = fmaxf(m, pm[s]);
  float num = 0.f, den = 0.f;
  for (int s = 0; s < splits; ++s) {
    if (pl[s] > 0.f) {
      const float w = __expf(pm[s] - m);
      num = fmaf(pa[s * kHeadDim + d], w, num);
      den = fmaf(pl[s], w, den);
    }
  }
  out[head * kHeadDim + d] = __float2half(num / den);
}

// h += x (when x is given), then y = LayerNorm(h). The residual is stored back in fp16,
// matching the reference fp16 residual stream. Two-pass variance for accuracy.
__global__ void __launch_bounds__(256) add_layernorm_kernel(
    half* __restrict__ h, const half* __restrict__ x, const half* __restrict__ gamma,
    const half* __restrict__ beta, half* __restrict__ y, float eps) {
  constexpr int kPer = kHidden / 256;
  __shared__ float red[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float vals[kPer];
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int idx = threadIdx.x + i * 256;
    float r = __half2float(h[idx]);
    if (x) {
      r = __half2float(__float2half(r + __half2float(x[idx])));
      h[idx] = __float2half(r);
    }
    vals[i] = r;
    sum += r;
  }

  sum = warp_sum(sum);
  if (lane == 0) red[warp] = sum;
  __syncthreads();
  float mean = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) mean += red[i];
  mean /= kHidden;
  __syncthreads();

  float sq = 0.f;
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const float d = vals[i] - mean;
    sq += d * d;
  }
  sq = warp_sum(sq);
  if (lane == 0) red[warp] = sq;
  __syncthreads();
  float var = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) var += red[i];
  const float inv = rsqrtf(var / kHidden + eps);

#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int idx = threadIdx.x + i * 256;
    y[idx] = __float2half((vals[i] - mean) * inv * __half2float(gamma[idx]) + __half2float(beta[idx]));
  }
}

const half* hp(const torch::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr<at::Half>()); }
half* hp_mut(torch::Tensor& t) { return reinterpret_cast<half*>(t.data_ptr<at::Half>()); }

void check_half(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kHalf, name, " must be contiguous CUDA fp16");
}

}  // namespace

void lm_head_argmax(torch::Tensor h, torch::Tensor w, torch::Tensor suppress, torch::Tensor best) {
  check_half(h, "h");
  check_half(w, "w");
  TORCH_CHECK(h.numel() == kHidden && w.size(1) == kHidden, "hidden size must be ", kHidden);
  TORCH_CHECK(suppress.scalar_type() == at::kBool && suppress.numel() == w.size(0), "suppress must be bool [vocab]");
  TORCH_CHECK(best.scalar_type() == at::kLong && best.numel() == 1, "best must be int64 [1]");
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  lm_head_argmax_kernel<<<sms * 4, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      hp(h), hp(w), suppress.data_ptr<bool>(), static_cast<int>(w.size(0)),
      reinterpret_cast<unsigned long long*>(best.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void argmax_finalize(torch::Tensor best, torch::Tensor tokens, torch::Tensor state, torch::Tensor cur, int64_t max_pos) {
  TORCH_CHECK(tokens.scalar_type() == at::kLong && state.scalar_type() == at::kLong && cur.scalar_type() == at::kLong,
              "tokens, state and cur must be int64");
  argmax_finalize_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<unsigned long long*>(best.data_ptr<int64_t>()), tokens.data_ptr<int64_t>(),
      state.data_ptr<int64_t>(), cur.data_ptr<int64_t>(), static_cast<int>(max_pos));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void decode_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
                      std::optional<torch::Tensor> len_state, std::optional<torch::Tensor> k_new,
                      std::optional<torch::Tensor> v_new, torch::Tensor part_m, torch::Tensor part_l,
                      torch::Tensor part_acc, double scale) {
  check_half(q, "q");
  check_half(k, "k");
  check_half(v, "v");
  check_half(out, "out");
  TORCH_CHECK(k.dim() == 3 && k.size(2) == kHeadDim && k.sizes() == v.sizes(), "k, v must be [heads, max_len, 64]");
  const int heads = static_cast<int>(k.size(0));
  const int max_len = static_cast<int>(k.size(1));
  const int splits = (max_len + kSplitKeys - 1) / kSplitKeys;
  TORCH_CHECK(q.numel() == heads * kHeadDim && out.numel() == heads * kHeadDim, "q, out must be [heads * 64]");
  TORCH_CHECK(k_new.has_value() == v_new.has_value() && k_new.has_value() == len_state.has_value(),
              "len_state, k_new and v_new must be given together");
  const half* kn = nullptr;
  const half* vn = nullptr;
  if (k_new) {
    check_half(*k_new, "k_new");
    check_half(*v_new, "v_new");
    TORCH_CHECK(k_new->numel() == heads * kHeadDim && v_new->numel() == heads * kHeadDim, "k_new, v_new must be [heads, 64]");
    kn = hp(*k_new);
    vn = hp(*v_new);
  }
  for (const torch::Tensor* t : {&part_m, &part_l, &part_acc}) {
    TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == at::kFloat, "workspace must be contiguous CUDA fp32");
  }
  TORCH_CHECK(part_m.numel() >= heads * splits && part_l.numel() >= heads * splits &&
                  part_acc.numel() >= heads * splits * kHeadDim,
              "workspace too small for ", splits, " splits");
  const int64_t* len_ptr = len_state ? len_state->data_ptr<int64_t>() : nullptr;
  const auto stream = at::cuda::getCurrentCUDAStream();
  decode_attention_split_kernel<<<dim3(heads, splits), 128, 0, stream>>>(
      hp(q), hp_mut(k), hp_mut(v), kn, vn, len_ptr, max_len, static_cast<float>(scale), part_m.data_ptr<float>(),
      part_l.data_ptr<float>(), part_acc.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  decode_attention_combine_kernel<<<heads, kHeadDim, 0, stream>>>(
      part_m.data_ptr<float>(), part_l.data_ptr<float>(), part_acc.data_ptr<float>(), hp_mut(out), splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void add_layernorm(torch::Tensor h, std::optional<torch::Tensor> x, torch::Tensor gamma, torch::Tensor beta,
                   torch::Tensor y, double eps) {
  check_half(h, "h");
  check_half(gamma, "gamma");
  check_half(beta, "beta");
  check_half(y, "y");
  TORCH_CHECK(h.numel() == kHidden, "hidden size must be ", kHidden);
  const half* xp = nullptr;
  if (x) {
    check_half(*x, "x");
    xp = hp(*x);
  }
  add_layernorm_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      hp_mut(h), xp, hp(gamma), hp(beta), hp_mut(y), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lm_head_argmax", &lm_head_argmax);
  m.def("argmax_finalize", &argmax_finalize);
  m.def("decode_attention", &decode_attention);
  m.def("add_layernorm", &add_layernorm);
}
