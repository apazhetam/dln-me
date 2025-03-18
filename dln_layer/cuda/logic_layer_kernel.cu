// Code modified from the "difflogic - A Library for Differentiable Logic Gate Networks" GitHub folder:
// https://github.com/Felix-Petersen/difflogic/blob/main/difflogic/cuda/difflogic_kernel.cu
// Petersen, Felix and Borgelt, Christian and Kuehne, Hilde and Deussen, Oliver.
// Deep Differentiable Logic Gate Networks.
// Conference on Neural Information Processing Systems (NeurIPS).
// 2022.

#include <torch/extension.h>

#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <vector>

#define BACKWARD_W_BATCH_THREADS 32

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

// adapted from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans)                                                                                                 \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(const cudaError_t code, const char *const file, const int line, const bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }


/**********************************************************************************************************************/


template <typename T> struct AtomicFPOp;

template <> struct AtomicFPOp<at::Half> {
    template <typename func_t> inline __device__ at::Half operator()(at::Half *address, at::Half val, const func_t &func) {
        unsigned int *address_as_ui = (unsigned int *)((char *)address - ((size_t)address & 2));
        unsigned int old = *address_as_ui;
        unsigned int assumed;

        at::Half hsum;
        do {
            assumed = old;
            hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
            hsum = func(hsum, val);
            old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
            old = atomicCAS(address_as_ui, assumed, old);
        } while (assumed != old);
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        return hsum;
    }
};

static inline __device__ at::Half gpuAtomicAdd(at::Half *address, at::Half val) {
#if defined(USE_ROCM) || ((defined(CUDA_VERSION) && CUDA_VERSION < 10000) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))

    unsigned int *aligned = (unsigned int *)((size_t)address - ((size_t)address & 2));
    unsigned int old = *aligned;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned short old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
        __half sum = c10::Half(__ushort_as_half(old_as_us)) + c10::Half(__float2half((float)val));
        unsigned short sum_as_us = __half_as_ushort(sum);
        unsigned int sum_as_ui = (size_t)address & 2 ? (sum_as_us << 16) | (old & 0xffff) : (old & 0xffff0000) | sum_as_us;
        old = atomicCAS(aligned, assumed, sum_as_ui);
    } while (assumed != old);
    unsigned short old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
    return c10::Half((__half_raw)__ushort_as_half(old_as_us));
#else
    return atomicAdd(reinterpret_cast<__half *>(address), val);
#endif
}

static inline __device__ float gpuAtomicAdd(float *address, float val) { return atomicAdd(address, val); }

static inline __device__ double gpuAtomicAdd(double *address, double val) { return atomicAdd(address, val); }




/**********************************************************************************************************************/
/**  TRAINING MODE  ***************************************************************************************************/
/**********************************************************************************************************************/


template <typename scalar_t>
__global__ void logic_layer_cuda_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y
) {

    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);
            col += blockDim.y * gridDim.y
        ) {
            const auto a_ = a[col][row];
            const auto b_ = b[col][row];
            const auto w_ = w[col];

            y[col][row] = (
                 ((w_[1] * (a_ * b_)
                 + w_[2] * (a_ - a_ * b_))
                + (w_[3] * a_
                 + w_[4] * (b_ - a_ * b_)))
               + ((w_[5] * b_
                 + w_[6] * (a_ + b_ - static_cast<scalar_t>(2) * a_ * b_))
                + (w_[7] * (a_ + b_ - a_ * b_)
                 + w_[8] * (static_cast<scalar_t>(1) - (a_ + b_ - a_ * b_)))))
              + (((w_[9] * (static_cast<scalar_t>(1) - (a_ + b_ - static_cast<scalar_t>(2) * a_ * b_))
                 + w_[10] * (static_cast<scalar_t>(1) - b_)) +
                  (w_[11] * (static_cast<scalar_t>(1) - b_ + a_ * b_)
                 + w_[12] * (static_cast<scalar_t>(1) - a_))) +
                  (w_[13] * (static_cast<scalar_t>(1) - a_ + a_ * b_)
                 + w_[14] * (static_cast<scalar_t>(1) - a_ * b_)
                 + w_[15])
            );
    }}
}


template <typename scalar_t>
__global__ void
logic_layer_cuda_backward_w_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_w_
) {

    const auto row_ = blockIdx.x * blockDim.x + threadIdx.x;

    for (  // neuron dim
        auto col = blockIdx.y * blockDim.y + threadIdx.y;
        col < grad_y.size(0);
        col += blockDim.y * gridDim.y
    ) {
        scalar_t grad_w_local_1 = 0;
        scalar_t grad_w_local_3 = 0;
        scalar_t grad_w_local_5 = 0;
        scalar_t grad_w_local_15 = 0;

        for (int row = row_; row < grad_y.size(1); row += BACKWARD_W_BATCH_THREADS) {  // batch dim
            const auto a_ = a[col][row];
            const auto b_ = b[col][row];
            const auto grad_y_ = grad_y[col][row];

            // compute grad_w
            grad_w_local_1 += (a_ * b_) * grad_y_;
            grad_w_local_3 += a_ * grad_y_;
            grad_w_local_5 += b_ * grad_y_;
            grad_w_local_15 += grad_y_;
        }

        grad_w_[col][row_][0] = grad_w_local_1;
        grad_w_[col][row_][1] = grad_w_local_3;
        grad_w_[col][row_][2] = grad_w_local_5;
        grad_w_[col][row_][3] = grad_w_local_15;
    }
}


template <typename scalar_t>
__global__ void
logic_layer_cuda_backward_ab_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_a,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_b
) {

    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < grad_a.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < grad_a.size(0);
            col += blockDim.y * gridDim.y
        ) {

            const auto a_ = a[col][row];
            const auto b_ = b[col][row];
            const auto grad_y_ = grad_y[col][row];

            // compute grad_a
            const auto dy_da = (
                (w[col][1] * b_
                + w[col][2] * (static_cast<scalar_t>(1) - b_)
                + w[col][3]) +
                (w[col][4] * -b_
                + w[col][6] * (static_cast<scalar_t>(1) - static_cast<scalar_t>(2) * b_)
                + w[col][7] * (static_cast<scalar_t>(1) - b_))
            ) + ((w[col][8] * (b_ - static_cast<scalar_t>(1))
                + w[col][9] * (static_cast<scalar_t>(2) * b_ - static_cast<scalar_t>(1))
                + w[col][11] * b_)
            + (-w[col][12]
                + w[col][13] * (b_ - static_cast<scalar_t>(1))
                + w[col][14] * -b_)
            );
            grad_a[col][row] = dy_da * grad_y_;

            // compute grad_b
            const auto dy_db = (
                (w[col][1] * a_
                + w[col][2] * -a_
                + w[col][4] * (static_cast<scalar_t>(1) - a_))
                + (w[col][5]
                + w[col][6] * (static_cast<scalar_t>(1) - static_cast<scalar_t>(2) * a_)
                + w[col][7] * (static_cast<scalar_t>(1) - a_))
            ) + ((w[col][8] * (a_ - static_cast<scalar_t>(1))
                + w[col][9] * (static_cast<scalar_t>(2) * a_ - static_cast<scalar_t>(1))
                - w[col][10])
                + (w[col][11] * (a_ - static_cast<scalar_t>(1))
                + w[col][13] * a_
                + w[col][14] * -a_)
            );
            grad_b[col][row] = dy_db * grad_y_;
    }}
}


torch::Tensor logic_layer_cuda_forward(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);

    const auto batch_size = a.size(1);
    // const auto in_size = a.size(0);
    const auto out_size = w.size(0);

    auto y = torch::empty({out_size, batch_size}, torch::dtype(a.dtype()).device(a.device()));

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "logic_layer_cuda_forward", ([&] {
                           logic_layer_cuda_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               a.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
                           );
                       }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return y;
}


torch::Tensor logic_layer_cuda_backward_w(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(grad_y);


    const auto batch_size = a.size(1);
    // const auto in_size = a.size(0);
    const auto out_size = grad_y.size(0);

    auto grad_w_4 = torch::empty({out_size, BACKWARD_W_BATCH_THREADS, 4}, torch::dtype(a.dtype()).device(a.device()));

    dim3 threads_per_block(BACKWARD_W_BATCH_THREADS, 1024 / BACKWARD_W_BATCH_THREADS);

    const dim3 blocks_per_grid(
        1,
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "logic_layer_cuda_backward_w", ([&] {
                           logic_layer_cuda_backward_w_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               a.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_w_4.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
                       }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    const auto grad_w_components = grad_w_4.sum(1);
    const auto grad_w_ab = grad_w_components.index({torch::indexing::Slice(), 0});
    const auto grad_w_a = grad_w_components.index({torch::indexing::Slice(), 1});
    const auto grad_w_b = grad_w_components.index({torch::indexing::Slice(), 2});
    const auto grad_w_ = grad_w_components.index({torch::indexing::Slice(), 3});

    const auto grad_w = torch::stack({
        torch::zeros({out_size}, torch::dtype(a.dtype()).device(a.device())),
        grad_w_ab,
        grad_w_a - grad_w_ab,
        grad_w_a,
        grad_w_b - grad_w_ab,
        grad_w_b,
        grad_w_a + grad_w_b - grad_w_ab - grad_w_ab,
        grad_w_a + grad_w_b - grad_w_ab,
        grad_w_ - grad_w_a - grad_w_b + grad_w_ab,
        grad_w_ - grad_w_a - grad_w_b + grad_w_ab + grad_w_ab,
        grad_w_ - grad_w_b,
        grad_w_ - grad_w_b + grad_w_ab,
        grad_w_ - grad_w_a,
        grad_w_ - grad_w_a + grad_w_ab,
        grad_w_ - grad_w_ab,
        grad_w_,
    }, 1);


    return grad_w;
}


std::tuple<torch::Tensor, torch::Tensor> logic_layer_cuda_backward_ab(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor grad_y
) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);
    CHECK_INPUT(grad_y);

    auto grad_a = torch::empty_like(a);
    auto grad_b = torch::empty_like(b);

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(grad_a.size(1), static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(grad_a.size(0), static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "logic_layer_cuda_backward_ab", ([&] {
                           logic_layer_cuda_backward_ab_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               a.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_a.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
                           );
                       }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return {grad_a, grad_b};
}


/**********************************************************************************************************************/
/**  INFERENCE MODE  **************************************************************************************************/
/**********************************************************************************************************************/


// | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
// |----|----------------------|-------|-------|-------|-------|
// | 0  | 0                    | 0     | 0     | 0     | 0     |
// | 1  | A and B              | 0     | 0     | 0     | 1     |
// | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
// | 3  | A                    | 0     | 0     | 1     | 1     |
// | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
// | 5  | B                    | 0     | 1     | 0     | 1     |
// | 6  | A xor B              | 0     | 1     | 1     | 0     |
// | 7  | A or B               | 0     | 1     | 1     | 1     |
// | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
// | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
// | 10 | not(B)               | 1     | 0     | 1     | 0     |
// | 11 | B implies A          | 1     | 0     | 1     | 1     |
// | 12 | not(A)               | 1     | 1     | 0     | 0     |
// | 13 | A implies B          | 1     | 1     | 0     | 1     |
// | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
// | 15 | 1                    | 1     | 1     | 1     | 1     |

template <typename T> __device__ __forceinline__ T bin_op_eval(const T a_, const T b_, const int op_idx) {
    bool A = (a_ != 0);
    bool B = (b_ != 0);

    switch (op_idx) {
    case 0:
        return static_cast<T>(0);
    case 1:
        return static_cast<T>(A && B);
    case 2:
        return static_cast<T>(A && !B);
    case 3:
        return static_cast<T>(A);
    case 4:
        return static_cast<T>(B && !A);
    case 5:
        return static_cast<T>(B);
    case 6:
        return static_cast<T>(A ^ B);
    case 7:
        return static_cast<T>(A || B);
    case 8:
        return static_cast<T>(!(A || B));
    case 9:
        return static_cast<T>(!(A ^ B));
    case 10:
        return static_cast<T>(!B);
    case 11:
        return static_cast<T>(!B || A);
    case 12:
        return static_cast<T>(!A);
    case 13:
        return static_cast<T>(!A || B);
    case 14:
        return static_cast<T>(!(A && B));
    default:
        return static_cast<T>(1);
    }
}

template <typename scalar_t>
__global__ void logic_layer_cuda_eval_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<uint8_t, 1, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y
) {
    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);
            col += blockDim.y * gridDim.y
        ) {

            const auto a_ = a[col][row];
            const auto b_ = b[col][row];
            const auto w_ = w[col];
            y[col][row] = bin_op_eval(a_, b_, w_);
        }
    }
}

torch::Tensor logic_layer_cuda_eval(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);

    const auto batch_size = a.size(1);
    const auto out_size = w.size(0);

    auto y = torch::zeros({out_size, batch_size}, torch::dtype(a.dtype()).device(a.device()));

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_INTEGRAL_TYPES(a.type(), "logic_layer_cuda_eval_kernel", ([&] {
                                   logic_layer_cuda_eval_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                       a.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       w.packed_accessor64<uint8_t, 1, torch::RestrictPtrTraits>(),
                                       y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
                                   );
                               }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return y;
}
