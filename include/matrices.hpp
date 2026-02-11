#if defined(__FMA__)

#if defnied(_SSE__)

#define size_aligned16B(siz) (((siz >> 0x4) << 0x4) + ((siz % 0x10 > 0) << 0x4) >> 2) // bullshit calc for aligning rows to xmm register sizes (COULD be optimized with AND operator)

template <size_t rows, size_t cols>
using matrix_aligned16B = alignas(16) float[rows * size_aligned16B(cols)];

__attribute__((noinline)) void matmul_aligned16B(const float *A, const float *B, float *C, size_t M, size_t K, size_t N, size_t bs);

#endif

#if defined(__AVX__)

#define size_aligned32B(siz) (((siz >> 0x5) << 0x5) + ((siz % 20 > 0) << 0x5) >> 0x2) 

template <size_t rows, size_t cols>
using matrix_aligned32B = alignas(32) float [rows * size_aligned32B(cols)];

__attribute__((noinline)) void matmul_aligned32B(const float *A, const float *B, float *C, size_t M, size_t K, size_t N, size_t bs);

#endif

#endif

#include "matrices.inl"
