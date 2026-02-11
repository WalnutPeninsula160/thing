#include <iostream>
#include <x86intrin.h>
#include <array>

template <size_t rows, size_t cols>
using matrix = float[rows * cols];

#if defined(__FMA__)

#if defined(__SSE__)

#define size_aligned16B(siz) (((siz >> 0x4) << 0x4) + ((siz % 0x10 > 0) << 0x4) >> 2) // bullshit calc for aligning the rows to xmm register sizes

template <size_t rows, size_t cols>
__attribute__((always_inline)) inline void print_matrix_aligned16B(float src[]) {
	constexpr size_t Ns = size_aligned16B(cols);
#pragma unroll rows
	for (size_t i = 0; i < rows; i++) {
#pragma unroll cols
		for (size_t j = 0; j < cols; j++) {
			std::cout << src[i * Ns + j] << '\t';
		}
		std::cout << '\n';
	}
}

// the std array inputted is stored in the rodata section by the g++ compiler, so this can be easily vectorized
template <size_t rows, size_t cols>
__attribute__((always_inline)) inline void set_matrix_aligned16B(std::array<float, rows * cols> src, float dst[]) {
	constexpr size_t Ns = size_aligned16B(cols);
	__m128 src_v;
#pragma unroll rows
	for (size_t i = 0; i < rows; i++) {
#pragma unroll (4 * (cols / 4))
		for (size_t j = 0; j < cols; j += 4) {
			src_v = _mm_loadu_ps(&src[i * cols + j]); // not sure if the std array is aligned by default, so unaligned load is safe for this
			_mm_store_ps(&dst[i * Ns + j], src_v); // since we know that dst is aligned, we can safely use aligned stores
		}
#pragma unroll (cols % 4)
		for (size_t j = cols % 4; j < cols; j++) {
		       dst[i * Ns + j] = src[i * cols + j];
		}	       
#pragma unroll (Ns - cols)
		for (size_t j = cols; j < Ns; j++) {
			dst[i * Ns + j] = 0.f;
	}}
}

template <size_t rows, size_t cols>
using matrix_aligned16B = alignas(16) float[rows * size_aligned16B(cols)];

__attribute__((noinline)) void matmul_aligned16B(const float *A, const float *B, float *C, size_t M, size_t K, size_t N, size_t bs) {
	size_t Ks = size_aligned16B(K);
	size_t Ns = size_aligned16B(N);
	__m128 Av, Bv, Cv;
	for (size_t Ib = 0; Ib < M; Ib += bs) {
		for (size_t Kb = 0; Kb < K; Kb += bs) {
			for (size_t Jb = 0; Jb < N; Jb += bs) {
				for (size_t i = Ib; i < std::min(Ib + bs, M); i++) {
					for (size_t k = Kb; k < std::min(Kb + bs, K); k++) {
						Av = _mm_set1_ps(A[i * Ks + k]);
						for (size_t j = Jb; j < std::min(Jb + bs, N); j += 4) {
							Bv = _mm_load_ps(&B[k * Ns + j]);
							Cv = _mm_load_ps(&C[i * Ns + j]);
							Cv = _mm_fmadd_ps(Av, Bv, Cv);
							_mm_store_ps(&C[i * Ns + j], Cv);
	}}}}}}
}

#endif

#if defined(__AVX__) 

#define size_aligned32B(siz) ((((siz >> 0x5) << 0x5) + (siz % 0x20 > 0) << 0x5) >> 2) // bullshit calc for aligning the rows to ymm register sizes

template <size_t rows, size_t cols>
__attribute__((always_inline)) inline void print_matrix_aligned32B(float src[]) {
	constexpr size_t Ns = size_aligned32B(cols);
#pragma unroll rows
	for (size_t i = 0; i < rows; i++) {
#pragma unroll cols
		for (size_t j = 0; j < cols; j++) {
			std::cout << src[i * Ns + j] << '\t';
		}
		std::cout << '\n';
	}
}

template <size_t rows, size_t cols>
__attribute__((always_inline)) inline void set_matrix_aligned32B(std::array<float, rows * cols> src, float dst[]) {
	constexpr size_t Ns = size_aligned32B(cols);
#pragma unroll rows
	for (size_t i = 0; i < rows; i++) {
#pragma unroll cols
		for (size_t j = 0; j < cols; j++) {
			dst[i * Ns + j] = src[i * cols + j];
		}
#pragma unroll (Ns - cols)
		for (size_t j = cols; j < Ns; j++) {
			dst[i * Ns + j] = 0.f;
	}}
}

template <size_t rows, size_t cols>
using matrix_aligned32B = alignas(32) float[rows * size_aligned32B(cols)];

__attribute__((noinline)) void matmul_aligned32B(const float *A, const float *B, float *C, size_t M, size_t K, size_t N, size_t bs) {
	size_t Ks = size_aligned32B(K);
	size_t Ns = size_aligned32B(N);
	__m256 Av, Bv, Cv;
	for (size_t Ib = 0; Ib < M; Ib += bs) {
		for (size_t Kb = 0; Kb < K; Kb += bs) {
			for (size_t Jb = 0; Jb < N; Jb += bs) {
				for (size_t i = Ib; i < std::min(Ib + bs, M); i++) {
					for (size_t k = Kb; k < std::min(Kb + bs, K); k++) {
						for (size_t j = Jb; j < std::min(Jb + bs, N); j += 8) {
							Bv = _mm256_load_ps(&B[k * Ns + k]);
							Cv = _mm256_load_ps(&C[i * Ns + j]);
							Cv = _mm256_fmadd_ps(Av, Bv, Cv);
							_mm256_store_ps(&C[i * Ns + j], Cv);
	}}}}}}
}

#endif

#endif

int main(int argc, char **argv) {
	size_t BLOCKSIZ = static_cast<size_t>(std::stoi(argv[1]));
	matrix_aligned16B<3, 3> A;
	matrix_aligned16B<3, 3> B;
	matrix_aligned16B<3, 3> C;
	set_matrix_aligned16B<3, 3>({1, 2, 3, 4, 5, 6, 7, 8, 9}, A);
	set_matrix_aligned16B<3, 3>({1, 2, 3, 4, 5, 6, 7, 8, 9}, B);
	matmul_aligned16B(A, B, C, 3, 3, 3, 8);
	print_matrix_aligned16B<3, 3>(A);
	return 0;
}
