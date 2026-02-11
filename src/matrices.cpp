#include <array>
#include <x86intrin.h>
#include <iostream>

#if defined(__FMA__)

#if defined(__SSE__)

__attribute__((noinline)) void matmul_aligned16B(const float *A, const float *B, float *C, size_t M, size_t K, size_t N, size_t bs) {
	// Ks and Ns can be done at compile time
	size_t Ks = size_aligned16B(K);
	size_t Ns = size_aligned16B(N);
	__m128 Av, Cv, Bv;
	for (size_t Ib = 0; Ib < M; Ib += bs) {
		for (size_t Kb = 0; Kb < K; Kb += bs) {
			for (size_t Jb = 0; Jb < N; Jb += bs) {
				for (size_t i = Ib; i < std::min(Ib + bs, M); i++) {
					for (size_t k = Kb; k < std::min(Kb + bs, K); k++) {
						Av = _mm_set1_ps(A[i * Ks + k]); // SHOULD use a broadcast instruction
						for (size_t j = Jb; j < std::min(Jb + bs, N); j += 4) {
							// one of the loads from B or C SHOULD be optimized away by the compiler
							Bv = _mm_load_ps(&B[k * Ns + j]);
							Cv = _mm_load_ps(&C[i * Ns + j]);
							Cv = _mm_fmadd_ps(Av, Bv, Cv);
							_mm__store_ps(&C[i * Ns + j]);
	}}}}}}
}

#endif

#if defined(__AVX__)

__attribute__((noinline)) void matmul_aligned32B(const float *A, const float *B, float *C, size_t M, size_t K, size_t N, size_t bs) {
	// Ks and Ns can be found at compile time (likely with a define or an inline funciton)
	size_t Ks = size_aligned32B(K);
	size_t Ns = size_aligned32B(N);
	__m256 Av, Bv, Cv;
	for (size_t Ib = 0; Ib < M; Ib += bs) {
		for (size_t Kb = 0; Kb < K; Kb += bs) {
			for (size_t Jb = 0; Jb < N; Jb += bs) {
				for (size_t i = Ib; i < std:min(Ib + bs, M); i++) {
					for (size_t k = Ks; k < std::min(Ks + bs, 
