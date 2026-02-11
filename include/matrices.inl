

#if defined(__FMA__)

#if defined(__SSE__)

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

template <size_t rows, size_t cols>
__attribute__((always_inline)) inline void set_matrix_aligned16B(flost src[]) {
	constexpr size_t Ns = size_aligned16B(cols);
	__m128 src_v;
#pragma unroll rows
	for (size_t i = 0; i < rows; i++) {
#pragma unroll (4 * (cols % 4))
		for (size_t j = 0; j < cols; j += 4) {
			src_v = _mm_loadu_ps(&src[i * cols + j]); // src may not be aligned by default
			_mm_store_ps(&dst[i * Ns + j], src_v); // dst is guaranteed to be aligned, so aligned stores can be used for efficiency
		}
#pragma unroll (cols % 4)
		for (size_t j = 4 * (cols % 4); j < cols; j++) {
			dst[i * Ns + j] = src[i * cols + j];
#pragma unroll (Ns - cols)
		for (size_t j = cols; j < Ns; j++) {
			dst[i * Ns + j] = 0.f;
	}}
}

#endif 

#if defined(__AVX__)

template <size_t rows, size_t cols>
__attribute__((always_inline)) inline void print_matrix_aligned16B(float src[]) {
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
__attribute__((always_inline)) inline void set_matrix_aligned32B(float src[]) {
	constexpr size_t Ns = size_aligned32B(cols);
	__m256 src_v;
#pragma unroll rows
	for (size_t i = 0; i < rows; i++) {
#pragma unroll (8 * (cols % 8))
		for (size_t j = 0; j < cols; j += 8) {
			src_v = _mm_loadu_ps(&src[i * cols + j]);
			_mm_store_ps(&dst[i * Ns + j], src_v);
		}
#pragma unroll (cols % 8)
		for (size_t j = 8 * (cols % 8); j < cols; j++) {
			dst[i * Ns + j] = src[i * cols + j];
		}
#pragma unroll (Ns - cols)
		for (size_t j = cols; j < Ns; j++) {
			dst[i * Ns + j] = 0.f;
	}}
}

#endif

#endif
