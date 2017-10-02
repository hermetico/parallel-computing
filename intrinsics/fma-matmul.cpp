#include <immintrin.h>
#include <stdio.h>

int main() {

    unsigned int VecSize = 4;

	double __attribute__((aligned(32))) A[VecSize] = {1, 2, 3, 4};
	double __attribute__((aligned(32))) B[VecSize] = {5, 6, 7, 8};
	double __attribute__((aligned(32))) C[VecSize];


	// assume A and B are unaligned
	__m256d a = _mm256_load_pd(A);
	__m256d b = _mm256_load_pd(B);
	__m256d zeros = _mm256_setzero_pd();

	// multiply and add
	__m256d acc = _mm256_fmadd_pd(a, b, zeros);
	_mm256_store_pd(&C[0], acc);

	for(unsigned int i = 0; i < VecSize; i++){
		printf("%ff\n",C[i]);
	}

    return 0;
}