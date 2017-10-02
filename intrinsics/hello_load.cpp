#include <immintrin.h>
#include <stdio.h>

int main() {

    unsigned int size = 4;
	// Check the aligned alloc!!
    double* nums = (double*) aligned_alloc (32, size * sizeof(double));

    for(unsigned int i = 0; i < size; i++){
        nums[i] = i;
    }

	__m256d doubles = _mm256_load_pd(nums);


	double* results = (double* )&doubles;
	for(unsigned int i = 0; i < size; i++){
		printf("%ff\n",results[i]);
	}

    return 0;
}