#include <stdio.h>
     #include <pmmintrin.h>

     // __m128d : these 128 bits can contain two double precision floating point numbers

     __m128d a, b, c;

     // _mm_load_pd and _mm_store_pd: The address must be 16-byte aligned


     double inp_sse1[2] __attribute__((aligned(16))) = { 3560000.7100100959, 3440000.2800070001 };
     double inp_sse2[2] __attribute__((aligned(16))) = { 4200000.5025000005, 4790000.4575000059 };
     double out_sse[2] __attribute__((aligned(16)));

     int main(void){
    
     printf("Usual way...\n");
    
     printf("Result 1: %.10f\n",inp_sse1[0] + inp_sse1[1]);    
     printf("Result 2: %.10f\n",inp_sse2[0] + inp_sse2[1]);

     
     printf("...with SSE3 instructions...\n");
    
     a = _mm_load_pd(inp_sse1);
     b = _mm_load_pd(inp_sse2);
     c = _mm_hadd_pd(a, b);
     _mm_store_pd(out_sse,c);
     
     printf("Result 1: %.10f\n",out_sse[0]);
     printf("Result 2: %.10f\n",out_sse[1]);


     return 0; 
     }

