const char* dgemm_desc = "dgemm custom implementation";
#include <iostream>
using namespace std;

#if !defined(KC)
#define KC 16
#endif

#if !defined(MC)
#define MC 2
#endif

#if !defined(NC)
#define NC 2
#endif

#define min(a,b) (((a)<(b))?(a):(b))


void gepp_var1(unsigned int lda, double* A, double* B, double* C, unsigned int kc)
{

    for (unsigned int i = 0; i < lda; i++)
    {
        for (unsigned int j = 0; j < lda; j++)
        {
            double cij = C[j * lda + i];

            for (unsigned int k = 0; k < kc; k++)
            {
                cij += A[k * lda + i ] * B[j * lda + k];
            }
            C[j * lda + i] = cij;
        }
    }
}

void gemm_var1(unsigned int lda, double* A, double* B, double* C)
{
    /**
    * Splices the matrices A and B by its K dimension
    */

    for (unsigned int kci = 0; kci < lda; kci += KC)
    {
        // where Kc ends if were are in an edge
        unsigned int kce = min(KC, lda - kci );
        gepp_var1(lda, A + kci * lda, B + kci, C, kce);
    }
}

void square_dgemm (int lda, double* A, double* B, double* C)
{
    gemm_var1(lda, A, B, C);

}

