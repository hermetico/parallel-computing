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


void gepp_var1(unsigned int lda, double* A, double* B, double* C, unsigned int ki, unsigned int ke)
{

    for (unsigned int i = 0; i < lda; i++)
    {
        for (unsigned int j = 0; j < lda; j++)
        {
            double cij = C[j * lda + i];

            for (unsigned int k = ki; k < ki + ke; k++)
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
    * Mc, Kc, Nn
    * lda
    * A, B, C
    * Reduces A to column panels, B to row panels
    * C and lda
    * A and
    */

    for (unsigned int ki = 0; ki < lda; ki += KC)
    {
        unsigned int ke = min(KC, lda - ki );
        gepp_var1(lda, A, B, C, ki, ke);
    }
}

void square_dgemm (int lda, double* A, double* B, double* C)
{
    gemm_var1(lda, A, B, C);

}

