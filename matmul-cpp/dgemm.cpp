const char* dgemm_desc = "dgemm custom implementation";
#include <iostream>
using namespace std;

#if !defined(KC)
#define KC 16
#endif

#if !defined(MC)
#define MC 16
#endif

#if !defined(NC)
#define NC 2
#endif

#define min(a,b) (((a)<(b))?(a):(b))

void gebp_var1(unsigned int lda, double* A, double* B, double* C, unsigned int kc, unsigned int mc)
{

    for (unsigned int m = 0; m < mc; m++)
    {
        for (unsigned int j = 0; j < lda; j++)
        {
            double cmj = C[j * lda + m];

            for (unsigned int k = 0; k < kc; k++)
            {
                cmj += A[k * lda + m ] * B[j * lda + k];
            }
            C[j * lda + m] = cmj;
        }
    }

}

void gepp_var1(unsigned int lda, double* A, double* B, double* C, unsigned int kc)
{

    /*
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
    */

    /**
    * Splices the matrices A and C by its M dimension
    */

    for(unsigned int mci = 0; mci < lda; mci += MC)
    {
        // checking mc edges
        unsigned int mc = min(MC, lda - mci );
        gebp_var1(lda, A + mci, B, C + mci, kc, mc);
    }
}

void gemm_var1(unsigned int lda, double* A, double* B, double* C)
{
    /**
    * Splices the matrices A and B by its K dimension
    */

    for (unsigned int kci = 0; kci < lda; kci += KC)
    {
        // checking kc edges
        unsigned int kc = min(KC, lda - kci );
        gepp_var1(lda, A + kci * lda, B + kci, C, kc);
    }
}

void square_dgemm (int lda, double* A, double* B, double* C)
{
    gemm_var1(lda, A, B, C);

}

