const char* dgemm_desc = "dgemm custom implementation";

#if !defined(BLOCK_K)
#define BLOCK_K 3
#endif

#define min(a,b) (((a)<(b))?(a):(b))


void gepp_variation1(int lda, double* A, double* B, double* C, int ki, int ke)
{
    for (int i = 0; i < lda; i++)
    {
        for (int j = 0; j < lda; j++)
        {
            double cij = C[(i * lda) + j];
            for (int k = ki; k < ki + ke; k++)
            {
               cij += A[(i * lda) + k ] * B[(k * lda) + j];
            }
            C[(i * lda) + j] = cij;
        }
    }
}

void gemm_variation1(unsigned int lda, double* A, double* B, double* C)
{
    /**
    * Mc, Kc, Nn
    * lda
    * A, B, C
    * Reduces A to column panels, B to row panels
    * C and lda
    * A and
    */

    for (int i = 0; i < lda; i += BLOCK_K)
    {
        int j = min(BLOCK_K, lda - i );
        gepp_variation1(lda, A, B, C, i, j);
    }
}

void square_dgemm (int lda, double* A, double* B, double* C)
{
    gemm_variation1(lda, A, B, C);
}





