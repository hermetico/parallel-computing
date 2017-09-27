const char* dgemm_desc = "dgemm custom implementation";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 36
#endif

#if !defined(BLOCK_M)
#define BLOCK_M 36
#endif

#if !defined(BLOCK_K)
#define BLOCK_K 3
#endif

#define min(a,b) (((a)<(b))?(a):(b))


void gepp_var1(int lda, double *A, double *B, double *C, int ki, int ke)
{
    for (int i = 0; i < lda; i++)
    {
        for (int j = 0; j < lda; j++)
        {
            for (int k = ki; k < ke; k++)
            {
                C[(i * lda) + j] += A[(i * lda) + k ] * B[(k * lda) + j];
            }

        }
    }
}

void gemm_var1( int lda, double *A, double *B, double *C)
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
        int j = min(BLOCK_K, lda - i ) + i;
        gepp_var1(lda, A, B, C, i, j);
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm (int lda, double* A, double* B, double* C)
{

    gemm_var1(lda, A, B, C);

}





