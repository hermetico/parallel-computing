
#include "matrix.h"

/* Make P(0,0,0)'s full size matrices */
void make_full_matrices(/* PARAMETERS */)
{
    
}

/* Allocate matrices */
void allocate_matrices(/* PARAMETERS */)
{
    
}

/* Reset the matrices for new use */
/void reset_matrices(/* PARAMETERS */)
{
    
}

/* Free the memory for matrices */
void destroy_matrices(/* PARAMETERS */)
{
    
}

/* Matrix multiplication */
void matrix_mult(int N, double* A, double* B, double* C)
{
  char TRANSA = 'N';
  char TRANSB = 'N';
  int M = N;
  int K = N;
  double ALPHA = 1.;
  double BETA = 1.;
  int LDA = N;
  int LDB = N;
  int LDC = N;
  DGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC); 
}