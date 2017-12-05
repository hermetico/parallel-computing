
#include "matrix.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#define DGEMM dgemm_
extern void DGEMM (char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

/* Make P(0,0,0)'s full size matrices */
void make_full_matrices(double *A, double *B, int n)
{
  srand48(time(NULL));
  for(int i = 0; i < n * n; ++i)
  {
    //A[i] = drand48();
    //B[i] = drand48();
    A[i] = (double) i + 1;
    B[i] = (double) i + n*n + 1;
  }
    
}

/* Allocate matrices */
void allocate_matrices(/* PARAMETERS */)
{
    
}

/* Reset the matrices for new use */
void reset_matrices(/* PARAMETERS */)
{
    
}

/* Free the memory for matrices */
void destroy_matrices(/* PARAMETERS */)
{
    
}

/* Matrix multiplication */

void matrix_mult(int M, int N, int K , double* A, double* B, double* C)
{
  char TRANSA = 'N';
  char TRANSB = 'N';

  double ALPHA = 1.;
  double BETA = 1.;

  int LDA = K;
  int LDB = N;
  int LDC = M;
  DGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC); 
}