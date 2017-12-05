
#ifndef MATRIX_H
#define MATRIX_H

/* reference_dgemm wraps a call to the BLAS-3 routine DGEMM, via the standard FORTRAN interface - hence the reference semantics. */


/* Function prototypes */
void make_full_matrices(double *A, double *B,  int n);
// void allocate_matrices(/* PARAMETERS */);
// void reset_matrices(/* PARAMETERS */);
// void destroy_matrices(/* PARAMETERS */);
void matrix_mult(int M, int N, int K , double* A, double* B, double* C);

#endif /* MATRIX_H */
