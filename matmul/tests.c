#include <stdio.h>  // For: perror
#include <stdlib.h>
/* Your function must have the following signature: */
extern const char* dgemm_desc;
extern void square_dgemm (int, double*, double*, double*);


int main (int argc, char **argv)
{
    int n=4;



    double* A = (double*) malloc (n * n * sizeof(double));
    double* B = (double*) malloc (n * n * sizeof(double));
    double* C = (double*) malloc (n * n * sizeof(double));

    for(int i=0; i < n * n; i++)
    {

        A[i] = i;
        B[i] = i;
        C[i] = 0;
    }
    printf(" \n");
    square_dgemm (n, A, B, C);
    for (int i=0; i<n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%lf ", C[i * n + j]);
        }
        printf("\n");
    }
    free(A);
    free(B);
    free(C);
    return 0;
}
