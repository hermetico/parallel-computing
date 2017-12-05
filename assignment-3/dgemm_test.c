#include <stdlib.h>
#include <stdio.h>
#define DGEMM dgemm_
extern void DGEMM (char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

int main(int argc, char **argv) {

	double A[9], B[9], C[9];
	int n = 3;

	for (int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			A[n*i+j] = n*i+j + 1;
			B[n*i+j] = n*n + n*i+j + 1;
			printf("A = %lf, B = %lf\n", A[n*i+j], B[n*i+j]);
			C[n*i+j] = (double) 0;
		}
	}

	char TRANSA = 'T';
	char TRANSB = 'T';

	double ALPHA = 1.;
	double BETA = 1.;

	int LDA = n;
	int LDB = n;
	int LDC = n;

	DGEMM(&TRANSA, &TRANSB, &n, &n, &n, &ALPHA, &A, &LDA, &B, &LDB, &BETA, &C, &LDC);

	printf("\n");
	for (int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			printf("%lf\t", C[n*i+j]);
		}
		printf("\n");
	}
	return 0;
}
