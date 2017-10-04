#include <iostream>
#include <stdlib.h>

using namespace std;
/* Your function must have the following signature: */
extern const char* dgemm_desc;
extern void square_dgemm (int, double*, double*, double*);


int main (int argc, char **argv)
{
    int n=5;



    double* A = (double*) malloc (n * n * sizeof(double));
    double* B = (double*) malloc (n * n * sizeof(double));
    double* C = (double*) malloc (n * n * sizeof(double));

    for(int i=0; i <  n ; i++)
    {
        for(int j=0; j <  n ; j++) {
            A[i*n+j] = (double) j*n + i + 1;
            B[i*n+j] = (double) j*n + i +1;
            C[i*n+j] = (double) 0;
        }
    }


    for (int i=0; i<n; i++) {
        for (int j = 0; j < n; j++) {
            cout <<  A[i +j*n] <<", ";
        }
        cout << endl;
    }


    cout << endl ;
    square_dgemm (n, A, B, C);
    for (int i=0; i<n; i++) {
        for (int j = 0; j < n; j++) {
            cout <<  C[i * n + j] <<", ";
        }
        cout << endl;
    }
    free(A);
    free(B);
    free(C);
    return 0;
}