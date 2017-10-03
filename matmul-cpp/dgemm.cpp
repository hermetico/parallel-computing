const char* dgemm_desc = "dgemm custom implementation";
#include <iostream>
#include <stdlib.h>
#include <immintrin.h>

using namespace std;

#if !defined(KC)
#define KC 120
#endif

#if !defined(MC)
#define MC 24
#endif

#if !defined(NR)
#define NR 24
#endif

#define min(a,b) (((a)<(b))?(a):(b))

// packed B and A
double* BP;
double* AP;


void pack_A(unsigned int lda, double* original, double* packed, unsigned int kc, unsigned int mc )
{
	for(unsigned int m = 0; m < mc; m++)
	{
		for(unsigned int k = 0; k < kc; k++)
		{
			packed[m * kc + k] = original[k * lda + m];
		}
	}
}

void pack_B(unsigned int lda, double* original, double* packed, unsigned int kc )
{
    for(unsigned int i = 0; i < lda; i++)
    {
        for(unsigned int k = 0; k < kc; k++)
        {
            packed[i * kc + k] = original[i * lda + k];
        }
    }
}

void do_block(unsigned int lda, double* AP, double* BP, double* C, unsigned int kc, unsigned int mc, unsigned int nr)
{

	double __attribute__((aligned(32))) accp[4]; //packed accumulator

    for (unsigned int m = 0; m < mc; m++)
    {
        for (unsigned int n = 0; n < nr; n++)
        {
	        unsigned int mkc = m * kc, nkc = n * kc;
            double cmn = C[n * lda + m];

	        unsigned int k = 0;

	        /*
	        cout << "kc " << kc << endl;
			cout << "mc " << mc << endl;
			cout << "nr " << nr << endl;
	         */
	        //TODO fix this hack!!
			if ( kc > 4 ){
				__m256d acc = _mm256_setzero_pd();

				for ( k = 0; k < kc - 4; k += 4)
				{
					//cout << "k " << k << endl;
					unsigned int mkck = mkc + k;
					unsigned int nkck = nkc + k;

					__m256d a = _mm256_loadu_pd(&AP[mkck]);
					__m256d b = _mm256_loadu_pd(&BP[nkck]);


					acc = _mm256_add_pd(acc, _mm256_mul_pd(a, b));


				}
				_mm256_store_pd(&accp[0], acc);
				cmn += accp[0] + accp[1] + accp[2]+ accp[3];
			}
			//cout << "after loop " << endl;
	        // in case there are remaining cells
	        for (k; k < kc; k++)
	        {
		        cmn += AP[mkc + k] * BP[nkc + k];
	        }

            C[n * lda + m] = cmn;
        }
    }
}

void gebp_var1(unsigned int lda, double* A, double* BP, double* C, unsigned int kc, unsigned int mc)
{

	// packing A
	pack_A(lda, A, AP, kc, mc);
    for(unsigned int nri = 0; nri < lda; nri += NR)
    {
        // checking nr edges
        unsigned int nr = min(NR, lda - nri );
        //do_block(lda, A, B + nri * lda, C + nri * lda, kc, mc, nr);
        do_block(lda, AP, BP + nri * kc, C + nri * lda, kc, mc, nr);
    }

}

void gepp_var1(unsigned int lda, double* A, double* B, double* C, unsigned int kc)
{
    /**
    * Slices the matrices A and C by its M dimension
    */

    // packing B
    pack_B(lda, B, BP, kc);
    for(unsigned int mci = 0; mci < lda; mci += MC)
    {
        // checking mc edges
        unsigned int mc = min(MC, lda - mci );
        gebp_var1(lda, A + mci, BP, C + mci, kc, mc);
    }
}

void gemm_var1(unsigned int lda, double* A, double* B, double* C)
{
    /**
    * Slices the matrices A and B by its K dimension
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
    // allocates BP for its possible maximum size
    BP = (double*) aligned_alloc (32, lda * KC * sizeof(double));
	AP = (double*) aligned_alloc (32, MC * KC * sizeof(double));

	gemm_var1(lda, A, B, C);

	free(BP);
	free(AP);

}

