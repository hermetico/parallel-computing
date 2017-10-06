const char* dgemm_desc = "dgemm custom implementation";
#include <iostream>
#include <stdlib.h>	 /* malloc, free, rand */
using namespace std;

#if !defined(KC)
#define KC 256
#endif

#if !defined(MC)
#define MC 64
#endif

#if !defined(NR)
#define NR 4
#endif

#if !defined(MR)
#define MR 8
#endif


/*
#if !defined(KC)
#define KC 5
#endif

#if !defined(MC)
#define MC 4
#endif

#if !defined(NR)
#define NR 3
#endif

#if !defined(MR)
#define MR 2
#endif
*/
#define min(a,b) (((a)<(b))?(a):(b))

// packed B and A
double* BP;
double* AP;
double* CP;


static void pack_A(unsigned int lda, double* original, double* packed, unsigned int kc, unsigned int mc )
{
	/*
	// uses zero padding
	cout << "pack A " <<endl;
	unsigned int k, m;
	for( k = 0; k < kc; k++)
	{
		for(m = 0; m < mc; m++)
		{
			packed[k * MC + m] = original[k * lda + m];
			cout << k * MC + m <<" -- "<< k * lda + m << " : "<< original[k * lda + m] <<endl;
		}
		for(m; m<MC; m++)
		{
			cout << k * MC + m <<" -- "<< 0 <<endl;
			packed[k * MC + m] = 0;
		}

	}

	cout << endl;
	*/
	for( unsigned int k = 0; k < kc; k++)
	{
		for(unsigned int m = 0; m < mc; m++)
		{
			packed[k * mc + m] = original[k * lda + m];
			//cout << k * mc + m <<" -- "<< k * lda + m << " : "<< original[k * lda + m] <<endl;
		}

	}
}

static void pack_B(unsigned int lda, double* original, double* packed, unsigned int kc )
{
	for(unsigned int i = 0; i < lda; i++)
	{
		for(unsigned int k = 0; k < kc; k++)
		{
			packed[i * kc + k] = original[i * lda + k];
		}
	}
}

static void pack_C(unsigned int lda, double* original, double* packed, unsigned int mr, unsigned int nr )
{
	//cout << "pack C" << endl;
	for(unsigned int n = 0; n < nr; n++)
	{
		for(unsigned int m = 0; m < mr; m++)
		{
			packed[n * mr + m] = original[n * lda + m];
			//cout <<n * mr + m <<" -- "<< n * lda + m << " : "<< original[n * lda + m] <<endl;
		}
	}
	//cout << endl;
}

static void unpack_C(unsigned int lda, double* original, double* packed, unsigned int mr, unsigned int nr )
{
	//cout << "unpack C" << endl;
	for(unsigned int n = 0; n < nr; n++)
	{
		for(unsigned int m = 0; m < mr; m++)
		{
			original[n * lda + m] = packed[n * mr + m];
			//cout <<n * lda + m <<" -- "<< n * mr + m << " : "<< packed[n * mr + m] <<endl;

		}
	}
	//cout << endl;
}

static void do_kernel(unsigned int lda, double* AP, double* BP, double* CP, unsigned int kc, unsigned int mc,  unsigned int mr, unsigned int nr)
{

	/*for (unsigned int m = 0; m < mr; m++)
	{
		for (unsigned int n = 0; n < nr; n++)
		{
			double cmn = CP[n * mr + m];

			for (unsigned int k = 0; k < kc; k++)
			{
				cmn += AP[m * kc + k ] * BP[n * kc + k];
			}
			CP[n * mr + m] = cmn;
		}
	}*/

	/*for (unsigned int k = 0; k < kc; k++)
	{
		for( unsigned int n = 0; n < nr; n++)
		{
			double fixedB = BP[n * kc + k];
			for (unsigned int m = 0; m < mr; m++)
			{
				CP[ k * mr + m] += AP[ k * mr + m] *fixedB;
			}
		}
	}
	 */
	//cout << "mr " << mr <<  endl;
	for (unsigned int k = 0; k < kc; k++)
	{
		//cout << "k "<< k << endl;
		for (unsigned int n = 0; n < nr; n++)
		{
			double fixedB = BP[n * kc + k];
			//cout << endl;
			//cout << "fixed " << fixedB <<  endl;
			for (unsigned int m = 0; m < mr; m++)
			{
				//cout << n * lda + m << " < -- AP[" << k * mr + m << "] "<< AP[k * mr + m]<<  endl;
				CP[n * lda + m] += AP[k * mc + m] * fixedB;
			}
		}
	}


}

static void do_block(unsigned int lda, double* AP, double* BP, double* c, unsigned int kc, unsigned int mc, unsigned int nr)
{

	/**
	* Slices the matrices A and C by its M dimension
	*/
	for(unsigned int mri = 0; mri < mc; mri += MR)
	{
		// checking mr edges
		unsigned int mr = min(MR, mc - mri );

		//pack_C(lda, C + mri, CP, mr, nr);

		do_kernel(lda, AP + mri, BP, c + mri, kc, mc, mr, nr);

		//unpack_C(lda, C + mri, CP, mr, nr);
	}
}

static void gebp_var1(unsigned int lda, double* ap, double* bp, double* c, unsigned int kc, unsigned int mc)
{
	/**
	* Slices the matrices B and C by its N dimension
	*/
	for(unsigned int nri = 0; nri < lda; nri += NR)
	{
		// checking nr edges
		unsigned int nr = min(NR, lda - nri );
		do_block(lda, ap, bp + nri * kc, c + nri * lda, kc, mc, nr);
	}

}

static void gepp_var1(unsigned int lda, double* a, double* bp, double* c, unsigned int kc)
{
	/**
	* Slices the matrices A and C by its M dimension
	*/
	for(unsigned int mci = 0; mci < lda; mci += MC)
	{
		// checking mc edges
		unsigned int mc = min(MC, lda - mci );

		pack_A(lda, a + mci, AP, kc, mc);

		gebp_var1(lda, AP, bp, c + mci, kc, mc);
	}
}

static void gemm_var1(unsigned int lda, double* a, double* b, double* c)
{
	/**
	* Slices the matrices A and B by its K dimension
	*/
	for (unsigned int kci = 0; kci < lda; kci += KC)
	{
		// checking kc edges
		unsigned int kc = min(KC, lda - kci );

		pack_B(lda, b + kci, BP, kc);

		gepp_var1(lda, a + kci * lda, BP, c, kc);
	}
}

void square_dgemm (int lda, double* A, double* B, double* C)
{
	// allocates BP for its possible maximum size
	BP = (double*) malloc (lda * KC * sizeof(double));
	// allocates AP for its possible maximum size
	AP = (double*) malloc (MC * KC * sizeof(double));
	CP = (double*) malloc (MC * NR * sizeof(double));

	gemm_var1(lda, A, B, C);

	free(BP);
	free(AP);

}

