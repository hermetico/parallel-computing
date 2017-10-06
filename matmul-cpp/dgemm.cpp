const char* dgemm_desc = "dgemm custom implementation";
#include <iostream>
#include <stdlib.h>	 /* malloc, free, rand */
#include <immintrin.h>
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


#define min(a,b) (((a)<(b))?(a):(b))

// packed B and A
double* BP;
double* AP;
double* CP;


static void pack_A(unsigned int lda, double* original, double* packed, unsigned int kc, unsigned int mc )
{

	for( unsigned int k = 0; k < kc; k++)
	{
		for(unsigned int m = 0; m < mc; m++)
		{
			packed[k * mc + m] = original[k * lda + m];
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

	for(unsigned int n = 0; n < nr; n++)
	{
		for(unsigned int m = 0; m < mr; m++)
		{
			packed[n * mr + m] = original[n * lda + m];
		}
	}
}

static void unpack_C(unsigned int lda, double* original, double* packed, unsigned int mr, unsigned int nr )
{

	for(unsigned int n = 0; n < nr; n++)
	{
		for(unsigned int m = 0; m < mr; m++)
		{
			original[n * lda + m] = packed[n * mr + m];
		}
	}
}



static void compute_kernel_mn84(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc) {

	__m256d ay0;
	__m256d ay1;

	__m256d bx;

	__m256d cx0y0 = _mm256_loadu_pd(&cp[0]);
	__m256d cx0y1 = _mm256_loadu_pd(&cp[4]);

	__m256d cx1y0 = _mm256_loadu_pd(&cp[8]);
	__m256d cx1y1 = _mm256_loadu_pd(&cp[12]);

	__m256d cx2y0 = _mm256_loadu_pd(&cp[16]);
	__m256d cx2y1 = _mm256_loadu_pd(&cp[20]);

	__m256d cx3y0 = _mm256_loadu_pd(&cp[24]);
	__m256d cx3y1 = _mm256_loadu_pd(&cp[28]);

	for (unsigned int k = 0; k < kc; k++, bp++, ap += mc)
	{
		ay0 = _mm256_loadu_pd(&ap[0]);
		ay1 = _mm256_loadu_pd(&ap[4]);

		// one element of B per column
		bx = _mm256_set1_pd(bp[0]);
		cx0y0 = _mm256_add_pd(cx0y0, _mm256_mul_pd(ay0, bx));
		cx0y1 = _mm256_add_pd(cx0y1, _mm256_mul_pd(ay1, bx));

		bx = _mm256_set1_pd(bp[kc]);
		cx1y0 = _mm256_add_pd(cx1y0, _mm256_mul_pd(ay0, bx));
		cx1y1 = _mm256_add_pd(cx1y1, _mm256_mul_pd(ay1, bx));


		bx = _mm256_set1_pd(bp[2 * kc]);
		cx2y0 = _mm256_add_pd(cx2y0, _mm256_mul_pd(ay0, bx));
		cx2y1 = _mm256_add_pd(cx2y1, _mm256_mul_pd(ay1, bx));

		bx = _mm256_set1_pd(bp[3 * kc]);
		cx3y0 = _mm256_add_pd(cx3y0, _mm256_mul_pd(ay0, bx));
		cx3y1 = _mm256_add_pd(cx3y1, _mm256_mul_pd(ay1, bx));

	}

	_mm256_storeu_pd(&cp[0], cx0y0);
	_mm256_storeu_pd(&cp[4], cx0y1);
	_mm256_storeu_pd(&cp[8], cx1y0);
	_mm256_storeu_pd(&cp[12], cx1y1);
	_mm256_storeu_pd(&cp[16], cx2y0);
	_mm256_storeu_pd(&cp[20], cx2y1);
	_mm256_storeu_pd(&cp[24], cx3y0);
	_mm256_storeu_pd(&cp[28], cx3y1);

}

static void compute_kernel_mn4(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc)
{

	__m256d ay0;

	__m256d bx;

	__m256d cx0 = _mm256_loadu_pd(&cp[0]);
	__m256d cx1 = _mm256_loadu_pd(&cp[4]);
	__m256d cx2 = _mm256_loadu_pd(&cp[8]);
	__m256d cx3 = _mm256_loadu_pd(&cp[12]);

	for (unsigned int k = 0; k < kc; k++, bp++, ap += mc)
	{
		ay0 = _mm256_loadu_pd(&ap[0]);
		// one element of B per column
		bx = _mm256_set1_pd(bp[0]);
		cx0 = _mm256_add_pd(cx0, _mm256_mul_pd(ay0, bx));

		bx = _mm256_set1_pd(bp[kc]);
		cx1 = _mm256_add_pd(cx1, _mm256_mul_pd(ay0, bx));

		bx = _mm256_set1_pd(bp[2 * kc]);
		cx2 = _mm256_add_pd(cx2, _mm256_mul_pd(ay0, bx));

		bx = _mm256_set1_pd(bp[3 * kc]);
		cx3 = _mm256_add_pd(cx3, _mm256_mul_pd(ay0, bx));

	}

	_mm256_storeu_pd(&cp[0], cx0);
	_mm256_storeu_pd(&cp[4], cx1);
	_mm256_storeu_pd(&cp[8], cx2);
	_mm256_storeu_pd(&cp[12], cx3);


}

static void compute_fallback_kernel(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc,  unsigned int mr, unsigned int nr){

	for (unsigned int k = 0; k < kc; k++)
	{
		for (unsigned int n = 0; n < nr; n++)
		{
			double fixedB = bp[n * kc + k];
			for (unsigned int m = 0; m < mr; m++)
			{
				// AP jumpes mc while CP jumpes just mr!!
				cp[n * mr + m] += ap[k * mc + m] * fixedB;
			}
		}
	}
}

static void do_kernel(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc,  unsigned int mr, unsigned int nr)
{


	switch(mr)
	{
		case 4:

			switch(nr)
			{
				case 4:
					compute_kernel_mn4(ap, bp, cp, kc, mc);
					break;

				default:
					compute_fallback_kernel(ap , bp, CP, kc, mc, mr, nr);
			}
		break;

		case 8:

			switch(nr)
			{
				case 4:
					compute_kernel_mn84(ap, bp, cp, kc, mc);
					break;
				default:
					compute_fallback_kernel(ap , bp, CP, kc, mc, mr, nr);
			}
			break;

		default:
			compute_fallback_kernel(ap , bp, CP, kc, mc, mr, nr);
	}


}

static void do_block(unsigned int lda, double* ap, double* bp, double* c, unsigned int kc, unsigned int mc, unsigned int nr)
{

	/**
	* Slices the matrices A and C by its M dimension
	*/
	for(unsigned int mri = 0; mri < mc; mri += MR)
	{
		// checking mr edges
		unsigned int mr = min(MR, mc - mri );

		pack_C(lda, c + mri, CP, mr, nr);

		do_kernel(ap + mri, bp, CP, kc, mc, mr, nr);

		unpack_C(lda, c + mri, CP, mr, nr);
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
	// allocates CP for its possible maximum size
	CP = (double*) malloc (MC * NR * sizeof(double));

	gemm_var1(lda, A, B, C);

	free(BP);
	free(AP);

}

