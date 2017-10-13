const char* dgemm_desc = "dgemm custom implementation";

#include <stdlib.h>	 /* malloc, free, rand */
#include <immintrin.h>



#define min(a,b) (((a)<(b))?(a):(b))


//Combination 1
//unsigned int KC = 512;
//unsigned int MC = 32;

//Combination 2
unsigned int KC = 256;
unsigned int MC = 64;

unsigned int NR = 4;
unsigned int MR = 8;

unsigned int PADD = 4;
unsigned int current_padding = 0;

// packed B and A
double* BP;
double* AP;
double* CP;


static void pack_A(unsigned int lda, double* original, double* packed, unsigned int kc, unsigned int mc, unsigned int padding)
{
	// global variable to keep track of the current padding :S ugly
	current_padding = padding;
	unsigned int m;

	for( unsigned int k = 0; k < kc; k++)
	{
		for(m = 0; m < mc - padding; m++)
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

static void pack_C(unsigned int lda, double* original, double* packed, unsigned int mr, unsigned int nr, unsigned int padding)
{

	unsigned int m;
	for(unsigned int n = 0; n < nr; n++)
	{
		for(m = 0; m < mr - padding; m++)
		{
			packed[n * mr + m] = original[n * lda + m];
		}
	}
}

static void unpack_C(unsigned int lda, double* original, double* packed, unsigned int mr, unsigned int nr, unsigned int padding)
{

	for(unsigned int n = 0; n < nr; n++)
	{
		for(unsigned int m = 0; m < mr - padding; m++)
		{
			original[n * lda + m] = packed[n * mr + m];
		}
	}
}

static void compute_kernel_mn84(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc)
{

	__m256d ay0;
	__m256d ay1;

	__m256d bx;

	__m256d cx0y0 = _mm256_load_pd(&cp[0]);
	__m256d cx0y1 = _mm256_load_pd(&cp[4]);

	__m256d cx1y0 = _mm256_load_pd(&cp[8]);
	__m256d cx1y1 = _mm256_load_pd(&cp[12]);

	__m256d cx2y0 = _mm256_load_pd(&cp[16]);
	__m256d cx2y1 = _mm256_load_pd(&cp[20]);

	__m256d cx3y0 = _mm256_load_pd(&cp[24]);
	__m256d cx3y1 = _mm256_load_pd(&cp[28]);

	for (unsigned int k = 0; k < kc; k++, bp++, ap += mc)
	{
		ay0 = _mm256_load_pd(&ap[0]);
		ay1 = _mm256_load_pd(&ap[4]);

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

	_mm256_store_pd(&cp[0], cx0y0);
	_mm256_store_pd(&cp[4], cx0y1);
	_mm256_store_pd(&cp[8], cx1y0);
	_mm256_store_pd(&cp[12], cx1y1);
	_mm256_store_pd(&cp[16], cx2y0);
	_mm256_store_pd(&cp[20], cx2y1);
	_mm256_store_pd(&cp[24], cx3y0);
	_mm256_store_pd(&cp[28], cx3y1);

}

static void compute_kernel_mn83(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc)
{

	__m256d ay0;
	__m256d ay1;

	__m256d bx;

	__m256d cx0y0 = _mm256_load_pd(&cp[0]);
	__m256d cx0y1 = _mm256_load_pd(&cp[4]);

	__m256d cx1y0 = _mm256_load_pd(&cp[8]);
	__m256d cx1y1 = _mm256_load_pd(&cp[12]);

	__m256d cx2y0 = _mm256_load_pd(&cp[16]);
	__m256d cx2y1 = _mm256_load_pd(&cp[20]);

	for (unsigned int k = 0; k < kc; k++, bp++, ap += mc)
	{
		ay0 = _mm256_load_pd(&ap[0]);
		ay1 = _mm256_load_pd(&ap[4]);

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

	}

	_mm256_store_pd(&cp[0], cx0y0);
	_mm256_store_pd(&cp[4], cx0y1);
	_mm256_store_pd(&cp[8], cx1y0);
	_mm256_store_pd(&cp[12], cx1y1);
	_mm256_store_pd(&cp[16], cx2y0);
	_mm256_store_pd(&cp[20], cx2y1);

}

static void compute_kernel_mn82(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc)
{

	__m256d ay0;
	__m256d ay1;

	__m256d bx;

	__m256d cx0y0 = _mm256_load_pd(&cp[0]);
	__m256d cx0y1 = _mm256_load_pd(&cp[4]);

	__m256d cx1y0 = _mm256_load_pd(&cp[8]);
	__m256d cx1y1 = _mm256_load_pd(&cp[12]);

	for (unsigned int k = 0; k < kc; k++, bp++, ap += mc)
	{
		ay0 = _mm256_load_pd(&ap[0]);
		ay1 = _mm256_load_pd(&ap[4]);

		// one element of B per column
		bx = _mm256_set1_pd(bp[0]);
		cx0y0 = _mm256_add_pd(cx0y0, _mm256_mul_pd(ay0, bx));
		cx0y1 = _mm256_add_pd(cx0y1, _mm256_mul_pd(ay1, bx));

		bx = _mm256_set1_pd(bp[kc]);
		cx1y0 = _mm256_add_pd(cx1y0, _mm256_mul_pd(ay0, bx));
		cx1y1 = _mm256_add_pd(cx1y1, _mm256_mul_pd(ay1, bx));

	}

	_mm256_store_pd(&cp[0], cx0y0);
	_mm256_store_pd(&cp[4], cx0y1);
	_mm256_store_pd(&cp[8], cx1y0);
	_mm256_store_pd(&cp[12], cx1y1);

}

static void compute_kernel_mn81(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc)
{

	__m256d ay0;
	__m256d ay1;

	__m256d bx;

	__m256d cx0y0 = _mm256_load_pd(&cp[0]);
	__m256d cx0y1 = _mm256_load_pd(&cp[4]);

	for (unsigned int k = 0; k < kc; k++, bp++, ap += mc)
	{
		ay0 = _mm256_load_pd(&ap[0]);
		ay1 = _mm256_load_pd(&ap[4]);

		// one element of B per column
		bx = _mm256_set1_pd(bp[0]);
		cx0y0 = _mm256_add_pd(cx0y0, _mm256_mul_pd(ay0, bx));
		cx0y1 = _mm256_add_pd(cx0y1, _mm256_mul_pd(ay1, bx));


	}

	_mm256_store_pd(&cp[0], cx0y0);
	_mm256_store_pd(&cp[4], cx0y1);

}

static void compute_kernel_mn44(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc)
{

	__m256d ay0;

	__m256d bx;

	__m256d cx0 = _mm256_load_pd(&cp[0]);
	__m256d cx1 = _mm256_load_pd(&cp[4]);
	__m256d cx2 = _mm256_load_pd(&cp[8]);
	__m256d cx3 = _mm256_load_pd(&cp[12]);

	for (unsigned int k = 0; k < kc; k++, bp++, ap += mc)
	{
		ay0 = _mm256_load_pd(&ap[0]);
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

	_mm256_store_pd(&cp[0], cx0);
	_mm256_store_pd(&cp[4], cx1);
	_mm256_store_pd(&cp[8], cx2);
	_mm256_store_pd(&cp[12], cx3);

}

static void compute_kernel_mn43(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc)
{

	__m256d ay0;

	__m256d bx;

	__m256d cx0 = _mm256_load_pd(&cp[0]);
	__m256d cx1 = _mm256_load_pd(&cp[4]);
	__m256d cx2 = _mm256_load_pd(&cp[8]);

	for (unsigned int k = 0; k < kc; k++, bp++, ap += mc)
	{
		ay0 = _mm256_load_pd(&ap[0]);
		// one element of B per column
		bx = _mm256_set1_pd(bp[0]);
		cx0 = _mm256_add_pd(cx0, _mm256_mul_pd(ay0, bx));

		bx = _mm256_set1_pd(bp[kc]);
		cx1 = _mm256_add_pd(cx1, _mm256_mul_pd(ay0, bx));

		bx = _mm256_set1_pd(bp[2 * kc]);
		cx2 = _mm256_add_pd(cx2, _mm256_mul_pd(ay0, bx));
	}

	_mm256_store_pd(&cp[0], cx0);
	_mm256_store_pd(&cp[4], cx1);
	_mm256_store_pd(&cp[8], cx2);

}

static void compute_kernel_mn42(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc)
{

	__m256d ay0;

	__m256d bx;

	__m256d cx0 = _mm256_load_pd(&cp[0]);
	__m256d cx1 = _mm256_load_pd(&cp[4]);

	for (unsigned int k = 0; k < kc; k++, bp++, ap += mc)
	{
		ay0 = _mm256_load_pd(&ap[0]);
		// one element of B per column
		bx = _mm256_set1_pd(bp[0]);
		cx0 = _mm256_add_pd(cx0, _mm256_mul_pd(ay0, bx));

		bx = _mm256_set1_pd(bp[kc]);
		cx1 = _mm256_add_pd(cx1, _mm256_mul_pd(ay0, bx));
	}

	_mm256_store_pd(&cp[0], cx0);
	_mm256_store_pd(&cp[4], cx1);

}

static void compute_kernel_mn41(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc)
{

	__m256d ay0;

	__m256d bx;

	__m256d cx0 = _mm256_load_pd(&cp[0]);

	for (unsigned int k = 0; k < kc; k++, bp++, ap += mc)
	{
		ay0 = _mm256_load_pd(&ap[0]);
		// one element of B per column
		bx = _mm256_set1_pd(bp[0]);
		cx0 = _mm256_add_pd(cx0, _mm256_mul_pd(ay0, bx));

	}

	_mm256_store_pd(&cp[0], cx0);

}

static void compute_fallback_kernel(double* ap, double* bp, double* cp, unsigned int kc, unsigned int mc,  unsigned int mr, unsigned int nr)
{
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
				case 1:
					compute_kernel_mn41(ap, bp, cp, kc, mc);
					break;
				case 2:
					compute_kernel_mn42(ap, bp, cp, kc, mc);
					break;
				case 3:
					compute_kernel_mn43(ap, bp, cp, kc, mc);
					break;
				case 4:
					compute_kernel_mn44(ap, bp, cp, kc, mc);
					break;
				default:
					// we shouldn't get here
					compute_fallback_kernel(ap , bp, cp, kc, mc, mr, nr);
			}
			break;
		case 8:
			switch(nr)
			{
				case 1:
					compute_kernel_mn81(ap, bp, cp, kc, mc);
					break;
				case 2:
					compute_kernel_mn82(ap, bp, cp, kc, mc);
					break;
				case 3:
					compute_kernel_mn83(ap, bp, cp, kc, mc);
					break;
				case 4:
					compute_kernel_mn84(ap, bp, cp, kc, mc);
					break;
				default:
					// we shouldn't get here
					compute_fallback_kernel(ap , bp, cp, kc, mc, mr, nr);
			}
			break;
		default:
			// we shouldn't get here
			compute_fallback_kernel(ap , bp, cp, kc, mc, mr, nr);
	}


}

static void compute_block(unsigned int lda, double* ap, double* bp, double* c, unsigned int kc, unsigned int mc, unsigned int nr)
{

	unsigned mri, mr;
	for(mri = 0; mri < mc; mri += MR)
	{
		mr = min(MR, mc - mri );
		// checks edges when there is padding on MC to not get out of bounds while reading from and writing to C
		if(mri + MR < mc)
		{
			pack_C(lda, c + mri, CP, mr, nr, 0);
			do_kernel(ap + mri, bp, CP, kc, mc, mr, nr);
			unpack_C(lda, c + mri, CP, mr, nr, 0);
		}
		else
		{
			pack_C(lda, c + mri, CP, mr, nr, current_padding);
			do_kernel(ap + mri, bp, CP, kc, mc, mr, nr);
			unpack_C(lda, c + mri, CP, mr, nr, current_padding);
		}
	}
}

static void gebpvar1(unsigned int lda, double* ap, double* bp, double* c, unsigned int kc, unsigned int mc)
{
	unsigned nri, nr;
	for(nri = 0; nri < lda; nri += NR)
	{
		nr = min(NR, lda - nri );
		compute_block(lda, ap, bp + nri * kc, c + nri * lda, kc, mc, nr);
	}

}

static void geppvar1(unsigned int lda, double* a, double* bp, double* c, unsigned int kc)
{
	unsigned int mci, mc, mc2;
	for(mci = 0; mci < lda; mci += MC)
	{
		// controlling AP padding in order to get data aligned
		mc = min(MC, lda - mci );
		mc2 = mc;
		mc += PADD - (mc % PADD);
		pack_A(lda, a + mci, AP, kc, mc, PADD - (mc2 % PADD));
		gebpvar1(lda, AP, bp, c + mci, kc, mc);
	}
}

void square_dgemm (int lda, double* A, double* B, double* C)
{
	// allocates BP for its possible maximum size
	BP = _mm_malloc (lda * KC * sizeof(double), 32);
	// allocates AP for its possible maximum size
	AP = _mm_malloc ((MC + (PADD - MC % PADD)) * KC * sizeof(double), 32); // enough to add zero padding
	// allocates CP for its possible maximum size
	CP = _mm_malloc (MR * NR * sizeof(double), 32);

	//gemm-var-1
	unsigned int kc, kci;
	for (kci = 0; kci < lda; kci += KC)
	{
		kc = min(KC, lda - kci );
		pack_B(lda, B + kci, BP, kc);
		geppvar1(lda, A + kci * lda, BP, C, kc);
	}

	_mm_free(BP);
	_mm_free(AP);
	_mm_free(CP);

}

