c1 = _mm_loadu_pd( C+0*lda )	//load unaligned block in C
     c2 = _mm_loadu_pd( C+1*lda )
     for( int i = 0; i < 2; i++ )
{
  a = _mm_load_pd( A+i*lda )		//load aligned i-th column of A
  b1 = _mm_load1_pd( B+i+0*lda )	//load i-th row of B
  b2 = _mm_load1_pd( B+i+1*lda )
  c1=_mm_add_pd( c1, _mm_mul_pd( a, b1 ) ); //rank-1 update
  c2=_mm_add_pd( c2, _mm_mul_pd( a, b2 ) );
}
_mm_storeu_pd( C+0*lda, c1 );	//store unaligned block in C
_mm_storeu_pd( C+1*lda, c2 );

