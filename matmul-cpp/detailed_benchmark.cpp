#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <float.h>
#include <math.h>

#include <sys/time.h>

//#define MAX_SPEED 19.2 // defining Max Gflops/s per core on Edison.
#define MAX_SPEED 24 // defining Max Gflops/s per core on intel i-7 4500U
//#define MAX_SPEED 27.2 // defining Max Gflops/s per core on intel i-7 2600K

//NOTE: #include <acml.h> //assumes AMD platform
extern "C" {
#include <cblas.h> //uses general CBLAS interface
}

//
//  Your function must have the following signature:
//
extern const char* dgemm_desc;
extern void square_dgemm( int M, double *A, double *B, double *C );

//
//  Helper functions
//

double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }

    gettimeofday( &end, NULL );

    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

void fill( double *p, int n )
{
    for( int i = 0; i < n; i++ )
        p[i] = 2 * drand48( ) - 1;
}

void absolute_value( double *p, int n )
{
    for( int i = 0; i < n; i++ )
        p[i] = fabs( p[i] );
}

//
//  The benchmarking program
//
int main( int argc, char **argv )
{
    printf ("Description:\t%s\n\n", dgemm_desc);

    //
    // These sizes should highlight performance dips at multiples of certain
    // powers-of-two
    //
    int test_sizes[] = {
        31, 32, 96, 97, 127, 128, 129, 191, 192, 229, 255, 256, 257,
        319, 320, 321, 417, 479, 480, 511, 512, 639, 640, 767, 768, 769,
    };

    int nsizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    double Mflops_s[nsizes], per[nsizes], aveper;

    for( int isize = 0; isize < sizeof(test_sizes)/sizeof(test_sizes[0]); isize++ )
    {
        int n = test_sizes[isize];

        double *A = (double*) malloc( n * n * sizeof(double) );
        double *B = (double*) malloc( n * n * sizeof(double) );
        double *C = (double*) malloc( n * n * sizeof(double) );

        fill( A, n * n );
        fill( B, n * n );
        fill( C, n * n );
        
        //
        //  measure Mflop/s rate
        //  time a sufficiently long sequence of calls to eliminate noise
        //
        double Mflop_s, seconds = -1.0;
        for( int n_iterations = 1; seconds < 0.1; n_iterations *= 2 ) 
        {
            //
            //  warm-up
            //
            square_dgemm( n, A, B, C );
            
            //
            //  measure time
            //
            seconds = read_timer( );
            for( int i = 0; i < n_iterations; i++ )
                square_dgemm( n, A, B, C );
            seconds = read_timer( ) - seconds;
            
            //
            //  compute Mflop/s rate
            //
            Mflop_s = 2e-6 * n_iterations * n * n * n / seconds;
        }
        Mflops_s[isize] = Mflop_s;
        per[isize] = Mflop_s*100/(MAX_SPEED*1000);
        printf ("Size: %d\tMflop/s: %8g\tPercentage:%6.2lf\n", n,  Mflops_s[isize],per[isize]);
        
        //
        //  Ensure that error does not exceed the theoretical error bound
        //
        memset( C, 0, sizeof( double ) * n * n );
        square_dgemm( n, A, B, C );
        //NOTE: dgemm( 'N','N', n,n,n, -1, A,n, B,n, 1, C,n );
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n,n,n, -1, A,n, B,n, 1, C,n );
        absolute_value( A, n * n );
        absolute_value( B, n * n );
        absolute_value( C, n * n );
        //NOTE: dgemm( 'N','N', n,n,n, -3.0*DBL_EPSILON*n, A,n, B,n, 1, C,n );
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n,n,n, -3.0*DBL_EPSILON*n, A,n, B,n, 1, C,n );
        for( int i = 0; i < n * n; i++ )
            if( C[i] > 0 )
            {
                printf( "FAILURE: error in matrix multiply exceeds an acceptable margin\n" );
                exit(-1);
            }

        free( C );
        free( B );
        free( A );
    }

    /* Calculating average percentage of peak reached by algorithm */
    aveper=0;
    for (int i=0; i<nsizes;i++)
        aveper+= per[i];
    aveper/=nsizes*1.0;

    /* Printing average percentage and grade to screen */
    printf("Average percentage of Peak = %g\n",aveper);

    return 0;
}
