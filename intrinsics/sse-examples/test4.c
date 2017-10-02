//gcc -Wimplicit-function-declaration -std=gnu99  -msse3 test4.c -lm


#include <stdio.h>
#include <pmmintrin.h>
#include <time.h>   
#include <math.h>   

void normal(float* a, int N)                                           
{                                                                        
  for (int i = 0; i < N; ++i)                                              
    a[i] = sqrt(a[i]);                                                 
}                                                                       
void sse(float* a, int N)                                              
{                      
  // We assume N % 4 == 0.                                             

  int nb_iters = N / 4;                                                  
  __m128* ptr = (__m128*)a;                                              
  for (int i = 0; i < nb_iters; ++i, ++ptr, a += 4)                        
    _mm_store_ps(a, _mm_sqrt_ps(*ptr));                                
}                                                                                                                                                                                                                
 
int main(int argc, char** argv)                                        
{  
  clock_t t1, t2;
  if (argc != 2)                                                           
    return 1;                                                            
  int N = atoi(argv[1]);                                                

  float* a;                                                              
  posix_memalign((void**)&a, 16,  N * sizeof(float));                   
  for (int i = 0; i < N; ++i)                                              
    a[i] = 3141592.65358;                                                
  {                                                                      
    t1=clock();
    normal(a, N);                                                        
    t2=clock();

    float diff = (((float)t2 - (float)t1) / 1000000.0F ) * 1000;   
    printf("time : %f\n",diff);  

  }                                                                                                                                             
  for (int i = 0; i < N; ++i)                                              
    a[i] = 3141592.65358;                                                
  {                                                                        
    t1=clock();
    sse(a, N);                                                           
    t2=clock();
    float diff = (((float)t2 - (float)t1) / 1000000.0F ) * 1000;   
    printf("time : %f\n",diff);  
  }
}
