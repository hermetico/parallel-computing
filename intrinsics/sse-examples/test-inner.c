#include <xmmintrin.h>
 
int main()
{
  int k,i;
  k = 100;
  float x[k]; float y[k]; // vectors of length k
  __m128 X, Y;
  // 128-bit values
  __m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
  float inner_prod, temp[4];
  for(i = 0; i < k - 4; i += 4) {
    X = _mm_load_ps(&x[i]); // load chunk of 4 floats
    Y = _mm_load_ps(y + i); // alternate way, pointer arithmetic
    acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
  }
  _mm_store_ps(&temp[0], acc); // store acc into an array of floats
  inner_prod = temp[0] + temp[1] + temp[2] + temp[3];
  // add the remaining values
  for(; i < k; i++)
    inner_prod += x[i] * y[i];
}
