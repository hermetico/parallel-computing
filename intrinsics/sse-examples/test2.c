  #include <pmmintrin.h>
  int main(){
    __m128 __A, __B,
      result;
    __A = _mm_set_ps(23.3,
		     43.7, 234.234, 98.746);
    __B = _mm_set_ps(15.4,
		     34.3, 4.1, 8.6);
    result = _mm_add_ps(__A,__B);
    return 0;
  }
