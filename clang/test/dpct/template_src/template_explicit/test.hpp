#include <cuda.h>

template <bool isTrue>
__global__ void foo(int parm)
#ifdef TEST
    ;
// CHECK: extern template void foo<true>(int);
// CHECK: extern template void foo<false>(int);
extern template __global__ void foo<true>(int);
extern template __global__ void foo<false>(int);
#else
{
  if (isTrue)
    blockIdx.x > parm;
}
#endif
