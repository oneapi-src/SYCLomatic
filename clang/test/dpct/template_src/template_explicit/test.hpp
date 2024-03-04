#include <cuda.h>

template<bool haveFreshList>
__global__ void nbnxn_kernel_prune_cuda(int numParts)
#ifdef TEST
	;
// CHECK: extern template void nbnxn_kernel_prune_cuda<true>(int,
// CHECK:                                                             const sycl::nd_item<3> &item_ct1);
// CHECK: extern template void nbnxn_kernel_prune_cuda<false>(int,
// CHECK:                                                             const sycl::nd_item<3> &item_ct1);
extern template __global__ void nbnxn_kernel_prune_cuda<true>(int);
extern template __global__ void nbnxn_kernel_prune_cuda<false>(int);
#else 
{    if (haveFreshList) {
bool a=	    blockIdx.x > numParts;
        // Code specific to having a fresh list
    } else {
        // Code for when there isn't a fresh list
    }
}
#endif
