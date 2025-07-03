// Option: --use-experimental-features=matrix
#include <mma.h>

__global__ void test() {
  // Start
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
  nvcuda::wmma::fill_fragment(acc_frag /* type fragment */,
                              1.0f /* type value */);
  // End
}
