// Option: --use-experimental-features=matrix
#include <mma.h>

template <typename T> __global__ void test(T val) {
  // Start
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
  nvcuda::wmma::fill_fragment(acc_frag, val /*const T&*/);
  // End
}
