// Option: --use-experimental-features=matrix
#include <mma.h>

__global__ void test() {
  // Start
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
      a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                         nvcuda::wmma::col_major>
      b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
  nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  // End
}
