// Option: --use-experimental-features=matrix
#include <mma.h>

__global__ void test(half *a, int row, int col, int lda) {
  // Start
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
      a_frag;
  nvcuda::wmma::load_matrix_sync(a_frag /* type fragment */,
                                 a + col + row * lda, lda);
  // End
}
