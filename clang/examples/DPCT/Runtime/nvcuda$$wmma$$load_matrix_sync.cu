// Option: --use-experimental-features=matrix
#include <mma.h>

template <typename T>
__global__ void test(const T *a, int row, int col, unsigned lda) {
  // Start
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
      a_frag;
  nvcuda::wmma::load_matrix_sync(a_frag, a + col + row * lda /*const T **/,
                                 lda /*unsigned*/);
  // End
}
