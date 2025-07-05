// Option: --use-experimental-features=matrix
#include <mma.h>

template <typename T>
__global__ void test(const T *c, int row, int col, unsigned ldc) {
  // Start
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
  nvcuda::wmma::store_matrix_sync(
      c + col + row * ldc /*const T **/, acc_frag, ldc /*unsigned*/,
      nvcuda::wmma::mem_col_major /*nvcuda::wmma::layout_t*/);
  nvcuda::wmma::store_matrix_sync(
      c + row + col * ldc /*const T **/, acc_frag, ldc /*unsigned*/,
      nvcuda::wmma::mem_row_major /*nvcuda::wmma::layout_t*/);
  // End
}
