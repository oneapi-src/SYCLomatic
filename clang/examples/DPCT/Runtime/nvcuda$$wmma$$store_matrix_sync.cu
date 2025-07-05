// Option: --use-experimental-features=matrix
#include <mma.h>

__global__ void test(float *c, int row, int col, int ldc) {
  // Start
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
  nvcuda::wmma::store_matrix_sync(
      c + col + row * ldc /*void **/, acc_frag, ldc /*int*/,
      nvcuda::wmma::mem_col_major /*memory order*/);
  nvcuda::wmma::store_matrix_sync(
      c + row + col * ldc /*void **/, acc_frag, ldc /*int*/,
      nvcuda::wmma::mem_row_major /*memory order*/);
  // End
}
