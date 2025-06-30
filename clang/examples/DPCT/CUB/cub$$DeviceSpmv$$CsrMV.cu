// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(float *d_values, int *d_row_offsets, int *d_column_indices, float *d_vector_x, float *d_vector_y, int num_rows, int num_cols, int num_nonzeros) {
  // Start
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSpmv::CsrMV(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_values/*float **/, d_row_offsets/*int **/, d_column_indices/*int **/, d_vector_x/*float **/, d_vector_y/*float **/, num_rows/*int*/, num_cols/*int*/, num_nonzeros/*int*/);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSpmv::CsrMV(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_values/*float **/, d_row_offsets/*int **/, d_column_indices/*int **/, d_vector_x/*float **/, d_vector_y/*float **/, num_rows/*int*/, num_cols/*int*/, num_nonzeros/*int*/);
  // End
}
