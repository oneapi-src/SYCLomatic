// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(int *d_keys_in, int *d_values_in, int *d_keys_out, int *d_values_out, int *d_num_selected_out, int num_items, cudaStream_t s) {
  // Start
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::UniqueByKey(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_values_in/*int **/, d_keys_out/*int **/, d_values_out/*int **/, d_num_selected_out/*int **/, num_items/*int*/, s/*cudaStream_t*/);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::UniqueByKey(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_values_in/*int **/, d_keys_out/*int **/, d_values_out/*int **/, d_num_selected_out/*int **/, num_items/*int*/, s/*cudaStream_t*/);
  // End
}
