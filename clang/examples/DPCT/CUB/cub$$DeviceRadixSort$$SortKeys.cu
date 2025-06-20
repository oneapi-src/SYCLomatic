// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(int n, int *d_keys_in, int *d_keys_out) {
  // Start
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes;
  cub::DeviceRadixSort::SortKeys(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceRadixSort::SortKeys(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/);
  // End
}
