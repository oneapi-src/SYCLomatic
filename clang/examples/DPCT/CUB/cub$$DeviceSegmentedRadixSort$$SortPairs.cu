// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(int n, int *d_keys_in, int *d_keys_out, int *d_values_in, int *d_values_out, int num_segments, int *d_offsets) {
  // Start
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes;
  cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
  // End
}
