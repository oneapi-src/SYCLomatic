// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(int *d_in, int *d_flags, int *d_out, int *d_num_selected_out, int num_items) {
  // Start
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DevicePartition::Flagged(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_in/*int **/, d_flags/*int **/, d_out/*int **/, d_num_selected_out/*int **/, num_items/*int*/);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DevicePartition::Flagged(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_in/*int **/, d_flags/*int **/, d_out/*int **/, d_num_selected_out/*int **/, num_items/*int*/);
  // End
}
