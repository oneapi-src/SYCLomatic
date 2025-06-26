// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

struct LessThan {
  int compare;
  inline LessThan(int compare) : compare(compare) {}
  __device__ bool operator()(const int &a) const {
    return (a < compare);
  }
};

void test(int *d_in, int *d_out, int *d_num_selected_out, int num_items, LessThan select_op) {
  // Start
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DevicePartition::If(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_in/*int **/, d_out/*int **/, d_num_selected_out/*int **/, num_items/*int*/, select_op/*SelectOp*/);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DevicePartition::If(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_in/*int **/, d_out/*int **/, d_num_selected_out/*int **/, num_items/*int*/, select_op/*SelectOp*/);
  // End
}
