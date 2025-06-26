// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

struct CustomOpT {
  template <typename DataType>
  __device__ bool operator()(const DataType &lhs, const DataType &rhs) {
    return lhs <= rhs;
  }
};

void test(int num_items, int *d_keys, int *d_values, CustomOpT op) {
  // Start
  void *temp_storage = nullptr;
  size_t temp_storage_size;
  cub::DeviceMergeSort::StableSortPairs(temp_storage/*void **/, temp_storage_size/*size_t*/, d_keys/*int **/, d_values/*int **/, num_items/*int*/, op/*CustomOpT*/);
  cudaMalloc(&temp_storage, temp_storage_size);
  cub::DeviceMergeSort::StableSortPairs(temp_storage/*void **/, temp_storage_size/*size_t*/, d_keys/*int **/, d_values/*int **/, num_items/*int*/, op/*CustomOpT*/);
  // End
}
