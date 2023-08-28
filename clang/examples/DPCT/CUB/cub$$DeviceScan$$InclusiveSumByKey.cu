// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

struct CustomEqual {
  template <typename T>
  __device__ __host__ inline bool operator()(const T &lhs, const T &rhs) const {
    return lhs == rhs;
  }
};

void test(void *temp_storage, size_t &temp_storage_bytes, int *d_keys_in, int * d_values_in,int *d_values_out,
          int num_items, CustomEqual equality_op, cudaStream_t stream) {
  // Start
  cub::DeviceScan::InclusiveSumByKey(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_keys_in/*KeysInputIteratorT*/, d_values_in/*ValuesInputIteratorT*/, d_values_out/*ValuesOutputIteratorT*/, num_items/*int*/, equality_op/*EqualityOpT*/, stream/*cudaStream_t*/);
  // End
}
