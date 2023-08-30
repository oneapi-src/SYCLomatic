// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

struct CustomSum {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct CustomEqual {
  template <typename T>
  __device__ __host__ inline bool operator()(const T &lhs, const T &rhs) const {
    return lhs == rhs;
  }
} custom_eq;

void test(void *temp_storage, size_t &temp_storage_bytes, int *d_keys_in, int * d_values_in,int *d_values_out,
          CustomSum op, int init_value,int num_items, CustomEqual equality_op) {
  // Start
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cub::DeviceScan::ExclusiveScanByKey(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_keys_in/*KeysInputIteratorT*/, d_values_in/*ValuesInputIteratorT*/, d_values_out/*ValuesOutputIteratorT*/, op/*ScanOpT*/, init_value/*InitValueT*/, num_items/*int*/, equality_op/*EqualityOpT*/, stream/*cudaStream_t*/);
  // End
}
