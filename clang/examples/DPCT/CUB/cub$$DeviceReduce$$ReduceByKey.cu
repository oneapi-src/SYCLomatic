// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

struct CustomMin {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return (b < a) ? b : a;
  }
};

void test(void *temp_storage, size_t &temp_storage_bytes, int *d_keys_in,
          int *d_unique_out, int *d_values_in, int *d_aggregates_out,
          int *d_num_runs_out, int num_items, CustomMin op) {
  // Start
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cub::DeviceReduce::ReduceByKey(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_keys_in/*KeysInputIteratorT*/, d_unique_out/*UniqueOutputIteratorT*/, d_values_in/*ValuesInputIteratorT*/, d_aggregates_out/*AggregatesOutputIteratorT*/, d_num_runs_out/*NumRunsOutputIteratorT*/, op/*ReductionOpT*/, num_items/*int*/, stream/*cudaStream_t*/);
  // End
}
