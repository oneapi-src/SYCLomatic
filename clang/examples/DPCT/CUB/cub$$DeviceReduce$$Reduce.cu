// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

struct CustomMin {
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

void test(void *temp_storage, size_t &temp_storage_bytes, int *d_in, int *d_out,
          int num_items, CustomMin op, int init_value, cudaStream_t stream) {
  // Start
  cub::DeviceReduce::Reduce(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_items/*int*/, op/*ReductionOpT*/, init_value/*T*/, stream/*cudaStream_t*/);
  // End
}
