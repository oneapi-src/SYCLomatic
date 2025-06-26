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

void test(void *temp_storage, size_t &temp_storage_bytes, int *d_in, int *d_out, int num_segments, int *d_offsets, CustomMin op, int init_value) {
  // Start
  cub::DeviceSegmentedReduce::Reduce(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/, op/*ReductionOpT*/, init_value/*T*/);
  // End
}
