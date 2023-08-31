// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

struct CustomSum {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

void test(void *temp_storage, size_t &temp_storage_bytes, int *d_in, int *d_out, CustomSum scan_op, int num_items) {
  // Start
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cub::DeviceScan::InclusiveScan(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, scan_op/*ScanOpT*/, num_items/*int*/, stream/*cudaStream_t*/);
  // End
}
