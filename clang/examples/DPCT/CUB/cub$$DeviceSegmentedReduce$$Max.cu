// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(void *temp_storage, size_t &temp_storage_bytes, int *d_in, int *d_out, int num_segments, int *d_offsets) {
  // Start
  cub::DeviceSegmentedReduce::Max(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
  // End
}
