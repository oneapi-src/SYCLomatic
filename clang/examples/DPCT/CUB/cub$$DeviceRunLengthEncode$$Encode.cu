// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(void *temp_storage, size_t &temp_storage_bytes, int *d_in, int *d_unique_out, int *d_counts_out,int *d_num_runs_out,
          int num_items, cudaStream_t stream) {
  // Start
  cub::DeviceRunLengthEncode::Encode(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_unique_out/*UniqueOutputIteratorT*/, d_counts_out/*LengthsOutputIteratorT*/, d_num_runs_out/*NumRunsOutputIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
  // End
}
