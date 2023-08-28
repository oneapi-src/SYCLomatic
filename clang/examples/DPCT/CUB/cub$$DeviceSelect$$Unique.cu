// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

void test(void *temp_storage, size_t &temp_storage_bytes, int *d_in, int * d_out,int *d_num_selected_out,
          int num_items, cudaStream_t stream) {
  // Start
  cub::DeviceSelect::Unique(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, d_num_selected_out/*NumSelectedIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
  // End
}
