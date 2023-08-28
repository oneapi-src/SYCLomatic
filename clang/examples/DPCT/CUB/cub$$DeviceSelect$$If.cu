// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

struct LessThan {
  int compare;
  inline LessThan(int compare) : compare(compare) {}
  __device__ bool operator()(const int &a) const {
    return (a < compare);
  }
};

void test(void *temp_storage, size_t &temp_storage_bytes, int *d_in, int * d_out,int *d_num_selected_out,
          int num_items, LessThan select_op, cudaStream_t stream) {
  // Start
  cub::DeviceSelect::If(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, d_num_selected_out/*NumSelectedIteratorT*/, num_items/*int*/, select_op/*SelectOp*/, stream/*cudaStream_t*/);
  // End
}
