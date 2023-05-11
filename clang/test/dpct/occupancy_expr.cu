// RUN: dpct --format-range=none -out-root %T/occupancy_expr %s --use-experimental-features=occupancy-calculation --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/occupancy_expr/occupancy_expr.dp.cpp

__global__ void k() {}

int main() {
  int num_blocks;
  int block_size = 128;
  size_t dynamic_shared_memory_size = 0 ;
  // CHECK: dpct::experimental::calculate_max_active_wg_per_xecore(&num_blocks, block_size, dynamic_shared_memory_size + dpct_placeholder /* static shared local memory size */);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, k, block_size, dynamic_shared_memory_size);
  return 0;
}
