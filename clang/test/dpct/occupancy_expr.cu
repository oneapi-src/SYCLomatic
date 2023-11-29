// RUN: dpct --format-range=none -out-root %T/occupancy_expr %s --use-experimental-features=occupancy-calculation --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/occupancy_expr/occupancy_expr.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/occupancy_expr/occupancy_expr.dp.cpp -o %T/occupancy_expr/occupancy_expr.dp.o %}

__global__ void k() {}

int main() {
  int num_blocks;
  int block_size = 128;
  size_t dynamic_shared_memory_size = 0;
  // CHECK: /*
  // CHECK: DPCT1111:{{[0-9]+}}: Please verify the input arguments of "dpct::experimental::calculate_max_active_wg_per_xecore" base on the target function "k".
  // CHECK: */
  // CHECK: dpct::experimental::calculate_max_active_wg_per_xecore(&num_blocks, block_size, dynamic_shared_memory_size + dpct_placeholder /* total share local memory size */);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, k, block_size, dynamic_shared_memory_size);

  CUfunction func;
  // CHECK: /*
  // CHECK: DPCT1111:{{[0-9]+}}: Please verify the input arguments of "dpct::experimental::calculate_max_active_wg_per_xecore" base on the target function "func".
  // CHECK: */
  // CHECK: dpct::experimental::calculate_max_active_wg_per_xecore(&num_blocks, block_size, dynamic_shared_memory_size + dpct_placeholder /* total share local memory size */);
  cuOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, func, block_size, dynamic_shared_memory_size);

  int min_grid_size;
  int block_size_limit;
  // CHECK: /*
  // CHECK-NEXT: DPCT1111:{{[0-9]+}}: Please verify the input arguments of "dpct::experimental::calculate_max_potential_wg" base on the target function "k".
  // CHECK-NEXT: */
  // CHECK-NEXT:dpct::experimental::calculate_max_potential_wg(&min_grid_size, &block_size, 0, dpct_placeholder /* total share local memory size */);
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, k);
  // CHECK: /*
  // CHECK-NEXT: DPCT1111:{{[0-9]+}}: Please verify the input arguments of "dpct::experimental::calculate_max_potential_wg" base on the target function "k".
  // CHECK-NEXT: */
  // CHECK-NEXT:dpct::experimental::calculate_max_potential_wg(&min_grid_size, &block_size, 0, dynamic_shared_memory_size + dpct_placeholder /* total share local memory size */);
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, k, dynamic_shared_memory_size);
  // CHECK: /*
  // CHECK-NEXT: DPCT1111:{{[0-9]+}}: Please verify the input arguments of "dpct::experimental::calculate_max_potential_wg" base on the target function "k".
  // CHECK-NEXT: */
  // CHECK-NEXT:dpct::experimental::calculate_max_potential_wg(&min_grid_size, &block_size, block_size_limit, dynamic_shared_memory_size + dpct_placeholder /* total share local memory size */);
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, k, dynamic_shared_memory_size, block_size_limit);
  return 0;
}
