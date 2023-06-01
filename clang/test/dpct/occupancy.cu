// RUN: dpct --format-range=none -out-root %T/occupancy %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/occupancy/occupancy.dp.cpp

__global__ void k() {}

int main() {
  int num_blocks;
  int block_size = 128;
  size_t dynamic_shared_memory_size = 0;
  //CHECK:/*
  //CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of cudaOccupancyMaxActiveBlocksPerMultiprocessor is not supported.
  //CHECK-NEXT:*/
  //CHECK-NEXT:cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, k, block_size, dynamic_shared_memory_size);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, k, block_size, dynamic_shared_memory_size);

  CUfunction func;
  //CHECK:/*
  //CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of cuOccupancyMaxActiveBlocksPerMultiprocessor is not supported.
  //CHECK-NEXT:*/
  //CHECK-NEXT:cuOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, func, block_size, dynamic_shared_memory_size);
  cuOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, func, block_size, dynamic_shared_memory_size);
  return 0;
}
