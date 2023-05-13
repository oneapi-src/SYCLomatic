// RUN: dpct --use-experimental-features=occupancy-calculation --use-custom-helper=api -out-root %T/Util/api_test30_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test30_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test30_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test30_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test30_out

// CHECK: 16
// TEST_FEATURE: Util_calculate_max_active_wg_per_xecore

__global__ void k() {}

int main() {
  int num_blocks;
  int block_size = 128;
  size_t dynamic_shared_memory_size = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, k, block_size, dynamic_shared_memory_size);
  return 0;
}