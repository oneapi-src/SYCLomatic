// RUN: dpct --format-range=none   --use-custom-helper=api -out-root %T/Memory/api_test45_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test45_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test45_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test45_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test45_out

// CHECK: 19
// TEST_FEATURE: Memory_usm_host_allocator_alias

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#define SIZE 4

int main(int argc, char *argv[]) {
  std::vector<float, thrust::system::cuda::experimental::pinned_allocator<float>> hVec(SIZE);
  std::fill(hVec.begin(), hVec.end(), 2);
  std::vector<float, thrust::cuda::experimental::pinned_allocator<float>> hVecCopy = hVec;
  return 0;
}
