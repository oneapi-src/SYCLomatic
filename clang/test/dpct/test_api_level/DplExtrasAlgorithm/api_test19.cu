// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test19_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test19_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test19_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test19_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test19_out

// CHECK: 35
// TEST_FEATURE: DplExtrasAlgorithm_stable_partition

#include <thrust/functional.h>
#include <thrust/partition.h>
#include <thrust/host_vector.h>

struct is_even {
  __host__ __device__ bool operator()(int x) const { return ((int)x % 2) == 0; }
};

int main() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int N = sizeof(data) / sizeof(int);
  thrust::stable_partition(data, data + N, is_even());
  return 0;
}
