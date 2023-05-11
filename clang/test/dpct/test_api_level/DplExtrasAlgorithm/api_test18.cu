// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test18_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test18_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test18_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test18_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test18_out

// CHECK: 37
// TEST_FEATURE: DplExtrasAlgorithm_stable_partition_copy

#include <thrust/functional.h>
#include <thrust/partition.h>
#include <thrust/host_vector.h>

template <typename T> struct is_even {
  __host__ __device__ bool operator()(T x) const { return ((int)x % 2) == 0; }
};

int main() {
  int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int result[10];
  const int N = sizeof(data) / sizeof(int);
  int *evens = result;
  int *odds = result + N / 2;
  thrust::stable_partition_copy(data, data + N, evens, odds, is_even<int>());
  return 0;
}
