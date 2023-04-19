// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test12_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test12_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test12_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test12_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test12_out

// CHECK: 39
// TEST_FEATURE: DplExtrasAlgorithm_transform_if

#include <thrust/transform.h>

struct is_odd {
  __host__ __device__ bool operator()(const int x) const {
    return x % 2;
  }
};

int main() {
  thrust::negate<int> neg;
  int *a;
  thrust::transform_if(thrust::seq, a, a, a, neg, is_odd());
  return 0;
}
