// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test29_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test29_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test29_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test29_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test29_out

// CHECK: 2
// TEST_FEATURE: DplExtrasAlgorithm_scatter_if

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/scatter.h>

void test_1() {
  const int N = 8;

  int V[N] = {10, 20, 30, 40, 50, 60, 70, 80};
  int M[N] = {0, 5, 1, 6, 2, 7, 3, 4};
  int S[N] = {1, 0, 1, 0, 1, 0, 1, 0};
  int D[N] = {0, 0, 0, 0, 0, 0, 0, 0};
  thrust::scatter_if(thrust::host, V, V + 8, M, S, D);
}
