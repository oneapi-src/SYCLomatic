// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test11_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test11_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test11_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test11_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test11_out

// CHECK: 3
// TEST_FEATURE: DplExtrasAlgorithm_partition_point

#include <thrust/partition.h>
#include <thrust/execution_policy.h>

int main() {
  int *a;
  thrust::partition_point(thrust::seq, a, a, [](int x) -> bool { return x < 23; });
  return 0;
}
