// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test1_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test1_out

// CHECK: 34
// TEST_FEATURE: DplExtrasAlgorithm_replace_if

#include <thrust/replace.h>

struct greater_than_zero
{
  __host__ __device__
  bool operator()(int x) const
  {
    return x > 0;
  }
  typedef int argument_type;
};

int main() {
  int A[10];
  greater_than_zero pred;
  thrust::replace_if(A, A + 10, A, pred, 0);
  return 0;
}
