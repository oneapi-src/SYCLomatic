// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Dpct/api_test6_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Dpct/api_test6_out/MainSourceFiles.yaml | wc -l > %T/Dpct/api_test6_out/count.txt
// RUN: FileCheck --input-file %T/Dpct/api_test6_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Dpct/api_test6_out


// CHECK: 40
// TEST_FEATURE: Dpct_no_feature

#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>

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
  greater_than_zero pred;
  thrust::device_vector<int> A(4);
  
  thrust::any_of(A.begin(), A.end(), pred);
  return 0;
}
