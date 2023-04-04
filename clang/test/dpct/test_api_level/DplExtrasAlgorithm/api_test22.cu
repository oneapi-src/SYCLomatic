// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test22_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test22_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test22_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test22_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test22_out

// CHECK: 41
// TEST_FEATURE: DplExtrasAlgorithm_gather_if

#include <thrust/gather.h>
#include <thrust/device_vector.h>

int main() {
  int init[10] = {0, 1, 2, 3};
  thrust::device_vector<int> MD(init, init + 4);
  thrust::device_vector<int> SD(init, init + 4);  
  thrust::device_vector<int> ID(init, init + 4);
  thrust::device_vector<int> RD(4);
  
  thrust::gather_if(MD.begin(), MD.end(), SD.begin(), ID.begin(), RD.begin());
  return 0;
}
