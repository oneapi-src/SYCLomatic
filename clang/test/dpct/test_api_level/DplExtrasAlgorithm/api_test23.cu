// UNSUPPORTED: cuda-8.0, cuda-12.0
// UNSUPPORTED: v8.0, v12.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test23_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test23_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test23_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test23_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test23_out

// CHECK: 43
// TEST_FEATURE: DplExtrasAlgorithm_merge

#include <thrust/merge.h>
#include <thrust/device_vector.h>

int main() {
  thrust::device_vector<int> AD(4);
  thrust::device_vector<int> BD(4);
  thrust::device_vector<int> CD(4);
  thrust::device_vector<int> DD(4);
  thrust::device_vector<int> ED(8);
  thrust::device_vector<int> FD(8);
  
  thrust::merge_by_key( AD.begin(), AD.end(), BD.begin(), BD.end(), CD.begin(), DD.begin(), ED.begin(), FD.begin());
  return 0;
}
