// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasMemory/api_test11_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasMemory/api_test11_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasMemory/api_test11_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasMemory/api_test11_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasMemory/api_test11_out

// CHECK: 37
// TEST_FEATURE: DplExtrasMemory_device_reference


#include <thrust/device_vector.h>

int main() {
  thrust::device_vector<int> dvec(1, 13);
  thrust::device_reference<int> r1 = dvec[0];
  thrust::device_reference<int> r2 = dvec[0];
  r2 = r1 + r2;
}
