// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/FftUtils/api_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/FftUtils/api_test3_out/MainSourceFiles.yaml | wc -l > %T/FftUtils/api_test3_out/count.txt
// RUN: FileCheck --input-file %T/FftUtils/api_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/FftUtils/api_test3_out

// CHECK: 22
// TEST_FEATURE: FftUtils_fft_engine

#include "cufft.h"

int main() {
  cufftHandle plan;
  float2* odata;
  float2* idata;
  cufftPlan1d(&plan, 10, CUFFT_C2C, 3);
  cufftExecC2C(plan, idata, odata, CUFFT_FORWARD);
  return 0;
}
