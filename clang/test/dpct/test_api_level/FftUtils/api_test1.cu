// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/FftUtils/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/FftUtils/api_test1_out/MainSourceFiles.yaml | wc -l > %T/FftUtils/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/FftUtils/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/FftUtils/api_test1_out

// CHECK: 2
// TEST_FEATURE: FftUtils_fft_direction

#include "cufft.h"

int main() {
  int a = CUFFT_FORWARD;
  return 0;
}
