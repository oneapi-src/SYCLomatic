// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/FftUtils/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/FftUtils/api_test1_out/MainSourceFiles.yaml | wc -l > %T/FftUtils/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/FftUtils/api_test1_out/count.txt --match-full-lines %s -check-prefix=FEATURE_NUMBER
// RUN: FileCheck --input-file %T/FftUtils/api_test1_out/api_test1.dp.cpp --match-full-lines %s -check-prefix=CODE
// RUN: rm -rf %T/FftUtils/api_test1_out

// FEATURE_NUMBER: 2
// TEST_FEATURE: FftUtils_fft_direction
// TEST_FEATURE: FftUtils_non_local_include_dependency

// CODE: // AAA
// CODE-NEXT:#include <sycl/sycl.hpp>
// CODE-NEXT:#include <dpct/dpct.hpp>
// CODE-NEXT:#include <oneapi/mkl.hpp>
// CODE-NEXT:// BBB

// AAA
#include "cufft.h"
// BBB

int main() {
  int a = CUFFT_FORWARD;
  return 0;
}
