// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/LibCommonUtils/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LibCommonUtils/api_test2_out/MainSourceFiles.yaml | wc -l > %T/LibCommonUtils/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/LibCommonUtils/api_test2_out/count.txt --match-full-lines %s -check-prefix=FEATURE_NUMBER
// RUN: FileCheck --input-file %T/LibCommonUtils/api_test2_out/api_test2.dp.cpp --match-full-lines %s -check-prefix=CODE
// RUN: rm -rf %T/LibCommonUtils/api_test2_out

// FEATURE_NUMBER: 3
// TEST_FEATURE: LibCommonUtils_mkl_get_version

// CODE: // AAA
// CODE-NEXT:#define DPCT_USM_LEVEL_NONE
// CODE-NEXT:#include <sycl/sycl.hpp>
// CODE-NEXT:#include <dpct/dpct.hpp>
// CODE-NEXT:#include <oneapi/mkl.hpp>
// CODE-NEXT:// BBB

// AAA
#include "cufft.h"
// BBB

int main() {
  int ver;
  cufftGetVersion(&ver);
  return 0;
}
