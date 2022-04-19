// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/LibCommonUtils/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LibCommonUtils/api_test2_out/MainSourceFiles.yaml | wc -l > %T/LibCommonUtils/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/LibCommonUtils/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LibCommonUtils/api_test2_out

// CHECK: 3
// TEST_FEATURE: LibCommonUtils_mkl_get_version

#include "cufft.h"

int main() {
  int ver;
  cufftGetVersion(&ver);
  return 0;
}
