// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/CclUtils/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/CclUtils/api_test1_out/MainSourceFiles.yaml | wc -l > %T/CclUtils/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/CclUtils/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/CclUtils/api_test1_out

// CHECK: 4
// TEST_FEATURE: CclUtils_get_version

#include <nccl.h>

int main() {
  int version;
  ncclGetVersion(&version);
  return 0;
}
