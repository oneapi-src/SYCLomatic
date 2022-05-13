// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/CclUtils/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/CclUtils/api_test2_out/MainSourceFiles.yaml | wc -l > %T/CclUtils/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/CclUtils/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/CclUtils/api_test2_out

// CHECK: 3
// TEST_FEATURE: CclUtils_create_kvs_address
// TEST_FEATURE: CclUtils_get_kvs_detail

#include <nccl.h>

int main() {
  ncclUniqueId Id;
  ncclGetUniqueId(&Id);
  return 0;
}
