// RUN: cat %s > %T/replace_callee_name_only.cu
// RUN: cat %S/replace_callee_name_only.yaml > %T/replace_callee_name_only.yaml
// RUN: cd %T
// RUN: rm -rf %T/replace_callee_name_only_output
// RUN: mkdir %T/replace_callee_name_only_output
// RUN: dpct -out-root %T/replace_callee_name_only_output replace_callee_name_only.cu --cuda-include-path="%cuda-path/include" --usm-level=none --rule-file=replace_callee_name_only.yaml -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/replace_callee_name_only_output/replace_callee_name_only.dp.cpp --match-full-lines replace_callee_name_only.cu

#include <cub/cub.cuh>
#include <stddef.h>

int n, *d_in, *d_out;
void *tmp;
size_t tmp_size;

#define CUB_WRAPPER(func, ...) do {                                       \
  void *temp_storage = nullptr;                                           \
  size_t temp_storage_bytes = 0;                                          \
  func(temp_storage, temp_storage_bytes, __VA_ARGS__);                    \
} while (false)

void test1() {
  // CHECK: CUB_WRAPPER(cub::DeviceScan::InclusiveSum, d_in, d_out, n);
  CUB_WRAPPER(cub::DeviceScan::InclusiveSum, d_in, d_out, n);
}

void test2() {
  // CHECK: cub::DeviceScan::InclusiveSum(tmp, tmp_size, d_in, d_out, n);
  cub::DeviceScan::InclusiveSum(tmp, tmp_size, d_in, d_out, n);
}
