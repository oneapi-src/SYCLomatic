// RUN: cat %s > %T/user_define_rule_header_order.cu
// RUN: cat %S/user_define_rule_header_order.yaml > %T/user_define_rule_header_order.yaml
// RUN: cd %T
// RUN: rm -rf %T/user_define_rule_header_order_output
// RUN: mkdir %T/user_define_rule_header_order_output
// RUN: dpct -out-root %T/user_define_rule_header_order_output user_define_rule_header_order.cu --cuda-include-path="%cuda-path/include" --usm-level=none --rule-file=user_define_rule_header_order.yaml -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/user_define_rule_header_order_output/user_define_rule_header_order.dp.cpp --match-full-lines user_define_rule_header_order.cu

// CHECK: #include <oneapi/dpl/execution>
// CHECK: #include <oneapi/dpl/algorithm>
// CHECK: #include <sycl/sycl.hpp>
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
  CUB_WRAPPER(cub::DeviceScan::InclusiveSum, d_in, d_out, n);
}
