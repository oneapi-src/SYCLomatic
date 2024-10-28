// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// RUN: cat %s > %T/user_define_rule_header_order.cu
// RUN: cat %S/user_define_rule_header_order.yaml > %T/user_define_rule_header_order.yaml
// RUN: cd %T
// RUN: rm -rf %T/user_define_rule_header_order_output
// RUN: mkdir %T/user_define_rule_header_order_output
// RUN: dpct -format-range=none -out-root %T/user_define_rule_header_order_output user_define_rule_header_order.cu --cuda-include-path="%cuda-path/include" --usm-level=none --rule-file=user_define_rule_header_order.yaml -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/user_define_rule_header_order_output/user_define_rule_header_order.dp.cpp --match-full-lines user_define_rule_header_order.cu
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/user_define_rule_header_order_output/user_define_rule_header_order.dp.cpp -o %T/user_define_rule_header_order_output/user_define_rule_header_order.dp.o %}

// CHECK: #include <oneapi/dpl/execution>
// CHECK: #include <oneapi/dpl/algorithm>
// CHECK: #include <sycl/sycl.hpp>
#include <cub/cub.cuh>
#include <stddef.h>

#ifdef NO_BUILD_TEST
namespace cub {
struct DeviceScan {
static void InclusiveSum(void *, size_t &, int *, int *, int) {}
};
} // namespace cub
#endif // NO_BUILD_TEST

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
