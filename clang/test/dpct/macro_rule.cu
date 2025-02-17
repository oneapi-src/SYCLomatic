// RUN: dpct --rule-file=%S/../../tools/dpct/extensions/opt_rules/macro_checks.yaml --format-range=none -out-root %T/macro_rule %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/macro_rule/macro_rule.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/macro_rule/macro_rule.dp.cpp -o %T/macro_rule/macro_rule.dp.o %}
#include <cuda_runtime.h>

#define CUDA_CHECK(err) err

int main() {
  int *ptr;
  // CHECK: DPCT_CHECK_ERROR(ptr = (int *)sycl::malloc_device(10, dpct::get_in_order_queue()));
  CUDA_CHECK(cudaMalloc(&ptr, 10));
  return 0;
}
