// RUN: dpct --format-range=none --no-dpcpp-extensions=assert -in-root %S -out-root %T/trap_no_assert %S/trap_no_assert.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/trap_no_assert/trap_no_assert.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/trap_no_assert/trap_no_assert.dp.cpp -o %T/trap_no_assert/trap_no_assert.dp.o %}
#include <cuda.h>

__global__ void kernel() {
// CHECK:  /*
// CHECK:  DPCT1122:0: Migration of '__trap' is not supported if 'assert' extension is disabled. You can migrate the code with 'assert' extension by not specifying --no-dpcpp-extensions=assert.
// CHECK:  */
// CHECK:  __trap();
      __trap();
}
