// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --no-dpcpp-extensions=assert -in-root %S -out-root %T/trap_no_assert %S/trap_no_assert.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/trap_no_assert/trap_no_assert.dp.cpp --match-full-lines %s
#include <cuda.h>

__global__ void kernel() {
// CHECK:  /*
// CHECK:  DPCT1028:{{[0-9]+}}: The __trap was not migrated because assert extension is disabled. You can migrate the code with assert extension by not specifying --no-dpcpp-extensions=assert.
// CHECK:  */
// CHECK:  __trap();
      __trap();
}
