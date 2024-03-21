// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/trap %S/trap.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/trap/trap.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/trap/trap.dp.cpp -o %T/trap/trap.dp.o %}
#include <cuda.h>

__global__ void kernel() {
      // CHECK: assert(0);
      __trap();
}
