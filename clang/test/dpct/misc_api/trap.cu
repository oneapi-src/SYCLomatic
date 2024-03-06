// RUN: dpct --format-range=none -in-root %S -out-root %T/trap %S/trap.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/trap/trap.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/trap/trap.dp.cpp -o %T/trap/trap.dp.o %}
#include <cuda.h>

__global__ void kernel() {
      // CHECK: assert(0);
      __trap();
}
