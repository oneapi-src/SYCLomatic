// UNSUPPORTED: system-windows
// RUN: dpct --format-range=none --out-root %T/nvtx %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nvtx/nvtx.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/nvtx/nvtx.dp.cpp -o %T/nvtx/nvtx.dp.o %}

#include "nvToolsExt.h"
#include "nvToolsExtCudaRt.h"

void foo(cudaStream_t stream) {
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to nvtxNameCudaStreamA was removed because it annotates source code to provide contextual information to the CUDA analysis tool. Consider using Intel(R) Instrumentation and Tracing Technology (ITT) API to implement a similar function.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to nvtxRangePushA was replaced with 0 because it annotates source code to provide contextual information to the CUDA analysis tool. Consider using Intel(R) Instrumentation and Tracing Technology (ITT) API to implement a similar function.
  // CHECK-NEXT: */
  // CHECK-NEXT: int a1 = 0;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to nvtxRangePushW was replaced with 0 because it annotates source code to provide contextual information to the CUDA analysis tool. Consider using Intel(R) Instrumentation and Tracing Technology (ITT) API to implement a similar function.
  // CHECK-NEXT: */
  // CHECK-NEXT: int a2 = 0;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to nvtxRangePop was replaced with 0 because it annotates source code to provide contextual information to the CUDA analysis tool. Consider using Intel(R) Instrumentation and Tracing Technology (ITT) API to implement a similar function.
  // CHECK-NEXT: */
  // CHECK-NEXT: int a3 = 0;
  nvtxNameCudaStreamA(stream, "abc");
  int a1 = nvtxRangePushA("abc");
  int a2 = nvtxRangePushW(L"abc");
  int a3 = nvtxRangePop();
}
