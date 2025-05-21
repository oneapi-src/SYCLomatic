// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none --out-root %T/nvtx3_a %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nvtx3_a/nvtx3_a.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/nvtx3_a/nvtx3_a.dp.cpp -o %T/nvtx3_a/nvtx3_a.dp.o %}

#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCudaRt.h"

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
