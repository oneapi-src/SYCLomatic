// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2, v12.3
// RUN: dpct --use-experimental-features=graph --format-range=none -out-root %T/cudaGraphNodeType_test %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cudaGraphNodeType_test/cudaGraphNodeType_test.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/cudaGraphNodeType_test/cudaGraphNodeType_test.dp.cpp -o %T/cudaGraphNodeType_test/cudaGraphNodeType_test.dp.o %}

#include <cuda.h>
#define CUDA_CHECK_THROW(x)  \
  do {                       \
    cudaError_t _result = x; \
  } while (0)

int main() {
  // CHECK: sycl::ext::oneapi::experimental::node_type nodeType;
  cudaGraphNodeType nodeType;

  // CHECK: nodeType = sycl::ext::oneapi::experimental::node_type::kernel;
  nodeType = cudaGraphNodeTypeKernel;

  // CHECK: nodeType = sycl::ext::oneapi::experimental::node_type::memcpy;
  nodeType = cudaGraphNodeTypeMemcpy;

  // CHECK: nodeType = sycl::ext::oneapi::experimental::node_type::memset;
  nodeType = cudaGraphNodeTypeMemset;

  // CHECK: nodeType = sycl::ext::oneapi::experimental::node_type::host_task;
  nodeType = cudaGraphNodeTypeHost;

  // CHECK: nodeType = sycl::ext::oneapi::experimental::node_type::subgraph;
  nodeType = cudaGraphNodeTypeGraph;

  // CHECK: nodeType = sycl::ext::oneapi::experimental::node_type::empty;
  nodeType = cudaGraphNodeTypeEmpty;

#ifndef NO_BUILD_TEST

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaGraphNodeTypeWaitEvent is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: nodeType = cudaGraphNodeTypeWaitEvent;
  nodeType = cudaGraphNodeTypeWaitEvent;

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaGraphNodeTypeEventRecord is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: nodeType = cudaGraphNodeTypeEventRecord;
  nodeType = cudaGraphNodeTypeEventRecord;

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaGraphNodeTypeExtSemaphoreSignal is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: nodeType = cudaGraphNodeTypeExtSemaphoreSignal;
  nodeType = cudaGraphNodeTypeExtSemaphoreSignal;

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaGraphNodeTypeExtSemaphoreWait is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: nodeType = cudaGraphNodeTypeExtSemaphoreWait;
  nodeType = cudaGraphNodeTypeExtSemaphoreWait;

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaGraphNodeTypeMemAlloc is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: nodeType = cudaGraphNodeTypeMemAlloc;
  nodeType = cudaGraphNodeTypeMemAlloc;

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaGraphNodeTypeMemFree is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: nodeType = cudaGraphNodeTypeMemFree;
  nodeType = cudaGraphNodeTypeMemFree;

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaGraphNodeTypeConditional is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: nodeType = cudaGraphNodeTypeConditional;
  nodeType = cudaGraphNodeTypeConditional;

#endif

  return 0;
}
