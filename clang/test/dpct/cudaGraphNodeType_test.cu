// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
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

  return 0;
}
