// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none -out-root %T/cudaGraph_test_default_option %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cudaGraph_test_default_option/cudaGraph_test_default_option.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DBUILD_TEST -fsycl %T/cudaGraph_test_default_option/cudaGraph_test_default_option.dp.cpp -o %T/cudaGraph_test_default_option/cudaGraph_test.dp.o %}

#ifndef BUILD_TEST
#include <cuda.h>

int main() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraph_t is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraph_t graph;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphExec_t is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphExec_t execGraph;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphInstantiate is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphInstantiate(&execGraph, graph, nullptr, nullptr, 0);

  cudaStream_t stream;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphLaunch is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphLaunch(execGraph, stream);

  return 0;
}

#endif
