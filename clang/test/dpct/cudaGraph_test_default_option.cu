// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none -out-root %T/cudaGraph_test_default_option %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cudaGraph_test_default_option/cudaGraph_test_default_option.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/cudaGraph_test_default_option/cudaGraph_test_default_option.dp.cpp -o %T/cudaGraph_test_default_option/cudaGraph_test.dp.o %}

#ifndef NO_BUILD_TEST
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

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaStreamBeginCapture is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaStreamEndCapture is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaStreamEndCapture(stream, &graph);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaStreamCaptureStatus is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaStreamCaptureStatus captureStatus;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaStreamCaptureStatusActive is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  captureStatus = cudaStreamCaptureStatusActive;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaStreamCaptureStatusNone is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  captureStatus = cudaStreamCaptureStatusNone;

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaStreamCaptureStatusInvalidated is not supported.
  // CHECK-NEXT: */
  captureStatus = cudaStreamCaptureStatusInvalidated;


  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaStreamIsCapturing is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaStreamIsCapturing(stream, &captureStatus);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphNode_t is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphNode_t node;

  // CHECK: /*
  // CHECK: DPCT1119:{{[0-9]+}}: Migration of cudaGraphAddEmptyNode is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphAddEmptyNode(&node, graph, NULL, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphAddDependencies is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphAddDependencies(graph, NULL, NULL, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphInstantiate is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphInstantiate(&execGraph, graph, nullptr, nullptr, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphLaunch is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphLaunch(execGraph, stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaGraphExecUpdateResult is not supported.
  // CHECK-NEXT: */
  cudaGraphExecUpdateResult status;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphExecUpdate is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphExecUpdate(execGraph, graph, nullptr, &status);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphExecDestroy is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphExecDestroy(execGraph);

  return 0;
}

#endif
