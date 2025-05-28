// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8
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
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaKernelNodeParams is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaKernelNodeParams params;
  params.blockDim = dim3(10);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphAddKernelNode is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphGetNodes is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphGetNodes(graph, NULL, nullptr);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphGetRootNodes is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphGetRootNodes(graph, NULL, nullptr);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphInstantiate is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphInstantiate(&execGraph, graph, nullptr, nullptr, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphLaunch is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphLaunch(execGraph, stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphExecUpdateResultInfo is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphExecUpdateResultInfo updateResult;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphExecUpdateResult is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphExecUpdateResult status;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphExecUpdate is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphExecUpdate(execGraph, graph, &updateResult);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphExecDestroy is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphExecDestroy(execGraph);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphNodeType is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphNodeType nodeType;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphNodeGetType is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphNodeGetType(node, &nodeType);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphNodeTypeKernel is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  nodeType = cudaGraphNodeTypeKernel;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaKernelNodeParams is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaKernelNodeParams kernelNodeParam0 = {};

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphDestroy is not supported, please try to remigrate with option: --use-experimental-features=graph.
  // CHECK-NEXT: */
  cudaGraphDestroy(graph);

  return 0;
}

#endif
