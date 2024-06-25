// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --use-experimental-features=graph --format-range=none -out-root %T/cudaStreamCapture_test %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cudaStreamCapture_test/cudaStreamCapture_test.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DBUILD_TEST -fsycl %T/cudaStreamCapture_test/cudaStreamCapture_test.dp.cpp -o %T/cudaStreamCapture_test/cudaStreamCapture_test.dp.o %}

#ifndef BUILD_TEST
#include <cuda.h>

int main() {
  cudaGraph_t graph;
  cudaGraph_t *graph2;
  cudaGraph_t **graph3;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // CHECK: dpct::experimental::command_graph_begin_recording(stream);
  // CHECK-NEXT: dpct::experimental::command_graph_end_recording(stream, graph);
  // CHECK-NEXT: dpct::experimental::command_graph_end_recording(stream, *graph2);
  // CHECK-NEXT: dpct::experimental::command_graph_end_recording(stream, **graph3);
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  cudaStreamEndCapture(stream, &graph);
  cudaStreamEndCapture(stream, graph2);
  cudaStreamEndCapture(stream, *graph3);

  return 0;
}
