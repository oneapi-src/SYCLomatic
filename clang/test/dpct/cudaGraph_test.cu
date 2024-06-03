// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none --use-experimental-features=graph -out-root %T/cudaGraph_test %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cudaGraph_test/cudaGraph_test.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST %T/cudaGraph_test/cudaGraph_test.dp.cpp -o %T/cudaGraph_test/cudaGraph_test.dp.o %}

#ifndef BUILD_TEST
#include <cuda.h>

int main() {
  // CHECK: dpct::experimental::command_graph_t graph;
  // CHECK-NEXT: dpct::experimental::command_graph_t *graph2;
  // CHECK-NEXT: dpct::experimental::command_graph_t **graph3;
  cudaGraph_t graph;
  cudaGraph_t *graph2;
  cudaGraph_t **graph3;

  // CHECK: dpct::experimental::command_graph_t graph4[10];
  cudaGraph_t graph4[10];

  // CHECK: dpct::experimental::command_graph_t graph5, *graph6, **graph7;
  cudaGraph_t graph5, *graph6, **graph7;

  // CHECK: dpct::experimental::command_graph_exec_t execGraph;
  // CHECK-NEXT: dpct::experimental::command_graph_exec_t *execGraph2;
  // CHECK-NEXT: dpct::experimental::command_graph_exec_t **execGraph3;
  cudaGraphExec_t execGraph;
  cudaGraphExec_t *execGraph2;
  cudaGraphExec_t **execGraph3;

  // CHECK: dpct::experimental::command_graph_exec_t execGraph4[10];
  cudaGraphExec_t execGraph4[10];

  // CHECK: dpct::experimental::command_graph_exec_t execGraph5, *execGraph6, **execGraph7;
  cudaGraphExec_t execGraph5, *execGraph6, **execGraph7;

  return 0;
}

#endif // BUILD_TEST
