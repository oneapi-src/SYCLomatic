// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --use-experimental-features=graph --format-range=none -out-root %T/cudaGraph_test %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cudaGraph_test/cudaGraph_test.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cudaGraph_test/cudaGraph_test.dp.cpp -o %T/cudaGraph_test/cudaGraph_test.dp.o %}

#include <cuda.h>

int main() {
  // CHECK: dpct::experimental::command_graph_ptr graph;
  // CHECK-NEXT: dpct::experimental::command_graph_ptr *graph2;
  // CHECK-NEXT: dpct::experimental::command_graph_ptr **graph3;
  cudaGraph_t graph;
  cudaGraph_t *graph2;
  cudaGraph_t **graph3;

  // CHECK: dpct::experimental::command_graph_ptr graph4[10];
  cudaGraph_t graph4[10];

  // CHECK: dpct::experimental::command_graph_ptr graph5, *graph6, **graph7;
  cudaGraph_t graph5, *graph6, **graph7;

  // CHECK: dpct::experimental::command_graph_exec_ptr execGraph;
  // CHECK-NEXT: dpct::experimental::command_graph_exec_ptr *execGraph2;
  // CHECK-NEXT: dpct::experimental::command_graph_exec_ptr **execGraph3;
  cudaGraphExec_t execGraph;
  cudaGraphExec_t *execGraph2;
  cudaGraphExec_t **execGraph3;

  // CHECK: dpct::experimental::command_graph_exec_ptr execGraph4[10];
  cudaGraphExec_t execGraph4[10];

  // CHECK: dpct::experimental::command_graph_exec_ptr execGraph5, *execGraph6, **execGraph7;
  cudaGraphExec_t execGraph5, *execGraph6, **execGraph7;

  // CHECK: execGraph = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>((*graph2)->finalize());
  // CHECK-NEXT: *execGraph2 = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize());
  // CHECK-NEXT: **execGraph3 = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>((*graph2)->finalize());
  cudaGraphInstantiate(&execGraph, *graph2, nullptr, nullptr, 0);
  cudaGraphInstantiate(execGraph2, graph, nullptr, nullptr, 0);
  cudaGraphInstantiate(*execGraph3, *graph2, nullptr, nullptr, 0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaStream_t *stream2;

  // CHECK: stream->ext_oneapi_graph(*execGraph);
  // CHECK-NEXT: (*stream2)->ext_oneapi_graph(**execGraph2);
  cudaGraphLaunch(execGraph, stream);
  cudaGraphLaunch(*execGraph2, *stream2);

  // CHECK: delete (execGraph);
  // CHECK-NEXT: delete (*execGraph2);
  // CHECK-NEXT:  delete (**execGraph3);
  cudaGraphExecDestroy(execGraph);
  cudaGraphExecDestroy(*execGraph2);
  cudaGraphExecDestroy(**execGraph3);

  return 0;
}
