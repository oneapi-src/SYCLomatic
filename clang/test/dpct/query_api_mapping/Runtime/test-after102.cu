// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1

// RUN: dpct --cuda-include-path="%cuda-path/include" -query-api-mapping=cudaGraphExecUpdate | FileCheck %s -check-prefix=cudaGraphExecUpdate
// cudaGraphExecUpdate: CUDA API:
// cudaGraphExecUpdate-NEXT:   cudaGraphExecUpdate(graph_exec /*cudaGraphExec_t*/, graph /*cudaGraph_t*/,
// cudaGraphExecUpdate-NEXT:                       node /*cudaGraphNode_t **/,
// cudaGraphExecUpdate-NEXT:                       result /*cudaGraphExecUpdateResult **/);
// cudaGraphExecUpdate-NEXT: Is migrated to (with the option --use-experimental-features=graph):
// cudaGraphExecUpdate-NEXT:   graph_exec->update(*graph);
