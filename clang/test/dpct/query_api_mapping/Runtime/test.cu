// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamGetFlags | FileCheck %s -check-prefix=CUDASTREAMGETFLAGS
// CUDASTREAMGETFLAGS: CUDA API:
// CUDASTREAMGETFLAGS-NEXT:   cudaStreamGetFlags(s /*cudaStream_t*/, f /*unsigned int **/);
// CUDASTREAMGETFLAGS-NEXT: Is migrated to (with some neccessary option):
// CUDASTREAMGETFLAGS-NEXT:   *(f) = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaEventDestroy | FileCheck %s -check-prefix=CUDAEVENTDESTROY
// CUDAEVENTDESTROY: CUDA API:
// CUDAEVENTDESTROY-NEXT:   cudaEventDestroy(e /*cudaEvent_t*/);
// CUDAEVENTDESTROY-NEXT: Is migrated to (with some neccessary option):
// CUDAEVENTDESTROY-NEXT:   dpct::destroy_event(e);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=aaa | FileCheck %s -check-prefix=AAA
// AAA: The API Mapping is not available
