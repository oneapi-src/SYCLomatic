// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaEventRecordWithFlags | FileCheck %s -check-prefix=cudaEventRecordWithFlags
// cudaEventRecordWithFlags: CUDA API:
// cudaEventRecordWithFlags-NEXT:   cudaEventRecordWithFlags(event /*cudaEvent_t*/, stream /*cudaStream_t*/,
// cudaEventRecordWithFlags-NEXT:                            cudaEventRecordDefault /*unsigned int*/);
// cudaEventRecordWithFlags-NEXT: Is migrated to (with the option --enable-profiling):
// cudaEventRecordWithFlags-NEXT:   dpct::sync_barrier(event /*cudaEvent_t*/, stream /*unsigned int*/);
