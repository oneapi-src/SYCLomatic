// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0

// RUN: dpct --enable-profiling  --format-range=none -out-root %T/event_record_with_flags_qb %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/event_record_with_flags_qb/event_record_with_flags_qb.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/event_record_with_flags_qb/event_record_with_flags_qb.dp.cpp -o %T/event_record_with_flags_qb/event_record_with_flags_qb.dp.o %}

// CHECK:#define DPCT_PROFILING_ENABLED
// CHECK-NEXT:#include <sycl/sycl.hpp>
// CHECK-NEXT:#include <dpct/dpct.hpp>
#include <cuda_runtime.h>

// CHECK: void cudaEventRecordWithFlags_1() {
// CHECK-NEXT:  dpct::event_ptr start;
// CHECK-NEXT:  dpct::queue_ptr stream;
// CHECK-NEXT:  dpct::sync_barrier(start, stream);
// CHECK-NEXT: }
void cudaEventRecordWithFlags_1() {
  cudaEvent_t start;
  cudaStream_t stream;
  cudaEventRecordWithFlags(start, stream, cudaEventRecordDefault);
}

#ifndef NO_BUILD_TEST
// CHECK: void cudaEventRecordWithFlags_2() {
// CHECK-NEXT:  dpct::event_ptr start;
// CHECK-NEXT:  dpct::queue_ptr stream;
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1028:{{[0-9a-f]+}}: The cudaEventRecordWithFlags was not migrated because parameter cudaEventRecordExternal is unsupported.
// CHECK-NEXT:  */
// CHECK-NEXT:  cudaEventRecordWithFlags(start, stream, cudaEventRecordExternal);
// CHECK-NEXT:}
void cudaEventRecordWithFlags_2() {
  cudaEvent_t start;
  cudaStream_t stream;
  cudaEventRecordWithFlags(start, stream, cudaEventRecordExternal);
}
#endif
