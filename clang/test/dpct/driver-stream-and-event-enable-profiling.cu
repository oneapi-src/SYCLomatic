// RUN: dpct --enable-profiling  -out-root %T/driver-stream-and-event-enable-profiling %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/driver-stream-and-event-enable-profiling/driver-stream-and-event-enable-profiling.dp.cpp %s

// CHECK:#define DPCT_PROFILING_ENABLED
// CHECK-NEXT:#include <sycl/sycl.hpp>
// CHECK-NEXT:#include <dpct/dpct.hpp>
// CHECK-NEXT:#include <vector>
#include "cuda.h"
#include <vector>

template <typename T>
void my_error_checker(T ReturnValue, char const *const FuncName) {
}

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)

void foo(){
  CUfunction f;
  CUstream s;
  CUevent e;

// CHECK: s->ext_oneapi_submit_barrier({*e});
  cuEventCreate(&e, CU_EVENT_DEFAULT);
  cuStreamWaitEvent(s, e, 0);

// CHECK: *e = s->ext_oneapi_submit_barrier();
// CHECK-NEXT: e->wait_and_throw();
  cuEventRecord(e, s);
  cuEventSynchronize(e);

// CHECK: sycl::info::event_command_status r;
// CHECK-NEXT: r = e->get_info<sycl::info::event::command_execution_status>();
  CUresult r;
  r = cuEventQuery(e);

// CHECK: dpct::event_ptr start, end;
// CHECK: *start = s->ext_oneapi_submit_barrier();
// CHECK: *end = s->ext_oneapi_submit_barrier();
// CHECK: start->wait_and_throw();
// CHECK: end->wait_and_throw();
// CHECK: float result_time;
// CHECK: result_time = (end->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
  CUevent start, end;
  cuEventRecord(start, s);
  cuEventRecord(end, s);
  cuEventSynchronize(start);
  cuEventSynchronize(end);
  float result_time;
  cuEventElapsedTime(&result_time, start, end);
}
