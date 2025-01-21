// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/driver-stream-and-event_after8.0 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/driver-stream-and-event_after8.0/driver-stream-and-event_after8.0.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/driver-stream-and-event_after8.0/driver-stream-and-event_after8.0.dp.cpp -o %T/driver-stream-and-event_after8.0/driver-stream-and-event_after8.0.dp.o %}

#include "cuda.h"

// CHECK: void test_stream_and_context(dpct::queue_ptr stream, int &context) {
void test_stream_and_context(CUstream stream, CUcontext& context) {
  // CHECK: context = dpct::get_device_id(stream->get_device());
  cuStreamGetCtx(stream, &context);
}
