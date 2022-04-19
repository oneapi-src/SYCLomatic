// RUN: dpct --format-range=none -out-root %T/test_for_test2018 %s --cuda-include-path="%cuda-path/include" --report-type=stats -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/test_for_test2018/test_for_test2018.dp.cpp --match-full-lines %s
#include <stdio.h>

// CHECK: void foo(){}
__global__ void foo(){}

// Code piece below is used to catch crash of SIGSEGV
#define HIPPER(name) cuda##name
int eventSynchronize(cudaEvent_t event) {
    return HIPPER(EventSynchronize)(event);
}

typedef HIPPER(Stream_t) stream_t;
typedef HIPPER(Event_t) event_t;

static const int streamDefault = HIPPER(StreamDefault);
static const int streamNonBlocking = HIPPER(StreamNonBlocking);

inline int streamCreate(stream_t *stream) {
  return HIPPER(StreamCreate)(stream);
}

inline int streamCreateWithFlags(stream_t *stream, unsigned int flags) {
  return HIPPER(StreamCreateWithFlags)(stream, flags);
}
