// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --use-experimental-features=graph --format-range=none -out-root %T/cudaStreamCaptureStatus_enum_test %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cudaStreamCaptureStatus_enum_test/cudaStreamCaptureStatus_enum_test.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/cudaStreamCaptureStatus_enum_test/cudaStreamCaptureStatus_enum_test.dp.cpp -o %T/cudaStreamCaptureStatus_enum_test/cudaStreamCaptureStatus_enum_test.dp.o %}

#ifndef NO_BUILD_TEST
#include <cuda.h>

int main() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // CHECK: sycl::ext::oneapi::experimental::queue_state captureStatus = sycl::ext::oneapi::experimental::queue_state::executing;
  // CHECK-NEXT: captureStatus = sycl::ext::oneapi::experimental::queue_state::recording;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaStreamCaptureStatusInvalidated is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: captureStatus = cudaStreamCaptureStatusInvalidated;
  cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
  captureStatus = cudaStreamCaptureStatusActive;
  captureStatus = cudaStreamCaptureStatusInvalidated;

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaStreamCaptureStatusInvalidated is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (captureStatus == cudaStreamCaptureStatusInvalidated) {
  if (captureStatus == cudaStreamCaptureStatusInvalidated) {
    return -1;
  }
}
#endif
