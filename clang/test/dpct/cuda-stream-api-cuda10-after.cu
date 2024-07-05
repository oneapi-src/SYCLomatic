// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none --use-experimental-features=graphs -out-root %T/cuda-stream-api-cuda10-after %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cuda-stream-api-cuda10-after/cuda-stream-api-cuda10-after.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST %T/cuda-stream-api-cuda10-after/cuda-stream-api-cuda10-after.dp.cpp -o %T/cuda-stream-api-cuda10-after/cuda-stream-api-cuda10-after.dp.o %}

#ifndef BUILD_TEST

template <typename T>
// CHECK: void my_error_checker(T ReturnValue, char const *const FuncName) {
void my_error_checker(T ReturnValue, char const *const FuncName) {
}

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)

int main() {
  cudaStream_t s0;
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
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaStreamIsCapturing was replaced with 0 because SYCL currently does not support capture operations on queues.
  // CHECK-NEXT: */
  // CHECK: MY_ERROR_CHECKER(0);
  MY_ERROR_CHECKER(cudaStreamIsCapturing(s0, &captureStatus));

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaStreamCaptureStatusInvalidated is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (captureStatus == cudaStreamCaptureStatusInvalidated) {
  if (captureStatus == cudaStreamCaptureStatusInvalidated) {
    return -1;
  }

  return 0;
}
#endif
