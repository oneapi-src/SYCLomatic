// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none -out-root %T/cuda-stream-api-cuda10-after %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cuda-stream-api-cuda10-after/cuda-stream-api-cuda10-after.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cuda-stream-api-cuda10-after/cuda-stream-api-cuda10-after.dp.cpp -o %T/cuda-stream-api-cuda10-after/cuda-stream-api-cuda10-after.dp.o %}

template <typename T>
// CHECK: void my_error_checker(T ReturnValue, char const *const FuncName) {
void my_error_checker(T ReturnValue, char const *const FuncName) {
}

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)

int main() {
  cudaStream_t s0;
  // CHECK: int captureStatus = 0;
  // CHECK-NEXT: captureStatus = 0;
  // CHECK-NEXT: captureStatus = 0;
  cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
  captureStatus = cudaStreamCaptureStatusActive;
  captureStatus = cudaStreamCaptureStatusInvalidated;
  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cudaStreamIsCapturing was replaced with 0 because SYCL currently does not support capture operations on queues.
  // CHECK-NEXT: */
  // CHECK: MY_ERROR_CHECKER(0);
  MY_ERROR_CHECKER(cudaStreamIsCapturing(s0, &captureStatus));

  return 0;
}
