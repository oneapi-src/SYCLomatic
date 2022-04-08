// RUN: c2s --usm-level=none -out-root %T/async-error-handler %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda --always-use-async-handler -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/async-error-handler/async-error-handler.dp.cpp --match-full-lines %s


int main() {
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK: sycl::queue *s0, *s1, *s2;
  cudaStream_t s0, s1, s2;

  // CHECK: s0 = dev_ct1.create_queue(true);
  cudaStreamCreate(&s0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag and priority options.
  // CHECK-NEXT: */
  // CHECK-NEXT: s1 = dev_ct1.create_queue(true);
  cudaStreamCreateWithFlags(&s1, cudaStreamDefault);

  // CHECK: /*
  // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag and priority options.
  // CHECK-NEXT: */
  // CHECK-NEXT: s2 = dev_ct1.create_queue(true);
  cudaStreamCreateWithPriority(&s2, cudaStreamDefault, 2);
}

