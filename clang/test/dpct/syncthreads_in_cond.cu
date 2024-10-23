// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/syncthreads_in_cond %s --cuda-include-path="%cuda-path/include" --use-experimental-features=nd_range_barrier,logical-group -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --input-file %T/syncthreads_in_cond/syncthreads_in_cond.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/syncthreads_in_cond/syncthreads_in_cond.dp.cpp -o %T/syncthreads_in_cond/syncthreads_in_cond.dp.o %}
#ifndef NO_BUILD_TEST
__device__ void test1(int *a) {
  unsigned tid =
      ((blockIdx.x + (blockIdx.y * gridDim.x)) * (blockDim.x * blockDim.y)) +
      (threadIdx.x + (threadIdx.y * blockDim.x));

  // check-achor-begin-1
  __syncthreads();
  // check-achor-end-1
  // CHECK: check-achor-begin-1
  // CHECK-NOT: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
  // CHECK: check-achor-end-1

  // switch
  switch (tid) {
  case 0:
    a[0] = 1;
    // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
    __syncthreads();
    break;
  case 1:
    a[1] = 1;
    // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
    __syncthreads();
    break;
  default:
    a[tid] = a[tid - 1] + a[tid - 2];
  }

  // if
  if (tid > 32) {
    // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
    __syncthreads();
  } else {
    a[tid] = 0;
    // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
    __syncthreads();
  }

  // do
  do {
    a[tid] = a[tid - 1] + a[tid - 2];
    // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
    __syncthreads();
  } while (a[tid]);

  // while
  while (a[tid])
    // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
    __syncthreads();

  // early return
  if (tid < 32)
    return;

  // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
  __syncthreads();
  a[tid] = a[tid - 1] + a[tid - 2];
}

__device__ void test2(int *a) {
  if (threadIdx.x > 10)
    // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
    test1(a);

  if (threadIdx.x < 5)
    return;

  // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
  test1(a);
}

constexpr int const_expr() {
  return 10;
}

__device__ void test3(int *a) {

  // check-achor-begin-2
  test2(a);
  // check-achor-end-2
  // CHECK: check-achor-begin-2
  // CHECK-NOT: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
  // CHECK: check-achor-end-2

  if (threadIdx.x > 10)
    // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
    test2(a);

  if (threadIdx.x < 5)
    return;
  // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
  test2(a);

  for (int i = 0; i < 10; ++i) {
    // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
    test2(a);
  }

  do {
    // CHECK: DPCT1118:{{.*}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
    test2(a);
  } while (*a < 10);
}

__global__ void test_no_warn() {

  // CHECK: for (;;)
  // CHECK: item_ct1.barrier();
  for (;;)
    __syncthreads();

  // CHECK: for (; const_expr();)
  // CHECK: item_ct1.barrier();
  for (; const_expr();)
    __syncthreads();

  // CHECK: if (const_expr())
  // CHECK: item_ct1.barrier();
  if (const_expr())
    __syncthreads();

  // CHECK: do {
  // CHECK: item_ct1.barrier();
  // CHECK: } while (const_expr());
  do {
    __syncthreads();
  } while (const_expr());

  // CHECK: while (const_expr())
  // CHECK: item_ct1.barrier();
  while (const_expr())
    __syncthreads();

  // CHECK: switch (const_expr()) {
  // CHECK-NEXT: case 10:
  // CHECK-NEXT: item_ct1.barrier();
  // CHECK-NEXT: break;
  // CHECK-NEXT: default:
  // CHECK-NEXT: }
  switch (const_expr()) {
  case 10:
    __syncthreads();
    break;
  default:
    break;
  }
}
#endif
