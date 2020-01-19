// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/types001.dp.cpp

// CHECK: dpct::device_info deviceProp;
cudaDeviceProp deviceProp;

// CHECK: const dpct::device_info deviceProp1 = {};
const cudaDeviceProp deviceProp1 = {};

// CHECK: volatile dpct::device_info deviceProp2;
volatile cudaDeviceProp deviceProp2;

// CHDCK: sycl::event events[23];
cudaEvent_t events[23];
// CHECK: const sycl::event *pevents[23];
const cudaEvent_t *pevents[23];
// CHECK: const sycl::event **ppevents[23];
const cudaEvent_t **ppevents[23];

// CHECK: int errors[23];
cudaError_t errors[23];
// CHECK: const int *perrors[23];
const cudaError_t *perrors[23];
// CHECK: const int **pperrors[23];
const cudaError_t **pperrors[23];

// CHECK: int errors1[23];
cudaError errors1[23];
// CHECK: const int *perrors1[23];
const cudaError *perrors1[23];
// CHECK: const int **pperrors1[23];
const cudaError **pperrors1[23];

// CHECK: sycl::range<3> dims[23];
dim3 dims[23];
// CHECK: const sycl::range<3> *pdims[23];
const dim3 *pdims[23];
// CHECK: const sycl::range<3> **ppdims[23];
const dim3 **ppdims[23];

struct s {
  // CHECK: sycl::event events[23];
  cudaEvent_t events[23];
  // CHECK: const sycl::event *pevents[23];
  const cudaEvent_t *pevents[23];
  // CHECK: const sycl::event **ppevents[23];
  const cudaEvent_t **ppevents[23];

  // CHECK: int errors[23];
  cudaError_t errors[23];
  // CHECK: const int *perrors[23];
  const cudaError_t *perrors[23];
  // CHECK: const int **pperrors[23];
  const cudaError_t **pperrors[23];

  // CHECK: int errors1[23];
  cudaError errors1[23];
  // CHECK: const int *perrors1[23];
  const cudaError *perrors1[23];
  // CHECK: const int **pperrors1[23];
  const cudaError **pperrors1[23];

  // CHECK: sycl::range<3> dims[23];
  dim3 dims[23];
  // CHECK: const sycl::range<3> *pdims[23];
  const dim3 *pdims[23];
  // CHECK: const sycl::range<3> **ppdims[23];
  const dim3 **ppdims[23];
};

// CHECK:  void foo(dpct::device_info p) {
void foo(cudaDeviceProp p) {
  return;
}

// CHECK: int e;
cudaError e;

// CHECK: int ee;
cudaError_t ee;

// CHECK: int foo_0(int);
cudaError_t foo_0(cudaError_t);

// CHECK: int foo_1(int);
cudaError foo_1(cudaError_t);

// CHECK: int apicall(int i) {
cudaError_t apicall(int i) {
  return cudaSuccess;
};

// CHECK: int err = apicall(0);
cudaError_t err = apicall(0);

template <typename T>
// CHECK: void check(T result, char const *const func) {
void check(T result, char const *const func) {
}

#define checkCudaErrors(val) check((val), #val)

int main(int argc, char **argv) {

  checkCudaErrors(apicall(0));
  return 0;
}

__global__ void foo() {
  void *p;
  // CHECK: (queue_p)p;
  // CHECK-NEXT: (queue_p *)p;
  // CHECK-NEXT: (queue_p **)p;
  // CHECK-NEXT: (queue_p ***)p;
  (cudaStream_t)p;
  (cudaStream_t *)p;
  (cudaStream_t **)p;
  (cudaStream_t ***)p;

  int i;
  // CHECK: (int)i;
  // CHECK-NEXT: (int *)p;
  // CHECK-NEXT: (int **)p;
  // CHECK-NEXT: (int ***)p;
  (cudaError)i;
  (cudaError *)p;
  (cudaError **)p;
  (cudaError ***)p;

  cudaDeviceProp cdp;
  // CHECK: dpct::device_info cdp2 = (dpct::device_info)cdp;
  // CHECK-NEXT: (dpct::device_info *)p;
  // CHECK-NEXT: (dpct::device_info **)p;
  // CHECK-NEXT: (dpct::device_info ***)p;
  cudaDeviceProp cdp2 = (cudaDeviceProp)cdp;
  (cudaDeviceProp *)p;
  (cudaDeviceProp **)p;
  (cudaDeviceProp ***)p;
}
