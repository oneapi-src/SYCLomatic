// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/types001.sycl.cpp

// CHECK: syclct::sycl_device_info deviceProp;
cudaDeviceProp deviceProp;

// CHECK: const syclct::sycl_device_info deviceProp1 = {};
const cudaDeviceProp deviceProp1 = {};

// CHECK: volatile syclct::sycl_device_info deviceProp2;
volatile cudaDeviceProp deviceProp2;

// CHECK:  void foo(syclct::sycl_device_info p) try {
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

// CHECK: int apicall(int i) try {
cudaError_t apicall(int i) {
  return cudaSuccess;
};

// CHECK: int err = apicall(0);
cudaError_t err = apicall(0);

template <typename T>
// CHECK: void check(T result, char const *const func) try {
void check(T result, char const *const func) {
}

#define checkCudaErrors(val) check((val), #val)

int main(int argc, char **argv) {

  checkCudaErrors(apicall(0));
  return 0;
}
