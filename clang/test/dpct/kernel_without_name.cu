// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel_without_name.dp.cpp --match-full-lines %s

__global__ void testKernel(int L, int M, int N);

__global__ void testKernelPtr(const int *L, const int *M, int N) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void testKernel(int L, int M, int N) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void helloFromGPU(int i) {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
}

__global__ void helloFromGPU(void) {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
}

__global__ void helloFromGPU2() {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
}

void testReference(const int &i) {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  // CHECK:         auto dpct_global_range = griddim * threaddim;
  // CHECK-NEXT:    cgh.parallel_for(
  helloFromGPU<<<griddim, threaddim>>>(i);

}

struct TestThis {
  struct TestMember {
    int arg1, arg2;
  } args;
  int arg3;
  dim3 griddim, threaddim;
  void test() {
    // CHECK:           auto dpct_global_range = griddim * threaddim;
    // CHECK-NEXT:      cgh.parallel_for(
    testKernel<<<griddim, threaddim>>>(args.arg1, args.arg2, arg3);
  }
};

int main() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK:           auto dpct_global_range = griddim * threaddim;
  // CHECK-NEXT:      cgh.parallel_for(
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);

  int karg1int = 1;
  int karg2int = 2;
  int karg3int = 3;
  int intvar = 20;
  // CHECK:     dpct::get_default_queue().submit(
  // CHECK-NEXT:  [&](sycl::handler &cgh) {
  // CHECK-NEXT:    cgh.parallel_for(
  testKernel<<<10, intvar>>>(karg1int, karg2int, karg3int);

  struct KernelPointer {
    const int *arg1, *arg2;
  } args;
  // CHECK: auto args_arg1_acc_ct0 = args_arg1_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:      auto args_arg2_acc_ct1 = args_arg2_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:      cgh.parallel_for(
  testKernelPtr<<<dim3(1), dim3(1, 2)>>>(args.arg1, args.arg2, karg3int);

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:  [&](sycl::handler &cgh) {
  // CHECK-NEXT:    cgh.parallel_for(
  testKernel <<<griddim.x, griddim.y + 2 >>>(karg1int, karg2int, karg3int);

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:  [&](sycl::handler &cgh) {
  // CHECK-NEXT:    cgh.parallel_for(
  helloFromGPU <<<2, 4>>>(23);

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:  [&](sycl::handler &cgh) {
  // CHECK-NEXT:    cgh.parallel_for(
  helloFromGPU <<<2, 4>>>();

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:  [&](sycl::handler &cgh) {
  // CHECK-NEXT:    cgh.parallel_for(
  helloFromGPU2 <<<2, 3>>>();

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:  [&](sycl::handler &cgh) {
  // CHECK-NEXT:    cgh.parallel_for(
  helloFromGPU<<<2, threaddim>>>();

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:  [&](sycl::handler &cgh) {
  // CHECK-NEXT:    cgh.parallel_for(
  helloFromGPU<<<griddim, 4>>>();
}

struct config {
  int b;
  struct subconfig {
    int d;
  } c;
};

__global__ void foo_kernel(int a, int b, int c) {}

class foo_class {
public:
  foo_class(int n) : a(n) {}

  // CHECK:  dpct::get_default_queue().submit(
  //CHECK-NEXT:      [&](sycl::handler &cgh) {
  //CHECK-NEXT:        cgh.parallel_for(
  int run_foo() {
    foo_kernel<<<1, 1>>>(a, aa.b, aa.c.d);
  }

private:
  int a;
  struct config aa;
};

__global__ void foo_kernel3(int *d) {
}
//CHECK:dpct::get_default_queue().submit(
//CHECK-NEXT:        [&](sycl::handler &cgh) {
//CHECK-NEXT:          auto acc_ct0 = buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
//CHECK-NEXT:          cgh.parallel_for(
void run_foo(dim3 c, dim3 d) {
  if (1)
    foo_kernel3<<<c, 1>>>(0);
}
//CHECK:dpct::get_default_queue().submit(
//CHECK-NEXT:        [&](sycl::handler &cgh) {
//CHECK-NEXT:          auto acc_ct0 = buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
//CHECK-NEXT:          auto dpct_global_range = c * d;
//CHECK-NEXT:          cgh.parallel_for(
//CHECK:  dpct::get_default_queue().submit(
//CHECK-NEXT:        [&](sycl::handler &cgh) {
//CHECK-NEXT:          auto acc_ct0 = buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
//CHECK-NEXT:          cgh.parallel_for(
void run_foo2(dim3 c, dim3 d) {
  if (1)
    foo_kernel3<<<c, d>>>(0);
  else
    foo_kernel3<<<c, 1>>>(0);
}
//CHECK:dpct::get_default_queue().submit(
//CHECK-NEXT:        [&](sycl::handler &cgh) {
//CHECK-NEXT:          auto acc_ct0 = buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
//CHECK-NEXT:          auto dpct_global_range = c * d;
//CHECK-NEXT:          cgh.parallel_for(
void run_foo3(dim3 c, dim3 d) {
  for (;;)
    foo_kernel3<<<c, d>>>(0);
}
//CHECK:dpct::get_default_queue().submit(
//CHECK-NEXT:       [&](sycl::handler &cgh) {
//CHECK-NEXT:         auto acc_ct0 = buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
//CHECK-NEXT:         cgh.parallel_for(
void run_foo4(dim3 c, dim3 d) {
 while (1)
   foo_kernel3<<<c, 1>>>(0);
}
