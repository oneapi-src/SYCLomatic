// RUN: dpct --format-range=none --usm-level=none -out-root %T/kernel_without_name %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel_without_name/kernel_without_name.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/kernel_without_name/kernel_without_name.dp.cpp -o %T/kernel_without_name/kernel_without_name.dp.o %}

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
  // CHECK: dpct::get_out_of_order_queue().parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(griddim * threaddim, threaddim),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         helloFromGPU(i, item_ct1);
  // CHECK-NEXT:       });
  helloFromGPU<<<griddim, threaddim>>>(i);

}

struct TestThis {
  struct TestMember {
    int arg1, arg2;
  } args;
  int arg3;
  dim3 griddim, threaddim;
  void test() {
    // CHECK: dpct::get_out_of_order_queue().submit(
    // CHECK-NEXT:   [&](sycl::handler &cgh) {
    // CHECK-NEXT:     int args_arg1_ct0 = args.arg1;
    // CHECK-NEXT:     int args_arg2_ct1 = args.arg2;
    // CHECK-NEXT:     int arg3_ct2 = arg3;
    // CHECK-EMPTY:
    // CHECK-NEXT:     cgh.parallel_for(
    // CHECK-NEXT:       sycl::nd_range<3>(griddim * threaddim, threaddim),
    // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:         testKernel(args_arg1_ct0, args_arg2_ct1, arg3_ct2, item_ct1);
    // CHECK-NEXT:       });
    // CHECK-NEXT:   });
    testKernel<<<griddim, threaddim>>>(args.arg1, args.arg2, arg3);
  }
};

int main() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
  dim3 griddim = 2;
  dim3 threaddim = 32;
  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK:  cgh.parallel_for(
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);

  int karg1int = 1;
  int karg2int = 2;
  int karg3int = 3;
  int intvar = 20;
  // CHECK:     q_ct1.parallel_for(
  testKernel<<<10, intvar>>>(karg1int, karg2int, karg3int);

  struct KernelPointer {
    const int *arg1, *arg2;
  } args;
  //CHECK:q_ct1.submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    dpct::access_wrapper<const int *> args_arg1_acc_ct0(args.arg1, cgh);
  //CHECK-NEXT:    dpct::access_wrapper<const int *> args_arg2_acc_ct1(args.arg2, cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:    cgh.parallel_for(
  testKernelPtr<<<dim3(1), dim3(1, 2)>>>(args.arg1, args.arg2, karg3int);

  // CHECK:   q_ct1.parallel_for(
  testKernel <<<griddim.x, griddim.y + 2 >>>(karg1int, karg2int, karg3int);

  // CHECK:   q_ct1.parallel_for(
  helloFromGPU <<<2, 4>>>(23);

  // CHECK:   q_ct1.parallel_for(
  helloFromGPU <<<2, 4>>>();

  // CHECK:   q_ct1.parallel_for(
  helloFromGPU2 <<<2, 3>>>();

  // CHECK:   q_ct1.parallel_for(
  helloFromGPU<<<2, threaddim>>>();

  // CHECK:   q_ct1.parallel_for(
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

  int run_foo() {
    // CHECK: dpct::get_out_of_order_queue().submit(
    // CHECK-NEXT:   [&](sycl::handler &cgh) {
    // CHECK-NEXT:     int a_ct0 = a;
    // CHECK-NEXT:     int aa_b_ct1 = aa.b;
    // CHECK-NEXT:     int aa_c_d_ct2 = aa.c.d;
    // CHECK-EMPTY:
    // CHECK-NEXT:     cgh.parallel_for(
    // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:         foo_kernel(a_ct0, aa_b_ct1, aa_c_d_ct2);
    // CHECK-NEXT:       });
    // CHECK-NEXT:   });
    foo_kernel<<<1, 1>>>(a, aa.b, aa.c.d);
  }

private:
  int a;
  struct config aa;
};

int *g_a;

__global__ void foo_kernel3(int *d) {
}
//CHECK:dpct::get_out_of_order_queue().submit(
//CHECK-NEXT:        [&](sycl::handler &cgh) {
//CHECK-NEXT:          dpct::access_wrapper<int *> g_a_acc_ct0(g_a, cgh);
//CHECK-EMPTY:
//CHECK-NEXT:          cgh.parallel_for(
void run_foo(dim3 c, dim3 d) {
  if (1)
    foo_kernel3<<<c, 1>>>(g_a);
}

void run_foo2(dim3 c, dim3 d) {
//CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
//CHECK:q_ct1.submit(
//CHECK-NEXT:        [&](sycl::handler &cgh) {
//CHECK-NEXT:          dpct::access_wrapper<int *> g_a_acc_ct0(g_a, cgh);
//CHECK-EMPTY:
//CHECK-NEXT:          cgh.parallel_for(
//CHECK:  q_ct1.submit(
//CHECK-NEXT:        [&](sycl::handler &cgh) {
//CHECK-NEXT:          dpct::access_wrapper<int *> g_a_acc_ct0(g_a, cgh);
//CHECK-EMPTY:
//CHECK-NEXT:          cgh.parallel_for(
  if (1)
    foo_kernel3<<<c, d>>>(g_a);
  else
    foo_kernel3<<<c, 1>>>(g_a);
}
//CHECK:dpct::get_out_of_order_queue().submit(
//CHECK-NEXT:        [&](sycl::handler &cgh) {
//CHECK-NEXT:          dpct::access_wrapper<int *> g_a_acc_ct0(g_a, cgh);
//CHECK-EMPTY:
//CHECK-NEXT:          cgh.parallel_for(
void run_foo3(dim3 c, dim3 d) {
  for (;;)
    foo_kernel3<<<c, d>>>(g_a);
}
//CHECK:dpct::get_out_of_order_queue().submit(
//CHECK-NEXT:       [&](sycl::handler &cgh) {
//CHECK-NEXT:         dpct::access_wrapper<int *> g_a_acc_ct0(g_a, cgh);
//CHECK-EMPTY:
//CHECK-NEXT:         cgh.parallel_for(
void run_foo4(dim3 c, dim3 d) {
 while (1)
   foo_kernel3<<<c, 1>>>(g_a);
}

template <class T> struct A {
  T *ptr;
  T *operator~() { return ptr; }
};

template <class T> __global__ void foo_kernel4(T *t) {}

template <class T> void run_foo5(A<T> &a) {
  //CHECK:dpct::get_out_of_order_queue().submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    auto a_ct0 = ~a;
  //CHECK-EMPTY:
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        foo_kernel4(a_ct0);
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  foo_kernel4<<<1, 1>>>(~a);
}

__global__ void foo_kernel5(unsigned int ui) {}

void run_foo6() {
  dim3 grid;
  //CHECK:q_ct1.submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    unsigned int grid_x_grid_y_ct0 = grid[2] * grid[1];
  //CHECK-EMPTY:
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        foo_kernel5(grid_x_grid_y_ct0);
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  foo_kernel5<<<1, 1>>>(grid.x * grid.y);
  //CHECK:q_ct1.submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    unsigned int grid_x_ct0 = ++grid[2];
  //CHECK-EMPTY:
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        foo_kernel5(grid_x_ct0);
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  foo_kernel5<<<1, 1>>>(++grid.x);
}

template <typename T>
__global__ void foo_kernel6(T *a, const T *b, const T *c, const T *d,
                            const T *e, const int f, const int g, const int h) {
  const int index = threadIdx.x;
}

template <typename T>
__global__ void foo_kernel6(T *a, const T *b, const T *c, const T *d,
                            const int f, const int g, const int h) {
  const int index = threadIdx.x;
}

template <typename T>
void run_foo7(T *a, const T *b, const T *c, const T *d, const T *e, const int f,
              const int g, const int h, cudaStream_t stream) {
  dim3 grid(1);
  dim3 block(1);
  if (a == e) {
    //CHECK:cgh.parallel_for(
    //CHECK-NEXT:  sycl::nd_range<3>(grid * block, block), 
    //CHECK-NEXT:  [=](sycl::nd_item<3> item_ct1) {
    //CHECK-NEXT:    foo_kernel6(a_acc_ct0.get_raw_pointer(), b_acc_ct1.get_raw_pointer(), c_acc_ct2.get_raw_pointer(), d_acc_ct3.get_raw_pointer(), f, g, h, item_ct1);
    //CHECK-NEXT:  });
    foo_kernel6<<<grid, block, 0, stream>>>(a, b, c, d, f, g, h);
  } else {
    //CHECK:cgh.parallel_for(
    //CHECK-NEXT:  sycl::nd_range<3>(grid * block, block), 
    //CHECK-NEXT:  [=](sycl::nd_item<3> item_ct1) {
    //CHECK-NEXT:    foo_kernel6(a_acc_ct0.get_raw_pointer(), b_acc_ct1.get_raw_pointer(), c_acc_ct2.get_raw_pointer(), d_acc_ct3.get_raw_pointer(), e_acc_ct4.get_raw_pointer(), f, g, h, item_ct1);
    //CHECK-NEXT:  });
    foo_kernel6<<<grid, block, 0, stream>>>(a, b, c, d, e, f, g, h);
  }
}
