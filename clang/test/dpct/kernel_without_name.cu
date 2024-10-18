// RUN: dpct --format-range=none --usm-level=none -out-root %T/kernel_without_name %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel_without_name/kernel_without_name.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/kernel_without_name/kernel_without_name.dp.cpp -o %T/kernel_without_name/kernel_without_name.dp.o %}

#include "cuda_fp16.h"

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
    // CHECK-NEXT:     auto args_arg1_ct0 = args.arg1;
    // CHECK-NEXT:     auto args_arg2_ct1 = args.arg2;
    // CHECK-NEXT:     auto arg3_ct2 = arg3;
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
  //CHECK-NEXT:    dpct::access_wrapper args_arg1_acc_ct0(args.arg1, cgh);
  //CHECK-NEXT:    dpct::access_wrapper args_arg2_acc_ct1(args.arg2, cgh);
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
    // CHECK-NEXT:     auto a_ct0 = a;
    // CHECK-NEXT:     auto aa_b_ct1 = aa.b;
    // CHECK-NEXT:     auto aa_c_d_ct2 = aa.c.d;
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
//CHECK-NEXT:          dpct::access_wrapper g_a_acc_ct0(g_a, cgh);
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
//CHECK-NEXT:          dpct::access_wrapper g_a_acc_ct0(g_a, cgh);
//CHECK-EMPTY:
//CHECK-NEXT:          cgh.parallel_for(
//CHECK:  q_ct1.submit(
//CHECK-NEXT:        [&](sycl::handler &cgh) {
//CHECK-NEXT:          dpct::access_wrapper g_a_acc_ct0(g_a, cgh);
//CHECK-EMPTY:
//CHECK-NEXT:          cgh.parallel_for(
  if (1)
    foo_kernel3<<<c, d>>>(g_a);
  else
    foo_kernel3<<<c, 1>>>(g_a);
}
//CHECK:dpct::get_out_of_order_queue().submit(
//CHECK-NEXT:        [&](sycl::handler &cgh) {
//CHECK-NEXT:          dpct::access_wrapper g_a_acc_ct0(g_a, cgh);
//CHECK-EMPTY:
//CHECK-NEXT:          cgh.parallel_for(
void run_foo3(dim3 c, dim3 d) {
  for (;;)
    foo_kernel3<<<c, d>>>(g_a);
}
//CHECK:dpct::get_out_of_order_queue().submit(
//CHECK-NEXT:       [&](sycl::handler &cgh) {
//CHECK-NEXT:         dpct::access_wrapper g_a_acc_ct0(g_a, cgh);
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
  //CHECK-NEXT:    auto grid_x_grid_y_ct0 = grid.x * grid.y;
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
  //CHECK-NEXT:    auto grid_x_ct0 = ++grid.x;
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

#ifndef NO_BUILD_TEST
template <typename T> struct kernel_type_t {
  using Type = T;
};

// CHECK: /*
// CHECK-NEXT: DPCT1125:{{[0-9]+}}: The type "Tk" defined in function "foo_device7" is used as the parameter type in all functions in the call path from the corresponding sycl::handler::parallel_for() to the current function. You may need to adjust the definition location of the type.
// CHECK-NEXT: */
// CHECK-NEXT: template <typename T> 
// CHECK-NEXT: void foo_device7(int a,
// CHECK-NEXT:                  int b,
// CHECK-NEXT:                  Tk *mem) {
// CHECK-NEXT:   using Tk = typename kernel_type_t<T>::Type;
template <typename T> __global__ 
void foo_device7(int a,
                 int b) {
  using Tk = typename kernel_type_t<T>::Type;
  __shared__ Tk mem[256];
}

// CHECK: /*
// CHECK-NEXT: DPCT1125:{{[0-9]+}}: The type "Tk" defined in function "foo_device7" is used as the parameter type in all functions in the call path from the corresponding sycl::handler::parallel_for() to the current function. You may need to adjust the definition location of the type.
// CHECK-NEXT: */
// CHECK-NEXT: template <typename T> 
// CHECK-NEXT: void foo_kernel7(int a,
// CHECK-NEXT:                  int b,
// CHECK-NEXT:                  Tk *mem) {
// CHECK-NEXT:   foo_device7<T>(a, b, mem);
template <typename T> __global__
void foo_kernel7(int a,
                 int b) {
  foo_device7<T>(a, b);
}

template <typename T> 
void run_foo8() {
  // CHECK: int i;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1126:{{[0-9]+}}: The type "Tk" defined in function "foo_device7" is used as the parameter type in all functions in the call path from the sycl::handler::parallel_for() to the function "foo_device7". You may need to adjust the definition location of the type.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::get_out_of_order_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::local_accessor<Tk, 1> mem_acc_ct1(sycl::range<1>(256), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         foo_kernel7<T>(i, i, mem_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  int i;
  foo_kernel7<T><<<1, 1>>>(i, i);
}
#endif

// CHECK: typedef float TK1;
// CHECK-NEXT: void foo_kernel8(int a, int b, TK1 *mem) {
// CHECK-NEXT:   //local mem
// CHECK-NEXT: }
// CHECK-NEXT: void run_foo9() {
// CHECK-NEXT:   int i;
// CHECK-NEXT:   dpct::get_out_of_order_queue().submit(
// CHECK-NEXT:     [&](sycl::handler &cgh) {
// CHECK-NEXT:       sycl::local_accessor<TK1, 1> mem_acc_ct1(sycl::range<1>(256), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:       cgh.parallel_for(
// CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
// CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:           foo_kernel8(i, i, mem_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
// CHECK-NEXT:         });
// CHECK-NEXT:     });
// CHECK-NEXT: }
typedef float TK1;
__global__ void foo_kernel8(int a, int b) {
  __shared__ TK1 mem[256];//local mem
}
void run_foo9() {
  int i;
  foo_kernel8<<<1, 1>>>(i, i);
}

template <typename T> __global__ void foo_kernel9() { __shared__ T mem[256]; }

template <typename T> void run_foo10() {
  //      CHECK:    cgh.parallel_for(
  // CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:        foo_kernel9<T>(mem_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
  // CHECK-NEXT:      });
  foo_kernel9<T><<<1, 1>>>();
}

__global__ void foo_kernel11(float2 Input) {}

void run_foo11() {
  // CHECK: dpct::get_out_of_order_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::float2 NAN_NAN_ct0 = {NAN, NAN};
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         foo_kernel11(NAN_NAN_ct0);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  foo_kernel11<<<1, 1>>>({NAN, NAN});
}

struct TYPE_A {
    float x, y;
    TYPE_A() = default;
    __host__ __device__ TYPE_A(const float &a, const float &b) : x(a), y(b) { }
};

__global__ void foo_kernel12(TYPE_A Input) {}

void run_foo12() {
  // CHECK: dpct::get_out_of_order_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     TYPE_A NAN_NAN_ct0 = {NAN, NAN};
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         foo_kernel12(NAN_NAN_ct0);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  foo_kernel12<<<1, 1>>>({NAN, NAN});
}

template<class T, bool B>
__global__ void foobar(T a) {}

// CHECK: template<class T>
// CHECK-NEXT: void foobar(T a) {
// CHECK-NEXT:   dpct::get_out_of_order_queue().parallel_for(
// CHECK-NEXT:     sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
// CHECK-NEXT:     [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:       foobar<T, true>(a);
// CHECK-NEXT:     });
// CHECK-NEXT: }
template<class T>
void foobar(T a) {
  foobar<T, true><<<1, 1>>>(a);
}

void run_foobar() {
  foobar<float>(1.0f);
}

class base {
public:
  base();
};
base::base() {}

class derived : public base {
public:
  derived();
  ~derived();
};
derived::derived() {}
derived::~derived() {}

struct A2 {
  A2(int) {}
  operator int() const { return 7; }
};

__global__ void kfunc(base arg, A2, int) {}

void func() {
  derived der;
  A2 a2(1);
  // CHECK: dpct::get_out_of_order_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     base der_ct0 = der;
  // CHECK-NEXT:     A2 ct1 = 1;
  // CHECK-NEXT:     int a2_ct2 = a2;
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 128) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kfunc(der_ct0, ct1, a2_ct2);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kfunc<<<128, 32>>>(der, 1, a2);
}

template <class T> __global__ void foo_kernel13(const T *a);

enum FLOATING_TYPE { FT_FLOAT, FT_DOUBLE };

struct Mat {
  template <class U> U *data() { return (U *)_data; }
  FLOATING_TYPE getType() { return _ft; }

  void *_data;
  FLOATING_TYPE _ft;
};

#define DISPATCH(type, functor)                                                \
  {                                                                            \
    switch (type) {                                                            \
    case FT_FLOAT: {                                                           \
      using scalar_t = float;                                                  \
      functor();                                                               \
      break;                                                                   \
    }                                                                          \
    case FT_DOUBLE: {                                                          \
      using scalar_t = double;                                                 \
      functor();                                                               \
      break;                                                                   \
    }                                                                          \
    }                                                                          \
  }

void run_foo13(Mat mat) {
  // CHECK: DISPATCH(mat.getType(), ([&] { dpct::get_out_of_order_queue().submit(
  // CHECK-NEXT: [&](sycl::handler &cgh) {
  // CHECK-NEXT:   dpct::access_wrapper mat_data_scalar_t_acc_ct0(mat.data<scalar_t>(), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:     sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:     [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:       foo_kernel13(mat_data_scalar_t_acc_ct0.get_raw_pointer());
  // CHECK-NEXT:     });
  // CHECK-NEXT: }); }));
  DISPATCH(mat.getType(), ([&] { foo_kernel13<<<1, 1>>>(mat.data<scalar_t>()); }));
}

template <class T> __global__ void foo_kernel13(const T *a) {}
#undef DISPATCH
