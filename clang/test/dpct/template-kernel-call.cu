// FIXME
// UNSUPPORTED: system-windows
// RUN: dpct --format-range=none --usm-level=none -out-root %T/template-kernel-call %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++11
// RUN: FileCheck --input-file %T/template-kernel-call/template-kernel-call.dp.cpp --match-full-lines %s

#include <vector>

void printf(const char *format, unsigned char data);

__global__ void kernel(int a, int b){
  return;
}
// CHECK: template<typename T>
// CHECK-NEXT: class testClass{
// CHECK-NEXT:   struct test {
// CHECK-NEXT:     int data;
// CHECK-NEXT:   };
// CHECK-NEXT:   test* ptest;
// CHECK-NEXT:   int a;
// CHECK-NEXT: public:
// CHECK-NEXT:   void run(){
// CHECK-NEXT:     dpct::get_out_of_order_queue().submit(
// CHECK-NEXT:       [&](sycl::handler &cgh) {
// CHECK-NEXT:         int this_a_ct0 = this->a;
// CHECK-NEXT:         int ptest_data_ct1 = ptest->data;
// CHECK-EMPTY:
// CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
// CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:             kernel(this_a_ct0, ptest_data_ct1);
// CHECK-NEXT:           });
// CHECK-NEXT:       });
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: };
template<typename T>
class testClass{
  struct test {
    int data;
  };
  test* ptest;
  int a;
public:
  void run(){
    kernel<<<1,1>>>(this->a, ptest->data);
    return;
  }
};

template<class T> void runTest();

template <class TName, unsigned N, class TData>
// CHECK: void testKernelPtr(const TData *L, const TData *M,
// CHECK-NEXT: const sycl::nd_item<3> &[[ITEMNAME:item_ct1]]) {
__global__ void testKernelPtr(const TData *L, const TData *M) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(2) * [[ITEMNAME]].get_local_range(2) + [[ITEMNAME]].get_local_id(2);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

template<class TData>
// CHECK: void testKernel(TData L, TData M, int N, const sycl::nd_item<3> &[[ITEMNAME:item_ct1]]) {
__global__ void testKernel(TData L, TData M, int N) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(2) * [[ITEMNAME]].get_local_range(2) + [[ITEMNAME]].get_local_id(2);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  L = M;
}

// CHECK: struct __dpct_align__(8) LA {
struct __align__(8) LA {
  unsigned int l, a;
};

template<class T>
class TestTemplate {
public:
  T data;
};

const unsigned ktarg = 80;
dim3 griddim = 2;
dim3 threaddim = 32;

class TestName;

template<class T>
void runTest() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
  typedef TestTemplate<T> TT;
  const void *karg1 = 0;
  const T *karg2 = 0;
  T *karg3 = 0;
  const TestTemplate<T> *karg4 = 0;
  TT *karg5 = 0;

  // CHECK:/*
  // CHECK-NEXT:DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT:*/
  // CHECK-NEXT: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<const T *> karg1_acc_ct0((const T *)karg1, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<const T *> karg2_acc_ct1(karg2, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, TestName, dpct_kernel_scalar<ktarg>, T>>(
  // CHECK-NEXT:       sycl::nd_range<3>(griddim * threaddim, threaddim),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         testKernelPtr<TestName, ktarg, T>(karg1_acc_ct0.get_raw_pointer(), karg2_acc_ct1.get_raw_pointer(), item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  testKernelPtr<TestName, ktarg, T><<<griddim, threaddim>>>((const T *)karg1, karg2);

  // CHECK:/*
  // CHECK-NEXT:DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT:*/
  // CHECK-NEXT: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<const T *> karg1_acc_ct0((const T *)karg1, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<T *> karg3_acc_ct1(karg3, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, class TestTemplate<T>, dpct_kernel_scalar<ktarg>, T>>(
  // CHECK-NEXT:       sycl::nd_range<3>(griddim * threaddim, threaddim),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         testKernelPtr<class TestTemplate<T>, ktarg, T>(karg1_acc_ct0.get_raw_pointer(), karg3_acc_ct1.get_raw_pointer(), item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  testKernelPtr<class TestTemplate<T>, ktarg, T><<<griddim, threaddim>>>((const T *)karg1, karg3);

  // CHECK:/*
  // CHECK-NEXT:DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT:*/
  // CHECK-NEXT: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<const TestTemplate<T> *> karg4_acc_ct0(karg4, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<TT *> karg5_acc_ct1(karg5, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, T, dpct_kernel_scalar<ktarg>, TestTemplate<T>>>(
  // CHECK-NEXT:       sycl::nd_range<3>(griddim * threaddim, threaddim),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         testKernelPtr<T, ktarg, TestTemplate<T>>(karg4_acc_ct0.get_raw_pointer(), karg5_acc_ct1.get_raw_pointer(), item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  testKernelPtr<T, ktarg, TestTemplate<T> ><<<griddim, threaddim>>>(karg4, karg5);

  T karg1T, karg2T;
  // CHECK:/*
  // CHECK-NEXT:DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto ktarg_ct2 = ktarg;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:         sycl::nd_range<3>(griddim * threaddim, threaddim),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel<T>(karg1T, karg2T, ktarg_ct2, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  testKernel<T><<<griddim, threaddim>>>(karg1T, karg2T, ktarg);

  TestTemplate<T> karg3TT;
  TT karg4TT;

  // CHECK:/*
  // CHECK-NEXT:DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT:*/
  // CHECK-NEXT:/*
  // CHECK-NEXT:DPCT1129:{{[0-9]+}}: The type "TestTemplate<T>" is used in the SYCL kernel, but it is not device copyable. The sycl::is_device_copyable specialization has been added for this type. Please review the code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto ktarg_ct2 = ktarg;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}, TestTemplate<T>>>(
  // CHECK-NEXT:         sycl::nd_range<3>(griddim * threaddim, threaddim),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel<TestTemplate<T>>(karg3TT, karg4TT, ktarg_ct2, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  testKernel<TestTemplate<T> ><<<griddim, threaddim>>>(karg3TT, karg4TT, ktarg);

  // CHECK:/*
  // CHECK-NEXT:DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT:*/
  // CHECK-NEXT:/*
  // CHECK-NEXT:DPCT1129:{{[0-9]+}}: The type "TestTemplate<T>" is used in the SYCL kernel, but it is not device copyable. The sycl::is_device_copyable specialization has been added for this type. Please review the code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto ktarg_ct2 = ktarg;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}, TT>>(
  // CHECK-NEXT:         sycl::nd_range<3>(griddim * threaddim, threaddim),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel<TT>(karg3TT, karg4TT, ktarg_ct2, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  testKernel<TT><<<griddim, threaddim>>>(karg3TT, karg4TT, ktarg);
}

template void runTest<int>();
template void runTest<float>();

int main() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
  void *karg1 = 0;
  LA *karg2 = 0;
  // CHECK:/*
  // CHECK-NEXT:DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT:*/
  // CHECK-NEXT: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<const LA *> karg1_acc_ct0((const LA *)karg1, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<const LA *> karg2_acc_ct1(karg2, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, TestName, dpct_kernel_scalar<ktarg>, LA>>(
  // CHECK-NEXT:       sycl::nd_range<3>(griddim * threaddim, threaddim),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         testKernelPtr<TestName, ktarg, LA>(karg1_acc_ct0.get_raw_pointer(), karg2_acc_ct1.get_raw_pointer(), item_ct1);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  testKernelPtr<TestName, ktarg, LA><<<griddim, threaddim>>>((const LA *)karg1, karg2);

  LA karg1LA, karg2LA;
  int intvar = 20;
  // CHECK:/*
  // CHECK-NEXT:DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       int ktarg_ct2 = ktarg;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}, LA>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 10) * sycl::range<3>(1, 1, intvar), sycl::range<3>(1, 1, intvar)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel<LA>(karg1LA, karg2LA, ktarg_ct2, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  testKernel<LA><<<10, intvar>>>(karg1LA, karg2LA, ktarg);
}


// CHECK:template<typename T>
// CHECK-NEXT:void convert_kernel(T b, const sycl::nd_item<3> &item_ct1, int *aaa,
// CHECK-NEXT:                    sycl::local_accessor<double, 2> bbb){
// CHECK:  T a = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
// CHECK-NEXT:}
template<typename T>
__global__ void convert_kernel(T b){
  __shared__ int aaa[0];
  __shared__ double bbb[8][0];
  T a = blockDim.x * blockIdx.x + threadIdx.x;
}

// CHECK:template<typename T>
// CHECK-NEXT:void convert(){
// CHECK-NEXT:  T b;
// CHECK-NEXT:  {
// CHECK-NEXT:  dpct::has_capability_or_fail(dpct::get_out_of_order_queue().get_device(), {sycl::aspect::fp64});
// CHECK-EMPTY:
// CHECK-NEXT:  dpct::get_out_of_order_queue().submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      sycl::local_accessor<int, 1> aaa_acc_ct1(sycl::range<1>(0), cgh);
// CHECK-NEXT:      sycl::local_accessor<double, 2> bbb_acc_ct1(sycl::range<2>(8, 0), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class convert_kernel_{{[a-f0-9]+}}, T>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 128) * sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          convert_kernel(b, item_ct1, aaa_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(), bbb_acc_ct1);
// CHECK-NEXT:        });
// CHECK-NEXT:    });
// CHECK-NEXT:  }
// CHECK-NEXT:}
template<typename T>
void convert(){
  T b;
  convert_kernel<<<128, 128>>>(b);
}


// CHECK:template <typename T>
// CHECK-NEXT:class Image {
// CHECK-NEXT:public:
// CHECK-NEXT:  T* dPtr;
// CHECK-NEXT:  dpct::queue_ptr s;
// CHECK-NEXT:};
// CHECK-NEXT:template <typename T>
// CHECK-NEXT:void my_kernel(T *A) {
// CHECK-NEXT:}
template <typename T>
class Image {
public:
  T* dPtr;
  cudaStream_t s;
};
template <typename T>
__global__ void my_kernel(T *A) {
}

// CHECK:template <typename T>
// CHECK-NEXT:static void multiply(int block_size, Image<T> &ptr, T value) {
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
// CHECK-NEXT:  */
// CHECK-NEXT:  ptr.s->submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      auto ptr_dPtr_ct0 = ptr.dPtr;
// CHECK-EMPTY:
// CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class my_kernel_{{[a-f0-9]+}}, T>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 8) * sycl::range<3>(1, 1, block_size), sycl::range<3>(1, 1, block_size)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          my_kernel<T>(ptr_dPtr_ct0);
// CHECK-NEXT:        });
// CHECK-NEXT:    });
// CHECK-NEXT:}
template <typename T>
static void multiply(int block_size, Image<T> &ptr, T value) {
  my_kernel<T><<<8, block_size, 0, ptr.s>>>(ptr.dPtr);
}

// CHECK:template <typename T, int size>
// CHECK-NEXT:void foo1(Image<T> &ptr, T value) {
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
// CHECK-NEXT:  */
// CHECK-NEXT:  ptr.s->submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      auto ptr_dPtr_ct0 = ptr.dPtr;
// CHECK-EMPTY:
// CHECK-NEXT:      /*
// CHECK-NEXT:      DPCT1050:{{[0-9]+}}: The template argument of the dpct_kernel_name could not be deduced. You need to update this code.
// CHECK-NEXT:      */
// CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class my_kernel_{{[a-f0-9]+}}, dpct_placeholder/*Fix the type mannually*/>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 8) * sycl::range<3>(1, 1, size), sycl::range<3>(1, 1, size)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          my_kernel(ptr_dPtr_ct0);
// CHECK-NEXT:        });
// CHECK-NEXT:    });
// CHECK-NEXT:}
template <typename T, int size>
void foo1(Image<T> &ptr, T value) {
  my_kernel<<<8, size, 0, ptr.s>>>(ptr.dPtr);
}

// CHECK:template <typename T, int size>
// CHECK-NEXT:void foo2(Image<T> &ptr, T value) {
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
// CHECK-NEXT:  */
// CHECK-NEXT:  ptr.s->submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      auto ptr_dPtr_ct0 = ptr.dPtr;
// CHECK-EMPTY:
// CHECK-NEXT:      /*
// CHECK-NEXT:      DPCT1050:{{[0-9]+}}: The template argument of the dpct_kernel_name could not be deduced. You need to update this code.
// CHECK-NEXT:      */
// CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class my_kernel_{{[a-f0-9]+}}, dpct_placeholder/*Fix the type mannually*/>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 8) * sycl::range<3>(2, size, 1), sycl::range<3>(2, size, 1)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          my_kernel(ptr_dPtr_ct0);
// CHECK-NEXT:        });
// CHECK-NEXT:    });
// CHECK-NEXT:}
template <typename T, int size>
void foo2(Image<T> &ptr, T value) {
  my_kernel<<<8, dim3(1, size, 2), 0, ptr.s>>>(ptr.dPtr);
}

template <typename T>
__global__ void my_kernel2(T A, int r) {}

template <typename V>
struct crs {
  typedef V val_t;
  crs();
  int rows;
};

template <typename T> struct spmv_driver{
  typedef T val_t;
  val_t alpha;
  crs<val_t> *crsmat;
};

namespace cuda {
template <class V> struct spmv_driver : public ::spmv_driver<V> {
  typedef ::spmv_driver<V> base_t;
  typedef typename base_t::val_t val_t;

  void run_naive() {
    val_t *dresult;
// CHECK:dresult = (val_t *)dpct::dpct_malloc(sizeof(val_t));
    cudaMalloc((void **)&dresult, sizeof(val_t));
// CHECK:q_ct1.submit(
// CHECK-NEXT:  [&](sycl::handler &cgh) {
// CHECK-NEXT:    auto base_t_alpha_ct0 = base_t::alpha;
// CHECK-NEXT:    auto base_t_crsmat_rows_ct1 = base_t::crsmat->rows;
// CHECK-EMPTY:
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1050:{{[0-9]+}}: The template argument of the dpct_kernel_name could not be deduced. You need to update this code.
// CHECK-NEXT:    */
// CHECK-NEXT:    cgh.parallel_for<dpct_kernel_name<class my_kernel2_{{[a-f0-9]+}}, dpct_placeholder/*Fix the type mannually*/>>(
// CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:        my_kernel2(base_t_alpha_ct0, base_t_crsmat_rows_ct1);
// CHECK-NEXT:      });
// CHECK-NEXT:  });
    my_kernel2<<<1,1>>>(base_t::alpha, base_t::crsmat->rows);
// CHECK:q_ct1.submit(
// CHECK-NEXT:  [&](sycl::handler &cgh) {
// CHECK-NEXT:    auto base_t_alpha_ct0 = base_t::alpha;
// CHECK-NEXT:    auto base_t_crsmat_rows_ct1 = base_t::crsmat->rows;
// CHECK-EMPTY:
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1050:{{[0-9]+}}: The template argument of the dpct_kernel_name could not be deduced. You need to update this code.
// CHECK-NEXT:    */
// CHECK-NEXT:    cgh.parallel_for<dpct_kernel_name<class my_kernel2_{{[a-f0-9]+}}, dpct_placeholder/*Fix the type mannually*/>>(
// CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, base_t::crsmat->rows) * sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)),
// CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:        my_kernel2(base_t_alpha_ct0, base_t_crsmat_rows_ct1);
// CHECK-NEXT:      });
// CHECK-NEXT:  });
    my_kernel2<<<base_t::crsmat->rows,2>>>(base_t::alpha, base_t::crsmat->rows);
// CHECK:q_ct1.submit(
// CHECK-NEXT:  [&](sycl::handler &cgh) {
// CHECK-NEXT:    auto base_t_alpha_ct0 = base_t::alpha;
// CHECK-NEXT:    auto base_t_crsmat_rows_ct1 = base_t::crsmat->rows;
// CHECK-EMPTY:
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1050:{{[0-9]+}}: The template argument of the dpct_kernel_name could not be deduced. You need to update this code.
// CHECK-NEXT:    */
// CHECK-NEXT:    cgh.parallel_for<dpct_kernel_name<class my_kernel2_{{[a-f0-9]+}}, dpct_placeholder/*Fix the type mannually*/>>(
// CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, base_t::crsmat->rows), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:        my_kernel2(base_t_alpha_ct0, base_t_crsmat_rows_ct1);
// CHECK-NEXT:      });
// CHECK-NEXT:  });
    my_kernel2<<<base_t::crsmat->rows,1>>>(base_t::alpha, base_t::crsmat->rows);
// CHECK:q_ct1.submit(
// CHECK-NEXT:  [&](sycl::handler &cgh) {
// CHECK-NEXT:    auto base_t_alpha_ct0 = base_t::alpha;
// CHECK-NEXT:    auto base_t_crsmat_rows_ct1 = base_t::crsmat->rows;
// CHECK-EMPTY:
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1050:{{[0-9]+}}: The template argument of the dpct_kernel_name could not be deduced. You need to update this code.
// CHECK-NEXT:    */
// CHECK-NEXT:    cgh.parallel_for<dpct_kernel_name<class my_kernel2_{{[a-f0-9]+}}, dpct_placeholder/*Fix the type mannually*/>>(
// CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, base_t::crsmat->rows), sycl::range<3>(1, 1, base_t::crsmat->rows)),
// CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:        my_kernel2(base_t_alpha_ct0, base_t_crsmat_rows_ct1);
// CHECK-NEXT:      });
// CHECK-NEXT:  });
    my_kernel2<<<2,base_t::crsmat->rows>>>(base_t::alpha, base_t::crsmat->rows);
  }
};
}

class IndexType {};

// CHECK: void thread_id(const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:  auto tidx = item_ct1.get_local_id(2);
// CHECK-NEXT:  auto tidx_int = static_cast<int>(item_ct1.get_local_id(2));
// CHECK-NEXT: }
__device__ void thread_id() {
  auto tidx = threadIdx.x;
  auto tidx_int = static_cast<int>(threadIdx.x);
}

// CHECK: template <typename IndexType = int> void thread_id(const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:   auto tidx = item_ct1.get_local_id(2);
// CHECK-NEXT:   auto tidx_template = static_cast<IndexType>(item_ct1.get_local_id(2));
// CHECK-NEXT:   auto tidx_int = static_cast<int>(item_ct1.get_local_id(2));
// CHECK-NEXT: }
template <typename IndexType = int> __device__ void thread_id() {
  auto tidx = threadIdx.x;
  auto tidx_template = static_cast<IndexType>(threadIdx.x);
  auto tidx_int = static_cast<int>(threadIdx.x);
}

// CHECK: template <typename IndexType = int> void kernel(const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:   auto tidx = item_ct1.get_local_id(2);
// CHECK-NEXT:   auto tidx_template = static_cast<IndexType>(item_ct1.get_local_id(2));
// CHECK-NEXT:   auto tidx_int = static_cast<int>(item_ct1.get_local_id(2));
// CHECK-NEXT: }
template <typename IndexType = int> __global__ void kernel() {
  auto tidx = threadIdx.x;
  auto tidx_template = static_cast<IndexType>(threadIdx.x);
  auto tidx_int = static_cast<int>(threadIdx.x);
}

#define TEST_LAMBDA(x) [&](){x();}()

template<class F>
__host__ __device__ void apply(F f, int a) {
        return f(a);
}

__device__ int lambda_call() {
        int a;
        TEST_LAMBDA([&](){apply([a] (int b) {},2);});
        return 0;
}

class Stream {};
namespace c10 {
namespace cuda {
class CUDAStream {
public:
  operator cudaStream_t() const {
    cudaStream_t tt;
    return tt;
  }
};
CUDAStream getCurrentCUDAStream(int device_index = -1);
} // namespace cuda
} // namespace c10

__global__ void kernel() {}
void foo() {
// CHECK:  auto stream = c10::cuda::getCurrentCUDAStream();
// CHECK-NEXT:  ((sycl::queue*)(stream))->parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
// CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:      kernel();
// CHECK-NEXT:    });
  auto stream = c10::cuda::getCurrentCUDAStream();
  kernel<<<1, 1, 0, stream>>>();
}

template<class T, int N>
class foo_class1{
public:
// CHECK: void foo(const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT: int a = item_ct1.get_local_id(2);
  __device__ void foo() {
    int a = threadIdx.x;
  }
};
template<int N>
class foo_class1<int, N>{
public:
// CHECK: void foo(const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT: int a = item_ct1.get_local_id(2);
  __device__ void foo() {
    int a = threadIdx.x;
  }
};

__global__ void test_fooclass1() {
  foo_class1<int, 10> a;
  a.foo();
  foo_class1<float, 10> b;
  b.foo();
}

__global__ void test_kernel();

template<class T>
void test_host() {
  std::vector<T> vec;
  // CHECK:  dpct::get_out_of_order_queue().parallel_for<dpct_kernel_name<class test_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:  sycl::nd_range<3>(sycl::range<3>(1, 1, vec.size()), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:  [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:    test_kernel();
  // CHECK-NEXT:  });
  test_kernel<<<vec.size(), 1>>>();
}
