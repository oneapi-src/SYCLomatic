// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/template-kernel-call.dp.cpp --match-full-lines %s

void printf(const char *format, unsigned char data);

template<class T> void runTest();

template <class TName, unsigned N, class TData>
// CHECK: void testKernelPtr(const TData *L, const TData *M, sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__ void testKernelPtr(const TData *L, const TData *M) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(2) * [[ITEMNAME]].get_local_range().get(2) + [[ITEMNAME]].get_local_id(2);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

template<class TData>
// CHECK: void testKernel(TData L, TData M, int N, sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__ void testKernel(TData L, TData M, int N) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(2) * [[ITEMNAME]].get_local_range().get(2) + [[ITEMNAME]].get_local_id(2);
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

template<class T>
void runTest() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  typedef TestTemplate<T> TT;
  const void *karg1 = 0;
  const T *karg2 = 0;
  T *karg3 = 0;
  const TestTemplate<T> *karg4 = 0;
  TT *karg5 = 0;

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg1_buf_ct0 = dpct::get_buffer_and_offset((const T *)karg1);
  // CHECK-NEXT:   size_t karg1_offset_ct0 = karg1_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg2_buf_ct1 = dpct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:   size_t karg2_offset_ct1 = karg2_buf_ct1.second;
  // CHECK-NEXT:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto karg1_acc_ct0 = karg1_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto karg2_acc_ct1 = karg2_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, class TestName, dpct_kernel_scalar<ktarg>, T>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           const T *karg1_ct0 = (const T *)(&karg1_acc_ct0[0] + karg1_offset_ct0);
  // CHECK-NEXT:           const T *karg2_ct1 = (const T *)(&karg2_acc_ct1[0] + karg2_offset_ct1);
  // CHECK-NEXT:           testKernelPtr<class TestName, ktarg, T>(karg1_ct0, karg2_ct1, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<class TestName, ktarg, T><<<griddim, threaddim>>>((const T *)karg1, karg2);

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg1_buf_ct0 = dpct::get_buffer_and_offset((const T *)karg1);
  // CHECK-NEXT:   size_t karg1_offset_ct0 = karg1_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg3_buf_ct1 = dpct::get_buffer_and_offset(karg3);
  // CHECK-NEXT:   size_t karg3_offset_ct1 = karg3_buf_ct1.second;
  // CHECK-NEXT:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto karg1_acc_ct0 = karg1_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto karg3_acc_ct1 = karg3_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, class TestTemplate<T>, dpct_kernel_scalar<ktarg>, T>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           const T *karg1_ct0 = (const T *)(&karg1_acc_ct0[0] + karg1_offset_ct0);
  // CHECK-NEXT:           T *karg3_ct1 = (T *)(&karg3_acc_ct1[0] + karg3_offset_ct1);
  // CHECK-NEXT:           testKernelPtr<class TestTemplate<T>, ktarg, T>(karg1_ct0, karg3_ct1, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<class TestTemplate<T>, ktarg, T><<<griddim, threaddim>>>((const T *)karg1, karg3);

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg4_buf_ct0 = dpct::get_buffer_and_offset(karg4);
  // CHECK-NEXT:   size_t karg4_offset_ct0 = karg4_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg5_buf_ct1 = dpct::get_buffer_and_offset(karg5);
  // CHECK-NEXT:   size_t karg5_offset_ct1 = karg5_buf_ct1.second;
  // CHECK-NEXT:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto karg4_acc_ct0 = karg4_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto karg5_acc_ct1 = karg5_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, T, dpct_kernel_scalar<ktarg>, TestTemplate<T>>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           const TestTemplate<T> *karg4_ct0 = (const TestTemplate<T> *)(&karg4_acc_ct0[0] + karg4_offset_ct0);
  // CHECK-NEXT:           TT *karg5_ct1 = (TT *)(&karg5_acc_ct1[0] + karg5_offset_ct1);
  // CHECK-NEXT:           testKernelPtr<T, ktarg, TestTemplate<T>>(karg4_ct0, karg5_ct1, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<T, ktarg, TestTemplate<T> ><<<griddim, threaddim>>>(karg4, karg5);

  T karg1T, karg2T;
  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-EMPTY:
  // CHECK-NEXT:       auto ktarg_ct2 = ktarg;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel<T>(karg1T, karg2T, ktarg_ct2, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  testKernel<T><<<griddim, threaddim>>>(karg1T, karg2T, ktarg);

  TestTemplate<T> karg3TT;
  TT karg4TT;

  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-EMPTY:
  // CHECK-NEXT:       auto ktarg_ct2 = ktarg;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}, TestTemplate<T>>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel<TestTemplate<T>>(karg3TT, karg4TT, ktarg_ct2, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  testKernel<TestTemplate<T> ><<<griddim, threaddim>>>(karg3TT, karg4TT, ktarg);

  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-EMPTY:
  // CHECK-NEXT:       auto ktarg_ct2 = ktarg;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}, TT>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
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
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  void *karg1 = 0;
  LA *karg2 = 0;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg1_buf_ct0 = dpct::get_buffer_and_offset((const LA *)karg1);
  // CHECK-NEXT:   size_t karg1_offset_ct0 = karg1_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg2_buf_ct1 = dpct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:   size_t karg2_offset_ct1 = karg2_buf_ct1.second;
  // CHECK-NEXT:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto karg1_acc_ct0 = karg1_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto karg2_acc_ct1 = karg2_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, class TestName, dpct_kernel_scalar<ktarg>, LA>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           const LA *karg1_ct0 = (const LA *)(&karg1_acc_ct0[0] + karg1_offset_ct0);
  // CHECK-NEXT:           const LA *karg2_ct1 = (const LA *)(&karg2_acc_ct1[0] + karg2_offset_ct1);
  // CHECK-NEXT:           testKernelPtr<class TestName, ktarg, LA>(karg1_ct0, karg2_ct1, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<class TestName, ktarg, LA><<<griddim, threaddim>>>((const LA *)karg1, karg2);

  LA karg1LA, karg2LA;
  int intvar = 20;
  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto ktarg_ct2 = ktarg;
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
// CHECK-NEXT:void convert_kernel(T b, sycl::nd_item<3> item_ct1, int *aaa,
// CHECK-NEXT:                    dpct::accessor<double, dpct::local, 2> bbb){
// CHECK:  T a = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
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
// CHECK-NEXT:  dpct::get_default_queue().submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      sycl::range<2> bbb_range_ct1(8, 0);
// CHECK-EMPTY:
// CHECK-NEXT:      sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> aaa_acc_ct1(sycl::range<1>(0), cgh);
// CHECK-NEXT:      sycl::accessor<double, 2, sycl::access::mode::read_write, sycl::access::target::local> bbb_acc_ct1(bbb_range_ct1, cgh);
// CHECK-EMPTY:
// CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class convert_kernel_{{[a-f0-9]+}}>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 128) * sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          convert_kernel(b, item_ct1, aaa_acc_ct1.get_pointer(), dpct::accessor<double, dpct::local, 2>(bbb_acc_ct1, bbb_range_ct1));
// CHECK-NEXT:        });
// CHECK-NEXT:    });
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
// CHECK-NEXT:  sycl::queue *s;
// CHECK-NEXT:};
// CHECK-NEXT:template <typename T>
// CHECK-NEXT:void my_kernel(T *A) {}
template <typename T>
class Image {
public:
  T* dPtr;
  cudaStream_t s;
};
template <typename T>
__global__ void my_kernel(T *A) {}

// CHECK:template <typename T>
// CHECK-NEXT:static void multiply(int block_size, Image<T> &ptr, T value) {
// CHECK-NEXT:  ptr.s->submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class my_kernel_{{[a-f0-9]+}}, T>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(0, 0, 8) * sycl::range<3>(0, 0, block_size), sycl::range<3>(0, 0, block_size)), 
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          my_kernel<T>(ptr.dPtr);
// CHECK-NEXT:        });
// CHECK-NEXT:    });
// CHECK-NEXT:}
template <typename T>
static void multiply(int block_size, Image<T> &ptr, T value) {
  my_kernel<T><<<8, block_size, 0, ptr.s>>>(ptr.dPtr);
}