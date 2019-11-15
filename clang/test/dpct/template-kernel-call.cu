// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/template-kernel-call.dp.cpp --match-full-lines %s

void printf(const char *format, unsigned char data);

template <class TName, unsigned N, class TData>
// CHECK: void testKernelPtr(const TData *L, const TData *M, cl::sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__ void testKernelPtr(const TData *L, const TData *M) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(2) * [[ITEMNAME]].get_local_range().get(2) + [[ITEMNAME]].get_local_id(2);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

template<class TData>
// CHECK: void testKernel(TData L, TData M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
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
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto karg1_acc_ct0 = karg1_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto karg2_acc_ct1 = karg2_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, class TestName, dpct_kernel_scalar<ktarg>, T>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           T const *karg1_ct0 = (T const *)(&karg1_acc_ct0[0] + karg1_offset_ct0);
  // CHECK-NEXT:           T const *karg2_ct1 = (T const *)(&karg2_acc_ct1[0] + karg2_offset_ct1);
  // CHECK-NEXT:           testKernelPtr<class TestName, ktarg, T>(karg1_ct0, karg2_ct1, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<class TestName, ktarg, T><<<griddim, threaddim>>>((const T *)karg1, karg2);

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg1_buf_ct0 = dpct::get_buffer_and_offset(karg1);
  // CHECK-NEXT:   size_t karg1_offset_ct0 = karg1_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg3_buf_ct1 = dpct::get_buffer_and_offset(karg3);
  // CHECK-NEXT:   size_t karg3_offset_ct1 = karg3_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto karg1_acc_ct0 = karg1_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto karg3_acc_ct1 = karg3_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, class TestTemplate<T>, dpct_kernel_scalar<ktarg>, T>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           void const *karg1_ct0 = (void const *)(&karg1_acc_ct0[0] + karg1_offset_ct0);
  // CHECK-NEXT:           T *karg3_ct1 = (T *)(&karg3_acc_ct1[0] + karg3_offset_ct1);
  // CHECK-NEXT:           testKernelPtr<class TestTemplate<T>, ktarg, T>(karg1_ct0, karg3_ct1, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<class TestTemplate<T>, ktarg, T><<<griddim, threaddim>>>(karg1, karg3);

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg4_buf_ct0 = dpct::get_buffer_and_offset(karg4);
  // CHECK-NEXT:   size_t karg4_offset_ct0 = karg4_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg5_buf_ct1 = dpct::get_buffer_and_offset(karg5);
  // CHECK-NEXT:   size_t karg5_offset_ct1 = karg5_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto karg4_acc_ct0 = karg4_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto karg5_acc_ct1 = karg5_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, T, dpct_kernel_scalar<ktarg>, TestTemplate<T>>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           TestTemplate<T> const *karg4_ct0 = (TestTemplate<T> const *)(&karg4_acc_ct0[0] + karg4_offset_ct0);
  // CHECK-NEXT:           TT *karg5_ct1 = (TT *)(&karg5_acc_ct1[0] + karg5_offset_ct1);
  // CHECK-NEXT:           testKernelPtr<T, ktarg, TestTemplate<T>>(karg4_ct0, karg5_ct1, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<T, ktarg, TestTemplate<T> ><<<griddim, threaddim>>>(karg4, karg5);

  T karg1T, karg2T;
  // CHECK: {
  // CHECK-NEXT:   auto ktarg_ct2 = ktarg;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel<T>(karg1T, karg2T, ktarg_ct2, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernel<T><<<griddim, threaddim>>>(karg1T, karg2T, ktarg);

  TestTemplate<T> karg3TT;
  TT karg4TT;

  // CHECK: {
  // CHECK-NEXT:   auto ktarg_ct2 = ktarg;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}, TestTemplate<T>>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel<TestTemplate<T>>(karg3TT, karg4TT, ktarg_ct2, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernel<TestTemplate<T> ><<<griddim, threaddim>>>(karg3TT, karg4TT, ktarg);

  // CHECK: {
  // CHECK-NEXT:   auto ktarg_ct2 = ktarg;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}, TT>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel<TT>(karg3TT, karg4TT, ktarg_ct2, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernel<TT><<<griddim, threaddim>>>(karg3TT, karg4TT, ktarg);
}

int main() {
  void *karg1 = 0;
  LA *karg2 = 0;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg1_buf_ct0 = dpct::get_buffer_and_offset((const LA *)karg1);
  // CHECK-NEXT:   size_t karg1_offset_ct0 = karg1_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg2_buf_ct1 = dpct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:   size_t karg2_offset_ct1 = karg2_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto karg1_acc_ct0 = karg1_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto karg2_acc_ct1 = karg2_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, class TestName, dpct_kernel_scalar<ktarg>, LA>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           LA const *karg1_ct0 = (LA const *)(&karg1_acc_ct0[0] + karg1_offset_ct0);
  // CHECK-NEXT:           LA const *karg2_ct1 = (LA const *)(&karg2_acc_ct1[0] + karg2_offset_ct1);
  // CHECK-NEXT:           testKernelPtr<class TestName, ktarg, LA>(karg1_ct0, karg2_ct1, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<class TestName, ktarg, LA><<<griddim, threaddim>>>((const LA *)karg1, karg2);

  LA karg1LA, karg2LA;
  int intvar = 20;
  // CHECK: {
  // CHECK-NEXT:   auto ktarg_ct2 = ktarg;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}, LA>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 10) * cl::sycl::range<3>(1, 1, intvar), cl::sycl::range<3>(1, 1, intvar)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel<LA>(karg1LA, karg2LA, ktarg_ct2, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernel<LA><<<10, intvar>>>(karg1LA, karg2LA, ktarg);
}
