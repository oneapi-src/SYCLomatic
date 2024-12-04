// RUN: dpct --format-range=none -out-root %T/kernel_without_name-usm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel_without_name-usm/kernel_without_name-usm.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/kernel_without_name-usm/kernel_without_name-usm.dp.cpp -o %T/kernel_without_name-usm/kernel_without_name-usm.dp.o %}

template <class T> __global__ void foo_kernel1(const T *a);

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

void run_foo1(Mat mat) {
  // CHECK: DISPATCH(mat.getType(), ([&] {   dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});
  // CHECK-EMPTY:
  // CHECK-NEXT: dpct::get_in_order_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::local_accessor<scalar_t, 1> shmem_acc_ct1(sycl::range<1>(100), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     auto mat_data_scalar_t_ct0 = mat.data<scalar_t>();
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         foo_kernel1(mat_data_scalar_t_ct0, (scalar_t *)shmem_acc_ct1.template get_multi_ptr<sycl::access::decorated::no>().get());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }); }));
  DISPATCH(mat.getType(), ([&] { foo_kernel1<<<1, 1>>>(mat.data<scalar_t>()); }));
}

template <class T> __global__ void foo_kernel1(const T *a) {
  __shared__ T shmem[100];
}
#undef DISPATCH

// CHECK: void foo_kernel2(uint8_t *dpct_local) {
// CHECK-NEXT:   auto smem = (int *)dpct_local;
// CHECK-NEXT:   char *out_cached = reinterpret_cast<char *>(smem);
// CHECK-NEXT: }
__global__ void foo_kernel2() {
  extern __shared__ int smem[];
  char *out_cached = reinterpret_cast<char *>(smem);
}

// CHECK: void run_foo2() {
// CHECK-NEXT:   [&] {
// CHECK-NEXT:     dpct::get_in_order_queue().submit(
// CHECK-NEXT:       [&](sycl::handler &cgh) {
// CHECK-NEXT:         sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(256), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:         cgh.parallel_for(
// CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
// CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:             foo_kernel2(dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
// CHECK-NEXT:           });
// CHECK-NEXT:       });
// CHECK-NEXT:   }();
// CHECK-NEXT: }
void run_foo2() {
  [&] {
    foo_kernel2<<<1, 1, 256>>>();
  }();
}

// CHECK: void foo_kernel3(const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:   auto lambda1 = [&](const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     item_ct1.get_local_id(2);
// CHECK-NEXT:   };
// CHECK-NEXT:   auto lambda2 = [&](const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     lambda1(item_ct1);
// CHECK-NEXT:   };
// CHECK-NEXT:   lambda2(item_ct1);
// CHECK-NEXT: }
__global__ void foo_kernel3() {
  auto lambda1 = [&]() {
    threadIdx.x;
  };
  auto lambda2 = [&]() {
    lambda1();
  };
  lambda2();
}
