// RUN: dpct --format-range=none -out-root %T/kernel_without_name-usm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel_without_name-usm/kernel_without_name-usm.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST %T/kernel_without_name-usm/kernel_without_name-usm.dp.cpp -o %T/kernel_without_name-usm/kernel_without_name-usm.dp.o %}

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
  // CHECK: DISPATCH(mat.getType(), ([&] { dpct::get_in_order_queue().submit(
  // CHECK-NEXT: [&](sycl::handler &cgh) {
  // CHECK-NEXT:   decltype(mat.data<scalar_t>()) mat_data_scalar_t_ct0 = mat.data<scalar_t>();
  // CHECK-EMPTY:
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:     sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:     [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:       foo_kernel1(mat_data_scalar_t_ct0);
  // CHECK-NEXT:     });
  // CHECK-NEXT: }); }));
  DISPATCH(mat.getType(), ([&] { foo_kernel1<<<1, 1>>>(mat.data<scalar_t>()); }));
}

template <class T> __global__ void foo_kernel1(const T *a) {}
#undef DISPATCH
