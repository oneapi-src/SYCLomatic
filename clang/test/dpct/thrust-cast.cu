// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T/thrust-cast %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/thrust-cast/thrust-cast.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/thrust-cast/thrust-cast.dp.cpp -o %T/thrust-cast/thrust-cast.dp.o %}
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <complex>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>

// CHECK: void kernel(std::complex<double> *det) {}
__global__ void kernel(thrust::complex<double> *det) {}

int main() {
// CHECK:  dpct::device_pointer<std::complex<double>> d_ptr = dpct::malloc_device<std::complex<double>>(1);
  thrust::device_ptr<thrust::complex<double>> d_ptr = thrust::device_malloc<thrust::complex<double>>(1);
  // CHECK:  q_ct1.submit(
  // CHECK-NEXT:    [&](sycl::handler &cgh) {
  // CHECK-NEXT:      auto thrust_raw_pointer_cast_d_ptr_ct0 = dpct::get_raw_pointer(d_ptr);
  // CHECK-EMPTY:
  // CHECK-NEXT:      cgh.parallel_for(
  // CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
  // CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:          kernel(thrust_raw_pointer_cast_d_ptr_ct0);
  // CHECK-NEXT:        });
  // CHECK-NEXT:    });
  kernel<<<1,256>>>(thrust::raw_pointer_cast(d_ptr));
  std::complex<double> res;
// CHECK:  q_ct1.memcpy(std::addressof(res), dpct::get_raw_pointer(d_ptr), sizeof(std::complex<double>)).wait();
  cudaMemcpy(std::addressof(res), thrust::raw_pointer_cast(d_ptr), sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
// CHECK:  dpct::free_device(d_ptr);
  thrust::device_free(d_ptr);
}

__global__ void kernel2(float *p) {}

void foo(thrust::device_vector<float> &vec, const int i, const int j) {
  //CHECK:dpct::get_in_order_queue().submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    auto thrust_raw_pointer_cast_vec_data_i_j_ct0 = dpct::get_raw_pointer(vec.data()) + i * j;
  //CHECK-EMPTY:
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        kernel2(thrust_raw_pointer_cast_vec_data_i_j_ct0);
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  kernel2<<<1, 1>>>(thrust::raw_pointer_cast(vec.data()) + i * j);
}

//CHECK: /*
//CHECK-NEXT: DPCT1044:{{[0-9]+}}: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may need to remove references to typedefs from thrust::unary_function in the class definition.
//CHECK-NEXT: */
//CHECK-NEXT: struct transpose_102_index {
//CHECK-NEXT:   const size_t m, n, p;
//CHECK-NEXT:   transpose_102_index(size_t _m, size_t _n, size_t _p) : m(_m), n(_n), p(_p) {}
//CHECK-NEXT:   size_t operator()(size_t linear_index) const {
//CHECK-NEXT:     size_t i = linear_index / (n * p);
//CHECK-NEXT:     size_t rmdr = linear_index % (n * p);
//CHECK-NEXT:     size_t j = rmdr / p;
//CHECK-NEXT:     size_t k = rmdr % p;
//CHECK-NEXT:     return m * p * j + p * i + k;
//CHECK-NEXT:   }
//CHECK-NEXT: };
struct transpose_102_index : public thrust::unary_function<size_t, size_t> {
  const size_t m, n, p;
  __host__ __device__ transpose_102_index(size_t _m, size_t _n, size_t _p) : m(_m), n(_n), p(_p) {}
  __host__ __device__ size_t operator()(size_t linear_index) {
    size_t i = linear_index / (n * p);
    size_t rmdr = linear_index % (n * p);
    size_t j = rmdr / p;
    size_t k = rmdr % p;
    return m * p * j + p * i + k;
  }
};

//CHECK: /*
//CHECK-NEXT: DPCT1044:{{[0-9]+}}: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may need to remove references to typedefs from thrust::unary_function in the class definition.
//CHECK-NEXT: */
//CHECK-NEXT:struct transpose_201_index {
//CHECK-NEXT:  const size_t m, n, p;
//CHECK-NEXT:  transpose_201_index(size_t _m, size_t _n, size_t _p) : m(_m), n(_n), p(_p) {}
//CHECK-NEXT:  size_t operator()(size_t linear_index) const {
//CHECK-NEXT:    size_t i = linear_index / (n * p);
//CHECK-NEXT:    size_t rmdr = linear_index % (n * p);
//CHECK-NEXT:    size_t j = rmdr / p;
//CHECK-NEXT:    size_t k = rmdr % p;
//CHECK-NEXT:    return m * n * k + n * i + j;
//CHECK-NEXT:  }
//CHECK-NEXT:};
struct transpose_201_index : public thrust::unary_function<size_t, size_t> {
  const size_t m, n, p;
  __host__ __device__ transpose_201_index(size_t _m, size_t _n, size_t _p) : m(_m), n(_n), p(_p) {}
  __host__ __device__ size_t operator()(size_t linear_index) {
    size_t i = linear_index / (n * p);
    size_t rmdr = linear_index % (n * p);
    size_t j = rmdr / p;
    size_t k = rmdr % p;
    return m * n * k + n * i + j;
  }
};

//CHECK: /*
//CHECK-NEXT: DPCT1044:{{[0-9]+}}: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may need to remove references to typedefs from thrust::unary_function in the class definition.
//CHECK-NEXT: */
//CHECK-NEXT:struct transpose_10_index {
//CHECK-NEXT:  const size_t m, n;
//CHECK-NEXT:  transpose_10_index(size_t _m, size_t _n) : m(_m), n(_n) {}
//CHECK-NEXT:  size_t operator()(size_t linear_index) const {
//CHECK-NEXT:    size_t i = linear_index / n;
//CHECK-NEXT:    size_t rmdr = linear_index % n;
//CHECK-NEXT:    size_t j = rmdr;
//CHECK-NEXT:    return j * m + i;
//CHECK-NEXT:  }
//CHECK-NEXT:};
struct transpose_10_index : public thrust::unary_function<size_t, size_t> {
  const size_t m, n;
  __host__ __device__ transpose_10_index(size_t _m, size_t _n) : m(_m), n(_n) {}
  __host__ __device__ size_t operator()(size_t linear_index) {
    size_t i = linear_index / n;
    size_t rmdr = linear_index % n;
    size_t j = rmdr;
    return j * m + i;
  }
};

void transpose_102(size_t m, size_t n, size_t p, thrust::device_vector<float> &src,
                   thrust::device_vector<float> &dst) {
  thrust::counting_iterator<size_t> indices(0);
  thrust::scatter(src.begin(), src.end(),
                  thrust::make_transform_iterator(indices, transpose_102_index(m, n, p)),
                  dst.begin());
}

void transpose_201(size_t m, size_t n, size_t p, thrust::device_vector<float> &src,
                   thrust::device_vector<float> &dst) {
  thrust::counting_iterator<size_t> indices(0);
  thrust::scatter(src.begin(), src.end(),
                  thrust::make_transform_iterator(indices, transpose_201_index(m, n, p)),
                  dst.begin());
}

template <typename T>
void transpose_2d(size_t m, size_t n, thrust::device_vector<T> &src, thrust::device_vector<T> &dst) {
  static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value || std::is_same<T, __half>::value,
                "T must be float type or integer type");
  thrust::counting_iterator<size_t> indices(0);
  thrust::scatter(src.begin(), src.end(),
                  thrust::make_transform_iterator(indices, transpose_10_index(m, n)),
                  dst.begin());
}

void bar(void) {
  thrust::device_vector<thrust::complex<double>> d_complex;
  thrust::device_vector<double> d_results(4);

  // CHECK:    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), d_complex.begin(), d_complex.end(), d_results.begin(), []  (std::complex<double> z) {
  // CHECK-NEXT:                          return std::arg(z);
  // CHECK-NEXT:                      });
  thrust::transform(d_complex.begin(), d_complex.end(), d_results.begin(),
                    [] __device__(thrust::complex<double> z) {
                      return thrust::arg(z);
                    });
}
