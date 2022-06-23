// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-placeholders %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/thrust-placeholders/thrust-placeholders.dp.cpp

// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
#include <thrust/memory.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>

// CHECK-EMPTY:
using namespace thrust::placeholders;

// CHECK-EMPTY:
namespace ph = thrust::placeholders;

__device__ void dev_fct(float *src, float *dst) {
  auto tmp1 = logf(2.0);
// CHECK:   auto tmp2 = [=](auto _1,auto _2){return _1 + logf(2.0) + _2;};
  auto tmp2 = _1 + logf(2.0) + ph::_2;
}

__global__ void kernel(float *src, float *dst) {
  auto tmp1 = logf(2.0);
// CHECK:   auto tmp2 = [=](auto _1,auto _2){return _1 + logf(2.0) + _2;};
  auto tmp2 = _1 + logf(2.0) + thrust::placeholders::_2;
}

int main() {
  float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float y[] = {2.0f, 1.0f, 1.0f, 1.0f};
  float a = 2.0f;
// CHECK:  std::transform(oneapi::dpl::execution::seq, x, x + 4, y, y, [=](auto _1,auto _2){return a * _1 + _2;});
  thrust::transform(x, x + 4, y, y, a * _1 + _2);

// CHECK:  std::transform(oneapi::dpl::execution::seq, x, x + 4, y, y, [=](auto _1,auto _2){return a*_1+_2;});
  thrust::transform(x, x + 4, y, y, a*_1+_2);

// CHECK:  std::transform(oneapi::dpl::execution::seq, x, x + 4, y, y, [=](auto _1,auto _2){return a * _1 + _2;});
  thrust::transform(x, x + 4, y, y, a * thrust::placeholders::_1 + thrust::placeholders::_2);

// CHECK:  std::transform(oneapi::dpl::execution::seq, x, x + 4, y, y, [=](auto _1,auto _2){return a * _1 + _2;});
  thrust::transform(x, x + 4, y, y, a * ph::_1 + ph::_2);

// CHECK:  auto tmp1 = [=](auto _1,auto _2,auto _3,auto _4,auto _5,auto _6,auto _7,auto _8,auto _9){return _1 + _2 + _3 + _4 + _5 + _6 + _7 + _8 + _9;};
  auto tmp1 = _1 + _2 + _3 + _4 + _5 + _6 + _7 + _8 + _9;

// CHECK:  auto tmp2 = [=](auto _1,auto _2,auto _3,auto _4,auto _5,auto _6,auto _7,auto _8,auto _9){return _1 + _2 + _3 + _4 + _5 + _6 + _7 + _8 + _9;};
  auto tmp2 = _1 + thrust::placeholders::_2 + ph::_3 + _4 + ph::_5 + _6 + _7 + _8 + thrust::placeholders::_9;

// CHECK:  auto tmp3 = [=](auto _1){return _1 + _1 + _1 + _1 + _1 + _1 + _1 + _1 + _1;};
  auto tmp3 = _1 + _1 + _1 + _1 + _1 + _1 + _1 + _1 + _1;

// CHECK:  auto tmp4 = [=](auto _1){return _1;};
  auto tmp4 = _1;

// CHECK:  auto tmp5 = [=](auto _1,auto _2){return _1 + sizeof(sycl::float2) + _2;};
  auto tmp5 = _1 + sizeof(float2) + _2;

}

