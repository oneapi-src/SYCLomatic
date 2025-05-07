// clang-format off
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none --use-experimental-features=matrix -out-root %T/wmma2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/wmma2/wmma2.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/wmma2/wmma2.dp.cpp -o %T/wmma2/wmma2.dp.o %}

#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <mma.h>
// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
namespace wmmaa = nvcuda::wmma;

template<typename T>
__global__ void simple_wmma_gemm(T *d) {
  wmmaa::fragment<wmmaa::accumulator, 16, 16, 16, T> c_frag;
// CHECK: sycl::ext::oneapi::experimental::matrix::joint_matrix_store(sycl::ext::oneapi::this_work_item::get_sub_group(), c_frag.get(), sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::no, T>(d), 1, sycl::ext::oneapi::experimental::matrix::layout::row_major);
  wmmaa::store_matrix_sync(d, c_frag, 1, wmmaa::mem_row_major);
}

int main() {

  simple_wmma_gemm<half><<<1, 1>>>(nullptr);

  simple_wmma_gemm<float><<<1, 1>>>(nullptr);

  return 0;
}


