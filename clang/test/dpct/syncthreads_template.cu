// UNSUPPORTED: system-windows
// RUN: dpct  --format-range=none -out-root %T/syncthreads_template %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/syncthreads_template/syncthreads_template.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/syncthreads_template/syncthreads_template.dp.cpp -o %T/syncthreads_template/syncthreads_template.dp.o %}

// Unsupport template kernel function
template<class T>
struct S1 {
  T data;
};
template<class Q>
__global__ void test1(S1<Q> s1) {
  s1.data;
  // CHECK:sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier();
  __syncthreads();
}

template<class Q>
__global__ void test2() {
  // CHECK:sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier(sycl::access::fence_space::local_space);
  __syncthreads();
}
