// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none -out-root %T/syncthreads_template %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/syncthreads_template/syncthreads_template.dp.cpp

// Unsupport template kernel function
template<class T>
struct S1 {
  T data;
};
template<class Q>
__global__ void test1(S1<Q> s1) {
  s1.data;
  // CHECK:item_ct1.barrier();
  __syncthreads();
}

template<class Q>
__global__ void test2() {
  // CHECK:item_ct1.barrier();
  __syncthreads();
}
