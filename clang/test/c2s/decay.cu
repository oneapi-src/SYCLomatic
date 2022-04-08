// RUN: c2s --format-range=none --usm-level=none -out-root %T/decay %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/decay/decay.dp.cpp %s
#include <cuda_runtime.h>

template<typename T>
__global__ void foo_kernel(T** R)
{
}


//CHECK: void foo(float* R[])
//CHECK-NEXT: {
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1069:{{[0-9]+}}: The argument 'R' of the kernel function contains virtual pointer(s), which cannot be dereferenced. Try to migrate the code with "usm-level=restricted".
//CHECK-NEXT:   */
//CHECK-NEXT:   c2s::get_default_queue().submit(
//CHECK-NEXT:     [&](sycl::handler &cgh) {
//CHECK-NEXT:       c2s::access_wrapper<float **> R_acc_ct0(R, cgh);
//CHECK-EMPTY:
//CHECK-NEXT:       cgh.parallel_for<c2s_kernel_name<class foo_kernel_{{[a-f0-9]+}}, float>>(
//CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:           foo_kernel<float>(R_acc_ct0.get_raw_pointer());
//CHECK-NEXT:         });
//CHECK-NEXT:     });
void foo(float* R[])
{
  foo_kernel<float><<<1, 1>>>(R);
}

