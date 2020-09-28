// RUN: cat %s > %T/checkKernelFormatMigrated.cu
// RUN: cd %T
// RUN: dpct --no-cl-namespace-inline -out-root %T checkKernelFormatMigrated.cu --cuda-include-path="%cuda-path/include" -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck -strict-whitespace checkKernelFormatMigrated.cu --match-full-lines --input-file %T/checkKernelFormatMigrated.dp.cpp

#include <cuda_runtime.h>

__global__ void k() {}

//     CHECK:void foo() {
//CHECK-NEXT:                dpct::get_default_queue().submit([&](cl::sycl::handler &cgh) {
//CHECK-NEXT:                                cgh.parallel_for(
//CHECK-NEXT:                                    cl::sycl::nd_range<3>(
//CHECK-NEXT:                                        cl::sycl::range<3>(1, 1, 1),
//CHECK-NEXT:                                        cl::sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:                                    [=](cl::sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:                                                    k();
//CHECK-NEXT:                                    });
//CHECK-NEXT:                });
//CHECK-NEXT:}
void foo() {
		k<<<1, 1>>>();
}

//     CHECK:void foo2() {
//CHECK-NEXT:		if (1)
//CHECK-NEXT:                                dpct::get_default_queue().submit(
//CHECK-NEXT:                                    [&](cl::sycl::handler &cgh) {
//CHECK-NEXT:                                                    cgh.parallel_for(
//CHECK-NEXT:                                                        cl::sycl::nd_range<3>(
//CHECK-NEXT:                                                            cl::sycl::range<3>(
//CHECK-NEXT:                                                                1, 1, 1),
//CHECK-NEXT:                                                            cl::sycl::range<3>(
//CHECK-NEXT:                                                                1, 1, 1)),
//CHECK-NEXT:                                                        [=](cl::sycl::nd_item<3>
//CHECK-NEXT:                                                                item_ct1) {
//CHECK-NEXT:                                                                        k();
//CHECK-NEXT:                                                        });
//CHECK-NEXT:                                    });
//CHECK-NEXT:}
void foo2() {
		if (1)
				k<<<1, 1>>>();
}

//     CHECK:void foo3() {
//CHECK-NEXT:		while (1) {
//CHECK-NEXT:                                dpct::get_default_queue().submit(
//CHECK-NEXT:                                    [&](cl::sycl::handler &cgh) {
//CHECK-NEXT:                                                    cgh.parallel_for(
//CHECK-NEXT:                                                        cl::sycl::nd_range<3>(
//CHECK-NEXT:                                                            cl::sycl::range<3>(
//CHECK-NEXT:                                                                1, 1, 1),
//CHECK-NEXT:                                                            cl::sycl::range<3>(
//CHECK-NEXT:                                                                1, 1, 1)),
//CHECK-NEXT:                                                        [=](cl::sycl::nd_item<3>
//CHECK-NEXT:                                                                item_ct1) {
//CHECK-NEXT:                                                                        k();
//CHECK-NEXT:                                                        });
//CHECK-NEXT:                                    });
//CHECK-NEXT:                }
//CHECK-NEXT:}
void foo3() {
		while (1) {
				k<<<1, 1>>>();
		}
}

//     CHECK:void foo4() {
//CHECK-NEXT:		for (;;) {
//CHECK-NEXT:                                dpct::get_default_queue().submit(
//CHECK-NEXT:                                    [&](cl::sycl::handler &cgh) {
//CHECK-NEXT:                                                    cgh.parallel_for(
//CHECK-NEXT:                                                        cl::sycl::nd_range<3>(
//CHECK-NEXT:                                                            cl::sycl::range<3>(
//CHECK-NEXT:                                                                1, 1, 1),
//CHECK-NEXT:                                                            cl::sycl::range<3>(
//CHECK-NEXT:                                                                1, 1, 1)),
//CHECK-NEXT:                                                        [=](cl::sycl::nd_item<3>
//CHECK-NEXT:                                                                item_ct1) {
//CHECK-NEXT:                                                                        k();
//CHECK-NEXT:                                                        });
//CHECK-NEXT:                                    });
//CHECK-NEXT:                }
//CHECK-NEXT:}
void foo4() {
		for (;;) {
				k<<<1, 1>>>();
		}
}