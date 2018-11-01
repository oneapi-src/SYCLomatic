// RUN: syclct -out-root %T %s -passes "FunctionAttrsRule,KernelIterationSpaceRule" -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/kernel-signature.sycl.cpp

// CHECK: void test_01(cl::sycl::nd_item<3> item_{{[a-f0-9]+}});
// CHECK: void test_06(cl::sycl::nd_item<3> item_{{[a-f0-9]+}});
// CHECK: void test_06(cl::sycl::nd_item<3> item_{{[a-f0-9]+}}, int *, int *);
// CHECK: void test_06(cl::sycl::nd_item<3> item_{{[a-f0-9]+}}, int *pA, int *pB) { }
__global__ void test_01();
__global__ void test_06();
__global__ void test_06(int *, int *);
__global__ void test_06(int *pA, int *pB) { }

// CHECK: void test_02();
__host__ void test_02();

// CHECK: void test_03();
__host__ __device__ void test_03();

// CHECK: void test_04() ;
void test_04() __device__;

// CHECK: void test_05() ;
__device__ void test_05() __device__;
