// RUN: cp %s %t
// RUN: cu2sycl %t -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %t

// CHECK: void test_00();
__device__ void test_00();

// CHECK: void test_01();
__global__ void test_01();

// CHECK: void test_02();
__host__ void test_02();

// CHECK: void test_03();
__host__ __device__ void test_03();

// CHECK: void test_04() ;
void test_04() __device__;

// CHECK: void test_05() ;
__device__ void test_05() __device__;

// Test that the attribute is properly removed from all function declarations
// even if there are several of them.
// CHECK: void test_06();
// CHECK: void test_06();
// CHECK: void test_06() { }
__global__ void test_06();
__global__ void test_06();
__global__ void test_06() { }
