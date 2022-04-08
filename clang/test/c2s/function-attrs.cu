// RUN: c2s --format-range=none -out-root %T/function-attrs %s -passes "IterationSpaceBuiltinRule" --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/function-attrs/function-attrs.dp.cpp

// CHECK: void test_00();
__device__ void test_00();

// CHECK: void test_01();
__global__ void test_01();

// CHECK: void test_02();
__host__ void test_02();

// CHECK: void test_03();
__host__ __device__ void test_03();

// CHECK: void test_04();
void test_04();

// CHECK: void test_05();
__device__ void test_05();

// Test that the attribute is properly removed from all function declarations
// even if there are several of them.
// CHECK: void test_06();
// CHECK: void test_06();
// CHECK: void test_06() { }
__global__ void test_06();
__global__ void test_06();
__global__ void test_06() { }
// CHECK: void test_07();
__global__    void test_07();
// CHECK: void test_08();
__global__	    void test_08();


