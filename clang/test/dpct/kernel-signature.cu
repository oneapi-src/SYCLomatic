// RUN: dpct --format-range=none -out-root %T/kernel-signature %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/kernel-signature/kernel-signature.dp.cpp

// CHECK: void test_01();
// CHECK: void test_06();
// CHECK: void test_06(int *, int *);
// CHECK: void test_06(int *pA, int *pB) { }
__global__ void test_01();
__global__ void test_06();
__global__ void test_06(int *, int *);
__global__ void test_06(int *pA, int *pB) { }

// CHECK: void test_02();
__host__ void test_02();

// CHECK: void test_03();
__host__ __device__ void test_03();

// CHECK: void test_04();
void test_04();

// CHECK: void test_05();
__device__ void test_05();


template <typename T>
struct foo
{
    //check:  T *getPointer()
    __device__ T *getPointer()
    {
        //check:  __device__ void error(void);
        extern __device__ void error(void);
        error();
        return NULL;
    }
};

