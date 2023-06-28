// RUN: dpct --format-range=none -out-root %T/pointer_to_device_array %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/pointer_to_device_array/pointer_to_device_array.dp.cpp

// CHECK: dpct::global_memory<int, 2> arr(sycl::range<2>(200, 4), {0});
__device__ int arr[200][4] = {0};

// CHECK: void my_kernel(dpct::accessor<int, dpct::global, 2> arr) {
// CHECK-NEXT:   int (*p)[4] = NULL;
// CHECK-NEXT:   p = (int (*)[4])arr.get_ptr();
// CHECK-NEXT: }
__device__ void my_kernel() {
  int (*p)[4] = NULL;
  p = arr;
}

typedef float *aaa_t[5][6];
// CHECK: static dpct::constant_memory<float *, 2> var1(5, 6);
__constant__ aaa_t var1;

__global__ void kernel1() {
  aaa_t *ptr;
  // CHECK: ptr = (aaa_t *)var1.get_ptr();
  ptr = &var1;
}

// CHECK: static dpct::constant_memory<float *, 2> var2(5, 6);
__constant__ float * var2[5][6];

__global__ void kernel2() {
  aaa_t *ptr;
  // CHECK: ptr = (float *(*)[5][6])var2.get_ptr();
  ptr = &var2;
}

typedef float *bbb_t[5];
// CHECK: static dpct::constant_memory<float *, 1> var3(5);
__constant__ bbb_t var3;

__global__ void kernel3() {
  bbb_t *ptr;
  // CHECK: ptr = (bbb_t *)&var3;
  ptr = &var3;
}

// CHECK: static dpct::constant_memory<float *, 1> var4(5);
__constant__ float * var4[5];

__global__ void kernel4() {
  bbb_t *ptr;
  // CHECK: ptr = (float *(*)[5])&var4;
  ptr = &var4;
}

typedef float *ccc_t;
// CHECK: static dpct::constant_memory<ccc_t, 0> var5;
__constant__ ccc_t var5;

__global__ void kernel5() {
  ccc_t *ptr;
  // CHECK: ptr = &var5;
  ptr = &var5;
}

// CHECK: static dpct::constant_memory<float *, 0> var6;
__constant__ float * var6;

__global__ void kernel6() {
  ccc_t *ptr;
  // CHECK: ptr = &var6;
  ptr = &var6;
}
