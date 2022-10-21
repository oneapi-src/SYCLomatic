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
