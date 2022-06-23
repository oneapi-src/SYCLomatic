// RUN: dpct --format-range=none --assume-nd-range-dim=1  -out-root %T/kernel_1d_range %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel_1d_range/kernel_1d_range.dp.cpp --match-full-lines %s


// k1(1D) -> d1 -> d2
// k2(3D) -> d3 -> d4

// k3(1D) -> d5 -> d6
// k4(3D) -> d7 -> d6

// k5(1D) -> d8 -> d10(no item)
// k6(3D) -> d9 -> d10(no item)

// k7(1D) -> d12 -> d14(3D)
// k8(1D) -> d13 -> d14(3D)

//CHECK:void d1(sycl::nd_item<1> item_ct1);
//CHECK-NEXT:void d2(sycl::nd_item<1> item_ct1);
//CHECK-NEXT:void d3(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void d4(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void d5(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void d6(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void d7(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void d8(sycl::nd_item<1> item_ct1);
//CHECK-NEXT:void d9(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void d10();
//CHECK-NEXT:void d11(sycl::nd_item<1> item_ct1);
//CHECK-NEXT:void d12(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void d13(sycl::nd_item<3> item_ct1);
__device__ void d1();
__device__ void d2();
__device__ void d3();
__device__ void d4();
__device__ void d5();
__device__ void d6();
__device__ void d7();
__device__ void d8();
__device__ void d9();
__device__ void d10();
__device__ void d11();
__device__ void d12();
__device__ void d13();
__device__ void d14();

//CHECK:void k1(sycl::nd_item<1> item_ct1);
//CHECK-NEXT:void k2(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void k3(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void k4(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void k5(sycl::nd_item<1> item_ct1);
//CHECK-NEXT:void k6(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void k7(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void k8(sycl::nd_item<3> item_ct1);
__global__ void k1();
__global__ void k2();
__global__ void k3();
__global__ void k4();
__global__ void k5();
__global__ void k6();
__global__ void k7();
__global__ void k8();

//CHECK:void d1(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(0);
//CHECK-NEXT:  a = item_ct1.get_group(0);
//CHECK-NEXT:  a = item_ct1.get_local_range(0);
//CHECK-NEXT:  a = item_ct1.get_group_range(0);
//CHECK-NEXT:  d2(item_ct1);
//CHECK-NEXT:}
__device__ void d1() {
  int a = threadIdx.x;
  a = blockIdx.x;
  a = blockDim.x;
  a = gridDim.x;
  d2();
}

//CHECK:void d2(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(0);
//CHECK-NEXT:  a = item_ct1.get_group(0);
//CHECK-NEXT:  a = item_ct1.get_local_range(0);
//CHECK-NEXT:  a = item_ct1.get_group_range(0);
//CHECK-NEXT:}
__device__ void d2() {
  int a = threadIdx.x;
  a = blockIdx.x;
  a = blockDim.x;
  a = gridDim.x;
}

//CHECK:void d3(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  a = item_ct1.get_local_id(1);
//CHECK-NEXT:  a = item_ct1.get_local_id(0);
//CHECK-NEXT:  a = item_ct1.get_group(2);
//CHECK-NEXT:  a = item_ct1.get_group(1);
//CHECK-NEXT:  a = item_ct1.get_group(0);
//CHECK-NEXT:  a = item_ct1.get_local_range(2);
//CHECK-NEXT:  a = item_ct1.get_local_range(1);
//CHECK-NEXT:  a = item_ct1.get_local_range(0);
//CHECK-NEXT:  a = item_ct1.get_group_range(2);
//CHECK-NEXT:  a = item_ct1.get_group_range(1);
//CHECK-NEXT:  a = item_ct1.get_group_range(0);
//CHECK-NEXT:  d4(item_ct1);
//CHECK-NEXT:}
__device__ void d3() {
  int a = threadIdx.x;
  a = threadIdx.y;
  a = threadIdx.z;
  a = blockIdx.x;
  a = blockIdx.y;
  a = blockIdx.z;
  a = blockDim.x;
  a = blockDim.y;
  a = blockDim.z;
  a = gridDim.x;
  a = gridDim.y;
  a = gridDim.z;
  d4();
}

//CHECK:void d4(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  a = item_ct1.get_local_id(1);
//CHECK-NEXT:  a = item_ct1.get_local_id(0);
//CHECK-NEXT:  a = item_ct1.get_group(2);
//CHECK-NEXT:  a = item_ct1.get_group(1);
//CHECK-NEXT:  a = item_ct1.get_group(0);
//CHECK-NEXT:  a = item_ct1.get_local_range(2);
//CHECK-NEXT:  a = item_ct1.get_local_range(1);
//CHECK-NEXT:  a = item_ct1.get_local_range(0);
//CHECK-NEXT:  a = item_ct1.get_group_range(2);
//CHECK-NEXT:  a = item_ct1.get_group_range(1);
//CHECK-NEXT:  a = item_ct1.get_group_range(0);
//CHECK-NEXT:}
__device__ void d4() {
  int a = threadIdx.x;
  a = threadIdx.y;
  a = threadIdx.z;
  a = blockIdx.x;
  a = blockIdx.y;
  a = blockIdx.z;
  a = blockDim.x;
  a = blockDim.y;
  a = blockDim.z;
  a = gridDim.x;
  a = gridDim.y;
  a = gridDim.z;
}

//CHECK:void d5(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  a = item_ct1.get_group(2);
//CHECK-NEXT:  a = item_ct1.get_local_range(2);
//CHECK-NEXT:  a = item_ct1.get_group_range(2);
//CHECK-NEXT:  d6(item_ct1);
//CHECK-NEXT:}
__device__ void d5() {
  int a = threadIdx.x;
  a = blockIdx.x;
  a = blockDim.x;
  a = gridDim.x;
  d6();
}

//CHECK:void d6(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  a = item_ct1.get_group(2);
//CHECK-NEXT:  a = item_ct1.get_local_range(2);
//CHECK-NEXT:  a = item_ct1.get_group_range(2);
//CHECK-NEXT:}
__device__ void d6() {
  int a = threadIdx.x;
  a = blockIdx.x;
  a = blockDim.x;
  a = gridDim.x;
}

//CHECK:void d7(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  a = item_ct1.get_local_id(1);
//CHECK-NEXT:  a = item_ct1.get_local_id(0);
//CHECK-NEXT:  a = item_ct1.get_group(2);
//CHECK-NEXT:  a = item_ct1.get_group(1);
//CHECK-NEXT:  a = item_ct1.get_group(0);
//CHECK-NEXT:  a = item_ct1.get_local_range(2);
//CHECK-NEXT:  a = item_ct1.get_local_range(1);
//CHECK-NEXT:  a = item_ct1.get_local_range(0);
//CHECK-NEXT:  a = item_ct1.get_group_range(2);
//CHECK-NEXT:  a = item_ct1.get_group_range(1);
//CHECK-NEXT:  a = item_ct1.get_group_range(0);
//CHECK-NEXT:  d6(item_ct1);
//CHECK-NEXT:}
__device__ void d7() {
  int a = threadIdx.x;
  a = threadIdx.y;
  a = threadIdx.z;
  a = blockIdx.x;
  a = blockIdx.y;
  a = blockIdx.z;
  a = blockDim.x;
  a = blockDim.y;
  a = blockDim.z;
  a = gridDim.x;
  a = gridDim.y;
  a = gridDim.z;
  d6();
}

//CHECK:void d8(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(0);
//CHECK-NEXT:  d10();
//CHECK-NEXT:}
__device__ void d8() {
  int a = threadIdx.x;
  d10();
}

//CHECK:void d9(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  d10();
//CHECK-NEXT:}
__device__ void d9() {
  int a = threadIdx.x;
  d10();
}

//CHECK:void d10() {
//CHECK-NEXT:  int a = 1;
//CHECK-NEXT:}
__device__ void d10() {
  int a = 1;
}

//CHECK:void d11(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(0);
//CHECK-NEXT:  double b = sqrt(item_ct1.get_local_id(0));
//CHECK-NEXT:  double c = sycl::atan2(acos(item_ct1.get_local_id(0)), acos(item_ct1.get_local_id(0)));
//CHECK-NEXT:}
__device__ void d11() {
  int a = threadIdx.x;
  double b = sqrt(threadIdx.x);
  double c = atan2(acos(threadIdx.x), acos(threadIdx.x));
}

//CHECK:void d12(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  d14(item_ct1);
//CHECK-NEXT:}
__device__ void d12() {
  int a = threadIdx.x;
  d14();
}

//CHECK:void d13(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  d14(item_ct1);
//CHECK-NEXT:}
__device__ void d13() {
  int a = threadIdx.x;
  d14();
}

//CHECK:void d14(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  double b = sqrt(item_ct1.get_local_id(2));
//CHECK-NEXT:  double c = sycl::atan2(acos(item_ct1.get_local_id(0)), acos(item_ct1.get_local_id(1)));
//CHECK-NEXT:}
__device__ void d14() {
  int a = threadIdx.x;
  double b = sqrt(threadIdx.x);
  double c = atan2(acos(threadIdx.z), acos(threadIdx.y));
}

//CHECK:void k1(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(0);
//CHECK-NEXT:  a = item_ct1.get_group(0);
//CHECK-NEXT:  a = item_ct1.get_local_range(0);
//CHECK-NEXT:  a = item_ct1.get_group_range(0);
//CHECK-NEXT:  d1(item_ct1);
//CHECK-NEXT:}
__global__ void k1() {
  int a = threadIdx.x;
  a = blockIdx.x;
  a = blockDim.x;
  a = gridDim.x;
  d1();
}

//CHECK:void k2(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  a = item_ct1.get_local_id(1);
//CHECK-NEXT:  a = item_ct1.get_local_id(0);
//CHECK-NEXT:  a = item_ct1.get_group(2);
//CHECK-NEXT:  a = item_ct1.get_group(1);
//CHECK-NEXT:  a = item_ct1.get_group(0);
//CHECK-NEXT:  a = item_ct1.get_local_range(2);
//CHECK-NEXT:  a = item_ct1.get_local_range(1);
//CHECK-NEXT:  a = item_ct1.get_local_range(0);
//CHECK-NEXT:  a = item_ct1.get_group_range(2);
//CHECK-NEXT:  a = item_ct1.get_group_range(1);
//CHECK-NEXT:  a = item_ct1.get_group_range(0);
//CHECK-NEXT:  d3(item_ct1);
//CHECK-NEXT:}
__global__ void k2() {
  int a = threadIdx.x;
  a = threadIdx.y;
  a = threadIdx.z;
  a = blockIdx.x;
  a = blockIdx.y;
  a = blockIdx.z;
  a = blockDim.x;
  a = blockDim.y;
  a = blockDim.z;
  a = gridDim.x;
  a = gridDim.y;
  a = gridDim.z;
  d3();
}

//CHECK:void k3(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  a = item_ct1.get_group(2);
//CHECK-NEXT:  a = item_ct1.get_local_range(2);
//CHECK-NEXT:  a = item_ct1.get_group_range(2);
//CHECK-NEXT:  d5(item_ct1);
//CHECK-NEXT:}
__global__ void k3() {
  int a = threadIdx.x;
  a = blockIdx.x;
  a = blockDim.x;
  a = gridDim.x;
  d5();
}

//CHECK:void k4(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  a = item_ct1.get_local_id(1);
//CHECK-NEXT:  a = item_ct1.get_local_id(0);
//CHECK-NEXT:  a = item_ct1.get_group(2);
//CHECK-NEXT:  a = item_ct1.get_group(1);
//CHECK-NEXT:  a = item_ct1.get_group(0);
//CHECK-NEXT:  a = item_ct1.get_local_range(2);
//CHECK-NEXT:  a = item_ct1.get_local_range(1);
//CHECK-NEXT:  a = item_ct1.get_local_range(0);
//CHECK-NEXT:  a = item_ct1.get_group_range(2);
//CHECK-NEXT:  a = item_ct1.get_group_range(1);
//CHECK-NEXT:  a = item_ct1.get_group_range(0);
//CHECK-NEXT:  d7(item_ct1);
//CHECK-NEXT:}
__global__ void k4() {
  int a = threadIdx.x;
  a = threadIdx.y;
  a = threadIdx.z;
  a = blockIdx.x;
  a = blockIdx.y;
  a = blockIdx.z;
  a = blockDim.x;
  a = blockDim.y;
  a = blockDim.z;
  a = gridDim.x;
  a = gridDim.y;
  a = gridDim.z;
  d7();
}


//CHECK:void k5(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(0);
//CHECK-NEXT:  d8(item_ct1);
//CHECK-NEXT:}
__global__ void k5() {
  int a = threadIdx.x;
  d8();
}

//CHECK:void k6(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  d9(item_ct1);
//CHECK-NEXT:}
__global__ void k6() {
  int a = threadIdx.x;
  d9();
}

//CHECK:void k7(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  d12(item_ct1);
//CHECK-NEXT:}
__global__ void k7() {
  int a = threadIdx.x;
  d12();
}

//CHECK:void k8(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  d13(item_ct1);
//CHECK-NEXT:}
__global__ void k8() {
  int a = threadIdx.x;
  d13();
}

int main() {
  int aa = 2;
  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<1>(sycl::range<1>(aa) * sycl::range<1>(3), sycl::range<1>(3)),
  //CHECK-NEXT:      [=](sycl::nd_item<1> item_ct1) {
  //CHECK-NEXT:        k1(item_ct1);
  //CHECK-NEXT:      });
  k1<<<aa, 3>>>();

  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(6, 5, 4), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k2(item_ct1);
  //CHECK-NEXT:      });
  k2<<<dim3(4 ,5, 6), 1>>>();

  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 7) * sycl::range<3>(1, 1, 8), sycl::range<3>(1, 1, 8)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k3(item_ct1);
  //CHECK-NEXT:      });
  k3<<<7, 8>>>();

  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 9) * sycl::range<3>(12, 11, 10), sycl::range<3>(12, 11, 10)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k4(item_ct1);
  //CHECK-NEXT:      });
  k4<<<9, dim3(10 ,11, 12)>>>();

  int bb = 14;
  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<1>(sycl::range<1>(13) * sycl::range<1>(bb), sycl::range<1>(bb)),
  //CHECK-NEXT:      [=](sycl::nd_item<1> item_ct1) {
  //CHECK-NEXT:        k5(item_ct1);
  //CHECK-NEXT:      });
  k5<<<dim3(13), dim3(bb)>>>();

  dim3 cc(1, 2, 3);
  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(cc) * sycl::range<3>(1, 1, 15), sycl::range<3>(1, 1, 15)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k6(item_ct1);
  //CHECK-NEXT:      });
  k6<<<dim3(cc), 15>>>();

  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k7(item_ct1);
  //CHECK-NEXT:      });
  k7<<<1, 1>>>();

  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k8(item_ct1);
  //CHECK-NEXT:      });
  k8<<<1, 1>>>();
  return 0;
}

//CHECK: #define MM __umul24
//CHECK-NEXT: #define MUL(a, b) sycl::mul24((unsigned int)a, (unsigned int)b)
//CHECK-NEXT: void foo1(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:   unsigned int tid = MUL(item_ct1.get_local_range(0), item_ct1.get_group(0)) + item_ct1.get_local_range(0);
//CHECK-NEXT:   unsigned int tid2 = sycl::mul24((unsigned int)item_ct1.get_local_range(0), (unsigned int)item_ct1.get_group_range(0));
//CHECK-NEXT: }
#define MM __umul24
#define MUL(a, b) __umul24(a, b)
__device__ void foo1() {
  unsigned int      tid = MUL(blockDim.x, blockIdx.x) + blockDim.x;
  unsigned int      tid2 = MM(blockDim.x, gridDim.x);
}
//CHECK: void foo2(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:   unsigned int tid = MUL(item_ct1.get_local_range(1), item_ct1.get_group(0)) + item_ct1.get_local_range(2);
//CHECK-NEXT:   unsigned int tid2 = sycl::mul24((unsigned int)item_ct1.get_local_range(1), (unsigned int)item_ct1.get_group_range(0));
//CHECK-NEXT: }
__device__ void foo2() {
  unsigned int      tid = MUL(blockDim.y, blockIdx.z) + blockDim.x;
  unsigned int      tid2 = MM(blockDim.y, gridDim.z);
}


//CHECK:void device1(sycl::nd_item<1> item_ct1);
//CHECK-NEXT:void device2(sycl::nd_item<1> item_ct1);
__device__ void device1();
__device__  void device2();

//CHECK:void device1(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(0);
//CHECK-NEXT:  device2(item_ct1);
//CHECK-NEXT:}
__device__ void device1() {
  int a = threadIdx.x;
  device2();
}

//CHECK:void device2(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(0);
//CHECK-NEXT:  device1(item_ct1);
//CHECK-NEXT:}
__device__ void device2() {
  int a = threadIdx.x;
  device1();
}

//CHECK:void global1(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(0);
//CHECK-NEXT:  device1(item_ct1);
//CHECK-NEXT:}
__global__ void global1() {
  int a = threadIdx.x;
  device1();
}


int foo3() {
  //CHECK:dpct::get_default_queue().parallel_for(
  //CHECK-NEXT:      sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
  //CHECK-NEXT:      [=](sycl::nd_item<1> item_ct1) {
  //CHECK-NEXT:        global1(item_ct1);
  //CHECK-NEXT:      });
  global1<<<1,1>>>();
  return 0;
}


//CHECK:void device3(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(1);
//CHECK-NEXT:}
__device__ void device3() {
  int a = threadIdx.y;
}

//CHECK:void device4(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  device3(item_ct1);
//CHECK-NEXT:}
__device__ void device4() {
  device3();
}

//CHECK:void global2(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(2);
//CHECK-NEXT:  device3(item_ct1);
//CHECK-NEXT:}
__global__ void global2() {
  int a = threadIdx.x;
  device3();
}

int foo4() {
  //CHECK:dpct::get_default_queue().parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 3) * sycl::range<3>(1, 1, 4), sycl::range<3>(1, 1, 4)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        global2(item_ct1);
  //CHECK-NEXT:      });
  global2<<<3 ,4>>>();
  return 0;
}

//CHECK:#define TIDx item_ct1.get_local_id(0)
#define TIDx threadIdx.x

// CHECK: void global3(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int t = TIDx;
//CHECK-NEXT:}
__global__ void global3() {
  int t = TIDx;
}

// CHECK: void global4(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int t = TIDx;
//CHECK-NEXT:}
__global__ void global4() {
  int t = TIDx;
}

int foo5() {
  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
  //CHECK-NEXT:      [=](sycl::nd_item<1> item_ct1) {
  //CHECK-NEXT:        global3(item_ct1);
  //CHECK-NEXT:      });
  global3<<<1,1>>>();

  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
  //CHECK-NEXT:      [=](sycl::nd_item<1> item_ct1) {
  //CHECK-NEXT:        global4(item_ct1);
  //CHECK-NEXT:      });
  global4<<<1,1>>>();
  return 0;
}

//CHECK:#define TIDx2 item_ct1.get_local_id(2)
#define TIDx2 threadIdx.x

//CHECK:void global5(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int t = TIDx2;
//CHECK-NEXT:}
__global__ void global5() {
  int t = TIDx2;
}

//CHECK:void global6(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  unsigned int tid = MUL(TIDx2, TIDx2);
//CHECK-NEXT:}
__global__ void global6() {
  unsigned int tid = MUL(TIDx2, TIDx2);
}

int foo6() {
  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        global5(item_ct1);
  //CHECK-NEXT:      });
  global5<<<1,1>>>();

  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(3, 2, 1), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        global6(item_ct1);
  //CHECK-NEXT:      });
  global6<<<dim3(1,2,3),1>>>();
  return 0;
}