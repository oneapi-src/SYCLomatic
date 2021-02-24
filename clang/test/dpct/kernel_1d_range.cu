// RUN: dpct --format-range=none --assume-nd-range-dim=1  -out-root %T/kernel_1d_range %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel_1d_range/kernel_1d_range.dp.cpp --match-full-lines %s


// k1(1D) -> d1 -> d2
// k2(3D) -> d3 -> d4

// k3(1D) -> d5 -> d6
// k4(3D) -> d7 -> d6

// k5(1D) -> d8 -> d10(no item)
// k6(3D) -> d9 -> d10(no item)

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

//CHECK:void k1(sycl::nd_item<1> item_ct1);
//CHECK-NEXT:void k2(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void k3(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void k4(sycl::nd_item<3> item_ct1);
//CHECK-NEXT:void k5(sycl::nd_item<1> item_ct1);
//CHECK-NEXT:void k6(sycl::nd_item<3> item_ct1);
__global__ void k1();
__global__ void k2();
__global__ void k3();
__global__ void k4();
__global__ void k5();
__global__ void k6();

//CHECK:void d1(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(0);
//CHECK-NEXT:  a = item_ct1.get_group(0);
//CHECK-NEXT:  a = item_ct1.get_local_range().get(0);
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
//CHECK-NEXT:  a = item_ct1.get_local_range().get(0);
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
//CHECK-NEXT:  a = item_ct1.get_local_range().get(2);
//CHECK-NEXT:  a = item_ct1.get_local_range().get(1);
//CHECK-NEXT:  a = item_ct1.get_local_range().get(0);
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
//CHECK-NEXT:  a = item_ct1.get_local_range().get(2);
//CHECK-NEXT:  a = item_ct1.get_local_range().get(1);
//CHECK-NEXT:  a = item_ct1.get_local_range().get(0);
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
//CHECK-NEXT:  a = item_ct1.get_local_range().get(2);
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
//CHECK-NEXT:  a = item_ct1.get_local_range().get(2);
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
//CHECK-NEXT:  a = item_ct1.get_local_range().get(2);
//CHECK-NEXT:  a = item_ct1.get_local_range().get(1);
//CHECK-NEXT:  a = item_ct1.get_local_range().get(0);
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
//CHECK-NEXT:  double b = sycl::sqrt((double)(item_ct1.get_local_id(0)));
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1064:{{[0-9]+}}: Migrated acos call is used in a macro definition and is not valid for all macro uses. Adjust the code.
//CHECK-NEXT:  */
//CHECK-NEXT:  double c = sycl::atan2((double)(sycl::acos((double)(item_ct1.get_local_id(0)))), (double)(sycl::acos((double)(item_ct1.get_local_id(0)))));
//CHECK-NEXT:}
__device__ void d11() {
  int a = threadIdx.x;
  double b = sqrt(threadIdx.x);
  double c = atan2(acos(threadIdx.x), acos(threadIdx.x));
}

//CHECK:void k1(sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  int a = item_ct1.get_local_id(0);
//CHECK-NEXT:  a = item_ct1.get_group(0);
//CHECK-NEXT:  a = item_ct1.get_local_range().get(0);
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
//CHECK-NEXT:  a = item_ct1.get_local_range().get(2);
//CHECK-NEXT:  a = item_ct1.get_local_range().get(1);
//CHECK-NEXT:  a = item_ct1.get_local_range().get(0);
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
//CHECK-NEXT:  a = item_ct1.get_local_range().get(2);
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
//CHECK-NEXT:  a = item_ct1.get_local_range().get(2);
//CHECK-NEXT:  a = item_ct1.get_local_range().get(1);
//CHECK-NEXT:  a = item_ct1.get_local_range().get(0);
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

int main() {
  int aa = 2;
  //CHECK:q_ct1.submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<1>(sycl::range<1>(aa) * sycl::range<1>(3), sycl::range<1>(3)), 
  //CHECK-NEXT:      [=](sycl::nd_item<1> item_ct1) {
  //CHECK-NEXT:        k1(item_ct1);
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  k1<<<aa, 3>>>();

  //CHECK:q_ct1.submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(6, 5, 4), sycl::range<3>(1, 1, 1)), 
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k2(item_ct1);
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  k2<<<dim3(4 ,5, 6), 1>>>();

  //CHECK:q_ct1.submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 7) * sycl::range<3>(1, 1, 8), sycl::range<3>(1, 1, 8)), 
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k3(item_ct1);
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  k3<<<7, 8>>>();

  //CHECK:q_ct1.submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 9) * sycl::range<3>(12, 11, 10), sycl::range<3>(12, 11, 10)), 
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k4(item_ct1);
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  k4<<<9, dim3(10 ,11, 12)>>>();

  int bb = 14;
  //CHECK:q_ct1.submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<1>(sycl::range<1>(13) * sycl::range<1>(bb), sycl::range<1>(bb)), 
  //CHECK-NEXT:      [=](sycl::nd_item<1> item_ct1) {
  //CHECK-NEXT:        k5(item_ct1);
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  k5<<<dim3(13), dim3(bb)>>>();

  dim3 cc(1, 2, 3);
  //CHECK:q_ct1.submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(cc) * sycl::range<3>(1, 1, 15), sycl::range<3>(1, 1, 15)), 
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k6(item_ct1);
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  k6<<<dim3(cc), 15>>>();
  return 0;
}

