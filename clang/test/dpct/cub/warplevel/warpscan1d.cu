// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --assume-nd-range-dim=1 -in-root %S -out-root %T/warplevel/warpscan1d %S/warpscan1d.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/warplevel/warpscan1d/warpscan1d.dp.cpp --match-full-lines %s

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define WARP_SIZE 32

void init_data(int* data, int num) {
  for(int i = 0; i < num; i++)
    data[i] = 1;
}
void verify_data(int* data, int num) {
  return;
}
void print_data(int* data, int num) {
  for (int i = 0; i < num; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

//CHECK: template<typename ScanTy, typename DataTy>
//CHECK-NEXT: void Scan1(ScanTy &s) {
//CHECK-NEXT:  DataTy d;
//CHECK-NEXT:  d = sycl::inclusive_scan_over_group(s, d, sycl::plus<>());
//CHECK-NEXT: }
template<typename ScanTy, typename DataTy>
__device__ void Scan1(ScanTy &s) {
  DataTy d;
  s.InclusiveSum(d, d);
}

//CHECK: void TemplateKernel1(int* data,
//CHECK-NEXT:  sycl::nd_item<1> item_ct1) {
//CHECK-NEXT:  typedef sycl::ext::oneapi::sub_group WarpScan;
//CHECK-NEXT:  typedef sycl::group<1> BlockScan;
//CHECK-EMPTY:
//CHECK-NEXT:  WarpScan ws(item_ct1.get_sub_group());
//CHECK-NEXT:  BlockScan bs(item_ct1.get_group());
//CHECK-NEXT:  Scan1<WarpScan, int>(ws);
//CHECK-NEXT:  Scan1<BlockScan, int>(bs);
//CHECK-NEXT:}
__global__ void TemplateKernel1(int* data) {
  typedef cub::WarpScan<int> WarpScan;
  typedef cub::BlockScan<int, 8> BlockScan;

  typename WarpScan::TempStorage temp1;
  typename BlockScan::TempStorage temp2;
  WarpScan ws(temp1);
  BlockScan bs(temp2);
  Scan1<WarpScan, int>(ws);
  Scan1<BlockScan, int>(bs);
}

int main() {
  int* dev_data = nullptr;

  dim3 GridSize(2);
  dim3 BlockSize(1 , 1, 128);
  int TotalThread = GridSize.x * BlockSize.x * BlockSize.y * BlockSize.z;

  cudaMallocManaged(&dev_data, TotalThread * sizeof(int));

  init_data(dev_data, TotalThread);

//CHECK: q_ct1.parallel_for(
//CHECK-NEXT:           sycl::nd_range<1>(sycl::range<1>(2) * sycl::range<1>(128), sycl::range<1>(128)),
//CHECK-NEXT:           [=](sycl::nd_item<1> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:             TemplateKernel1(dev_data, item_ct1);
//CHECK-NEXT:           });
  TemplateKernel1<<<2, 128>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  return 0;
}
