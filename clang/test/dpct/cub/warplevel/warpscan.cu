// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/warplevel/warpscan %S/warpscan.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/warplevel/warpscan/warpscan.dp.cpp --match-full-lines %s

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

//CHECK: void ExclusiveScanKernel1(int* data,
//CHECK-NEXT:   sycl::nd_item<3> item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT: int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT: int input = data[threadid];
//CHECK-NEXT: int output = 0;
//CHECK-NEXT: output = sycl::exclusive_scan_over_group(item_ct1.get_sub_group(), input, sycl::plus<>());
//CHECK-NEXT: data[threadid] = output;
//CHECK-NEXT: }
__global__ void ExclusiveScanKernel1(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  WarpScan(temp1).ExclusiveScan(input, output, cub::Sum());
  data[threadid] = output;
}

//CHECK: void ExclusiveScanKernel2(int* data,
//CHECK-NEXT:   sycl::nd_item<3> item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT: int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT: int input = data[threadid];
//CHECK-NEXT: int output = 0;
//CHECK-NEXT: output = sycl::exclusive_scan_over_group(item_ct1.get_sub_group(), input, 0, sycl::plus<>());
//CHECK-NEXT: data[threadid] = output;
//CHECK-NEXT: }
__global__ void ExclusiveScanKernel2(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1[10];

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  WarpScan(temp1[0]).ExclusiveScan(input, output, 0, cub::Sum());
  data[threadid] = output;
}

//CHECK: void InclusiveScanKernel(int* data,
//CHECK-NEXT:   sycl::nd_item<3> item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT: int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT: int input = data[threadid];
//CHECK-NEXT: int output = 0;
//CHECK-NEXT: output = sycl::inclusive_scan_over_group(item_ct1.get_sub_group(), input, sycl::plus<>());
//CHECK-NEXT: data[threadid] = output;
//CHECK-NEXT: }
__global__ void InclusiveScanKernel(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  WarpScan(temp1).InclusiveScan(input, output, cub::Sum());
  data[threadid] = output;
}

//CHECK: void ExclusiveSumKernel(int* data,
//CHECK-NEXT:   sycl::nd_item<3> item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT: int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT: int input = data[threadid];
//CHECK-NEXT: int output = 0;
//CHECK-NEXT: output = sycl::exclusive_scan_over_group(item_ct1.get_sub_group(), input, sycl::plus<>());
//CHECK-NEXT: data[threadid] = output;
//CHECK-NEXT: }
__global__ void ExclusiveSumKernel(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  WarpScan(temp1).ExclusiveSum(input, output);
  data[threadid] = output;
}

//CHECK: void InclusiveSumKernel(int* data,
//CHECK-NEXT:   sycl::nd_item<3> item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT: int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT: int input = data[threadid];
//CHECK-NEXT: int output = 0;
//CHECK-NEXT: output = sycl::inclusive_scan_over_group(item_ct1.get_sub_group(), input, sycl::plus<>());
//CHECK-NEXT: data[threadid] = output;
//CHECK-NEXT: }
__global__ void InclusiveSumKernel(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  WarpScan(temp1).InclusiveSum(input, output);
  data[threadid] = output;
}

//CHECK: void BroadcastKernel(int* data,
//CHECK-NEXT:   sycl::nd_item<3> item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT: int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT: int input = data[threadid];
//CHECK-NEXT: int output = 0;
//CHECK-NEXT: output = sycl::group_broadcast(item_ct1.get_sub_group(), input, 0);
//CHECK-NEXT: data[threadid] = output;
//CHECK-NEXT: }
__global__ void BroadcastKernel(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  output = WarpScan(temp1).Broadcast(input, 0);
  data[threadid] = output;
}

//CHECK: void WarningTestKernel1(int* data,
//CHECK-NEXT:   sycl::nd_item<3> item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT:  int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT:  int input = data[threadid];
//CHECK-NEXT:  int output = 0;
//CHECK-NEXT:  output = sycl::inclusive_scan_over_group(item_ct1.get_sub_group(), input, sycl::plus<>());
//CHECK-NEXT:  data[threadid] = output;
//CHECK-EMPTY:
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1085:{{[0-9]+}}: The function inclusive_scan_over_group requires sub-group size to be 16, while other sub-group functions in the same SYCL kernel require a different sub-group size. You may need to adjust the code.
//CHECK-NEXT:  */
//CHECK-NEXT:  output = sycl::inclusive_scan_over_group(item_ct1.get_sub_group(), input, sycl::plus<>());
//CHECK-NEXT:  data[threadid] = output + data[threadid];
//CHECK-NEXT:}
__global__ void WarningTestKernel1(int* data) {
  typedef cub::WarpScan<int> WarpScan;
  typedef cub::WarpScan<int, 16> WarpScan16;

  typename WarpScan::TempStorage temp1;

  typename WarpScan16::TempStorage temp2;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  WarpScan(temp1).InclusiveSum(input, output);
  data[threadid] = output;

  WarpScan16(temp2).InclusiveSum(input, output);
  data[threadid] = output + data[threadid];
}

//CHECK: void WarpScanTest(sycl::nd_item<3> item_ct1){
//CHECK-EMPTY:
//CHECK-NEXT:  int data;
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1085:{{[0-9]+}}: The function inclusive_scan_over_group requires sub-group size to be 32, while other sub-group functions in the same SYCL kernel require a different sub-group size. You may need to adjust the code.
//CHECK-NEXT:  */
//CHECK-NEXT:  data = sycl::inclusive_scan_over_group(item_ct1.get_sub_group(), data, sycl::plus<>());
//CHECK-NEXT:}
__device__ void WarpScanTest(){
  typedef cub::WarpScan<int> WarpScan;
  typename WarpScan::TempStorage temp1;
  int data;
  WarpScan(temp1).InclusiveSum(data, data);
}

//CHECK: void WarpReduceTest(sycl::nd_item<3> item_ct1){
//CHECK-EMPTY:
//CHECK-NEXT:  int data;
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1085:{{[0-9]+}}: The function reduce_over_group requires sub-group size to be 16, while other sub-group functions in the same SYCL kernel require a different sub-group size. You may need to adjust the code.
//CHECK-NEXT:  */
//CHECK-NEXT:  data = sycl::reduce_over_group(item_ct1.get_sub_group(), data, sycl::plus<>());
//CHECK-NEXT:}
__device__ void WarpReduceTest(){
  typedef cub::WarpReduce<int, 16> WarpReduce;
  typename WarpReduce::TempStorage temp1;
  int data;
  data = WarpReduce(temp1).Sum(data);
}

//CHECK: void WarningTestKernel2(int* data,
//CHECK-NEXT:  sycl::nd_item<3> item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT: int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT: int input = data[threadid];
//CHECK-NEXT: int output = 0;
//CHECK-NEXT: output = sycl::inclusive_scan_over_group(item_ct1.get_sub_group(), input, sycl::plus<>());
//CHECK-NEXT: data[threadid] = output;
//CHECK-EMPTY:
//CHECK-NEXT: WarpScanTest(item_ct1);
//CHECK-EMPTY:
//CHECK-NEXT: WarpReduceTest(item_ct1);
//CHECK-NEXT: }
__global__ void WarningTestKernel2(int* data) {
  typedef cub::WarpScan<int, 8> WarpScan;

  typename WarpScan::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output = 0;
  WarpScan(temp1).InclusiveSum(input, output);
  data[threadid] = output;

  WarpScanTest();

  WarpReduceTest();
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
//CHECK-NEXT:   sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  typedef sycl::ext::oneapi::sub_group WarpScan;
//CHECK-NEXT:  typedef sycl::group<3> BlockScan;
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

__global__ void ScanKernel(int* data) {
  typedef cub::WarpScan<int> WarpScan;

  __shared__ typename WarpScan::TempStorage temp1;

  int threadid = threadIdx.x;

  int input = data[threadid];
  int output1 = 0, output2 = 0;
// CHECK: /*
// CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cub::WarpScan.Scan is not supported.
// CHECK-NEXT: */
// CHECK-NEXT: WarpScan(temp1).Scan(input, output1, output2, sycl::plus<>());
  WarpScan(temp1).Scan(input, output1, output2, cub::Sum());
  data[threadid] = output1 + output2;
}

int main() {
  int* dev_data = nullptr;

  dim3 GridSize(2);
  dim3 BlockSize(1 , 1, 128);
  int TotalThread = GridSize.x * BlockSize.x * BlockSize.y * BlockSize.z;

  cudaMallocManaged(&dev_data, TotalThread * sizeof(int));
  
  init_data(dev_data, TotalThread);
//CHECK:   q_ct1.parallel_for(
//CHECK-NEXT:         sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:           ExclusiveScanKernel1(dev_data, item_ct1);
//CHECK-NEXT:         });
  ExclusiveScanKernel1<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK:   q_ct1.parallel_for(
//CHECK-NEXT:         sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:           ExclusiveScanKernel2(dev_data, item_ct1);
//CHECK-NEXT:         });
  ExclusiveScanKernel2<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK:   q_ct1.parallel_for(
//CHECK-NEXT:         sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:           InclusiveScanKernel(dev_data, item_ct1);
//CHECK-NEXT:         });
  InclusiveScanKernel<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK:   q_ct1.parallel_for(
//CHECK-NEXT:         sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:           ExclusiveSumKernel(dev_data, item_ct1);
//CHECK-NEXT:         });
  ExclusiveSumKernel<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK:   q_ct1.parallel_for(
//CHECK-NEXT:         sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:           InclusiveSumKernel(dev_data, item_ct1);
//CHECK-NEXT:         });
  InclusiveSumKernel<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK:   q_ct1.parallel_for(
//CHECK-NEXT:         sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:           BroadcastKernel(dev_data, item_ct1);
//CHECK-NEXT:         });
  BroadcastKernel<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);


  init_data(dev_data, TotalThread);
//CHECK:  q_ct1.parallel_for(
//CHECK-NEXT:        sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:          WarningTestKernel1(dev_data, item_ct1);
//CHECK-NEXT:        });
  WarningTestKernel1<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK:  q_ct1.parallel_for(
//CHECK-NEXT:        sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(8){{\]\]}} {
//CHECK-NEXT:          WarningTestKernel2(dev_data, item_ct1);
//CHECK-NEXT:        });
  WarningTestKernel2<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  init_data(dev_data, TotalThread);
//CHECK:  q_ct1.parallel_for(
//CHECK-NEXT:        sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
//CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:          TemplateKernel1(dev_data, item_ct1);
//CHECK-NEXT:        });
  TemplateKernel1<<<GridSize, BlockSize>>>(dev_data);
  cudaDeviceSynchronize();
  verify_data(dev_data, TotalThread);

  return 0;
}
