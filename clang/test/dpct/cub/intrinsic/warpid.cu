// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/intrinsic/warpid %S/warpid.cu --cuda-include-path="%cuda-path/include" -- -std=c++17 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/intrinsic/warpid/warpid.dp.cpp %s

// CHECK:#include <sycl/sycl.hpp>
// CHECK:#include <dpct/dpct.hpp>
#include <cub/cub.cuh>
#include <limits>
#include <map>
#include <stdio.h>

// CHECK: void warpid(int *id, int *ws, const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:   int tid = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
// CHECK-NEXT:   id[tid] = item_ct1.get_sub_group().get_group_linear_id();
// CHECK-NEXT:   *ws = item_ct1.get_sub_group().get_local_range().get(0);
// CHECK-NEXT: }
__global__ void warpid(int *id, int *ws) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  id[tid] = cub::WarpId();
  *ws = warpSize;
}

int main() {
  int *id = nullptr;
  int *WS = nullptr;
  cudaMallocManaged(&id, sizeof(int) * 64);
  cudaMallocManaged(&WS, sizeof(int));
  warpid<<<1, 64>>>(id, WS);
  cudaDeviceSynchronize();

  std::map<int, int> cnt;

  for (int i = 0; i < 64; ++i) {
    cnt[id[i]]++;
  }

  for (const auto &[WarpId, WarpIdNum] : cnt) {
    if (WarpIdNum != *WS) {
      printf("Incorrect result: warpId: %d : warpIdNum: %d : warpSize: %d\n", WarpId, WarpIdNum, *WS);
    }
  }

  return 0;
}
