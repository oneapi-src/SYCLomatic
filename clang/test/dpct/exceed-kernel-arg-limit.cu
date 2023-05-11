// RUN: dpct --format-range=none --usm-level=none -out-root %T/exceed-kernel-arg-limit %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/exceed-kernel-arg-limit/exceed-kernel-arg-limit.dp.cpp --match-full-lines %s

#include <stdio.h>
#include <cuda_runtime.h>
const int array_size = 2;

__constant__ int device1[array_size][array_size][array_size];
__constant__ int device2[array_size][array_size][array_size];
__constant__ int device3[array_size][array_size][array_size];
__constant__ int device4[array_size][array_size][array_size];
__constant__ int device5[array_size][array_size][array_size];
__constant__ int device6[array_size][array_size][array_size];
__constant__ int device7[array_size][array_size][array_size];
__constant__ int device8[array_size][array_size][array_size];
__constant__ int device9[array_size][array_size][array_size];
__constant__ int device10[array_size][array_size][array_size];
__constant__ int device11[array_size][array_size][array_size];
__constant__ int device12[array_size][array_size][array_size];
__constant__ int device13[array_size][array_size][array_size];


__global__ void kernel(int *out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    out[i] = device1[i][i][i] + device2[i][i][i] + device3[i][i][i] + device4[i][i][i] + device5[i][i][i]
           + device6[i][i][i] + device7[i][i][i] + device8[i][i][i] + device9[i][i][i] + device10[i][i][i]
           + device11[i][i][i] + device12[i][i][i] + device13[i][i][i];
}

__constant__ int a = 1;
static texture<uint2, 1> tex21;
__device__ int b[36][36];

__device__ void test() {
  __shared__ int c[36];
  c[0] = b[0][0] + a;
}

__device__ uint2 d[16];
__global__ void kernel2(int4 i4, int* ip) {
  test();
  __shared__ int e[12][12];
  d[0] = tex1D(tex21, 1);
  int i = 0;
  e[0][0] = i4.w + device1[i][i][i] + device2[i][i][i] + device3[i][i][i] + device4[i][i][i] + device5[i][i][i]
                 + device6[i][i][i] + device7[i][i][i];
  printf("test\n");
}

int main() {
    // 8(global) + 13*80 = 1048
    //CHECK: int *global;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1042:{{[0-9]+}}: The size of the arguments passed to the SYCL kernel exceeds the minimum size limit (1024) for a non-custom SYCL device. You can get the hardware argument size limit by querying info::device::max_parameter_size. You may need to rewrite this code if the size of the arguments exceeds the hardware limit.
    //CHECK-NEXT: */
    int *global;
    kernel<<<dim3(1, 1, 1), dim3(array_size, 1, 1)>>>(global);

    // 208(stream) + 16(i4) + 8(ip) + 48(image accessor + sampler) + 32(a) + 56(b) + 32(c) + 32(d) + 56(e) + 7*80 = 1048
    //CHECK: sycl::int4 i4;
    //CHECK: int* ip = 0;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1042:{{[0-9]+}}: The size of the arguments passed to the SYCL kernel exceeds the minimum size limit (1024) for a non-custom SYCL device. You can get the hardware argument size limit by querying info::device::max_parameter_size. You may need to rewrite this code if the size of the arguments exceeds the hardware limit.
    //CHECK-NEXT: */
    int4 i4;
    int* ip = 0;
    kernel2<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(i4, ip);
}


