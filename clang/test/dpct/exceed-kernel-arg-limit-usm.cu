// RUN: dpct --format-range=none -out-root %T/exceed-kernel-arg-limit-usm %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/exceed-kernel-arg-limit-usm/exceed-kernel-arg-limit-usm.dp.cpp --match-full-lines %s

#include <stdio.h>
#include <cuda_runtime.h>
const int array_size = 33;

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
__constant__ int device14[array_size][array_size][array_size];
__constant__ int device15[array_size][array_size][array_size];
__constant__ int device16[array_size][array_size][array_size];
__constant__ int device17[array_size][array_size][array_size];
__constant__ int device18[array_size][array_size][array_size];
__constant__ int device19[array_size][array_size][array_size];
__constant__ int device20[array_size][array_size][array_size];
__constant__ int device21[array_size][array_size][array_size];
__constant__ int device22[array_size][array_size][array_size];
__constant__ int device23[array_size][array_size][array_size];
__constant__ int device24[array_size][array_size][array_size];
__constant__ int device25[array_size][array_size][array_size];
__constant__ int device26[array_size][array_size][array_size];
__constant__ int device27[array_size][array_size][array_size];
__constant__ int device28[array_size][array_size][array_size];
__constant__ int device29[array_size][array_size][array_size];
__constant__ int device30[array_size][array_size][array_size];
__constant__ int device31[array_size][array_size][array_size];
__constant__ int device32[array_size][array_size][array_size];


__global__ void kernel(int *out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    out[i] = device1[i][i][i] + device2[i][i][i] + device3[i][i][i] + device4[i][i][i] + device5[i][i][i]
           + device6[i][i][i] + device7[i][i][i] + device8[i][i][i] + device9[i][i][i] + device10[i][i][i]
           + device11[i][i][i] + device12[i][i][i] + device13[i][i][i] + device14[i][i][i] + device15[i][i][i]
           + device16[i][i][i] + device17[i][i][i] + device18[i][i][i] + device19[i][i][i] + device20[i][i][i]
           + device21[i][i][i] + device22[i][i][i] + device23[i][i][i] + device24[i][i][i] + device25[i][i][i]
           + device26[i][i][i] + device27[i][i][i] + device28[i][i][i] + device29[i][i][i] + device30[i][i][i]
           + device31[i][i][i] + device32[i][i][i];
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
                 + device6[i][i][i] + device7[i][i][i] + device8[i][i][i] + device9[i][i][i] + device10[i][i][i]
                 + device11[i][i][i] + device12[i][i][i] + device13[i][i][i] + device14[i][i][i] + device15[i][i][i]
                 + device16[i][i][i] + device17[i][i][i] + device18[i][i][i] + device19[i][i][i] + device20[i][i][i]
                 + device21[i][i][i] + device22[i][i][i];
  printf("test\n");
}

int main() {
    // 8(global) + 32*32 = 1032
    //CHECK: int *global;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1042:{{[0-9]+}}: The size of the arguments passed to the SYCL kernel exceeds the minimum size limit (1024) for a non-custom SYCL device. You can get the hardware argument size limit by querying info::device::max_parameter_size. You may need to rewrite this code if the size of the arguments exceeds the hardware limit.
    //CHECK-NEXT: */
    int *global;
    kernel<<<dim3(1, 1, 1), dim3(array_size, 1, 1)>>>(global);

    // 208(stream) + 16(i4) + 8(ip) + 48(image accessor + sampler) + 8(a) + 24(b) + 8(c) + 8(d) + 24(e) + 22*32 = 1056
    //CHECK: sycl::int4 i4;
    //CHECK: int* ip = 0;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1042:{{[0-9]+}}: The size of the arguments passed to the SYCL kernel exceeds the minimum size limit (1024) for a non-custom SYCL device. You can get the hardware argument size limit by querying info::device::max_parameter_size. You may need to rewrite this code if the size of the arguments exceeds the hardware limit.
    //CHECK-NEXT: */
    int4 i4;
    int* ip = 0;
    kernel2<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(i4, ip);
}


