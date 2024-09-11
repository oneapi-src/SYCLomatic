// RUN: dpct --format-range=none -out-root %T/texture_syclcompat %s -use-syclcompat --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture_syclcompat/texture_syclcompat.dp.cpp --match-full-lines %s

#include <stdio.h>

struct texObjWrapper {
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "cudaTextureObject_t" is not currently supported with SYCLcompat. Please adjust the code manually.
  // CHECK-NEXT: */
  cudaTextureObject_t tex;
};

void func(int i) {}

template <typename T>
void funcT(T t) {}

// CHECK: /*
// CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "cudaTextureObject_t" is not currently supported with SYCLcompat. Please adjust the code manually.
// CHECK-NEXT: */
// CHECK-NEXT: void gather_force(const cudaTextureObject_t gridTexObj){}
__global__ void gather_force(const cudaTextureObject_t gridTexObj){}

// CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaResourceDesc" is not currently supported with SYCLcompat. Please adjust the code manually.
// CHECK-NEXT: */
// CHECK-NEXT: cudaResourceDesc res42;
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "cudaTextureDesc" is not currently supported with SYCLcompat. Please adjust the code manually.
// CHECK-NEXT: */
// CHECK-NEXT: cudaTextureDesc texDesc42;
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "cudaCreateTextureObject" is not currently supported with SYCLcompat. Please adjust the code manually.
// CHECK-NEXT: */
// CHECK-NEXT: cudaCreateTextureObject(&tex, &res42, &texDesc42, NULL);
template <class T> void BindTextureObject(cudaArray_t &data, cudaTextureObject_t &tex) {
  cudaResourceDesc res42;
  cudaTextureDesc texDesc42;
  cudaCreateTextureObject(&tex, &res42, &texDesc42, NULL);
}

// CHECK: DPCT1131:{{[0-9]+}}: The migration of "tex1D" is not currently supported with SYCLcompat. Please adjust the code manually.

// CHECK: DPCT1131:{{[0-9]+}}: The migration of "tex1Dfetch" is not currently supported with SYCLcompat. Please adjust the code manually.
__device__ void device01(cudaTextureObject_t tex21) {
  uint2 u21;
  tex1D(&u21, tex21, 0.5f);
  tex1Dfetch(&u21, tex21, 1);
}

// CHECK: DPCT1131:{{[0-9]+}}: The migration of "tex2D" is not currently supported with SYCLcompat. Please adjust the code manually.
__global__ void kernel(cudaTextureObject_t tex2, cudaTextureObject_t tex4) {
  float4 f42;
  device01(tex2);
  tex2D(&f42, tex4, 0.5f, 0.5f);
}

int main() {
  using CudaRGBA = float4;

// CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaArray_t" is not currently supported with SYCLcompat. Please adjust the code manually.
// CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaChannelFormatDesc" is not currently supported with SYCLcompat. Please adjust the code manually.
  cudaArray_t a42;
  cudaChannelFormatDesc desc42;
  desc42 = cudaCreateChannelDesc<CudaRGBA>();

  cudaTextureObject_t tex42;
  cudaResourceDesc res42;
  cudaTextureDesc texDesc42;

// CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaGetTextureObjectTextureDesc" is not currently supported with SYCLcompat. Please adjust the code manually.
// CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaGetTextureObjectResourceDesc" is not currently supported with SYCLcompat. Please adjust the code manually.
  cudaGetTextureObjectTextureDesc(&texDesc42, tex42);
  cudaGetTextureObjectResourceDesc(&res42, tex42);

// CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaDestroyTextureObject" is not currently supported with SYCLcompat. Please adjust the code manually.
  cudaDestroyTextureObject(tex42);
}

__global__ void mipmap_kernel(cudaTextureObject_t tex) {
  int i;
  float j, k, l, m;
// CHECK: DPCT1131:{{[0-9]+}}: The migration of "tex1DLod" is not currently supported with SYCLcompat. Please adjust the code manually.
  tex1DLod<short2>(tex, j, l);
// CHECK: DPCT1131:{{[0-9]+}}: The migration of "tex1DLod" is not currently supported with SYCLcompat. Please adjust the code manually.
  tex1DLod(&i, tex, j, l);
// CHECK: DPCT1131:{{[0-9]+}}: The migration of "tex2DLod" is not currently supported with SYCLcompat. Please adjust the code manually.
  tex2DLod<short2>(tex, j, k, l);
// CHECK: DPCT1131:{{[0-9]+}}: The migration of "tex2DLod" is not currently supported with SYCLcompat. Please adjust the code manually.
  tex2DLod(&i, tex, j, k, l);
// CHECK: DPCT1131:{{[0-9]+}}: The migration of "tex3DLod" is not currently supported with SYCLcompat. Please adjust the code manually.
  tex3DLod<short2>(tex, j, k, m, l);
// CHECK: DPCT1131:{{[0-9]+}}: The migration of "tex3DLod" is not currently supported with SYCLcompat. Please adjust the code manually.
  tex3DLod(&i, tex, j, k, m, l);
}

void mipmap() {
  unsigned int flag, l;
  cudaExtent e;
  cudaChannelFormatDesc desc;
  cudaArray_t pArr;
// CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaMipmappedArray_t" is not currently supported with SYCLcompat. Please adjust the code manually.
  cudaMipmappedArray_t pMipMapArr;
}
