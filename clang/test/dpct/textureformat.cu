// RUN: c2s --format-range=none --usm-level=none -out-root %T/textureformat %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/textureformat/textureformat.dp.cpp --match-full-lines %s
#include "cuda_runtime.h"
__global__ void transformKernel(float* output, cudaTextureObject_t texObj, int width, int height, float theta)
{

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  float u = x / (float)width;
  float v = y / (float)height;

  u -= 0.5f;
  v -= 0.5f;
  float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
  float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

  output[y * width + x] = tex2D<float>(texObj, tu, tv);
}
// CHECK: c2s::image_wrapper<sycl::float4, 2> tex42;
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
// CHECK-NEXT: */
// CHECK-NEXT: c2s::image_wrapper<sycl::float3, 2> tex32;
static texture<float4, 2> tex42;
static texture<float3, 2> tex32;

int main()
{
  int width = 10;
  int height = 10;
  int size = 10, angle = 10;
  int *h_data;

  // CHECK: /*
  // CHECK-NEXT: DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2s::image_channel channelDesc = c2s::image_channel(32, 32, 0, 0, c2s::image_channel_data_type::fp);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

  // CHECK: /*
  // CHECK-NEXT: DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: c2s::image_channel channelDesc1 = c2s::image_channel::create<sycl::float3>();
  cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc<float3>();

  cudaArray* cuArray;
  cudaMallocArray(&cuArray, &channelDesc, width, height);

  cudaMemcpyToArray(cuArray, 0, 0, h_data, size, cudaMemcpyHostToDevice);

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  float* output;
  cudaMalloc(&output, width * height * sizeof(float));

  dim3 dimBlock(16, 16);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
  transformKernel<<<dimGrid, dimBlock>>>(output, texObj, width, height, angle);

  cudaDestroyTextureObject(texObj);

  cudaFreeArray(cuArray);
  cudaFree(output);
  return 0;
}

