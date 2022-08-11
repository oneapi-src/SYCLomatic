// RUN: dpct --format-range=none -out-root %T/Out %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/Out/cudaPointerAttributes.dp.cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
int main() {
  int N = 2048;
  size_t size = N * sizeof(float);

  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);

  float *d_A;
  float *d_B;
  float *d_C;
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  // CHECK: dpct::pointer_attributes attributes;
  cudaPointerAttributes attributes;
  // CHECK: dpct::get_pointer_attributes(attributes, h_A);
  cudaPointerGetAttributes (&attributes, h_A);
  // CHECK: std::cout << attributes.device << std::endl;
  // CHECK: std::cout << attributes.memory_type << std::endl;
  // CHECK: std::cout << attributes.host_pointer << std::endl;
  // CHECK: std::cout << attributes.device_pointer << std::endl;
  std::cout << attributes.device << std::endl;
  std::cout << attributes.type << std::endl;
  std::cout << attributes.hostPointer << std::endl;
  std::cout << attributes.devicePointer << std::endl;
  // CHECK: dpct::pointer_attributes *attributes2 = new dpct::pointer_attributes();
  cudaPointerAttributes *attributes2 = new cudaPointerAttributes();
  // CHECK: dpct::get_pointer_attributes(*attributes2, h_A);
  cudaPointerGetAttributes (attributes2, h_A);
  // CHECK: std::cout << attributes2->device << std::endl;
  // CHECK: std::cout << attributes2->memory_type << std::endl;
  std::cout << attributes2->device << std::endl;
  std::cout << attributes2->type << std::endl;
  // CHECK: if (attributes2->memory_type == sycl::usm::alloc::host) {
  // CHECK: } else if (attributes2->memory_type == sycl::usm::alloc::device) {
  // CHECK: } else if (attributes2->memory_type == sycl::usm::alloc::unknown) {
  // CHECK: } else if (attributes2->memory_type == sycl::usm::alloc::unknown) {
  if (attributes2->type == cudaMemoryTypeHost) {
    return -1;
  } else if (attributes2->type == cudaMemoryTypeDevice) {
    return -1;
  } else if (attributes2->type == cudaMemoryTypeManaged) {
    return -1;
  } else if (attributes2->type == cudaMemoryTypeUnregistered) {
    return -1;
  }
}
