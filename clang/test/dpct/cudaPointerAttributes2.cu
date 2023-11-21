// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none -out-root %T/Out/cudaPointerAttributes2 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/Out/cudaPointerAttributes2/cudaPointerAttributes2.dp.cpp
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
  // CHECK: attributes.init(h_A);
  cudaPointerGetAttributes(&attributes, h_A);
  // CHECK: if (attributes.get_device_id() != (unsigned int)-1) {
  // CHECK: attributes.get_memory_type();
  // CHECK: std::cout << attributes.get_host_pointer() << std::endl;
  // CHECK: std::cout << attributes.get_device_pointer() << std::endl;
  std::cout << "====== Host Attributes =======" << std::endl;
  // The malloc memory is not a USM allocation memory. So the device attr is the default value(-1);
  if (attributes.device != (unsigned int)-1) {
    return -1;
  }
  attributes.type;
  std::cout << attributes.hostPointer << std::endl;
  std::cout << attributes.devicePointer << std::endl;

  void * malloc_host;
  cudaMallocHost((void **)&malloc_host, size);
  cudaPointerAttributes attributes2;
  cudaPointerGetAttributes (&attributes2, malloc_host);
  std::cout << "====== Malloc Host Attributes =======" << std::endl;
  std::cout << "malloc host " << malloc_host << std::endl;
  std::cout << attributes2.device << std::endl;
  attributes2.type;
  std::cout << attributes2.hostPointer << std::endl;
  std::cout << attributes2.devicePointer << std::endl;

  // CHECK: dpct::pointer_attributes *attributes3 = new dpct::pointer_attributes();
  cudaPointerAttributes *attributes3 = new cudaPointerAttributes();
  // CHECK: attributes3->init(d_A);
  cudaPointerGetAttributes (attributes3, d_A);
  // CHECK: std::cout << attributes3->get_device_id() << std::endl;
  // CHECK: attributes3->get_memory_type();
  std::cout << "====== Device Attributes =======" << std::endl;
  std::cout << attributes3->device << std::endl;
  attributes3->type;
  std::cout << attributes3->hostPointer << std::endl;
  std::cout << attributes3->devicePointer << std::endl;
  // CHECK: if (attributes3->get_memory_type() == sycl::usm::alloc::host) {
  // CHECK: } else if (attributes3->get_memory_type() == sycl::usm::alloc::device) {
  // CHECK: } else if (attributes3->get_memory_type() == sycl::usm::alloc::shared) {
  // CHECK: } else if (attributes3->get_memory_type() == sycl::usm::alloc::unknown) {
  if (attributes3->type == cudaMemoryTypeHost) {
    return 0;
  } else if (attributes3->type == cudaMemoryTypeDevice) {
    return 1;
  } else if (attributes3->type == cudaMemoryTypeManaged) {
    return 2;
  } else if (attributes3->type == cudaMemoryTypeUnregistered) {
    return 3;
  }
}
