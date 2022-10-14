// UNSUPPORTED: v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8
// UNSUPPORTED: cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// RUN: dpct --format-range=none -out-root %T/Out/cudaPointerAttributes %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/Out/cudaPointerAttributes/cudaPointerAttributes.dp.cpp
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
  cudaPointerGetAttributes (&attributes, h_A);
  // CHECK: std::cout << attributes.get_device_id() << std::endl;
  // CHECK: std::cout << attributes.get_memory_type() << std::endl;
  // CHECK: std::cout << attributes.get_host_pointer() << std::endl;
  // CHECK: std::cout << attributes.get_device_pointer() << std::endl;
  // CHECK: std::cout << attributes.is_memory_shared() << std::endl;
  std::cout << "====== Host Attributes =======" << std::endl;
  std::cout << attributes.device << std::endl;
  std::cout << attributes.memoryType << std::endl;
  std::cout << attributes.hostPointer << std::endl;
  std::cout << attributes.devicePointer << std::endl;
  std::cout << attributes.isManaged << std::endl;

  void * malloc_host;
  cudaMallocHost((void **)&malloc_host, size);
  cudaPointerAttributes attributes2;
  cudaPointerGetAttributes (&attributes2, malloc_host);
  std::cout << "====== Malloc Host Attributes =======" << std::endl;
  std::cout << "malloc host " << malloc_host << std::endl;
  std::cout << attributes2.device << std::endl;
  std::cout << attributes2.memoryType << std::endl;
  std::cout << attributes2.hostPointer << std::endl;
  std::cout << attributes2.devicePointer << std::endl;
  std::cout << attributes2.isManaged << std::endl;

  // CHECK: dpct::pointer_attributes *attributes3 = new dpct::pointer_attributes();
  cudaPointerAttributes *attributes3 = new cudaPointerAttributes();
  // CHECK: attributes3->init(d_A);
  cudaPointerGetAttributes (attributes3, d_A);
  // CHECK: std::cout << attributes3->get_device_id() << std::endl;
  // CHECK: std::cout << attributes3->get_memory_type() << std::endl;
  std::cout << "====== Device Attributes =======" << std::endl;
  std::cout << attributes3->device << std::endl;
  std::cout << attributes3->memoryType << std::endl;
  std::cout << attributes3->hostPointer << std::endl;
  std::cout << attributes3->devicePointer << std::endl;
  std::cout << attributes3->isManaged << std::endl;
  // CHECK: if (attributes3->get_memory_type() == sycl::usm::alloc::host) {
  // CHECK: } else if (attributes3->get_memory_type() == sycl::usm::alloc::device) {
  // CHECK: } else if (attributes3->is_memory_shared()) {
  if (attributes3->memoryType == cudaMemoryTypeHost) {
    return 0;
  } else if (attributes3->memoryType == cudaMemoryTypeDevice) {
    return 1;
  } else if (attributes3->isManaged) {
    return 2;
  }
}
