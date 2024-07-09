// UNSUPPORTED: v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2, v12.3 , v12.4
// UNSUPPORTED: cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4
// RUN: dpct --format-range=none -out-root %T/Out/cudaPointerAttributes %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/Out/cudaPointerAttributes/cudaPointerAttributes.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/Out/cudaPointerAttributes/cudaPointerAttributes.dp.cpp -o %T/Out/cudaPointerAttributes/cudaPointerAttributes.dp.o %}
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
int main() {
  int N = 2048;
  size_t size = N * sizeof(float);

  float *h_A = (float *)malloc(size);

  float *d_A;
  cudaMalloc((void **)&d_A, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  unsigned int numAttributes = 5;

  // CHECK: dpct::attribute_type attributes[] = {
  // CHECK:   dpct::attribute_type::memory_type, dpct::attribute_type::device_pointer,
  // CHECK:   dpct::attribute_type::host_pointer, dpct::attribute_type::is_managed,
  // CHECK:   dpct::attribute_type::device_id};
  CUpointer_attribute attributes[] = {
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
    CU_POINTER_ATTRIBUTE_HOST_POINTER,
    CU_POINTER_ATTRIBUTE_IS_MANAGED,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
  };

  int memType;
  // CHECK: dpct::device_ptr devPtr;
  CUdeviceptr devPtr;
  void* hostPtr;
  unsigned int isManaged;
  int deviceID;

  void* attributeValues[] = {
    &memType,
    &devPtr,
    &hostPtr,
    &isManaged,
    &deviceID
  };

  // CHECK: dpct::pointer_attributes.get( 
  cuPointerGetAttributes(
    numAttributes,
    attributes,
    attributeValues,
    h_A
  );

  std::cout << "====== Host Attributes =======" << std::endl;
  std::cout << deviceID << std::endl;
  std::cout << memType << std::endl;
  std::cout << hostPtr << std::endl;
  std::cout << devPtr << std::endl;
  std::cout << isManaged << std::endl;

  void * malloc_host;
  cudaMallocHost((void **)&malloc_host, size);
  // CHECK: dpct::pointer_attributes.get( 
  cuPointerGetAttributes(
    numAttributes,
    attributes,
    attributeValues,
    malloc_host
  );
  std::cout << "====== Malloc Host Attributes =======" << std::endl;
  std::cout << "malloc host " << malloc_host << std::endl;
  std::cout << deviceID << std::endl;
  std::cout << memType << std::endl;
  std::cout << hostPtr << std::endl;
  std::cout << devPtr << std::endl;
  std::cout << isManaged << std::endl;

  // CHECK: dpct::pointer_attributes.get( 
  cuPointerGetAttributes(
    numAttributes,
    attributes,
    attributeValues,
    d_A
  );
  std::cout << "====== Device Attributes =======" << std::endl;
  std::cout << *static_cast<int *>(attributeValues[0]) << std::endl;
  std::cout << attributeValues[1] << std::endl;
  std::cout << attributeValues[2] << std::endl;
  std::cout << *static_cast<unsigned int *>(attributeValues[3]) << std::endl;
  std::cout << *static_cast<int *>(attributeValues[4]) << std::endl;
  // CHECK: if (memType == sycl::usm::alloc::host) {
  // CHECK: } else if (memType == sycl::usm::alloc::device) {
  // CHECK: } else if (isManaged) {
  if (memType == cudaMemoryTypeHost) {
    return 0;
  } else if (memType == cudaMemoryTypeDevice) {
    return 1;
  } else if (isManaged) {
    return 2;
  }
  // CHECK: if (memType == sycl::usm::alloc::unknown) {
  // CHECK: } else if (memType == sycl::usm::alloc::host) {
  // CHECK: } else if (memType == sycl::usm::alloc::device) {
  // CHECK: } else if (memType == sycl::usm::alloc::shared) {
  if (memType == 0) {
    return 0;
  } else if (memType == 1) {
    return 1;
  } else if (memType == 2) {
    return 2;
  } else if (memType == 3) {
    return 3;
  }
}
