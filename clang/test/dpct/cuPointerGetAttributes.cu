// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/Out/cuPointerGetAttributes %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/Out/cuPointerGetAttributes/cuPointerGetAttributes.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/Out/cuPointerGetAttributes/cuPointerGetAttributes.dp.cpp -o %T/Out/cuPointerGetAttributes/cuPointerGetAttributes.dp.o %}
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

  unsigned int numAttributes = 5;

  // CHECK: dpct::attribute_type attributes[] = {
  // CHECK:   dpct::attribute_type::memory_type,
  // CHECK:   dpct::attribute_type::device_pointer,
  // CHECK:   dpct::attribute_type::host_pointer,
  // CHECK:   dpct::attribute_type::is_managed,
  // CHECK:   dpct::attribute_type::device_id
  CUpointer_attribute attributes[] = {
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
    CU_POINTER_ATTRIBUTE_HOST_POINTER,
    CU_POINTER_ATTRIBUTE_IS_MANAGED,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
  };

  // CHECK: sycl::usm::alloc memType;
  CUmemorytype memType;
  void* hostPtr;
  unsigned int isManaged;
  int deviceID;
  // CHECK: dpct::device_ptr devPtr;
  CUdeviceptr devPtr;

  void* attributeValues[] = {
    &memType,
    &devPtr,
    &hostPtr,
    &isManaged,
    &deviceID
  };

  // CHECK: dpct::pointer_attributes::get(numAttributes, attributes, attributeValues, (dpct::device_ptr) h_A);
  cuPointerGetAttributes(
    numAttributes,
    attributes,
    attributeValues,
    (CUdeviceptr) h_A
  );

  std::cout << "====== Host Attributes =======" << std::endl;
  std::cout << deviceID << std::endl;
  std::cout << static_cast<int>(memType) << std::endl;
  std::cout << hostPtr << std::endl;
  std::cout << devPtr << std::endl;
  std::cout << isManaged << std::endl;

  void * malloc_host;
  cudaMallocHost((void **)&malloc_host, size);
  // CHECK: dpct::pointer_attributes::get(numAttributes, attributes, attributeValues, (dpct::device_ptr) malloc_host);
  cuPointerGetAttributes(
    numAttributes,
    attributes,
    attributeValues,
    (CUdeviceptr) malloc_host
  );
  std::cout << "====== Malloc Host Attributes =======" << std::endl;
  std::cout << "malloc host " << malloc_host << std::endl;
  std::cout << deviceID << std::endl;
  std::cout << static_cast<int>(memType) << std::endl;
  std::cout << hostPtr << std::endl;
  std::cout << devPtr << std::endl;
  std::cout << isManaged << std::endl;

  // CHECK: dpct::pointer_attributes::get(numAttributes, attributes, attributeValues, (dpct::device_ptr) d_A);
  cuPointerGetAttributes(
    numAttributes,
    attributes,
    attributeValues,
    (CUdeviceptr) d_A
  );
  std::cout << "====== Device Attributes =======" << std::endl;
  std::cout << *static_cast<int *>(attributeValues[0]) << std::endl;
  std::cout << attributeValues[1] << std::endl;
  std::cout << attributeValues[2] << std::endl;
  std::cout << *static_cast<unsigned int *>(attributeValues[3]) << std::endl;
  std::cout << *static_cast<int *>(attributeValues[4]) << std::endl;

  // CHECK: if (memType == sycl::usm::alloc::host) {
  if (memType == CU_MEMORYTYPE_HOST) {
    return 0;
  // CHECK: } else if (memType == sycl::usm::alloc::device) {
  } else if (memType == CU_MEMORYTYPE_DEVICE) {
    return 1;
  } else if (isManaged) {
    return 2;
  }
}
