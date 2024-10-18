// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --use-syclcompat --format-range=none --out-root %T/syclcompat_test3 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/syclcompat_test3/syclcompat_test3.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -DNO_BUILD_TEST -c -fsycl %T/syclcompat_test3/syclcompat_test3.dp.cpp -o %T/syclcompat_test3/syclcompat_test3.dp.o %}

#include <cudnn.h>
#include <cuda.h>

void f1() {
  size_t reserve_size;
  // CHECK: syclcompat::err1 status = {0};
  cudnnStatus_t status = {CUDNN_STATUS_SUCCESS};
  // CHECK: size_t version = dpct::dnnl::get_version();
  // CHECK-NEXT: dpct::dnnl::memory_desc_ext dataTensor, outTensor;
  // CHECK-NEXT: dpct::dnnl::engine_ext::derive_batch_normalization_memory_desc(outTensor, dataTensor, dpct::dnnl::batch_normalization_mode::per_activation);
  // CHECK-NEXT: dpct::dnnl::engine_ext::derive_batch_normalization_memory_desc(outTensor, outTensor, dataTensor, dpct::dnnl::batch_normalization_mode::per_activation);
  // CHECK-NEXT: reserve_size = dpct::dnnl::engine_ext::get_dropout_workspace_size(dataTensor);
  size_t version = cudnnGetVersion();
  cudnnTensorDescriptor_t dataTensor, outTensor;
  cudnnDeriveBNTensorDescriptor(outTensor, dataTensor, CUDNN_BATCHNORM_PER_ACTIVATION);
  cudnnDeriveNormTensorDescriptor(outTensor, outTensor, dataTensor, CUDNN_NORM_PER_ACTIVATION, 1);
  cudnnDropoutGetReserveSpaceSize(dataTensor, &reserve_size);
}

void f2() {
  int result;
  CUdevice device;
  // CHECK: result = syclcompat::get_device(device).get_mem_base_addr_align() / 8;
  cuDeviceGetAttribute(&result, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, device);
}

void f3() {
  int result;
  int dev_id;
  // CHECK: result = syclcompat::get_device(dev_id).get_mem_base_addr_align() / 8;
  cudaDeviceGetAttribute(&result, cudaDevAttrTextureAlignment, dev_id);
}
