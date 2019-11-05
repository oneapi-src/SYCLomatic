// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture_object.dp.cpp --match-full-lines %s
// UNSUPPORTED: cdua-9.0, cuda-9.2, cuda-10.0, cuda-10.1
// UNSUPPORTED: v9.0, v9.2, v10.0, v10.1

// CHECK: void device01(dpct::dpct_image_accessor<cl::sycl::uint2, 1> tex21) {
// CHECK-NEXT: cl::sycl::uint2 u21;
// CHECK-NEXT: dpct::dpct_read_image(&u21, tex21, 0.5f);
// CHECK-NEXT: dpct::dpct_read_image(&u21, tex21, 1);
__device__ void device01(cudaTextureObject_t tex21) {
  uint2 u21;
  tex1D(&u21, tex21, 0.5f);
  tex1Dfetch(&u21, tex21, 1);
}

// CHECK: void kernel(dpct::dpct_image_accessor<cl::sycl::uint2, 1> tex21, dpct::dpct_image_accessor<cl::sycl::float4, 2> tex42) {
// CHECK-NEXT: device01(tex21);
// CHECK-NEXT: cl::sycl::float4 f42;
// CHECK-NEXT: dpct::dpct_read_image(&f42, tex42, 0.5f, 0.5f);
/// Texture accessors should be passed down to __global__/__device__ function if used.
__global__ void kernel(cudaTextureObject_t tex21, cudaTextureObject_t tex42) {
  device01(tex21);
  float4 f42;
  tex2D(&f42, tex42, 0.5f, 0.5f);
}

int main() {

  // CHECK: cl::sycl::float4 *d_data42;
  // CHECK-NEXT: dpct::dpct_image_matrix_p a42;
  // CHECK-NEXT: dpct::dpct_malloc(&d_data42, sizeof(cl::sycl::float4) * 32 * 32);
  // CHECK-NEXT: dpct::dpct_image_channel desc42 = dpct::create_image_channel(32, 32, 32, 32, dpct::channel_float);
  // CHECK-NEXT: dpct::dpct_malloc_matrix(&a42, &desc42, 32, 32);
  // CHECK-NEXT: dpct::dpct_memcpy_to_matrix(a42, 0, 0, d_data42, 32 * 32 * sizeof(cl::sycl::float4));
  // CHECK-NEXT: dpct::dpct_image_base_p tex42;
  // CHECK-NEXT: dpct::dpct_image_data res42;
  // CHECK-NEXT: dpct::dpct_image_info texDesc42;
  // CHECK-NEXT: res42.type = dpct::data_matrix;
  // CHECK-NEXT: res42.data.matrix = a42;
  // CHECK-NEXT: texDesc42.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: texDesc42.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: texDesc42.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: texDesc42.filter_mode() = cl::sycl::filtering_mode::nearest;
  // CHECK-NEXT: dpct::dpct_create_image(&tex42, &res42, &texDesc42);
  float4 *d_data42;
  cudaArray_t a42;
  cudaMalloc(&d_data42, sizeof(float4) * 32 * 32);
  cudaChannelFormatDesc desc42 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaMallocArray(&a42, &desc42, 32, 32);
  cudaMemcpyToArray(a42, 0, 0, d_data42, 32 * 32 * sizeof(float4), cudaMemcpyDeviceToDevice);
  cudaTextureObject_t tex42;
  cudaResourceDesc res42;
  cudaTextureDesc texDesc42;
  res42.resType = cudaResourceTypeArray;
  res42.res.array.array = a42;
  texDesc42.addressMode[0] = cudaAddressModeClamp;
  texDesc42.addressMode[1] = cudaAddressModeClamp;
  texDesc42.addressMode[2] = cudaAddressModeClamp;
  texDesc42.filterMode = cudaFilterModePoint;
  cudaCreateTextureObject(&tex42, &res42, &texDesc42, NULL);

  // CHECK: cl::sycl::uint2 *d_data21;
  // CHECK-NEXT: dpct::dpct_malloc(&d_data21, sizeof(cl::sycl::uint2) * 32);
  // CHECK-NEXT: dpct::dpct_image_channel desc21 = dpct::create_image_channel(32, 32, 0, 0, dpct::channel_unsigned);
  // CHECK-NEXT: dpct::dpct_image_base_p tex21;
  // CHECK-NEXT: dpct::dpct_image_data res21;
  // CHECK-NEXT: dpct::dpct_image_info texDesc21;
  // CHECK-NEXT: res21.type = dpct::data_linear;
  // CHECK-NEXT: res21.data.linear.data = d_data21;
  // CHECK-NEXT: res21.data.linear.size = sizeof(cl::sycl::uint2) * 32;
  // CHECK-NEXT: res21.data.linear.chn = desc21;
  // CHECK-NEXT: texDesc21.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: texDesc21.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: texDesc21.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: texDesc21.filter_mode() = cl::sycl::filtering_mode::linear;
  // CHECK-NEXT: dpct::dpct_create_image(&tex21, &res21, &texDesc21);
  uint2 *d_data21;
  cudaMalloc(&d_data21, sizeof(uint2) * 32);
  cudaChannelFormatDesc desc21 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
  cudaTextureObject_t tex21;
  cudaResourceDesc res21;
  cudaTextureDesc texDesc21;
  res21.resType = cudaResourceTypeLinear;
  res21.res.linear.devPtr = d_data21;
  res21.res.linear.sizeInBytes = sizeof(uint2) * 32;
  res21.res.linear.desc = desc21;
  texDesc21.addressMode[0] = cudaAddressModeClamp;
  texDesc21.addressMode[1] = cudaAddressModeClamp;
  texDesc21.addressMode[2] = cudaAddressModeClamp;
  texDesc21.filterMode = cudaFilterModeLinear;
  cudaCreateTextureObject(&tex21, &res21, &texDesc21, NULL);

  // CHECK:   {
  // CHECK-NEXT: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:     auto tex21_acc = static_cast<dpct::dpct_image<cl::sycl::uint2, 1> *>(tex21)->get_access(cgh);
  // CHECK-NEXT:     auto tex42_acc = static_cast<dpct::dpct_image<cl::sycl::float4, 2> *>(tex42)->get_access(cgh);
  // CHECK-NEXT:     auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:     auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:       [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel(tex21_acc, tex42_acc);
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  // CHECK-NEXT: }
  kernel<<<1, 1>>>(tex21, tex42);

  // CHECK: dpct::dpct_free(tex42);
  // CHECK-NEXT: dpct::dpct_free(tex21);
  cudaDestroyTextureObject(tex42);
  cudaDestroyTextureObject(tex21);

  // CHECK: dpct::dpct_free(a42);
  cudaFreeArray(a42);

  // CHECK: dpct::dpct_free(d_data42);
  // CHECK-NEXT: dpct::dpct_free(d_data21);
  cudaFree(d_data42);
  cudaFree(d_data21);
}
