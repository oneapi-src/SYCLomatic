// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/texture.sycl.cpp --match-full-lines %s

// CHECK: syclct::syclct_texture<cl::sycl::float4, 2> tex42;
static texture<float4, 2> tex42;
// CHECK: syclct::syclct_texture<cl::sycl::uint2, 1> tex21;
static texture<uint2, 1> tex21;
/// TODO: Expect to support 3D array in future.
// TODO-CHECK: syclct::syclct_texture<int, 3> tex13;
// static texture<int, 3> tex13;

// CHECK: void device01(syclct::syclct_texture_accessor<cl::sycl::uint2, 1> tex21) {
// CHECK-NEXT: cl::sycl::uint2 u21 = syclct::syclct_read_texture(tex21, 1.0f);
// CHECK-NEXT: cl::sycl::uint2 u21_fetch = syclct::syclct_read_texture(tex21, 1);
__device__ void device01() {
  uint2 u21 = tex1D(tex21, 1.0f);
  uint2 u21_fetch = tex1Dfetch(tex21, 1);
}

// CHECK: void kernel(syclct::syclct_texture_accessor<cl::sycl::float4, 2> tex42, syclct::syclct_texture_accessor<cl::sycl::uint2, 1> tex21) {
// CHECK-NEXT: device01(tex21);
// CHECK-NEXT: cl::sycl::float4 f42 = syclct::syclct_read_texture(tex42, 1.0f, 1.0f);
/// Texture accessors should be passed down to __global__/__device__ function if used.
__global__ void kernel() {
  device01();
  float4 f42 = tex2D(tex42, 1.0f, 1.0f);
}

int main() {

  // CHECK: syclct::syclct_array a42;
  // CHECK-NEXT: cl::sycl::float4 *d_data42;
  // CHECK-NEXT: syclct::sycl_malloc(&d_data42, sizeof(cl::sycl::float4) * 32 * 32);
  // CHECK-NEXT: syclct::syclct_channel_desc desc42 = syclct::create_channel_desc(32, 32, 32, 32, syclct::channel_float);
  // CHECK-NEXT: syclct::syclct_malloc_array(&a42, &desc42, 32, 32);
  // CHECK-NEXT: syclct::syclct_memcpy_to_array(a42, d_data42);
  // CHECK-NEXT: tex42.set_addr_mode( cl::sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT: tex42.set_addr_mode( cl::sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT: tex42.set_addr_mode( cl::sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT: tex42.set_filter_mode( cl::sycl::filtering_mode::nearest);
  // CHECK-NEXT: syclct::syclct_bind_texture(tex42, a42);
  cudaArray_t a42;
  float4 *d_data42;
  cudaMalloc(&d_data42, sizeof(float4) * 32 * 32);
  cudaChannelFormatDesc desc42 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaMallocArray(&a42, &desc42, 32, 32);
  cudaMemcpyToArray(a42, 0, 0, d_data42, 32 * 32, cudaMemcpyDeviceToDevice);
  tex42.addressMode[0] = cudaAddressModeClamp;
  tex42.addressMode[1] = cudaAddressModeClamp;
  tex42.addressMode[2] = cudaAddressModeClamp;
  tex42.filterMode = cudaFilterModePoint;
  cudaBindTextureToArray(tex42, a42, desc42);

  // CHECK: syclct::syclct_array a21;
  // CHECK-NEXT: cl::sycl::uint2 *d_data21;
  // CHECK-NEXT: syclct::sycl_malloc(&d_data21, sizeof(cl::sycl::uint2) * 32);
  // CHECK-NEXT: syclct::syclct_channel_desc desc21 = syclct::create_channel_desc(32, 32, 0, 0, syclct::channel_unsigned);
  // CHECK-NEXT: syclct::syclct_malloc_array(&a21, &desc21, 32, 0);
  // CHECK-NEXT: syclct::syclct_memcpy_to_array(a21, d_data21);
  // CHECK-NEXT: tex21.set_addr_mode( cl::sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT: tex21.set_addr_mode( cl::sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT: tex21.set_addr_mode( cl::sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT: tex21.set_filter_mode( cl::sycl::filtering_mode::linear);
  // CHECK-NEXT: syclct::syclct_bind_texture(tex21, a21);
  cudaArray *a21;
  uint2 *d_data21;
  cudaMalloc(&d_data21, sizeof(uint2) * 32);
  cudaChannelFormatDesc desc21 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
  cudaMallocArray(&a21, &desc21, 32);
  cudaMemcpyToArray(a21, 0, 0, d_data21, 32, cudaMemcpyDeviceToDevice);
  tex21.addressMode[0] = cudaAddressModeClamp;
  tex21.addressMode[1] = cudaAddressModeClamp;
  tex21.addressMode[2] = cudaAddressModeClamp;
  tex21.filterMode = cudaFilterModeLinear;
  cudaBindTextureToArray(tex21, a21, desc21);

  // CHECK:   {
  // CHECK-NEXT:   syclct::get_default_queue().submit(
  // CHECK-NEXT:       [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:         auto tex42_acc = tex42.get_access(cgh);
  // CHECK-NEXT:         auto tex21_acc = tex21.get_access(cgh);
  // CHECK-NEXT:         cgh.parallel_for<syclct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:             cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:             [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               kernel(tex42_acc, tex21_acc);
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  // CHECK-NEXT: }
  kernel<<<1, 1>>>();

  // CHECK: syclct::syclct_unbind_texture(tex42);
  // CHECK-NEXT: syclct::syclct_unbind_texture(tex21);
  cudaUnbindTexture(tex42);
  cudaUnbindTexture(tex21);

  // CHECK: syclct::syclct_free_array(a42);
  // CHECK-NEXT: syclct::syclct_free_array(a21);
  cudaFreeArray(a42);
  cudaFreeArray(a21);
 
  // CHECK: syclct::sycl_free(d_data42);
  // CHECK-NEXT: syclct::sycl_free(d_data21);
  cudaFree(d_data42);
  cudaFree(d_data21);
}
