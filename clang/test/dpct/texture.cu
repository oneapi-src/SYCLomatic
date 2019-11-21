// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture.dp.cpp --match-full-lines %s

// CHECK: dpct::image<cl::sycl::float4, 2> tex42;
static texture<float4, 2> tex42;
// CHECK: dpct::image<cl::sycl::uint2, 1> tex21;
static texture<uint2, 1> tex21;
/// TODO: Expect to support 3D array in future.
// TODO-CHECK: dpct::image<int, 3> tex13;
// static texture<int, 3> tex13;

// CHECK: void device01(dpct::image_accessor<cl::sycl::uint2, 1> tex21) {
// CHECK-NEXT: cl::sycl::uint2 u21 = dpct::read_image(tex21, 1.0f);
// CHECK-NEXT: cl::sycl::uint2 u21_fetch = dpct::read_image(tex21, 1);
__device__ void device01() {
  uint2 u21 = tex1D(tex21, 1.0f);
  uint2 u21_fetch = tex1Dfetch(tex21, 1);
}

// CHECK: void kernel(dpct::image_accessor<cl::sycl::float4, 2> tex42, dpct::image_accessor<cl::sycl::uint2, 1> tex21) {
// CHECK-NEXT: device01(tex21);
// CHECK-NEXT: cl::sycl::float4 f42 = dpct::read_image(tex42, 1.0f, 1.0f);
/// Texture accessors should be passed down to __global__/__device__ function if used.
__global__ void kernel() {
  device01();
  float4 f42 = tex2D(tex42, 1.0f, 1.0f);
}

int main() {

  // CHECK: dpct::image_channel halfChn = dpct::create_image_channel<cl::sycl::cl_half>();
  cudaChannelFormatDesc halfChn = cudaCreateChannelDescHalf();

  // CHECK: cl::sycl::float4 *d_data42;
  // CHECK-NEXT: dpct::image_matrix_p a42;
  // CHECK-NEXT: dpct::dpct_malloc(&d_data42, sizeof(cl::sycl::float4) * 32 * 32);
  // CHECK-NEXT: dpct::image_channel desc42 = dpct::create_image_channel(32, 32, 32, 32, dpct::channel_float);
  // CHECK-NEXT: dpct::malloc_matrix(&a42, &desc42, 32, 32);
  // CHECK-NEXT: dpct::memcpy_to_matrix(a42, 0, 0, d_data42, 32 * 32 * sizeof(cl::sycl::float4));
  // CHECK-NEXT: tex42.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex42.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex42.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex42.filter_mode() = cl::sycl::filtering_mode::nearest;
  // CHECK-NEXT: dpct::attach_image(tex42, a42);
  float4 *d_data42;
  cudaArray_t a42;
  cudaMalloc(&d_data42, sizeof(float4) * 32 * 32);
  cudaChannelFormatDesc desc42 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaMallocArray(&a42, &desc42, 32, 32);
  cudaMemcpyToArray(a42, 0, 0, d_data42, 32 * 32 * sizeof(float4), cudaMemcpyDeviceToDevice);
  tex42.addressMode[0] = cudaAddressModeClamp;
  tex42.addressMode[1] = cudaAddressModeClamp;
  tex42.addressMode[2] = cudaAddressModeClamp;
  tex42.filterMode = cudaFilterModePoint;
  cudaBindTextureToArray(tex42, a42, desc42);

  // CHECK: cl::sycl::uint2 *d_data21;
  // CHECK-NEXT: dpct::dpct_malloc(&d_data21, sizeof(cl::sycl::uint2) * 32);
  // CHECK-NEXT: dpct::image_channel desc21 = dpct::create_image_channel(32, 32, 0, 0, dpct::channel_unsigned);
  // CHECK-NEXT: tex21.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex21.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex21.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex21.filter_mode() = cl::sycl::filtering_mode::linear;
  // CHECK-NEXT: dpct::attach_image(tex21, d_data21, desc21, 32 * sizeof(cl::sycl::uint2));
  // CHECK-NEXT: dpct::attach_image(tex21, d_data21, 32 * sizeof(cl::sycl::uint2));
  uint2 *d_data21;
  cudaMalloc(&d_data21, sizeof(uint2) * 32);
  cudaChannelFormatDesc desc21 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
  tex21.addressMode[0] = cudaAddressModeClamp;
  tex21.addressMode[1] = cudaAddressModeClamp;
  tex21.addressMode[2] = cudaAddressModeClamp;
  tex21.filterMode = cudaFilterModeLinear;
  cudaBindTexture(0, tex21, d_data21, desc21, 32 * sizeof(uint2));
  cudaBindTexture(0, tex21, d_data21, 32 * sizeof(uint2));

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:       [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:         auto tex42_acc = tex42.get_access(cgh);
  // CHECK-NEXT:         auto tex21_acc = tex21.get_access(cgh);
  // CHECK-NEXT:         auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:         auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:             [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               kernel(tex42_acc, tex21_acc);
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  kernel<<<1, 1>>>();

  // CHECK: dpct::detach_image(tex42);
  // CHECK-NEXT: dpct::detach_image(&tex21);
  cudaUnbindTexture(tex42);
  cudaUnbindTexture(&tex21);

  // CHECK: dpct::dpct_free(a42);
  cudaFreeArray(a42);

  // CHECK: dpct::dpct_free(d_data42);
  // CHECK-NEXT: dpct::dpct_free(d_data21);
  cudaFree(d_data42);
  cudaFree(d_data21);

  // CHECK:  dpct::image<unsigned int, 1> tex_tmp;
  // CHECK-NEXT:   tex_tmp.coord_normalized() = false;
  // CHECK-NEXT:   tex_tmp.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT:   tex_tmp.filter_mode() = cl::sycl::filtering_mode::nearest;
  texture<unsigned int, 1, cudaReadModeElementType> tex_tmp;
  tex_tmp.normalized = false;
  tex_tmp.addressMode[0] = cudaAddressModeClamp;
  tex_tmp.filterMode = cudaFilterModePoint;
}

// Before the patch for CTST-1078 is merged, when dpct parses device function foo(),
// dpct parser will emit parser error: use of undeclared identifier '__nv_tex_surf_handler',
// the patch is to fix this issue.
__device__ void foo() {
   cudaTextureObject_t foo;
   float *ret;
   tex1D(ret, foo, 1.0);
}
