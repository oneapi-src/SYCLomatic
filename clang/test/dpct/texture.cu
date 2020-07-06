// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture.dp.cpp --match-full-lines %s

#include <stdio.h>

#define cudaCheck(stmt) do {                         \
  cudaError_t err = stmt;                            \
  if (err != cudaSuccess) {                          \
    char msg[256];                                   \
    sprintf(msg, "%s in file %s, function %s, line %d\n", #stmt,__FILE__,__FUNCTION__,__LINE__); \
  }                                                  \
} while(0)

void func(int i) {}

template <typename T>
void funcT(T t) {}

// CHECK: dpct::image<int, 4> tex_no_ref;
static texture<int, 4> tex_no_ref;
// CHECK: dpct::image<sycl::float4, 2> tex42;
static texture<float4, 2> tex42;
// CHECK: dpct::image<sycl::uint2, 1> tex21;
static texture<uint2, 1> tex21;
/// TODO: Expect to support 3D array in future.
// TODO-CHECK: dpct::image<int, 3> tex13;
// static texture<int, 3> tex13;

// CHECK: void device01(dpct::image_accessor<sycl::uint2, 1> tex21) {
// CHECK-NEXT: sycl::uint2 u21 = tex21.read(1.0f);
// CHECK-NEXT: sycl::uint2 u21_fetch = tex21.read(1);
__device__ void device01() {
  uint2 u21 = tex1D(tex21, 1.0f);
  uint2 u21_fetch = tex1Dfetch(tex21, 1);
}

// CHECK: void kernel(dpct::image_accessor<sycl::float4, 2> tex42,
// CHECK-NEXT:        dpct::image_accessor<sycl::uint2, 1> tex21) {
// CHECK-NEXT: device01(tex21);
// CHECK-NEXT: sycl::float4 f42 = tex42.read(1.0f, 1.0f);
/// Texture accessors should be passed down to __global__/__device__ function if used.
__global__ void kernel() {
  device01();
  float4 f42 = tex2D(tex42, 1.0f, 1.0f);
}

int main() {

  // CHECK: dpct::image_channel halfChn = dpct::create_image_channel<sycl::cl_half>();
  cudaChannelFormatDesc halfChn = cudaCreateChannelDescHalf();

  // CHECK: dpct::image_channel float4Chn = dpct::create_image_channel<sycl::float4>();
  cudaChannelFormatDesc float4Chn = cudaCreateChannelDesc<float4>();

  auto tex42_ptr = &tex42;
  // CHECK: sycl::float4 *d_data42;
  // CHECK-NEXT: dpct::image_matrix_p a42;
  // CHECK-NEXT: dpct::dpct_malloc(&d_data42, sizeof(sycl::float4) * 32 * 32);
  // CHECK-NEXT: dpct::image_channel desc42 = dpct::create_image_channel(32, 32, 32, 32, dpct::channel_float);
  // CHECK-NEXT: a42 = new dpct::image_matrix(desc42, sycl::range<2>(32, 32));
  // CHECK-NEXT: dpct::dpct_memcpy(a42->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(d_data42, 32 * 32 * sizeof(sycl::float4), 32 * 32 * sizeof(sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(32 * 32 * sizeof(sycl::float4), 1, 1));
  // CHECK-NEXT: tex42.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex42.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex42.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex42.filter_mode() = sycl::filtering_mode::nearest;
  // CHECK-NEXT: tex42_ptr->attach(d_data42, 32 * sizeof(sycl::float4), 32, 32 * sizeof(sycl::float4), desc42);
  // CHECK-NEXT: tex42.attach(d_data42, 32 * sizeof(sycl::float4), 32, 32 * sizeof(sycl::float4));
  // CHECK-NEXT: tex42.attach(d_data42, 32 * sizeof(sycl::float4), 32, 32 * sizeof(sycl::float4), desc42);
  // CHECK-NEXT: tex42.attach(d_data42, 32 * sizeof(sycl::float4), 32, 32 * sizeof(sycl::float4), desc42);
  // CHECK-NEXT: tex42.attach(a42);
  // CHECK-NEXT: tex42.attach(a42, desc42);
  // CHECK-NEXT: tex42.attach(a42, desc42);
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
  cudaBindTexture2D(0, tex42_ptr, d_data42, &desc42, 32 * sizeof(float4), 32, 32 * sizeof(float4));
  cudaBindTexture2D(0, tex42, d_data42, 32 * sizeof(float4), 32, 32 * sizeof(float4));
  cudaBindTexture2D(0, tex42, d_data42, desc42, 32 * sizeof(float4), 32, 32 * sizeof(float4));
  cudaBindTexture2D(0, &tex42, d_data42, &desc42, 32 * sizeof(float4), 32, 32 * sizeof(float4));
  cudaBindTextureToArray(tex42, a42);
  cudaBindTextureToArray(&tex42, a42, &desc42);
  cudaBindTextureToArray(tex42, a42, desc42);

  // CHECK: desc42 = a42->get_channel();
  cudaGetChannelDesc(&desc42, a42);

  // CHECK: sycl::uint2 *d_data21;
  // CHECK-NEXT: dpct::dpct_malloc(&d_data21, sizeof(sycl::uint2) * 32);
  // CHECK-NEXT: dpct::image_channel desc21 = dpct::create_image_channel(32, 32, 0, 0, dpct::channel_unsigned);
  // CHECK-NEXT: tex21.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex21.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex21.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex21.filter_mode() = sycl::filtering_mode::linear;
  // CHECK-NEXT: tex21.attach(d_data21, 32 * sizeof(sycl::uint2), desc21);
  // CHECK-NEXT: tex21.attach(d_data21, 32 * sizeof(sycl::uint2));
  // CHECK-NEXT: tex21.attach(d_data21, 32 * sizeof(sycl::uint2), desc21);
  uint2 *d_data21;
  cudaMalloc(&d_data21, sizeof(uint2) * 32);
  cudaChannelFormatDesc desc21 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
  tex21.addressMode[0] = cudaAddressModeClamp;
  tex21.addressMode[1] = cudaAddressModeClamp;
  tex21.addressMode[2] = cudaAddressModeClamp;
  tex21.filterMode = cudaFilterModeLinear;
  cudaBindTexture(0, tex21, d_data21, desc21, 32 * sizeof(uint2));
  cudaBindTexture(0, tex21, d_data21, 32 * sizeof(uint2));
  cudaBindTexture(0, &tex21, d_data21, &desc21, 32 * sizeof(uint2));

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto tex42_acc = tex42.get_access(cgh);
  // CHECK-NEXT:         auto tex21_acc = tex21.get_access(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         auto tex42_smpl = tex42.get_sampler();
  // CHECK-NEXT:         auto tex21_smpl = tex21.get_sampler();
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               kernel(dpct::image_accessor<sycl::float4, 2>(tex42_smpl, tex42_acc), dpct::image_accessor<sycl::uint2, 1>(tex21_smpl, tex21_acc));
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  kernel<<<1, 1>>>();

  // CHECK: tex42.detach();
  // CHECK-NEXT: tex21.detach();
  cudaUnbindTexture(tex42);
  cudaUnbindTexture(&tex21);

  // CHECK: delete a42;
  cudaFreeArray(a42);

  // CHECK: dpct::dpct_free(d_data42);
  // CHECK-NEXT: dpct::dpct_free(d_data21);
  cudaFree(d_data42);
  cudaFree(d_data21);

  // CHECK:  dpct::image<unsigned int, 1> tex_tmp;
  // CHECK-NEXT:   tex_tmp.coord_normalized() = false;
  // CHECK-NEXT:   tex_tmp.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT:   tex_tmp.filter_mode() = sycl::filtering_mode::nearest;
  texture<unsigned int, 1, cudaReadModeElementType> tex_tmp;
  tex_tmp.normalized = false;
  tex_tmp.addressMode[0] = cudaAddressModeClamp;
  tex_tmp.filterMode = cudaFilterModePoint;

  // Test IsAssigned
  {
    int errorCode;
    // CHECK: errorCode = (tex42.attach(a42, desc42), 0);
    errorCode = cudaBindTextureToArray(tex42, a42, desc42);
    // CHECK: cudaCheck((tex42.attach(a42, desc42), 0));
    cudaCheck(cudaBindTextureToArray(tex42, a42, desc42));
    // CHECK: func((tex42.attach(a42, desc42), 0));
    func(cudaBindTextureToArray(tex42, a42, desc42));
    // CHECK: funcT((tex42.attach(a42, desc42), 0));
    funcT(cudaBindTextureToArray(tex42, a42, desc42));

    // CHECK: errorCode = (tex21.attach(d_data21, 32 * sizeof(sycl::uint2)), 0);
    errorCode = cudaBindTexture(0, tex21, d_data21, 32 * sizeof(uint2));
    // CHECK: cudaCheck((tex21.attach(d_data21, 32 * sizeof(sycl::uint2)), 0));
    cudaCheck(cudaBindTexture(0, tex21, d_data21, 32 * sizeof(uint2)));
    // CHECK: func((tex21.attach(d_data21, 32 * sizeof(sycl::uint2)), 0));
    func(cudaBindTexture(0, tex21, d_data21, 32 * sizeof(uint2)));
    // CHECK: funcT((tex21.attach(d_data21, 32 * sizeof(sycl::uint2)), 0));
    funcT(cudaBindTexture(0, tex21, d_data21, 32 * sizeof(uint2)));

    // CHECK: errorCode = (tex42.detach(), 0);
    errorCode = cudaUnbindTexture(tex42);
    // CHECK: cudaCheck((tex42.detach(), 0));
    cudaCheck(cudaUnbindTexture(tex42));
    // CHECK: func((tex42.detach(), 0));
    func(cudaUnbindTexture(tex42));
    // CHECK: funcT((tex42.detach(), 0));
    funcT(cudaUnbindTexture(tex42));

    // CHECK: errorCode = (tex42.detach(), 0);
    errorCode = cudaUnbindTexture(&tex42);
    // CHECK: cudaCheck((tex42.detach(), 0));
    cudaCheck(cudaUnbindTexture(&tex42));
    // CHECK: func((tex42.detach(), 0));
    func(cudaUnbindTexture(&tex42));
    // CHECK: funcT((tex42.detach(), 0));
    funcT(cudaUnbindTexture(&tex42));

    // CHECK: errorCode = (delete a42, 0);
    errorCode = cudaFreeArray(a42);
    // CHECK: cudaCheck((delete a42, 0));
    cudaCheck(cudaFreeArray(a42));
    // CHECK: func((delete a42, 0));
    func(cudaFreeArray(a42));
    // CHECK: funcT((delete a42, 0));
    funcT(cudaFreeArray(a42));

    // CHECK: errorCode = (a42 = new dpct::image_matrix(desc42, sycl::range<2>(32, 32)), 0);
    errorCode = cudaMallocArray(&a42, &desc42, 32, 32);
    // CHECK: cudaCheck((a42 = new dpct::image_matrix(desc42, sycl::range<2>(32, 32)), 0));
    cudaCheck(cudaMallocArray(&a42, &desc42, 32, 32));
    // CHECK: func((a42 = new dpct::image_matrix(desc42, sycl::range<2>(32, 32)), 0));
    func(cudaMallocArray(&a42, &desc42, 32, 32));
    // CHECK: funcT((a42 = new dpct::image_matrix(desc42, sycl::range<2>(32, 32)), 0));
    funcT(cudaMallocArray(&a42, &desc42, 32, 32));
  }
}

// Before the patch for CTST-1078 is merged, when dpct parses device function foo(),
// dpct parser will emit parser error: use of undeclared identifier '__nv_tex_surf_handler',
// the patch is to fix this issue.
__device__ void foo() {
   cudaTextureObject_t foo;
   float *ret;
   tex1D(ret, foo, 1.0);
}
