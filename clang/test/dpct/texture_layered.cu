// RUN: dpct --format-range=none --usm-level=none -out-root %T/texture_layered %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture_layered/texture_layered.dp.cpp --match-full-lines %s

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
// CHECK: /*
// CHECK: DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
// CHECK: */
// CHECK: dpct::image_wrapper<int, 2, true> tex_no_ref;
static texture<int, cudaTextureType1DLayered> tex_no_ref;
// CHECK: dpct::image_wrapper<sycl::float4, 3, true> tex42;
static texture<float4, cudaTextureType2DLayered> tex42;
// CHECK: /*
// CHECK: DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
// CHECK: */
// CHECK: dpct::image_wrapper<sycl::uint2, 2, true> tex21;
static texture<uint2, cudaTextureType1DLayered> tex21;

// CHECK: void device01(sycl::float4 *out, dpct::image_accessor_ext<sycl::uint2, 1, true> tex21) {
// CHECK-NEXT: sycl::uint2 u21 = tex21.read(12, 1.0f);
// CHECK-NEXT: out[0].x() =  u21.x();
__device__ void device01(float4 *out) {
  uint2 u21 = tex1DLayered(tex21, 1.0f, 12);
  out[0].x = u21.x;
}

// CHECK: void kernel(sycl::float4 *out, dpct::image_accessor_ext<sycl::float4, 2, true> tex42,
// CHECK-NEXT:        dpct::image_accessor_ext<sycl::uint2, 1, true> tex21) {
// CHECK-NEXT: device01(out, tex21);
// CHECK-NEXT: out[1] = tex42.read(12, 1.0f, 1.0f);
/// Texture accessors should be passed down to __global__/__device__ function if used.
__global__ void kernel(float4 *out) {
  device01(out);
  out[1] = tex2DLayered(tex42, 1.0f, 1.0f, 12);
}

int main() {

  // CHECK: dpct::image_channel halfChn = dpct::image_channel::create<sycl::half>();
  cudaChannelFormatDesc halfChn = cudaCreateChannelDescHalf();

  // CHECK: sycl::float4 *d_data42;
  // CHECK-NEXT: dpct::image_matrix_p a42;
  // CHECK-NEXT: d_data42 = (sycl::float4 *)dpct::dpct_malloc(sizeof(sycl::float4) * 32 * 32);
  // CHECK-NEXT: dpct::image_channel desc42 = dpct::image_channel(32, 32, 32, 32, dpct::image_channel_data_type::fp);
  // CHECK-NEXT: a42 = new dpct::image_matrix(desc42, sycl::range<2>(32, 32));
  // CHECK-NEXT: dpct::dpct_memcpy(a42->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(d_data42, 32 * 32 * sizeof(sycl::float4), 32 * 32 * sizeof(sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(32 * 32 * sizeof(sycl::float4), 1, 1));
  // CHECK-NEXT: tex42.set(sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT: tex42.set(sycl::filtering_mode::nearest);
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
  cudaBindTextureToArray(tex42, a42, desc42);

  // CHECK: sycl::uint2 *d_data21;
  // CHECK-NEXT: d_data21 = (sycl::uint2 *)dpct::dpct_malloc(sizeof(sycl::uint2) * 32);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::image_channel desc21 = dpct::image_channel(32, 32, 0, 0, dpct::image_channel_data_type::unsigned_int);
  // CHECK-NEXT: tex21.set(sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT: tex21.set(sycl::filtering_mode::linear);
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

  float4 *d;
  cudaMalloc(&d, sizeof(float4) * 4);

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_acc_ct0 = dpct::get_access(d, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         auto tex42_acc = tex42.get_access(cgh);
  // CHECK-NEXT:         auto tex21_acc = tex21.get_access(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         auto tex42_smpl = tex42.get_sampler();
  // CHECK-NEXT:         auto tex21_smpl = tex21.get_sampler();
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               kernel((sycl::float4 *)(&d_acc_ct0[0]), dpct::image_accessor_ext<sycl::float4, 2, true>(tex42_smpl, tex42_acc), dpct::image_accessor_ext<sycl::uint2, 1, true>(tex21_smpl, tex21_acc));
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  kernel<<<1, 1>>>(d);

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

  // CHECK:  dpct::image_wrapper<unsigned int, 1> tex_tmp;
  // CHECK-NEXT:   tex_tmp.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest, sycl::coordinate_normalization_mode::unnormalized);
  texture<unsigned int, 1, cudaReadModeElementType> tex_tmp;
  tex_tmp.normalized = false;
  tex_tmp.addressMode[0] = cudaAddressModeClamp;
  tex_tmp.filterMode = cudaFilterModePoint;

}

