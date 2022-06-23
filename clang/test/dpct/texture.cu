// RUN: dpct --format-range=none --usm-level=none -out-root %T/texture %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14 -fno-delayed-template-parsing
// RUN: FileCheck --input-file %T/texture/texture.dp.cpp --match-full-lines %s

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
// CHECK: dpct::image_wrapper<int, 4> tex_no_ref;
static texture<int, 4> tex_no_ref;
// CHECK: dpct::image_wrapper<sycl::float4, 2> tex42;
static texture<float4, 2> tex42;
// CHECK: /*
// CHECK: DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
// CHECK: */
// CHECK: dpct::image_wrapper<sycl::uint2, 1> tex21;
static texture<uint2, 1> tex21;
/// TODO: Expect to support 3D array in future.
// TODO-CHECK: dpct::image<int, 3> tex13;
// static texture<int, 3> tex13;

// CHECK: void device01(dpct::image_accessor_ext<sycl::uint2, 1> tex21) {
// CHECK-NEXT: sycl::uint2 u21 = tex21.read(1.0f);
// CHECK-NEXT: sycl::uint2 u21_fetch = tex21.read(1);
__device__ void device01() {
  uint2 u21 = tex1D(tex21, 1.0f);
  uint2 u21_fetch = tex1Dfetch(tex21, 1);
}

// CHECK: void kernel(dpct::image_accessor_ext<sycl::float4, 2> tex42,
// CHECK-NEXT:        dpct::image_accessor_ext<sycl::uint2, 1> tex21) {
// CHECK-NEXT: device01(tex21);
// CHECK-NEXT: sycl::float4 f42 = tex42.read(1.0f, 1.0f);
/// Texture accessors should be passed down to __global__/__device__ function if used.
__global__ void kernel() {
  device01();
  float4 f42 = tex2D(tex42, 1.0f, 1.0f);
}

int main() {

  // CHECK: dpct::image_channel halfChn = dpct::image_channel::create<sycl::half>();
  cudaChannelFormatDesc halfChn = cudaCreateChannelDescHalf();

  // CHECK: dpct::image_channel float4Chn = dpct::image_channel::create<sycl::float4>();
  cudaChannelFormatDesc float4Chn = cudaCreateChannelDesc<float4>();

  auto tex42_ptr = &tex42;

  // CHECK: dpct::image_matrix **a_ptr = new dpct::image_matrix_p;
  // CHECK-NEXT: sycl::float4 *d_test;
  // CHECK-NEXT: d_test = (sycl::float4 *)dpct::dpct_malloc(sizeof(sycl::float4) * 32 * 32);
  // CHECK-NEXT: *a_ptr = new dpct::image_matrix(tex42.get_channel(), sycl::range<2>(32, 32));
  // CHECK-NEXT: dpct::dpct_memcpy((*a_ptr)->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(d_test, 32 * 32 * sizeof(sycl::float4), 32 * 32 * sizeof(sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(32 * 32 * sizeof(sycl::float4), 1, 1));
  // CHECK-NEXT: delete *a_ptr;
  // CHECK-NEXT: dpct::dpct_free(d_test);
  // CHECK-NEXT: delete a_ptr;

  cudaArray **a_ptr = new cudaArray_t;
  float4 *d_test;
  cudaMalloc(&d_test, sizeof(float4) * 32 * 32);
  cudaMallocArray(a_ptr, &tex42.channelDesc, 32, 32);
  cudaMemcpyToArray(*a_ptr, 0, 0, d_test, 32 * 32 * sizeof(float4), cudaMemcpyDeviceToDevice);
  cudaFreeArray(*a_ptr);
  cudaFree(d_test);
  delete a_ptr;

  // CHECK: sycl::float4 *d_data42;
  // CHECK-NEXT: dpct::image_matrix_p a42;
  // CHECK-NEXT: d_data42 = (sycl::float4 *)dpct::dpct_malloc(sizeof(sycl::float4) * 32 * 32);
  // CHECK-NEXT: a42 = new dpct::image_matrix(tex42.get_channel(), sycl::range<2>(32, 32));
  // CHECK-NEXT: dpct::dpct_memcpy(a42->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(d_data42, 32 * 32 * sizeof(sycl::float4), 32 * 32 * sizeof(sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(32 * 32 * sizeof(sycl::float4), 1, 1));
  // CHECK-NEXT: tex42.set(sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT: tex42.set(sycl::filtering_mode::nearest);
  // CHECK-NEXT: tex42_ptr->attach(d_data42, 32 * sizeof(sycl::float4), 32, 32 * sizeof(sycl::float4), tex42.get_channel());
  // CHECK-NEXT: tex42.attach(d_data42, 32 * sizeof(sycl::float4), 32, 32 * sizeof(sycl::float4));
  // CHECK-NEXT: tex42.attach(d_data42, 32 * sizeof(sycl::float4), 32, 32 * sizeof(sycl::float4), tex42.get_channel());
  // CHECK-NEXT: tex42.attach(d_data42, 32 * sizeof(sycl::float4), 32, 32 * sizeof(sycl::float4), tex42.get_channel());
  // CHECK-NEXT: tex42.attach(a42);
  // CHECK-NEXT: tex42.attach(a42, tex42.get_channel());
  // CHECK-NEXT: tex42.attach(a42, tex42.get_channel());
  // CHECK-NEXT: tex42.set_channel(dpct::image_channel(32, 32, 32, 32, dpct::image_channel_data_type::fp));
  float4 *d_data42;
  cudaArray_t a42;
  cudaMalloc(&d_data42, sizeof(float4) * 32 * 32);
  cudaMallocArray(&a42, &tex42.channelDesc, 32, 32);
  cudaMemcpyToArray(a42, 0, 0, d_data42, 32 * 32 * sizeof(float4), cudaMemcpyDeviceToDevice);
  tex42.addressMode[0] = cudaAddressModeClamp;
  tex42.addressMode[1] = cudaAddressModeClamp;
  tex42.addressMode[2] = cudaAddressModeClamp;
  tex42.filterMode = cudaFilterModePoint;
  cudaBindTexture2D(0, tex42_ptr, d_data42, &tex42.channelDesc, 32 * sizeof(float4), 32, 32 * sizeof(float4));
  cudaBindTexture2D(0, tex42, d_data42, 32 * sizeof(float4), 32, 32 * sizeof(float4));
  cudaBindTexture2D(0, tex42, d_data42, tex42.channelDesc, 32 * sizeof(float4), 32, 32 * sizeof(float4));
  cudaBindTexture2D(0, &tex42, d_data42, &tex42.channelDesc, 32 * sizeof(float4), 32, 32 * sizeof(float4));
  cudaBindTextureToArray(tex42, a42);
  cudaBindTextureToArray(&tex42, a42, &tex42.channelDesc);
  cudaBindTextureToArray(tex42, a42, tex42.channelDesc);
  tex42.channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

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

  // CHECK: desc21 = a42->get_channel();
  cudaGetChannelDesc(&desc21, a42);

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
  // CHECK-NEXT:               kernel(dpct::image_accessor_ext<sycl::float4, 2>(tex42_smpl, tex42_acc), dpct::image_accessor_ext<sycl::uint2, 1>(tex21_smpl, tex21_acc));
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

  // CHECK:  dpct::image_wrapper<unsigned int, 1> tex_tmp;
  // CHECK-NEXT:   tex_tmp.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest, sycl::coordinate_normalization_mode::unnormalized);
  // CHECK-NEXT:   sycl::addressing_mode addr = tex_tmp.get_addressing_mode();
  // CHECK-NEXT:   sycl::filtering_mode filter = tex_tmp.get_filtering_mode();
  // CHECK-NEXT:   int normalized = tex_tmp.is_coordinate_normalized();
  // CHECK-NEXT:   unsigned chn_x = tex_tmp.get_channel_size();
  // CHECK-NEXT:   dpct::image_channel_data_type kind = tex_tmp.get_channel_data_type();
  // CHECK-NEXT:   dpct::image_channel chn = tex_tmp.get_channel();
  // CHECK-NEXT:   tex_tmp.set_channel_size(3, chn_x);
  // CHECK-NEXT:   tex_tmp.set_channel_data_type(dpct::image_channel_data_type::fp);
  texture<unsigned int, 1, cudaReadModeElementType> tex_tmp;
  tex_tmp.normalized = false;
  tex_tmp.addressMode[0] = cudaAddressModeClamp;
  tex_tmp.filterMode = cudaFilterModePoint;
  cudaTextureAddressMode addr = tex_tmp.addressMode[0];
  cudaTextureFilterMode filter = tex_tmp.filterMode;
  int normalized = tex_tmp.normalized;
  unsigned chn_x = tex_tmp.channelDesc.x;
  cudaChannelFormatKind kind = tex_tmp.channelDesc.f;
  cudaChannelFormatDesc chn = tex_tmp.channelDesc;
  tex_tmp.channelDesc.z = chn_x;
  tex_tmp.channelDesc.f = cudaChannelFormatKindFloat;

  // CHECK:   tex_tmp.set_coordinate_normalization_mode(normalized);
  // CHECK-NEXT:   tex_tmp.set(sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT:   if (true) {
  // CHECK-NEXT:     tex_tmp.set(sycl::addressing_mode::repeat, sycl::filtering_mode::nearest, sycl::coordinate_normalization_mode::normalized);
  // CHECK-NEXT:   }
  // CHECK-NEXT:   tex_tmp.set(sycl::filtering_mode::linear);
  // CHECK-NEXT:   addr = tex_tmp.get_addressing_mode();
  // CHECK-NEXT:   tex_tmp.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest, sycl::coordinate_normalization_mode::normalized);
  tex_tmp.normalized = normalized;
  tex_tmp.addressMode[0] = cudaAddressModeClamp;
  if (true) {
    tex_tmp.filterMode = cudaFilterModePoint;
    tex_tmp.addressMode[1] = cudaAddressModeWrap;
    tex_tmp.addressMode[2] = cudaAddressModeWrap;
    tex_tmp.normalized = 1;
  }
  tex_tmp.filterMode = cudaFilterModeLinear;
  addr = tex_tmp.addressMode[0];
  tex_tmp.filterMode = cudaFilterModePoint;
  tex_tmp.addressMode[2] = cudaAddressModeClamp;
  tex_tmp.normalized = 1;


  // Test IsAssigned
  {
    cudaChannelFormatDesc desc42;
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

// once when dpct parses device function foo(),
// dpct parser will emit parser error: use of undeclared identifier '__nv_tex_surf_handler',
// the patch is to fix this issue.
__device__ void foo() {
   cudaTextureObject_t foo;
   float *ret;
   tex1D(ret, foo, 1.0);
}

template <class T>
__device__ T fooFilter(float w0x, float w1x, float w2x, float w3x, T c0, T c1,
                       T c2, T c3) {
  T resultVal = (int)(c0 * w0x + c1 * w1x + c2 * w2x + c3 * w3x + 0.5f);
  return resultVal;
}

// CHECK:template <class T, class R>
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
// CHECK-NEXT:*/
// CHECK-NEXT:R foo(dpct::image_accessor_ext<T, 2> texref,
// CHECK-NEXT:                            float x, float y) {
template <class T, class R>
__device__ R foo(const texture<T, 2, cudaReadModeElementType> texref,
                            float x, float y) {
  float px = floor(x - 2) + 1.0f;
  float py = floor(y - 2) + 1.0f;

  float fx = x - px;
  float fy = y - py;
  float w0x = 0;
  float w1x = 0;
  float w2x = 0;
  float w3x = 0;

  float w0y = 0;
  float w1y = 0;
  float w2y = 0;
  float w3y = 0;

  // CHECK:return fooFilter<R>(
  // CHECK-NEXT:    w0x, w1x, w2x, w3x,
  // CHECK-NEXT:    fooFilter<R>(w0y, w1y, w2y, w3y, texref.read(px, py),
  // CHECK-NEXT:                 texref.read(px, py + 1), texref.read(px, py + 2),
  // CHECK-NEXT:                 texref.read(px, py + 3)),
  // CHECK-NEXT:    fooFilter<R>(w0y, w1y, w2y, w3y, texref.read(px + 1, py),
  // CHECK-NEXT:                 texref.read(px + 1, py + 1), texref.read(px + 1, py + 2),
  // CHECK-NEXT:                 texref.read(px + 1, py + 3)),
  // CHECK-NEXT:    fooFilter<R>(w0y, w1y, w2y, w3y, texref.read(px + 2, py),
  // CHECK-NEXT:                 texref.read(px + 2, py + 1), texref.read(px + 2, py + 2),
  // CHECK-NEXT:                 texref.read(px + 2, py + 3)),
  // CHECK-NEXT:    fooFilter<R>(w0y, w1y, w2y, w3y, texref.read(px + 3, py),
  // CHECK-NEXT:                 texref.read(px + 3, py + 1), texref.read(px + 3, py + 2),
  // CHECK-NEXT:                 texref.read(px + 3, py + 3)));
  return fooFilter<R>(
      w0x, w1x, w2x, w3x,
      fooFilter<R>(w0y, w1y, w2y, w3y, tex2D(texref, px, py),
                   tex2D(texref, px, py + 1), tex2D(texref, px, py + 2),
                   tex2D(texref, px, py + 3)),
      fooFilter<R>(w0y, w1y, w2y, w3y, tex2D(texref, px + 1, py),
                   tex2D(texref, px + 1, py + 1), tex2D(texref, px + 1, py + 2),
                   tex2D(texref, px + 1, py + 3)),
      fooFilter<R>(w0y, w1y, w2y, w3y, tex2D(texref, px + 2, py),
                   tex2D(texref, px + 2, py + 1), tex2D(texref, px + 2, py + 2),
                   tex2D(texref, px + 2, py + 3)),
      fooFilter<R>(w0y, w1y, w2y, w3y, tex2D(texref, px + 3, py),
                   tex2D(texref, px + 3, py + 1), tex2D(texref, px + 3, py + 2),
                   tex2D(texref, px + 3, py + 3)));
}

// CHECK:/*
// CHECK-NEXT:DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
// CHECK-NEXT:*/
// CHECK-NEXT:dpct::image_wrapper<unsigned int, 1> tex_tmp;
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1062:{{[0-9]+}}: SYCL Image doesn't support normalized read mode.
// CHECK-NEXT:*/
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
// CHECK-NEXT:*/
// CHECK-NEXT:dpct::image_wrapper<unsigned char, 2> tex;
texture<unsigned int, 1, cudaReadModeElementType> tex_tmp;
texture<unsigned char, 2, cudaReadModeNormalizedFloat> tex;

// CHECK:template <class T, class R>
// CHECK-NEXT:R tex2D_bar(
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1062:{{[0-9]+}}: SYCL Image doesn't support normalized read mode.
// CHECK-NEXT:    */
// CHECK-NEXT:    /*
// CHECK-NEXT:    DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
// CHECK-NEXT:    */
// CHECK-NEXT:    dpct::image_accessor_ext<T, 2> tex, float x, float y) {
// CHECK-NEXT:  float px = floor(x - 2) + 1.0f;
// CHECK-NEXT:  float py = floor(y - 2) + 1.0f;
// CHECK-NEXT:  float fx = x - px;
// CHECK-NEXT:  float fy = y - py;
// CHECK-NEXT:  float w0x = 0; // w0_2(fx);
// CHECK-NEXT:  float w1x = 0; // w1_2(fx);
// CHECK-NEXT:  float w2x = 0; // w2_2(fx);
// CHECK-NEXT:  float w3x = 0; // w3_2(fx);
// CHECK-NEXT:  float w0y = 0; // w0_2(fy);
// CHECK-NEXT:  float w1y = 0; // w1_2(fy);
// CHECK-NEXT:  float w2y = 0; // w2_2(fy);
// CHECK-NEXT:  float w3y = 0; // w3_2(fy);
// CHECK-NEXT:  return fooFilter<R>(
// CHECK-NEXT:      w0x, w1x, w2x, w3x,
// CHECK-NEXT:      fooFilter<R>(w0y, w1y, w2y, w3y, tex.read(px, py),
// CHECK-NEXT:                   tex.read(px, py + 1), tex.read(px, py + 2),
// CHECK-NEXT:                   tex.read(px, py + 3)),
// CHECK-NEXT:      fooFilter<R>(w0y, w1y, w2y, w3y, tex.read(px + 1, py),
// CHECK-NEXT:                   tex.read(px + 1, py + 1), tex.read(px + 1, py + 2),
// CHECK-NEXT:                   tex.read(px + 1, py + 3)),
// CHECK-NEXT:      fooFilter<R>(w0y, w1y, w2y, w3y, tex.read(px + 2, py),
// CHECK-NEXT:                   tex.read(px + 2, py + 1), tex.read(px + 2, py + 2),
// CHECK-NEXT:                   tex.read(px + 2, py + 3)),
// CHECK-NEXT:      fooFilter<R>(w0y, w1y, w2y, w3y, tex.read(px + 3, py),
// CHECK-NEXT:                   tex.read(px + 3, py + 1), tex.read(px + 3, py + 2),
// CHECK-NEXT:                   tex.read(px + 3, py + 3)));
// CHECK-NEXT:}
template <class T, class R>
__device__ R tex2D_bar(
    const texture<T, 2, cudaReadModeNormalizedFloat> tex, float x, float y) {
  float px = floor(x - 2) + 1.0f;
  float py = floor(y - 2) + 1.0f;
  float fx = x - px;
  float fy = y - py;
  float w0x = 0; // w0_2(fx);
  float w1x = 0; // w1_2(fx);
  float w2x = 0; // w2_2(fx);
  float w3x = 0; // w3_2(fx);
  float w0y = 0; // w0_2(fy);
  float w1y = 0; // w1_2(fy);
  float w2y = 0; // w2_2(fy);
  float w3y = 0; // w3_2(fy);
  return fooFilter<R>(
      w0x, w1x, w2x, w3x,
      fooFilter<R>(w0y, w1y, w2y, w3y, tex2D(tex, px, py),
                   tex2D(tex, px, py + 1), tex2D(tex, px, py + 2),
                   tex2D(tex, px, py + 3)),
      fooFilter<R>(w0y, w1y, w2y, w3y, tex2D(tex, px + 1, py),
                   tex2D(tex, px + 1, py + 1), tex2D(tex, px + 1, py + 2),
                   tex2D(tex, px + 1, py + 3)),
      fooFilter<R>(w0y, w1y, w2y, w3y, tex2D(tex, px + 2, py),
                   tex2D(tex, px + 2, py + 1), tex2D(tex, px + 2, py + 2),
                   tex2D(tex, px + 2, py + 3)),
      fooFilter<R>(w0y, w1y, w2y, w3y, tex2D(tex, px + 3, py),
                   tex2D(tex, px + 3, py + 1), tex2D(tex, px + 3, py + 2),
                   tex2D(tex, px + 3, py + 3)));
}

// CHECK:void test_call(sycl::uchar4 *d_output,
// CHECK-NEXT:                          unsigned int srcImgWidth,
// CHECK-NEXT:                          unsigned int srcImgHeight,
// CHECK-NEXT:                          float inverseOfScale, float tx,
// CHECK-NEXT:                          float ty, sycl::nd_item<3> item_ct1,
// CHECK-NEXT:                          dpct::image_accessor_ext<unsigned char, 2> tex) {
// CHECK-NEXT:  unsigned int x = sycl::mul24((unsigned int)item_ct1.get_group(2), (unsigned int)item_ct1.get_local_range(2)) + item_ct1.get_local_id(2);
// CHECK-NEXT:  unsigned int y = sycl::mul24((unsigned int)item_ct1.get_group(1), (unsigned int)item_ct1.get_local_range(1)) + item_ct1.get_local_id(1);
// CHECK-NEXT:  unsigned int i = sycl::mul24(y, (unsigned int)(srcImgWidth / inverseOfScale)) + x; // mabinbin
// CHECK-NEXT:  float u = x * inverseOfScale + tx; // mabinbin
// CHECK-NEXT:  float v = y * inverseOfScale + ty; // mabinbin
// CHECK-NEXT:  if ((x < srcImgWidth / inverseOfScale) &&
// CHECK-NEXT:      (y < srcImgHeight / inverseOfScale)) {
// CHECK-NEXT:    float c = tex2D_bar<unsigned char, float>(tex, u, v);
// CHECK-NEXT:  }
// CHECK-NEXT:}
__global__ void test_call(uchar4 *d_output,
                          unsigned int srcImgWidth,
                          unsigned int srcImgHeight,
                          float inverseOfScale, float tx,
                          float ty) {
  unsigned int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int i = __umul24(y, srcImgWidth / inverseOfScale) + x; // mabinbin
  float u = x * inverseOfScale + tx; // mabinbin
  float v = y * inverseOfScale + ty; // mabinbin
  if ((x < srcImgWidth / inverseOfScale) &&
      (y < srcImgHeight / inverseOfScale)) {
    float c = tex2D_bar<unsigned char, float>(tex, u, v);
  }
}


