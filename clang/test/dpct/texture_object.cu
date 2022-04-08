// RUN: c2s --format-range=none --usm-level=none -out-root %T/texture_object %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14 -fno-delayed-template-parsing
// RUN: FileCheck --input-file %T/texture_object/texture_object.dp.cpp --match-full-lines %s

#include <stdio.h>

#define cudaCheck(stmt) do {                         \
  cudaError_t err = stmt;                            \
  if (err != cudaSuccess) {                          \
    char msg[256];                                   \
    sprintf(msg, "%s in file %s, function %s, line %d\n", #stmt,__FILE__,__FUNCTION__,__LINE__); \
  }                                                  \
} while(0)

struct texObjWrapper {
  // CHECK: c2s::image_wrapper_base_p tex;
  cudaTextureObject_t tex;
};

void func(int i) {}

template <typename T>
void funcT(T t) {}

// CHECK: DPCT1050:{{[0-9]+}}: The template argument of  the image_accessor_ext could not be deduced. You need to update this code.
// CHECK: void gather_force(const c2s::image_accessor_ext<c2s_placeholder/*Fix the type manually*/, 1> gridTexObj){}
__global__ void gather_force(const cudaTextureObject_t gridTexObj){}

// CHECK: void gather_force(const c2s::image_wrapper_base_p gridTexObj, sycl::queue *stream) {
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1050:{{[0-9]+}}: The template argument of the image_accessor_ext could not be deduced. You need to update this code.
// CHECK-NEXT:  */
// CHECK-NEXT:  stream->submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      auto gridTexObj_acc = static_cast<c2s::image_wrapper<c2s_placeholder/*Fix the type manually*/, 1> *>(gridTexObj)->get_access(cgh);
// CHECK-EMPTY:
// CHECK-NEXT:      auto gridTexObj_smpl = gridTexObj->get_sampler();
// CHECK-EMPTY:
// CHECK-NEXT:      cgh.parallel_for<c2s_kernel_name<class gather_force_{{[a-f0-9]+}}>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          gather_force(c2s::image_accessor_ext<c2s_placeholder/*Fix the type manually*/, 1>(gridTexObj_smpl, gridTexObj_acc));
// CHECK-NEXT:        });
// CHECK-NEXT:    });
// CHECK-NEXT: }
void gather_force(const cudaTextureObject_t gridTexObj, cudaStream_t stream) {
  gather_force <<< 1, 1, 1, stream >>>(gridTexObj);
}

// CHECK: template <class T> void BindTextureObject(c2s::image_matrix_p &data, c2s::image_wrapper_base_p &tex) {
// CHECK-NEXT:   c2s::image_data res42;
// CHECK-NEXT:   c2s::sampling_info texDesc42;
// CHECK:   res42.set_data(data);
// CHECK-NEXT:   texDesc42.set(sycl::addressing_mode::clamp_to_edge);
// CHECK-NEXT:   texDesc42.set(sycl::filtering_mode::nearest);
// CHECK-NEXT:   data = (c2s::image_matrix_p)res42.get_data_ptr();
// CHECK-NEXT:   tex = c2s::create_image_wrapper(res42, texDesc42);
// CHECK-NEXT: }
template <class T> void BindTextureObject(cudaArray_t &data, cudaTextureObject_t &tex) {
  cudaResourceDesc res42;
  cudaTextureDesc texDesc42;
  res42.resType = cudaResourceTypeArray;
  res42.res.array.array = data;
  texDesc42.addressMode[0] = cudaAddressModeClamp;
  texDesc42.addressMode[1] = cudaAddressModeClamp;
  texDesc42.addressMode[2] = cudaAddressModeClamp;
  texDesc42.filterMode = cudaFilterModePoint;
  data = res42.res.array.array;
  cudaCreateTextureObject(&tex, &res42, &texDesc42, NULL);
}

// CHECK: void device01(c2s::image_accessor_ext<sycl::uint2, 1> tex21) {
// CHECK-NEXT: sycl::uint2 u21;
// CHECK-NEXT: u21 = tex21.read(0.5f);
// CHECK-NEXT: u21 = tex21.read(1);
__device__ void device01(cudaTextureObject_t tex21) {
  uint2 u21;
  tex1D(&u21, tex21, 0.5f);
  tex1Dfetch(&u21, tex21, 1);
}

// CHECK: void kernel(c2s::image_accessor_ext<sycl::uint2, 1> tex2, c2s::image_accessor_ext<sycl::float4, 2> tex4) {
// CHECK-NEXT: device01(tex2);
// CHECK-NEXT: sycl::float4 f42;
// CHECK-NEXT: f42 = tex4.read(0.5f, 0.5f);
/// Texture accessors should be passed down to __global__/__device__ function if used.
__global__ void kernel(cudaTextureObject_t tex2, cudaTextureObject_t tex4) {
  device01(tex2);
  float4 f42;
  tex2D(&f42, tex4, 0.5f, 0.5f);
}

int main() {

  // CHECK: sycl::float4 *d_data42;
  // CHECK-NEXT: c2s::image_matrix_p a42;
  // CHECK-NEXT: d_data42 = (sycl::float4 *)c2s::c2s_malloc(sizeof(sycl::float4) * 32 * 32);
  // CHECK-NEXT: c2s::image_channel desc42 = c2s::image_channel(32, 32, 32, 32, c2s::image_channel_data_type::fp);
  // CHECK-NEXT: a42 = new c2s::image_matrix(desc42, sycl::range<2>(32, 32));
  // CHECK-NEXT: c2s::c2s_memcpy(a42->to_pitched_data(), sycl::id<3>(0, 0, 0), c2s::pitched_data(d_data42, 32 * 32 * sizeof(sycl::float4), 32 * 32 * sizeof(sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(32 * 32 * sizeof(sycl::float4), 1, 1));
  // CHECK-NEXT: c2s::image_wrapper_base_p tex42;
  // CHECK-NEXT: c2s::image_data res42;
  // CHECK-NEXT: c2s::sampling_info texDesc42;
  // CHECK: res42.set_data(d_data42, sizeof(sycl::float4) * 32, 32, sizeof(sycl::float4) * 32, desc42);
  // CHECK: res42.set_data(a42);
  // CHECK-NEXT: texDesc42.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest, sycl::coordinate_normalization_mode::normalized);
  // CHECK-NEXT: tex42 = c2s::create_image_wrapper(res42, texDesc42);
  float4 *d_data42;
  cudaArray_t a42;
  cudaMalloc(&d_data42, sizeof(float4) * 32 * 32);
  cudaChannelFormatDesc desc42 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaMallocArray(&a42, &desc42, 32, 32);
  cudaMemcpyToArray(a42, 0, 0, d_data42, 32 * 32 * sizeof(float4), cudaMemcpyDeviceToDevice);
  cudaTextureObject_t tex42;
  cudaResourceDesc res42;
  cudaTextureDesc texDesc42;
  res42.resType = cudaResourceTypePitch2D;
  res42.res.pitch2D.devPtr = d_data42;
  res42.res.pitch2D.desc = desc42;
  res42.res.pitch2D.width = sizeof(float4) * 32;
  res42.res.pitch2D.height = 32;
  res42.res.pitch2D.pitchInBytes = sizeof(float4) * 32;
  res42.resType = cudaResourceTypeArray;
  res42.res.array.array = a42;
  texDesc42.addressMode[0] = cudaAddressModeClamp;
  texDesc42.addressMode[1] = cudaAddressModeClamp;
  texDesc42.addressMode[2] = cudaAddressModeClamp;
  texDesc42.filterMode = cudaFilterModePoint;
  texDesc42.normalizedCoords = 1;
  cudaCreateTextureObject(&tex42, &res42, &texDesc42, NULL);

  // CHECK: sycl::uint2 *d_data21;
  // CHECK-NEXT: d_data21 = (sycl::uint2 *)c2s::c2s_malloc(sizeof(sycl::uint2) * 32);
  // CHECK-NEXT: c2s::image_wrapper_base_p tex21;
  // CHECK-NEXT: c2s::image_data res21;
  // CHECK-NEXT: c2s::sampling_info texDesc21;
  // CHECK-NEXT: res21.set_data_type(c2s::image_data_type::linear);
  // CHECK-NEXT: res21.set_data_ptr(d_data21);
  // CHECK-NEXT: res21.set_channel_data_type(c2s::image_channel_data_type::unsigned_int);
  // CHECK-NEXT: res21.set_channel_size(1, sizeof(unsigned)*8); // bits per channel
  // CHECK-NEXT: if (sizeof(sycl::uint2) >= sizeof(unsigned)*2) res21.set_channel_size(2, sizeof(unsigned)*8);
  // CHECK-NEXT: if (sizeof(sycl::uint2) >= sizeof(unsigned)*3) res21.set_channel_size(3, sizeof(unsigned)*8);
  // CHECK-NEXT: if (sizeof(sycl::uint2) >= sizeof(unsigned)*4) res21.set_channel_size(4, sizeof(unsigned)*8);
  // CHECK-NEXT: res21.set_x(32*sizeof(sycl::uint2));
  // CHECK-NEXT: unsigned chnX = res21.get_channel_size();
  // CHECK-NEXT: c2s::image_channel_data_type formatKind = res21.get_channel_data_type();
  // CHECK-NEXT: texDesc21.set(sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT: texDesc21.set(sycl::filtering_mode::linear);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaTextureDesc::readMode is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: texDesc21.readMode = cudaReadModeElementType;
  // CHECK-NEXT: tex21 = c2s::create_image_wrapper(res21, texDesc21);
  uint2 *d_data21;
  cudaMalloc(&d_data21, sizeof(uint2) * 32);
  cudaTextureObject_t tex21;
  cudaResourceDesc res21;
  cudaTextureDesc texDesc21;
  res21.resType = cudaResourceTypeLinear;
  res21.res.linear.devPtr = d_data21;
  res21.res.linear.desc.f = cudaChannelFormatKindUnsigned;
  res21.res.linear.desc.x = sizeof(unsigned)*8; // bits per channel
  if (sizeof(uint2) >= sizeof(unsigned)*2) res21.res.linear.desc.y = sizeof(unsigned)*8;
  if (sizeof(uint2) >= sizeof(unsigned)*3) res21.res.linear.desc.z = sizeof(unsigned)*8;
  if (sizeof(uint2) >= sizeof(unsigned)*4) res21.res.linear.desc.w = sizeof(unsigned)*8;
  res21.res.linear.sizeInBytes = 32*sizeof(uint2);
  unsigned chnX = res21.res.linear.desc.x;
  cudaChannelFormatKind formatKind = res21.res.linear.desc.f;
  texDesc21.addressMode[0] = cudaAddressModeClamp;
  texDesc21.addressMode[1] = cudaAddressModeClamp;
  texDesc21.addressMode[2] = cudaAddressModeClamp;
  texDesc21.filterMode = cudaFilterModeLinear;
  texDesc21.readMode = cudaReadModeElementType;
  cudaCreateTextureObject(&tex21, &res21, &texDesc21, NULL);

  // CHECK: c2s::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     auto tex21_acc = static_cast<c2s::image_wrapper<sycl::uint2, 1> *>(tex21)->get_access(cgh);
  // CHECK-NEXT:     auto tex42_acc = static_cast<c2s::image_wrapper<sycl::float4, 2> *>(tex42)->get_access(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     auto tex21_smpl = tex21->get_sampler();
  // CHECK-NEXT:     auto tex42_smpl = tex42->get_sampler();
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<c2s_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel(c2s::image_accessor_ext<sycl::uint2, 1>(tex21_smpl, tex21_acc), c2s::image_accessor_ext<sycl::float4, 2>(tex42_smpl, tex42_acc));
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel<<<1, 1>>>(tex21, tex42);
  
  // CHECK: texDesc42 = tex42->get_sampling_info();
  // CHECK-NEXT: res42 = tex42->get_data();
  cudaGetTextureObjectTextureDesc(&texDesc42, tex42);
  cudaGetTextureObjectResourceDesc(&res42, tex42);

  // CHECK: delete tex42;
  // CHECK-NEXT: delete tex21;
  cudaDestroyTextureObject(tex42);
  cudaDestroyTextureObject(tex21);

  // CHECK: delete a42;
  cudaFreeArray(a42);

  // CHECK: c2s::c2s_free(d_data42);
  // CHECK-NEXT: c2s::c2s_free(d_data21);
  cudaFree(d_data42);
  cudaFree(d_data21);

  // Test IsAssigned
  {
    int errorCode;
    // CHECK: errorCode = (tex21 = c2s::create_image_wrapper(res21, texDesc21), 0);
    errorCode = cudaCreateTextureObject(&tex21, &res21, &texDesc21, NULL);
    // CHECK: cudaCheck((tex21 = c2s::create_image_wrapper(res21, texDesc21), 0));
    cudaCheck(cudaCreateTextureObject(&tex21, &res21, &texDesc21, NULL));
    // CHECK: func((tex21 = c2s::create_image_wrapper(res21, texDesc21), 0));
    func(cudaCreateTextureObject(&tex21, &res21, &texDesc21, NULL));
    // CHECK: funcT((tex21 = c2s::create_image_wrapper(res21, texDesc21), 0));
    funcT(cudaCreateTextureObject(&tex21, &res21, &texDesc21, NULL));

    // CHECK: errorCode = (delete tex21, 0);
    errorCode = cudaDestroyTextureObject(tex21);
    // CHECK: cudaCheck((delete tex21, 0));
    cudaCheck(cudaDestroyTextureObject(tex21));
    // CHECK: func((delete tex21, 0));
    func(cudaDestroyTextureObject(tex21));
    // CHECK: funcT((delete tex21, 0));
    funcT(cudaDestroyTextureObject(tex21));
  }
}

void foo(){
  cudaArray_t a42;
  cudaResourceDesc res42;
  //CHECK: res42.set_data(a42);
  res42.resType = cudaResourceTypeArray;
  res42.res.array.array = a42;

  float4 *d_data42;
  cudaChannelFormatDesc desc42 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  //CHECK: res42.set_data(d_data42, sizeof(sycl::float4) * 32, 32, sizeof(sycl::float4) * 32, desc42);
  res42.resType = cudaResourceTypePitch2D;
  res42.res.pitch2D.devPtr = d_data42;
  res42.res.pitch2D.desc = desc42;
  res42.res.pitch2D.width = sizeof(float4) * 32;
  res42.res.pitch2D.height = 32;
  res42.res.pitch2D.pitchInBytes = sizeof(float4) * 32;

  uint2 *d_data21;
  //CHECK: res42.set_data(d_data21, sizeof(sycl::float4) * 32, desc42);
  res42.resType = cudaResourceTypeLinear;
  res42.res.linear.devPtr = d_data21;
  res42.res.linear.sizeInBytes = sizeof(float4) * 32;
  res42.res.linear.desc = desc42;

  // CHECK:  c2s::sampling_info tex_tmp;
  // CHECK-NEXT:   tex_tmp.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest, sycl::coordinate_normalization_mode::unnormalized);
  // CHECK-NEXT:   sycl::addressing_mode addr = tex_tmp.get_addressing_mode();
  // CHECK-NEXT:   sycl::filtering_mode filter = tex_tmp.get_filtering_mode();
  // CHECK-NEXT:   int normalized = tex_tmp.is_coordinate_normalized();
  cudaTextureDesc tex_tmp;
  tex_tmp.normalizedCoords = false;
  tex_tmp.addressMode[0] = cudaAddressModeClamp;
  tex_tmp.filterMode = cudaFilterModePoint;
  cudaTextureAddressMode addr = tex_tmp.addressMode[0];
  cudaTextureFilterMode filter = tex_tmp.filterMode;
  int normalized = tex_tmp.normalizedCoords;

  // CHECK:   tex_tmp.set_coordinate_normalization_mode(normalized);
  // CHECK-NEXT:   tex_tmp.set(sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT:   if (true) {
  // CHECK-NEXT:     tex_tmp.set(sycl::addressing_mode::repeat, sycl::filtering_mode::nearest, sycl::coordinate_normalization_mode::normalized);
  // CHECK-NEXT:   }
  // CHECK-NEXT:   tex_tmp.set(sycl::filtering_mode::linear);
  // CHECK-NEXT:   addr = tex_tmp.get_addressing_mode();
  // CHECK-NEXT:   tex_tmp.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest, sycl::coordinate_normalization_mode::normalized);
  tex_tmp.normalizedCoords = normalized;
  tex_tmp.addressMode[0] = cudaAddressModeClamp;
  if (true) {
    tex_tmp.filterMode = cudaFilterModePoint;
    tex_tmp.addressMode[1] = cudaAddressModeWrap;
    tex_tmp.addressMode[2] = cudaAddressModeWrap;
    tex_tmp.normalizedCoords = 1;
  }
  tex_tmp.filterMode = cudaFilterModeLinear;
  addr = tex_tmp.addressMode[0];
  tex_tmp.filterMode = cudaFilterModePoint;
  tex_tmp.addressMode[2] = cudaAddressModeClamp;
  tex_tmp.normalizedCoords = 1;
}

