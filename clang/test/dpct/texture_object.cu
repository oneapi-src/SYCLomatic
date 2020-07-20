// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14 -fno-delayed-template-parsing
// RUN: FileCheck --input-file %T/texture_object.dp.cpp --match-full-lines %s

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

// CHECK: DPCT1050:{{[0-9]+}}: The template argument of  the image_accessor could not be deduced. You need to update this code.
// CHECK: void gather_force(const dpct::image_accessor<dpct_placeholder/*Fix the type manually*/, 1> gridTexObj){}
__global__ void gather_force(const cudaTextureObject_t gridTexObj){}

// CHECK: void gather_force(const dpct::image_base_p gridTexObj, sycl::queue *stream) {
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1050:{{[0-9]+}}: The template argument of the image_accessor could not be deduced. You need to update this code.
// CHECK-NEXT:  */
// CHECK-NEXT:  stream->submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      auto gridTexObj_acc = static_cast<dpct::image<dpct_placeholder/*Fix the type manually*/, 1> *>(gridTexObj)->get_access(cgh);
// CHECK-EMPTY:
// CHECK-NEXT:      auto gridTexObj_smpl = gridTexObj->get_sampler();
// CHECK-EMPTY:
// CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class gather_force_{{[a-f0-9]+}}>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          gather_force(dpct::image_accessor<dpct_placeholder/*Fix the type manually*/, 1>(gridTexObj_smpl, gridTexObj_acc));
// CHECK-NEXT:        });
// CHECK-NEXT:    });
// CHECK-NEXT: }
void gather_force(const cudaTextureObject_t gridTexObj, cudaStream_t stream) {
  gather_force <<< 1, 1, 1, stream >>>(gridTexObj);
}

// CHECK: template <class T> void BindTextureObject(dpct::image_matrix_p &data, dpct::image_base_p &tex) {
// CHECK-NEXT:   dpct::image_data res42;
// CHECK-NEXT:   dpct::image_info texDesc42;
// CHECK-NEXT:   res42.type = dpct::data_matrix;
// CHECK-NEXT:   res42.data = data;
// CHECK-NEXT:   texDesc42.addr_mode() = sycl::addressing_mode::clamp_to_edge;
// CHECK-NEXT:   texDesc42.addr_mode() = sycl::addressing_mode::clamp_to_edge;
// CHECK-NEXT:   texDesc42.addr_mode() = sycl::addressing_mode::clamp_to_edge;
// CHECK-NEXT:   texDesc42.filter_mode() = sycl::filtering_mode::nearest;
// CHECK-NEXT:   dpct::create_image(&tex, &res42, &texDesc42);
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
  cudaCreateTextureObject(&tex, &res42, &texDesc42, NULL);
}

// CHECK: void device01(dpct::image_accessor<sycl::uint2, 1> tex21) {
// CHECK-NEXT: sycl::uint2 u21;
// CHECK-NEXT: u21 = tex21.read(0.5f);
// CHECK-NEXT: u21 = tex21.read(1);
__device__ void device01(cudaTextureObject_t tex21) {
  uint2 u21;
  tex1D(&u21, tex21, 0.5f);
  tex1Dfetch(&u21, tex21, 1);
}

// CHECK: void kernel(dpct::image_accessor<sycl::uint2, 1> tex2, dpct::image_accessor<sycl::float4, 2> tex4) {
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
  // CHECK-NEXT: dpct::image_matrix_p a42;
  // CHECK-NEXT: dpct::dpct_malloc(&d_data42, sizeof(sycl::float4) * 32 * 32);
  // CHECK-NEXT: dpct::image_channel desc42 = dpct::create_image_channel(32, 32, 32, 32, dpct::channel_float);
  // CHECK-NEXT: a42 = new dpct::image_matrix(desc42, sycl::range<2>(32, 32));
  // CHECK-NEXT: dpct::dpct_memcpy(a42->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(d_data42, 32 * 32 * sizeof(sycl::float4), 32 * 32 * sizeof(sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(32 * 32 * sizeof(sycl::float4), 1, 1));
  // CHECK-NEXT: dpct::image_base_p tex42;
  // CHECK-NEXT: dpct::image_data res42;
  // CHECK-NEXT: dpct::image_info texDesc42;
  // CHECK-NEXT: res42.type = dpct::data_pitch;
  // CHECK-NEXT: res42.data = d_data42;
  // CHECK-NEXT: res42.chn = desc42;
  // CHECK-NEXT: res42.x = sizeof(sycl::float4) * 32;
  // CHECK-NEXT: res42.y = 32;
  // CHECK-NEXT: res42.pitch = sizeof(sycl::float4) * 32;
  // CHECK-NEXT: res42.type = dpct::data_matrix;
  // CHECK-NEXT: res42.data = a42;
  // CHECK-NEXT: texDesc42.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: texDesc42.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: texDesc42.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: texDesc42.filter_mode() = sycl::filtering_mode::nearest;
  // CHECK-NEXT: texDesc42.coord_normalized() = 1;
  // CHECK-NEXT: dpct::create_image(&tex42, &res42, &texDesc42);
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
  // CHECK-NEXT: dpct::dpct_malloc(&d_data21, sizeof(sycl::uint2) * 32);
  // CHECK-NEXT: dpct::image_base_p tex21;
  // CHECK-NEXT: dpct::image_data res21;
  // CHECK-NEXT: dpct::image_info texDesc21;
  // CHECK-NEXT: res21.type = dpct::data_linear;
  // CHECK-NEXT: res21.data = d_data21;
  // CHECK-NEXT: res21.chn.type = dpct::channel_unsigned;
  // CHECK-NEXT: res21.chn.set_channel_size(1, sizeof(unsigned)*8); // bits per channel
  // CHECK-NEXT: if (sizeof(sycl::uint2) >= sizeof(unsigned)*2) res21.chn.set_channel_size(2, sizeof(unsigned)*8);
  // CHECK-NEXT: if (sizeof(sycl::uint2) >= sizeof(unsigned)*3) res21.chn.set_channel_size(3, sizeof(unsigned)*8);
  // CHECK-NEXT: if (sizeof(sycl::uint2) >= sizeof(unsigned)*4) res21.chn.set_channel_size(4, sizeof(unsigned)*8);
  // CHECK-NEXT: res21.x = 32*sizeof(sycl::uint2);
  // CHECK-NEXT: texDesc21.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: texDesc21.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: texDesc21.addr_mode() = sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: texDesc21.filter_mode() = sycl::filtering_mode::linear;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  // CHECK-NEXT: texDesc21.readMode = cudaReadModeElementType;
  // CHECK-NEXT: dpct::create_image(&tex21, &res21, &texDesc21);
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
  texDesc21.addressMode[0] = cudaAddressModeClamp;
  texDesc21.addressMode[1] = cudaAddressModeClamp;
  texDesc21.addressMode[2] = cudaAddressModeClamp;
  texDesc21.filterMode = cudaFilterModeLinear;
  texDesc21.readMode = cudaReadModeElementType;
  cudaCreateTextureObject(&tex21, &res21, &texDesc21, NULL);

  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     auto tex21_acc = static_cast<dpct::image<sycl::uint2, 1> *>(tex21)->get_access(cgh);
  // CHECK-NEXT:     auto tex42_acc = static_cast<dpct::image<sycl::float4, 2> *>(tex42)->get_access(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     auto tex21_smpl = tex21->get_sampler();
  // CHECK-NEXT:     auto tex42_smpl = tex42->get_sampler();
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel(dpct::image_accessor<sycl::uint2, 1>(tex21_smpl, tex21_acc), dpct::image_accessor<sycl::float4, 2>(tex42_smpl, tex42_acc));
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel<<<1, 1>>>(tex21, tex42);
  
  // CHECK: texDesc42 = tex42->get_info();
  // CHECK-NEXT: res42 = tex42->get_data();
  cudaGetTextureObjectTextureDesc(&texDesc42, tex42);
  cudaGetTextureObjectResourceDesc(&res42, tex42);

  // CHECK: delete tex42;
  // CHECK-NEXT: delete tex21;
  cudaDestroyTextureObject(tex42);
  cudaDestroyTextureObject(tex21);

  // CHECK: delete a42;
  cudaFreeArray(a42);

  // CHECK: dpct::dpct_free(d_data42);
  // CHECK-NEXT: dpct::dpct_free(d_data21);
  cudaFree(d_data42);
  cudaFree(d_data21);

  // Test IsAssigned
  {
    int errorCode;
    // CHECK: errorCode = (dpct::create_image(&tex21, &res21, &texDesc21), 0);
    errorCode = cudaCreateTextureObject(&tex21, &res21, &texDesc21, NULL);
    // CHECK: cudaCheck((dpct::create_image(&tex21, &res21, &texDesc21), 0));
    cudaCheck(cudaCreateTextureObject(&tex21, &res21, &texDesc21, NULL));
    // CHECK: func((dpct::create_image(&tex21, &res21, &texDesc21), 0));
    func(cudaCreateTextureObject(&tex21, &res21, &texDesc21, NULL));
    // CHECK: funcT((dpct::create_image(&tex21, &res21, &texDesc21), 0));
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
