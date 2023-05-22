// RUN: dpct --format-range=none --usm-level=none -out-root %T/texture_object_driver %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14 -fno-delayed-template-parsing
// RUN: FileCheck --input-file %T/texture_object_driver/texture_object_driver.dp.cpp --match-full-lines %s

#include "cuda.h"
#include <stdio.h>

#define cudaCheck(stmt) do {                         \
  int err = stmt;                            \
  if (err != cudaSuccess) {                          \
    char msg[256];                                   \
    sprintf(msg, "%s in file %s, function %s, line %d\n", #stmt,__FILE__,__FUNCTION__,__LINE__); \
  }                                                  \
} while(0)

void func(int i) {}

template <typename T>
void funcT(T t) {}

// CHECK: template <class T> void BindTextureObject(dpct::image_matrix_p &data, dpct::image_wrapper_base_p &tex) {
// CHECK-NEXT:   dpct::image_data res42;
// CHECK-NEXT:   dpct::sampling_info texDesc42;
// CHECK-NEXT:   res42.set_data_type(dpct::image_data_type::matrix);
// CHECK-NEXT:   res42.set_data_ptr(data);
// CHECK-NEXT:   texDesc42.set(sycl::addressing_mode::clamp_to_edge);
// CHECK-NEXT:   texDesc42.set(sycl::filtering_mode::nearest);
// CHECK-NEXT:   data = (dpct::image_matrix_p)res42.get_data_ptr();
// CHECK-NEXT:   tex = dpct::create_image_wrapper(res42, texDesc42);
// CHECK-NEXT: }
template <class T> void BindTextureObject(CUarray &data, CUtexObject &tex) {
  CUDA_RESOURCE_DESC res42;
  CUDA_TEXTURE_DESC texDesc42;
  res42.resType = CU_RESOURCE_TYPE_ARRAY;
  res42.res.array.hArray = data;
  texDesc42.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc42.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc42.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc42.filterMode = CU_TR_FILTER_MODE_POINT;
  data = res42.res.array.hArray;
  cuTexObjectCreate(&tex, &res42, &texDesc42, NULL);
}

int main() {

  // CHECK: sycl::float4 *d_data42;
  // CHECK-NEXT: dpct::image_matrix_p a42;
  // CHECK-NEXT: size_t desc42_x_ct1, desc42_y_ct1;
  // CHECK-NEXT: unsigned desc42_channel_num_ct1;
  // CHECK-NEXT: sycl::image_channel_type desc42_channel_type_ct1;
  // CHECK-NEXT: desc42_channel_num_ct1 = 4;
  // CHECK-NEXT: desc42_channel_type_ct1 = sycl::image_channel_type::fp32;
  // CHECK-NEXT: desc42_x_ct1 = 32;
  // CHECK-NEXT: desc42_y_ct1 = 32;
  // CHECK-NEXT: a42 = new dpct::image_matrix(desc42_channel_type_ct1, desc42_channel_num_ct1, desc42_x_ct1, desc42_y_ct1);
  // CHECK-NEXT: dpct::image_wrapper_base_p tex42;
  // CHECK-NEXT: dpct::image_data res42;
  // CHECK-NEXT: dpct::sampling_info texDesc42;
  // CHECK-NEXT: res42.set_data_type(dpct::image_data_type::pitch);
  // CHECK-NEXT: res42.set_data_ptr((dpct::device_ptr)d_data42);
  // CHECK-NEXT: res42.set_x(sizeof(sycl::float4) * 32);
  // CHECK-NEXT: res42.set_y(32);
  // CHECK-NEXT: res42.set_pitch(sizeof(sycl::float4) * 32);
  // CHECK-NEXT: res42.set_channel_num(4);
  // CHECK-NEXT: res42.set_channel_type(sycl::image_channel_type::fp32);
  // CHECK-NEXT: res42.set_data_type(dpct::image_data_type::matrix);
  // CHECK-NEXT: res42.set_data_ptr(a42);
  // CHECK-NEXT: texDesc42.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest, sycl::coordinate_normalization_mode::normalized);
  // CHECK-NEXT: tex42 = dpct::create_image_wrapper(res42, texDesc42);
  float4 *d_data42;
  CUarray a42;
  CUDA_ARRAY_DESCRIPTOR desc42;
  desc42.NumChannels = 4;
  desc42.Format = CU_AD_FORMAT_FLOAT;
  desc42.Width = 32;
  desc42.Height = 32;
  cuArrayCreate(&a42, &desc42);
  CUtexObject tex42;
  CUDA_RESOURCE_DESC res42;
  CUDA_TEXTURE_DESC texDesc42;
  res42.resType = CU_RESOURCE_TYPE_PITCH2D;
  res42.res.pitch2D.devPtr = (CUdeviceptr)d_data42;
  res42.res.pitch2D.width = sizeof(float4) * 32;
  res42.res.pitch2D.height = 32;
  res42.res.pitch2D.pitchInBytes = sizeof(float4) * 32;
  res42.res.pitch2D.numChannels = 4;
  res42.res.pitch2D.format = CU_AD_FORMAT_FLOAT;
  res42.resType = CU_RESOURCE_TYPE_ARRAY;
  res42.res.array.hArray = a42;
  texDesc42.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc42.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc42.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc42.filterMode = CU_TR_FILTER_MODE_POINT;
  texDesc42.flags = CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES;
  cuTexObjectCreate(&tex42, &res42, &texDesc42, NULL);

  // CHECK: sycl::uint2 *d_data21;
  // CHECK-NEXT: d_data21 = (sycl::uint2 *)dpct::dpct_malloc(sizeof(sycl::uint2) * 32);
  // CHECK-NEXT: dpct::image_wrapper_base_p tex21;
  // CHECK-NEXT: dpct::image_data res21;
  // CHECK-NEXT: dpct::sampling_info texDesc21;
  // CHECK-NEXT: res21.set_data_type(dpct::image_data_type::linear);
  // CHECK-NEXT: res21.set_data_ptr((dpct::device_ptr)d_data21);
  // CHECK-NEXT: res21.set_channel_num(2);
  // CHECK-NEXT: res21.set_channel_type(sycl::image_channel_type::unsigned_int32);
  // CHECK-NEXT: res21.set_x(32*sizeof(sycl::uint2));
  // CHECK-NEXT: unsigned chnX = res21.get_channel_num();
  // CHECK-NEXT: sycl::image_channel_type formatKind = res21.get_channel_type();
  // CHECK-NEXT: texDesc21.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::linear, sycl::coordinate_normalization_mode::normalized);
  // CHECK-NEXT: tex21 = dpct::create_image_wrapper(res21, texDesc21);
  uint2 *d_data21;
  cudaMalloc(&d_data21, sizeof(uint2) * 32);
  CUtexObject tex21;
  CUDA_RESOURCE_DESC res21;
  CUDA_TEXTURE_DESC texDesc21;
  res21.resType = CU_RESOURCE_TYPE_LINEAR;
  res21.res.linear.devPtr = (CUdeviceptr)d_data21;
  res21.res.linear.numChannels = 2;
  res21.res.linear.format = CU_AD_FORMAT_UNSIGNED_INT32;
  res21.res.linear.sizeInBytes = 32*sizeof(uint2);
  unsigned chnX = res21.res.linear.numChannels;
  CUarray_format formatKind = res21.res.linear.format;
  texDesc21.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc21.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc21.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc21.filterMode = CU_TR_FILTER_MODE_LINEAR;
  texDesc21.flags = CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_READ_AS_INTEGER;
  cuTexObjectCreate(&tex21, &res21, &texDesc21, NULL);
  
  // CHECK: texDesc42 = tex42->get_sampling_info();
  // CHECK-NEXT: res42 = tex42->get_data();
  cuTexObjectGetTextureDesc(&texDesc42, tex42);
  cuTexObjectGetResourceDesc(&res42, tex42);

  // CHECK: delete tex42;
  // CHECK-NEXT: delete tex21;
  cuTexObjectDestroy(tex42);
  cuTexObjectDestroy(tex21);

  // CHECK: delete a42;
  cuArrayDestroy(a42);

  // CHECK: dpct::dpct_free(d_data42);
  // CHECK-NEXT: dpct::dpct_free(d_data21);
  cudaFree(d_data42);
  cudaFree(d_data21);

  // Test IsAssigned
  {
    int errorCode;
    // CHECK: errorCode = CHECK_SYCL_ERROR(tex21 = dpct::create_image_wrapper(res21, texDesc21));
    errorCode = cuTexObjectCreate(&tex21, &res21, &texDesc21, NULL);
    // CHECK: cudaCheck(CHECK_SYCL_ERROR(tex21 = dpct::create_image_wrapper(res21, texDesc21)));
    cudaCheck(cuTexObjectCreate(&tex21, &res21, &texDesc21, NULL));
    // CHECK: func(CHECK_SYCL_ERROR(tex21 = dpct::create_image_wrapper(res21, texDesc21)));
    func(cuTexObjectCreate(&tex21, &res21, &texDesc21, NULL));
    // CHECK: funcT(CHECK_SYCL_ERROR(tex21 = dpct::create_image_wrapper(res21, texDesc21)));
    funcT(cuTexObjectCreate(&tex21, &res21, &texDesc21, NULL));

    // CHECK: errorCode = CHECK_SYCL_ERROR(delete tex21);
    errorCode = cuTexObjectDestroy(tex21);
    // CHECK: cudaCheck(CHECK_SYCL_ERROR(delete tex21));
    cudaCheck(cuTexObjectDestroy(tex21));
    // CHECK: func(CHECK_SYCL_ERROR(delete tex21));
    func(cuTexObjectDestroy(tex21));
    // CHECK: funcT(CHECK_SYCL_ERROR(delete tex21));
    funcT(cuTexObjectDestroy(tex21));
  }
}

void foo(){
  CUarray a42;
  CUDA_RESOURCE_DESC res42;
  // CHECK: res42.set_data_type(dpct::image_data_type::matrix);
  // CHECK-NEXT: res42.set_data_ptr(a42);
  res42.resType = CU_RESOURCE_TYPE_ARRAY;
  res42.res.array.hArray = a42;

  float4 *d_data42;
  // CHECK: res42.set_data_type(dpct::image_data_type::pitch);
  // CHECK-NEXT: res42.set_data_ptr((dpct::device_ptr)d_data42);
  // CHECK-NEXT: res42.set_channel_num(4);
  // CHECK-NEXT: res42.set_channel_type(sycl::image_channel_type::fp32);
  // CHECK-NEXT: res42.set_x(sizeof(sycl::float4) * 32);
  // CHECK-NEXT: res42.set_y(32);
  // CHECK-NEXT: res42.set_pitch(sizeof(sycl::float4) * 32);
  res42.resType = CU_RESOURCE_TYPE_PITCH2D;
  res42.res.pitch2D.devPtr = (CUdeviceptr)d_data42;
  res42.res.pitch2D.numChannels = 4;
  res42.res.pitch2D.format = CU_AD_FORMAT_FLOAT;
  res42.res.pitch2D.width = sizeof(float4) * 32;
  res42.res.pitch2D.height = 32;
  res42.res.pitch2D.pitchInBytes = sizeof(float4) * 32;

  uint2 *d_data21;
  // CHECK: res42.set_data_type(dpct::image_data_type::linear);
  // CHECK-NEXT: res42.set_data_ptr((dpct::device_ptr)d_data21);
  // CHECK-NEXT: res42.set_x(sizeof(sycl::float4) * 32);
  // CHECK-NEXT: res42.set_channel_num(4);
  // CHECK-NEXT: res42.set_channel_type(sycl::image_channel_type::fp32);
  res42.resType = CU_RESOURCE_TYPE_LINEAR;
  res42.res.linear.devPtr = (CUdeviceptr)d_data21;
  res42.res.linear.sizeInBytes = sizeof(float4) * 32;
  res42.res.pitch2D.numChannels = 4;
  res42.res.pitch2D.format = CU_AD_FORMAT_FLOAT;

  // CHECK:  dpct::sampling_info tex_tmp;
  // CHECK-NEXT:   tex_tmp.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest, sycl::coordinate_normalization_mode::unnormalized);
  // CHECK-NEXT:   sycl::addressing_mode addr = tex_tmp.get_addressing_mode();
  // CHECK-NEXT:   sycl::filtering_mode filter = tex_tmp.get_filtering_mode();
  // CHECK-NEXT:   int flags = tex_tmp.is_coordinate_normalized();
  CUDA_TEXTURE_DESC tex_tmp;
  tex_tmp.flags = CU_TRSF_READ_AS_INTEGER;
  tex_tmp.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
  tex_tmp.filterMode = CU_TR_FILTER_MODE_POINT;
  CUaddress_mode_enum addr = tex_tmp.addressMode[0];
  CUfilter_mode_enum filter = tex_tmp.filterMode;
  int flags = tex_tmp.flags;
  
  // CHECK: /*
  // CHECK-NEXT: DPCT1074:{{[0-9]+}}: The SYCL Image class does not support some of the flags used in the original code. Unsupported flags were ignored. Data read from SYCL Image could not be normalized as specified in the original code.
  // CHECK-NEXT: */
  // CHECK-NEXT: tex_tmp.set_coordinate_normalization_mode(flags & 0x02);
  // CHECK-NEXT: tex_tmp.set(sycl::addressing_mode::clamp_to_edge);
  // CHECK-NEXT: if (true) {
  // CHECK-NEXT:   tex_tmp.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest, sycl::coordinate_normalization_mode::normalized);
  // CHECK-NEXT: }
  // CHECK-NEXT: tex_tmp.set(sycl::filtering_mode::linear);
  // CHECK-NEXT: addr = tex_tmp.get_addressing_mode();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1074:{{[0-9]+}}: The SYCL Image class does not support some of the flags used in the original code. Unsupported flags were ignored. Data read from SYCL Image could not be normalized as specified in the original code.
  // CHECK-NEXT: */
  // CHECK-NEXT: tex_tmp.set(sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest, sycl::coordinate_normalization_mode::normalized);
  tex_tmp.flags = flags;
  tex_tmp.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
  if (true) {
    tex_tmp.filterMode = CU_TR_FILTER_MODE_POINT;
    tex_tmp.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    tex_tmp.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    tex_tmp.flags = 3;
  }
  tex_tmp.filterMode = CU_TR_FILTER_MODE_LINEAR;
  addr = tex_tmp.addressMode[0];
  tex_tmp.filterMode = CU_TR_FILTER_MODE_POINT;
  tex_tmp.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
  tex_tmp.flags = CU_TRSF_NORMALIZED_COORDINATES;
}

