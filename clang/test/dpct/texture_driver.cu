// RUN: dpct --format-range=none --usm-level=none -out-root %T/texture_driver %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture_driver/texture_driver.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/texture_driver/texture_driver.dp.cpp -o %T//texture_driver/texture_driver.dp.o %}

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

int main() {
  // CHECK: dpct::image_matrix_desc p3DDesc;
  // CHECK-NEXT: p3DDesc.width = 1;
  // CHECK-NEXT: p3DDesc.height = 2;
  // CHECK-NEXT: p3DDesc.depth = 3;
  // CHECK-NEXT: p3DDesc.channel_type = sycl::image_channel_type::signed_int8;
  // CHECK-NEXT: p3DDesc.num_channels = 4;
  CUDA_ARRAY3D_DESCRIPTOR p3DDesc;
  p3DDesc.Width = 1;
  p3DDesc.Height = 2;
  p3DDesc.Depth = 3;
  p3DDesc.Format = CU_AD_FORMAT_SIGNED_INT8;
  p3DDesc.Flags = 5;
  p3DDesc.NumChannels = 4;

  // CHECK: dpct::image_matrix_desc halfDesc;
  // CHECK-NEXT: halfDesc.height = 32;
  // CHECK-NEXT: halfDesc.width = 64;
  // CHECK-NEXT: halfDesc.channel_type = sycl::image_channel_type::fp16;
  // CHECK-NEXT: halfDesc.num_channels = 1;
  CUDA_ARRAY_DESCRIPTOR halfDesc;
  halfDesc.Height = 32;
  halfDesc.Width = 64;
  halfDesc.Format = CU_AD_FORMAT_HALF;
  halfDesc.NumChannels = 1;

  // CHECK: dpct::image_matrix_desc float4Desc;
  // CHECK-NEXT: float4Desc.width = 64;
  // CHECK-NEXT: float4Desc.channel_type = sycl::image_channel_type::fp32;
  // CHECK-NEXT: float4Desc.num_channels = 4;
  // CHECK-NEXT: float4Desc.height = 32;
  CUDA_ARRAY_DESCRIPTOR float4Desc;
  float4Desc.Width = 64;
  float4Desc.Format = CU_AD_FORMAT_FLOAT;
  float4Desc.NumChannels = 4;
  float4Desc.Height = 32;

  // CHECK: dpct::image_matrix **a3d_ptr = new dpct::image_matrix_p;
  // CHECK-NEXT: *a3d_ptr = new dpct::image_matrix(&p3DDesc);
  // CHECK-NEXT: delete (*a3d_ptr);
  // CHECK-NEXT: delete a3d_ptr;
  CUarray_st **a3d_ptr = new CUarray;
  cuArray3DCreate(a3d_ptr, &p3DDesc);
  cuArrayDestroy(*a3d_ptr);
  delete a3d_ptr;

  // CHECK: dpct::image_matrix **a_ptr = new dpct::image_matrix_p;
  // CHECK-NEXT: dpct::image_matrix_p a42;
  // CHECK-NEXT: *a_ptr = new dpct::image_matrix(&halfDesc);
  // CHECK-NEXT: a42 = new dpct::image_matrix(&float4Desc);
  // CHECK-NEXT: delete (*a_ptr);
  // CHECK-NEXT: delete a42;
  // CHECK-NEXT: delete a_ptr;
  CUarray_st **a_ptr = new CUarray;
  CUarray a42;
  cuArrayCreate(a_ptr, &halfDesc);
  cuArrayCreate(&a42, &float4Desc);
  cuArrayDestroy(*a_ptr);
  cuArrayDestroy(a42);
  delete a_ptr;

  // Test IsAssigned
  {
    int errorCode;

    // CHECK: errorCode = DPCT_CHECK_ERROR(a42 = new dpct::image_matrix(&float4Desc));
    errorCode = cuArrayCreate(&a42, &float4Desc);
    // CHECK: errorCode = DPCT_CHECK_ERROR(delete a42);
    errorCode = cuArrayDestroy(a42);


    // CHECK: cudaCheck(DPCT_CHECK_ERROR(a42 = new dpct::image_matrix(&float4Desc)));
    cudaCheck(cuArrayCreate(&a42, &float4Desc));
    // CHECK: cudaCheck(DPCT_CHECK_ERROR(delete a42));
    cudaCheck(cuArrayDestroy(a42));


    // CHECK: func(DPCT_CHECK_ERROR(a42 = new dpct::image_matrix(&float4Desc)));
    func(cuArrayCreate(&a42, &float4Desc));
    // CHECK: func(DPCT_CHECK_ERROR(delete a42));
    func(cuArrayDestroy(a42));


    // CHECK: funcT(DPCT_CHECK_ERROR(a42 = new dpct::image_matrix(&float4Desc)));
    funcT(cuArrayCreate(&a42, &float4Desc));
    // CHECK: funcT(DPCT_CHECK_ERROR(delete a42));
    funcT(cuArrayDestroy(a42));
  }
}

void create_array_fail() {
  CUarray a;
  unsigned i;
  // CHECK: dpct::image_matrix_desc d[20], *p;
  CUDA_ARRAY_DESCRIPTOR d[20], *p;
  p = &d[5];

  // CHECK: a = new dpct::image_matrix(d);
  // CHECK-NEXT: a = new dpct::image_matrix(p);
  // CHECK-NEXT: a = new dpct::image_matrix(p + i);
  // CHECK-NEXT: a = new dpct::image_matrix(&d[i]);
  cuArrayCreate(&a, d);
  cuArrayCreate(&a, p);
  cuArrayCreate(&a, p + i);
  cuArrayCreate(&a, &d[i]);
}

void test_texref() {
  // CHECK: sycl::addressing_mode addr_mode;
  // CHECK-NEXT: sycl::filtering_mode filter_mode;
  // CHECK-NEXT: sycl::image_channel_type format;
  // CHECK-NEXT: dpct::image_matrix_p arr;
  // CHECK-NEXT: dpct::image_wrapper_base_p tex;
  // CHECK-NEXT: int err_code;
  CUaddress_mode addr_mode;
  CUfilter_mode filter_mode;
  CUarray_format format;
  CUarray arr;
  CUtexref tex;
  CUresult err_code;
  int flags, chn_num;

  // CHECK: tex->set_channel_type(format);
  // CHECK-NEXT: tex->set_channel_num(4);
  // CHECK-NEXT: err_code = DPCT_CHECK_ERROR((tex->set_channel_type(sycl::image_channel_type::fp32), tex->set_channel_num(chn_num)));
  // CHECK-NEXT: cudaCheck(DPCT_CHECK_ERROR((tex->set_channel_type(format), tex->set_channel_num(4))));
  // CHECK-NEXT: func(DPCT_CHECK_ERROR((tex->set_channel_type(format), tex->set_channel_num(4))));
  // CHECK-NEXT: funcT(DPCT_CHECK_ERROR((tex->set_channel_type(format), tex->set_channel_num(4))));
  cuTexRefSetFormat(tex, format, 4);
  err_code = cuTexRefSetFormat(tex, CU_AD_FORMAT_FLOAT, chn_num);
  cudaCheck(cuTexRefSetFormat(tex, format, 4));
  func(cuTexRefSetFormat(tex,format,4));
  funcT(cuTexRefSetFormat(tex,format,4));

  // CHECK: /*
  // CHECK-NEXT: DPCT1074:{{[0-9]+}}: The SYCL Image class does not support some of the flags used in the original code. Unsupported flags were ignored. Data read from SYCL Image could not be normalized as specified in the original code.
  // CHECK-NEXT: */
  // CHECK-NEXT: tex->set_coordinate_normalization_mode(flags & 0x02);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1074:{{[0-9]+}}: The SYCL Image class does not support some of the flags used in the original code. Unsupported flags were ignored. Data read from SYCL Image could not be normalized as specified in the original code.
  // CHECK-NEXT: */
  // CHECK-NEXT: err_code = DPCT_CHECK_ERROR(tex->set(sycl::coordinate_normalization_mode::normalized));
  // CHECK-NEXT: cudaCheck(DPCT_CHECK_ERROR(tex->set(sycl::coordinate_normalization_mode::normalized)));
  // CHECK-NEXT: func(DPCT_CHECK_ERROR(tex->set(sycl::coordinate_normalization_mode::normalized)));
  // CHECK-NEXT: funcT(DPCT_CHECK_ERROR(tex->set(sycl::coordinate_normalization_mode::unnormalized)));
  cuTexRefSetFlags(tex, flags);
  err_code = cuTexRefSetFlags(tex, CU_TRSF_NORMALIZED_COORDINATES);
  cudaCheck(cuTexRefSetFlags(tex,  CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_READ_AS_INTEGER));
  func(cuTexRefSetFlags(tex,3));
  funcT(cuTexRefSetFlags(tex,1));
  unsigned int uflag;
  // CHECK: uflag = tex->is_coordinate_normalized() << 1;
  cuTexRefGetFlags(&uflag, tex);

  // CHECK: tex->set(addr_mode);
  // CHECK-NEXT: err_code = DPCT_CHECK_ERROR(tex->set(sycl::addressing_mode::clamp_to_edge));
  // CHECK-NEXT: cudaCheck(DPCT_CHECK_ERROR(tex->set(addr_mode)));
  // CHECK-NEXT: func(DPCT_CHECK_ERROR(tex->set(addr_mode)));
  // CHECK-NEXT: funcT(DPCT_CHECK_ERROR(tex->set(addr_mode)));
  cuTexRefSetAddressMode(tex, 0, addr_mode);
  err_code = cuTexRefSetAddressMode(tex, 1, CU_TR_ADDRESS_MODE_CLAMP);
  cudaCheck(cuTexRefSetAddressMode(tex, 2, addr_mode));
  func(cuTexRefSetAddressMode(tex,0,addr_mode));
  funcT(cuTexRefSetAddressMode(tex,0,addr_mode));

  // CHECK: addr_mode = tex->get_addressing_mode();
  cuTexRefGetAddressMode(&addr_mode, tex, 0);

  // CHECK: tex->set(filter_mode);
  // CHECK-NEXT: err_code = DPCT_CHECK_ERROR(tex->set(sycl::filtering_mode::linear));
  // CHECK-NEXT: cudaCheck(DPCT_CHECK_ERROR(tex->set(filter_mode)));
  // CHECK-NEXT: func(DPCT_CHECK_ERROR(tex->set(filter_mode)));
  // CHECK-NEXT: funcT(DPCT_CHECK_ERROR(tex->set(filter_mode)));
  cuTexRefSetFilterMode(tex, filter_mode);
  err_code = cuTexRefSetFilterMode(tex, CU_TR_FILTER_MODE_LINEAR);
  cudaCheck(cuTexRefSetFilterMode(tex, filter_mode));
  func(cuTexRefSetFilterMode(tex,filter_mode));
  funcT(cuTexRefSetFilterMode(tex,filter_mode));

  // CHECK: filter_mode = tex->get_filtering_mode();
  cuTexRefGetFilterMode(&filter_mode, tex);

  // CHECK: tex->attach(dpct::image_data(arr));
  // CHECK-NEXT: err_code = DPCT_CHECK_ERROR(tex->attach(dpct::image_data(arr)));
  // CHECK-NEXT: cudaCheck(DPCT_CHECK_ERROR(tex->attach(dpct::image_data(arr))));
  // CHECK-NEXT: func(DPCT_CHECK_ERROR(tex->attach(dpct::image_data(arr))));
  // CHECK-NEXT: funcT(DPCT_CHECK_ERROR(tex->attach(dpct::image_data(arr))));
  cuTexRefSetArray(tex, arr, CU_TRSA_OVERRIDE_FORMAT);
  err_code = cuTexRefSetArray(tex, arr, 0x01);
  cudaCheck(cuTexRefSetArray(tex, arr, CU_TRSA_OVERRIDE_FORMAT));
  func(cuTexRefSetArray(tex,arr, CU_TRSA_OVERRIDE_FORMAT));
  funcT(cuTexRefSetArray(tex,arr, CU_TRSA_OVERRIDE_FORMAT));

  // CHECK: dpct::device_ptr dptr;
  // CHECK-Next: size_t s, b;
  // CHECK-Next: tex->attach(dptr, b);
  // CHECK-Next: size_t desc_x_ct1, desc_y_ct1;
  // CHECK-Next: unsigned desc_channel_num_ct1;
  // CHECK-Next: sycl::image_channel_type desc_channel_type_ct1;
  // CHECK-Next: tex->attach(dptr, desc_x_ct1, desc_y_ct1, b);
  CUdeviceptr dptr;
  size_t s, b;
  cuTexRefSetAddress(&s, tex, dptr, b);
  CUDA_ARRAY_DESCRIPTOR desc;
  cuTexRefSetAddress2D(tex, &desc, dptr, b);
}

// CHECK: sycl::addressing_mode AddrMode[] =
// CHECK-NEXT: {
// CHECK-NEXT:   sycl::addressing_mode::repeat,
// CHECK-NEXT:   sycl::addressing_mode::clamp_to_edge,
// CHECK-NEXT:   sycl::addressing_mode::clamp
// CHECK-NEXT: };
CUaddress_mode AddrMode[] =
{
  CU_TR_ADDRESS_MODE_WRAP,
  CU_TR_ADDRESS_MODE_CLAMP,
  CU_TR_ADDRESS_MODE_BORDER
};

// CHECK: sycl::filtering_mode FltMode[] =
// CHECK-NEXT: {
// CHECK-NEXT:   sycl::filtering_mode::nearest,
// CHECK-NEXT:   sycl::filtering_mode::linear
// CHECK-NEXT: };
CUfilter_mode FltMode[] =
{
  CU_TR_FILTER_MODE_POINT,
  CU_TR_FILTER_MODE_LINEAR
};

// CHECK: void TestAssignment(sycl::addressing_mode a) {
// CHECK-NEXT:   if (a == sycl::addressing_mode::repeat);
// CHECK-NEXT:   if (a == sycl::addressing_mode::clamp_to_edge);
// CHECK-NEXT: }
void TestAssignment(CUaddress_mode a) {
  if (a == CU_TR_ADDRESS_MODE_WRAP);
  if (a == CU_TR_ADDRESS_MODE_CLAMP);
}
