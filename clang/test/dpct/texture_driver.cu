// RUN: dpct --format-range=none --usm-level=none -out-root %T/texture_driver %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14 -fno-delayed-template-parsing
// RUN: FileCheck --input-file %T/texture_driver/texture_driver.dp.cpp --match-full-lines %s

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

  // CHECK: size_t halfDesc_x_ct1, halfDesc_y_ct1;
  // CHECK-NEXT: unsigned halfDesc_channel_num_ct1;
  // CHECK-NEXT: sycl::image_channel_type halfDesc_channel_type_ct1;
  // CHECK-NEXT: halfDesc_y_ct1 = 32;
  // CHECK-NEXT: halfDesc_x_ct1 = 64;
  // CHECK-NEXT: halfDesc_channel_type_ct1 = sycl::image_channel_type::fp16;
  // CHECK-NEXT: halfDesc_channel_num_ct1 = 1;
  CUDA_ARRAY_DESCRIPTOR halfDesc;
  halfDesc.Height = 32;
  halfDesc.Width = 64;
  halfDesc.Format = CU_AD_FORMAT_HALF;
  halfDesc.NumChannels = 1;

  // CHECK: size_t float4Desc_x_ct1, float4Desc_y_ct1;
  // CHECK-NEXT: unsigned float4Desc_channel_num_ct1;
  // CHECK-NEXT: sycl::image_channel_type float4Desc_channel_type_ct1;
  // CHECK-NEXT: float4Desc_x_ct1 = 64;
  // CHECK-NEXT: float4Desc_channel_type_ct1 = sycl::image_channel_type::fp32;
  // CHECK-NEXT: float4Desc_channel_num_ct1 = 4;
  // CHECK-NEXT: float4Desc_y_ct1 = 32;
  CUDA_ARRAY_DESCRIPTOR float4Desc;
  float4Desc.Width = 64;
  float4Desc.Format = CU_AD_FORMAT_FLOAT;
  float4Desc.NumChannels = 4;
  float4Desc.Height = 32;

  // CHECK: dpct::image_matrix **a_ptr = new dpct::image_matrix_p;
  // CHECK-NEXT: dpct::image_matrix_p a42;
  // CHECK-NEXT: *a_ptr = new dpct::image_matrix(halfDesc_channel_type_ct1, halfDesc_channel_num_ct1, halfDesc_x_ct1, halfDesc_y_ct1);
  // CHECK-NEXT: a42 = new dpct::image_matrix(float4Desc_channel_type_ct1, float4Desc_channel_num_ct1, float4Desc_x_ct1, float4Desc_y_ct1);
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

    // CHECK: errorCode = (a42 = new dpct::image_matrix(float4Desc_channel_type_ct1, float4Desc_channel_num_ct1, float4Desc_x_ct1, float4Desc_y_ct1), 0);
    errorCode = cuArrayCreate(&a42, &float4Desc);
    // CHECK: errorCode = (delete a42, 0);
    errorCode = cuArrayDestroy(a42);


    // CHECK: cudaCheck((a42 = new dpct::image_matrix(float4Desc_channel_type_ct1, float4Desc_channel_num_ct1, float4Desc_x_ct1, float4Desc_y_ct1), 0));
    cudaCheck(cuArrayCreate(&a42, &float4Desc));
    // CHECK: cudaCheck((delete a42, 0));
    cudaCheck(cuArrayDestroy(a42));


    // CHECK: func((a42 = new dpct::image_matrix(float4Desc_channel_type_ct1, float4Desc_channel_num_ct1, float4Desc_x_ct1, float4Desc_y_ct1), 0));
    func(cuArrayCreate(&a42, &float4Desc));
    // CHECK: func((delete a42, 0));
    func(cuArrayDestroy(a42));


    // CHECK: funcT((a42 = new dpct::image_matrix(float4Desc_channel_type_ct1, float4Desc_channel_num_ct1, float4Desc_x_ct1, float4Desc_y_ct1), 0));
    funcT(cuArrayCreate(&a42, &float4Desc));
    // CHECK: funcT((delete a42, 0));
    funcT(cuArrayDestroy(a42));
  }
}

void create_array_fail() {
  CUarray a;
  unsigned i;
  // CHECK: CUDA_ARRAY_DESCRIPTOR d[20], *p;
  CUDA_ARRAY_DESCRIPTOR d[20], *p;
  p = &d[5];

  // CHECK: /*
  // CHECK-NEXT:  DPCT1073:{{[0-9]+}}: The field values of parameter 'd' could not be deduced, so the call was not migrated. You need to update this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cuArrayCreate(&a, d);
  // CHECK-NEXT: /*
  // CHECK-NEXT:  DPCT1073:{{[0-9]+}}: The field values of parameter 'p' could not be deduced, so the call was not migrated. You need to update this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cuArrayCreate(&a, p);
  // CHECK-NEXT: /*
  // CHECK-NEXT:  DPCT1073:{{[0-9]+}}: The field values of parameter 'p + i' could not be deduced, so the call was not migrated. You need to update this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cuArrayCreate(&a, p + i);
  // CHECK-NEXT: /*
  // CHECK-NEXT:  DPCT1073:{{[0-9]+}}: The field values of parameter '&d[i]' could not be deduced, so the call was not migrated. You need to update this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cuArrayCreate(&a, &d[i]);
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
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: err_code = (tex->set_channel_type(sycl::image_channel_type::fp32), tex->set_channel_num(chn_num), 0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaCheck((tex->set_channel_type(format), tex->set_channel_num(4), 0));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: func((tex->set_channel_type(format), tex->set_channel_num(4), 0));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: funcT((tex->set_channel_type(format), tex->set_channel_num(4), 0));
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
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1074:{{[0-9]+}}: The SYCL Image class does not support some of the flags used in the original code. Unsupported flags were ignored. Data read from SYCL Image could not be normalized as specified in the original code.
  // CHECK-NEXT: */
  // CHECK-NEXT: err_code = (tex->set(sycl::coordinate_normalization_mode::normalized), 0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaCheck((tex->set(sycl::coordinate_normalization_mode::normalized), 0));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: func((tex->set(sycl::coordinate_normalization_mode::normalized), 0));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: funcT((tex->set(sycl::coordinate_normalization_mode::unnormalized), 0));
  cuTexRefSetFlags(tex, flags);
  err_code = cuTexRefSetFlags(tex, CU_TRSF_NORMALIZED_COORDINATES);
  cudaCheck(cuTexRefSetFlags(tex,  CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_READ_AS_INTEGER));
  func(cuTexRefSetFlags(tex,3));
  funcT(cuTexRefSetFlags(tex,1));

  // CHECK: tex->set(addr_mode);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: err_code = (tex->set(sycl::addressing_mode::clamp_to_edge), 0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaCheck((tex->set(addr_mode), 0));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: func((tex->set(addr_mode), 0));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: funcT((tex->set(addr_mode), 0));
  cuTexRefSetAddressMode(tex, 0, addr_mode);
  err_code = cuTexRefSetAddressMode(tex, 1, CU_TR_ADDRESS_MODE_CLAMP);
  cudaCheck(cuTexRefSetAddressMode(tex, 2, addr_mode));
  func(cuTexRefSetAddressMode(tex,0,addr_mode));
  funcT(cuTexRefSetAddressMode(tex,0,addr_mode));

  // CHECK: tex->set(filter_mode);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: err_code = (tex->set(sycl::filtering_mode::linear), 0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaCheck((tex->set(filter_mode), 0));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: func((tex->set(filter_mode), 0));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: funcT((tex->set(filter_mode), 0));
  cuTexRefSetFilterMode(tex, filter_mode);
  err_code = cuTexRefSetFilterMode(tex, CU_TR_FILTER_MODE_LINEAR);
  cudaCheck(cuTexRefSetFilterMode(tex, filter_mode));
  func(cuTexRefSetFilterMode(tex,filter_mode));
  funcT(cuTexRefSetFilterMode(tex,filter_mode));

  // CHECK: tex->attach(dpct::image_data(arr));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: err_code = (tex->attach(dpct::image_data(arr)), 0);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaCheck((tex->attach(dpct::image_data(arr)), 0));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: func((tex->attach(dpct::image_data(arr)), 0));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: funcT((tex->attach(dpct::image_data(arr)), 0));
  cuTexRefSetArray(tex, arr, CU_TRSA_OVERRIDE_FORMAT);
  err_code = cuTexRefSetArray(tex, arr, 0x01);
  cudaCheck(cuTexRefSetArray(tex, arr, CU_TRSA_OVERRIDE_FORMAT));
  func(cuTexRefSetArray(tex,arr, CU_TRSA_OVERRIDE_FORMAT));
  funcT(cuTexRefSetArray(tex,arr, CU_TRSA_OVERRIDE_FORMAT));
}