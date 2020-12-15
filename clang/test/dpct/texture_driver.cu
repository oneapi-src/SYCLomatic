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
  // CHECK-NEXT:  DPCT1073:0: The fields' values of parameter 'd' could not be deduced, so the call was not migrated. You need to update this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cuArrayCreate(&a, d);
  // CHECK-NEXT: /*
  // CHECK-NEXT:  DPCT1073:1: The fields' values of parameter 'p' could not be deduced, so the call was not migrated. You need to update this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cuArrayCreate(&a, p);
  // CHECK-NEXT: /*
  // CHECK-NEXT:  DPCT1073:2: The fields' values of parameter 'p + i' could not be deduced, so the call was not migrated. You need to update this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cuArrayCreate(&a, p + i);
  // CHECK-NEXT: /*
  // CHECK-NEXT:  DPCT1073:3: The fields' values of parameter '&d[i]' could not be deduced, so the call was not migrated. You need to update this code manually.
  // CHECK-NEXT: */
  // CHECK-NEXT: cuArrayCreate(&a, &d[i]);
  cuArrayCreate(&a, d);
  cuArrayCreate(&a, p);
  cuArrayCreate(&a, p + i);
  cuArrayCreate(&a, &d[i]);
}