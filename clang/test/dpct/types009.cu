// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8
// RUN: dpct --format-range=none --out-root %T/types009 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/types009/types009.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/types009/types009.dp.cpp -o %T/types009/types009.dp.o %}

#include <library_types.h>

void foo(cudaDataType_t t) {
  //      CHECK: t = dpct::library_data_t::real_half;
  // CHECK-NEXT: t = dpct::library_data_t::complex_half;
  // CHECK-NEXT: t = dpct::library_data_t::real_bfloat16;
  // CHECK-NEXT: t = dpct::library_data_t::complex_bfloat16;
  // CHECK-NEXT: t = dpct::library_data_t::real_float;
  // CHECK-NEXT: t = dpct::library_data_t::complex_float;
  // CHECK-NEXT: t = dpct::library_data_t::real_double;
  // CHECK-NEXT: t = dpct::library_data_t::complex_double;
  // CHECK-NEXT: t = dpct::library_data_t::real_int4;
  // CHECK-NEXT: t = dpct::library_data_t::complex_int4;
  // CHECK-NEXT: t = dpct::library_data_t::real_uint4;
  // CHECK-NEXT: t = dpct::library_data_t::complex_uint4;
  // CHECK-NEXT: t = dpct::library_data_t::real_int8;
  // CHECK-NEXT: t = dpct::library_data_t::complex_int8;
  // CHECK-NEXT: t = dpct::library_data_t::real_uint8;
  // CHECK-NEXT: t = dpct::library_data_t::complex_uint8;
  // CHECK-NEXT: t = dpct::library_data_t::real_int16;
  // CHECK-NEXT: t = dpct::library_data_t::complex_int16;
  // CHECK-NEXT: t = dpct::library_data_t::real_uint16;
  // CHECK-NEXT: t = dpct::library_data_t::complex_uint16;
  // CHECK-NEXT: t = dpct::library_data_t::real_int32;
  // CHECK-NEXT: t = dpct::library_data_t::complex_int32;
  // CHECK-NEXT: t = dpct::library_data_t::real_uint32;
  // CHECK-NEXT: t = dpct::library_data_t::complex_uint32;
  // CHECK-NEXT: t = dpct::library_data_t::real_int64;
  // CHECK-NEXT: t = dpct::library_data_t::complex_int64;
  // CHECK-NEXT: t = dpct::library_data_t::real_uint64;
  // CHECK-NEXT: t = dpct::library_data_t::complex_uint64;
  // CHECK-NEXT: t = dpct::library_data_t::real_f8_e4m3;
  // CHECK-NEXT: t = dpct::library_data_t::real_f8_e5m2;
  t = CUDA_R_16F;
  t = CUDA_C_16F;
  t = CUDA_R_16BF;
  t = CUDA_C_16BF;
  t = CUDA_R_32F;
  t = CUDA_C_32F;
  t = CUDA_R_64F;
  t = CUDA_C_64F;
  t = CUDA_R_4I;
  t = CUDA_C_4I;
  t = CUDA_R_4U;
  t = CUDA_C_4U;
  t = CUDA_R_8I;
  t = CUDA_C_8I;
  t = CUDA_R_8U;
  t = CUDA_C_8U;
  t = CUDA_R_16I;
  t = CUDA_C_16I;
  t = CUDA_R_16U;
  t = CUDA_C_16U;
  t = CUDA_R_32I;
  t = CUDA_C_32I;
  t = CUDA_R_32U;
  t = CUDA_C_32U;
  t = CUDA_R_64I;
  t = CUDA_C_64I;
  t = CUDA_R_64U;
  t = CUDA_C_64U;
  t = CUDA_R_8F_E4M3;
  t = CUDA_R_8F_E5M2;
}
