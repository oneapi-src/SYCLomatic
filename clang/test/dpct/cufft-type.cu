// RUN: dpct --format-range=none -out-root %T/cufft-type %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft-type/cufft-type.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>

size_t size;

int main() {
  //CHECK:float fftreal;
  //CHECK-NEXT:double fftdreal;
  //CHECK-NEXT:sycl::float2 fftcomplex;
  //CHECK-NEXT:sycl::double2 fftdcomplex;
  //CHECK-NEXT:sycl::float2 ccomplex;
  //CHECK-NEXT:sycl::double2 cdcomplex;
  //CHECK-NEXT:size = sizeof(float);
  //CHECK-NEXT:size = sizeof(double);
  //CHECK-NEXT:size = sizeof(sycl::float2);
  //CHECK-NEXT:size = sizeof(sycl::double2);
  //CHECK-NEXT:size = sizeof(sycl::float2);
  //CHECK-NEXT:size = sizeof(sycl::double2);
  cufftReal fftreal;
  cufftDoubleReal fftdreal;
  cufftComplex fftcomplex;
  cufftDoubleComplex fftdcomplex;
  cuComplex ccomplex;
  cuDoubleComplex cdcomplex;
  size = sizeof(cufftReal);
  size = sizeof(cufftDoubleReal);
  size = sizeof(cufftComplex);
  size = sizeof(cufftDoubleComplex);
  size = sizeof(cuComplex);
  size = sizeof(cuDoubleComplex);

  //CHECK:int forward = dpct::fft::fft_direction::forward;
  //CHECK-NEXT:int inverse = dpct::fft::fft_direction::backward;
  int forward = CUFFT_FORWARD;
  int inverse = CUFFT_INVERSE;

  //CHECK:dpct::fft::fft_type fftt_t;
  //CHECK-NEXT:dpct::fft::fft_type fftt;
  //CHECK-NEXT:size = sizeof(dpct::fft::fft_type);
  //CHECK-NEXT:size = sizeof(dpct::fft::fft_type);
  //CHECK-NEXT:fftt = dpct::fft::fft_type::real_float_to_complex_float;
  //CHECK-NEXT:fftt = dpct::fft::fft_type::complex_float_to_real_float;
  //CHECK-NEXT:fftt = dpct::fft::fft_type::complex_float_to_complex_float;
  //CHECK-NEXT:fftt = dpct::fft::fft_type::real_double_to_complex_double;
  //CHECK-NEXT:fftt = dpct::fft::fft_type::complex_double_to_real_double;
  //CHECK-NEXT:fftt = dpct::fft::fft_type::complex_double_to_complex_double;
  cufftType_t fftt_t;
  cufftType fftt;
  size = sizeof(cufftType_t);
  size = sizeof(cufftType);
  fftt = CUFFT_R2C;
  fftt = CUFFT_C2R;
  fftt = CUFFT_C2C;
  fftt = CUFFT_D2Z;
  fftt = CUFFT_Z2D;
  fftt = CUFFT_Z2Z;

  //CHECK:dpct::fft::fft_engine* ffth;
  //CHECK-NEXT:size = sizeof(dpct::fft::fft_engine*);
  cufftHandle ffth;
  size = sizeof(cufftHandle);

  //CHECK:int fftr_t;
  //CHECK-NEXT:int fftr;
  //CHECK-NEXT:size = sizeof(int);
  //CHECK-NEXT:size = sizeof(int);
  //CHECK-NEXT:fftr = 0;
  //CHECK-NEXT:fftr = 1;
  //CHECK-NEXT:fftr = 2;
  //CHECK-NEXT:fftr = 3;
  //CHECK-NEXT:fftr = 4;
  //CHECK-NEXT:fftr = 5;
  //CHECK-NEXT:fftr = 6;
  //CHECK-NEXT:fftr = 7;
  //CHECK-NEXT:fftr = 8;
  //CHECK-NEXT:fftr = 9;
  //CHECK-NEXT:fftr = 10;
  //CHECK-NEXT:fftr = 11;
  //CHECK-NEXT:fftr = 12;
  //CHECK-NEXT:fftr = 13;
  //CHECK-NEXT:fftr = 14;
  //CHECK-NEXT:fftr = 15;
  //CHECK-NEXT:fftr = 16;
  cufftResult_t fftr_t;
  cufftResult fftr;
  size = sizeof(cufftResult_t);
  size = sizeof(cufftResult);
  fftr = CUFFT_SUCCESS;
  fftr = CUFFT_INVALID_PLAN;
  fftr = CUFFT_ALLOC_FAILED;
  fftr = CUFFT_INVALID_TYPE;
  fftr = CUFFT_INVALID_VALUE;
  fftr = CUFFT_INTERNAL_ERROR;
  fftr = CUFFT_EXEC_FAILED;
  fftr = CUFFT_SETUP_FAILED;
  fftr = CUFFT_INVALID_SIZE;
  fftr = CUFFT_UNALIGNED_DATA;
  fftr = CUFFT_INCOMPLETE_PARAMETER_LIST;
  fftr = CUFFT_INVALID_DEVICE;
  fftr = CUFFT_PARSE_ERROR;
  fftr = CUFFT_NO_WORKSPACE;
  fftr = CUFFT_NOT_IMPLEMENTED;
  fftr = CUFFT_LICENSE_ERROR;
  fftr = CUFFT_NOT_SUPPORTED;

  return 0;
}


//CHECK:template<
//CHECK-NEXT:typename A = float,
//CHECK-NEXT:typename B = double,
//CHECK-NEXT:typename C = sycl::float2,
//CHECK-NEXT:typename D = sycl::double2,
//CHECK-NEXT:typename E = sycl::float2,
//CHECK-NEXT:typename F = sycl::double2,
//CHECK-NEXT:typename G = dpct::fft::fft_type,
//CHECK-NEXT:typename H = dpct::fft::fft_type,
//CHECK-NEXT:typename I = dpct::fft::fft_engine*,
//CHECK-NEXT:typename J = int,
//CHECK-NEXT:typename K = int>
//CHECK-NEXT:void foo1(
//CHECK-NEXT:float a,
//CHECK-NEXT:double b,
//CHECK-NEXT:sycl::float2 c,
//CHECK-NEXT:sycl::double2 d,
//CHECK-NEXT:sycl::float2 e,
//CHECK-NEXT:sycl::double2 f,
//CHECK-NEXT:dpct::fft::fft_type g,
//CHECK-NEXT:dpct::fft::fft_type h,
//CHECK-NEXT:dpct::fft::fft_engine* i,
//CHECK-NEXT:int j,
//CHECK-NEXT:int k
//CHECK-NEXT:){}
template<
typename A = cufftReal,
typename B = cufftDoubleReal,
typename C = cufftComplex,
typename D = cufftDoubleComplex,
typename E = cuComplex,
typename F = cuDoubleComplex,
typename G = cufftType_t,
typename H = cufftType,
typename I = cufftHandle,
typename J = cufftResult_t,
typename K = cufftResult>
void foo1(
cufftReal a,
cufftDoubleReal b,
cufftComplex c,
cufftDoubleComplex d,
cuComplex e,
cuDoubleComplex f,
cufftType_t g,
cufftType h,
cufftHandle i,
cufftResult_t j,
cufftResult k
){}


//CHECK:template<
//CHECK-NEXT:dpct::fft::fft_type A1 = dpct::fft::fft_type::real_float_to_complex_float,
//CHECK-NEXT:dpct::fft::fft_type A2 = dpct::fft::fft_type::complex_float_to_real_float,
//CHECK-NEXT:dpct::fft::fft_type A3 = dpct::fft::fft_type::complex_float_to_complex_float,
//CHECK-NEXT:dpct::fft::fft_type A4 = dpct::fft::fft_type::real_double_to_complex_double,
//CHECK-NEXT:dpct::fft::fft_type A5 = dpct::fft::fft_type::complex_double_to_real_double,
//CHECK-NEXT:dpct::fft::fft_type A6 = dpct::fft::fft_type::complex_double_to_complex_double,
//CHECK-NEXT:int B1 = 0,
//CHECK-NEXT:int B2 = 1,
//CHECK-NEXT:int B3 = 2,
//CHECK-NEXT:int B4 = 3,
//CHECK-NEXT:int B5 = 4,
//CHECK-NEXT:int B6 = 5,
//CHECK-NEXT:int B7 = 6,
//CHECK-NEXT:int B8 = 7,
//CHECK-NEXT:int B9 = 8,
//CHECK-NEXT:int B10 = 9,
//CHECK-NEXT:int B11 = 10,
//CHECK-NEXT:int B12 = 11,
//CHECK-NEXT:int B13 = 12,
//CHECK-NEXT:int B14 = 13,
//CHECK-NEXT:int B15 = 14,
//CHECK-NEXT:int B16 = 15,
//CHECK-NEXT:int B17 = 16>
//CHECK-NEXT:void foo2(
//CHECK-NEXT:dpct::fft::fft_type a1 = dpct::fft::fft_type::real_float_to_complex_float,
//CHECK-NEXT:dpct::fft::fft_type a2 = dpct::fft::fft_type::complex_float_to_real_float,
//CHECK-NEXT:dpct::fft::fft_type a3 = dpct::fft::fft_type::complex_float_to_complex_float,
//CHECK-NEXT:dpct::fft::fft_type a4 = dpct::fft::fft_type::real_double_to_complex_double,
//CHECK-NEXT:dpct::fft::fft_type a5 = dpct::fft::fft_type::complex_double_to_real_double,
//CHECK-NEXT:dpct::fft::fft_type a6 = dpct::fft::fft_type::complex_double_to_complex_double,
//CHECK-NEXT:int b1 = 0,
//CHECK-NEXT:int b2 = 1,
//CHECK-NEXT:int b3 = 2,
//CHECK-NEXT:int b4 = 3,
//CHECK-NEXT:int b5 = 4,
//CHECK-NEXT:int b6 = 5,
//CHECK-NEXT:int b7 = 6,
//CHECK-NEXT:int b8 = 7,
//CHECK-NEXT:int b9 = 8,
//CHECK-NEXT:int b10 = 9,
//CHECK-NEXT:int b11 = 10,
//CHECK-NEXT:int b12 = 11,
//CHECK-NEXT:int b13 = 12,
//CHECK-NEXT:int b14 = 13,
//CHECK-NEXT:int b15 = 14,
//CHECK-NEXT:int b16 = 15,
//CHECK-NEXT:int b17 = 16
//CHECK-NEXT:){}
template<
cufftType A1 = CUFFT_R2C,
cufftType A2 = CUFFT_C2R,
cufftType A3 = CUFFT_C2C,
cufftType A4 = CUFFT_D2Z,
cufftType A5 = CUFFT_Z2D,
cufftType A6 = CUFFT_Z2Z,
cufftResult B1 = CUFFT_SUCCESS,
cufftResult B2 = CUFFT_INVALID_PLAN,
cufftResult B3 = CUFFT_ALLOC_FAILED,
cufftResult B4 = CUFFT_INVALID_TYPE,
cufftResult B5 = CUFFT_INVALID_VALUE,
cufftResult B6 = CUFFT_INTERNAL_ERROR,
cufftResult B7 = CUFFT_EXEC_FAILED,
cufftResult B8 = CUFFT_SETUP_FAILED,
cufftResult B9 = CUFFT_INVALID_SIZE,
cufftResult B10 = CUFFT_UNALIGNED_DATA,
cufftResult B11 = CUFFT_INCOMPLETE_PARAMETER_LIST,
cufftResult B12 = CUFFT_INVALID_DEVICE,
cufftResult B13 = CUFFT_PARSE_ERROR,
cufftResult B14 = CUFFT_NO_WORKSPACE,
cufftResult B15 = CUFFT_NOT_IMPLEMENTED,
cufftResult B16 = CUFFT_LICENSE_ERROR,
cufftResult B17 = CUFFT_NOT_SUPPORTED>
void foo2(
cufftType a1 = CUFFT_R2C,
cufftType a2 = CUFFT_C2R,
cufftType a3 = CUFFT_C2C,
cufftType a4 = CUFFT_D2Z,
cufftType a5 = CUFFT_Z2D,
cufftType a6 = CUFFT_Z2Z,
cufftResult b1 = CUFFT_SUCCESS,
cufftResult b2 = CUFFT_INVALID_PLAN,
cufftResult b3 = CUFFT_ALLOC_FAILED,
cufftResult b4 = CUFFT_INVALID_TYPE,
cufftResult b5 = CUFFT_INVALID_VALUE,
cufftResult b6 = CUFFT_INTERNAL_ERROR,
cufftResult b7 = CUFFT_EXEC_FAILED,
cufftResult b8 = CUFFT_SETUP_FAILED,
cufftResult b9 = CUFFT_INVALID_SIZE,
cufftResult b10 = CUFFT_UNALIGNED_DATA,
cufftResult b11 = CUFFT_INCOMPLETE_PARAMETER_LIST,
cufftResult b12 = CUFFT_INVALID_DEVICE,
cufftResult b13 = CUFFT_PARSE_ERROR,
cufftResult b14 = CUFFT_NO_WORKSPACE,
cufftResult b15 = CUFFT_NOT_IMPLEMENTED,
cufftResult b16 = CUFFT_LICENSE_ERROR,
cufftResult b17 = CUFFT_NOT_SUPPORTED
){}


//CHECK:template<typename T>
//CHECK-NEXT:float foo3(){}
template<typename T>
cufftReal foo3(){}

//CHECK:template<typename T>
//CHECK-NEXT:double foo4(){}
template<typename T>
cufftDoubleReal foo4(){}

//CHECK:template<typename T>
//CHECK-NEXT:sycl::float2 foo5(){}
template<typename T>
cufftComplex foo5(){}

//CHECK:template<typename T>
//CHECK-NEXT:sycl::double2 foo6(){}
template<typename T>
cufftDoubleComplex foo6(){}

//CHECK:template<typename T>
//CHECK-NEXT:sycl::float2 foo7(){}
template<typename T>
cuComplex foo7(){}

//CHECK:template<typename T>
//CHECK-NEXT:sycl::double2 foo8(){}
template<typename T>
cuDoubleComplex foo8(){}

//CHECK:template<typename T>
//CHECK-NEXT:dpct::fft::fft_type foo9(){}
template<typename T>
cufftType_t foo9(){}

//CHECK:template<typename T>
//CHECK-NEXT:dpct::fft::fft_type foo10(){}
template<typename T>
cufftType foo10(){}

//CHECK:template<typename T>
//CHECK-NEXT:dpct::fft::fft_engine* foo11(){}
template<typename T>
cufftHandle foo11(){}

//CHECK:template<typename T>
//CHECK-NEXT:int foo12(){}
template<typename T>
cufftResult_t foo12(){}

//CHECK:template<typename T>
//CHECK-NEXT:int foo13(){}
template<typename T>
cufftResult foo13(){}

//     CHECK:void bar1(dpct::fft::fft_engine* const &aaa) {}
//CHECK-NEXT:void bar2(dpct::fft::fft_engine* const &aaa) {}
//CHECK-NEXT:void bar3(dpct::fft::fft_engine* const aaa) {}
//CHECK-NEXT:void bar4(dpct::fft::fft_engine* const aaa) {}
void bar1(cufftHandle const &aaa) {}
void bar2(const cufftHandle &aaa) {}
void bar3(cufftHandle const aaa) {}
void bar4(const cufftHandle aaa) {}

class MyStruct {
//     CHECK: ::dpct::fft::fft_engine* handle;
//CHECK-NEXT: ::dpct::fft::fft_engine* & foo() { return handle; }
//CHECK-NEXT: ::dpct::fft::fft_engine* const & foo() const { return handle; }
  ::cufftHandle handle;
  ::cufftHandle & foo() { return handle; }
  const ::cufftHandle & foo() const { return handle; }
};
