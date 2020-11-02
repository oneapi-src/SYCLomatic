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

  //CHECK:int forward = -1;
  //CHECK-NEXT:int inverse = 1;
  int forward = CUFFT_FORWARD;
  int inverse = CUFFT_INVERSE;

  //CHECK:int fftt_t;
  //CHECK-NEXT:int fftt;
  //CHECK-NEXT:size = sizeof(int);
  //CHECK-NEXT:size = sizeof(int);
  //CHECK-NEXT:fftt = 42;
  //CHECK-NEXT:fftt = 44;
  //CHECK-NEXT:fftt = 41;
  //CHECK-NEXT:fftt = 106;
  //CHECK-NEXT:fftt = 108;
  //CHECK-NEXT:fftt = 105;
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

  //CHECK:/*
  //CHECK-NEXT:DPCT1050:{{[0-9]+}}: The template argument of the FFT precision and domain type could not be deduced. You need to update this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:std::shared_ptr<oneapi::mkl::dft::descriptor<dpct_placeholder/*Fix the precision and domain type manually*/>> ffth;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1050:{{[0-9]+}}: The template argument of the FFT precision and domain type could not be deduced. You need to update this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:size = sizeof(std::shared_ptr<oneapi::mkl::dft::descriptor<dpct_placeholder/*Fix the precision and domain type manually*/>>);
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
//CHECK-NEXT:typename G = int,
//CHECK-NEXT:typename H = int,
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1050:2: The template argument of the FFT precision and domain type could not be deduced. You need to update this code.
//CHECK-NEXT:*/
//CHECK-NEXT:typename I = std::shared_ptr<oneapi::mkl::dft::descriptor<dpct_placeholder/*Fix the precision and domain type manually*/>>,
//CHECK-NEXT:typename J = int,
//CHECK-NEXT:typename K = int>
//CHECK-NEXT:void foo1(
//CHECK-NEXT:float a,
//CHECK-NEXT:double b,
//CHECK-NEXT:sycl::float2 c,
//CHECK-NEXT:sycl::double2 d,
//CHECK-NEXT:sycl::float2 e,
//CHECK-NEXT:sycl::double2 f,
//CHECK-NEXT:int g,
//CHECK-NEXT:int h,
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1050:3: The template argument of the FFT precision and domain type could not be deduced. You need to update this code.
//CHECK-NEXT:*/
//CHECK-NEXT:std::shared_ptr<oneapi::mkl::dft::descriptor<dpct_placeholder/*Fix the precision and domain type manually*/>> i,
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
//CHECK-NEXT:int A1 = 42,
//CHECK-NEXT:int A2 = 44,
//CHECK-NEXT:int A3 = 41,
//CHECK-NEXT:int A4 = 106,
//CHECK-NEXT:int A5 = 108,
//CHECK-NEXT:int A6 = 105,
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
//CHECK-NEXT:int a1 = 42,
//CHECK-NEXT:int a2 = 44,
//CHECK-NEXT:int a3 = 41,
//CHECK-NEXT:int a4 = 106,
//CHECK-NEXT:int a5 = 108,
//CHECK-NEXT:int a6 = 105,
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
//CHECK-NEXT:int foo9(){}
template<typename T>
cufftType_t foo9(){}

//CHECK:template<typename T>
//CHECK-NEXT:int foo10(){}
template<typename T>
cufftType foo10(){}

//CHECK:template<typename T>
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1050:4: The template argument of the FFT precision and domain type could not be deduced. You need to update this code.
//CHECK:*/
//CHECK-NEXT:std::shared_ptr<oneapi::mkl::dft::descriptor<dpct_placeholder/*Fix the precision and domain type manually*/>> foo11(){}
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
