//===--------------- CallExprRewriterCUBLASLevel2.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUBLAS.h"

using namespace clang::dpct;

RewriterMap dpct::createCUBLASLevel2RewriterMap() {
  return RewriterMap{
#define GEMV(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC,                                                                   \
      CALL("oneapi::mkl::blas::column_major::gemv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           ARG(2), ARG(3), SCALAR_INPUT(ARG(4), #TYPE1),                       \
           BUFFER_OR_USM_PTR(ARG(5), #TYPE1), ARG(6),                          \
           BUFFER_OR_USM_PTR(ARG(7), #TYPE1), ARG(8),                          \
           SCALAR_INPUT(ARG(9), #TYPE1), BUFFER_OR_USM_PTR(ARG(10), #TYPE1),   \
           ARG(11))))
      // cublasSgemv_v2
      GEMV(cublasSgemv_v2, float)
      // cublasDgemv_v2
      GEMV(cublasDgemv_v2, double)
      // cublasCgemv_v2
      GEMV(cublasCgemv_v2, std::complex<float>)
      // cublasZgemv_v2
      GEMV(cublasZgemv_v2, std::complex<double>)
      // cublasSgemv_v2_64
      GEMV(cublasSgemv_v2_64, float)
      // cublasDgemv_v2_64
      GEMV(cublasDgemv_v2_64, double)
      // cublasCgemv_v2_64
      GEMV(cublasCgemv_v2_64, std::complex<float>)
      // cublasZgemv_v2_64
      GEMV(cublasZgemv_v2_64, std::complex<double>)
#undef GEMV

#define GBMV(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC,                                                                   \
      CALL("oneapi::mkl::blas::column_major::gbmv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           ARG(2), ARG(3), ARG(4), ARG(5), SCALAR_INPUT(ARG(6), #TYPE1),       \
           BUFFER_OR_USM_PTR(ARG(7), #TYPE1), ARG(8),                          \
           BUFFER_OR_USM_PTR(ARG(9), #TYPE1), ARG(10),                         \
           SCALAR_INPUT(ARG(11), #TYPE1), BUFFER_OR_USM_PTR(ARG(12), #TYPE1),  \
           ARG(13))))
      // cublasSgbmv_v2
      GBMV(cublasSgbmv_v2, float)
      // cublasDgbmv_v2
      GBMV(cublasDgbmv_v2, double)
      // cublasCgbmv_v2
      GBMV(cublasCgbmv_v2, std::complex<float>)
      // cublasZgbmv_v2
      GBMV(cublasZgbmv_v2, std::complex<double>)
      // cublasSgbmv_v2_64
      GBMV(cublasSgbmv_v2_64, float)
      // cublasDgbmv_v2_64
      GBMV(cublasDgbmv_v2_64, double)
      // cublasCgbmv_v2_64
      GBMV(cublasCgbmv_v2_64, std::complex<float>)
      // cublasZgbmv_v2_64
      GBMV(cublasZgbmv_v2_64, std::complex<double>)
#undef GBMV

#define TRMV(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC,                                                                   \
      CALL("oneapi::mkl::blas::column_major::trmv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),    \
           ARG(4), BUFFER_OR_USM_PTR(ARG(5), #TYPE1), ARG(6),                  \
           BUFFER_OR_USM_PTR(ARG(7), #TYPE1), ARG(8))))
      // cublasStrmv_v2
      TRMV(cublasStrmv_v2, float)
      // cublasDtrmv_v2
      TRMV(cublasDtrmv_v2, double)
      // cublasCtrmv_v2
      TRMV(cublasCtrmv_v2, std::complex<float>)
      // cublasZtrmv_v2
      TRMV(cublasZtrmv_v2, std::complex<double>)
      // cublasStrmv_v2_64
      TRMV(cublasStrmv_v2_64, float)
      // cublasDtrmv_v2_64
      TRMV(cublasDtrmv_v2_64, double)
      // cublasCtrmv_v2_64
      TRMV(cublasCtrmv_v2_64, std::complex<float>)
      // cublasZtrmv_v2_64
      TRMV(cublasZtrmv_v2_64, std::complex<double>)
#undef TRMV

#define TBMV(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC,                                                                   \
      CALL("oneapi::mkl::blas::column_major::tbmv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),    \
           ARG(4), ARG(5), BUFFER_OR_USM_PTR(ARG(6), #TYPE1), ARG(7),          \
           BUFFER_OR_USM_PTR(ARG(8), #TYPE1), ARG(9))))
      // cublasStbmv_v2
      TBMV(cublasStbmv_v2, float)
      // cublasDtbmv_v2
      TBMV(cublasDtbmv_v2, double)
      // cublasCtbmv_v2
      TBMV(cublasCtbmv_v2, std::complex<float>)
      // cublasZtbmv_v2
      TBMV(cublasZtbmv_v2, std::complex<double>)
      // cublasStbmv_v2_64
      TBMV(cublasStbmv_v2_64, float)
      // cublasDtbmv_v2_64
      TBMV(cublasDtbmv_v2_64, double)
      // cublasCtbmv_v2_64
      TBMV(cublasCtbmv_v2_64, std::complex<float>)
      // cublasZtbmv_v2_64
      TBMV(cublasZtbmv_v2_64, std::complex<double>)
#undef TBMV

#define TPMV(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC,                                                                   \
      CALL("oneapi::mkl::blas::column_major::tpmv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),    \
           ARG(4), BUFFER_OR_USM_PTR(ARG(5), #TYPE1),                          \
           BUFFER_OR_USM_PTR(ARG(6), #TYPE1), ARG(7))))
      // cublasStpmv_v2
      TPMV(cublasStpmv_v2, float)
      // cublasDtpmv_v2
      TPMV(cublasDtpmv_v2, double)
      // cublasCtpmv_v2
      TPMV(cublasCtpmv_v2, std::complex<float>)
      // cublasZtpmv_v2
      TPMV(cublasZtpmv_v2, std::complex<double>)
      // cublasStpmv_v2_64
      TPMV(cublasStpmv_v2_64, float)
      // cublasDtpmv_v2_64
      TPMV(cublasDtpmv_v2_64, double)
      // cublasCtpmv_v2_64
      TPMV(cublasCtpmv_v2_64, std::complex<float>)
      // cublasZtpmv_v2_64
      TPMV(cublasZtpmv_v2_64, std::complex<double>)
#undef TPMV

#define TRSV(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC,                                                                   \
      CALL("oneapi::mkl::blas::column_major::trsv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),    \
           ARG(4), BUFFER_OR_USM_PTR(ARG(5), #TYPE1), ARG(6),                  \
           BUFFER_OR_USM_PTR(ARG(7), #TYPE1), ARG(8))))
      // cublasStrsv_v2
      TRSV(cublasStrsv_v2, float)
      // cublasDtrsv_v2
      TRSV(cublasDtrsv_v2, double)
      // cublasCtrsv_v2
      TRSV(cublasCtrsv_v2, std::complex<float>)
      // cublasZtrsv_v2
      TRSV(cublasZtrsv_v2, std::complex<double>)
      // cublasStrsv_v2_64
      TRSV(cublasStrsv_v2_64, float)
      // cublasDtrsv_v2_64
      TRSV(cublasDtrsv_v2_64, double)
      // cublasCtrsv_v2_64
      TRSV(cublasCtrsv_v2_64, std::complex<float>)
      // cublasZtrsv_v2_64
      TRSV(cublasZtrsv_v2_64, std::complex<double>)
#undef TRSV

#define TPSV(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC,                                                                   \
      CALL("oneapi::mkl::blas::column_major::tpsv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),    \
           ARG(4), BUFFER_OR_USM_PTR(ARG(5), #TYPE1),                          \
           BUFFER_OR_USM_PTR(ARG(6), #TYPE1), ARG(7))))
      // cublasStpsv_v2
      TPSV(cublasStpsv_v2, float)
      // cublasDtpsv_v2
      TPSV(cublasDtpsv_v2, double)
      // cublasCtpsv_v2
      TPSV(cublasCtpsv_v2, std::complex<float>)
      // cublasZtpsv_v2
      TPSV(cublasZtpsv_v2, std::complex<double>)
      // cublasStpsv_v2_64
      TPSV(cublasStpsv_v2_64, float)
      // cublasDtpsv_v2_64
      TPSV(cublasDtpsv_v2_64, double)
      // cublasCtpsv_v2_64
      TPSV(cublasCtpsv_v2_64, std::complex<float>)
      // cublasZtpsv_v2_64
      TPSV(cublasZtpsv_v2_64, std::complex<double>)
#undef TPSV

#define HER2(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::her2",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(4), TYPE), ARG(5),                            \
           BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),                            \
           BUFFER_OR_USM_PTR(ARG(8), TYPE), ARG(9))))
      // cublasCher2_v2
      HER2(cublasCher2_v2, "std::complex<float>")
      // cublasZher2_v2
      HER2(cublasZher2_v2, "std::complex<double>")
      // cublasCher2_v2_64
      HER2(cublasCher2_v2_64, "std::complex<float>")
      // cublasZher2_v2_64
      HER2(cublasZher2_v2_64, "std::complex<double>")
#undef HER2

#define SPR2(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::spr2",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(4), TYPE), ARG(5),                            \
           BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),                            \
           BUFFER_OR_USM_PTR(ARG(8), TYPE))))
      // cublasSspr2_v2
      SPR2(cublasSspr2_v2, "float")
      // cublasDspr2_v2
      SPR2(cublasDspr2_v2, "double")
      // cublasSspr2_v2_64
      SPR2(cublasSspr2_v2_64, "float")
      // cublasDspr2_v2_64
      SPR2(cublasDspr2_v2_64, "double")
#undef SPR2

#define HPR2(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::hpr2",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(4), TYPE), ARG(5),                            \
           BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),                            \
           BUFFER_OR_USM_PTR(ARG(8), TYPE))))
      // cublasChpr2_v2
      HPR2(cublasChpr2_v2, "std::complex<float>")
      // cublasZhpr2_v2
      HPR2(cublasZhpr2_v2, "std::complex<double>")
      // cublasChpr2_v2_64
      HPR2(cublasChpr2_v2_64, "std::complex<float>")
      // cublasZhpr2_v2_64
      HPR2(cublasZhpr2_v2_64, "std::complex<double>")
#undef HPR2

#define SYR(NAME, TYPE)                                                        \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::syr",                             \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(4), TYPE), ARG(5),                            \
           BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7))))
      // cublasSsyr_v2
      SYR(cublasSsyr_v2, "float")
      // cublasDsyr_v2
      SYR(cublasDsyr_v2, "double")
      // cublasCsyr_v2
      SYR(cublasCsyr_v2, "std::complex<float>")
      // cublasZsyr_v2
      SYR(cublasZsyr_v2, "std::complex<double>")
      // cublasSsyr_v2_64
      SYR(cublasSsyr_v2_64, "float")
      // cublasDsyr_v2_64
      SYR(cublasDsyr_v2_64, "double")
      // cublasCsyr_v2_64
      SYR(cublasCsyr_v2_64, "std::complex<float>")
      // cublasZsyr_v2_64
      SYR(cublasZsyr_v2_64, "std::complex<double>")
#undef SYR

#define HER(NAME, TYPE1, TYPE2)                                                \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::her",                             \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE1),                                \
           BUFFER_OR_USM_PTR(ARG(4), TYPE2), ARG(5),                           \
           BUFFER_OR_USM_PTR(ARG(6), TYPE2), ARG(7))))
      // cublasCher_v2
      HER(cublasCher_v2, "float", "std::complex<float>")
      // cublasZher_v2
      HER(cublasZher_v2, "double", "std::complex<double>")
      // cublasCher_v2_64
      HER(cublasCher_v2_64, "float", "std::complex<float>")
      // cublasZher_v2_64
      HER(cublasZher_v2_64, "double", "std::complex<double>")
#undef HER

#define SPR(NAME, TYPE)                                                        \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::spr",                             \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(4), TYPE), ARG(5),                            \
           BUFFER_OR_USM_PTR(ARG(6), TYPE))))
      // cublasSspr_v2
      SPR(cublasSspr_v2, "float")
      // cublasDspr_v2
      SPR(cublasDspr_v2, "double")
      // cublasSspr_v2_64
      SPR(cublasSspr_v2_64, "float")
      // cublasDspr_v2_64
      SPR(cublasDspr_v2_64, "double")
#undef SPR

#define HPR(NAME, TYPE1, TYPE2)                                                \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::hpr",                             \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE1),                                \
           BUFFER_OR_USM_PTR(ARG(4), TYPE2), ARG(5),                           \
           BUFFER_OR_USM_PTR(ARG(6), TYPE2))))
      // cublasChpr_v2
      HPR(cublasChpr_v2, "float", "std::complex<float>")
      // cublasZhpr_v2
      HPR(cublasZhpr_v2, "double", "std::complex<double>")
      // cublasChpr_v2_64
      HPR(cublasChpr_v2_64, "float", "std::complex<float>")
      // cublasZhpr_v2_64
      HPR(cublasZhpr_v2_64, "double", "std::complex<double>")
#undef HPR

#define TBSV(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::tbsv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),    \
           ARG(4), ARG(5), BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),            \
           BUFFER_OR_USM_PTR(ARG(8), TYPE), ARG(9))))
      // cublasStbsv_v2
      TBSV(cublasStbsv_v2, "float")
      // cublasDtbsv_v2
      TBSV(cublasDtbsv_v2, "double")
      // cublasCtbsv_v2
      TBSV(cublasCtbsv_v2, "std::complex<float>")
      // cublasZtbsv_v2
      TBSV(cublasZtbsv_v2, "std::complex<double>")
      // cublasStbsv_v2_64
      TBSV(cublasStbsv_v2_64, "float")
      // cublasDtbsv_v2_64
      TBSV(cublasDtbsv_v2_64, "double")
      // cublasCtbsv_v2_64
      TBSV(cublasCtbsv_v2_64, "std::complex<float>")
      // cublasZtbsv_v2_64
      TBSV(cublasZtbsv_v2_64, "std::complex<double>")
#undef TBSV

#define SYMV(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::symv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(4), TYPE), ARG(5),                            \
           BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),                            \
           SCALAR_INPUT(ARG(8), TYPE), BUFFER_OR_USM_PTR(ARG(9), TYPE),        \
           ARG(10))))
      // cublasSsymv_v2
      SYMV(cublasSsymv_v2, "float")
      // cublasDsymv_v2
      SYMV(cublasDsymv_v2, "double")
      // cublasCsymv_v2
      SYMV(cublasCsymv_v2, "std::complex<float>")
      // cublasZsymv_v2
      SYMV(cublasZsymv_v2, "std::complex<double>")
      // cublasSsymv_v2_64
      SYMV(cublasSsymv_v2_64, "float")
      // cublasDsymv_v2_64
      SYMV(cublasDsymv_v2_64, "double")
      // cublasCsymv_v2_64
      SYMV(cublasCsymv_v2_64, "std::complex<float>")
      // cublasZsymv_v2_64
      SYMV(cublasZsymv_v2_64, "std::complex<double>")
#undef SYMV

#define HEMV(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::hemv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(4), TYPE), ARG(5),                            \
           BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),                            \
           SCALAR_INPUT(ARG(8), TYPE), BUFFER_OR_USM_PTR(ARG(9), TYPE),        \
           ARG(10))))
      // cublasChemv_v2
      HEMV(cublasChemv_v2, "std::complex<float>")
      // cublasZhemv_v2
      HEMV(cublasZhemv_v2, "std::complex<double>")
      // cublasChemv_v2_64
      HEMV(cublasChemv_v2_64, "std::complex<float>")
      // cublasZhemv_v2_64
      HEMV(cublasZhemv_v2_64, "std::complex<double>")
#undef HEMV

#define SBMV(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::sbmv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), ARG(3), SCALAR_INPUT(ARG(4), TYPE),                         \
           BUFFER_OR_USM_PTR(ARG(5), TYPE), ARG(6),                            \
           BUFFER_OR_USM_PTR(ARG(7), TYPE), ARG(8),                            \
           SCALAR_INPUT(ARG(9), TYPE), BUFFER_OR_USM_PTR(ARG(10), TYPE),       \
           ARG(11))))
      // cublasSsbmv_v2
      SBMV(cublasSsbmv_v2, "float")
      // cublasDsbmv_v2
      SBMV(cublasDsbmv_v2, "double")
      // cublasSsbmv_v2_64
      SBMV(cublasSsbmv_v2_64, "float")
      // cublasDsbmv_v2_64
      SBMV(cublasDsbmv_v2_64, "double")
#undef SBMV

#define HBMV(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::hbmv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), ARG(3), SCALAR_INPUT(ARG(4), TYPE),                         \
           BUFFER_OR_USM_PTR(ARG(5), TYPE), ARG(6),                            \
           BUFFER_OR_USM_PTR(ARG(7), TYPE), ARG(8),                            \
           SCALAR_INPUT(ARG(9), TYPE), BUFFER_OR_USM_PTR(ARG(10), TYPE),       \
           ARG(11))))
      // cublasChbmv_v2
      HBMV(cublasChbmv_v2, "std::complex<float>")
      // cublasZhbmv_v2
      HBMV(cublasZhbmv_v2, "std::complex<double>")
      // cublasChbmv_v2_64
      HBMV(cublasChbmv_v2_64, "std::complex<float>")
      // cublasZhbmv_v2_64
      HBMV(cublasZhbmv_v2_64, "std::complex<double>")
#undef HBMV

#define SPMV(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::spmv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(4), TYPE), BUFFER_OR_USM_PTR(ARG(5), TYPE),   \
           ARG(6), SCALAR_INPUT(ARG(7), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(8), TYPE), ARG(9))))
      // cublasSspmv_v2
      SPMV(cublasSspmv_v2, "float")
      // cublasDspmv_v2
      SPMV(cublasDspmv_v2, "double")
      // cublasSspmv_v2_64
      SPMV(cublasSspmv_v2_64, "float")
      // cublasDspmv_v2_64
      SPMV(cublasDspmv_v2_64, "double")
#undef SPMV

#define HPMV(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::hpmv",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(4), TYPE), BUFFER_OR_USM_PTR(ARG(5), TYPE),   \
           ARG(6), SCALAR_INPUT(ARG(7), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(8), TYPE), ARG(9))))
      // cublasChpmv_v2
      HPMV(cublasChpmv_v2, "std::complex<float>")
      // cublasZhpmv_v2
      HPMV(cublasZhpmv_v2, "std::complex<double>")
      // cublasChpmv_v2_64
      HPMV(cublasChpmv_v2_64, "std::complex<float>")
      // cublasZhpmv_v2_64
      HPMV(cublasZhpmv_v2_64, "std::complex<double>")
#undef HPMV

#define GER(NAME, TYPE)                                                        \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL("oneapi::mkl::blas::column_major::ger",                      \
                  MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1), ARG(2),      \
                  SCALAR_INPUT(ARG(3), TYPE), BUFFER_OR_USM_PTR(ARG(4), TYPE), \
                  ARG(5), BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),             \
                  BUFFER_OR_USM_PTR(ARG(8), TYPE), ARG(9))))
      // cublasSger_v2
      GER(cublasSger_v2, "float")
      // cublasDger_v2
      GER(cublasDger_v2, "double")
      // cublasCger_v2
      GER(cublasSger_v2_64, "float")
      // cublasZger_v2
      GER(cublasDger_v2_64, "double")
#undef GER

#define GERU(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL("oneapi::mkl::blas::column_major::geru",                     \
                  MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1), ARG(2),      \
                  SCALAR_INPUT(ARG(3), TYPE), BUFFER_OR_USM_PTR(ARG(4), TYPE), \
                  ARG(5), BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),             \
                  BUFFER_OR_USM_PTR(ARG(8), TYPE), ARG(9))))
      // cublasCgeru_v2
      GERU(cublasCgeru_v2, "std::complex<float>")
      // cublasZgeru_v2
      GERU(cublasZgeru_v2, "std::complex<double>")
      // cublasCgeru_v2_64
      GERU(cublasCgeru_v2_64, "std::complex<float>")
      // cublasZgeru_v2_64
      GERU(cublasZgeru_v2_64, "std::complex<double>")
#undef GERU

#define GERC(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL("oneapi::mkl::blas::column_major::gerc",                     \
                  MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1), ARG(2),      \
                  SCALAR_INPUT(ARG(3), TYPE), BUFFER_OR_USM_PTR(ARG(4), TYPE), \
                  ARG(5), BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),             \
                  BUFFER_OR_USM_PTR(ARG(8), TYPE), ARG(9))))
      // cublasCgerc_v2
      GERC(cublasCgerc_v2, "std::complex<float>")
      // cublasZgerc_v2
      GERC(cublasZgerc_v2, "std::complex<double>")
      // cublasCgerc_v2_64
      GERC(cublasCgerc_v2_64, "std::complex<float>")
      // cublasZgerc_v2_64
      GERC(cublasZgerc_v2_64, "std::complex<double>")
#undef GERC

  };
}
