//===--------------- APINamesCUBLAS.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCommon.h"
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasGetVersion_v2",
                       CALL(MapNames::getLibraryHelperNamespace() + "mkl_get_version",
                            ARG(MapNames::getLibraryHelperNamespace() + "version_field::major"), ARG(1)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasGetVersion",
                       CALL(MapNames::getLibraryHelperNamespace() + "mkl_get_version",
                            ARG(MapNames::getLibraryHelperNamespace() + "version_field::major"), ARG(0)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGN_FACTORY_ENTRY("cublasCreate_v2", DEREF(0),
                         NEW(MapNames::getLibraryHelperNamespace() +
                             "blas::descriptor"))))
ASSIGNABLE_FACTORY(DELETE_FACTORY_ENTRY("cublasDestroy_v2", ARG(0)))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cublasSetStream_v2", ARG(0), true,
                                             "set_queue", ARG(1)))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cublasGetStream_v2", DEREF(1),
                                        UO(UnaryOperatorKind::UO_AddrOf,
                                           PAREN(MEMBER_CALL(ARG(0), true,
                                                             "get_queue")))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSetKernelStream",
    CALL(MapNames::getLibraryHelperNamespace() + "blas::descriptor::set_saved_queue",
         ARG(0))))

ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cublasSetMathMode", ARG(0), true,
                                             "set_math_mode", ARG(1)))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cublasGetMathMode", DEREF(1),
                                        MEMBER_CALL(ARG(0), true,
                                                    "get_math_mode")))
#define SETGET_VECTOR(NAME)                                                    \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          #NAME,                                                               \
          CALL(MapNames::getLibraryHelperNamespace() + "blas::matrix_mem_copy", ARG(4), \
               ARG(2), ARG(5), ARG(3), ARG("1"), ARG(0), ARG(1)))))
SETGET_VECTOR(cublasSetVector)
SETGET_VECTOR(cublasGetVector)
SETGET_VECTOR(cublasSetVector_64)
SETGET_VECTOR(cublasGetVector_64)
#undef SETGET_VECTOR

#define SETGET_VECTOR_ASYNC(NAME)                                              \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(#NAME,                                                \
                         CALL(MapNames::getLibraryHelperNamespace() +          \
                                  "blas::matrix_mem_copy",                     \
                              ARG(4), ARG(2), ARG(5), ARG(3), ARG("1"),        \
                              ARG(0), ARG(1),                                  \
                              ARG(MapNames::getLibraryHelperNamespace() +      \
                                  "cs::memcpy_direction::automatic"),          \
                              DEREF(6), ARG("true")))))
SETGET_VECTOR_ASYNC(cublasSetVectorAsync)
SETGET_VECTOR_ASYNC(cublasGetVectorAsync)
SETGET_VECTOR_ASYNC(cublasSetVectorAsync_64)
SETGET_VECTOR_ASYNC(cublasGetVectorAsync_64)
#undef SETGET_VECTOR_ASYNC

#define SETGET_MATRIX(NAME)                                                    \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          #NAME,                                                               \
          CALL(MapNames::getLibraryHelperNamespace() + "blas::matrix_mem_copy", ARG(5), \
               ARG(3), ARG(6), ARG(4), ARG(0), ARG(1), ARG(2)))))
SETGET_MATRIX(cublasSetMatrix)
SETGET_MATRIX(cublasGetMatrix)
SETGET_MATRIX(cublasSetMatrix_64)
SETGET_MATRIX(cublasGetMatrix_64)
#undef SETGET_MATRIX

#define SETGET_MATRIX_ASYNC(NAME)                                              \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          #NAME, CALL(MapNames::getLibraryHelperNamespace() +                  \
                          "blas::matrix_mem_copy",                             \
                      ARG(5), ARG(3), ARG(6), ARG(4), ARG(0), ARG(1), ARG(2),  \
                      ARG(MapNames::getLibraryHelperNamespace() +              \
                          "cs::memcpy_direction::automatic"),                  \
                      DEREF(7), ARG("true")))))
SETGET_MATRIX_ASYNC(cublasSetMatrixAsync)
SETGET_MATRIX_ASYNC(cublasGetMatrixAsync)
SETGET_MATRIX_ASYNC(cublasSetMatrixAsync_64)
SETGET_MATRIX_ASYNC(cublasGetMatrixAsync_64)
#undef SETGET_MATRIX_ASYNC

#define GEMM(NAME, TYPE, IS_COMPLEX)                                           \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::gemm",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           ARG(3), ARG(4), ARG(5), SCALAR_INPUT(ARG(6), TYPE),                 \
           BUFFER_OR_USM_PTR(ARG(7), TYPE), ARG(8),                            \
           BUFFER_OR_USM_PTR(ARG(9), TYPE), ARG(10),                           \
           SCALAR_INPUT(ARG(11), TYPE), BUFFER_OR_USM_PTR(ARG(12), TYPE),      \
           ARG(13))))
GEMM(cublasHgemm, MapNames::getClNamespace() + "half", false)
GEMM(cublasSgemm_v2, "float", false)
GEMM(cublasDgemm_v2, "double", false)
GEMM(cublasCgemm_v2, "std::complex<float>", true)
GEMM(cublasZgemm_v2, "std::complex<double>", true)
GEMM(cublasHgemm_64, MapNames::getClNamespace() + "half", false)
GEMM(cublasSgemm_v2_64, "float", false)
GEMM(cublasDgemm_v2_64, "double", false)
GEMM(cublasCgemm_v2_64, "std::complex<float>", true)
GEMM(cublasZgemm_v2_64, "std::complex<double>", true)
#undef GEMM

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasNrm2Ex",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::nrm2", ARG(0),
             ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasNrm2Ex_64",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::nrm2", ARG(0),
             ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasDotEx",
                       CALL(MapNames::getLibraryHelperNamespace() + "blas::dot",
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasDotEx_64",
                       CALL(MapNames::getLibraryHelperNamespace() + "blas::dot",
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasDotcEx",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "blas::dotc",
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasDotcEx_64",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "blas::dotc",
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasScalEx",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::scal", ARG(0),
             ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasScalEx_64",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::scal", ARG(0),
             ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasAxpyEx",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "blas::axpy",
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasAxpyEx_64",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "blas::axpy",
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasRotEx",
                       CALL(MapNames::getLibraryHelperNamespace() + "blas::rot",
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasRotEx_64",
                       CALL(MapNames::getLibraryHelperNamespace() + "blas::rot",
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9), ARG(10)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasGemmBatchedEx",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch", ARG(0),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(7), BOOL(true),
                                       BOOL(false)),
             ARG(8), ARG(9),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(10), BOOL(true),
                                       BOOL(false)),
             ARG(11), ARG(12), ARG(13),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(14),
                                       BOOL(false), BOOL(false)),
             ARG(15), ARG(16), ARG(17), ARG(18)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasGemmBatchedEx_64",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch", ARG(0),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(7), BOOL(true),
                                       BOOL(false)),
             ARG(8), ARG(9),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(10), BOOL(true),
                                       BOOL(false)),
             ARG(11), ARG(12), ARG(13),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(14),
                                       BOOL(false), BOOL(false)),
             ARG(15), ARG(16), ARG(17), ARG(18)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasGemmStridedBatchedEx",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch", ARG(0),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17),
             ARG(18), ARG(19), ARG(20), ARG(21)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasGemmStridedBatchedEx_64",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch", ARG(0),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17),
             ARG(18), ARG(19), ARG(20), ARG(21)))))

#define SYRKX(FUNC)                                                            \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          #FUNC, CALL(MapNames::getLibraryHelperNamespace() + "blas::syrk", ARG(0),     \
                      BLAS_ENUM_ARG(                                           \
                          1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),   \
                      BLAS_ENUM_ARG(                                           \
                          2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),  \
                      ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9),  \
                      ARG(10), ARG(11), ARG(12)))))
SYRKX(cublasSsyrkx)
SYRKX(cublasDsyrkx)
SYRKX(cublasCsyrkx)
SYRKX(cublasZsyrkx)
SYRKX(cublasSsyrkx_64)
SYRKX(cublasDsyrkx_64)
SYRKX(cublasCsyrkx_64)
SYRKX(cublasZsyrkx_64)
#undef SYRKX

#define CHERKX(FUNC)                                                           \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          #FUNC, CALL(MapNames::getLibraryHelperNamespace() + "blas::herk", ARG(0),     \
                      BLAS_ENUM_ARG(                                           \
                          1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),   \
                      BLAS_ENUM_ARG(                                           \
                          2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),  \
                      ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9),  \
                      ARG(10), ARG(11), ARG(12)))))
CHERKX(cublasCherkx)
CHERKX(cublasZherkx)
CHERKX(cublasCherkx_64)
CHERKX(cublasZherkx_64)
#undef CHERKX

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSgeam",
    CALL("oneapi::mkl::blas::column_major::omatadd",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "float"), ARG(10),
         BUFFER_OR_USM_PTR(ARG(11), "float"), ARG(12))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDgeam",
    CALL("oneapi::mkl::blas::column_major::omatadd",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "double"), ARG(10),
         BUFFER_OR_USM_PTR(ARG(11), "double"), ARG(12))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCgeam",
    CALL("oneapi::mkl::blas::column_major::omatadd",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<float>"), ARG(10),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<float>"), ARG(12))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZgeam",
    CALL("oneapi::mkl::blas::column_major::omatadd",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<double>"), ARG(10),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSdgmm",
    CALL("oneapi::mkl::blas::column_major::dgmm", MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "float"), ARG(5),
         BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "float"), ARG(9))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDdgmm",
    CALL("oneapi::mkl::blas::column_major::dgmm", MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "double"), ARG(5),
         BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "double"), ARG(9))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCdgmm",
    CALL("oneapi::mkl::blas::column_major::dgmm", MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "std::complex<float>"),
         ARG(5), BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZdgmm",
    CALL("oneapi::mkl::blas::column_major::dgmm", MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "std::complex<double>"),
         ARG(5), BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasStrsmBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::trsm_batch", ARG(0),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
             ARG(5), ARG(6), ARG(7), CAST(makeLiteral("const void**"), ARG(8)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
             ARG(9), CAST(makeLiteral("void**"), ARG(10)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
             ARG(11), ARG(12),
             ARG(MapNames::getLibraryHelperNamespace() +
                 "library_data_t::real_float")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDtrsmBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::trsm_batch", ARG(0),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
             ARG(5), ARG(6), ARG(7), CAST(makeLiteral("const void**"), ARG(8)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"),
             ARG(9), CAST(makeLiteral("void**"), ARG(10)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"),
             ARG(11), ARG(12),
             ARG(MapNames::getLibraryHelperNamespace() +
                 "library_data_t::real_double")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCtrsmBatched",
        CALL(
            MapNames::getLibraryHelperNamespace() + "blas::trsm_batch", ARG(0),
            BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
            BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
            BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
            BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
            ARG(5), ARG(6), ARG(7), CAST(makeLiteral("const void**"), ARG(8)),
            ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float"),
            ARG(9), CAST(makeLiteral("void**"), ARG(10)),
            ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float"),
            ARG(11), ARG(12),
            ARG(MapNames::getLibraryHelperNamespace() +
                "library_data_t::complex_float")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZtrsmBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::trsm_batch", ARG(0),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
             ARG(5), ARG(6), ARG(7), CAST(makeLiteral("const void**"), ARG(8)),
             ARG(MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_double"),
             ARG(9), CAST(makeLiteral("void**"), ARG(10)),
             ARG(MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_double"),
             ARG(11), ARG(12),
             ARG(MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_double")))))

#define TRMM(FUNC)                                                             \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          #FUNC,                                                               \
          CALL(                                                                \
              MapNames::getLibraryHelperNamespace() + "blas::trmm", ARG(0),             \
              BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side), \
              BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo), \
              BLAS_ENUM_ARG(3,                                                 \
                            clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
              BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag), \
              ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),        \
              ARG(12), ARG(13)))))
TRMM(cublasStrmm_v2)
TRMM(cublasDtrmm_v2)
TRMM(cublasCtrmm_v2)
TRMM(cublasZtrmm_v2)
TRMM(cublasStrmm_v2_64)
TRMM(cublasDtrmm_v2_64)
TRMM(cublasCtrmm_v2_64)
TRMM(cublasZtrmm_v2_64)
#undef TRMM

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasHgemmBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch", ARG(0),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6),
             CAST(makeLiteral("const void**"), ARG(7)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_half"),
             ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_half"),
             ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_half"),
             ARG(13), ARG(14),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_half")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgemmBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch", ARG(0),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6),
             CAST(makeLiteral("const void**"), ARG(7)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
             ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
             ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
             ARG(13), ARG(14),
             ARG(MapNames::getLibraryHelperNamespace() +
                 "library_data_t::real_float")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgemmBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch", ARG(0),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6),
             CAST(makeLiteral("const void**"), ARG(7)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"),
             ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"),
             ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"),
             ARG(13), ARG(14),
             ARG(MapNames::getLibraryHelperNamespace() +
                 "library_data_t::real_double")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgemmBatched",
        CALL(
            MapNames::getLibraryHelperNamespace() + "blas::gemm_batch", ARG(0),
            BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
            BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
            ARG(3), ARG(4), ARG(5), ARG(6),
            CAST(makeLiteral("const void**"), ARG(7)),
            ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float"),
            ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
            ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float"),
            ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)),
            ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float"),
            ARG(13), ARG(14),
            ARG(MapNames::getLibraryHelperNamespace() +
                "library_data_t::complex_float")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgemmBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch", ARG(0),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6),
             CAST(makeLiteral("const void**"), ARG(7)),
             ARG(MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_double"),
             ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_double"),
             ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)),
             ARG(MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_double"),
             ARG(13), ARG(14),
             ARG(MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_double")))))

// getrf
WARNING_FACTORY_ENTRY("cublasSgetrfBatched",
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgetrfBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "getrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(2), BOOL(false), BOOL(false)), ARG(3), ARG(4), ARG(5), ARG(6))))),
Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4), ARG(MapNames::getLibraryHelperNamespace() + "getrf_batch_wrapper"), ARG("cublasSgetrfBatched"))
WARNING_FACTORY_ENTRY("cublasDgetrfBatched",
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgetrfBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "getrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(2), BOOL(false), BOOL(false)), ARG(3), ARG(4), ARG(5), ARG(6))))),
Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4), ARG(MapNames::getLibraryHelperNamespace() + "getrf_batch_wrapper"), ARG("cublasDgetrfBatched"))
WARNING_FACTORY_ENTRY("cublasCgetrfBatched",
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgetrfBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "getrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(2), BOOL(false), BOOL(false)), ARG(3), ARG(4), ARG(5), ARG(6))))),
Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4), ARG(MapNames::getLibraryHelperNamespace() + "getrf_batch_wrapper"), ARG("cublasCgetrfBatched"))
WARNING_FACTORY_ENTRY("cublasZgetrfBatched",
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgetrfBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "getrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(2), BOOL(false), BOOL(false)), ARG(3), ARG(4), ARG(5), ARG(6))))),
Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4), ARG(MapNames::getLibraryHelperNamespace() + "getrf_batch_wrapper"), ARG("cublasZgetrfBatched"))

// getrs
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgetrsBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "getrs_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(2), ARG(3), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(4), BOOL(true), BOOL(false)),
             ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(7), BOOL(false), BOOL(false)), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgetrsBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "getrs_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(2), ARG(3), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(4), BOOL(true), BOOL(false)),
             ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(7), BOOL(false), BOOL(false)), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgetrsBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "getrs_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(2), ARG(3), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(4), BOOL(true), BOOL(false)),
             ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(7), BOOL(false), BOOL(false)), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgetrsBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "getrs_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(2), ARG(3), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(4), BOOL(true), BOOL(false)),
             ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(7), BOOL(false), BOOL(false)), ARG(8), ARG(9), ARG(10)))))

// getri
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgetriBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "getri_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(2), BOOL(true), BOOL(false)),
             ARG(3), ARG(4), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7), ARG(8)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgetriBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "getri_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(2), BOOL(true), BOOL(false)),
             ARG(3), ARG(4), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7), ARG(8)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgetriBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "getri_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(2), BOOL(true), BOOL(false)),
             ARG(3), ARG(4), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7), ARG(8)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgetriBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "getri_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(2), BOOL(true), BOOL(false)),
             ARG(3), ARG(4), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7), ARG(8)))))

// geqrf
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgeqrfBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "geqrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), ARG(2), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(3), BOOL(false), BOOL(false)), ARG(4),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgeqrfBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "geqrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), ARG(2), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(3), BOOL(false), BOOL(false)), ARG(4),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgeqrfBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "geqrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), ARG(2), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(3), BOOL(false), BOOL(false)), ARG(4),
             DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgeqrfBatched",
        CALL(MapNames::getLibraryHelperNamespace() + "geqrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), ARG(2), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(3), BOOL(false), BOOL(false)), ARG(4),
             DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7)))))

#define GELS_BATCHED(NAME)                                                     \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          #NAME,                                                               \
          CALL(MapNames::getLibraryHelperNamespace() + "blas::gels_batch_wrapper",      \
               ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), \
               ARG(8), ARG(9), ARG(10), ARG(11)))))
GELS_BATCHED(cublasSgelsBatched)
GELS_BATCHED(cublasDgelsBatched)
GELS_BATCHED(cublasCgelsBatched)
GELS_BATCHED(cublasZgelsBatched)
#undef GELS_BATCHED

WARNING_FACTORY_ENTRY("cublasGetStatusString",
                      CALL_FACTORY_ENTRY("cublasGetStatusString",
                                         CALL(MapNames::getDpctNamespace() +
                                                  "get_error_string_dummy",
                                              ARG_WC(0))),
                      Diagnostics::ERROR_HANDLING_API_REPLACED_BY_DUMMY)

#define GEMM_EX(NAME, COMPUTE_TYPE)                                            \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          #NAME,                                                               \
          CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm", ARG(0),   \
               BLAS_ENUM_ARG(1,                                                \
                             clang::dpct::BLASEnumExpr::BLASEnumType::Trans),  \
               BLAS_ENUM_ARG(2,                                                \
                             clang::dpct::BLASEnumExpr::BLASEnumType::Trans),  \
               ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9),         \
               ARG(10), ARG(11), ARG(12), ARG(13), ARG(14), ARG(15), ARG(16),  \
               ARG(COMPUTE_TYPE)))))
GEMM_EX(cublasSgemmEx,
        MapNames::getLibraryHelperNamespace() + "library_data_t::real_float")
GEMM_EX(cublasCgemmEx,
        MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float")
GEMM_EX(cublasCgemm3mEx,
        MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float")
GEMM_EX(cublasGemmEx, 17)
GEMM_EX(cublasSgemmEx_64,
        MapNames::getLibraryHelperNamespace() + "library_data_t::real_float")
GEMM_EX(cublasCgemmEx_64,
        MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float")
GEMM_EX(cublasCgemm3mEx_64,
        MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float")
GEMM_EX(cublasGemmEx_64, 17)
#undef GEMM_EX

#define SYHERK(NAME, IS_HERMITIAN, COMPUTE_TYPE)                               \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          #NAME, CALL(MapNames::getLibraryHelperNamespace() +                  \
                          "blas::syherk<" + #IS_HERMITIAN + ">",               \
                      ARG(0),                                                  \
                      BLAS_ENUM_ARG(                                           \
                          1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),   \
                      BLAS_ENUM_ARG(                                           \
                          2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),  \
                      ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9),  \
                      ARG(10), ARG(11), ARG(12), ARG(COMPUTE_TYPE)))))
SYHERK(cublasCsyrkEx, false,
       MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float")
SYHERK(cublasCsyrk3mEx, false,
       MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float")
SYHERK(cublasCherkEx, true,
       MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float")
SYHERK(cublasCherk3mEx, true,
       MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float")
SYHERK(cublasCsyrkEx_64, false,
       MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float")
SYHERK(cublasCsyrk3mEx_64, false,
       MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float")
SYHERK(cublasCherkEx_64, true,
       MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float")
SYHERK(cublasCherk3mEx_64, true,
       MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float")
#undef SYHERK

#define SYRK(NAME, TYPE, IS_COMPLEX)                                           \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::syrk",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           ARG(3), ARG(4), SCALAR_INPUT(ARG(5), #TYPE),                        \
           BUFFER_OR_USM_PTR(ARG(6), #TYPE), ARG(7),                           \
           SCALAR_INPUT(ARG(8), #TYPE), BUFFER_OR_USM_PTR(ARG(9), #TYPE),      \
           ARG(10))))
SYRK(cublasSsyrk_v2, float, false)
SYRK(cublasDsyrk_v2, double, false)
SYRK(cublasCsyrk_v2, std::complex<float>, true)
SYRK(cublasZsyrk_v2, std::complex<double>, true)
SYRK(cublasSsyrk_v2_64, float, false)
SYRK(cublasDsyrk_v2_64, double, false)
SYRK(cublasCsyrk_v2_64, std::complex<float>, true)
SYRK(cublasZsyrk_v2_64, std::complex<double>, true)
#undef SYRK

#define SYMM(NAME, TYPE, IS_COMPLEX)                                           \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::symm",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(3), ARG(4), SCALAR_INPUT(ARG(5), #TYPE),                        \
           BUFFER_OR_USM_PTR(ARG(6), #TYPE), ARG(7),                           \
           BUFFER_OR_USM_PTR(ARG(8), #TYPE), ARG(9),                           \
           SCALAR_INPUT(ARG(10), #TYPE), BUFFER_OR_USM_PTR(ARG(11), #TYPE),    \
           ARG(12))))
SYMM(cublasSsymm_v2, float, false)
SYMM(cublasDsymm_v2, double, false)
SYMM(cublasCsymm_v2, std::complex<float>, true)
SYMM(cublasZsymm_v2, std::complex<double>, true)
SYMM(cublasSsymm_v2_64, float, false)
SYMM(cublasDsymm_v2_64, double, false)
SYMM(cublasCsymm_v2_64, std::complex<float>, true)
SYMM(cublasZsymm_v2_64, std::complex<double>, true)
#undef SYMM

#define TRSM(NAME, TYPE, IS_COMPLEX)                                           \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::trsm",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),    \
           ARG(5), ARG(6), SCALAR_INPUT(ARG(7), #TYPE),                        \
           BUFFER_OR_USM_PTR(ARG(8), #TYPE), ARG(9),                           \
           BUFFER_OR_USM_PTR(ARG(10), #TYPE), ARG(11))))
TRSM(cublasStrsm_v2, float, false)
TRSM(cublasDtrsm_v2, double, false)
TRSM(cublasCtrsm_v2, std::complex<float>, true)
TRSM(cublasZtrsm_v2, std::complex<double>, true)
TRSM(cublasStrsm_v2_64, float, false)
TRSM(cublasDtrsm_v2_64, double, false)
TRSM(cublasCtrsm_v2_64, std::complex<float>, true)
TRSM(cublasZtrsm_v2_64, std::complex<double>, true)
#undef TRSM

#define HEMM(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::hemm",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(3), ARG(4), SCALAR_INPUT(ARG(5), #TYPE),                        \
           BUFFER_OR_USM_PTR(ARG(6), #TYPE), ARG(7),                           \
           BUFFER_OR_USM_PTR(ARG(8), #TYPE), ARG(9),                           \
           SCALAR_INPUT(ARG(10), #TYPE), BUFFER_OR_USM_PTR(ARG(11), #TYPE),    \
           ARG(12))))
HEMM(cublasChemm_v2, std::complex<float>)
HEMM(cublasZhemm_v2, std::complex<double>)
HEMM(cublasChemm_v2_64, std::complex<float>)
HEMM(cublasZhemm_v2_64, std::complex<double>)
#undef HEMM

#define HERK(NAME, TYPE1, TYPE2)                                               \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::herk",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           ARG(3), ARG(4), SCALAR_INPUT(ARG(5), #TYPE2),                       \
           BUFFER_OR_USM_PTR(ARG(6), #TYPE1), ARG(7),                          \
           SCALAR_INPUT(ARG(8), #TYPE2), BUFFER_OR_USM_PTR(ARG(9), #TYPE1),    \
           ARG(10))))
HERK(cublasCherk_v2, std::complex<float>, float)
HERK(cublasZherk_v2, std::complex<double>, double)
HERK(cublasCherk_v2_64, std::complex<float>, float)
HERK(cublasZherk_v2_64, std::complex<double>, double)
#undef HERK

#define SYR2K(NAME, TYPE, IS_COMPLEX)                                          \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::syr2k",                           \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           ARG(3), ARG(4), SCALAR_INPUT(ARG(5), #TYPE),                        \
           BUFFER_OR_USM_PTR(ARG(6), #TYPE), ARG(7),                           \
           BUFFER_OR_USM_PTR(ARG(8), #TYPE), ARG(9),                           \
           SCALAR_INPUT(ARG(10), #TYPE), BUFFER_OR_USM_PTR(ARG(11), #TYPE),    \
           ARG(12))))
SYR2K(cublasSsyr2k_v2, float, false)
SYR2K(cublasDsyr2k_v2, double, false)
SYR2K(cublasCsyr2k_v2, std::complex<float>, true)
SYR2K(cublasZsyr2k_v2, std::complex<double>, true)
SYR2K(cublasSsyr2k_v2_64, float, false)
SYR2K(cublasDsyr2k_v2_64, double, false)
SYR2K(cublasCsyr2k_v2_64, std::complex<float>, true)
SYR2K(cublasZsyr2k_v2_64, std::complex<double>, true)
#undef SYR2K

#define HER2K(NAME, TYPE1, TYPE2)                                              \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::her2k",                           \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           ARG(3), ARG(4), SCALAR_INPUT(ARG(5), #TYPE1),                       \
           BUFFER_OR_USM_PTR(ARG(6), #TYPE1), ARG(7),                          \
           BUFFER_OR_USM_PTR(ARG(8), #TYPE1), ARG(9),                          \
           SCALAR_INPUT(ARG(10), #TYPE2), BUFFER_OR_USM_PTR(ARG(11), #TYPE1),  \
           ARG(12))))
HER2K(cublasCher2k_v2, std::complex<float>, float)
HER2K(cublasZher2k_v2, std::complex<double>, double)
HER2K(cublasCher2k_v2_64, std::complex<float>, float)
HER2K(cublasZher2k_v2_64, std::complex<double>, double)
#undef HER2K

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSgeam_64",
    CALL("oneapi::mkl::blas::column_major::omatadd",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "float"), ARG(10),
         BUFFER_OR_USM_PTR(ARG(11), "float"), ARG(12))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDgeam_64",
    CALL("oneapi::mkl::blas::column_major::omatadd",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "double"), ARG(10),
         BUFFER_OR_USM_PTR(ARG(11), "double"), ARG(12))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCgeam_64",
    CALL("oneapi::mkl::blas::column_major::omatadd",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<float>"), ARG(10),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<float>"), ARG(12))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZgeam_64",
    CALL("oneapi::mkl::blas::column_major::omatadd",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         CALL(MapNames::getLibraryHelperNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<double>"), ARG(10),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSdgmm_64",
    CALL("oneapi::mkl::blas::column_major::dgmm", MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "float"), ARG(5),
         BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "float"), ARG(9))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDdgmm_64",
    CALL("oneapi::mkl::blas::column_major::dgmm", MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "double"), ARG(5),
         BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "double"), ARG(9))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCdgmm_64",
    CALL("oneapi::mkl::blas::column_major::dgmm", MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "std::complex<float>"),
         ARG(5), BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZdgmm_64",
    CALL("oneapi::mkl::blas::column_major::dgmm", MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "std::complex<double>"),
         ARG(5), BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9))))

WARNING_FACTORY_ENTRY(
    "cublasIsamax_v2_64",
    LAMBDA_FACTORY_ENTRY(
        "cublasIsamax_v2_64",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamax",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "float"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)
WARNING_FACTORY_ENTRY(
    "cublasIdamax_v2_64",
    LAMBDA_FACTORY_ENTRY(
        "cublasIdamax_v2_64",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamax",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "double"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)
WARNING_FACTORY_ENTRY(
    "cublasIcamax_v2_64",
    LAMBDA_FACTORY_ENTRY(
        "cublasIcamax_v2_64",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamax",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "std::complex<float>"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)
WARNING_FACTORY_ENTRY(
    "cublasIzamax_v2_64",
    LAMBDA_FACTORY_ENTRY(
        "cublasIzamax_v2_64",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamax",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "std::complex<double>"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)

WARNING_FACTORY_ENTRY(
    "cublasIsamin_v2_64",
    LAMBDA_FACTORY_ENTRY(
        "cublasIsamin_v2_64",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamin",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "float"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)
WARNING_FACTORY_ENTRY(
    "cublasIdamin_v2_64",
    LAMBDA_FACTORY_ENTRY(
        "cublasIdamin_v2_64",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamin",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "double"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)
WARNING_FACTORY_ENTRY(
    "cublasIcamin_v2_64",
    LAMBDA_FACTORY_ENTRY(
        "cublasIcamin_v2_64",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamin",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "std::complex<float>"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)
WARNING_FACTORY_ENTRY(
    "cublasIzamin_v2_64",
    LAMBDA_FACTORY_ENTRY(
        "cublasIzamin_v2_64",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamin",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "std::complex<double>"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)

WARNING_FACTORY_ENTRY(
    "cublasIsamax_v2",
    LAMBDA_FACTORY_ENTRY(
        "cublasIsamax_v2",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int_to_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamax",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "float"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)
WARNING_FACTORY_ENTRY(
    "cublasIdamax_v2",
    LAMBDA_FACTORY_ENTRY(
        "cublasIdamax_v2",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int_to_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamax",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "double"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)
WARNING_FACTORY_ENTRY(
    "cublasIcamax_v2",
    LAMBDA_FACTORY_ENTRY(
        "cublasIcamax_v2",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int_to_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamax",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "std::complex<float>"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)
WARNING_FACTORY_ENTRY(
    "cublasIzamax_v2",
    LAMBDA_FACTORY_ENTRY(
        "cublasIzamax_v2",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int_to_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamax",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "std::complex<double>"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)

WARNING_FACTORY_ENTRY(
    "cublasIsamin_v2",
    LAMBDA_FACTORY_ENTRY(
        "cublasIsamin_v2",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int_to_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamin",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "float"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)
WARNING_FACTORY_ENTRY(
    "cublasIdamin_v2",
    LAMBDA_FACTORY_ENTRY(
        "cublasIdamin_v2",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int_to_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamin",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "double"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)
WARNING_FACTORY_ENTRY(
    "cublasIcamin_v2",
    LAMBDA_FACTORY_ENTRY(
        "cublasIcamin_v2",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int_to_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamin",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "std::complex<float>"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)
WARNING_FACTORY_ENTRY(
    "cublasIzamin_v2",
    LAMBDA_FACTORY_ENTRY(
        "cublasIzamin_v2",
        DECL(MapNames::getLibraryHelperNamespace() + "blas::wrapper_int_to_int64_out",
             "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"), ARG(4)),
        CALL("oneapi::mkl::blas::column_major::iamin",
             MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
             BUFFER_OR_USM_PTR(ARG(2), "std::complex<double>"), ARG(3),
             BUFFER_OR_USM_PTR(MEMBER_CALL(ARG("res_wrapper_ct4"), false,
                                           "get_ptr"),
                               "std::int64_t"),
             ARG("oneapi::mkl::index_base::one")),
        LITERAL("return 0")),
    Diagnostics::NOERROR_RETURN_LAMBDA)

#define NRM2(FUNC, TYPE1, TYPE2, TYPE3)                                        \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECL(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(4)),                                                        \
          CALL("oneapi::mkl::blas::column_major::nrm2",                        \
               MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                 \
               BUFFER_OR_USM_PTR(ARG(2), #TYPE2), ARG(3),                      \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),   \
                   #TYPE3)),                                                   \
          LITERAL("return 0")),                                                \
      Diagnostics::NOERROR_RETURN_LAMBDA)
NRM2(cublasSnrm2_v2, wrapper_float_out, float, float)
NRM2(cublasDnrm2_v2, wrapper_double_out, double, double)
NRM2(cublasScnrm2_v2, wrapper_float_out, std::complex<float>, float)
NRM2(cublasDznrm2_v2, wrapper_double_out, std::complex<double>, double)
NRM2(cublasSnrm2_v2_64, wrapper_float_out, float, float)
NRM2(cublasDnrm2_v2_64, wrapper_double_out, double, double)
NRM2(cublasScnrm2_v2_64, wrapper_float_out, std::complex<float>, float)
NRM2(cublasDznrm2_v2_64, wrapper_double_out, std::complex<double>, double)
#undef NRM2

#define DOT(FUNC, NEW_FUNC, TYPE1, TYPE2, TYPE3)                               \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECL(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct6", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(6)),                                                        \
          CALL("oneapi::mkl::blas::column_major::" #NEW_FUNC,                  \
               MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                 \
               BUFFER_OR_USM_PTR(ARG(2), #TYPE2), ARG(3),                      \
               BUFFER_OR_USM_PTR(ARG(4), #TYPE2), ARG(5),                      \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct6"), false, "get_ptr"),   \
                   #TYPE3)),                                                   \
          LITERAL("return 0")),                                                \
      Diagnostics::NOERROR_RETURN_LAMBDA)
DOT(cublasSdot_v2, dot, wrapper_float_out, float, float)
DOT(cublasDdot_v2, dot, wrapper_double_out, double, double)
DOT(cublasCdotu_v2, dotu, wrapper_float2_out, std::complex<float>, std::complex<float>)
DOT(cublasCdotc_v2, dotc, wrapper_float2_out, std::complex<float>, std::complex<float>)
DOT(cublasZdotu_v2, dotu, wrapper_double2_out, std::complex<double>, std::complex<double>)
DOT(cublasZdotc_v2, dotc, wrapper_double2_out, std::complex<double>, std::complex<double>)
DOT(cublasSdot_v2_64, dot, wrapper_float_out, float, float)
DOT(cublasDdot_v2_64, dot, wrapper_double_out, double, double)
DOT(cublasCdotu_v2_64, dotu, wrapper_float2_out, std::complex<float>, std::complex<float>)
DOT(cublasCdotc_v2_64, dotc, wrapper_float2_out, std::complex<float>, std::complex<float>)
DOT(cublasZdotu_v2_64, dotu, wrapper_double2_out, std::complex<double>, std::complex<double>)
DOT(cublasZdotc_v2_64, dotc, wrapper_double2_out, std::complex<double>, std::complex<double>)
#undef DOT

#define SCAL(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC, CALL("oneapi::mkl::blas::column_major::scal",                     \
                  MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),              \
                  SCALAR_INPUT(ARG(2), #TYPE1),                                \
                  BUFFER_OR_USM_PTR(ARG(3), #TYPE1), ARG(4))))
SCAL(cublasSscal_v2, float)
SCAL(cublasDscal_v2, double)
SCAL(cublasCscal_v2, std::complex<float>)
SCAL(cublasCsscal_v2, std::complex<float>)
SCAL(cublasZscal_v2, std::complex<double>)
SCAL(cublasZdscal_v2, std::complex<double>)
SCAL(cublasSscal_v2_64, float)
SCAL(cublasDscal_v2_64, double)
SCAL(cublasCscal_v2_64, std::complex<float>)
SCAL(cublasCsscal_v2_64, std::complex<float>)
SCAL(cublasZscal_v2_64, std::complex<double>)
SCAL(cublasZdscal_v2_64, std::complex<double>)
#undef SCAL

#define AXPY(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC,                                                                   \
      CALL("oneapi::mkl::blas::column_major::axpy",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                     \
           SCALAR_INPUT(ARG(2), #TYPE1), BUFFER_OR_USM_PTR(ARG(3), #TYPE1),    \
           ARG(4), BUFFER_OR_USM_PTR(ARG(5), #TYPE1), ARG(6))))
AXPY(cublasSaxpy_v2, float)
AXPY(cublasDaxpy_v2, double)
AXPY(cublasCaxpy_v2, std::complex<float>)
AXPY(cublasZaxpy_v2, std::complex<double>)
AXPY(cublasSaxpy_v2_64, float)
AXPY(cublasDaxpy_v2_64, double)
AXPY(cublasCaxpy_v2_64, std::complex<float>)
AXPY(cublasZaxpy_v2_64, std::complex<double>)
#undef AXPY

#define COPY(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC, CALL("oneapi::mkl::blas::column_major::copy",                     \
                  MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),              \
                  BUFFER_OR_USM_PTR(ARG(2), #TYPE1), ARG(3),                   \
                  BUFFER_OR_USM_PTR(ARG(4), #TYPE1), ARG(5))))
COPY(cublasScopy_v2, float)
COPY(cublasDcopy_v2, double)
COPY(cublasCcopy_v2, std::complex<float>)
COPY(cublasZcopy_v2, std::complex<double>)
COPY(cublasScopy_v2_64, float)
COPY(cublasDcopy_v2_64, double)
COPY(cublasCcopy_v2_64, std::complex<float>)
COPY(cublasZcopy_v2_64, std::complex<double>)
#undef COPY

#define SWAP(FUNC, TYPE1)                                                      \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC, CALL("oneapi::mkl::blas::column_major::swap",                     \
                  MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),              \
                  BUFFER_OR_USM_PTR(ARG(2), #TYPE1), ARG(3),                   \
                  BUFFER_OR_USM_PTR(ARG(4), #TYPE1), ARG(5))))
SWAP(cublasSswap_v2, float)
SWAP(cublasDswap_v2, double)
SWAP(cublasCswap_v2, std::complex<float>)
SWAP(cublasZswap_v2, std::complex<double>)
SWAP(cublasSswap_v2_64, float)
SWAP(cublasDswap_v2_64, double)
SWAP(cublasCswap_v2_64, std::complex<float>)
SWAP(cublasZswap_v2_64, std::complex<double>)
#undef SWAP

#define ASUM(FUNC, TYPE1, TYPE2, TYPE3)                                        \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECL(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(4)),                                                        \
          CALL("oneapi::mkl::blas::column_major::asum",                        \
               MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                 \
               BUFFER_OR_USM_PTR(ARG(2), #TYPE2), ARG(3),                      \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),   \
                   #TYPE3)),                                                   \
          LITERAL("return 0")),                                                \
      Diagnostics::NOERROR_RETURN_LAMBDA)
ASUM(cublasSasum_v2, wrapper_float_out, float, float)
ASUM(cublasDasum_v2, wrapper_double_out, double, double)
ASUM(cublasScasum_v2, wrapper_float_out, std::complex<float>, float)
ASUM(cublasDzasum_v2, wrapper_double_out, std::complex<double>, double)
ASUM(cublasSasum_v2_64, wrapper_float_out, float, float)
ASUM(cublasDasum_v2_64, wrapper_double_out, double, double)
ASUM(cublasScasum_v2_64, wrapper_float_out, std::complex<float>, float)
ASUM(cublasDzasum_v2_64, wrapper_double_out, std::complex<double>, double)
#undef ASUM

#define ROT(FUNC, TYPE1)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC,                                                                   \
      CALL("oneapi::mkl::blas::column_major::rot",                             \
           MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                     \
           BUFFER_OR_USM_PTR(ARG(2), #TYPE1), ARG(3),                          \
           BUFFER_OR_USM_PTR(ARG(4), #TYPE1), ARG(5),                          \
           SCALAR_INPUT(ARG(6), #TYPE1), SCALAR_INPUT(ARG(7), #TYPE1))))
ROT(cublasSrot_v2, float)
ROT(cublasDrot_v2, double)
ROT(cublasCrot_v2, std::complex<float>)
ROT(cublasCsrot_v2, std::complex<float>)
ROT(cublasZrot_v2, std::complex<double>)
ROT(cublasZdrot_v2, std::complex<double>)
ROT(cublasSrot_v2_64, float)
ROT(cublasDrot_v2_64, double)
ROT(cublasCrot_v2_64, std::complex<float>)
ROT(cublasCsrot_v2_64, std::complex<float>)
ROT(cublasZrot_v2_64, std::complex<double>)
ROT(cublasZdrot_v2_64, std::complex<double>)
#undef ROT

#define ROTG(FUNC, TYPE1, TYPE2, TYPE3, TYPE4, TYPE5)                          \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECL(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct1", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(1)),                                                        \
          DECL(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct2", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(2)),                                                        \
          DECL(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE2,                 \
               "res_wrapper_ct3", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(3)),                                                        \
          DECL(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE3,                 \
               "res_wrapper_ct4", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(4)),                                                        \
          CALL("oneapi::mkl::blas::column_major::rotg",                        \
               MEMBER_CALL(ARG(0), true, "get_queue"),                         \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct1"), false, "get_ptr"),      \
                   #TYPE4),                                                    \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct2"), false, "get_ptr"),      \
                   #TYPE4),                                                    \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct3"), false, "get_ptr"),      \
                   #TYPE5),                                                    \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct4"), false, "get_ptr"),      \
                   #TYPE4)),                                                   \
          LITERAL("return 0")),                                                \
      Diagnostics::NOERROR_RETURN_LAMBDA)
ROTG(cublasSrotg_v2, wrapper_float_inout, wrapper_float_out, wrapper_float_out, float, float)
ROTG(cublasDrotg_v2, wrapper_double_inout, wrapper_double_out, wrapper_double_out, double, double)
ROTG(cublasCrotg_v2, wrapper_float2_inout, wrapper_float_out, wrapper_float2_out, std::complex<float>, float)
ROTG(cublasZrotg_v2, wrapper_double2_inout, wrapper_double_out, wrapper_double2_out, std::complex<double>, double)
#undef ROTG

#define ROTM(FUNC, TYPE1, TYPE2)                                               \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECL(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct6", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(6), ARG("5")),                                              \
          CALL("oneapi::mkl::blas::column_major::rotm",                        \
               MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                 \
               BUFFER_OR_USM_PTR(ARG(2), #TYPE2), ARG(3),                      \
               BUFFER_OR_USM_PTR(ARG(4), #TYPE2), ARG(5),                      \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct6"), false, "get_ptr"),      \
                   #TYPE2)),                                                   \
          LITERAL("return 0")),                                                \
      Diagnostics::NOERROR_RETURN_LAMBDA)
ROTM(cublasSrotm_v2, wrapper_float_in, float)
ROTM(cublasDrotm_v2, wrapper_double_in, double)
ROTM(cublasSrotm_v2_64, wrapper_float_in, float)
ROTM(cublasDrotm_v2_64, wrapper_double_in, double)
#undef ROTM

#define ROTMG(FUNC, TYPE1, TYPE2, TYPE3)                                       \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECL(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct1", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(1)),                                                        \
          DECL(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct2", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(2)),                                                        \
          DECL(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct3", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(3)),                                                        \
          DECL(MapNames::getLibraryHelperNamespace() + "blas::" #TYPE2,                 \
               "res_wrapper_ct5", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(5), ARG("5")),                                              \
          CALL("oneapi::mkl::blas::column_major::rotmg",                       \
               MEMBER_CALL(ARG(0), true, "get_queue"),                         \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct1"), false, "get_ptr"),      \
                   #TYPE3),                                                    \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct2"), false, "get_ptr"),      \
                   #TYPE3),                                                    \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct3"), false, "get_ptr"),      \
                   #TYPE3),                                                    \
               SCALAR_INPUT(ARG(4), #TYPE3),                                   \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct5"), false, "get_ptr"),      \
                   #TYPE3)),                                                   \
          LITERAL("return 0")),                                                \
      Diagnostics::NOERROR_RETURN_LAMBDA)
ROTMG(cublasSrotmg_v2, wrapper_float_inout, wrapper_float_out, float)
ROTMG(cublasDrotmg_v2, wrapper_double_inout, wrapper_double_out, double)
#undef ROTMG

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
GEMV(cublasSgemv_v2, float)
GEMV(cublasDgemv_v2, double)
GEMV(cublasCgemv_v2, std::complex<float>)
GEMV(cublasZgemv_v2, std::complex<double>)
GEMV(cublasSgemv_v2_64, float)
GEMV(cublasDgemv_v2_64, double)
GEMV(cublasCgemv_v2_64, std::complex<float>)
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
GBMV(cublasSgbmv_v2, float)
GBMV(cublasDgbmv_v2, double)
GBMV(cublasCgbmv_v2, std::complex<float>)
GBMV(cublasZgbmv_v2, std::complex<double>)
GBMV(cublasSgbmv_v2_64, float)
GBMV(cublasDgbmv_v2_64, double)
GBMV(cublasCgbmv_v2_64, std::complex<float>)
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
TRMV(cublasStrmv_v2, float)
TRMV(cublasDtrmv_v2, double)
TRMV(cublasCtrmv_v2, std::complex<float>)
TRMV(cublasZtrmv_v2, std::complex<double>)
TRMV(cublasStrmv_v2_64, float)
TRMV(cublasDtrmv_v2_64, double)
TRMV(cublasCtrmv_v2_64, std::complex<float>)
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
TBMV(cublasStbmv_v2, float)
TBMV(cublasDtbmv_v2, double)
TBMV(cublasCtbmv_v2, std::complex<float>)
TBMV(cublasZtbmv_v2, std::complex<double>)
TBMV(cublasStbmv_v2_64, float)
TBMV(cublasDtbmv_v2_64, double)
TBMV(cublasCtbmv_v2_64, std::complex<float>)
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
TPMV(cublasStpmv_v2, float)
TPMV(cublasDtpmv_v2, double)
TPMV(cublasCtpmv_v2, std::complex<float>)
TPMV(cublasZtpmv_v2, std::complex<double>)
TPMV(cublasStpmv_v2_64, float)
TPMV(cublasDtpmv_v2_64, double)
TPMV(cublasCtpmv_v2_64, std::complex<float>)
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
TRSV(cublasStrsv_v2, float)
TRSV(cublasDtrsv_v2, double)
TRSV(cublasCtrsv_v2, std::complex<float>)
TRSV(cublasZtrsv_v2, std::complex<double>)
TRSV(cublasStrsv_v2_64, float)
TRSV(cublasDtrsv_v2_64, double)
TRSV(cublasCtrsv_v2_64, std::complex<float>)
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
TPSV(cublasStpsv_v2, float)
TPSV(cublasDtpsv_v2, double)
TPSV(cublasCtpsv_v2, std::complex<float>)
TPSV(cublasZtpsv_v2, std::complex<double>)
TPSV(cublasStpsv_v2_64, float)
TPSV(cublasDtpsv_v2_64, double)
TPSV(cublasCtpsv_v2_64, std::complex<float>)
TPSV(cublasZtpsv_v2_64, std::complex<double>)
#undef TPSV

#define GEMM_3M(FUNC, TYPE1)                                                   \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #FUNC,                                                                   \
      CALL("oneapi::mkl::blas::column_major::gemm",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           ARG(3), ARG(4), ARG(5), SCALAR_INPUT(ARG(6), TYPE1),                \
           BUFFER_OR_USM_PTR(ARG(7), TYPE1), ARG(8),                           \
           BUFFER_OR_USM_PTR(ARG(9), TYPE1), ARG(10),                          \
           SCALAR_INPUT(ARG(11), TYPE1), BUFFER_OR_USM_PTR(ARG(12), TYPE1),    \
           ARG(13), ARG("oneapi::mkl::blas::compute_mode::complex_3m"))))
GEMM_3M(cublasCgemm3m, "std::complex<float>")
GEMM_3M(cublasZgemm3m, "std::complex<double>")
GEMM_3M(cublasCgemm3m_64, "std::complex<float>")
GEMM_3M(cublasZgemm3m_64, "std::complex<double>")
#undef GEMM_3M

#define GEMM_BATCH(NAME, TYPE, IS_COMPLEX)                                     \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::gemm_batch",                      \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
           ARG(3), ARG(4), ARG(5), SCALAR_INPUT(ARG(6), TYPE),                 \
           BUFFER_OR_USM_PTR(ARG(7), TYPE), ARG(8), ARG(9),                    \
           BUFFER_OR_USM_PTR(ARG(10), TYPE), ARG(11), ARG(12),                 \
           SCALAR_INPUT(ARG(13), TYPE), BUFFER_OR_USM_PTR(ARG(14), TYPE),      \
           ARG(15), ARG(16), ARG(17))))
GEMM_BATCH(cublasHgemmStridedBatched, MapNames::getClNamespace() + "half", false)
GEMM_BATCH(cublasSgemmStridedBatched, "float", false)
GEMM_BATCH(cublasDgemmStridedBatched, "double", false)
GEMM_BATCH(cublasCgemmStridedBatched, "std::complex<float>", true)
GEMM_BATCH(cublasZgemmStridedBatched, "std::complex<double>", true)
#undef GEMM_BATCH

#define COPY_EX(NAME)                                                          \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL(MapNames::getLibraryHelperNamespace() + "blas::copy", ARG(0),       \
           ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7))))
COPY_EX(cublasCopyEx)
COPY_EX(cublasCopyEx_64)
#undef COPY_EX

#define SWAP_EX(NAME)                                                          \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL(MapNames::getLibraryHelperNamespace() + "blas::swap", ARG(0),       \
           ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7))))
SWAP_EX(cublasSwapEx)
SWAP_EX(cublasSwapEx_64)
#undef SWAP_EX

#define ASUM_EX(NAME)                                                          \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL(MapNames::getLibraryHelperNamespace() + "blas::asum",        \
                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))))
ASUM_EX(cublasAsumEx)
ASUM_EX(cublasAsumEx_64)
#undef ASUM_EX

#define ROTM_EX(NAME)                                                          \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL(MapNames::getLibraryHelperNamespace() + "blas::rotm",        \
                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),      \
                  ARG(7), ARG(8), ARG(9))))
ROTM_EX(cublasRotmEx)
ROTM_EX(cublasRotmEx_64)
#undef ROTM_EX

#define IAMAX_EX(NAME)                                                         \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL(MapNames::getLibraryHelperNamespace() + "blas::iamax",       \
                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))
IAMAX_EX(cublasIamaxEx)
IAMAX_EX(cublasIamaxEx_64)

#define IAMIN_EX(NAME)                                                         \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL(MapNames::getLibraryHelperNamespace() + "blas::iamin",       \
                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))
IAMIN_EX(cublasIaminEx)
IAMIN_EX(cublasIaminEx_64)
#undef IAMIN_EX

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
TBSV(cublasStbsv_v2, "float")
TBSV(cublasDtbsv_v2, "double")
TBSV(cublasCtbsv_v2, "std::complex<float>")
TBSV(cublasZtbsv_v2, "std::complex<double>")
TBSV(cublasStbsv_v2_64, "float")
TBSV(cublasDtbsv_v2_64, "double")
TBSV(cublasCtbsv_v2_64, "std::complex<float>")
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
SYMV(cublasSsymv_v2, "float")
SYMV(cublasDsymv_v2, "double")
SYMV(cublasCsymv_v2, "std::complex<float>")
SYMV(cublasZsymv_v2, "std::complex<double>")
SYMV(cublasSsymv_v2_64, "float")
SYMV(cublasDsymv_v2_64, "double")
SYMV(cublasCsymv_v2_64, "std::complex<float>")
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
HEMV(cublasChemv_v2, "std::complex<float>")
HEMV(cublasZhemv_v2, "std::complex<double>")
HEMV(cublasChemv_v2_64, "std::complex<float>")
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
SBMV(cublasSsbmv_v2, "float")
SBMV(cublasDsbmv_v2, "double")
SBMV(cublasSsbmv_v2_64, "float")
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
HBMV(cublasChbmv_v2, "std::complex<float>")
HBMV(cublasZhbmv_v2, "std::complex<double>")
HBMV(cublasChbmv_v2_64, "std::complex<float>")
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
SPMV(cublasSspmv_v2, "float")
SPMV(cublasDspmv_v2, "double")
SPMV(cublasSspmv_v2_64, "float")
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
HPMV(cublasChpmv_v2, "std::complex<float>")
HPMV(cublasZhpmv_v2, "std::complex<double>")
HPMV(cublasChpmv_v2_64, "std::complex<float>")
HPMV(cublasZhpmv_v2_64, "std::complex<double>")
#undef HPMV

#define GER(NAME, TYPE)                                                        \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL("oneapi::mkl::blas::column_major::ger",                      \
                  MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1), ARG(2),      \
                  SCALAR_INPUT(ARG(3), TYPE), BUFFER_OR_USM_PTR(ARG(4), TYPE), \
                  ARG(5), BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),             \
                  BUFFER_OR_USM_PTR(ARG(8), TYPE), ARG(9))))
GER(cublasSger_v2, "float")
GER(cublasDger_v2, "double")
GER(cublasSger_v2_64, "float")
GER(cublasDger_v2_64, "double")
#undef GER

#define GERU(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL("oneapi::mkl::blas::column_major::geru",                     \
                  MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1), ARG(2),      \
                  SCALAR_INPUT(ARG(3), TYPE), BUFFER_OR_USM_PTR(ARG(4), TYPE), \
                  ARG(5), BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),             \
                  BUFFER_OR_USM_PTR(ARG(8), TYPE), ARG(9))))
GERU(cublasCgeru_v2, "std::complex<float>")
GERU(cublasZgeru_v2, "std::complex<double>")
GERU(cublasCgeru_v2_64, "std::complex<float>")
GERU(cublasZgeru_v2_64, "std::complex<double>")
#undef GERU

#define GERC(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL("oneapi::mkl::blas::column_major::gerc",                     \
                  MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1), ARG(2),      \
                  SCALAR_INPUT(ARG(3), TYPE), BUFFER_OR_USM_PTR(ARG(4), TYPE), \
                  ARG(5), BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),             \
                  BUFFER_OR_USM_PTR(ARG(8), TYPE), ARG(9))))
GERC(cublasCgerc_v2, "std::complex<float>")
GERC(cublasZgerc_v2, "std::complex<double>")
GERC(cublasCgerc_v2_64, "std::complex<float>")
GERC(cublasZgerc_v2_64, "std::complex<double>")
#undef GERC

#define SYR(NAME, TYPE)                                                        \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::syr",                             \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(4), TYPE), ARG(5),                            \
           BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7))))
SYR(cublasSsyr_v2, "float")
SYR(cublasDsyr_v2, "double")
SYR(cublasCsyr_v2, "std::complex<float>")
SYR(cublasZsyr_v2, "std::complex<double>")
SYR(cublasSsyr_v2_64, "float")
SYR(cublasDsyr_v2_64, "double")
SYR(cublasCsyr_v2_64, "std::complex<float>")
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
HER(cublasCher_v2, "float", "std::complex<float>")
HER(cublasZher_v2, "double", "std::complex<double>")
HER(cublasCher_v2_64, "float", "std::complex<float>")
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
SPR(cublasSspr_v2, "float")
SPR(cublasDspr_v2, "double")
SPR(cublasSspr_v2_64, "float")
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
HPR(cublasChpr_v2, "float", "std::complex<float>")
HPR(cublasZhpr_v2, "double", "std::complex<double>")
HPR(cublasChpr_v2_64, "float", "std::complex<float>")
HPR(cublasZhpr_v2_64, "double", "std::complex<double>")
#undef HPR

#define SYR2(NAME, TYPE)                                                       \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL("oneapi::mkl::blas::column_major::syr2",                            \
           MEMBER_CALL(ARG(0), true, "get_queue"),                             \
           BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),    \
           ARG(2), SCALAR_INPUT(ARG(3), TYPE),                                 \
           BUFFER_OR_USM_PTR(ARG(4), TYPE), ARG(5),                            \
           BUFFER_OR_USM_PTR(ARG(6), TYPE), ARG(7),                            \
           BUFFER_OR_USM_PTR(ARG(8), TYPE), ARG(9))))
SYR2(cublasSsyr2_v2, "float")
SYR2(cublasDsyr2_v2, "double")
SYR2(cublasCsyr2_v2, "std::complex<float>")
SYR2(cublasZsyr2_v2, "std::complex<double>")
SYR2(cublasSsyr2_v2_64, "float")
SYR2(cublasDsyr2_v2_64, "double")
SYR2(cublasCsyr2_v2_64, "std::complex<float>")
SYR2(cublasZsyr2_v2_64, "std::complex<double>")
#undef SYR2

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
HER2(cublasCher2_v2, "std::complex<float>")
HER2(cublasZher2_v2, "std::complex<double>")
HER2(cublasCher2_v2_64, "std::complex<float>")
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
SPR2(cublasSspr2_v2, "float")
SPR2(cublasDspr2_v2, "double")
SPR2(cublasSspr2_v2_64, "float")
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
HPR2(cublasChpr2_v2, "std::complex<float>")
HPR2(cublasZhpr2_v2, "std::complex<double>")
HPR2(cublasChpr2_v2_64, "std::complex<float>")
HPR2(cublasZhpr2_v2_64, "std::complex<double>")
#undef HPR2

ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cublasLtCreate", DEREF(0),
    NEW(MapNames::getLibraryHelperNamespace() + "blas_gemm::experimental::descriptor")))
ASSIGNABLE_FACTORY(DELETE_FACTORY_ENTRY("cublasLtDestroy", ARG(0)))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cublasLtMatmulDescCreate", DEREF(0),
    NEW(MapNames::getLibraryHelperNamespace() + "blas_gemm::experimental::matmul_desc_t",
        ARG(1), ARG(2))))
ASSIGNABLE_FACTORY(DELETE_FACTORY_ENTRY("cublasLtMatmulDescDestroy", ARG(0)))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cublasLtMatmulDescSetAttribute",
                                             ARG(0), true, "set_attribute",
                                             ARG(1), ARG(2)))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cublasLtMatmulDescGetAttribute",
                                             ARG(0), true, "get_attribute",
                                             ARG(1), ARG(2)))

ASSIGNABLE_FACTORY(
    ASSIGN_FACTORY_ENTRY("cublasLtMatrixLayoutCreate", DEREF(0),
                         NEW(MapNames::getLibraryHelperNamespace() +
                                 "blas_gemm::experimental::matrix_layout_t",
                             ARG(1), ARG(2), ARG(3), ARG(4))))
ASSIGNABLE_FACTORY(DELETE_FACTORY_ENTRY("cublasLtMatrixLayoutDestroy", ARG(0)))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cublasLtMatrixLayoutSetAttribute",
                                             ARG(0), true, "set_attribute",
                                             ARG(1), ARG(2)))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cublasLtMatrixLayoutGetAttribute",
                                             ARG(0), true, "get_attribute",
                                             ARG(1), ARG(2)))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasLtMatmul",
    CALL(MapNames::getLibraryHelperNamespace() + "blas_gemm::experimental::matmul",
         ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8),
         ARG(9), ARG(10), ARG(11), ARG(15))))
REMOVE_API_FACTORY_ENTRY("cublasLtMatmulPreferenceCreate")
REMOVE_API_FACTORY_ENTRY("cublasLtMatmulPreferenceDestroy")
REMOVE_API_FACTORY_ENTRY("cublasLtMatmulPreferenceSetAttribute")
REMOVE_API_FACTORY_ENTRY("cublasLtMatmulPreferenceGetAttribute")
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cublasLtMatmulAlgoGetHeuristic",
                                        DEREF(9), ARG("1")))

ASSIGNABLE_FACTORY(
    ASSIGN_FACTORY_ENTRY("cublasLtMatrixTransformDescCreate", DEREF(0),
                         NEW(MapNames::getLibraryHelperNamespace() +
                                 "blas_gemm::experimental::transform_desc_t",
                             ARG(1))))
ASSIGNABLE_FACTORY(DELETE_FACTORY_ENTRY("cublasLtMatrixTransformDescDestroy",
                                        ARG(0)))
ASSIGNABLE_FACTORY(
    MEMBER_CALL_FACTORY_ENTRY("cublasLtMatrixTransformDescSetAttribute", ARG(0),
                              true, "set_attribute", ARG(1), ARG(2)))
ASSIGNABLE_FACTORY(
    MEMBER_CALL_FACTORY_ENTRY("cublasLtMatrixTransformDescGetAttribute", ARG(0),
                              true, "get_attribute", ARG(1), ARG(2)))
ASSIGNABLE_FACTORY(
    CALL_FACTORY_ENTRY("cublasLtMatrixTransform",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "blas_gemm::experimental::matrix_transform",
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10))))
WARNING_FACTORY_ENTRY("cublasLtGetVersion",
                      CALL_FACTORY_ENTRY("cublasLtGetVersion",
                                         CALL(MapNames::getLibraryHelperNamespace() +
                                              "dnnl::get_version")),
                      Diagnostics::TYPE_MISMATCH)
