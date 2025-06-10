//===---------------- CallExprRewriterCUBLASExt.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUBLAS.h"

using namespace clang::dpct;

RewriterMap dpct::createCUBLASExtRewriterMap() {
  return RewriterMap{
      // cublasNrm2Ex
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasNrm2Ex",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::nrm2", ARG(0),
                   ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))
      // cublasNrm2Ex_64
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasNrm2Ex_64",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::nrm2", ARG(0),
                   ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))
      // cublasDotEx
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasDotEx",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::dot", ARG(0),
                   ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                   ARG(8), ARG(9)))))
      // cublasDotEx_64
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasDotEx_64",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::dot", ARG(0),
                   ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                   ARG(8), ARG(9)))))
      // cublasDotcEx
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasDotcEx",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::dotc", ARG(0),
                   ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                   ARG(8), ARG(9)))))
      // cublasDotcEx_64
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasDotcEx_64",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::dotc", ARG(0),
                   ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                   ARG(8), ARG(9)))))
      // cublasScalEx
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasScalEx",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::scal", ARG(0),
                   ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))
      // cublasScalEx_64
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasScalEx_64",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::scal", ARG(0),
                   ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))
      // cublasAxpyEx
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasAxpyEx",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::axpy", ARG(0),
                   ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                   ARG(8), ARG(9)))))
      // cublasAxpyEx_64
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasAxpyEx_64",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::axpy", ARG(0),
                   ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                   ARG(8), ARG(9)))))
      // cublasRotEx
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasRotEx",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::rot", ARG(0),
                   ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                   ARG(8), ARG(9), ARG(10)))))
      // cublasRotEx_64
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasRotEx_64",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::rot", ARG(0),
                   ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
                   ARG(8), ARG(9), ARG(10)))))
      // cublasGemmBatchedEx
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasGemmBatchedEx",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(
                       1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(
                       2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   ARG(3), ARG(4), ARG(5), ARG(6),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(7),
                                             BOOL(true), BOOL(false)),
                   ARG(8), ARG(9),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(10),
                                             BOOL(true), BOOL(false)),
                   ARG(11), ARG(12), ARG(13),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(14),
                                             BOOL(false), BOOL(false)),
                   ARG(15), ARG(16), ARG(17), ARG(18)))))
      // cublasGemmBatchedEx_64
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasGemmBatchedEx_64",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(
                       1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(
                       2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   ARG(3), ARG(4), ARG(5), ARG(6),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(7),
                                             BOOL(true), BOOL(false)),
                   ARG(8), ARG(9),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(10),
                                             BOOL(true), BOOL(false)),
                   ARG(11), ARG(12), ARG(13),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(14),
                                             BOOL(false), BOOL(false)),
                   ARG(15), ARG(16), ARG(17), ARG(18)))))
      // cublasGemmStridedBatchedEx
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasGemmStridedBatchedEx",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(
                       1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(
                       2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9),
                   ARG(10), ARG(11), ARG(12), ARG(13), ARG(14), ARG(15),
                   ARG(16), ARG(17), ARG(18), ARG(19), ARG(20), ARG(21)))))
      // cublasGemmStridedBatchedEx_64
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasGemmStridedBatchedEx_64",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(
                       1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(
                       2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9),
                   ARG(10), ARG(11), ARG(12), ARG(13), ARG(14), ARG(15),
                   ARG(16), ARG(17), ARG(18), ARG(19), ARG(20), ARG(21)))))
      // cublasSgeam
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
      // cublasDgeam
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
      // cublasCgeam
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
      // cublasZgeam
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
      // cublasSdgmm
      ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
          "cublasSdgmm",
          CALL("oneapi::mkl::blas::column_major::dgmm",
               MEMBER_CALL(ARG(0), true, "get_queue"),
               BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
               ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "float"), ARG(5),
               BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
               BUFFER_OR_USM_PTR(ARG(8), "float"), ARG(9))))
      // cublasDdgmm
      ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
          "cublasDdgmm",
          CALL("oneapi::mkl::blas::column_major::dgmm",
               MEMBER_CALL(ARG(0), true, "get_queue"),
               BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
               ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "double"), ARG(5),
               BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
               BUFFER_OR_USM_PTR(ARG(8), "double"), ARG(9))))
      // cublasCdgmm
      ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
          "cublasCdgmm",
          CALL("oneapi::mkl::blas::column_major::dgmm",
               MEMBER_CALL(ARG(0), true, "get_queue"),
               BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
               ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "std::complex<float>"),
               ARG(5), BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
               BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9))))
      // cublasZdgmm
      ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
          "cublasZdgmm",
          CALL("oneapi::mkl::blas::column_major::dgmm",
               MEMBER_CALL(ARG(0), true, "get_queue"),
               BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
               ARG(2), ARG(3),
               BUFFER_OR_USM_PTR(ARG(4), "std::complex<double>"), ARG(5),
               BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
               BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9))))

      // cublasSgetrfBatched
      WARNING_FACTORY_ENTRY(
          "cublasSgetrfBatched",
          ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              CALL_FACTORY_ENTRY(
                  "cublasSgetrfBatched",
                  CALL(MapNames::getLibraryHelperNamespace() +
                           "getrf_batch_wrapper",
                       MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                       DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(2),
                                                 BOOL(false), BOOL(false)),
                       ARG(3), ARG(4), ARG(5), ARG(6))))),
          Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4),
          ARG(MapNames::getLibraryHelperNamespace() + "getrf_batch_wrapper"),
          ARG("cublasSgetrfBatched"))
      // cublasDgetrfBatched
      WARNING_FACTORY_ENTRY(
          "cublasDgetrfBatched",
          ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              CALL_FACTORY_ENTRY(
                  "cublasDgetrfBatched",
                  CALL(MapNames::getLibraryHelperNamespace() +
                           "getrf_batch_wrapper",
                       MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                       DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(2),
                                                 BOOL(false), BOOL(false)),
                       ARG(3), ARG(4), ARG(5), ARG(6))))),
          Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4),
          ARG(MapNames::getLibraryHelperNamespace() + "getrf_batch_wrapper"),
          ARG("cublasDgetrfBatched"))
      // cublasCgetrfBatched
      WARNING_FACTORY_ENTRY(
          "cublasCgetrfBatched",
          ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              CALL_FACTORY_ENTRY(
                  "cublasCgetrfBatched",
                  CALL(MapNames::getLibraryHelperNamespace() +
                           "getrf_batch_wrapper",
                       MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                       DOUBLE_POINTER_CONST_CAST(
                           makeLiteral(MapNames::getClNamespace() + "float2"),
                           ARG(2), BOOL(false), BOOL(false)),
                       ARG(3), ARG(4), ARG(5), ARG(6))))),
          Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4),
          ARG(MapNames::getLibraryHelperNamespace() + "getrf_batch_wrapper"),
          ARG("cublasCgetrfBatched"))
      // cublasZgetrfBatched
      WARNING_FACTORY_ENTRY(
          "cublasZgetrfBatched",
          ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
              HelperFeatureEnum::device_ext,
              CALL_FACTORY_ENTRY(
                  "cublasZgetrfBatched",
                  CALL(MapNames::getLibraryHelperNamespace() +
                           "getrf_batch_wrapper",
                       MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                       DOUBLE_POINTER_CONST_CAST(
                           makeLiteral(MapNames::getClNamespace() + "double2"),
                           ARG(2), BOOL(false), BOOL(false)),
                       ARG(3), ARG(4), ARG(5), ARG(6))))),
          Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4),
          ARG(MapNames::getLibraryHelperNamespace() + "getrf_batch_wrapper"),
          ARG("cublasZgetrfBatched"))
      // cublasSgetrsBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasSgetrsBatched",
              CALL(
                  MapNames::getLibraryHelperNamespace() + "getrs_batch_wrapper",
                  MEMBER_CALL(ARG(0), true, "get_queue"),
                  BLAS_ENUM_ARG(1,
                                clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                  ARG(2), ARG(3),
                  DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(4),
                                            BOOL(true), BOOL(false)),
                  ARG(5), ARG(6),
                  DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(7),
                                            BOOL(false), BOOL(false)),
                  ARG(8), ARG(9), ARG(10)))))
      // cublasDgetrsBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasDgetrsBatched",
              CALL(
                  MapNames::getLibraryHelperNamespace() + "getrs_batch_wrapper",
                  MEMBER_CALL(ARG(0), true, "get_queue"),
                  BLAS_ENUM_ARG(1,
                                clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                  ARG(2), ARG(3),
                  DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(4),
                                            BOOL(true), BOOL(false)),
                  ARG(5), ARG(6),
                  DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(7),
                                            BOOL(false), BOOL(false)),
                  ARG(8), ARG(9), ARG(10)))))
      // cublasCgetrsBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasCgetrsBatched",
              CALL(
                  MapNames::getLibraryHelperNamespace() + "getrs_batch_wrapper",
                  MEMBER_CALL(ARG(0), true, "get_queue"),
                  BLAS_ENUM_ARG(1,
                                clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                  ARG(2), ARG(3),
                  DOUBLE_POINTER_CONST_CAST(
                      makeLiteral(MapNames::getClNamespace() + "float2"),
                      ARG(4), BOOL(true), BOOL(false)),
                  ARG(5), ARG(6),
                  DOUBLE_POINTER_CONST_CAST(
                      makeLiteral(MapNames::getClNamespace() + "float2"),
                      ARG(7), BOOL(false), BOOL(false)),
                  ARG(8), ARG(9), ARG(10)))))
      // cublasZgetrsBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasZgetrsBatched",
              CALL(
                  MapNames::getLibraryHelperNamespace() + "getrs_batch_wrapper",
                  MEMBER_CALL(ARG(0), true, "get_queue"),
                  BLAS_ENUM_ARG(1,
                                clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                  ARG(2), ARG(3),
                  DOUBLE_POINTER_CONST_CAST(
                      makeLiteral(MapNames::getClNamespace() + "double2"),
                      ARG(4), BOOL(true), BOOL(false)),
                  ARG(5), ARG(6),
                  DOUBLE_POINTER_CONST_CAST(
                      makeLiteral(MapNames::getClNamespace() + "double2"),
                      ARG(7), BOOL(false), BOOL(false)),
                  ARG(8), ARG(9), ARG(10)))))
      // cublasSgetriBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasSgetriBatched",
              CALL(MapNames::getLibraryHelperNamespace() +
                       "getri_batch_wrapper",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(2),
                                             BOOL(true), BOOL(false)),
                   ARG(3), ARG(4),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(5),
                                             BOOL(false), BOOL(false)),
                   ARG(6), ARG(7), ARG(8)))))
      // cublasDgetriBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasDgetriBatched",
              CALL(MapNames::getLibraryHelperNamespace() +
                       "getri_batch_wrapper",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(2),
                                             BOOL(true), BOOL(false)),
                   ARG(3), ARG(4),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(5),
                                             BOOL(false), BOOL(false)),
                   ARG(6), ARG(7), ARG(8)))))
      // cublasCgetriBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasCgetriBatched",
              CALL(MapNames::getLibraryHelperNamespace() +
                       "getri_batch_wrapper",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   DOUBLE_POINTER_CONST_CAST(
                       makeLiteral(MapNames::getClNamespace() + "float2"),
                       ARG(2), BOOL(true), BOOL(false)),
                   ARG(3), ARG(4),
                   DOUBLE_POINTER_CONST_CAST(
                       makeLiteral(MapNames::getClNamespace() + "float2"),
                       ARG(5), BOOL(false), BOOL(false)),
                   ARG(6), ARG(7), ARG(8)))))
      // cublasZgetriBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasZgetriBatched",
              CALL(MapNames::getLibraryHelperNamespace() +
                       "getri_batch_wrapper",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),
                   DOUBLE_POINTER_CONST_CAST(
                       makeLiteral(MapNames::getClNamespace() + "double2"),
                       ARG(2), BOOL(true), BOOL(false)),
                   ARG(3), ARG(4),
                   DOUBLE_POINTER_CONST_CAST(
                       makeLiteral(MapNames::getClNamespace() + "double2"),
                       ARG(5), BOOL(false), BOOL(false)),
                   ARG(6), ARG(7), ARG(8)))))
      // cublasSgeqrfBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasSgeqrfBatched",
              CALL(MapNames::getLibraryHelperNamespace() +
                       "geqrf_batch_wrapper",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1), ARG(2),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(3),
                                             BOOL(false), BOOL(false)),
                   ARG(4),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(5),
                                             BOOL(false), BOOL(false)),
                   ARG(6), ARG(7)))))
      // cublasDgeqrfBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasDgeqrfBatched",
              CALL(MapNames::getLibraryHelperNamespace() +
                       "geqrf_batch_wrapper",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1), ARG(2),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(3),
                                             BOOL(false), BOOL(false)),
                   ARG(4),
                   DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(5),
                                             BOOL(false), BOOL(false)),
                   ARG(6), ARG(7)))))
      // cublasCgeqrfBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasCgeqrfBatched",
              CALL(MapNames::getLibraryHelperNamespace() +
                       "geqrf_batch_wrapper",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1), ARG(2),
                   DOUBLE_POINTER_CONST_CAST(
                       makeLiteral(MapNames::getClNamespace() + "float2"),
                       ARG(3), BOOL(false), BOOL(false)),
                   ARG(4),
                   DOUBLE_POINTER_CONST_CAST(
                       makeLiteral(MapNames::getClNamespace() + "float2"),
                       ARG(5), BOOL(false), BOOL(false)),
                   ARG(6), ARG(7)))))
      // cublasZgeqrfBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasZgeqrfBatched",
              CALL(MapNames::getLibraryHelperNamespace() +
                       "geqrf_batch_wrapper",
                   MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1), ARG(2),
                   DOUBLE_POINTER_CONST_CAST(
                       makeLiteral(MapNames::getClNamespace() + "double2"),
                       ARG(3), BOOL(false), BOOL(false)),
                   ARG(4),
                   DOUBLE_POINTER_CONST_CAST(
                       makeLiteral(MapNames::getClNamespace() + "double2"),
                       ARG(5), BOOL(false), BOOL(false)),
                   ARG(6), ARG(7)))))

#define GELS_BATCHED(NAME)                                                     \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(#NAME, CALL(MapNames::getLibraryHelperNamespace() +   \
                                         "blas::gels_batch_wrapper",           \
                                     ARG(0), ARG(1), ARG(2), ARG(3), ARG(4),   \
                                     ARG(5), ARG(6), ARG(7), ARG(8), ARG(9),   \
                                     ARG(10), ARG(11)))))
      // cublasSgelsBatched
      GELS_BATCHED(cublasSgelsBatched)
      // cublasDgelsBatched
      GELS_BATCHED(cublasDgelsBatched)
      // cublasCgelsBatched
      GELS_BATCHED(cublasCgelsBatched)
      // cublasZgelsBatched
      GELS_BATCHED(cublasZgelsBatched)
#undef GELS_BATCHED

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
      // cublasSgemmEx
      GEMM_EX(cublasSgemmEx, MapNames::getLibraryHelperNamespace() +
                                 "library_data_t::real_float")
      // cublasCgemmEx
      GEMM_EX(cublasCgemmEx, MapNames::getLibraryHelperNamespace() +
                                 "library_data_t::complex_float")
      // cublasCgemm3mEx
      GEMM_EX(cublasCgemm3mEx, MapNames::getLibraryHelperNamespace() +
                                   "library_data_t::complex_float")
      // cublasGemmEx
      GEMM_EX(cublasGemmEx, 17)
      // cublasSgemmEx_64
      GEMM_EX(cublasSgemmEx_64, MapNames::getLibraryHelperNamespace() +
                                    "library_data_t::real_float")
      // cublasCgemmEx_64
      GEMM_EX(cublasCgemmEx_64, MapNames::getLibraryHelperNamespace() +
                                    "library_data_t::complex_float")
      // cublasCgemm3mEx_64
      GEMM_EX(cublasCgemm3mEx_64, MapNames::getLibraryHelperNamespace() +
                                      "library_data_t::complex_float")
      // cublasGemmEx_64
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
      // cublasCsyrkEx
      SYHERK(cublasCsyrkEx, false,
             MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_float")
      // cublasCsyrk3mEx
      SYHERK(cublasCsyrk3mEx, false,
             MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_float")
      // cublasCherkEx
      SYHERK(cublasCherkEx, true,
             MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_float")
      // cublasCherk3mEx
      SYHERK(cublasCherk3mEx, true,
             MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_float")
      // cublasCsyrkEx_64
      SYHERK(cublasCsyrkEx_64, false,
             MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_float")
      // cublasCsyrk3mEx_64
      SYHERK(cublasCsyrk3mEx_64, false,
             MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_float")
      // cublasCherkEx_64
      SYHERK(cublasCherkEx_64, true,
             MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_float")
      // cublasCherk3mEx_64
      SYHERK(cublasCherk3mEx_64, true,
             MapNames::getLibraryHelperNamespace() +
                 "library_data_t::complex_float")
#undef SYHERK

      // cublasSgeam_64
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
      // cublasDgeam_64
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
      // cublasCgeam_64
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
      // cublasZgeam_64
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
      // cublasSdgmm_64
      ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
          "cublasSdgmm_64",
          CALL("oneapi::mkl::blas::column_major::dgmm",
               MEMBER_CALL(ARG(0), true, "get_queue"),
               BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
               ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "float"), ARG(5),
               BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
               BUFFER_OR_USM_PTR(ARG(8), "float"), ARG(9))))
      // cublasDdgmm_64
      ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
          "cublasDdgmm_64",
          CALL("oneapi::mkl::blas::column_major::dgmm",
               MEMBER_CALL(ARG(0), true, "get_queue"),
               BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
               ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "double"), ARG(5),
               BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
               BUFFER_OR_USM_PTR(ARG(8), "double"), ARG(9))))
      // cublasCdgmm_64
      ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
          "cublasCdgmm_64",
          CALL("oneapi::mkl::blas::column_major::dgmm",
               MEMBER_CALL(ARG(0), true, "get_queue"),
               BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
               ARG(2), ARG(3), BUFFER_OR_USM_PTR(ARG(4), "std::complex<float>"),
               ARG(5), BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
               BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9))))
      // cublasZdgmm_64
      ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
          "cublasZdgmm_64",
          CALL("oneapi::mkl::blas::column_major::dgmm",
               MEMBER_CALL(ARG(0), true, "get_queue"),
               BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
               ARG(2), ARG(3),
               BUFFER_OR_USM_PTR(ARG(4), "std::complex<double>"), ARG(5),
               BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
               BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9))))

#define COPY_EX(NAME)                                                          \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL(MapNames::getLibraryHelperNamespace() + "blas::copy", ARG(0),       \
           ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7))))
      // cublasCopyEx
      COPY_EX(cublasCopyEx)
      // cublasCopyEx_64
      COPY_EX(cublasCopyEx_64)
#undef COPY_EX

#define SWAP_EX(NAME)                                                          \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME,                                                                   \
      CALL(MapNames::getLibraryHelperNamespace() + "blas::swap", ARG(0),       \
           ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7))))
      // cublasSwapEx
      SWAP_EX(cublasSwapEx)
      // cublasSwapEx_64
      SWAP_EX(cublasSwapEx_64)
#undef SWAP_EX

#define ASUM_EX(NAME)                                                          \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL(MapNames::getLibraryHelperNamespace() + "blas::asum",        \
                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))))
      // cublasAsumEx
      ASUM_EX(cublasAsumEx)
      // cublasAsumEx_64
      ASUM_EX(cublasAsumEx_64)
#undef ASUM_EX

#define ROTM_EX(NAME)                                                          \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL(MapNames::getLibraryHelperNamespace() + "blas::rotm",        \
                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),      \
                  ARG(7), ARG(8), ARG(9))))
      // cublasRotmEx
      ROTM_EX(cublasRotmEx)
      // cublasRotmEx_64
      ROTM_EX(cublasRotmEx_64)
#undef ROTM_EX

#define IAMAX_EX(NAME)                                                         \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL(MapNames::getLibraryHelperNamespace() + "blas::iamax",       \
                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))
      // cublasIamaxEx
      IAMAX_EX(cublasIamaxEx)
      // cublasIamaxEx_64
      IAMAX_EX(cublasIamaxEx_64)

#define IAMIN_EX(NAME)                                                         \
  ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(                                       \
      #NAME, CALL(MapNames::getLibraryHelperNamespace() + "blas::iamin",       \
                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))
      // cublasIaminEx
      IAMIN_EX(cublasIaminEx)
      // cublasIaminEx_64
      IAMIN_EX(cublasIaminEx_64)
#undef IAMIN_EX

  };
}
