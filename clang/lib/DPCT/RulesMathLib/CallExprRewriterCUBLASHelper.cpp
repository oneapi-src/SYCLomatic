//===--------------- CallExprRewriterCUBLASHelper.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUBLAS.h"

using namespace clang::dpct;

RewriterMap dpct::createCUBLASHelperRewriterMap() {
  return RewriterMap{
      // cublasGetVersion_v2
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasGetVersion_v2",
              CALL(MapNames::getLibraryHelperNamespace() + "mkl_get_version",
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "version_field::major"),
                   ARG(1)))))
      // cublasGetVersion
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasGetVersion",
              CALL(MapNames::getLibraryHelperNamespace() + "mkl_get_version",
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "version_field::major"),
                   ARG(0)))))
      // cublasCreate_v2
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          ASSIGN_FACTORY_ENTRY(
              "cublasCreate_v2", DEREF(0),
              NEW(MapNames::getLibraryHelperNamespace() + "blas::descriptor"))))
      // cublasDestroy_v2
      ASSIGNABLE_FACTORY(DELETE_FACTORY_ENTRY("cublasDestroy_v2", ARG(0)))
      // cublasSetStream_v2
      ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cublasSetStream_v2", ARG(0),
                                                   true, "set_queue", ARG(1)))
      // cublasGetStream_v2
      ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
          "cublasGetStream_v2", DEREF(1),
          UO(UnaryOperatorKind::UO_AddrOf,
             PAREN(MEMBER_CALL(ARG(0), true, "get_queue")))))
      // cublasSetKernelStream
      ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
          "cublasSetKernelStream", CALL(MapNames::getLibraryHelperNamespace() +
                                            "blas::descriptor::set_saved_queue",
                                        ARG(0))))
      // cublasSetMathMode
      ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
          "cublasSetMathMode", ARG(0), true, "set_math_mode", ARG(1)))
      // cublasGetMathMode
      ASSIGNABLE_FACTORY(
          ASSIGN_FACTORY_ENTRY("cublasGetMathMode", DEREF(1),
                               MEMBER_CALL(ARG(0), true, "get_math_mode")))

#define SETGET_VECTOR(NAME)                                                    \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(#NAME, CALL(MapNames::getLibraryHelperNamespace() +   \
                                         "blas::matrix_mem_copy",              \
                                     ARG(4), ARG(2), ARG(5), ARG(3), ARG("1"), \
                                     ARG(0), ARG(1)))))
      // cublasSetVector
      SETGET_VECTOR(cublasSetVector)
      // cublasGetVector
      SETGET_VECTOR(cublasGetVector)
      // cublasSetVector_64
      SETGET_VECTOR(cublasSetVector_64)
      // cublasGetVector_64
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
      // cublasSetVectorAsync
      SETGET_VECTOR_ASYNC(cublasSetVectorAsync)
      // cublasGetVectorAsync
      SETGET_VECTOR_ASYNC(cublasGetVectorAsync)
      // cublasSetVectorAsync_64
      SETGET_VECTOR_ASYNC(cublasSetVectorAsync_64)
      // cublasGetVectorAsync_64
      SETGET_VECTOR_ASYNC(cublasGetVectorAsync_64)
#undef SETGET_VECTOR_ASYNC

#define SETGET_MATRIX(NAME)                                                    \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(#NAME, CALL(MapNames::getLibraryHelperNamespace() +   \
                                         "blas::matrix_mem_copy",              \
                                     ARG(5), ARG(3), ARG(6), ARG(4), ARG(0),   \
                                     ARG(1), ARG(2)))))
      // cublasSetMatrix
      SETGET_MATRIX(cublasSetMatrix)
      // cublasGetMatrix
      SETGET_MATRIX(cublasGetMatrix)
      // cublasSetMatrix_64
      SETGET_MATRIX(cublasSetMatrix_64)
      // cublasGetMatrix_64
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
      // cublasSetMatrixAsync
      SETGET_MATRIX_ASYNC(cublasSetMatrixAsync)
      // cublasGetMatrixAsync
      SETGET_MATRIX_ASYNC(cublasGetMatrixAsync)
      // cublasSetMatrixAsync_64
      SETGET_MATRIX_ASYNC(cublasSetMatrixAsync_64)
      // cublasGetMatrixAsync_64
      SETGET_MATRIX_ASYNC(cublasGetMatrixAsync_64)
#undef SETGET_MATRIX_ASYNC
      // cublasGetStatusString
      WARNING_FACTORY_ENTRY(
          "cublasGetStatusString",
          CALL_FACTORY_ENTRY(
              "cublasGetStatusString",
              CALL(MapNames::getDpctNamespace() + "get_error_string_dummy",
                   ARG_WC(0))),
          Diagnostics::ERROR_HANDLING_API_REPLACED_BY_DUMMY)};
}
