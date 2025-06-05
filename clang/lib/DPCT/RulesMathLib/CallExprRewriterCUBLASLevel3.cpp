//===--------------- CallExprRewriterCUBLASLevel3.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUBLAS.h"

using namespace clang::dpct;

RewriterMap dpct::createCUBLASLevel3RewriterMap() {
  return RewriterMap{
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
      // cublasHgemm
      GEMM(cublasHgemm, MapNames::getClNamespace() + "half", false)
      // cublasSgemm_v2
      GEMM(cublasSgemm_v2, "float", false)
      // cublasDgemm_v2
      GEMM(cublasDgemm_v2, "double", false)
      // cublasCgemm_v2
      GEMM(cublasCgemm_v2, "std::complex<float>", true)
      // cublasZgemm_v2
      GEMM(cublasZgemm_v2, "std::complex<double>", true)
      // cublasHgemm_64
      GEMM(cublasHgemm_64, MapNames::getClNamespace() + "half", false)
      // cublasSgemm_v2_64
      GEMM(cublasSgemm_v2_64, "float", false)
      // cublasDgemm_v2_64
      GEMM(cublasDgemm_v2_64, "double", false)
      // cublasCgemm_v2_64
      GEMM(cublasCgemm_v2_64, "std::complex<float>", true)
      // cublasZgemm_v2_64
      GEMM(cublasZgemm_v2_64, "std::complex<double>", true)
#undef GEMM

#define SYRKX(FUNC)                                                            \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          #FUNC,                                                               \
          CALL(                                                                \
              MapNames::getLibraryHelperNamespace() + "blas::syrk", ARG(0),    \
              BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo), \
              BLAS_ENUM_ARG(2,                                                 \
                            clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
              ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), \
              ARG(11), ARG(12)))))
      // cublasSsyrkx
      SYRKX(cublasSsyrkx)
      // cublasDsyrkx
      SYRKX(cublasDsyrkx)
      // cublasCsyrkx
      SYRKX(cublasCsyrkx)
      // cublasZsyrkx
      SYRKX(cublasZsyrkx)
      // cublasSsyrkx_64
      SYRKX(cublasSsyrkx_64)
      // cublasDsyrkx_64
      SYRKX(cublasDsyrkx_64)
      // cublasCsyrkx_64
      SYRKX(cublasCsyrkx_64)
      // cublasZsyrkx_64
      SYRKX(cublasZsyrkx_64)
#undef SYRKX

#define CHERKX(FUNC)                                                           \
  ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(                                  \
      HelperFeatureEnum::device_ext,                                           \
      CALL_FACTORY_ENTRY(                                                      \
          #FUNC,                                                               \
          CALL(                                                                \
              MapNames::getLibraryHelperNamespace() + "blas::herk", ARG(0),    \
              BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo), \
              BLAS_ENUM_ARG(2,                                                 \
                            clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
              ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), \
              ARG(11), ARG(12)))))
      // cublasCherkx
      CHERKX(cublasCherkx)
      // cublasZherkx
      CHERKX(cublasZherkx)
      // cublasCherkx_64
      CHERKX(cublasCherkx_64)
      // cublasZherkx_64
      CHERKX(cublasZherkx_64)
#undef CHERKX

      // cublasStrsmBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasStrsmBatched",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::trsm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(1,
                                 clang::dpct::BLASEnumExpr::BLASEnumType::Side),
                   BLAS_ENUM_ARG(2,
                                 clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
                   BLAS_ENUM_ARG(
                       3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(4,
                                 clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
                   ARG(5), ARG(6), ARG(7),
                   CAST(makeLiteral("const void**"), ARG(8)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_float"),
                   ARG(9), CAST(makeLiteral("void**"), ARG(10)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_float"),
                   ARG(11), ARG(12),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_float")))))
      // cublasDtrsmBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasDtrsmBatched",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::trsm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(1,
                                 clang::dpct::BLASEnumExpr::BLASEnumType::Side),
                   BLAS_ENUM_ARG(2,
                                 clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
                   BLAS_ENUM_ARG(
                       3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(4,
                                 clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
                   ARG(5), ARG(6), ARG(7),
                   CAST(makeLiteral("const void**"), ARG(8)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_double"),
                   ARG(9), CAST(makeLiteral("void**"), ARG(10)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_double"),
                   ARG(11), ARG(12),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_double")))))
      // cublasCtrsmBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasCtrsmBatched",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::trsm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(1,
                                 clang::dpct::BLASEnumExpr::BLASEnumType::Side),
                   BLAS_ENUM_ARG(2,
                                 clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
                   BLAS_ENUM_ARG(
                       3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(4,
                                 clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
                   ARG(5), ARG(6), ARG(7),
                   CAST(makeLiteral("const void**"), ARG(8)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::complex_float"),
                   ARG(9), CAST(makeLiteral("void**"), ARG(10)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::complex_float"),
                   ARG(11), ARG(12),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::complex_float")))))
      // cublasZtrsmBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasZtrsmBatched",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::trsm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(1,
                                 clang::dpct::BLASEnumExpr::BLASEnumType::Side),
                   BLAS_ENUM_ARG(2,
                                 clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
                   BLAS_ENUM_ARG(
                       3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(4,
                                 clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
                   ARG(5), ARG(6), ARG(7),
                   CAST(makeLiteral("const void**"), ARG(8)),
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
              MapNames::getLibraryHelperNamespace() + "blas::trmm", ARG(0),    \
              BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side), \
              BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo), \
              BLAS_ENUM_ARG(3,                                                 \
                            clang::dpct::BLASEnumExpr::BLASEnumType::Trans),   \
              BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag), \
              ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),        \
              ARG(12), ARG(13)))))
      // cublasStrmm_v2
      TRMM(cublasStrmm_v2)
      // cublasDtrmm_v2
      TRMM(cublasDtrmm_v2)
      // cublasCtrmm_v2
      TRMM(cublasCtrmm_v2)
      // cublasZtrmm_v2
      TRMM(cublasZtrmm_v2)
      // cublasStrmm_v2_64
      TRMM(cublasStrmm_v2_64)
      // cublasDtrmm_v2_64
      TRMM(cublasDtrmm_v2_64)
      // cublasCtrmm_v2_64
      TRMM(cublasCtrmm_v2_64)
      // cublasZtrmm_v2_64
      TRMM(cublasZtrmm_v2_64)
#undef TRMM

      // cublasHgemmBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasHgemmBatched",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(
                       1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(
                       2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   ARG(3), ARG(4), ARG(5), ARG(6),
                   CAST(makeLiteral("const void**"), ARG(7)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_half"),
                   ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_half"),
                   ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_half"),
                   ARG(13), ARG(14),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_half")))))
      // cublasSgemmBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasSgemmBatched",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(
                       1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(
                       2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   ARG(3), ARG(4), ARG(5), ARG(6),
                   CAST(makeLiteral("const void**"), ARG(7)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_float"),
                   ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_float"),
                   ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_float"),
                   ARG(13), ARG(14),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_float")))))
      // cublasDgemmBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasDgemmBatched",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(
                       1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(
                       2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   ARG(3), ARG(4), ARG(5), ARG(6),
                   CAST(makeLiteral("const void**"), ARG(7)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_double"),
                   ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_double"),
                   ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_double"),
                   ARG(13), ARG(14),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::real_double")))))
      // cublasCgemmBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasCgemmBatched",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(
                       1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(
                       2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   ARG(3), ARG(4), ARG(5), ARG(6),
                   CAST(makeLiteral("const void**"), ARG(7)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::complex_float"),
                   ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::complex_float"),
                   ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::complex_float"),
                   ARG(13), ARG(14),
                   ARG(MapNames::getLibraryHelperNamespace() +
                       "library_data_t::complex_float")))))
      // cublasZgemmBatched
      ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          CALL_FACTORY_ENTRY(
              "cublasZgemmBatched",
              CALL(MapNames::getLibraryHelperNamespace() + "blas::gemm_batch",
                   ARG(0),
                   BLAS_ENUM_ARG(
                       1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
                   BLAS_ENUM_ARG(
                       2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
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
      // cublasSsyrk_v2
      SYRK(cublasSsyrk_v2, float, false)
      // cublasDsyrk_v2
      SYRK(cublasDsyrk_v2, double, false)
      // cublasCsyrk_v2
      SYRK(cublasCsyrk_v2, std::complex<float>, true)
      // cublasZsyrk_v2
      SYRK(cublasZsyrk_v2, std::complex<double>, true)
      // cublasSsyrk_v2_64
      SYRK(cublasSsyrk_v2_64, float, false)
      // cublasDsyrk_v2_64
      SYRK(cublasDsyrk_v2_64, double, false)
      // cublasCsyrk_v2_64
      SYRK(cublasCsyrk_v2_64, std::complex<float>, true)
      // cublasZsyrk_v2_64
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
      // cublasSsymm_v2
      SYMM(cublasSsymm_v2, float, false)
      // cublasDsymm_v2
      SYMM(cublasDsymm_v2, double, false)
      // cublasCsymm_v2
      SYMM(cublasCsymm_v2, std::complex<float>, true)
      // cublasZsymm_v2
      SYMM(cublasZsymm_v2, std::complex<double>, true)
      // cublasSsymm_v2_64
      SYMM(cublasSsymm_v2_64, float, false)
      // cublasDsymm_v2_64
      SYMM(cublasDsymm_v2_64, double, false)
      // cublasCsymm_v2_64
      SYMM(cublasCsymm_v2_64, std::complex<float>, true)
      // cublasZsymm_v2_64
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
      // cublasStrsm_v2
      TRSM(cublasStrsm_v2, float, false)
      // cublasDtrsm_v2
      TRSM(cublasDtrsm_v2, double, false)
      // cublasCtrsm_v2
      TRSM(cublasCtrsm_v2, std::complex<float>, true)
      // cublasZtrsm_v2
      TRSM(cublasZtrsm_v2, std::complex<double>, true)
      // cublasStrsm_v2_64
      TRSM(cublasStrsm_v2_64, float, false)
      // cublasDtrsm_v2_64
      TRSM(cublasDtrsm_v2_64, double, false)
      // cublasCtrsm_v2_64
      TRSM(cublasCtrsm_v2_64, std::complex<float>, true)
      // cublasZtrsm_v2_64
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
      // cublasChemm_v2
      HEMM(cublasChemm_v2, std::complex<float>)
      // cublasZhemm_v2
      HEMM(cublasZhemm_v2, std::complex<double>)
      // cublasChemm_v2_64
      HEMM(cublasChemm_v2_64, std::complex<float>)
      // cublasZhemm_v2_64
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
      // cublasCherk_v2
      HERK(cublasCherk_v2, std::complex<float>, float)
      // cublasZherk_v2
      HERK(cublasZherk_v2, std::complex<double>, double)
      // cublasCherk_v2_64
      HERK(cublasCherk_v2_64, std::complex<float>, float)
      // cublasZherk_v2_64
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
      // cublasSsyr2k_v2
      SYR2K(cublasSsyr2k_v2, float, false)
      // cublasDsyr2k_v2
      SYR2K(cublasDsyr2k_v2, double, false)
      // cublasCsyr2k_v2
      SYR2K(cublasCsyr2k_v2, std::complex<float>, true)
      // cublasZsyr2k_v2
      SYR2K(cublasZsyr2k_v2, std::complex<double>, true)
      // cublasSsyr2k_v2_64
      SYR2K(cublasSsyr2k_v2_64, float, false)
      // cublasDsyr2k_v2_64
      SYR2K(cublasDsyr2k_v2_64, double, false)
      // cublasCsyr2k_v2_64
      SYR2K(cublasCsyr2k_v2_64, std::complex<float>, true)
      // cublasZsyr2k_v2_64
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
      // cublasCher2k_v2
      HER2K(cublasCher2k_v2, std::complex<float>, float)
      // cublasZher2k_v2
      HER2K(cublasZher2k_v2, std::complex<double>, double)
      // cublasCher2k_v2_64
      HER2K(cublasCher2k_v2_64, std::complex<float>, float)
      // cublasZher2k_v2_64
      HER2K(cublasZher2k_v2_64, std::complex<double>, double)
#undef HER2K

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
      // cublasSsyr2_v2
      SYR2(cublasSsyr2_v2, "float")
      // cublasDsyr2_v2
      SYR2(cublasDsyr2_v2, "double")
      // cublasCsyr2_v2
      SYR2(cublasCsyr2_v2, "std::complex<float>")
      // cublasZsyr2_v2
      SYR2(cublasZsyr2_v2, "std::complex<double>")
      // cublasSsyr2_v2_64
      SYR2(cublasSsyr2_v2_64, "float")
      // cublasDsyr2_v2_64
      SYR2(cublasDsyr2_v2_64, "double")
      // cublasCsyr2_v2_64
      SYR2(cublasCsyr2_v2_64, "std::complex<float>")
      // cublasZsyr2_v2_64
      SYR2(cublasZsyr2_v2_64, "std::complex<double>")
#undef SYR2

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
      // cublasCgemm3m
      GEMM_3M(cublasCgemm3m, "std::complex<float>")
      // cublasZgemm3m
      GEMM_3M(cublasZgemm3m, "std::complex<double>")
      // cublasCgemm3m_64
      GEMM_3M(cublasCgemm3m_64, "std::complex<float>")
      // cublasZgemm3m_64
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
      // cublasHgemmStridedBatched
      GEMM_BATCH(cublasHgemmStridedBatched, MapNames::getClNamespace() + "half",
                 false)
      // cublasSgemmStridedBatched
      GEMM_BATCH(cublasSgemmStridedBatched, "float", false)
      // cublasDgemmStridedBatched
      GEMM_BATCH(cublasDgemmStridedBatched, "double", false)
      // cublasCgemmStridedBatched
      GEMM_BATCH(cublasCgemmStridedBatched, "std::complex<float>", true)
      // cublasZgemmStridedBatched
      GEMM_BATCH(cublasZgemmStridedBatched, "std::complex<double>", true)
#undef GEMM_BATCH

  };
}
