//===--------------- APINamesCUBLAS.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasGetVersion_v2",
                       CALL(MapNames::getDpctNamespace() + "mkl_get_version",
                            ARG(MapNames::getDpctNamespace() + "version_field::major"), ARG(1)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasGetVersion",
                       CALL(MapNames::getDpctNamespace() + "mkl_get_version",
                            ARG(MapNames::getDpctNamespace() + "version_field::major"), ARG(0)))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSgemm_v2",
    CALL("oneapi::mkl::blas::column_major::gemm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6), DEREF(0)),
         BUFFER_OR_USM_PTR(7, "float"), ARG(8), BUFFER_OR_USM_PTR(9, "float"),
         ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11), DEREF(0)),
         BUFFER_OR_USM_PTR(12, "float"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDgemm_v2",
    CALL("oneapi::mkl::blas::column_major::gemm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6), DEREF(0)),
         BUFFER_OR_USM_PTR(7, "double"), ARG(8), BUFFER_OR_USM_PTR(9, "double"),
         ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11), DEREF(0)),
         BUFFER_OR_USM_PTR(12, "double"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCgemm_v2",
    CALL("oneapi::mkl::blas::column_major::gemm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6), DEREF(0)),
         BUFFER_OR_USM_PTR(7, "std::complex<float>"), ARG(8),
         BUFFER_OR_USM_PTR(9, "std::complex<float>"), ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11), DEREF(0)),
         BUFFER_OR_USM_PTR(12, "std::complex<float>"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZgemm_v2",
    CALL("oneapi::mkl::blas::column_major::gemm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6), DEREF(0)),
         BUFFER_OR_USM_PTR(7, "std::complex<double>"), ARG(8),
         BUFFER_OR_USM_PTR(9, "std::complex<double>"), ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11), DEREF(0)),
         BUFFER_OR_USM_PTR(12, "std::complex<double>"), ARG(13))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasNrm2Ex",
                       CALL(MapNames::getDpctNamespace() + "nrm2", MEMBER_CALL(ARG(0), true, "get_queue"),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasDotEx",
                       CALL(MapNames::getDpctNamespace() + "dot", MEMBER_CALL(ARG(0), true, "get_queue"),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasDotcEx",
                       CALL(MapNames::getDpctNamespace() + "dotc", MEMBER_CALL(ARG(0), true, "get_queue"),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasScalEx",
                       CALL(MapNames::getDpctNamespace() + "scal", MEMBER_CALL(ARG(0), true, "get_queue"),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasAxpyEx",
                       CALL(MapNames::getDpctNamespace() + "axpy", MEMBER_CALL(ARG(0), true, "get_queue"),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasRotEx",
                       CALL(MapNames::getDpctNamespace() + "rot", MEMBER_CALL(ARG(0), true, "get_queue"),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasGemmEx",
        CALL(MapNames::getDpctNamespace() + "gemm", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasGemmBatchedEx",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(7), BOOL(true), BOOL(false)), ARG(8), ARG(9),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(10), BOOL(true), BOOL(false)),
             ARG(11), ARG(12), ARG(13), DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(14), BOOL(false), BOOL(false)), ARG(15), ARG(16), ARG(17), ARG(18)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasGemmStridedBatchedEx",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17),
             ARG(18), ARG(19), ARG(20), ARG(21)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSsyrkx",
        CALL(MapNames::getDpctNamespace() + "syrk", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDsyrkx",
        CALL(MapNames::getDpctNamespace() + "syrk", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCsyrkx",
        CALL(MapNames::getDpctNamespace() + "syrk", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZsyrkx",
        CALL(MapNames::getDpctNamespace() + "syrk", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCherkx",
        CALL(MapNames::getDpctNamespace() + "herk", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZherkx",
        CALL(MapNames::getDpctNamespace() + "herk", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasStrsmBatched",
        CALL(MapNames::getDpctNamespace() + "trsm_batch", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
             ARG(5), ARG(6), ARG(7), CAST(makeLiteral("const void**"), ARG(8)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"), ARG(9), CAST(makeLiteral("void**"), ARG(10)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"), ARG(11), ARG(12), ARG(MapNames::getDpctNamespace() + "library_data_t::real_float")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDtrsmBatched",
        CALL(MapNames::getDpctNamespace() + "trsm_batch", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
             ARG(5), ARG(6), ARG(7), CAST(makeLiteral("const void**"), ARG(8)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"), ARG(9), CAST(makeLiteral("void**"), ARG(10)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"), ARG(11), ARG(12), ARG(MapNames::getDpctNamespace() + "library_data_t::real_double")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCtrsmBatched",
        CALL(MapNames::getDpctNamespace() + "trsm_batch", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
             ARG(5), ARG(6), ARG(7), CAST(makeLiteral("const void**"), ARG(8)), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"), ARG(9), CAST(makeLiteral("void**"), ARG(10)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"), ARG(11), ARG(12), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZtrsmBatched",
        CALL(MapNames::getDpctNamespace() + "trsm_batch", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
             ARG(5), ARG(6), ARG(7), CAST(makeLiteral("const void**"), ARG(8)), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_double"), ARG(9), CAST(makeLiteral("void**"), ARG(10)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::complex_double"), ARG(11), ARG(12), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_double")))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasStrmm_v2",
        CALL(MapNames::getDpctNamespace() + "trmm", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
             ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDtrmm_v2",
        CALL(MapNames::getDpctNamespace() + "trmm", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
             ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCtrmm_v2",
        CALL(MapNames::getDpctNamespace() + "trmm", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
             ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZtrmm_v2",
        CALL(MapNames::getDpctNamespace() + "trmm", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
             ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasHgemmBatched",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), CAST(makeLiteral("const void**"), ARG(7)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_half"), ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_half"), ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_half"),
             ARG(13), ARG(14), ARG(MapNames::getDpctNamespace() + "library_data_t::real_half")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgemmBatched",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), CAST(makeLiteral("const void**"), ARG(7)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"), ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"), ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"),
             ARG(13), ARG(14), ARG(MapNames::getDpctNamespace() + "library_data_t::real_float")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgemmBatched",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), CAST(makeLiteral("const void**"), ARG(7)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"), ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"), ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"),
             ARG(13), ARG(14), ARG(MapNames::getDpctNamespace() + "library_data_t::real_double")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgemmBatched",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), CAST(makeLiteral("const void**"), ARG(7)), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"), ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"), ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"),
             ARG(13), ARG(14), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgemmBatched",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), CAST(makeLiteral("const void**"), ARG(7)), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_double"), ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::complex_double"), ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_double"),
             ARG(13), ARG(14), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_double")))))

// getrf
WARNING_FACTORY_ENTRY("cublasSgetrfBatched",
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgetrfBatched",
        CALL(MapNames::getDpctNamespace() + "getrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(2), BOOL(false), BOOL(false)), ARG(3), ARG(4), ARG(5), ARG(6))))),
Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4), ARG(MapNames::getDpctNamespace() + "getrf_batch_wrapper"), ARG("cublasSgetrfBatched"))
WARNING_FACTORY_ENTRY("cublasDgetrfBatched",
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgetrfBatched",
        CALL(MapNames::getDpctNamespace() + "getrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(2), BOOL(false), BOOL(false)), ARG(3), ARG(4), ARG(5), ARG(6))))),
Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4), ARG(MapNames::getDpctNamespace() + "getrf_batch_wrapper"), ARG("cublasDgetrfBatched"))
WARNING_FACTORY_ENTRY("cublasCgetrfBatched",
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgetrfBatched",
        CALL(MapNames::getDpctNamespace() + "getrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(2), BOOL(false), BOOL(false)), ARG(3), ARG(4), ARG(5), ARG(6))))),
Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4), ARG(MapNames::getDpctNamespace() + "getrf_batch_wrapper"), ARG("cublasCgetrfBatched"))
WARNING_FACTORY_ENTRY("cublasZgetrfBatched",
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgetrfBatched",
        CALL(MapNames::getDpctNamespace() + "getrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(2), BOOL(false), BOOL(false)), ARG(3), ARG(4), ARG(5), ARG(6))))),
Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4), ARG(MapNames::getDpctNamespace() + "getrf_batch_wrapper"), ARG("cublasZgetrfBatched"))

// getrs
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgetrsBatched",
        CALL(MapNames::getDpctNamespace() + "getrs_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(2), ARG(3), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(4), BOOL(true), BOOL(false)),
             ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(7), BOOL(false), BOOL(false)), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgetrsBatched",
        CALL(MapNames::getDpctNamespace() + "getrs_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(2), ARG(3), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(4), BOOL(true), BOOL(false)),
             ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(7), BOOL(false), BOOL(false)), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgetrsBatched",
        CALL(MapNames::getDpctNamespace() + "getrs_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(2), ARG(3), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(4), BOOL(true), BOOL(false)),
             ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(7), BOOL(false), BOOL(false)), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgetrsBatched",
        CALL(MapNames::getDpctNamespace() + "getrs_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(2), ARG(3), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(4), BOOL(true), BOOL(false)),
             ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(7), BOOL(false), BOOL(false)), ARG(8), ARG(9), ARG(10)))))

// getri
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgetriBatched",
        CALL(MapNames::getDpctNamespace() + "getri_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(2), BOOL(true), BOOL(false)),
             ARG(3), ARG(4), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7), ARG(8)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgetriBatched",
        CALL(MapNames::getDpctNamespace() + "getri_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(2), BOOL(true), BOOL(false)),
             ARG(3), ARG(4), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7), ARG(8)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgetriBatched",
        CALL(MapNames::getDpctNamespace() + "getri_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(2), BOOL(true), BOOL(false)),
             ARG(3), ARG(4), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7), ARG(8)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgetriBatched",
        CALL(MapNames::getDpctNamespace() + "getri_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(2), BOOL(true), BOOL(false)),
             ARG(3), ARG(4), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7), ARG(8)))))

// geqrf
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgeqrfBatched",
        CALL(MapNames::getDpctNamespace() + "geqrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), ARG(2), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(3), BOOL(false), BOOL(false)), ARG(4),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgeqrfBatched",
        CALL(MapNames::getDpctNamespace() + "geqrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), ARG(2), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(3), BOOL(false), BOOL(false)), ARG(4),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgeqrfBatched",
        CALL(MapNames::getDpctNamespace() + "geqrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), ARG(2), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(3), BOOL(false), BOOL(false)), ARG(4),
             DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgeqrfBatched",
        CALL(MapNames::getDpctNamespace() + "geqrf_batch_wrapper", MEMBER_CALL(ARG(0), true, "get_queue"),
             ARG(1), ARG(2), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(3), BOOL(false), BOOL(false)), ARG(4),
             DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasCrot_v2",
                       CALL(MapNames::getDpctNamespace() + "rot", MEMBER_CALL(ARG(0), true, "get_queue"),
                            ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"), ARG(3), ARG(4),
                            ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"),
                            ARG(5), ARG(6), ARG(7), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float")))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasZrot_v2",
                       CALL(MapNames::getDpctNamespace() + "rot", MEMBER_CALL(ARG(0), true, "get_queue"),
                            ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_double"), ARG(3), ARG(4),
                            ARG(MapNames::getDpctNamespace() + "library_data_t::complex_double"),
                            ARG(5), ARG(6), ARG(7), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_double")))))

WARNING_FACTORY_ENTRY(
    "cublasGetStatusString",
    INSERT_AROUND_FACTORY(CALL_FACTORY_ENTRY("cublasGetStatusString",
                                             CALL("cublasGetStatusString",
                                                  ARG_WC(0))),
                          "\"cublasGetStatusString is not supported\"/*", "*/"),
    Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_COMMENTED)

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgemmEx",
        CALL(MapNames::getDpctNamespace() + "gemm",
             MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13), ARG(14), ARG(15), ARG(16),
             ARG(MapNames::getDpctNamespace() +
                 "library_data_t::real_float")))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgemmEx",
        CALL(MapNames::getDpctNamespace() + "gemm",
             MEMBER_CALL(ARG(0), true, "get_queue"),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13), ARG(14), ARG(15), ARG(16),
             ARG(MapNames::getDpctNamespace() +
                 "library_data_t::complex_float")))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSgemm_v2_64",
    CALL("oneapi::mkl::blas::column_major::gemm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6), DEREF(0)),
         BUFFER_OR_USM_PTR(7, "float"), ARG(8), BUFFER_OR_USM_PTR(9, "float"),
         ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11), DEREF(0)),
         BUFFER_OR_USM_PTR(12, "float"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDgemm_v2_64",
    CALL("oneapi::mkl::blas::column_major::gemm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6), DEREF(0)),
         BUFFER_OR_USM_PTR(7, "double"), ARG(8), BUFFER_OR_USM_PTR(9, "double"),
         ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11), DEREF(0)),
         BUFFER_OR_USM_PTR(12, "double"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCgemm_v2_64",
    CALL("oneapi::mkl::blas::column_major::gemm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6), DEREF(0)),
         BUFFER_OR_USM_PTR(7, "std::complex<float>"), ARG(8),
         BUFFER_OR_USM_PTR(9, "std::complex<float>"), ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11), DEREF(0)),
         BUFFER_OR_USM_PTR(12, "std::complex<float>"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZgemm_v2_64",
    CALL("oneapi::mkl::blas::column_major::gemm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6), DEREF(0)),
         BUFFER_OR_USM_PTR(7, "std::complex<double>"), ARG(8),
         BUFFER_OR_USM_PTR(9, "std::complex<double>"), ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11), DEREF(0)),
         BUFFER_OR_USM_PTR(12, "std::complex<double>"), ARG(13))))
