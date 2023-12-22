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
                       CALL(MapNames::getDpctNamespace() + "nrm2", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasDotEx",
                       CALL(MapNames::getDpctNamespace() + "dot", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasDotcEx",
                       CALL(MapNames::getDpctNamespace() + "dotc", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasScalEx",
                       CALL(MapNames::getDpctNamespace() + "scal", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasAxpyEx",
                       CALL(MapNames::getDpctNamespace() + "axpy", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasRotEx",
                       CALL(MapNames::getDpctNamespace() + "rot", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasGemmEx",
        CALL(MapNames::getDpctNamespace() + "gemm", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasGemmBatchedEx",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(7), BOOL(true), BOOL(false)), ARG(8), ARG(9),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(10), BOOL(true), BOOL(false)),
             ARG(11), ARG(12), ARG(13), DOUBLE_POINTER_CONST_CAST(makeLiteral("void"), ARG(14), BOOL(false), BOOL(false)), ARG(15), ARG(16), ARG(17), ARG(18)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasGemmStridedBatchedEx",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17),
             ARG(18), ARG(19), ARG(20), ARG(21)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSsyrkx",
        CALL(MapNames::getDpctNamespace() + "syrk", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDsyrkx",
        CALL(MapNames::getDpctNamespace() + "syrk", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCsyrkx",
        CALL(MapNames::getDpctNamespace() + "syrk", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZsyrkx",
        CALL(MapNames::getDpctNamespace() + "syrk", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCherkx",
        CALL(MapNames::getDpctNamespace() + "herk", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZherkx",
        CALL(MapNames::getDpctNamespace() + "herk", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSgeam",
    CALL("oneapi::mkl::blas::column_major::omatadd", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "float"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "float"), ARG(10), BUFFER_OR_USM_PTR(11, "float"),
         ARG(12))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDgeam",
    CALL("oneapi::mkl::blas::column_major::omatadd", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "double"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "double"), ARG(10),
         BUFFER_OR_USM_PTR(11, "double"), ARG(12))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCgeam",
    CALL("oneapi::mkl::blas::column_major::omatadd", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "std::complex<float>"), ARG(10),
         BUFFER_OR_USM_PTR(11, "std::complex<float>"), ARG(12))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZgeam",
    CALL("oneapi::mkl::blas::column_major::omatadd", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "std::complex<double>"), ARG(10),
         BUFFER_OR_USM_PTR(11, "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSdgmm",
    CALL("oneapi::mkl::blas::column_major::dgmm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(4, "float"), ARG(5),
         BUFFER_OR_USM_PTR(6, "float"), ARG(7), BUFFER_OR_USM_PTR(8, "float"),
         ARG(9))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDdgmm",
    CALL("oneapi::mkl::blas::column_major::dgmm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(4, "double"), ARG(5),
         BUFFER_OR_USM_PTR(6, "double"), ARG(7), BUFFER_OR_USM_PTR(8, "double"),
         ARG(9))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCdgmm",
    CALL("oneapi::mkl::blas::column_major::dgmm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(4, "std::complex<float>"), ARG(5),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<float>"), ARG(9))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZdgmm",
    CALL("oneapi::mkl::blas::column_major::dgmm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(4, "std::complex<double>"), ARG(5),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<double>"), ARG(9))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasStrsmBatched",
        CALL(MapNames::getDpctNamespace() + "trsm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
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
        CALL(MapNames::getDpctNamespace() + "trsm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
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
        CALL(MapNames::getDpctNamespace() + "trsm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
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
        CALL(MapNames::getDpctNamespace() + "trsm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
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
        CALL(MapNames::getDpctNamespace() + "trmm", DEREF(makeDerefArgCreatorWithCall(0)),
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
        CALL(MapNames::getDpctNamespace() + "trmm", DEREF(makeDerefArgCreatorWithCall(0)),
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
        CALL(MapNames::getDpctNamespace() + "trmm", DEREF(makeDerefArgCreatorWithCall(0)),
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
        CALL(MapNames::getDpctNamespace() + "trmm", DEREF(makeDerefArgCreatorWithCall(0)),
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
        CALL(MapNames::getDpctNamespace() + "gemm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), CAST(makeLiteral("const void**"), ARG(7)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_half"), ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_half"), ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_half"),
             ARG(13), ARG(14), ARG(MapNames::getDpctNamespace() + "library_data_t::real_half")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgemmBatched",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), CAST(makeLiteral("const void**"), ARG(7)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"), ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"), ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"),
             ARG(13), ARG(14), ARG(MapNames::getDpctNamespace() + "library_data_t::real_float")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgemmBatched",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), CAST(makeLiteral("const void**"), ARG(7)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"), ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"), ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)), ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"),
             ARG(13), ARG(14), ARG(MapNames::getDpctNamespace() + "library_data_t::real_double")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgemmBatched",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), CAST(makeLiteral("const void**"), ARG(7)), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"), ARG(8), CAST(makeLiteral("const void**"), ARG(9)),
             ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"), ARG(10), ARG(11), CAST(makeLiteral("void**"), ARG(12)), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"),
             ARG(13), ARG(14), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float")))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgemmBatched",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
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
        CALL(MapNames::getDpctNamespace() + "getrf_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(2), BOOL(false), BOOL(false)), ARG(3), ARG(4), ARG(5), ARG(6))))),
Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4), ARG(MapNames::getDpctNamespace() + "getrf_batch_wrapper"), ARG("cublasSgetrfBatched"))
WARNING_FACTORY_ENTRY("cublasDgetrfBatched",
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgetrfBatched",
        CALL(MapNames::getDpctNamespace() + "getrf_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(2), BOOL(false), BOOL(false)), ARG(3), ARG(4), ARG(5), ARG(6))))),
Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4), ARG(MapNames::getDpctNamespace() + "getrf_batch_wrapper"), ARG("cublasDgetrfBatched"))
WARNING_FACTORY_ENTRY("cublasCgetrfBatched",
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgetrfBatched",
        CALL(MapNames::getDpctNamespace() + "getrf_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(2), BOOL(false), BOOL(false)), ARG(3), ARG(4), ARG(5), ARG(6))))),
Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4), ARG(MapNames::getDpctNamespace() + "getrf_batch_wrapper"), ARG("cublasCgetrfBatched"))
WARNING_FACTORY_ENTRY("cublasZgetrfBatched",
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgetrfBatched",
        CALL(MapNames::getDpctNamespace() + "getrf_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(2), BOOL(false), BOOL(false)), ARG(3), ARG(4), ARG(5), ARG(6))))),
Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(4), ARG(MapNames::getDpctNamespace() + "getrf_batch_wrapper"), ARG("cublasZgetrfBatched"))

// getrs
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgetrsBatched",
        CALL(MapNames::getDpctNamespace() + "getrs_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(2), ARG(3), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(4), BOOL(true), BOOL(false)),
             ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(7), BOOL(false), BOOL(false)), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgetrsBatched",
        CALL(MapNames::getDpctNamespace() + "getrs_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(2), ARG(3), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(4), BOOL(true), BOOL(false)),
             ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(7), BOOL(false), BOOL(false)), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgetrsBatched",
        CALL(MapNames::getDpctNamespace() + "getrs_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(2), ARG(3), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(4), BOOL(true), BOOL(false)),
             ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(7), BOOL(false), BOOL(false)), ARG(8), ARG(9), ARG(10)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgetrsBatched",
        CALL(MapNames::getDpctNamespace() + "getrs_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(2), ARG(3), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(4), BOOL(true), BOOL(false)),
             ARG(5), ARG(6), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(7), BOOL(false), BOOL(false)), ARG(8), ARG(9), ARG(10)))))

// getri
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgetriBatched",
        CALL(MapNames::getDpctNamespace() + "getri_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(2), BOOL(true), BOOL(false)),
             ARG(3), ARG(4), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7), ARG(8)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgetriBatched",
        CALL(MapNames::getDpctNamespace() + "getri_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(2), BOOL(true), BOOL(false)),
             ARG(3), ARG(4), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7), ARG(8)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgetriBatched",
        CALL(MapNames::getDpctNamespace() + "getri_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(2), BOOL(true), BOOL(false)),
             ARG(3), ARG(4), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7), ARG(8)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgetriBatched",
        CALL(MapNames::getDpctNamespace() + "getri_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             ARG(1), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(2), BOOL(true), BOOL(false)),
             ARG(3), ARG(4), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7), ARG(8)))))

// geqrf
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasSgeqrfBatched",
        CALL(MapNames::getDpctNamespace() + "geqrf_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             ARG(1), ARG(2), DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(3), BOOL(false), BOOL(false)), ARG(4),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("float"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasDgeqrfBatched",
        CALL(MapNames::getDpctNamespace() + "geqrf_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             ARG(1), ARG(2), DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(3), BOOL(false), BOOL(false)), ARG(4),
             DOUBLE_POINTER_CONST_CAST(makeLiteral("double"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasCgeqrfBatched",
        CALL(MapNames::getDpctNamespace() + "geqrf_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             ARG(1), ARG(2), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(3), BOOL(false), BOOL(false)), ARG(4),
             DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "float2"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7)))))
ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cublasZgeqrfBatched",
        CALL(MapNames::getDpctNamespace() + "geqrf_batch_wrapper", DEREF(makeDerefArgCreatorWithCall(0)),
             ARG(1), ARG(2), DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(3), BOOL(false), BOOL(false)), ARG(4),
             DOUBLE_POINTER_CONST_CAST(makeLiteral(MapNames::getClNamespace() + "double2"), ARG(5), BOOL(false), BOOL(false)), ARG(6), ARG(7)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasCrot_v2",
                       CALL(MapNames::getDpctNamespace() + "rot", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"), ARG(3), ARG(4),
                            ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"),
                            ARG(5), ARG(6), ARG(7), ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float")))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cublasZrot_v2",
                       CALL(MapNames::getDpctNamespace() + "rot", DEREF(makeDerefArgCreatorWithCall(0)),
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
             DEREF(makeDerefArgCreatorWithCall(0)),
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
             DEREF(makeDerefArgCreatorWithCall(0)),
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

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSsyrk_v2",
    CALL("oneapi::mkl::blas::column_major::syrk", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "float"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "float"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDsyrk_v2",
    CALL("oneapi::mkl::blas::column_major::syrk", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "double"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "double"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCsyrk_v2",
    CALL("oneapi::mkl::blas::column_major::syrk", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "std::complex<float>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZsyrk_v2",
    CALL("oneapi::mkl::blas::column_major::syrk", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "std::complex<double>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSsyrk_v2_64",
    CALL("oneapi::mkl::blas::column_major::syrk", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "float"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "float"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDsyrk_v2_64",
    CALL("oneapi::mkl::blas::column_major::syrk", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "double"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "double"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCsyrk_v2_64",
    CALL("oneapi::mkl::blas::column_major::syrk", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "std::complex<float>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZsyrk_v2_64",
    CALL("oneapi::mkl::blas::column_major::syrk", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "std::complex<double>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSsymm_v2",
    CALL("oneapi::mkl::blas::column_major::symm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "float"), ARG(7), BUFFER_OR_USM_PTR(8, "float"),
         ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "float"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDsymm_v2",
    CALL("oneapi::mkl::blas::column_major::symm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "double"), ARG(7), BUFFER_OR_USM_PTR(8, "double"),
         ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "double"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCsymm_v2",
    CALL("oneapi::mkl::blas::column_major::symm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZsymm_v2",
    CALL("oneapi::mkl::blas::column_major::symm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSsymm_v2_64",
    CALL("oneapi::mkl::blas::column_major::symm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "float"), ARG(7), BUFFER_OR_USM_PTR(8, "float"),
         ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "float"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDsymm_v2_64",
    CALL("oneapi::mkl::blas::column_major::symm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "double"), ARG(7), BUFFER_OR_USM_PTR(8, "double"),
         ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "double"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCsymm_v2_64",
    CALL("oneapi::mkl::blas::column_major::symm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZsymm_v2_64",
    CALL("oneapi::mkl::blas::column_major::symm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasStrsm_v2",
    CALL("oneapi::mkl::blas::column_major::trsm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7), DEREF(0)),
         BUFFER_OR_USM_PTR(8, "float"), ARG(9), BUFFER_OR_USM_PTR(10, "float"),
         ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDtrsm_v2",
    CALL("oneapi::mkl::blas::column_major::trsm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7), DEREF(0)),
         BUFFER_OR_USM_PTR(8, "double"), ARG(9),
         BUFFER_OR_USM_PTR(10, "double"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCtrsm_v2",
    CALL("oneapi::mkl::blas::column_major::trsm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7), DEREF(0)),
         BUFFER_OR_USM_PTR(8, "std::complex<float>"), ARG(9),
         BUFFER_OR_USM_PTR(10, "std::complex<float>"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZtrsm_v2",
    CALL("oneapi::mkl::blas::column_major::trsm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7), DEREF(0)),
         BUFFER_OR_USM_PTR(8, "std::complex<double>"), ARG(9),
         BUFFER_OR_USM_PTR(10, "std::complex<double>"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasStrsm_v2_64",
    CALL("oneapi::mkl::blas::column_major::trsm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7), DEREF(0)),
         BUFFER_OR_USM_PTR(8, "float"), ARG(9), BUFFER_OR_USM_PTR(10, "float"),
         ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDtrsm_v2_64",
    CALL("oneapi::mkl::blas::column_major::trsm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7), DEREF(0)),
         BUFFER_OR_USM_PTR(8, "double"), ARG(9),
         BUFFER_OR_USM_PTR(10, "double"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCtrsm_v2_64",
    CALL("oneapi::mkl::blas::column_major::trsm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7), DEREF(0)),
         BUFFER_OR_USM_PTR(8, "std::complex<float>"), ARG(9),
         BUFFER_OR_USM_PTR(10, "std::complex<float>"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZtrsm_v2_64",
    CALL("oneapi::mkl::blas::column_major::trsm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7), DEREF(0)),
         BUFFER_OR_USM_PTR(8, "std::complex<double>"), ARG(9),
         BUFFER_OR_USM_PTR(10, "std::complex<double>"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasChemm_v2",
    CALL("oneapi::mkl::blas::column_major::hemm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZhemm_v2",
    CALL("oneapi::mkl::blas::column_major::hemm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasChemm_v2_64",
    CALL("oneapi::mkl::blas::column_major::hemm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZhemm_v2_64",
    CALL("oneapi::mkl::blas::column_major::hemm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCherk_v2",
    CALL("oneapi::mkl::blas::column_major::herk", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "std::complex<float>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZherk_v2",
    CALL("oneapi::mkl::blas::column_major::herk", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "std::complex<double>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCherk_v2_64",
    CALL("oneapi::mkl::blas::column_major::herk", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "std::complex<float>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZherk_v2_64",
    CALL("oneapi::mkl::blas::column_major::herk", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "std::complex<double>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSsyr2k_v2",
    CALL("oneapi::mkl::blas::column_major::syr2k", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "float"), ARG(7), BUFFER_OR_USM_PTR(8, "float"),
         ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "float"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDsyr2k_v2",
    CALL("oneapi::mkl::blas::column_major::syr2k", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "double"), ARG(7), BUFFER_OR_USM_PTR(8, "double"),
         ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "double"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCsyr2k_v2",
    CALL("oneapi::mkl::blas::column_major::syr2k", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZsyr2k_v2",
    CALL("oneapi::mkl::blas::column_major::syr2k", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSsyr2k_v2_64",
    CALL("oneapi::mkl::blas::column_major::syr2k", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "float"), ARG(7), BUFFER_OR_USM_PTR(8, "float"),
         ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "float"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDsyr2k_v2_64",
    CALL("oneapi::mkl::blas::column_major::syr2k", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "double"), ARG(7), BUFFER_OR_USM_PTR(8, "double"),
         ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "double"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCsyr2k_v2_64",
    CALL("oneapi::mkl::blas::column_major::syr2k", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZsyr2k_v2_64",
    CALL("oneapi::mkl::blas::column_major::syr2k", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCher2k_v2",
    CALL("oneapi::mkl::blas::column_major::her2k", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZher2k_v2",
    CALL("oneapi::mkl::blas::column_major::her2k", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCher2k_v2_64",
    CALL("oneapi::mkl::blas::column_major::her2k", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZher2k_v2_64",
    CALL("oneapi::mkl::blas::column_major::her2k", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10), DEREF(0)),
         BUFFER_OR_USM_PTR(11, "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSgeam_64",
    CALL("oneapi::mkl::blas::column_major::omatadd", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "float"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "float"), ARG(10), BUFFER_OR_USM_PTR(11, "float"),
         ARG(12))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDgeam_64",
    CALL("oneapi::mkl::blas::column_major::omatadd", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "double"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "double"), ARG(10),
         BUFFER_OR_USM_PTR(11, "double"), ARG(12))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCgeam_64",
    CALL("oneapi::mkl::blas::column_major::omatadd", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "std::complex<float>"), ARG(10),
         BUFFER_OR_USM_PTR(11, "std::complex<float>"), ARG(12))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZgeam_64",
    CALL("oneapi::mkl::blas::column_major::omatadd", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5), DEREF(0)),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8), DEREF(0)),
         BUFFER_OR_USM_PTR(9, "std::complex<double>"), ARG(10),
         BUFFER_OR_USM_PTR(11, "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSdgmm_64",
    CALL("oneapi::mkl::blas::column_major::dgmm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(4, "float"), ARG(5),
         BUFFER_OR_USM_PTR(6, "float"), ARG(7), BUFFER_OR_USM_PTR(8, "float"),
         ARG(9))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDdgmm_64",
    CALL("oneapi::mkl::blas::column_major::dgmm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(4, "double"), ARG(5),
         BUFFER_OR_USM_PTR(6, "double"), ARG(7), BUFFER_OR_USM_PTR(8, "double"),
         ARG(9))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCdgmm_64",
    CALL("oneapi::mkl::blas::column_major::dgmm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(4, "std::complex<float>"), ARG(5),
         BUFFER_OR_USM_PTR(6, "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<float>"), ARG(9))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZdgmm_64",
    CALL("oneapi::mkl::blas::column_major::dgmm", DEREF(0),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         ARG(2), ARG(3), BUFFER_OR_USM_PTR(4, "std::complex<double>"), ARG(5),
         BUFFER_OR_USM_PTR(6, "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(8, "std::complex<double>"), ARG(9))))
