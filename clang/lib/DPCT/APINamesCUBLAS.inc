//===---APINamesCUBLAS.inc ---------------------------------------------===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LibCommonUtils_mkl_get_version,
    CALL_FACTORY_ENTRY("cublasGetVersion_v2",
                       CALL(MapNames::getDpctNamespace() + "mkl_get_version",
                            ARG(MapNames::getDpctNamespace() + "version_field::major"), ARG(1)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LibCommonUtils_mkl_get_version,
    CALL_FACTORY_ENTRY("cublasGetVersion",
                       CALL(MapNames::getDpctNamespace() + "mkl_get_version",
                            ARG(MapNames::getDpctNamespace() + "version_field::major"), ARG(0)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_nrm2,
    CALL_FACTORY_ENTRY("cublasNrm2Ex",
                       CALL(MapNames::getDpctNamespace() + "nrm2", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_dot,
    CALL_FACTORY_ENTRY("cublasDotEx",
                       CALL(MapNames::getDpctNamespace() + "dot", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_dotc,
    CALL_FACTORY_ENTRY("cublasDotcEx",
                       CALL(MapNames::getDpctNamespace() + "dotc", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_scal,
    CALL_FACTORY_ENTRY("cublasScalEx",
                       CALL(MapNames::getDpctNamespace() + "scal", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_axpy,
    CALL_FACTORY_ENTRY("cublasAxpyEx",
                       CALL(MapNames::getDpctNamespace() + "axpy", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_rot,
    CALL_FACTORY_ENTRY("cublasRotEx",
                       CALL(MapNames::getDpctNamespace() + "rot", DEREF(makeDerefArgCreatorWithCall(0)),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_gemm,
    CALL_FACTORY_ENTRY(
        "cublasGemmEx",
        CALL(MapNames::getDpctNamespace() + "gemm", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_gemm_batch,
    CALL_FACTORY_ENTRY(
        "cublasGemmBatchedEx",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17), ARG(18)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_gemm_batch_stride,
    CALL_FACTORY_ENTRY(
        "cublasGemmStridedBatchedEx",
        CALL(MapNames::getDpctNamespace() + "gemm_batch", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17),
             ARG(18), ARG(19), ARG(20), ARG(21)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_syrk,
    CALL_FACTORY_ENTRY(
        "cublasSsyrkx",
        CALL(MapNames::getDpctNamespace() + "syrk", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_syrk,
    CALL_FACTORY_ENTRY(
        "cublasDsyrkx",
        CALL(MapNames::getDpctNamespace() + "syrk", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_syrk,
    CALL_FACTORY_ENTRY(
        "cublasCsyrkx",
        CALL(MapNames::getDpctNamespace() + "syrk", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_syrk,
    CALL_FACTORY_ENTRY(
        "cublasZsyrkx",
        CALL(MapNames::getDpctNamespace() + "syrk", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_herk,
    CALL_FACTORY_ENTRY(
        "cublasCherkx",
        CALL(MapNames::getDpctNamespace() + "herk", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::BlasUtils_herk,
    CALL_FACTORY_ENTRY(
        "cublasZherkx",
        CALL(MapNames::getDpctNamespace() + "herk", DEREF(makeDerefArgCreatorWithCall(0)),
             BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
             BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
             ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
             ARG(11), ARG(12)))))
