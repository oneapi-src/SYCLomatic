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

ASSIGNABLE_FACTORY(FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGN_FACTORY_ENTRY("cublasCreate_v2", DEREF(0),
                         NEW(MapNames::getDpctNamespace() +
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
    CALL(MapNames::getDpctNamespace() + "blas::descriptor::set_saved_queue",
         ARG(0))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSgemm_v2",
    CALL("oneapi::mkl::blas::column_major::gemm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(7), "float"), ARG(8),
         BUFFER_OR_USM_PTR(ARG(9), "float"), ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(12), "float"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDgemm_v2",
    CALL("oneapi::mkl::blas::column_major::gemm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(7), "double"), ARG(8),
         BUFFER_OR_USM_PTR(ARG(9), "double"), ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(12), "double"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCgemm_v2",
    CALL("oneapi::mkl::blas::column_major::gemm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(7), "std::complex<float>"), ARG(8),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<float>"), ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(12), "std::complex<float>"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZgemm_v2",
    CALL("oneapi::mkl::blas::column_major::gemm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(7), "std::complex<double>"), ARG(8),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<double>"), ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(12), "std::complex<double>"), ARG(13))))

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

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSgeam",
    CALL("oneapi::mkl::blas::column_major::omatadd",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
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
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
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
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
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
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
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

WARNING_FACTORY_ENTRY(
    "cublasGetStatusString",
    TOSTRING_FACTORY_ENTRY("cublasGetStatusString",
                           LITERAL("\"<Placeholder string>\"")),
    Diagnostics::TRNA_WARNING_ERROR_HANDLING_API_COMMENTED,
    ARG("The call was replaced by a placeholder string"))

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
    CALL("oneapi::mkl::blas::column_major::gemm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(7), "float"), ARG(8),
         BUFFER_OR_USM_PTR(ARG(9), "float"), ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(12), "float"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDgemm_v2_64",
    CALL("oneapi::mkl::blas::column_major::gemm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(7), "double"), ARG(8),
         BUFFER_OR_USM_PTR(ARG(9), "double"), ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(12), "double"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCgemm_v2_64",
    CALL("oneapi::mkl::blas::column_major::gemm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(7), "std::complex<float>"), ARG(8),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<float>"), ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(12), "std::complex<float>"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZgemm_v2_64",
    CALL("oneapi::mkl::blas::column_major::gemm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4), ARG(5),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(6),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(7), "std::complex<double>"), ARG(8),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<double>"), ARG(10),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(11),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(12), "std::complex<double>"), ARG(13))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSsyrk_v2",
    CALL("oneapi::mkl::blas::column_major::syrk",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "float"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDsyrk_v2",
    CALL("oneapi::mkl::blas::column_major::syrk",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "double"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCsyrk_v2",
    CALL("oneapi::mkl::blas::column_major::syrk",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<float>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZsyrk_v2",
    CALL("oneapi::mkl::blas::column_major::syrk",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<double>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSsyrk_v2_64",
    CALL("oneapi::mkl::blas::column_major::syrk",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "float"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDsyrk_v2_64",
    CALL("oneapi::mkl::blas::column_major::syrk",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "double"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCsyrk_v2_64",
    CALL("oneapi::mkl::blas::column_major::syrk",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<float>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZsyrk_v2_64",
    CALL("oneapi::mkl::blas::column_major::syrk",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<double>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSsymm_v2",
    CALL("oneapi::mkl::blas::column_major::symm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "float"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "float"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDsymm_v2",
    CALL("oneapi::mkl::blas::column_major::symm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "double"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "double"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCsymm_v2",
    CALL("oneapi::mkl::blas::column_major::symm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZsymm_v2",
    CALL("oneapi::mkl::blas::column_major::symm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSsymm_v2_64",
    CALL("oneapi::mkl::blas::column_major::symm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "float"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "float"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDsymm_v2_64",
    CALL("oneapi::mkl::blas::column_major::symm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "double"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "double"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCsymm_v2_64",
    CALL("oneapi::mkl::blas::column_major::symm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZsymm_v2_64",
    CALL("oneapi::mkl::blas::column_major::symm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasStrsm_v2",
    CALL("oneapi::mkl::blas::column_major::trsm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(8), "float"), ARG(9),
         BUFFER_OR_USM_PTR(ARG(10), "float"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDtrsm_v2",
    CALL("oneapi::mkl::blas::column_major::trsm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(8), "double"), ARG(9),
         BUFFER_OR_USM_PTR(ARG(10), "double"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCtrsm_v2",
    CALL("oneapi::mkl::blas::column_major::trsm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9),
         BUFFER_OR_USM_PTR(ARG(10), "std::complex<float>"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZtrsm_v2",
    CALL("oneapi::mkl::blas::column_major::trsm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9),
         BUFFER_OR_USM_PTR(ARG(10), "std::complex<double>"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasStrsm_v2_64",
    CALL("oneapi::mkl::blas::column_major::trsm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(8), "float"), ARG(9),
         BUFFER_OR_USM_PTR(ARG(10), "float"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDtrsm_v2_64",
    CALL("oneapi::mkl::blas::column_major::trsm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(8), "double"), ARG(9),
         BUFFER_OR_USM_PTR(ARG(10), "double"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCtrsm_v2_64",
    CALL("oneapi::mkl::blas::column_major::trsm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9),
         BUFFER_OR_USM_PTR(ARG(10), "std::complex<float>"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZtrsm_v2_64",
    CALL("oneapi::mkl::blas::column_major::trsm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(3, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(4, clang::dpct::BLASEnumExpr::BLASEnumType::Diag),
         ARG(5), ARG(6),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(7),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9),
         BUFFER_OR_USM_PTR(ARG(10), "std::complex<double>"), ARG(11))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasChemm_v2",
    CALL("oneapi::mkl::blas::column_major::hemm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZhemm_v2",
    CALL("oneapi::mkl::blas::column_major::hemm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasChemm_v2_64",
    CALL("oneapi::mkl::blas::column_major::hemm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZhemm_v2_64",
    CALL("oneapi::mkl::blas::column_major::hemm",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Side),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCherk_v2",
    CALL("oneapi::mkl::blas::column_major::herk",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<float>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZherk_v2",
    CALL("oneapi::mkl::blas::column_major::herk",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<double>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCherk_v2_64",
    CALL("oneapi::mkl::blas::column_major::herk",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<float>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZherk_v2_64",
    CALL("oneapi::mkl::blas::column_major::herk",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(9), "std::complex<double>"), ARG(10))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSsyr2k_v2",
    CALL("oneapi::mkl::blas::column_major::syr2k",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "float"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "float"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDsyr2k_v2",
    CALL("oneapi::mkl::blas::column_major::syr2k",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "double"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "double"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCsyr2k_v2",
    CALL("oneapi::mkl::blas::column_major::syr2k",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZsyr2k_v2",
    CALL("oneapi::mkl::blas::column_major::syr2k",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSsyr2k_v2_64",
    CALL("oneapi::mkl::blas::column_major::syr2k",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "float"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "float"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasDsyr2k_v2_64",
    CALL("oneapi::mkl::blas::column_major::syr2k",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "double"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "double"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCsyr2k_v2_64",
    CALL("oneapi::mkl::blas::column_major::syr2k",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZsyr2k_v2_64",
    CALL("oneapi::mkl::blas::column_major::syr2k",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCher2k_v2",
    CALL("oneapi::mkl::blas::column_major::her2k",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZher2k_v2",
    CALL("oneapi::mkl::blas::column_major::her2k",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasCher2k_v2_64",
    CALL("oneapi::mkl::blas::column_major::her2k",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<float>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<float>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasZher2k_v2_64",
    CALL("oneapi::mkl::blas::column_major::her2k",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Uplo),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         BUFFER_OR_USM_PTR(ARG(8), "std::complex<double>"), ARG(9),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(10),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(11), "std::complex<double>"), ARG(12))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cublasSgeam_64",
    CALL("oneapi::mkl::blas::column_major::omatadd",
         MEMBER_CALL(ARG(0), true, "get_queue"),
         BLAS_ENUM_ARG(1, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         BLAS_ENUM_ARG(2, clang::dpct::BLASEnumExpr::BLASEnumType::Trans),
         ARG(3), ARG(4),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "float"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
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
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "double"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
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
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<float>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
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
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(5),
              MEMBER_CALL(ARG(0), true, "get_queue")),
         BUFFER_OR_USM_PTR(ARG(6), "std::complex<double>"), ARG(7),
         CALL(MapNames::getDpctNamespace() + "get_value", ARG(8),
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int_to_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int_to_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int_to_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int_to_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int_to_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int_to_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int_to_int64_out",
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
        DECL(MapNames::getDpctNamespace() + "blas::wrapper_int_to_int64_out",
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
          DECL(MapNames::getDpctNamespace() + "blas::" #TYPE1,                 \
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
          DECL(MapNames::getDpctNamespace() + "blas::" #TYPE1,                 \
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
          DECL(MapNames::getDpctNamespace() + "blas::" #TYPE1,                 \
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
      CALL("oneapi::mkl::blas::column_major::gemv",                            \
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
          DECL(MapNames::getDpctNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct1", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(1)),                                                        \
          DECL(MapNames::getDpctNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct2", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(2)),                                                        \
          DECL(MapNames::getDpctNamespace() + "blas::" #TYPE2,                 \
               "res_wrapper_ct3", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(3)),                                                        \
          DECL(MapNames::getDpctNamespace() + "blas::" #TYPE3,                 \
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
ROTG(cublasSrotg_v2_64, wrapper_float_inout, wrapper_float_out, wrapper_float_out, float, float)
ROTG(cublasDrotg_v2_64, wrapper_double_inout, wrapper_double_out, wrapper_double_out, double, double)
ROTG(cublasCrotg_v2_64, wrapper_float2_inout, wrapper_float_out, wrapper_float2_out, std::complex<float>, float)
ROTG(cublasZrotg_v2_64, wrapper_double2_inout, wrapper_double_out, wrapper_double2_out, std::complex<double>, double)
#undef ROTG

#define ROTM(FUNC, TYPE1, TYPE2)                                               \
  WARNING_FACTORY_ENTRY(                                                       \
      #FUNC,                                                                   \
      LAMBDA_FACTORY_ENTRY(                                                    \
          #FUNC,                                                               \
          DECL(MapNames::getDpctNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct6", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(6), ARG("5")),                                              \
          CALL("oneapi::mkl::blas::column_major::rotm",                        \
               MEMBER_CALL(ARG(0), true, "get_queue"), ARG(1),                 \
               BUFFER_OR_USM_PTR(ARG(2), #TYPE2), ARG(3),                      \
               BUFFER_OR_USM_PTR(ARG(4), #TYPE2), ARG(5),                      \
               BUFFER_OR_USM_PTR(                                              \
                   MEMBER_CALL(ARG("res_wrapper_ct6"), false, "get_ptr"),      \
                   #TYPE2),                                                    \
               LITERAL("return 0"))),                                          \
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
          DECL(MapNames::getDpctNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct1", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(1)),                                                        \
          DECL(MapNames::getDpctNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct2", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(2)),                                                        \
          DECL(MapNames::getDpctNamespace() + "blas::" #TYPE1,                 \
               "res_wrapper_ct3", MEMBER_CALL(ARG(0), true, "get_queue"),      \
               ARG(3)),                                                        \
          DECL(MapNames::getDpctNamespace() + "blas::" #TYPE2,                 \
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
ROTMG(cublasSrotmg_v2_64, wrapper_float_inout, wrapper_float_out, float)
ROTMG(cublasDrotmg_v2_64, wrapper_double_inout, wrapper_double_out, double)
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
