//===------------------- APINamesCUSOLVER.inc ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusolverDnSetStream", ARG(0), ARG(1)))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusolverDnGetStream", DEREF(1),
                                        ARG(0)))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_sygvd,
    CALL_FACTORY_ENTRY("cusolverDnSsygvd",
                       CALL(MapNames::getDpctNamespace() + "lapack::sygvd",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                            ARG(12))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnSsygvd_bufferSize", DEREF(10),
    CALL("oneapi::mkl::lapack::sygvd_scratchpad_size<float>", DEREF(0), ARG(1),
         ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_sygvd,
    CALL_FACTORY_ENTRY("cusolverDnDsygvd",
                       CALL(MapNames::getDpctNamespace() + "lapack::sygvd",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                            ARG(12))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnDsygvd_bufferSize", DEREF(10),
    CALL("oneapi::mkl::lapack::sygvd_scratchpad_size<double>", DEREF(0), ARG(1),
         ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_hegvd,
    CALL_FACTORY_ENTRY("cusolverDnChegvd",
                       CALL(MapNames::getDpctNamespace() + "lapack::hegvd",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                            ARG(12))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnChegvd_bufferSize", DEREF(10),
    CALL("oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<float>>",
         DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_hegvd,
    CALL_FACTORY_ENTRY("cusolverDnZhegvd",
                       CALL(MapNames::getDpctNamespace() + "lapack::hegvd",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                            ARG(12))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnZhegvd_bufferSize", DEREF(10),
    CALL("oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<double>>",
         DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::LapackUtils_potrf_batch,
                        CALL_FACTORY_ENTRY("cusolverDnSpotrfBatched",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "lapack::potrf_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5),
                                                ARG(6))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::LapackUtils_potrf_batch,
                        CALL_FACTORY_ENTRY("cusolverDnDpotrfBatched",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "lapack::potrf_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5),
                                                ARG(6))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::LapackUtils_potrf_batch,
                        CALL_FACTORY_ENTRY("cusolverDnCpotrfBatched",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "lapack::potrf_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5),
                                                ARG(6))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::LapackUtils_potrf_batch,
                        CALL_FACTORY_ENTRY("cusolverDnZpotrfBatched",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "lapack::potrf_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5),
                                                ARG(6))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::LapackUtils_potrs_batch,
                        CALL_FACTORY_ENTRY("cusolverDnSpotrsBatched",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "lapack::potrs_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5), ARG(6),
                                                ARG(7), ARG(8), ARG(9))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::LapackUtils_potrs_batch,
                        CALL_FACTORY_ENTRY("cusolverDnDpotrsBatched",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "lapack::potrs_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5), ARG(6),
                                                ARG(7), ARG(8), ARG(9))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::LapackUtils_potrs_batch,
                        CALL_FACTORY_ENTRY("cusolverDnCpotrsBatched",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "lapack::potrs_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5), ARG(6),
                                                ARG(7), ARG(8), ARG(9))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::LapackUtils_potrs_batch,
                        CALL_FACTORY_ENTRY("cusolverDnZpotrsBatched",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "lapack::potrs_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5), ARG(6),
                                                ARG(7), ARG(8), ARG(9))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_getrf_scratchpad_size,
    COMMA_OP_FACTORY_ENTRY(
        BINARY_OP_FACTORY_ENTRY("cusolverDnXgetrf_bufferSize",
                                BinaryOperatorKind::BO_Assign, DEREF(9),
                                ARG("0")),
        CALL_FACTORY_ENTRY(
            "cusolverDnXgetrf_bufferSize",
            CALL(MapNames::getDpctNamespace() + "lapack::getrf_scratchpad_size",
                 DEREF(0), ARG(2), ARG(3), ARG(4), ARG(6), ARG(8)))))

WARNING_FACTORY_ENTRY(
    "cusolverDnXgetrf",
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::LapackUtils_getrf,
        CALL_FACTORY_ENTRY("cusolverDnXgetrf",
                           CALL(MapNames::getDpctNamespace() + "lapack::getrf",
                                DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5),
                                ARG(6), ARG(7), ARG(9), ARG(10), ARG(13)))),
    Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(7),
    ARG(MapNames::getDpctNamespace() + "lapack::getrf"),
    ARG("cusolverDnXgetrf"))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_getrs,
    CALL_FACTORY_ENTRY("cusolverDnXgetrs",
                       CALL(MapNames::getDpctNamespace() + "lapack::getrs",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10), ARG(11), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_geqrf_scratchpad_size,
    COMMA_OP_FACTORY_ENTRY(
        BINARY_OP_FACTORY_ENTRY("cusolverDnXgeqrf_bufferSize",
                                BinaryOperatorKind::BO_Assign, DEREF(11),
                                ARG("0")),
        CALL_FACTORY_ENTRY(
            "cusolverDnXgeqrf_bufferSize",
            CALL(MapNames::getDpctNamespace() + "lapack::geqrf_scratchpad_size",
                 DEREF(0), ARG(2), ARG(3), ARG(4), ARG(6), ARG(10)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_geqrf,
    CALL_FACTORY_ENTRY("cusolverDnXgeqrf",
                       CALL(MapNames::getDpctNamespace() + "lapack::geqrf",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(10), ARG(11), ARG(14))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_getrf_scratchpad_size,
    CALL_FACTORY_ENTRY("cusolverDnGetrf_bufferSize",
                       CALL(MapNames::getDpctNamespace() +
                                "lapack::getrf_scratchpad_size",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))

WARNING_FACTORY_ENTRY(
    "cusolverDnGetrf",
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::LapackUtils_getrf,
        CALL_FACTORY_ENTRY("cusolverDnGetrf",
                           CALL(MapNames::getDpctNamespace() + "lapack::getrf",
                                DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5),
                                ARG(6), ARG(7), ARG(9), ARG(10), ARG(11)))),
    Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(7),
    ARG(MapNames::getDpctNamespace() + "lapack::getrf"), ARG("cusolverDnGetrf"))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_getrs,
    CALL_FACTORY_ENTRY("cusolverDnGetrs",
                       CALL(MapNames::getDpctNamespace() + "lapack::getrs",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10), ARG(11), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_geqrf_scratchpad_size,
    CALL_FACTORY_ENTRY("cusolverDnGeqrf_bufferSize",
                       CALL(MapNames::getDpctNamespace() +
                                "lapack::geqrf_scratchpad_size",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(6), ARG(10))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_geqrf,
    CALL_FACTORY_ENTRY("cusolverDnGeqrf",
                       CALL(MapNames::getDpctNamespace() + "lapack::geqrf",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(10), ARG(11), ARG(12))))

REMOVE_API_FACTORY_ENTRY("cusolverDnCreateParams")
REMOVE_API_FACTORY_ENTRY("cusolverDnDestroyParams")

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_gesvd_scratchpad_size,
    COMMA_OP_FACTORY_ENTRY(
        BINARY_OP_FACTORY_ENTRY("cusolverDnXgesvd_bufferSize",
                                BinaryOperatorKind::BO_Assign, DEREF(19),
                                ARG("0")),
        CALL_FACTORY_ENTRY(
            "cusolverDnXgesvd_bufferSize",
            CALL(MapNames::getDpctNamespace() + "lapack::gesvd_scratchpad_size",
                 DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(8),
                 ARG(11), ARG(13), ARG(14), ARG(16), ARG(18)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_gesvd,
    CALL_FACTORY_ENTRY("cusolverDnXgesvd",
                       CALL(MapNames::getDpctNamespace() + "lapack::gesvd",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10), ARG(11), ARG(12),
                            ARG(13), ARG(14), ARG(15), ARG(16), ARG(18),
                            ARG(19), ARG(22))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_gesvd_scratchpad_size,
    CALL_FACTORY_ENTRY(
        "cusolverDnGesvd_bufferSize",
        CALL(MapNames::getDpctNamespace() + "lapack::gesvd_scratchpad_size",
             DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(8), ARG(11),
             ARG(13), ARG(14), ARG(16), ARG(18))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_gesvd,
    CALL_FACTORY_ENTRY("cusolverDnGesvd",
                       CALL(MapNames::getDpctNamespace() + "lapack::gesvd",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10), ARG(11), ARG(12),
                            ARG(13), ARG(14), ARG(15), ARG(16), ARG(18),
                            ARG(19), ARG(20))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_potrf_scratchpad_size,
    COMMA_OP_FACTORY_ENTRY(
        BINARY_OP_FACTORY_ENTRY("cusolverDnXpotrf_bufferSize",
                                BinaryOperatorKind::BO_Assign, DEREF(9),
                                ARG("0")),
        CALL_FACTORY_ENTRY(
            "cusolverDnXpotrf_bufferSize",
            CALL(MapNames::getDpctNamespace() + "lapack::potrf_scratchpad_size",
                 DEREF(0), ARG(2), ARG(3), ARG(4), ARG(6), ARG(8)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_potrf,
    CALL_FACTORY_ENTRY("cusolverDnXpotrf",
                       CALL(MapNames::getDpctNamespace() + "lapack::potrf",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(8), ARG(9), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_potrf_scratchpad_size,
    CALL_FACTORY_ENTRY("cusolverDnPotrf_bufferSize",
                       CALL(MapNames::getDpctNamespace() +
                                "lapack::potrf_scratchpad_size",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_potrf,
    CALL_FACTORY_ENTRY("cusolverDnPotrf",
                       CALL(MapNames::getDpctNamespace() + "lapack::potrf",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(8), ARG(9), ARG(10))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_potrs,
    CALL_FACTORY_ENTRY("cusolverDnXpotrs",
                       CALL(MapNames::getDpctNamespace() + "lapack::potrs",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10), ARG(11))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_potrs,
    CALL_FACTORY_ENTRY("cusolverDnPotrs",
                       CALL(MapNames::getDpctNamespace() + "lapack::potrs",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10), ARG(11))))

REMOVE_API_FACTORY_ENTRY("cusolverDnCreateGesvdjInfo")
REMOVE_API_FACTORY_ENTRY("cusolverDnDestroyGesvdjInfo")

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_gesvd_scratchpad_size_2,
    CALL_FACTORY_ENTRY(
        "cusolverDnSgesvdj_bufferSize",
        CALL(MapNames::getDpctNamespace() + "lapack::gesvd_scratchpad_size",
             DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"),
             ARG(6),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"),
             ARG(9),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"),
             ARG(11), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_gesvd_2,
    CALL_FACTORY_ENTRY(
        "cusolverDnSgesvdj",
        CALL(MapNames::getDpctNamespace() + "lapack::gesvd", DEREF(0), ARG(1),
             ARG(2), ARG(3), ARG(4),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"),
             ARG(5), ARG(6),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"),
             ARG(7),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"),
             ARG(8), ARG(9),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"),
             ARG(10), ARG(11), ARG(12), ARG(13), ARG(14))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_gesvd_scratchpad_size_2,
    CALL_FACTORY_ENTRY(
        "cusolverDnDgesvdj_bufferSize",
        CALL(MapNames::getDpctNamespace() + "lapack::gesvd_scratchpad_size",
             DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"),
             ARG(6),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"),
             ARG(9),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"),
             ARG(11), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_gesvd_2,
    CALL_FACTORY_ENTRY(
        "cusolverDnDgesvdj",
        CALL(MapNames::getDpctNamespace() + "lapack::gesvd", DEREF(0), ARG(1),
             ARG(2), ARG(3), ARG(4),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"),
             ARG(5), ARG(6),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"),
             ARG(7),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"),
             ARG(8), ARG(9),
             ARG(MapNames::getDpctNamespace() + "library_data_t::real_double"),
             ARG(10), ARG(11), ARG(12), ARG(13), ARG(14))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_gesvd_scratchpad_size_2,
    CALL_FACTORY_ENTRY(
        "cusolverDnCgesvdj_bufferSize",
        CALL(
            MapNames::getDpctNamespace() + "lapack::gesvd_scratchpad_size",
            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
            ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"),
            ARG(6),
            ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"),
            ARG(9),
            ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"),
            ARG(11), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_gesvd_2,
    CALL_FACTORY_ENTRY(
        "cusolverDnCgesvdj",
        CALL(
            MapNames::getDpctNamespace() + "lapack::gesvd", DEREF(0), ARG(1),
            ARG(2), ARG(3), ARG(4),
            ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"),
            ARG(5), ARG(6),
            ARG(MapNames::getDpctNamespace() + "library_data_t::real_float"),
            ARG(7),
            ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"),
            ARG(8), ARG(9),
            ARG(MapNames::getDpctNamespace() + "library_data_t::complex_float"),
            ARG(10), ARG(11), ARG(12), ARG(13), ARG(14))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_gesvd_scratchpad_size_2,
    CALL_FACTORY_ENTRY("cusolverDnZgesvdj_bufferSize",
                       CALL(MapNames::getDpctNamespace() +
                                "lapack::gesvd_scratchpad_size",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
                            ARG(MapNames::getDpctNamespace() +
                                "library_data_t::complex_double"),
                            ARG(6),
                            ARG(MapNames::getDpctNamespace() +
                                "library_data_t::complex_double"),
                            ARG(9),
                            ARG(MapNames::getDpctNamespace() +
                                "library_data_t::complex_double"),
                            ARG(11), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::LapackUtils_gesvd_2,
    CALL_FACTORY_ENTRY("cusolverDnZgesvdj",
                       CALL(MapNames::getDpctNamespace() + "lapack::gesvd",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
                            ARG(MapNames::getDpctNamespace() +
                                "library_data_t::complex_double"),
                            ARG(5), ARG(6),
                            ARG(MapNames::getDpctNamespace() +
                                "library_data_t::real_double"),
                            ARG(7),
                            ARG(MapNames::getDpctNamespace() +
                                "library_data_t::complex_double"),
                            ARG(8), ARG(9),
                            ARG(MapNames::getDpctNamespace() +
                                "library_data_t::complex_double"),
                            ARG(10), ARG(11), ARG(12), ARG(13), ARG(14))))
