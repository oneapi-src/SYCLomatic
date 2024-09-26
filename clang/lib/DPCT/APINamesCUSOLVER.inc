//===------------------- APINamesCUSOLVER.inc ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

// clang-format off
REMOVE_API_FACTORY_ENTRY("cusolverDnSetAdvOptions")

ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusolverDnSetStream", ARG(0), ARG(1)))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusolverDnGetStream", DEREF(1),
                                        ARG(0)))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnSsygvd",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::sygvd",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                            ARG(12))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnSsygvd_bufferSize", DEREF(10),
    CALL("oneapi::mkl::lapack::sygvd_scratchpad_size<float>", DEREF(0), ARG(1),
         ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnDsygvd",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::sygvd",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                            ARG(12))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnDsygvd_bufferSize", DEREF(10),
    CALL("oneapi::mkl::lapack::sygvd_scratchpad_size<double>", DEREF(0), ARG(1),
         ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnChegvd",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::hegvd",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                            ARG(12))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnChegvd_bufferSize", DEREF(10),
    CALL("oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<float>>",
         DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnZhegvd",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::hegvd",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                            ARG(12))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnZhegvd_bufferSize", DEREF(10),
    CALL("oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<double>>",
         DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("cusolverDnSpotrfBatched",
                                           CALL(MapNames::getLibraryHelperNamespace() +
                                                    "lapack::potrf_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5),
                                                ARG(6))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("cusolverDnDpotrfBatched",
                                           CALL(MapNames::getLibraryHelperNamespace() +
                                                    "lapack::potrf_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5),
                                                ARG(6))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("cusolverDnCpotrfBatched",
                                           CALL(MapNames::getLibraryHelperNamespace() +
                                                    "lapack::potrf_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5),
                                                ARG(6))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("cusolverDnZpotrfBatched",
                                           CALL(MapNames::getLibraryHelperNamespace() +
                                                    "lapack::potrf_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5),
                                                ARG(6))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("cusolverDnSpotrsBatched",
                                           CALL(MapNames::getLibraryHelperNamespace() +
                                                    "lapack::potrs_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5), ARG(6),
                                                ARG(7), ARG(8), ARG(9))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("cusolverDnDpotrsBatched",
                                           CALL(MapNames::getLibraryHelperNamespace() +
                                                    "lapack::potrs_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5), ARG(6),
                                                ARG(7), ARG(8), ARG(9))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("cusolverDnCpotrsBatched",
                                           CALL(MapNames::getLibraryHelperNamespace() +
                                                    "lapack::potrs_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5), ARG(6),
                                                ARG(7), ARG(8), ARG(9))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        CALL_FACTORY_ENTRY("cusolverDnZpotrsBatched",
                                           CALL(MapNames::getLibraryHelperNamespace() +
                                                    "lapack::potrs_batch",
                                                DEREF(0), ARG(1), ARG(2),
                                                ARG(3), ARG(4), ARG(5), ARG(6),
                                                ARG(7), ARG(8), ARG(9))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cusolverDnXgetrf_bufferSize",
        CALL(MapNames::getLibraryHelperNamespace() + "lapack::getrf_scratchpad_size",
             DEREF(0), ARG(2), ARG(3), ARG(4), ARG(6), ARG(8), ARG(9))))

WARNING_FACTORY_ENTRY(
    "cusolverDnXgetrf",
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY("cusolverDnXgetrf",
                           CALL(MapNames::getLibraryHelperNamespace() + "lapack::getrf",
                                DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5),
                                ARG(6), ARG(7), ARG(9), ARG(10), ARG(13)))),
    Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(7),
    ARG(MapNames::getLibraryHelperNamespace() + "lapack::getrf"),
    ARG("cusolverDnXgetrf"))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnXgetrs",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::getrs",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10), ARG(11), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cusolverDnXgeqrf_bufferSize",
        CALL(MapNames::getLibraryHelperNamespace() + "lapack::geqrf_scratchpad_size",
             DEREF(0), ARG(2), ARG(3), ARG(4), ARG(6), ARG(10), ARG(11))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnXgeqrf",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::geqrf",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(10), ARG(11), ARG(14))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnGetrf_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "lapack::getrf_scratchpad_size",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))

WARNING_FACTORY_ENTRY(
    "cusolverDnGetrf",
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        CALL_FACTORY_ENTRY("cusolverDnGetrf",
                           CALL(MapNames::getLibraryHelperNamespace() + "lapack::getrf",
                                DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5),
                                ARG(6), ARG(7), ARG(9), ARG(10), ARG(11)))),
    Diagnostics::DIFFERENT_LU_FACTORIZATION, ARG(7),
    ARG(MapNames::getLibraryHelperNamespace() + "lapack::getrf"), ARG("cusolverDnGetrf"))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnGetrs",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::getrs",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10), ARG(11), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnGeqrf_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "lapack::geqrf_scratchpad_size",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(6), ARG(10))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnGeqrf",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::geqrf",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(10), ARG(11), ARG(12))))

REMOVE_API_FACTORY_ENTRY("cusolverDnCreateParams")
REMOVE_API_FACTORY_ENTRY("cusolverDnDestroyParams")

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cusolverDnXgesvd_bufferSize",
        CALL(MapNames::getLibraryHelperNamespace() + "lapack::gesvd_scratchpad_size",
             DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(8), ARG(11),
             ARG(13), ARG(14), ARG(16), ARG(18), ARG(19))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnXgesvd",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::gesvd",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10), ARG(11), ARG(12),
                            ARG(13), ARG(14), ARG(15), ARG(16), ARG(18),
                            ARG(19), ARG(22))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cusolverDnGesvd_bufferSize",
        CALL(MapNames::getLibraryHelperNamespace() + "lapack::gesvd_scratchpad_size",
             DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(8), ARG(11),
             ARG(13), ARG(14), ARG(16), ARG(18))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnGesvd",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::gesvd",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10), ARG(11), ARG(12),
                            ARG(13), ARG(14), ARG(15), ARG(16), ARG(18),
                            ARG(19), ARG(20))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cusolverDnXpotrf_bufferSize",
        CALL(MapNames::getLibraryHelperNamespace() + "lapack::potrf_scratchpad_size",
             DEREF(0), ARG(2), ARG(3), ARG(4), ARG(6), ARG(8), ARG(9))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnXpotrf",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::potrf",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(8), ARG(9), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnPotrf_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "lapack::potrf_scratchpad_size",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(6), ARG(8))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnPotrf",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::potrf",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(8), ARG(9), ARG(10))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnXpotrs",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::potrs",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10), ARG(11))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnPotrs",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::potrs",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(10), ARG(11))))

REMOVE_API_FACTORY_ENTRY("cusolverDnCreateGesvdjInfo")
REMOVE_API_FACTORY_ENTRY("cusolverDnDestroyGesvdjInfo")

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cusolverDnSgesvdj_bufferSize",
        CALL(MapNames::getLibraryHelperNamespace() + "lapack::gesvd_scratchpad_size",
             DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
             ARG(6),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
             ARG(9),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
             ARG(11), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cusolverDnSgesvdj",
        CALL(MapNames::getLibraryHelperNamespace() + "lapack::gesvd", DEREF(0), ARG(1),
             ARG(2), ARG(3), ARG(4),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
             ARG(5), ARG(6),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
             ARG(7),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
             ARG(8), ARG(9),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
             ARG(10), ARG(11), ARG(12), ARG(13), ARG(14))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cusolverDnDgesvdj_bufferSize",
        CALL(MapNames::getLibraryHelperNamespace() + "lapack::gesvd_scratchpad_size",
             DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"),
             ARG(6),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"),
             ARG(9),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"),
             ARG(11), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cusolverDnDgesvdj",
        CALL(MapNames::getLibraryHelperNamespace() + "lapack::gesvd", DEREF(0), ARG(1),
             ARG(2), ARG(3), ARG(4),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"),
             ARG(5), ARG(6),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"),
             ARG(7),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"),
             ARG(8), ARG(9),
             ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_double"),
             ARG(10), ARG(11), ARG(12), ARG(13), ARG(14))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cusolverDnCgesvdj_bufferSize",
        CALL(
            MapNames::getLibraryHelperNamespace() + "lapack::gesvd_scratchpad_size",
            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
            ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float"),
            ARG(6),
            ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float"),
            ARG(9),
            ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float"),
            ARG(11), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cusolverDnCgesvdj",
        CALL(
            MapNames::getLibraryHelperNamespace() + "lapack::gesvd", DEREF(0), ARG(1),
            ARG(2), ARG(3), ARG(4),
            ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float"),
            ARG(5), ARG(6),
            ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::real_float"),
            ARG(7),
            ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float"),
            ARG(8), ARG(9),
            ARG(MapNames::getLibraryHelperNamespace() + "library_data_t::complex_float"),
            ARG(10), ARG(11), ARG(12), ARG(13), ARG(14))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnZgesvdj_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "lapack::gesvd_scratchpad_size",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
                            ARG(MapNames::getLibraryHelperNamespace() +
                                "library_data_t::complex_double"),
                            ARG(6),
                            ARG(MapNames::getLibraryHelperNamespace() +
                                "library_data_t::complex_double"),
                            ARG(9),
                            ARG(MapNames::getLibraryHelperNamespace() +
                                "library_data_t::complex_double"),
                            ARG(11), ARG(12))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnZgesvdj",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::gesvd",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
                            ARG(MapNames::getLibraryHelperNamespace() +
                                "library_data_t::complex_double"),
                            ARG(5), ARG(6),
                            ARG(MapNames::getLibraryHelperNamespace() +
                                "library_data_t::real_double"),
                            ARG(7),
                            ARG(MapNames::getLibraryHelperNamespace() +
                                "library_data_t::complex_double"),
                            ARG(8), ARG(9),
                            ARG(MapNames::getLibraryHelperNamespace() +
                                "library_data_t::complex_double"),
                            ARG(10), ARG(11), ARG(12), ARG(13), ARG(14))))

ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnSgeqrf_bufferSize", DEREF(5),
    CALL("oneapi::mkl::lapack::geqrf_scratchpad_size<float>", DEREF(0), ARG(1),
         ARG(2), ARG(4))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnDgeqrf_bufferSize", DEREF(5),
    CALL("oneapi::mkl::lapack::geqrf_scratchpad_size<double>", DEREF(0), ARG(1),
         ARG(2), ARG(4))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnCgeqrf_bufferSize", DEREF(5),
    CALL("oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<float>>",
         DEREF(0), ARG(1), ARG(2), ARG(4))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
    "cusolverDnZgeqrf_bufferSize", DEREF(5),
    CALL("oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<double>>",
         DEREF(0), ARG(1), ARG(2), ARG(4))))

// cusolverDn<T>syevdx_bufferSize, cusolverDn<T>heevdx_bufferSize
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnSsyevdx_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syheevx_scratchpad_size<float, float>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6),
                        ARG(7), ARG(8), ARG(9), ARG(10), ARG(13))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnDsyevdx_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syheevx_scratchpad_size<double, double>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6),
                        ARG(7), ARG(8), ARG(9), ARG(10), ARG(13))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnCheevdx_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syheevx_scratchpad_size<" +
                            MapNames::getClNamespace() + "float2, float>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6),
                        ARG(7), ARG(8), ARG(9), ARG(10), ARG(13))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnZheevdx_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syheevx_scratchpad_size<" +
                            MapNames::getClNamespace() + "double2, double>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6),
                        ARG(7), ARG(8), ARG(9), ARG(10), ARG(13))))

// cusolverDn<T>syevdx, cusolverDn<T>heevdx
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnSsyevdx",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syheevx<float, float>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                        ARG(12), ARG(13), ARG(14), ARG(15))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnDsyevdx",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syheevx<double, double>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                        ARG(12), ARG(13), ARG(14), ARG(15))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnCheevdx",
                   CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevx<" +
                            MapNames::getClNamespace() + "float2, float>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                        ARG(12), ARG(13), ARG(14), ARG(15))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnZheevdx",
                   CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevx<" +
                            MapNames::getClNamespace() + "double2, double>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                        ARG(12), ARG(13), ARG(14), ARG(15))))

// cusolverDnSyevdx_bufferSize
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnSyevdx_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syheevx_scratchpad_size",
                        DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                        ARG(8), ARG(9), ARG(10), ARG(11), ARG(12), ARG(14),
                        ARG(17))))
// cusolverDnSyevdx
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnSyevdx",
                   CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevx",
                        DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                        ARG(7), ARG(8), ARG(9), ARG(10), ARG(11), ARG(12),
                        ARG(13), ARG(14), ARG(15), ARG(17), ARG(18), ARG(19))))
// cusolverDnXsyevdx_bufferSize
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnXsyevdx_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syheevx_scratchpad_size",
                        DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                        ARG(8), ARG(9), ARG(10), ARG(11), ARG(12), ARG(14),
                        ARG(17), ARG(18))))
// cusolverDnXsyevdx
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnXsyevdx",
                   CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevx",
                        DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                        ARG(7), ARG(8), ARG(9), ARG(10), ARG(11), ARG(12),
                        ARG(13), ARG(14), ARG(15), ARG(17), ARG(18), ARG(21))))

// cusolverDn<T>sygvdx_bufferSize, cusolverDn<T>hegvdx_bufferSize
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnSsygvdx_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syhegvx_scratchpad_size<float, float>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(7), ARG(9), ARG(10), ARG(11), ARG(12), ARG(13),
                        ARG(16))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnDsygvdx_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syhegvx_scratchpad_size<double, double>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(7), ARG(9), ARG(10), ARG(11), ARG(12), ARG(13),
                        ARG(16))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnChegvdx_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syhegvx_scratchpad_size<" +
                            MapNames::getClNamespace() + "float2, float>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(7), ARG(9), ARG(10), ARG(11), ARG(12), ARG(13),
                        ARG(16))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnZhegvdx_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syhegvx_scratchpad_size<" +
                            MapNames::getClNamespace() + "double2, double>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(7), ARG(9), ARG(10), ARG(11), ARG(12), ARG(13),
                        ARG(16))))

// cusolverDn<T>sygvdx, cusolverDn<T>hegvdx
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnSsygvdx",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syhegvx<float, float>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                        ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17),
                        ARG(18))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnDsygvdx",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syhegvx<double, double>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                        ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17),
                        ARG(18))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnChegvdx",
                   CALL(MapNames::getLibraryHelperNamespace() + "lapack::syhegvx<" +
                            MapNames::getClNamespace() + "float2, float>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                        ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17),
                        ARG(18))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnZhegvdx",
                   CALL(MapNames::getLibraryHelperNamespace() + "lapack::syhegvx<" +
                            MapNames::getClNamespace() + "double2, double>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                        ARG(12), ARG(13), ARG(14), ARG(15), ARG(16), ARG(17),
                        ARG(18))))

// cusolverDn<T>sygvj_bufferSize, cusolverDn<T>hegvj_bufferSize
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnSsygvj_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syhegvd_scratchpad_size<float>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6),
                        ARG(8), ARG(10))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnDsygvj_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syhegvd_scratchpad_size<float>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6),
                        ARG(8), ARG(10))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnChegvj_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syhegvd_scratchpad_size<" +
                            MapNames::getClNamespace() + "float2>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6),
                        ARG(8), ARG(10))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnZhegvj_bufferSize",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syhegvd_scratchpad_size<" +
                            MapNames::getClNamespace() + "double2>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6),
                        ARG(8), ARG(10))))

// cusolverDn<T>sygvj, cusolverDn<T>hegvj
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnSsygvj",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syhegvd<float, float>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                        ARG(12))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnDsygvj",
                   CALL(MapNames::getLibraryHelperNamespace() +
                            "lapack::syhegvd<double, double>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                        ARG(12))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnChegvj",
                   CALL(MapNames::getLibraryHelperNamespace() + "lapack::syhegvd<" +
                            MapNames::getClNamespace() + "float2, float>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                        ARG(12))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
CALL_FACTORY_ENTRY("cusolverDnZhegvj",
                   CALL(MapNames::getLibraryHelperNamespace() + "lapack::syhegvd<" +
                            MapNames::getClNamespace() + "double2, double>",
                        DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                        ARG(6), ARG(7), ARG(8), ARG(9), ARG(10), ARG(11),
                        ARG(12))))

REMOVE_API_FACTORY_ENTRY("cusolverDnCreateSyevjInfo")
REMOVE_API_FACTORY_ENTRY("cusolverDnDestroySyevjInfo")

// cusolverDn<T>syevj_bufferSize, cusolverDn<T>heevj_bufferSize
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnSsyevj_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "lapack::syheev_scratchpad_size<float>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(5), ARG(7))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnDsyevj_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "lapack::syheev_scratchpad_size<double>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(5), ARG(7))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnCheevj_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "lapack::syheev_scratchpad_size<" +
                                MapNames::getClNamespace() + "float2>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(5), ARG(7))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnZheevj_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "lapack::syheev_scratchpad_size<" +
                                MapNames::getClNamespace() + "double2>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(5), ARG(7))))

// cusolverDn<T>syevj, cusolverDn<T>heevj
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnSsyevj",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "lapack::syheev<float, float>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnDsyevj",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "lapack::syheev<double, double>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnCheevj",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheev<" +
                                MapNames::getClNamespace() + "float2, float>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnZheevj",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheev<" +
                                MapNames::getClNamespace() + "double2, double>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(9))))

// cusolverDnXsyevd_bufferSize
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnXsyevd_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "lapack::syheevd_scratchpad_size",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(7),
                            ARG(8), ARG(11), ARG(12))))

// cusolverDnXsyevd
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnXsyevd",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevd",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(11), ARG(12), ARG(15))))

// cusolverDnSyevd_bufferSize
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnSyevd_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() +
                                "lapack::syheevd_scratchpad_size",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(7),
                            ARG(8), ARG(11))))

// cusolverDnSyevd
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnSyevd",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevd",
                            DEREF(0), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9), ARG(11), ARG(12), ARG(13))))

// cusolverDnXtrtri_bufferSize
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY(
        "cusolverDnXtrtri_bufferSize",
        CALL(MapNames::getLibraryHelperNamespace() + "lapack::trtri_scratchpad_size",
             DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(6), ARG(7), ARG(8))))

// cusolverDnXtrtri
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnXtrtri",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::trtri",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                            ARG(6), ARG(7), ARG(8), ARG(11))))

// cusolverDn<T>syevd_bufferSize/cusolverDn<T>heevd_bufferSize
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnSsyevd_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevd_scratchpad_size<float>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(5), ARG(7))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnDsyevd_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevd_scratchpad_size<double>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(5), ARG(7))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnCheevd_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevd_scratchpad_size<std::complex<float>>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(5), ARG(7))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnZheevd_bufferSize",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevd_scratchpad_size<std::complex<double>>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(5), ARG(7))))

// cusolverDn<T>syevd/cusolverDn<T>heevd
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnSsyevd",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevd<float, float>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnDsyevd",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevd<double, double>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnCheevd",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevd<" +
                                MapNames::getClNamespace() + "float2, float>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    CALL_FACTORY_ENTRY("cusolverDnZheevd",
                       CALL(MapNames::getLibraryHelperNamespace() + "lapack::syheevd<" +
                                MapNames::getClNamespace() + "double2, double>",
                            DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                            ARG(7), ARG(8), ARG(9))))

// clang-format on
