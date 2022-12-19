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
