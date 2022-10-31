//===------------------- APINamesCUSOLVER.inc ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusolverDnSetStream", ARG(0), ARG(1)))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusolverDnGetStream", DEREF(1), ARG(0)))

//CONDITIONAL_FACTORY_ENTRY(
//    checkIsUSM(),
//    MULTI_STMTS_FACTORY_ENTRY(
//        "cusolverDnSpotrfBatched", 
//        STR("{"),
//        getbuffer(), getbuffer(), getbuffer(),
//        ASSIGNABLE_FACTORY("cusolverDnSpotrfBatched",
//                           CALL("oneapi::mkl::lapack::potrf_batched", DEREF(0), ARG(1),
//                                ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))),
//        STR("}")),
//    UNSUPPORT_FACTORY_ENTRY("cusolverDnSpotrfBatched",
//                            Diagnostics::API_NOT_MIGRATED,
//                            ARG("cusolverDnSpotrfBatched")))
