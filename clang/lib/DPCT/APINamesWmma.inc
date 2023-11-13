//===--------------- APINamesWmma.inc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

CONDITIONAL_FACTORY_ENTRY(
    checkEnableJointMatrix(),
    CALL_FACTORY_ENTRY(
        "nvcuda::wmma::fill_fragment",
        CALL(MapNames::getClNamespace() +
                 "ext::oneapi::experimental::matrix::joint_matrix_fill",
             SUBGROUP, ARG(0), ARG(1))),
    ENTRY_UNSUPPORTED("nvcuda::wmma::fill_fragment",
                      Diagnostics::API_NOT_MIGRATED))

CONDITIONAL_FACTORY_ENTRY(
    checkEnableJointMatrix(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgCount(3),
        CALL_FACTORY_ENTRY(
            "nvcuda::wmma::load_matrix_sync",
            CALL(MapNames::getClNamespace() +
                     "ext::oneapi::experimental::matrix::joint_matrix_load",
                 SUBGROUP, ARG(0),
                 CALL(TEMPLATED_CALLEE_WITH_ARGS(
                          MapNames::getClNamespace() + "address_space_cast",
                          LITERAL(MapNames::getClNamespace() +
                                  "access::address_space::generic_space"),
                          LITERAL(MapNames::getClNamespace() +
                                  "access::decorated::no"),
                          getDerefedType(1)),
                      ARG(1)),
                 ARG(2))),
        CALL_FACTORY_ENTRY(
            "nvcuda::wmma::load_matrix_sync",
            CALL(MapNames::getClNamespace() +
                     "ext::oneapi::experimental::matrix::joint_matrix_load",
                 SUBGROUP, ARG(0),
                 CALL(TEMPLATED_CALLEE_WITH_ARGS(
                          MapNames::getClNamespace() + "address_space_cast",
                          LITERAL(MapNames::getClNamespace() +
                                  "access::address_space::generic_space"),
                          LITERAL(MapNames::getClNamespace() +
                                  "access::decorated::no"),
                          getDerefedType(1)),
                      ARG(1)),
                 ARG(2), ARG(3)))),
    ENTRY_UNSUPPORTED("nvcuda::wmma::load_matrix_sync",
                      Diagnostics::API_NOT_MIGRATED))

CONDITIONAL_FACTORY_ENTRY(
    checkEnableJointMatrix(),
    CALL_FACTORY_ENTRY(
        "nvcuda::wmma::mma_sync",
        CALL(MapNames::getClNamespace() +
                 "ext::oneapi::experimental::matrix::joint_matrix_mad",
             SUBGROUP, ARG(0), ARG(1), ARG(2), ARG(3))),
    ENTRY_UNSUPPORTED("nvcuda::wmma::mma_sync", Diagnostics::API_NOT_MIGRATED))

CONDITIONAL_FACTORY_ENTRY(
    checkEnableJointMatrix(),
    CALL_FACTORY_ENTRY(
        "nvcuda::wmma::store_matrix_sync",
        CALL(MapNames::getClNamespace() +
                 "ext::oneapi::experimental::matrix::joint_matrix_store",
             SUBGROUP, ARG(1),
             CALL(TEMPLATED_CALLEE_WITH_ARGS(
                      MapNames::getClNamespace() + "address_space_cast",
                      LITERAL(MapNames::getClNamespace() +
                              "access::address_space::generic_space"),
                      LITERAL(MapNames::getClNamespace() +
                              "access::decorated::no"),
                      getDerefedType(0)),
                  ARG(0)),
             ARG(2), ARG(3))),
    ENTRY_UNSUPPORTED("nvcuda::wmma::store_matrix_sync",
                      Diagnostics::API_NOT_MIGRATED))
