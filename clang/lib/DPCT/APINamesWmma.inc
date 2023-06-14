//===--------------- APINamesWmma.inc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

CONDITIONAL_FACTORY_ENTRY(
        checkEnableJointMatrix(),
        CALL_FACTORY_ENTRY("nvcuda::wmma::fill_fragment",
                           CALL(MapNames::getClNamespace() +
                                "ext::oneapi::experimental::matrix::joint_matrix_fill",
                                SUBGROUP, ARG(0), ARG(1))),
        ENTRY_UNSUPPORTED("nvcuda::wmma::fill_fragment", Diagnostics::API_NOT_MIGRATED))
