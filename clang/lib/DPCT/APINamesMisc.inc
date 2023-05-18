//===--------------- APINamesMisc.inc -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

CONDITIONAL_FACTORY_ENTRY(
    UseOccupancyCalculation,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Util_calculate_max_active_wg_per_xecore,
        CALL_FACTORY_ENTRY(
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
            CALL(MapNames::getDpctNamespace() +
                     "experimental::calculate_max_active_wg_per_xecore",
                 ARG(0), ARG(2),
                 BO(BinaryOperatorKind::BO_Add, ARG(3),
                    ARG("dpct_placeholder /* static shared local memory "
                        "size */"))))),
    ENTRY_UNSUPPORTED("cudaOccupancyMaxActiveBlocksPerMultiprocessor",
                      Diagnostics::API_NOT_MIGRATED))
