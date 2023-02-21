//===------------------- APINamesCUSPARSE.inc ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseSetMatType", ARG(0),
                                                  true, "set_matrix_type",
                                                  ARG(1)))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseSetMatIndexBase",
                                                  ARG(0), true,
                                                  "set_index_base", ARG(1)))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseSetMatDiagType",
                                                  ARG(0), true, "set_diag",
                                                  ARG(1)))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseSetMatFillMode",
                                                  ARG(0), true, "set_uplo",
                                                  ARG(1)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseGetMatType", ARG(0),
                                                  true, "get_matrix_type"))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseGetMatIndexBase",
                                                  ARG(0), true,
                                                  "get_index_base"))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseGetMatDiagType",
                                                  ARG(0), true, "get_diag"))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseGetMatFillMode",
                                                  ARG(0), true, "get_uplo"))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::SparseUtils_matrix_info,
    ASSIGN_FACTORY_ENTRY("cusparseCreateMatDescr", DEREF(0),
                         CALL("std::make_shared<" +
                              MapNames::getDpctNamespace() +
                              "sparse::matrix_info>")))
REMOVE_API_FACTORY_ENTRY("cusparseDestroyMatDescr")
