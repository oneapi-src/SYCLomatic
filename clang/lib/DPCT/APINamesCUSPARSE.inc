//===------------------- APINamesCUSPARSE.inc ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_sparse_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseSetMatType", ARG(0),
                                                  true, "set", ARG(1)))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_sparse_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseSetMatIndexBase",
                                                  ARG(0), true, "set", ARG(1)))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_sparse_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseSetMatDiagType",
                                                  ARG(0), true, "set", ARG(1)))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_sparse_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseSetMatFillMode",
                                                  ARG(0), true, "set", ARG(1)))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::SparseUtils_sparse_matrix_info,
    MEMBER_CALL_FACTORY_ENTRY("cusparseGetMatType", ARG(0), true,
                              "get<" + MapNames::getDpctNamespace() +
                                  "sparse::sparse_matrix_info::matrix_type>"))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::SparseUtils_sparse_matrix_info,
    MEMBER_CALL_FACTORY_ENTRY("cusparseGetMatIndexBase", ARG(0), true,
                              "get<oneapi::mkl::index_base>"))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_sparse_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseGetMatDiagType",
                                                  ARG(0), true,
                                                  "get<oneapi::mkl::diag>"))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_sparse_matrix_info,
                        MEMBER_CALL_FACTORY_ENTRY("cusparseGetMatFillMode",
                                                  ARG(0), true,
                                                  "get<oneapi::mkl::uplo>"))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::SparseUtils_sparse_matrix_info,
    ASSIGN_FACTORY_ENTRY("cusparseCreateMatDescr", DEREF(0),
                         CALL("std::make_shared<" +
                              MapNames::getDpctNamespace() +
                              "sparse::sparse_matrix_info>")))
REMOVE_API_FACTORY_ENTRY("cusparseDestroyMatDescr")
