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

ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusparseCreate", DEREF(0),
                                        QUEUEPTRSTR))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusparseDestroy", ARG(0),
                                        ARG("nullptr")))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusparseSetStream", ARG(0), ARG(1)))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusparseGetStream", DEREF(1), ARG(0)))
REMOVE_API_FACTORY_ENTRY("cusparseGetPointerMode")
REMOVE_API_FACTORY_ENTRY("cusparseSetPointerMode")

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_optimize_info,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cusparseCreateSolveAnalysisInfo", DEREF(0),
                            CALL("std::make_shared<" +
                                 MapNames::getDpctNamespace() +
                                 "sparse::optimize_info>"))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_optimize_info,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cusparseDestroySolveAnalysisInfo", ARG(0), false,
                            "reset")))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::SparseUtils_optimize_csrsv,
    ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
        "cusparseScsrsv_analysis",
        CALL(MapNames::getDpctNamespace() + "sparse::optimize_csrsv", DEREF(0),
             ARG(1), ARG(2), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8)))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::SparseUtils_optimize_csrsv,
    ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
        "cusparseDcsrsv_analysis",
        CALL(MapNames::getDpctNamespace() + "sparse::optimize_csrsv", DEREF(0),
             ARG(1), ARG(2), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8)))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::SparseUtils_optimize_csrsv,
    ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
        "cusparseCcsrsv_analysis",
        CALL(MapNames::getDpctNamespace() + "sparse::optimize_csrsv", DEREF(0),
             ARG(1), ARG(2), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8)))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::SparseUtils_optimize_csrsv,
    ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
        "cusparseZcsrsv_analysis",
        CALL(MapNames::getDpctNamespace() + "sparse::optimize_csrsv", DEREF(0),
             ARG(1), ARG(2), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8)))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_csrmv,
                        ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
                            "cusparseScsrmv",
                            CALL(MapNames::getDpctNamespace() + "sparse::csrmv",
                                 DEREF(0), ARG(1), ARG(2), ARG(3), ARG(5),
                                 ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
                                 ARG(11), ARG(12)))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_csrmv,
                        ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
                            "cusparseDcsrmv",
                            CALL(MapNames::getDpctNamespace() + "sparse::csrmv",
                                 DEREF(0), ARG(1), ARG(2), ARG(3), ARG(5),
                                 ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
                                 ARG(11), ARG(12)))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_csrmv,
                        ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
                            "cusparseCcsrmv",
                            CALL(MapNames::getDpctNamespace() + "sparse::csrmv",
                                 DEREF(0), ARG(1), ARG(2), ARG(3), ARG(5),
                                 ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
                                 ARG(11), ARG(12)))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_csrmv,
                        ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
                            "cusparseZcsrmv",
                            CALL(MapNames::getDpctNamespace() + "sparse::csrmv",
                                 DEREF(0), ARG(1), ARG(2), ARG(3), ARG(5),
                                 ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
                                 ARG(11), ARG(12)))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_csrmm,
                        ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
                            "cusparseScsrmm",
                            CALL(MapNames::getDpctNamespace() + "sparse::csrmm",
                                 DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
                                 ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
                                 ARG(11), ARG(12), ARG(13), ARG(14), ARG(15)))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_csrmm,
                        ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
                            "cusparseDcsrmm",
                            CALL(MapNames::getDpctNamespace() + "sparse::csrmm",
                                 DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
                                 ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
                                 ARG(11), ARG(12), ARG(13), ARG(14), ARG(15)))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_csrmm,
                        ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
                            "cusparseCcsrmm",
                            CALL(MapNames::getDpctNamespace() + "sparse::csrmm",
                                 DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
                                 ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
                                 ARG(11), ARG(12), ARG(13), ARG(14), ARG(15)))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::SparseUtils_csrmm,
                        ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
                            "cusparseZcsrmm",
                            CALL(MapNames::getDpctNamespace() + "sparse::csrmm",
                                 DEREF(0), ARG(1), ARG(2), ARG(3), ARG(4),
                                 ARG(6), ARG(7), ARG(8), ARG(9), ARG(10),
                                 ARG(11), ARG(12), ARG(13), ARG(14), ARG(15)))))

ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cusparseCreateCsr",
    CALL(MapNames::getDpctNamespace() + "sparse::sparse_matrix_desc::create_csr",
         ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8),
         ARG(9), ARG(10))))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cusparseDestroySpMat",
    CALL(MapNames::getDpctNamespace() + "sparse::sparse_matrix_desc::destroy",
         ARG(0))))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
    "cusparseCsrGet", ARG(0), true, "get_desc",
    ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8), ARG(9),
    ARG(10)))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
    "cusparseSpMatGetFormat", ARG(0), true,
    "get_format", ARG(1)))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
    "cusparseSpMatGetIndexBase", ARG(0), true,
    "get_base", ARG(1)))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
    "cusparseSpMatGetValues", ARG(0), true,
    "get_value", ARG(1)))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
    "cusparseSpMatSetValues", ARG(0), true,
    "set_value", ARG(1)))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusparseCreateDnMat", DEREF(0),
                                        CALL("std::make_shared<" +
                                                 MapNames::getDpctNamespace() +
                                                 "sparse::dense_matrix_desc>",
                                             ARG(1), ARG(2), ARG(3), ARG(4),
                                             ARG(5), ARG(6))))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cusparseDestroyDnMat", ARG(0),
                                             false, "reset"))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
    "cusparseDnMatGet", ARG(0), true, "get_desc",
    ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusparseDnMatGetValues", DEREF(1),
                                        MEMBER_EXPR(DEREF(0), false,
                                                    LITERAL("_value"))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusparseDnMatSetValues",
                                        MEMBER_EXPR(DEREF(0), false,
                                                    LITERAL("_value")),
                                        ARG(1)))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusparseCreateDnVec", DEREF(0),
                                        CALL("std::make_shared<" +
                                                 MapNames::getDpctNamespace() +
                                                 "sparse::dense_vector_desc>",
                                             ARG(1), ARG(2), ARG(3))))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cusparseDestroyDnVec", ARG(0),
                                             false, "reset"))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cusparseDnVecGet", ARG(0), true,
                                             "get_desc",
                                             ARG(1), ARG(2), ARG(3)))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusparseDnVecGetValues", DEREF(1),
                                        MEMBER_EXPR(DEREF(0), false,
                                                    LITERAL("_value"))))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusparseDnVecSetValues",
                                        MEMBER_EXPR(DEREF(0), false,
                                                    LITERAL("_value")),
                                        ARG(1)))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cusparseCsrSetPointers", ARG(0),
                                             true, "set_pointers", ARG(1),
                                             ARG(2), ARG(3)))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cusparseSpMatGetSize", ARG(0),
                                             true, "get_size", ARG(1), ARG(2),
                                             ARG(3)))
ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
    "cusparseGetProperty",
    CALL(MapNames::getDpctNamespace() + "mkl_get_version", ARG(0), ARG(1))))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cusparseSpMatGetAttribute",
                                             ARG(0), true, "get_attribute",
                                             ARG(1), ARG(2), ARG(3)))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cusparseSpMatSetAttribute",
                                             ARG(0), true, "set_attribute",
                                             ARG(1), ARG(2), ARG(3)))

ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusparseCreateConstDnVec", DEREF(0),
                                        CALL("std::make_shared<" +
                                                 MapNames::getDpctNamespace() +
                                                 "sparse::dense_vector_desc>",
                                             ARG(1), ARG(2), ARG(3))))
ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
    "cusparseConstDnVecGet", ARG(0), true,
    "get_desc", ARG(1), ARG(2), ARG(3)))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cusparseConstDnVecGetValues", DEREF(1),
                                        MEMBER_EXPR(DEREF(0), false,
                                                    LITERAL("_value"))))
