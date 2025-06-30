//===----------------- CallExprRewriterCUBLASLt.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUBLAS.h"

using namespace clang::dpct;

RewriterMap dpct::createCUBLASLtRewriterMap() {
  return RewriterMap{
      // cublasLtCreate
      ASSIGNABLE_FACTORY(
          ASSIGN_FACTORY_ENTRY("cublasLtCreate", DEREF(0),
                               NEW(MapNames::getLibraryHelperNamespace() +
                                   "blas_gemm::experimental::descriptor")))
      // cublasLtDestroy
      ASSIGNABLE_FACTORY(DELETE_FACTORY_ENTRY("cublasLtDestroy", ARG(0)))
      // cublasLtMatmulDescCreate
      ASSIGNABLE_FACTORY(
          ASSIGN_FACTORY_ENTRY("cublasLtMatmulDescCreate", DEREF(0),
                               NEW(MapNames::getLibraryHelperNamespace() +
                                       "blas_gemm::experimental::matmul_desc_t",
                                   ARG(1), ARG(2))))
      // cublasLtMatmulDescDestroy
      ASSIGNABLE_FACTORY(
          DELETE_FACTORY_ENTRY("cublasLtMatmulDescDestroy", ARG(0)))
      // cublasLtMatmulDescSetAttribute
      ASSIGNABLE_FACTORY(
          MEMBER_CALL_FACTORY_ENTRY("cublasLtMatmulDescSetAttribute", ARG(0),
                                    true, "set_attribute", ARG(1), ARG(2)))
      // cublasLtMatmulDescGetAttribute
      ASSIGNABLE_FACTORY(
          MEMBER_CALL_FACTORY_ENTRY("cublasLtMatmulDescGetAttribute", ARG(0),
                                    true, "get_attribute", ARG(1), ARG(2)))
      // cublasLtMatrixLayoutCreate
      ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
          "cublasLtMatrixLayoutCreate", DEREF(0),
          NEW(MapNames::getLibraryHelperNamespace() +
                  "blas_gemm::experimental::matrix_layout_t",
              ARG(1), ARG(2), ARG(3), ARG(4))))
      // cublasLtMatrixLayoutDestroy
      ASSIGNABLE_FACTORY(
          DELETE_FACTORY_ENTRY("cublasLtMatrixLayoutDestroy", ARG(0)))
      // cublasLtMatrixLayoutSetAttribute
      ASSIGNABLE_FACTORY(
          MEMBER_CALL_FACTORY_ENTRY("cublasLtMatrixLayoutSetAttribute", ARG(0),
                                    true, "set_attribute", ARG(1), ARG(2)))
      // cublasLtMatrixLayoutGetAttribute
      ASSIGNABLE_FACTORY(
          MEMBER_CALL_FACTORY_ENTRY("cublasLtMatrixLayoutGetAttribute", ARG(0),
                                    true, "get_attribute", ARG(1), ARG(2)))
      // cublasLtMatmul
      ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY(
          "cublasLtMatmul",
          CALL(MapNames::getLibraryHelperNamespace() +
                   "blas_gemm::experimental::matmul",
               ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7),
               ARG(8), ARG(9), ARG(10), ARG(11), ARG(15))))
      // cublasLtMatmulPreferenceCreate
      REMOVE_API_FACTORY_ENTRY("cublasLtMatmulPreferenceCreate")
      // cublasLtMatmulPreferenceDestroy
      REMOVE_API_FACTORY_ENTRY("cublasLtMatmulPreferenceDestroy")
      // cublasLtMatmulPreferenceSetAttribute
      REMOVE_API_FACTORY_ENTRY("cublasLtMatmulPreferenceSetAttribute")
      // cublasLtMatmulPreferenceGetAttribute
      REMOVE_API_FACTORY_ENTRY("cublasLtMatmulPreferenceGetAttribute")
      // cublasLtMatmulAlgoGetHeuristic
      ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cublasLtMatmulAlgoGetHeuristic",
                                              DEREF(9), ARG("1")))
      // cublasLtMatrixTransformDescCreate
      ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
          "cublasLtMatrixTransformDescCreate", DEREF(0),
          NEW(MapNames::getLibraryHelperNamespace() +
                  "blas_gemm::experimental::transform_desc_t",
              ARG(1))))
      // cublasLtMatrixTransformDescDestroy
      ASSIGNABLE_FACTORY(
          DELETE_FACTORY_ENTRY("cublasLtMatrixTransformDescDestroy", ARG(0)))
      // cublasLtMatrixTransformDescSetAttribute
      ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
          "cublasLtMatrixTransformDescSetAttribute", ARG(0), true,
          "set_attribute", ARG(1), ARG(2)))
      // cublasLtMatrixTransformDescGetAttribute
      ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
          "cublasLtMatrixTransformDescGetAttribute", ARG(0), true,
          "get_attribute", ARG(1), ARG(2)))
      // cublasLtMatrixTransform
      ASSIGNABLE_FACTORY(
          CALL_FACTORY_ENTRY("cublasLtMatrixTransform",
                             CALL(MapNames::getLibraryHelperNamespace() +
                                      "blas_gemm::experimental::"
                                      "matrix_transform",
                                  ARG(1), ARG(2), ARG(3), ARG(4), ARG(5),
                                  ARG(6), ARG(7), ARG(8), ARG(9), ARG(10))))
      // cublasLtGetVersion
      WARNING_FACTORY_ENTRY(
          "cublasLtGetVersion",
          CALL_FACTORY_ENTRY("cublasLtGetVersion",
                             CALL(MapNames::getLibraryHelperNamespace() +
                                  "dnnl::get_version")),
          Diagnostics::TYPE_MISMATCH)};
}
