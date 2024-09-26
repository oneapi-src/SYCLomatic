//===--------------- RewriterDeviceSpmv.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"
#include "CallExprRewriterCommon.h"
#include "MapNames.h"

using namespace clang::dpct;

RewriterMap dpct::createDeviceSpmvRewriterMap() {
  return RewriterMap{
      // cub::DeviceSpmv::CsrMV
      CONDITIONAL_FACTORY_ENTRY(
          CheckCubRedundantFunctionCall(),
          REMOVE_API_FACTORY_ENTRY("cub::DeviceSpmv::CsrMV"),
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPCT_SPBLAS_Utils,
              REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                  makeCheckAnd(CheckArgCount(11, std::greater_equal<>(),
                                             /* IncludeDefaultArg */ false),
                               makeCheckNot(CheckArgIsDefaultCudaStream(10))),
                  CALL_FACTORY_ENTRY(
                      "cub::DeviceSpmv::CsrMV",
                      CALL(MapNames::getLibraryHelperNamespace() + "sparse::csrmv",
                           STREAM(10), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                           ARG(7), ARG(8))),
                  CALL_FACTORY_ENTRY(
                      "cub::DeviceSpmv::CsrMV",
                      CALL(MapNames::getLibraryHelperNamespace() + "sparse::csrmv",
                           QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
                           ARG(7), ARG(8)))))))};
}
