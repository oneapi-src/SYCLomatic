//===--------------- RewriterClassMethods.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"

using namespace clang::dpct;

RewriterMap dpct::createClassMethodsRewriterMap() {
  return RewriterMap{
      // cub::ArgIndexInputIterator.normalize
      FEATURE_REQUEST_FACTORY(
          HelperFeatureEnum::device_ext,
          HEADER_INSERT_FACTORY(
              HeaderType::HT_DPCT_DPL_Utils,
              ASSIGN_FACTORY_ENTRY("cub::ArgIndexInputIterator.normalize",
                                   MemberExprBase(),
                                   MEMBER_CALL(MemberExprBase(), false,
                                               LITERAL("create_normalize")))))};
}
