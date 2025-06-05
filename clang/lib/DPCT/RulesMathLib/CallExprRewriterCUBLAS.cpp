//===--------------- CallExprRewriterCUBLAS.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUBLAS.h"

namespace clang {
namespace dpct {

void CallExprRewriterFactoryBase::initRewriterMapCUBLAS() {
  RewriterMap->merge(createCUBLASLevel1RewriterMap());
  RewriterMap->merge(createCUBLASLevel2RewriterMap());
  RewriterMap->merge(createCUBLASLevel3RewriterMap());
  RewriterMap->merge(createCUBLASHelperRewriterMap());
  RewriterMap->merge(createCUBLASExtRewriterMap());
  RewriterMap->merge(createCUBLASLtRewriterMap());
}

} // namespace dpct
} // namespace clang
