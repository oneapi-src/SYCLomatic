//===--------------- ThrustAPIMigration.cpp
//---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThrustAPIMigration.h"

#include "ASTTraversal.h"
#include "ExprAnalysis.h"

namespace clang {
namespace dpct {

using namespace clang;
using namespace clang::dpct;
using namespace clang::ast_matchers;


void ThrustRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto LIBCUAPIHasNames = [&]() {
      return hasAnyName("cuda::atomic_thread_fence",
                        "cuda::std::atomic_thread_fence");
  };
}

void ThrustRule::runRule(const ast_matchers::MatchFinder::MatchResult &Result) {

}
} // namespace dpct
} // namespace clang