//===--------------- ASTmatcherCommon.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_AST_MATCHER_COMMON_H
#define DPCT_AST_MATCHER_COMMON_H
#include "clang/ASTMatchers/ASTMatchersInternal.h"

using namespace clang::ast_matchers;
namespace clang {
namespace dpct {

inline auto parentStmtCub = []() {
  return anyOf(hasParent(compoundStmt()), hasParent(forStmt()),
               hasParent(whileStmt()), hasParent(doStmt()),
               hasParent(ifStmt()));
};

inline auto parentStmt = []() {
    return anyOf(
        hasParent(compoundStmt()), hasParent(forStmt()), hasParent(whileStmt()),
        hasParent(doStmt()), hasParent(ifStmt()),
        hasParent(exprWithCleanups(anyOf(
            hasParent(compoundStmt()), hasParent(forStmt()),
            hasParent(whileStmt()), hasParent(doStmt()), hasParent(ifStmt())))));
};


} //namespace dpct
} //namespace clang

#endif //!DPCT_AST_MATCHER_COMMON_H