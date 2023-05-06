//===----------------------- AsmTokenKinds.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsmTokenKinds.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::dpct;

static const char * const TokNames[] = {
#define TOK(X) #X,
#define KEYWORD(X,Y) #X,
#include "AsmTokenKinds.def"
  nullptr
};

const char *asmtok::getTokenName(asmtok::TokenKind Kind) {
  if (Kind < asmtok::NUM_TOKENS)
    return TokNames[Kind];
  llvm_unreachable("unknown TokenKind");
  return nullptr;
}

const char *asmtok::getPunctuatorSpelling(asmtok::TokenKind Kind) {
  switch (Kind) {
#define PUNCTUATOR(X,Y) case X: return Y;
#include "AsmTokenKinds.def"
  default: break;
  }
  return nullptr;
}

const char *asmtok::getKeywordSpelling(asmtok::TokenKind Kind) {
  switch (Kind) {
#define KEYWORD(X,Y) case kw_ ## X: return #X;
#include "AsmTokenKinds.def"
    default: break;
  }
  return nullptr;
}

