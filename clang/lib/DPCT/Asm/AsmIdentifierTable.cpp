//===------------------ AsmIdentifierTable.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsmIdentifierTable.h"

using namespace clang::dpct;

DpctAsmIdentifierInfoLookup::~DpctAsmIdentifierInfoLookup() = default;

DpctAsmIdentifierTable::DpctAsmIdentifierTable(
    DpctAsmIdentifierInfoLookup *ExternalLookup)
    : HashTable(128), // Start with space for 8K identifiers.
      ExternalLookup(ExternalLookup) {
  // Populate the identifier table with info about keywords for the current
  // language.
  AddKeywords();
}

void DpctAsmIdentifierTable::AddKeywords() {
#define KEYWORD(X, Y) (void)get(#X, asmtok::kw_##X);
#define INSTRUCTION(X) (void)get(#X, asmtok::op_##X);
#include "AsmTokenKinds.def"
}

bool DpctAsmIdentifierInfo::isInstruction() const {
  switch (getTokenID()) {
#define KEYWORD(X, Y)                                                          \
  case asmtok::kw_##X:                                                         \
    return true;
#define INSTRUCTION(X)                                                         \
  case asmtok::op_##X:                                                         \
    return true;
#include "AsmTokenKinds.def"
  default:
    break;
  }
  return false;
}

bool DpctAsmIdentifierInfo::isBuiltinType() const {
  switch (getTokenID()) {
#define BUILTIN_TYPE(X, Y)                                                     \
  case asmtok::kw_##X:                                                         \
    return true;
#include "AsmTokenKinds.def"
  default:
    break;
  }
  return false;
}
