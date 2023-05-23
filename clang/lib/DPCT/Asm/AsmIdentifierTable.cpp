//===------------------ AsmIdentifierTable.cpp ------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsmIdentifierTable.h"

using namespace clang::dpct;

InlineAsmIdentifierInfoLookup::~InlineAsmIdentifierInfoLookup() = default;

InlineAsmIdentifierTable::InlineAsmIdentifierTable(
    InlineAsmIdentifierInfoLookup *ExternalLookup)
    : HashTable(128), // Start with space for 8K identifiers.
      ExternalLookup(ExternalLookup) {
  // Populate the identifier table with info about keywords for the current
  // language.
  AddKeywords();
}

void InlineAsmIdentifierTable::AddKeywords() {
#define KEYWORD(X, Y) get(#X, asmtok::kw_##X);
#define BUILTIN_TYPE(X, Y) get(#X, asmtok::kw_##X).setBuiltinType();
#define INSTRUCTION(X) get(#X, asmtok::op_##X).setInstruction();
#include "AsmTokenKinds.def"
}
