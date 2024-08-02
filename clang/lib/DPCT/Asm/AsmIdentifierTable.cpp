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
      ExternalLookup(ExternalLookup) {}

void InlineAsmIdentifierTable::AddKeywords() {
#define KEYWORD(X, Y) get(Y, asmtok::kw_##X);
#define SPECIAL_REG(X, Y, Z)                                                   \
  get(Y, asmtok::bi_##X).setFlag(InlineAsmIdentifierInfo::SpecialReg);
#define BUILTIN_TYPE(X, Y)                                                     \
  get(Y, asmtok::kw_##X).setFlag(InlineAsmIdentifierInfo::BuiltinType);
#define INSTRUCTION(X)                                                         \
  get(#X, asmtok::op_##X).setFlag(InlineAsmIdentifierInfo::Instruction);
#define MODIFIER(X, Y)                                                         \
  get(Y, asmtok::kw_##X).setFlag(InlineAsmIdentifierInfo::Modifier);
#define STATE_SPACE(X, Y)                                                      \
  get(Y, asmtok::kw_##X).setFlag(InlineAsmIdentifierInfo::StateSpace);
#include "AsmTokenKinds.def"
}
