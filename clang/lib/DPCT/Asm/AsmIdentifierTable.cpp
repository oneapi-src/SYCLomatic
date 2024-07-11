//===------------------ AsmIdentifierTable.cpp ------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsmIdentifierTable.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::dpct;

InlineAsmIdentifierInfoLookup::~InlineAsmIdentifierInfoLookup() = default;

InlineAsmIdentifierTable::InlineAsmIdentifierTable(
    InlineAsmIdentifierInfoLookup *ExternalLookup)
    : HashTable(128), // Start with space for 8K identifiers.
      ExternalLookup(ExternalLookup) {}

void InlineAsmIdentifierTable::AddKeywords() {
#define KEYWORD(X, Y) get(Y, asmtok::kw_##X);
#define BUILTIN_ID(X, Y, Z)                                                    \
  get(Y, asmtok::bi_##X).setFlag(InlineAsmIdentifierInfo::BuiltinID);
#define BUILTIN_TYPE(X, Y)                                                     \
  get(Y, asmtok::kw_##X).setFlag(InlineAsmIdentifierInfo::BuiltinType);
#define INSTRUCTION(X)                                                         \
  get(#X, asmtok::op_##X).setFlag(InlineAsmIdentifierInfo::Instruction);
#define ROUND_MOD(X, Y)                                                        \
  get(Y, asmtok::kw_##X).setFlag(InlineAsmIdentifierInfo::InstAttr);
#define SAT_MOD(X, Y)                                                          \
  get(Y, asmtok::kw_##X).setFlag(InlineAsmIdentifierInfo::InstAttr);
#define MUL_MOD(X, Y)                                                          \
  get(Y, asmtok::kw_##X).setFlag(InlineAsmIdentifierInfo::InstAttr);
#define CMP_OP(X, Y)                                                           \
  get(Y, asmtok::kw_##X).setFlag(InlineAsmIdentifierInfo::InstAttr);
#define BIN_OP(X, Y)                                                           \
  get(Y, asmtok::kw_##X).setFlag(InlineAsmIdentifierInfo::InstAttr);
#define SYNC_OP(X, Y)                                                          \
  get(Y, asmtok::kw_##X).setFlag(InlineAsmIdentifierInfo::InstAttr);
#include "AsmTokenKinds.def"
}
