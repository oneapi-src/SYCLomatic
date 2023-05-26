//===----------------------- AsmLexer.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_ASM_LEXER_H
#define CLANG_DPCT_ASM_LEXER_H

#include "Asm/AsmIdentifierTable.h"
#include "AsmToken.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::dpct {

/// InlineAsmLexer - This provides a simple interface that truns a text buffer
/// into a stream of tokens. This provides no support for buffering, or
/// buffering/seeking of tokens, only forward lexing is supported.
class InlineAsmLexer {

  // Start of the buffer.
  const char *BufferStart = nullptr;

  // End of the buffer.
  const char *BufferEnd = nullptr;

  // BufferPtr - Current pointer into the buffer.  This is the next character
  // to be lexed.
  const char *BufferPtr = nullptr;

  /// Mapping/lookup information for all identifiers in
  /// the program, including program keywords.
  mutable InlineAsmIdentifierTable Identifiers;

public:
  InlineAsmLexer(llvm::MemoryBufferRef Input);
  InlineAsmLexer(const InlineAsmLexer &) = delete;
  InlineAsmLexer &operator=(const InlineAsmLexer &) = delete;
  ~InlineAsmLexer();

  bool lex(InlineAsmToken &Result);

  InlineAsmIdentifierInfo *getIdentifierInfo(StringRef Name) const {
    return &Identifiers.get(Name);
  }

  InlineAsmIdentifierTable &getIdentifiertable() { return Identifiers; }

  const InlineAsmIdentifierTable &getIdentifiertable() const {
    return Identifiers;
  }

  void cleanIdentifier(llvm::SmallVectorImpl<char> &Buf, StringRef Input) const;
  InlineAsmIdentifierInfo *
  lookupIdentifierInfo(InlineAsmToken &Identifier) const;

private:
  /// formToken - When we lex a token, we have identified a span
  /// starting at BufferPtr, going to TokEnd that forms the token.  This method
  /// takes that range and assigns it to the token as its location and size.  In
  /// addition, since tokens cannot overlap, this also updates BufferPtr to be
  /// TokEnd.
  void formToken(InlineAsmToken &Result, const char *TokEnd,
                 asmtok::TokenKind Kind) {
    unsigned TokLen = TokEnd - BufferPtr;
    Result.setLength(TokLen);
    Result.setKind(Kind);
    BufferPtr = TokEnd;
  }

  bool isHexLiteral(const char *Start) const;

  char getChar(const char *Ptr) const {
    if (Ptr >= BufferEnd)
      return 0;
    return *Ptr;
  }

  char getAndAdvanceChar(const char *&Ptr) const { return getChar(Ptr++); }
  bool lexIdentifierContinue(InlineAsmToken &Result, const char *CurPtr);
  bool lexNumericConstant(InlineAsmToken &Result, const char *CurPtr);
  bool lexTokenInternal(InlineAsmToken &Result);
};

} // namespace clang::dpct

#endif // CLANG_DPCT_ASM_LEXER_H
