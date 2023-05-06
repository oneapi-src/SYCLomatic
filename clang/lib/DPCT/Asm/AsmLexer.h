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

/// DpctAsmLexer - This provides a simple interface that truns a text buffer into a
/// stream of tokens. This provides no support for buffering, or buffering/seeking
/// of tokens, only forward lexing is supported.
class DpctAsmLexer {

  // Start of the buffer.
  const char *BufferStart = nullptr;

  // End of the buffer.
  const char *BufferEnd = nullptr;

  // BufferPtr - Current pointer into the buffer.  This is the next character
  // to be lexed.
  const char *BufferPtr = nullptr;

  /// Mapping/lookup information for all identifiers in
  /// the program, including program keywords.
  mutable DpctAsmIdentifierTable Identifiers;

  /// Cached tokens state.
  using CachedTokensTy = SmallVector<DpctAsmToken, 1>;

  /// Cached tokens are stored here when we do backtracking or
  /// lookahead.
  CachedTokensTy CachedTokens;

  /// The position of the cached token that should "lex" next.
  ///
  /// If it points beyond the CachedTokens vector, it means that a normal
  /// lex() should be invoked.
  CachedTokensTy::size_type CachedLexPos = 0;

public:
  DpctAsmLexer(llvm::MemoryBufferRef Input);
  DpctAsmLexer(const DpctAsmLexer &) = delete;
  DpctAsmLexer &operator=(const DpctAsmLexer &) = delete;
  ~DpctAsmLexer();

  void setBuffer(StringRef Buf);
  bool lex(DpctAsmToken &Result);

  const DpctAsmToken &peekAhead(unsigned N) {
    assert(CachedLexPos + N > CachedTokens.size() && "Confused caching.");
    for (size_t C = CachedLexPos + N - CachedTokens.size(); C > 0; --C) {
      CachedTokens.push_back(DpctAsmToken());
      lex(CachedTokens.back());
    }
    return CachedTokens.back();
  }

  const DpctAsmToken &lookAhead(unsigned N) {
    if (CachedLexPos + N < CachedTokens.size())
      return CachedTokens[CachedLexPos + N];
    return peekAhead(N + 1);
  }

  DpctAsmIdentifierInfo *getIdentifierInfo(StringRef Name) const {
    return &Identifiers.get(Name);
  }

  DpctAsmIdentifierTable &getIdentifiertable() {
    return Identifiers;
  }

  const DpctAsmIdentifierTable &getIdentifiertable() const {
    return Identifiers;
  }

  void cleanIdentifier(llvm::SmallVectorImpl<char> &Buf, StringRef Input) const;
  DpctAsmIdentifierInfo *lookupIdentifierInfo(DpctAsmToken &Identifier) const;

private:
  /// FormTokenWithChars - When we lex a token, we have identified a span
  /// starting at BufferPtr, going to TokEnd that forms the token.  This method
  /// takes that range and assigns it to the token as its location and size.  In
  /// addition, since tokens cannot overlap, this also updates BufferPtr to be
  /// TokEnd.
  void formTokenWithChars(DpctAsmToken &Result, const char *TokEnd,
                          asmtok::TokenKind Kind) {
    unsigned TokLen = TokEnd - BufferPtr;
    Result.setLength(TokLen);
    Result.setLocation(SMLoc::getFromPointer(BufferPtr));
    Result.setKind(Kind);
    BufferPtr = TokEnd;
  }

  bool isHexLiteral(const char *Start) const;

  char getChar(const char *Ptr) const {
    if (Ptr == BufferEnd)
      return 0;
    return *Ptr;
  }

  char getAndAdvanceChar(const char *&Ptr) const { return getChar(Ptr++); }

  const char *consumeChar(const char *Ptr, unsigned Size = 1) const {
    if (Ptr + Size >= BufferEnd)
      return BufferEnd;
    return Ptr + Size;
  }

  bool lexIdentifierContinue(DpctAsmToken &Result, const char *CurPtr);
  bool lexNumericConstant(DpctAsmToken &Result, const char *CurPtr);
  bool skipWhitespace(DpctAsmToken &Result, const char *CurPtr);
  bool lexTokenInternal(DpctAsmToken &Result);
};

} // namespace clang::dpct

#endif // CLANG_DPCT_ASM_LEXER_H
