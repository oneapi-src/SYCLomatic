//===----------------------- AsmToken.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_ASM_TOKEN_H
#define CLANG_DPCT_ASM_TOKEN_H

#include "AsmIdentifierTable.h"
#include "AsmTokenKinds.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include <cstddef>

namespace clang::dpct {

using llvm::SMLoc;

class DpctAsmToken {
  SMLoc Loc;
  void *PtrData;
  size_t Length;
  asmtok::TokenKind Kind;
  unsigned Flags;

public:
  enum TokenFlags {
    NeedsCleaning = 0x01, // This identifier contains special characters.
    Placeholder = 0x02,   // This identifier is an inline asm placeholder.
    StartOfDot = 0x04,    // This identifier is an dot identifier.
  };

  asmtok::TokenKind getKind() const { return Kind; }
  void setKind(asmtok::TokenKind K) { Kind = K; }

  /// Return a source location identifier for the specified
  /// offset in the current file.
  SMLoc getLocation() const { return Loc; }
  unsigned getLength() const { return Length; }

  void setLocation(SMLoc L) { Loc = L; }
  void setLength(unsigned Len) { Length = Len; }

  const char *getName() const { return asmtok::getTokenName(Kind); }

  void startToken() {
    Kind = asmtok::unknown;
    PtrData = nullptr;
    Length = 0;
    Flags = 0;
    Loc = SMLoc();
  }

  bool hasPtrData() const { return PtrData != nullptr; }

  DpctAsmIdentifierInfo *getIdentifier() const {
    assert(isNot(asmtok::raw_identifier) &&
           "getIdentifierInfo() on a tok::raw_identifier token!");
    if (is(asmtok::numeric_constant))
      return nullptr;
    if (is(asmtok::eof))
      return nullptr;
    return (DpctAsmIdentifierInfo *)PtrData;
  }

  void setIdentifier(DpctAsmIdentifierInfo *II) { PtrData = (void *)II; }

  StringRef getRawIdentifier() const {
    assert(is(asmtok::raw_identifier));
    return StringRef(reinterpret_cast<const char *>(PtrData), getLength());
  }
  void setRawIdentifierData(const char *Ptr) {
    assert(is(asmtok::raw_identifier));
    PtrData = const_cast<char *>(Ptr);
  }

  const char *getLiteralData() const {
    assert(is(asmtok::numeric_constant) &&
           "Cannot get literal data of non-literal");
    return reinterpret_cast<const char *>(PtrData);
  }

  void setLiteralData(const char *Ptr) {
    assert(is(asmtok::numeric_constant) &&
           "Cannot set literal data of non-literal");
    PtrData = const_cast<char *>(Ptr);
  }

  /// is/isNot - Predicates to check if this token is a specific kind, as in
  /// "if (Tok.is(tok::l_brace)) {...}".
  bool is(asmtok::TokenKind K) const { return Kind == K; }
  bool isNot(asmtok::TokenKind K) const { return Kind != K; }
  bool isOneOf(asmtok::TokenKind K1, asmtok::TokenKind K2) const {
    return is(K1) || is(K2);
  }
  template <typename... Ts> bool isOneOf(asmtok::TokenKind K1, Ts... Ks) const {
    return is(K1) || isOneOf(Ks...);
  }

  /// Set the specified flag.
  void setFlag(TokenFlags Flag) { Flags |= Flag; }

  /// Get the specified flag.
  bool getFlag(TokenFlags Flag) const { return (Flags & Flag) != 0; }

  /// Unset the specified flag.
  void clearFlag(TokenFlags Flag) { Flags &= ~Flag; }

  /// Return the internal represtation of the flags.
  ///
  /// This is only intended for low-level operations such as writing tokens to
  /// disk.
  unsigned getFlags() const { return Flags; }

  /// Return true if this token has trigraphs or escaped newlines in it.
  bool needsCleaning() const { return getFlag(NeedsCleaning); }

  /// Return true if this token is an inline asm placeholder.
  bool isPlaceholder() const { return getFlag(Placeholder); }

  /// Return true if this token is an dot identifier.
  bool startOfDot() const { return getFlag(StartOfDot); }
};

} // namespace clang::dpct

#endif // CLANG_DPCT_ASM_TOKEN_H
