//===----------------------- AsmLexer.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_ASM_LEXER_H
#define CLANG_DPCT_ASM_LEXER_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::dpct {

using llvm::StringRef;

class AsmToken {
public:
  enum TokenKind {
    // Markers
    Eof,
    Error,

    // String values.
    Identifier,
    DotIdentifier,
    String,

    // Integer values.
    Integer,
    Unsigned,

    // IEEE754 floating point values.
    Float,
    Double,

    // Comments
    Comment,
    HashDirective,
    // No-value.
    EndOfStatement,
    Sink,     // '_'
    Question, // '?'
    Colon,
    Space,
    Plus,
    Minus,
    Tilde,
    Slash,     // '/'
    BackSlash, // '\'
    LParen,
    RParen,
    LBrac,
    RBrac,
    LCurly,
    RCurly,
    Star,
    Dot,
    Comma,
    Dollar,
    Equal,
    EqualEqual,

    Pipe,
    PipePipe,
    Caret,
    Amp,
    AmpAmp,
    Exclaim,
    ExclaimEqual,
    Percent,
    Hash,
    Less,
    LessEqual,
    LessLess,
    LessGreater,
    Greater,
    GreaterEqual,
    GreaterGreater,
    At,
    MinusGreater
  };

private:
  TokenKind Kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef Str;

  union {
    int64_t i64;
    uint64_t u64;
    float f32;
    double f64;
  };
public:
  AsmToken() = default;
  AsmToken(TokenKind Kind, StringRef Str)
      : Kind(Kind), Str(Str), u64(0U) {}
  AsmToken(TokenKind Kind, StringRef Str, int64_t I64)
      : Kind(Kind), Str(Str), i64(I64) {}
  AsmToken(TokenKind Kind, StringRef Str, uint64_t U64)
      : Kind(Kind), Str(Str), u64(U64) {}
  AsmToken(TokenKind Kind, StringRef Str, float FpVal)
      : Kind(Kind), Str(Str), f32(FpVal) {}
  AsmToken(TokenKind Kind, StringRef Str, double FpVal)
      : Kind(Kind), Str(Str), f64(FpVal) {}

  TokenKind getKind() const { return Kind; }
  bool is(TokenKind K) const { return Kind == K; }
  bool isNot(TokenKind K) const { return Kind != K; }

  template <typename K, typename... Ks>
  bool is(K k, Ks... ks) const {
    return is(k) || (is(ks) || ...);
  }

  /// Get the contents of a string token (without quotes).
  StringRef getStringContents() const {
    assert(Kind == String && "This token isn't a string!");
    return Str.slice(1, Str.size() - 1);
  }

  bool isStorageClass() const;

  bool isInstructionStorageClass() const;

  bool isTypeName() const;

  bool isVarAttributes() const;

  /// Get the identifier string for the current token, which should be an
  /// identifier or a string. This gets the portion of the string which should
  /// be used as the identifier, e.g., it does not include the quotes on
  /// strings.
  StringRef getIdentifier() const {
    if (Kind == Identifier)
      return getString();
    return getStringContents();
  }

  /// Get the string for the current token, this includes all characters (for
  /// example, the quotes on strings) in the token.
  ///
  /// The returned StringRef points into the source manager's memory buffer, and
  /// is safe to store across calls to Lex().
  StringRef getString() const { return Str; }

  // FIXME: Don't compute this in advance, it makes every token larger, and is
  // also not generally what we want (it is nicer for recovery etc. to lex 123br
  // as a single token, then diagnose as an invalid number).
  int64_t getIntVal() const {
    assert(Kind == Integer && "This token isn't an integer!");
    return i64;
  }

  uint64_t getUnsignedVal() const {
    assert(Kind == Unsigned  && "This token isn't an integer!");
    return u64;
  }

  float getF32Val() const {
    assert(Kind == Float  && "This token isn't an integer!");
    return f32;
  }

  double getF64Val() const {
    assert(Kind == Double  && "This token isn't an integer!");
    return f64;
  }

  void dump(raw_ostream &OS) const;
};
class PtxLexer {
  const char *TokStart = nullptr;
  const char *CurPtr = nullptr;
  StringRef CurBuf;
  bool IsAtStartOfLine = true;
  bool IsAtStartOfStatement = true;
  bool IsPeeking = false;
  bool EndStatementAtEOF = true;
  SmallVector<AsmToken, 1> CurTok;

protected:
  /// LexToken - Read the next token and return its code.
  AsmToken LexToken();

public:
  PtxLexer();
  PtxLexer(const PtxLexer &) = delete;
  PtxLexer &operator=(const PtxLexer &) = delete;
  ~PtxLexer();

  void setBuffer(StringRef Buf, const char *Ptr = nullptr,
                 bool EndStatementAtEOF = true);

  StringRef LexUntilEndOfStatement();

  size_t peekTokens(MutableArrayRef<AsmToken> Buf);

  const AsmToken &Lex();
  void UnLex(AsmToken const &Token);

  bool isAtStartOfStatement() { return IsAtStartOfStatement; }

  /// Get the current (last) lexed token.
  const AsmToken &getTok() const { return CurTok[0]; }

  /// Look ahead at the next token to be lexed.
  const AsmToken peekTok() {
    AsmToken Tok;

    MutableArrayRef<AsmToken> Buf(Tok);
    size_t ReadCount = peekTokens(Buf);

    assert(ReadCount == 1);
    (void)ReadCount;

    return Tok;
  }

private:
  bool isAtStartOfComment(const char *Ptr);
  bool isAtStatementSeparator(const char *Ptr);
  int getNextChar();
  int peekNextChar();
  AsmToken ConsumeIntegerSuffix(unsigned Radix);
  AsmToken ReturnError(const char *Loc, const std::string &Msg);

  AsmToken LexIdentifier();
  AsmToken LexSlash();
  AsmToken LexLineComment();
  AsmToken LexDigit();
  AsmToken LexQuote();

  StringRef LexUntilEndOfLine();
};

} // namespace clang::dpct

#endif // CLANG_DPCT_ASM_LEXER_H
