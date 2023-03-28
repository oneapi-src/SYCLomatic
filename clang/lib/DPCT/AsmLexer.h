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

using llvm::APInt;
using llvm::StringRef;

class AsmToken {
public:
  enum TokenKind {
    // Markers
    Eof,
    Error,

    // String values.
    Identifier,
    String,

    // Integer values.
    Integer,
    BigNum, // larger than 64 bits

    // Real values.
    Real,

    // Comments
    Comment,
    HashDirective,
    // No-value.
    EndOfStatement,
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

  APInt IntVal;

public:
  AsmToken() = default;
  AsmToken(TokenKind Kind, StringRef Str, APInt IntVal)
      : Kind(Kind), Str(Str), IntVal(std::move(IntVal)) {}
  AsmToken(TokenKind Kind, StringRef Str, int64_t IntVal = 0)
      : Kind(Kind), Str(Str), IntVal(64, IntVal, true) {}

  TokenKind getKind() const { return Kind; }
  bool is(TokenKind K) const { return Kind == K; }
  bool isNot(TokenKind K) const { return Kind != K; }

  /// Get the contents of a string token (without quotes).
  StringRef getStringContents() const {
    assert(Kind == String && "This token isn't a string!");
    return Str.slice(1, Str.size() - 1);
  }

  /// Check if this token is a builtin identifier
  /// .address_size   .explicitcluster  .maxnreg            .section
  /// .alias          .extern           .maxntid            .shared
  /// .align          .file             .minnctapersm       .sreg
  /// .branchtargets  .func             .noreturn           .target
  /// .callprototype  .global           .param              .tex
  /// .calltargets    .loc              .pragma             .version
  /// .common         .local            .reg                .visible
  /// .const          .maxclusterrank   .reqnctapercluster  .weak
  /// .entry          .maxnctapersm     .reqntid
  bool isDirective() const;

  /// Check if this token is a builtin identifier
  /// %clock      %laneid       %lanemask_gt    %pm0, ..., %pm7
  /// %clock64    %lanemask_eq  %nctaid         %smid
  /// %ctaid      %lanemask_le  %ntid           %tid
  /// %envreg<32> %lanemask_lt  %nsmid          %warpid
  /// %gridid     %lanemask_ge  %nwarpid        WARP_SZ
  bool isBuiltinIdentifier() const;

  /// Check if this token is a 'Fundamental Type Specifiers'
  /// Signed integer    .s8,  .s16,   .s32, .s64
  /// Unsigned integer  .u8,  .u16,   .u32, .u64
  /// Floating-point    .f16, .f16x2, .f32, .f64
  /// Bits (untyped)    .b8,  .b16,   .b32, .b64
  /// Predicate         .pred
  bool isFundamentalTypeSpecifier() const;

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
    return IntVal.getZExtValue();
  }

  APInt getAPIntVal() const {
    assert((Kind == Integer || Kind == BigNum) &&
           "This token isn't an integer!");
    return IntVal;
  }

  void dump(raw_ostream &OS) const;
};
class AsmLexer {
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
  AsmLexer();
  AsmLexer(const AsmLexer &) = delete;
  AsmLexer &operator=(const AsmLexer &) = delete;
  ~AsmLexer();

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
  AsmToken ReturnError(const char *Loc, const std::string &Msg);

  AsmToken LexIdentifier();
  AsmToken LexSlash();
  AsmToken LexLineComment();
  AsmToken LexDigit();
  AsmToken LexSingleQuote();
  AsmToken LexQuote();
  AsmToken LexFloatLiteral();
  AsmToken LexHexFloatLiteral(bool NoIntDigits);

  StringRef LexUntilEndOfLine();
};

} // namespace clang::dpct

#endif // CLANG_DPCT_ASM_LEXER_H
