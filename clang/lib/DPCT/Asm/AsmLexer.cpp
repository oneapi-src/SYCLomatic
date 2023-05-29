//===----------------------- AsmLexer.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsmLexer.h"
#include "Asm/AsmIdentifierTable.h"
#include "Asm/AsmToken.h"
#include "Asm/AsmTokenKinds.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::dpct;

InlineAsmLexer::InlineAsmLexer(llvm::MemoryBufferRef Input)
    : BufferStart(Input.getBufferStart()), BufferEnd(Input.getBufferEnd()),
      BufferPtr(Input.getBufferStart()) {}

InlineAsmLexer::~InlineAsmLexer() = default;

static inline bool isIdentifierStart(unsigned char C) {
  return clang::isAsciiIdentifierStart(C, /*AllowDollar*/ true) || C == '%';
}

static inline bool isIdentifierContinue(unsigned char C) {
  return clang::isAsciiIdentifierContinue(C, /*AllowDollar*/ true);
}

/// Clean the special character in identifier.
/// Rule: 1. replace '$' with '_d_'
///       2. replace '%' with '_p_'
///       3. add a '_' character to escape '_d_' and '_p_'
/// %r -> _p_r
/// %$r -> _p__d_r
/// %r$ -> _p_r_d_
/// %r% -> _p_r_p_
/// %%r -> _p__p_r
/// %r%_d_ -> _p_r_p__d_
/// %r_d_% => _p_r__d__p_
void InlineAsmLexer::cleanIdentifier(SmallVectorImpl<char> &Buf,
                                     StringRef Input) const {
  Buf.clear();
  auto Push = [&Buf](char C) {
    Buf.push_back('_');
    Buf.push_back(C);
    Buf.push_back('_');
  };

  for (const char *Ptr = Input.begin(); Ptr != Input.end(); ++Ptr) {
    char C = *Ptr;
    StringRef SubStr(Ptr, Input.end() - Ptr);
    switch (C) {
    case '$':
      Push('d');
      break;
    case '%':
      Push('p');
      break;
    case '_':
      if (SubStr.starts_with("_d_") || SubStr.starts_with("_p_"))
        Buf.push_back('_');
      [[fallthrough]];
    default:
      Buf.push_back(C);
      break;
    }
  }
}

InlineAsmIdentifierInfo *
InlineAsmLexer::lookupIdentifierInfo(InlineAsmToken &Identifier) const {
  assert(!Identifier.getRawIdentifier().empty() && "No raw identifier data!");
  InlineAsmIdentifierInfo *II;
  StringRef Raw = Identifier.getRawIdentifier();
  if (!getIdentifiertable().contains(
          Raw) && // Maybe a builtin identifier, e.g. %laneid
      (Identifier.needsCleaning() || Raw.contains("_d_") ||
       Raw.contains("_p_"))) {
    SmallString<64> IdentifierBuffer;
    cleanIdentifier(IdentifierBuffer, Identifier.getRawIdentifier());
    II = getIdentifierInfo(IdentifierBuffer);
  } else {
    II = getIdentifierInfo(Identifier.getRawIdentifier());
  }
  Identifier.setIdentifier(II);
  Identifier.setKind(II->getTokenID());
  return II;
}

bool InlineAsmLexer::isHexLiteral(const char *Start) const {
  char C = getChar(Start);
  if (C != '0')
    return false;
  C = getChar(Start + 1);
  return (C == 'x' || C == 'X');
}

bool InlineAsmLexer::lex(InlineAsmToken &Result) {
  Result.startToken();
  /// TODO: Set up misc whitespace flags
  return lexTokenInternal(Result);
}

bool InlineAsmLexer::lexTokenInternal(InlineAsmToken &Result) {
LexNextToken:
  assert(!Result.hasPtrData() && "Result has not been reset");

  // CurPtr - Cache BufferPtr in an automatic variable.
  const char *CurPtr = BufferPtr;

  // Small amounts of horizontal whitespace is very common between tokens.
  if (isHorizontalWhitespace(*CurPtr)) {
    do {
      ++CurPtr;
    } while (isHorizontalWhitespace(*CurPtr));
    BufferPtr = CurPtr;
  }

  // Read a character, advancing over it.
  char Char = getAndAdvanceChar(CurPtr);
  asmtok::TokenKind Kind;

  switch (Char) {
  case 0: // Null.
    if (CurPtr - 1 == BufferEnd) {
      BufferPtr = CurPtr;
      Kind = asmtok::eof;
      break;
    }
    [[fallthrough]];
  case '\r':
  case '\n':
    BufferPtr = CurPtr;
    goto LexNextToken;
  // clang-format off
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
  // clang-format on  
    return lexNumericConstant(Result, CurPtr);
  // clang-format off
  case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G':
  case 'H': case 'I': case 'J': case 'K': case 'L': case 'M': case 'N':
  case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T': case 'U':
  case 'V': case 'W': case 'X': case 'Y': case 'Z':
  case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
  case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
  case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
  case 'v': case 'w': case 'x': case 'y': case 'z':
    // clang-format on
    return lexIdentifierContinue(Result, CurPtr);
  case '$':
    Result.setFlag(InlineAsmToken::NeedsCleaning);
    return lexIdentifierContinue(Result, CurPtr);
  case '_':
    Char = getChar(CurPtr);
    if (isIdentifierContinue(Char))
      return lexIdentifierContinue(Result, CurPtr);
    Kind = asmtok::underscore;
    break;
  case '%':
    Char = getChar(CurPtr);
    if (isDigit(Char)) {
      Result.setFlag(InlineAsmToken::Placeholder);
      return lexIdentifierContinue(Result, CurPtr);
    }
    if (Char == '%') {
      BufferPtr = CurPtr;
      ++CurPtr;
      Char = getChar(CurPtr);
      if (isIdentifierContinue(Char)) {
        Result.setFlag(InlineAsmToken::NeedsCleaning);
        return lexIdentifierContinue(Result, CurPtr);
      }
      Kind = asmtok::percent;
    } else if (Char == '{') {
      ++CurPtr;
      Kind = asmtok::l_brace;
    } else if (Char == '}') {
      ++CurPtr;
      Kind = asmtok::r_brace;
    } else if (Char == '|') {
      ++CurPtr;
      Kind = asmtok::pipe;
    } else {
      // error: unknown escaped characher.
      return false;
    }
    break;
  case '?':
    Kind = asmtok::question;
    break;
  case '[':
    Kind = asmtok::l_square;
    break;
  case ']':
    Kind = asmtok::r_square;
    break;
  case '(':
    Kind = asmtok::l_paren;
    break;
  case ')':
    Kind = asmtok::r_paren;
    break;
  case '{':
    Kind = asmtok::l_brace;
    break;
  case '}':
    Kind = asmtok::r_brace;
    break;
  case '.':
    Char = getChar(CurPtr);
    if (isIdentifierStart(Char)) {
      Result.setFlag(InlineAsmToken::StartOfDot);
      return lexIdentifierContinue(Result, CurPtr);
    }
    if (isDigit(Char)) {
      return lexNumericConstant(Result, CurPtr);
    }
    Kind = asmtok::period;
    break;
  case '&':
    Char = getChar(CurPtr);
    if (Char == '&') {
      Kind = asmtok::ampamp;
      CurPtr++;
    } else {
      Kind = asmtok::amp;
    }
    break;
  case '|':
    Char = getChar(CurPtr);
    if (Char == '|') {
      Kind = asmtok::pipepipe;
      CurPtr++;
    } else {
      Kind = asmtok::pipe;
    }
    break;
  case '*':
    Kind = asmtok::star;
    break;
  case '+':
    Kind = asmtok::plus;
    break;
  case '-':
    Kind = asmtok::minus;
    break;
  case '~':
    Kind = asmtok::tilde;
    break;
  case '!':
    if (getChar(CurPtr) == '=') {
      Kind = asmtok::exclaimequal;
      CurPtr++;
    } else {
      Kind = asmtok::exclaim;
    }
    break;
  case '/':
    Kind = asmtok::slash;
    break;
  case '<':
    Char = getChar(CurPtr);
    if (Char == '<') {
      CurPtr++;
      Kind = asmtok::lessless;
    } else if (Char == '=') {
      CurPtr++;
      Kind = asmtok::lessequal;
    } else {
      Kind = asmtok::less;
    }
    break;
  case '>':
    Char = getChar(CurPtr);
    if (Char == '>') {
      CurPtr++;
      Kind = asmtok::greatergreater;
    } else if (Char == '=') {
      CurPtr++;
      Kind = asmtok::greaterequal;
    } else {
      Kind = asmtok::greater;
    }
    break;
  case '^':
    Kind = asmtok::caret;
    break;
  case ':':
    Char = getChar(CurPtr);
    if (Char == ':') {
      CurPtr++;
      Kind = asmtok::coloncolon;
    } else {
      Kind = asmtok::colon;
    }
    break;
  case '=':
    Char = getChar(CurPtr);
    if (Char == '=') {
      CurPtr++;
      Kind = asmtok::equalequal;
    } else {
      Kind = asmtok::equal;
    }
    break;
  case ',':
    Kind = asmtok::comma;
    break;
  case '@':
    Kind = asmtok::at;
    break;
  case ';':
    Kind = asmtok::semi;
    break;
  default:
    Kind = asmtok::unknown;
    break;
  }
  formToken(Result, CurPtr, Kind);
  return true;
}

// followsym:   [a-zA-Z0-9_$]
// identifier:  [a-zA-Z]{followsym}* | {[_$%]{followsym}+
// directive:   .identifier
bool InlineAsmLexer::lexIdentifierContinue(InlineAsmToken &Result,
                                           const char *CurPtr) {
  // Match [a-zA-Z0-9_$]*, we have already matched an identifier start.
  while (true) {
    char C = getChar(CurPtr);

    if (isDigit(C)) {
      ++CurPtr;
      continue;
    }

    if (isIdentifierContinue(C)) {
      if (Result.isPlaceholder())
        break;
      if (C == '$' || C == '%')
        Result.setFlag(InlineAsmToken::NeedsCleaning);
      ++CurPtr;
      continue;
    }

    break;
  }

  if (Result.startOfDot())
    ++BufferPtr; // Skip '.'

  const char *IdStart = BufferPtr;
  formToken(Result, CurPtr, asmtok::raw_identifier);
  Result.setRawIdentifierData(IdStart);

  (void)lookupIdentifierInfo(Result);
  return true;
}

bool InlineAsmLexer::lexNumericConstant(InlineAsmToken &Result,
                                        const char *CurPtr) {
  char C = getChar(CurPtr);
  char PrevCh = 0;
  while (isPreprocessingNumberBody(C)) {
    CurPtr++;
    PrevCh = C;
    C = getChar(CurPtr);
  }

  // If we fell out, check for a sign, due to 1e+12.  If we have one, continue.
  if ((C == '-' || C == '+') && (PrevCh == 'E' || PrevCh == 'e')) {
    // If we are in Microsoft mode, don't continue if the constant is hex.
    // For example, MSVC will accept the following as 3 tokens: 0x1234567e+1
    if (!isHexLiteral(BufferPtr))
      return lexNumericConstant(Result, CurPtr++);
  }

  // If we have a hex FP constant, continue.
  if ((C == '-' || C == '+') && (PrevCh == 'P' || PrevCh == 'p')) {
    return lexNumericConstant(Result, CurPtr++);
  }

  const char *TokStart = BufferPtr;
  formToken(Result, CurPtr, asmtok::numeric_constant);
  Result.setLiteralData(TokStart);
  return true;
}
