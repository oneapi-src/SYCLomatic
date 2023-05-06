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

DpctAsmLexer::DpctAsmLexer(llvm::MemoryBufferRef Input)
    : BufferStart(Input.getBufferStart()), BufferEnd(Input.getBufferEnd()),
      BufferPtr(Input.getBufferStart()) {
  Identifiers.AddKeywords();
}

DpctAsmLexer::~DpctAsmLexer() = default;

static inline bool isIdentifierStart(unsigned char C) {
  return clang::isAsciiIdentifierStart(C, /*AllowDollar*/ true) || C == '%';
}

static inline bool isIdentifierContinue(unsigned char C) {
  return clang::isAsciiIdentifierContinue(C, /*AllowDollar*/ true);
}

void DpctAsmLexer::cleanIdentifier(SmallVectorImpl<char> &Buf,
                                   StringRef Input) const {
  Buf.clear();
  Input = Input.drop_while([](char C) { return C == '%'; });
  for (char C : Input) {
    switch (C) {
    case '$':
      Buf.push_back('s');
      break;
    case '%':
      Buf.push_back('p');
      break;
    default:
      Buf.push_back(C);
      break;
    }
  }
  Buf.push_back('_');
  Buf.push_back('c');
  Buf.push_back('t');
}

DpctAsmIdentifierInfo *
DpctAsmLexer::lookupIdentifierInfo(DpctAsmToken &Identifier) const {
  assert(!Identifier.getRawIdentifier().empty() && "No raw identifier data!");
  DpctAsmIdentifierInfo *II;
  if (!Identifier.needsCleaning())
    II = getIdentifierInfo(Identifier.getRawIdentifier());
  else {
    SmallString<64> IdentifierBuffer;
    cleanIdentifier(IdentifierBuffer, Identifier.getRawIdentifier());
    II = getIdentifierInfo(IdentifierBuffer);
  }
  Identifier.setIdentifier(II);
  Identifier.setKind(II->getTokenID());
  return II;
}

bool DpctAsmLexer::isHexLiteral(const char *Start) const {
  char C = getChar(Start);
  if (C != 0)
    return false;
  C = getChar(Start + 1);
  return (C == 'x' || C == 'X');
}

bool DpctAsmLexer::lex(DpctAsmToken &Result) {
  
  if (CachedLexPos < CachedTokens.size()) {
    Result = CachedTokens[CachedLexPos++];
    return true;
  } 
  
  if (!CachedTokens.empty()) {
    // All cached tokens were consumed.
    CachedTokens.clear();
    CachedLexPos = 0;
  }

  Result.startToken();
  /// TODO: Set up misc whitespace flags
  return lexTokenInternal(Result);
}

bool DpctAsmLexer::lexTokenInternal(DpctAsmToken &Result) {
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
      formTokenWithChars(Result, CurPtr, asmtok::eof);
      return true;
    }
    BufferPtr = CurPtr;
    goto LexNextToken;
  case '\r':
  case '\n':
  case ' ':
  case '\t':
  case '\f':
  case '\v':
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
  case '$':
    // clang-format on
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
      Result.setFlag(DpctAsmToken::Placeholder);
      return lexIdentifierContinue(Result, CurPtr);
    }
    if (Char == '%') {
      Char = getAndAdvanceChar(CurPtr);
      if (isIdentifierStart(Char)) {
        if (Char == '%' || Char == '$')
          Result.setFlag(DpctAsmToken::NeedsCleaning);
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
      Result.setFlag(DpctAsmToken::StartOfDot);
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
      CurPtr = consumeChar(CurPtr);
    } else {
      Kind = asmtok::amp;
    }
    break;
   case '|':
    Char = getChar(CurPtr);
    if (Char == '|') {
      Kind = asmtok::pipepipe;
      CurPtr = consumeChar(CurPtr);
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
      CurPtr = consumeChar(CurPtr);
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
      CurPtr = consumeChar(CurPtr);
      Kind = asmtok::lessless;
    } else if (Char == '=') {
      CurPtr = consumeChar(CurPtr);
      Kind = asmtok::lessequal;
    } else {
      Kind = asmtok::less;
    }
    break;
  case '>':
    Char = getChar(CurPtr);
    if (Char == '>') {
      CurPtr = consumeChar(CurPtr);
      Kind = asmtok::greatergreater;
    } else if (Char == '=') {
      CurPtr = consumeChar(CurPtr);
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
      CurPtr = consumeChar(CurPtr);
      Kind = asmtok::coloncolon;
    } else {
      Kind = asmtok::colon;
    }
    break;
  case '=':
    Char = getChar(CurPtr);
    if (Char == '=') {
      CurPtr = consumeChar(CurPtr);
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
  formTokenWithChars(Result, CurPtr, Kind);
  return true;
}

// followsym:   [a-zA-Z0-9_$]
// identifier:  [a-zA-Z]{followsym}* | {[_$%]{followsym}+
// directive:   .identifier
bool DpctAsmLexer::lexIdentifierContinue(DpctAsmToken &Result,
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
      if (!Result.needsCleaning() && (C == '$' || C == '%'))
        Result.setFlag(DpctAsmToken::NeedsCleaning);
      ++CurPtr;
      continue;
    }

    break;
  }

  if (Result.startOfDot())
    ++BufferPtr; // Skip '.'

  const char *IdStart = BufferPtr;
  formTokenWithChars(Result, CurPtr, asmtok::raw_identifier);
  Result.setRawIdentifierData(IdStart);

  DpctAsmIdentifierInfo *II = lookupIdentifierInfo(Result);

  if (Result.isPlaceholder() && II->getName().starts_with("%")) {
    // error: Inline asm placeholder must be resolved in LookUpIdentifierInfo.
    return false;
  }

  if (!Result.isPlaceholder() &&
      (II->getName().contains('%') || II->getName().contains('$'))) {
    // error: This identifier must be cleaned in LookUpIdentifierInfo.
    return false;
  }
  return true;
}

bool DpctAsmLexer::lexNumericConstant(DpctAsmToken &Result,
                                      const char *CurPtr) {
  char C = getChar(CurPtr);
  char PrevCh = 0;
  while (isPreprocessingNumberBody(C)) {
    CurPtr = consumeChar(CurPtr);
    PrevCh = C;
    C = getChar(CurPtr);
  }

  // If we fell out, check for a sign, due to 1e+12.  If we have one, continue.
  if ((C == '-' || C == '+') && (PrevCh == 'E' || PrevCh == 'e')) {
    // If we are in Microsoft mode, don't continue if the constant is hex.
    // For example, MSVC will accept the following as 3 tokens: 0x1234567e+1
    if (isHexLiteral(BufferPtr))
      return lexNumericConstant(Result, consumeChar(CurPtr));
  }

  // If we have a hex FP constant, continue.
  if ((C == '-' || C == '+') && (PrevCh == 'P' || PrevCh == 'p')) {
    return lexNumericConstant(Result, consumeChar(CurPtr));
  }

  const char *TokStart = BufferPtr;
  formTokenWithChars(Result, CurPtr, asmtok::numeric_constant);
  Result.setLiteralData(TokStart);
  return true;
}
