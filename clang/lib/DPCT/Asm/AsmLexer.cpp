//===----------------------- AsmLexer.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsmLexer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::dpct;

using llvm::hexDigitValue;
using llvm::isAlnum;
using llvm::isDigit;
using llvm::isHexDigit;
using llvm::SaveAndRestore;
using llvm::StringSet;

bool PtxToken::isStorageClass() const {
  return llvm::StringSwitch<bool>(getString())
      .Case(".reg", true)
      .Case(".const", true)
      .Case(".global", true)
      .Case(".local", true)
      .Case(".param", true)
      .Case(".shared", true)
      .Case(".tex", true)
      .Default(false);
}

bool PtxToken::isInstructionStorageClass() const {
  return llvm::StringSwitch<bool>(getString())
      .Case(".const", true)
      .Case(".global", true)
      .Case(".local", true)
      .Case(".param", true)
      .Case(".shared", true)
      .Case(".tex", true)
      .Default(false);
}

bool PtxToken::isTypeName() const {
  static StringSet<> TypeNames{".s2",     ".s4",     ".s8",   ".s16",   ".s32",
                               ".s64",    ".u2",     ".u4",   ".u8",    ".u16",
                               ".u32",    ".u64",    ".byte", ".4byte", ".b8",
                               ".b16",    ".b32",    ".b64",  ".b128",  ".f16",
                               ".f16x2",  ".f32",    ".f64",  ".e4m3",  ".e5m2",
                               ".e4m3x2", ".e5m2x2", ".quad", ".pred"};
  return TypeNames.contains(getString());
}

bool PtxToken::isVarAttributes() const {
  return getString() == ".align" || getString() == ".attribute" ||
         isStorageClass();
}

void PtxToken::dump(raw_ostream &OS) const {
  switch (Kind) {
  case PtxToken::Error:
    OS << "error";
    break;
  case PtxToken::Identifier:
    OS << "identifier";
    break;
  case PtxToken::DotIdentifier:
    OS << "dot identifier";
    break;
  case PtxToken::Integer:
    OS << "int";
    break;
  case PtxToken::Unsigned:
    OS << "unsigned";
    break;
  case PtxToken::Float:
    OS << "float";
    break;
  case PtxToken::Double:
    OS << "double";
    break;
  case PtxToken::String:
    OS << "string";
    break;

  case PtxToken::Amp:
    OS << "Amp";
    break;
  case PtxToken::AmpAmp:
    OS << "AmpAmp";
    break;
  case PtxToken::At:
    OS << "At";
    break;
  case PtxToken::BackSlash:
    OS << "BackSlash";
    break;
  case PtxToken::Caret:
    OS << "Caret";
    break;
  case PtxToken::Colon:
    OS << "Colon";
    break;
  case PtxToken::Comma:
    OS << "Comma";
    break;
  case PtxToken::Comment:
    OS << "Comment";
    break;
  case PtxToken::Dollar:
    OS << "Dollar";
    break;
  case PtxToken::Dot:
    OS << "Dot";
    break;
  case PtxToken::EndOfStatement:
    OS << "EndOfStatement";
    break;
  case PtxToken::Eof:
    OS << "Eof";
    break;
  case PtxToken::Equal:
    OS << "Equal";
    break;
  case PtxToken::EqualEqual:
    OS << "EqualEqual";
    break;
  case PtxToken::Exclaim:
    OS << "Exclaim";
    break;
  case PtxToken::ExclaimEqual:
    OS << "ExclaimEqual";
    break;
  case PtxToken::Greater:
    OS << "Greater";
    break;
  case PtxToken::GreaterEqual:
    OS << "GreaterEqual";
    break;
  case PtxToken::GreaterGreater:
    OS << "GreaterGreater";
    break;
  case PtxToken::Hash:
    OS << "Hash";
    break;
  case PtxToken::HashDirective:
    OS << "HashDirective";
    break;
  case PtxToken::LBrac:
    OS << "LBrac";
    break;
  case PtxToken::LCurly:
    OS << "LCurly";
    break;
  case PtxToken::LParen:
    OS << "LParen";
    break;
  case PtxToken::Less:
    OS << "Less";
    break;
  case PtxToken::LessEqual:
    OS << "LessEqual";
    break;
  case PtxToken::LessGreater:
    OS << "LessGreater";
    break;
  case PtxToken::LessLess:
    OS << "LessLess";
    break;
  case PtxToken::Minus:
    OS << "Minus";
    break;
  case PtxToken::MinusGreater:
    OS << "MinusGreater";
    break;
  case PtxToken::Percent:
    OS << "Percent";
    break;
  case PtxToken::Pipe:
    OS << "Pipe";
    break;
  case PtxToken::PipePipe:
    OS << "PipePipe";
    break;
  case PtxToken::Plus:
    OS << "Plus";
    break;
  case PtxToken::RBrac:
    OS << "RBrac";
    break;
  case PtxToken::RCurly:
    OS << "RCurly";
    break;
  case PtxToken::RParen:
    OS << "RParen";
    break;
  case PtxToken::Slash:
    OS << "Slash";
    break;
  case PtxToken::Space:
    OS << "Space";
    break;
  case PtxToken::Star:
    OS << "Star";
    break;
  case PtxToken::Tilde:
    OS << "Tilde";
    break;
  case Sink:
    OS << "Sink";
    break;
  case Question:
    OS << "Question";
    break;
  }

  // Print the token string.
  OS << " (\"";
  OS.write_escaped(getString());
  OS << "\")";
}

PtxLexer::PtxLexer() {
  TokStart = nullptr;
  CurTok.emplace_back(PtxToken::Space, StringRef());
}

PtxLexer::~PtxLexer() = default;

void PtxLexer::setBuffer(StringRef Buf, const char *Ptr,
                         bool EndStatementAtEOF) {
  CurBuf = Buf;

  if (Ptr)
    CurPtr = Ptr;
  else
    CurPtr = CurBuf.begin();

  TokStart = nullptr;
  this->EndStatementAtEOF = EndStatementAtEOF;
}

const PtxToken &PtxLexer::Lex() {
  assert(!CurTok.empty());
  // Mark if we parsing out a EndOfStatement.
  IsAtStartOfStatement = CurTok.front().getKind() == PtxToken::EndOfStatement;
  CurTok.erase(CurTok.begin());
  // LexToken may generate multiple tokens via UnLex but will always return
  // the first one. Place returned value at head of CurTok vector.
  if (CurTok.empty()) {
    PtxToken T = LexToken();
    CurTok.insert(CurTok.begin(), T);
  }
  return CurTok.front();
}

void PtxLexer::UnLex(PtxToken const &Token) {
  IsAtStartOfStatement = false;
  CurTok.insert(CurTok.begin(), Token);
}

PtxToken PtxLexer::ReturnError(const char *Loc, const std::string &Msg) {
  llvm::errs() << llvm::raw_ostream::RED << Msg << llvm::raw_ostream::RESET << "\n";
  return PtxToken(PtxToken::Error, StringRef(Loc, CurPtr - Loc));
}

int PtxLexer::getNextChar() {
  if (CurPtr == CurBuf.end())
    return EOF;
  return (unsigned char)*CurPtr++;
}

int PtxLexer::peekNextChar() {
  if (CurPtr == CurBuf.end())
    return EOF;
  return (unsigned char)*CurPtr;
}

static bool isDirectiveStart(char C) { return C == '.'; }

static bool isIdentifierHead(char C) {
  return llvm::isAlpha(C) || C == '_' || C == '$' || C == '%';
}

static bool isIdentifierBody(char C) {
  return isAlnum(C) || C == '_' || C == '$';
}

/// LexIdentifier: [a-zA-Z_$.@?][a-zA-Z0-9_$.@#?]*
static bool isIdentifierChar(char C) {
  return isAlnum(C) || C == '_' || C == '$' || C == '.' || C == '%';
}

// followsym:   [a-zA-Z0-9_$]
// identifier:  [a-zA-Z]{followsym}* | {[_$%]{followsym}+
// directive:   .identifier
PtxToken PtxLexer::LexIdentifier() {
  auto Kind = PtxToken::Identifier;
  if (CurPtr[-1] == '.')
    Kind = PtxToken::DotIdentifier;
  while (isIdentifierBody(*CurPtr))
    ++CurPtr;

  // Handle . as a special case.
  if (CurPtr == TokStart + 1 && TokStart[0] == '.')
    return PtxToken(PtxToken::Dot, StringRef(TokStart, 1));
  return PtxToken(Kind, StringRef(TokStart, CurPtr - TokStart));
}

/// LexSlash: Slash: /
///           C-Style Comment: /* ... */
///           C-style Comment: // ...
PtxToken PtxLexer::LexSlash() {

  switch (*CurPtr) {
  case '*':
    IsAtStartOfStatement = false;
    break; // C style comment.
  case '/':
    ++CurPtr;
    return LexLineComment();
  default:
    IsAtStartOfStatement = false;
    return PtxToken(PtxToken::Slash, StringRef(TokStart, 1));
  }

  // C Style comment.
  ++CurPtr; // skip the star.
  while (CurPtr != CurBuf.end()) {
    switch (*CurPtr++) {
    case '*':
      // End of the comment?
      if (*CurPtr != '/')
        break;
      ++CurPtr; // End the */.
      return PtxToken(PtxToken::Comment,
                      StringRef(TokStart, CurPtr - TokStart));
    }
  }
  return ReturnError(TokStart, "unterminated comment");
}

/// LexLineComment: Comment: #[^\n]*
///                        : //[^\n]*
PtxToken PtxLexer::LexLineComment() {
  // Mark This as an end of statement with a body of the
  // comment. While it would be nicer to leave this two tokens,
  // backwards compatability with TargetParsers makes keeping this in this form
  // better.
  int CurChar = getNextChar();
  while (CurChar != '\n' && CurChar != '\r' && CurChar != EOF)
    CurChar = getNextChar();
  if (CurChar == '\r' && CurPtr != CurBuf.end() && *CurPtr == '\n')
    ++CurPtr;

  IsAtStartOfLine = true;
  // This is a whole line comment. leave newline
  if (IsAtStartOfStatement)
    return PtxToken(PtxToken::EndOfStatement,
                    StringRef(TokStart, CurPtr - TokStart));
  IsAtStartOfStatement = true;

  return PtxToken(PtxToken::EndOfStatement,
                  StringRef(TokStart, CurPtr - 1 - TokStart));
}

PtxToken PtxLexer::ConsumeIntegerSuffix() {
  if (CurPtr[0] == 'U') {
    uint64_t Result = 0;
    if (StringRef(TokStart, CurPtr - TokStart).getAsInteger(0, Result))
      return ReturnError(TokStart, "invalid hexadecimal number");

    ++CurPtr; // Skip 'U' suffix
    return PtxToken(PtxToken::Unsigned, StringRef(TokStart, CurPtr - TokStart),
                    Result);
  }

  int64_t Result = 0;
  if (StringRef(TokStart, CurPtr - TokStart).getAsInteger(0, Result))
    return ReturnError(TokStart, "invalid hexadecimal number");
 
  return PtxToken(PtxToken::Integer, StringRef(TokStart, CurPtr - TokStart),
                  Result);
}

/// 0[fF]{hexdigit}{8}      // single-precision floating point
/// 0[dD]{hexdigit}{16}     // double-precision floating point
/// hexadecimal literal:  0[xX]{hexdigit}+U?
/// octal literal:        0{octal digit}+U?
/// binary literal:       0[bB]{bit}+U?
/// decimal literal       {nonzero-digit}{digit}*U?
PtxToken PtxLexer::LexDigit() {

  if (*CurPtr == 'f' || *CurPtr == 'F') {
    ++CurPtr;
    const char *Begin = CurPtr;

    while (isHexDigit(*CurPtr))
      ++CurPtr;

    if (CurPtr - Begin < 8U)
      return ReturnError(
          CurPtr, "Invalid IEEE 754 single-precision floating point values");

    float F32;
    uint32_t F32bytes = 0;
    (void)StringRef(Begin, 8).getAsInteger(16, F32bytes);
    std::memcpy(&F32, &F32bytes, sizeof(float));
    return PtxToken(PtxToken::Float, StringRef(TokStart, CurPtr - TokStart),
                    F32);
  }

  if (*CurPtr == 'd' || *CurPtr == 'D') {
    ++CurPtr;
    const char *Begin = CurPtr;
    while (isHexDigit(*CurPtr))
      ++CurPtr;

    if (CurPtr - Begin < 16U)
      return ReturnError(
          CurPtr, "Invalid IEEE 754 double-precision floating point values");

    double F64;
    uint64_t F64bytes = 0;
    (void)StringRef(Begin, 8).getAsInteger(16, F64bytes);
    std::memcpy(&F64, &F64bytes, sizeof(double));
    return PtxToken(PtxToken::Double, StringRef(TokStart, CurPtr - TokStart),
                    F64);
  }

  if ((*CurPtr == 'x') || (*CurPtr == 'X')) {
    ++CurPtr;
    const char *NumStart = CurPtr;
    while (isHexDigit(CurPtr[0]))
      ++CurPtr;

    // Otherwise requires at least one hex digit.
    if (CurPtr == NumStart)
      return ReturnError(CurPtr - 2, "invalid hexadecimal number");

    return ConsumeIntegerSuffix();
  }

  if ((*CurPtr == 'b') || (*CurPtr == 'B')) {
    ++CurPtr;
    const char *NumStart = CurPtr;
    while (CurPtr[0] == '0' || CurPtr[0] == '1')
      ++CurPtr;

    if (CurPtr == NumStart)
      return ReturnError(CurPtr - 2, "invalid binary number");

    return ConsumeIntegerSuffix();
  }

  while (isDigit(*CurPtr)) ++CurPtr;

  return ConsumeIntegerSuffix();
}

/// LexQuote: String: "..."
PtxToken PtxLexer::LexQuote() {
  int CurChar = getNextChar();

  // TODO: does gas allow multiline string constants?
  while (CurChar != '"') {
    if (CurChar == '\\') {
      // Allow \", etc.
      CurChar = getNextChar();
    }

    if (CurChar == EOF)
      return ReturnError(TokStart, "unterminated string constant");

    CurChar = getNextChar();
  }

  return PtxToken(PtxToken::String, StringRef(TokStart, CurPtr - TokStart));
}

StringRef PtxLexer::LexUntilEndOfStatement() {
  TokStart = CurPtr;

  while (!isAtStartOfComment(CurPtr) &&     // Start of line comment.
         !isAtStatementSeparator(CurPtr) && // End of statement marker.
         *CurPtr != '\n' && *CurPtr != '\r' && CurPtr != CurBuf.end()) {
    ++CurPtr;
  }
  return StringRef(TokStart, CurPtr - TokStart);
}

StringRef PtxLexer::LexUntilEndOfLine() {
  TokStart = CurPtr;

  while (*CurPtr != '\n' && *CurPtr != '\r' && CurPtr != CurBuf.end()) {
    ++CurPtr;
  }
  return StringRef(TokStart, CurPtr - TokStart);
}

size_t PtxLexer::peekTokens(MutableArrayRef<PtxToken> Buf) {
  SaveAndRestore SavedTokenStart(TokStart);
  SaveAndRestore SavedCurPtr(CurPtr);
  SaveAndRestore SavedAtStartOfLine(IsAtStartOfLine);
  SaveAndRestore SavedAtStartOfStatement(IsAtStartOfStatement);
  SaveAndRestore SavedIsPeeking(IsPeeking, true);

  size_t ReadCount;
  for (ReadCount = 0; ReadCount < Buf.size(); ++ReadCount) {
    PtxToken Token = LexToken();

    Buf[ReadCount] = Token;

    if (Token.is(PtxToken::Eof))
      break;
  }

  return ReadCount;
}

static const char *getSeparatorString() { return ";"; }

bool PtxLexer::isAtStartOfComment(const char *Ptr) {

  StringRef CommentString = "//";

  if (CommentString.size() == 1)
    return CommentString[0] == Ptr[0];

  // Allow # preprocessor comments also be counted as comments for "##" cases
  if (CommentString[1] == '#')
    return CommentString[0] == Ptr[0];

  return strncmp(Ptr, CommentString.data(), CommentString.size()) == 0;
}

bool PtxLexer::isAtStatementSeparator(const char *Ptr) {
  return strncmp(Ptr, getSeparatorString(), strlen(getSeparatorString())) == 0;
}

PtxToken PtxLexer::LexToken() {
  TokStart = CurPtr;
  // This always consumes at least one character.
  int CurChar = getNextChar();

  if (isAtStartOfComment(TokStart))
    return LexLineComment();

  if (isAtStatementSeparator(TokStart)) {
    CurPtr += strlen(getSeparatorString()) - 1;
    IsAtStartOfLine = true;
    IsAtStartOfStatement = true;
    return PtxToken(PtxToken::EndOfStatement,
                    StringRef(TokStart, strlen(getSeparatorString())));
  }

  // If we're missing a newline at EOF, make sure we still get an
  // EndOfStatement token before the Eof token.
  if (CurChar == EOF && !IsAtStartOfStatement && EndStatementAtEOF) {
    IsAtStartOfLine = true;
    IsAtStartOfStatement = true;
    return PtxToken(PtxToken::EndOfStatement, StringRef(TokStart, 0));
  }
  IsAtStartOfLine = false;
  bool OldIsAtStartOfStatement = IsAtStartOfStatement;
  IsAtStartOfStatement = false;
  switch (CurChar) {
  default:
    // Handle identifier: [a-zA-Z_.?][a-zA-Z0-9_$.@#?]*
    if (isalpha(CurChar) || CurChar == '%')
      return LexIdentifier();

    // Unknown character, emit an error.
    return ReturnError(TokStart, "invalid character in input");
  case EOF:
    if (EndStatementAtEOF) {
      IsAtStartOfLine = true;
      IsAtStartOfStatement = true;
    }
    return PtxToken(PtxToken::Eof, StringRef(TokStart, 0));
  case 0:
  case ' ':
  case '\t':
    IsAtStartOfStatement = OldIsAtStartOfStatement;
    while (*CurPtr == ' ' || *CurPtr == '\t')
      CurPtr++;
    return LexToken(); // Ignore whitespace.
  case '\r': {
    IsAtStartOfLine = true;
    IsAtStartOfStatement = true;
    // If this is a CR followed by LF, treat that as one token.
    if (CurPtr != CurBuf.end() && *CurPtr == '\n')
      ++CurPtr;
    return PtxToken(PtxToken::EndOfStatement,
                    StringRef(TokStart, CurPtr - TokStart));
  }
  case '\n':
    IsAtStartOfLine = true;
    return LexToken();  // Ignore whitespace.
  case ':':
    return PtxToken(PtxToken::Colon, StringRef(TokStart, 1));
  case '+':
    return PtxToken(PtxToken::Plus, StringRef(TokStart, 1));
  case '~':
    return PtxToken(PtxToken::Tilde, StringRef(TokStart, 1));
  case '(':
    return PtxToken(PtxToken::LParen, StringRef(TokStart, 1));
  case ')':
    return PtxToken(PtxToken::RParen, StringRef(TokStart, 1));
  case '[':
    return PtxToken(PtxToken::LBrac, StringRef(TokStart, 1));
  case ']':
    return PtxToken(PtxToken::RBrac, StringRef(TokStart, 1));
  case '{':
    return PtxToken(PtxToken::LCurly, StringRef(TokStart, 1));
  case '}':
    return PtxToken(PtxToken::RCurly, StringRef(TokStart, 1));
  case '*':
    return PtxToken(PtxToken::Star, StringRef(TokStart, 1));
  case ',':
    return PtxToken(PtxToken::Comma, StringRef(TokStart, 1));
  case '$':
  case '.':
    return LexIdentifier();
  case '@': {
    return PtxToken(PtxToken::At, StringRef(TokStart, 1));
  }
  case '\\':
    return PtxToken(PtxToken::BackSlash, StringRef(TokStart, 1));
  case '=':
    if (*CurPtr == '=') {
      ++CurPtr;
      return PtxToken(PtxToken::EqualEqual, StringRef(TokStart, 2));
    }
    return PtxToken(PtxToken::Equal, StringRef(TokStart, 1));
  case '-':
    if (*CurPtr == '>') {
      ++CurPtr;
      return PtxToken(PtxToken::MinusGreater, StringRef(TokStart, 2));
    }
    return PtxToken(PtxToken::Minus, StringRef(TokStart, 1));
  case '|':
    if (*CurPtr == '|') {
      ++CurPtr;
      return PtxToken(PtxToken::PipePipe, StringRef(TokStart, 2));
    }
    return PtxToken(PtxToken::Pipe, StringRef(TokStart, 1));
  case '^':
    return PtxToken(PtxToken::Caret, StringRef(TokStart, 1));
  case '&':
    if (*CurPtr == '&') {
      ++CurPtr;
      return PtxToken(PtxToken::AmpAmp, StringRef(TokStart, 2));
    }
    return PtxToken(PtxToken::Amp, StringRef(TokStart, 1));
  case '!':
    if (*CurPtr == '=') {
      ++CurPtr;
      return PtxToken(PtxToken::ExclaimEqual, StringRef(TokStart, 2));
    }
    return PtxToken(PtxToken::Exclaim, StringRef(TokStart, 1));
  case '%': {
    // Check next token is NVPTX builtin identifier
    auto NextTok = peekTok();
    if (NextTok.is(PtxToken::Identifier) || NextTok.is(PtxToken::Integer))
      return LexIdentifier();
    return PtxToken(PtxToken::Percent, StringRef(TokStart, 1));
  }
  case '/':
    IsAtStartOfStatement = OldIsAtStartOfStatement;
    return LexSlash();
  case '#': {
    return PtxToken(PtxToken::Hash, StringRef(TokStart, 1));
  }
  case '"':
    return LexQuote();
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    return LexDigit();
  case '<':
    switch (*CurPtr) {
    case '<':
      ++CurPtr;
      return PtxToken(PtxToken::LessLess, StringRef(TokStart, 2));
    case '=':
      ++CurPtr;
      return PtxToken(PtxToken::LessEqual, StringRef(TokStart, 2));
    case '>':
      ++CurPtr;
      return PtxToken(PtxToken::LessGreater, StringRef(TokStart, 2));
    default:
      return PtxToken(PtxToken::Less, StringRef(TokStart, 1));
    }
  case '>':
    switch (*CurPtr) {
    case '>':
      ++CurPtr;
      return PtxToken(PtxToken::GreaterGreater, StringRef(TokStart, 2));
    case '=':
      ++CurPtr;
      return PtxToken(PtxToken::GreaterEqual, StringRef(TokStart, 2));
    default:
      return PtxToken(PtxToken::Greater, StringRef(TokStart, 1));
    }
  case '_':
    if (isAlnum(*CurPtr) || *CurPtr == '$' || *CurPtr == '%')
      return LexIdentifier();
    return PtxToken(PtxToken::Sink, StringRef(TokStart, 1));
  case '?':
    return PtxToken(PtxToken::Question, StringRef(TokStart, 1));
  }
}
