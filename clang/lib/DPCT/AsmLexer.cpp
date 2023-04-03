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

using namespace clang::dpct;

using llvm::hexDigitValue;
using llvm::isAlnum;
using llvm::isDigit;
using llvm::isHexDigit;
using llvm::SaveAndRestore;
using llvm::StringSet;

bool AsmToken::isDirective() const {
  if (isNot(Identifier))
    return false;
  static StringSet<> Dectives{"address_size",
                              "explicitcluster",
                              "maxnreg",
                              "section",
                              "alias",
                              "extern",
                              "maxntid",
                              "shared",
                              "align",
                              "file",
                              "minnctapersm",
                              "sreg",
                              "branchtargets",
                              "func",
                              "noreturn",
                              "target",
                              "callprototype",
                              "global",
                              "param",
                              "tex",
                              "calltargets",
                              "loc",
                              "pragma",
                              "version",
                              "common",
                              "local",
                              "reg",
                              "visible",
                              "const",
                              "maxclusterrank",
                              "reqnctapercluster",
                              "weak",
                              "entry",
                              "maxnctapersm",
                              "reqntid"};
  return Dectives.contains(getString());
}

bool AsmToken::isBuiltinIdentifier() const {
  if (isNot(Identifier))
    return false;
  static StringSet<> BuiltinIdentifiers{
      "cloc",        "laneid",      "lanemask_gt", "pm0",         "pm1",
      "pm2",         "pm3",         "pm4",         "pm5",         "pm7",
      "clock64",     "lanemask_eq", "nctaid",      "smid",        "ctaid",
      "lanemask_le", "ntid",        "tid",         "envreg<32>",  "lanemask_lt",
      "nsmid",       "warpid",      "gridid",      "lanemask_ge", "nwarpid",
      "WARP_SZ"};
  return BuiltinIdentifiers.contains(getString());
}

bool AsmToken::isFundamentalTypeSpecifier() const {
  if (isNot(Identifier))
    return false;
  static StringSet<> Types{"s8",  "s16", "s32", "s64",   "u8",  "u16",
                           "u32", "u64", "f16", "f16x2", "f32", "f64",
                           "b8",  "b16", "b32", "b64",   "pred"};
  return Types.contains(getString());
}

bool AsmToken::isStorageClass() const {
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

bool AsmToken::isInstructionStorageClass() const {
  return llvm::StringSwitch<bool>(getString())
      .Case(".const", true)
      .Case(".global", true)
      .Case(".local", true)
      .Case(".param", true)
      .Case(".shared", true)
      .Case(".tex", true)
      .Default(false);
}

bool AsmToken::isTypeName() const {
  static StringSet<> TypeNames{".s2",     ".s4",     ".s8",   ".s16",   ".s32",
                               ".s64",    ".u2",     ".u4",   ".u8",    ".u16",
                               ".u32",    ".u64",    ".byte", ".4byte", ".b8",
                               ".b16",    ".b32",    ".b64",  ".b128",  ".f16",
                               ".f16x2",  ".f32",    ".f64",  ".e4m3",  ".e5m2",
                               ".e4m3x2", ".e5m2x2", ".quad", ".pred"};
  return TypeNames.contains(getString());
}

bool AsmToken::isVarAttributes() const {
  return getString() == ".align" || getString() == ".attribute" ||
         isStorageClass();
}

void AsmToken::dump(raw_ostream &OS) const {
  switch (Kind) {
  case AsmToken::Error:
    OS << "error";
    break;
  case AsmToken::Identifier:
    OS << "identifier";
    break;
  case AsmToken::DotIdentifier:
    OS << "dot identifier";
    break;
  case AsmToken::Integer:
    OS << "int";
    break;
  case AsmToken::Unsigned:
    OS << "unsigned";
    break;
  case AsmToken::Float:
    OS << "float";
    break;
  case AsmToken::Double:
    OS << "double";
    break;
  case AsmToken::String:
    OS << "string";
    break;

  case AsmToken::Amp:
    OS << "Amp";
    break;
  case AsmToken::AmpAmp:
    OS << "AmpAmp";
    break;
  case AsmToken::At:
    OS << "At";
    break;
  case AsmToken::BackSlash:
    OS << "BackSlash";
    break;
  case AsmToken::Caret:
    OS << "Caret";
    break;
  case AsmToken::Colon:
    OS << "Colon";
    break;
  case AsmToken::Comma:
    OS << "Comma";
    break;
  case AsmToken::Comment:
    OS << "Comment";
    break;
  case AsmToken::Dollar:
    OS << "Dollar";
    break;
  case AsmToken::Dot:
    OS << "Dot";
    break;
  case AsmToken::EndOfStatement:
    OS << "EndOfStatement";
    break;
  case AsmToken::Eof:
    OS << "Eof";
    break;
  case AsmToken::Equal:
    OS << "Equal";
    break;
  case AsmToken::EqualEqual:
    OS << "EqualEqual";
    break;
  case AsmToken::Exclaim:
    OS << "Exclaim";
    break;
  case AsmToken::ExclaimEqual:
    OS << "ExclaimEqual";
    break;
  case AsmToken::Greater:
    OS << "Greater";
    break;
  case AsmToken::GreaterEqual:
    OS << "GreaterEqual";
    break;
  case AsmToken::GreaterGreater:
    OS << "GreaterGreater";
    break;
  case AsmToken::Hash:
    OS << "Hash";
    break;
  case AsmToken::HashDirective:
    OS << "HashDirective";
    break;
  case AsmToken::LBrac:
    OS << "LBrac";
    break;
  case AsmToken::LCurly:
    OS << "LCurly";
    break;
  case AsmToken::LParen:
    OS << "LParen";
    break;
  case AsmToken::Less:
    OS << "Less";
    break;
  case AsmToken::LessEqual:
    OS << "LessEqual";
    break;
  case AsmToken::LessGreater:
    OS << "LessGreater";
    break;
  case AsmToken::LessLess:
    OS << "LessLess";
    break;
  case AsmToken::Minus:
    OS << "Minus";
    break;
  case AsmToken::MinusGreater:
    OS << "MinusGreater";
    break;
  case AsmToken::Percent:
    OS << "Percent";
    break;
  case AsmToken::Pipe:
    OS << "Pipe";
    break;
  case AsmToken::PipePipe:
    OS << "PipePipe";
    break;
  case AsmToken::Plus:
    OS << "Plus";
    break;
  case AsmToken::RBrac:
    OS << "RBrac";
    break;
  case AsmToken::RCurly:
    OS << "RCurly";
    break;
  case AsmToken::RParen:
    OS << "RParen";
    break;
  case AsmToken::Slash:
    OS << "Slash";
    break;
  case AsmToken::Space:
    OS << "Space";
    break;
  case AsmToken::Star:
    OS << "Star";
    break;
  case AsmToken::Tilde:
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

AsmLexer::AsmLexer() {
  TokStart = nullptr;
  CurTok.emplace_back(AsmToken::Space, StringRef());
}

AsmLexer::~AsmLexer() = default;

void AsmLexer::setBuffer(StringRef Buf, const char *Ptr,
                         bool EndStatementAtEOF) {
  CurBuf = Buf;

  if (Ptr)
    CurPtr = Ptr;
  else
    CurPtr = CurBuf.begin();

  TokStart = nullptr;
  this->EndStatementAtEOF = EndStatementAtEOF;
}

const AsmToken &AsmLexer::Lex() {
  assert(!CurTok.empty());
  // Mark if we parsing out a EndOfStatement.
  IsAtStartOfStatement = CurTok.front().getKind() == AsmToken::EndOfStatement;
  CurTok.erase(CurTok.begin());
  // LexToken may generate multiple tokens via UnLex but will always return
  // the first one. Place returned value at head of CurTok vector.
  if (CurTok.empty()) {
    AsmToken T = LexToken();
    CurTok.insert(CurTok.begin(), T);
  }
  return CurTok.front();
}

void AsmLexer::UnLex(AsmToken const &Token) {
  IsAtStartOfStatement = false;
  CurTok.insert(CurTok.begin(), Token);
}

AsmToken AsmLexer::ReturnError(const char *Loc, const std::string &Msg) {
  llvm::errs() << llvm::raw_ostream::RED << Msg << llvm::raw_ostream::RESET << "\n";
  return AsmToken(AsmToken::Error, StringRef(Loc, CurPtr - Loc));
}

int AsmLexer::getNextChar() {
  if (CurPtr == CurBuf.end())
    return EOF;
  return (unsigned char)*CurPtr++;
}

int AsmLexer::peekNextChar() {
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
AsmToken AsmLexer::LexIdentifier() {
  auto Kind = AsmToken::Identifier;
  if (CurPtr[-1] == '.')
    Kind = AsmToken::DotIdentifier;
  while (isIdentifierBody(*CurPtr))
    ++CurPtr;

  // Handle . as a special case.
  if (CurPtr == TokStart + 1 && TokStart[0] == '.')
    return AsmToken(AsmToken::Dot, StringRef(TokStart, 1));
  return AsmToken(Kind, StringRef(TokStart, CurPtr - TokStart));
}

/// LexSlash: Slash: /
///           C-Style Comment: /* ... */
///           C-style Comment: // ...
AsmToken AsmLexer::LexSlash() {

  switch (*CurPtr) {
  case '*':
    IsAtStartOfStatement = false;
    break; // C style comment.
  case '/':
    ++CurPtr;
    return LexLineComment();
  default:
    IsAtStartOfStatement = false;
    return AsmToken(AsmToken::Slash, StringRef(TokStart, 1));
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
      return AsmToken(AsmToken::Comment,
                      StringRef(TokStart, CurPtr - TokStart));
    }
  }
  return ReturnError(TokStart, "unterminated comment");
}

/// LexLineComment: Comment: #[^\n]*
///                        : //[^\n]*
AsmToken AsmLexer::LexLineComment() {
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
    return AsmToken(AsmToken::EndOfStatement,
                    StringRef(TokStart, CurPtr - TokStart));
  IsAtStartOfStatement = true;

  return AsmToken(AsmToken::EndOfStatement,
                  StringRef(TokStart, CurPtr - 1 - TokStart));
}

// Look ahead to search for first non-hex digit, if it's [hH], then we treat the
// integer as a hexadecimal, possibly with leading zeroes.
static unsigned doHexLookAhead(const char *&CurPtr, unsigned DefaultRadix,
                               bool LexHex) {
  const char *FirstNonDec = nullptr;
  const char *LookAhead = CurPtr;
  while (true) {
    if (isDigit(*LookAhead)) {
      ++LookAhead;
    } else {
      if (!FirstNonDec)
        FirstNonDec = LookAhead;

      // Keep going if we are looking for a 'h' suffix.
      if (LexHex && isHexDigit(*LookAhead))
        ++LookAhead;
      else
        break;
    }
  }
  bool isHex = LexHex && (*LookAhead == 'h' || *LookAhead == 'H');
  CurPtr = isHex || !FirstNonDec ? LookAhead : FirstNonDec;
  if (isHex)
    return 16;
  return DefaultRadix;
}

static const char *findLastDigit(const char *CurPtr, unsigned DefaultRadix) {
  while (hexDigitValue(*CurPtr) < DefaultRadix) {
    ++CurPtr;
  }
  return CurPtr;
}

AsmToken AsmLexer::ConsumeIntegerSuffix(unsigned Radix) {
  if (CurPtr[0] == 'U') {
    uint64_t Result;
    if (StringRef(TokStart, CurPtr - TokStart).getAsInteger(Radix, Result))
      return ReturnError(TokStart, "invalid hexadecimal number");

    ++CurPtr; // Skip 'U' suffix
    return AsmToken(AsmToken::Unsigned, StringRef(TokStart, CurPtr - TokStart),
                    Result);
  }

  int64_t Result;
  if (StringRef(TokStart, CurPtr - TokStart).getAsInteger(Radix, Result))
    return ReturnError(TokStart, "invalid hexadecimal number");

  return AsmToken(AsmToken::Integer, StringRef(TokStart, CurPtr - TokStart),
                  Result);
}

/// 0[fF]{hexdigit}{8}      // single-precision floating point
/// 0[dD]{hexdigit}{16}     // double-precision floating point
/// hexadecimal literal:  0[xX]{hexdigit}+U?
/// octal literal:        0{octal digit}+U?
/// binary literal:       0[bB]{bit}+U?
/// decimal literal       {nonzero-digit}{digit}*U?
AsmToken AsmLexer::LexDigit() {

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
    return AsmToken(AsmToken::Float, StringRef(TokStart, CurPtr - TokStart),
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
    return AsmToken(AsmToken::Double, StringRef(TokStart, CurPtr - TokStart),
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

    return ConsumeIntegerSuffix(16);
  }

  if ((*CurPtr == 'b') || (*CurPtr == 'B')) {
    ++CurPtr;
    const char *NumStart = CurPtr;
    while (CurPtr[0] == '0' || CurPtr[0] == '1')
      ++CurPtr;

    if (CurPtr == NumStart)
      return ReturnError(CurPtr - 2, "invalid binary number");

    return ConsumeIntegerSuffix(2);
  }

  // Either octal or hexadecimal.
  return ConsumeIntegerSuffix(doHexLookAhead(CurPtr, 8, false));
}

/// LexQuote: String: "..."
AsmToken AsmLexer::LexQuote() {
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

  return AsmToken(AsmToken::String, StringRef(TokStart, CurPtr - TokStart));
}

StringRef AsmLexer::LexUntilEndOfStatement() {
  TokStart = CurPtr;

  while (!isAtStartOfComment(CurPtr) &&     // Start of line comment.
         !isAtStatementSeparator(CurPtr) && // End of statement marker.
         *CurPtr != '\n' && *CurPtr != '\r' && CurPtr != CurBuf.end()) {
    ++CurPtr;
  }
  return StringRef(TokStart, CurPtr - TokStart);
}

StringRef AsmLexer::LexUntilEndOfLine() {
  TokStart = CurPtr;

  while (*CurPtr != '\n' && *CurPtr != '\r' && CurPtr != CurBuf.end()) {
    ++CurPtr;
  }
  return StringRef(TokStart, CurPtr - TokStart);
}

size_t AsmLexer::peekTokens(MutableArrayRef<AsmToken> Buf) {
  SaveAndRestore SavedTokenStart(TokStart);
  SaveAndRestore SavedCurPtr(CurPtr);
  SaveAndRestore SavedAtStartOfLine(IsAtStartOfLine);
  SaveAndRestore SavedAtStartOfStatement(IsAtStartOfStatement);
  SaveAndRestore SavedIsPeeking(IsPeeking, true);

  size_t ReadCount;
  for (ReadCount = 0; ReadCount < Buf.size(); ++ReadCount) {
    AsmToken Token = LexToken();

    Buf[ReadCount] = Token;

    if (Token.is(AsmToken::Eof))
      break;
  }

  return ReadCount;
}

static const char *getSeparatorString() { return ";"; }

bool AsmLexer::isAtStartOfComment(const char *Ptr) {

  StringRef CommentString = "//";

  if (CommentString.size() == 1)
    return CommentString[0] == Ptr[0];

  // Allow # preprocessor comments also be counted as comments for "##" cases
  if (CommentString[1] == '#')
    return CommentString[0] == Ptr[0];

  return strncmp(Ptr, CommentString.data(), CommentString.size()) == 0;
}

bool AsmLexer::isAtStatementSeparator(const char *Ptr) {
  return strncmp(Ptr, getSeparatorString(), strlen(getSeparatorString())) == 0;
}

AsmToken AsmLexer::LexToken() {
  TokStart = CurPtr;
  // This always consumes at least one character.
  int CurChar = getNextChar();

  if (isAtStartOfComment(TokStart))
    return LexLineComment();

  if (isAtStatementSeparator(TokStart)) {
    CurPtr += strlen(getSeparatorString()) - 1;
    IsAtStartOfLine = true;
    IsAtStartOfStatement = true;
    return AsmToken(AsmToken::EndOfStatement,
                    StringRef(TokStart, strlen(getSeparatorString())));
  }

  // If we're missing a newline at EOF, make sure we still get an
  // EndOfStatement token before the Eof token.
  if (CurChar == EOF && !IsAtStartOfStatement && EndStatementAtEOF) {
    IsAtStartOfLine = true;
    IsAtStartOfStatement = true;
    return AsmToken(AsmToken::EndOfStatement, StringRef(TokStart, 0));
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
    return AsmToken(AsmToken::Eof, StringRef(TokStart, 0));
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
    return AsmToken(AsmToken::EndOfStatement,
                    StringRef(TokStart, CurPtr - TokStart));
  }
  case '\n':
    IsAtStartOfLine = true;
    IsAtStartOfStatement = true;
    return AsmToken(AsmToken::EndOfStatement, StringRef(TokStart, 1));
  case ':':
    return AsmToken(AsmToken::Colon, StringRef(TokStart, 1));
  case '+':
    return AsmToken(AsmToken::Plus, StringRef(TokStart, 1));
  case '~':
    return AsmToken(AsmToken::Tilde, StringRef(TokStart, 1));
  case '(':
    return AsmToken(AsmToken::LParen, StringRef(TokStart, 1));
  case ')':
    return AsmToken(AsmToken::RParen, StringRef(TokStart, 1));
  case '[':
    return AsmToken(AsmToken::LBrac, StringRef(TokStart, 1));
  case ']':
    return AsmToken(AsmToken::RBrac, StringRef(TokStart, 1));
  case '{':
    return AsmToken(AsmToken::LCurly, StringRef(TokStart, 1));
  case '}':
    return AsmToken(AsmToken::RCurly, StringRef(TokStart, 1));
  case '*':
    return AsmToken(AsmToken::Star, StringRef(TokStart, 1));
  case ',':
    return AsmToken(AsmToken::Comma, StringRef(TokStart, 1));
  case '$':
  case '.':
    return LexIdentifier();
  case '@': {
    return AsmToken(AsmToken::At, StringRef(TokStart, 1));
  }
  case '\\':
    return AsmToken(AsmToken::BackSlash, StringRef(TokStart, 1));
  case '=':
    if (*CurPtr == '=') {
      ++CurPtr;
      return AsmToken(AsmToken::EqualEqual, StringRef(TokStart, 2));
    }
    return AsmToken(AsmToken::Equal, StringRef(TokStart, 1));
  case '-':
    if (*CurPtr == '>') {
      ++CurPtr;
      return AsmToken(AsmToken::MinusGreater, StringRef(TokStart, 2));
    }
    return AsmToken(AsmToken::Minus, StringRef(TokStart, 1));
  case '|':
    if (*CurPtr == '|') {
      ++CurPtr;
      return AsmToken(AsmToken::PipePipe, StringRef(TokStart, 2));
    }
    return AsmToken(AsmToken::Pipe, StringRef(TokStart, 1));
  case '^':
    return AsmToken(AsmToken::Caret, StringRef(TokStart, 1));
  case '&':
    if (*CurPtr == '&') {
      ++CurPtr;
      return AsmToken(AsmToken::AmpAmp, StringRef(TokStart, 2));
    }
    return AsmToken(AsmToken::Amp, StringRef(TokStart, 1));
  case '!':
    if (*CurPtr == '=') {
      ++CurPtr;
      return AsmToken(AsmToken::ExclaimEqual, StringRef(TokStart, 2));
    }
    return AsmToken(AsmToken::Exclaim, StringRef(TokStart, 1));
  case '%': {
    // Check next token is NVPTX builtin identifier
    auto NextTok = peekTok();
    if (NextTok.getString() != "WARP_SZ" && NextTok.isBuiltinIdentifier())
      return LexIdentifier();
    return AsmToken(AsmToken::Percent, StringRef(TokStart, 1));
  }
  case '/':
    IsAtStartOfStatement = OldIsAtStartOfStatement;
    return LexSlash();
  case '#': {
    return AsmToken(AsmToken::Hash, StringRef(TokStart, 1));
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
      return AsmToken(AsmToken::LessLess, StringRef(TokStart, 2));
    case '=':
      ++CurPtr;
      return AsmToken(AsmToken::LessEqual, StringRef(TokStart, 2));
    case '>':
      ++CurPtr;
      return AsmToken(AsmToken::LessGreater, StringRef(TokStart, 2));
    default:
      return AsmToken(AsmToken::Less, StringRef(TokStart, 1));
    }
  case '>':
    switch (*CurPtr) {
    case '>':
      ++CurPtr;
      return AsmToken(AsmToken::GreaterGreater, StringRef(TokStart, 2));
    case '=':
      ++CurPtr;
      return AsmToken(AsmToken::GreaterEqual, StringRef(TokStart, 2));
    default:
      return AsmToken(AsmToken::Greater, StringRef(TokStart, 1));
    }
  case '_':
    if (isAlnum(*CurPtr) || *CurPtr == '$' || *CurPtr == '%')
      return LexIdentifier();
    return AsmToken(AsmToken::Sink, StringRef(TokStart, 1));
  case '?':
    return AsmToken(AsmToken::Question, StringRef(TokStart, 1));

    // TODO: Quoted identifiers (objc methods etc)
    // local labels: [0-9][:]
    // Forward/backward labels: [0-9][fb]
    // Integers, fp constants, character constants.
  }
}
