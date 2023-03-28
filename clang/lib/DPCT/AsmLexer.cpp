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

using llvm::isDigit;
using llvm::isHexDigit;
using llvm::isAlnum;
using llvm::hexDigitValue;
using llvm::SaveAndRestore;
using llvm::StringSet;

bool AsmToken::isDirective() const {
  if (isNot(Identifier))
    return false;
  static StringSet<> Dectives{
    "address_size",   "explicitcluster",  "maxnreg",            "section",
    "alias",          "extern",           "maxntid",            "shared",
    "align",          "file",             "minnctapersm",       "sreg",
    "branchtargets",  "func",             "noreturn",           "target",
    "callprototype",  "global",           "param",              "tex",
    "calltargets",    "loc",              "pragma",             "version",
    "common",         "local",            "reg",                "visible",
    "const",          "maxclusterrank",   "reqnctapercluster",  "weak",
    "entry",          "maxnctapersm",     "reqntid"
  };
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
  static StringSet<> Types{
    "s8",   "s16",   "s32", "s64",
    "u8",   "u16",   "u32", "u64",
    "f16",  "f16x2", "f32", "f64",
    "b8",   "b16",   "b32", "b64",
    "pred"
  };
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
  static StringSet<> TypeNames{
    ".s2", ".s4", ".s8", ".s16", ".s32", ".s64",
    ".u2", ".u4", ".u8", ".u16", ".u32", ".u64",
    ".byte", ".4byte", ".b8", ".b16", ".b32", ".b64", ".b128",
    ".f16", ".f16x2", ".f32", ".f64", ".e4m3", ".e5m2", ".e4m3x2", ".e5m2x2",
    ".quad", ".pred"
  };
  return TypeNames.contains(getString());
}

bool AsmToken::isVarAttributes() const {
  return getString() == ".align" || getString() == ".attribute" || isStorageClass();
}

void AsmToken::dump(raw_ostream &OS) const {
  switch (Kind) {
  case AsmToken::Error:
    OS << "error";
    break;
  case AsmToken::Identifier:
    OS << "identifier: " << getString();
    break;
  case AsmToken::Integer:
    OS << "int: " << getString();
    break;
  case AsmToken::Real:
    OS << "real: " << getString();
    break;
  case AsmToken::String:
    OS << "string: " << getString();
    break;

  case AsmToken::Amp:                OS << "Amp"; break;
  case AsmToken::AmpAmp:             OS << "AmpAmp"; break;
  case AsmToken::At:                 OS << "At"; break;
  case AsmToken::BackSlash:          OS << "BackSlash"; break;
  case AsmToken::BigNum:             OS << "BigNum"; break;
  case AsmToken::Caret:              OS << "Caret"; break;
  case AsmToken::Colon:              OS << "Colon"; break;
  case AsmToken::Comma:              OS << "Comma"; break;
  case AsmToken::Comment:            OS << "Comment"; break;
  case AsmToken::Dollar:             OS << "Dollar"; break;
  case AsmToken::Dot:                OS << "Dot"; break;
  case AsmToken::EndOfStatement:     OS << "EndOfStatement"; break;
  case AsmToken::Eof:                OS << "Eof"; break;
  case AsmToken::Equal:              OS << "Equal"; break;
  case AsmToken::EqualEqual:         OS << "EqualEqual"; break;
  case AsmToken::Exclaim:            OS << "Exclaim"; break;
  case AsmToken::ExclaimEqual:       OS << "ExclaimEqual"; break;
  case AsmToken::Greater:            OS << "Greater"; break;
  case AsmToken::GreaterEqual:       OS << "GreaterEqual"; break;
  case AsmToken::GreaterGreater:     OS << "GreaterGreater"; break;
  case AsmToken::Hash:               OS << "Hash"; break;
  case AsmToken::HashDirective:      OS << "HashDirective"; break;
  case AsmToken::LBrac:              OS << "LBrac"; break;
  case AsmToken::LCurly:             OS << "LCurly"; break;
  case AsmToken::LParen:             OS << "LParen"; break;
  case AsmToken::Less:               OS << "Less"; break;
  case AsmToken::LessEqual:          OS << "LessEqual"; break;
  case AsmToken::LessGreater:        OS << "LessGreater"; break;
  case AsmToken::LessLess:           OS << "LessLess"; break;
  case AsmToken::Minus:              OS << "Minus"; break;
  case AsmToken::MinusGreater:       OS << "MinusGreater"; break;
  case AsmToken::Percent:            OS << "Percent"; break;
  case AsmToken::Pipe:               OS << "Pipe"; break;
  case AsmToken::PipePipe:           OS << "PipePipe"; break;
  case AsmToken::Plus:               OS << "Plus"; break;
  case AsmToken::RBrac:              OS << "RBrac"; break;
  case AsmToken::RCurly:             OS << "RCurly"; break;
  case AsmToken::RParen:             OS << "RParen"; break;
  case AsmToken::Slash:              OS << "Slash"; break;
  case AsmToken::Space:              OS << "Space"; break;
  case AsmToken::Star:               OS << "Star"; break;
  case AsmToken::Tilde:              OS << "Tilde"; break;
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

/// The leading integral digit sequence and dot should have already been
/// consumed, some or all of the fractional digit sequence *can* have been
/// consumed.
AsmToken AsmLexer::LexFloatLiteral() {
  // Skip the fractional digit sequence.
  while (isDigit(*CurPtr))
    ++CurPtr;

  if (*CurPtr == '-' || *CurPtr == '+')
    return ReturnError(CurPtr, "invalid sign in float literal");

  // Check for exponent
  if ((*CurPtr == 'e' || *CurPtr == 'E')) {
    ++CurPtr;

    if (*CurPtr == '-' || *CurPtr == '+')
      ++CurPtr;

    while (isDigit(*CurPtr))
      ++CurPtr;
  }

  return AsmToken(AsmToken::Real,
                  StringRef(TokStart, CurPtr - TokStart));
}

/// LexHexFloatLiteral matches essentially (.[0-9a-fA-F]*)?[pP][+-]?[0-9a-fA-F]+
/// while making sure there are enough actual digits around for the constant to
/// be valid.
///
/// The leading "0x[0-9a-fA-F]*" (i.e. integer part) has already been consumed
/// before we get here.
AsmToken AsmLexer::LexHexFloatLiteral(bool NoIntDigits) {
  assert((*CurPtr == 'p' || *CurPtr == 'P' || *CurPtr == '.') &&
         "unexpected parse state in floating hex");
  bool NoFracDigits = true;

  // Skip the fractional part if there is one
  if (*CurPtr == '.') {
    ++CurPtr;

    const char *FracStart = CurPtr;
    while (isHexDigit(*CurPtr))
      ++CurPtr;

    NoFracDigits = CurPtr == FracStart;
  }

  if (NoIntDigits && NoFracDigits)
    return ReturnError(TokStart, "invalid hexadecimal floating-point constant: "
                                 "expected at least one significand digit");

  // Make sure we do have some kind of proper exponent part
  if (*CurPtr != 'p' && *CurPtr != 'P')
    return ReturnError(TokStart, "invalid hexadecimal floating-point constant: "
                                 "expected exponent part 'p'");
  ++CurPtr;

  if (*CurPtr == '+' || *CurPtr == '-')
    ++CurPtr;

  // N.b. exponent digits are *not* hex
  const char *ExpStart = CurPtr;
  while (isDigit(*CurPtr))
    ++CurPtr;

  if (CurPtr == ExpStart)
    return ReturnError(TokStart, "invalid hexadecimal floating-point constant: "
                                 "expected at least one exponent digit");

  return AsmToken(AsmToken::Real, StringRef(TokStart, CurPtr - TokStart));
}

static bool isDirectiveStart(char C) {
  return C == '.';
}

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
  // Check for floating point literals.
  if (CurPtr[-1] == '.' && isDigit(*CurPtr))
      return LexFloatLiteral();

  while (isIdentifierBody(*CurPtr))
    ++CurPtr;

  // Handle . as a special case.
  if (CurPtr == TokStart+1 && TokStart[0] == '.')
    return AsmToken(AsmToken::Dot, StringRef(TokStart, 1));
  return AsmToken(AsmToken::Identifier, StringRef(TokStart, CurPtr - TokStart));
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
  ++CurPtr;  // skip the star.
  while (CurPtr != CurBuf.end()) {
    switch (*CurPtr++) {
    case '*':
      // End of the comment?
      if (*CurPtr != '/')
        break;
      ++CurPtr;   // End the */.
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

static void SkipIgnoredIntegerSuffix(const char *&CurPtr) {
  // Skip case-insensitive ULL, UL, U, L and LL suffixes.
  if (CurPtr[0] == 'U' || CurPtr[0] == 'u')
    ++CurPtr;
  if (CurPtr[0] == 'L' || CurPtr[0] == 'l')
    ++CurPtr;
  if (CurPtr[0] == 'L' || CurPtr[0] == 'l')
    ++CurPtr;
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

static AsmToken intToken(StringRef Ref, APInt &Value) {
  if (Value.isIntN(64))
    return AsmToken(AsmToken::Integer, Ref, Value);
  return AsmToken(AsmToken::BigNum, Ref, Value);
}

static std::string radixName(unsigned Radix) {
  switch (Radix) {
  case 2:
    return "binary";
  case 8:
    return "octal";
  case 10:
    return "decimal";
  case 16:
    return "hexadecimal";
  default:
    return "base-" + std::to_string(Radix);
  }
}

/// LexDigit: First character is [0-9].
///   Local Label: [0-9][:]
///   Forward/Backward Label: [0-9][fb]
///   Binary integer: 0b[01]+
///   Octal integer: 0[0-7]+
///   Hex integer: 0x[0-9a-fA-F]+ or [0x]?[0-9][0-9a-fA-F]*[hH]
///   Decimal integer: [1-9][0-9]*
AsmToken AsmLexer::LexDigit() {
  if ((*CurPtr == 'x') || (*CurPtr == 'X')) {
    ++CurPtr;
    const char *NumStart = CurPtr;
    while (isHexDigit(CurPtr[0]))
      ++CurPtr;

    // "0x.0p0" is valid, and "0x0p0" (but not "0xp0" for example, which will be
    // diagnosed by LexHexFloatLiteral).
    if (CurPtr[0] == '.' || CurPtr[0] == 'p' || CurPtr[0] == 'P')
      return LexHexFloatLiteral(NumStart == CurPtr);

    // Otherwise requires at least one hex digit.
    if (CurPtr == NumStart)
      return ReturnError(CurPtr-2, "invalid hexadecimal number");

    APInt Result(128, 0);
    if (StringRef(TokStart, CurPtr - TokStart).getAsInteger(0, Result))
      return ReturnError(TokStart, "invalid hexadecimal number");

    // The darwin/x86 (and x86-64) assembler accepts and ignores ULL and LL
    // suffixes on integer literals.
    SkipIgnoredIntegerSuffix(CurPtr);

    return intToken(StringRef(TokStart, CurPtr - TokStart), Result);
  }

  // Either octal or hexadecimal.
  APInt Value(128, 0, true);
  unsigned Radix = doHexLookAhead(CurPtr, 8, false);
  StringRef Result(TokStart, CurPtr - TokStart);
  if (Result.getAsInteger(Radix, Value))
    return ReturnError(TokStart, "invalid " + radixName(Radix) + " number");

  // Consume the [hH].
  if (Radix == 16)
    ++CurPtr;

  // The darwin/x86 (and x86-64) assembler accepts and ignores ULL and LL
  // suffixes on integer literals.
  SkipIgnoredIntegerSuffix(CurPtr);

  return intToken(Result, Value);
}

/// LexSingleQuote: Integer: 'b'
AsmToken AsmLexer::LexSingleQuote() {
  int CurChar = getNextChar();

  if (CurChar == '\\')
    CurChar = getNextChar();

  if (CurChar == EOF)
    return ReturnError(TokStart, "unterminated single quote");

  CurChar = getNextChar();

  if (CurChar != '\'')
    return ReturnError(TokStart, "single quote way too long");

  // The idea here being that 'c' is basically just an integral
  // constant.
  StringRef Res = StringRef(TokStart,CurPtr - TokStart);
  long long Value;

  if (Res.startswith("\'\\")) {
    char theChar = Res[2];
    switch (theChar) {
      default: Value = theChar; break;
      case '\'': Value = '\''; break;
      case 't': Value = '\t'; break;
      case 'n': Value = '\n'; break;
      case 'b': Value = '\b'; break;
      case 'f': Value = '\f'; break;
      case 'r': Value = '\r'; break;
    }
  } else
    Value = TokStart[1];

  return AsmToken(AsmToken::Integer, Res, Value);
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
  return StringRef(TokStart, CurPtr-TokStart);
}

StringRef AsmLexer::LexUntilEndOfLine() {
  TokStart = CurPtr;

  while (*CurPtr != '\n' && *CurPtr != '\r' && CurPtr != CurBuf.end()) {
    ++CurPtr;
  }
  return StringRef(TokStart, CurPtr-TokStart);
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

static const char *getSeparatorString() {
  return ";";
}

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
  return strncmp(Ptr,getSeparatorString(),
                 strlen(getSeparatorString())) == 0;
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
    if (isalpha(CurChar) || CurChar == '_' || CurChar == '%')
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
  case ':': return AsmToken(AsmToken::Colon, StringRef(TokStart, 1));
  case '+': return AsmToken(AsmToken::Plus, StringRef(TokStart, 1));
  case '~': return AsmToken(AsmToken::Tilde, StringRef(TokStart, 1));
  case '(': return AsmToken(AsmToken::LParen, StringRef(TokStart, 1));
  case ')': return AsmToken(AsmToken::RParen, StringRef(TokStart, 1));
  case '[': return AsmToken(AsmToken::LBrac, StringRef(TokStart, 1));
  case ']': return AsmToken(AsmToken::RBrac, StringRef(TokStart, 1));
  case '{': return AsmToken(AsmToken::LCurly, StringRef(TokStart, 1));
  case '}': return AsmToken(AsmToken::RCurly, StringRef(TokStart, 1));
  case '*': return AsmToken(AsmToken::Star, StringRef(TokStart, 1));
  case ',': return AsmToken(AsmToken::Comma, StringRef(TokStart, 1));
  case '$':
  case '.':
    return LexIdentifier();
  case '@': {
    return AsmToken(AsmToken::At, StringRef(TokStart, 1));
  }
  case '\\': return AsmToken(AsmToken::BackSlash, StringRef(TokStart, 1));
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
  case '^': return AsmToken(AsmToken::Caret, StringRef(TokStart, 1));
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
  case '\'': return LexSingleQuote();
  case '"': return LexQuote();
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
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

  // TODO: Quoted identifiers (objc methods etc)
  // local labels: [0-9][:]
  // Forward/backward labels: [0-9][fb]
  // Integers, fp constants, character constants.
  }
}

