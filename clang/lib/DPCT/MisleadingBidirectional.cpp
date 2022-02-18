//===--- MisleadingBidirectional.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Diagnostics.h"
#include "MisleadingBidirectional.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"

using namespace clang;
using namespace clang::dpct;
using namespace clang::ast_matchers;

static bool containsMisleadingBidi(StringRef Buffer,
                                   bool HonorLineBreaks = true) {
  const char *CurPtr = Buffer.begin();

  enum BidiChar {
    PS = 0x2029,
    RLO = 0x202E,
    RLE = 0x202B,
    LRO = 0x202D,
    LRE = 0x202A,
    PDF = 0x202C,
    RLI = 0x2067,
    LRI = 0x2066,
    FSI = 0x2068,
    PDI = 0x2069
  };

  SmallVector<BidiChar> BidiContexts;

  // Scan each character while maintaining a stack of opened bidi context.
  // RLO/RLE/LRO/LRE all are closed by PDF while RLI LRI and FSI are closed by
  // PDI. New lines reset the context count. Extra PDF / PDI are ignored.
  //
  // Warn if we end up with an unclosed context.
  while (CurPtr < Buffer.end()) {
    unsigned char C = *CurPtr;
    if (isASCII(C)) {
      ++CurPtr;
      bool IsParagrapSep =
          (C == 0xA || C == 0xD || (0x1C <= C && C <= 0x1E) || C == 0x85);
      bool IsSegmentSep = (C == 0x9 || C == 0xB || C == 0x1F);
      if (IsParagrapSep || IsSegmentSep)
        BidiContexts.clear();
      continue;
    }
    llvm::UTF32 CodePoint;
    llvm::ConversionResult Result = llvm::convertUTF8Sequence(
        (const llvm::UTF8 **)&CurPtr, (const llvm::UTF8 *)Buffer.end(),
        &CodePoint, llvm::strictConversion);

    // If conversion fails, utf-8 is designed so that we can just try next char.
    if (Result != llvm::conversionOK) {
      ++CurPtr;
      continue;
    }

    // Open a PDF context.
    if (CodePoint == RLO || CodePoint == RLE || CodePoint == LRO ||
        CodePoint == LRE)
      BidiContexts.push_back(PDF);
    // Close PDF Context.
    else if (CodePoint == PDF) {
      if (!BidiContexts.empty() && BidiContexts.back() == PDF)
        BidiContexts.pop_back();
    }
    // Open a PDI Context.
    else if (CodePoint == RLI || CodePoint == LRI || CodePoint == FSI)
      BidiContexts.push_back(PDI);
    // Close a PDI Context.
    else if (CodePoint == PDI) {
      auto R = std::find(BidiContexts.rbegin(), BidiContexts.rend(), PDI);
      if (R != BidiContexts.rend())
        BidiContexts.resize(BidiContexts.rend() - R - 1);
    }
    // Line break or equivalent
    else if (CodePoint == PS)
      BidiContexts.clear();
  }
  return !BidiContexts.empty();
}

bool MisleadingBidirectionalHandler::HandleComment(Preprocessor &PP,
                                                   SourceRange Range) {
  StringRef Text =
      Lexer::getSourceText(CharSourceRange::getCharRange(Range),
                           PP.getSourceManager(), PP.getLangOpts());

  if (containsMisleadingBidi(Text, true)) {
    DiagnosticsUtils::report(
        Range.getBegin(), Diagnostics::MISLEADING_BIDIRECTIONAL,
        dpct::DpctGlobalInfo::getCompilerInstance(), &TransformSet, false);
  }
  return false;
}

void MisleadingBidirectionalRule::registerMatcher(
    ast_matchers::MatchFinder &MF) {
  if (DpctGlobalInfo::getCheckUnicodeSecurityFlag()) {
    MF.addMatcher(stringLiteral().bind("strlit"), this);
  }
}

void MisleadingBidirectionalRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const StringLiteral *SL =
          getNodeAsType<StringLiteral>(Result, "strlit")) {
    StringRef Literal = SL->getBytes();
    if (containsMisleadingBidi(Literal, false)) {
      report(SL->getBeginLoc(), Diagnostics::MISLEADING_BIDIRECTIONAL, false);
    }
  }
}
