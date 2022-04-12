//===--- Homoglyph.cpp - clang-tidy----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "Diagnostics.h"
#include "Homoglyph.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"

namespace {
// Preprocessed version of
// https://www.unicode.org/Public/security/latest/confusables.txt
//
// This contains a sorted array of { UTF32 codepoint; UTF32 values[N];}
#include "Confusables.inc"
} // namespace

using namespace clang;
using namespace clang::c2s;
using namespace clang::ast_matchers;

std::string ConfusableIdentifierDetectionRule::skeleton(StringRef Name) {
  std::string SName = Name.str();
  std::string Skeleton;
  Skeleton.reserve(1 + Name.size());

  char const *Curr = SName.c_str();
  char const *End = Curr + SName.size();
  while (Curr < End) {

    char const *Prev = Curr;
    llvm::UTF32 CodePoint;
    llvm::ConversionResult Result = llvm::convertUTF8Sequence(
        (const llvm::UTF8 **)&Curr, (const llvm::UTF8 *)End, &CodePoint,
        llvm::strictConversion);
    if (Result != llvm::conversionOK) {
      llvm::errs() << "Unicode conversion issue\n";
      break;
    }

    StringRef Key(Prev, Curr - Prev);
    auto Where = std::lower_bound(
        std::begin(ConfusableEntries), std::end(ConfusableEntries), CodePoint,
        [](decltype(ConfusableEntries[0]) x, llvm::UTF32 y) {
          return x.codepoint < y;
        });
    if (Where == std::end(ConfusableEntries) || CodePoint != Where->codepoint) {
      Skeleton.append(Prev, Curr);
    } else {
      llvm::UTF8 Buffer[32];
      llvm::UTF8 *BufferStart = std::begin(Buffer);
      llvm::UTF8 *IBuffer = BufferStart;
      const llvm::UTF32 *ValuesStart = std::begin(Where->values);
      const llvm::UTF32 *ValuesEnd =
          std::find(std::begin(Where->values), std::end(Where->values), '\0');
      if (llvm::ConvertUTF32toUTF8(&ValuesStart, ValuesEnd, &IBuffer,
                                   std::end(Buffer), llvm::strictConversion) !=
          llvm::conversionOK) {
        llvm::errs() << "Unicode conversion issue\n";
        break;
      }
      Skeleton.append((char *)BufferStart, (char *)IBuffer);
    }
  }
  return Skeleton;
}

void ConfusableIdentifierDetectionRule::registerMatcher(
    ast_matchers::MatchFinder &MF) {
  if (C2SGlobalInfo::getCheckUnicodeSecurityFlag()) {
    MF.addMatcher(namedDecl().bind("confusabledecl"), this);
  }
}

void ConfusableIdentifierDetectionRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const NamedDecl *ND =
          getNodeAsType<NamedDecl>(Result, "confusabledecl")) {
    StringRef NDName = ND->getName();
    auto &Mapped = Mapper[skeleton(NDName)];
    auto *NDDecl = ND->getDeclContext();
    for (auto *OND : Mapped) {
      if (!NDDecl->isDeclInLexicalTraversal(OND) &&
          !OND->getDeclContext()->isDeclInLexicalTraversal(ND))
        continue;
      if (OND->getName() != NDName) {
        report(OND->getBeginLoc(), Diagnostics::CONFUSABLE_IDENTIFIER, false,
               OND->getName(), NDName);
        report(ND->getBeginLoc(), Diagnostics::CONFUSABLE_IDENTIFIER, false,
               NDName, OND->getName());
      }
    }
    Mapped.push_back(ND);
  }
}
