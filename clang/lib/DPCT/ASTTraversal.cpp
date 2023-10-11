//===--------------- ASTTraversal.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "AsmMigration.h"
#include "BarrierFenceSpaceAnalyzer.h"
#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"
#include "DNNAPIMigration.h"
#include "ExprAnalysis.h"
#include "FFTAPIMigration.h"
#include "GroupFunctionAnalyzer.h"
#include "Homoglyph.h"
#include "LIBCUAPIMigration.h"
#include "MemberExprRewriter.h"
#include "MigrationRuleManager.h"
#include "MisleadingBidirectional.h"
#include "NCCLAPIMigration.h"
#include "OptimizeMigration.h"
#include "SaveNewFiles.h"
#include "TextModification.h"
#include "ThrustAPIMigration.h"
#include "Utility.h"
#include "WMMAAPIMigration.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/Cuda.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::dpct;
using namespace clang::tooling;

extern std::string CudaPath;
extern std::string DpctInstallPath; // Installation directory for this tool
extern llvm::cl::opt<UsmLevel> USMLevel;
extern bool ProcessAllFlag;
extern bool ExplicitClNamespace;

TextModification *clang::dpct::replaceText(SourceLocation Begin, SourceLocation End,
                              std::string &&Str, const SourceManager &SM) {
  auto Length = SM.getFileOffset(End) - SM.getFileOffset(Begin);
  if (Length > 0) {
    return new ReplaceText(Begin, Length, std::move(Str));
  }
  return nullptr;
}

SourceLocation getArgEndLocation(const CallExpr *C, unsigned Idx,
                                 const SourceManager &SM) {
  auto SL = getStmtExpansionSourceRange(C->getArg(Idx)).getEnd();
  return SL.getLocWithOffset(Lexer::MeasureTokenLength(
      SL, SM, DpctGlobalInfo::getContext().getLangOpts()));
}

/// Return a TextModication that removes nth argument of the CallExpr,
/// together with the preceding comma.
TextModification *clang::dpct::removeArg(const CallExpr *C, unsigned n,
                            const SourceManager &SM) {
  if (C->getNumArgs() <= n)
    return nullptr;
  if (C->getArg(n)->isDefaultArgument())
    return nullptr;

  SourceLocation Begin, End;
  if (n) {
    Begin = getArgEndLocation(C, n - 1, SM);
    End = getArgEndLocation(C, n, SM);
  } else {
    Begin = getStmtExpansionSourceRange(C->getArg(n)).getBegin();
    if (C->getNumArgs() > 1) {
      End = getStmtExpansionSourceRange(C->getArg(n + 1)).getBegin();
    } else {
      End = getArgEndLocation(C, n, SM);
    }
  }
  return replaceText(Begin, End, "", SM);
}

auto parentStmt = []() {
  return anyOf(
      hasParent(compoundStmt()), hasParent(forStmt()), hasParent(whileStmt()),
      hasParent(doStmt()), hasParent(ifStmt()),
      hasParent(exprWithCleanups(anyOf(
          hasParent(compoundStmt()), hasParent(forStmt()),
          hasParent(whileStmt()), hasParent(doStmt()), hasParent(ifStmt())))));
};

static const CXXConstructorDecl *getIfConstructorDecl(const Decl *ND) {
  if (const auto *Tmpl = dyn_cast<FunctionTemplateDecl>(ND))
    ND = Tmpl->getTemplatedDecl();
  return dyn_cast<CXXConstructorDecl>(ND);
}

static internal::Matcher<NamedDecl> vectorTypeName() {
  std::vector<std::string> TypeNames(MapNames::SupportedVectorTypes.begin(),
                                     MapNames::SupportedVectorTypes.end());
  return internal::Matcher<NamedDecl>(new internal::HasNameMatcher(TypeNames));
}

unsigned MigrationRule::PairID = 0;
bool IncludesCallbacks::isInAnalysisScope(SourceLocation Loc) {
  return DpctGlobalInfo::isInAnalysisScope(Loc);
}

int IncludesCallbacks::findPoundSign(SourceLocation DirectiveStart) {
  std::pair<FileID, unsigned> LocInfo =
      SM.getDecomposedSpellingLoc(DirectiveStart);

  bool CharDataInvalid = false;
  auto Entry = SM.getSLocEntry(LocInfo.first, &CharDataInvalid);
  if (CharDataInvalid || !Entry.isFile()) {
    return -1;
  }
  std::optional<llvm::MemoryBufferRef> Buffer =
      Entry.getFile().getContentCache().getBufferOrNone(
          SM.getDiagnostics(), SM.getFileManager(), SourceLocation());
  if (!Buffer.has_value())
    return -1;
  const char *BufferStart = Buffer->getBufferStart();
  const char *Pos = BufferStart + LocInfo.second - 1;
  while (Pos >= BufferStart) {
    if (*Pos == '#')
      return BufferStart + LocInfo.second - Pos;
    if (*Pos != ' ' && *Pos != '\t')
      return -1;
    Pos--;
  }
  return -1;
}

void IncludesCallbacks::insertCudaArchRepl(
    std::shared_ptr<clang::dpct::ExtReplacement> Repl) {
  auto FilePath = Repl->getFilePath().str();
  auto Offset = Repl->getOffset();
  auto &Map = DpctGlobalInfo::getInstance().getCudaArchMacroReplMap();
  std::string Key = FilePath + std::to_string(Offset);
  Map.insert({Key, Repl});
  return;
}

bool IncludesCallbacks::ReplaceCuMacro(const Token &MacroNameTok) {
  bool IsInAnalysisScope = isInAnalysisScope(MacroNameTok.getLocation());
  if (!IsInAnalysisScope) {
    return false;
  }
  if (!MacroNameTok.getIdentifierInfo()) {
    return false;
  }
  std::string MacroName = MacroNameTok.getIdentifierInfo()->getName().str();
  auto Iter = MapNames::MacrosMap.find(MacroName);
  if (Iter != MapNames::MacrosMap.end()) {
    std::string ReplacedMacroName = Iter->second;
    auto Repl = std::make_shared<ReplaceToken>(MacroNameTok.getLocation(),
                                               std::move(ReplacedMacroName));
    if (MacroName == "__CUDA_ARCH__") {
      if (DpctGlobalInfo::getInstance().getContext().getLangOpts().CUDA) {
        requestFeature(HelperFeatureEnum::device_ext);
        insertCudaArchRepl(Repl->getReplacement(DpctGlobalInfo::getContext()));
        return true;
      }
      return false;
    }
    if (MacroName == "__CUDACC__" &&
        !MacroNameTok.getIdentifierInfo()->hasMacroDefinition())
      return false;
    if (MacroName == "CUDART_VERSION" || MacroName == "__CUDART_API_VERSION") {
      auto LocInfo = DpctGlobalInfo::getLocInfo(MacroNameTok.getLocation());
      DpctGlobalInfo::getInstance()
          .insertFile(LocInfo.first)
          ->setRTVersionValue(
              clang::CudaVersionToMacroDefStr(DpctGlobalInfo::getSDKVersion()));
    }
    TransformSet.emplace_back(Repl);
    return true;
  }
  return false;
}

void IncludesCallbacks::MacroDefined(const Token &MacroNameTok,
                                     const MacroDirective *MD) {
  bool IsInAnalysisScope = isInAnalysisScope(MacroNameTok.getLocation());

  size_t i;
  // Record all macro define locations
  auto MI = MD->getMacroInfo();
  if (!MI) {
    return;
  }
  for (i = 0; i < MI->getNumTokens(); i++) {
    std::shared_ptr<dpct::DpctGlobalInfo::MacroDefRecord> R =
        std::make_shared<dpct::DpctGlobalInfo::MacroDefRecord>(
            MacroNameTok.getLocation(), IsInAnalysisScope);
    dpct::DpctGlobalInfo::getMacroTokenToMacroDefineLoc()[getHashStrFromLoc(
        MI->getReplacementToken(i).getLocation())] = R;
  }

  if (!IsInAnalysisScope) {
    return;
  }

  for (auto Iter = MI->tokens_begin(); Iter != MI->tokens_end(); ++Iter) {
    auto II = Iter->getIdentifierInfo();
    if (!II)
      continue;

    if (MapNames::MacrosMap.find(II->getName().str()) !=
        MapNames::MacrosMap.end()) {
      std::string ReplacedMacroName =
          MapNames::MacrosMap.at(II->getName().str());
      TransformSet.emplace_back(
          new ReplaceToken(Iter->getLocation(), std::move(ReplacedMacroName)));
      if (II->getName().str() == "__CUDA_ARCH__" ||
          II->getName().str() == "__NVCC__") {
        requestFeature(HelperFeatureEnum::device_ext);
      }
    }

    if (II->hasMacroDefinition() && (II->getName().str() == "__host__" ||
                                     II->getName().str() == "__device__" ||
                                     II->getName().str() == "__global__" ||
                                     II->getName().str() == "__constant__" ||
                                     II->getName().str() == "__shared__")) {
      TransformSet.emplace_back(removeMacroInvocationAndTrailingSpaces(
          SourceRange(Iter->getLocation(), Iter->getEndLoc())));
    } else if (II->hasMacroDefinition() && II->getName().str() == "CUDART_CB") {
#ifdef _WIN32
      TransformSet.emplace_back(
          new ReplaceText(Iter->getLocation(), 9, "__stdcall"));
#else
      TransformSet.emplace_back(removeMacroInvocationAndTrailingSpaces(
          SourceRange(Iter->getLocation(), Iter->getEndLoc())));
#endif
    }

    if (MapNames::AtomicFuncNamesMap.find(II->getName().str()) !=
        MapNames::AtomicFuncNamesMap.end()) {
      std::string HashStr =
          getHashStrFromLoc(MI->getReplacementToken(0).getLocation());
      DpctGlobalInfo::getInstance().insertAtomicInfo(
          HashStr, MacroNameTok.getLocation(), II->getName().str());
    } else if (MacroNameTok.getLocation().isValid() &&
               MacroNameTok.getIdentifierInfo() &&
               MapNames::VectorTypeMigratedTypeSizeMap.find(
                   MacroNameTok.getIdentifierInfo()->getName().str()) !=
                   MapNames::VectorTypeMigratedTypeSizeMap.end()) {
      DiagnosticsUtils::report(
          MacroNameTok.getLocation(), Diagnostics::MACRO_SAME_AS_SYCL_TYPE,
          &TransformSet, false,
          MacroNameTok.getIdentifierInfo()->getName().str());
    }
  }
}

void IncludesCallbacks::MacroExpands(const Token &MacroNameTok,
                                     const MacroDefinition &MD,
                                     SourceRange Range, const MacroArgs *Args) {
  bool IsInAnalysisScope = isInAnalysisScope(SM.getSpellingLoc(MacroNameTok.getLocation()));
  auto MI = MD.getMacroInfo();
  if (!MI) {
    return;
  }
  if (MI->getNumTokens() > 0) {
    std::string HashKey = "";
    if (MI->getReplacementToken(0).getLocation().isValid()) {
      HashKey = getCombinedStrFromLoc(MI->getReplacementToken(0).getLocation());
    } else {
      HashKey = "InvalidLoc";
    }
    auto DefRange = Range;
    if(Range.getBegin().isMacroID() || Range.getEnd().isMacroID()) {
      DefRange = getDefinitionRange(Range.getBegin(), Range.getEnd());
    }

    dpct::DpctGlobalInfo::getExpansionRangeBeginMap()[getCombinedStrFromLoc(DefRange.getBegin())] =
        SourceRange(MI->getReplacementToken(0).getLocation(), MI->getDefinitionEndLoc());
    if (dpct::DpctGlobalInfo::getMacroDefines().find(HashKey) ==
        dpct::DpctGlobalInfo::getMacroDefines().end()) {
      // Record all processed macro definition
      dpct::DpctGlobalInfo::getMacroDefines()[HashKey] = true;
      size_t i;
      // Record all tokens in the macro definition
      for (i = 0; i < MI->getNumTokens(); i++) {
        std::shared_ptr<dpct::DpctGlobalInfo::MacroExpansionRecord> R =
            std::make_shared<dpct::DpctGlobalInfo::MacroExpansionRecord>(
                MacroNameTok.getIdentifierInfo(), MI, Range, IsInAnalysisScope, i);
        dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord()
            [getCombinedStrFromLoc(MI->getReplacementToken(i).getLocation())] =
                R;
      }
      std::shared_ptr<dpct::DpctGlobalInfo::MacroExpansionRecord> R =
          std::make_shared<dpct::DpctGlobalInfo::MacroExpansionRecord>(
              MacroNameTok.getIdentifierInfo(), MI, Range, IsInAnalysisScope,
              MI->getNumTokens());
      auto EndOfLastToken = Lexer::getLocForEndOfToken(
          MI->getReplacementToken(MI->getNumTokens() - 1).getLocation(), 0, SM,
          DpctGlobalInfo::getContext().getLangOpts());
      dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord()
          [getCombinedStrFromLoc(EndOfLastToken)] = R;
    }

    // If PredefinedStreamName is used with concatenated macro token,
    // detect the previous macro expansion
    std::string MacroNameStr;
    if (auto Identifier = MacroNameTok.getIdentifierInfo())
      MacroNameStr = Identifier->getName().str();
    if (MacroNameStr == "cudaStreamDefault"
        || MacroNameStr == "cudaStreamNonBlocking") {
      // Currently, only support examples like,
      // #define CONCATE(name) cuda##name
      // which contains 3 tokens, and the 2nd token is ##.
      // To support more complicated cases like,
      // #define CONCATE(name1, name2) cuda##name1##name2
      // will need to calculate the complete replaced string of the previous
      // macro.
      if (std::get<0>(dpct::DpctGlobalInfo::LastMacroRecord) == 3 &&
          std::get<1>(dpct::DpctGlobalInfo::LastMacroRecord) == "hashhash") {
        auto DefRange = getDefinitionRange(
            std::get<2>(dpct::DpctGlobalInfo::LastMacroRecord).getBegin(),
            std::get<2>(dpct::DpctGlobalInfo::LastMacroRecord).getEnd());
        auto Length = Lexer::MeasureTokenLength(
            DefRange.getEnd(), SM,
            dpct::DpctGlobalInfo::getContext().getLangOpts());
        Length += SM.getDecomposedLoc(DefRange.getEnd()).second -
                  SM.getDecomposedLoc(DefRange.getBegin()).second;
        requestFeature(HelperFeatureEnum::device_ext);
        TransformSet.emplace_back(new ReplaceText(
            DefRange.getBegin(), Length,
            "0"));
      }
    }

    // Record (#tokens, name of the 2nd token, range) as a tuple
    SourceRange LastRange = Range;
    dpct::DpctGlobalInfo::LastMacroRecord =
        std::make_tuple<unsigned int, std::string, SourceRange>(
            MI->getNumTokens(),
            MI->getNumTokens() >= 3
                ? std::string(MI->getReplacementToken(1).getName())
                : "",
            std::move(LastRange));
  } else {
    // Extend the Range to include comments/whitespaces before next token
    auto EndLoc = Range.getEnd();
    Token Tok;
    do {
      EndLoc = SM.getExpansionLoc(EndLoc);
      Lexer::getRawToken(
          EndLoc.getLocWithOffset(Lexer::MeasureTokenLength(
              EndLoc, SM, dpct::DpctGlobalInfo::getContext().getLangOpts())),
          Tok, SM, dpct::DpctGlobalInfo::getContext().getLangOpts(), true);
      EndLoc = Tok.getEndLoc();
    } while (Tok.isNot(tok::eof) && Tok.is(tok::comment));

    if (Tok.isNot(tok::eof)) {
      dpct::DpctGlobalInfo::getEndOfEmptyMacros()[getHashStrFromLoc(
          Tok.getLocation())] = Range.getBegin();
      dpct::DpctGlobalInfo::getBeginOfEmptyMacros()[getHashStrFromLoc(
          Range.getBegin())] =
          dpct::DpctGlobalInfo::getLocInfo(Range.getEnd()).second -
          dpct::DpctGlobalInfo::getLocInfo(Range.getBegin()).second;
    }
  }

  // In order to check whether __constant__ macro is empty, we first record
  // the expansion location of the __constant__, then check each __annotate__
  // macro, if the expansion locations are same and the content is empty, then
  // it means this __constant__ variable is used in host.
  // In this case, we need add "host_constant" flag in the replacement of
  // removing "__constant__"; and record the offset of the beginning of this
  // line for finding this replacement in MemVarRule. Since the variable name
  // is difficult to get here, the warning is also emitted in MemVarRule.
  if (MacroNameTok.getKind() == tok::identifier &&
      MacroNameTok.getIdentifierInfo() &&
      MacroNameTok.getIdentifierInfo()->getName() == "__annotate__" && MI &&
      !MI->param_empty()) {
    SourceLocation Loc = SM.getExpansionLoc(Range.getBegin());

    if (auto TM = DpctGlobalInfo::getInstance().findConstantMacroTMInfo(Loc)) {
      TM->setLineBeginOffset(getOffsetOfLineBegin(Loc, SM));
      if (MI->getNumTokens() == 0) {
        if (TM->getConstantFlag() == dpct::ConstantFlagType::Default ||
            TM->getConstantFlag() == dpct::ConstantFlagType::Host)
          TM->setConstantFlag(dpct::ConstantFlagType::Host);
        else
          TM->setConstantFlag(dpct::ConstantFlagType::HostDeviceInOnePass);
      } else {
        if (TM->getConstantFlag() == dpct::ConstantFlagType::Default ||
            TM->getConstantFlag() == dpct::ConstantFlagType::Device)
          TM->setConstantFlag(dpct::ConstantFlagType::Device);
        else
          TM->setConstantFlag(dpct::ConstantFlagType::HostDeviceInOnePass);
      }
    }
  }

  if (!IsInAnalysisScope) {
    return;
  }
  
  if (ReplaceCuMacro(MacroNameTok)){
    return ;
  }

  // For the un-specialized struct, there is no AST for the extern function
  // declaration in its member function body in Windows. e.g: template <typename
  // T> struct foo
  // {
  //    __device__ T *getPointer()
  //    {
  //        extern __device__ void error(void); // No AST for this line
  //        error();
  //        return NULL;
  //    }
  // };
  auto TKind = MacroNameTok.getKind();
  if (!MacroNameTok.getIdentifierInfo()) {
    return;
  }

  auto Name = MacroNameTok.getIdentifierInfo()->getName();
  if (TKind == tok::identifier &&
      (Name == "__host__" || Name == "__device__" || Name == "__global__" ||
       Name == "__constant__" || Name == "__launch_bounds__" ||
       Name == "__shared__" || Name == "__grid_constant__")) {
    auto TM = removeMacroInvocationAndTrailingSpaces(
        SourceRange(SM.getSpellingLoc(Range.getBegin()),
                    SM.getSpellingLoc(Range.getEnd())));
    if (Name == "__constant__") {
      if (!DpctGlobalInfo::getInstance().findConstantMacroTMInfo(
              SM.getExpansionLoc(Range.getBegin()))) {
        DpctGlobalInfo::getInstance().insertConstantMacroTMInfo(
            SM.getExpansionLoc(Range.getBegin()), TM);
        TransformSet.emplace_back(TM);
      }
    } else {
      TransformSet.emplace_back(TM);
    }
  }

  auto ItRule = MapNames::MacroRuleMap.find(Name.str());
  if (ItRule != MapNames::MacroRuleMap.end()) {
    std::string OutStr = ItRule->second.Out;
    TransformSet.emplace_back(
        new ReplaceToken(Range.getBegin(), std::move(OutStr)));
    requestFeature(ItRule->second.HelperFeature);
    for (auto ItHeader = ItRule->second.Includes.begin();
         ItHeader != ItRule->second.Includes.end(); ItHeader++) {
      DpctGlobalInfo::getInstance().insertHeader(Range.getBegin(), *ItHeader);
    }
  }

  if (TKind == tok::identifier && Name == "CUDART_CB") {
#ifdef _WIN32
    TransformSet.emplace_back(
        new ReplaceText(Range.getBegin(), 9, "__stdcall"));
#else
    TransformSet.emplace_back(removeMacroInvocationAndTrailingSpaces(Range));
#endif
  }

  auto Iter = MapNames::HostAllocSet.find(Name.str());
  if (TKind == tok::identifier && Iter != MapNames::HostAllocSet.end()) {
    if (MI->getNumTokens() == 1) {
      auto ReplToken = MI->getReplacementToken(0);
      if (ReplToken.getKind() == tok::numeric_constant) {
        TransformSet.emplace_back(new ReplaceToken(Range.getBegin(), "0"));
        DiagnosticsUtils::report(Range.getBegin(),
                                 Diagnostics::HOSTALLOCMACRO_NO_MEANING,
                                 &TransformSet, false, Name.str());
      }
    }
  }

  if (MI->getNumTokens() > 0) {
    DpctGlobalInfo::getInstance().removeAtomicInfo(
        getHashStrFromLoc(MI->getReplacementToken(0).getLocation()));
  }
}
std::shared_ptr<TextModification>
IncludesCallbacks::removeMacroInvocationAndTrailingSpaces(SourceRange Range) {
  const char *C = SM.getCharacterData(Range.getBegin());
  int Offset = 0;
  // Skip '\\', '\n' and '\r' when in macro define
  while (*(C + Offset) == '\\' || *(C + Offset) == '\n' ||
         *(C + Offset) == '\r') {
    Offset += 1;
  }
  Range =
      SourceRange(Range.getBegin().getLocWithOffset(Offset), Range.getEnd());
  return std::make_shared<ReplaceText>(
      Range.getBegin(), getLenIncludingTrailingSpaces(Range, SM), "", true);
}
void IncludesCallbacks::Else(SourceLocation Loc, SourceLocation IfLoc) {
  if (isInAnalysisScope(Loc)) {
    auto &Map = DpctGlobalInfo::getInstance()
                    .getCudaArchPPInfoMap()[SM.getFilename(Loc).str()];
    unsigned Offset = SM.getFileOffset(IfLoc);
    DirectiveInfo DI;
    DI.DirectiveLoc = SM.getFileOffset(Loc);
    int NSLoc = findPoundSign(Loc);
    if (NSLoc == -1) {
      DI.NumberSignLoc = UINT_MAX;
    } else {
      DI.NumberSignLoc = DI.DirectiveLoc - NSLoc;
    }
    if (Map.count(Offset)) {
      Map[Offset].ElseInfo = DI;
    } else {
      CudaArchPPInfo Info;
      Info.DT = IfType::IT_Unknow;
      Info.IfInfo.DirectiveLoc = Offset;
      Info.ElseInfo = DI;
      Map[Offset] = Info;
    }
  }
}
void IncludesCallbacks::Ifdef(SourceLocation Loc, const Token &MacroNameTok,
                              const MacroDefinition &MD) {
  if (!isInAnalysisScope(Loc))
    return;
  SourceLocation MacroLoc = MacroNameTok.getLocation();
  if (!MacroNameTok.getIdentifierInfo()) {
    return;
  }
  std::string MacroName = MacroNameTok.getIdentifierInfo()->getName().str();
  if (MacroName == "__CUDA_ARCH__" && DpctGlobalInfo::getRunRound() == 0) {
    requestFeature(HelperFeatureEnum::device_ext);
    auto &Map = DpctGlobalInfo::getInstance()
                    .getCudaArchPPInfoMap()[SM.getFilename(Loc).str()];
    unsigned Offset = SM.getFileOffset(Loc);
    CudaArchPPInfo Info;
    Info.DT = IfType::IT_Ifdef;
    int NSLoc = findPoundSign(Loc);
    if (NSLoc == -1) {
      Info.IfInfo.NumberSignLoc = UINT_MAX;
    } else {
      Info.IfInfo.NumberSignLoc = Offset - NSLoc;
    }
    Info.IfInfo.DirectiveLoc = Offset;
    Info.IfInfo.ConditionLoc = SM.getFileOffset(MacroLoc);
    Info.IfInfo.Condition = MacroName;
    Map[Offset] = Info;
  }
  ReplaceCuMacro(MacroNameTok);
}
void IncludesCallbacks::Ifndef(SourceLocation Loc, const Token &MacroNameTok,
                               const MacroDefinition &MD) {
  if (!isInAnalysisScope(Loc))
    return;
  SourceLocation MacroLoc = MacroNameTok.getLocation();
  if (!MacroNameTok.getIdentifierInfo()) {
    return;
  }
  std::string MacroName = MacroNameTok.getIdentifierInfo()->getName().str();
  if (MacroName == "__CUDA_ARCH__" && DpctGlobalInfo::getRunRound() == 0) {
    requestFeature(HelperFeatureEnum::device_ext);
    auto &Map = DpctGlobalInfo::getInstance()
                    .getCudaArchPPInfoMap()[SM.getFilename(Loc).str()];
    unsigned Offset = SM.getFileOffset(Loc);
    CudaArchPPInfo Info;
    Info.DT = IfType::IT_Ifndef;
    Info.IfInfo.DirectiveLoc = Offset;
    int NSLoc = findPoundSign(Loc);
    if (NSLoc == -1) {
      Info.IfInfo.NumberSignLoc = UINT_MAX;
    } else {
      Info.IfInfo.NumberSignLoc = Offset - NSLoc;
    }
    Info.IfInfo.ConditionLoc = SM.getFileOffset(MacroLoc);
    Info.IfInfo.Condition = MacroName;
    Map[Offset] = Info;
  }
  ReplaceCuMacro(MacroNameTok);
}

void IncludesCallbacks::Defined(const Token &MacroNameTok,
                                const MacroDefinition &MD, SourceRange Range) {
  SourceLocation MacroLoc = MacroNameTok.getLocation();
  if (!MacroNameTok.getIdentifierInfo()) {
    return;
  }
  std::string MacroName = MacroNameTok.getIdentifierInfo()->getName().str();
  if (!isInAnalysisScope(MacroLoc))
    return;
  if (MacroName == "__CUDA_ARCH__") {
    requestFeature(HelperFeatureEnum::device_ext);
    auto &Map =
        DpctGlobalInfo::getInstance()
            .getCudaArchDefinedMap()[SM.getFilename(Range.getBegin()).str()];
    if (Map.count(SM.getFileOffset(MacroLoc))) {
      Map[SM.getFileOffset(MacroLoc)] = SM.getFileOffset(Range.getBegin());
    } else {
      Map.insert(
          {SM.getFileOffset(MacroLoc), SM.getFileOffset(Range.getBegin())});
    }
  }
  ReplaceCuMacro(MacroNameTok);
}

void IncludesCallbacks::Endif(SourceLocation Loc, SourceLocation IfLoc) {
  bool IsInAnalysisScope = isInAnalysisScope(Loc);
  if (IsInAnalysisScope) {
    dpct::DpctGlobalInfo::getEndifLocationOfIfdef()[getHashStrFromLoc(IfLoc)] =
        Loc;
    dpct::DpctGlobalInfo::getConditionalCompilationLoc().emplace_back(
        DpctGlobalInfo::getInstance().getLocInfo(Loc));
    dpct::DpctGlobalInfo::getConditionalCompilationLoc().emplace_back(
        DpctGlobalInfo::getInstance().getLocInfo(IfLoc));
    if (DpctGlobalInfo::getRunRound() == 0) {
      auto &Map = DpctGlobalInfo::getInstance()
                      .getCudaArchPPInfoMap()[SM.getFilename(Loc).str()];
      unsigned Offset = SM.getFileOffset(IfLoc);
      DirectiveInfo DI;
      DI.DirectiveLoc = SM.getFileOffset(Loc);
      int NSLoc = findPoundSign(Loc);
      if (NSLoc == -1) {
        DI.NumberSignLoc = UINT_MAX;
      } else {
        DI.NumberSignLoc = DI.DirectiveLoc - NSLoc;
      }
      if (Map.count(Offset)) {
        Map[Offset].EndInfo = DI;
      } else {
        CudaArchPPInfo Info;
        Info.DT = IfType::IT_Unknow;
        Info.IfInfo.DirectiveLoc = Offset;
        Info.EndInfo = DI;
        Map[Offset] = Info;
      }
    }
  }
}
void IncludesCallbacks::ReplaceCuMacro(SourceRange ConditionRange, IfType IT,
                                       SourceLocation IfLoc,
                                       SourceLocation ElifLoc) {
  auto Begin = SM.getExpansionLoc(ConditionRange.getBegin());
  auto End = SM.getExpansionLoc(ConditionRange.getEnd());
  const char *BP = SM.getCharacterData(Begin);
  const char *EP = SM.getCharacterData(End);
  unsigned Size = EP - BP;
  Token Tok;
  if (!Lexer::getRawToken(End, Tok, SM, LangOptions()))
    Size = Size + Tok.getLength();
  std::string E(BP, Size);
  for (auto &MacroMap : MapNames::MacrosMap) {
    size_t Pos = 0;
    std::string MacroName = MacroMap.first;
    std::string ReplacedMacroName = MacroMap.second;

    std::size_t Found = E.find(MacroName, Pos);
    if (Found != std::string::npos && MacroName == "__CUDA_ARCH__") {
      auto &Map = DpctGlobalInfo::getInstance()
                      .getCudaArchPPInfoMap()[SM.getFilename(ElifLoc).str()];
      unsigned Offset = SM.getFileOffset(IfLoc);
      int NSLoc = -1;
      if (Map.count(Offset)) {
        if (IT == IfType::IT_If) {
          Map[Offset].DT = IfType::IT_If;
          Map[Offset].IfInfo.DirectiveLoc = Offset;
          Map[Offset].IfInfo.ConditionLoc = SM.getFileOffset(Begin);
          Map[Offset].IfInfo.Condition = E;
          NSLoc = findPoundSign(IfLoc);
          if (NSLoc == -1) {
            Map[Offset].IfInfo.NumberSignLoc = UINT_MAX;
          } else {
            Map[Offset].IfInfo.NumberSignLoc = Offset - NSLoc;
          }
        } else {
          if (Map[Offset].ElInfo.count(SM.getFileOffset(ElifLoc)))
            return;
          DirectiveInfo DI;
          DI.DirectiveLoc = SM.getFileOffset(ElifLoc);
          DI.ConditionLoc = SM.getFileOffset(Begin);
          DI.Condition = E;
          NSLoc = findPoundSign(IfLoc);
          if (NSLoc == -1) {
            DI.NumberSignLoc = UINT_MAX;
          } else {
            DI.NumberSignLoc = DI.DirectiveLoc - NSLoc;
          }
          Map[Offset].ElInfo[SM.getFileOffset(ElifLoc)] = DI;
        }
      } else {
        CudaArchPPInfo Info;
        DirectiveInfo DI;
        if (IT == IfType::IT_If) {
          DI.DirectiveLoc = Offset;
          DI.ConditionLoc = SM.getFileOffset(Begin);
          DI.Condition = E;
          NSLoc = findPoundSign(IfLoc);
          if (NSLoc == -1) {
            DI.NumberSignLoc = UINT_MAX;
          } else {
            DI.NumberSignLoc = Offset - NSLoc;
          }
          Info.IfInfo = DI;
          Info.DT = IfType::IT_If;
        } else {
          DI.DirectiveLoc = SM.getFileOffset(ElifLoc);
          DI.ConditionLoc = SM.getFileOffset(Begin);
          DI.Condition = E;
          NSLoc = findPoundSign(IfLoc);
          if (NSLoc == -1) {
            DI.NumberSignLoc = UINT_MAX;
          } else {
            DI.NumberSignLoc = DI.DirectiveLoc - NSLoc;
          }
          Info.ElInfo[SM.getFileOffset(ElifLoc)] = DI;
          Info.DT = IfType::IT_Unknow;
        }
        Map[Offset] = Info;
      }
    }
    while (Found != std::string::npos) {
      // found one, insert replace for it
      SourceLocation IB = Begin.getLocWithOffset(Found);
      SourceLocation IE = IB.getLocWithOffset(MacroName.length());
      CharSourceRange InsertRange(SourceRange(IB, IE), false);
      auto Repl =
          std::make_shared<ReplaceInclude>(InsertRange, ReplacedMacroName);
      if (MacroName == "__CUDA_ARCH__" &&
          DpctGlobalInfo::getInstance().getContext().getLangOpts().CUDA) {
        insertCudaArchRepl(Repl->getReplacement(DpctGlobalInfo::getContext()));
        requestFeature(HelperFeatureEnum::device_ext);
      } else if ((MacroName != "__CUDACC__" ||
                  DpctGlobalInfo::getMacroDefines().count(MacroName)) &&
                 MacroName != "__CUDA_ARCH__") {
        TransformSet.emplace_back(Repl);
      }
      // check next
      Pos = Found + MacroName.length();
      if ((Pos + MacroName.length()) > Size) {
        break;
      }
      Found = E.find(MacroName, Pos);
    }
  }
}
void IncludesCallbacks::If(SourceLocation Loc, SourceRange ConditionRange,
                           ConditionValueKind ConditionValue) {
  bool IsInAnalysisScope = isInAnalysisScope(Loc);

  if (!IsInAnalysisScope) {
    return;
  }
  ReplaceCuMacro(ConditionRange, IfType::IT_If, Loc, Loc);
}
void IncludesCallbacks::Elif(SourceLocation Loc, SourceRange ConditionRange,
                             ConditionValueKind ConditionValue,
                             SourceLocation IfLoc) {
  bool IsInAnalysisScope = isInAnalysisScope(Loc);

  if (!IsInAnalysisScope) {
    return;
  }

  ReplaceCuMacro(ConditionRange, IfType::IT_Elif, IfLoc, Loc);
}
bool IncludesCallbacks::ShouldEnter(StringRef FileName, bool IsAngled) {
#ifdef _WIN32
  std::string Name = FileName.str();
  return !IsAngled || !MapNames::isInSet(MapNames::ThrustFileExcludeSet, Name);
#else
  return true;
#endif
}

void IncludesCallbacks::FileChanged(SourceLocation Loc, FileChangeReason Reason,
                                    SrcMgr::CharacteristicKind FileType,
                                    FileID PrevFID) {
  if (DpctGlobalInfo::isQueryAPIMapping())
    return;
  // Record the location when a file is entered
  if (Reason == clang::PPCallbacks::EnterFile) {
    DpctGlobalInfo::getInstance().setFileEnterLocation(Loc);

    bool IsInAnalysisScope = isInAnalysisScope(Loc);

    if (!IsInAnalysisScope) {
      return;
    }

    std::string InFile = SM.getFilename(Loc).str();
    InFile = getAbsolutePath(InFile);
    makeCanonical(InFile);
    if (IsFileInCmd || ProcessAllFlag ||
        GetSourceFileType(InFile) & SPT_CudaSource) {
      IncludeFileMap[DpctGlobalInfo::removeSymlinks(SM.getFileManager(),
                                                    InFile)] = false;
    }
    IsFileInCmd = false;

    loadYAMLIntoFileInfo(InFile);
  }
}

void MigrationRule::print(llvm::raw_ostream &OS) {
  const auto &EmittedTransformations = getEmittedTransformations();
  if (EmittedTransformations.empty()) {
    return;
  }

  OS << "[" << getName() << "]" << getNL();
  constexpr char Indent[] = "  ";
  for (const auto &TM : EmittedTransformations) {
    OS << Indent;
    TM->print(OS, DpctGlobalInfo::getContext(),
              /* Print parent */ false);
  }
}

void MigrationRule::printStatistics(llvm::raw_ostream &OS) {
  const auto &EmittedTransformations = getEmittedTransformations();
  if (EmittedTransformations.empty()) {
    return;
  }

  OS << "<Statistics of " << getName() << ">" << getNL();
  std::unordered_map<std::string, size_t> TMNameCountMap;
  for (const auto &TM : EmittedTransformations) {
    const std::string Name = TM->getName();
    if (TMNameCountMap.count(Name) == 0) {
      TMNameCountMap.emplace(std::make_pair(Name, 1));
    } else {
      ++TMNameCountMap[Name];
    }
  }

  constexpr char Indent[] = "  ";
  for (const auto &Pair : TMNameCountMap) {
    const std::string &Name = Pair.first;
    const size_t &Numbers = Pair.second;
    OS << Indent << "Emitted # of replacement <" << Name << ">: " << Numbers
       << getNL();
  }
}

void MigrationRule::emplaceTransformation(TextModification *TM) {
  auto T = std::shared_ptr<TextModification>(TM);
  Transformations.emplace_back(T);
  TransformSet->emplace_back(T);
}

void IterationSpaceBuiltinRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      memberExpr(hasObjectExpression(opaqueValueExpr(hasSourceExpression(
                     declRefExpr(to(varDecl(hasAnyName("threadIdx", "blockDim",
                                                       "blockIdx", "gridDim"))
                                        .bind("varDecl")))
                         .bind("declRefExpr")))),
                 hasAncestor(functionDecl().bind("func")))
          .bind("memberExpr"),
      this);
  MF.addMatcher(declRefExpr(to(varDecl(hasAnyName("warpSize")).bind("varDecl")))
                    .bind("declRefExpr"),
                this);

  MF.addMatcher(declRefExpr(to(varDecl(hasAnyName("threadIdx", "blockDim",
                                                  "blockIdx", "gridDim"))),
                            hasAncestor(functionDecl().bind("funcDecl")))
                    .bind("declRefExprUnTempFunc"),
                this);
}

bool IterationSpaceBuiltinRule::renameBuiltinName(const DeclRefExpr *DRE,
                                                  std::string &NewName) {
  auto BuiltinName = DRE->getDecl()->getName();
  if (BuiltinName == "threadIdx")
    NewName = DpctGlobalInfo::getItem(DRE) + ".get_local_id(";
  else if (BuiltinName == "blockDim")
    NewName = DpctGlobalInfo::getItem(DRE) + ".get_local_range(";
  else if (BuiltinName == "blockIdx")
    NewName = DpctGlobalInfo::getItem(DRE) + ".get_group(";
  else if (BuiltinName == "gridDim")
    NewName = DpctGlobalInfo::getItem(DRE) + ".get_group_range(";
  else if (BuiltinName == "warpSize")
    NewName = DpctGlobalInfo::getSubGroup(DRE) + ".get_local_range().get(0)";
  else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected field name: " << BuiltinName;
    return false;
  }

  return true;
}

void IterationSpaceBuiltinRule::runRule(
    const MatchFinder::MatchResult &Result) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  if (const DeclRefExpr *DRE =
          getNodeAsType<DeclRefExpr>(Result, "declRefExprUnTempFunc")) {
    // Take the case of instantiated template function for example:
    // template <typename IndexType = int> __device__ void thread_id() {
    //  auto tidx_template = static_cast<IndexType>(threadIdx.x);
    //}
    // On Linux platform, .x(MemberExpr, __cuda_builtin_threadIdx_t) in
    // static_cast statement is not available in AST, while 'threadIdx' is
    // available, so dpct migrates it by 'threadIdx' matcher to identify the
    // SourceLocation of 'threadIdx', then look forward 2 tokens to check
    // whether .x appears.
    auto FD = getAssistNodeAsType<FunctionDecl>(Result, "funcDecl");
    if (!FD)
      return;
    const auto Begin = SM.getSpellingLoc(DRE->getBeginLoc());
    auto End = SM.getSpellingLoc(DRE->getEndLoc());
    End = End.getLocWithOffset(
        Lexer::MeasureTokenLength(End, *Result.SourceManager, LangOptions()));

    const auto Type = DRE->getDecl()
                          ->getType()
                          .getCanonicalType()
                          .getUnqualifiedType()
                          .getAsString();

    if (Type.find("__cuda_builtin") == std::string::npos)
      return;

    const auto Tok2Ptr = Lexer::findNextToken(End, SM, LangOptions());
    if (!Tok2Ptr.has_value())
      return;

    const auto Tok2 = Tok2Ptr.value();
    if (Tok2.getKind() == tok::raw_identifier) {
      std::string TypeStr = Tok2.getRawIdentifier().str();
      const char *StartPos = SM.getCharacterData(Begin);
      const char *EndPos = SM.getCharacterData(Tok2.getEndLoc());
      const auto TyLen = EndPos - StartPos;

      if (TyLen <= 0)
        return;

      std::string Replacement;
      if (!renameBuiltinName(DRE, Replacement))
        return;

      const auto FieldName = Tok2.getRawIdentifier().str();
      unsigned Dimension;
      auto DFI = DeviceFunctionDecl::LinkRedecls(FD);
      if (!DFI)
        return;

      if (FieldName == "x") {
        DpctGlobalInfo::getInstance().insertBuiltinVarInfo(Begin, TyLen,
                                                           Replacement, DFI);
        DpctGlobalInfo::updateSpellingLocDFIMaps(DRE->getBeginLoc(), DFI);
        return;
      } else if (FieldName == "y") {
        Dimension = 1;
        DFI->getVarMap().Dim = 3;
      } else if (FieldName == "z") {
        Dimension = 0;
        DFI->getVarMap().Dim = 3;
      } else
        return;

      Replacement += std::to_string(Dimension);
      Replacement += ")";

      emplaceTransformation(
          new ReplaceText(Begin, TyLen, std::move(Replacement)));
    }
    return;
  }

  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "memberExpr");
  const VarDecl *VD = getAssistNodeAsType<VarDecl>(Result, "varDecl");
  const DeclRefExpr *DRE = getNodeAsType<DeclRefExpr>(Result, "declRefExpr");
  std::shared_ptr<DeviceFunctionInfo> DFI = nullptr;
  if (!VD || !DRE) {
    return;
  }
  bool IsME = false;
  if (ME) {
    auto FD = getAssistNodeAsType<FunctionDecl>(Result, "func");
    if (!FD)
      return;
    DFI = DeviceFunctionDecl::LinkRedecls(FD);
    if (!DFI)
      return;
    IsME = true;
  } else {
    std::string InFile = dpct::DpctGlobalInfo::getSourceManager()
                             .getFilename(VD->getBeginLoc())
                             .str();

    if (!isChildOrSamePath(DpctInstallPath, InFile)) {
      return;
    }
  }

  std::string Replacement;
  StringRef BuiltinName = VD->getName();
  if (!renameBuiltinName(DRE, Replacement))
    return;

  if (IsME) {
    ValueDecl *Field = ME->getMemberDecl();
    StringRef FieldName = Field->getName();
    unsigned Dimension;
    if (FieldName == "__fetch_builtin_x") {
      auto Range = getDefinitionRange(ME->getBeginLoc(), ME->getEndLoc());
      SourceLocation Begin = Range.getBegin();
      SourceLocation End = Range.getEnd();

      End = End.getLocWithOffset(Lexer::MeasureTokenLength(
          End, SM, DpctGlobalInfo::getContext().getLangOpts()));

      unsigned int Len =
          SM.getDecomposedLoc(End).second - SM.getDecomposedLoc(Begin).second;
      DpctGlobalInfo::getInstance().insertBuiltinVarInfo(Begin, Len,
                                                         Replacement, DFI);
      DpctGlobalInfo::updateSpellingLocDFIMaps(ME->getBeginLoc(), DFI);
      return;
    } else if (FieldName == "__fetch_builtin_y") {
      Dimension = 1;
      DFI->getVarMap().Dim = 3;
    } else if (FieldName == "__fetch_builtin_z") {
      Dimension = 0;
      DFI->getVarMap().Dim = 3;
    } else {
      llvm::dbgs() << "[" << getName()
                   << "] Unexpected field name: " << FieldName;
      return;
    }

    Replacement += std::to_string(Dimension);
    Replacement += ")";
  }
  if (IsME) {
    emplaceTransformation(new ReplaceStmt(ME, std::move(Replacement)));
  } else {
    auto isDefaultParmWarpSize = [=](const FunctionDecl *&FD,
                                     const ParmVarDecl *&PVD) -> bool {
      if (BuiltinName != "warpSize")
        return false;
      PVD = DpctGlobalInfo::findAncestor<ParmVarDecl>(DRE);
      if (!PVD || !PVD->hasDefaultArg())
        return false;
      FD = dyn_cast_or_null<FunctionDecl>(PVD->getParentFunctionOrMethod());
      if (!FD)
        return false;
      if (FD->hasAttr<CUDADeviceAttr>())
        return true;
      return false;
    };

    const ParmVarDecl *PVD = nullptr;
    const FunctionDecl *FD = nullptr;
    if (isDefaultParmWarpSize(FD, PVD)) {
      SourceManager &SM = DpctGlobalInfo::getSourceManager();
      bool IsConstQualified = PVD->getType().isConstQualified();
      emplaceTransformation(new ReplaceStmt(DRE, "0"));
      unsigned int Idx = PVD->getFunctionScopeIndex();

      for (const auto FDIter : FD->redecls()) {
        if (IsConstQualified) {
          SourceRange SR;
          const ParmVarDecl *CurrentPVD = FDIter->getParamDecl(Idx);
          if (getTypeRange(CurrentPVD, SR)) {
            auto Length =
                SM.getFileOffset(SR.getEnd()) - SM.getFileOffset(SR.getBegin());
            QualType NewType = CurrentPVD->getType();
            NewType.removeLocalConst();
            std::string NewTypeStr =
                DpctGlobalInfo::getReplacedTypeName(NewType);
            emplaceTransformation(
                new ReplaceText(SR.getBegin(), Length, std::move(NewTypeStr)));
          }
        }

        const Stmt *Body = FD->getBody();
        if (!Body)
          continue;
        if (const CompoundStmt *BodyCS = dyn_cast<CompoundStmt>(Body)) {
          if (BodyCS->child_begin() != BodyCS->child_end()) {
            SourceLocation InsertLoc =
                SM.getExpansionLoc((*(BodyCS->child_begin()))->getBeginLoc());
            std::string IndentStr = getIndent(InsertLoc, SM).str();
            std::string Text =
                "if (!" + PVD->getName().str() + ") " + PVD->getName().str() +
                " = " + DpctGlobalInfo::getSubGroup(BodyCS) +
                ".get_local_range().get(0);" + getNL() + IndentStr;
            emplaceTransformation(new InsertText(InsertLoc, std::move(Text)));
          }
        }
      }
    } else
      emplaceTransformation(new ReplaceStmt(DRE, std::move(Replacement)));
  }
}

REGISTER_RULE(IterationSpaceBuiltinRule, PassKind::PK_Analysis)

void ErrorHandlingIfStmtRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      // Match if-statement that does not have else and has a condition of
      // either an operator!= or a variable of type enum.
      ifStmt(unless(hasElse(anything())),
             hasCondition(
                 anyOf(binaryOperator(hasOperatorName("!=")).bind("op!="),
                       ignoringImpCasts(
                           declRefExpr(hasType(hasCanonicalType(enumType())))
                               .bind("var")))))
          .bind("errIf"),
      this);
  MF.addMatcher(
      // Match if-statement that does not have else and has a condition of
      // operator==.
      ifStmt(unless(hasElse(anything())),
             hasCondition(binaryOperator(hasOperatorName("==")).bind("op==")))
          .bind("errIfSpecial"),
      this);
}

static bool isVarRef(const Expr *E) {
  if (auto D = dyn_cast<DeclRefExpr>(E))
    return isa<VarDecl>(D->getDecl());
  else
    return false;
}

static std::string getVarType(const Expr *E) {
  return E->getType().getCanonicalType().getUnqualifiedType().getAsString();
}

static bool isCudaFailureCheck(const BinaryOperator *Op, bool IsEq = false) {
  auto Lhs = Op->getLHS()->IgnoreImplicit();
  auto Rhs = Op->getRHS()->IgnoreImplicit();

  const Expr *Literal = nullptr;

  if (isVarRef(Lhs) && (getVarType(Lhs) == "enum cudaError" ||
                        getVarType(Lhs) == "enum cudaError_enum")) {
    Literal = Rhs;
  } else if (isVarRef(Rhs) && (getVarType(Rhs) == "enum cudaError" ||
                               getVarType(Rhs) == "enum cudaError_enum")) {
    Literal = Lhs;
  } else
    return false;

  if (auto IntLit = dyn_cast<IntegerLiteral>(Literal)) {
    if (IsEq ^ (IntLit->getValue() != 0))
      return false;
  } else if (auto D = dyn_cast<DeclRefExpr>(Literal)) {
    auto EnumDecl = dyn_cast<EnumConstantDecl>(D->getDecl());
    if (!EnumDecl)
      return false;
    if (IsEq ^ (EnumDecl->getInitVal() != 0))
      return false;
  } else {
    // The expression is neither an int literal nor an enum value.
    return false;
  }

  return true;
}

static bool isCudaFailureCheck(const DeclRefExpr *E) {
  return isVarRef(E) && (getVarType(E) == "enum cudaError" ||
                         getVarType(E) == "enum cudaError_enum");
}

void ErrorHandlingIfStmtRule::runRule(const MatchFinder::MatchResult &Result) {
  static std::vector<std::string> NameList = {"errIf", "errIfSpecial"};
  const IfStmt *If = getNodeAsType<IfStmt>(Result, "errIf");
  if (!If)
    if (!(If = getNodeAsType<IfStmt>(Result, "errIfSpecial")))
      return;
  auto EmitNotRemoved = [&](SourceLocation SL, const Stmt *R) {
    report(SL, Diagnostics::STMT_NOT_REMOVED, false);
  };
  auto isErrorHandlingSafeToRemove = [&](const Stmt *S) {
    if (const auto *CE = dyn_cast<CallExpr>(S)) {
      if (!CE->getDirectCallee()) {
        EmitNotRemoved(S->getSourceRange().getBegin(), S);
        return false;
      }
      auto Name = CE->getDirectCallee()->getNameAsString();
      static const llvm::StringSet<> SafeCallList = {
          "printf", "puts", "exit", "cudaDeviceReset", "fprintf"};
      if (SafeCallList.find(Name) == SafeCallList.end()) {
        EmitNotRemoved(S->getSourceRange().getBegin(), S);
        return false;
      }
      return true;
    }
    EmitNotRemoved(S->getSourceRange().getBegin(), S);
    return false;
  };

  auto isErrorHandling = [&](const Stmt *Block) {
    if (!isa<CompoundStmt>(Block))
      return isErrorHandlingSafeToRemove(Block);
    const CompoundStmt *CS = cast<CompoundStmt>(Block);
    for (const auto *S : CS->children()) {
      if (auto *E = dyn_cast_or_null<Expr>(S)) {
        if (!isErrorHandlingSafeToRemove(E->IgnoreImplicit())) {
          return false;
        }
      }
    }
    return true;
  };

  if (![&] {
        bool IsIfstmtSpecialCase = false;
        SourceLocation Ip;
        if (auto Op = getNodeAsType<BinaryOperator>(Result, "op!=")) {
          if (!isCudaFailureCheck(Op))
            return false;
        } else if (auto Op = getNodeAsType<BinaryOperator>(Result, "op==")) {
          if (!isCudaFailureCheck(Op, true))
            return false;
          IsIfstmtSpecialCase = true;
          Ip = Op->getBeginLoc();

        } else {
          auto CondVar = getNodeAsType<DeclRefExpr>(Result, "var");
          if (!isCudaFailureCheck(CondVar))
            return false;
        }
        // We know that it's error checking condition, check the body
        if (!isErrorHandling(If->getThen())) {
          if (IsIfstmtSpecialCase) {
            report(Ip, Diagnostics::IFSTMT_SPECIAL_CASE, false);
          } else {
            report(If->getSourceRange().getBegin(),
                   Diagnostics::IFSTMT_NOT_REMOVED, false);
          }
          return false;
        }
        return true;
      }()) {

    return;
  }

  emplaceTransformation(new ReplaceStmt(If, ""));

  // if the last token right after the ifstmt is ";"
  // remove the token
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto EndLoc = Lexer::getLocForEndOfToken(
      SM.getSpellingLoc(If->getEndLoc()), 0, SM, Result.Context->getLangOpts());
  Token Tok;
  Lexer::getRawToken(EndLoc, Tok, SM, Result.Context->getLangOpts(), true);
  if (Tok.getKind() == tok::semi) {
    emplaceTransformation(new ReplaceText(EndLoc, 1, ""));
  }
}

REGISTER_RULE(ErrorHandlingIfStmtRule, PassKind::PK_Migration)

void ErrorHandlingHostAPIRule::registerMatcher(MatchFinder &MF) {
  auto isMigratedHostAPI = [&]() {
    return allOf(
        anyOf(returns(asString("cudaError_t")),
              returns(asString("cublasStatus_t")),
              returns(asString("nvgraphStatus_t")),
              returns(asString("cusparseStatus_t")),
              returns(asString("cusolverStatus_t")),
              returns(asString("cufftResult_t")),
              returns(asString("curandStatus_t")),
              returns(asString("ncclResult_t"))),
        // cudaGetLastError returns cudaError_t but won't fail in the call
        unless(hasName("cudaGetLastError")),
        anyOf(unless(hasAttr(attr::CUDADevice)), hasAttr(attr::CUDAHost)));
  };

  // Match host API call in the condition session of flow control
  MF.addMatcher(
      functionDecl(
          allOf(
              unless(hasDescendant(functionDecl())),
              unless(
                  anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal))),
              anyOf(
                  hasDescendant(ifStmt(hasCondition(expr(hasDescendant(
                      callExpr(callee(functionDecl(isMigratedHostAPI())))))))),
                  hasDescendant(doStmt(hasCondition(expr(hasDescendant(
                      callExpr(callee(functionDecl(isMigratedHostAPI())))))))),
                  hasDescendant(whileStmt(hasCondition(expr(hasDescendant(
                      callExpr(callee(functionDecl(isMigratedHostAPI())))))))),
                  hasDescendant(switchStmt(hasCondition(expr(hasDescendant(
                      callExpr(callee(functionDecl(isMigratedHostAPI())))))))),
                  hasDescendant(
                      forStmt(hasCondition(expr(hasDescendant(callExpr(
                          callee(functionDecl(isMigratedHostAPI())))))))))))
          .bind("inConditionHostAPI"),
      this);

  // Match host API call whose return value used inside flow control or return
  MF.addMatcher(
      functionDecl(
          allOf(unless(hasDescendant(functionDecl())),
                unless(anyOf(hasAttr(attr::CUDADevice),
                             hasAttr(attr::CUDAGlobal))),
                hasDescendant(callExpr(allOf(
                    callee(functionDecl(isMigratedHostAPI())),
                    anyOf(hasAncestor(binaryOperator(allOf(
                              hasLHS(declRefExpr()), isAssignmentOperator()))),
                          hasAncestor(varDecl())),
                    anyOf(hasAncestor(ifStmt()), hasAncestor(doStmt()),
                          hasAncestor(switchStmt()), hasAncestor(whileStmt()),
                          hasAncestor(callExpr()), hasAncestor(forStmt())))))))
          .bind("inLoopHostAPI"),
      this);

  MF.addMatcher(
      functionDecl(allOf(unless(hasDescendant(functionDecl())),
                         unless(anyOf(hasAttr(attr::CUDADevice),
                                      hasAttr(attr::CUDAGlobal))),
                         hasDescendant(callExpr(
                             allOf(callee(functionDecl(isMigratedHostAPI())),
                                   hasAncestor(returnStmt()))))))
          .bind("inReturnHostAPI"),
      this);

  // Match host API call whose return value captured and used
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(isMigratedHostAPI())),
                     anyOf(hasAncestor(binaryOperator(
                               allOf(hasLHS(declRefExpr().bind("targetLHS")),
                                     isAssignmentOperator()))),
                           hasAncestor(varDecl().bind("targetVarDecl"))),
                     unless(hasDescendant(functionDecl())),
                     hasAncestor(
                         functionDecl(unless(anyOf(hasAttr(attr::CUDADevice),
                                                   hasAttr(attr::CUDAGlobal))))
                             .bind("savedHostAPI"))))
          .bind("referencedHostAPI"),
      this);
}

void ErrorHandlingHostAPIRule::runRule(const MatchFinder::MatchResult &Result) {
  // if host API call in the condition session of flow control
  // or host API call whose return value used inside flow control or return
  // then add try catch.
  auto FD = getNodeAsType<FunctionDecl>(Result, "inConditionHostAPI");
  if (!FD) {
    FD = getNodeAsType<FunctionDecl>(Result, "inLoopHostAPI");
  }
  if (!FD) {
    FD = getNodeAsType<FunctionDecl>(Result, "inReturnHostAPI");
  }
  if (FD) {
    insertTryCatch(FD);
    return;
  }

  // Check if the return value is saved in a variable,
  // if yes, get the varDecl as the target varDecl TD.
  FD = getAssistNodeAsType<FunctionDecl>(Result, "savedHostAPI");
  if (!FD)
    return;
  auto TVD = getAssistNodeAsType<VarDecl>(Result, "targetVarDecl");
  auto TLHS = getAssistNodeAsType<DeclRefExpr>(Result, "targetLHS");
  const ValueDecl *TD = nullptr;
  if (TVD || TLHS) {
    TD = TVD ? TVD : TLHS->getDecl();
  }

  if (!TD)
    return;

  // Get the location of the API call to make sure the variable is referenced
  // AFTER the API call.
  auto CE = getAssistNodeAsType<CallExpr>(Result, "referencedHostAPI");

  if (!CE)
    return;

  // For each reference of TD, check if the location is after CE,
  // if yes, add try catch.
  std::vector<const DeclRefExpr *> Refs;
  VarReferencedInFD(FD->getBody(), TD, Refs);
  SourceManager &SM = DpctGlobalInfo::getSourceManager();
  auto CallLoc = SM.getExpansionLoc(CE->getBeginLoc());
  for (auto It = Refs.begin(); It != Refs.end(); ++It) {
    auto RefLoc = SM.getExpansionLoc((*It)->getBeginLoc());
    if (SM.getCharacterData(RefLoc) - SM.getCharacterData(CallLoc) > 0) {
      insertTryCatch(FD);
      return;
    }
  }
}

void ErrorHandlingHostAPIRule::insertTryCatch(const FunctionDecl *FD) {
  SourceManager &SM = DpctGlobalInfo::getSourceManager();
  bool IsLambda = false;
  bool IsInMacro = false;
  if (auto CMD = dyn_cast<CXXMethodDecl>(FD)) {
    if (CMD->getParent()->isLambda()) {
      IsLambda = true;
    }
  }

  auto BodyRange = getDefinitionRange(FD->getBody()->getBeginLoc(),
                                      FD->getBody()->getEndLoc());
  auto It = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      getCombinedStrFromLoc(BodyRange.getEnd()));
  if (It != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
    IsInMacro = true;
  }

  std::string IndentStr = getIndent(FD->getBeginLoc(), SM).str();
  std::string InnerIndentStr = IndentStr + "  ";

  std::string NewLine = getNL();
  if(IsInMacro)
    NewLine = "\\" + NewLine;

  if (IsLambda) {
    if (auto CSM = dyn_cast<CompoundStmt>(FD->getBody())) {
      // IndentStr = getIndent((*(CSM->body_begin()))->getBeginLoc(), SM).str();
      std::string TryStr = "try{ " + std::string(getNL()) + IndentStr;
      emplaceTransformation(
          new InsertBeforeStmt(*(CSM->body_begin()), std::move(TryStr)));
    }
  } else if (const CXXConstructorDecl *CDecl = getIfConstructorDecl(FD)) {
    emplaceTransformation(new InsertBeforeCtrInitList(CDecl, " try "));
  } else {
    emplaceTransformation(new InsertBeforeStmt(FD->getBody(), " try "));
  }

  std::string ReplaceStr =
      NewLine + IndentStr +
      std::string("catch (" + MapNames::getClNamespace(true) +
                  "exception const &exc) {") +
      NewLine + InnerIndentStr +
      std::string("std::cerr << exc.what() << \"Exception caught at file:\" << "
                  "__FILE__ << "
                  "\", line:\" << __LINE__ << std::endl;") +
      NewLine + InnerIndentStr + std::string("std::exit(1);") + NewLine +
      IndentStr + "}";
  if (IsLambda) {
    ReplaceStr += NewLine + IndentStr + "}";
  }
  emplaceTransformation(
      new InsertAfterStmt(FD->getBody(), std::move(ReplaceStr)));
}

REGISTER_RULE(ErrorHandlingHostAPIRule, PassKind::PK_Migration)

void AtomicFunctionRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> AtomicFuncNames(MapNames::AtomicFuncNamesMap.size());
  std::transform(
      MapNames::AtomicFuncNamesMap.begin(), MapNames::AtomicFuncNamesMap.end(),
      AtomicFuncNames.begin(),
      [](const std::pair<std::string, std::string> &p) { return p.first; });

  auto hasAnyAtomicFuncName = [&]() {
    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(AtomicFuncNames));
  };

  // Support all integer type, float and double
  // Type half and half2 are not supported
  auto supportedTypes = [&]() {
    return anyOf(hasType(pointsTo(isInteger())),
                 hasType(pointsTo(asString("float"))),
                 hasType(pointsTo(asString("double"))));
  };

  auto supportedAtomicFunctions = [&]() {
    return allOf(hasAnyAtomicFuncName(), hasParameter(0, supportedTypes()));
  };

  auto unsupportedAtomicFunctions = [&]() {
    return allOf(hasAnyAtomicFuncName(),
                 unless(hasParameter(0, supportedTypes())));
  };

  MF.addMatcher(callExpr(callee(functionDecl(supportedAtomicFunctions())))
                    .bind("supportedAtomicFuncCall"),
                this);

  MF.addMatcher(callExpr(callee(functionDecl(unsupportedAtomicFunctions())))
                    .bind("unsupportedAtomicFuncCall"),
                this);
}

void AtomicFunctionRule::ReportUnsupportedAtomicFunc(const CallExpr *CE) {
  if (!CE)
    return;

  std::ostringstream OSS;
  // Atomic functions with __half and half2 are not supported.
  if (!CE->getDirectCallee())
    return;
  OSS << "half version of "
      << MapNames::ITFName.at(CE->getDirectCallee()->getName().str());
  report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, OSS.str());
}

void AtomicFunctionRule::MigrateAtomicFunc(
    const CallExpr *CE, const ast_matchers::MatchFinder::MatchResult &Result) {
  if (!CE)
    return;

  // Don't migrate user defined function
  if (auto *CalleeDecl = CE->getDirectCallee()) {
    if (isUserDefinedDecl(CalleeDecl))
      return;
  } else {
    return;
  };

  const std::string FuncName = CE->getDirectCallee()->getName().str();
  if (!CallExprRewriterFactoryBase::RewriterMap)
    return;
  auto Iter = CallExprRewriterFactoryBase::RewriterMap->find(FuncName);
  if (Iter != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
}

void AtomicFunctionRule::runRule(const MatchFinder::MatchResult &Result) {
  ReportUnsupportedAtomicFunc(
      getNodeAsType<CallExpr>(Result, "unsupportedAtomicFuncCall"));

  MigrateAtomicFunc(getNodeAsType<CallExpr>(Result, "supportedAtomicFuncCall"),
                    Result);
}

REGISTER_RULE(AtomicFunctionRule, PassKind::PK_Migration)

void ZeroLengthArrayRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(typeLoc(loc(constantArrayType())).bind("ConstantArrayType"),
                this);
}
void ZeroLengthArrayRule::runRule(
    const MatchFinder::MatchResult &Result) {
  auto TL = getNodeAsType<TypeLoc>(Result, "ConstantArrayType");
  if (!TL)
    return;
  const ConstantArrayType *CAT =
      dyn_cast_or_null<ConstantArrayType>(TL->getTypePtr());
  if (!CAT)
    return;

  // Check the array length
  if (!(CAT->getSize().isZero()))
    return;

  const clang::FieldDecl *MemberVariable =
      DpctGlobalInfo::findAncestor<clang::FieldDecl>(TL);
  if (MemberVariable) {
    report(TL->getBeginLoc(), Diagnostics::ZERO_LENGTH_ARRAY, false);
  } else {
    const clang::FunctionDecl *FD = DpctGlobalInfo::getParentFunction(TL);
    if (FD) {
      // Check if the array is in device code
      if (!(FD->getAttr<CUDADeviceAttr>()) && !(FD->getAttr<CUDAGlobalAttr>()))
        return;
    }
  }

  // Check if the array is a shared variable
  const VarDecl* VD = DpctGlobalInfo::findAncestor<VarDecl>(TL);
  if (VD && VD->getAttr<CUDASharedAttr>())
    return;

  report(TL->getBeginLoc(), Diagnostics::ZERO_LENGTH_ARRAY, false);
}
REGISTER_RULE(ZeroLengthArrayRule, PassKind::PK_Migration)

void MiscAPIRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName("cudaOccupancyMaxActiveBlocksPerMultiprocessor",
                      "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                      "cudaOccupancyMaxPotentialBlockSize");
  };

  MF.addMatcher(
      callExpr(callee(functionDecl(functionName()))).bind("FunctionCall"),
      this);
}
void MiscAPIRule::runRule(const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  ExprAnalysis EA(CE);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}
REGISTER_RULE(MiscAPIRule, PassKind::PK_Migration)

// Rule for types migration in var declarations and field declarations
void TypeInDeclRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      typeLoc(
          loc(qualType(hasDeclaration(namedDecl(hasAnyName(
              "cudaError", "curandStatus", "cublasStatus", "CUstream",
              "CUstream_st", "thrust::complex", "thrust::device_vector",
              "thrust::device_ptr", "thrust::device_reference",
              "thrust::host_vector", "cublasHandle_t", "CUevent_st", "__half",
              "half", "__half2", "half2", "cudaMemoryAdvise", "cudaError_enum",
              "cudaDeviceProp", "cudaPitchedPtr", "thrust::counting_iterator",
              "thrust::transform_iterator", "thrust::permutation_iterator",
              "thrust::iterator_difference", "cusolverDnHandle_t",
              "cusolverDnParams_t", "gesvdjInfo_t", "syevjInfo_t",
              "thrust::device_malloc_allocator", "thrust::divides",
              "thrust::tuple", "thrust::maximum", "thrust::multiplies",
              "thrust::plus", "cudaDataType_t", "cudaError_t", "CUresult",
              "CUdevice", "cudaEvent_t", "cublasStatus_t", "cuComplex",
              "cuFloatComplex", "cuDoubleComplex", "CUevent",
              "cublasFillMode_t", "cublasDiagType_t", "cublasSideMode_t",
              "cublasOperation_t", "cusolverStatus_t", "cusolverEigType_t",
              "cusolverEigMode_t", "curandStatus_t", "cudaStream_t",
              "cusparseStatus_t", "cusparseDiagType_t", "cusparseFillMode_t",
              "cusparseIndexBase_t", "cusparseMatrixType_t", "cusparseAlgMode_t",
              "cusparseOperation_t", "cusparseMatDescr_t", "cusparseHandle_t",
              "CUcontext", "cublasPointerMode_t", "cusparsePointerMode_t",
              "cublasGemmAlgo_t", "cusparseSolveAnalysisInfo_t", "cudaDataType",
              "cublasDataType_t", "curandState_t", "curandState",
              "curandStateXORWOW_t", "curandStateXORWOW",
              "curandStatePhilox4_32_10_t", "curandStatePhilox4_32_10",
              "curandStateMRG32k3a_t", "curandStateMRG32k3a", "thrust::minus",
              "thrust::negate", "thrust::logical_or", "thrust::equal_to",
              "thrust::less", "cudaSharedMemConfig", "curandGenerator_t",
              "curandRngType_t", "cufftHandle", "cufftReal", "cufftDoubleReal",
              "cufftComplex", "cufftDoubleComplex", "cufftResult_t",
              "cufftResult", "cufftType_t", "cufftType", "thrust::pair",
              "CUdeviceptr", "cudaDeviceAttr", "CUmodule", "CUjit_option",
              "CUfunction", "cudaMemcpyKind", "cudaComputeMode",
              "__nv_bfloat16", "cooperative_groups::__v1::thread_group",
              "cooperative_groups::__v1::thread_block_tile",
              "cooperative_groups::__v1::thread_block", "libraryPropertyType_t",
              "libraryPropertyType", "cudaDataType_t", "cudaDataType",
              "cublasComputeType_t", "cublasAtomicsMode_t", "CUmem_advise_enum",
              "CUmem_advise", "thrust::tuple_element", "thrust::tuple_size",
              "cublasMath_t", "cudaPointerAttributes", "thrust::zip_iterator",
              "cusolverEigRange_t", "cudaUUID_t", "cusolverDnFunction_t",
              "cusolverAlgMode_t", "cusparseIndexType_t", "cusparseFormat_t",
              "cusparseDnMatDescr_t", "cusparseOrder_t", "cusparseDnVecDescr_t",
              "cusparseConstDnVecDescr_t", "cusparseSpMatDescr_t",
              "cusparseSpMMAlg_t", "cusparseSpMVAlg_t", "cusparseSpGEMMDescr_t",
              "cusparseSpSVDescr_t", "cusparseSpGEMMAlg_t",
              "cusparseSpSVAlg_t", "cudaFuncAttributes"))))))
          .bind("cudaTypeDef"),
      this);
  MF.addMatcher(varDecl(hasType(classTemplateSpecializationDecl(
                            hasAnyTemplateArgument(refersToType(hasDeclaration(
                                namedDecl(hasName("use_default"))))))))
                    .bind("useDefaultVarDeclInTemplateArg"),
                this);
}

template <typename T>
bool getLocation(const Type *TypePtr, SourceLocation &SL) {
  auto TType = TypePtr->getAs<T>();
  if (TType) {
    auto TypeDecl = TType->getDecl();
    if (TypeDecl) {
      SL = TypeDecl->getLocation();
      return true;
    } else {
      return false;
    }
  }
  return false;
}

bool getTypeDeclLocation(const Type *TypePtr, SourceLocation &SL) {
  if (getLocation<EnumType>(TypePtr, SL)) {
    return true;
  } else if (getLocation<TypedefType>(TypePtr, SL)) {
    return true;
  } else if (getLocation<RecordType>(TypePtr, SL)) {
    return true;
  }
  return false;
}

bool getTemplateTypeReplacement(std::string TypeStr, std::string &Replacement,
                                unsigned &Len) {
  auto P1 = TypeStr.find('<');
  if (P1 != std::string::npos) {
    auto P2 = Replacement.find('<');
    if (P2 != std::string::npos) {
      Replacement = Replacement.substr(0, P2);
    }
    Len = P1;
    return true;
  }
  return false;
}

bool isAuto(const char *StrChar, unsigned Len) {
  return std::string(StrChar, Len) == "auto";
}

void insertComplexHeader(SourceLocation SL, std::string &Replacement) {
  if (SL.isValid() && Replacement.substr(0, 12) == "std::complex") {
    DpctGlobalInfo::getInstance().insertHeader(SL, HT_Complex);
  }
}

bool TypeInDeclRule::replaceTemplateSpecialization(
    SourceManager *SM, LangOptions &LOpts, SourceLocation BeginLoc,
    const TemplateSpecializationTypeLoc TSL) {

  for (unsigned i = 0; i < TSL.getNumArgs(); ++i) {
    auto ArgLoc = TSL.getArgLoc(i);
    if (ArgLoc.getArgument().getKind() != TemplateArgument::Type)
      continue;
    auto TSI = ArgLoc.getTypeSourceInfo();
    if (!TSI)
      continue;
    auto UTL = TSI->getTypeLoc().getUnqualifiedLoc();

    if (UTL.getTypeLocClass() == clang::TypeLoc::Elaborated) {
      auto ETC = UTL.getAs<ElaboratedTypeLoc>();

      auto ETBeginLoc = ETC.getQualifierLoc().getBeginLoc();
      auto ETEndLoc = ETC.getQualifierLoc().getEndLoc();

      if (ETBeginLoc.isInvalid() || ETEndLoc.isInvalid())
        continue;

      const char *Start = SM->getCharacterData(ETBeginLoc);
      const char *End = SM->getCharacterData(ETEndLoc);
      auto TyLen = End - Start;
      if (TyLen <= 0)
        return false;

      std::string RealTypeNameStr(Start, TyLen);

      auto Pos = RealTypeNameStr.find('<');
      if (Pos != std::string::npos) {
        RealTypeNameStr = RealTypeNameStr.substr(0, Pos);
        TyLen = Pos;
      }

      requestHelperFeatureForTypeNames(RealTypeNameStr);
      std::string Replacement =
          MapNames::findReplacedName(MapNames::TypeNamesMap, RealTypeNameStr);
      insertHeaderForTypeRule(RealTypeNameStr, ETBeginLoc);

      if (!Replacement.empty()) {
        SrcAPIStaticsMap[RealTypeNameStr]++;
        emplaceTransformation(
            new ReplaceText(ETBeginLoc, TyLen, std::move(Replacement)));
      }
    }
  }

  Token Tok;
  Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
  if (!Tok.isAnyIdentifier()) {
    return false;
  }

  auto TypeNameStr = Tok.getRawIdentifier().str();
  // skip to the next identifier after keyword "typename" or "const"
  if (TypeNameStr == "typename" || TypeNameStr == "const") {
    Tok = Lexer::findNextToken(BeginLoc, *SM, LOpts).value();
    BeginLoc = Tok.getLocation();
  }
  auto LAngleLoc = TSL.getLAngleLoc();

  const char *Start = SM->getCharacterData(BeginLoc);
  const char *End = SM->getCharacterData(LAngleLoc);
  auto TyLen = End - Start;
  if (TyLen <= 0)
    return false;

  const std::string RealTypeNameStr(Start, TyLen);

  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
      RealTypeNameStr.find("device_malloc_allocator") != std::string::npos) {
    report(BeginLoc, Diagnostics::KNOWN_UNSUPPORTED_TYPE, false,
            RealTypeNameStr);
    return true;
  }

  requestHelperFeatureForTypeNames(RealTypeNameStr);
  std::string Replacement =
      MapNames::findReplacedName(MapNames::TypeNamesMap, RealTypeNameStr);
  insertHeaderForTypeRule(RealTypeNameStr, BeginLoc);
  if (!Replacement.empty()) {
    insertComplexHeader(BeginLoc, Replacement);
    emplaceTransformation(
        new ReplaceText(BeginLoc, TyLen, std::move(Replacement)));
    return true;
  }
  return false;
}

// There's no AST matcher for dealing with DependentNameTypeLocs so
// it is handled 'manually'
bool TypeInDeclRule::replaceDependentNameTypeLoc(SourceManager *SM,
                                                 LangOptions &LOpts,
                                                 const TypeLoc *TL) {
  auto D = DpctGlobalInfo::findAncestor<Decl>(TL);
  TypeSourceInfo *TSI = nullptr;
  if (auto TD = dyn_cast<TypedefDecl>(D))
    TSI = TD->getTypeSourceInfo();
  else if (auto VD = dyn_cast<VarDecl>(D))
    TSI = VD->getTypeSourceInfo();
  else if (auto FD = dyn_cast<FieldDecl>(D))
    TSI = FD->getTypeSourceInfo();
  else if (auto TAD = dyn_cast<TypeAliasDecl>(D))
    TSI = TAD->getTypeSourceInfo();
  else
    return false;

  auto TTL = TSI->getTypeLoc();
  auto SR = TTL.getSourceRange();
  auto DTL = TTL.getAs<DependentNameTypeLoc>();
  if (!DTL)
    return false;

  auto NNSL = DTL.getQualifierLoc();
  auto NNTL = NNSL.getTypeLoc();

  auto BeginLoc = SR.getBegin();
  if (NNTL.getTypeLocClass() == clang::TypeLoc::TemplateSpecialization &&
      NNTL.getBeginLoc() == TL->getBeginLoc()) {
    auto TSL = NNTL.getUnqualifiedLoc().getAs<TemplateSpecializationTypeLoc>();
    if (replaceTemplateSpecialization(SM, LOpts, BeginLoc, TSL)) {
      // Check if "::type" needs replacement (only needed for
      // thrust::iterator_difference)
      Token Tok;
      Lexer::getRawToken(SR.getEnd(), Tok, *SM, LOpts, true);
      auto TypeNameStr =
          Tok.isAnyIdentifier() ? Tok.getRawIdentifier().str() : "";
      Lexer::getRawToken(TSL.getBeginLoc(), Tok, *SM, LOpts, true);
      auto TemplateNameStr =
          Tok.isAnyIdentifier() ? Tok.getRawIdentifier().str() : "";
      if (TypeNameStr == "type" && TemplateNameStr == "iterator_difference") {
        emplaceTransformation(
            new ReplaceText(SR.getEnd(), 4, std::string("difference_type")));
      }
      return true;
    }
  }
  return false;
}

// Make the necessary replacements for thrust::transform_iterator.
// The mapping requires swapping of the two template parameters, i.e.
//   thrust::transform_iterator<Functor, Iterator> ->
//     oenapi::dpl::transform_iterator<Iterator, Functor>
// This is a special transformation, because it requires the template
// parameters to be processed as part of the top level processing of
// the transform_iterator itself.  Simply processing the TypeLocs
// representing the template arguments when they are matched would
// result in wrong replacements being produced.
//
// For example:
//   thrust::transform_iterator<F, thrust::transform_iterator<F,I>>
// Should produce:
//   oneapi::dpl::transform_iterator<oneapi::dpl::transform_iterator<I,F>, F>
//
// The processing is therefore done by recursively walking all the
// TypeLocs that can be reached from the template arguments, and
// marking them as processed, so they won't be processed again, when
// their TypeLocs are matched by the matcher
bool TypeInDeclRule::replaceTransformIterator(SourceManager *SM,
                                              LangOptions &LOpts,
                                              const TypeLoc *TL) {

  // Local helper functions

  auto getFileLoc = [&](SourceLocation Loc) -> SourceLocation {
    // The EndLoc of some TypeLoc objects are Extension Locs, even
    // when the BeginLoc is a regular FileLoc.  This seems to happen
    // when the last typearg in a template specialization
    // is itself a template type.  For example.
    // SomeType<T1, AnotherType<T2>>.  The EndLoc for the TypeLoc for
    // AnotherType<T2> is an extension Loc.
    return SM->getFileLoc(Loc);
  };

  // Get the string from the source between [B, E].  The E location
  // is extended to the end of the token.  Special handling of the
  // '>' token is required in case it's followed by another '>'
  // For example: T<F, I<X>>
  // Without the special case condition, '>>' is considered one token
  auto getStr = [&](SourceLocation B, SourceLocation E) {
    B = getFileLoc(B);
    E = getFileLoc(E);
    if (*(SM->getCharacterData(E)) == '>')
      E = E.getLocWithOffset(1);
    else
      E = Lexer::getLocForEndOfToken(E, 0, *SM, LOpts);
    return std::string(SM->getCharacterData(B), SM->getCharacterData(E));
  };

  // Strip the 'typename' keyword when used in front of template types
  // This is necessary when looking up the typename string in the TypeNamesMap
  auto stripTypename = [](std::string &Str) {
    if (Str.substr(0, 8) == "typename") {
      Str = Str.substr(8);
      Str.erase(Str.begin(), std::find_if(Str.begin(), Str.end(), [](char ch) {
                  return !std::isspace(ch);
                }));
      return true;
    }
    return false;
  };

  // Get the Typename without potential template arguments.
  // For example:
  //   thrust::transform_iterator<F, I>
  //     -> thrust::transform_iterator
  auto getBaseTypeName = [&](const TypeLoc *TL) -> std::string {
    std::string Name = getStr(TL->getBeginLoc(), TL->getEndLoc());
    auto LAnglePos = Name.find("<");
    if (LAnglePos != std::string::npos)
      return Name.substr(0, LAnglePos);
    else
      return Name;
  };
  // Get the mapped typename, if one exists.  If not return the input
  auto mapName = [&](std::string Name) -> std::string {
    std::string NameToMap = Name;
    bool Stripped = stripTypename(NameToMap);
    std::string Replacement =
        MapNames::findReplacedName(MapNames::TypeNamesMap, NameToMap);
    insertHeaderForTypeRule(NameToMap, TL->getBeginLoc());
    requestHelperFeatureForTypeNames(NameToMap);
    if (Replacement.empty())
      return Name;
    else if (Stripped)
      return std::string("typename ") + Replacement;
    else
      return Replacement;
  };

  // Returns whether a TypeLoc has a template specialization, if
  // so the specialization is returned as well
  auto hasTemplateSpecialization =
      [&](const TypeLoc *TL, TemplateSpecializationTypeLoc &TSTL) -> bool {
    if (TL->getTypeLocClass() == clang::TypeLoc::TemplateSpecialization) {
      TSTL = TL->getAs<TemplateSpecializationTypeLoc>();
      return true;
    }
    if (TL->getTypeLocClass() == clang::TypeLoc::Elaborated) {
      auto ETL = TL->getAs<ElaboratedTypeLoc>();
      auto NTL = ETL.getNamedTypeLoc();
      if (NTL.getTypeLocClass() == clang::TypeLoc::TemplateSpecialization) {
        TSTL = NTL.getUnqualifiedLoc().getAs<TemplateSpecializationTypeLoc>();
        return true;
      } else
        return false;
    }
    return false;
  };

  // Returns whether a TypeLoc represents the thrust::transform_iterator
  // type with exactly 2 template arguments
  auto isTransformIterator = [&](const TypeLoc *TL) -> bool {
    TemplateSpecializationTypeLoc TSTL;
    if (hasTemplateSpecialization(TL, TSTL)) {
      std::string TypeStr = getStr(TSTL.getBeginLoc(), TSTL.getBeginLoc());
      if (TypeStr == "transform_iterator" && TSTL.getNumArgs() == 2) {
        return true;
      }
    }
    return false;
  };

  // Returns the full string replacement for a TypeLoc.  If necessary
  // template arguments are recursively walked to get potential replacement
  // for those as well.
  std::function<std::string(const TypeLoc *)> getNewTypeStr =
      [&](const TypeLoc *TL) -> std::string {
    std::string BaseTypeStr = getBaseTypeName(TL);
    std::string NewBaseTypeStr = mapName(BaseTypeStr);
    TemplateSpecializationTypeLoc TSTL;
    bool hasTSTL = hasTemplateSpecialization(TL, TSTL);
    // Mark this TL as having been processed
    ProcessedTypeLocs.insert(*TL);
    if (!hasTSTL) {
      // Not a template specialization, so recursion can terminate
      return NewBaseTypeStr;
    }
    // Mark the TSTL TypeLoc as having been processed
    ProcessedTypeLocs.insert(TSTL);
    if (isTransformIterator(TL) &&
        TSTL.getArgLoc(0).getArgument().getKind() == TemplateArgument::Type &&
        TSTL.getArgLoc(1).getArgument().getKind() == TemplateArgument::Type) {
      // Two template arguments must be swapped
      auto TSI1 = TSTL.getArgLoc(0).getTypeSourceInfo();
      auto TSI2 = TSTL.getArgLoc(1).getTypeSourceInfo();
      if (TSI1 && TSI2) {
        auto Arg1 = TSI1->getTypeLoc();
        auto Arg2 = TSI2->getTypeLoc();
        std::string Arg1Str = getNewTypeStr(&Arg1);
        std::string Arg2Str = getNewTypeStr(&Arg2);
        return NewBaseTypeStr + "<" + Arg2Str + ", " + Arg1Str + ">";
      }
    }
    // Recurse down through the template arguments
    std::string NewTypeStr = NewBaseTypeStr + "<";
    for (unsigned i = 0; i < TSTL.getNumArgs(); ++i) {
      std::string ArgStr;
      if (TSTL.getArgLoc(i).getArgument().getKind() == TemplateArgument::Type) {
        if (auto TSI = TSTL.getArgLoc(i).getTypeSourceInfo()) {
          auto ArgLoc = TSI->getTypeLoc();
          ArgStr = getNewTypeStr(&ArgLoc);
        }
      } else {
        ExprAnalysis EA;
        EA.analyze(TSTL.getArgLoc(i));
        ArgStr = EA.getReplacedString();
      }
      if (i != 0)
        NewTypeStr += ", ";
      NewTypeStr += ArgStr;
    }
    NewTypeStr += ">";
    return NewTypeStr;
  };

  // Main function:
  // Perform the complete replacement for the input TypeLoc.
  // TypeLocs that are being processed during the walk are inserted
  // into the ProcessedTypeLocs set, to prevent further processing
  // by the main matcher function
  if (!isTransformIterator(TL)) {
    return false;
  }
  std::string NewTypeStr = getNewTypeStr(TL);
  emplaceTransformation(new ReplaceToken(getFileLoc(TL->getBeginLoc()),
                                         getFileLoc(TL->getEndLoc()),
                                         std::move(NewTypeStr)));
  return true;
}

void TypeInDeclRule::processCudaStreamType(const DeclaratorDecl *DD) {
  auto SD = getAllDecls(DD);

  auto replaceInitParam = [&](const clang::Expr *replExpr) {
    if (!replExpr)
      return;

    if (auto type = DpctGlobalInfo::getUnqualifiedTypeName(replExpr->getType());
        !(type == "CUstream" || type == "cudaStream_t"))
      return;

    if (isDefaultStream(replExpr)) {
      int Index = getPlaceholderIdx(replExpr);
      if (Index == 0) {
        Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      }
      buildTempVariableMap(Index, replExpr, HelperFuncType::HFT_DefaultQueue);
      std::string Repl = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
      emplaceTransformation(new ReplaceStmt(replExpr, "&" + Repl));
    }
  };

  for (auto It = SD.begin(); It != SD.end(); ++It) {
    const clang::Expr *replExpr = nullptr;
    if (const auto VD = dyn_cast<clang::VarDecl>(*It))
      replExpr = VD->getInit();
    else if (const auto FD = dyn_cast<clang::FieldDecl>(*It))
      replExpr = FD->getInClassInitializer();

    if (!replExpr)
      continue;

    if (const auto VarInitExpr = dyn_cast<InitListExpr>(replExpr)) {
      auto arrayReplEXpr = VarInitExpr->inits();
      for (auto replExpr : arrayReplEXpr) {
        replaceInitParam(replExpr);
      }
      return;
    }

    replaceInitParam(replExpr);
  }
}

void TypeInDeclRule::runRule(const MatchFinder::MatchResult &Result) {
  SourceManager *SM = Result.SourceManager;
  auto LOpts = Result.Context->getLangOpts();

  if (auto TL = getNodeAsType<TypeLoc>(Result, "cudaTypeDef")) {

    // if TL is the T in
    // template<typename T> void foo(T a);
    if (TL->getType()->getTypeClass() == clang::Type::SubstTemplateTypeParm ||
        TL->getBeginLoc().isInvalid()) {
      return;
    }

    if(isCapturedByLambda(TL))
      return;

    auto TypeStr =
        DpctGlobalInfo::getTypeName(TL->getType().getUnqualifiedType());

    if (auto FD = DpctGlobalInfo::getParentFunction(TL)) {
      if (FD->isImplicit())
        return;
    }

    if (ProcessedTypeLocs.find(*TL) != ProcessedTypeLocs.end())
      return;

    // Try to migrate cudaSuccess to sycl::info::event_command_status if it is
    // used in cases like "cudaSuccess == cudaEventQuery()".
    if (EventAPICallRule::getEventQueryTraversal().startFromTypeLoc(*TL))
      return;

    // when the following code is not in AnalysisScope
    // #define MACRO_SHOULD_NOT_BE_MIGRATED (MatchedType)3
    // Even if MACRO_SHOULD_NOT_BE_MIGRATED used in AnalysisScope, DPCT should not
    // migrate MatchedType.
    if (!DpctGlobalInfo::isInAnalysisScope(SM->getSpellingLoc(TL->getBeginLoc())) &&
        isPartOfMacroDef(SM->getSpellingLoc(TL->getBeginLoc()),
                         SM->getSpellingLoc(TL->getEndLoc()))) {
      return;
    }

    auto Range = getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc());
    auto BeginLoc = Range.getBegin();
    auto EndLoc = Range.getEnd();

    // WA for concatinated macro token
    if (SM->isWrittenInScratchSpace(SM->getSpellingLoc(TL->getBeginLoc()))) {
      BeginLoc = SM->getExpansionRange(TL->getBeginLoc()).getBegin();
      EndLoc = SM->getExpansionRange(TL->getBeginLoc()).getEnd();
    }

    std::string CanonicalTypeStr =
      DpctGlobalInfo::getUnqualifiedTypeName(
        TL->getType().getCanonicalType());
    StringRef CanonicalTypeStrRef(CanonicalTypeStr);
    if (CanonicalTypeStrRef.startswith(
            "cooperative_groups::__v1::thread_block_tile<")) {
      if (auto ETL = TL->getUnqualifiedLoc().getAs<ElaboratedTypeLoc>()) {
        SourceLocation Begin = ETL.getBeginLoc();
        SourceLocation End = ETL.getEndLoc();
        if (Begin.isMacroID() || End.isMacroID())
          return;
        End = End.getLocWithOffset(Lexer::MeasureTokenLength(
            End, *SM, DpctGlobalInfo::getContext().getLangOpts()));
        if (End.isMacroID())
          return;
        if (CanonicalTypeStrRef.equals(
                "cooperative_groups::__v1::thread_block_tile<32>")) {
          emplaceTransformation(new ReplaceText(
              Begin, End.getRawEncoding() - Begin.getRawEncoding(),
              MapNames::getClNamespace() + "sub_group"));
        } else if (DpctGlobalInfo::useLogicalGroup()) {
          emplaceTransformation(new ReplaceText(
              Begin, End.getRawEncoding() - Begin.getRawEncoding(),
              MapNames::getDpctNamespace() + "experimental::logical_group"));
          requestFeature(HelperFeatureEnum::device_ext);
        }
        return;
      }
    }

    if (CanonicalTypeStr == "cooperative_groups::__v1::thread_group" ||
        CanonicalTypeStr == "cooperative_groups::__v1::thread_block") {
      if (DpctGlobalInfo::findAncestor<clang::CompoundStmt>(TL) &&
          DpctGlobalInfo::findAncestor<clang::FunctionDecl>(TL))
        return;
      if (auto ETL = TL->getUnqualifiedLoc().getAs<ElaboratedTypeLoc>()) {
        SourceLocation Begin = ETL.getBeginLoc();
        SourceLocation End = ETL.getEndLoc();
        if (Begin.isMacroID() || End.isMacroID())
          return;
        End = End.getLocWithOffset(Lexer::MeasureTokenLength(
            End, *SM, DpctGlobalInfo::getContext().getLangOpts()));
        if (End.isMacroID())
          return;
        const auto *FD = DpctGlobalInfo::getParentFunction(TL);
        if (!FD)
          return;
        auto DFI = DeviceFunctionDecl::LinkRedecls(FD);
        auto Index = DpctGlobalInfo::getCudaKernelDimDFIIndexThenInc();
        DpctGlobalInfo::insertCudaKernelDimDFIMap(Index, DFI);
        std::string group_type = "";
        if (DpctGlobalInfo::useLogicalGroup())
          group_type = MapNames::getDpctNamespace() + "experimental::group_base";
        if (CanonicalTypeStr == "cooperative_groups::__v1::thread_block")
          group_type = MapNames::getClNamespace() + "group";
        if (!group_type.empty())
          emplaceTransformation(new ReplaceText(
              Begin, End.getRawEncoding() - Begin.getRawEncoding(),
              group_type + "<{{NEEDREPLACEG" + std::to_string(Index) + "}}>"));
        return;
      }
    }

    if (replaceDependentNameTypeLoc(SM, LOpts, TL)) {
      return;
    }

    if (replaceTransformIterator(SM, LOpts, TL)) {
      return;
    }

    if (TL->getTypeLocClass() == clang::TypeLoc::Elaborated) {
      auto ETC = TL->getAs<ElaboratedTypeLoc>();
      auto NTL = ETC.getNamedTypeLoc();
      if (NTL.getTypeLocClass() == clang::TypeLoc::TemplateSpecialization) {
        auto TSL =
            NTL.getUnqualifiedLoc().getAs<TemplateSpecializationTypeLoc>();

        if (replaceTemplateSpecialization(SM, LOpts, BeginLoc, TSL)) {
          return;
        }
      } else if (NTL.getTypeLocClass() == clang::TypeLoc::Record) {
        if (TypeStr == "__nv_bfloat16" && !DpctGlobalInfo::useBFloat16()) {
          return;
        }

        auto TSL = NTL.getUnqualifiedLoc().getAs<RecordTypeLoc>();

        const std::string TyName =
            dpct::DpctGlobalInfo::getTypeName(TSL.getType());
        std::string Replacement =
            MapNames::findReplacedName(MapNames::TypeNamesMap, TyName);
        requestHelperFeatureForTypeNames(TyName);
        insertHeaderForTypeRule(TyName, TL->getBeginLoc());

        if (!Replacement.empty()) {
          SrcAPIStaticsMap[TyName]++;
          emplaceTransformation(new ReplaceToken(BeginLoc, TSL.getEndLoc(),
                                                 std::move(Replacement)));
          return;
        }
      }
    } else if (TL->getTypeLocClass() == clang::TypeLoc::Qualified) {
      // To process the case like "typename
      // thrust::device_vector<int>::iterator itr;".
      auto ETL = TL->getUnqualifiedLoc().getAs<ElaboratedTypeLoc>();
      if (ETL) {
        auto NTL = ETL.getNamedTypeLoc();
        if (NTL.getTypeLocClass() == clang::TypeLoc::TemplateSpecialization) {
          auto TSL =
              NTL.getUnqualifiedLoc().getAs<TemplateSpecializationTypeLoc>();
          if (replaceTemplateSpecialization(SM, LOpts, BeginLoc, TSL)) {
            return;
          }
        }
      }
    } else if (TL->getTypeLocClass() ==
               clang::TypeLoc::TemplateSpecialization) {
      // To process cases like "tuple_element<0, TupleTy>" in
      // "typename thrust::tuple_element<0, TupleTy>::type"
      auto TSL = TL->getAs<TemplateSpecializationTypeLoc>();
      auto Parents = Result.Context->getParents(TSL);
      if (!Parents.empty()) {
        if (auto NNSL = Parents[0].get<NestedNameSpecifierLoc>()) {
          if (replaceTemplateSpecialization(SM, LOpts, NNSL->getBeginLoc(),
                                            TSL)) {
            return;
          }
        }
      }
    }

    std::string Str =
        MapNames::findReplacedName(MapNames::TypeNamesMap, TypeStr);
    insertHeaderForTypeRule(TypeStr, BeginLoc);
    requestHelperFeatureForTypeNames(TypeStr);
    if (Str.empty()) {
      auto Itr = MapNames::DeviceRandomGeneratorTypeMap.find(TypeStr);
      if (Itr != MapNames::DeviceRandomGeneratorTypeMap.end()) {
        if (TypeStr == "curandState_t" || TypeStr == "curandState" ||
            TypeStr == "curandStateXORWOW_t" ||
            TypeStr == "curandStateXORWOW") {
          report(BeginLoc, Diagnostics::DIFFERENT_GENERATOR, false);
        }
        Str = Itr->second;
      }
    }
    // Add '#include <complex>' directive to the file only once
    if (TypeStr == "cuComplex" || TypeStr == "cuDoubleComplex" ||
        TypeStr == "cuFloatComplex") {
      DpctGlobalInfo::getInstance().insertHeader(BeginLoc, HT_Complex);
    }
    // Add '#include <dpct/lib_common_utils.hpp>' directive to the file only
    // once
    if (TypeStr == "libraryPropertyType" ||
        TypeStr == "libraryPropertyType_t" || TypeStr == "cudaDataType_t" ||
        TypeStr == "cudaDataType" || TypeStr == "cublasComputeType_t") {
      DpctGlobalInfo::getInstance().insertHeader(BeginLoc,
                                                 HT_DPCT_COMMON_Utils);
    }

    const DeclaratorDecl *DD = nullptr;
    const VarDecl *VarD = DpctGlobalInfo::findAncestor<VarDecl>(TL);
    const FieldDecl *FieldD = DpctGlobalInfo::findAncestor<FieldDecl>(TL);
    const FunctionDecl *FD = DpctGlobalInfo::findAncestor<FunctionDecl>(TL);
    if (FD &&
        (FD->hasAttr<CUDADeviceAttr>() || FD->hasAttr<CUDAGlobalAttr>())) {
      if (DpctGlobalInfo::getUnqualifiedTypeName(TL->getType()) == "cublasHandle_t") {
        report(BeginLoc, Diagnostics::HANDLE_IN_DEVICE, false, TypeStr);
        return;
      }
    }
    if (VarD) {
      DD = VarD;
    } else if (FieldD) {
      DD = FieldD;
    } else if (FD) {
      DD = FD;
    }

    if (DD) {
      if (TL->getType().getCanonicalType()->isPointerType()) {
        const auto *PtrTy =
            TL->getType().getCanonicalType()->getAs<PointerType>();
        if (!PtrTy)
          return;
        if (PtrTy->getPointeeType()->isRecordType()) {
          const auto *RecordTy = PtrTy->getPointeeType()->getAs<RecordType>();
          if (!RecordTy)
            return;
          const auto *RD = RecordTy->getAsRecordDecl();
          if (!RD)
            return;
          if (RD->getName() == "CUstream_st" &&
              DpctGlobalInfo::isInCudaPath(RD->getBeginLoc()))
            processCudaStreamType(DD);
        }
      }
    }

    if (!Str.empty()) {
      SrcAPIStaticsMap[TypeStr]++;

      auto Len = Lexer::MeasureTokenLength(
          EndLoc, *SM, DpctGlobalInfo::getContext().getLangOpts());
      Len += SM->getDecomposedLoc(EndLoc).second -
             SM->getDecomposedLoc(BeginLoc).second;
      emplaceTransformation(new ReplaceText(BeginLoc, Len, std::move(Str)));
      return;
    }
  }

  if (auto VD =
          getNodeAsType<VarDecl>(Result, "useDefaultVarDeclInTemplateArg")) {
    auto TL = VD->getTypeSourceInfo()->getTypeLoc();

    auto TSTL = TL.getAsAdjusted<TemplateSpecializationTypeLoc>();
    if (!TSTL)
      return;
    auto TST = TSTL.getType()->getAsAdjusted<TemplateSpecializationType>();
    if (!TST || TST->template_arguments().empty())
      return;

    if (!DpctGlobalInfo::getTypeName(TST->template_arguments()[0].getAsType())
             .compare("thrust::use_default")) {
      auto ArgBeginLoc = TSTL.getArgLoc(0).getSourceRange().getBegin();
      auto ArgEndLoc = TSTL.getArgLoc(0).getSourceRange().getEnd();
      emplaceTransformation(new ReplaceToken(ArgBeginLoc, ArgEndLoc, ""));
    }
  }
}

REGISTER_RULE(TypeInDeclRule, PassKind::PK_Migration)

// Rule for types replacements in var. declarations.
void VectorTypeNamespaceRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(
                            anyOf(namedDecl(vectorTypeName()),
                                  typedefDecl(vectorTypeName()))))))
                    .bind("vectorTypeTL"),
                this);

  MF.addMatcher(
      cxxRecordDecl(isDirectlyDerivedFrom(hasAnyName(SUPPORTEDVECTORTYPENAMES)))
          .bind("inheritanceType"),
      this);

  auto Vec3Types = [&]() {
    return hasAnyName("char3", "uchar3", "short3", "ushort3", "int3", "uint3",
                      "long3", "ulong3", "float3", "double3", "longlong3",
                      "ulonglong3");
  };

  MF.addMatcher(stmt(sizeOfExpr(hasArgumentOfType(hasCanonicalType(
                         hasDeclaration(namedDecl(Vec3Types()))))))
                    .bind("SizeofVector3Warn"),
                this);
  MF.addMatcher(cxxRecordDecl(isDirectlyDerivedFrom(hasAnyName(
                                  "char1", "uchar1", "short1", "ushort1",
                                  "int1", "uint1", "long1", "ulong1", "float1",
                                  "longlong1", "ulonglong1", "double1", "__half_raw")))
                    .bind("inherit"),
                this);
  // Matcher for __half_raw implicitly convert to half.
  MF.addMatcher(
      declRefExpr(allOf(unless(hasParent(memberExpr())),
                        unless(hasParent(unaryOperator(hasOperatorName("&")))),
                        to(varDecl(hasType(qualType(hasDeclaration(
                                       namedDecl(hasAnyName("__half_raw"))))))),
                        hasParent(implicitCastExpr())))
          .bind("halfRawExpr"),
      this);
}

void VectorTypeNamespaceRule::runRule(const MatchFinder::MatchResult &Result) {
  SourceManager *SM = Result.SourceManager;
  if (auto TL = getNodeAsType<TypeLoc>(Result, "vectorTypeTL")) {
    if (TL->getBeginLoc().isInvalid())
      return;

    // To skip user-defined type.
    if (const auto *ND = getNamedDecl(TL->getTypePtr())) {
      auto Loc = ND->getBeginLoc();
      if (DpctGlobalInfo::isInAnalysisScope(Loc))
        return;
    }

    auto BeginLoc =
        getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc()).getBegin();

    bool IsInScratchspace = false;
    // WA for concatinated macro token
    if (SM->isWrittenInScratchSpace(SM->getSpellingLoc(TL->getBeginLoc()))) {
      BeginLoc = SM->getExpansionLoc(TL->getBeginLoc());
      IsInScratchspace = true;
    }

    const FieldDecl *FD = DpctGlobalInfo::findAncestor<FieldDecl>(TL);
    if (auto D = dyn_cast_or_null<CXXRecordDecl>(getParentDecl(FD))) {
      // To process cases like "union struct_union {float2 data;};".
      auto Type = FD->getType();
      if (D && D->isUnion() && !Type->isPointerType() && !Type->isArrayType()) {
        // To add a default member initializer list "{}" to the
        // vector variant member of the union, because a union contains a
        // non-static data member with a non-trivial default constructor, the
        // default constructor of the union will be deleted by default.
        auto Loc = FD->getEndLoc().getLocWithOffset(Lexer::MeasureTokenLength(
            FD->getEndLoc(), *SM, Result.Context->getLangOpts()));
        emplaceTransformation(new InsertText(Loc, "{}"));
      }
    }
    bool NeedRemoveVolatile = true;
    Token Tok;
    auto LOpts = Result.Context->getLangOpts();
    Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
    if (Tok.isAnyIdentifier()) {
      const std::string TypeStr = Tok.getRawIdentifier().str();
      if (TypeStr == "__nv_bfloat162" && !DpctGlobalInfo::useBFloat16()) {
        return;
      }
      std::string Str =
          MapNames::findReplacedName(MapNames::TypeNamesMap, TypeStr);
      insertHeaderForTypeRule(TypeStr, BeginLoc);
      requestHelperFeatureForTypeNames(TypeStr);
      if (!Str.empty()) {
        SrcAPIStaticsMap[TypeStr]++;
        emplaceTransformation(new ReplaceToken(BeginLoc, std::move(Str)));
      }
      if (TypeStr.back() == '1') {
        NeedRemoveVolatile = false;
      }
    }

    if (IsInScratchspace) {
      std::string TypeStr = TL->getType().getUnqualifiedType().getAsString();
      auto Begin = SM->getImmediateExpansionRange(TL->getBeginLoc()).getBegin();
      auto End = SM->getImmediateExpansionRange(TL->getEndLoc()).getEnd();
      if (TypeStr.back() == '1') {
        // Make (Begin, End) be the range of "##1"
        Begin = SM->getSpellingLoc(Begin);
        End = SM->getSpellingLoc(End);
        Begin = Begin.getLocWithOffset(Lexer::MeasureTokenLength(
            Begin, *SM, DpctGlobalInfo::getContext().getLangOpts()));
        End = End.getLocWithOffset(Lexer::MeasureTokenLength(
            End, *SM, DpctGlobalInfo::getContext().getLangOpts()));
        auto Length = SM->getFileOffset(End) - SM->getFileOffset(Begin);
        return emplaceTransformation(new ReplaceText(Begin, Length, ""));
      } else {
        // Make Begin be the begin of "MACROARG##1"
        Begin = SM->getSpellingLoc(Begin);
        return emplaceTransformation(
            new InsertText(Begin, MapNames::getClNamespace()));
      }
    }

    // check whether the vector has volatile qualifier, if so, remove the
    // qualifier and emit a warning.
    if (!NeedRemoveVolatile)
      return;
    const ValueDecl *VD = DpctGlobalInfo::findAncestor<ValueDecl>(TL);
    if (VD) {
      bool isPointerToVolatile = false;
      if (const auto PT = dyn_cast<PointerType>(VD->getType())) {
        isPointerToVolatile = PT->getPointeeType().isVolatileQualified();
      }
      if (isPointerToVolatile || VD->getType().isVolatileQualified()) {
        SourceLocation Loc = SM->getExpansionLoc(VD->getBeginLoc());
        report(Loc, Diagnostics::VOLATILE_VECTOR_ACCESS, false);

        // remove the volatile qualifier and trailing spaces
        Token Tok;
        Lexer::getRawToken(Loc, Tok, *SM,
                           DpctGlobalInfo::getContext().getLangOpts(), true);
        unsigned int EndLocOffset =
            SM->getDecomposedExpansionLoc(VD->getEndLoc()).second;
        while (SM->getDecomposedExpansionLoc(Tok.getEndLoc()).second <=
               EndLocOffset) {
          if (Tok.is(tok::TokenKind::raw_identifier) &&
              Tok.getRawIdentifier().str() == "volatile") {
            emplaceTransformation(new ReplaceText(
                Tok.getLocation(),
                getLenIncludingTrailingSpaces(
                    SourceRange(Tok.getLocation(), Tok.getEndLoc()), *SM),
                ""));
            break;
          }
          Lexer::getRawToken(Tok.getEndLoc(), Tok, *SM,
                             DpctGlobalInfo::getContext().getLangOpts(), true);
        }
      }
    }
  }

  if (auto RD = getNodeAsType<CXXRecordDecl>(Result, "inherit")) {
    report(RD->getBeginLoc(), Diagnostics::VECTYPE_INHERITATED, false);
  }

  if (const auto *UETT =
          getNodeAsType<UnaryExprOrTypeTraitExpr>(Result, "SizeofVector3Warn")) {

    // Ignore shared variables.
    // .e.g: __shared__ int a[sizeof(float3)], b[sizeof(float3)], ...;
    if (const auto *V = DpctGlobalInfo::findAncestor<VarDecl>(UETT)) {
      if (V->hasAttr<CUDASharedAttr>())
        return;
    }
    std::string argTypeName = DpctGlobalInfo::getTypeName(UETT->getTypeOfArgument());
    std::string argCanTypeName = DpctGlobalInfo::getTypeName(UETT->getTypeOfArgument().getCanonicalType());
    if (argTypeName != argCanTypeName)
      argTypeName += " (aka " + argCanTypeName + ")";

    report(UETT, Diagnostics::SIZEOF_WARNING, true, argTypeName);
  }
  // Runrule for __half_raw implicitly convert to half.
  if (auto DRE = getNodeAsType<DeclRefExpr>(Result, "halfRawExpr")) {
    ExprAnalysis EA;
    std::string Replacement;
    llvm::raw_string_ostream OS(Replacement);
    OS << MapNames::getClNamespace() + "bit_cast<" +
              MapNames::getClNamespace() + "half>(";
    EA.analyze(DRE);
    OS << EA.getReplacedString();
    OS << ")";
    OS.flush();
    emplaceTransformation(new ReplaceStmt(DRE, Replacement));
    return;
  }
}

REGISTER_RULE(VectorTypeNamespaceRule, PassKind::PK_Migration)

void VectorTypeMemberAccessRule::registerMatcher(MatchFinder &MF) {
  auto memberAccess = [&]() {
    return hasObjectExpression(
        hasType(qualType(anyOf(hasCanonicalType(recordType(hasDeclaration(
                                   cxxRecordDecl(vectorTypeName())))),
                               hasDeclaration(namedDecl(vectorTypeName()))))));
  };

  // int2.x => int2.x()
  MF.addMatcher(
      memberExpr(allOf(memberAccess(), unless(hasParent(binaryOperator(allOf(
                                           hasLHS(memberExpr(memberAccess())),
                                           isAssignmentOperator()))))))
          .bind("VecMemberExpr"),
      this);

  // class A : int2{ void foo(){x = 3;}}
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(
                               pointsTo(cxxRecordDecl(vectorTypeName())))))
                    .bind("DerivedVecMemberExpr"),
                this);

  // int2.x += xxx => int2.x() += xxx
  MF.addMatcher(
      binaryOperator(allOf(hasLHS(memberExpr(memberAccess())
                                      .bind("VecMemberExprAssignmentLHS")),
                           isAssignmentOperator()))
          .bind("VecMemberExprAssignment"),
      this);

  // int2 *a; a->x = 1;
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(pointerType(pointee(qualType(
              anyOf(recordType(hasDeclaration(cxxRecordDecl(vectorTypeName()))),
                    hasDeclaration(namedDecl(vectorTypeName())))))))))
          .bind("VecMemberExprArrow"),
      this);

  // No inner filter is available for decltypeType(). Thus, this matcher will
  // match all decltypeType. Detail control flow for different types is in
  // runRule().
  MF.addMatcher(typeLoc(loc(decltypeType())).bind("TypeLoc"), this);
}

void VectorTypeMemberAccessRule::renameMemberField(const MemberExpr *ME) {
  ExprAnalysis EA(ME);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}

void VectorTypeMemberAccessRule::runRule(
    const MatchFinder::MatchResult &Result) {
  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "VecMemberExpr")) {
    auto Parents = Result.Context->getParents(*ME);
    if (Parents.size() == 0) {
      return;
    }
    renameMemberField(ME);
  }

  if (auto ME = getNodeAsType<MemberExpr>(Result, "DerivedVecMemberExpr")) {
    renameMemberField(ME);
  }

  if (auto ME =
          getNodeAsType<MemberExpr>(Result, "VecMemberExprAssignmentLHS")) {
    renameMemberField(ME);
  }

  if (auto ME = getNodeAsType<MemberExpr>(Result, "VecMemberExprArrow")) {
    renameMemberField(ME);
  }

  if (auto *TL = getNodeAsType<DecltypeTypeLoc>(Result, "TypeLoc")) {
    ExprAnalysis EA;
    EA.analyze(*TL);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

REGISTER_RULE(VectorTypeMemberAccessRule, PassKind::PK_Migration)

namespace clang {
namespace ast_matchers {

AST_MATCHER(FunctionDecl, overloadedVectorOperator) {
  if (!DpctGlobalInfo::isInAnalysisScope(Node.getBeginLoc()))
    return false;

  switch (Node.getOverloadedOperator()) {
  default: {
    return false;
  }
#define OVERLOADED_OPERATOR_MULTI(...)
#define OVERLOADED_OPERATOR(Name, ...)                                         \
  case OO_##Name: {                                                            \
    break;                                                                     \
  }
#include "clang/Basic/OperatorKinds.def"
#undef OVERLOADED_OPERATOR
#undef OVERLOADED_OPERATOR_MULTI
  }

  // Check parameter is vector type
  auto SupportedParamType = [&](const ParmVarDecl *PD) {
    if (!PD)
      return false;
    const IdentifierInfo *IDInfo =
        PD->getOriginalType().getBaseTypeIdentifier();
    if (!IDInfo)
      return false;

    const std::string TypeName = IDInfo->getName().str();
    return (MapNames::SupportedVectorTypes.find(TypeName) !=
            MapNames::SupportedVectorTypes.end());
  };

  // As long as one parameter is vector type
  for (unsigned i = 0, End = Node.getNumParams(); i != End; ++i) {
    if (SupportedParamType(Node.getParamDecl(i))) {
      return true;
    }
  }

  return false;
}

} // namespace ast_matchers
} // namespace clang

void VectorTypeOperatorRule::registerMatcher(MatchFinder &MF) {
  auto vectorTypeOverLoadedOperator = [&]() {
    return functionDecl(overloadedVectorOperator(),
                        unless(hasAncestor(cxxRecordDecl())));
  };

  // Matches user overloaded operator declaration
  MF.addMatcher(vectorTypeOverLoadedOperator().bind("overloadedOperatorDecl"),
                this);

  // Matches call of user overloaded operator
  MF.addMatcher(cxxOperatorCallExpr(callee(vectorTypeOverLoadedOperator()))
                    .bind("callOverloadedOperator"),
                this);
}

const char VectorTypeOperatorRule::NamespaceName[] =
    "dpct_operator_overloading";

void VectorTypeOperatorRule::MigrateOverloadedOperatorDecl(
    const MatchFinder::MatchResult &Result, const FunctionDecl *FD) {
  if (!FD)
    return;

  // Helper function to get the scope of function declaration
  // Eg.:
  //
  //    void test();
  //   ^            ^
  //   |            |
  // Begin         End
  //
  //    void test() {}
  //   ^              ^
  //   |              |
  // Begin           End
  auto GetFunctionSourceRange = [&](const SourceManager &SM,
                                    const SourceLocation &StartLoc,
                                    const SourceLocation &EndLoc) {
    const std::pair<FileID, unsigned> StartLocInfo =
        SM.getDecomposedExpansionLoc(StartLoc);
    llvm::StringRef Buffer(SM.getCharacterData(EndLoc));
    const std::string Str = std::string(";}");
    size_t Offset = Buffer.find_first_of(Str);
    assert(Offset != llvm::StringRef::npos);
    const std::pair<FileID, unsigned> EndLocInfo =
        SM.getDecomposedExpansionLoc(EndLoc.getLocWithOffset(Offset + 1));
    assert(StartLocInfo.first == EndLocInfo.first);

    return SourceRange(
        SM.getComposedLoc(StartLocInfo.first, StartLocInfo.second),
        SM.getComposedLoc(EndLocInfo.first, EndLocInfo.second));
  };

  // Add namespace to user overloaded operator declaration
  // double2& operator+=(double2& lhs, const double2& rhs)
  // =>
  // namespace dpct_operator_overloading {
  //
  // double2& operator+=(double2& lhs, const double2& rhs)
  //
  // }
  const auto &SM = *Result.SourceManager;
  const std::string NL = getNL(FD->getBeginLoc(), SM);

  std::ostringstream Prologue;
  // clang-format off
  Prologue << "namespace " << NamespaceName << " {" << NL
           << NL;
  // clang-format on

  std::ostringstream Epilogue;
  // clang-format off
  Epilogue << NL
           << "}  // namespace " << NamespaceName << NL
           << NL;
  // clang-format on
  SourceRange SR;
  auto P = getParentDecl(FD);
  // Deal with functions as well as function templates
  if (auto FTD = dyn_cast<FunctionTemplateDecl>(P)) {
    SR = GetFunctionSourceRange(SM, FTD->getBeginLoc(), FTD->getEndLoc());
  } else {
    SR = GetFunctionSourceRange(SM, FD->getBeginLoc(), FD->getEndLoc());
  }
  report(SR.getBegin(), Diagnostics::TRNA_WARNING_OVERLOADED_API_FOUND, false);
  emplaceTransformation(new InsertText(SR.getBegin(), Prologue.str()));
  emplaceTransformation(new InsertText(SR.getEnd(), Epilogue.str()));
}

void VectorTypeOperatorRule::MigrateOverloadedOperatorCall(
    const MatchFinder::MatchResult &Result, const CXXOperatorCallExpr *CE) {
  if (!CE)
    return;

  // Explicitly call user overloaded operator
  //
  // For non-assignment operator:
  // a == b
  // =>
  // dpct_operator_overloading::operator==(a, b)
  //
  // For assignment operator:
  // a += b
  // =>
  // dpct_operator_overloading::operator+=(a, b)
  if (!clang::getOperatorSpelling(CE->getOperator()))
    return;
  const std::string OperatorName =
      std::string(clang::getOperatorSpelling(CE->getOperator()));
  std::ostringstream FuncCall;

  FuncCall << NamespaceName << "::operator" << OperatorName;

  std::string OperatorReplacement = (CE->getNumArgs() == 1)
                                        ? /* Unary operator */ ""
                                        : /* Binary operator */ ",";
  emplaceTransformation(
      new ReplaceToken(CE->getOperatorLoc(), std::move(OperatorReplacement)));
  insertAroundStmt(CE, FuncCall.str() + "(", ")");
}

void VectorTypeOperatorRule::runRule(const MatchFinder::MatchResult &Result) {
  // Add namespace to user overloaded operator declaration
  MigrateOverloadedOperatorDecl(
      Result, getNodeAsType<FunctionDecl>(Result, "overloadedOperatorDecl"));

  // Explicitly call user overloaded operator
  MigrateOverloadedOperatorCall(Result, getNodeAsType<CXXOperatorCallExpr>(
                                            Result, "callOverloadedOperator"));
}

REGISTER_RULE(VectorTypeOperatorRule, PassKind::PK_Migration)

void ReplaceDim3CtorRule::registerMatcher(MatchFinder &MF) {
  // Find dim3 constructors which are part of different casts (representing
  // different syntaxes). This includes copy constructors. All constructors
  // will be visited once.
  MF.addMatcher(cxxConstructExpr(hasType(namedDecl(hasName("dim3"))),
                                 argumentCountIs(1),
                                 unless(hasAncestor(cxxConstructExpr(
                                     hasType(namedDecl(hasName("dim3")))))))
                    .bind("dim3Top"),
                this);

  MF.addMatcher(cxxConstructExpr(
                    hasType(namedDecl(hasName("dim3"))), argumentCountIs(3),
                    anyOf(hasParent(varDecl()), hasParent(exprWithCleanups())),
                    unless(hasParent(initListExpr())),
                    unless(hasAncestor(
                        cxxConstructExpr(hasType(namedDecl(hasName("dim3")))))))
                    .bind("dim3CtorDecl"),
                this);

  MF.addMatcher(
      cxxConstructExpr(hasType(namedDecl(hasName("dim3"))), argumentCountIs(3),
                       // skip fields in a struct.  The source loc is
                       // messed up (points to the start of the struct)
                       unless(hasParent(initListExpr())),
                       unless(hasAncestor(cxxRecordDecl())),
                       unless(hasParent(varDecl())),
                       unless(hasParent(exprWithCleanups())),
                       unless(hasAncestor(cxxConstructExpr(
                           hasType(namedDecl(hasName("dim3")))))))
          .bind("dim3CtorNoDecl"),
      this);

  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(anyOf(
                  namedDecl(hasAnyName("dim3")),
                  typedefDecl(hasAnyName("dim3")))))))
          .bind("dim3Type"),
      this);
}

ReplaceDim3Ctor *ReplaceDim3CtorRule::getReplaceDim3Modification(
    const MatchFinder::MatchResult &Result) {
  if (auto Ctor = getNodeAsType<CXXConstructExpr>(Result, "dim3CtorDecl")) {
    if(getParentKernelCall(Ctor))
      return nullptr;
    // dim3 a; or dim3 a(1);
    return new ReplaceDim3Ctor(Ctor, true /*isDecl*/);
  } else if (auto Ctor =
                 getNodeAsType<CXXConstructExpr>(Result, "dim3CtorNoDecl")) {
    if(getParentKernelCall(Ctor))
      return nullptr;
    // deflt = dim3(3);
    return new ReplaceDim3Ctor(Ctor, false /*isDecl*/);
  } else if (auto Ctor = getNodeAsType<CXXConstructExpr>(Result, "dim3Top")) {
    if(getParentKernelCall(Ctor))
      return nullptr;
    // dim3 d3_6_3 = dim3(ceil(test.x + NUM), NUM + test.y, NUM + test.z + NUM);
    if (auto A = ReplaceDim3Ctor::getConstructExpr(Ctor->getArg(0))) {
      // strip the top CXXConstructExpr, if there's a CXXConstructExpr further
      // down
      return new ReplaceDim3Ctor(Ctor, A);
    } else {
      // Copy constructor case: dim3 a(copyfrom)
      // No replacements are needed
      return nullptr;
    }
  }

  return nullptr;
}

void ReplaceDim3CtorRule::runRule(const MatchFinder::MatchResult &Result) {
  ReplaceDim3Ctor *R = getReplaceDim3Modification(Result);
  if (R) {
    emplaceTransformation(R);
  }

  if (auto TL = getNodeAsType<TypeLoc>(Result, "dim3Type")) {
    if (TL->getBeginLoc().isInvalid())
      return;

    auto BeginLoc =
        getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc()).getBegin();
    SourceManager *SM = Result.SourceManager;

    // WA for concatenated macro token
    if (SM->isWrittenInScratchSpace(SM->getSpellingLoc(TL->getBeginLoc()))) {
      BeginLoc = SM->getExpansionLoc(TL->getBeginLoc());
    }

    Token Tok;
    auto LOpts = Result.Context->getLangOpts();
    Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
    if (Tok.isAnyIdentifier()) {
      if (TL->getType()->isElaboratedTypeSpecifier()) {
        // To handle case like "struct cudaExtent extent;"
        auto ETC = TL->getUnqualifiedLoc().getAs<ElaboratedTypeLoc>();
        auto NTL = ETC.getNamedTypeLoc();

        if (NTL.getTypeLocClass() == clang::TypeLoc::Record) {
          auto TSL = NTL.getUnqualifiedLoc().getAs<RecordTypeLoc>();

          const std::string TyName =
              dpct::DpctGlobalInfo::getTypeName(TSL.getType());
          std::string Str =
              MapNames::findReplacedName(MapNames::TypeNamesMap, TyName);
          insertHeaderForTypeRule(TyName, BeginLoc);
          requestHelperFeatureForTypeNames(TyName);

          if (!Str.empty()) {
            emplaceTransformation(
                new ReplaceToken(BeginLoc, TSL.getEndLoc(), std::move(Str)));
            return;
          }
        }
      }

      std::string TypeName = Tok.getRawIdentifier().str();
      std::string Str =
          MapNames::findReplacedName(MapNames::TypeNamesMap, TypeName);
      insertHeaderForTypeRule(TypeName, BeginLoc);
      requestHelperFeatureForTypeNames(TypeName);
      if (auto VD = DpctGlobalInfo::findAncestor<VarDecl>(TL)) {
        auto TypeStr = VD->getType().getAsString();
        if (VD->getKind() == Decl::Var && TypeStr == "dim3") {
          std::string Replacement;
          std::string ReplacedType = "range";
          llvm::raw_string_ostream OS(Replacement);
          DpctGlobalInfo::printCtadClass(
              OS, buildString(MapNames::getClNamespace(), ReplacedType), 3);
          Str = OS.str();
        }
      }

      if (!Str.empty()) {
        SrcAPIStaticsMap[TypeName]++;
        emplaceTransformation(new ReplaceToken(BeginLoc, std::move(Str)));
        return;
      }
    }
  }
}

REGISTER_RULE(ReplaceDim3CtorRule, PassKind::PK_Migration)

// rule for dim3 types member fields replacements.
void Dim3MemberFieldsRule::registerMatcher(MatchFinder &MF) {
  // dim3->x/y/z => (*dim3)[0]/[1]/[2]
  // dim3.x/y/z => dim3[0]/[1]/[2]
  // int64_t{dim3->x/y/z} => int64_t((*dim3)[0]/[1]/[2])
  // int64_t{dim3.x/y/z} => int64_t(dim3[0]/[1]/[2])
  auto Dim3MemberExpr = [&]() {
    return memberExpr(anyOf(
        has(implicitCastExpr(hasType(pointsTo(typedefDecl(hasName("dim3")))))),
        hasObjectExpression(hasType(qualType(hasCanonicalType(
            recordType(hasDeclaration(cxxRecordDecl(hasName("dim3"))))))))));
  };
  MF.addMatcher(Dim3MemberExpr().bind("Dim3MemberExpr"), this);
  MF.addMatcher(
      cxxFunctionalCastExpr(
          allOf(hasTypeLoc(loc(isSignedInteger())),
                hasDescendant(
                    initListExpr(hasInit(0, ignoringImplicit(Dim3MemberExpr())))
                        .bind("InitListExpr")))),
      this);
}

void Dim3MemberFieldsRule::runRule(const MatchFinder::MatchResult &Result) {
  // E.g.
  // dim3 *pd3, d3;
  // pd3->z; d3.z;
  // int64_t{d3.x}, int64_t{pd3->x};
  // will migrate to:
  // (*pd3)[0]; d3[0];
  // sycl::range<3> *pd3, d3;
  // int64_t(d3[0]), int64_t((*pd3)[0]);
  ExprAnalysis EA;
  if (const auto *ILE = getNodeAsType<InitListExpr>(Result, "InitListExpr")) {
    EA.analyze(ILE);
  } else if (const auto *ME =
                 getNodeAsType<MemberExpr>(Result, "Dim3MemberExpr")) {
    EA.analyze(ME);
  } else {
    return;
  }
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}

REGISTER_RULE(Dim3MemberFieldsRule, PassKind::PK_Migration)

void DeviceInfoVarRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(anyOf(
              hasType(qualType(hasCanonicalType(recordType(
                  hasDeclaration(cxxRecordDecl(hasAnyName(
                    "cudaDeviceProp", "cudaPointerAttributes"))))))),
              hasType(
                  pointsTo(qualType(hasCanonicalType(recordType(hasDeclaration(
                    cxxRecordDecl(hasAnyName(
                      "cudaDeviceProp", "cudaPointerAttributes")))))))))))
          .bind("FieldVar"),
      this);
}

void DeviceInfoVarRule::runRule(const MatchFinder::MatchResult &Result) {
  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "FieldVar");
  if (!ME)
    return;
  auto MemberName = ME->getMemberNameInfo().getAsString();

  auto BaseType = ME->getBase()->getType();
  if (BaseType->isPointerType()) {
    BaseType = BaseType->getPointeeType();
  }
  std::string MemberExprName =
                      DpctGlobalInfo::getTypeName(BaseType.getCanonicalType())
                        + "." + MemberName;
  if (MemberExprRewriterFactoryBase::MemberExprRewriterMap->find(MemberExprName)
        != MemberExprRewriterFactoryBase::MemberExprRewriterMap->end()) {
      ExprAnalysis EA;
      EA.analyze(ME);
      emplaceTransformation(EA.getReplacement());
      EA.applyAllSubExprRepl();
      return;
  }
  // unmigrated properties
  if (MemberName == "regsPerBlock") {
    report(ME->getBeginLoc(), Diagnostics::UNMIGRATED_DEVICE_PROP, false,
           MemberName);
    return;
  }

  // not functionally compatible properties
  if (MemberName == "deviceOverlap" || MemberName == "concurrentKernels") {
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, "true");
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), "true"));
    return;
  } else if (MemberName == "canMapHostMemory" ||
             MemberName == "kernelExecTimeoutEnabled") {
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, "false");
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), "false"));
    return;
  } else if (MemberName == "pciDomainID" || MemberName == "pciBusID") {
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, "-1");
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), "-1"));
    return;
  } else if (MemberName == "memPitch") {
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, "INT_MAX");
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), "INT_MAX"));
    return;
  } else if (MemberName == "textureAlignment") {
    requestFeature(HelperFeatureEnum::device_ext);
    std::string Repl =
        MapNames::getDpctNamespace() + "get_current_device().get_info<" +
        MapNames::getClNamespace() + "info::device::mem_base_addr_align>()";
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, Repl);
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), std::move(Repl)));
    return;
  } else if (MemberName == "l2CacheSize") {
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, "global_mem_cache_size");
  } else if (MemberName == "ECCEnabled") {
    requestFeature(HelperFeatureEnum::device_ext);
    std::string Repl = MapNames::getDpctNamespace() +
                       "get_current_device().get_info<" +
                       MapNames::getClNamespace() +
                       "info::device::error_correction_support>()";
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), std::move(Repl)));
    return;
  }

  if (MemberName == "sharedMemPerBlock" ||
      MemberName == "sharedMemPerMultiprocessor") {
    report(ME->getBeginLoc(), Diagnostics::LOCAL_MEM_SIZE, false, MemberName);
  } else if (MemberName == "maxGridSize") {
    report(ME->getBeginLoc(), Diagnostics::MAX_GRID_SIZE, false);
  }

  if (!DpctGlobalInfo::useDeviceInfo() &&
      (MemberName == "pciDeviceID" || MemberName == "uuid")) {
    report(ME->getBeginLoc(), Diagnostics::UNMIGRATED_DEVICE_PROP, false,
           MemberName);
    return;
  }

  auto Search = PropNamesMap.find(MemberName);
  if (Search == PropNamesMap.end()) {
    return;
  }
  
  // migrate to get_XXX() eg. "b=a.minor" to "b=a.get_minor_version()"
  auto Parents = Result.Context->getParents(*ME);
  if (Parents.size() < 1)
    return;
  if ((Search->second.compare(0, 13, "major_version") == 0) ||
      (Search->second.compare(0, 13, "minor_version") == 0)) {
    report(ME->getBeginLoc(), Comments::VERSION_COMMENT, false);
  }
  if (Search->second.compare(0, 10, "integrated") == 0) {
    report(ME->getBeginLoc(), Comments::NOT_SUPPORT_API_INTEGRATEDORNOT, false);
  }
  std::string TmplArg = "";
  if (MemberName == "maxGridSize" ||
      MemberName == "maxThreadsDim") {
    // Similar code in ExprAnalysis.cpp
    TmplArg = "<int *>";
  }
  if (auto *BO = Parents[0].get<clang::BinaryOperator>()) {
  // migrate to set_XXX() eg. "a.minor = 1" to "a.set_minor_version(1)"
    if (BO->getOpcode() == clang::BO_Assign) {
      requestFeature(MapNames::PropToSetFeatureMap.at(MemberName));
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, "set_" + Search->second));
      emplaceTransformation(new ReplaceText(BO->getOperatorLoc(), 1, "("));
      emplaceTransformation(new InsertAfterStmt(BO, ")"));
      return ;
    }
  } else if (auto *OCE = Parents[0].get<clang::CXXOperatorCallExpr>()) {
  // migrate to set_XXX() for types with an overloaded = operator
    if (OCE->getOperator() == clang::OverloadedOperatorKind::OO_Equal) {
      requestFeature(MapNames::PropToSetFeatureMap.at(MemberName));
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, "set_" + Search->second));
      emplaceTransformation(new ReplaceText(OCE->getOperatorLoc(), 1, "("));
      emplaceTransformation(new InsertAfterStmt(OCE, ")"));
      return ;
    }
  }
  requestFeature(MapNames::PropToGetFeatureMap.at(MemberName));
  emplaceTransformation(new RenameFieldInMemberExpr(
    ME, "get_" + Search->second + TmplArg + "()")); 
  return ;
}

REGISTER_RULE(DeviceInfoVarRule, PassKind::PK_Migration)

// Rule for Enums constants.
void EnumConstantRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(
          to(enumConstantDecl(anyOf(
              hasType(enumDecl(hasAnyName(
                  "cudaComputeMode", "cudaMemcpyKind", "cudaMemoryAdvise",
                  "cudaDeviceAttr", "libraryPropertyType_t", "cudaDataType_t",
                  "cublasComputeType_t", "CUmem_advise_enum", "cufftType_t",
                  "cufftType", "cudaMemoryType", "CUctx_flags_enum"))),
              matchesName("CUDNN_.*"), matchesName("CUSOLVER_.*")))))
          .bind("EnumConstant"),
      this);
}

void EnumConstantRule::handleComputeMode(std::string EnumName,
                                         const DeclRefExpr *E) {
  report(E->getBeginLoc(), Diagnostics::COMPUTE_MODE, false);
  auto P = getParentStmt(E);
  if (auto ICE = dyn_cast<ImplicitCastExpr>(P)) {
    P = getParentStmt(ICE);
    if (auto BO = dyn_cast<BinaryOperator>(P)) {
      auto LHS = BO->getLHS()->IgnoreImpCasts();
      auto RHS = BO->getRHS()->IgnoreImpCasts();
      const MemberExpr *ME = nullptr;
      if (auto MEL = dyn_cast<MemberExpr>(LHS))
        ME = MEL;
      else if (auto MER = dyn_cast<MemberExpr>(RHS))
        ME = MER;
      if (ME) {
        auto MD = ME->getMemberDecl();
        auto BaseTy = DpctGlobalInfo::getUnqualifiedTypeName(
            ME->getBase()->getType().getCanonicalType(),
            DpctGlobalInfo::getContext());
        if (MD->getNameAsString() == "computeMode" &&
            BaseTy == "cudaDeviceProp") {
          if (EnumName == "cudaComputeModeDefault") {
            if (BO->getOpcodeStr() == "==")
              emplaceTransformation(new ReplaceStmt(P, "true"));
            else if (BO->getOpcodeStr() == "!=")
              emplaceTransformation(new ReplaceStmt(P, "false"));
          } else {
            if (BO->getOpcodeStr() == "==")
              emplaceTransformation(new ReplaceStmt(P, "false"));
            else if (BO->getOpcodeStr() == "!=")
              emplaceTransformation(new ReplaceStmt(P, "true"));
          }
          return;
        }
      }
    }
  }
  // default => 1
  // others  => 0
  if (EnumName == "cudaComputeModeDefault") {
    emplaceTransformation(new ReplaceStmt(E, "1"));
    return;
  } else if (EnumName == "cudaComputeModeExclusive" ||
             EnumName == "cudaComputeModeProhibited" ||
             EnumName == "cudaComputeModeExclusiveProcess") {
    emplaceTransformation(new ReplaceStmt(E, "0"));
    return;
  }
}

void EnumConstantRule::runRule(const MatchFinder::MatchResult &Result) {
  const DeclRefExpr *E = getNodeAsType<DeclRefExpr>(Result, "EnumConstant");
  if (!E)
    return;
  std::string EnumName = E->getNameInfo().getName().getAsString();
  if (EnumName == "cudaComputeModeDefault" ||
      EnumName == "cudaComputeModeExclusive" ||
      EnumName == "cudaComputeModeProhibited" ||
      EnumName == "cudaComputeModeExclusiveProcess") {
    handleComputeMode(EnumName, E);
    return;
  } else if (auto ET = dyn_cast<EnumType>(E->getType())) {
    if (auto ETD = ET->getDecl()) {
      auto EnumTypeName = ETD->getName().str();
      if (EnumTypeName == "cudaMemoryAdvise" ||
          EnumTypeName == "CUmem_advise_enum") {
        report(E->getBeginLoc(), Diagnostics::DEFAULT_MEM_ADVICE, false,
               " and was set to 0");
      } else if (EnumTypeName == "cudaDeviceAttr") {
        auto &Context = DpctGlobalInfo::getContext();
        auto Parent = Context.getParents(*E)[0];
        if (auto PCE = Parent.get<CallExpr>()) {
          if (auto DC = PCE->getDirectCallee()) {
            if (DC->getNameAsString() == "cudaDeviceGetAttribute")
              return;
          }
        }
        if (auto EC = dyn_cast<EnumConstantDecl>(E->getDecl())) {
          std::string Repl = toString(EC->getInitVal(), 10);
          emplaceTransformation(new ReplaceStmt(E, Repl));
          return;
        }
      }
    }
  }

  auto Search = EnumNamesMap.find(EnumName);
  if (Search == EnumNamesMap.end()) {
    return;
  }
  if (auto ET = dyn_cast<EnumType>(E->getType())) {
    if (auto ETD = ET->getDecl()) {
      if (ETD->getName().str() == "libraryPropertyType_t" ||
          ETD->getName().str() == "cudaDataType_t" ||
          ETD->getName().str() == "cublasComputeType_t") {
        DpctGlobalInfo::getInstance().insertHeader(
            DpctGlobalInfo::getSourceManager().getExpansionLoc(
                E->getBeginLoc()),
            HT_DPCT_COMMON_Utils);
      }
    }
  }
  emplaceTransformation(new ReplaceStmt(E, Search->second->NewName));
  requestHelperFeatureForEnumNames(EnumName);
}

REGISTER_RULE(EnumConstantRule, PassKind::PK_Migration)

void ErrorConstantsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(
                              hasDeclContext(enumDecl(anyOf(
                                hasName("cudaError"), hasName("cufftResult_t"),
                                hasName("cudaError_enum"),
                                hasName("cudaSharedMemConfig")))))))
                    .bind("ErrorConstants"),
                this);
}

void ErrorConstantsRule::runRule(const MatchFinder::MatchResult &Result) {
  const DeclRefExpr *DE = getNodeAsType<DeclRefExpr>(Result, "ErrorConstants");
  if (!DE)
    return;
  if (EventAPICallRule::getEventQueryTraversal().startFromEnumRef(DE))
    return;
  auto *EC = cast<EnumConstantDecl>(DE->getDecl());
  std::string Repl = toString(EC->getInitVal(), 10);

  // If the cudaErrorNotReady is one operand of binary operator "==" or "!=",
  // and the other operand is the function call "cudaEventQuery", and the whole
  // binary is in the condition of if/while/for/do/switch,
  // then cudaErrorNotReady will be migrated to "0" while "==" will be migrated
  // to "!=".
  if (EC->getDeclName().getAsString() == "cudaErrorNotReady" &&
      isConditionOfFlowControl(DE, true)) {
    auto &Context = dpct::DpctGlobalInfo::getContext();
    auto ParentNodes = Context.getParents(*DE);
    DynTypedNode ParentNode;
    bool MatchFunction = false;
    const BinaryOperator *BO = nullptr;
    while (!ParentNodes.empty()) {
      ParentNode = ParentNodes[0];
      BO = ParentNode.get<BinaryOperator>();
      if (BO && (BO->getOpcode() == BinaryOperatorKind::BO_EQ ||
                 BO->getOpcode() == BinaryOperatorKind::BO_NE)) {
        auto LHSCall = dyn_cast<CallExpr>(BO->getLHS()->IgnoreImpCasts());
        auto RHSCall = dyn_cast<CallExpr>(BO->getRHS()->IgnoreImpCasts());

        if ((LHSCall && LHSCall->getDirectCallee() &&
             (LHSCall->getDirectCallee()
                      ->getNameInfo()
                      .getName()
                      .getAsString() == "cudaEventQuery" ||
              LHSCall->getDirectCallee()
                      ->getNameInfo()
                      .getName()
                      .getAsString() == "cuEventQuery")) ||
            (RHSCall && RHSCall->getDirectCallee() &&
             (RHSCall->getDirectCallee()
                      ->getNameInfo()
                      .getName()
                      .getAsString() == "cudaEventQuery" ||
              RHSCall->getDirectCallee()
                      ->getNameInfo()
                      .getName()
                      .getAsString() == "cuEventQuery"))) {
          MatchFunction = true;
          break;
        }
      }
      ParentNodes = Context.getParents(ParentNode);
    }

    if (MatchFunction) {
      Repl = "0";
      std::string OperatorRepl;
      if (BO->getOpcode() == BinaryOperatorKind::BO_EQ)
        OperatorRepl = "!=";
      else
        OperatorRepl = "==";
      emplaceTransformation(
          new ReplaceToken(DpctGlobalInfo::getSourceManager().getSpellingLoc(
                               BO->getOperatorLoc()),
                           std::move(OperatorRepl)));
    }
  }

  emplaceTransformation(new ReplaceStmt(DE, Repl));
}

REGISTER_RULE(ErrorConstantsRule, PassKind::PK_Migration)

void LinkageSpecDeclRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(linkageSpecDecl().bind("LinkageSpecDecl"), this);
}

void LinkageSpecDeclRule::runRule(const MatchFinder::MatchResult &Result) {
  const LinkageSpecDecl *LSD =
      getNodeAsType<LinkageSpecDecl>(Result, "LinkageSpecDecl");
  if (!LSD)
    return;
  if (LSD->getLanguage() != clang::LinkageSpecDecl::LanguageIDs::lang_c)
    return;
  if (!LSD->hasBraces())
    return;

  SourceLocation Begin =
      DpctGlobalInfo::getSourceManager().getExpansionLoc(LSD->getExternLoc());
  SourceLocation End =
      DpctGlobalInfo::getSourceManager().getExpansionLoc(LSD->getRBraceLoc());
  auto BeginLocInfo = DpctGlobalInfo::getLocInfo(Begin);
  auto EndLocInfo = DpctGlobalInfo::getLocInfo(End);
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(BeginLocInfo.first);

  FileInfo->getExternCRanges().push_back(
      std::make_pair(BeginLocInfo.second, EndLocInfo.second));
}

REGISTER_RULE(LinkageSpecDeclRule, PassKind::PK_Migration)

void ManualMigrateEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName("NCCL_.*"))))
                    .bind("NCCLConstants"),
                this);
}

void ManualMigrateEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "NCCLConstants")) {
    auto *ECD = cast<EnumConstantDecl>(DE->getDecl());
    if (DpctGlobalInfo::isInAnalysisScope(ECD->getBeginLoc())) {
      return;
    }
    report(dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
               DE->getBeginLoc()),
           Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
           "Intel(R) oneAPI Collective Communications Library");
  }
}

REGISTER_RULE(ManualMigrateEnumsRule, PassKind::PK_Migration,
              RuleGroupKind::RK_NCCL)

// Rule for FFT enums.
void FFTEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(
          to(enumConstantDecl(matchesName(
              "(CUFFT_SUCCESS|CUFFT_INVALID_PLAN|CUFFT_ALLOC_FAILED|CUFFT_"
              "INVALID_TYPE|CUFFT_INVALID_VALUE|CUFFT_INTERNAL_ERROR|CUFFT_"
              "EXEC_FAILED|CUFFT_SETUP_FAILED|CUFFT_INVALID_SIZE|CUFFT_"
              "UNALIGNED_DATA|CUFFT_INCOMPLETE_PARAMETER_LIST|CUFFT_INVALID_"
              "DEVICE|CUFFT_PARSE_ERROR|CUFFT_NO_WORKSPACE|CUFFT_NOT_"
              "IMPLEMENTED|CUFFT_LICENSE_ERROR|CUFFT_NOT_SUPPORTED)"))))
          .bind("FFTConstants"),
      this);
}

void FFTEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "FFTConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));
    return;
  }
}

REGISTER_RULE(FFTEnumsRule, PassKind::PK_Migration, RuleGroupKind::RK_FFT)

// Rule for CU_JIT enums.
void CU_JITEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(
          to(enumConstantDecl(matchesName(
              "(CU_JIT_*)"))))
          .bind("CU_JITConstants"),
      this);
}

void CU_JITEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "CU_JITConstants")) {
    emplaceTransformation(new ReplaceStmt(DE, "0"));

    report(DE->getBeginLoc(),
           Diagnostics::HOSTALLOCMACRO_NO_MEANING,
           true, DE->getDecl()->getNameAsString());
  }
}

REGISTER_RULE(CU_JITEnumsRule, PassKind::PK_Migration)

// Rule for BLAS enums.
// Migrate BLAS status values to corresponding int values
// Other BLAS named values are migrated to corresponding named values
void BLASEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName(
                                "(CUBLAS_STATUS.*)|(CUDA_R_.*)|(CUDA_C_.*)|("
                                "CUBLAS_GEMM_.*)|(CUBLAS_POINTER_MODE.*)"))))
                    .bind("BLASStatusConstants"),
                this);
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName(
                                "(CUBLAS_OP.*)|(CUBLAS_SIDE.*)|(CUBLAS_FILL_"
                                "MODE.*)|(CUBLAS_DIAG.*)"))))
                    .bind("BLASNamedValueConstants"),
                this);
}

void BLASEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "BLASStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));
  }

  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "BLASNamedValueConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    std::string Name = EC->getNameAsString();
    auto Search = MapNames::BLASEnumsMap.find(Name);
    if (Search == MapNames::BLASEnumsMap.end()) {
      llvm::dbgs() << "[" << getName()
                   << "] Unexpected enum variable: " << Name;
      return;
    }
    std::string Replacement = Search->second;
    emplaceTransformation(new ReplaceStmt(DE, std::move(Replacement)));
  }
}

REGISTER_RULE(BLASEnumsRule, PassKind::PK_Migration, RuleGroupKind::RK_BLas)

// Rule for RANDOM enums.
void RandomEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName("CURAND_STATUS.*"))))
          .bind("RANDOMStatusConstants"),
      this);
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName("CURAND_RNG.*"))))
                    .bind("RandomTypeEnum"),
                this);
}

void RandomEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "RANDOMStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));
  }
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "RandomTypeEnum")) {
    std::string EnumStr = DE->getNameInfo().getName().getAsString();
    auto Search = MapNames::RandomEngineTypeMap.find(EnumStr);
    if (Search == MapNames::RandomEngineTypeMap.end()) {
      report(DE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, EnumStr);
      return;
    }
    if (EnumStr == "CURAND_RNG_PSEUDO_XORWOW" ||
        EnumStr == "CURAND_RNG_QUASI_SOBOL64" ||
        EnumStr == "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64") {
      report(DE->getBeginLoc(), Diagnostics::DIFFERENT_GENERATOR, false);
    } else if (EnumStr == "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32") {
      report(DE->getBeginLoc(), Diagnostics::DIFFERENT_BASIC_GENERATOR, false);
    }
    emplaceTransformation(new ReplaceStmt(DE, Search->second));
  }
}

REGISTER_RULE(RandomEnumsRule, PassKind::PK_Migration, RuleGroupKind::RK_Rng)

// Rule for spBLAS enums.
// Migrate spBLAS status values to corresponding int values
// Other spBLAS named values are migrated to corresponding named values
void SPBLASEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName(
                                "(CUSPARSE_STATUS.*)|(CUSPARSE_POINTER_MODE.*)|"
                                "(CUSPARSE_ALG.*)"))))
                    .bind("SPBLASStatusConstants"),
                this);
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName(
                      "(CUSPARSE_OPERATION_.*)|(CUSPARSE_FILL_MODE_.*)|("
                      "CUSPARSE_DIAG_TYPE_.*)|(CUSPARSE_INDEX_.*)|(CUSPARSE_"
                      "MATRIX_TYPE_.*)|(CUSPARSE_ORDER_.*)"))))
          .bind("SPBLASNamedValueConstants"),
      this);
}

void SPBLASEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SPBLASStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));
  }

  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SPBLASNamedValueConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    std::string Name = EC->getNameAsString();
    auto Search = MapNames::SPBLASEnumsMap.find(Name);
    if (Search == MapNames::SPBLASEnumsMap.end()) {
      llvm::dbgs() << "[" << getName()
                   << "] Unexpected enum variable: " << Name;
      return;
    }
    std::string Replacement = Search->second;
    emplaceTransformation(new ReplaceStmt(DE, std::move(Replacement)));
  }
}

REGISTER_RULE(SPBLASEnumsRule, PassKind::PK_Migration, RuleGroupKind::RK_Sparse)

/// The function returns the migrated arguments of the scalar parameters.
/// In the original code, the type of this parameter is pointer.
/// (1) If original type is float/double and argument is like "&alpha",
///     this function will return "alpha".
/// (2) If original type is float2/double2 and argument is like "&alpha",
///     this function will return
///     "std::complex<float/double>(alpha.x(), alpha.x())".
/// (3) If original argument is like "alpha", this function will return
///     "dpct::get_value(alpha, q)".
/// \p Expr is used to distinguish case(1,2) and case(3)
/// \p ExprStr and \p QueueStr are used for case(3)
/// \p ValueType is used for case(2)
std::string getValueStr(const Expr *Expr, std::string ExprStr,
                        std::string QueueStr, std::string ValueType = "") {
  if (auto UO = dyn_cast_or_null<UnaryOperator>(Expr->IgnoreImpCasts())) {
    if (UO->getOpcode() == UO_AddrOf && UO->getSubExpr()) {
      ExprAnalysis EA;
      std::string NewStr = EA.ref(UO->getSubExpr());
      if (ValueType == "std::complex<float>" ||
          ValueType == "std::complex<double>")
        return ValueType + "(" + NewStr + ".x(), " + NewStr + ".y())";
      else
        return NewStr;
    }
  } else if (auto COCE =
                 dyn_cast<CXXOperatorCallExpr>(Expr->IgnoreImpCasts())) {
    if (COCE->getOperator() == OO_Amp && COCE->getArg(0)) {
      ExprAnalysis EA;
      std::string NewStr = EA.ref(COCE->getArg(0));
      if (ValueType == "std::complex<float>" ||
          ValueType == "std::complex<double>")
        return ValueType + "(" + NewStr + ".x(), " + NewStr + ".y())";
      else
        return NewStr;
    }
  }
  requestFeature(HelperFeatureEnum::device_ext);
  return MapNames::getDpctNamespace() + "get_value(" + ExprStr + ", *" +
         QueueStr + ")";
}

// Rule for spBLAS function calls.
void SPBLASFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        /*management*/
        "cusparseCreate", "cusparseDestroy", "cusparseSetStream",
        "cusparseGetStream", "cusparseGetPointerMode", "cusparseSetPointerMode",
        "cusparseGetErrorName", "cusparseGetErrorString", "cusparseGetProperty",
        /*helper*/
        "cusparseCreateMatDescr", "cusparseDestroyMatDescr",
        "cusparseSetMatType", "cusparseGetMatType", "cusparseSetMatIndexBase",
        "cusparseGetMatIndexBase", "cusparseSetMatDiagType",
        "cusparseGetMatDiagType", "cusparseSetMatFillMode",
        "cusparseGetMatFillMode", "cusparseCreateSolveAnalysisInfo",
        "cusparseDestroySolveAnalysisInfo",
        /*level 2*/
        "cusparseScsrmv", "cusparseDcsrmv", "cusparseCcsrmv", "cusparseZcsrmv",
        "cusparseScsrmv_mp", "cusparseDcsrmv_mp", "cusparseCcsrmv_mp",
        "cusparseZcsrmv_mp", "cusparseCsrmvEx", "cusparseCsrmvEx_bufferSize",
        "cusparseScsrsv_analysis", "cusparseDcsrsv_analysis",
        "cusparseCcsrsv_analysis", "cusparseZcsrsv_analysis",
        /*level 3*/
        "cusparseScsrmm", "cusparseDcsrmm", "cusparseCcsrmm", "cusparseZcsrmm",
        /*Generic*/
        "cusparseCreateCsr", "cusparseDestroySpMat", "cusparseCsrGet",
        "cusparseSpMatGetFormat", "cusparseSpMatGetIndexBase",
        "cusparseSpMatGetValues", "cusparseSpMatSetValues",
        "cusparseCreateDnMat", "cusparseDestroyDnMat", "cusparseDnMatGet",
        "cusparseDnMatGetValues", "cusparseDnMatSetValues",
        "cusparseCreateDnVec", "cusparseDestroyDnVec", "cusparseDnVecGet",
        "cusparseDnVecGetValues", "cusparseDnVecSetValues",
        "cusparseCsrSetPointers", "cusparseSpMatGetSize",
        "cusparseSpMatGetAttribute", "cusparseSpMatSetAttribute",
        "cusparseCreateConstDnVec", "cusparseConstDnVecGet",
        "cusparseConstDnVecGetValues", "cusparseSpMM",
        "cusparseSpMM_bufferSize", "cusparseSpMV", "cusparseSpMV_bufferSize",
        "cusparseSpMM_preprocess", "cusparseSpGEMM_compute",
        "cusparseSpGEMM_copy", "cusparseSpGEMM_createDescr",
        "cusparseSpGEMM_destroyDescr", "cusparseSpGEMM_workEstimation",
        "cusparseSpSV_createDescr", "cusparseSpSV_destroyDescr",
        "cusparseSpSV_solve", "cusparseSpSV_bufferSize",
        "cusparseSpSV_analysis");
  };
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), parentStmt()))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               unless(parentStmt())))
                    .bind("FunctionCallUsed"),
                this);
}

void SPBLASFunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCallUsed")))
      return;
  }

  if (!CE->getDirectCallee())
    return;

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  StringRef FuncNameRef(FuncName);
  if (FuncNameRef.endswith("csrmv") || FuncNameRef.endswith("csrmv_mp")) {
    report(
        DpctGlobalInfo::getSourceManager().getExpansionLoc(CE->getBeginLoc()),
        Diagnostics::UNSUPPORT_MATRIX_TYPE, true,
        "general/symmetric/triangular");
  } else if (FuncNameRef.endswith("csrmm")) {
    report(
        DpctGlobalInfo::getSourceManager().getExpansionLoc(CE->getBeginLoc()),
        Diagnostics::UNSUPPORT_MATRIX_TYPE, true, "general");
  }

  if (MapNames::SPARSEAPIWithRewriter.find(FuncName) !=
      MapNames::SPARSEAPIWithRewriter.end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
}

REGISTER_RULE(SPBLASFunctionCallRule, PassKind::PK_Migration,
              RuleGroupKind::RK_Sparse)

// Rule for Random function calls. Currently only support host APIs.
void RandomFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "curandCreateGenerator", "curandSetPseudoRandomGeneratorSeed",
        "curandSetGeneratorOffset", "curandSetQuasiRandomGeneratorDimensions",
        "curandDestroyGenerator", "curandGenerate", "curandGenerateLongLong",
        "curandGenerateLogNormal", "curandGenerateLogNormalDouble",
        "curandGenerateNormal", "curandGenerateNormalDouble",
        "curandGeneratePoisson", "curandGenerateUniform",
        "curandGenerateUniformDouble", "curandSetStream",
        "curandCreateGeneratorHost");
  };
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), parentStmt()))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               unless(parentStmt())))
                    .bind("FunctionCallUsed"),
                this);
}

void RandomFunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCallUsed")))
      return;
    IsAssigned = true;
  }

  if (!CE->getDirectCallee())
    return;

  auto &SM = DpctGlobalInfo::getSourceManager();

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // Below code can distinguish this kind of function like macro
  //  #define CHECK_STATUS(x) x
  //  CHECK_STATUS(anAPICall());
  bool IsMacroArg = SM.isMacroArgExpansion(CE->getBeginLoc()) &&
                    SM.isMacroArgExpansion(CE->getEndLoc());

  if (FuncNameBegin.isMacroID() && IsMacroArg) {
    FuncNameBegin = SM.getImmediateSpellingLoc(FuncNameBegin);
    FuncNameBegin = SM.getExpansionLoc(FuncNameBegin);
  } else if (FuncNameBegin.isMacroID()) {
    FuncNameBegin = SM.getExpansionLoc(FuncNameBegin);
  }

  if (FuncCallEnd.isMacroID() && IsMacroArg) {
    FuncCallEnd = SM.getImmediateSpellingLoc(FuncCallEnd);
    FuncCallEnd = SM.getExpansionLoc(FuncCallEnd);
  } else if (FuncCallEnd.isMacroID()) {
    FuncCallEnd = SM.getExpansionLoc(FuncCallEnd);
  }

  // Offset 1 is the length of the last token ")"
  FuncCallEnd = SM.getExpansionLoc(FuncCallEnd).getLocWithOffset(1);

  if (IsAssigned) {
    requestFeature(HelperFeatureEnum::device_ext);
    insertAroundStmt(CE, "DPCT_CHECK_ERROR(", ")");
  }

  if (FuncName == "curandCreateGenerator" ||
      FuncName == "curandCreateGeneratorHost") {
    const auto *const Arg0 = CE->getArg(0);
    requestFeature(HelperFeatureEnum::device_ext);
    if (Arg0->getStmtClass() == Stmt::UnaryOperatorClass) {
      const auto *const UO = cast<const UnaryOperator>(Arg0);
      auto SE = UO->getSubExpr();
      if (UO->getOpcode() == UO_AddrOf &&
          (SE->getStmtClass() == Stmt::DeclRefExprClass ||
           SE->getStmtClass() == Stmt::MemberExprClass)) {
        return emplaceTransformation(new ReplaceStmt(
            CE, false,
            buildString(ExprAnalysis::ref(SE),
                        " = " + MapNames::getDpctNamespace() +
                            "rng::create_host_rng(",
                        ExprAnalysis::ref(CE->getArg(1)), ")")));
      }
    }
    return emplaceTransformation(
        new ReplaceStmt(CE, false,
                        buildString("*(", ExprAnalysis::ref(CE->getArg(0)),
                                    ") = " + MapNames::getDpctNamespace() +
                                        "rng::create_host_rng(",
                                    ExprAnalysis::ref(CE->getArg(1)), ")")));
  }
  if (FuncName == "curandDestroyGenerator") {
    return emplaceTransformation(new ReplaceStmt(
        CE, false, buildString(ExprAnalysis::ref(CE->getArg(0)), ".reset()")));
  }
  if (FuncName == "curandSetPseudoRandomGeneratorSeed") {
    return emplaceTransformation(new ReplaceStmt(
        CE, false,
        buildString(ExprAnalysis::ref(CE->getArg(0)), "->set_seed(",
                    ExprAnalysis::ref(CE->getArg(1)), ")")));
  }
  if (FuncName == "curandSetQuasiRandomGeneratorDimensions") {
    return emplaceTransformation(new ReplaceStmt(
        CE, false,
        buildString(ExprAnalysis::ref(CE->getArg(0)), "->set_dimensions(",
                    ExprAnalysis::ref(CE->getArg(1)), ")")));
  }
  if (MapNames::RandomGenerateFuncMap.find(FuncName) !=
      MapNames::RandomGenerateFuncMap.end()) {
    auto ArgStr = ExprAnalysis::ref(CE->getArg(1));
    for (unsigned i = 2; i < CE->getNumArgs(); ++i) {
      ArgStr += buildString(", ", ExprAnalysis::ref(CE->getArg(i)));
    }
    return emplaceTransformation(new ReplaceStmt(
        CE, false,
        buildString(
            ExprAnalysis::ref(CE->getArg(0)),
            "->" + MapNames::RandomGenerateFuncMap.find(FuncName)->second + "(",
            ArgStr, ")")));
  }
  if (FuncName == "curandSetGeneratorOffset") {
    return emplaceTransformation(new ReplaceStmt(
        CE, false,
        buildString(ExprAnalysis::ref(CE->getArg(0)), "->skip_ahead(",
                    ExprAnalysis::ref(CE->getArg(1)), ")")));
  }
  if (FuncName == "curandSetStream") {
    return emplaceTransformation(new ReplaceStmt(
        CE, false,
        buildString(ExprAnalysis::ref(CE->getArg(0)), "->set_queue(",
                    ExprAnalysis::ref(CE->getArg(1)), ")")));
  }
}

REGISTER_RULE(RandomFunctionCallRule, PassKind::PK_Migration,
              RuleGroupKind::RK_Rng)

// Rule for device Random function calls.
void DeviceRandomFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "curand_init", "curand", "curand4", "curand_normal", "curand_normal4",
        "curand_normal2", "curand_normal2_double", "curand_normal_double",
        "curand_log_normal", "curand_log_normal2", "curand_log_normal2_double",
        "curand_log_normal4", "curand_log_normal_double", "curand_uniform",
        "curand_uniform2_double", "curand_uniform4", "curand_uniform_double",
        "curand_poisson", "curand_poisson4", "skipahead", "skipahead_sequence",
        "skipahead_subsequence", "curand_uniform4_double", "curand_normal4_double",
        "curand_log_normal4_double");
  };
  MF.addMatcher(
      callExpr(callee(functionDecl(functionName()))).bind("FunctionCall"),
      this);
}

void DeviceRandomFunctionCallRule::runRule(
    const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE)
    return;
  if (!CE->getDirectCallee())
    return;

  auto &SM = DpctGlobalInfo::getSourceManager();
  auto SL = SM.getExpansionLoc(CE->getBeginLoc());
  std::string Key =
      SM.getFilename(SL).str() + std::to_string(SM.getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // Offset 1 is the length of the last token ")"
  FuncCallEnd = SM.getExpansionLoc(FuncCallEnd).getLocWithOffset(1);
  auto FuncCallLength =
      SM.getCharacterData(FuncCallEnd) - SM.getCharacterData(FuncNameBegin);
  std::string IndentStr = getIndent(FuncNameBegin, SM).str();

  if (FuncName == "curand_init") {
    if (CE->getNumArgs() < 4) {
      report(FuncNameBegin, Diagnostics::API_NOT_MIGRATED, false, FuncName);
      return;
    }

    std::string Arg0Type = DpctGlobalInfo::getTypeName(
        CE->getArg(0)->getType().getCanonicalType());
    std::string Arg1Type = DpctGlobalInfo::getTypeName(
        CE->getArg(1)->getType().getCanonicalType());
    std::string Arg2Type = DpctGlobalInfo::getTypeName(
        CE->getArg(2)->getType().getCanonicalType());
    std::string DRefArg3Type;

    if (Arg0Type == "unsigned long long" && Arg1Type == "unsigned long long" &&
        Arg2Type == "unsigned long long" &&
        CE->getArg(3)->getType().getCanonicalType()->isPointerType()) {
      DRefArg3Type = DpctGlobalInfo::getTypeName(
          CE->getArg(3)->getType().getCanonicalType()->getPointeeType());
      if (MapNames::DeviceRandomGeneratorTypeMap.find(DRefArg3Type) ==
          MapNames::DeviceRandomGeneratorTypeMap.end()) {
        report(FuncNameBegin, Diagnostics::NOT_SUPPORTED_PARAMETER, false,
               FuncName,
               "parameter " + getStmtSpelling(CE->getArg(3)) +
                   " is unsupported");
        return;
      }
    } else {
      report(FuncNameBegin, Diagnostics::API_NOT_MIGRATED, false, FuncName);
      return;
    }

    auto IsLiteral = [=](const Expr *E) {
      if (dyn_cast<IntegerLiteral>(E->IgnoreCasts()) ||
          dyn_cast<FloatingLiteral>(E->IgnoreCasts()) ||
          dyn_cast<FixedPointLiteral>(E->IgnoreCasts())) {
        return true;
      }
      return false;
    };

    std::string GeneratorType =
        MapNames::DeviceRandomGeneratorTypeMap.find(DRefArg3Type)->second;
    std::string RNGSeed = ExprAnalysis::ref(CE->getArg(0));
    bool IsRNGSubseqLiteral = IsLiteral(CE->getArg(1));
    std::string RNGSubseq = ExprAnalysis::ref(CE->getArg(1));
    bool IsRNGOffsetLiteral = IsLiteral(CE->getArg(2));
    std::string RNGOffset = ExprAnalysis::ref(CE->getArg(2));
    std::string RNGStateName = getDrefName(CE->getArg(3));

    std::string FirstOffsetArg, SecondOffsetArg;
    if (IsRNGOffsetLiteral) {
      FirstOffsetArg = RNGOffset;
    } else {
      FirstOffsetArg = "static_cast<std::uint64_t>(" + RNGOffset + ")";
    }

    std::string ReplStr;
    if (DRefArg3Type == "curandStateXORWOW") {
      report(FuncNameBegin, Diagnostics::SUBSEQUENCE_IGNORED, false, RNGSubseq);
      ReplStr = RNGStateName + " = " + GeneratorType + "(" + RNGSeed + ", " +
                FirstOffsetArg + ")";
    } else {
      std::string Factor = "8";
      if (GeneratorType == MapNames::getDpctNamespace() +
                               "rng::device::rng_generator<oneapi::"
                               "mkl::rng::device::philox4x32x10<1>>" &&
          DRefArg3Type == "curandStatePhilox4_32_10") {
        Factor = "4";
      }

      if (needExtraParens(CE->getArg(1))) {
        RNGSubseq = "(" + RNGSubseq + ")";
      }
      if (IsRNGSubseqLiteral) {
        SecondOffsetArg = RNGSubseq + " * " + Factor;
      } else {
        SecondOffsetArg =
            "static_cast<std::uint64_t>(" + RNGSubseq + " * " + Factor + ")";
      }

      ReplStr = RNGStateName + " = " + GeneratorType + "(" + RNGSeed + ", {" +
                FirstOffsetArg + ", " + SecondOffsetArg + "})";
    }
    emplaceTransformation(
        new ReplaceText(FuncNameBegin, FuncCallLength, std::move(ReplStr)));
  } else if (FuncName == "skipahead" || FuncName == "skipahead_sequence" ||
             FuncName == "skipahead_subsequence") {
    if (FuncName == "skipahead") {
      std::string Arg1Type = CE->getArg(1)->getType().getAsString();
      if (Arg1Type != "curandStateMRG32k3a_t *" &&
          Arg1Type != "curandStatePhilox4_32_10_t *" &&
          Arg1Type != "curandStateXORWOW_t *") {
        // Do not support Sobol32 state and Sobol64 state
        report(FuncNameBegin, Diagnostics::API_NOT_MIGRATED, false, FuncName);
        return;
      }
    }
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  } else {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

REGISTER_RULE(DeviceRandomFunctionCallRule, PassKind::PK_Migration,
              RuleGroupKind::RK_Rng)

void BLASFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        /*Regular BLAS API*/
        /*Regular helper*/
        "cublasCreate_v2", "cublasDestroy_v2", "cublasSetVector",
        "cublasGetVector", "cublasSetVectorAsync", "cublasGetVectorAsync",
        "cublasSetMatrix", "cublasGetMatrix", "cublasSetMatrixAsync",
        "cublasGetMatrixAsync", "cublasSetStream_v2", "cublasGetStream_v2",
        "cublasGetPointerMode_v2", "cublasSetPointerMode_v2",
        "cublasGetAtomicsMode", "cublasSetAtomicsMode", "cublasGetVersion_v2",
        "cublasGetMathMode", "cublasSetMathMode", "cublasGetStatusString",
        /*Regular level 1*/
        "cublasIsamax_v2", "cublasIdamax_v2", "cublasIcamax_v2",
        "cublasIzamax_v2", "cublasIsamin_v2", "cublasIdamin_v2",
        "cublasIcamin_v2", "cublasIzamin_v2", "cublasSasum_v2",
        "cublasDasum_v2", "cublasScasum_v2", "cublasDzasum_v2",
        "cublasSaxpy_v2", "cublasDaxpy_v2", "cublasCaxpy_v2", "cublasZaxpy_v2",
        "cublasScopy_v2", "cublasDcopy_v2", "cublasCcopy_v2", "cublasZcopy_v2",
        "cublasSdot_v2", "cublasDdot_v2", "cublasCdotu_v2", "cublasCdotc_v2",
        "cublasZdotu_v2", "cublasZdotc_v2", "cublasSnrm2_v2", "cublasDnrm2_v2",
        "cublasScnrm2_v2", "cublasDznrm2_v2", "cublasSrot_v2", "cublasDrot_v2",
        "cublasCsrot_v2", "cublasZdrot_v2", "cublasCrot_v2", "cublasZrot_v2",
        "cublasSrotg_v2", "cublasDrotg_v2",
        "cublasCrotg_v2", "cublasZrotg_v2", "cublasSrotm_v2", "cublasDrotm_v2",
        "cublasSrotmg_v2", "cublasDrotmg_v2", "cublasSscal_v2",
        "cublasDscal_v2", "cublasCscal_v2", "cublasCsscal_v2", "cublasZscal_v2",
        "cublasZdscal_v2", "cublasSswap_v2", "cublasDswap_v2", "cublasCswap_v2",
        "cublasZswap_v2",
        /*Regular level 2*/
        "cublasSgbmv_v2", "cublasDgbmv_v2", "cublasCgbmv_v2", "cublasZgbmv_v2",
        "cublasSgemv_v2", "cublasDgemv_v2", "cublasCgemv_v2", "cublasZgemv_v2",
        "cublasSger_v2", "cublasDger_v2", "cublasCgeru_v2", "cublasCgerc_v2",
        "cublasZgeru_v2", "cublasZgerc_v2", "cublasSsbmv_v2", "cublasDsbmv_v2",
        "cublasSspmv_v2", "cublasDspmv_v2", "cublasSspr_v2", "cublasDspr_v2",
        "cublasSspr2_v2", "cublasDspr2_v2", "cublasSsymv_v2", "cublasDsymv_v2",
        "cublasCsymv_v2", "cublasZsymv_v2",
        "cublasSsyr_v2", "cublasDsyr_v2", "cublasSsyr2_v2", "cublasDsyr2_v2",
        "cublasCsyr_v2", "cublasZsyr_v2", "cublasCsyr2_v2", "cublasZsyr2_v2",
        "cublasStbmv_v2", "cublasDtbmv_v2", "cublasCtbmv_v2", "cublasZtbmv_v2",
        "cublasStbsv_v2", "cublasDtbsv_v2", "cublasCtbsv_v2", "cublasZtbsv_v2",
        "cublasStpmv_v2", "cublasDtpmv_v2", "cublasCtpmv_v2", "cublasZtpmv_v2",
        "cublasStpsv_v2", "cublasDtpsv_v2", "cublasCtpsv_v2", "cublasZtpsv_v2",
        "cublasStrmv_v2", "cublasDtrmv_v2", "cublasCtrmv_v2", "cublasZtrmv_v2",
        "cublasStrsv_v2", "cublasDtrsv_v2", "cublasCtrsv_v2", "cublasZtrsv_v2",
        "cublasChemv_v2", "cublasZhemv_v2", "cublasChbmv_v2", "cublasZhbmv_v2",
        "cublasChpmv_v2", "cublasZhpmv_v2", "cublasCher_v2", "cublasZher_v2",
        "cublasCher2_v2", "cublasZher2_v2", "cublasChpr_v2", "cublasZhpr_v2",
        "cublasChpr2_v2", "cublasZhpr2_v2",
        /*Regular level 3*/
        "cublasSgemm_v2", "cublasDgemm_v2", "cublasCgemm_v2", "cublasZgemm_v2",
        "cublasHgemm", "cublasCgemm3m", "cublasZgemm3m",
        "cublasHgemmStridedBatched", "cublasSgemmStridedBatched",
        "cublasDgemmStridedBatched", "cublasCgemmStridedBatched",
        "cublasZgemmStridedBatched", "cublasSsymm_v2", "cublasDsymm_v2",
        "cublasCsymm_v2", "cublasZsymm_v2", "cublasSsyrk_v2", "cublasDsyrk_v2",
        "cublasCsyrk_v2", "cublasZsyrk_v2", "cublasSsyr2k_v2",
        "cublasDsyr2k_v2", "cublasCsyr2k_v2", "cublasZsyr2k_v2",
        "cublasStrsm_v2", "cublasDtrsm_v2", "cublasCtrsm_v2", "cublasZtrsm_v2",
        "cublasChemm_v2", "cublasZhemm_v2", "cublasCherk_v2", "cublasZherk_v2",
        "cublasCher2k_v2", "cublasZher2k_v2", "cublasSsyrkx", "cublasDsyrkx",
        "cublasCsyrkx", "cublasZsyrkx", "cublasCherkx", "cublasZherkx",
        "cublasStrmm_v2", "cublasDtrmm_v2", "cublasCtrmm_v2", "cublasZtrmm_v2",
        "cublasHgemmBatched", "cublasSgemmBatched", "cublasDgemmBatched",
        "cublasCgemmBatched", "cublasZgemmBatched", "cublasStrsmBatched",
        "cublasDtrsmBatched", "cublasCtrsmBatched", "cublasZtrsmBatched",
        /*Extensions*/
        "cublasSgetrfBatched", "cublasDgetrfBatched", "cublasCgetrfBatched",
        "cublasZgetrfBatched", "cublasSgetrsBatched", "cublasDgetrsBatched",
        "cublasCgetrsBatched", "cublasZgetrsBatched", "cublasSgetriBatched",
        "cublasDgetriBatched", "cublasCgetriBatched", "cublasZgetriBatched",
        "cublasSgeqrfBatched", "cublasDgeqrfBatched", "cublasCgeqrfBatched",
        "cublasZgeqrfBatched", "cublasGemmEx", "cublasSgemmEx", "cublasCgemmEx",
        "cublasNrm2Ex", "cublasDotEx", "cublasDotcEx", "cublasScalEx",
        "cublasAxpyEx", "cublasRotEx", "cublasGemmBatchedEx",
        "cublasGemmStridedBatchedEx", "cublasSdgmm", "cublasDdgmm",
        "cublasCdgmm", "cublasZdgmm", "cublasSgeam", "cublasDgeam",
        "cublasCgeam", "cublasZgeam",
        /*Legacy API*/
        "cublasInit", "cublasShutdown", "cublasGetError",
        "cublasSetKernelStream", "cublasGetVersion",
        /*level 1*/
        "cublasSnrm2", "cublasDnrm2", "cublasScnrm2", "cublasDznrm2",
        "cublasSdot", "cublasDdot", "cublasCdotu", "cublasCdotc", "cublasZdotu",
        "cublasZdotc", "cublasSscal", "cublasDscal", "cublasCscal",
        "cublasZscal", "cublasCsscal", "cublasZdscal", "cublasSaxpy",
        "cublasDaxpy", "cublasCaxpy", "cublasZaxpy", "cublasScopy",
        "cublasDcopy", "cublasCcopy", "cublasZcopy", "cublasSswap",
        "cublasDswap", "cublasCswap", "cublasZswap", "cublasIsamax",
        "cublasIdamax", "cublasIcamax", "cublasIzamax", "cublasIsamin",
        "cublasIdamin", "cublasIcamin", "cublasIzamin", "cublasSasum",
        "cublasDasum", "cublasScasum", "cublasDzasum", "cublasSrot",
        "cublasDrot", "cublasCsrot", "cublasZdrot", "cublasCrot",
        "cublasZrot", "cublasSrotg",
        "cublasDrotg", "cublasSrotm", "cublasDrotm", "cublasSrotmg",
        "cublasDrotmg",
        /*level 2*/
        "cublasSgemv", "cublasDgemv", "cublasCgemv", "cublasZgemv",
        "cublasSgbmv", "cublasDgbmv", "cublasCgbmv", "cublasZgbmv",
        "cublasStrmv", "cublasDtrmv", "cublasCtrmv", "cublasZtrmv",
        "cublasStbmv", "cublasDtbmv", "cublasCtbmv", "cublasZtbmv",
        "cublasStpmv", "cublasDtpmv", "cublasCtpmv", "cublasZtpmv",
        "cublasStrsv", "cublasDtrsv", "cublasCtrsv", "cublasZtrsv",
        "cublasStpsv", "cublasDtpsv", "cublasCtpsv", "cublasZtpsv",
        "cublasStbsv", "cublasDtbsv", "cublasCtbsv", "cublasZtbsv",
        "cublasSsymv", "cublasDsymv", "cublasChemv", "cublasZhemv",
        "cublasSsbmv", "cublasDsbmv", "cublasChbmv", "cublasZhbmv",
        "cublasSspmv", "cublasDspmv", "cublasChpmv", "cublasZhpmv",
        "cublasSger", "cublasDger", "cublasCgeru", "cublasCgerc", "cublasZgeru",
        "cublasZgerc", "cublasSsyr", "cublasDsyr", "cublasCher", "cublasZher",
        "cublasSspr", "cublasDspr", "cublasChpr", "cublasZhpr", "cublasSsyr2",
        "cublasDsyr2", "cublasCher2", "cublasZher2", "cublasSspr2",
        "cublasDspr2", "cublasChpr2", "cublasZhpr2",
        /*level 3*/
        "cublasSgemm", "cublasDgemm", "cublasCgemm", "cublasZgemm",
        "cublasSsyrk", "cublasDsyrk", "cublasCsyrk", "cublasZsyrk",
        "cublasCherk", "cublasZherk", "cublasSsyr2k", "cublasDsyr2k",
        "cublasCsyr2k", "cublasZsyr2k", "cublasCher2k", "cublasZher2k",
        "cublasSsymm", "cublasDsymm", "cublasCsymm", "cublasZsymm",
        "cublasChemm", "cublasZhemm", "cublasStrsm", "cublasDtrsm",
        "cublasCtrsm", "cublasZtrsm", "cublasStrmm", "cublasDtrmm",
        "cublasCtrmm", "cublasZtrmm");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               hasAncestor(functionDecl(
                                   anyOf(hasAttr(attr::CUDADevice),
                                         hasAttr(attr::CUDAGlobal))))))
                    .bind("kernelCall"),
                this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), parentStmt(),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), unless(parentStmt()),
                unless(hasAncestor(varDecl())),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedNotInitializeVarDecl"),
      this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), hasAncestor(varDecl()),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedInitializeVarDecl"),
      this);

  MF.addMatcher(
      unresolvedLookupExpr(
          hasAnyDeclaration(namedDecl(functionName())),
          hasParent(callExpr(unless(parentStmt())).bind("callExprUsed")))
          .bind("unresolvedCallUsed"),
      this);
}

void BLASFunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  auto getArgWithTypeCast = [&](const Expr* E, const std::string& CastType) {
    if (auto Cast = dyn_cast<CStyleCastExpr>(E->IgnoreImpCasts())) {
      return "(" + CastType + ")" + ExprAnalysis::ref(Cast->getSubExpr());
    } else {
      return "(" + CastType + ")" + ExprAnalysis::ref(E);
    }
  };

  bool IsAssigned = false;
  bool IsInitializeVarDecl = false;
  bool HasDeviceAttr = false;
  std::string FuncName = "";
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "kernelCall");
  if (CE) {
    HasDeviceAttr = true;
  } else if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCall"))) {
    if ((CE = getNodeAsType<CallExpr>(
             Result, "FunctionCallUsedNotInitializeVarDecl"))) {
      IsAssigned = true;
    } else if ((CE = getNodeAsType<CallExpr>(
                    Result, "FunctionCallUsedInitializeVarDecl"))) {
      IsAssigned = true;
      IsInitializeVarDecl = true;
    } else if (auto *ULE = getNodeAsType<UnresolvedLookupExpr>(
                   Result, "unresolvedCallUsed")) {
      CE = getNodeAsType<CallExpr>(Result, "callExprUsed");
      FuncName = ULE->getName().getAsString();
    } else {
      return;
    }
  }

  if (FuncName == "") {
    if (!CE->getDirectCallee())
      return;
    FuncName = CE->getDirectCallee()->getNameInfo().getName().getAsString();
  }

  const SourceManager *SM = Result.SourceManager;
  auto Loc = DpctGlobalInfo::getLocInfo(SM->getExpansionLoc(CE->getBeginLoc()));
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Loc.first +
                                              std::to_string(Loc.second)));

  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // There are some macros like "#define API API_v2"
  // so the function names we match should have the
  // suffix "_v2".
  bool IsMacroArg = SM->isMacroArgExpansion(CE->getBeginLoc()) &&
                    SM->isMacroArgExpansion(CE->getEndLoc());

  if (FuncNameBegin.isMacroID() && IsMacroArg) {
    FuncNameBegin = SM->getImmediateSpellingLoc(FuncNameBegin);
    FuncNameBegin = SM->getExpansionLoc(FuncNameBegin);
  } else if (FuncNameBegin.isMacroID()) {
    FuncNameBegin = SM->getExpansionLoc(FuncNameBegin);
  }

  if (FuncCallEnd.isMacroID() && IsMacroArg) {
    FuncCallEnd = SM->getImmediateSpellingLoc(FuncCallEnd);
    FuncCallEnd = SM->getExpansionLoc(FuncCallEnd);
  } else if (FuncCallEnd.isMacroID()) {
    FuncCallEnd = SM->getExpansionLoc(FuncCallEnd);
  }

  // Offset 1 is the length of the last token ")"
  FuncCallEnd = FuncCallEnd.getLocWithOffset(1);
  auto SR = getScopeInsertRange(CE, FuncNameBegin, FuncCallEnd);
  SourceLocation PrefixInsertLoc = SR.getBegin(), SuffixInsertLoc = SR.getEnd();

  auto FuncCallLength =
      SM->getCharacterData(FuncCallEnd) - SM->getCharacterData(FuncNameBegin);

  bool CanAvoidUsingLambda = false;
  SourceLocation OuterInsertLoc;
  std::string OriginStmtType;
  bool NeedUseLambda = isConditionOfFlowControl(
      CE, OriginStmtType, CanAvoidUsingLambda, OuterInsertLoc);
  bool IsInReturnStmt = isInReturnStmt(CE, OuterInsertLoc);
  bool CanAvoidBrace = false;
  const CompoundStmt *CS = findImmediateBlock(CE);
  if (CS && (CS->size() == 1)) {
    const Stmt *S = *(CS->child_begin());
    if (CE == S || dyn_cast<ReturnStmt>(S))
      CanAvoidBrace = true;
  }

  if (NeedUseLambda) {
    PrefixInsertLoc = FuncNameBegin;
    SuffixInsertLoc = FuncCallEnd;
  } else if (IsMacroArg) {
    NeedUseLambda = true;
    SourceRange SR = getFunctionRange(CE);
    PrefixInsertLoc = SR.getBegin();
    SuffixInsertLoc = SR.getEnd();
  } else if (IsInReturnStmt) {
    NeedUseLambda = true;
    CanAvoidUsingLambda = true;
    OriginStmtType = "return";
    // For some Legacy BLAS API (return the calculated value), below two
    // variables are needed. Although the function call is in return stmt, it
    // cannot move out and must use lambda.
    PrefixInsertLoc = FuncNameBegin;
    SuffixInsertLoc = FuncCallEnd;
  }

  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
      (FuncName == "cublasHgemmBatched" || FuncName == "cublasSgemmBatched" ||
       FuncName == "cublasDgemmBatched" || FuncName == "cublasCgemmBatched" ||
       FuncName == "cublasZgemmBatched" || FuncName == "cublasStrsmBatched" ||
       FuncName == "cublasDtrsmBatched" || FuncName == "cublasCtrsmBatched" ||
       FuncName == "cublasZtrsmBatched" || FuncName == "cublasGemmBatchedEx")) {
    report(FuncNameBegin, Diagnostics::API_NOT_MIGRATED, false, FuncName);
    return;
  }

  std::string IndentStr = getIndent(PrefixInsertLoc, *SM).str();
  // PrefixInsertStr: stmt + NL + indent
  // SuffixInsertStr: NL + indent + stmt
  std::string PrefixInsertStr, SuffixInsertStr;
  // Clean below five member variables before starting migration
  CallExprArguReplVec.clear();
  CallExprReplStr = "";
  NeedWaitAPICall = false;
  SyncAPIBufferAssignmentInThenBlock.clear();
  SyncAPIBufferAssignmentInElseBlock.clear();
  // TODO: Need to process the situation when scalar pointers (alpha, beta)
  // are device pointers.

  auto Item = MapNames::BLASAPIWithRewriter.find(FuncName);
  if (Item != MapNames::BLASAPIWithRewriter.end()) {
    std::string NewFunctionName = Item->second;
    if (HasDeviceAttr && !NewFunctionName.empty()) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), NewFunctionName);
      return;
    }
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  } else if (FuncName == "cublasSdgmm" || FuncName == "cublasDdgmm" ||
             FuncName == "cublasCdgmm" || FuncName == "cublasZdgmm") {
    std::string Replacement = "oneapi::mkl::blas::column_major::dgmm_batch";
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }
    BLASEnumInfo EnumInfo({}, -1, 1, -1);
    std::string BufferType;
    if (FuncName == "cublasSdgmm") {
      BufferType = "float";
    } else if (FuncName == "cublasDdgmm") {
      BufferType = "double";
    } else if (FuncName == "cublasCdgmm") {
      BufferType = "std::complex<float>";
    } else {
      BufferType = "std::complex<double>";
    }

    // initialize the replacement of each argument
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(i));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    std::string LdcTimesN =
        (needExtraParens(CE->getArg(9)) ? ("(" + CallExprArguReplVec[9] + ")")
                                        : CallExprArguReplVec[9]) +
        " * " +
        (needExtraParens(CE->getArg(3)) ? ("(" + CallExprArguReplVec[3] + ")")
                                        : CallExprArguReplVec[3]);

    // update the replacement of four enmu arguments
    if (const CStyleCastExpr *CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(1))) {
      std::string CurrentArgumentRepl;
      processParamIntCastToBLASEnum(CE->getArg(1), CSCE, 1, IndentStr, EnumInfo,
                                    PrefixInsertStr, CurrentArgumentRepl);
      CallExprArguReplVec[1] = CurrentArgumentRepl;
    }

    // update the replacement of three buffers
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      requestFeature(HelperFeatureEnum::device_ext);
      std::string BufferDecl;
      CallExprArguReplVec[4] = getBufferNameAndDeclStr(
          CE->getArg(4), BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[6] = getBufferNameAndDeclStr(
          CE->getArg(6), BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[8] = getBufferNameAndDeclStr(
          CE->getArg(8), BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
    } else {
      if (FuncName == "cublasCdgmm") {
        CallExprArguReplVec[4] =
            getArgWithTypeCast(CE->getArg(4), "std::complex<float>*");
        CallExprArguReplVec[6] =
            getArgWithTypeCast(CE->getArg(6), "std::complex<float>*");
        CallExprArguReplVec[8] =
            getArgWithTypeCast(CE->getArg(8), "std::complex<float>*");
      } else if (FuncName == "cublasZdgmm") {
        CallExprArguReplVec[4] =
            getArgWithTypeCast(CE->getArg(4), "std::complex<double>*");
        CallExprArguReplVec[6] =
            getArgWithTypeCast(CE->getArg(6), "std::complex<double>*");
        CallExprArguReplVec[8] =
            getArgWithTypeCast(CE->getArg(8), "std::complex<double>*");
      }
    }

    // Insert some arguments since we are now using batch API
    // If we have new dedicated API for migrating dgmm in the future,
    // then we can remove the argument insertion.
    CallExprArguReplVec.push_back(LdcTimesN); // stride_c
    CallExprArguReplVec.push_back("1"); // batch_size
    auto Iter = CallExprArguReplVec.begin();
    std::advance(Iter, 8);
    CallExprArguReplVec.insert(Iter, "0"); // stride_b
    Iter = CallExprArguReplVec.begin();
    std::advance(Iter, 6);
    CallExprArguReplVec.insert(Iter, "0"); // stride_a

    CallExprReplStr = getFinalCallExprStr(Replacement) + CallExprReplStr;

    if (NeedUseLambda) {
      if (PrefixInsertStr.empty() && SuffixInsertStr.empty()) {
        NeedUseLambda = false;
      }
    }

    applyMigrationText(NeedUseLambda, IsMacroArg, CanAvoidBrace,
                       CanAvoidUsingLambda, OriginStmtType, IsAssigned,
                       OuterInsertLoc, PrefixInsertLoc, SuffixInsertLoc,
                       FuncNameBegin, FuncCallEnd, FuncCallLength, IndentStr,
                       PrefixInsertStr, SuffixInsertStr);
  } else if (MapNames::BLASFuncReplInfoMap.find(FuncName) !=
             MapNames::BLASFuncReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::BLASFuncReplInfoMap.find(FuncName);
    MapNames::BLASFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;
    BLASEnumInfo EnumInfo(
        ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
        ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      std::string CurrentArgumentRepl;
      const CStyleCastExpr *CSCE = nullptr;
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            requestFeature(HelperFeatureEnum::device_ext);
            auto DefaultQueue = DpctGlobalInfo::getDefaultQueue(CE->getArg(i));
            std::string ResultTempPtr =
                "res_temp_ptr_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            std::string ResultTempHost =
                "res_temp_host_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "int64_t* " + ResultTempPtr +
                              " = " + MapNames::getClNamespace() +
                              "malloc_shared<int64_t>(" + "1, " + DefaultQueue +
                              ");" + getNL() + IndentStr;
            SuffixInsertStr = SuffixInsertStr + getNL() + IndentStr + "int " +
                              ResultTempHost + " = (int)*" + ResultTempPtr +
                              ";" + getNL() + IndentStr +
                              MapNames::getDpctNamespace() + "dpct_memcpy(" +
                              ExprAnalysis::ref(CE->getArg(i)) + ", &" +
                              ResultTempHost + ", sizeof(int));" + getNL() +
                              IndentStr + MapNames::getClNamespace() + "free(" +
                              ResultTempPtr + ", " + DefaultQueue + ");";
            CurrentArgumentRepl = ResultTempPtr;
          } else {
            CurrentArgumentRepl = ExprAnalysis::ref(CE->getArg(i));
          }
        } else {
          std::string BufferDecl = "";
          std::string BufferName = "";
          auto MaySyncAPIIter = MapNames::MaySyncBLASFunc.find(FuncName);
          auto MaySyncAPIIWithMultiArgsIter =
              MapNames::MaySyncBLASFuncWithMultiArgs.find(FuncName);
          if (MaySyncAPIIter != MapNames::MaySyncBLASFunc.end() &&
              i == MaySyncAPIIter->second.second) {
            BufferName = processSyncAPIBufferArg(
                FuncName, CE, PrefixInsertStr, IndentStr,
                MaySyncAPIIter->second.first, i);
          } else if (MaySyncAPIIWithMultiArgsIter !=
                         MapNames::MaySyncBLASFuncWithMultiArgs.end() &&
                     MaySyncAPIIWithMultiArgsIter->second.find(i) !=
                         MaySyncAPIIWithMultiArgsIter->second.end()) {
            auto ArgIter = MaySyncAPIIWithMultiArgsIter->second.find(i);
            BufferName = processSyncAPIBufferArg(FuncName, CE, PrefixInsertStr,
                                                 IndentStr, ArgIter->second, i);
          } else if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            BufferName = processSyncAPIBufferArg(FuncName, CE, PrefixInsertStr,
                                                 IndentStr, "int", i);
          } else {
            requestFeature(HelperFeatureEnum::device_ext);
            BufferName = getBufferNameAndDeclStr(
                CE->getArg(i), ReplInfo.BufferTypeInfo[IndexTemp], IndentStr,
                BufferDecl);
            PrefixInsertStr = PrefixInsertStr + BufferDecl;
          }

          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            std::string ResultTempBuf =
                "res_temp_buf_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + MapNames::getClNamespace() +
                              "buffer<int64_t> " + ResultTempBuf + "(" +
                              MapNames::getClNamespace() + "range<1>(1));" +
                              getNL() + IndentStr;
            SuffixInsertStr =
                SuffixInsertStr + getNL() + IndentStr + BufferName +
                ".get_access<" + MapNames::getClNamespace() +
                "access_mode::" + "write>()[0] = (int)" + ResultTempBuf + "." +
                "get_access<" + MapNames::getClNamespace() +
                "access_mode::read>()[0];";
            CurrentArgumentRepl = ResultTempBuf;
          } else {
            CurrentArgumentRepl = BufferName;
          }
        }
      } else if (isReplIndex(i, ReplInfo.PointerIndexInfo, IndexTemp)) {
        ExprAnalysis EA(CE->getArg(i));
        CurrentArgumentRepl = getValueStr(CE->getArg(i), EA.getReplacedString(),
                                          CallExprArguReplVec[0]);
      } else if ((CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(i)))) {
        processParamIntCastToBLASEnum(CE->getArg(i), CSCE, i, IndentStr,
                                      EnumInfo, PrefixInsertStr,
                                      CurrentArgumentRepl);
      } else {
        ExprAnalysis EA;
        EA.analyze(CE->getArg(i));
        CurrentArgumentRepl = EA.getReplacedString();
      }
      CallExprArguReplVec.push_back(CurrentArgumentRepl);
    }

    if (FuncName == "cublasIsamax_v2" || FuncName == "cublasIdamax_v2" ||
        FuncName == "cublasIsamin_v2" || FuncName == "cublasIdamin_v2") {
      CallExprArguReplVec.push_back("oneapi::mkl::index_base::one");
    }

    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
      if (FuncName == "cublasSrotm_v2") {
        CallExprArguReplVec[6] =
            "const_cast<float*>(" + CallExprArguReplVec[6] + ")";
      } else if (FuncName == "cublasDrotm_v2") {
        CallExprArguReplVec[6] =
            "const_cast<double*>(" + CallExprArguReplVec[6] + ")";
      }
      addWait(FuncName, CE, PrefixInsertStr, SuffixInsertStr, IndentStr);
      if (MapNames::MustSyncBLASFunc.find(FuncName) !=
          MapNames::MustSyncBLASFunc.end())
        NeedWaitAPICall = true;
    } else {
      printIfStmt(FuncName, CE, PrefixInsertStr, IndentStr);
    }
    CallExprReplStr = getFinalCallExprStr(Replacement) + CallExprReplStr;

    if (NeedUseLambda) {
      if (PrefixInsertStr.empty() && SuffixInsertStr.empty()) {
        // If there is one API call in the migrated code, it is unnecessary to
        // use a lambda expression
        NeedUseLambda = false;
      }
    }

    applyMigrationText(NeedUseLambda, IsMacroArg, CanAvoidBrace,
                       CanAvoidUsingLambda, OriginStmtType, IsAssigned,
                       OuterInsertLoc, PrefixInsertLoc, SuffixInsertLoc,
                       FuncNameBegin, FuncCallEnd, FuncCallLength, IndentStr,
                       PrefixInsertStr, SuffixInsertStr);
  } else if (MapNames::BLASFuncComplexReplInfoMap.find(FuncName) !=
             MapNames::BLASFuncComplexReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::BLASFuncComplexReplInfoMap.find(FuncName);
    MapNames::BLASFuncComplexReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;
    BLASEnumInfo EnumInfo(
        ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
        ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

    int ArgNum = CE->getNumArgs();

    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      std::string CurrentArgumentRepl;
      const CStyleCastExpr *CSCE = nullptr;
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            auto DefaultQueue = DpctGlobalInfo::getDefaultQueue(CE->getArg(i));
            requestFeature(HelperFeatureEnum::device_ext);
            std::string ResultTempPtr =
                "res_temp_ptr_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            std::string ResultTempHost =
                "res_temp_host_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "int64_t* " + ResultTempPtr +
                              " = " + MapNames::getClNamespace() +
                              "malloc_shared<int64_t>(" + "1, " + DefaultQueue +
                              ");" + getNL() + IndentStr;
            SuffixInsertStr = SuffixInsertStr + getNL() + IndentStr + "int " +
                              ResultTempHost + " = (int)*" + ResultTempPtr +
                              ";" + getNL() + IndentStr +
                              MapNames::getDpctNamespace() + "dpct_memcpy(" +
                              ExprAnalysis::ref(CE->getArg(i)) + ", &" +
                              ResultTempHost + ", sizeof(int));" + getNL() +
                              IndentStr + MapNames::getClNamespace() + "free(" +
                              ResultTempPtr + ", " + DefaultQueue + ");";
            CurrentArgumentRepl = ResultTempPtr;
          } else if (ReplInfo.BufferTypeInfo[IndexTemp] ==
                         "std::complex<float>" ||
                     ReplInfo.BufferTypeInfo[IndexTemp] ==
                         "std::complex<double>") {
            CurrentArgumentRepl = getArgWithTypeCast(
                CE->getArg(i), ReplInfo.BufferTypeInfo[IndexTemp] + "*");
          } else {
            CurrentArgumentRepl = ExprAnalysis::ref(CE->getArg(i));
          }
        } else {
          std::string BufferDecl = "";
          std::string BufferName = "";
          auto MaySyncAPIIter = MapNames::MaySyncBLASFunc.find(FuncName);
          auto MaySyncAPIIWithMultiArgsIter =
              MapNames::MaySyncBLASFuncWithMultiArgs.find(FuncName);
          if (MaySyncAPIIter != MapNames::MaySyncBLASFunc.end() &&
              i == MaySyncAPIIter->second.second) {
            BufferName = processSyncAPIBufferArg(
                FuncName, CE, PrefixInsertStr, IndentStr,
                MaySyncAPIIter->second.first, i);
          } else if (MaySyncAPIIWithMultiArgsIter !=
                         MapNames::MaySyncBLASFuncWithMultiArgs.end() &&
                     MaySyncAPIIWithMultiArgsIter->second.find(i) !=
                         MaySyncAPIIWithMultiArgsIter->second.end()) {
            auto ArgIter = MaySyncAPIIWithMultiArgsIter->second.find(i);
            BufferName = processSyncAPIBufferArg(FuncName, CE, PrefixInsertStr,
                                                 IndentStr, ArgIter->second, i);
          } else if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            BufferName = processSyncAPIBufferArg(FuncName, CE, PrefixInsertStr,
                                                 IndentStr, "int", i);
          } else {
            requestFeature(HelperFeatureEnum::device_ext);
            BufferName = getBufferNameAndDeclStr(
                CE->getArg(i), ReplInfo.BufferTypeInfo[IndexTemp], IndentStr,
                BufferDecl);
            PrefixInsertStr = PrefixInsertStr + BufferDecl;
          }

          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            std::string ResultTempBuf =
                "res_temp_buf_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + MapNames::getClNamespace() +
                              "buffer<int64_t> " + ResultTempBuf + "(" +
                              MapNames::getClNamespace() + "range<1>(1));" +
                              getNL() + IndentStr;
            SuffixInsertStr =
                SuffixInsertStr + getNL() + IndentStr + BufferName +
                ".get_access<" + MapNames::getClNamespace() +
                "access_mode::" + "write>()[0] = (int)" + ResultTempBuf + "." +
                "get_access<" + MapNames::getClNamespace() +
                "access_mode::read>()[0];";
            CurrentArgumentRepl = ResultTempBuf;
          } else {
            CurrentArgumentRepl = BufferName;
          }
        }
      } else if (isReplIndex(i, ReplInfo.PointerIndexInfo, IndexTemp)) {
        ExprAnalysis EA(CE->getArg(i));
        CurrentArgumentRepl = getValueStr(CE->getArg(i), EA.getReplacedString(),
                                          CallExprArguReplVec[0],
                                          ReplInfo.PointerTypeInfo[IndexTemp]);
      } else if ((CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(i)))) {
        processParamIntCastToBLASEnum(CE->getArg(i), CSCE, i, IndentStr,
                                      EnumInfo, PrefixInsertStr,
                                      CurrentArgumentRepl);
      } else {
        ExprAnalysis EA;
        EA.analyze(CE->getArg(i));
        CurrentArgumentRepl = EA.getReplacedString();
      }

      CallExprArguReplVec.push_back(CurrentArgumentRepl);
    }

    if (FuncName == "cublasIcamax_v2" || FuncName == "cublasIzamax_v2" ||
        FuncName == "cublasIcamin_v2" || FuncName == "cublasIzamin_v2") {
      CallExprArguReplVec.push_back("oneapi::mkl::index_base::one");
    }

    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
      addWait(FuncName, CE, PrefixInsertStr, SuffixInsertStr, IndentStr);
      if (MapNames::MustSyncBLASFunc.find(FuncName) !=
          MapNames::MustSyncBLASFunc.end())
        NeedWaitAPICall = true;
    } else {
      printIfStmt(FuncName, CE, PrefixInsertStr, IndentStr);
    }

    CallExprReplStr = getFinalCallExprStr(Replacement) + CallExprReplStr;

    if (NeedUseLambda) {
      if (PrefixInsertStr.empty() && SuffixInsertStr.empty()) {
        // If there is one API call in the migrated code, it is unnecessary to
        // use a lambda expression
        NeedUseLambda = false;
      }
    }

    applyMigrationText(NeedUseLambda, IsMacroArg, CanAvoidBrace,
                       CanAvoidUsingLambda, OriginStmtType, IsAssigned,
                       OuterInsertLoc, PrefixInsertLoc, SuffixInsertLoc,
                       FuncNameBegin, FuncCallEnd, FuncCallLength, IndentStr,
                       PrefixInsertStr, SuffixInsertStr);
  } else if (MapNames::LegacyBLASFuncReplInfoMap.find(FuncName) !=
             MapNames::LegacyBLASFuncReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::LegacyBLASFuncReplInfoMap.find(FuncName);
    MapNames::BLASFuncComplexReplInfo ReplInfo = ReplInfoPair->second;
    requestFeature(HelperFeatureEnum::device_ext);
    CallExprReplStr = CallExprReplStr + ReplInfo.ReplName + "(*" +
                      MapNames::getDpctNamespace() +
                      "get_current_device().get_saved_queue()";
    std::string IndentStr =
        getIndent(PrefixInsertLoc, (Result.Context)->getSourceManager()).str();

    std::string VarType;
    std::string VarName;
    std::string DeclOutOfBrace;
    const VarDecl *VD = 0;
    if (IsInitializeVarDecl) {
      VD = getAncestralVarDecl(CE);
      if (VD) {
        VarType = VD->getType().getAsString();
        if (VarType == "cuComplex" || VarType == "cuFloatComplex") {
          VarType = MapNames::getClNamespace() + "float2";
        }
        if (VarType == "cuDoubleComplex") {
          VarType = MapNames::getClNamespace() + "double2";
        }
        VarName = VD->getNameAsString();
      } else {
        assert(0 && "Fail to get VarDecl information");
        return;
      }
      DeclOutOfBrace = VarType + " " + VarName + ";" + getNL() + IndentStr;
    }
    std::vector<std::string> ParamsStrsVec =
        getParamsAsStrs(CE, *(Result.Context));
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
          if ((FuncName == "cublasSrotm" || FuncName == "cublasDrotm") &&
              i == 5) {
            CallExprReplStr = CallExprReplStr + ", const_cast<" +
                              ReplInfo.BufferTypeInfo[IndexTemp] + "*>(" +
                              ExprAnalysis::ref(CE->getArg(5)) + ")";
          } else if (ReplInfo.BufferTypeInfo[IndexTemp] ==
                         "std::complex<float>" ||
                     ReplInfo.BufferTypeInfo[IndexTemp] ==
                         "std::complex<double>") {
            CallExprReplStr =
                CallExprReplStr + ", " +
                getArgWithTypeCast(CE->getArg(i),
                                   ReplInfo.BufferTypeInfo[IndexTemp] + "*");
          } else {
            CallExprReplStr = CallExprReplStr + ", " + ParamsStrsVec[i];
          }
        } else {
          requestFeature(HelperFeatureEnum::device_ext);
          std::string BufferDecl;
          std::string BufferName = getBufferNameAndDeclStr(
              CE->getArg(i), ReplInfo.BufferTypeInfo[IndexTemp], IndentStr,
              BufferDecl);
          CallExprReplStr = CallExprReplStr + ", " + BufferName;
          PrefixInsertStr = PrefixInsertStr + BufferDecl;
        }
      } else if (isReplIndex(i, ReplInfo.PointerIndexInfo, IndexTemp)) {
        if (ReplInfo.PointerTypeInfo[IndexTemp] == "float" ||
            ReplInfo.PointerTypeInfo[IndexTemp] == "double") {
          // This code path is only for legacy cublasSrotmg and cublasDrotmg
          CallExprReplStr = CallExprReplStr + ", " +
                            getDrefName(CE->getArg(i)->IgnoreImplicit());
        } else {
          if (isAnIdentifierOrLiteral(CE->getArg(i)))
            CallExprReplStr =
                CallExprReplStr + ", " + ReplInfo.PointerTypeInfo[IndexTemp] +
                "(" + ParamsStrsVec[i] + ".x()," + ParamsStrsVec[i] + ".y())";
          else
            CallExprReplStr = CallExprReplStr + ", " +
                              ReplInfo.PointerTypeInfo[IndexTemp] + "((" +
                              ParamsStrsVec[i] + ").x(),(" + ParamsStrsVec[i] +
                              ").y())";
        }
      } else if (isReplIndex(i, ReplInfo.OperationIndexInfo, IndexTemp)) {
        Expr::EvalResult ER;
        if (CE->getArg(i)->EvaluateAsInt(ER, *Result.Context) &&
            !CE->getArg(i)->getBeginLoc().isMacroID()) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 'N' || Value == 'n') {
            CallExprReplStr =
                CallExprReplStr + ", oneapi::mkl::transpose::nontrans";
          } else if (Value == 'T' || Value == 't') {
            CallExprReplStr =
                CallExprReplStr + ", oneapi::mkl::transpose::trans";
          } else {
            CallExprReplStr =
                CallExprReplStr + ", oneapi::mkl::transpose::conjtrans";
          }
        } else {
          std::string TransParamName;
          if (CE->getArg(i)->HasSideEffects(DpctGlobalInfo::getContext())) {
            TransParamName =
                "transpose_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "auto " + TransParamName +
                              " = " + ParamsStrsVec[i] + ";" + getNL() +
                              IndentStr;
          } else {
            TransParamName = ParamsStrsVec[i];
          }
          CallExprReplStr = CallExprReplStr + ", " + "(" + TransParamName +
                            "=='N'||" + TransParamName +
                            "=='n') ? oneapi::mkl::transpose::nontrans: ((" +
                            TransParamName + "=='T'||" + TransParamName +
                            "=='t') ? oneapi::mkl::transpose::"
                            "trans : oneapi::mkl::transpose::conjtrans)";
        }
      } else if (ReplInfo.FillModeIndexInfo == i) {
        Expr::EvalResult ER;
        if (CE->getArg(i)->EvaluateAsInt(ER, *Result.Context) &&
            !CE->getArg(i)->getBeginLoc().isMacroID()) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 'U' || Value == 'u') {
            CallExprReplStr = CallExprReplStr + ", oneapi::mkl::uplo::upper";
          } else {
            CallExprReplStr = CallExprReplStr + ", oneapi::mkl::uplo::lower";
          }
        } else {
          std::string FillParamName;
          if (CE->getArg(i)->HasSideEffects(DpctGlobalInfo::getContext())) {
            FillParamName =
                "fillmode_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "auto " + FillParamName +
                              " = " + ParamsStrsVec[i] + ";" + getNL() +
                              IndentStr;
          } else {
            FillParamName = ParamsStrsVec[i];
          }
          CallExprReplStr =
              CallExprReplStr + ", " + "(" + FillParamName + "=='L'||" +
              FillParamName +
              "=='l') ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper";
        }
      } else if (ReplInfo.SideModeIndexInfo == i) {
        Expr::EvalResult ER;
        if (CE->getArg(i)->EvaluateAsInt(ER, *Result.Context) &&
            !CE->getArg(i)->getBeginLoc().isMacroID()) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 'L' || Value == 'l') {
            CallExprReplStr = CallExprReplStr + ", oneapi::mkl::side::left";
          } else {
            CallExprReplStr = CallExprReplStr + ", oneapi::mkl::side::right";
          }
        } else {
          std::string SideParamName;
          if (CE->getArg(i)->HasSideEffects(DpctGlobalInfo::getContext())) {
            SideParamName =
                "sidemode_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "auto " + SideParamName +
                              " = " + ParamsStrsVec[i] + ";" + getNL() +
                              IndentStr;
          } else {
            SideParamName = ParamsStrsVec[i];
          }
          CallExprReplStr =
              CallExprReplStr + ", " + "(" + SideParamName + "=='L'||" +
              SideParamName +
              "=='l') ? oneapi::mkl::side::left : oneapi::mkl::side::right";
        }
      } else if (ReplInfo.DiagTypeIndexInfo == i) {
        Expr::EvalResult ER;
        if (CE->getArg(i)->EvaluateAsInt(ER, *Result.Context) &&
            !CE->getArg(i)->getBeginLoc().isMacroID()) {
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 'N' || Value == 'n') {
            CallExprReplStr = CallExprReplStr + ", oneapi::mkl::diag::nonunit";
          } else {
            CallExprReplStr = CallExprReplStr + ", oneapi::mkl::diag::unit";
          }
        } else {
          std::string DiagParamName;
          if (CE->getArg(i)->HasSideEffects(DpctGlobalInfo::getContext())) {
            DiagParamName =
                "diagtype_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "auto " + DiagParamName +
                              " = " + ParamsStrsVec[i] + ";" + getNL() +
                              IndentStr;
          } else {
            DiagParamName = ParamsStrsVec[i];
          }
          CallExprReplStr =
              CallExprReplStr + ", " + "(" + DiagParamName + "=='N'||" +
              DiagParamName +
              "=='n') ? oneapi::mkl::diag::nonunit : oneapi::mkl::diag::unit";
        }
      } else {
        CallExprReplStr = CallExprReplStr + ", " + ParamsStrsVec[i];
      }
    }

    if (FuncName == "cublasIsamax" || FuncName == "cublasIdamax" ||
        FuncName == "cublasIcamax" || FuncName == "cublasIzamax" ||
        FuncName == "cublasIsamin" || FuncName == "cublasIdamin" ||
        FuncName == "cublasIcamin" || FuncName == "cublasIzamin") {
      CallExprArguReplVec.push_back("oneapi::mkl::index_base::one");
    }

    // All legacy APIs are synchronous
    if (FuncName == "cublasSnrm2" || FuncName == "cublasDnrm2" ||
        FuncName == "cublasScnrm2" || FuncName == "cublasDznrm2" ||
        FuncName == "cublasSdot" || FuncName == "cublasDdot" ||
        FuncName == "cublasCdotu" || FuncName == "cublasCdotc" ||
        FuncName == "cublasZdotu" || FuncName == "cublasZdotc" ||
        FuncName == "cublasIsamax" || FuncName == "cublasIdamax" ||
        FuncName == "cublasIcamax" || FuncName == "cublasIzamax" ||
        FuncName == "cublasIsamin" || FuncName == "cublasIdamin" ||
        FuncName == "cublasIcamin" || FuncName == "cublasIzamin" ||
        FuncName == "cublasSasum" || FuncName == "cublasDasum" ||
        FuncName == "cublasScasum" || FuncName == "cublasDzasum") {
      // APIs which have return value
      std::string ResultTempPtr =
          "res_temp_ptr_ct" +
          std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string ResultTempVal =
          "res_temp_val_ct" +
          std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string ResultTempBuf =
          "res_temp_buf_ct" +
          std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string ResultType =
          ReplInfo.BufferTypeInfo[ReplInfo.BufferTypeInfo.size() - 1];
      std::string ReturnValueParamsStr;
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
        requestFeature(HelperFeatureEnum::device_ext);
        auto DefaultQueue = DpctGlobalInfo::getDefaultQueue(CE);
        PrefixInsertStr = PrefixInsertStr + ResultType + "* " + ResultTempPtr +
                          " = " + MapNames::getClNamespace() +
                          "malloc_shared<" + ResultType + ">(1, " + DefaultQueue + ");" +
                          getNL() + IndentStr + CallExprReplStr + ", " +
                          ResultTempPtr + ").wait();" + getNL() + IndentStr;

        ReturnValueParamsStr =
            "(" + ResultTempPtr + "->real(), " + ResultTempPtr + "->imag())";

        if (NeedUseLambda) {
          PrefixInsertStr = PrefixInsertStr + ResultType + " " + ResultTempVal +
                            " = *" + ResultTempPtr + ";" + getNL() + IndentStr +
                            MapNames::getClNamespace() + "free(" +
                            ResultTempPtr + ", " + DefaultQueue + ");" +
                            getNL() + IndentStr;
          ReturnValueParamsStr =
              "(" + ResultTempVal + ".real(), " + ResultTempVal + ".imag())";
        } else {
          SuffixInsertStr = SuffixInsertStr + getNL() + IndentStr +
                            MapNames::getClNamespace() + "free(" +
                            ResultTempPtr + ", " + DefaultQueue + ");";
        }
      } else {
        PrefixInsertStr = PrefixInsertStr + MapNames::getClNamespace() +
                          "buffer<" + ResultType + "> " + ResultTempBuf + "(" +
                          MapNames::getClNamespace() + "range<1>(1));" +
                          getNL() + IndentStr + CallExprReplStr + ", " +
                          ResultTempBuf + ");" + getNL() + IndentStr;
        ReturnValueParamsStr =
            "(" + ResultTempBuf + ".get_access<" + MapNames::getClNamespace() +
            "access_mode::read>()[0].real(), " + ResultTempBuf +
            ".get_access<" + MapNames::getClNamespace() +
            "access_mode::read>()[0].imag())";
      }

      std::string Repl;
      if (FuncName == "cublasCdotu" || FuncName == "cublasCdotc") {
        Repl = MapNames::getClNamespace() + "float2" + ReturnValueParamsStr;
      } else if (FuncName == "cublasZdotu" || FuncName == "cublasZdotc") {
        Repl = MapNames::getClNamespace() + "double2" + ReturnValueParamsStr;
      } else {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
          if (NeedUseLambda)
            Repl = ResultTempVal;
          else
            Repl = "*" + ResultTempPtr;
        } else {
          Repl = ResultTempBuf + ".get_access<" + MapNames::getClNamespace() +
                 "access_mode::read>()[0]";
        }
      }
      if (NeedUseLambda) {
        std::string CallRepl = "return " + Repl;

        insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                          std::string("[&](){") + getNL() + IndentStr +
                              PrefixInsertStr,
                          ";" + SuffixInsertStr + getNL() + IndentStr + "}()");
        emplaceTransformation(new ReplaceStmt(CE, CallRepl));
      } else {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            DeclOutOfBrace + "{" + getNL() + IndentStr +
                                PrefixInsertStr,
                            SuffixInsertStr + getNL() + IndentStr + "}");
        else
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            DeclOutOfBrace + PrefixInsertStr,
                            std::move(SuffixInsertStr));

        if (IsInitializeVarDecl) {
          auto ParentNodes = (Result.Context)->getParents(*VD);
          const DeclStmt *DS = 0;
          if ((DS = ParentNodes[0].get<DeclStmt>())) {
            emplaceTransformation(
                new ReplaceStmt(DS, VarName + " = " + Repl + ";"));
          } else {
            assert(0 && "Fail to get Var Decl Stmt");
            return;
          }
        } else {
          emplaceTransformation(new ReplaceStmt(CE, Repl));
        }
      }
    } else {
      // APIs which haven't return value
      if (NeedUseLambda) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
          CallExprReplStr = CallExprReplStr + ").wait()";
        } else {
          CallExprReplStr = CallExprReplStr + ")";
        }
        if (CanAvoidUsingLambda) {
          std::string InsertStr;
          if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
            InsertStr = DeclOutOfBrace + "{" + getNL() + IndentStr +
                        PrefixInsertStr + CallExprReplStr + ";" +
                        SuffixInsertStr + getNL() + IndentStr + "}" + getNL() +
                        IndentStr;
          else
            InsertStr = DeclOutOfBrace + PrefixInsertStr + CallExprReplStr +
                        ";" + SuffixInsertStr + getNL() + IndentStr;
          emplaceTransformation(
              new InsertText(OuterInsertLoc, std::move(InsertStr)));
          // APIs in this code path haven't return value, so remove the CallExpr
          emplaceTransformation(
              new ReplaceText(FuncNameBegin, FuncCallLength, ""));
        } else {
          emplaceTransformation(
              new ReplaceStmt(CE, std::move(CallExprReplStr)));
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            std::string("[&](){") + getNL() + IndentStr +
                                PrefixInsertStr,
                            getNL() + IndentStr + std::string("}()"));
        }
      } else {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
          CallExprReplStr = CallExprReplStr + ").wait()";
        } else {
          CallExprReplStr = CallExprReplStr + ")";
        }
        emplaceTransformation(new ReplaceStmt(CE, std::move(CallExprReplStr)));
        if (!PrefixInsertStr.empty()) {
          if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
            insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                              std::string("{") + getNL() + IndentStr +
                                  PrefixInsertStr,
                              getNL() + IndentStr + std::string("}"));
          else
            insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                              std::move(PrefixInsertStr), "");
        }
      }
    }
  } else if (FuncName == "cublasCreate_v2" || FuncName == "cublasDestroy_v2" ||
             FuncName == "cublasSetStream_v2" ||
             FuncName == "cublasGetStream_v2" ||
             FuncName == "cublasSetKernelStream") {
    SourceRange SR = getFunctionRange(CE);
    auto Len = SM->getDecomposedLoc(SR.getEnd()).second -
               SM->getDecomposedLoc(SR.getBegin()).second;

    std::string Repl;

    if (FuncName == "cublasCreate_v2") {
      std::string LHS = getDrefName(CE->getArg(0));
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
      Repl = LHS + " = &{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
    } else if (FuncName == "cublasDestroy_v2") {
      dpct::ExprAnalysis EA(CE->getArg(0));
      Repl = EA.getReplacedString() + " = nullptr";
    } else if (FuncName == "cublasSetStream_v2") {
      dpct::ExprAnalysis EA0(CE->getArg(0));
      dpct::ExprAnalysis EA1(CE->getArg(1));
      Repl = EA0.getReplacedString() + " = " + EA1.getReplacedString();
    } else if (FuncName == "cublasGetStream_v2") {
      dpct::ExprAnalysis EA0(CE->getArg(0));
      std::string LHS = getDrefName(CE->getArg(1));
      Repl = LHS + " = " + EA0.getReplacedString();
    } else if (FuncName == "cublasSetKernelStream") {
      dpct::ExprAnalysis EA(CE->getArg(0));
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_CurrentDevice);
      requestFeature(HelperFeatureEnum::device_ext);
      Repl = "{{NEEDREPLACED" + std::to_string(Index) + "}}.set_saved_queue(" +
             EA.getReplacedString() + ")";
    } else {
      return;
    }

    if (SM->isMacroArgExpansion(CE->getBeginLoc()) &&
        SM->isMacroArgExpansion(CE->getEndLoc())) {
      if (IsAssigned) {
        requestFeature(HelperFeatureEnum::device_ext);
        emplaceTransformation(new ReplaceText(
            SR.getBegin(), Len, "DPCT_CHECK_ERROR(" + Repl + ")"));
      } else {
        emplaceTransformation(
            new ReplaceText(SR.getBegin(), Len, std::move(Repl)));
      }
    } else {
      if (IsAssigned) {
        requestFeature(HelperFeatureEnum::device_ext);
        emplaceTransformation(
            new ReplaceStmt(CE, true, "DPCT_CHECK_ERROR(" + Repl + ")"));
      } else {
        emplaceTransformation(new ReplaceStmt(CE, true, Repl));
      }
    }
  } else if (FuncName == "cublasInit" || FuncName == "cublasShutdown" ||
             FuncName == "cublasGetError") {
    // Remove these three function calls.
    // TODO: migrate functions when they are in template
    auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
    SourceRange SR = getFunctionRange(CE);
    auto Len = SM->getDecomposedLoc(SR.getEnd()).second -
               SM->getDecomposedLoc(SR.getBegin()).second;
    if (SM->isMacroArgExpansion(CE->getBeginLoc()) &&
        SM->isMacroArgExpansion(CE->getEndLoc())) {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceText(SR.getBegin(), Len, "0"));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceText(SR.getBegin(), Len, "0"));
      }
    } else {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceStmt(CE, false, "0"));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceStmt(CE, false, ""));
      }
    }
  } else if (FuncName == "cublasGetPointerMode_v2" ||
             FuncName == "cublasSetPointerMode_v2" ||
             FuncName == "cublasGetAtomicsMode" ||
             FuncName == "cublasSetAtomicsMode" ||
             FuncName == "cublasGetMathMode" ||
             FuncName == "cublasSetMathMode") {
    std::string Msg = "this call is redundant in SYCL.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, true, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, true, ""));
    }
  } else if (FuncName == "cublasSetVector" || FuncName == "cublasGetVector" ||
             FuncName == "cublasSetVectorAsync" ||
             FuncName == "cublasGetVectorAsync") {
    if (HasDeviceAttr) {
      report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName),
             MapNames::getDpctNamespace() + "matrix_mem_copy");
      return;
    }

    std::vector<std::string> ParamsStrsVec =
        getParamsAsStrs(CE, *(Result.Context));
    const Expr *IncxExpr = CE->getArg(3);
    const Expr *IncyExpr = CE->getArg(5);
    Expr::EvalResult IncxExprResult, IncyExprResult;

    if (IncxExpr->EvaluateAsInt(IncxExprResult, *Result.Context) &&
        IncyExpr->EvaluateAsInt(IncyExprResult, *Result.Context)) {
      std::string IncxStr =
          IncxExprResult.Val.getAsString(*Result.Context, IncxExpr->getType());
      std::string IncyStr =
          IncyExprResult.Val.getAsString(*Result.Context, IncyExpr->getType());
      if (IncxStr != IncyStr) {
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE,
               false, MapNames::ITFName.at(FuncName),
               "parameter " + ParamsStrsVec[3] +
                   " does not equal to parameter " + ParamsStrsVec[5]);
      } else if ((IncxStr == IncyStr) && (IncxStr != "1")) {
        // incx equals to incy, but does not equal to 1. Performance issue may
        // occur.
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE,
               false, MapNames::ITFName.at(FuncName),
               "parameter " + ParamsStrsVec[3] + " equals to parameter " +
                   ParamsStrsVec[5] + " but greater than 1");
      }
    } else {
      report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE, false,
             MapNames::ITFName.at(FuncName),
             "parameter(s) " + ParamsStrsVec[3] + " and/or " +
                 ParamsStrsVec[5] + " could not be evaluated");
    }

    std::string XStr = "(void*)" + getExprString(CE->getArg(2), true);
    std::string YStr = "(void*)" + getExprString(CE->getArg(4), true);
    std::string IncX = getExprString(CE->getArg(3), false);
    std::string IncY = getExprString(CE->getArg(5), false);
    std::string ElemSize = getExprString(CE->getArg(1), false);
    std::string ElemNum = getExprString(CE->getArg(0), false);

    std::string Replacement = "(" + YStr + ", " + XStr + ", " + IncY + ", " +
                              IncX + ", 1, " + ElemNum + ", " + ElemSize;

    requestFeature(HelperFeatureEnum::device_ext);
    if (FuncName == "cublasGetVector" || FuncName == "cublasSetVector") {
      Replacement =
          MapNames::getDpctNamespace() + "matrix_mem_copy" + Replacement + ")";
    } else {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(6));
      Replacement = MapNames::getDpctNamespace() + "matrix_mem_copy" +
                    Replacement + ", " + MapNames::getDpctNamespace() +
                    "automatic, *" + EA.getReplacedString() + ", true)";
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));

    if (IsAssigned) {
      requestFeature(HelperFeatureEnum::device_ext);
      insertAroundStmt(CE, "DPCT_CHECK_ERROR(", ")");
    }
  } else if (FuncName == "cublasSetMatrix" || FuncName == "cublasGetMatrix" ||
             FuncName == "cublasSetMatrixAsync" ||
             FuncName == "cublasGetMatrixAsync") {
    if (HasDeviceAttr) {
      report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName),
             MapNames::getDpctNamespace() + "matrix_mem_copy");
      return;
    }

    std::vector<std::string> ParamsStrsVec =
        getParamsAsStrs(CE, *(Result.Context));

    const Expr *LdaExpr = CE->getArg(4);
    const Expr *LdbExpr = CE->getArg(6);
    Expr::EvalResult LdaExprResult, LdbExprResult;
    if (LdaExpr->EvaluateAsInt(LdaExprResult, *Result.Context) &&
        LdbExpr->EvaluateAsInt(LdbExprResult, *Result.Context)) {
      std::string LdaStr =
          LdaExprResult.Val.getAsString(*Result.Context, LdaExpr->getType());
      std::string LdbStr =
          LdbExprResult.Val.getAsString(*Result.Context, LdbExpr->getType());
      if (LdaStr != LdbStr) {
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE,
               false, MapNames::ITFName.at(FuncName),
               "parameter " + ParamsStrsVec[4] +
                   " does not equal to parameter " + ParamsStrsVec[6]);
      } else {
        const Expr *RowsExpr = CE->getArg(0);
        Expr::EvalResult RowsExprResult;
        if (RowsExpr->EvaluateAsInt(RowsExprResult, *Result.Context)) {
          std::string RowsStr = RowsExprResult.Val.getAsString(
              *Result.Context, RowsExpr->getType());
          if (std::stoi(LdaStr) > std::stoi(RowsStr)) {
            // lda > rows. Performance issue may occur.
            report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE,
                   false, MapNames::ITFName.at(FuncName),
                   "parameter " + ParamsStrsVec[0] +
                       " is smaller than parameter " + ParamsStrsVec[4]);
          }
        } else {
          // rows cannot be evaluated. Performance issue may occur.
          report(
              CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE, false,
              MapNames::ITFName.at(FuncName),
              "parameter " + ParamsStrsVec[0] +
                  " could not be evaluated and may be smaller than parameter " +
                  ParamsStrsVec[4]);
        }
      }
    } else {
      report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMANCE_ISSUE, false,
             MapNames::ITFName.at(FuncName),
             "parameter(s) " + ParamsStrsVec[4] + " and/or " +
                 ParamsStrsVec[6] + " could not be evaluated");
    }

    std::string AStr = "(void*)" + getExprString(CE->getArg(3), true);
    std::string BStr = "(void*)" + getExprString(CE->getArg(5), true);
    std::string LdA = getExprString(CE->getArg(4), false);
    std::string LdB = getExprString(CE->getArg(6), false);
    std::string Rows = getExprString(CE->getArg(0), false);
    std::string Cols = getExprString(CE->getArg(1), false);
    std::string ElemSize = getExprString(CE->getArg(2), false);

    std::string Replacement = "(" + BStr + ", " + AStr + ", " + LdB + ", " +
                              LdA + ", " + Rows + ", " + Cols + ", " + ElemSize;

    requestFeature(HelperFeatureEnum::device_ext);
    if (FuncName == "cublasGetMatrix" || FuncName == "cublasSetMatrix") {
      Replacement =
          MapNames::getDpctNamespace() + "matrix_mem_copy" + Replacement + ")";
    } else {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(7));
      Replacement = MapNames::getDpctNamespace() + "matrix_mem_copy" +
                    Replacement + ", " + MapNames::getDpctNamespace() +
                    "automatic, *" + EA.getReplacedString() + ", true)";
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));

    if (IsAssigned) {
      requestFeature(HelperFeatureEnum::device_ext);
      insertAroundStmt(CE, "DPCT_CHECK_ERROR(", ")");
    }
  } else if (FuncName == "cublasGetVersion" ||
             FuncName == "cublasGetVersion_v2") {
    if (FuncName == "cublasGetVersion" || FuncName == "cublasGetVersion_v2") {
      DpctGlobalInfo::getInstance().insertHeader(
          SM->getExpansionLoc(CE->getBeginLoc()), HT_DPCT_COMMON_Utils);
    }

    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  } else {
    assert(0 && "Unknown function name");
  }
}

// Check whether the input expression is CallExpr or UnaryExprOrTypeTraitExpr
// (sizeof or alignof) or an identifier or literal
bool BLASFunctionCallRule::isCEOrUETTEOrAnIdentifierOrLiteral(const Expr *E) {
  auto CE = dyn_cast<CallExpr>(E->IgnoreImpCasts());
  if (CE != nullptr) {
    return true;
  }
  auto UETTE = dyn_cast<UnaryExprOrTypeTraitExpr>(E->IgnoreImpCasts());
  if (UETTE != nullptr) {
    return true;
  }
  if (isAnIdentifierOrLiteral(E)) {
    return true;
  }
  return false;
}

// Get the replacement string of the input expression.
// If the AddparenthesisIfNecessary is true, the output string will add "(...)"
// if it is necessary.
std::string
BLASFunctionCallRule::getExprString(const Expr *E,
                                    bool AddparenthesisIfNecessary) {
  ExprAnalysis EA;
  EA.analyze(E);
  std::string Res = EA.getReplacedString();
  if (AddparenthesisIfNecessary && !isCEOrUETTEOrAnIdentifierOrLiteral(E)) {
    return "(" + Res + ")";
  } else {
    return Res;
  }
}

bool BLASFunctionCallRule::isReplIndex(int Input,
                                       const std::vector<int> &IndexInfo,
                                       int &IndexTemp) {
  for (int i = 0; i < static_cast<int>(IndexInfo.size()); ++i) {
    if (IndexInfo[i] == Input) {
      IndexTemp = i;
      return true;
    }
  }
  return false;
}

std::vector<std::string>
BLASFunctionCallRule::getParamsAsStrs(const CallExpr *CE,
                                      const ASTContext &Context) {
  std::vector<std::string> ParamsStrVec;
  for (auto Arg : CE->arguments())
    ParamsStrVec.emplace_back(ExprAnalysis::ref(Arg));
  return ParamsStrVec;
}

// sample code looks like:
//   Complex-type res1 = API(...);
//   res2 = API(...);
//
// migrated code looks like:
//   Complex-type res1;
//   {
//   buffer res_buffer;
//   mklAPI(res_buffer);
//   res1 = res_buffer.get_access()[0];
//   }
//   {
//   buffer res_buffer;
//   mklAPI(res_buffer);
//   res2 = res_buffer.get_access()[0];
//   }
//
// If the API return value initializes the var declaration, we need to put the
// var declaration out of the scope and assign it in the scope, otherwise users
// cannot use this var out of the scope.
// So we need to find the Decl node of the var, which is the CallExpr node's
// ancestor.
// The initial node is the matched CallExpr node. Then visit the parent node of
// the current node until the current node is a VarDecl node.
const clang::VarDecl *
BLASFunctionCallRule::getAncestralVarDecl(const clang::CallExpr *CE) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*CE);
  while (Parents.size() == 1) {
    auto *Parent = Parents[0].get<VarDecl>();
    if (Parent) {
      return Parent;
    } else {
      Parents = Context.getParents(Parents[0]);
    }
  }
  return nullptr;
}

std::string BLASFunctionCallRule::processParamIntCastToBLASEnum(
    const Expr *E, const CStyleCastExpr *CSCE, const int DistinctionID,
    const std::string IndentStr, const BLASEnumInfo &EnumInfo,
    std::string &PrefixInsertStr, std::string &CurrentArgumentRepl) {
  std::string DpctTempVarName;
  auto &Context = DpctGlobalInfo::getContext();
  const Expr *SubExpr = CSCE->getSubExpr();
  std::string SubExprStr;
  if (SubExpr->getBeginLoc().isMacroID() && isOuterMostMacro(CSCE)) {
    // when type casting syntax is in a macro,
    // analyze the entire CSCE by ExprAnalysis
    ExprAnalysis SEA;
    SEA.analyze(CSCE);
    SubExprStr = SEA.getReplacedString();

    // Since the type cast in the macro definition need be kept (it may be used
    // in more than one places), so we need add the type cast to int for the
    // current argument.
    CurrentArgumentRepl = "(int)";
  } else {
    // To eliminate the redundant cast of non-macro cases
    SubExprStr = ExprAnalysis::ref(SubExpr);
  }

  int IndexTemp = -1;
  if (isReplIndex(DistinctionID, EnumInfo.OperationIndexInfo, IndexTemp)) {
    Expr::EvalResult ER;
    if (E->EvaluateAsInt(ER, Context) && !E->getBeginLoc().isMacroID()) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 0) {
        CurrentArgumentRepl += "oneapi::mkl::transpose::nontrans";
      } else if (Value == 1) {
        CurrentArgumentRepl += "oneapi::mkl::transpose::trans";
      } else {
        CurrentArgumentRepl += "oneapi::mkl::transpose::conjtrans";
      }
    } else {
      if (E->HasSideEffects(DpctGlobalInfo::getContext())) {
        DpctTempVarName =
            "transpose_ct" +
            std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
        PrefixInsertStr = PrefixInsertStr + "auto " + DpctTempVarName + " = " +
                          SubExprStr + ";" + getNL() + IndentStr;
        CurrentArgumentRepl += "(int)" + DpctTempVarName +
                               "==2 ? oneapi::mkl::transpose::conjtrans : "
                               "(oneapi::mkl::transpose)" +
                               DpctTempVarName;
      } else {
        CurrentArgumentRepl += SubExprStr +
                               "==2 ? oneapi::mkl::transpose::conjtrans : "
                               "(oneapi::mkl::transpose)" +
                               SubExprStr;
      }
    }
  }
  if (EnumInfo.FillModeIndexInfo == DistinctionID) {
    Expr::EvalResult ER;
    if (E->EvaluateAsInt(ER, Context) && !E->getBeginLoc().isMacroID()) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 0) {
        CurrentArgumentRepl += "oneapi::mkl::uplo::lower";
      } else {
        CurrentArgumentRepl += "oneapi::mkl::uplo::upper";
      }
    } else {
      CurrentArgumentRepl +=
          SubExprStr +
          "==0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper";
    }
  }
  if (EnumInfo.SideModeIndexInfo == DistinctionID) {
    Expr::EvalResult ER;
    if (E->EvaluateAsInt(ER, Context) && !E->getBeginLoc().isMacroID()) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 0) {
        CurrentArgumentRepl += "oneapi::mkl::side::left";
      } else {
        CurrentArgumentRepl += "oneapi::mkl::side::right";
      }
    } else {
      CurrentArgumentRepl += "(oneapi::mkl::side)" + SubExprStr;
    }
  }
  if (EnumInfo.DiagTypeIndexInfo == DistinctionID) {
    Expr::EvalResult ER;
    if (E->EvaluateAsInt(ER, Context) && !E->getBeginLoc().isMacroID()) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 0) {
        CurrentArgumentRepl += "oneapi::mkl::diag::nonunit";
      } else {
        CurrentArgumentRepl += "oneapi::mkl::diag::unit";
      }
    } else {
      CurrentArgumentRepl += "(oneapi::mkl::diag)" + SubExprStr;
    }
  }

  return DpctTempVarName;
}

REGISTER_RULE(BLASFunctionCallRule, PassKind::PK_Migration,
              RuleGroupKind::RK_BLas)

// Rule for SOLVER enums.
// Migrate SOLVER status values to corresponding int values
// Other SOLVER named values are migrated to corresponding named values
void SOLVEREnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName(
                                "(CUSOLVER_STATU.*)|(CUSOLVER_ALG.*)|("
                                "CUSOLVERDN_GETRF)|(CUSOLVERDN_POTRF)"))))
                    .bind("SOLVERConstants"),
                this);
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName(
                      "(CUSOLVER_EIG_TYPE.*)|(CUSOLVER_EIG_MODE.*)"))))
          .bind("SLOVERNamedValueConstants"),
      this);
}

void SOLVEREnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SOLVERConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));
  }

  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SLOVERNamedValueConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    std::string Name = EC->getNameAsString();
    auto Search = MapNames::SOLVEREnumsMap.find(Name);
    if (Search == MapNames::SOLVEREnumsMap.end()) {
      llvm::dbgs() << "[" << getName()
                   << "] Unexpected enum variable: " << Name;
      return;
    }
    std::string Replacement = Search->second;
    emplaceTransformation(new ReplaceStmt(DE, std::move(Replacement)));
  }
}

REGISTER_RULE(SOLVEREnumsRule, PassKind::PK_Migration, RuleGroupKind::RK_Solver)

void SOLVERFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "cusolverDnSetAdvOptions", "cusolverDnGetStream", "cusolverDnSetStream",
        "cusolverDnCreateParams", "cusolverDnDestroyParams", "cusolverDnCreate",
        "cusolverDnDestroy", "cusolverDnSpotrf_bufferSize",
        "cusolverDnDpotrf_bufferSize", "cusolverDnCpotrf_bufferSize",
        "cusolverDnZpotrf_bufferSize", "cusolverDnSpotri_bufferSize",
        "cusolverDnDpotri_bufferSize", "cusolverDnCpotri_bufferSize",
        "cusolverDnZpotri_bufferSize", "cusolverDnSgetrf_bufferSize",
        "cusolverDnDgetrf_bufferSize", "cusolverDnCgetrf_bufferSize",
        "cusolverDnZgetrf_bufferSize", "cusolverDnSpotrf", "cusolverDnDpotrf",
        "cusolverDnCpotrf", "cusolverDnZpotrf", "cusolverDnSpotrs",
        "cusolverDnDpotrs", "cusolverDnCpotrs", "cusolverDnZpotrs",
        "cusolverDnSpotri", "cusolverDnDpotri", "cusolverDnCpotri",
        "cusolverDnZpotri", "cusolverDnSgetrf", "cusolverDnDgetrf",
        "cusolverDnCgetrf", "cusolverDnZgetrf", "cusolverDnSgetrs",
        "cusolverDnDgetrs", "cusolverDnCgetrs", "cusolverDnZgetrs",
        "cusolverDnSgeqrf_bufferSize", "cusolverDnDgeqrf_bufferSize",
        "cusolverDnCgeqrf_bufferSize", "cusolverDnZgeqrf_bufferSize",
        "cusolverDnSgeqrf", "cusolverDnDgeqrf", "cusolverDnCgeqrf",
        "cusolverDnZgeqrf", "cusolverDnSormqr_bufferSize",
        "cusolverDnDormqr_bufferSize", "cusolverDnSormqr", "cusolverDnDormqr",
        "cusolverDnCunmqr_bufferSize", "cusolverDnZunmqr_bufferSize",
        "cusolverDnCunmqr", "cusolverDnZunmqr", "cusolverDnSorgqr_bufferSize",
        "cusolverDnDorgqr_bufferSize", "cusolverDnCungqr_bufferSize",
        "cusolverDnZungqr_bufferSize", "cusolverDnSorgqr", "cusolverDnDorgqr",
        "cusolverDnCungqr", "cusolverDnZungqr", "cusolverDnSsytrf_bufferSize",
        "cusolverDnDsytrf_bufferSize", "cusolverDnCsytrf_bufferSize",
        "cusolverDnZsytrf_bufferSize", "cusolverDnSsytrf", "cusolverDnDsytrf",
        "cusolverDnCsytrf", "cusolverDnZsytrf", "cusolverDnSgebrd_bufferSize",
        "cusolverDnDgebrd_bufferSize", "cusolverDnCgebrd_bufferSize",
        "cusolverDnZgebrd_bufferSize", "cusolverDnSgebrd", "cusolverDnDgebrd",
        "cusolverDnCgebrd", "cusolverDnZgebrd", "cusolverDnSorgbr_bufferSize",
        "cusolverDnDorgbr_bufferSize", "cusolverDnCungbr_bufferSize",
        "cusolverDnZungbr_bufferSize", "cusolverDnSorgbr", "cusolverDnDorgbr",
        "cusolverDnCungbr", "cusolverDnZungbr", "cusolverDnSsytrd_bufferSize",
        "cusolverDnDsytrd_bufferSize", "cusolverDnChetrd_bufferSize",
        "cusolverDnZhetrd_bufferSize", "cusolverDnSsytrd", "cusolverDnDsytrd",
        "cusolverDnChetrd", "cusolverDnZhetrd", "cusolverDnSormtr_bufferSize",
        "cusolverDnDormtr_bufferSize", "cusolverDnCunmtr_bufferSize",
        "cusolverDnZunmtr_bufferSize", "cusolverDnSormtr", "cusolverDnDormtr",
        "cusolverDnCunmtr", "cusolverDnZunmtr", "cusolverDnSorgtr_bufferSize",
        "cusolverDnDorgtr_bufferSize", "cusolverDnCungtr_bufferSize",
        "cusolverDnZungtr_bufferSize", "cusolverDnSorgtr", "cusolverDnDorgtr",
        "cusolverDnCungtr", "cusolverDnZungtr", "cusolverDnSgesvd_bufferSize",
        "cusolverDnDgesvd_bufferSize", "cusolverDnCgesvd_bufferSize",
        "cusolverDnZgesvd_bufferSize", "cusolverDnSgesvd", "cusolverDnDgesvd",
        "cusolverDnCgesvd", "cusolverDnZgesvd", "cusolverDnSpotrfBatched",
        "cusolverDnDpotrfBatched", "cusolverDnCpotrfBatched",
        "cusolverDnZpotrfBatched", "cusolverDnSpotrsBatched",
        "cusolverDnDpotrsBatched", "cusolverDnCpotrsBatched",
        "cusolverDnZpotrsBatched", "cusolverDnSsygvd", "cusolverDnDsygvd",
        "cusolverDnSsygvd_bufferSize", "cusolverDnDsygvd_bufferSize",
        "cusolverDnChegvd", "cusolverDnZhegvd", "cusolverDnChegvd_bufferSize",
        "cusolverDnZhegvd_bufferSize", "cusolverDnXgetrf",
        "cusolverDnXgetrf_bufferSize", "cusolverDnXgetrs", "cusolverDnXgeqrf",
        "cusolverDnXgeqrf_bufferSize", "cusolverDnGetrf",
        "cusolverDnGetrf_bufferSize", "cusolverDnGetrs", "cusolverDnGeqrf",
        "cusolverDnGeqrf_bufferSize", "cusolverDnCreateGesvdjInfo",
        "cusolverDnDestroyGesvdjInfo", "cusolverDnCreateSyevjInfo",
        "cusolverDnDestroySyevjInfo", "cusolverDnSgesvdj_bufferSize",
        "cusolverDnDgesvdj_bufferSize", "cusolverDnCgesvdj_bufferSize",
        "cusolverDnZgesvdj_bufferSize", "cusolverDnXgesvd_bufferSize",
        "cusolverDnGesvd_bufferSize", "cusolverDnSgesvdj", "cusolverDnDgesvdj",
        "cusolverDnCgesvdj", "cusolverDnZgesvdj", "cusolverDnXgesvd",
        "cusolverDnGesvd", "cusolverDnXpotrf_bufferSize",
        "cusolverDnPotrf_bufferSize", "cusolverDnXpotrf", "cusolverDnPotrf",
        "cusolverDnXpotrs", "cusolverDnPotrs", "cusolverDnSsyevdx",
        "cusolverDnDsyevdx", "cusolverDnSsyevdx_bufferSize",
        "cusolverDnDsyevdx_bufferSize", "cusolverDnCheevdx",
        "cusolverDnZheevdx", "cusolverDnCheevdx_bufferSize",
        "cusolverDnZheevdx_bufferSize", "cusolverDnSsygvdx",
        "cusolverDnDsygvdx", "cusolverDnSsygvdx_bufferSize",
        "cusolverDnDsygvdx_bufferSize", "cusolverDnChegvdx",
        "cusolverDnZhegvdx", "cusolverDnChegvdx_bufferSize",
        "cusolverDnZhegvdx_bufferSize", "cusolverDnSsygvj", "cusolverDnDsygvj",
        "cusolverDnSsygvj_bufferSize", "cusolverDnDsygvj_bufferSize",
        "cusolverDnChegvj", "cusolverDnZhegvj", "cusolverDnChegvj_bufferSize",
        "cusolverDnZhegvj_bufferSize", "cusolverDnXsyevdx",
        "cusolverDnXsyevdx_bufferSize", "cusolverDnSyevdx",
        "cusolverDnSyevdx_bufferSize", "cusolverDnSsyevj", "cusolverDnDsyevj",
        "cusolverDnSsyevj_bufferSize", "cusolverDnDsyevj_bufferSize",
        "cusolverDnCheevj", "cusolverDnZheevj", "cusolverDnCheevj_bufferSize",
        "cusolverDnZheevj_bufferSize", "cusolverDnXsyevd",
        "cusolverDnXsyevd_bufferSize", "cusolverDnSyevd",
        "cusolverDnSyevd_bufferSize", "cusolverDnXtrtri",
        "cusolverDnXtrtri_bufferSize", "cusolverDnSsyevd_bufferSize",
        "cusolverDnDsyevd_bufferSize", "cusolverDnCheevd_bufferSize",
        "cusolverDnZheevd_bufferSize", "cusolverDnSsyevd", "cusolverDnDsyevd",
        "cusolverDnCheevd", "cusolverDnZheevd");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               hasAncestor(functionDecl(
                                   anyOf(hasAttr(attr::CUDADevice),
                                         hasAttr(attr::CUDAGlobal))))))
                    .bind("kernelCall"),
                this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), parentStmt(),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), unless(parentStmt()),
                unless(hasParent(varDecl())),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedNotInitializeVarDecl"),
      this);

  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(functionName())), hasParent(varDecl()),
                hasAncestor(functionDecl(unless(allOf(
                    hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))))
          .bind("FunctionCallUsedInitializeVarDecl"),
      this);
}

void SOLVERFunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  bool IsInitializeVarDecl = false;
  bool HasDeviceAttr = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "kernelCall");
  if (CE) {
    HasDeviceAttr = true;
  } else if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCall"))) {
    if ((CE = getNodeAsType<CallExpr>(
             Result, "FunctionCallUsedNotInitializeVarDecl"))) {
      IsAssigned = true;
    } else if ((CE = getNodeAsType<CallExpr>(
                    Result, "FunctionCallUsedInitializeVarDecl"))) {
      IsAssigned = true;
      IsInitializeVarDecl = true;
    } else {
      return;
    }
  }

  const SourceManager *SM = Result.SourceManager;
  auto SL = SM->getExpansionLoc(CE->getBeginLoc());
  std::string Key = SM->getFilename(SL).str() +
                    std::to_string(SM->getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));

  // Collect sourceLocations of the function call
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());

  // Correct sourceLocations for macros
  if (FuncNameBegin.isMacroID())
    FuncNameBegin = SM->getExpansionLoc(FuncNameBegin);
  if (FuncCallEnd.isMacroID())
    FuncCallEnd = SM->getExpansionLoc(FuncCallEnd);
  // Offset 1 is the length of the last token ")"
  FuncCallEnd = FuncCallEnd.getLocWithOffset(1);

  // Collect sourceLocations for creating new scope
  std::string PrefixBeforeScope, PrefixInsertStr, SuffixInsertStr;
  auto SR = getScopeInsertRange(CE, FuncNameBegin, FuncCallEnd);
  SourceLocation StmtBegin = SR.getBegin(), StmtEndAfterSemi = SR.getEnd();
  std::string IndentStr =
      getIndent(StmtBegin, (Result.Context)->getSourceManager()).str();
  Token Tok;
  Lexer::getRawToken(FuncNameBegin, Tok, *SM, LangOptions());
  SourceLocation FuncNameEnd = Tok.getEndLoc();
  auto FuncNameLength =
      SM->getCharacterData(FuncNameEnd) - SM->getCharacterData(FuncNameBegin);

  // Prepare the prefix and the postfix for assignment
  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  std::string AssignPrefix = "";
  std::string AssignPostfix = "";

  if (IsAssigned) {
    AssignPrefix = "DPCT_CHECK_ERROR(";
    AssignPostfix = ")";
  }

  if (HasDeviceAttr) {
    report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
           MapNames::ITFName.at(FuncName),
           MapNames::getDpctNamespace() + "dpct_memcpy");
    return;
  }

  const VarDecl *VD = 0;
  if (IsInitializeVarDecl) {
    // Create the prefix for VarDecl before scope and remove the VarDecl inside
    // the scope
    VD = getAncestralVarDecl(CE);
    std::string VarType, VarName;
    if (VD) {
      VarType = VD->getType().getAsString();
      VarName = VD->getNameAsString();

      requestHelperFeatureForTypeNames(VarType);
      insertHeaderForTypeRule(VarType, VD->getBeginLoc());
      auto Itr = MapNames::TypeNamesMap.find(VarType);
      if (Itr == MapNames::TypeNamesMap.end())
        return;
      VarType = Itr->second->NewName;
      PrefixBeforeScope = VarType + " " + VarName + ";" + getNL() + IndentStr +
                          PrefixBeforeScope;
      SourceLocation typeBegin =
          VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
      SourceLocation nameBegin = VD->getLocation();
      SourceLocation nameEnd = Lexer::getLocForEndOfToken(
          nameBegin, 0, *SM, Result.Context->getLangOpts());
      auto replLen =
          SM->getCharacterData(nameEnd) - SM->getCharacterData(typeBegin);
      for (auto ItHeader = Itr->second->Includes.begin();
           ItHeader != Itr->second->Includes.end(); ItHeader++) {
        DpctGlobalInfo::getInstance().insertHeader(typeBegin, *ItHeader);
      }
      emplaceTransformation(
          new ReplaceText(typeBegin, replLen, std::move(VarName)));
    } else {
      assert(0 && "Fail to get VarDecl information");
      return;
    }
  }

  if (MapNames::SOLVERAPIWithRewriter.find(FuncName) !=
      MapNames::SOLVERAPIWithRewriter.end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  } else if (MapNames::SOLVERFuncReplInfoMap.find(FuncName) !=
      MapNames::SOLVERFuncReplInfoMap.end()) {
    // Find replacement string
    auto ReplInfoPair = MapNames::SOLVERFuncReplInfoMap.find(FuncName);
    MapNames::SOLVERFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;

    // Migrate arguments one by one
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      int IndexTemp = -1;
      // MyFunction(float* a);
      // In usm: Keep it and type cast if needed.
      // In non-usm: Create a buffer, pass the buffer
      // If type is int: MKL takes int64_t, so need to create a temp
      //                 buffer/variable and copy the value back
      //                 after the function call.
      // Some API migration requires MoveFrom and MoveTo.
      // e.g., move arg#1 to arg#0
      // MyFunction(float* a, float* b);
      // ---> MyFunction(float* b, float*a);
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
          requestFeature(HelperFeatureEnum::device_ext);
          std::string BufferDecl;
          std::string BufferName = getBufferNameAndDeclStr(
              CE->getArg(i), *(Result.Context),
              ReplInfo.BufferTypeInfo[IndexTemp], StmtBegin, BufferDecl, i);
          PrefixInsertStr = PrefixInsertStr + BufferDecl;
          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            PrefixInsertStr =
                PrefixInsertStr + IndentStr + MapNames::getClNamespace() +
                "buffer<int64_t> "
                "result_temp_buffer" +
                std::to_string(i) + "(" + MapNames::getClNamespace() +
                "range<1>(1));" + getNL();
            SuffixInsertStr = SuffixInsertStr + BufferName + ".get_access<" +
                              MapNames::getClNamespace() +
                              "access_mode::write>()[0] = "
                              "(int)result_temp_buffer" +
                              std::to_string(i) + ".get_access<" +
                              MapNames::getClNamespace() +
                              "access_mode::read>()[0];" + getNL() + IndentStr;
            BufferName = "result_temp_buffer" + std::to_string(i);
          }
          bool Moved = false;
          for (size_t j = 0; j < ReplInfo.MoveFrom.size(); j++) {
            if (ReplInfo.MoveFrom[j] == i) {
              Moved = true;
              if (CE->getArg(ReplInfo.MoveTo[j])) {
                emplaceTransformation(new InsertAfterStmt(
                    CE->getArg(ReplInfo.MoveTo[j] - 1),
                    ", result_temp_buffer" + std::to_string(i)));
              }
              ReplInfo.RedundantIndexInfo.push_back(i);
              break;
            }
          }
          if (!Moved) {
            emplaceTransformation(new ReplaceStmt(CE->getArg(i), BufferName));
          }
        } else {
          std::string ArgName = ExprAnalysis::ref(CE->getArg(i));
          if (ReplInfo.BufferTypeInfo[IndexTemp] == "int") {
            PrefixInsertStr = IndentStr + "int64_t result_temp_pointer" +
                              std::to_string(i) + ";" + getNL();
            SuffixInsertStr = SuffixInsertStr + " *" +
                              ExprAnalysis::ref(CE->getArg(i)) +
                              " = result_temp_pointer" + std::to_string(i) +
                              ";" + getNL() + IndentStr;
            ArgName = "&result_temp_pointer" + std::to_string(i);
          }
          bool Moved = false;
          for (size_t j = 0; j < ReplInfo.MoveFrom.size(); j++) {
            if (ReplInfo.MoveFrom[j] == i) {
              Moved = true;
              if (CE->getArg(ReplInfo.MoveTo[j])) {
                emplaceTransformation(new InsertAfterStmt(
                    CE->getArg(ReplInfo.MoveTo[j] - 1), ", " + ArgName));
              }
              ReplInfo.RedundantIndexInfo.push_back(i);
              break;
            }
          }
          if (!Moved) {
            std::string TypeStr =
                ReplInfo.BufferTypeInfo[IndexTemp].compare("int")
                    ? "(" + ReplInfo.BufferTypeInfo[IndexTemp] + "*)"
                    : "";
            emplaceTransformation(
                new ReplaceStmt(CE->getArg(i), TypeStr + ArgName));
          }
        }
      }
      // Remove the redundant args including the leading ","
      if (isReplIndex(i, ReplInfo.RedundantIndexInfo, IndexTemp)) {
        SourceLocation RemoveBegin, RemoveEnd;
        if (i == 0) {
          RemoveBegin = CE->getArg(i)->getBeginLoc();
        } else {
          RemoveBegin = CE->getArg(i - 1)->getEndLoc().getLocWithOffset(
              Lexer::MeasureTokenLength(
                  CE->getArg(i - 1)->getEndLoc(), *SM,
                  dpct::DpctGlobalInfo::getContext().getLangOpts()));
        }
        RemoveEnd = CE->getArg(i)->getEndLoc().getLocWithOffset(
            Lexer::MeasureTokenLength(
                CE->getArg(i)->getEndLoc(), *SM,
                dpct::DpctGlobalInfo::getContext().getLangOpts()));
        auto ParameterLength =
            SM->getCharacterData(RemoveEnd) - SM->getCharacterData(RemoveBegin);
        emplaceTransformation(
            new ReplaceText(RemoveBegin, ParameterLength, ""));
      }
      // OldFoo(float* out); --> *(out) = NewFoo();
      // In current case, return value is always the last arg
      if (ReplInfo.ReturnValue && i == ArgNum - 1) {
        Replacement = "*(" +
                      ExprAnalysis::ref(CE->getArg(CE->getNumArgs() - 1)) +
                      ") = " + Replacement;
      }
      // The arg#0 is always the handler and will always be migrated to queue.
      if (i == 0) {
        // process handle argument
        emplaceTransformation(new ReplaceStmt(
            CE->getArg(i), "*" + ExprAnalysis::ref(CE->getArg(i))));
      }
    }
    // Declare new args if it is used in MKL
    if (!ReplInfo.MissedArgumentFinalLocation.empty()) {
      std::string ReplStr;
      for (size_t i = 0; i < ReplInfo.MissedArgumentFinalLocation.size(); ++i) {
        if (ReplInfo.MissedArgumentIsBuffer[i]) {
          PrefixInsertStr =
              PrefixInsertStr + IndentStr + MapNames::getClNamespace() +
              "buffer<" + ReplInfo.MissedArgumentType[i] + "> " +
              ReplInfo.MissedArgumentName[i] + "(" +
              MapNames::getClNamespace() + "range<1>(1));" + getNL();
        } else {
          PrefixInsertStr = PrefixInsertStr + IndentStr +
                            ReplInfo.MissedArgumentType[i] + " " +
                            ReplInfo.MissedArgumentName[i] + ";" + getNL();
        }
        ReplStr = ReplStr + ", " + ReplInfo.MissedArgumentName[i];
        if (i == ReplInfo.MissedArgumentFinalLocation.size() - 1 ||
            ReplInfo.MissedArgumentInsertBefore[i + 1] !=
                ReplInfo.MissedArgumentInsertBefore[i]) {
          if (ReplInfo.MissedArgumentInsertBefore[i] > 0) {
            emplaceTransformation(new InsertAfterStmt(
                CE->getArg(ReplInfo.MissedArgumentInsertBefore[i] - 1),
                std::move(ReplStr)));
          }
          ReplStr = "";
        }
      }
    }

    // Copy an arg. e.g. copy arg#0 to arg#2
    // OldFoo(int m, int n); --> NewFoo(int m, int n, int m)
    if (!ReplInfo.CopyFrom.empty()) {
      std::string InsStr = "";
      for (size_t i = 0; i < ReplInfo.CopyFrom.size(); ++i) {
        InsStr =
            InsStr + ", " + ExprAnalysis::ref(CE->getArg(ReplInfo.CopyFrom[i]));
        if (i == ReplInfo.CopyTo.size() - 1 ||
            ReplInfo.CopyTo[i + 1] != ReplInfo.CopyTo[i]) {
          emplaceTransformation(new InsertAfterStmt(
              CE->getArg(ReplInfo.CopyTo[i - 1]), std::move(InsStr)));
          InsStr = "";
        }
      }
    }
    // Type cast, for enum type migration
    if (!ReplInfo.CastIndexInfo.empty()) {
      for (size_t i = 0; i < ReplInfo.CastIndexInfo.size(); ++i) {
        std::string CastStr = "(" + ReplInfo.CastTypeInfo[i] + ")";
        emplaceTransformation(new InsertBeforeStmt(
            CE->getArg(ReplInfo.CastIndexInfo[i]), std::move(CastStr)));
      }
    }
    // Create scratchpad and scratchpad_size if required in MKL
    if (!ReplInfo.WorkspaceIndexInfo.empty()) {
      std::string BufferSizeArgStr = "";
      for (size_t i = 0; i < ReplInfo.WorkspaceSizeInfo.size(); ++i) {
        BufferSizeArgStr += i ? " ," : "";
        BufferSizeArgStr +=
            ExprAnalysis::ref(CE->getArg(ReplInfo.WorkspaceSizeInfo[i]));
      }
      std::string ScratchpadSizeNameStr =
          "scratchpad_size_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string ScratchpadNameStr =
          "scratchpad_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string EventNameStr =
          "event_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      std::string WSVectorNameStr =
          "ws_vec_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      PrefixInsertStr += IndentStr + "std::int64_t " + ScratchpadSizeNameStr +
                         " = " + ReplInfo.WorkspaceSizeFuncName + "(*" +
                         BufferSizeArgStr + ");" + getNL();
      std::string BufferTypeStr = "float";
      if (ReplInfo.BufferTypeInfo.size() > 0) {
        BufferTypeStr = ReplInfo.BufferTypeInfo[0];
      }
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
        DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(),
                                                   HT_Thread);

        PrefixInsertStr +=
            IndentStr + BufferTypeStr + " *" + ScratchpadNameStr + " = " +
            MapNames::getClNamespace() + "malloc_device<" + BufferTypeStr +
            ">(" + ScratchpadSizeNameStr + ", *" +
            ExprAnalysis::ref(CE->getArg(0)) + ");" + getNL();
        PrefixInsertStr += IndentStr + MapNames::getClNamespace() + "event " +
                           EventNameStr + ";" + getNL();

        Replacement = EventNameStr + " = " + Replacement;

        SuffixInsertStr += "std::vector<void *> " + WSVectorNameStr + "{" +
                           ScratchpadNameStr + "};" + getNL() + IndentStr;
        SuffixInsertStr += MapNames::getDpctNamespace() + "async_dpct_free(" +
                           WSVectorNameStr + ", {" + EventNameStr + "}, *" +
                           ExprAnalysis::ref(CE->getArg(0)) + ");" + getNL() +
                           IndentStr;
        requestFeature(HelperFeatureEnum::device_ext);
      } else {
        PrefixInsertStr += IndentStr + MapNames::getClNamespace() + "buffer<" +
                           BufferTypeStr + ", 1> " + ScratchpadNameStr + "{" +
                           MapNames::getClNamespace() + "range<1>(" +
                           ScratchpadSizeNameStr + ")};" + getNL();
      }
      if (ReplInfo.WorkspaceIndexInfo[0] > 0) {
        emplaceTransformation(new InsertAfterStmt(
            CE->getArg(ReplInfo.WorkspaceIndexInfo[0]),
            ", " + ScratchpadNameStr + ", " + ScratchpadSizeNameStr));
      }
    }

    // Create scratchpad_size if only scratchpad_size is required in MKL
    if (!ReplInfo.WSSizeInsertAfter.empty()) {
      std::string BufferSizeArgStr = "";
      for (size_t i = 0; i < ReplInfo.WSSizeInfo.size(); ++i) {
        BufferSizeArgStr += i ? " ," : "";
        BufferSizeArgStr +=
            ExprAnalysis::ref(CE->getArg(ReplInfo.WSSizeInfo[i]));
      }
      std::string ScratchpadSizeNameStr =
          "scratchpad_size_ct" +
          std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      PrefixInsertStr += IndentStr + "std::int64_t " + ScratchpadSizeNameStr +
                         " = " + ReplInfo.WSSFuncName + "(*" +
                         BufferSizeArgStr + ");" + getNL();
      if (ReplInfo.WSSizeInsertAfter[0] > 0) {
        emplaceTransformation(
            new InsertAfterStmt(CE->getArg(ReplInfo.WSSizeInsertAfter[0]),
                                ", " + ScratchpadSizeNameStr));
      }
    }

    // Check PrefixInsertStr and SuffixInsertStr to decide whether to add
    // bracket
    std::string PrefixWithBracket = "";
    std::string SuffixWithBracket = "";
    if (!PrefixInsertStr.empty() || !SuffixInsertStr.empty()) {
      PrefixWithBracket =
          "{" + std::string(getNL()) + PrefixInsertStr + IndentStr;
      SuffixWithBracket = getNL() + IndentStr + SuffixInsertStr + "}";
    }

    std::string ReplaceFuncName = Replacement;
    emplaceTransformation(
        new ReplaceText(FuncNameBegin, FuncNameLength, std::move(Replacement)));
    insertAroundRange(StmtBegin, StmtEndAfterSemi,
                      PrefixBeforeScope + PrefixWithBracket,
                      std::move(SuffixWithBracket));

    StringRef FuncNameRef(FuncName);
    if (FuncNameRef.endswith("getrf")) {
      report(StmtBegin, Diagnostics::DIFFERENT_LU_FACTORIZATION, true,
             getStmtSpelling(CE->getArg(6)), ReplaceFuncName,
             MapNames::ITFName.at(FuncName));
    }
    if (IsAssigned) {
      insertAroundRange(FuncNameBegin, FuncCallEnd, std::move(AssignPrefix),
                        std::move(AssignPostfix));
      requestFeature(HelperFeatureEnum::device_ext);
    }
  } else if (FuncName == "cusolverDnCreate" ||
             FuncName == "cusolverDnDestroy") {
    std::string Repl;
    if (FuncName == "cusolverDnCreate") {
      std::string LHS = getDrefName(CE->getArg(0));
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
      Repl = LHS + " = &{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
    } else if (FuncName == "cusolverDnDestroy") {
      dpct::ExprAnalysis EA(CE->getArg(0));
      Repl = EA.getReplacedString() + " = nullptr";
    } else {
      return;
    }

    if (IsAssigned) {
      requestFeature(HelperFeatureEnum::device_ext);
      emplaceTransformation(
          new ReplaceStmt(CE, true, "DPCT_CHECK_ERROR(" + Repl + ")"));
    } else {
      emplaceTransformation(new ReplaceStmt(CE, true, Repl));
    }
  }
}

void SOLVERFunctionCallRule::getParameterEnd(
    const SourceLocation &ParameterEnd, SourceLocation &ParameterEndAfterComma,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  std::optional<Token> TokSharedPtr;
  TokSharedPtr = Lexer::findNextToken(ParameterEnd, *(Result.SourceManager),
                                      LangOptions());
  Token TokComma = TokSharedPtr.value();
  if (TokComma.getKind() == tok::comma) {
    ParameterEndAfterComma = TokComma.getEndLoc();
  } else {
    ParameterEndAfterComma = TokComma.getLocation();
  }
}

bool SOLVERFunctionCallRule::isReplIndex(int Input, std::vector<int> &IndexInfo,
                                         int &IndexTemp) {
  for (int i = 0; i < static_cast<int>(IndexInfo.size()); ++i) {
    if (IndexInfo[i] == Input) {
      IndexTemp = i;
      return true;
    }
  }
  return false;
}

std::string SOLVERFunctionCallRule::getBufferNameAndDeclStr(
    const Expr *Arg, const ASTContext &AC, const std::string &TypeAsStr,
    SourceLocation SL, std::string &BufferDecl, int DistinctionID) {

  std::string PointerName = ExprAnalysis::ref(Arg);
  std::string BufferTempName =
      getTempNameForExpr(Arg, true, true) + "buf_ct" +
      std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());

  requestFeature(HelperFeatureEnum::device_ext);
  BufferDecl = getIndent(SL, AC.getSourceManager()).str() + "auto " +
               BufferTempName + " = " + MapNames::getDpctNamespace() +
               "get_buffer<" + TypeAsStr + ">(" + PointerName + ");" + getNL();
  return BufferTempName;
}

const clang::VarDecl *
SOLVERFunctionCallRule::getAncestralVarDecl(const clang::CallExpr *CE) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*CE);
  while (Parents.size() == 1) {
    auto *Parent = Parents[0].get<VarDecl>();
    if (Parent) {
      return Parent;
    } else {
      Parents = Context.getParents(Parents[0]);
    }
  }
  return nullptr;
}

REGISTER_RULE(SOLVERFunctionCallRule, PassKind::PK_Migration,
              RuleGroupKind::RK_Solver)

void FunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "cudaGetDeviceCount", "cudaGetDeviceProperties",
        "cudaGetDeviceProperties_v2", "cudaDeviceReset", "cudaSetDevice",
        "cudaDeviceGetAttribute", "cudaDeviceGetP2PAttribute",
        "cudaDeviceGetPCIBusId", "cudaGetDevice", "cudaDeviceSetLimit",
        "cudaGetLastError", "cudaPeekAtLastError", "cudaDeviceSynchronize",
        "cudaThreadSynchronize", "cudnnGetErrorString", "cudaGetErrorString", "cudaGetErrorName",
        "cudaDeviceSetCacheConfig", "cudaDeviceGetCacheConfig", "clock",
        "cudaOccupancyMaxPotentialBlockSize", "cudaThreadSetLimit",
        "cudaFuncSetCacheConfig", "cudaThreadExit", "cudaDeviceGetLimit",
        "cudaDeviceSetSharedMemConfig", "cudaIpcCloseMemHandle",
        "cudaIpcGetEventHandle", "cudaIpcGetMemHandle",
        "cudaIpcOpenEventHandle", "cudaIpcOpenMemHandle", "cudaSetDeviceFlags",
        "cudaDeviceCanAccessPeer", "cudaDeviceDisablePeerAccess",
        "cudaDeviceEnablePeerAccess", "cudaDriverGetVersion",
        "cudaRuntimeGetVersion", "clock64",
        "cudaFuncSetSharedMemConfig", "cuFuncSetCacheConfig",
        "cudaPointerGetAttributes", "cuCtxSetCacheConfig", "cuCtxSetLimit",
        "cudaCtxResetPersistingL2Cache", "cuCtxResetPersistingL2Cache",
        "cudaStreamSetAttribute", "cudaStreamGetAttribute", "cudaFuncSetAttribute");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), parentStmt()))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               unless(parentStmt())))
                    .bind("FunctionCallUsed"),
                this);
}

std::string FunctionCallRule::findValueofAttrVar(const Expr *AttrArg,
                                                 const CallExpr *CE) {
  std::string AttributeName;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &CT = DpctGlobalInfo::getContext();
  int MinDistance = INT_MAX;
  int RecognizedMinDistance = INT_MAX;
  if (!AttrArg || !CE)
    return "";
  auto DRE = dyn_cast<DeclRefExpr>(AttrArg->IgnoreImpCasts());
  if (!DRE)
    return "";
  auto Decl = dyn_cast<VarDecl>(DRE->getDecl());
  if (!Decl || CT.getParents(*Decl)[0].get<TranslationUnitDecl>())
    return "";
  int DRELocOffset = SM.getFileOffset(SM.getExpansionLoc(DRE->getBeginLoc()));

  if (Decl->hasInit()) {
    // get the attribute name from definition
    if (auto Init = dyn_cast<DeclRefExpr>(Decl->getInit())) {
      SourceLocation InitLoc = SM.getExpansionLoc(Init->getLocation());
      MinDistance = DRELocOffset - SM.getFileOffset(InitLoc);
      RecognizedMinDistance = MinDistance;
      AttributeName = Init->getNameInfo().getName().getAsString();
    }
  }
  std::string AttrVarName = DRE->getNameInfo().getName().getAsString();
  auto AttrVarScope = findImmediateBlock(Decl);
  if (!AttrVarScope)
    return "";

  // we need to track the reference of attr var in its scope
  auto AttrVarMatcher =
      findAll(declRefExpr(to(varDecl(hasName(AttrVarName)))).bind("AttrVar"));
  auto MatchResult = ast_matchers::match(AttrVarMatcher, *AttrVarScope,
                                         DpctGlobalInfo::getContext());

  for (auto &SubResult : MatchResult) {
    const DeclRefExpr *AugDRE = SubResult.getNodeAs<DeclRefExpr>("AttrVar");
    if (!AugDRE)
      break;
    SourceLocation AugLoc = SM.getExpansionLoc(AugDRE->getBeginLoc());
    int CurrentDistance = DRELocOffset - SM.getFileOffset(AugLoc);
    // we need to skip no effect reference
    if (CurrentDistance <= 0 || !isModifiedRef(AugDRE)) {
      continue;
    }
    MinDistance = MinDistance > CurrentDistance ? CurrentDistance : MinDistance;

    auto BO = CT.getParents(*AugDRE)[0].get<BinaryOperator>();
    if (BO && BO->getOpcode() == BinaryOperatorKind::BO_Assign) {
      auto Condition = [&](const clang::DynTypedNode &Node) -> bool {
        if (Node.get<IfStmt>() || Node.get<WhileStmt>() ||
            Node.get<ForStmt>() || Node.get<DoStmt>() || Node.get<CaseStmt>() ||
            Node.get<SwitchStmt>() || Node.get<CompoundStmt>()) {
          return true;
        }
        return false;
      };
      auto BOCS = DpctGlobalInfo::findAncestor<CompoundStmt>(BO, Condition);
      auto CECS = DpctGlobalInfo::findAncestor<CompoundStmt>(CE, Condition);
      if (!(BOCS && CECS && BOCS == CECS))
        continue;
      if (auto RHS = dyn_cast<DeclRefExpr>(BO->getRHS())) {
        RecognizedMinDistance = CurrentDistance < RecognizedMinDistance
                                    ? CurrentDistance
                                    : RecognizedMinDistance;
        AttributeName = RHS->getNameInfo().getName().getAsString();
      }
    }
  }
  // if there is a non-recognized reference closer than recognized reference,
  // then we need to clear current attribute name
  if (RecognizedMinDistance > MinDistance)
    AttributeName.clear();
  return AttributeName;
}

void FunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCallUsed")))
      return;
    IsAssigned = true;
  }
  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  if (!CallExprRewriterFactoryBase::RewriterMap)
    return;
  auto Iter = CallExprRewriterFactoryBase::RewriterMap->find(FuncName);
  if (Iter != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  std::string Prefix, Suffix;
  if (IsAssigned) {
    Prefix = "DPCT_CHECK_ERROR(";
    Suffix = ")";
  }

  if (FuncName == "cudaGetDeviceCount") {
    if (IsAssigned) {
      requestFeature(HelperFeatureEnum::device_ext);
    }
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(
        new InsertBeforeStmt(CE, Prefix + ResultVarName + " = "));
    emplaceTransformation(
        new ReplaceStmt(CE, MapNames::getDpctNamespace() +
                                "dev_mgr::instance().device_count()" + Suffix));
    requestFeature(HelperFeatureEnum::device_ext);
  } else if (FuncName == "cudaGetDeviceProperties" ||
             FuncName == "cudaGetDeviceProperties_v2") {
    if (IsAssigned) {
      requestFeature(HelperFeatureEnum::device_ext);
    }
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(new ReplaceCalleeName(
        CE, Prefix + MapNames::getDpctNamespace() + "get_device_info"));
    emplaceTransformation(new ReplaceStmt(CE->getArg(0), ResultVarName));
    if (DpctGlobalInfo::useNoQueueDevice()) {
      emplaceTransformation(new ReplaceStmt(
          CE->getArg(1), DpctGlobalInfo::getGlobalDeviceName()));
    } else {
      emplaceTransformation(new ReplaceStmt(
          CE->getArg(1), MapNames::getDpctNamespace() +
                             "dev_mgr::instance().get_device(" +
                             getStmtSpelling(CE->getArg(1)) + ")"));
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new InsertAfterStmt(CE, std::move(Suffix)));
  } else if (FuncName == "cudaDriverGetVersion" ||
             FuncName == "cudaRuntimeGetVersion") {
    if (IsAssigned) {
      requestFeature(HelperFeatureEnum::device_ext);
    }
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(
        new InsertBeforeStmt(CE, Prefix + ResultVarName + " = "));

    std::string ReplStr = MapNames::getDpctNamespace() + "get_major_version(";
    if (DpctGlobalInfo::useNoQueueDevice()) {
      ReplStr += DpctGlobalInfo::getGlobalDeviceName();
      ReplStr += ")";
    } else {
      ReplStr += MapNames::getDpctNamespace();
      ReplStr += "get_current_device())";
    }
    emplaceTransformation(new ReplaceStmt(CE, ReplStr + Suffix));
    report(CE->getBeginLoc(), Warnings::TYPE_MISMATCH, false);
    requestFeature(HelperFeatureEnum::device_ext);
  } else if (FuncName == "cudaDeviceReset" || FuncName == "cudaThreadExit") {
    if (IsAssigned) {
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (isPlaceholderIdxDuplicated(CE))
      return;
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::HFT_CurrentDevice);
    emplaceTransformation(new ReplaceStmt(CE, Prefix + "{{NEEDREPLACED" +
                                                  std::to_string(Index) +
                                                  "}}.reset()" + Suffix));
    requestFeature(HelperFeatureEnum::device_ext);
  } else if (FuncName == "cudaSetDevice") {
    if (DpctGlobalInfo::useNoQueueDevice()) {
      emplaceTransformation(new ReplaceStmt(CE, "0"));
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             "cudaSetDevice",
             "it is redundant if it is migrated with option "
             "--helper-function-preference=no-queue-device "
             "which declares a global SYCL device and queue.");
    } else {
      DpctGlobalInfo::setDeviceChangedFlag(true);
      report(CE->getBeginLoc(), Diagnostics::DEVICE_ID_DIFFERENT, false,
             getStmtSpelling(CE->getArg(0)));
      emplaceTransformation(new ReplaceStmt(
          CE->getCallee(),
          Prefix + MapNames::getDpctNamespace() + "select_device"));
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (IsAssigned)
      emplaceTransformation(new InsertAfterStmt(CE, ")"));
  } else if (FuncName == "cudaDeviceGetAttribute") {
    std::string ResultVarName = getDrefName(CE->getArg(0));
    auto AttrArg = CE->getArg(1);
    std::string AttributeName;
    if (auto DRE = dyn_cast<DeclRefExpr>(AttrArg)) {
      AttributeName = DRE->getNameInfo().getName().getAsString();
    } else {
      AttributeName = findValueofAttrVar(AttrArg, CE);
      if (AttributeName.empty()) {
        report(CE->getBeginLoc(), Diagnostics::UNPROCESSED_DEVICE_ATTRIBUTE,
               false);
        return;
      }
    }
    std::string ReplStr{ResultVarName};
    auto StmtStrArg2 = getStmtSpelling(CE->getArg(2));

    if (AttributeName == "cudaDevAttrConcurrentManagedAccess" &&
        DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      std::string ReplStr = getDrefName(CE->getArg(0));
      ReplStr += " = false";
      if (IsAssigned)
        ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
      emplaceTransformation(new ReplaceStmt(CE, ReplStr));
      return;
    }

    if (AttributeName == "cudaDevAttrComputeMode") {
      report(CE->getBeginLoc(), Diagnostics::COMPUTE_MODE, false);
      ReplStr += " = 1";
    } else {
      auto Search = EnumConstantRule::EnumNamesMap.find(AttributeName);
      if (Search == EnumConstantRule::EnumNamesMap.end()) {
        return;
      }
      requestHelperFeatureForEnumNames(AttributeName);

      ReplStr += " = " + MapNames::getDpctNamespace() +
                 "dev_mgr::instance().get_device(";
      ReplStr += StmtStrArg2;
      ReplStr += ").";
      ReplStr += Search->second->NewName;
      ReplStr += "()";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (IsAssigned)
      ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaDeviceGetP2PAttribute") {
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(new ReplaceStmt(CE, ResultVarName + " = 0"));
    report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
           "cudaDeviceGetP2PAttribute");
  } else if (FuncName == "cudaDeviceGetPCIBusId") {
    report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
           "cudaDeviceGetPCIBusId");
  } else if (FuncName == "cudaGetDevice") {
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(new InsertBeforeStmt(CE, ResultVarName + " = "));
    if (DpctGlobalInfo::useNoQueueDevice()) {
      emplaceTransformation(new ReplaceStmt(CE, "0"));
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             "cudaGetDevice",
             "it is redundant if it is migrated with option "
             "--helper-function-preference=no-queue-device "
             "which declares a global SYCL device and queue.");
    } else {
      emplaceTransformation(
          new ReplaceStmt(CE, MapNames::getDpctNamespace() +
                                  "dev_mgr::instance().current_device_id()"));
      requestFeature(HelperFeatureEnum::device_ext);
    }
  } else if (FuncName == "cudaDeviceSynchronize" ||
             FuncName == "cudaThreadSynchronize") {
    if (isPlaceholderIdxDuplicated(CE))
      return;
    std::string ReplStr;
    if (DpctGlobalInfo::useNoQueueDevice()) {
      ReplStr = DpctGlobalInfo::getGlobalQueueName() + ".wait_and_throw()";
    } else {
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_CurrentDevice);
      ReplStr = "{{NEEDREPLACED" + std::to_string(Index) +
                "}}.queues_wait_and_throw()";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (IsAssigned) {
      ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));

  } else if (FuncName == "cudaGetLastError" ||
             FuncName == "cudaPeekAtLastError" ||
             FuncName == "cudaGetErrorString" ||
             FuncName == "cudaGetErrorName") {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  } else if (FuncName == "clock" || FuncName == "clock64") {
    if (CE->getDirectCallee()->hasAttr<CUDAGlobalAttr>() ||
        CE->getDirectCallee()->hasAttr<CUDADeviceAttr>()) {
      report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED_SYCL_UNDEF, false,
             FuncName);
    }
    // Add '#include <time.h>' directive to the file only once
    auto Loc = CE->getBeginLoc();
    DpctGlobalInfo::getInstance().insertHeader(Loc, HT_Time);
  } else if (FuncName == "cudaDeviceSetLimit" ||
             FuncName == "cudaThreadSetLimit" ||
             FuncName == "cudaDeviceSetCacheConfig" ||
             FuncName == "cudaDeviceGetCacheConfig" ||
             FuncName == "cuCtxSetCacheConfig" || FuncName == "cuCtxSetLimit" ||
             FuncName == "cudaCtxResetPersistingL2Cache" ||
             FuncName == "cuCtxResetPersistingL2Cache") {
    auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
  } else if(FuncName == "cudaStreamSetAttribute" ||
             FuncName == "cudaStreamGetAttribute" ){
    std::string ArgStr = getStmtSpelling(CE->getArg(1));
    if (ArgStr == "cudaStreamAttributeAccessPolicyWindow") {
      if (IsAssigned) {
        report(
            CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
            MapNames::ITFName.at(FuncName),
            "SYCL currently does not support setting cache config on devices.");
        emplaceTransformation(new ReplaceStmt(CE, "0"));
      } else {
        report(
            CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
            MapNames::ITFName.at(FuncName),
            "SYCL currently does not support setting cache config on devices.");
        emplaceTransformation(new ReplaceStmt(CE, ""));
      }
    } else if (ArgStr == "cudaLaunchAttributeIgnore") {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName),
               "this call is redundant in SYCL.");
        emplaceTransformation(new ReplaceStmt(CE, "0"));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName),
               "this call is redundant in SYCL.");
        emplaceTransformation(new ReplaceStmt(CE, ""));
      }
    } else {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName),
               "SYCL currently does not support corresponding setting.");
        emplaceTransformation(new ReplaceStmt(CE, "0"));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName),
               "SYCL currently does not support corresponding setting.");
        emplaceTransformation(new ReplaceStmt(CE, ""));
      }
    }
  } else if(FuncName == "cudaFuncSetAttribute"){
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName),
             "SYCL currently does not support corresponding setting.");
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName),
             "SYCL currently does not support corresponding setting.");
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
  }else if (FuncName == "cudaOccupancyMaxPotentialBlockSize") {
    report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
           MapNames::ITFName.at(FuncName));
  } else if (FuncName == "cudaDeviceGetLimit") {
    ExprAnalysis EA;
    EA.analyze(CE->getArg(0));
    auto Arg0Str = EA.getReplacedString();
    std::string ReplStr{"*"};
    ReplStr += Arg0Str;
    ReplStr += " = 0";
    if (IsAssigned) {
      ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    report(CE->getBeginLoc(), Diagnostics::DEVICE_LIMIT_NOT_SUPPORTED, false);
  } else if (FuncName == "cudaDeviceSetSharedMemConfig" ||
             FuncName == "cudaFuncSetSharedMemConfig" ||
             FuncName == "cudaFuncSetCacheConfig" ||
             FuncName == "cuFuncSetCacheConfig") {
    std::string Msg = "SYCL currently does not support configuring shared "
                      "memory on devices.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
  } else if (FuncName == "cudaSetDeviceFlags") {
    std::string Msg =
        "SYCL currently does not support setting flags for devices.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
  } else if (FuncName == "cudaDeviceEnablePeerAccess" ||
             FuncName == "cudaDeviceDisablePeerAccess") {
    std::string Msg =
        "SYCL currently does not support memory access across peer devices.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
  } else if (FuncName == "cudaDeviceCanAccessPeer") {
    ExprAnalysis EA;
    EA.analyze(CE->getArg(0));
    auto Arg0Str = EA.getReplacedString();
    std::string ReplStr{"*"};
    ReplStr += Arg0Str;
    ReplStr += " = 0";
    if (IsAssigned) {
      ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    report(CE->getBeginLoc(), Diagnostics::EXPLICIT_PEER_ACCESS, false,
           MapNames::ITFName.at(FuncName));
  } else if (FuncName == "cudaIpcGetEventHandle" ||
             FuncName == "cudaIpcOpenEventHandle" ||
             FuncName == "cudaIpcGetMemHandle" ||
             FuncName == "cudaIpcOpenMemHandle" ||
             FuncName == "cudaIpcCloseMemHandle") {
    report(CE->getBeginLoc(), Diagnostics::IPC_NOT_SUPPORTED, false,
           MapNames::ITFName.at(FuncName));
  } else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected function name: " << FuncName;
    return;
  }
}

REGISTER_RULE(FunctionCallRule, PassKind::PK_Migration)

EventAPICallRule *EventAPICallRule::CurrentRule = nullptr;
void EventAPICallRule::registerMatcher(MatchFinder &MF) {
  auto eventAPIName = [&]() {
    return hasAnyName(
        "cudaEventCreate", "cudaEventCreateWithFlags", "cudaEventDestroy",
        "cudaEventRecord", "cudaEventElapsedTime", "cudaEventSynchronize",
                      "cudaEventQuery", "cuEventCreate", "cuEventRecord",
        "cuEventSynchronize", "cuEventQuery", "cuEventElapsedTime",
        "cuEventDestroy_v2");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(eventAPIName())), parentStmt()))
          .bind("eventAPICall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(eventAPIName())),
                               unless(parentStmt())))
                    .bind("eventAPICallUsed"),
                this);
}

bool isEqualOperator(const Stmt *S) {
  if (!S)
    return false;
  if (auto BO = dyn_cast<BinaryOperator>(S))
    return BO->getOpcode() == BO_EQ || BO->getOpcode() == BO_NE;

  if (auto COCE = dyn_cast<CXXOperatorCallExpr>(S))
    return COCE->getOperator() == OO_EqualEqual ||
           COCE->getOperator() == OO_ExclaimEqual;

  return false;
}
bool isAssignOperator(const Stmt *);
const Expr *getLhs(const Stmt *);
const Expr *getRhs(const Stmt *);
const VarDecl *getAssignTargetDecl(const Stmt *E) {
  if (isAssignOperator(E))
    if (auto L = getLhs(E))
      if (auto DRE = dyn_cast<DeclRefExpr>(L->IgnoreImpCasts()))
        return dyn_cast<VarDecl>(DRE->getDecl());

  return nullptr;
}

const VarDecl *EventQueryTraversal::getAssignTarget(const CallExpr *Call) {
  auto ParentMap = Context.getParents(*Call);
  if (ParentMap.size() == 0)
    return nullptr;

  auto &Parent = ParentMap[0];
  if (auto VD = Parent.get<VarDecl>()) {
    return VD;
  }
  if (auto BO = Parent.get<BinaryOperator>())
    return getAssignTargetDecl(BO);

  if (auto COE = Parent.get<CXXOperatorCallExpr>())
    return getAssignTargetDecl(COE);

  return nullptr;
}

bool EventQueryTraversal::isEventQuery(const CallExpr *Call) {
  if (!Call)
    return false;
  if (auto Callee = Call->getDirectCallee())
    if (Callee->getName() == "cudaEventQuery" ||
        Callee->getName() == "cuEventQuery")
      return QueryCallUsed = true;
  return false;
}

std::string EventQueryTraversal::getReplacedEnumValue(const DeclRefExpr *DRE) {
  if (!DRE)
    return std::string();
  if (auto ECD = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
    auto Name = ECD->getName();
    if (Name == "cudaSuccess") {
      return MapNames::getClNamespace() +
             "info::event_command_status::complete";
    }
  }
  return std::string();
}

TextModification *
EventQueryTraversal::buildCallReplacement(const CallExpr *Call) {
  static std::string MemberName = "get_info<" + MapNames::getClNamespace() +
                                  "info::event::command_execution_status>";
  std::string ReplStr;
  MemberCallPrinter<const Expr *, StringRef> Printer(Call->getArg(0), true,
                                                     MemberName);
  llvm::raw_string_ostream OS(ReplStr);
  Printer.print(OS);
  return new ReplaceStmt(Call, std::move(OS.str()));
}

bool EventQueryTraversal::checkVarDecl(const VarDecl *VD,
                                       const FunctionDecl *TargetFD) {
  if (!VD || !TargetFD)
    return false;
  if (dyn_cast<FunctionDecl>(VD->getDeclContext()) == TargetFD &&
      VD->getKind() == Decl::Var) {
    auto DS = DpctGlobalInfo::findParent<DeclStmt>(VD);
    return DS && DS->isSingleDecl();
  }
  return false;
}

bool EventQueryTraversal::traverseFunction(const FunctionDecl *FD,
                                           const VarDecl *VD) {
  if (!checkVarDecl(VD, FD))
    return Rule->VarDeclCache[VD] = false;
  ResultTy Result;
  auto Ret = traverseStmt(FD->getBody(), VD, Result) && QueryCallUsed;

  for (const auto &R : Result) {
    Rule->ExprCache[R.first] = Ret;
    if (Ret)
      Rule->emplaceTransformation(R.second);
  }
  return Rule->VarDeclCache[VD] = Ret;
}

bool EventQueryTraversal::traverseAssignRhs(const Expr *Rhs, ResultTy &Result) {
  if (!Rhs)
    return true;
  auto Call = dyn_cast<CallExpr>(Rhs->IgnoreImpCasts());
  if (!isEventQuery(Call))
    return false;

  Result.emplace_back(Call, buildCallReplacement(Call));
  return true;
}

bool EventQueryTraversal::traverseEqualStmt(const Stmt *S, const VarDecl *VD,
                                            ResultTy &Result) {
  const Expr *L = getLhs(S), *R = getRhs(S);
  do {
    if (!L || !R)
      break;
    const DeclRefExpr *LRef = dyn_cast<DeclRefExpr>(L->IgnoreImpCasts()),
                      *RRef = dyn_cast<DeclRefExpr>(R->IgnoreImpCasts());
    if (!LRef || !RRef)
      break;

    const DeclRefExpr *TargetExpr = nullptr;
    if (LRef->getDecl() == VD)
      TargetExpr = RRef;
    else if (RRef->getDecl() == VD)
      TargetExpr = LRef;

    auto Replaced = getReplacedEnumValue(TargetExpr);
    if (Replaced.empty())
      break;
    Result.emplace_back(TargetExpr, new ReplaceStmt(TargetExpr, Replaced));
    return true;
  } while (false);
  for (auto Child : S->children())
    if (!traverseStmt(Child, VD, Result))
      return false;
  return true;
}

bool EventQueryTraversal::traverseStmt(const Stmt *S, const VarDecl *VD,
                                       ResultTy &Result) {
  if (!S)
    return true;
  switch (S->getStmtClass()) {
  case Stmt::DeclStmtClass: {
    auto DS = static_cast<const DeclStmt *>(S);
    if (DS->isSingleDecl() && VD == DS->getSingleDecl()) {
      Result.emplace_back(
          S, new ReplaceTypeInDecl(VD, MapNames::getClNamespace() +
                                           "info::event_command_status"));
      return traverseAssignRhs(VD->getInit(), Result);
    }
    for (auto D : DS->decls())
      if (auto VDecl = dyn_cast<VarDecl>(D))
        if (!traverseStmt(VDecl->getInit(), VD, Result))
          return false;
    break;
  }
  case Stmt::DeclRefExprClass:
    if (auto D =
            dyn_cast<VarDecl>(static_cast<const DeclRefExpr *>(S)->getDecl()))
      return D != VD;
    break;
  case Stmt::BinaryOperatorClass:
  case Stmt::CXXOperatorCallExprClass:
    if (getAssignTargetDecl(S) == VD)
      return traverseAssignRhs(getRhs(S), Result);
    if (isEqualOperator(S))
      return traverseEqualStmt(S, VD, Result);
    LLVM_FALLTHROUGH;
  default:
    for (auto Child : S->children())
      if (!traverseStmt(Child, VD, Result))
        return false;
    break;
  }
  return true;
}

bool EventQueryTraversal::startFromStmt(
    const Stmt *S, const std::function<const VarDecl *()> &VDGetter) {
  if (!Rule)
    return false;
  auto ExprIter = Rule->ExprCache.find(S);
  if (ExprIter != Rule->ExprCache.end())
    return ExprIter->second;

  const VarDecl *VD = VDGetter();
  if (!VD)
    return Rule->ExprCache[S] = false;
  auto VarDeclIter = Rule->VarDeclCache.find(VD);
  if (VarDeclIter != Rule->VarDeclCache.end())
    return Rule->ExprCache[S] = VarDeclIter->second;

  return traverseFunction(DpctGlobalInfo::findAncestor<FunctionDecl>(S), VD);
}

// Handle case like "cudaSuccess == cudaEventQuery()" or "cudaSuccess !=
// cudaEventQeury()".
void EventQueryTraversal::handleDirectEqualStmt(const DeclRefExpr *DRE,
                                                const CallExpr *Call) {
  if (!isEventQuery(Call))
    return;
  auto DREReplaceStr = getReplacedEnumValue(DRE);
  if (DREReplaceStr.empty())
    return;
  Rule->emplaceTransformation(new ReplaceStmt(DRE, DREReplaceStr));
  Rule->emplaceTransformation(buildCallReplacement(Call));
  Rule->ExprCache[DRE] = Rule->ExprCache[Call] = true;
  return;
}

bool EventQueryTraversal::startFromQuery(const CallExpr *Call) {
  return startFromStmt(Call, [&]() -> const VarDecl * {
    if (isEventQuery(Call))
      return getAssignTarget(Call);
    return nullptr;
  });
}

bool EventQueryTraversal::startFromEnumRef(const DeclRefExpr *DRE) {
  if (getReplacedEnumValue(DRE).empty())
    return false;

  return startFromStmt(DRE, [&]() -> const VarDecl * {
    auto ImpCast = DpctGlobalInfo::findParent<ImplicitCastExpr>(DRE);
    if (!ImpCast)
      return nullptr;
    auto S = DpctGlobalInfo::findParent<Stmt>(ImpCast);
    if (!isEqualOperator(S))
      return nullptr;
    const Expr *TargetExpr = nullptr, *L = getLhs(S), *R = getRhs(S);
    if (L == ImpCast)
      TargetExpr = R;
    else if (R == ImpCast)
      TargetExpr = L;

    if (!TargetExpr)
      return nullptr;
    if (auto TargetDRE = dyn_cast<DeclRefExpr>(TargetExpr->IgnoreImpCasts()))
      return dyn_cast<VarDecl>(TargetDRE->getDecl());
    else if (auto Call = dyn_cast<CallExpr>(TargetExpr->IgnoreImpCasts()))
      handleDirectEqualStmt(DRE, Call);
    return nullptr;
  });
}

bool EventQueryTraversal::startFromTypeLoc(TypeLoc TL) {
  if (DpctGlobalInfo::getUnqualifiedTypeName(QualType(TL.getTypePtr(), 0)) ==
      "cudaError_t")
    if (auto DS = DpctGlobalInfo::findAncestor<DeclStmt>(&TL))
      return startFromStmt(DS, [&]() -> const VarDecl * {
        if (DS->isSingleDecl())
          if (auto VD = dyn_cast<VarDecl>(DS->getSingleDecl()))
            return (VD->getTypeSourceInfo()->getTypeLoc() == TL) ? VD : nullptr;
        return nullptr;
      });
  return false;
}

EventQueryTraversal EventAPICallRule::getEventQueryTraversal() {
  return EventQueryTraversal(CurrentRule);
}

bool EventAPICallRule::isEventElapsedTimeFollowed(const CallExpr *Expr) {
  bool IsMeasureTime = false;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto CELoc = SM.getExpansionLoc(Expr->getBeginLoc()).getRawEncoding();
  auto FD = getImmediateOuterFuncDecl(Expr);
  if (!FD)
    return false;
  auto FuncBody = FD->getBody();
  for (auto It = FuncBody->child_begin(); It != FuncBody->child_end(); ++It) {
    auto Loc = SM.getExpansionLoc(It->getBeginLoc()).getRawEncoding();
    if (Loc < CELoc)
      continue;

    const CallExpr *Call = nullptr;
    findEventAPI(*It, Call, "cudaEventElapsedTime");
    if (!Call) {
      findEventAPI(*It, Call, "cuEventElapsedTime");
    }

    if (Call) {
      // To check the argment of "cudaEventQuery" is same as the second argument
      // of "cudaEventElapsedTime", in the code pieces:
      // ...
      // unsigned long int counter = 0;
      // while (cudaEventQuery(stop) == cudaErrorNotReady) {
      //  counter++;
      // }
      // cudaEventElapsedTime(&gpu_time, start, stop);
      // ...
      auto Arg2 = getStmtSpelling(Call->getArg(2));
      auto Arg0 = getStmtSpelling(Expr->getArg(0));
      if (Arg2 == Arg0)
        IsMeasureTime = true;
    }
  }
  return IsMeasureTime;
}

void EventAPICallRule::runRule(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "eventAPICall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "eventAPICallUsed")))
      return;
    IsAssigned = true;
  }

  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(FuncName);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  if (FuncName == "cudaEventQuery" || FuncName == "cuEventQuery") {
    if (getEventQueryTraversal().startFromQuery(CE))
      return;

    // Pattern-based solution for migration of time measurement code is enabled
    // only when option '--enable-profiling' is disabled.
    if (!isEventElapsedTimeFollowed(CE) &&
        !DpctGlobalInfo::getEnablepProfilingFlag()) {
      auto FD = getImmediateOuterFuncDecl(CE);
      if (!FD)
        return;
      auto FuncBody = FD->getBody();

      if (!FuncBody)
        return;
      reset();
      TimeElapsedCE = CE;
      if (FuncName == "cudaEventQuery") {
        updateAsyncRange(FuncBody, "cudaEventCreate");
      } else {
        updateAsyncRange(FuncBody, "cuEventCreate");
      }
      if (RecordBegin && RecordEnd) {
        processAsyncJob(FuncBody);

        if (!IsKernelInLoopStmt) {
          DpctGlobalInfo::getInstance().updateTimeStubTypeInfo(
              RecordBegin->getBeginLoc(), TimeElapsedCE->getEndLoc());
        }
      }
    }

    ExprAnalysis EA(CE->getArg(0));
    std::string ReplStr = "(int)" + EA.getReplacedString() + "->get_info<" +
                          MapNames::getClNamespace() +
                          "info::event::command_execution_status>()";
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaEventRecord" || FuncName == "cuEventRecord") {
    handleEventRecord(CE, Result, IsAssigned);
  } else if (FuncName == "cudaEventElapsedTime" ||
             FuncName == "cuEventElapsedTime") {
    // Reset from last migration on time measurement.
    // Do NOT delete me.
    reset();
    TimeElapsedCE = CE;
    handleEventElapsedTime(IsAssigned);
  } else if (FuncName == "cudaEventSynchronize" ||
             FuncName == "cuEventSynchronize") {
    if(DpctGlobalInfo::getEnablepProfilingFlag()) {
      // Option '--enable-profiling' is enabled
      std::string ReplStr{getStmtSpelling(CE->getArg(0))};
      ReplStr += "->wait_and_throw()";
      if (IsAssigned) {
        ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
        requestFeature(HelperFeatureEnum::device_ext);
      }
      emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    } else {
      // Option '--enable-profiling' is not enabled
      bool NeedReport = false;
      std::string ReplStr{getStmtSpelling(CE->getArg(0))};
      ReplStr += "->wait_and_throw()";
      if (IsAssigned) {
        ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
        NeedReport = true;
      }

      auto &Context = dpct::DpctGlobalInfo::getContext();
      const auto &TM = ReplaceStmt(CE, ReplStr);
      const auto R = TM.getReplacement(Context);
      DpctGlobalInfo::getInstance().insertEventSyncTypeInfo(R, NeedReport,
                                                            IsAssigned);
    }
  } else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected function name: " << FuncName;
    return;
  }
}

// Gets the declared size of the array referred in E, if E is either
// ConstantArrayType or VariableArrayType; otherwise return an empty
// string.
// For example:
// int maxSize = 100;
// int a[10];
// int b[maxSize];
//
// E1(const Expr *)-> a[2]
// E2(const Expr *)-> b[3]
// getArraySize(E1) => 10
// getArraySize(E2) => maxSize
std::string getArrayDeclSize(const Expr *E) {
  const ArrayType *AT = nullptr;
  if (auto ME = dyn_cast<MemberExpr>(E))
    AT = ME->getMemberDecl()->getType()->getAsArrayTypeUnsafe();
  else if (auto DRE = dyn_cast<DeclRefExpr>(E))
    AT = DRE->getDecl()->getType()->getAsArrayTypeUnsafe();

  if (!AT)
    return {};

  if (auto CAT = dyn_cast<ConstantArrayType>(AT))
    return std::to_string(*CAT->getSize().getRawData());
  if (auto VAT = dyn_cast<VariableArrayType>(AT))
    return ExprAnalysis::ref(VAT->getSizeExpr());
  return {};
}

// Returns true if E is array type for a MemberExpr or DeclRefExpr; returns
// false if E is pointer type.
// Requires: E is the base of ArraySubscriptExpr
bool isArrayType(const Expr *E) {
  if (auto ME = dyn_cast<MemberExpr>(E)) {
    auto AT = ME->getMemberDecl()->getType()->getAsArrayTypeUnsafe();
    return AT ? true : false;
  }
  if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
    auto AT = DRE->getDecl()->getType()->getAsArrayTypeUnsafe();
    return AT ? true : false;
  }
  return false;
}

// Get the time point helper variable name for an event. The helper variable is
// declared right after its corresponding event variable.
std::string getTimePointNameForEvent(const Expr *E, bool IsDecl) {
  std::string TimePointName;
  E = E->IgnoreImpCasts();
  if (auto UO = dyn_cast<UnaryOperator>(E))
    return getTimePointNameForEvent(UO->getSubExpr(), IsDecl);
  if (auto ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    auto Base = ASE->getBase()->IgnoreImpCasts();
    if (isArrayType(Base))
      return getTimePointNameForEvent(Base, IsDecl) + "[" +
             (IsDecl ? getArrayDeclSize(Base)
                     : ExprAnalysis::ref(ASE->getIdx())) +
             "]";
    return getTimePointNameForEvent(Base, IsDecl) + "_" +
           ExprAnalysis::ref(ASE->getIdx());
  }
  if (auto ME = dyn_cast<MemberExpr>(E)) {
    auto Base = ME->getBase()->IgnoreImpCasts();
    return ((IsDecl || ME->isImplicitAccess())
                ? ""
                : ExprAnalysis::ref(Base) + (ME->isArrow() ? "->" : ".")) +
           ME->getMemberDecl()->getNameAsString() + getCTFixedSuffix();
  }
  if (auto DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getDecl()->getNameAsString() + getCTFixedSuffix();
  return TimePointName;
}

// Get the (potentially inner) decl of E for common Expr types, including
// UnaryOperator, ArraySubscriptExpr, MemberExpr and DeclRefExpr; otherwise
// returns nullptr;
const ValueDecl *getDecl(const Expr *E) {
  E = E->IgnoreImpCasts();
  if (auto UO = dyn_cast<UnaryOperator>(E))
    return getDecl(UO->getSubExpr());
  if (auto ASE = dyn_cast<ArraySubscriptExpr>(E))
    return getDecl(ASE->getBase()->IgnoreImpCasts());
  if (auto ME = dyn_cast<MemberExpr>(E))
    return ME->getMemberDecl();
  if (auto DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getDecl();
  return nullptr;
}

void EventAPICallRule::findEventAPI(const Stmt *Node, const CallExpr *&Call,
                                    const std::string EventAPIName) {
  if (!Node)
    return;

  if (auto CE = dyn_cast<CallExpr>(Node)) {
    if (CE->getDirectCallee()) {
      if (CE->getDirectCallee()->getNameAsString() == EventAPIName) {
        Call = CE;
        return;
      }
    }
  }
  for (auto It = Node->child_begin(); It != Node->child_end(); ++It) {
    findEventAPI(*It, Call, EventAPIName);
  }
}

void EventAPICallRule::handleEventRecordWithProfilingEnabled(
    const CallExpr *CE, const MatchFinder::MatchResult &Result,
    bool IsAssigned) {
  auto StreamArg = CE->getArg(CE->getNumArgs() - 1);
  auto StreamName = getStmtSpelling(StreamArg);
  auto ArgName = getStmtSpelling(CE->getArg(0));
  bool IsDefaultStream = isDefaultStream(StreamArg);
  auto IndentLoc = CE->getBeginLoc();
  auto &SM = DpctGlobalInfo::getSourceManager();

  if (IndentLoc.isMacroID())
    IndentLoc = SM.getExpansionLoc(IndentLoc);

  if (IsAssigned) {

    std::string StmtStr;
    if (IsDefaultStream) {
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
      std::string Str;
      if (!DpctGlobalInfo::useEnqueueBarrier()) {
        // ext_oneapi_submit_barrier is specified in the value of option
        // --no-dpcpp-extensions.
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {

          Str = MapNames::getDpctNamespace() +
                "get_current_device().queues_wait_and_throw();";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          std::string SubStr = "{{NEEDREPLACEQ" + std::to_string(Index) +
                               "}}.single_task([=](){});";
          SubStr = "*" + ArgName + " = " + SubStr;
          Str += SubStr;

          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += MapNames::getDpctNamespace() +
                 "get_current_device().queues_wait_and_throw();";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += "return 0;";

          Str = "[](){" + Str + "}()";
          emplaceTransformation(new ReplaceStmt(CE, std::move(Str)));
          return;

        } else {
          Str = "{{NEEDREPLACEQ" + std::to_string(Index) +
                "}}.single_task([=](){})";
        }

      } else {
        Str = "{{NEEDREPLACEQ" + std::to_string(Index) +
              "}}.ext_oneapi_submit_barrier()";
      }
      StmtStr = "*" + ArgName + " = " + Str;
    } else {
      std::string Str;
      if (!DpctGlobalInfo::useEnqueueBarrier()) {
        // ext_oneapi_submit_barrier is specified in the value of option
        // --no-dpcpp-extensions.

        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {

          Str = MapNames::getDpctNamespace() +
                "get_current_device().queues_wait_and_throw();";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += StreamName + "->" + "single_task([=](){});";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += MapNames::getDpctNamespace() +
                 "get_current_device().queues_wait_and_throw()";

          Str = "[](){" + Str + "}()";
          emplaceTransformation(new ReplaceStmt(CE, std::move(Str)));
          return;
        } else {
          Str = StreamName + "->" + "single_task([=](){})";
        }

      } else {
        Str = StreamName + "->" + "ext_oneapi_submit_barrier()";
      }
      StmtStr = "*" + ArgName + " = " + Str;
    }
    StmtStr = "DPCT_CHECK_ERROR(" + StmtStr + ")";

    emplaceTransformation(new ReplaceStmt(CE, std::move(StmtStr)));

    report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_ZERO, false);

  } else {
    std::string ReplStr;
    if (IsDefaultStream) {
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
      std::string Str;
      if (!DpctGlobalInfo::useEnqueueBarrier()) {
        // ext_oneapi_submit_barrier is specified in the value of option
        // --no-dpcpp-extensions.

        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {

          Str = MapNames::getDpctNamespace() +
                "get_current_device().queues_wait_and_throw();";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += "*" + ArgName + " = {{NEEDREPLACEQ" + std::to_string(Index) +
                 "}}.single_task([=](){});";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += MapNames::getDpctNamespace() +
                 "get_current_device().queues_wait_and_throw()";

        } else {
          Str = "*" + ArgName + " = {{NEEDREPLACEQ" + std::to_string(Index) +
                "}}.single_task([=](){})";
        }

      } else {
        Str = "*" + ArgName + " = {{NEEDREPLACEQ" + std::to_string(Index) +
              "}}.ext_oneapi_submit_barrier()";
      }
      ReplStr += Str;
    } else {

      std::string Str;
      if (!DpctGlobalInfo::useEnqueueBarrier()) {
        // ext_oneapi_submit_barrier is specified in the value of option
        // --no-dpcpp-extensions.

        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {

          Str = MapNames::getDpctNamespace() +
                "get_current_device().queues_wait_and_throw();";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();

          Str += "*" + ArgName + " = " + StreamName + "->single_task([=](){});";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += MapNames::getDpctNamespace() +
                 "get_current_device().queues_wait_and_throw()";

        } else {
          Str = "*" + ArgName + " = " + StreamName + "->single_task([=](){})";
        }

      } else {
        Str = "*" + ArgName + " = " + StreamName +
              "->ext_oneapi_submit_barrier()";
      }
      ReplStr += Str;
    }

    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  }
}

void EventAPICallRule::handleEventRecordWithProfilingDisabled(
    const CallExpr *CE, const MatchFinder::MatchResult &Result,
    bool IsAssigned) {

  // Insert the helper variable right after the event variables
  static std::set<std::pair<const Decl *, std::string>> DeclDupFilter;
  auto &SM = DpctGlobalInfo::getSourceManager();
  const ValueDecl *MD = getDecl(CE->getArg(0));
  std::string InsertStr;
  bool IsParmVarDecl = isa<ParmVarDecl>(MD);

  if (!IsParmVarDecl)
    report(CE->getBeginLoc(), Diagnostics::TIME_MEASUREMENT_FOUND, false);

  DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), HT_Chrono);

  if (isInMacroDefinition(MD->getBeginLoc(), MD->getEndLoc())) {
    InsertStr += "\\";
  }
  InsertStr += getNL();
  InsertStr += getIndent(MD->getBeginLoc(), SM).str();
  InsertStr += "std::chrono::time_point<std::chrono::steady_clock> ";
  InsertStr += getTimePointNameForEvent(CE->getArg(0), true);
  InsertStr += ";";
  auto Pair = std::make_pair(MD, InsertStr);
  if (DeclDupFilter.find(Pair) == DeclDupFilter.end()) {
    DeclDupFilter.insert(Pair);
    if (!IsParmVarDecl)
      emplaceTransformation(new InsertAfterDecl(MD, std::move(InsertStr)));
  }

  std::ostringstream Repl;
  // Replace event recording with std::chrono timing
  if (!IsParmVarDecl) {
    Repl << getTimePointNameForEvent(CE->getArg(0), false)
        << " = std::chrono::steady_clock::now()";
  }

  const std::string Name =
      CE->getCalleeDecl()->getAsFunction()->getNameAsString();

  auto StreamArg = CE->getArg(CE->getNumArgs() - 1);
  auto StreamName = getStmtSpelling(StreamArg);
  auto ArgName = getStmtSpelling(CE->getArg(0));
  bool IsDefaultStream = isDefaultStream(StreamArg);
  auto IndentLoc = CE->getBeginLoc();
  auto &Context = dpct::DpctGlobalInfo::getContext();

  if (IsAssigned) {
    if (!DpctGlobalInfo::useEnqueueBarrier()) {
      // ext_oneapi_submit_barrier is specified in the value of option
      // --no-dpcpp-extensions.
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      std::string StmtStr;

      if (IsDefaultStream) {
        if (isPlaceholderIdxDuplicated(CE))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);

        std::string Str = "{{NEEDREPLACEQ" + std::to_string(Index) +
                          "}}.ext_oneapi_submit_barrier()";
        StmtStr = "*" + ArgName + " = " + Str;
      } else {
        std::string Str = StreamName + "->" + "ext_oneapi_submit_barrier()";
        StmtStr = "*" + ArgName + " = " + Str;
      }
      StmtStr = "DPCT_CHECK_ERROR(" + StmtStr + ")";

      auto ReplWithSubmitBarrier =
          ReplaceStmt(CE, StmtStr).getReplacement(Context);
      auto ReplWithoutSubmitBarrier =
          ReplaceStmt(CE, "0").getReplacement(Context);
      DpctGlobalInfo::getInstance().insertTimeStubTypeInfo(
          ReplWithSubmitBarrier, ReplWithoutSubmitBarrier);
    }
    if (!IsParmVarDecl)
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_ZERO, false);

    auto OuterStmt = findNearestNonExprNonDeclAncestorStmt(CE);

    if (!IsParmVarDecl)
      Repl << "; ";

    if (IndentLoc.isMacroID())
      IndentLoc = SM.getExpansionLoc(IndentLoc);

    if (!IsParmVarDecl)
      Repl << getNL() << getIndent(IndentLoc, SM).str();

    auto TM = new InsertText(SM.getExpansionLoc(OuterStmt->getBeginLoc()),
                             std::move(Repl.str()));
    TM->setInsertPosition(IP_Right);
    emplaceTransformation(TM);
  } else {
    if (!DpctGlobalInfo::useEnqueueBarrier()) {
      // ext_oneapi_submit_barrier is specified in the value of option
      // --no-dpcpp-extensions.
      auto TM = new ReplaceStmt(CE, std::move(Repl.str()));
      TM->setInsertPosition(IP_Right);
      emplaceTransformation(TM);
    } else {
      std::string StrWithoutSubmitBarrier = Repl.str();
      auto ReplWithoutSB =
          ReplaceStmt(CE, StrWithoutSubmitBarrier).getReplacement(Context);
      std::string ReplStr;
      if (!IsParmVarDecl)
        ReplStr += ";";
      if (isInMacroDefinition(MD->getBeginLoc(), MD->getEndLoc())) {
        ReplStr += "\\";
      }
      if (IsDefaultStream) {
        if (isPlaceholderIdxDuplicated(CE))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
        std::string Str = "*" + ArgName + " = {{NEEDREPLACEQ" +
                          std::to_string(Index) +
                          "}}.ext_oneapi_submit_barrier()";
        if (!IsParmVarDecl)
          ReplStr += getNL();
        ReplStr += getIndent(IndentLoc, SM).str();
        ReplStr += Str;
      } else {
        std::string Str = "*" + ArgName + " = " + StreamName +
                          "->ext_oneapi_submit_barrier()";
        if (!IsParmVarDecl)
          ReplStr += getNL();
        ReplStr += getIndent(IndentLoc, SM).str();
        ReplStr += Str;
      }
      Repl << ReplStr;
      auto ReplWithSB = ReplaceStmt(CE, Repl.str()).getReplacement(Context);
      DpctGlobalInfo::getInstance().insertTimeStubTypeInfo(ReplWithSB,
                                                           ReplWithoutSB);
    }
  }
}

void EventAPICallRule::handleEventRecord(const CallExpr *CE,
                                         const MatchFinder::MatchResult &Result,
                                         bool IsAssigned) {
  if (!getDecl(CE->getArg(0)))
    return;

  if (DpctGlobalInfo::getEnablepProfilingFlag()) {
    // Option '--enable-profiling' is enabled
    handleEventRecordWithProfilingEnabled(CE, Result, IsAssigned);
  } else {
    // Option '--enable-profiling' is disabled
    handleEventRecordWithProfilingDisabled(CE, Result, IsAssigned);
  }
}

void EventAPICallRule::handleEventElapsedTime(bool IsAssigned) {
  if(DpctGlobalInfo::getEnablepProfilingFlag()) {
    // Option '--enable-profiling' is enabled
    auto StmtStrArg0 = getStmtSpelling(TimeElapsedCE->getArg(0));
    auto StmtStrArg1 = getStmtSpelling(TimeElapsedCE->getArg(1));
    auto StmtStrArg2 = getStmtSpelling(TimeElapsedCE->getArg(2));

    std::ostringstream Repl;
    std::string Assginee = "*(" + StmtStrArg0 + ")";
    if (auto UO = dyn_cast<UnaryOperator>(TimeElapsedCE->getArg(0))) {
      if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf)
        Assginee = getStmtSpelling(UO->getSubExpr());
    }

    auto StartTimeStr = StmtStrArg1 + "->get_profiling_info<"
                            "sycl::info::event_profiling::command_start>()";
    auto StopTimeStr =  StmtStrArg2 + "->get_profiling_info<"
                            "sycl::info::event_profiling::command_end>()";

    Repl << Assginee << " = ("
        << StopTimeStr << " - " << StartTimeStr << ") / 1000000.0f";
    if (IsAssigned) {
      std::ostringstream Temp;
      Temp << "DPCT_CHECK_ERROR(" << Repl.str() << ")";
      Repl = std::move(Temp);
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(TimeElapsedCE, std::move(Repl.str())));
  } else {
    // Option '--enable-profiling' is not enabled
    auto StmtStrArg0 = getStmtSpelling(TimeElapsedCE->getArg(0));
    auto StmtStrArg1 = getTimePointNameForEvent(TimeElapsedCE->getArg(1), false);
    auto StmtStrArg2 = getTimePointNameForEvent(TimeElapsedCE->getArg(2), false);
    std::ostringstream Repl;
    std::string Assginee = "*(" + StmtStrArg0 + ")";
    if (auto UO = dyn_cast<UnaryOperator>(TimeElapsedCE->getArg(0))) {
      if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf)
        Assginee = getStmtSpelling(UO->getSubExpr());
    }
    Repl << Assginee << " = std::chrono::duration<float, std::milli>("
        << StmtStrArg2 << " - " << StmtStrArg1 << ").count()";
    if (IsAssigned) {
      std::ostringstream Temp;
      Temp << "DPCT_CHECK_ERROR((" << Repl.str() << "))";
      Repl = std::move(Temp);
      requestFeature(HelperFeatureEnum::device_ext);
    }
    const std::string Name =
        TimeElapsedCE->getCalleeDecl()->getAsFunction()->getNameAsString();
    emplaceTransformation(new ReplaceStmt(TimeElapsedCE, std::move(Repl.str())));
    handleTimeMeasurement();
  }
}

bool EventAPICallRule::IsEventArgArraySubscriptExpr(const Expr *E) {
  E = E->IgnoreImpCasts();
  if (auto UO = dyn_cast<UnaryOperator>(E))
    return IsEventArgArraySubscriptExpr(UO->getSubExpr());
  if (auto PE = dyn_cast<ParenExpr>(E))
    return IsEventArgArraySubscriptExpr(PE->getSubExpr());
  if (dyn_cast<ArraySubscriptExpr>(E))
    return true;
  return false;
}

const Expr *EventAPICallRule::findNextRecordedEvent(const Stmt *Node,
                                                    unsigned KCallLoc) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  for (auto Iter = Node->child_begin(); Iter != Node->child_end(); ++Iter) {

    const CallExpr *Call = nullptr;
    findEventAPI(*Iter, Call, "cudaEventRecord");

    if (!Call)
      findEventAPI(*Iter, Call, "cuEventRecord");

    if (Call) {
      if (SM.getExpansionLoc(Call->getBeginLoc()).getRawEncoding() > KCallLoc)
        return Call->getArg(0);
    }
  }
  return nullptr;
}

//  The following is a typical code piece, in which three
//  locations are used to help migrate:
//
//  cudaEventRecord(start);                 // <<== RecordBeginLoc
//  ...
//  mem_calls();
//  kernel_calls();
//  mem_calls();
//  ...
//  cudaEventRecord(stop);                  // <<== RecordEndLoc
//  ...
//  sync_calls();
//  ...
//  cudaEventElapsedTime(&et, start, stop); // <<== TimeElapsedLoc
//
//  or
//
//  cudaEventCreate(&stop);               // <<== RecordBeginLoc
//  ...
//  async_mem_calls
//  kernel_calls
//  async_mem_calls
//  ...
//  cudaEventRecord(stop);
//    while (cudaEventQuery(stop) == cudaErrorNotReady) { // <<== RecordEndLoc
//                                                             /TimeElapsedLoc
//        ...
//    }
//  processAsyncJob is used to process all sync calls between
//  RecordEndLoc and RecordEndLoc, and RecordEndLoc and TimeElapsedLoc.
void EventAPICallRule::processAsyncJob(const Stmt *Node) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  RecordBeginLoc =
      SM.getExpansionLoc(RecordBegin->getBeginLoc()).getRawEncoding();
  RecordEndLoc = SM.getExpansionLoc(RecordEnd->getBeginLoc()).getRawEncoding();
  TimeElapsedLoc =
      SM.getExpansionLoc(TimeElapsedCE->getBeginLoc()).getRawEncoding();

  // Handle the kernel calls and async memory operations between start and stop
  handleTargetCalls(Node);

  if (USMLevel == UsmLevel::UL_Restricted) {
    for (const auto &NewEventName : Events2Wait) {
      std::ostringstream SyncStmt;
      SyncStmt
          << NewEventName << getNL()
          << getIndent(SM.getExpansionLoc(RecordEnd->getBeginLoc()), SM).str();
      auto TM = new InsertText(SM.getExpansionLoc(RecordEnd->getBeginLoc()),
                               SyncStmt.str());
      TM->setInsertPosition(IP_AlwaysLeft);
      emplaceTransformation(TM);
    }
  } else {
    for (const auto &T : Queues2Wait) {
      std::ostringstream SyncStmt;
      SyncStmt
          << std::get<0>(T) << getNL()
          << getIndent(SM.getExpansionLoc(RecordEnd->getBeginLoc()), SM).str();
      auto TM = new InsertBeforeStmt(RecordEnd, SyncStmt.str(), 0 /*PairID*/,
                                     true /*DoMacroExpansion*/);
      TM->setInsertPosition(IP_AlwaysLeft);
      emplaceTransformation(TM);
    }
  }
}

void EventAPICallRule::findThreadSyncLocation(const Stmt *Node) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  const CallExpr *Call = nullptr;
  findEventAPI(Node, Call, "cudaThreadSynchronize");

  if (Call) {
    ThreadSyncLoc = SM.getExpansionLoc(Call->getBeginLoc()).getRawEncoding();
  }
}

void EventAPICallRule::updateAsyncRangRecursive(
    const Stmt *Node, const CallExpr *AsyncCE, const std::string EventAPIName) {
  if (!Node)
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto CELoc = SM.getExpansionLoc(AsyncCE->getBeginLoc()).getRawEncoding();
  for (auto Iter = Node->child_begin(); Iter != Node->child_end(); ++Iter) {
    if (*Iter == nullptr)
      continue;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
      findThreadSyncLocation(*Iter);

    if (SM.getExpansionLoc(Iter->getBeginLoc()).getRawEncoding() > CELoc) {
      return;
    }

    if (EventAPIName == "cudaEventRecord" || EventAPIName == "cuEventRecord") {
      const CallExpr *Call = nullptr;
      findEventAPI(*Iter, Call, EventAPIName);

      if (Call) {
        // Find the last call of Event Record on start and stop before
        // calculate the time elapsed
        auto Arg0 = getStmtSpelling(Call->getArg(0));
        if (Arg0 == getStmtSpelling(AsyncCE->getArg(1))) {
          RecordBegin = findNearestNonExprNonDeclAncestorStmt(Call);
        } else if (Arg0 == getStmtSpelling(AsyncCE->getArg(2))) {
          RecordEnd = findNearestNonExprNonDeclAncestorStmt(Call);
        }
      }

    } else if (EventAPIName == "cudaEventCreate" ||
               EventAPIName == "cuEventCreate") {

      const CallExpr *Call = nullptr;
      findEventAPI(*Iter, Call, EventAPIName);

      if (Call) {
        std::string Arg0;
        if (auto UO = dyn_cast<UnaryOperator>(Call->getArg(0))) {
          if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf) {
            Arg0 = getStmtSpelling(UO->getSubExpr());
          }
        }
        if (Arg0.empty())
          Arg0 = getStmtSpelling(Call->getArg(0));

        if (Arg0 == getStmtSpelling(AsyncCE->getArg(0)))
          RecordBegin = findNearestNonExprNonDeclAncestorStmt(Call);
      }

      // To update RecordEnd
      Call = nullptr;
      findEventAPI(*Iter, Call, "cudaEventRecord");
      if (!Call)
        findEventAPI(*Iter, Call, "cuEventRecord");

      if (Call) {
        auto Arg0 = getStmtSpelling(Call->getArg(0));
        if (Arg0 == getStmtSpelling(AsyncCE->getArg(0))) {
          RecordEnd = findNearestNonExprNonDeclAncestorStmt(Call);
        }
      }
    }

    // Recursively update range in deeper code structures
    updateAsyncRangRecursive(*Iter, AsyncCE, EventAPIName);
  }
}

//  The following is a typical code piece, in which three
//  locations are used to help migrate:
//
//  cudaEventRecord(start);                 // <<== RecordBeginLoc
//  ...
//  mem_calls();
//  kernel_calls();
//  mem_calls();
//  ...
//  cudaEventRecord(stop);                  // <<== RecordEndLoc
//  ...
//  sync_calls();
//  ...
//  cudaEventElapsedTime(&et, start, stop); // <<== TimeElapsedLoc
//
//  or
//
//  cudaEventCreate(&stop);               // <<== RecordBeginLoc
//  ...
//  async_mem_calls
//  kernel_calls
//  async_mem_calls
//  ...
//  cudaEventRecord(stop);
//    while (cudaEventQuery(stop) == cudaErrorNotReady) { // <<== RecordEndLoc
//                                                             /TimeElapsedLoc
//        ...
//    }
// \p FuncBody is the body of the function which calls function pointed by
// TimeElapsedCE \p EventAPIName is the EventAPI name (.i.e cudaEventRecord or
// cudaEventCreate) to help to locate RecordEndLoc.
void EventAPICallRule::updateAsyncRange(const Stmt *FuncBody,
                                        const std::string EventAPIName) {
  auto EventArg = TimeElapsedCE->getArg(0);
  if (IsEventArgArraySubscriptExpr(EventArg)) {
    // If the event arg is a ArraySubscriptExpr, if not async range is not
    // identified, mark all kernels in the current function to wait.
    updateAsyncRangRecursive(FuncBody, TimeElapsedCE, EventAPIName);
    if (!RecordEnd) {
      IsKernelSync = true;
      RecordBegin = *FuncBody->child_begin();
      RecordEnd = TimeElapsedCE;
    }
  } else {
    updateAsyncRangRecursive(FuncBody, TimeElapsedCE, EventAPIName);
  }
}

//  The following is a typical piece of time-measurement code, in which three
//  locations are used to help migrate:
//
//  cudaEventRecord(start);                 // <<== RecordBeginLoc
//  ...
//  mem_calls();
//  kernel_calls();
//  mem_calls();
//  ...
//  cudaEventRecord(stop);                  // <<== RecordEndLoc
//  ...
//  sync_calls();
//  ...
//  cudaEventElapsedTime(&et, start, stop); // <<== TimeElapsedLoc
void EventAPICallRule::handleTimeMeasurement() {

  auto FD = getImmediateOuterFuncDecl(TimeElapsedCE);
  if (!FD)
    return;

  const Stmt *FuncBody = nullptr;
  if (FD->isTemplateInstantiation()) {
    auto FTD = FD->getPrimaryTemplate();
    if (!FTD)
      return;
    FuncBody = FTD->getTemplatedDecl()->getBody();
  } else {
    FuncBody = FD->getBody();
  }

  if (!FuncBody)
    return;

  updateAsyncRange(FuncBody, "cudaEventRecord");
  updateAsyncRange(FuncBody, "cuEventRecord");

  if (!RecordBegin || !RecordEnd) {
    return;
  }

  // To store the range of code where time measurement takes place.
  processAsyncJob(FuncBody);

  if (!IsKernelInLoopStmt) {
    DpctGlobalInfo::getInstance().updateTimeStubTypeInfo(
        RecordBegin->getBeginLoc(), TimeElapsedCE->getEndLoc());
  }
}

// To get the redundant parent ParenExpr for \p Call to handle case like
// "(cudaEventSynchronize(stop))".
const clang::Stmt *
EventAPICallRule::getRedundantParenExpr(const CallExpr *Call) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*Call);
  if (Parents.size()) {
    auto &Parent = Parents[0];
    if (auto PE = Parent.get<ParenExpr>()) {
      if (auto ParentStmt = getParentStmt(PE)) {
        auto ParentStmtClass = ParentStmt->getStmtClass();
        bool Ret = ParentStmtClass == Stmt::StmtClass::IfStmtClass ||
                   ParentStmtClass == Stmt::StmtClass::WhileStmtClass ||
                   ParentStmtClass == Stmt::StmtClass::DoStmtClass ||
                   ParentStmtClass == Stmt::StmtClass::CallExprClass ||
                   ParentStmtClass == Stmt::StmtClass::ImplicitCastExprClass ||
                   ParentStmtClass == Stmt::StmtClass::BinaryOperatorClass ||
                   ParentStmtClass == Stmt::StmtClass::ForStmtClass;
        if (!Ret) {
          return PE;
        }
      }
    }
  }
  return nullptr;
}

//  cudaEventRecord(start);                 // <<== RecordBeginLoc
//  ...
//  mem_calls();
//  kernel_calls();
//  mem_calls();
//  ...
//  cudaEventRecord(stop);                  // <<== RecordEndLoc
//  ...
//  sync_calls();
//  ...
//  cudaEventElapsedTime(&et, start, stop); // <<== TimeElapsedLoc
void EventAPICallRule::handleTargetCalls(const Stmt *Node, const Stmt *Last) {
  if (!Node)
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();

  for (auto It = Node->child_begin(); It != Node->child_end(); ++It) {
    if (*It == nullptr)
      continue;
    auto Loc = SM.getExpansionLoc(It->getBeginLoc()).getRawEncoding();

    // Skip statements before RecordBeginLoc or after TimeElapsedLoc
    if (Loc > RecordBeginLoc && Loc <= TimeElapsedLoc) {

      // Handle cudaEventSynchronize between RecordEndLoc and TimeElapsedLoc
      if (Loc > RecordEndLoc && Loc < TimeElapsedLoc) {

        const CallExpr *Call = nullptr;
        std::string OriginalAPIName = "";
        findEventAPI(*It, Call, "cudaEventSynchronize");
        if (Call) {
          OriginalAPIName = "cudaEventSynchronize";
        } else {
          findEventAPI(*It, Call, "cuEventSynchronize");
          if (Call)
            OriginalAPIName = "cuEventSynchronize";
        }

        if (Call) {
          if (const clang::Stmt *S = getRedundantParenExpr(Call)) {
            // To remove statement like "(cudaEventSynchronize(stop));"
            emplaceTransformation(new ReplaceStmt(S, false, true, ""));
          }

          const auto &TM = ReplaceStmt(Call, "");
          auto &Context = dpct::DpctGlobalInfo::getContext();
          auto R = TM.getReplacement(Context);
          DpctGlobalInfo::getInstance().updateEventSyncTypeInfo(R);
        }
      }

      // Now handle all statements between RecordBeginLoc and RecordEndLoc
      switch (It->getStmtClass()) {
      case Stmt::CallExprClass: {
        handleOrdinaryCalls(dyn_cast<CallExpr>(*It));
        break;
      }
      case Stmt::CUDAKernelCallExprClass: {

        if (Last && (Last->getStmtClass() == Stmt::DoStmtClass ||
                     Last->getStmtClass() == Stmt::WhileStmtClass ||
                     Last->getStmtClass() == Stmt::ForStmtClass)) {
          IsKernelInLoopStmt = true;
        }

        auto FD = getImmediateOuterFuncDecl(Node);
        if (FD)
          handleKernelCalls(FD->getBody(), dyn_cast<CUDAKernelCallExpr>(*It));
        break;
      }
      case Stmt::ExprWithCleanupsClass: {
        auto ExprS = dyn_cast<ExprWithCleanups>(*It);
        auto *SubExpr = ExprS->getSubExpr();
        if (auto *KCall = dyn_cast<CUDAKernelCallExpr>(SubExpr)) {

          if (Last && (Last->getStmtClass() == Stmt::DoStmtClass ||
                       Last->getStmtClass() == Stmt::WhileStmtClass ||
                       Last->getStmtClass() == Stmt::ForStmtClass)) {
            IsKernelInLoopStmt = true;
          }

          auto FD = getImmediateOuterFuncDecl(Node);
          if (FD)
            handleKernelCalls(FD->getBody(), KCall);
        }
        break;
      }

      default:
        break;
      }
    }

    handleTargetCalls(*It, Node);
  }
}

void EventAPICallRule::handleKernelCalls(const Stmt *Node,
                                         const CUDAKernelCallExpr *KCall) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto KCallLoc = SM.getExpansionLoc(KCall->getBeginLoc()).getRawEncoding();
  auto K = DpctGlobalInfo::getInstance().insertKernelCallExpr(KCall);
  auto EventExpr = findNextRecordedEvent(Node, KCallLoc);
  if (!EventExpr && TimeElapsedCE->getNumArgs() == 3)
    EventExpr = TimeElapsedCE->getArg(2);

  auto ArgName = ExprAnalysis::ref(EventExpr);
  // Skip statements before RecordBeginLoc or after RecordEndLoc
  if (KCallLoc < RecordBeginLoc || KCallLoc > RecordEndLoc)
    return;

  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
    bool NeedWait = false;
    // In usm none mode, if cudaThreadSynchronize apears after kernel call,
    // kernel wait is not needed.
    NeedWait = ThreadSyncLoc > KCallLoc;

    if (KCallLoc > RecordBeginLoc && !NeedWait) {
      if (IsKernelSync) {
        K->setEvent(ArgName);
        K->setSync();
      } else {
        Queues2Wait.emplace_back(MapNames::getDpctNamespace() +
                                     "get_current_device()."
                                     "queues_wait_and_throw();",
                                 nullptr);
        requestFeature(HelperFeatureEnum::device_ext);
      }
    }
  }

  if (USMLevel == UsmLevel::UL_Restricted) {
    if (KCallLoc > RecordBeginLoc) {
      if (!IsKernelInLoopStmt && !IsKernelSync) {
        K->setEvent(ArgName);
        Events2Wait.push_back(ArgName + "->wait();");
      } else if (IsKernelSync) {
        K->setEvent(ArgName);
        K->setSync();
        // Events2Wait.push_back("(" + ArgName + ")" + ".wait();");
      } else {
        std::string WaitQueue = MapNames::getDpctNamespace() +
                                "get_current_device()."
                                "queues_wait_and_throw();";
        Events2Wait.push_back(WaitQueue);
        requestFeature(HelperFeatureEnum::device_ext);
      }
    }
  }
}

void EventAPICallRule::handleOrdinaryCalls(const CallExpr *Call) {
  auto Callee = Call->getDirectCallee();
  if (!Callee)
    return;
  auto CalleeName = Callee->getName();
  if (CalleeName.startswith("cudaMemcpy") && CalleeName.endswith("Async")) {
    auto StreamArg = Call->getArg(Call->getNumArgs() - 1);
    bool IsDefaultStream = isDefaultStream(StreamArg);
    bool NeedStreamWait = false;

    if (StreamArg->IgnoreImpCasts()->getStmtClass() ==
        Stmt::ArraySubscriptExprClass)
      NeedStreamWait = true;

    if (USMLevel == UsmLevel::UL_Restricted) {
      // std::string EventName = getTempNameForExpr(TimeElapsedCE->getArg(2));
      std::string EventName;
      if (TimeElapsedCE->getNumArgs() == 3) {
        EventName = getTempNameForExpr(TimeElapsedCE->getArg(2));
      } else {
        EventName = getTempNameForExpr(TimeElapsedCE->getArg(0));
      }
      std::string QueueName =
          IsDefaultStream ? "q_ct1_" : getTempNameForExpr(StreamArg);
      std::string NewEventName =
          EventName + QueueName + std::to_string(++QueueCounter[QueueName]);
      Events2Wait.push_back(NewEventName + ".wait();");
      auto &SM = DpctGlobalInfo::getSourceManager();
      std::ostringstream SyncStmt;
      SyncStmt << MapNames::getClNamespace() << "event " << NewEventName << ";"
               << getNL()
               << getIndent(SM.getExpansionLoc(RecordBegin->getBeginLoc()), SM)
                      .str();
      emplaceTransformation(new InsertText(
          SM.getExpansionLoc(RecordBegin->getBeginLoc()), SyncStmt.str()));

      auto TM = new InsertBeforeStmt(Call, NewEventName + " = ");
      TM->setInsertPosition(IP_Right);
      emplaceTransformation(TM);
    } else {
      std::tuple<bool, std::string, const CallExpr *> T;
      if (IsDefaultStream && !DefaultQueueAdded) {
        DefaultQueueAdded = true;
        if (isPlaceholderIdxDuplicated(Call))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        auto &SM = DpctGlobalInfo::getSourceManager();
        std::ostringstream SyncStmt;
        SyncStmt << "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.wait();"
                 << getNL()
                 << getIndent(SM.getExpansionLoc(RecordEnd->getBeginLoc()), SM)
                        .str();
        buildTempVariableMap(Index, Call, HelperFuncType::HFT_DefaultQueue);

        emplaceTransformation(new InsertText(
            SM.getExpansionLoc(RecordEnd->getBeginLoc()), SyncStmt.str()));
      } else if (!IsDefaultStream) {
        if (NeedStreamWait) {
          Queues2Wait.emplace_back(MapNames::getDpctNamespace() +
                                       "get_current_device()."
                                       "queues_wait_and_throw();",
                                   nullptr);
          requestFeature(HelperFeatureEnum::device_ext);
        } else {
          auto ArgName = getStmtSpelling(StreamArg);
          Queues2Wait.emplace_back(ArgName + "->wait();", nullptr);
        }
      }
    }
  }
}

REGISTER_RULE(EventAPICallRule, PassKind::PK_Migration)

void StreamAPICallRule::registerMatcher(MatchFinder &MF) {
  auto streamFunctionName = [&]() {
    return hasAnyName("cudaStreamCreate", "cudaStreamCreateWithFlags",
                      "cudaStreamCreateWithPriority", "cudaStreamDestroy",
                      "cudaStreamSynchronize", "cudaStreamGetPriority",
                      "cudaStreamGetFlags", "cudaDeviceGetStreamPriorityRange",
                      "cudaStreamAttachMemAsync", "cudaStreamBeginCapture",
                      "cudaStreamEndCapture", "cudaStreamIsCapturing",
                      "cudaStreamQuery", "cudaStreamWaitEvent",
                      "cudaStreamAddCallback", "cuStreamCreate",
                      "cuStreamSynchronize", "cuStreamWaitEvent",
                      "cuStreamDestroy_v2", "cuStreamAttachMemAsync",
                      "cuStreamAddCallback");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(streamFunctionName())), parentStmt()))
          .bind("streamAPICall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(streamFunctionName())),
                               unless(parentStmt())))
                    .bind("streamAPICallUsed"),
                this);
}

std::string getNewQueue(int Index) {
  extern bool AsyncHandlerFlag;
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  printPartialArguments(OS << "{{NEEDREPLACED" << std::to_string(Index)
                           << "}}.create_queue(",
                        AsyncHandlerFlag ? 1 : 0, "true")
      << ")";
  return OS.str();
}

void StreamAPICallRule::runRule(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "streamAPICall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "streamAPICallUsed")))
      return;
    IsAssigned = true;
  }

  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  if (!CallExprRewriterFactoryBase::RewriterMap)
    return;
  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(FuncName);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  if (FuncName == "cudaStreamCreate" || FuncName == "cuStreamCreate" ||
      FuncName == "cudaStreamCreateWithFlags" ||
      FuncName == "cudaStreamCreateWithPriority") {
    std::string ReplStr;
    auto StmtStr0 = getStmtSpelling(CE->getArg(0));
    // TODO: simplify expression
    if (StmtStr0[0] == '&')
      ReplStr = StmtStr0.substr(1);
    else
      ReplStr = "*(" + StmtStr0 + ")";

    if (DpctGlobalInfo::useNoQueueDevice()) {
      // Now the UsmLevel must not be UL_None here.
      ReplStr += " = new " + MapNames::getClNamespace() + "queue(" +
                DpctGlobalInfo::getGlobalDeviceName() + ", " +
                MapNames::getClNamespace() + "property_list{" +
                MapNames::getClNamespace() + "property::queue::in_order()";
      if (DpctGlobalInfo::getEnablepProfilingFlag()) {
        ReplStr += ", " + MapNames::getClNamespace() +
                   "property::queue::enable_profiling()";
      }
      ReplStr += "})";
    } else {
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_CurrentDevice);
      ReplStr += " = " + getNewQueue(Index);
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (IsAssigned) {
      ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
    }
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
    if (FuncName == "cudaStreamCreateWithFlags" ||
        FuncName == "cudaStreamCreateWithPriority") {
      report(CE->getBeginLoc(), Diagnostics::QUEUE_CREATED_IGNORING_OPTIONS,
             false);
    }
  } else if (FuncName == "cudaStreamDestroy") {
    auto StmtStr0 = getStmtSpelling(CE->getArg(0));
    if (isPlaceholderIdxDuplicated(CE))
      return;
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::HFT_CurrentDevice);
    auto ReplStr = "{{NEEDREPLACED" + std::to_string(Index) +
                   "}}.destroy_queue(" + StmtStr0 + ")";
    requestFeature(HelperFeatureEnum::device_ext);
    if (IsAssigned) {
      ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
    }
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaStreamSynchronize" ||
             FuncName == "cuStreamSynchronize") {
    auto StmtStr = getStmtSpelling(CE->getArg(0));
    std::string ReplStr;
    if (StmtStr == "0" || StmtStr == "cudaStreamDefault" ||
        StmtStr == "cudaStreamPerThread" || StmtStr == "cudaStreamLegacy") {
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
      ReplStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.";
    } else {
      ReplStr = StmtStr + "->";
    }
    ReplStr += "wait()";
    const std::string Name =
        CE->getCalleeDecl()->getAsFunction()->getNameAsString();
    if (IsAssigned) {
      ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaStreamGetFlags" ||
             FuncName == "cudaStreamGetPriority") {
    report(CE->getBeginLoc(), Diagnostics::STREAM_FLAG_PRIORITY_NOT_SUPPORTED,
           false);
    auto StmtStr1 = getStmtSpelling(CE->getArg(1));
    std::string ReplStr{"*("};
    ReplStr += StmtStr1;
    ReplStr += ") = 0";
    if (IsAssigned) {
      ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    const std::string Name =
        CE->getCalleeDecl()->getAsFunction()->getNameAsString();
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaDeviceGetStreamPriorityRange") {
    report(CE->getBeginLoc(), Diagnostics::STREAM_FLAG_PRIORITY_NOT_SUPPORTED,
           false);
    auto StmtStr0 = getStmtSpelling(CE->getArg(0));
    auto StmtStr1 = getStmtSpelling(CE->getArg(1));
    std::string ReplStr{"*("};
    ReplStr += StmtStr0;
    ReplStr += ") = 0, *(";
    ReplStr += StmtStr1;
    ReplStr += ") = 0";
    if (IsAssigned) {
      ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    const std::string Name =
        CE->getCalleeDecl()->getAsFunction()->getNameAsString();
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaStreamAttachMemAsync" ||
             FuncName == "cudaStreamBeginCapture" ||
             FuncName == "cudaStreamEndCapture" ||
             FuncName == "cudaStreamIsCapturing" ||
             FuncName == "cudaStreamQuery") {
    auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
  } else if (FuncName == "cudaStreamWaitEvent" ||
             FuncName == "cuStreamWaitEvent") {
    std::string ReplStr;
    auto StmtStr1 = getStmtSpelling(CE->getArg(1));
    if (!DpctGlobalInfo::useEnqueueBarrier()) {
      // ext_oneapi_submit_barrier is specified in the value of option
      // --no-dpcpp-extensions.
      ReplStr = StmtStr1 + "->wait()";
    } else {
      StmtStr1 = "*" + StmtStr1;
      auto StreamArg = CE->getArg(0);
      bool IsDefaultStream = isDefaultStream(StreamArg);
      std::string StmtStr0;
      if (IsDefaultStream) {
        if (isPlaceholderIdxDuplicated(CE))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);

        StmtStr0 = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.";
      } else {
        StmtStr0 = getStmtSpelling(CE->getArg(0)) + "->";
      }
      ReplStr = StmtStr0 + "ext_oneapi_submit_barrier({" +
                StmtStr1 + "})";
    }
    if (IsAssigned) {
      ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "cudaStreamAddCallback") {
    auto StmtStr0 = getStmtSpelling(CE->getArg(0));
    auto StmtStr1 = getStmtSpelling(CE->getArg(1));
    auto StmtStr2 = getStmtSpelling(CE->getArg(2));
    std::string ReplStr{"std::async([&]() { "};
    ReplStr += StmtStr0;
    ReplStr += "->wait(); ";
    ReplStr += StmtStr1;
    ReplStr += "(";
    ReplStr += StmtStr0;
    ReplStr += ", 0, ";
    ReplStr += StmtStr2;
    ReplStr += "); ";
    ReplStr += "})";
    if (IsAssigned) {
      ReplStr = "DPCT_CHECK_ERROR(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
    DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), HT_Future);
  } else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected function name: " << FuncName;
    return;
  }
}

REGISTER_RULE(StreamAPICallRule, PassKind::PK_Migration)

// kernel call information collection
void KernelCallRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      cudaKernelCallExpr(hasAncestor(functionDecl().bind("callContext")))
          .bind("kernelCall"),
      this);

  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName("cudaLaunchKernel",
                                              "cudaLaunchCooperativeKernel"))))
          .bind("launch"),
      this);
}

void KernelCallRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto KCall =
          getAssistNodeAsType<CUDAKernelCallExpr>(Result, "kernelCall")) {
    auto FD = getAssistNodeAsType<FunctionDecl>(Result, "callContext");
    if (!FD)
      return;
    const auto &SM = (*Result.Context).getSourceManager();

    if (SM.isMacroArgExpansion(KCall->getCallee()->getBeginLoc())) {
      // Report warning message
      report(KCall->getBeginLoc(), Diagnostics::KERNEL_CALLEE_MACRO_ARG, false);
    }

    // Remove KCall in the original location
    auto KCallSpellingRange = getTheLastCompleteImmediateRange(
        KCall->getBeginLoc(), KCall->getEndLoc());
    auto KCallLen = SM.getCharacterData(KCallSpellingRange.second) -
                    SM.getCharacterData(KCallSpellingRange.first) +
                    Lexer::MeasureTokenLength(KCallSpellingRange.second, SM,
                                              Result.Context->getLangOpts());
    emplaceTransformation(
        new ReplaceText(KCallSpellingRange.first, KCallLen, ""));
    removeTrailingSemicolon(KCall, Result);

    bool Flag = true;
    unsigned int IndentLen = calculateIndentWidth(
        KCall, SM.getExpansionLoc(KCall->getBeginLoc()), Flag);
    if (Flag)
      DpctGlobalInfo::insertKCIndentWidth(IndentLen);

    // Add kernel call to map,
    // will do code generation in Global.buildReplacements();
    if (!FD->isTemplateInstantiation()){
      DpctGlobalInfo::getInstance().insertKernelCallExpr(KCall);
    }
    const CallExpr *Config = KCall->getConfig();
    if (Config) {
      if (Config->getNumArgs() > 2) {
        const Expr *SharedMemSize = Config->getArg(2);
        if (containSizeOfType(SharedMemSize)) {
          auto KCallInfo =
              DpctGlobalInfo::getInstance().insertKernelCallExpr(KCall);
          KCallInfo->setEmitSizeofWarningFlag(true);
        } else {
          const Expr *ExprContainSizeofType = nullptr;
          if (checkIfContainSizeofTypeRecursively(SharedMemSize,
                                                  ExprContainSizeofType)) {
            if (ExprContainSizeofType) {
              report(ExprContainSizeofType->getBeginLoc(),
                     Diagnostics::SIZEOF_WARNING, false, "local memory");
            }
          }
        }
      }
    }

    // Filter out compiler generated methods
    if (const CXXMethodDecl *CXXMDecl = dyn_cast<CXXMethodDecl>(FD)) {
      if (!CXXMDecl->isUserProvided()) {
        return;
      }
    }

    auto BodySLoc = FD->getBody()->getSourceRange().getBegin().getRawEncoding();
    if (Insertions.find(BodySLoc) != Insertions.end())
      return;

    Insertions.insert(BodySLoc);
  } else if (auto LaunchKernelCall =
                 getNodeAsType<CallExpr>(Result, "launch")) {
    if (DpctGlobalInfo::getInstance().buildLaunchKernelInfo(LaunchKernelCall)) {
      emplaceTransformation(new ReplaceStmt(LaunchKernelCall, true, false, ""));
      removeTrailingSemicolon(LaunchKernelCall, Result);
    } else {
      auto FuncName = LaunchKernelCall->getDirectCallee()
                          ->getNameInfo()
                          .getName()
                          .getAsString();
      report(LaunchKernelCall->getBeginLoc(), Diagnostics::API_NOT_MIGRATED,
             false, FuncName);
    }
  }
}

// Find and remove the semicolon after the kernel call
void KernelCallRule::removeTrailingSemicolon(
    const CallExpr *KCall,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto &SM = (*Result.Context).getSourceManager();
  auto KELoc = getTheLastCompleteImmediateRange(KCall->getBeginLoc(), KCall->getEndLoc()).second;
  auto Tok = Lexer::findNextToken(KELoc, SM, LangOptions()).value();
  if (Tok.is(tok::TokenKind::semi))
    emplaceTransformation(new ReplaceToken(Tok.getLocation(), ""));
}

REGISTER_RULE(KernelCallRule, PassKind::PK_Analysis)


bool isRecursiveDeviceFuncDecl(const FunctionDecl* FD) {
  // Build call graph for FunctionDecl and look for cycles in call graph.
  // Emit the warning message when the recursive call exists in kernel function.
  if (!FD) return false;
  CallGraph CG;
  CG.addToCallGraph(const_cast<FunctionDecl *>(FD));
  bool FDIsRecursive = false;
  for (llvm::scc_iterator<CallGraph *> SCCI = llvm::scc_begin(&CG),
                              SCCE = llvm::scc_end(&CG);
                              SCCI != SCCE; ++SCCI) {
    if (SCCI.hasCycle()) FDIsRecursive = true;
  }
  return FDIsRecursive;
}

bool isRecursiveDeviceCallExpr(const CallExpr* CE) {
  if (isRecursiveDeviceFuncDecl(CE->getDirectCallee()))
    return true;
  return false;
}

static void checkCallGroupFunctionInControlFlow(FunctionDecl *FD) {
  GroupFunctionCallInControlFlowAnalyzer A(DpctGlobalInfo::getContext());
  (void)A.checkCallGroupFunctionInControlFlow(FD);
}

// __device__ function call information collection
void DeviceFunctionDeclRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto DeviceFunctionMatcher =
      functionDecl(anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))
          .bind("funcDecl");

  MF.addMatcher(callExpr(hasAncestor(DeviceFunctionMatcher)).bind("callExpr"),
                this);
  MF.addMatcher(
      cxxConstructExpr(hasAncestor(DeviceFunctionMatcher)).bind("CtorExpr"),
      this);

  MF.addMatcher(DeviceFunctionMatcher, this);

  MF.addMatcher(callExpr(hasAncestor(DeviceFunctionMatcher),
                         callee(functionDecl(hasName("printf"))))
                    .bind("PrintfExpr"),
                this);

  MF.addMatcher(varDecl(hasAncestor(DeviceFunctionMatcher)).bind("varGrid"),
                this);

  MF.addMatcher(
      functionDecl(anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))
          .bind("deviceFuncDecl"),
      this);

  MF.addMatcher(
      cxxNewExpr(hasAncestor(DeviceFunctionMatcher)).bind("CxxNewExpr"), this);
  MF.addMatcher(
      cxxDeleteExpr(hasAncestor(DeviceFunctionMatcher)).bind("CxxDeleteExpr"),
      this);
  MF.addMatcher(callExpr(hasAncestor(DeviceFunctionMatcher),
                         callee(functionDecl(hasAnyName(
                             "malloc", "free", "delete", "__builtin_alloca"))))
                    .bind("MemoryManipulation"),
                this);

  MF.addMatcher(typeLoc(hasAncestor(DeviceFunctionMatcher),
                        loc(qualType(hasDeclaration(namedDecl(hasAnyName(
                            "__half", "half", "__half2", "half2"))))))
                    .bind("fp16"),
                this);

  MF.addMatcher(
      typeLoc(hasAncestor(DeviceFunctionMatcher), loc(asString("double")))
          .bind("fp64"),
      this);
}

void DeviceFunctionDeclRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {

  if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "deviceFuncDecl")) {
    if (FD->isTemplateInstantiation())
      return;

    if (FD->hasAttr<CUDADeviceAttr>() &&
        FD->getAttr<CUDADeviceAttr>()->isImplicit())
      return;

    const auto &FTL = FD->getFunctionTypeLoc();
    if (!FTL)
      return;

    auto BeginLoc = FD->getBeginLoc();
    if (auto DFT = FD->getDescribedFunctionTemplate())
      BeginLoc = DFT->getBeginLoc();
    auto EndLoc = FTL.getRParenLoc();

    auto BeginLocInfo = DpctGlobalInfo::getLocInfo(BeginLoc);
    auto EndLocInfo = DpctGlobalInfo::getLocInfo(EndLoc);
    auto FileInfo =
        DpctGlobalInfo::getInstance().insertFile(BeginLocInfo.first);
    auto &Map = FileInfo->getFuncDeclRangeMap();
    auto Name = FD->getNameAsString();
    auto Iter = Map.find(Name);
    if (Iter == Map.end()) {
      std::vector<std::pair<unsigned int, unsigned int>> Vec;
      Vec.push_back(std::make_pair(BeginLocInfo.second, EndLocInfo.second));
      Map[Name] = Vec;
    } else {
      Iter->second.push_back(
          std::make_pair(BeginLocInfo.second, EndLocInfo.second));
    }
  }

  std::shared_ptr<DeviceFunctionInfo> FuncInfo;
  auto FD = getAssistNodeAsType<FunctionDecl>(Result, "funcDecl");
  if (!FD || (FD->hasAttr<CUDADeviceAttr>() && FD->hasAttr<CUDAHostAttr>() &&
              DpctGlobalInfo::getRunRound() == 1))
    return;

  if (FD->hasAttr<CUDADeviceAttr>() &&
      FD->getAttr<CUDADeviceAttr>()->isImplicit())
    return;

  if (FD->isVariadic()) {
    report(FD->getBeginLoc(), Warnings::DEVICE_VARIADIC_FUNCTION, false);
  }

  if (FD->isVirtualAsWritten()) {
    report(FD->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION,
                              false, "Virtual functions");
  }

  if(isRecursiveDeviceFuncDecl(FD))
    report(FD->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION,
                            false, "Recursive functions");

  FuncInfo = DeviceFunctionDecl::LinkRedecls(FD);
  if (!FuncInfo)
    return;

  if (FD->doesThisDeclarationHaveABody()) {
    size_t ParamCounter = 0;
    for (auto &Param : FD->parameters()) {
      FuncInfo->setParameterReferencedStatus(ParamCounter,
                                             Param->isReferenced());
      ParamCounter++;
    }
    auto &SM = DpctGlobalInfo::getSourceManager();
    auto &CTX = DpctGlobalInfo::getContext();
    size_t LocalVariableSize = 0;
    for(auto D : FD->decls()){
      if(auto VD = dyn_cast_or_null<VarDecl>(D)) {
        if(VD->hasAttr<CUDASharedAttr>() || !VD->isLocalVarDecl() || isCubVar(VD)){
          continue;
        }
        auto Size = CTX.getTypeSizeInCharsIfKnown(VD->getType());
        if(Size.has_value()) {
          LocalVariableSize += Size.value().getQuantity();
        }
      }
    }
    // For Xe-LP architecture, if the sub-group size is 32, then each work-item
    // can use 128 * 32 Byte / 32 = 128 Byte registers.
    if(LocalVariableSize > 128) {
      report(SM.getExpansionLoc(FD->getBeginLoc()), Warnings::REGISTER_USAGE,
             false, FD->getDeclName().getAsString());
    }
  }
  if (isLambda(FD) && !FuncInfo->isLambda()) {
    FuncInfo->setLambda();
  }
  if (FD->hasAttr<CUDAGlobalAttr>()) {
    FuncInfo->setKernel();
  }
  if (FD->isInlined()) {
    FuncInfo->setInlined();
  }
  if (auto CE = getAssistNodeAsType<CallExpr>(Result, "callExpr")) {
    if (CE->getDirectCallee()) {
      if (CE->getDirectCallee()->isVirtualAsWritten())
        report(CE->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION,
                        false, "Virtual functions");
    }

    if (isRecursiveDeviceCallExpr(CE))
      report(CE->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION,
                                false, "Recursive functions");
    auto CallInfo = FuncInfo->addCallee(CE);
    checkCallGroupFunctionInControlFlow(const_cast<FunctionDecl *>(FD));
    if (CallInfo->hasSideEffects())
      report(CE->getBeginLoc(), Diagnostics::CALL_GROUP_FUNC_IN_COND, false);
  } else if (CE = getAssistNodeAsType<CallExpr>(Result, "PrintfExpr")) {
    if (FD->hasAttr<CUDAHostAttr>()) {
      report(CE->getBeginLoc(), Warnings::PRINTF_FUNC_NOT_SUPPORT, false);
      return;
    }
    std::string ReplacedStmt;
    llvm::raw_string_ostream OS(ReplacedStmt);
    OS << DpctGlobalInfo::getStreamName() << " << ";
    CE->getArg(0)->printPretty(OS, nullptr,
                               Result.Context->getPrintingPolicy());
    emplaceTransformation(new ReplaceStmt(CE, std::move(OS.str())));
    if (CE->getNumArgs() > 1 ||
        CE->getArg(0)->IgnoreImplicitAsWritten()->getStmtClass() !=
            Stmt::StringLiteralClass)
      report(CE->getBeginLoc(), Warnings::PRINTF_FUNC_MIGRATION_WARNING, false);
    FuncInfo->setStream();
  } else if (auto Ctor =
                 getAssistNodeAsType<CXXConstructExpr>(Result, "CtorExpr")) {
    FuncInfo->addCallee(Ctor);
  }

  if (auto CXX = getAssistNodeAsType<CXXNewExpr>(Result, "CxxNewExpr")) {
    report(CXX->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION,
           false, "The usage of dynamic memory allocation and deallocation APIs");
  }

  if (auto CXX = getAssistNodeAsType<CXXDeleteExpr>(Result, "CxxDeleteExpr")) {
    report(CXX->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION,
           false, "The usage of dynamic memory allocation and deallocation APIs");
  }

  if (auto CE = getAssistNodeAsType<CallExpr>(Result, "MemoryManipulation")) {
    report(CE->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION, false,
           "The usage of dynamic memory allocation and deallocation APIs");
  }

  if (auto Var = getAssistNodeAsType<VarDecl>(Result, "varGrid")) {

    if (!Var->getInit())
      return;

    if (auto CE =
            dyn_cast<CallExpr>(Var->getInit()->IgnoreUnlessSpelledInSource())) {
      if (CE->getType().getCanonicalType().getAsString() !=
          "class cooperative_groups::__v1::grid_group")
        return;

      if (!DpctGlobalInfo::useNdRangeBarrier()) {
        auto Name = Var->getNameAsString();
        report(Var->getBeginLoc(), Diagnostics::ND_RANGE_BARRIER, false,
               "this_grid()");
        return;
      }

      FuncInfo->setSync();
      auto Begin = Var->getBeginLoc();
      auto End = Var->getEndLoc();
      const auto &SM = *Result.SourceManager;

      End = End.getLocWithOffset(Lexer::MeasureTokenLength(
          End, SM, dpct::DpctGlobalInfo::getContext().getLangOpts()));

      Token Tok;
      Tok = Lexer::findNextToken(
                End, SM, dpct::DpctGlobalInfo::getContext().getLangOpts())
                .value();
      End = Tok.getLocation();

      auto Length = SM.getFileOffset(End) - SM.getFileOffset(Begin);

      // Remove statement "cg::grid_group grid = cg::this_grid();"
      emplaceTransformation(new ReplaceText(Begin, Length, ""));
    }
  }
  if (getAssistNodeAsType<TypeLoc>(Result, "fp64")) {
    FuncInfo->setBF64();
  }
  if (getAssistNodeAsType<TypeLoc>(Result, "fp16")) {
    FuncInfo->setBF16();
  }
}

REGISTER_RULE(DeviceFunctionDeclRule, PassKind::PK_Analysis)

/// __constant__/__shared__/__device__ var information collection
void MemVarRule::registerMatcher(MatchFinder &MF) {
  auto DeclMatcher =
      varDecl(anyOf(hasAttr(attr::CUDAConstant), hasAttr(attr::CUDADevice),
                    hasAttr(attr::CUDAShared), hasAttr(attr::HIPManaged)),
              unless(hasAnyName("threadIdx", "blockDim", "blockIdx", "gridDim",
                                "warpSize")));
  MF.addMatcher(DeclMatcher.bind("var"), this);
  MF.addMatcher(
      declRefExpr(anyOf(hasParent(implicitCastExpr(
                                      unless(hasParent(arraySubscriptExpr())))
                                      .bind("impl")),
                        anything()),
                  to(DeclMatcher.bind("decl")),
                  hasAncestor(functionDecl().bind("func")))
          .bind("used"),
      this);

  MF.addMatcher(varDecl(hasParent(translationUnitDecl())).bind("hostGlobalVar"),
                this);
}

void MemVarRule::previousHCurrentD(const VarDecl *VD, tooling::Replacement &R) {
  // 1. emit DPCT1055 warning
  // 2. add a new variable for host
  // 3. insert dpct::constant_memory and add the info from that replacement
  //     into current replacement.
  // 4. remove the replacement of removing "__constant__". In yaml case, clang
  //    replacement merging mechanism will occur error due to overlapping.
  //    The reason of setting offset as 0 is to avoid doing merge.
  //    4.1 About removing the replacement of removing "__constant__",
  //        e.g., the code is: static __constant__ a;
  //        the repl of removing "__constant__" and repl of replacing
  //        "static __constant__ a;" to "static dpct::constant_memory<float, 0>
  //        a;" are overlapped. And this merging is not in dpct but in clang's
  //        file (Replacement.cpp), clang's merging mechanism will occur error
  //        due to overlapping.
  //    4.2 About setting the offset equals to 0,
  //        if we keep the original offset, in clang's merging, a new merged
  //        replacement will be saved, and it will not contain the additional
  //        info we added. So we need to avoid this merge.
  // 5. remove previous DPCT1056 warning (will be handled in
  // removeHostConstantWarning)

  auto &SM = DpctGlobalInfo::getSourceManager();

  std::string HostVariableName = VD->getNameAsString() + "_host_ct1";
  report(VD->getBeginLoc(), Diagnostics::HOST_DEVICE_CONSTANT, false,
         VD->getNameAsString(), HostVariableName);

  std::string InitStr =
      VD->hasInit() ? ExprAnalysis::ref(VD->getInit()) : std::string("");
  std::string NewDecl =
      DpctGlobalInfo::getReplacedTypeName(VD->getType()) + " " +
      HostVariableName +
      (InitStr.empty() ? InitStr : std::string(" = " + InitStr)) + ";" +
      getNL() + getIndent(SM.getExpansionLoc(VD->getBeginLoc()), SM).str();
  if (VD->getStorageClass() == SC_Static)
    NewDecl = "static " + NewDecl;
  emplaceTransformation(new InsertText(SM.getExpansionLoc(VD->getBeginLoc()),
                                       std::move(NewDecl)));

  if (auto DeviceRepl = ReplaceVarDecl::getVarDeclReplacement(
          VD, MemVarInfo::buildMemVarInfo(VD)->getDeclarationReplacement(VD))) {
    DeviceRepl->setConstantFlag(dpct::ConstantFlagType::HostDevice);
    DeviceRepl->setConstantOffset(R.getConstantOffset());
    DeviceRepl->setInitStr(InitStr);
    DeviceRepl->setNewHostVarName(HostVariableName);
    emplaceTransformation(DeviceRepl);
  }

  R = tooling::Replacement(R.getFilePath(), 0, 0, "");
}

void MemVarRule::previousDCurrentH(const VarDecl *VD, tooling::Replacement &R) {
  // 1. change DeviceConstant to HostDeviceConstant
  // 2. emit DPCT1055 warning (warning info is from previous device case)
  // 3. add a new variable for host (decl info is from previous device case)

  auto &SM = DpctGlobalInfo::getSourceManager();
  R.setConstantFlag(dpct::ConstantFlagType::HostDevice);

  std::string HostVariableName = R.getNewHostVarName();
  std::string InitStr = R.getInitStr();
  std::string NewDecl =
      DpctGlobalInfo::getReplacedTypeName(VD->getType()) + " " +
      HostVariableName +
      (InitStr.empty() ? InitStr : std::string(" = " + InitStr)) + ";" +
      getNL() + getIndent(SM.getExpansionLoc(VD->getBeginLoc()), SM).str();
  if (VD->getStorageClass() == SC_Static)
    NewDecl = "static " + NewDecl;

  SourceLocation SL = SM.getComposedLoc(
      SM.getDecomposedLoc(SM.getExpansionLoc(VD->getBeginLoc())).first,
      R.getConstantOffset());
  SL = DiagnosticsUtils::getStartOfLine(
      SL, SM, DpctGlobalInfo::getContext().getLangOpts(), false);
  report(SL, Diagnostics::HOST_DEVICE_CONSTANT, false, VD->getNameAsString(),
         HostVariableName);

  emplaceTransformation(new InsertText(SL, std::move(NewDecl)));
}

void MemVarRule::removeHostConstantWarning(Replacement &R) {
  std::string ReplStr = R.getReplacementText().str();

  // warning text of Diagnostics::HOST_CONSTANT
  std::string Warning = "The use of [_a-zA-Z][_a-zA-Z0-9]+ in device "
                        "code was not detected. If this variable is also used "
                        "in device code, you need to rewrite the code.";
  std::string Pattern =
      "/\\*\\s+DPCT" +
      std::to_string(static_cast<int>(Diagnostics::HOST_CONSTANT)) +
      ":[0-9]+: " + Warning + "\\s+\\*/" + getNL();
  std::regex RE(Pattern);
  std::smatch MRes;
  std::string Result;
  std::regex_replace(std::back_inserter(Result), ReplStr.begin(), ReplStr.end(),
                     RE, "");
  R.setReplacementText(Result);
}

void MemVarRule::processTypeDeclaredLocal(const VarDecl *MemVar,
                                          std::shared_ptr<MemVarInfo> Info) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto DS = Info->getDeclStmtOfVarType();
  if (!DS)
    return;
  // this token is ';'
  auto InsertSL = SM.getExpansionLoc(DS->getEndLoc()).getLocWithOffset(1);
  auto GenDeclStmt = [=, &SM](
                         StringRef TypeName) -> std::string {
    bool IsReference = !Info->getType()->getDimension();
    std::string Ret;
    llvm::raw_string_ostream OS(Ret);
    OS << getNL() << getIndent(InsertSL, SM);
    OS << TypeName << ' ';
    if (IsReference)
      OS << '&';
    else
      OS << '*';
    OS << Info->getName();
    OS << " = ";
    if (IsReference)
      OS << '*';
    // add typecast for the __shared__ variable, since after migration the
    // __shared__ variable type will be uint8_t*
    OS << '(' << TypeName << " *)";
    OS << Info->getNameAppendSuffix() << ';';
    return OS.str();
  };
  if (Info->isAnonymousType()) {
    // keep the origin type declaration, only remove variable name
    //  }  a_variable  ,  b_variable ;
    //   |                |
    // begin             end
    // ReplaceToken replacing [begin, end]
    SourceLocation Begin =
        SM.getExpansionLoc(Info->getDeclOfVarType()->getBraceRange().getEnd());
    Begin = Begin.getLocWithOffset(1); // this token is }
    SourceLocation End = SM.getExpansionLoc(MemVar->getEndLoc());
    emplaceTransformation(new ReplaceToken(Begin, End, ""));

    std::string NewTypeName = Info->getLocalTypeName();

    // add a typename
    emplaceTransformation(new InsertText(
        SM.getExpansionLoc(
            Info->getDeclOfVarType()->getBraceRange().getBegin()),
        " " + NewTypeName));

    // add typecast for the __shared__ variable, since after migration the
    // __shared__ variable type will be uint8_t*
    emplaceTransformation(new InsertText(InsertSL, GenDeclStmt(NewTypeName)));
  } else if (DS) {
    // remove var decl
    emplaceTransformation(ReplaceVarDecl::getVarDeclReplacement(
        MemVar, Info->getDeclarationReplacement(MemVar)));

    Info->setLocalTypeName(Info->getType()->getBaseName());
    emplaceTransformation(
        new InsertText(InsertSL, GenDeclStmt(Info->getType()->getBaseName())));
  }
}

bool MemVarRule::currentIsDevice(const VarDecl *MemVar,
                                 std::shared_ptr<MemVarInfo> Info) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto BeginLoc = SM.getExpansionLoc(MemVar->getBeginLoc());
  auto OffsetOfLineBegin = getOffsetOfLineBegin(BeginLoc, SM);
  auto BeginLocInfo = DpctGlobalInfo::getLocInfo(BeginLoc);
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(BeginLocInfo.first);
  auto &S = FileInfo->getConstantMacroTMSet();
  for (auto &TM : S) {
    if (TM == nullptr)
      continue;
    if ((TM->getConstantFlag() == dpct::ConstantFlagType::Device ||
         TM->getConstantFlag() == dpct::ConstantFlagType::HostDeviceInOnePass) &&
        TM->getLineBeginOffset() == OffsetOfLineBegin) {
      TM->setIgnoreTM(true);
      // current __constant__ variable used in device, using
      // OffsetOfLineBegin link the R(reomving __constant__) and
      // R(dcpt::constant_memery):
      // 1. check previous processed replacements, if found, do not check
      // info from yaml
      if (!FileInfo->getRepls())
        return false;
      auto &M = FileInfo->getRepls()->getReplMap();
      bool RemoveWarning = false;
      for (auto &R : M) {
        if ((R.second->getConstantFlag() == dpct::ConstantFlagType::Host ||
             R.second->getConstantFlag() == dpct::ConstantFlagType::HostDeviceInOnePass) &&
            R.second->getConstantOffset() == TM->getConstantOffset()) {
          // using flag and the offset of __constant__ to link
          // R(dcpt::constant_memery)  and R(reomving __constant__) from
          // previous execution, previous is host, current is device:
          previousHCurrentD(MemVar, *(R.second));
          dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(
              MemVar->getNameAsString());
          RemoveWarning = true;
          break;
        } else if ((R.second->getConstantFlag() ==
                        dpct::ConstantFlagType::Device ||
                    R.second->getConstantFlag() ==
                        dpct::ConstantFlagType::HostDevice) &&
                   R.second->getConstantOffset() == TM->getConstantOffset()) {
          TM->setIgnoreTM(true);
          return true;
        }
      }
      if (RemoveWarning) {
        for (auto &R : M) {
          if (R.second->getConstantOffset() == TM->getConstantOffset()) {
            removeHostConstantWarning(*(R.second));
            TM->setIgnoreTM(true);
            return true;
          }
        }
        TM->setIgnoreTM(true);
        return true;
      }

      // 2. if no info found, check info from yaml
      if (FileInfo->PreviousTUReplFromYAML) {
        auto &ReplsFromYAML = FileInfo->getReplacements();
        for (auto &R : ReplsFromYAML) {
          if (R.getConstantFlag() == dpct::ConstantFlagType::Host &&
              R.getConstantOffset() == TM->getConstantOffset()) {
            // using flag and the offset of __constant__ to link
            // R(dcpt::constant_memery) and R(reomving __constant__) from
            // previous execution previous is host, current is device:
            previousHCurrentD(MemVar, R);
            dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(
                MemVar->getNameAsString());
            RemoveWarning = true;
            break;
          } else if ((R.getConstantFlag() == dpct::ConstantFlagType::Device ||
                      R.getConstantFlag() ==
                          dpct::ConstantFlagType::HostDevice) &&
                     R.getConstantOffset() == TM->getConstantOffset()) {
            TM->setIgnoreTM(true);
            return true;
          }
        }
        if (RemoveWarning) {
          for (auto &R : ReplsFromYAML) {
            if (R.getConstantOffset() == TM->getConstantOffset()) {
              removeHostConstantWarning(R);
              TM->setIgnoreTM(true);
              return true;
            }
          }
          TM->setIgnoreTM(true);
          return true;
        }
      }
      if (Info->getType()->getDimension() > 3 &&
          DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
        report(MemVar->getBeginLoc(), Diagnostics::EXCEED_MAX_DIMENSION, true);
      }
      // Code here means this is the first migration, need save info to
      // replacement
      Info->setIgnoreFlag(false);
      TM->setIgnoreTM(true);
      auto ReplaceStr = Info->getDeclarationReplacement(MemVar);
      auto SourceFileType = GetSourceFileType(Info->getFilePath());
      if ((SourceFileType == SPT_CudaHeader ||
           SourceFileType == SPT_CppHeader) &&
          !Info->isStatic()) {
        ReplaceStr = "inline " + ReplaceStr;
      }
      auto RVD =
          ReplaceVarDecl::getVarDeclReplacement(MemVar, std::move(ReplaceStr));
      if (!RVD)
        return true;
      RVD->setConstantFlag(TM->getConstantFlag());
      RVD->setConstantOffset(TM->getConstantOffset());
      RVD->setInitStr(MemVar->hasInit() ? ExprAnalysis::ref(MemVar->getInit())
                                        : std::string(""));
      RVD->setNewHostVarName(MemVar->getNameAsString() + "_host_ct1");
      emplaceTransformation(RVD);
      return true;
    }
  }
  return false;
}

bool MemVarRule::currentIsHost(const VarDecl *VD, std::string VarName) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto BeginLoc = SM.getExpansionLoc(VD->getBeginLoc());
  auto OffsetOfLineBegin = getOffsetOfLineBegin(BeginLoc, SM);
  auto BeginLocInfo = DpctGlobalInfo::getLocInfo(BeginLoc);
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(BeginLocInfo.first);
  auto &S = FileInfo->getConstantMacroTMSet();
  for (auto &TM : S) {
    if (TM == nullptr)
      continue;
    if ((TM->getConstantFlag() == dpct::ConstantFlagType::Host ||
         TM->getConstantFlag() == dpct::ConstantFlagType::HostDeviceInOnePass) &&
        TM->getLineBeginOffset() == OffsetOfLineBegin) {
      // current __constant__ variable used in host, using OffsetOfLineBegin
      // link the R(reomving __constant__) and here

      // 1. check previous processed replacements, if found, do not check
      // info from yaml
      if (!FileInfo->getRepls())
        return false;
      auto &M = FileInfo->getRepls()->getReplMap();
      for (auto &R : M) {
        if ((R.second->getConstantFlag() == dpct::ConstantFlagType::Device ||
             R.second->getConstantFlag() == dpct::ConstantFlagType::HostDeviceInOnePass) &&
            R.second->getConstantOffset() == TM->getConstantOffset()) {
          // using flag and the offset of __constant__ to link previous
          // execution of previous is device, current is host:
          previousDCurrentH(VD, *(R.second));
          dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(VarName);
          TM->setIgnoreTM(true);
          return true;
        } else if ((R.second->getConstantFlag() ==
                        dpct::ConstantFlagType::Host ||
                    R.second->getConstantFlag() ==
                        dpct::ConstantFlagType::HostDevice) &&
                   R.second->getConstantOffset() == TM->getConstantOffset()) {
          if (R.second->getConstantFlag() == dpct::ConstantFlagType::HostDevice)
            dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(VarName);
          TM->setIgnoreTM(true);
          return true;
        }
      }

      // 2. if no info found, check info from yaml
      if (FileInfo->PreviousTUReplFromYAML) {
        auto &ReplsFromYAML = FileInfo->getReplacements();
        for (auto &R : ReplsFromYAML) {
          if (R.getConstantFlag() == dpct::ConstantFlagType::Device &&
              R.getConstantOffset() == TM->getConstantOffset()) {
            // using flag and the offset of __constant__ to link here and
            // R(reomving __constant__) from previous execution, previous is
            // device, current is host.
            previousDCurrentH(VD, R);
            dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(VarName);
            TM->setIgnoreTM(true);
            return true;
          } else if ((R.getConstantFlag() == dpct::ConstantFlagType::Host ||
                      R.getConstantFlag() ==
                          dpct::ConstantFlagType::HostDevice) &&
                     R.getConstantOffset() == TM->getConstantOffset()) {
            if (R.getConstantFlag() == dpct::ConstantFlagType::HostDevice)
              dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(VarName);
            TM->setIgnoreTM(true);
            return true;
          }
        }
      }

      // Code here means this is the first migration, only emit a warning
      // Add the constant offset in the replacement
      // The constant offset will be used in previousHCurrentD to distinguish
      // unnecessary warnings.
      if (report(VD->getBeginLoc(), Diagnostics::HOST_CONSTANT, false,
                 VD->getNameAsString())) {
        TransformSet->back()->setConstantOffset(TM->getConstantOffset());
      }
    }
  }
  return false;
}

void MemVarRule::runRule(const MatchFinder::MatchResult &Result) {
  auto getRHSOfTheNonConstAssignedVar =
      [](const DeclRefExpr *DRE) -> const Expr * {
    auto isExpectedRHS = [](const Expr *E, DynTypedNode Current,
                            QualType QT) -> bool {
      return (E == Current.get<Expr>()) && QT->isPointerType() &&
             !QT->getPointeeType().isConstQualified();
    };

    auto &Context = DpctGlobalInfo::getContext();
    DynTypedNode Current = DynTypedNode::create(*DRE);
    DynTypedNodeList Parents = Context.getParents(Current);
    while (!Parents.empty()) {
      const BinaryOperator *BO = Parents[0].get<BinaryOperator>();
      const VarDecl *VD = Parents[0].get<VarDecl>();
      if (BO) {
        if (BO->isAssignmentOp() &&
            isExpectedRHS(BO->getRHS(), Current, BO->getLHS()->getType()))
          return BO->getRHS();
      } else if (VD) {
        if (VD->hasInit() &&
            isExpectedRHS(VD->getInit(), Current, VD->getType()))
          return VD->getInit();
        return nullptr;
      }
      Current = Parents[0];
      Parents = Context.getParents(Current);
    }
    return nullptr;
  };

  std::string CanonicalType;
  if (auto MemVar = getAssistNodeAsType<VarDecl>(Result, "var")) {
    if (isCubVar(MemVar)) {
      return;
    }
    CanonicalType = MemVar->getType().getCanonicalType().getAsString();
    if (CanonicalType.find("block_tile_memory") != std::string::npos) {
      emplaceTransformation(new ReplaceVarDecl(MemVar, ""));
      return;
    }
    auto Info = MemVarInfo::buildMemVarInfo(MemVar);
    if (!Info)
      return;

    if (Info->isTypeDeclaredLocal()) {
      processTypeDeclaredLocal(MemVar, Info);
    } else {
      // This IgnoreFlag is used to disable the replacement of
      // "dpct::constant_memory<T, 0> a;"
      // The reason is in previous migration executions, this variable has been
      // migrated and saved the replacement in yaml. That replacement maybe has
      // both host and device replacements, which is different from current
      // replacement (only device). We want to use the Repl in yaml. so ignore
      // this replacement.
      Info->setIgnoreFlag(true);
      if (MemVar->hasAttr<CUDAConstantAttr>()) {
        if (currentIsDevice(MemVar, Info))
          return;
      }

      Info->setIgnoreFlag(false);

      if (!Info->isShared() && Info->getType()->getDimension() > 3 &&
          DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
        report(MemVar->getBeginLoc(), Diagnostics::EXCEED_MAX_DIMENSION, true);
      }
      emplaceTransformation(ReplaceVarDecl::getVarDeclReplacement(
          MemVar, Info->getDeclarationReplacement(MemVar)));
    }
    return;
  }
  auto MemVarRef = getNodeAsType<DeclRefExpr>(Result, "used");
  auto Func = getAssistNodeAsType<FunctionDecl>(Result, "func");
  auto Decl = getAssistNodeAsType<VarDecl>(Result, "decl");
  DpctGlobalInfo &Global = DpctGlobalInfo::getInstance();
  if (MemVarRef && Func && Decl) {
    if (isCubVar(Decl)) {
      return;
    }
    auto GetReplRange =
        [&](const Stmt *ReplaceNode) -> std::pair<SourceLocation, unsigned> {
      auto Range = getDefinitionRange(ReplaceNode->getBeginLoc(),
                                      ReplaceNode->getEndLoc());
      auto &SM = DpctGlobalInfo::getSourceManager();
      auto Begin = Range.getBegin();
      auto End = Range.getEnd();
      auto Length = Lexer::MeasureTokenLength(
          End, SM, dpct::DpctGlobalInfo::getContext().getLangOpts());
      Length +=
          SM.getDecomposedLoc(End).second - SM.getDecomposedLoc(Begin).second;
      return std::make_pair(Begin, Length);
    };
    const auto *Parent = getParentStmt(MemVarRef);
    bool HasTypeCasted = false;
    // 1. Handle assigning a 2 or more dimensions array pointer to a variable.
    if (const auto *const ICE = dyn_cast_or_null<ImplicitCastExpr>(Parent)) {
      if (const auto *arrType = MemVarRef->getType()->getAsArrayTypeUnsafe()) {
        if (ICE->getCastKind() == CK_ArrayToPointerDecay &&
            arrType->getElementType()->isArrayType() &&
            isAssignOperator(getParentStmt(Parent))) {
          auto Range = GetReplRange(MemVarRef);
          emplaceTransformation(
              new ReplaceText(Range.first, Range.second,
                              buildString("(", ICE->getType(), ")",
                                          Decl->getName(), ".get_ptr()")));
          HasTypeCasted = true;
        }
      }
    }
    // 2. Handle address-of operation.
    else if (const UnaryOperator *UO =
                 dyn_cast_or_null<UnaryOperator>(Parent)) {
      if (!Decl->hasAttr<CUDASharedAttr>() && UO->getOpcode() == UO_AddrOf) {
        CtTypeInfo TypeAnalysis(Decl, false);
        auto Range = GetReplRange(UO);
        if (TypeAnalysis.getDimension() >= 2) {
          // Dim >= 2
          emplaceTransformation(new ReplaceText(
              Range.first, Range.second,
              buildString("reinterpret_cast<", UO->getType(), ">(",
                          Decl->getName(), ".get_ptr())")));
          HasTypeCasted = true;
        } else if (TypeAnalysis.getDimension() == 1) {
          // Dim == 1
          emplaceTransformation(
              new ReplaceText(Range.first, Range.second,
                              buildString("reinterpret_cast<", UO->getType(),
                                          ">(&", Decl->getName(), ")")));
          HasTypeCasted = true;
        } else {
          // Dim == 0
          if (Decl->hasAttr<CUDAConstantAttr>() &&
              (MemVarRef->getType()->getTypeClass() !=
               Type::TypeClass::Elaborated)) {
            const Expr *RHS = getRHSOfTheNonConstAssignedVar(MemVarRef);
            if (RHS) {
              auto Range = GetReplRange(RHS);
              emplaceTransformation(new ReplaceText(
                  Range.first, Range.second,
                  buildString("const_cast<", RHS->getType(), ">(",
                              ExprAnalysis::ref(RHS), ")")));
              HasTypeCasted = true;
            }
          }
        }
      }
    }
    if (!HasTypeCasted && Decl->hasAttr<CUDAConstantAttr>() &&
        (MemVarRef->getType()->getTypeClass() ==
         Type::TypeClass::ConstantArray)) {
      const Expr *RHS = getRHSOfTheNonConstAssignedVar(MemVarRef);
      if (RHS) {
        auto Range = GetReplRange(RHS);
        emplaceTransformation(
            new ReplaceText(Range.first, Range.second,
                            buildString("const_cast<", RHS->getType(), ">(",
                                        ExprAnalysis::ref(RHS), ")")));
      }
    }
    auto VD = dyn_cast<VarDecl>(MemVarRef->getDecl());
    if (Func->isImplicit() ||
        Func->getTemplateSpecializationKind() == TSK_ImplicitInstantiation)
      return;
    if (VD == nullptr)
      return;

    auto Var = Global.findMemVarInfo(VD);
    if (Func->hasAttr<CUDAGlobalAttr>() || Func->hasAttr<CUDADeviceAttr>()) {
      if (DpctGlobalInfo::useGroupLocalMemory() &&
          VD->hasAttr<CUDASharedAttr>() && VD->getStorageClass() != SC_Extern) {
        if (!Var)
          return;
        if (auto B = dyn_cast_or_null<CompoundStmt>(Func->getBody())) {
          if (B->body_empty())
            return;
          emplaceTransformation(new InsertBeforeStmt(
              B->body_front(), Var->getDeclarationReplacement(VD)));
          return;
        }
      } else {
        if (Var) {
          DeviceFunctionDecl::LinkRedecls(Func)->addVar(Var);
        }
      }
    } else {
      if (Var && !VD->getType()->isArrayType() &&
          VD->hasAttr<HIPManagedAttr>()) {
        emplaceTransformation(new InsertAfterStmt(MemVarRef, "[0]"));
      }
    }
  }

  if (auto VD = getNodeAsType<VarDecl>(Result, "hostGlobalVar")) {
    auto VarName = VD->getNameAsString();
    bool IsHost =
        !(VD->hasAttr<CUDAConstantAttr>() || VD->hasAttr<CUDADeviceAttr>() ||
          VD->hasAttr<CUDASharedAttr>() || VD->hasAttr<HIPManagedAttr>());
    if (IsHost) {
      dpct::DpctGlobalInfo::getGlobalVarNameSet().insert(VarName);

      if (currentIsHost(VD, VarName))
        return;
    }
  }
}

REGISTER_RULE(MemVarRule, PassKind::PK_Analysis)

std::string MemoryMigrationRule::getTypeStrRemovedAddrOf(const Expr *E,
                                                         bool isCOCE) {
  QualType QT;
  if (isCOCE) {
    auto COCE = dyn_cast<CXXOperatorCallExpr>(E);
    if (!COCE) {
      return "";
    }
    QT = COCE->getArg(0)->getType();
  } else {
    auto UO = dyn_cast<UnaryOperator>(E);
    if (!UO) {
      return "";
    }
    QT = UO->getSubExpr()->getType();
  }
  std::string ReplType = DpctGlobalInfo::getReplacedTypeName(QT);
  return ReplType;
}

/// Get the assigned part of the malloc function call.
/// \param [in] E The expression needs to be analyzed.
/// \param [in] Arg0Str The original string of the first arg of the malloc.
/// e.g.:
/// origin code:
///   int2 const * d_data;
///   cudaMalloc((void **)&d_data, sizeof(int2));
/// This function will return a string "d_data = (sycl::int2 const *)"
/// In this example, \param E is "&d_data", \param Arg0Str is "(void **)&d_data"
std::string MemoryMigrationRule::getAssignedStr(const Expr *E,
                                                const std::string &Arg0Str) {
  std::ostringstream Repl;
  std::string Type;
  printDerefOp(Repl, E, &Type);
  Repl << " = (" << Type << ")";

  return Repl.str();
}

const ArraySubscriptExpr *
MemoryMigrationRule::getArraySubscriptExpr(const Expr *E) {
  if (const auto MTE = dyn_cast<MaterializeTemporaryExpr>(E)) {
    if (auto TE = MTE->getSubExpr()) {
      if (auto UO = dyn_cast<UnaryOperator>(TE)) {
        if (auto Arg = dyn_cast<ArraySubscriptExpr>(UO->getSubExpr())) {
          return Arg;
        }
      }
    }
  }
  return nullptr;
}

const Expr *MemoryMigrationRule::getUnaryOperatorExpr(const Expr *E) {
  if (const auto MTE = dyn_cast<MaterializeTemporaryExpr>(E)) {
    if (auto TE = MTE->getSubExpr()) {
      if (auto UO = dyn_cast<UnaryOperator>(TE)) {
        return UO->getSubExpr();
      }
    }
  }
  return nullptr;
}

llvm::raw_ostream &printMemcpy3DParmsName(llvm::raw_ostream &OS,
                                          StringRef BaseName,
                                          StringRef MemberName) {
  return OS << BaseName << "_" << MemberName << getCTFixedSuffix();
}

void MemoryMigrationRule::replaceMemAPIArg(
    const Expr *E, const ast_matchers::MatchFinder::MatchResult &Result,
    const std::string &StreamStr, std::string OffsetFromBaseStr) {

  StringRef VarName;
  auto Sub = E->IgnoreImplicitAsWritten();
  if (auto MTE = dyn_cast<MaterializeTemporaryExpr>(Sub)) {
    Sub = MTE->getSubExpr()->IgnoreImplicitAsWritten();
  }
  if (auto UO = dyn_cast<UnaryOperator>(Sub)) {
    if (UO->getOpcode() == UO_AddrOf) {
      Sub = UO->getSubExpr()->IgnoreImplicitAsWritten();
    }
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(Sub)) {
    if (COCE->getOperator() == OO_Amp) {
      Sub = COCE->getArg(0);
    }
  }
  std::string ArrayOffset;
  if (auto ASE = dyn_cast<ArraySubscriptExpr>(Sub)) {
    Sub = ASE->getBase()->IgnoreImplicitAsWritten();
    auto Idx = ASE->getIdx();
    Expr::EvalResult ER;
    ArrayOffset = ExprAnalysis::ref(Idx);
    if (!Idx->isValueDependent() && Idx->EvaluateAsInt(ER, *Result.Context)) {
      if (ER.Val.getInt().getZExtValue() == 0) {
        ArrayOffset.clear();
      }
    }
  }
  if (auto DRE = dyn_cast<DeclRefExpr>(Sub)) {
    if (auto VI = DpctGlobalInfo::getInstance().findMemVarInfo(
            dyn_cast<VarDecl>(DRE->getDecl()))) {
      VarName = VI->getName();
    }
  } else if (auto SL = dyn_cast<StringLiteral>(Sub)) {
    VarName = SL->getString();
  }

  if (VarName.empty())
    return;

  std::string Replaced;
  llvm::raw_string_ostream OS(Replaced);

  auto PrintVarName = [&](llvm::raw_ostream &Out) {
    Out << VarName << ".get_ptr(";
    if (!StreamStr.empty()) {
      requestFeature(HelperFeatureEnum::device_ext);
      Out << "*" << StreamStr;
    } else {
      requestFeature(HelperFeatureEnum::device_ext);
    }
    Out << ")";
    if (!ArrayOffset.empty())
      Out << " + " << ArrayOffset;
  };

  if (OffsetFromBaseStr.empty()) {
    PrintVarName(OS);
  } else {
    OS << "(char *)(";
    PrintVarName(OS);
    OS << ") + " << OffsetFromBaseStr;
  }
  emplaceTransformation(
      new ReplaceToken(E->getBeginLoc(), E->getEndLoc(), std::move(OS.str())));
}

bool MemoryMigrationRule::canUseTemplateStyleMigration(
    const Expr *AllocatedExpr, const Expr *SizeExpr, std::string &ReplType,
    std::string &ReplSize) {
  const Expr *AE = nullptr;
  if (auto CSCE = dyn_cast<CStyleCastExpr>(AllocatedExpr)) {
    AE = CSCE->getSubExpr()->IgnoreImplicitAsWritten();
  } else {
    AE = AllocatedExpr;
  }

  QualType DerefQT = AE->getType();
  if (DerefQT->isPointerType()) {
    DerefQT = DerefQT->getPointeeType();
    if (DerefQT->isPointerType()) {
      DerefQT = DerefQT->getPointeeType();
    } else {
      return false;
    }
  } else {
    return false;
  }

  std::string TypeStr = DpctGlobalInfo::getReplacedTypeName(DerefQT);
  // ReplType will be used as the template argument in memory API.
  ReplType = getFinalCastTypeNameStr(TypeStr);

  auto BO = dyn_cast<BinaryOperator>(SizeExpr);
  if (BO && BO->getOpcode() == BinaryOperatorKind::BO_Mul) {
    std::string Repl;
    if (!isContainMacro(BO->getLHS()) &&
        isSameSizeofTypeWithTypeStr(BO->getLHS(), TypeStr)) {
      // case 1: sizeof(b) * a
      ArgumentAnalysis AA;
      AA.setCallSpelling(BO);
      AA.analyze(BO->getRHS());
      Repl = AA.getRewritePrefix() + AA.getRewriteString() +
             AA.getRewritePostfix();
    } else if (!isContainMacro(BO->getRHS()) &&
               isSameSizeofTypeWithTypeStr(BO->getRHS(), TypeStr)) {
      // case 2: a * sizeof(b)
      ArgumentAnalysis AA;
      AA.setCallSpelling(BO);
      AA.analyze(BO->getLHS());
      Repl = AA.getRewritePrefix() + AA.getRewriteString() +
             AA.getRewritePostfix();
    } else {
      return false;
    }

    SourceLocation RemoveBegin, RemoveEnd;
    SourceRange RemoveRange = getStmtExpansionSourceRange(BO);
    RemoveBegin = RemoveRange.getBegin();
    RemoveEnd = RemoveRange.getEnd();
    RemoveEnd = RemoveEnd.getLocWithOffset(
        Lexer::MeasureTokenLength(RemoveEnd, DpctGlobalInfo::getSourceManager(),
                                  DpctGlobalInfo::getContext().getLangOpts()));
    emplaceTransformation(replaceText(RemoveBegin, RemoveEnd, std::move(Repl),
                                      DpctGlobalInfo::getSourceManager()));
    return true;
  } else {
    // case 3: sizeof(b)
    if (!isContainMacro(SizeExpr) &&
        isSameSizeofTypeWithTypeStr(SizeExpr, TypeStr)) {
      SourceLocation RemoveBegin, RemoveEnd;
      SourceRange RemoveRange = getStmtExpansionSourceRange(SizeExpr);
      RemoveBegin = RemoveRange.getBegin();
      RemoveEnd = RemoveRange.getEnd();
      RemoveEnd = RemoveEnd.getLocWithOffset(Lexer::MeasureTokenLength(
          RemoveEnd, DpctGlobalInfo::getSourceManager(),
          DpctGlobalInfo::getContext().getLangOpts()));
      emplaceTransformation(replaceText(RemoveBegin, RemoveEnd, "1",
                                        DpctGlobalInfo::getSourceManager()));

      return true;
    }
  }

  return false;
}

/// Transform cudaMallocxxx() to xxx = mallocxxx();
void MemoryMigrationRule::mallocMigrationWithTransformation(
    SourceManager &SM, const CallExpr *C, const std::string &CallName,
    std::string &&ReplaceName, const std::string &PaddingArgs,
    bool NeedTypeCast, size_t AllocatedArgIndex, size_t SizeArgIndex) {
  std::string ReplSize, ReplType;
  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted &&
      CallName != "cudaMallocArray" && CallName != "cudaMalloc3DArray" &&
      CallName != "cublasAlloc" &&
      canUseTemplateStyleMigration(C->getArg(AllocatedArgIndex),
                                   C->getArg(SizeArgIndex), ReplType,
                                   ReplSize)) {
    auto TM = new InsertBeforeStmt(
        C, getTransformedMallocPrefixStr(C->getArg(AllocatedArgIndex),
                                         NeedTypeCast, true));
    TM->setInsertPosition(IP_Right);
    emplaceTransformation(TM);

    emplaceTransformation(
        new ReplaceCalleeName(C, ReplaceName + "<" + ReplType + ">"));
  } else {
    auto TM = new InsertBeforeStmt(
        C, getTransformedMallocPrefixStr(C->getArg(AllocatedArgIndex),
                                         NeedTypeCast));
    TM->setInsertPosition(IP_Right);
    emplaceTransformation(TM);

    emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceName)));
  }
  emplaceTransformation(removeArg(C, AllocatedArgIndex, SM));
  if (!PaddingArgs.empty())
    emplaceTransformation(
        new InsertText(C->getRParenLoc(), ", " + PaddingArgs));
}

/// e.g., for int *a and cudaMalloc(&a, size), print "a = ".
/// If \p DerefType is not null, assign a string "int *".
void printDerefOp(std::ostream &OS, const Expr *E, std::string *DerefType) {
  E = E->IgnoreImplicitAsWritten();
  bool NeedDerefOp = true;
  if (auto UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == clang::UO_AddrOf) {
      E = UO->getSubExpr()->IgnoreImplicitAsWritten();
      NeedDerefOp = false;
    }
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (COCE->getOperator() == clang::OO_Amp && COCE->getNumArgs() == 1) {
      E = COCE->getArg(0)->IgnoreImplicitAsWritten();
      NeedDerefOp = false;
    }
  }
  E = E->IgnoreParens();

  std::unique_ptr<ParensPrinter<std::ostream>> PP;
  if (NeedDerefOp) {
    OS << "*";
    switch (E->getStmtClass()) {
    case Stmt::DeclRefExprClass:
    case Stmt::MemberExprClass:
    case Stmt::ParenExprClass:
    case Stmt::CallExprClass:
    case Stmt::IntegerLiteralClass:
      break;
    default:
      PP = std::make_unique<ParensPrinter<std::ostream>>(OS);
      break;
    }
  }
  ExprAnalysis EA(E);
  EA.analyze();
  OS << EA.getReplacedString();

  if (DerefType) {
    QualType DerefQT;
    if (auto ArraySub = dyn_cast<ArraySubscriptExpr>(E)) {
      QualType BaseType = ArraySub->getBase()->getType();
      if (BaseType->isArrayType()) {
        if (auto Array = BaseType->getAsArrayTypeUnsafe()) {
          DerefQT = Array->getElementType();
        }
      } else if (BaseType->isPointerType()) {
        DerefQT = BaseType->getPointeeType();
      }
    }
    if (DerefQT.isNull()) {
      DerefQT = E->getType();
    }
    if (NeedDerefOp)
      DerefQT = DerefQT->getPointeeType();
    *DerefType = DpctGlobalInfo::getReplacedTypeName(DerefQT);
  }
}

/// e.g., for int *a and cudaMalloc(&a, size), return "a = (int *)".
/// If \p NeedTypeCast is false, return "a = ";
/// If \p TemplateStyle is true, \p NeedTypeCast will be specified as false
/// always
std::string MemoryMigrationRule::getTransformedMallocPrefixStr(
    const Expr *MallocOutArg, bool NeedTypeCast, bool TemplateStyle) {
  if (TemplateStyle)
    NeedTypeCast = false;
  std::ostringstream OS;
  std::string CastTypeName;
  MallocOutArg = MallocOutArg->IgnoreImplicitAsWritten();
  if (auto CSCE = dyn_cast<CStyleCastExpr>(MallocOutArg)) {
    MallocOutArg = CSCE->getSubExpr()->IgnoreImplicitAsWritten();
    if (!TemplateStyle)
      NeedTypeCast = true;
  }
  printDerefOp(OS, MallocOutArg, NeedTypeCast ? &CastTypeName : nullptr);

  OS << " = ";
  if (!CastTypeName.empty())
    OS << "(" << getFinalCastTypeNameStr(CastTypeName) << ")";

  return OS.str();
}

/// Common migration for cudaMallocArray and cudaMalloc3DArray.
void MemoryMigrationRule::mallocArrayMigration(const CallExpr *C,
                                               const std::string &Name,
                                               size_t FlagIndex,
                                               SourceManager &SM) {

  requestFeature(HelperFeatureEnum::device_ext);
  mallocMigrationWithTransformation(
      SM, C, Name, "new " + MapNames::getDpctNamespace() + "image_matrix", "",
      false);

  emplaceTransformation(removeArg(C, FlagIndex, SM));

  std::ostringstream OS;
  printDerefOp(OS, C->getArg(1));
  emplaceTransformation(new ReplaceStmt(C->getArg(1), OS.str()));
}

void MemoryMigrationRule::mallocMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  if (isPlaceholderIdxDuplicated(C))
    return;
  int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();

  if (Name == "cudaMalloc" || Name == "cuMemAlloc_v2") {
    if (USMLevel == UsmLevel::UL_Restricted) {
      // Leverage CallExprRewritter to migrate the USM version
      ExprAnalysis EA(C);
      auto LocInfo = DpctGlobalInfo::getLocInfo(C->getBeginLoc());
      auto Info = std::make_shared<PriorityReplInfo>();
      if (auto TM = EA.getReplacement())
        Info->Repls.push_back(TM->getReplacement(DpctGlobalInfo::getContext()));
      Info->Repls.insert(Info->Repls.end(), EA.getSubExprRepl().begin(),
                         EA.getSubExprRepl().end());
      DpctGlobalInfo::addPriorityReplInfo(
          LocInfo.first + std::to_string(LocInfo.second), Info);
    } else {
      DpctGlobalInfo::getInstance().insertCudaMalloc(C);
      auto LocInfo = DpctGlobalInfo::getLocInfo(C->getBeginLoc());
      auto Action = []() {
        requestFeature(HelperFeatureEnum::device_ext);
      };
      auto Info = std::make_shared<PriorityReplInfo>();
      auto &Context = DpctGlobalInfo::getContext();
      auto &SM = *Result.SourceManager;
      Info->RelatedAction.emplace_back(Action);
      if (auto TM = removeArg(C, 0, SM))
        Info->Repls.push_back(TM->getReplacement(Context));
      ExprAnalysis EA(C);
      if (auto TM = EA.getReplacement())
        Info->Repls.push_back(TM->getReplacement(DpctGlobalInfo::getContext()));
      Info->Repls.insert(Info->Repls.end(), EA.getSubExprRepl().begin(),
                         EA.getSubExprRepl().end());

      DpctGlobalInfo::addPriorityReplInfo(
          LocInfo.first + std::to_string(LocInfo.second), Info);
    }
  } else if (Name == "cudaHostAlloc" || Name == "cudaMallocHost" ||
             Name == "cuMemHostAlloc" || Name == "cuMemAllocHost_v2" ||
             Name == "cuMemAllocPitch_v2" || Name == "cudaMallocPitch") {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  } else if (Name == "cudaMallocManaged" || Name == "cuMemAllocManaged") {
    if (USMLevel == UsmLevel::UL_Restricted) {
      // Leverage CallExprRewriter to migrate the USM version
      ExprAnalysis EA(C);
      auto LocInfo = DpctGlobalInfo::getLocInfo(C->getBeginLoc());
      auto Info = std::make_shared<PriorityReplInfo>();
      if (auto TM = EA.getReplacement())
        Info->Repls.push_back(TM->getReplacement(DpctGlobalInfo::getContext()));
      Info->Repls.insert(Info->Repls.end(), EA.getSubExprRepl().begin(),
                         EA.getSubExprRepl().end());
      DpctGlobalInfo::addPriorityReplInfo(
          LocInfo.first + std::to_string(LocInfo.second), Info);
    } else {
      ManagedPointerAnalysis MPA(C, IsAssigned);
      MPA.RecursiveAnalyze();
      MPA.applyAllSubExprRepl();
    }
  } else if (Name == "cublasAlloc") {
    // TODO: migrate functions when they are in template
    // TODO: migrate functions when they are in macro body
    auto ArgRange0 = getStmtExpansionSourceRange(C->getArg(0));
    auto ArgEnd0 = ArgRange0.getEnd().getLocWithOffset(
        Lexer::MeasureTokenLength(ArgRange0.getEnd(), *(Result.SourceManager),
                                  Result.Context->getLangOpts()));
    auto ArgRange1 = getStmtExpansionSourceRange(C->getArg(1));
    emplaceTransformation(
        replaceText(ArgEnd0, ArgRange1.getBegin(), "*", *Result.SourceManager));
    insertAroundStmt(C->getArg(0), "(", ")");
    insertAroundStmt(C->getArg(1), "(", ")");
    DpctGlobalInfo::getInstance().insertCublasAlloc(C);
    emplaceTransformation(removeArg(C, 2, *Result.SourceManager));
    if (USMLevel == UsmLevel::UL_Restricted) {
      buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
      if (IsAssigned)
        emplaceTransformation(new InsertBeforeStmt(C, "DPCT_CHECK_ERROR("));
      mallocMigrationWithTransformation(
          *Result.SourceManager, C, Name,
          MapNames::getClNamespace() + "malloc_device",
          "{{NEEDREPLACEQ" + std::to_string(Index) + "}}", true, 2);
      if (IsAssigned) {
        emplaceTransformation(new InsertAfterStmt(C, ")"));
        requestFeature(HelperFeatureEnum::device_ext);
      }
    } else {
      ExprAnalysis EA(C->getArg(2));
      EA.analyze();
      std::ostringstream OS;
      std::string Type;
      if (IsAssigned)
        OS << "DPCT_CHECK_ERROR(";
      printDerefOp(OS, C->getArg(2)->IgnoreCasts()->IgnoreParens(), &Type);
      if (Type != "NULL TYPE" && Type != "void *")
        OS << " = (" << Type << ")";
      else
        OS << " = ";

      emplaceTransformation(new InsertBeforeStmt(C, OS.str()));
      emplaceTransformation(new ReplaceCalleeName(
          C, MapNames::getDpctNamespace() + "dpct_malloc"));
      requestFeature(HelperFeatureEnum::device_ext);
      if (IsAssigned) {
        emplaceTransformation(new InsertAfterStmt(C, ")"));
        requestFeature(HelperFeatureEnum::device_ext);
      }
    }
  } else if (Name == "cudaMalloc3D") {
    std::ostringstream OS;
    std::string Type;
    if (IsAssigned)
      OS << "DPCT_CHECK_ERROR(";
    printDerefOp(OS, C->getArg(0)->IgnoreCasts()->IgnoreParens(), &Type);
    if (Name != "cudaMalloc3D" && Type != "NULL TYPE" && Type != "void *")
      OS << " = (" << Type << ")";
    else
      OS << " = ";

    requestFeature(HelperFeatureEnum::device_ext);
    emplaceTransformation(new InsertBeforeStmt(C, OS.str()));
    emplaceTransformation(
        new ReplaceCalleeName(C, MapNames::getDpctNamespace() + "dpct_malloc"));
    emplaceTransformation(removeArg(C, 0, *Result.SourceManager));
    std::ostringstream OS2;
    printDerefOp(OS2, C->getArg(1));
    if (IsAssigned) {
      emplaceTransformation(new InsertAfterStmt(C, ")"));
      requestFeature(HelperFeatureEnum::device_ext);
    }
  } else if (Name == "cudaMalloc3DArray") {
    mallocArrayMigration(C, Name, 3, *Result.SourceManager);
  } else if (Name == "cudaMallocArray") {
    mallocArrayMigration(C, Name, 4, *Result.SourceManager);
    static std::string SizeClassName =
        DpctGlobalInfo::getCtadClass(MapNames::getClNamespace() + "range", 2);
    if (C->getArg(3)->isDefaultArgument())
      aggregateArgsToCtor(C, SizeClassName, 2, 2, ", 0", *Result.SourceManager);
    else
      aggregateArgsToCtor(C, SizeClassName, 2, 3, "", *Result.SourceManager);
  }
}

void MemoryMigrationRule::memcpyMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  for (unsigned I = 0, E = C->getNumArgs(); I != E; ++I) {
    if (isa<PackExpansionExpr>(C->getArg(I))) {
      return;
    }
  }

  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  if (!CallExprRewriterFactoryBase::RewriterMap)
    return;
  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(Name);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  std::string ReplaceStr;
  // Detect if there is Async in the func name and crop the async substr
  std::string NameRef = Name;
  bool IsAsync = false;
  size_t AsyncLoc = NameRef.find("Async");
  if (AsyncLoc != std::string::npos) {
    IsAsync = true;
    NameRef = NameRef.substr(0, AsyncLoc);
  }
  if (!NameRef.compare("cudaMemcpy2D")) {
    handleDirection(C, 6);
    handleAsync(C, 7, Result);
  } else if (!NameRef.compare("cudaMemcpy3D") ||
             NameRef.rfind("cuMemcpy3D", 0) == 0 ||
             NameRef.rfind("cuMemcpy2D", 0) == 0) {
    handleAsync(C, 1, Result);
    if (auto UO =
            dyn_cast<UnaryOperator>(C->getArg(0)->IgnoreImplicitAsWritten())) {
      if (auto DRE = dyn_cast<DeclRefExpr>(
              UO->getSubExpr()->IgnoreImplicitAsWritten())) {
        std::string DirectionStr =
            IsAsync && NameRef.compare("cudaMemcpy3D")
                ? (", " + MapNames::getDpctNamespace() + "automatic")
                : "";
        emplaceTransformation(new ReplaceStmt(
            C->getArg(0),
            MemoryDataTypeRule::getMemcpy3DArguments(
                DRE->getDecl()->getName(), !NameRef.compare("cudaMemcpy3D")) +
                DirectionStr));
      }
    }
  } else if (!NameRef.compare("cudaMemcpy") ||
             NameRef.rfind("cuMemcpyDtoH", 0) == 0) {
    if (!NameRef.compare("cudaMemcpy")) {
      handleDirection(C, 3);
    }
    std::string AsyncQueue;
    bool NeedTypeCast = false;

    size_t StreamIndex = NameRef.compare("cudaMemcpy") ? 3 : 4;
    if (StreamIndex < C->getNumArgs()) {
      auto StreamArg = C->getArg(StreamIndex);
      // Is the stream argument a default stream handle we recognize?
      // Note: the value for the default stream argument in
      // cudaMemcpyAsync is 0, aka the default stream
      if (StreamArg->isDefaultArgument() || isDefaultStream(StreamArg)) {
        AsyncQueue = "";
      }
      // Are we casting from an integer?
      else if (auto Cast = dyn_cast<CastExpr>(StreamArg);
               Cast && Cast->getCastKind() != clang::CK_LValueToRValue &&
               Cast->getSubExpr()->getType()->isIntegerType()) {
        requestFeature(HelperFeatureEnum::device_ext);
        AsyncQueue = MapNames::getDpctNamespace() + "int_as_queue_ptr(" +
                     ExprAnalysis::ref(Cast->getSubExpr()) + ")";
      } else {
        // If we are implicitly casting from something other than
        // an int (e.g. a user defined class), we need to explicitly
        // insert that cast in the migration to use member access (->).
        if (auto ICE = dyn_cast<ImplicitCastExpr>(StreamArg))
          NeedTypeCast = ICE->getCastKind() != clang::CK_LValueToRValue;
        AsyncQueue = ExprAnalysis::ref(StreamArg);
      }
    }

    replaceMemAPIArg(C->getArg(0), Result, AsyncQueue);
    replaceMemAPIArg(C->getArg(1), Result, AsyncQueue);
    if (USMLevel == UsmLevel::UL_Restricted) {
      // Since the range of removeArg is larger than the range of
      // handleDirection, the handle direction replacement will be removed.
      emplaceTransformation(removeArg(C, 3, *Result.SourceManager));
      if (IsAsync) {
        emplaceTransformation(removeArg(C, 4, *Result.SourceManager));
      } else {
        if (NameRef.compare("cudaMemcpy") || !canOmitMemcpyWait(C)) {
          // wait is needed when FuncName is not cudaMemcpy or
          // cudaMemcpy really needs wait
          emplaceTransformation(new InsertAfterStmt(C, ".wait()"));
        }
      }
      if (AsyncQueue.empty()) {
        if (isPlaceholderIdxDuplicated(C))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
        ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memcpy";
      } else {
        if (NeedTypeCast)
          AsyncQueue = buildString("((sycl::queue *)(", AsyncQueue, "))");

        ReplaceStr = AsyncQueue + "->memcpy";
      }
    } else {
      if (!NameRef.compare("cudaMemcpy")) {
        handleAsync(C, 4, Result);
      } else {
        emplaceTransformation(new InsertAfterStmt(
            C->getArg(2), ", " + MapNames::getDpctNamespace() + "automatic"));
        handleAsync(C, 3, Result);
      }
    }
  }

  if (ReplaceStr.empty()) {
    if (IsAsync) {
      ReplaceStr = MapNames::getDpctNamespace() + "async_dpct_memcpy";
      requestFeature(HelperFeatureEnum::device_ext);
    } else {
      ReplaceStr = MapNames::getDpctNamespace() + "dpct_memcpy";
      requestFeature(HelperFeatureEnum::device_ext);
    }
  }

  if (ULExpr) {
    auto BeginLoc = ULExpr->getBeginLoc();
    auto EndLoc = ULExpr->hasExplicitTemplateArgs()
                 ? ULExpr->getLAngleLoc().getLocWithOffset(-1)
                 : ULExpr->getEndLoc();
    emplaceTransformation(new ReplaceToken(BeginLoc, EndLoc, std::move(ReplaceStr)));
  } else {
    emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
  }
}

void MemoryMigrationRule::arrayMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  auto& SM = *Result.SourceManager;
  std::string ReplaceStr;
  StringRef NameRef(Name);
  auto EndPos = C->getNumArgs() - 1;
  bool IsAsync = NameRef.endswith("Async");
  if (NameRef == "cuMemcpyAtoH_v2" || NameRef == "cuMemcpyHtoA_v2" ||
      NameRef == "cuMemcpyAtoHAsync_v2" || NameRef == "cuMemcpyHtoAAsync_v2" ||
      NameRef == "cuMemcpyAtoD_v2" || NameRef == "cuMemcpyDtoA_v2" ||
      NameRef == "cuMemcpyAtoA_v2") {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  if (IsAsync) {
    NameRef = NameRef.drop_back(5 /* len of "Async" */);
    ReplaceStr = MapNames::getDpctNamespace() + "async_dpct_memcpy";

    auto StreamExpr = C->getArg(EndPos);
    std::string Str;
    if (isDefaultStream(StreamExpr)) {
      emplaceTransformation(removeArg(C, EndPos, SM));
      emplaceTransformation(removeArg(C, --EndPos, SM));
    } else {
      auto Begin = getArgEndLocation(C, EndPos - 2, SM),
           End = getArgEndLocation(C, EndPos, SM);
      llvm::raw_string_ostream OS(Str);
      OS << ", " << MapNames::getDpctNamespace() << "automatic";
      OS << ", ";
      DerefExpr(StreamExpr, C).print(OS);
      emplaceTransformation(replaceText(Begin, End, std::move(Str), SM));
    }
    requestFeature(HelperFeatureEnum::device_ext);
  } else {
    ReplaceStr = MapNames::getDpctNamespace() + "dpct_memcpy";
    emplaceTransformation(removeArg(C, EndPos, SM));
    requestFeature(HelperFeatureEnum::device_ext);
  }

  if (NameRef == "cudaMemcpy2DArrayToArray") {
    insertToPitchedData(C, 0);
    aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
    insertToPitchedData(C, 3);
    aggregate3DVectorClassCtor(C, "id", 4, "0", SM);
    aggregate3DVectorClassCtor(C, "range", 6, "1", SM);
  } else if (NameRef == "cudaMemcpy2DFromArray") {
    aggregatePitchedData(C, 0, 1, SM);
    insertZeroOffset(C, 2);
    insertToPitchedData(C, 2);
    aggregate3DVectorClassCtor(C, "id", 3, "0", SM);
    aggregate3DVectorClassCtor(C, "range", 5, "1", SM);
  } else if (NameRef == "cudaMemcpy2DToArray") {
    insertToPitchedData(C, 0);
    aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
    aggregatePitchedData(C, 3, 4, SM);
    insertZeroOffset(C, 5);
    aggregate3DVectorClassCtor(C, "range", 5, "1", SM);
  } else if (NameRef == "cudaMemcpyArrayToArray") {
    insertToPitchedData(C, 0);
    aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
    insertToPitchedData(C, 3);
    aggregate3DVectorClassCtor(C, "id", 4, "0", SM);
    aggregate3DVectorClassCtor(C, "range", 6, "1", SM, 1);
  } else if (NameRef == "cudaMemcpyFromArray") {
    aggregatePitchedData(C, 0, 4, SM, true);
    insertZeroOffset(C, 1);
    insertToPitchedData(C, 1);
    aggregate3DVectorClassCtor(C, "id", 2, "0", SM);
    aggregate3DVectorClassCtor(C, "range", 4, "1", SM, 1);
  } else if (NameRef == "cudaMemcpyToArray") {
    insertToPitchedData(C, 0);
    aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
    aggregatePitchedData(C, 3, 4, SM, true);
    insertZeroOffset(C, 4);
    aggregate3DVectorClassCtor(C, "range", 4, "1", SM, 1);
  }

  if (ULExpr) {
    auto BeginLoc = ULExpr->getBeginLoc();
    auto EndLoc = ULExpr->hasExplicitTemplateArgs()
                      ? ULExpr->getLAngleLoc().getLocWithOffset(-1)
                      : ULExpr->getEndLoc();
    emplaceTransformation(new ReplaceToken(BeginLoc, EndLoc, std::move(ReplaceStr)));
  } else {
    emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
  }
}

void MemoryMigrationRule::memcpySymbolMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string DirectionName;
  // Currently, if memory API occurs in a template, we will migrate the API call
  // under the undeclared decl AST node and the explicit specialization AST
  // node. The API call in explicit specialization is same as without template.
  // But if the API has non-specified default parameters and it is in an
  // undeclared decl, these default parameters will not be counted into the
  // number of call arguments. So, we need check the argument number before get
  // it.
  if (C->getNumArgs() >= 5 && !C->getArg(4)->isDefaultArgument()) {
    const Expr *Direction = C->getArg(4);
    const DeclRefExpr *DD = dyn_cast_or_null<DeclRefExpr>(Direction);
    if (DD && isa<EnumConstantDecl>(DD->getDecl())) {
      DirectionName = DD->getNameInfo().getName().getAsString();
      auto Search = EnumConstantRule::EnumNamesMap.find(DirectionName);
      if (Search == EnumConstantRule::EnumNamesMap.end())
        return;
      requestHelperFeatureForEnumNames(DirectionName);
      Direction = nullptr;
      DirectionName = Search->second->NewName;
    }
  }

  DpctGlobalInfo &Global = DpctGlobalInfo::getInstance();
  auto MallocInfo = Global.findCudaMalloc(C->getArg(1));
  auto VD = CudaMallocInfo::getDecl(C->getArg(0));
  if (MallocInfo && VD) {
    if (auto Var = Global.findMemVarInfo(VD)) {
      requestFeature(HelperFeatureEnum::device_ext);
      emplaceTransformation(new ReplaceStmt(
          C, Var->getName() + ".assign(" +
                 MallocInfo->getAssignArgs(Var->getType()->getBaseName()) +
                 ")"));
      return;
    }
  }

  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  std::string ReplaceStr;
  std::string StreamStr;
  if (isPlaceholderIdxDuplicated(C))
    return;
  int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
  if (Name == "cudaMemcpyToSymbol" || Name == "cudaMemcpyFromSymbol") {
    if (USMLevel == UsmLevel::UL_Restricted) {
      buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
      ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memcpy";
    } else {
      requestFeature(HelperFeatureEnum::device_ext);
      ReplaceStr = MapNames::getDpctNamespace() + "dpct_memcpy";
    }
  } else {
    if (C->getNumArgs() == 6 && !C->getArg(5)->isDefaultArgument()) {
      if (!isDefaultStream(C->getArg(5))) {
        StreamStr = ExprAnalysis::ref(C->getArg(5));
      }
    }
    if (USMLevel == UsmLevel::UL_Restricted) {
      if (StreamStr.empty()) {
        buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
        ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memcpy";
      } else {
        ReplaceStr = StreamStr + "->memcpy";
      }
    } else {
      requestFeature(HelperFeatureEnum::device_ext);
      ReplaceStr = MapNames::getDpctNamespace() + "async_dpct_memcpy";
    }
  }

  if (ULExpr) {
    auto BeginLoc = ULExpr->getBeginLoc();
    auto EndLoc = ULExpr->hasExplicitTemplateArgs()
                 ? ULExpr->getLAngleLoc().getLocWithOffset(-1)
                 : ULExpr->getEndLoc();
    emplaceTransformation(new ReplaceToken(BeginLoc, EndLoc, std::move(ReplaceStr)));
  } else {
    emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
  }

  ExprAnalysis EA;
  std::string OffsetFromBaseStr;
  if (C->getNumArgs() >= 4 && !C->getArg(3)->isDefaultArgument()) {
    EA.analyze(C->getArg(3));
    OffsetFromBaseStr = EA.getReplacedString();
  } else {
    OffsetFromBaseStr = "0";
  }

  if ((Name == "cudaMemcpyToSymbol" || Name == "cudaMemcpyToSymbolAsync") &&
      OffsetFromBaseStr != "0") {
    replaceMemAPIArg(C->getArg(0), Result, StreamStr, OffsetFromBaseStr);
  } else {
    replaceMemAPIArg(C->getArg(0), Result, StreamStr);
  }

  if ((Name == "cudaMemcpyFromSymbol" || Name == "cudaMemcpyFromSymbolAsync") &&
      OffsetFromBaseStr != "0") {
    replaceMemAPIArg(C->getArg(1), Result, StreamStr, OffsetFromBaseStr);
  } else {
    replaceMemAPIArg(C->getArg(1), Result, StreamStr);
  }

  // Remove C->getArg(3)
  if (C->getNumArgs() >= 4 && !C->getArg(3)->isDefaultArgument()) {
    if (auto TM = removeArg(C, 3, *Result.SourceManager))
      emplaceTransformation(TM);
  }

  if (C->getNumArgs() >= 5 && !C->getArg(4)->isDefaultArgument()) {
    emplaceTransformation(
        new ReplaceStmt(C->getArg(4), std::move(DirectionName)));
  }

  // Async
  if (Name == "cudaMemcpyToSymbolAsync" ||
      Name == "cudaMemcpyFromSymbolAsync") {
    if (C->getNumArgs() == 6 && !C->getArg(4)->isDefaultArgument()) {
      if (USMLevel == UsmLevel::UL_Restricted) {
        if (auto TM = removeArg(C, 4, *Result.SourceManager))
          emplaceTransformation(TM);
        if (!C->getArg(5)->isDefaultArgument()) {
          if (auto TM = removeArg(C, 5, *Result.SourceManager))
            emplaceTransformation(TM);
        }
      } else {
        handleAsync(C, 5, Result);
      }
    } else if (C->getNumArgs() == 5 && !C->getArg(4)->isDefaultArgument()) {
      if (USMLevel == UsmLevel::UL_Restricted) {
        if (auto TM = removeArg(C, 4, *Result.SourceManager))
          emplaceTransformation(TM);
      }
    }
  } else {
    if (USMLevel == UsmLevel::UL_Restricted) {
      if (C->getNumArgs() == 5 && !C->getArg(4)->isDefaultArgument()) {
        if (auto TM = removeArg(C, 4, *Result.SourceManager))
          emplaceTransformation(TM);
      }
      if (!canOmitMemcpyWait(C)) {
        emplaceTransformation(new InsertAfterStmt(C, ".wait()"));
      }
    }
  }
}

void MemoryMigrationRule::freeMigration(const MatchFinder::MatchResult &Result,
                                        const CallExpr *C,
                                        const UnresolvedLookupExpr *ULExpr,
                                        bool IsAssigned) {

  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }
  if (isPlaceholderIdxDuplicated(C))
    return;

  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(Name);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
  int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
  if (Name == "cudaFree" || Name == "cublasFree") {
    if (USMLevel == UsmLevel::UL_Restricted) {
      ArgumentAnalysis AA;
      AA.setCallSpelling(C);
      AA.analyze(C->getArg(0));
      auto ArgStr = AA.getRewritePrefix() + AA.getRewriteString() +
                    AA.getRewritePostfix();
      std::ostringstream Repl;
      buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
      if (hasManagedAttr(0)(C)) {
          ArgStr = "*(" + ArgStr + ".get_ptr())";
      }
      Repl << MapNames::getClNamespace() + "free(" << ArgStr
           << ", {{NEEDREPLACEQ" + std::to_string(Index) + "}})";
      emplaceTransformation(new ReplaceStmt(C, std::move(Repl.str())));
    } else {
      requestFeature(HelperFeatureEnum::device_ext);
      emplaceTransformation(
          new ReplaceCalleeName(C, MapNames::getDpctNamespace() + "dpct_free"));
    }
  } else if (Name == "cudaFreeHost" || Name == "cuMemFreeHost") {
    if (USMLevel == UsmLevel::UL_Restricted) {
      CheckCanUseCLibraryMallocOrFree Checker(0, true);
      ExprAnalysis EA;
      EA.analyze(C->getArg(0));
      std::ostringstream Repl;
      if(Checker(C)) {
        Repl << "free(" << EA.getReplacedString() << ")";
      } else {
        buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
        Repl << MapNames::getClNamespace() + "free(" << EA.getReplacedString()
           << ", {{NEEDREPLACEQ" + std::to_string(Index) + "}})";
      }
      emplaceTransformation(new ReplaceStmt(C, std::move(Repl.str())));
    } else {
      emplaceTransformation(new ReplaceCalleeName(C, "free"));
    }
  } else if (Name == "cudaFreeArray") {
    ExprAnalysis EA(C->getArg(0));
    EA.analyze();
    emplaceTransformation(
        new ReplaceStmt(C, "delete " + EA.getReplacedString()));
  }
}

void MemoryMigrationRule::memsetMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(Name);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  std::string ReplaceStr;
  StringRef NameRef(Name);
  bool IsAsync = NameRef.endswith("Async");
  if (IsAsync) {
    NameRef = NameRef.drop_back(5 /* len of "Async" */);
    ReplaceStr = MapNames::getDpctNamespace() + "async_dpct_memset";
    requestFeature(HelperFeatureEnum::device_ext);
  } else {
    ReplaceStr = MapNames::getDpctNamespace() + "dpct_memset";
    requestFeature(HelperFeatureEnum::device_ext);
  }

  if (NameRef == "cudaMemset2D") {
    handleAsync(C, 5, Result);
  } else if (NameRef == "cudaMemset3D") {
    handleAsync(C, 3, Result);
  } else if (NameRef == "cudaMemset") {
    std::string AsyncQueue;
    bool NeedTypeCast = false;
    if (C->getNumArgs() > 3 && !C->getArg(3)->isDefaultArgument()) {
      if (auto ICE = dyn_cast<ImplicitCastExpr>(C->getArg(3)))
        NeedTypeCast = ICE->getCastKind() != clang::CK_LValueToRValue;

      if (!isDefaultStream(C->getArg(3)))
        AsyncQueue = ExprAnalysis::ref(C->getArg(3));
    }
    replaceMemAPIArg(C->getArg(0), Result, AsyncQueue);
    if (USMLevel == UsmLevel::UL_Restricted) {
      if (IsAsync) {
        emplaceTransformation(removeArg(C, 3, *Result.SourceManager));
      } else {
        emplaceTransformation(new InsertAfterStmt(C, ".wait()"));
      }
      if (AsyncQueue.empty()) {
        if (isPlaceholderIdxDuplicated(C))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
        ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memset";
      } else {
        if (NeedTypeCast)
          AsyncQueue = buildString("((sycl::queue *)(", AsyncQueue, "))");

        ReplaceStr = AsyncQueue + "->memset";
      }
    } else {
      handleAsync(C, 3, Result);
    }
  }

  emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
}

void MemoryMigrationRule::getSymbolSizeMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  // Here only handle ordinary variable name reference, for accessing the
  // size of something residing on the device directly from host side should
  // not be possible.
  std::string Replacement;
  ExprAnalysis EA;
  EA.analyze(C->getArg(0));
  auto StmtStrArg0 = EA.getReplacedString();
  EA.analyze(C->getArg(1));
  auto StmtStrArg1 = EA.getReplacedString();

  requestFeature(HelperFeatureEnum::device_ext);
  Replacement = getDrefName(C->getArg(0)) + " = " + StmtStrArg1 + ".get_size()";
  emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
}

void MemoryMigrationRule::prefetchMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string FuncName;
  if (ULExpr) {
    FuncName = ULExpr->getName().getAsString();
  } else {
    FuncName = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(FuncName);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end() &&
      USMLevel == UsmLevel::UL_Restricted) {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  if (USMLevel == UsmLevel::UL_Restricted) {
    const SourceManager *SM = Result.SourceManager;
    std::string Replacement;
    ExprAnalysis EA;
    EA.analyze(C->getArg(0));
    auto StmtStrArg0 = EA.getReplacedString();
    EA.analyze(C->getArg(1));
    auto StmtStrArg1 = EA.getReplacedString();
    EA.analyze(C->getArg(2));
    auto StmtStrArg2 = EA.getReplacedString();
    std::string StmtStrArg3;
    if (C->getNumArgs() == 4 && !C->getArg(3)->isDefaultArgument()) {
      if (!isDefaultStream(C->getArg(3)))
        StmtStrArg3 = ExprAnalysis::ref(C->getArg(3));
    } else {
      StmtStrArg3 = "0";
    }

    // In clang "define NULL __null"
    if (StmtStrArg3 == "0" || StmtStrArg3 == "") {
      const auto Prefix =
          MapNames::getDpctNamespace() +
          (StmtStrArg2 == "cudaCpuDeviceId"
               ? +"cpu_device()"
               : "dev_mgr::instance().get_device(" + StmtStrArg2 + ")");
      requestFeature(HelperFeatureEnum::device_ext);
      Replacement = Prefix + "." + DpctGlobalInfo::getDeviceQueueName() +
                    "().prefetch(" + StmtStrArg0 + "," + StmtStrArg1 + ")";
    } else {
      if (SM->getCharacterData(C->getArg(3)->getBeginLoc()) -
              SM->getCharacterData(C->getArg(3)->getEndLoc()) ==
          0) {
        Replacement =
            StmtStrArg3 + "->prefetch(" + StmtStrArg0 + "," + StmtStrArg1 + ")";
      } else {
        Replacement = "(" + StmtStrArg3 + ")->prefetch(" + StmtStrArg0 + "," +
                      StmtStrArg1 + ")";
      }
    }
    emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
  } else {
    report(C->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, FuncName);
  }
}

void MemoryMigrationRule::miscMigration(const MatchFinder::MatchResult &Result,
                                        const CallExpr *C,
                                        const UnresolvedLookupExpr *ULExpr,
                                        bool IsAssigned) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  if (Name == "cudaHostGetDevicePointer" ||
      Name == "cuMemHostGetDevicePointer_v2") {
    if (USMLevel == UsmLevel::UL_Restricted) {
      ExprAnalysis EA(C);
      auto LocInfo = DpctGlobalInfo::getLocInfo(C->getBeginLoc());
      auto Info = std::make_shared<PriorityReplInfo>();
      if (auto TM = EA.getReplacement())
        Info->Repls.push_back(TM->getReplacement(DpctGlobalInfo::getContext()));
      Info->Repls.insert(Info->Repls.end(), EA.getSubExprRepl().begin(),
                         EA.getSubExprRepl().end());
      DpctGlobalInfo::addPriorityReplInfo(
          LocInfo.first + std::to_string(LocInfo.second), Info);
    } else {
      report(C->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             MapNames::ITFName.at(Name));
    }
  } else if (Name == "make_cudaExtent" || Name == "make_cudaPos") {
    std::string CtorName;
    llvm::raw_string_ostream OS(CtorName);
    DpctGlobalInfo::printCtadClass(
        OS,
        buildString(MapNames::getClNamespace(),
                    (Name == "make_cudaPos") ? "id" : "range"),
        3);
    emplaceTransformation(new ReplaceCalleeName(C, std::move(OS.str())));
  } else if (Name == "cudaGetChannelDesc") {
    std::ostringstream OS;
    printDerefOp(OS, C->getArg(0));
    OS << " = " << ExprAnalysis::ref(C->getArg(1)) << "->get_channel()";
    emplaceTransformation(new ReplaceStmt(C, OS.str()));
    requestFeature(HelperFeatureEnum::device_ext);
  } else if (Name == "cuMemGetInfo_v2" || Name == "cudaMemGetInfo") {
    if (DpctGlobalInfo::useDeviceInfo()) {
      std::ostringstream OS;
      if (IsAssigned)
        OS << "DPCT_CHECK_ERROR(";
      OS << MapNames::getDpctNamespace() + "get_current_device().get_memory_info";
      OS << "(";
      printDerefOp(OS, C->getArg(0));
      OS << ", ";
      printDerefOp(OS, C->getArg(1));
      OS << ")";

      emplaceTransformation(new ReplaceStmt(C, OS.str()));
      if (IsAssigned) {
        OS << ")";
      }
      emplaceTransformation(new ReplaceStmt(C, OS.str()));
      requestFeature(HelperFeatureEnum::device_ext);
      report(C->getBeginLoc(), Diagnostics::EXTENSION_DEVICE_INFO, false,
             Name == "cuMemGetInfo_v2" ? "cuMemGetInfo" : Name);
    } else {
      auto &SM = DpctGlobalInfo::getSourceManager();
      std::ostringstream OS;
      if (IsAssigned)
        OS << "DPCT_CHECK_ERROR(";

      auto SecondArg = C->getArg(1);
      printDerefOp(OS, SecondArg);
      OS << " = " << MapNames::getDpctNamespace()
         << "get_current_device().get_device_info()"
            ".get_global_mem_size()";
      requestFeature(HelperFeatureEnum::device_ext);
      if (IsAssigned) {
        OS << ")";
      }
      SourceLocation CallBegin(C->getBeginLoc());
      SourceLocation CallEnd(C->getEndLoc());

      bool IsMacroArg =
          SM.isMacroArgExpansion(CallBegin) && SM.isMacroArgExpansion(CallEnd);

      if (CallBegin.isMacroID() && IsMacroArg) {
        CallBegin = SM.getImmediateSpellingLoc(CallBegin);
        CallBegin = SM.getExpansionLoc(CallBegin);
      } else if (CallBegin.isMacroID()) {
        CallBegin = SM.getExpansionLoc(CallBegin);
      }

      if (CallEnd.isMacroID() && IsMacroArg) {
        CallEnd = SM.getImmediateSpellingLoc(CallEnd);
        CallEnd = SM.getExpansionLoc(CallEnd);
      } else if (CallEnd.isMacroID()) {
        CallEnd = SM.getExpansionLoc(CallEnd);
      }
      CallEnd = CallEnd.getLocWithOffset(1);

      emplaceTransformation(replaceText(CallBegin, CallEnd, OS.str(), SM));
      report(C->getBeginLoc(), Diagnostics::UNSUPPORT_FREE_MEMORY_SIZE, false);
    }
  } else {
    auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(Name);
    if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
      ExprAnalysis EA(C);
      emplaceTransformation(EA.getReplacement());
      EA.applyAllSubExprRepl();
      return;
    }
  }
}

void MemoryMigrationRule::cudaArrayGetInfo(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string IndentStr =
      getIndent(C->getBeginLoc(), *Result.SourceManager).str();
  if (IsAssigned)
    IndentStr += "  ";
  std::ostringstream OS;
  std::string Arg3Str = ExprAnalysis::ref(C->getArg(3));
  printDerefOp(OS, C->getArg(0));
  OS << " = " << Arg3Str << "->get_channel();" << getNL() << IndentStr;
  printDerefOp(OS, C->getArg(1));
  OS << " = " << Arg3Str << "->get_range();" << getNL() << IndentStr;
  printDerefOp(OS, C->getArg(2));
  OS << " = 0";
  emplaceTransformation(new ReplaceStmt(C, OS.str()));
  requestFeature(HelperFeatureEnum::device_ext);
}

void MemoryMigrationRule::cudaMemAdvise(const MatchFinder::MatchResult &Result,
                                        const CallExpr *C,
                                        const UnresolvedLookupExpr *ULExpr,
                                        bool IsAssigned) {
  auto FuncName = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  // Do nothing if USM is disabled
  if (USMLevel == UsmLevel::UL_None) {
    report(C->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, FuncName);
    return;
  }

  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(FuncName);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end() &&
      USMLevel == UsmLevel::UL_Restricted) {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  auto Arg2Expr = C->getArg(2);
  if (auto NamedCaster = dyn_cast<ExplicitCastExpr>(Arg2Expr)) {
    if (NamedCaster->getTypeAsWritten()->isIntegerType()) {
      Arg2Expr = NamedCaster->getSubExpr();
    } else if (DpctGlobalInfo::getUnqualifiedTypeName(
                   NamedCaster->getTypeAsWritten()) == "cudaMemoryAdvise" &&
               NamedCaster->getSubExpr()->getType()->isIntegerType()) {
      Arg2Expr = NamedCaster->getSubExpr();
    }
  }
  auto Arg0Str = ExprAnalysis::ref(C->getArg(0));
  auto Arg1Str = ExprAnalysis::ref(C->getArg(1));
  auto Arg3Str = ExprAnalysis::ref(C->getArg(3));

  std::string Arg2Str;
  if (Arg2Expr->getStmtClass() == Stmt::IntegerLiteralClass) {
    Arg2Str = "0";
  } else {
    Arg2Str = ExprAnalysis::ref(Arg2Expr);
  }

  if (Arg2Str == "0") {
    report(C->getBeginLoc(), Diagnostics::DEFAULT_MEM_ADVICE, false,
           " and was set to 0");
  } else {
    report(C->getBeginLoc(), Diagnostics::DEFAULT_MEM_ADVICE, false, "");
  }

  std::ostringstream OS;
  if (getStmtSpelling(C->getArg(3)) == "cudaCpuDeviceId") {
    OS << MapNames::getDpctNamespace() + "cpu_device()." +
              DpctGlobalInfo::getDeviceQueueName() + "().mem_advise("
       << Arg0Str << ", " << Arg1Str << ", " << Arg2Str << ")";
    emplaceTransformation(new ReplaceStmt(C, OS.str()));
    requestFeature(HelperFeatureEnum::device_ext);
    return;
  }
  OS << MapNames::getDpctNamespace() + "get_device(" << Arg3Str
     << ")." + DpctGlobalInfo::getDeviceQueueName() + "().mem_advise("
     << Arg0Str << ", " << Arg1Str << ", " << Arg2Str << ")";
  emplaceTransformation(new ReplaceStmt(C, OS.str()));
  requestFeature(HelperFeatureEnum::device_ext);
}

// Memory migration rules live here.
void MemoryMigrationRule::registerMatcher(MatchFinder &MF) {
  auto memoryAPI = [&]() {
    return hasAnyName(
        "cudaMalloc", "cudaMemcpy", "cudaMemcpyAsync", "cudaMemcpyToSymbol",
        "cudaMemcpyToSymbolAsync", "cudaMemcpyFromSymbol",
        "cudaMemcpyFromSymbolAsync", "cudaFree", "cudaMemset",
        "cudaMemsetAsync", "cublasFree", "cublasAlloc", "cudaGetSymbolAddress",
        "cudaFreeHost", "cudaHostAlloc", "cudaHostGetDevicePointer",
        "cudaHostRegister", "cudaHostUnregister", "cudaMallocHost",
        "cudaMallocManaged", "cudaGetSymbolSize", "cudaMemPrefetchAsync",
        "cudaMalloc3D", "cudaMallocPitch", "cudaMemset2D", "cudaMemset3D",
        "cudaMemset2DAsync", "cudaMemset3DAsync", "cudaMemcpy2D",
        "cudaMemcpy3D", "cudaMemcpy2DAsync", "cudaMemcpy3DAsync",
        "cudaMemcpy2DArrayToArray", "cudaMemcpy2DToArray",
        "cudaMemcpy2DToArrayAsync", "cudaMemcpy2DFromArray",
        "cudaMemcpy2DFromArrayAsync", "cudaMemcpyArrayToArray",
        "cudaMemcpyToArray", "cudaMemcpyToArrayAsync", "cudaMemcpyFromArray",
        "cudaMemcpyFromArrayAsync", "cudaMallocArray", "cudaMalloc3DArray",
        "cudaFreeArray", "cudaArrayGetInfo", "cudaHostGetFlags",
        "cudaMemAdvise", "cuMemAdvise", "cudaGetChannelDesc", "cuMemHostAlloc",
        "cuMemFreeHost", "cuMemGetInfo_v2", "cuMemAlloc_v2", "cuMemcpyHtoD_v2",
        "cuMemcpyDtoH_v2", "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2",
        "cuMemcpy2D_v2", "cuMemcpy2DAsync_v2", "cuMemcpy3D_v2",
        "cuMemcpy3DAsync_v2", "cudaMemGetInfo", "cuMemAllocManaged",
        "cuMemAllocHost_v2", "cuMemHostGetDevicePointer_v2",
        "cuMemcpyDtoDAsync_v2", "cuMemcpyDtoD_v2", "cuMemAllocPitch_v2",
        "cuMemPrefetchAsync", "cuMemFree_v2", "cuDeviceTotalMem_v2",
        "cuMemHostGetFlags", "cuMemHostRegister_v2", "cuMemHostUnregister",
        "cuMemcpy", "cuMemcpyAsync", "cuMemcpyHtoA_v2", "cuMemcpyAtoH_v2",
        "cuMemcpyHtoAAsync_v2", "cuMemcpyAtoHAsync_v2", "cuMemcpyDtoA_v2",
        "cuMemcpyAtoD_v2", "cuMemcpyAtoA_v2", "cuMemsetD16_v2", "cuMemsetD16Async",
        "cuMemsetD2D16_v2", "cuMemsetD2D16Async", "cuMemsetD2D32_v2",
        "cuMemsetD2D32Async", "cuMemsetD2D8_v2", "cuMemsetD2D8Async",
        "cuMemsetD32_v2", "cuMemsetD32Async", "cuMemsetD8_v2",
        "cuMemsetD8Async");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(memoryAPI())), parentStmt()))
                    .bind("call"),
                this);

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(memoryAPI())), unless(parentStmt())))
          .bind("callUsed"),
      this);

  MF.addMatcher(
      unresolvedLookupExpr(
          hasAnyDeclaration(namedDecl(memoryAPI())),
          hasParent(callExpr(unless(parentStmt())).bind("callExprUsed")))
          .bind("unresolvedCallUsed"),
      this);

  MF.addMatcher(
      unresolvedLookupExpr(hasAnyDeclaration(namedDecl(memoryAPI())),
                           hasParent(callExpr(parentStmt()).bind("callExpr")))
          .bind("unresolvedCall"),
      this);
}

void MemoryMigrationRule::runRule(const MatchFinder::MatchResult &Result) {
  auto MigrateCallExpr = [&](const CallExpr *C, const bool IsAssigned,
                             const UnresolvedLookupExpr *ULExpr = NULL) {
    if (!C)
      return;

    std::string Name;
    if (ULExpr && C) {
      Name = ULExpr->getName().getAsString();
    } else {
      Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
    }
    if (MigrationDispatcher.find(Name) == MigrationDispatcher.end())
      return;

    // If there is a malloc function call in a template function, and the
    // template function is implicitly instantiated with two types. Then there
    // will be three FunctionDecl nodes in the AST. We should do replacement on
    // the FunctionDecl node which is not implicitly instantiated.
    auto &Context = dpct::DpctGlobalInfo::getContext();
    auto Parents = Context.getParents(*C);
    while (Parents.size() == 1) {
      auto *Parent = Parents[0].get<FunctionDecl>();
      if (Parent) {
        if (Parent->getTemplateSpecializationKind() ==
                TSK_ExplicitSpecialization ||
            Parent->getTemplateSpecializationKind() == TSK_Undeclared)
          break;
        else
          return;
      } else {
        Parents = Context.getParents(Parents[0]);
      }
    }

    MigrationDispatcher.at(Name)(Result, C, ULExpr, IsAssigned);
    // if API is removed, then no need to add (*, 0)
    // There are some cases where (*, 0) has already been added.
    // If the API is processed with rewriter in APINamesMemory.inc,
    // need to exclude the API from additional processing.
    if (IsAssigned && Name.compare("cudaHostRegister") &&
        Name.compare("cudaHostUnregister") && Name.compare("cudaMemAdvise") &&
        Name.compare("cudaArrayGetInfo") && Name.compare("cudaMalloc") &&
        Name.compare("cudaMallocPitch") && Name.compare("cudaMalloc3D") &&
        Name.compare("cublasAlloc") && Name.compare("cuMemGetInfo_v2") &&
        Name.compare("cudaHostAlloc") && Name.compare("cudaMallocHost") &&
        Name.compare("cuMemHostAlloc") && Name.compare("cudaMemGetInfo") &&
        Name.compare("cudaMallocManaged") &&
        Name.compare("cuMemAllocManaged") &&
        Name.compare("cuMemAllocHost_v2") &&
        Name.compare("cudaHostGetDevicePointer") &&
        Name.compare("cuMemHostGetDevicePointer_v2") &&
        Name.compare("cuMemcpyDtoDAsync_v2") &&
        Name.compare("cuMemcpyDtoD_v2") && Name.compare("cuMemAdvise") &&
        Name.compare("cuMemPrefetchAsync") &&
        Name.compare("cuMemcpyHtoDAsync_v2") &&
        Name.compare("cuMemcpyDtoD_v2") &&
        Name.compare("cuMemHostUnregister") &&
        Name.compare("cuMemHostRegister_v2") &&
        Name.compare("cudaHostGetFlags") && Name.compare("cuMemHostGetFlags") &&
        Name.compare("cuMemcpy") && Name.compare("cuMemcpyAsync") &&
        Name.compare("cuMemAllocPitch_v2")) {
      requestFeature(HelperFeatureEnum::device_ext);
      insertAroundStmt(C, "DPCT_CHECK_ERROR(", ")");
    } else if (IsAssigned && !Name.compare("cudaMemAdvise") &&
               USMLevel != UsmLevel::UL_None) {
      requestFeature(HelperFeatureEnum::device_ext);
      insertAroundStmt(C, "DPCT_CHECK_ERROR(", ")");
    } else if (IsAssigned && !Name.compare("cudaArrayGetInfo")) {
      requestFeature(HelperFeatureEnum::device_ext);
      std::string IndentStr =
          getIndent(C->getBeginLoc(), *Result.SourceManager).str();
      IndentStr += "  ";
      std::string PreStr{"DPCT_CHECK_ERROR([&](){"};
      PreStr += getNL();
      PreStr += IndentStr;
      std::string PostStr{";"};
      PostStr += getNL();
      PostStr += IndentStr;
      PostStr += "}())";
      insertAroundStmt(C, std::move(PreStr), std::move(PostStr));
    }
  };

  MigrateCallExpr(getAssistNodeAsType<CallExpr>(Result, "call"),
                  /* IsAssigned */ false);
  MigrateCallExpr(getAssistNodeAsType<CallExpr>(Result, "callUsed"),
                  /* IsAssigned */ true);
  MigrateCallExpr(
      getAssistNodeAsType<CallExpr>(Result, "callExprUsed"),
      /* IsAssigned */ true,
      getAssistNodeAsType<UnresolvedLookupExpr>(Result, "unresolvedCallUsed"));

  MigrateCallExpr(
      getAssistNodeAsType<CallExpr>(Result, "callExpr"),
      /* IsAssigned */ false,
      getAssistNodeAsType<UnresolvedLookupExpr>(Result, "unresolvedCall"));
}

void MemoryMigrationRule::getSymbolAddressMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  // Here only handle ordinary variable name reference, for accessing the
  // address of something residing on the device directly from host side should
  // not be possible.
  std::string Replacement;
  ExprAnalysis EA;
  EA.analyze(C->getArg(0));
  auto StmtStrArg0 = EA.getReplacedString();
  EA.analyze(C->getArg(1));
  auto StmtStrArg1 = EA.getReplacedString();
  Replacement = "*(" + StmtStrArg0 + ")" + " = " + StmtStrArg1 + ".get_ptr()";
  requestFeature(HelperFeatureEnum::device_ext);
  emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
}

MemoryMigrationRule::MemoryMigrationRule() {
  std::map<
      std::string,
      std::function<void(MemoryMigrationRule *,
                         const ast_matchers::MatchFinder::MatchResult &,
                         const CallExpr *, const UnresolvedLookupExpr *, bool)>>
      Dispatcher{
          {"cudaMalloc", &MemoryMigrationRule::mallocMigration},
          {"cuMemAlloc_v2", &MemoryMigrationRule::mallocMigration},
          {"cudaHostAlloc", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocHost", &MemoryMigrationRule::mallocMigration},
          {"cuMemAllocHost_v2", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocManaged", &MemoryMigrationRule::mallocMigration},
          {"cuMemAllocManaged", &MemoryMigrationRule::mallocMigration},
          {"cublasAlloc", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocPitch", &MemoryMigrationRule::mallocMigration},
          {"cudaMalloc3D", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocArray", &MemoryMigrationRule::mallocMigration},
          {"cudaMalloc3DArray", &MemoryMigrationRule::mallocMigration},
          {"cudaMemcpy", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyHtoD_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyDtoH_v2", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpyAsync", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyDtoHAsync_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyHtoDAsync_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyDtoDAsync_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyDtoD_v2", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpyToSymbol", &MemoryMigrationRule::memcpySymbolMigration},
          {"cudaMemcpyToSymbolAsync",
           &MemoryMigrationRule::memcpySymbolMigration},
          {"cudaMemcpyFromSymbol", &MemoryMigrationRule::memcpySymbolMigration},
          {"cudaMemcpyFromSymbolAsync",
           &MemoryMigrationRule::memcpySymbolMigration},
          {"cudaMemcpy2D", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpy2D_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpy2DAsync_v2", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy3D", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpy3D_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpy3DAsync_v2", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy2DAsync", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy3DAsync", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy2DArrayToArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DFromArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DFromArrayAsync", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DToArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DToArrayAsync", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyArrayToArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyToArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyToArrayAsync", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyFromArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyFromArrayAsync", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyAtoH_v2", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyHtoA_v2", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyAtoHAsync_v2", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyHtoAAsync_v2", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyAtoD_v2", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyDtoA_v2", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyAtoA_v2", &MemoryMigrationRule::arrayMigration},
          {"cudaFree", &MemoryMigrationRule::freeMigration},
          {"cuMemFree_v2", &MemoryMigrationRule::freeMigration},
          {"cudaFreeArray", &MemoryMigrationRule::freeMigration},
          {"cudaFreeHost", &MemoryMigrationRule::freeMigration},
          {"cuMemFreeHost", &MemoryMigrationRule::freeMigration},
          {"cublasFree", &MemoryMigrationRule::freeMigration},
          {"cudaMemset", &MemoryMigrationRule::memsetMigration},
          {"cudaMemsetAsync", &MemoryMigrationRule::memsetMigration},
          {"cudaMemset2D", &MemoryMigrationRule::memsetMigration},
          {"cudaMemset2DAsync", &MemoryMigrationRule::memsetMigration},
          {"cudaMemset3D", &MemoryMigrationRule::memsetMigration},
          {"cudaMemset3DAsync", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD16_v2", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD16Async", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD2D16_v2", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD2D16Async", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD2D32_v2", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD2D32Async", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD2D8_v2", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD2D8Async", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD32_v2", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD32Async", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD8_v2", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD8Async", &MemoryMigrationRule::memsetMigration},
          {"cudaGetSymbolAddress",
           &MemoryMigrationRule::getSymbolAddressMigration},
          {"cudaGetSymbolSize", &MemoryMigrationRule::getSymbolSizeMigration},
          {"cudaHostGetDevicePointer", &MemoryMigrationRule::miscMigration},
          {"cuMemHostGetDevicePointer_v2", &MemoryMigrationRule::miscMigration},
          {"cudaHostRegister", &MemoryMigrationRule::miscMigration},
          {"cudaHostUnregister", &MemoryMigrationRule::miscMigration},
          {"cuMemHostRegister_v2", &MemoryMigrationRule::miscMigration},
          {"cuMemHostUnregister", &MemoryMigrationRule::miscMigration},
          {"cuMemHostGetFlags", &MemoryMigrationRule::miscMigration},
          {"cudaMemPrefetchAsync", &MemoryMigrationRule::prefetchMigration},
          {"cuMemPrefetchAsync", &MemoryMigrationRule::prefetchMigration},
          {"cudaArrayGetInfo", &MemoryMigrationRule::cudaArrayGetInfo},
          {"cudaHostGetFlags", &MemoryMigrationRule::miscMigration},
          {"cudaMemAdvise", &MemoryMigrationRule::cudaMemAdvise},
          {"cuMemAdvise", &MemoryMigrationRule::cudaMemAdvise},
          {"cudaGetChannelDesc", &MemoryMigrationRule::miscMigration},
          {"cuMemHostAlloc", &MemoryMigrationRule::mallocMigration},
          {"cuMemAllocPitch_v2", &MemoryMigrationRule::mallocMigration},
          {"cuMemGetInfo_v2", &MemoryMigrationRule::miscMigration},
          {"cudaMemGetInfo", &MemoryMigrationRule::miscMigration},
          {"cuDeviceTotalMem_v2", &MemoryMigrationRule::miscMigration},
          {"cuMemcpy", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyAsync", &MemoryMigrationRule::memcpyMigration}};

  for (auto &P : Dispatcher)
    MigrationDispatcher[P.first] =
        std::bind(P.second, this, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4);
}

/// Convert a raw pointer argument and a pitch argument to a dpct::pitched_data
/// constructor. If \p ExcludeSizeArg is true, the argument represents the
/// pitch size will not be included in the constructor.
/// e.g. (...data, pitch, ...) => (...dpct::pitched_data(data, pitch, pitch, 1),
/// ...).
/// If \p ExcludeSizeArg is true, e.g. (...data, ..., pitch, ...) =>
/// (...dpct::pitched_data(data, pitch, pitch, 1), ..., pitch, ...)
void MemoryMigrationRule::aggregatePitchedData(const CallExpr *C,
                                               size_t DataArgIndex,
                                               size_t SizeArgIndex,
                                               SourceManager &SM,
                                               bool ExcludeSizeArg) {
  if (C->getNumArgs() <= DataArgIndex || C->getNumArgs() <= SizeArgIndex)
    return;
  size_t EndArgIndex = SizeArgIndex;
  std::string PaddingArgs, SizeArg;
  llvm::raw_string_ostream PaddingOS(PaddingArgs);
  ArgumentAnalysis A(C->getArg(SizeArgIndex), false);
  A.analyze();
  SizeArg = A.getReplacedString();
  if (ExcludeSizeArg) {
    PaddingOS << ", " << SizeArg;
    EndArgIndex = DataArgIndex;
  }
  PaddingOS << ", " << SizeArg << ", 1";
  aggregateArgsToCtor(C, MapNames::getDpctNamespace() + "pitched_data",
                      DataArgIndex, EndArgIndex, PaddingOS.str(), SM);
  requestFeature(HelperFeatureEnum::device_ext);
}

/// Convert several arguments to a constructor of class \p ClassName.
/// e.g. (...width, height, ...) => (...sycl::range<3>(width, height, 1), ...)
void MemoryMigrationRule::aggregateArgsToCtor(
    const CallExpr *C, const std::string &ClassName, size_t StartArgIndex,
    size_t EndArgIndex, const std::string &PaddingArgs, SourceManager &SM) {
  auto EndLoc = getStmtExpansionSourceRange(C->getArg(EndArgIndex)).getEnd();
  EndLoc = EndLoc.getLocWithOffset(Lexer::MeasureTokenLength(
      EndLoc, SM, DpctGlobalInfo::getContext().getLangOpts()));
  insertAroundRange(
      getStmtExpansionSourceRange(C->getArg(StartArgIndex)).getBegin(), EndLoc,
      ClassName + "(", PaddingArgs + ")");
}

/// Convert several arguments to a 3D vector constructor, like id<3> or
/// range<3>.
/// e.g. (...width, height, ...) => (...sycl::range<3>(width, height, 1), ...)
void MemoryMigrationRule::aggregate3DVectorClassCtor(
    const CallExpr *C, StringRef ClassName, size_t StartArgIndex,
    StringRef DefaultValue, SourceManager &SM, size_t ArgsNum) {
  if (C->getNumArgs() <= StartArgIndex + ArgsNum - 1)
    return;
  std::string Class, Padding;
  llvm::raw_string_ostream ClassOS(Class), PaddingOS(Padding);
  ClassOS << MapNames::getClNamespace();
  DpctGlobalInfo::printCtadClass(ClassOS, ClassName, 3);
  for (size_t i = 0; i < 3 - ArgsNum; ++i) {
    PaddingOS << ", " << DefaultValue;
  }
  aggregateArgsToCtor(C, ClassOS.str(), StartArgIndex,
                      StartArgIndex + ArgsNum - 1, PaddingOS.str(), SM);
}

void MemoryMigrationRule::handleDirection(const CallExpr *C, unsigned i) {
  if (C->getNumArgs() > i && !C->getArg(i)->isDefaultArgument()) {
    if (auto DRE = dyn_cast<DeclRefExpr>(C->getArg(i))) {
      if (auto Enum = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
        auto &ReplaceDirection = MapNames::findReplacedName(
            EnumConstantRule::EnumNamesMap, Enum->getName().str());
        if (!ReplaceDirection.empty()) {
          emplaceTransformation(new ReplaceStmt(DRE, ReplaceDirection));
          requestHelperFeatureForEnumNames(Enum->getName().str());
        }
      }
    }
  }
}

void MemoryMigrationRule::handleAsync(const CallExpr *C, unsigned i,
                                      const MatchFinder::MatchResult &Result) {
  if (C->getNumArgs() > i && !C->getArg(i)->isDefaultArgument()) {
    auto StreamExpr = C->getArg(i)->IgnoreImplicitAsWritten();
    if (isDefaultStream(StreamExpr)) {
      emplaceTransformation(removeArg(C, i, *Result.SourceManager));
      return;
    }
    emplaceTransformation(new InsertBeforeStmt(StreamExpr, "*"));
    if (!isa<DeclRefExpr>(StreamExpr)) {
      insertAroundStmt(StreamExpr, "(", ")");
    }
  }
}

REGISTER_RULE(MemoryMigrationRule, PassKind::PK_Migration)

const Expr *getRhs(const Stmt *);
TextModification *ReplaceMemberAssignAsSetMethod(SourceLocation EndLoc,
                                                 const MemberExpr *ME,
                                                 StringRef MethodName,
                                                 StringRef ReplacedArg,
                                                 StringRef ExtraArg = "") {
  return new ReplaceToken(
      ME->getMemberLoc(), EndLoc,
      buildString("set", MethodName.empty() ? "" : "_", MethodName, "(",
                  ExtraArg, ExtraArg.empty() ? "" : ", ", ReplacedArg, ")"));
}

TextModification *ReplaceMemberAssignAsSetMethod(const Expr *E,
                                                 const MemberExpr *ME,
                                                 StringRef MethodName,
                                                 StringRef ReplacedArg = "",
                                                 StringRef ExtraArg = "") {
  if (ReplacedArg.empty()) {
    if (auto RHS = getRhs(E)) {
      return ReplaceMemberAssignAsSetMethod(
          getStmtExpansionSourceRange(E).getEnd(), ME, MethodName,
          ExprAnalysis::ref(RHS), ExtraArg);
    }
  }
  return ReplaceMemberAssignAsSetMethod(getStmtExpansionSourceRange(E).getEnd(),
                                        ME, MethodName, ReplacedArg, ExtraArg);
}

void MemoryDataTypeRule::emplaceCuArrayDescDeclarations(const VarDecl *VD) {
  if (DpctGlobalInfo::isCommentsEnabled()) {
    emplaceTransformation(ReplaceVarDecl::getVarDeclReplacement(
        VD, "// These variables are defined for info of image_matrix."));
  }
  emplaceParamDecl(VD, "size_t", false, "0", "x", "y");
  emplaceParamDecl(VD, "unsigned", false, "0", "channel_num");
  emplaceParamDecl(VD, MapNames::getClNamespace() + "image_channel_type", false,
                   "0", "channel_type");
}

void MemoryDataTypeRule::emplaceMemcpy3DDeclarations(const VarDecl *VD,
                                                     bool hasDirection) {
  if (DpctGlobalInfo::isCommentsEnabled()) {
    emplaceTransformation(ReplaceVarDecl::getVarDeclReplacement(
        VD, "// These variables are defined for 3d matrix memory copy."));
  }
  emplaceParamDecl(VD, MapNames::getDpctNamespace() + "pitched_data", false,
                   "0", "from_data", "to_data");
  requestFeature(HelperFeatureEnum::device_ext);
  emplaceParamDecl(VD, getCtadType("id"), true, "0", "from_pos", "to_pos");
  emplaceParamDecl(VD, getCtadType("range"), true, "1", "size");
  if (hasDirection) {
    emplaceParamDecl(VD, MapNames::getDpctNamespace() + "memcpy_direction",
                     false, "0", "direction");
    requestFeature(HelperFeatureEnum::device_ext);
  }
}

std::string MemoryDataTypeRule::getMemcpy3DArguments(StringRef BaseName,
                                                     bool hasDirection) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  printParamName(OS, BaseName, "to_data") << ", ";
  printParamName(OS, BaseName, "to_pos") << ", ";
  printParamName(OS, BaseName, "from_data") << ", ";
  printParamName(OS, BaseName, "from_pos") << ", ";
  if (hasDirection) {
    printParamName(OS, BaseName, "size") << ", ";
    printParamName(OS, BaseName, "direction");
  } else {
    printParamName(OS, BaseName, "size");
  }
  return OS.str();
}

void MemoryDataTypeRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(varDecl(hasType(namedDecl(hasAnyName(
                            "cudaMemcpy3DParms", "CUDA_ARRAY_DESCRIPTOR",
                            "CUDA_MEMCPY3D", "CUDA_MEMCPY2D"))))
                    .bind("decl"),
                this);
  MF.addMatcher(
      memberExpr(hasObjectExpression(declRefExpr(hasType(namedDecl(
                     hasAnyName("cudaMemcpy3DParms", "CUDA_ARRAY_DESCRIPTOR",
                                "CUDA_MEMCPY3D", "CUDA_MEMCPY2D"))))))
          .bind("parmsMember"),
      this);
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(recordDecl(hasAnyName(
                               "cudaExtent", "cudaPos", "cudaPitchedPtr")))))
                    .bind("otherMember"),
                this);
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName("make_cudaExtent", "make_cudaPos",
                                              "make_cudaPitchedPtr"))))
          .bind("makeData"),
      this);
}

std::string MemoryDataTypeRule::getPitchMemberSetter(StringRef BaseName,
                                                     const std::string &Member,
                                                     const Expr *E) {
  std::string BaseStr;
  llvm::raw_string_ostream OS(BaseStr);
  auto Itr = MemberNames.find(Member);
  auto Itr2 = PitchMemberToSetter.find(Member);
  if (Itr != MemberNames.end() && Itr2 != PitchMemberToSetter.end()) {
    requestFeature(PitchMemberToFeature.at(Member));
    printParamName(OS, BaseName, Itr->second);
    OS.flush();
    MemberCallPrinter<std::string, std::string, const Expr *> MCP(
        BaseStr, false, Itr2->second, std::move(E));
    std::string ResultStr;
    llvm::raw_string_ostream OS2(ResultStr);
    MCP.print(OS2);
    OS2.flush();
    return OS2.str();
  }
  return "";
}

std::string MemoryDataTypeRule::getSizeOrPosMember(StringRef BaseName,
                                                   const std::string &Member) {
  std::string ResultStr;
  llvm::raw_string_ostream OS(ResultStr);
  auto Itr = SizeOrPosToMember.find(Member);
  if (Itr != SizeOrPosToMember.end()) {
    if (Member.rfind("src", 0) == 0)
      printParamName(OS, BaseName, "from_pos");
    else if (Member.rfind("dst", 0) == 0)
      printParamName(OS, BaseName, "to_pos");
    else
      printParamName(OS, BaseName, "size");
    OS << Itr->second;
    OS.flush();
    return ResultStr;
  }
  return "";
}

void MemoryDataTypeRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto VD = getNodeAsType<VarDecl>(Result, "decl")) {
    if (isa<ParmVarDecl>(VD))
      return;
    auto TypeName = DpctGlobalInfo::getUnqualifiedTypeName(VD->getType());
    if (TypeName == "cudaMemcpy3DParms")
      emplaceMemcpy3DDeclarations(VD, true);
    else if (TypeName == "CUDA_ARRAY_DESCRIPTOR")
      emplaceCuArrayDescDeclarations(VD);
    else
      emplaceMemcpy3DDeclarations(VD, false);
  } else if (auto ME = getNodeAsType<MemberExpr>(Result, "parmsMember")) {
    if (auto BO = DpctGlobalInfo::findAncestor<BinaryOperator>(ME)) {
      if (BO->getOpcode() == BinaryOperatorKind::BO_Assign &&
          ME == BO->getLHS()) {
        std::string QualName =
            DpctGlobalInfo::getUnqualifiedTypeName(ME->getType());
        if (QualName == "cudaArray_t" || QualName == "CUarray") {
          requestFeature(HelperFeatureEnum::device_ext);
          emplaceTransformation(
              new InsertAfterStmt(BO->getRHS(), "->to_pitched_data()"));
        } else if (QualName == "CUarray_st") {
          requestFeature(HelperFeatureEnum::device_ext);
          emplaceTransformation(
              new InsertAfterStmt(BO->getRHS(), ".to_pitched_data()"));
        } else if (auto DRE = dyn_cast<DeclRefExpr>(
                       ME->getBase()->IgnoreImplicitAsWritten())) {
          // if the member expr need to be removed
          if (isRemove(ME->getMemberDecl()->getName().str())) {
            emplaceTransformation(new ReplaceStmt(BO, ""));
            return;
          }
          // if the member expr need to be migrated to setter of
          // dpct::pitched_data
          std::string SetterStr = getPitchMemberSetter(
              DRE->getDecl()->getName(), ME->getMemberDecl()->getName().str(),
              BO->getRHS());
          if (!SetterStr.empty()) {
            emplaceTransformation(new ReplaceStmt(BO, SetterStr));
            return;
          }
          // if the member expr need to be migrated to pos or size assignment

          auto BaseName =
              DpctGlobalInfo::getUnqualifiedTypeName(ME->getBase()->getType());
          if (BaseName == "CUDA_MEMCPY2D" || BaseName == "CUDA_MEMCPY3D") {
            std::string SizeOrPosStr =
                getSizeOrPosMember(DRE->getDecl()->getName(),
                                   ME->getMemberDecl()->getName().str());
            if (!SizeOrPosStr.empty()) {
              emplaceTransformation(
                  new ReplaceStmt(BO->getLHS(), SizeOrPosStr));
              return;
            }
          }
        }
      }
    }
    if (auto DRE =
            dyn_cast<DeclRefExpr>(ME->getBase()->IgnoreImplicitAsWritten())) {
      emplaceTransformation(new ReplaceStmt(
          ME, getMemberName(DRE->getDecl()->getName(),
                            ME->getMemberDecl()->getName().str())));
    }
  } else if (auto CE = getNodeAsType<CallExpr>(Result, "makeData")) {
    if (auto FD = CE->getDirectCallee()) {
      auto Name = FD->getName();
      std::string ReplaceName;
      if (Name == "make_cudaExtent") {
        ReplaceName = DpctGlobalInfo::getCtadClass(
            MapNames::getClNamespace() + "range", 3);
      } else if (Name == "make_cudaPos") {
        ReplaceName =
            DpctGlobalInfo::getCtadClass(MapNames::getClNamespace() + "id", 3);
      } else if (Name == "make_cudaPitchedPtr") {
        ReplaceName = MapNames::getDpctNamespace() + "pitched_data";
        requestFeature(HelperFeatureEnum::device_ext);
      } else {
        DpctDiags() << "Unexpected function name [" << Name
                    << "] in MemoryDataTypeRule";
        return;
      }
      emplaceTransformation(new ReplaceCalleeName(CE, std::move(ReplaceName)));
    }
  } else if (auto M = getNodeAsType<MemberExpr>(Result, "otherMember")) {
    auto BaseName =
        DpctGlobalInfo::getUnqualifiedTypeName(M->getBase()->getType());
    auto MemberName = M->getMemberDecl()->getName();
    if (BaseName == "cudaPos") {
      auto &Replace = MapNames::findReplacedName(MapNames::Dim3MemberNamesMap,
                                                 MemberName.str());
      if (!Replace.empty())
        emplaceTransformation(new ReplaceToken(
            M->getOperatorLoc(), M->getEndLoc(), std::string(Replace)));
    } else if (BaseName == "cudaExtent") {
      auto &Replace =
          MapNames::findReplacedName(ExtentMemberNames, MemberName.str());
      if (!Replace.empty())
        emplaceTransformation(new ReplaceToken(
            M->getOperatorLoc(), M->getEndLoc(), std::string(Replace)));
    } else if (BaseName == "cudaPitchedPtr") {
      auto &Replace =
          MapNames::findReplacedName(PitchMemberNames, MemberName.str());
      if (Replace.empty())
        return;
      static const std::unordered_map<std::string, HelperFeatureEnum>
          PitchMemberNameToSetFeatureMap = {
              {"pitch", HelperFeatureEnum::device_ext},
              {"ptr", HelperFeatureEnum::device_ext},
              {"xsize", HelperFeatureEnum::device_ext},
              {"ysize", HelperFeatureEnum::device_ext}};
      static const std::unordered_map<std::string, HelperFeatureEnum>
          PitchMemberNameToGetFeatureMap = {
              {"pitch", HelperFeatureEnum::device_ext},
              {"ptr", HelperFeatureEnum::device_ext},
              {"xsize", HelperFeatureEnum::device_ext},
              {"ysize", HelperFeatureEnum::device_ext}};
      if (auto BO = DpctGlobalInfo::findParent<BinaryOperator>(M)) {
        if (BO->getOpcode() == BO_Assign) {
          requestFeature(PitchMemberNameToSetFeatureMap.at(MemberName.str()));
          emplaceTransformation(ReplaceMemberAssignAsSetMethod(BO, M, Replace));
          return;
        }
      }
      emplaceTransformation(new ReplaceToken(
          M->getMemberLoc(), buildString("get_", Replace, "()")));
      requestFeature(PitchMemberNameToGetFeatureMap.at(MemberName.str()));
    }
  }
}

REGISTER_RULE(MemoryDataTypeRule, PassKind::PK_Migration)

void UnnamedTypesRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(typedefDecl(hasDescendant(loc(recordType(hasDeclaration(
                    cxxRecordDecl(unless(anyOf(has(cxxRecordDecl(isImplicit())),
                                               isImplicit())),
                                  hasDefinition())
                        .bind("unnamedType")))))),
                this);
}

void UnnamedTypesRule::runRule(const MatchFinder::MatchResult &Result) {
  auto D = getNodeAsType<CXXRecordDecl>(Result, "unnamedType");
  if (D && D->getName().empty())
    emplaceTransformation(new InsertClassName(D));
}

REGISTER_RULE(UnnamedTypesRule, PassKind::PK_Migration)

void CMemoryAPIRule::registerMatcher(MatchFinder &MF) {
  auto cMemoryAPI = [&]() { return hasAnyName("calloc", "realloc", "malloc"); };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(cMemoryAPI())),
                     hasParent(implicitCastExpr().bind("implicitCast")))),
      this);
}

void CMemoryAPIRule::runRule(const MatchFinder::MatchResult &Result) {
  auto ICE = getNodeAsType<ImplicitCastExpr>(Result, "implicitCast");
  if (!ICE)
    return;

  emplaceTransformation(new InsertText(
      ICE->getBeginLoc(),
      "(" + DpctGlobalInfo::getReplacedTypeName(ICE->getType()) + ")"));
}

REGISTER_RULE(CMemoryAPIRule, PassKind::PK_Migration)

void GuessIndentWidthRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      functionDecl(allOf(hasParent(translationUnitDecl()),
                         hasBody(compoundStmt(unless(anyOf(
                             statementCountIs(0), statementCountIs(1)))))))
          .bind("FunctionDecl"),
      this);
  MF.addMatcher(
      cxxMethodDecl(hasParent(cxxRecordDecl(hasParent(translationUnitDecl()))))
          .bind("CXXMethodDecl"),
      this);
  MF.addMatcher(
      fieldDecl(hasParent(cxxRecordDecl(hasParent(translationUnitDecl()))))
          .bind("FieldDecl"),
      this);
}

void GuessIndentWidthRule::runRule(const MatchFinder::MatchResult &Result) {
  if (DpctGlobalInfo::getGuessIndentWidthMatcherFlag())
    return;
  SourceManager &SM = DpctGlobalInfo::getSourceManager();
  // Case 1:
  // TranslationUnitDecl
  // `-FunctionDecl
  //   `-CompoundStmt
  //     |-Stmt_1
  //     |-Stmt_2
  //     ...
  //     |-Stmt_n-1
  //     `-Stmt_n
  // The stmt in the compound stmt should >= 2, then we use the indent of the
  // first stmt as IndentWidth.
  auto FD = getNodeAsType<FunctionDecl>(Result, "FunctionDecl");
  if (FD) {
    CompoundStmt *CS = nullptr;
    Stmt *S = nullptr;
    if ((CS = dyn_cast<CompoundStmt>(FD->getBody())) &&
        (!CS->children().empty()) && (S = *(CS->children().begin()))) {
      DpctGlobalInfo::setIndentWidth(
          getIndent(SM.getExpansionLoc(S->getBeginLoc()), SM).size());
      DpctGlobalInfo::setGuessIndentWidthMatcherFlag(true);
      return;
    }
  }

  // Case 2:
  // TranslationUnitDecl
  // `-CXXRecordDecl
  //   |-CXXRecordDecl
  //   `-CXXMethodDecl
  // Use the indent of the CXXMethodDecl as the IndentWidth.
  auto CMD = getNodeAsType<CXXMethodDecl>(Result, "CXXMethodDecl");
  if (CMD) {
    DpctGlobalInfo::setIndentWidth(
        getIndent(SM.getExpansionLoc(CMD->getBeginLoc()), SM).size());
    DpctGlobalInfo::setGuessIndentWidthMatcherFlag(true);
    return;
  }

  // Case 3:
  // TranslationUnitDecl
  // `-CXXRecordDecl
  //   |-CXXRecordDecl
  //   `-FieldDecl
  // Use the indent of the FieldDecl as the IndentWidth.
  auto FieldD = getNodeAsType<FieldDecl>(Result, "FieldDecl");
  if (FieldD) {
    DpctGlobalInfo::setIndentWidth(
        getIndent(SM.getExpansionLoc(FieldD->getBeginLoc()), SM).size());
    DpctGlobalInfo::setGuessIndentWidthMatcherFlag(true);
    return;
  }
}

REGISTER_RULE(GuessIndentWidthRule, PassKind::PK_Migration)

void MathFunctionsRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> MathFunctionsCallExpr = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_OPERATOR(APINAME, OPKIND) APINAME,
#define ENTRY_TYPECAST(APINAME) APINAME,
#define ENTRY_UNSUPPORTED(APINAME) APINAME,
#define ENTRY_REWRITE(APINAME) APINAME,
#include "APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_NO_REWRITE
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
#undef ENTRY_REWRITE
  };

  std::vector<std::string> MathFunctionsUnresolvedLookupExpr = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME)
#define ENTRY_UNSUPPORTED(APINAME)
#define ENTRY_REWRITE(APINAME) APINAME,
#include "APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_NO_REWRITE
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
#undef ENTRY_REWRITE
  };

  MF.addMatcher(
      callExpr(callee(functionDecl(
                   internal::Matcher<NamedDecl>(
                       new internal::HasNameMatcher(MathFunctionsCallExpr)),
                   anyOf(unless(hasDeclContext(namespaceDecl(anything()))),
                         hasDeclContext(namespaceDecl(hasName("std")))))),
               unless(hasAncestor(
                   cxxConstructExpr(hasType(typedefDecl(hasName("dim3")))))))
          .bind("math"),
      this);

  MF.addMatcher(
      callExpr(callee(unresolvedLookupExpr(hasAnyDeclaration(namedDecl(
                   internal::Matcher<NamedDecl>(new internal::HasNameMatcher(
                       MathFunctionsUnresolvedLookupExpr)))))))
          .bind("unresolved"),
      this);
}

void MathFunctionsRule::runRule(const MatchFinder::MatchResult &Result) {
   const CallExpr *CE = getAssistNodeAsType<CallExpr>(Result, "math");
   if (!CE)
     CE = getNodeAsType<CallExpr>(Result, "unresolved");
   if (!CE)
     return;

  ExprAnalysis EA(CE);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();

  auto FD = CE->getDirectCallee();
  // For CUDA file, nvcc can include math header files implicitly.
  // So we need add the cmath header file if the API is not from SDK
  // header.
  bool NeedInsertCmath = false;
  if (FD) {
    std::string Name = FD->getNameInfo().getName().getAsString();
    if (Name == "__brev" || Name == "__brevll") {
      requestFeature(HelperFeatureEnum::device_ext);
    } else if (Name == "__byte_perm") {
      requestFeature(HelperFeatureEnum::device_ext);
    } else if (Name == "__ffs" || Name == "__ffsll") {
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (!math::IsDefinedInCUDA()(CE)) {
      NeedInsertCmath = true;
    }
  } else {
    NeedInsertCmath = true;
  }
  if (NeedInsertCmath) {
    DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), HT_Math);
  }
}

REGISTER_RULE(MathFunctionsRule, PassKind::PK_Migration)

void WarpFunctionsRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> WarpFunctions = {"__reduce_add_sync",
                                            "__reduce_min_sync",
                                            "__reduce_and_sync",
                                            "__reduce_or_sync",
                                            "__reduce_xor_sync",
                                            "__reduce_max_sync",
                                            "__shfl_up_sync",
                                            "__shfl_down_sync",
                                            "__shfl_sync",
                                            "__shfl_up",
                                            "__shfl_down",
                                            "__shfl",
                                            "__shfl_xor",
                                            "__shfl_xor_sync",
                                            "__all",
                                            "__all_sync",
                                            "__any",
                                            "__any_sync",
                                            "__ballot",
                                            "__ballot_sync",
                                            "__match_any_sync",
                                            "__match_all_sync",
                                            "__activemask"};

  MF.addMatcher(callExpr(callee(functionDecl(internal::Matcher<NamedDecl>(
                             new internal::HasNameMatcher(WarpFunctions)))),
                         hasAncestor(functionDecl().bind("ancestor")))
                    .bind("warp"),
                this);
}

void WarpFunctionsRule::runRule(const MatchFinder::MatchResult &Result) {
  auto CE = getNodeAsType<CallExpr>(Result, "warp");
  if (!CE)
    return;

  if (auto *CalleeDecl = CE->getDirectCallee()) {
    if (isUserDefinedDecl(CalleeDecl)) {
      return;
    }
  }

  ExprAnalysis EA(CE);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}
REGISTER_RULE(WarpFunctionsRule, PassKind::PK_Analysis)

void CooperativeGroupsFunctionRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> CGAPI;
  CGAPI.insert(CGAPI.end(), MapNames::CooperativeGroupsAPISet.begin(),
               MapNames::CooperativeGroupsAPISet.end());
  MF.addMatcher(
      callExpr(
          allOf(callee(functionDecl(
                    internal::Matcher<NamedDecl>(
                        new internal::HasNameMatcher(CGAPI)),
                    hasAncestor(namespaceDecl(hasName("cooperative_groups"))))),
                hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                               hasAttr(attr::CUDAGlobal))))))
          .bind("FuncCall"),
      this);
  MF.addMatcher(
      declRefExpr(
          hasAncestor(
              implicitCastExpr(
                  hasImplicitDestinationType(qualType(hasCanonicalType(
                      recordType(hasDeclaration(cxxRecordDecl(hasName(
                          "cooperative_groups::__v1::thread_group"))))))))))
          .bind("declRef"),
      this);
}

void CooperativeGroupsFunctionRule::runRule(
    const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FuncCall");
  const DeclRefExpr *DR = getNodeAsType<DeclRefExpr>(Result, "declRef");
  const SourceManager &SM = DpctGlobalInfo::getSourceManager();
  if (DR && DpctGlobalInfo::useLogicalGroup()) {
    std::string ReplacedStr = MapNames::getDpctNamespace() + "experimental::group" +
                  "(" + DR->getNameInfo().getAsString() + ", " +
                  DpctGlobalInfo::getItem(DR) + ")";
    SourceRange DefRange = getDefinitionRange(DR->getBeginLoc(),  DR->getEndLoc());
    SourceLocation Begin = DefRange.getBegin();
    SourceLocation End = DefRange.getEnd();
    End = End.getLocWithOffset(Lexer::MeasureTokenLength(
        End, SM, DpctGlobalInfo::getContext().getLangOpts()));
    emplaceTransformation(replaceText(Begin, End, std::move(ReplacedStr),
                                      DpctGlobalInfo::getSourceManager()));
    return;
  }
  if (!CE)
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  struct ReportUnsupportedWarning {
    ReportUnsupportedWarning(SourceLocation SL, std::string FunctionName,
                             CooperativeGroupsFunctionRule *ThisPtrOfRule)
        : SL(SL), FunctionName(FunctionName), ThisPtrOfRule(ThisPtrOfRule) {}
    ~ReportUnsupportedWarning() {
      if (NeedReport) {
        ThisPtrOfRule->report(SL, Diagnostics::API_NOT_MIGRATED, true,
                              FunctionName);
      }
    }
    bool NeedReport = true;
  private:
    SourceLocation SL;
    std::string FunctionName;
    CooperativeGroupsFunctionRule *ThisPtrOfRule = nullptr;
  };

  ReportUnsupportedWarning RUW(CE->getBeginLoc(), FuncName, this);

  if (FuncName == "sync" || FuncName == "thread_rank" || FuncName == "size" ||
      FuncName == "shfl_down" || FuncName == "shfl_up" ||
      FuncName == "shfl_xor" || FuncName == "meta_group_rank" ||
      FuncName == "reduce" || FuncName == "thread_index" ||
      FuncName == "group_index" || FuncName == "num_threads") {
    // There are 3 usages of cooperative groups APIs.
    // 1. cg::thread_block tb; tb.sync(); // member function
    // 2. cg::thread_block tb; cg::sync(tb); // free function
    // 3. cg::thread_block::sync(); // static function
    // Value meaning: is_migration_support/is_original_code_support
    // FunctionName  Case1 Case2 Case3
    // sync          1/1   1/1   0/1
    // thread_rank   1/1   1/1   0/1
    // size          1/1   0/0   1/1
    // num_threads   1/1   0/0   1/1
    // shfl_down     1/1   0/0   0/0
    // shfl_up       1/1   0/0   0/0
    // shfl_xor      1/1   0/0   0/0
    // meta_group_rank 1/1   0/0   0/0
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    RUW.NeedReport = false;
  } else if (FuncName == "this_thread_block") {
    if (auto P = getAncestorDeclStmt(CE)) {
      if (auto VD = dyn_cast<VarDecl>(*P->decl_begin())) {
        emplaceTransformation(new ReplaceTypeInDecl(VD, "auto"));
      }
    }
    RUW.NeedReport = false;
    emplaceTransformation(
        new ReplaceStmt(CE, DpctGlobalInfo::getGroup(CE)));
  } else if (FuncName == "tiled_partition") {
    RUW.NeedReport = false;
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();

    CheckParamType Checker1(
        0, "const class cooperative_groups::__v1::thread_block &");
    CheckIntergerTemplateArgValueNE Checker2(0, 32);
    CheckIntergerTemplateArgValueLE Checker3(0, 32);
    if (Checker1(CE) && Checker3(CE)) {
      auto FuncInfo = DeviceFunctionDecl::LinkRedecls(
          DpctGlobalInfo::getParentFunction(CE));
      if (FuncInfo) {
        FuncInfo->getVarMap().Dim = 3;
        if (Checker2(CE) && DpctGlobalInfo::useLogicalGroup()) {
          FuncInfo->addSubGroupSizeRequest(32, CE->getBeginLoc(),
                                           MapNames::getDpctNamespace() +
                                               "experimental::logical_group");
        } else {
          FuncInfo->addSubGroupSizeRequest(32, CE->getBeginLoc(),
                                           DpctGlobalInfo::getSubGroup(CE));
        }
      }
    }
  }
}

#undef EMIT_WARNING_AND_RETURN
REGISTER_RULE(CooperativeGroupsFunctionRule, PassKind::PK_Analysis)

void SyncThreadsRule::registerMatcher(MatchFinder &MF) {
  auto SyncAPI = [&]() {
    return hasAnyName("__syncthreads", "__threadfence_block", "__threadfence",
                      "__threadfence_system", "__syncthreads_and",
                      "__syncthreads_or", "__syncthreads_count", "__syncwarp");
  };
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(SyncAPI())), parentStmt(),
                     hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                                    hasAttr(attr::CUDAGlobal)))
                                     .bind("FuncDecl"))))
          .bind("SyncFuncCall"),
      this);
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(SyncAPI())), unless(parentStmt()),
                     hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                                    hasAttr(attr::CUDAGlobal)))
                                     .bind("FuncDeclUsed"))))
          .bind("SyncFuncCallUsed"),
      this);
}

void SyncThreadsRule::runRule(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "SyncFuncCall");
  const FunctionDecl *FD =
      getAssistNodeAsType<FunctionDecl>(Result, "FuncDecl");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "SyncFuncCallUsed")))
      return;
    FD = getAssistNodeAsType<FunctionDecl>(Result, "FuncDeclUsed");
    IsAssigned = true;
  }
  if (!FD)
    return;

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  if (FuncName == "__syncthreads") {
    DpctGlobalInfo::registerNDItemUser(CE);
    const FunctionDecl *FD = nullptr;
    if (FD = getAssistNodeAsType<FunctionDecl>(Result, "FuncDecl")) {
      GroupFunctionCallInControlFlowAnalyzer A(DpctGlobalInfo::getContext());
      A.checkCallGroupFunctionInControlFlow(const_cast<FunctionDecl *>(FD));
      auto FnInfo = DeviceFunctionDecl::LinkRedecls(FD);
      auto CallInfo = FnInfo->addCallee(CE);
      if (CallInfo->hasSideEffects())
        report(CE->getBeginLoc(), Diagnostics::CALL_GROUP_FUNC_IN_COND, false);
    }
  } else if (FuncName == "this_thread_block") {
    if (auto P = getAncestorDeclStmt(CE)) {
      if (auto VD = dyn_cast<VarDecl>(*P->decl_begin())) {
        emplaceTransformation(new ReplaceTypeInDecl(VD, "auto"));
      }
    }
    emplaceTransformation(
        new ReplaceStmt(CE, DpctGlobalInfo::getGroup(CE, FD)));
  } else if (FuncName == "__threadfence_block") {
    std::string CLNS = MapNames::getClNamespace();
    std::string ReplStr = CLNS + "atomic_fence(" + CLNS +
                          "memory_order::acq_rel, " + CLNS +
                          "memory_scope::work_group" + ")";
    report(CE->getBeginLoc(), Diagnostics::MEMORY_ORDER_PERFORMANCE_TUNNING,
           true);
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "__threadfence") {
    std::string CLNS = MapNames::getClNamespace();
    std::string ReplStr = CLNS + "atomic_fence(" + CLNS +
                          "memory_order::acq_rel, " + CLNS +
                          "memory_scope::device" + ")";
    report(CE->getBeginLoc(), Diagnostics::MEMORY_ORDER_PERFORMANCE_TUNNING,
           true);
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "__threadfence_system") {
    std::string CLNS = MapNames::getClNamespace();
    std::string ReplStr = CLNS + "atomic_fence(" + CLNS +
                          "memory_order::acq_rel, " + CLNS +
                          "memory_scope::system" + ")";
    report(CE->getBeginLoc(), Diagnostics::MEMORY_ORDER_PERFORMANCE_TUNNING,
           true);
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "__syncthreads_and" ||
             FuncName == "__syncthreads_or" ||
             FuncName == "__syncthreads_count") {
    std::string ReplStr;
    if (IsAssigned) {
      ReplStr = "(";
      ReplStr += DpctGlobalInfo::getItem(CE) + ".barrier(), ";
    } else {
      ReplStr += DpctGlobalInfo::getItem(CE) + ".barrier();" + getNL();
      ReplStr += getIndent(CE->getBeginLoc(), *Result.SourceManager).str();
    }
    if (FuncName == "__syncthreads_and") {
      ReplStr += MapNames::getClNamespace() + "all_of_group(";
    } else if (FuncName == "__syncthreads_or") {
      ReplStr += MapNames::getClNamespace() + "any_of_group(";
    } else {
      ReplStr += MapNames::getClNamespace() + "reduce_over_group(";
    }
    ReplStr += DpctGlobalInfo::getGroup(CE) + ", ";
    if (FuncName == "__syncthreads_count") {
      ReplStr += ExprAnalysis::ref(CE->getArg(0)) + " == 0 ? 0 : 1, " +
                 MapNames::getClNamespace() + "ext::oneapi::plus<>()";
    } else {
      ReplStr += ExprAnalysis::ref(CE->getArg(0));
    }

    ReplStr += ")";
    if (IsAssigned)
      ReplStr += ")";
    report(CE->getBeginLoc(), Diagnostics::BARRIER_PERFORMANCE_TUNNING, true,
           "nd_item");
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "__syncwarp") {
    std::string ReplStr;
    ReplStr = MapNames::getClNamespace() + "group_barrier(" +
              DpctGlobalInfo::getSubGroup(CE) + ")";
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  }
}

REGISTER_RULE(SyncThreadsRule, PassKind::PK_Analysis)

void SyncThreadsMigrationRule::registerMatcher(MatchFinder &MF) {
  auto SyncAPI = [&]() { return hasAnyName("__syncthreads"); };
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(SyncAPI())), parentStmt(),
                     hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                                    hasAttr(attr::CUDAGlobal)))
                                     .bind("FuncDecl"))))
          .bind("SyncFuncCall"),
      this);
}

void SyncThreadsMigrationRule::runRule(const MatchFinder::MatchResult &Result) {
  static std::map<std::string, bool> LocationResultMapForTemplate;
  auto emplaceReplacement = [&](BarrierFenceSpaceAnalyzerResult Res,
                                const CallExpr *CE) {
    std::string Replacement;
    if (Res.CanUseLocalBarrier) {
      if (Res.MayDependOn1DKernel) {
        report(CE->getBeginLoc(), Diagnostics::ONE_DIMENSION_KERNEL_BARRIER,
               true, Res.GlobalFunctionName);
      }
      Replacement = DpctGlobalInfo::getItem(CE) + ".barrier(" +
                    MapNames::getClNamespace() +
                    "access::fence_space::local_space)";
    } else {
      report(CE->getBeginLoc(), Diagnostics::BARRIER_PERFORMANCE_TUNNING, true,
             "nd_item");
      Replacement = DpctGlobalInfo::getItem(CE) + ".barrier()";
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
  };

  const CallExpr *CE = getAssistNodeAsType<CallExpr>(Result, "SyncFuncCall");
  const FunctionDecl *FD =
      getAssistNodeAsType<FunctionDecl>(Result, "FuncDecl");
  if (!CE || !FD)
    return;

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  if (FuncName == "__syncthreads") {
    BarrierFenceSpaceAnalyzer A;
    const FunctionTemplateDecl *FTD = FD->getDescribedFunctionTemplate();
    if (FTD) {
      if (FTD->specializations().empty()) {
        emplaceReplacement(A.analyze(CE), CE);
      }
    } else {
      if (FD->getTemplateSpecializationKind() ==
          TemplateSpecializationKind::TSK_Undeclared) {
        emplaceReplacement(A.analyze(CE), CE);
      } else {
        auto CurRes = A.analyze(CE, true);
        std::string LocHash = getHashStrFromLoc(CE->getBeginLoc());
        auto Iter = LocationResultMapForTemplate.find(LocHash);
        if (Iter != LocationResultMapForTemplate.end()) {
          if (Iter->second != CurRes.CanUseLocalBarrier) {
            report(CE->getBeginLoc(),
                   Diagnostics::CANNOT_UNIFY_FUNCTION_CALL_IN_MACRO_OR_TEMPLATE,
                   false, FuncName);
          }
        } else {
          LocationResultMapForTemplate[LocHash] = CurRes.CanUseLocalBarrier;
          emplaceReplacement(CurRes, CE);
        }
      }
    }
  }
}

REGISTER_RULE(SyncThreadsMigrationRule, PassKind::PK_Migration)

void KernelFunctionInfoRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName("cudaFuncGetAttributes"))))
          .bind("call"),
      this);
  MF.addMatcher(callExpr(callee(functionDecl(hasAnyName("cuFuncGetAttribute"))))
                    .bind("callFuncGetAttribute"),
                this);
  MF.addMatcher(
      memberExpr(anyOf(has(implicitCastExpr(hasType(pointsTo(
                           recordDecl(hasName("cudaFuncAttributes")))))),
                       hasObjectExpression(
                           hasType(recordDecl(hasName("cudaFuncAttributes"))))))
          .bind("member"),
      this);
}

void KernelFunctionInfoRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto C = getNodeAsType<CallExpr>(Result, "call")) {
    if (isAssigned(C)) {
      emplaceTransformation(new ReplaceToken(
          C->getBeginLoc(), "DPCT_CHECK_ERROR(" + MapNames::getDpctNamespace() +
                                "get_kernel_function_info"));
      emplaceTransformation(new InsertAfterStmt(C, ")"));
    } else {
      emplaceTransformation(
          new ReplaceToken(C->getBeginLoc(), MapNames::getDpctNamespace() +
                                                 "get_kernel_function_info"));
    }
    requestFeature(HelperFeatureEnum::device_ext);
    auto FuncArg = C->getArg(1);
    emplaceTransformation(new InsertBeforeStmt(FuncArg, "(const void *)"));
  } else if (auto C = getNodeAsType<CallExpr>(Result, "callFuncGetAttribute")) {
    ExprAnalysis EA;
    EA.analyze(C);
    emplaceTransformation(EA.getReplacement());
  } else if (auto M = getNodeAsType<MemberExpr>(Result, "member")) {
    auto MemberName = M->getMemberNameInfo();
    auto NameMap = AttributesNamesMap.find(MemberName.getAsString());
    if (NameMap != AttributesNamesMap.end())
      emplaceTransformation(new ReplaceToken(MemberName.getBeginLoc(),
                                             std::string(NameMap->second)));
  }
}

REGISTER_RULE(KernelFunctionInfoRule, PassKind::PK_Migration)

std::vector<std::vector<std::string>>
RecognizeAPINameRule::splitAPIName(std::vector<std::string> &AllAPINames) {
  std::vector<std::vector<std::string>> Result;
  std::vector<std::string> FuncNames, FuncNamesHasNS, FuncNamespaces;
  size_t ScopeResolutionOpSize = 2; // The length of string("::")
  for (auto &APIName : AllAPINames) {
    size_t ScopeResolutionOpPos = APIName.rfind("::");
    // 1. FunctionName
    if (ScopeResolutionOpPos == std::string::npos) {
      FuncNames.emplace_back(APIName);
    } else {
      // 2. NameSpace::FunctionName
      if (std::find(FuncNamespaces.begin(), FuncNamespaces.end(),
                    APIName.substr(0, ScopeResolutionOpPos)) ==
          FuncNamespaces.end()) {
        FuncNamespaces.emplace_back(APIName.substr(0, ScopeResolutionOpPos));
      }
      FuncNamesHasNS.emplace_back(
          APIName.substr(ScopeResolutionOpPos + ScopeResolutionOpSize));
    }
  }
  return {FuncNames, FuncNamesHasNS, FuncNamespaces};
}

void RecognizeAPINameRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> AllAPINames = MigrationStatistics::GetAllAPINames();
  // AllAPIComponent[0] : FuncNames
  // AllAPIComponent[1] : FuncNamesHasNS
  // AllAPIComponent[2] : FuncNamespaces
  std::vector<std::vector<std::string>> AllAPIComponent =
      splitAPIName(AllAPINames);
  if (!AllAPIComponent[0].empty()) {
    MF.addMatcher(
        callExpr(
            allOf(callee(functionDecl(internal::Matcher<NamedDecl>(
                      new internal::HasNameMatcher(AllAPIComponent[0])))),
                  unless(hasAncestor(cudaKernelCallExpr())),
                  unless(callee(hasDeclContext(namedDecl(hasName("std")))))))
            .bind("APINamesUsed"),
        this);
  }

  if (!AllAPIComponent[1].empty() && !AllAPIComponent[2].empty()) {
    MF.addMatcher(
        callExpr(
            callee(functionDecl(allOf(
                namedDecl(internal::Matcher<NamedDecl>(
                    new internal::HasNameMatcher(AllAPIComponent[1]))),
                hasAncestor(
                    namespaceDecl(namedDecl(internal::Matcher<NamedDecl>(
                        new internal::HasNameMatcher(AllAPIComponent[2])))))))))
            .bind("APINamesHasNSUsed"),
        this);
  }
}

const std::string
RecognizeAPINameRule::getFunctionSignature(const FunctionDecl *Func,
                                           std::string ObjName) {
  std::string Buf;
  llvm::raw_string_ostream OS(Buf);
  OS << Func->getReturnType().getAsString() << " " << ObjName
     << Func->getQualifiedNameAsString() << "(";

  for (unsigned int Index = 0; Index < Func->getNumParams(); Index++) {
    if (Index > 0) {
      OS << ",";
    }
    OS << QualType::getAsString(Func->parameters()[Index]->getType().split(),
                                PrintingPolicy{{}})
       << " " << Func->parameters()[Index]->getQualifiedNameAsString();
  }
  OS << ")";
  return OS.str();
}

void RecognizeAPINameRule::processFuncCall(const CallExpr *CE) {
  const NamedDecl *ND;
  std::string Namespace = "";
  std::string ObjName = "";
  std::string APIName = "";
  if (dyn_cast<CXXOperatorCallExpr>(CE))
    return;
  if (auto MD = dyn_cast<CXXMemberCallExpr>(CE)) {
    QualType ObjType = MD->getImplicitObjectArgument()
                           ->IgnoreImpCasts()
                           ->getType()
                           .getCanonicalType();
    ND = getNamedDecl(ObjType.getTypePtr());
    if (!ND)
      return;
    ObjName = ND->getNameAsString();
  } else {
    // Match the static member function call, like: A a; a.staticCall();
    if (auto ME = dyn_cast<MemberExpr>(CE->getCallee()->IgnoreImpCasts())) {
      auto ObjType = ME->getBase()->getType().getCanonicalType();
      ND = getNamedDecl(ObjType.getTypePtr());
      ObjName = ND->getNameAsString();
    // Match the static call, like: A::staticCall();
    } else if (auto RT = dyn_cast<RecordDecl>(
                   CE->getCalleeDecl()->getDeclContext())) {
      ObjName = RT->getNameAsString();
      ND = dyn_cast<NamedDecl>(RT);
    } else {
      ND = dyn_cast<NamedDecl>(CE->getCalleeDecl());
    }
  }

  if (!dpct::DpctGlobalInfo::isInCudaPath(ND->getLocation()) &&
      !isChildOrSamePath(DpctInstallPath,
                         dpct::DpctGlobalInfo::getLocInfo(ND).first)) {
    if ( ND->getIdentifier() && !ND->getName().startswith("cudnn") && !ND->getName().startswith("nccl"))
      return;
  }

  auto *NSD = dyn_cast<NamespaceDecl>(ND->getDeclContext());
  Namespace = getNameSpace(NSD);
  APIName = CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  if (!ObjName.empty())
    APIName = ObjName + "::" + APIName;
  if (!Namespace.empty())
    APIName = Namespace + "::" + APIName;
  SrcAPIStaticsMap[getFunctionSignature(CE->getCalleeDecl()->getAsFunction(),
                                        "")]++;

  if (!MigrationStatistics::IsMigrated(APIName)) {
    const SourceManager &SM = DpctGlobalInfo::getSourceManager();
    const SourceLocation FileLoc = SM.getFileLoc(CE->getBeginLoc());

    std::string SLStr = FileLoc.printToString(SM);

    std::size_t PosCol = SLStr.rfind(':');
    std::size_t PosRow = SLStr.rfind(':', PosCol - 1);
    std::string FileName = SLStr.substr(0, PosRow);
    LOCStaticsMap[FileName][2]++;

    auto Iter = MapNames::ITFName.find(APIName.c_str());
    if (Iter != MapNames::ITFName.end())
      report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             Iter->second);
    else
      report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, APIName);
  }
}

void RecognizeAPINameRule::runRule(const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = nullptr;
  if ((CE = getNodeAsType<CallExpr>(Result, "APINamesUsed")) ||
      (CE = getNodeAsType<CallExpr>(Result, "APINamesHasNSUsed")))
    processFuncCall(CE);
}

REGISTER_RULE(RecognizeAPINameRule, PassKind::PK_Migration)

void RecognizeTypeRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto TypeTable = MigrationStatistics::GetTypeTable();
  std::vector<std::string> UnsupportedType;
  std::vector<std::string> UnsupportedPointerType;
  for (auto &Type : TypeTable) {
    if (!Type.second) {
      if (Type.first.find("*") != std::string::npos) {
        UnsupportedPointerType.push_back(
            Type.first.substr(0, Type.first.length() - 1));
      } else {
        UnsupportedType.push_back(Type.first);
      }
    }
  }
  MF.addMatcher(
      typeLoc(
          anyOf(loc(qualType(
                    hasDeclaration(namedDecl(internal::Matcher<NamedDecl>(
                        new internal::HasNameMatcher(UnsupportedType)))))),
                loc(pointerType(pointee(qualType(hasDeclaration(namedDecl(
                    internal::Matcher<NamedDecl>(new internal::HasNameMatcher(
                        UnsupportedPointerType))))))))))
          .bind("typeloc"),
      this);
}

void RecognizeTypeRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const TypeLoc *TL = getNodeAsType<TypeLoc>(Result, "typeloc");
  if (!TL)
    return;
  auto &Context = DpctGlobalInfo::getContext();
  QualType QTy = TL->getType();
  if (QTy.isCanonical())
    return;
  std::string TypeName =
      DpctGlobalInfo::getTypeName(QTy.getUnqualifiedType(), Context);
  // process pointer type
  if (!QTy->isTypedefNameType() && QTy->isPointerType()) {
    std::string PointeeTy = DpctGlobalInfo::getTypeName(
        QTy->getPointeeType().getUnqualifiedType(), Context);
    report(TL->getBeginLoc(), Diagnostics::KNOWN_UNSUPPORTED_TYPE, false,
           PointeeTy + " *");
    return;
  }
  report(TL->getBeginLoc(), Diagnostics::KNOWN_UNSUPPORTED_TYPE, false,
         TypeName);
}

REGISTER_RULE(RecognizeTypeRule, PassKind::PK_Migration)

void TextureMemberSetRule::registerMatcher(MatchFinder &MF) {
  auto ObjectType =
      hasObjectExpression(hasType(namedDecl(hasAnyName("cudaResourceDesc"))));
  // myres.res.array.array = a;
  auto AssignResArrayArray = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("array")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("array")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("ArrayMember"))))))))));
  // myres.resType = cudaResourceTypeArray;
  auto ArraySetCompound = binaryOperator(allOf(
      isAssignmentOperator(),
      hasLHS(memberExpr(allOf(ObjectType, member(hasName("resType"))))
                 .bind("ResTypeMemberExpr")),
      hasRHS(declRefExpr(
          hasDeclaration(enumConstantDecl(hasName("cudaResourceTypeArray"))))),
      hasParent(
          compoundStmt(has(AssignResArrayArray.bind("AssignResArrayArray"))))));
  MF.addMatcher(ArraySetCompound.bind("ArraySetCompound"), this);
  // myres.res.pitch2D.devPtr = p;
  auto AssignRes2DPtr = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("devPtr")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("pitch2D")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("PtrMember"))))))))));
  // myres.res.pitch2D.desc = desc42;
  auto AssignRes2DDesc = cxxOperatorCallExpr(
      allOf(isAssignmentOperator(),
            has(memberExpr(allOf(
                member(hasName("desc")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("pitch2D")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("DescMember"))))))))));
  // myres.res.pitch2D.width = sizeof(float4) * 32;
  auto AssignRes2DWidth = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("width")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("pitch2D")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("WidthMember"))))))))));
  // myres.res.pitch2D.height = 32;
  auto AssignRes2DHeight = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("height")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("pitch2D")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("HeightMember"))))))))));
  // myres.res.pitch2D.pitchInBytes = sizeof(float4) * 32;
  auto AssignRes2DPitchInBytes = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("pitchInBytes")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("pitch2D")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("PitchMember"))))))))));
  // myres.resType = cudaResourceTypePitch2D;
  auto Pitch2DSetCompound = binaryOperator(allOf(
      isAssignmentOperator(),
      hasLHS(memberExpr(allOf(ObjectType, member(hasName("resType"))))
                 .bind("ResTypeMemberExpr")),
      hasRHS(declRefExpr(hasDeclaration(
          enumConstantDecl(hasName("cudaResourceTypePitch2D"))))),
      hasParent(compoundStmt(allOf(
          has(AssignRes2DPtr.bind("AssignRes2DPtr")),
          has(AssignRes2DDesc.bind("AssignRes2DDesc")),
          has(AssignRes2DWidth.bind("AssignRes2DWidth")),
          has(AssignRes2DHeight.bind("AssignRes2DHeight")),
          has(AssignRes2DPitchInBytes.bind("AssignRes2DPitchInBytes")))))));
  MF.addMatcher(Pitch2DSetCompound.bind("Pitch2DSetCompound"), this);
  // myres.res.linear.devPtr = d_data21;
  auto AssignResLinearPtr = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("devPtr")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("linear")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("PtrMember"))))))))));
  // myres.res.linear.sizeInBytes = sizeof(float4) * 32;
  auto AssignResLinearSize = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("sizeInBytes")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("linear")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("SizeMember"))))))))));
  // myres.res.linear.desc = desc42;
  auto AssignResLinearDesc = cxxOperatorCallExpr(
      allOf(isAssignmentOperator(),
            has(memberExpr(allOf(
                member(hasName("desc")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("linear")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("DescMember"))))))))));
  // myres.resType = cudaResourceTypeLinear;
  auto LinearSetCompound = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(ObjectType, member(hasName("resType"))))
                       .bind("ResTypeMemberExpr")),
            hasRHS(declRefExpr(hasDeclaration(
                enumConstantDecl(hasName("cudaResourceTypeLinear"))))),
            hasParent(compoundStmt(
                allOf(has(AssignResLinearPtr.bind("AssignResLinearPtr")),
                      has(AssignResLinearSize.bind("AssignResLinearSize")),
                      has(AssignResLinearDesc.bind("AssignResLinearDesc")))))));
  MF.addMatcher(LinearSetCompound.bind("LinearSetCompound"), this);
}

void TextureMemberSetRule::removeRange(SourceRange R) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &LO = DpctGlobalInfo::getContext().getLangOpts();
  auto End = R.getEnd();
  End = End.getLocWithOffset(Lexer::MeasureTokenLength(End, SM, LO));
  End = End.getLocWithOffset(Lexer::MeasureTokenLength(End, SM, LO));
  emplaceTransformation(replaceText(R.getBegin(), End, "", SM));
}

void TextureMemberSetRule::runRule(const MatchFinder::MatchResult &Result) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &LO = DpctGlobalInfo::getContext().getLangOpts();
  if (auto BO = getNodeAsType<BinaryOperator>(Result, "Pitch2DSetCompound")) {
    auto AssignPtrExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignRes2DPtr");
    auto AssignWidthExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignRes2DWidth");
    auto AssignHeightExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignRes2DHeight");
    auto AssignDescExpr =
        getNodeAsType<CXXOperatorCallExpr>(Result, "AssignRes2DDesc");
    auto AssignPitchExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignRes2DPitchInBytes");
    auto ResTypeMemberExpr =
        getNodeAsType<MemberExpr>(Result, "ResTypeMemberExpr");
    auto PtrMemberExpr = getNodeAsType<MemberExpr>(Result, "PtrMember");
    auto WidthMemberExpr = getNodeAsType<MemberExpr>(Result, "WidthMember");
    auto HeightMemberExpr = getNodeAsType<MemberExpr>(Result, "HeightMember");
    auto PitchMemberExpr = getNodeAsType<MemberExpr>(Result, "PitchMember");
    auto DescMemberExpr = getNodeAsType<MemberExpr>(Result, "DescMember");

    if (!AssignPtrExpr || !AssignWidthExpr || !AssignHeightExpr ||
        !AssignDescExpr || !AssignPitchExpr || !ResTypeMemberExpr ||
        !PtrMemberExpr || !WidthMemberExpr || !HeightMemberExpr ||
        !PitchMemberExpr || !DescMemberExpr)
      return;

    // Compare the name of all resource obj
    std::string ResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(ResTypeMemberExpr->getBase())) {
      ResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string PtrResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(PtrMemberExpr->getBase())) {
      PtrResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string WidthResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(WidthMemberExpr->getBase())) {
      WidthResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string HeightResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(HeightMemberExpr->getBase())) {
      HeightResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string PitchResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(PitchMemberExpr->getBase())) {
      PitchResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string DescResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(DescMemberExpr->getBase())) {
      DescResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    if (ResName.compare(PtrResName) || ResName.compare(WidthResName) ||
        ResName.compare(HeightResName) || ResName.compare(PitchResName) ||
        ResName.compare(DescResName)) {
      // Won't do pretty code if the resource name is different
      return;
    }
    // Calculate insert location
    std::string MemberOpt = ResTypeMemberExpr->isArrow() ? "->" : ".";
    auto BORange = getStmtExpansionSourceRange(BO);
    auto AssignPtrRange = getStmtExpansionSourceRange(AssignPtrExpr);
    auto AssignWidthRange = getStmtExpansionSourceRange(AssignWidthExpr);
    auto AssignHeightRange = getStmtExpansionSourceRange(AssignHeightExpr);
    auto AssignPitchRange = getStmtExpansionSourceRange(AssignPitchExpr);
    auto AssignDescRange = getStmtExpansionSourceRange(AssignDescExpr);
    auto LastPos = BORange.getEnd();
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignPtrRange.getEnd()).second) {
      LastPos = AssignPtrRange.getEnd();
    }
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignWidthRange.getEnd()).second) {
      LastPos = AssignWidthRange.getEnd();
    }
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignHeightRange.getEnd()).second) {
      LastPos = AssignHeightRange.getEnd();
    }
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignPitchRange.getEnd()).second) {
      LastPos = AssignPitchRange.getEnd();
    }
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignDescRange.getEnd()).second) {
      LastPos = AssignDescRange.getEnd();
    }
    // Skip the last token
    LastPos =
        LastPos.getLocWithOffset(Lexer::MeasureTokenLength(LastPos, SM, LO));
    // Skip ";"
    LastPos =
        LastPos.getLocWithOffset(Lexer::MeasureTokenLength(LastPos, SM, LO));
    // Generate insert str
    ExprAnalysis EA;
    EA.analyze(AssignPtrExpr->getRHS());
    std::string AssignPtrRHS = EA.getReplacedString();
    EA.analyze(AssignWidthExpr->getRHS());
    std::string AssignWidthRHS = EA.getReplacedString();
    EA.analyze(AssignHeightExpr->getRHS());
    std::string AssignHeightRHS = EA.getReplacedString();
    EA.analyze(AssignPitchExpr->getRHS());
    std::string AssignPitchRHS = EA.getReplacedString();
    EA.analyze(AssignDescExpr->getArg(1));
    std::string AssignDescRHS = EA.getReplacedString();
    std::string IndentStr = getIndent(AssignPtrExpr->getBeginLoc(), SM).str();
    std::string InsertStr = getNL() + IndentStr + ResName + MemberOpt +
                            "set_data(" + AssignPtrRHS + ", " + AssignWidthRHS +
                            ", " + AssignHeightRHS + ", " + AssignPitchRHS +
                            ", " + AssignDescRHS + ");";
    requestFeature(HelperFeatureEnum::device_ext);
    // Remove all the assign expr
    removeRange(BORange);
    removeRange(AssignPtrRange);
    removeRange(AssignWidthRange);
    removeRange(AssignHeightRange);
    removeRange(AssignPitchRange);
    removeRange(AssignDescRange);
    emplaceTransformation(new InsertText(LastPos, std::move(InsertStr)));
  } else if (auto BO =
                 getNodeAsType<BinaryOperator>(Result, "LinearSetCompound")) {
    auto AssignPtrExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignResLinearPtr");
    auto AssignSizeExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignResLinearSize");
    auto AssignDescExpr =
        getNodeAsType<CXXOperatorCallExpr>(Result, "AssignResLinearDesc");
    auto ResTypeMemberExpr =
        getNodeAsType<MemberExpr>(Result, "ResTypeMemberExpr");
    auto PtrMemberExpr = getNodeAsType<MemberExpr>(Result, "PtrMember");
    auto SizeMemberExpr = getNodeAsType<MemberExpr>(Result, "SizeMember");
    auto DescMemberExpr = getNodeAsType<MemberExpr>(Result, "DescMember");

    if (!BO || !AssignPtrExpr || !AssignSizeExpr || !AssignDescExpr ||
        !ResTypeMemberExpr || !PtrMemberExpr || !SizeMemberExpr ||
        !DescMemberExpr)
      return;

    // Compare the name of all resource obj
    std::string ResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(ResTypeMemberExpr->getBase())) {
      ResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string PtrResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(PtrMemberExpr->getBase())) {
      PtrResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string SizeResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(SizeMemberExpr->getBase())) {
      SizeResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string DescResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(DescMemberExpr->getBase())) {
      DescResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    if (ResName.compare(PtrResName) || ResName.compare(SizeResName) ||
        ResName.compare(DescResName)) {
      // Won't do pretty code if the resource name is different
      return;
    }
    // Calculate insert location
    std::string MemberOpt = ResTypeMemberExpr->isArrow() ? "->" : ".";
    auto BORange = getStmtExpansionSourceRange(BO);
    auto AssignPtrRange = getStmtExpansionSourceRange(AssignPtrExpr);
    auto AssignSizeRange = getStmtExpansionSourceRange(AssignSizeExpr);
    auto AssignDescRange = getStmtExpansionSourceRange(AssignDescExpr);
    auto LastPos = BORange.getEnd();
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignPtrRange.getEnd()).second) {
      LastPos = AssignPtrRange.getEnd();
    }
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignSizeRange.getEnd()).second) {
      LastPos = AssignSizeRange.getEnd();
    }
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignDescRange.getEnd()).second) {
      LastPos = AssignDescRange.getEnd();
    }
    // Skip the last token
    LastPos =
        LastPos.getLocWithOffset(Lexer::MeasureTokenLength(LastPos, SM, LO));
    // Skip ";"
    LastPos =
        LastPos.getLocWithOffset(Lexer::MeasureTokenLength(LastPos, SM, LO));
    // Generate insert str
    ExprAnalysis EA;
    EA.analyze(AssignPtrExpr->getRHS());
    std::string AssignPtrRHS = EA.getReplacedString();
    EA.analyze(AssignSizeExpr->getRHS());
    std::string AssignSizeRHS = EA.getReplacedString();
    EA.analyze(AssignDescExpr->getArg(1));
    std::string AssignDescRHS = EA.getReplacedString();
    std::string IndentStr = getIndent(AssignPtrExpr->getBeginLoc(), SM).str();
    std::string InsertStr = getNL() + IndentStr + ResName + MemberOpt +
                            "set_data(" + AssignPtrRHS + ", " + AssignSizeRHS +
                            ", " + AssignDescRHS + ");";
    requestFeature(HelperFeatureEnum::device_ext);
    // Remove all the assign expr
    removeRange(BORange);
    removeRange(AssignPtrRange);
    removeRange(AssignSizeRange);
    removeRange(AssignDescRange);
    emplaceTransformation(new InsertText(LastPos, std::move(InsertStr)));
  } else if (auto BO =
                 getNodeAsType<BinaryOperator>(Result, "ArraySetCompound")) {
    auto AssignArrayExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignResArrayArray");
    auto ResTypeMemberExpr =
        getNodeAsType<MemberExpr>(Result, "ResTypeMemberExpr");
    auto ArrayMemberExpr = getNodeAsType<MemberExpr>(Result, "ArrayMember");

    if (!BO || !AssignArrayExpr || !ResTypeMemberExpr || !ArrayMemberExpr)
      return;

    // Compare the name of all resource obj
    std::string ResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(ResTypeMemberExpr->getBase())) {
      ResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string ArrayResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(ArrayMemberExpr->getBase())) {
      ArrayResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }

    if (ResName.compare(ArrayResName)) {
      // Won't do pretty code if the resource name is different
      return;
    }
    // Calculate insert location
    std::string MemberOpt = ResTypeMemberExpr->isArrow() ? "->" : ".";
    auto BORange = getStmtExpansionSourceRange(BO);
    auto AssignArrayRange = getStmtExpansionSourceRange(AssignArrayExpr);

    auto LastPos = BORange.getEnd();
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignArrayRange.getEnd()).second) {
      LastPos = AssignArrayRange.getEnd();
    }

    // Skip the last token
    LastPos =
        LastPos.getLocWithOffset(Lexer::MeasureTokenLength(LastPos, SM, LO));
    // Skip ";"
    LastPos =
        LastPos.getLocWithOffset(Lexer::MeasureTokenLength(LastPos, SM, LO));
    // Generate insert str
    ExprAnalysis EA;
    EA.analyze(AssignArrayExpr->getRHS());
    std::string AssignArrayRHS = EA.getReplacedString();
    std::string IndentStr = getIndent(AssignArrayExpr->getBeginLoc(), SM).str();
    std::string InsertStr = getNL() + IndentStr + ResName + MemberOpt +
                            "set_data(" + AssignArrayRHS + ");";
    requestFeature(HelperFeatureEnum::device_ext);
    // Remove all the assign expr
    removeRange(BORange);
    removeRange(AssignArrayRange);

    emplaceTransformation(new InsertText(LastPos, std::move(InsertStr)));
  }
}

REGISTER_RULE(TextureMemberSetRule, PassKind::PK_Migration)

void TextureRule::registerMatcher(MatchFinder &MF) {
  auto DeclMatcher = varDecl(hasType(classTemplateSpecializationDecl(
      hasName("texture"))));

  auto DeclMatcherUTF = varDecl(hasType(classTemplateDecl(hasName("texture"))));
  MF.addMatcher(DeclMatcherUTF.bind("texDeclForUnspecializedTemplateFunc"),
                this);

  // Match texture object's declaration
  MF.addMatcher(DeclMatcher.bind("texDecl"), this);
  MF.addMatcher(
      declRefExpr(
          hasDeclaration(DeclMatcher.bind("texDecl")),
          anyOf(hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                               hasAttr(attr::CUDAGlobal)))
                                .bind("texFunc")),
                // Match the __global__/__device__ functions inside which
                // texture object is referenced
                anything()) // Make this matcher available whether it has
                            // ancestors as before
          )
          .bind("tex"),
      this);
  MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(typedefDecl(hasAnyName(
                            "cudaTextureObject_t", "CUtexObject"))))))
                    .bind("texObj"),
                this);
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(namedDecl(hasAnyName(
                               "cudaChannelFormatDesc", "cudaTextureDesc",
                               "cudaResourceDesc", "textureReference",
                               "CUDA_RESOURCE_DESC", "CUDA_TEXTURE_DESC")))))
                    .bind("texMember"),
                this);
  MF.addMatcher(
      typeLoc(
          loc(qualType(hasDeclaration(namedDecl(hasAnyName(
              "cudaChannelFormatDesc", "cudaChannelFormatKind",
              "cudaTextureDesc", "cudaResourceDesc", "cudaResourceType",
              "cudaTextureAddressMode", "cudaTextureFilterMode", "cudaArray",
              "cudaArray_t", "CUarray_st", "CUarray", "CUarray_format",
              "CUarray_format_enum", "CUdeviceptr", "CUresourcetype",
              "CUresourcetype_enum", "CUaddress_mode", "CUaddress_mode_enum",
              "CUfilter_mode", "CUfilter_mode_enum", "CUDA_RESOURCE_DESC",
              "CUDA_TEXTURE_DESC", "CUtexref", "textureReference"))))))
          .bind("texType"),
      this);

  MF.addMatcher(
      declRefExpr(
          to(enumConstantDecl(hasType(enumDecl(hasAnyName(
              "cudaTextureAddressMode", "cudaTextureFilterMode",
              "cudaChannelFormatKind", "cudaResourceType",
              "CUarray_format_enum", "CUaddress_mode", "CUfilter_mode"))))))
          .bind("texEnum"),
      this);

  std::vector<std::string> APINamesSelected = {
      "cudaCreateChannelDesc",
      "cudaCreateChannelDescHalf",
      "cudaUnbindTexture",
      "cudaBindTextureToArray",
      "cudaBindTexture",
      "cudaBindTexture2D",
      "tex1D",
      "tex2D",
      "tex3D",
      "tex1Dfetch",
      "tex1DLayered",
      "tex2DLayered",
      "cudaCreateTextureObject",
      "cudaDestroyTextureObject",
      "cudaGetTextureObjectResourceDesc",
      "cudaGetTextureObjectTextureDesc",
      "cudaGetTextureObjectResourceViewDesc",
      "cuArrayCreate_v2",
      "cuArrayDestroy",
      "cuTexObjectCreate",
      "cuTexObjectDestroy",
      "cuTexObjectGetTextureDesc",
      "cuTexObjectGetResourceDesc",
      "cuTexRefSetArray",
      "cuTexRefSetFormat",
      "cuTexRefSetAddressMode",
      "cuTexRefSetFilterMode",
      "cuTexRefSetFlags",
      "cuTexRefGetAddressMode",
      "cuTexRefGetFilterMode",
      "cuTexRefGetFlags",
      "cuTexRefSetAddress_v2",
      "cuTexRefSetAddress2D_v3",
  };

  auto hasAnyFuncName = [&]() {
    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(APINamesSelected));
  };

  MF.addMatcher(callExpr(callee(functionDecl(hasAnyFuncName()))).bind("call"),
                this);

  MF.addMatcher(unresolvedLookupExpr(
                    hasAnyDeclaration(namedDecl(hasAnyFuncName())),
                    hasParent(callExpr(unless(parentStmt())).bind("callExpr")))
                    .bind("unresolvedLookupExpr"),
                this);
}

bool TextureRule::removeExtraMemberAccess(const MemberExpr *ME) {
  if (auto ParentME = getParentMemberExpr(ME)) {
    emplaceTransformation(new ReplaceToken(ME->getMemberLoc(), ""));
    emplaceTransformation(new ReplaceToken(ParentME->getOperatorLoc(), ""));
    return true;
  }
  return false;
}

bool TextureRule::tryMerge(const MemberExpr *ME, const Expr *BO) {
  static std::unordered_map<std::string, std::vector<std::string>> MergeMap = {
      {"textureReference", {"addressMode", "filterMode", "normalized"}},
      {"cudaTextureDesc", {"addressMode", "filterMode", "normalizedCoords"}},
      {"CUDA_TEXTURE_DESC", {"addressMode", "filterMode", "flags"}},
  };

  auto Iter = MergeMap.find(
      DpctGlobalInfo::getUnqualifiedTypeName(ME->getBase()->getType()));
  if (Iter == MergeMap.end())
    return false;

  SettersMerger Merger(Iter->second, this);
  return Merger.tryMerge(BO);
}

void TextureRule::replaceTextureMember(const MemberExpr *ME,
                                       ASTContext &Context, SourceManager &SM) {
  auto AssignedBO = getParentAsAssignedBO(ME, Context);
  if (tryMerge(ME, AssignedBO))
    return;

  auto Field = ME->getMemberNameInfo().getAsString();
  if (Field == "channelDesc") {
    if (removeExtraMemberAccess(ME))
      return;
  }
  auto ReplField = MapNames::findReplacedName(TextureMemberNames, Field);
  if (ReplField.empty()) {
    report(ME->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
           DpctGlobalInfo::getOriginalTypeName(ME->getBase()->getType()) +
               "::" + Field);
    if (AssignedBO) {
      emplaceTransformation(new ReplaceStmt(AssignedBO, ""));
    } else {
      emplaceTransformation(new ReplaceStmt(ME, "0"));
    }
    return;
  }

  if (AssignedBO) {
    StringRef MethodName;
    auto AssignedValue = getMemberAssignedValue(AssignedBO, Field, MethodName);
    if (MethodName.empty()) {
      requestFeature(HelperFeatureEnum::device_ext);
    } else {
      if (MapNames::SamplingInfoToSetFeatureMap.count(MethodName.str())) {
        requestFeature(
            MapNames::SamplingInfoToSetFeatureMap.at(MethodName.str()));
      }
      if (MapNames::ImageWrapperBaseToSetFeatureMap.count(MethodName.str())) {
        requestFeature(
            MapNames::ImageWrapperBaseToSetFeatureMap.at(MethodName.str()));
      }
    }
    emplaceTransformation(ReplaceMemberAssignAsSetMethod(
        AssignedBO, ME, MethodName, AssignedValue));
  } else {
    if (ReplField == "coordinate_normalization_mode") {
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, "is_coordinate_normalized()"));
      requestFeature(HelperFeatureEnum::device_ext);
    } else {
      emplaceTransformation(new RenameFieldInMemberExpr(
          ME, buildString("get_", ReplField, "()")));
      if (MapNames::SamplingInfoToGetFeatureMap.count(ReplField)) {
        requestFeature(MapNames::SamplingInfoToGetFeatureMap.at(ReplField));
      }
      if (MapNames::ImageWrapperBaseToGetFeatureMap.count(ReplField)) {
        requestFeature(MapNames::ImageWrapperBaseToGetFeatureMap.at(ReplField));
      }
    }
  }
}

const Expr *TextureRule::getParentAsAssignedBO(const Expr *E,
                                               ASTContext &Context) {
  auto Parents = Context.getParents(*E);
  if (Parents.size() > 0)
    return getAssignedBO(Parents[0].get<Expr>(), Context);
  return nullptr;
}

// Return the binary operator if E is the lhs of an assign expression, otherwise
// nullptr.
const Expr *TextureRule::getAssignedBO(const Expr *E, ASTContext &Context) {
  if (dyn_cast<MemberExpr>(E)) {
    // Continue finding parents when E is MemberExpr.
    return getParentAsAssignedBO(E, Context);
  } else if (auto ICE = dyn_cast<ImplicitCastExpr>(E)) {
    // Stop finding parents and return nullptr when E is ImplicitCastExpr,
    // except for ArrayToPointerDecay cast.
    if (ICE->getCastKind() == CK_ArrayToPointerDecay) {
      return getParentAsAssignedBO(E, Context);
    }
  } else if (auto ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    // Continue finding parents when E is ArraySubscriptExpr, and remove
    // subscript operator anyway for texture object's member.
    emplaceTransformation(new ReplaceToken(
        Lexer::getLocForEndOfToken(ASE->getLHS()->getEndLoc(), 0,
                                   Context.getSourceManager(),
                                   Context.getLangOpts()),
        ASE->getRBracketLoc(), ""));
    return getParentAsAssignedBO(E, Context);
  } else if (auto BO = dyn_cast<BinaryOperator>(E)) {
    // If E is BinaryOperator, return E only when it is assign expression,
    // otherwise return nullptr.
    if (BO->getOpcode() == BO_Assign)
      return BO;
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (COCE->getOperator() == OO_Equal) {
      return COCE;
    }
  }
  return nullptr;
}

bool TextureRule::processTexVarDeclInDevice(const VarDecl *VD) {
  if (auto FD =
          dyn_cast_or_null<FunctionDecl>(VD->getParentFunctionOrMethod())) {
    if (FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) {
      auto Tex = DpctGlobalInfo::getInstance().insertTextureInfo(VD);

      auto DataType = Tex->getType()->getDataType();
      if (DataType.back() != '4') {
        report(VD->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_FORMAT, true);
      }

      ParameterStream PS;
      Tex->getFuncDecl(PS);
      emplaceTransformation(new ReplaceToken(VD->getBeginLoc(), VD->getEndLoc(),
                                             std::move(PS.Str)));
      return true;
    }
  }
  return false;
}

void TextureRule::runRule(const MatchFinder::MatchResult &Result) {

  if (getAssistNodeAsType<UnresolvedLookupExpr>(Result,
                                                "unresolvedLookupExpr")) {
    const CallExpr *CE = getAssistNodeAsType<CallExpr>(Result, "callExpr");
    ExprAnalysis A;
    A.analyze(CE);
    emplaceTransformation(A.getReplacement());
  }

  if (auto VD = getAssistNodeAsType<VarDecl>(
          Result, "texDeclForUnspecializedTemplateFunc")) {

    auto TST = VD->getType()->getAs<TemplateSpecializationType>();
    if (!TST)
      return;

    std::string Name =
        TST->getTemplateName().getAsTemplateDecl()->getNameAsString();

    if (Name == "texture") {
      auto Args = TST->template_arguments();

      if (!isa<ParmVarDecl>(VD) || Args.size() != 3)
        return;

      if (getStmtSpelling(Args[2].getAsExpr()) == "cudaReadModeNormalizedFloat")
        report(VD->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_NORM_READ_MODE,
               true);

      processTexVarDeclInDevice(VD);
    }
  } else if (auto VD = getAssistNodeAsType<VarDecl>(Result, "texDecl")) {
    auto TST = VD->getType()->getAs<TemplateSpecializationType>();
    if (!TST)
      return;

    auto Args = TST->template_arguments();

    if (Args.size() == 3) {
      if (getStmtSpelling(Args[2].getAsExpr()) == "cudaReadModeNormalizedFloat")
        report(VD->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_NORM_READ_MODE,
               true);
    }

    auto Tex = DpctGlobalInfo::getInstance().insertTextureInfo(VD);

    if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "texFunc")) {

      if (!isa<ParmVarDecl>(VD))
        DeviceFunctionDecl::LinkRedecls(FD)->addTexture(Tex);
    }

    if (processTexVarDeclInDevice(VD))
      return;

    auto DataType = Tex->getType()->getDataType();
    if (DataType.back() != '4') {
      report(VD->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_FORMAT, true);
    }
    emplaceTransformation(new ReplaceVarDecl(VD, Tex->getHostDeclString()));

  } else if (auto ME = getNodeAsType<MemberExpr>(Result, "texMember")) {
    auto BaseTy = DpctGlobalInfo::getUnqualifiedTypeName(
        ME->getBase()->getType(), *Result.Context);
    auto MemberName = ME->getMemberNameInfo().getAsString();
    if (BaseTy == "cudaResourceDesc" || BaseTy == "CUDA_RESOURCE_DESC_st" ||
        BaseTy == "CUDA_RESOURCE_DESC") {
      if (MemberName == "res") {
        removeExtraMemberAccess(ME);
        replaceResourceDataExpr(getParentMemberExpr(ME), *Result.Context);
      } else if (MemberName == "resType") {
        if (auto BO = getParentAsAssignedBO(ME, *Result.Context)) {
          requestFeature(HelperFeatureEnum::device_ext);
          emplaceTransformation(
              ReplaceMemberAssignAsSetMethod(BO, ME, "data_type"));
        } else {
          requestFeature(HelperFeatureEnum::device_ext);
          emplaceTransformation(
              new RenameFieldInMemberExpr(ME, "get_data_type()"));
        }
      }
    } else if (BaseTy == "cudaChannelFormatDesc") {
      static std::map<std::string, std::string> MethodNameMap = {
          {"x", "channel_size"},
          {"y", "channel_size"},
          {"z", "channel_size"},
          {"w", "channel_size"},
          {"f", "channel_data_type"}};
      static const std::unordered_map<std::string, HelperFeatureEnum>
          MethodNameToGetFeatureMap = {
              {"x", HelperFeatureEnum::device_ext},
              {"y", HelperFeatureEnum::device_ext},
              {"z", HelperFeatureEnum::device_ext},
              {"w", HelperFeatureEnum::device_ext},
              {"f", HelperFeatureEnum::device_ext}};
      static const std::unordered_map<std::string, HelperFeatureEnum>
          MethodNameToSetFeatureMap = {
              {"x", HelperFeatureEnum::device_ext},
              {"y", HelperFeatureEnum::device_ext},
              {"z", HelperFeatureEnum::device_ext},
              {"w", HelperFeatureEnum::device_ext},
              {"f", HelperFeatureEnum::device_ext}};
      static std::map<std::string, std::string> ExtraArgMap = {
          {"x", "1"}, {"y", "2"}, {"z", "3"}, {"w", "4"}, {"f", ""}};
      std::string MemberName = ME->getMemberNameInfo().getAsString();
      if (auto BO = getParentAsAssignedBO(ME, *Result.Context)) {
        requestFeature(HelperFeatureEnum::device_ext);
        requestFeature(MethodNameToSetFeatureMap.at(MemberName));
        emplaceTransformation(ReplaceMemberAssignAsSetMethod(
            BO, ME, MethodNameMap[MemberName], "", ExtraArgMap[MemberName]));
      } else {
        requestFeature(HelperFeatureEnum::device_ext);
        requestFeature(MethodNameToGetFeatureMap.at(MemberName));
        emplaceTransformation(new RenameFieldInMemberExpr(
            ME, buildString("get_", MethodNameMap[MemberName], "()")));
      }
    } else {
      replaceTextureMember(ME, *Result.Context, *Result.SourceManager);
    }
  } else if (auto TL = getNodeAsType<TypeLoc>(Result, "texType")) {
    if (isCapturedByLambda(TL))
      return;
    const std::string &ReplType = MapNames::findReplacedName(
        MapNames::TypeNamesMap,
        DpctGlobalInfo::getUnqualifiedTypeName(TL->getType(), *Result.Context));

    requestHelperFeatureForTypeNames(
        DpctGlobalInfo::getUnqualifiedTypeName(TL->getType(), *Result.Context));
    insertHeaderForTypeRule(
        DpctGlobalInfo::getUnqualifiedTypeName(TL->getType(), *Result.Context),
        TL->getBeginLoc());
    if (!ReplType.empty())
      emplaceTransformation(new ReplaceToken(TL->getBeginLoc(), TL->getEndLoc(),
                                             std::string(ReplType)));
  } else if (auto CE = getNodeAsType<CallExpr>(Result, "call")) {
    auto Name = CE->getDirectCallee()->getNameAsString();
    if (Name == "cuTexRefSetFlags") {
      StringRef MethodName;
      auto Value = getTextureFlagsSetterInfo(CE->getArg(1), MethodName);
      requestFeature(HelperFeatureEnum::device_ext);
      std::shared_ptr<CallExprRewriter> Rewriter =
          std::make_shared<AssignableRewriter>(
              CE, std::make_shared<PrinterRewriter<MemberCallPrinter<
                      const Expr *, RenameWithSuffix, StringRef>>>(
                      CE, Name, CE->getArg(0), true,
                      RenameWithSuffix("set", MethodName), Value));
      std::optional<std::string> Result = Rewriter->rewrite();
      if (Result.has_value())
        emplaceTransformation(
            new ReplaceStmt(CE, true, std::move(Result).value()));
      return;
    }
    if (Name == "cudaCreateChannelDesc") {
      auto Callee =
          dyn_cast<DeclRefExpr>(CE->getCallee()->IgnoreImplicitAsWritten());
      if (Callee) {
        auto TemArg = Callee->template_arguments();
        if (TemArg.size() != 0) {
          auto ChnType = TemArg[0]
                             .getArgument()
                             .getAsType()
                             .getCanonicalType()
                             .getAsString();
          if (ChnType.back() != '4') {
            report(CE->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_FORMAT,
                   true);
          }
        } else if (getStmtSpelling(CE->getArg(0)) == "0" ||
                   getStmtSpelling(CE->getArg(1)) == "0" ||
                   getStmtSpelling(CE->getArg(2)) == "0" ||
                   getStmtSpelling(CE->getArg(3)) == "0") {
          report(CE->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_FORMAT,
                 true);
        }
      }
    }
    ExprAnalysis A;
    A.analyze(CE);
    emplaceTransformation(A.getReplacement());
    A.applyAllSubExprRepl();
  } else if (auto DRE = getNodeAsType<DeclRefExpr>(Result, "texEnum")) {
    if (auto ECD = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
      std::string EnumName = ECD->getName().str();
      requestHelperFeatureForEnumNames(EnumName);
      if (MapNames::replaceName(EnumConstantRule::EnumNamesMap, EnumName)) {
        emplaceTransformation(new ReplaceStmt(DRE, EnumName));
      } else {
        report(DRE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
               EnumName);
      }
    }
  } else if (auto TL = getNodeAsType<TypeLoc>(Result, "texObj")) {
    if (auto FD = DpctGlobalInfo::getParentFunction(TL)) {
      if (FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) {
        return;
      }
    } else if (auto VD = DpctGlobalInfo::findAncestor<VarDecl>(TL)) {
      if (!VD->hasGlobalStorage()) {
        return;
      }
    }
    emplaceTransformation(new ReplaceToken(TL->getBeginLoc(), TL->getEndLoc(),
                                           MapNames::getDpctNamespace() +
                                               "image_wrapper_base_p"));
    requestFeature(HelperFeatureEnum::device_ext);
  }
}

void TextureRule::replaceResourceDataExpr(const MemberExpr *ME,
                                          ASTContext &Context) {
  if (!ME)
    return;
  auto TopMember = getParentMemberExpr(ME);
  if (!TopMember)
    return;

  removeExtraMemberAccess(ME);

  auto AssignedBO = getParentAsAssignedBO(TopMember, Context);
  auto FieldName =
      ResourceTypeNames[TopMember->getMemberNameInfo().getAsString()];
  if (FieldName.empty()) {
    report(ME->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
           DpctGlobalInfo::getOriginalTypeName(ME->getBase()->getType()) +
               "::" + ME->getMemberDecl()->getName().str());
  }

  if (FieldName == "channel") {
    if (removeExtraMemberAccess(TopMember))
      return;
  }

  if (AssignedBO) {
    static const std::unordered_map<std::string, HelperFeatureEnum>
        ResourceTypeNameToSetFeature = {
            {"devPtr", HelperFeatureEnum::device_ext},
            {"desc", HelperFeatureEnum::device_ext},
            {"array", HelperFeatureEnum::device_ext},
            {"width", HelperFeatureEnum::device_ext},
            {"height", HelperFeatureEnum::device_ext},
            {"pitchInBytes", HelperFeatureEnum::device_ext},
            {"sizeInBytes", HelperFeatureEnum::device_ext},
            {"hArray", HelperFeatureEnum::device_ext},
            {"format", HelperFeatureEnum::device_ext},
            {"numChannels", HelperFeatureEnum::device_ext}};
    requestFeature(ResourceTypeNameToSetFeature.at(
                       TopMember->getMemberNameInfo().getAsString()));
    emplaceTransformation(
        ReplaceMemberAssignAsSetMethod(AssignedBO, TopMember, FieldName));
  } else {
    auto MemberName = TopMember->getMemberDecl()->getName();
    if (MemberName == "array" || MemberName == "hArray") {
      emplaceTransformation(new InsertBeforeStmt(
          TopMember, "(" + MapNames::getDpctNamespace() + "image_matrix_p)"));
      requestFeature(HelperFeatureEnum::device_ext);
    }
    static const std::unordered_map<std::string, HelperFeatureEnum>
        ResourceTypeNameToGetFeature = {
            {"devPtr", HelperFeatureEnum::device_ext},
            {"desc", HelperFeatureEnum::device_ext},
            {"array", HelperFeatureEnum::device_ext},
            {"width", HelperFeatureEnum::device_ext},
            {"height", HelperFeatureEnum::device_ext},
            {"pitchInBytes", HelperFeatureEnum::device_ext},
            {"sizeInBytes", HelperFeatureEnum::device_ext},
            {"hArray", HelperFeatureEnum::device_ext},
            {"format", HelperFeatureEnum::device_ext},
            {"numChannels", HelperFeatureEnum::device_ext}};
    requestFeature(ResourceTypeNameToGetFeature.at(
                       TopMember->getMemberNameInfo().getAsString()));
    emplaceTransformation(new RenameFieldInMemberExpr(
        TopMember, buildString("get_", FieldName, "()")));
    if(TopMember->getMemberNameInfo().getAsString() == "devPtr"){
        emplaceTransformation(new InsertBeforeStmt(
        ME, buildString("(char *)")));
    }
  }
}

bool isAssignOperator(const Stmt *S) {
  if (auto BO = dyn_cast<BinaryOperator>(S)) {
    return BO->getOpcode() == BO_Assign;
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(S)) {
    return COCE->getOperator() == OO_Equal;
  }
  return false;
}

const Expr *getLhs(const Stmt *S) {
  if (auto BO = dyn_cast<BinaryOperator>(S)) {
    return BO->getLHS();
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (COCE->getNumArgs() > 0) {
      return COCE->getArg(0);
    }
  }
  return nullptr;
}

const Expr *getRhs(const Stmt *S) {
  if (auto BO = dyn_cast<BinaryOperator>(S)) {
    return BO->getRHS();
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (COCE->getNumArgs() > 1) {
      return COCE->getArg(1);
    }
  }
  return nullptr;
}

void TextureRule::SettersMerger::traverseBinaryOperator(const Stmt *S) {
  do {
    if (S != Target && Result.empty())
      return;

    if (!isAssignOperator(S))
      break;

    const Expr *LHS = getLhs(S);
    if (!LHS)
      break;
    if (auto ASE = dyn_cast<ArraySubscriptExpr>(LHS)) {
      LHS = ASE->getBase()->IgnoreImpCasts();
    }
    if (const MemberExpr *ME = dyn_cast<MemberExpr>(LHS)) {
      auto Method = ME->getMemberDecl()->getName();
      if (auto DRE = dyn_cast<DeclRefExpr>(ME->getBase()->IgnoreImpCasts())) {
        if (Result.empty()) {
          D = DRE->getDecl();
          IsArrow = ME->isArrow();
        } else if (DRE->getDecl() != D) {
          break;
        }
        unsigned i = 0;
        for (const auto &Name : MethodNames) {
          if (Method == Name) {
            Result.emplace_back(i, S);
            return;
          }
          ++i;
        }
      }
    }
  } while (false);
  traverse(getLhs(S));
  traverse(getRhs(S));
}

void TextureRule::SettersMerger::traverse(const Stmt *S) {
  if (Stop || !S)
    return;

  switch (S->getStmtClass()) {
  case Stmt::BinaryOperatorClass:
  case Stmt::CXXOperatorCallExprClass:
    traverseBinaryOperator(S);
    break;
  case Stmt::DeclRefExprClass:
    if (static_cast<const DeclRefExpr *>(S)->getDecl() != D) {
      break;
    }
    LLVM_FALLTHROUGH;
  case Stmt::IfStmtClass:
  case Stmt::WhileStmtClass:
  case Stmt::DoStmtClass:
  case Stmt::SwitchStmtClass:
  case Stmt::ForStmtClass:
  case Stmt::CaseStmtClass:
    if (!Result.empty()) {
      Stop = true;
    }
    break;
  default:
    for (auto Child : S->children()) {
      traverse(Child);
    }
  }
}

StringRef getCoordinateNormalizationStr(bool IsNormalized) {
  if (IsNormalized) {
    static std::string NormalizedName =
        MapNames::getClNamespace() +
        "coordinate_normalization_mode::normalized";
    return NormalizedName;
  } else {
    static std::string UnnormalizedName =
        MapNames::getClNamespace() +
        "coordinate_normalization_mode::unnormalized";
    return UnnormalizedName;
  }
}

std::string TextureRule::getTextureFlagsSetterInfo(const Expr *Flags,
                                                   StringRef &SetterName) {
  SetterName = "";
  if (!Flags->isValueDependent()) {
    Expr::EvalResult Result;
    if (Flags->EvaluateAsInt(Result, DpctGlobalInfo::getContext())) {
      auto Val = Result.Val.getInt().getZExtValue();
      if (Val != 1 && Val != 3) {
        report(Flags, Diagnostics::TEX_FLAG_UNSUPPORT, false,
               ExprAnalysis::ref(Flags));
      }
      return getCoordinateNormalizationStr(Val & 0x02).str();
    }
  }
  SetterName = "coordinate_normalization_mode";
  report(Flags, Diagnostics::TEX_FLAG_UNSUPPORT, false,
         ExprAnalysis::ref(Flags));
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  printWithParens(OS, Flags);
  OS << " & 0x02";
  return OS.str();
}

std::string TextureRule::getMemberAssignedValue(const Stmt *AssignStmt,
                                                StringRef MemberName,
                                                StringRef &SetMethodName) {
  SetMethodName = "";
  if (auto RHS = getRhs(AssignStmt)) {
    RHS = RHS->IgnoreImpCasts();
    if (MemberName == "normalizedCoords" || MemberName == "normalized") {
      if (auto IL = dyn_cast<IntegerLiteral>(RHS)) {
        return getCoordinateNormalizationStr(IL->getValue().getZExtValue())
            .str();
      } else if (auto BL = dyn_cast<CXXBoolLiteralExpr>(RHS)) {
        return getCoordinateNormalizationStr(BL->getValue()).str();
      }
      SetMethodName = "coordinate_normalization_mode";
    } else if (MemberName == "flags") {
      return getTextureFlagsSetterInfo(RHS, SetMethodName);
    } else if (MemberName == "channelDesc") {
      SetMethodName = "channel";
    }
    return ExprAnalysis::ref(RHS);
  } else {
    return std::string();
  }
}

bool TextureRule::SettersMerger::applyResult() {
  class ResultMapInserter {
    unsigned LastIndex = 0;
    std::vector<const Stmt *> LatestStmts;
    std::vector<const Stmt *> DuplicatedStmts;
    TextureRule *Rule;
    std::map<const Stmt *, bool> &ResultMap;

  public:
    ResultMapInserter(size_t MethodNum, TextureRule *TexRule)
        : LatestStmts(MethodNum, nullptr), Rule(TexRule),
          ResultMap(TexRule->ProcessedBO) {}
    ~ResultMapInserter() {
      for (auto S : DuplicatedStmts) {
        Rule->emplaceTransformation(new ReplaceStmt(S, ""));
        ResultMap[S] = true;
      }
      for (auto S : LatestStmts) {
        ResultMap[S] = false;
      }
    }
    void update(size_t Index, const Stmt *S) {
      auto &Latest = LatestStmts[Index];
      if (Latest)
        DuplicatedStmts.push_back(Latest);
      Latest = S;
      LastIndex = Index;
    }
    void success(std::string &Replaced) {
      Rule->emplaceTransformation(
          new ReplaceStmt(LatestStmts[LastIndex], true, std::move(Replaced)));
      DuplicatedStmts.insert(DuplicatedStmts.end(), LatestStmts.begin(),
                             LatestStmts.begin() + LastIndex);
      DuplicatedStmts.insert(DuplicatedStmts.end(),
                             LatestStmts.begin() + LastIndex + 1,
                             LatestStmts.end());
      LatestStmts.clear();
    }
  };

  ResultMapInserter Inserter(MethodNames.size(), Rule);
  std::vector<std::string> ArgsList(MethodNames.size());
  unsigned ActualArgs = 0;
  for (const auto &R : Result) {
    if (ArgsList[R.first].empty())
      ++ActualArgs;
    Inserter.update(R.first, R.second);
    static StringRef Dummy;
    ArgsList[R.first] =
        Rule->getMemberAssignedValue(R.second, MethodNames[R.first], Dummy);
  }
  if (ActualArgs != ArgsList.size()) {
    return false;
  }

  std::string ReplacedText;
  llvm::raw_string_ostream OS(ReplacedText);
  MemberCallPrinter<StringRef, StringRef, std::vector<std::string>> Printer(
      D->getName(), IsArrow, "set", std::move(ArgsList));
  Printer.print(OS);

  Inserter.success(OS.str());
  return true;
}

bool TextureRule::SettersMerger::tryMerge(const Stmt *BO) {
  auto Iter = ProcessedBO.find(BO);
  if (Iter != ProcessedBO.end())
    return Iter->second;

  Target = BO;
  auto CS = DpctGlobalInfo::findAncestor<CompoundStmt>(
      BO, [&](const DynTypedNode &Node) -> bool {
        if (Node.get<IfStmt>() || Node.get<WhileStmt>() ||
            Node.get<ForStmt>() || Node.get<DoStmt>() || Node.get<CaseStmt>() ||
            Node.get<SwitchStmt>() || Node.get<CompoundStmt>()) {
          return true;
        }
        return false;
      });

  if (!CS) {
    return ProcessedBO[BO] = false;
  }

  traverse(CS);
  if (applyResult()) {
    requestFeature(HelperFeatureEnum::device_ext);
    return true;
  } else {
    return false;
  }
}

REGISTER_RULE(TextureRule, PassKind::PK_Analysis)

void CXXNewExprRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(cxxNewExpr().bind("newExpr"), this);
}

void CXXNewExprRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto CNE = getAssistNodeAsType<CXXNewExpr>(Result, "newExpr")) {
    // E.g., new cudaEvent_t *()
    Token Tok;
    auto LOpts = Result.Context->getLangOpts();
    SourceManager *SM = Result.SourceManager;
    auto BeginLoc =
        CNE->getAllocatedTypeSourceInfo()->getTypeLoc().getBeginLoc();
    Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
    if (Tok.isAnyIdentifier()) {
      std::string Str = MapNames::findReplacedName(
          MapNames::TypeNamesMap, Tok.getRawIdentifier().str());
      insertHeaderForTypeRule(Tok.getRawIdentifier().str(), BeginLoc);
      requestHelperFeatureForTypeNames(Tok.getRawIdentifier().str());

      SourceManager &SM = DpctGlobalInfo::getSourceManager();
      BeginLoc = SM.getExpansionLoc(BeginLoc);
      if (!Str.empty()) {
        emplaceTransformation(new ReplaceToken(BeginLoc, std::move(Str)));
        return;
      }
    }

    // E.g., #define NEW_STREAM new cudaStream_t
    //      stream = NEW_STREAM;
    auto TypeName = CNE->getAllocatedType().getAsString();
    auto ReplName = std::string(
        MapNames::findReplacedName(MapNames::TypeNamesMap, TypeName));
    insertHeaderForTypeRule(TypeName, BeginLoc);
    requestHelperFeatureForTypeNames(TypeName);

    if (!ReplName.empty()) {
      auto BeginLoc =
          CNE->getAllocatedTypeSourceInfo()->getTypeLoc().getBeginLoc();
      emplaceTransformation(new ReplaceToken(BeginLoc, std::move(ReplName)));
    }
  }
}

REGISTER_RULE(CXXNewExprRule, PassKind::PK_Migration)

void NamespaceRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(usingDirectiveDecl().bind("usingDirective"), this);
  MF.addMatcher(namespaceAliasDecl().bind("namespaceAlias"), this);
  MF.addMatcher(usingDecl().bind("using"), this);
}

void NamespaceRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto UDD =
          getAssistNodeAsType<UsingDirectiveDecl>(Result, "usingDirective")) {
    std::string Namespace = UDD->getNominatedNamespace()->getNameAsString();
    if (Namespace == "cooperative_groups" || Namespace == "placeholders" ||
        Namespace == "nvcuda")
      emplaceTransformation(new ReplaceDecl(UDD, ""));
  } else if (auto NAD = getAssistNodeAsType<NamespaceAliasDecl>(
                 Result, "namespaceAlias")) {
    std::string Namespace = NAD->getNamespace()->getNameAsString();
    if (Namespace == "cooperative_groups" || Namespace == "placeholders")
      emplaceTransformation(new ReplaceDecl(NAD, ""));
  } else if (auto UD = getAssistNodeAsType<UsingDecl>(Result, "using")) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    SourceLocation Beg, End;
    unsigned int Len, Toklen;
    if (UD->getBeginLoc().isMacroID()) {
      // For scenario like "#define USING_1(FUNC) using std::FUNC; int a = 1;".
      // The macro include other statement or decl, we keep the origin code.
      if (auto CS = DpctGlobalInfo::findAncestor<CompoundStmt>(UD)) {
        const DeclStmt *DS =
            DpctGlobalInfo::getContext().getParents(*UD)[0].get<DeclStmt>();
        if (!DS) {
          return;
        }
        for (auto child : CS->children()) {
          if (child == DS) {
            continue;
          } else if (child->getBeginLoc().isMacroID() &&
                     SM.getExpansionLoc(child->getBeginLoc()) ==
                         SM.getExpansionLoc(UD->getBeginLoc())) {
            return;
          }
        }
      } else if (auto TS =
                     DpctGlobalInfo::findAncestor<TranslationUnitDecl>(UD)) {
        for (const auto &child : TS->decls()) {
          if (child == UD) {
            continue;
          } else if (auto USD = dyn_cast<UsingShadowDecl>(child)) {
            // To process implicit UsingShadowDecl node generated by UsingDecl
            // in global scope
            if (USD->getIntroducer() == UD) {
              continue;
            }
          } else if (child->getBeginLoc().isMacroID() &&
                     SM.getExpansionLoc(child->getBeginLoc()) ==
                         SM.getExpansionLoc(UD->getBeginLoc())) {
            return;
          }
        }
      } else {
        return;
      }
      auto Range = SM.getExpansionRange(UD->getBeginLoc());
      Beg = Range.getBegin();
      End = Range.getEnd();
    } else {
      Beg = UD->getBeginLoc();
      End = UD->getEndLoc();
    }
    Toklen = Lexer::MeasureTokenLength(
        End, SM, DpctGlobalInfo::getContext().getLangOpts());
    Len = SM.getFileOffset(End) - SM.getFileOffset(Beg) + Toklen;

    bool IsAllTargetsInCUDA = true;
    for (const auto &child : UD->getDeclContext()->decls()) {
      if (child == UD) {
        continue;
      } else if (const clang::UsingShadowDecl *USD =
                     dyn_cast<UsingShadowDecl>(child)) {
        if (USD->getIntroducer() == UD) {
          if (const auto *FD = dyn_cast<FunctionDecl>(USD->getTargetDecl())) {
            if (!isFromCUDA(FD)) {
              IsAllTargetsInCUDA = false;
              break;
            }
          } else if (const auto *FTD =
                         dyn_cast<FunctionTemplateDecl>(USD->getTargetDecl())) {
            if (!isFromCUDA(FTD)) {
              IsAllTargetsInCUDA = false;
              break;
            }
          } else {
            IsAllTargetsInCUDA = false;
            break;
          }
        }
      }
    }

    if (IsAllTargetsInCUDA) {
      auto NextTok = Lexer::findNextToken(
          End, SM, DpctGlobalInfo::getContext().getLangOpts());
      if (NextTok.has_value() && NextTok.value().is(tok::semi)) {
        Len = SM.getFileOffset(NextTok.value().getLocation()) -
              SM.getFileOffset(Beg) + 1;
      }
      emplaceTransformation(new ReplaceText(Beg, Len, ""));
    }
  }
}

REGISTER_RULE(NamespaceRule, PassKind::PK_Migration)

void RemoveBaseClassRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(cxxRecordDecl(isDirectlyDerivedFrom(hasAnyName(
                                  "unary_function", "binary_function")))
                    .bind("derivedFrom"),
                this);
}

void RemoveBaseClassRule::runRule(const MatchFinder::MatchResult &Result) {
  auto SM = Result.SourceManager;
  auto LOpts = Result.Context->getLangOpts();
  auto findColon = [&](SourceRange SR) {
    Token Tok;
    auto E = SR.getEnd();
    SourceLocation Loc = SR.getBegin();
    Lexer::getRawToken(Loc, Tok, *SM, LOpts, true);
    bool ColonFound = false;
    while (Loc <= E) {
      if (Tok.is(tok::TokenKind::colon)) {
        ColonFound = true;
        break;
      }
      Tok = Lexer::findNextToken(Tok.getLocation(), *SM, LOpts).value();
      Loc = Tok.getLocation();
    }
    if (ColonFound)
      return Loc;
    else
      return SourceLocation();
  };

  auto getBaseDecl = [](QualType QT) {
    const Type *T = QT.getTypePtr();
    const NamedDecl *ND = nullptr;
    if (const auto *E = dyn_cast<ElaboratedType>(T)) {
      T = E->desugar().getTypePtr();
      if (const auto *TT = dyn_cast<TemplateSpecializationType>(T))
        ND = TT->getTemplateName().getAsTemplateDecl();
    } else
      ND = T->getAsCXXRecordDecl();
    return ND;
  };

  if (auto D = getNodeAsType<CXXRecordDecl>(Result, "derivedFrom")) {
    if (D->getNumBases() != 1)
      return;
    auto SR = SourceRange(D->getInnerLocStart(), D->getBraceRange().getBegin());
    auto ColonLoc = findColon(SR);
    if (ColonLoc.isValid()) {
      auto QT = D->bases().begin()->getType();
      const NamedDecl *BaseDecl = getBaseDecl(QT);
      if (BaseDecl) {
        auto BaseName = BaseDecl->getDeclName().getAsString();
        auto ThrustName = "thrust::" + BaseName;
        auto StdName = "std::" + BaseName;
        report(ColonLoc, Diagnostics::DEPRECATED_BASE_CLASS, false, ThrustName,
               StdName);
        auto Len = SM->getFileOffset(D->getBraceRange().getBegin()) -
                   SM->getFileOffset(ColonLoc);
        emplaceTransformation(new ReplaceText(ColonLoc, Len, ""));
      }
    }
  }
}

REGISTER_RULE(RemoveBaseClassRule, PassKind::PK_Migration)
REGISTER_RULE(AsmRule, PassKind::PK_Analysis)

// Rule for FFT function calls.
void FFTFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName("cufftPlan1d", "cufftPlan2d", "cufftPlan3d",
                      "cufftPlanMany", "cufftMakePlan1d", "cufftMakePlan2d",
                      "cufftMakePlan3d", "cufftMakePlanMany",
                      "cufftMakePlanMany64", "cufftExecC2C", "cufftExecR2C",
                      "cufftExecC2R", "cufftExecZ2Z", "cufftExecZ2D",
                      "cufftExecD2Z", "cufftCreate", "cufftDestroy",
                      "cufftSetStream", "cufftGetVersion", "cufftGetProperty",
                      "cufftXtMakePlanMany", "cufftXtExec", "cufftGetSize1d",
                      "cufftGetSize2d", "cufftGetSize3d", "cufftGetSizeMany",
                      "cufftGetSize", "cufftEstimate1d", "cufftEstimate2d",
                      "cufftEstimate3d", "cufftEstimateMany",
                      "cufftSetAutoAllocation", "cufftGetSizeMany64",
                      "cufftSetWorkArea");
  };
  MF.addMatcher(callExpr(callee(functionDecl(functionName()))).bind("FuncCall"),
                this);

  // Currently, only exec functions support function pointer migration
  auto execFunctionName = [&]() {
    return hasAnyName("cufftExecC2C", "cufftExecR2C", "cufftExecC2R",
                      "cufftExecZ2Z", "cufftExecZ2D", "cufftExecD2Z");
  };
  MF.addMatcher(unaryOperator(hasOperatorName("&"),
                              hasUnaryOperand(declRefExpr(hasDeclaration(
                                  functionDecl(execFunctionName())))))
                    .bind("FuncPtr"),
                this);
}

void FFTFunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FuncCall");
  const UnaryOperator *UO = getNodeAsType<UnaryOperator>(Result, "FuncPtr");

  if (!CE) {
    auto TM = processFunctionPointer(UO);
    if (TM) {
      emplaceTransformation(TM);
    }
    return;
  }

  auto &SM = DpctGlobalInfo::getSourceManager();
  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  if (FuncName == "cufftGetVersion" || FuncName == "cufftGetProperty") {
    DpctGlobalInfo::getInstance().insertHeader(
        SM.getExpansionLoc(CE->getBeginLoc()), HT_DPCT_COMMON_Utils);
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  } else if (FuncName == "cufftSetStream" ||
             FuncName == "cufftCreate" || FuncName == "cufftDestroy" ||
             FuncName == "cufftPlan1d" || FuncName == "cufftMakePlan1d" ||
             FuncName == "cufftPlan2d" || FuncName == "cufftMakePlan2d" ||
             FuncName == "cufftPlan3d" || FuncName == "cufftMakePlan3d" ||
             FuncName == "cufftPlanMany" || FuncName == "cufftMakePlanMany" ||
             FuncName == "cufftMakePlanMany64" || FuncName == "cufftXtMakePlanMany" ||
             FuncName == "cufftExecC2C" || FuncName == "cufftExecZ2Z" ||
             FuncName == "cufftExecC2R" || FuncName == "cufftExecR2C" ||
             FuncName == "cufftExecZ2D" || FuncName == "cufftExecD2Z" ||
             FuncName == "cufftXtExec" || FuncName == "cufftGetSize1d" ||
             FuncName == "cufftGetSize2d" || FuncName == "cufftGetSize3d" ||
             FuncName == "cufftGetSizeMany" || FuncName == "cufftGetSize" ||
             FuncName == "cufftEstimate1d" || FuncName == "cufftEstimate2d" ||
             FuncName == "cufftEstimate3d" || FuncName == "cufftEstimateMany" ||
             FuncName == "cufftSetAutoAllocation" || FuncName == "cufftGetSizeMany64" ||
             FuncName == "cufftSetWorkArea") {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
}

REGISTER_RULE(FFTFunctionCallRule, PassKind::PK_Migration,
              RuleGroupKind::RK_FFT)

void DriverModuleAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto DriverModuleAPI = [&]() {
    return hasAnyName("cuModuleLoad", "cuModuleLoadData", "cuModuleLoadDataEx",
                      "cuModuleUnload", "cuModuleGetFunction", "cuLaunchKernel",
                      "cuModuleGetTexRef");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(DriverModuleAPI())), parentStmt()))
          .bind("call"),
      this);

  MF.addMatcher(callExpr(allOf(callee(functionDecl(DriverModuleAPI())),
                               unless(parentStmt())))
                    .bind("callUsed"),
                this);
}

void DriverModuleAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "callUsed"))) {
      return;
    }
  }

  std::string APIName = "";
  if (auto DC = CE->getDirectCallee()) {
    APIName = DC->getNameAsString();
  } else {
    return;
  }

  if (APIName == "cuModuleLoad") {
    report(CE->getBeginLoc(), Diagnostics::MODULE_LOAD, false,
           getStmtSpelling(CE->getArg(1)));
  } else if (APIName == "cuModuleLoadData" || APIName == "cuModuleLoadDataEx") {
    report(CE->getBeginLoc(), Diagnostics::MODULE_LOAD_DATA, false,
           getStmtSpelling(CE->getArg(1)));
  }

  if (isAssigned(CE) &&
      (APIName == "cuModuleLoad" || APIName == "cuModuleLoadData" ||
       APIName == "cuModuleLoadDataEx" || APIName == "cuModuleGetFunction")) {
    requestFeature(HelperFeatureEnum::device_ext);
    insertAroundStmt(CE, "DPCT_CHECK_ERROR(", ")");
  }

  ExprAnalysis EA;
  EA.analyze(CE);
  emplaceTransformation(EA.getReplacement());
}

REGISTER_RULE(DriverModuleAPIRule, PassKind::PK_Migration)

void DriverDeviceAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {

  auto DriverDeviceAPI = [&]() {
    return hasAnyName("cuDeviceGet", "cuDeviceComputeCapability",
                      "cuDriverGetVersion", "cuDeviceGetCount",
                      "cuDeviceGetAttribute", "cuDeviceGetName");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(DriverDeviceAPI())), parentStmt()))
          .bind("call"),
      this);

  MF.addMatcher(callExpr(allOf(callee(functionDecl(DriverDeviceAPI())),
                               unless(parentStmt())))
                    .bind("callUsed"),
                this);
}

void DriverDeviceAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  std::string APIName;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "callUsed"))) {
      return;
    }
    IsAssigned = true;
  }
  if (auto DC = CE->getDirectCallee()) {
    APIName = DC->getNameAsString();
  } else {
    return;
  }
  std::ostringstream OS;


  if (APIName == "cuDeviceGet") {
    if (IsAssigned)
      OS << "DPCT_CHECK_ERROR(";
    auto FirArg = CE->getArg(0)->IgnoreImplicitAsWritten();
    auto SecArg = CE->getArg(1)->IgnoreImplicitAsWritten();

    ExprAnalysis SecEA(SecArg);
    SecEA.analyze();
    std::string Rep;
    printDerefOp(OS, FirArg);
    OS << " = " << SecEA.getReplacedString();
    if (IsAssigned) {
      OS << ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDeviceGetName") {
    if (IsAssigned)
      OS << "DPCT_CHECK_ERROR(";
    auto FirArg = CE->getArg(0)->IgnoreImplicitAsWritten();
    auto SecArg = CE->getArg(1)->IgnoreImplicitAsWritten();
    auto ThrArg = CE->getArg(2)->IgnoreImplicitAsWritten();
    ExprAnalysis FirEA(FirArg);
    ExprAnalysis SecEA(SecArg);
    ExprAnalysis ThrEA(ThrArg);
    FirEA.analyze();
    SecEA.analyze();
    ThrEA.analyze();
    OS << "memcpy(" << FirEA.getReplacedString()
       << ", " + MapNames::getDpctNamespace() +
              "dev_mgr::instance().get_device("
       << ThrEA.getReplacedString() << ").get_info<"
       << MapNames::getClNamespace() << "info::device::name>().c_str(), "
       << SecEA.getReplacedString() << ")";
    requestFeature(HelperFeatureEnum::device_ext);
    if (IsAssigned) {
      OS << ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDeviceComputeCapability") {
    auto &SM = DpctGlobalInfo::getSourceManager();
    std::string Indent =
        getIndent(SM.getExpansionLoc(CE->getBeginLoc()), SM).str();
    if (IsAssigned)
      OS << "[&](){" << getNL();
    auto FirArg = CE->getArg(0)->IgnoreImplicitAsWritten();
    auto SecArg = CE->getArg(1)->IgnoreImplicitAsWritten();
    auto ThrArg = CE->getArg(2)->IgnoreImplicitAsWritten();
    std::string device_str;
    if (DpctGlobalInfo::useNoQueueDevice()) {
      device_str = DpctGlobalInfo::getGlobalDeviceName();
    } else {
      std::string ThrRep;
      ExprAnalysis EA(ThrArg);
      EA.analyze();
      ThrRep = EA.getReplacedString();
      device_str = MapNames::getDpctNamespace() +
                   "dev_mgr::instance().get_device(" + ThrRep + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (IsAssigned) {
      OS << Indent << "  ";
      printDerefOp(OS, FirArg);
      OS << " = " << MapNames::getDpctNamespace() << "get_major_version("
         << device_str << ");" << getNL();
      OS << Indent << "  ";
      printDerefOp(OS, SecArg);
      OS << " = " << MapNames::getDpctNamespace() << "get_minor_version("
         << device_str << ");" << getNL();
      OS << Indent << "  "
         << "return 0;" << getNL();
    } else {
      printDerefOp(OS, FirArg);
      OS << " = " << MapNames::getDpctNamespace() << "get_major_version("
         << device_str << ");" << getNL() << Indent;
      printDerefOp(OS, SecArg);
      OS << " = " << MapNames::getDpctNamespace() << "get_minor_version("
         << device_str << ")";
    }
    if (IsAssigned) {
      OS << Indent << "}()";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_LAMBDA, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDeviceGetCount") {
    if (IsAssigned)
      OS << "DPCT_CHECK_ERROR(";
    auto Arg = CE->getArg(0)->IgnoreImplicitAsWritten();
    printDerefOp(OS, Arg);
    OS << " = "
       << MapNames::getDpctNamespace() + "dev_mgr::instance().device_count()";
    requestFeature(HelperFeatureEnum::device_ext);
    if (IsAssigned) {
      OS << ")";
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDeviceGetAttribute") {
    auto SecArg = CE->getArg(1);
    if (auto DRE = dyn_cast<DeclRefExpr>(SecArg)) {
      auto AttributeName = DRE->getNameInfo().getAsString();
      auto Search = EnumConstantRule::EnumNamesMap.find(AttributeName);
      if (Search == EnumConstantRule::EnumNamesMap.end()) {
        report(CE->getBeginLoc(), Diagnostics::NOT_SUPPORTED_PARAMETER, false,
               APIName,
               "parameter " + getStmtSpelling(SecArg) + " is unsupported");
        return;
      }
      if (AttributeName == "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY" ||
          AttributeName == "CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT" ||
          AttributeName == "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK") {
        report(CE->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
          AttributeName, Search->second->NewName);
      }
    } else {
      report(CE->getBeginLoc(), Diagnostics::UNPROCESSED_DEVICE_ATTRIBUTE,
            false);
      return;
    }
  }
  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(APIName);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
}

REGISTER_RULE(DriverDeviceAPIRule, PassKind::PK_Migration)

void DriverContextAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto contextAPI = [&]() {
    return hasAnyName(
        "cuInit", "cuCtxCreate_v2", "cuCtxSetCurrent", "cuCtxGetCurrent",
        "cuCtxSynchronize", "cuCtxDestroy_v2", "cuDevicePrimaryCtxRetain",
        "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease",
        "cuCtxGetDevice", "cuCtxGetApiVersion", "cuCtxGetLimit");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(contextAPI())), parentStmt()))
          .bind("call"),
      this);

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(contextAPI())), unless(parentStmt())))
          .bind("callUsed"),
      this);
}

void DriverContextAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  std::string APIName;
  std::ostringstream OS;
  auto &SM = DpctGlobalInfo::getSourceManager();
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "callUsed")))
      return;
    IsAssigned = true;
  }

  if (auto DC = CE->getDirectCallee()) {
    APIName = DC->getNameAsString();
  } else {
    return;
  }

  if (!CallExprRewriterFactoryBase::RewriterMap)
    return;

  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(APIName);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  if (IsAssigned) {
    OS << "DPCT_CHECK_ERROR(";
  }
  if (APIName == "cuInit") {
    std::string Msg = "this call is redundant in SYCL.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             APIName, Msg);
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false, APIName,
             Msg);
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
    return;
  } else if (APIName == "cuCtxDestroy_v2" ||
             APIName == "cuDevicePrimaryCtxRelease_v2" ||
             APIName == "cuDevicePrimaryCtxRelease") {
    SourceLocation CallBegin(CE->getBeginLoc());
    SourceLocation CallEnd(CE->getEndLoc());

    bool IsMacroArg =
        SM.isMacroArgExpansion(CallBegin) && SM.isMacroArgExpansion(CallEnd);

    if (CallBegin.isMacroID() && IsMacroArg) {
      CallBegin = SM.getImmediateSpellingLoc(CallBegin);
      CallBegin = SM.getExpansionLoc(CallBegin);
    } else if (CallBegin.isMacroID()) {
      CallBegin = SM.getExpansionLoc(CallBegin);
    }

    if (CallEnd.isMacroID() && IsMacroArg) {
      CallEnd = SM.getImmediateSpellingLoc(CallEnd);
      CallEnd = SM.getExpansionLoc(CallEnd);
    } else if (CallEnd.isMacroID()) {
      CallEnd = SM.getExpansionLoc(CallEnd);
    }
    CallEnd = CallEnd.getLocWithOffset(1);

    std::string Msg = "this call is redundant in SYCL.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             APIName, Msg);
      emplaceTransformation(replaceText(CallBegin, CallEnd, "0", SM));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false, APIName,
             Msg);
      CallEnd = CallEnd.getLocWithOffset(1);
      emplaceTransformation(replaceText(CallBegin, CallEnd, "", SM));
    }
    return;
  } else if (APIName == "cuCtxSetCurrent") {
    if (DpctGlobalInfo::useNoQueueDevice()) {
      OS << "0";
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             "cuCtxSetCurrent",
             "it is redundant if it is migrated with option "
             "--helper-function-preference=no-queue-device "
             "which declares a global SYCL device and queue.");
    } else {
      auto Arg = CE->getArg(0)->IgnoreImplicitAsWritten();
      ExprAnalysis EA(Arg);
      EA.analyze();
      OS << MapNames::getDpctNamespace() + "select_device("
         << EA.getReplacedString() << ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
  } else if (APIName == "cuCtxGetCurrent") {
    auto Arg = CE->getArg(0)->IgnoreImplicitAsWritten();
    printDerefOp(OS, Arg);
    OS << " = ";
    if (DpctGlobalInfo::useNoQueueDevice()) {
      OS << "0";
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             "cuCtxGetCurrent",
             "it is redundant if it is migrated with option "
             "--helper-function-preference=no-queue-device "
             "which declares a global SYCL device and queue.");
    } else {
      OS << MapNames::getDpctNamespace() +
                "dev_mgr::instance().current_device_id()";
      requestFeature(HelperFeatureEnum::device_ext);
    }
  } else if (APIName == "cuCtxSynchronize") {
    OS << MapNames::getDpctNamespace() +
              "get_current_device().queues_wait_and_throw()";
    requestFeature(HelperFeatureEnum::device_ext);
  } else if (APIName == "cuCtxGetLimit") {
    auto SecArg = CE->getArg(1);
    if (auto DRE = dyn_cast<DeclRefExpr>(SecArg)) {
      std::string AttributeName = DRE->getNameInfo().getAsString();
      auto Search = EnumConstantRule::EnumNamesMap.find(AttributeName);
      if (Search != EnumConstantRule::EnumNamesMap.end()) {
        printDerefOp(OS, CE->getArg(0));
        OS << " = " << Search->second->NewName;
      } else {
        auto Msg = MapNames::RemovedAPIWarningMessage.find(APIName);
        if (IsAssigned) {
          report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
                MapNames::ITFName.at(APIName), Msg->second);
          emplaceTransformation(new ReplaceStmt(CE, "0"));
        } else {
          report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
                MapNames::ITFName.at(APIName), Msg->second);
          emplaceTransformation(new ReplaceStmt(CE, ""));
        }
        return;
      }
    }
  }
  if (IsAssigned) {
    OS << ")";
    requestFeature(HelperFeatureEnum::device_ext);
  }
  emplaceTransformation(new ReplaceStmt(CE, OS.str()));
}

REGISTER_RULE(DriverContextAPIRule, PassKind::PK_Migration)
// In host device function, macro __CUDA_ARCH__ is used to differentiate
// different code blocks. And migration of the two code blocks will result in
// different requirement on parameter of the function signature, device code
// block requires sycl::nd_item. Current design does two round parses and
// generates an extra host version function. The first-round parse dpct defines
// macro __CUDA_ARCH__ and the second-round parse dpct undefines macro
// __CUDA_ARCH__ to generate replacement for different code blocks.
// Implementation steps as follow:
//   1. Match all host device function's declaration and caller.
//   2. Check if macro CUDA_ARCH used to differentiate different
//   code blocks and called in host side, then record relative information.
//   3. Record host device function call expression.
//   4. All these information will be used in post-process stage and generate
//   final replacements.
// Condition to trigger the rule:
//   1. The function has host and device attribute.
//   2. The function uses macro CUDA_ARCH used to differentiate different code
//   blocks.
//   3. The function has been called in host side.
// Example code:
// __host__ __device__ int foo() {
//    #ifdef __CUDA_ARCH__
//      return threadIdx.x;
//    #else
//      return -1;
//    #endif
// }
//
// __global__ void kernel() {
//   foo();
// }
//
// int main() {
//   foo();
// }
//
// After migration:
// int foo(sycl::nd_item<3> item) {
//   return item.get_local_id(2);
// }
//
// int foo_host_ct1() {
//   return -1;
// }
//
// void kernel(sycl::nd_item<3> item) {
//   foo(item);
// }
//
// int main() {
//   foo_host_ct1();
// }
void CudaArchMacroRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto HostDeviceFunctionMatcher =
      functionDecl(allOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAHost),
                         unless(cxxMethodDecl())));
  MF.addMatcher(callExpr(callee(HostDeviceFunctionMatcher)).bind("callExpr"),
                this);
  MF.addMatcher(HostDeviceFunctionMatcher.bind("funcDecl"), this);
}
void CudaArchMacroRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &Global = DpctGlobalInfo::getInstance();
  auto &CT = DpctGlobalInfo::getContext();
  DpctNameGenerator DNG;
  const FunctionDecl *FD =
      getAssistNodeAsType<FunctionDecl>(Result, "funcDecl");
  auto &HDFIMap = Global.getHostDeviceFuncInfoMap();
  HostDeviceFuncLocInfo HDFLI;
  // process __host__ __device__ function definition except overloaded operator
  if (FD && (Global.getRunRound() == 0) && !FD->isOverloadedOperator() &&
      FD->getTemplateSpecializationKind() ==
          TemplateSpecializationKind::TSK_Undeclared) {
    auto NameInfo = FD->getNameInfo();
    // TODO: add support for macro
    if (NameInfo.getBeginLoc().isMacroID())
      return;
    auto BeginLoc = SM.getExpansionLoc(FD->getBeginLoc());
    if (FD->isTemplated()) {
      auto P = CT.getParents(*FD);
      if (!P.size())
        return;
      const FunctionTemplateDecl *FTD = P[0].get<FunctionTemplateDecl>();
      if (FTD)
        BeginLoc = SM.getExpansionLoc(FTD->getBeginLoc());
    }
    auto EndLoc = SM.getExpansionLoc(FD->getEndLoc());
    auto Beg = Global.getLocInfo(BeginLoc);
    auto End = Global.getLocInfo(EndLoc);
    auto T = Lexer::findNextToken(EndLoc, SM, LangOptions());
    if (T.has_value() && T.value().is(tok::TokenKind::semi)) {
      End = Global.getLocInfo(T.value().getLocation());
    }
    auto FileInfo = DpctGlobalInfo::getInstance().insertFile(Beg.first);
    std::string &FileContent = FileInfo->getFileContent();
    auto NameLocInfo = Global.getLocInfo(NameInfo.getBeginLoc());
    std::string ManglingName = DNG.getName(FD);
    Global.getMainSourceFileMap()[NameLocInfo.first].push_back(
        Global.getMainFile()->getFilePath());
    HDFLI.FuncStartOffset = Beg.second;
    HDFLI.FuncEndOffset = End.second;
    HDFLI.FuncNameOffset = NameLocInfo.second + NameInfo.getAsString().length();
    HDFLI.FuncContentCache =
        FileContent.substr(Beg.second, End.second - Beg.second + 1);
    HDFLI.FilePath = NameLocInfo.first;
    if (!FD->isThisDeclarationADefinition()) {
      HDFLI.Type = HDFuncInfoType::HDFI_Decl;
      HDFIMap[ManglingName].LocInfos.insert(
          {HDFLI.FilePath + "Decl" + std::to_string(HDFLI.FuncEndOffset),
           HDFLI});
      return;
    }
    HDFLI.Type = HDFuncInfoType::HDFI_Def;
    bool NeedInsert = false;
    for (auto &Info : Global.getCudaArchPPInfoMap()[FileInfo->getFilePath()]) {
      if ((Info.first > Beg.second) && (Info.first < End.second) &&
          (!Info.second.ElInfo.empty() ||
           (Info.second.IfInfo.DirectiveLoc &&
            (Info.second.DT != IfType::IT_Unknow)))) {
        Info.second.isInHDFunc = true;
        NeedInsert = true;
      }
    }
    if (NeedInsert) {
      HDFIMap[ManglingName].isDefInserted = true;
      HDFIMap[ManglingName].LocInfos.insert(
          {HDFLI.FilePath + "Def" + std::to_string(HDFLI.FuncEndOffset),
           HDFLI});
    }
  } // address __host__ __device__ function call
  else if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "callExpr")) {
    // TODO: add support for macro
    if (CE->getBeginLoc().isMacroID())
      return;
    if (auto *PF = DpctGlobalInfo::getParentFunction(CE)) {
      if ((PF->hasAttr<CUDADeviceAttr>() && !PF->hasAttr<CUDAHostAttr>()) ||
          PF->hasAttr<CUDAGlobalAttr>()) {
        return;
      } else if (PF->hasAttr<CUDADeviceAttr>() && PF->hasAttr<CUDAHostAttr>()) {
        HDFLI.CalledByHostDeviceFunction = true;
      }
    }
    const FunctionDecl *DC = CE->getDirectCallee();
    if (DC) {
      unsigned int Offset = DC->getNameAsString().length();
      std::string ManglingName(DNG.getName(DC));
      if (DC->isTemplateInstantiation()) {
        if (auto DFT = DC->getPrimaryTemplate()) {
          const FunctionDecl *TFD = DFT->getTemplatedDecl();
          if (TFD)
            ManglingName = DNG.getName(TFD);
        }
      }
      auto LocInfo = Global.getLocInfo(CE->getBeginLoc());
      Global.getMainSourceFileMap()[LocInfo.first].push_back(
        Global.getMainFile()->getFilePath());
      HDFLI.Type = HDFuncInfoType::HDFI_Call;
      HDFLI.FilePath = LocInfo.first;
      HDFLI.FuncEndOffset = LocInfo.second + Offset;
      HDFIMap[ManglingName].LocInfos.insert(
          {HDFLI.FilePath + "Call" + std::to_string(HDFLI.FuncEndOffset),
           HDFLI});
      HDFIMap[ManglingName].isCalledInHost = true;
    }
  }
}
REGISTER_RULE(CudaArchMacroRule, PassKind::PK_Migration)

REGISTER_RULE(ConfusableIdentifierDetectionRule, PassKind::PK_Migration)

REGISTER_RULE(MisleadingBidirectionalRule, PassKind::PK_Migration)

REGISTER_RULE(CuDNNTypeRule, PassKind::PK_Migration, RuleGroupKind::RK_DNN)

REGISTER_RULE(CuDNNAPIRule, PassKind::PK_Migration, RuleGroupKind::RK_DNN)

REGISTER_RULE(NCCLRule, PassKind::PK_Migration, RuleGroupKind::RK_NCCL)

REGISTER_RULE(LIBCURule, PassKind::PK_Migration, RuleGroupKind::RK_Libcu)

REGISTER_RULE(ThrustAPIRule, PassKind::PK_Migration, RuleGroupKind::RK_Thrust)

REGISTER_RULE(ThrustTypeRule, PassKind::PK_Migration, RuleGroupKind::RK_Thrust)

REGISTER_RULE(WMMARule, PassKind::PK_Analysis)

REGISTER_RULE(ForLoopUnrollRule, PassKind::PK_Migration)

void ComplexAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto ComplexAPI = [&]() {
    return hasAnyName("make_cuDoubleComplex", "cuCreal", "cuCrealf", "cuCimag",
                      "cuCimagf", "cuCadd", "cuCsub", "cuCmul", "cuCdiv",
                      "cuCabs", "cuConj", "make_cuFloatComplex", "cuCaddf",
                      "cuCsubf", "cuCmulf", "cuCdivf", "cuCabsf", "cuConjf",
                      "make_cuComplex", "__saturatef", "cuComplexDoubleToFloat",
                      "cuComplexFloatToDouble");
  };

  MF.addMatcher(callExpr(callee(functionDecl(ComplexAPI()))).bind("call"),
                this);
}

void ComplexAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call")) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

REGISTER_RULE(ComplexAPIRule, PassKind::PK_Migration)

void TemplateSpecializationTypeLocRule::registerMatcher(
    ast_matchers::MatchFinder &MF) {
  auto TargetTypeName = [&]() {
    return hasAnyName("thrust::not_equal_to", "thrust::constant_iterator",
                      "thrust::system::cuda::experimental::pinned_allocator",
                      "thrust::random::default_random_engine",
                      "thrust::random::uniform_real_distribution",
                      "thrust::random::normal_distribution",
                      "thrust::random::linear_congruential_engine",
                      "thrust::random::uniform_int_distribution");
  };

  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(TargetTypeName())))))
          .bind("loc"),
      this);

  MF.addMatcher(declRefExpr().bind("declRefExpr"), this);
}

void TemplateSpecializationTypeLocRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {

  const DeclRefExpr *DRE = getNodeAsType<DeclRefExpr>(Result, "declRefExpr");
  if (DRE) {
    std::string TypeName = DpctGlobalInfo::getTypeName(DRE->getType());
    std::string Name = DRE->getNameInfo().getName().getAsString();
    if (TypeName.find("thrust::random::linear_congruential_engine") !=
            std::string::npos &&
        Name == "max") {
      emplaceTransformation(
          new ReplaceStmt(DRE, "oneapi::dpl::default_engine::max()"));
    }
  }

  if (auto TL = getNodeAsType<TypeLoc>(Result, "loc")) {
    ExprAnalysis EA;
    EA.analyze(*TL);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

REGISTER_RULE(TemplateSpecializationTypeLocRule, PassKind::PK_Migration)

void CudaStreamCastRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
     castExpr(hasType(qualType(hasCanonicalType(
        qualType(pointsTo(namedDecl(hasName("CUstream_st"))))))))
     .bind("cast"),
     this);
}

void CudaStreamCastRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto CE = getNodeAsType<CastExpr>(Result, "cast")) {
    if (CE->getCastKind() == clang::CK_LValueToRValue ||
        CE->getCastKind() == clang::CK_NoOp)
      return;

    if (isDefaultStream(CE->getSubExpr())) {
      if (isPlaceholderIdxDuplicated(CE->getSubExpr()))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE->getSubExpr(),
                           HelperFuncType::HFT_DefaultQueue);
      emplaceTransformation(new ReplaceStmt(
          CE, "&{{NEEDREPLACEQ" + std::to_string(Index) + "}}"));
    } else if (CE->getSubExpr()->getType()->isIntegerType()) {
      requestFeature(HelperFeatureEnum::device_ext);
      emplaceTransformation(new ReplaceStmt(
          CE, MapNames::getDpctNamespace() + "int_as_queue_ptr(" +
                  ExprAnalysis::ref(CE->getSubExpr()) + ")"));
    }
  }
}

REGISTER_RULE(CudaStreamCastRule, PassKind::PK_Migration)

void CudaExtentRule::registerMatcher(ast_matchers::MatchFinder &MF) {

  // 1. Match any cudaExtent TypeLoc.
  MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(
                            namedDecl(hasAnyName("cudaExtent", "cudaPos"))))))
                    .bind("loc"),
                this);

  // 2. Match cudaExtent default ctor.
  //    cudaExtent()    - CXXTemporaryObjectExpr, handled by (1) and (2).
  //    cudaExtent a    - VarDecl, handled by (1) and (2)
  MF.addMatcher(
      cxxConstructExpr(hasType(namedDecl(hasAnyName("cudaExtent", "cudaPos"))))
          .bind("defaultCtor"),
      this);

  // 3. Match field declaration, which doesn't has an in-class initializer.
  //    The in-class initializer case will handled by other matchers.
  MF.addMatcher(
      fieldDecl(hasType(namedDecl(hasAnyName("cudaExtent", "cudaPos"))),
                unless(hasInClassInitializer(anything())))
          .bind("fieldDeclHasNoInit"),
      this);

  // 4. Match c++ initializer_list, which has cudaExtent type.
  MF.addMatcher(
      initListExpr(hasType(namedDecl(hasAnyName("cudaExtent", "cudaPos"))))
          .bind("initListExpr"),
      this);
}

void CudaExtentRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {

  // cudaExtent -> sycl::range<3>
  if (const TypeLoc *TL = getAssistNodeAsType<TypeLoc>(Result, "loc")) {
    ExprAnalysis EA;
    EA.analyze(*TL);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  // cudaExtent a;  -> sycl::range<3> a{0, 0, 0};
  // cudaExtent()   -> sycl::range<3>{0, 0, 0};
  // struct Foo { cudaExtent e; Foo() : e() {} }; -> struct Foo { sycl::range<3> e; Foo() : e{0, 0, 0} {} };
  if (const CXXConstructExpr *Ctor =
          getNodeAsType<CXXConstructExpr>(Result, "defaultCtor")) {

    // Ignore implicit move/copy ctor
    if (Ctor->getNumArgs() != 0)
      return;
    CharSourceRange CSR;
    SourceRange SR = Ctor->getParenOrBraceRange();
    auto &SM = DpctGlobalInfo::getSourceManager();
    std::string Replacement = "{0, 0, 0}";

    if (SR.isInvalid()) {
      auto CtorLoc = Ctor->getLocation().isMacroID()
                         ? SM.getSpellingLoc(Ctor->getLocation())
                         : Ctor->getLocation();
      auto CtorEndLoc = Lexer::getLocForEndOfToken(
          CtorLoc, 0, SM, DpctGlobalInfo::getContext().getLangOpts());
      CSR = CharSourceRange(SourceRange(CtorEndLoc, CtorEndLoc), false);
      DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            SM, CSR, Replacement, nullptr));
    } else {
      auto CtorEndLoc = Lexer::getLocForEndOfToken(
          SR.getEnd(), 0, SM, DpctGlobalInfo::getContext().getLangOpts());
      CharSourceRange CSR(SourceRange(SR.getBegin(), CtorEndLoc), false);
      DpctGlobalInfo::getInstance().addReplacement(
          std::make_shared<ExtReplacement>(
              SM, CSR, Replacement, nullptr));
    }
    return;
  }

  // struct Foo { cudaExtent a; }; -> struct Foo { syc::range<3> a{0, 0, 0}; };
  if (const FieldDecl *FD =
          getNodeAsType<FieldDecl>(Result, "fieldDeclHasNoInit")) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    auto IdentBeginLoc = FD->getEndLoc().isMacroID()
                             ? SM.getSpellingLoc(FD->getEndLoc())
                             : FD->getEndLoc();
    auto IdentEndLoc = Lexer::getLocForEndOfToken(
        IdentBeginLoc, 0, SM, DpctGlobalInfo::getContext().getLangOpts());
    CharSourceRange CSR =
        CharSourceRange(SourceRange(IdentEndLoc, IdentEndLoc), false);
    std::string Replacement = "{0, 0, 0}";
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            SM, CSR, Replacement, nullptr));
    return;
  }

  // cudaExtent a{};          -> sycl::range<3> a{0, 0, 0};
  // cudaExtent b{1};         -> sycl::range<3> b{1, 0, 0};
  // cudaExtent c{1, 1};      -> sycl::range<3> c{1, 1, 0};
  // cudaExtent d{1, 1, 1};   -> sycl::range<3> d{1, 1, 1};
  // cudaExtent({1, 1, 1});   -> sycl::range<3>({1, 1, 1});
  if (const InitListExpr *Init =
          getNodeAsType<InitListExpr>(Result, "initListExpr")) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    std::string Replacement;
    llvm::raw_string_ostream OS(Replacement);
    OS << "{";
    for (size_t I = 0; I < Init->getNumInits(); ++I) {
      const Expr *E = Init->getInit(I);
      if (isa<ImplicitValueInitExpr>(E)) {
        OS << "0";
      } else {
        ExprAnalysis EA;
        EA.analyze(E);
        OS << EA.getReplacedString();
      }
      if (I + 1 < Init->getNumInits())
        OS << ", ";
    }
    OS << "}";
    OS.flush();
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            SM, Init, Replacement, nullptr));
    return;
  }
}

REGISTER_RULE(CudaExtentRule, PassKind::PK_Analysis)

void CudaUuidRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(namedDecl(
                               hasAnyName("CUuuid_st", "cudaUUID_t")))),
                           member(hasName("bytes")))
                    .bind("UUID_bytes"),
                this);
}

void CudaUuidRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto ME = Result.Nodes.getNodeAs<MemberExpr>("UUID_bytes")) {
    const auto SM = Result.SourceManager;
    const auto Begin = SM->getSpellingLoc(ME->getOperatorLoc());
    return emplaceTransformation(new ReplaceText(Begin, 6, ""));
  }
}

REGISTER_RULE(CudaUuidRule, PassKind::PK_Analysis)

void TypeRemoveRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(typedefDecl(
                            hasAnyName("cudaLaunchAttributeValue"))))))
                    .bind("TypeWarning"),
                this);
  MF.addMatcher(
      binaryOperator(allOf(isAssignmentOperator(),
                           hasLHS(hasDescendant(memberExpr(hasType(namedDecl(
                               hasAnyName("cudaAccessPolicyWindow"))))))))
          .bind("AssignStmtRemove"),
      this);
}

void TypeRemoveRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto TL = getNodeAsType<TypeLoc>(Result, "TypeWarning")) {
    report(getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc()).getBegin(),
           Diagnostics::API_NOT_MIGRATED, false,
           getStmtSpelling(TL->getSourceRange()));
  }
  if (auto BO = getNodeAsType<BinaryOperator>(Result, "AssignStmtRemove"))
    emplaceTransformation(new ReplaceStmt(BO, ""));
  return;
}

REGISTER_RULE(TypeRemoveRule, PassKind::PK_Analysis)
