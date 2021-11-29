//===--- ASTTraversal.cpp --------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "CallExprRewriter.h"
#include "CustomHelperFiles.h"
#include "ExprAnalysis.h"
#include "GAnalytics.h"
#include "SaveNewFiles.h"
#include "TextModification.h"
#include "Utility.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Tooling/Tooling.h"
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

TextModification *replaceText(SourceLocation Begin, SourceLocation End,
                              std::string &&Str, const SourceManager &SM) {
  auto Length = SM.getFileOffset(End) - SM.getFileOffset(Begin);
  if (Length > 0) {
    return new ReplaceText(Begin, Length, std::move(Str));
  }
  return nullptr;
}

/// Return a TextModication that removes nth argument of the CallExpr,
/// together with the preceding comma.
TextModification *removeArg(const CallExpr *C, unsigned n,
                            const SourceManager &SM) {
  if (C->getNumArgs() <= n)
    return nullptr;
  if (C->getArg(n)->isDefaultArgument())
    return nullptr;

  SourceLocation Begin, End;
  if (n) {
    Begin = getStmtExpansionSourceRange(C->getArg(n - 1)).getEnd();
    Begin = Begin.getLocWithOffset(Lexer::MeasureTokenLength(
        Begin, SM, dpct::DpctGlobalInfo::getContext().getLangOpts()));
    End = getStmtExpansionSourceRange(C->getArg(n)).getEnd();
    End = End.getLocWithOffset(Lexer::MeasureTokenLength(
        End, SM, dpct::DpctGlobalInfo::getContext().getLangOpts()));
  } else {
    Begin = getStmtExpansionSourceRange(C->getArg(n)).getBegin();
    if (C->getNumArgs() > 1) {
      End = getStmtExpansionSourceRange(C->getArg(n + 1)).getBegin();
    } else {
      End = getStmtExpansionSourceRange(C->getArg(n)).getEnd();
      End = End.getLocWithOffset(Lexer::MeasureTokenLength(
          End, SM, dpct::DpctGlobalInfo::getContext().getLangOpts()));
    }
  }
  return replaceText(Begin, End, "", SM);
}

auto parentStmt = []() {
  return anyOf(hasParent(compoundStmt()), hasParent(forStmt()),
               hasParent(whileStmt()), hasParent(doStmt()),
               hasParent(ifStmt()));
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
bool IncludesCallbacks::isInRoot(SourceLocation Loc) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(Loc).str();
  return !isDirectory(InFile) && isChildOrSamePath(InRoot, InFile);
}

int IncludesCallbacks::findPoundSign(SourceLocation DirectiveStart) {
  std::pair<FileID, unsigned> LocInfo =
      SM.getDecomposedSpellingLoc(DirectiveStart);

  bool CharDataInvalid = false;
  auto Entry = SM.getSLocEntry(LocInfo.first, &CharDataInvalid);
  if (CharDataInvalid || !Entry.isFile()) {
    return -1;
  }
  llvm::Optional<llvm::MemoryBufferRef> Buffer =
      Entry.getFile().getContentCache().getBufferOrNone(
          SM.getDiagnostics(), SM.getFileManager(), SourceLocation());
  if (!Buffer.hasValue())
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
void IncludesCallbacks::ReplaceCuMacro(const Token &MacroNameTok) {
  bool IsInRoot = isInRoot(MacroNameTok.getLocation());
  if (!IsInRoot) {
    return;
  }
  if (!MacroNameTok.getIdentifierInfo()) {
    return;
  }
  std::string MacroName = MacroNameTok.getIdentifierInfo()->getName().str();
  auto Iter = MapNames::MacrosMap.find(MacroName);
  if (Iter != MapNames::MacrosMap.end()) {
    std::string ReplacedMacroName = Iter->second;
    auto Repl = std::make_shared<ReplaceToken>(MacroNameTok.getLocation(),
                                               std::move(ReplacedMacroName));
    if (MacroName == "__CUDA_ARCH__") {
      requestFeature(HelperFeatureEnum::Dpct_dpct_compatibility_temp,
                     MacroNameTok.getLocation());
      auto &Map = DpctGlobalInfo::getInstance().getCudaArchMacroReplSet();
      Map.insert(Repl->getReplacement(DpctGlobalInfo::getContext()));
      return;
    }
    if (MacroName == "__CUDACC__" &&
        !MacroNameTok.getIdentifierInfo()->hasMacroDefinition())
      return;
    TransformSet.emplace_back(Repl);
  }
}

void IncludesCallbacks::MacroDefined(const Token &MacroNameTok,
                                     const MacroDirective *MD) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(MacroNameTok.getLocation()).str();
  bool IsInRoot = !isDirectory(InFile) && isChildOrSamePath(InRoot, InFile);

  size_t i;
  // Record all macro define locations
  for (i = 0; i < MD->getMacroInfo()->getNumTokens(); i++) {
    std::shared_ptr<dpct::DpctGlobalInfo::MacroDefRecord> R =
        std::make_shared<dpct::DpctGlobalInfo::MacroDefRecord>(
            MacroNameTok.getLocation(), IsInRoot);
    dpct::DpctGlobalInfo::getMacroTokenToMacroDefineLoc()[getHashStrFromLoc(
        MD->getMacroInfo()->getReplacementToken(i).getLocation())] = R;
  }

  if (!IsInRoot) {
    return;
  }

  auto MI = MD->getMacroInfo();
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
        requestFeature(HelperFeatureEnum::Dpct_dpct_compatibility_temp,
                       Iter->getLocation());
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
          dpct::DpctGlobalInfo::getCompilerInstance(), &TransformSet, false,
          MacroNameTok.getIdentifierInfo()->getName().str());
    }
  }
}

void IncludesCallbacks::MacroExpands(const Token &MacroNameTok,
                                     const MacroDefinition &MD,
                                     SourceRange Range, const MacroArgs *Args) {
  std::string InRoot = ATM.InRoot;
  std::string InFile =
      SM.getFilename(SM.getSpellingLoc(MacroNameTok.getLocation())).str();
  bool IsInRoot = !isDirectory(InFile) && isChildOrSamePath(InRoot, InFile);

  if (!MD.getMacroInfo())
    return;
  if (MD.getMacroInfo()->getNumTokens() > 0) {
    std::string HashKey = "";
    if (MD.getMacroInfo()->getReplacementToken(0).getLocation().isValid()) {
      HashKey = getCombinedStrFromLoc(
          MD.getMacroInfo()->getReplacementToken(0).getLocation());
    } else {
      HashKey = "InvalidLoc";
    }

    dpct::DpctGlobalInfo::getExpansionRangeBeginSet().insert(
        getCombinedStrFromLoc(Range.getBegin()));
    if (dpct::DpctGlobalInfo::getMacroDefines().find(HashKey) ==
        dpct::DpctGlobalInfo::getMacroDefines().end()) {
      // Record all processed macro definition
      dpct::DpctGlobalInfo::getMacroDefines()[HashKey] = true;
      size_t i;
      // Record all tokens in the macro definition
      for (i = 0; i < MD.getMacroInfo()->getNumTokens(); i++) {
        std::shared_ptr<dpct::DpctGlobalInfo::MacroExpansionRecord> R =
            std::make_shared<dpct::DpctGlobalInfo::MacroExpansionRecord>(
                MacroNameTok.getIdentifierInfo(), MD.getMacroInfo(), Range,
                IsInRoot, i);
        dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord()
            [getCombinedStrFromLoc(
                MD.getMacroInfo()->getReplacementToken(i).getLocation())] = R;
      }
    }

    // If PredefinedStreamName is used with concatinated macro token,
    // detect the previous macro expansion and
    std::string MacroNameStr;
    if (auto Identifier = MacroNameTok.getIdentifierInfo())
      MacroNameStr = Identifier->getName().str();
    if (MapNames::PredefinedStreamName.find(MacroNameStr) !=
        MapNames::PredefinedStreamName.end()) {
      // Currently, only support examples like,
      // #define CONCATE(name) cuda##name
      // which contains 3 tokens and the 2nd token is ##.
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
        requestFeature(HelperFeatureEnum::Device_get_default_queue,
                       DefRange.getBegin());
        TransformSet.emplace_back(new ReplaceText(
            DefRange.getBegin(), Length,
            "&" + MapNames::getDpctNamespace() + "get_default_queue()"));
      }
    }

    // Record (#tokens, name of the 2nd token, range) as a tuple
    SourceRange LastRange = Range;
    dpct::DpctGlobalInfo::LastMacroRecord =
        std::make_tuple<unsigned int, std::string, SourceRange>(
            MD.getMacroInfo()->getNumTokens(),
            MD.getMacroInfo()->getNumTokens() >= 3
                ? std::string(
                      MD.getMacroInfo()->getReplacementToken(1).getName())
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
          Range.getBegin())] = Range.getEnd();
    }
  }

  // In order to check whether __constant__ macro is empty, we first record
  // the expansion location of the __constant__, then check each __annotate__
  // macro, if the expansion locations are same and the content is empty, then
  // it means this __constant__ variable is used in host.
  // In this case, we need add "host_constant" flag in the replacement of
  // removing
  // "__constant__"; and record the offset of the begining of this line for
  // finding this replacement in MemVarRule. Since the varible name is difficult
  // to get here, the warning is also emitted in MemVarRule.
  if (MacroNameTok.getKind() == tok::identifier &&
      MacroNameTok.getIdentifierInfo() &&
      MacroNameTok.getIdentifierInfo()->getName() == "__annotate__" &&
      MD.getMacroInfo() && !MD.getMacroInfo()->param_empty()) {
    SourceLocation Loc = SM.getExpansionLoc(Range.getBegin());

    if (auto TM = DpctGlobalInfo::getInstance().findConstantMacroTMInfo(Loc)) {
      TM->setLineBeginOffset(getOffsetOfLineBegin(Loc, SM));
      if (MD.getMacroInfo()->getNumTokens() == 0) {
        TM->setConstantFlag(dpct::ConstantFlagType::Host);
      } else {
        TM->setConstantFlag(dpct::ConstantFlagType::Device);
      }
    }
  }

  if (!IsInRoot) {
    return;
  }

  if (MacroNameTok.getIdentifierInfo() &&
      MacroNameTok.getIdentifierInfo()->getName() == "__CUDA_ARCH__") {
    requestFeature(HelperFeatureEnum::Dpct_dpct_compatibility_temp,
                   Range.getBegin());
    auto &ReplMap = DpctGlobalInfo::getInstance().getCudaArchMacroReplSet();
    auto Repl = std::make_shared<ReplaceText>(Range.getBegin(), 13,
                                              "DPCT_COMPATIBILITY_TEMP");
    ReplMap.insert(Repl->getReplacement(DpctGlobalInfo::getContext()));
  } else if (MacroNameTok.getIdentifierInfo() &&
             MacroNameTok.getIdentifierInfo()->getName() == "CUFFT_FORWARD") {
    TransformSet.emplace_back(new ReplaceText(Range.getBegin(), 13, "-1"));
  } else if (MacroNameTok.getIdentifierInfo() &&
             MacroNameTok.getIdentifierInfo()->getName() == "CUFFT_INVERSE") {
    TransformSet.emplace_back(new ReplaceText(Range.getBegin(), 13, "1"));
  }

  // For the un-specialized struct, there is no AST for the extern function
  // declaration in its member function body in Windows. e.g: template <typename
  // T> struct foo
  //{
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
       Name == "__shared__")) {
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

  if (TKind == tok::identifier && Name == "__forceinline__") {
    TransformSet.emplace_back(
        new ReplaceToken(Range.getBegin(), "__dpct_inline__"));
  } else if (TKind == tok::identifier && Name == "CUDART_CB") {
#ifdef _WIN32
    TransformSet.emplace_back(
        new ReplaceText(Range.getBegin(), 9, "__stdcall"));
#else
    TransformSet.emplace_back(removeMacroInvocationAndTrailingSpaces(Range));
#endif
  }

  auto Iter = MapNames::HostAllocSet.find(Name.str());
  if (TKind == tok::identifier && Iter != MapNames::HostAllocSet.end()) {
    if (MD.getMacroInfo()->getNumTokens() == 1) {
      auto ReplToken = MD.getMacroInfo()->getReplacementToken(0);
      if (ReplToken.getKind() == tok::numeric_constant) {
        TransformSet.emplace_back(new ReplaceToken(Range.getBegin(), "0"));
        DiagnosticsUtils::report(Range.getBegin(),
                                 Diagnostics::HOSTALLOCMACRO_NO_MEANING,
                                 dpct::DpctGlobalInfo::getCompilerInstance(),
                                 &TransformSet, false, Name.str());
      }
    }
  }

  if (auto MI = MD.getMacroInfo()) {
    if (MI->getNumTokens() > 0) {
      DpctGlobalInfo::getInstance().removeAtomicInfo(
          getHashStrFromLoc(MI->getReplacementToken(0).getLocation()));
    }
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
  if (isInRoot(Loc)) {
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
  if (!isInRoot(Loc))
    return;
  SourceLocation MacroLoc = MacroNameTok.getLocation();
  if (!MacroNameTok.getIdentifierInfo()) {
    return;
  }
  std::string MacroName = MacroNameTok.getIdentifierInfo()->getName().str();
  if (MacroName == "__CUDA_ARCH__" && DpctGlobalInfo::getRunRound() == 0) {
    requestFeature(HelperFeatureEnum::Dpct_dpct_compatibility_temp, Loc);
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
  if (!isInRoot(Loc))
    return;
  SourceLocation MacroLoc = MacroNameTok.getLocation();
  if (!MacroNameTok.getIdentifierInfo()) {
    return;
  }
  std::string MacroName = MacroNameTok.getIdentifierInfo()->getName().str();
  if (MacroName == "__CUDA_ARCH__" && DpctGlobalInfo::getRunRound() == 0) {
    requestFeature(HelperFeatureEnum::Dpct_dpct_compatibility_temp, Loc);
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
  if (!isInRoot(MacroLoc))
    return;
  if (MacroName == "__CUDA_ARCH__") {
    requestFeature(HelperFeatureEnum::Dpct_dpct_compatibility_temp, MacroLoc);
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
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(Loc).str();
  bool IsInRoot = !isDirectory(InFile) && isChildOrSamePath(InRoot, InFile);
  if (IsInRoot) {
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
    auto &ReplMap = DpctGlobalInfo::getInstance().getCudaArchMacroReplSet();
    while (Found != std::string::npos) {
      // found one, insert replace for it
      SourceLocation IB = Begin.getLocWithOffset(Found);
      SourceLocation IE = IB.getLocWithOffset(MacroName.length());
      CharSourceRange InsertRange(SourceRange(IB, IE), false);
      auto Repl = std::make_shared<ReplaceInclude>(
          InsertRange, std::move(ReplacedMacroName));
      if (MacroName == "__CUDA_ARCH__") {
        ReplMap.insert(Repl->getReplacement(DpctGlobalInfo::getContext()));
        requestFeature(HelperFeatureEnum::Dpct_dpct_compatibility_temp, Begin);
      } else if (MacroName != "__CUDACC__" ||
                 DpctGlobalInfo::getMacroDefines().count(MacroName)) {
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
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(Loc).str();
  bool IsInRoot = !isDirectory(InFile) && isChildOrSamePath(InRoot, InFile);

  if (!IsInRoot) {
    return;
  }
  ReplaceCuMacro(ConditionRange, IfType::IT_If, Loc, Loc);
}
void IncludesCallbacks::Elif(SourceLocation Loc, SourceRange ConditionRange,
                             ConditionValueKind ConditionValue,
                             SourceLocation IfLoc) {
  std::string InRoot = ATM.InRoot;
  std::string InFile = SM.getFilename(Loc).str();
  bool IsInRoot = !isDirectory(InFile) && isChildOrSamePath(InRoot, InFile);

  if (!IsInRoot) {
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

// A class that uses the RAII idiom to selectively update the locations of the
// last inclusion directives.
class LastInclusionLocationUpdater {
public:
  LastInclusionLocationUpdater(SourceLocation Loc, bool UpdateNeeded = true)
      : Loc(Loc), UpdateNeeded(UpdateNeeded) {}
  ~LastInclusionLocationUpdater() {
    if (UpdateNeeded)
      DpctGlobalInfo::getInstance().setLastIncludeLocation(Loc);
  }
  void update(bool UpdateNeeded) { this->UpdateNeeded = UpdateNeeded; }

private:
  SourceLocation Loc;
  bool UpdateNeeded;
};

void IncludesCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  // Record the locations of the first and last inclusion directives in a file
  DpctGlobalInfo::getInstance().setFirstIncludeLocation(HashLoc);
  LastInclusionLocationUpdater Updater{FilenameRange.getEnd()};

  std::string IncludePath = SearchPath.str();
  makeCanonical(IncludePath);

  std::string IncludingFile = DpctGlobalInfo::getLocInfo(HashLoc).first;

  // eg. '/home/path/util.h' -> '/home/path'
  StringRef Directory = llvm::sys::path::parent_path(IncludingFile);
  std::string InRoot = ATM.InRoot;

  bool IsIncludingFileInInRoot =
      !isDirectory(IncludingFile) && isChildOrSamePath(InRoot, Directory.str());

  // If the header file included can not be found, just return.
  if (!File) {
    return;
  }

  std::string FilePath;
  if (!File->tryGetRealPathName().empty()) {
    FilePath = File->tryGetRealPathName().str();
  } else {
    llvm::SmallString<512> FilePathAbs(File->getName());
    DpctGlobalInfo::getSourceManager().getFileManager().makeAbsolutePath(
        FilePathAbs);
    llvm::sys::path::native(FilePathAbs);
    llvm::sys::path::remove_dots(FilePathAbs, true);
    FilePath = std::string(FilePathAbs.str());
  }

  std::string DirPath = llvm::sys::path::parent_path(FilePath).str();
  bool IsFileInInRoot = !isChildPath(DpctInstallPath, DirPath) &&
                        (isChildOrSamePath(InRoot, DirPath));
  bool IsExcluded = DpctGlobalInfo::isExcluded(FilePath);

  bool NeedMigrate = !IsExcluded && IsFileInInRoot;

  if (IsFileInInRoot) {
    auto FilePathWithoutSymlinks =
        DpctGlobalInfo::removeSymlinks(SM.getFileManager(), FilePath);
    IncludeFileMap[FilePathWithoutSymlinks] = false;
    dpct::DpctGlobalInfo::getIncludingFileSet().insert(FilePathWithoutSymlinks);
  }

  if ((!SM.isWrittenInMainFile(HashLoc) && !IsIncludingFileInInRoot) ||
      IsExcluded) {
    return;
  }

  // The "FilePath" is included by the "IncludingFile".
  // If "FilePath" is not under the Inroot folder, do not record the including
  // relationship information.
  if (DpctGlobalInfo::isInRoot(FilePath, false))
    DpctGlobalInfo::getInstance().recordIncludingRelationship(IncludingFile,
                                                              FilePath);

  // Record that math header is included in this file
  if (IsAngled && (FileName.compare(StringRef("math.h")) == 0 ||
                   FileName.compare(StringRef("cmath")) == 0)) {
    DpctGlobalInfo::getInstance().setMathHeaderInserted(HashLoc, true);
  }

  // Record that time header is included in this file
  if (IsAngled && (FileName.compare(StringRef("time.h")) == 0)) {
    DpctGlobalInfo::getInstance().setTimeHeaderInserted(HashLoc, true);
  }

  // Record that algorithm header is included in this file
  if (IsAngled && FileName.compare(StringRef("algorithm")) == 0) {
    DpctGlobalInfo::getInstance().setAlgorithmHeaderInserted(HashLoc, true);
  }

  // Replace with
  // <mkl_blas_sycl.hpp>, <mkl_lapack_sycl.hpp> and <mkl_sycl_types.hpp>
  if ((FileName.compare(StringRef("cublas_v2.h")) == 0) ||
      (FileName.compare(StringRef("cublas.h")) == 0) ||
      (FileName.compare(StringRef("cusolverDn.h")) == 0)) {
    if (DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
            HelperFilesCustomizationLevel::HFCL_None ||
        DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
            HelperFilesCustomizationLevel::HFCL_All) {
      DpctGlobalInfo::getInstance().insertHeader(HashLoc, HT_MKL_BLAS_Solver);
    } else {
      DpctGlobalInfo::getInstance().insertHeader(
          HashLoc, HT_MKL_BLAS_Solver_Without_Util);
    }

    DpctGlobalInfo::setMKLHeaderUsed(true);

    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        ""));
    Updater.update(false);
  }

  // Replace with <mkl_rng_sycl.hpp> and <mkl_rng_sycl_device.hpp>
  if ((FileName.compare(StringRef("curand.h")) == 0) ||
      (FileName.compare(StringRef("curand_kernel.h")) == 0)) {
    DpctGlobalInfo::getInstance().insertHeader(HashLoc, HT_MKL_RNG);
    DpctGlobalInfo::setMKLHeaderUsed(true);
    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        ""));
    Updater.update(false);
  }

  // Replace with <mkl_spblas_sycl.hpp>
  if ((FileName.compare(StringRef("cusparse.h")) == 0) ||
      (FileName.compare(StringRef("cusparse_v2.h")) == 0)) {
    if (DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
            HelperFilesCustomizationLevel::HFCL_None ||
        DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
            HelperFilesCustomizationLevel::HFCL_All) {
      DpctGlobalInfo::getInstance().insertHeader(HashLoc, HT_MKL_SPBLAS);
    } else {
      DpctGlobalInfo::getInstance().insertHeader(HashLoc,
                                                 HT_MKL_SPBLAS_Without_Util);
    }

    DpctGlobalInfo::setMKLHeaderUsed(true);

    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        ""));
    Updater.update(false);
  }

  if (FileName.compare(StringRef("cufft.h")) == 0) {

    DpctGlobalInfo::setMKLHeaderUsed(true);

    DpctGlobalInfo::getInstance().insertHeader(HashLoc, HT_MKL_FFT);
    TransformSet.emplace_back(new ReplaceInclude(
        CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                        /*IsTokenRange=*/false),
        ""));
    Updater.update(false);
  }

  if (FileName.startswith(StringRef("nccl"))) {
    if (isChildOrSamePath(InRoot, DirPath)) {
      return;
    }
    DiagnosticsUtils::report(
        HashLoc, Diagnostics::MANUAL_MIGRATION_LIBRARY,
        dpct::DpctGlobalInfo::getCompilerInstance(), &TransformSet, false,
        "Intel(R) oneAPI Collective Communications Library");
    Updater.update(false);
  }
  if (FileName.startswith(StringRef("cudnn"))) {
    if (isChildOrSamePath(InRoot, DirPath)) {
      return;
    }
    DiagnosticsUtils::report(
        HashLoc, Diagnostics::MANUAL_MIGRATION_LIBRARY,
        dpct::DpctGlobalInfo::getCompilerInstance(), &TransformSet, false,
        "Intel(R) oneAPI Deep Neural Network Library (oneDNN)");
    Updater.update(false);
  }

  if (!isChildPath(CudaPath, IncludePath) &&
      IncludePath.compare(0, 15, "/usr/local/cuda", 15)) {

    // Replace "#include "*.cuh"" with "include "*.dp.hpp""
    if (NeedMigrate && FileName.endswith(".cuh")) {
      CharSourceRange InsertRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                                  /* IsTokenRange */ false);
      std::string NewFileName = "#include \"" +
                                FileName.drop_back(strlen(".cuh")).str() +
                                ".dp.hpp\"";
      TransformSet.emplace_back(
          new ReplaceInclude(InsertRange, std::move(NewFileName)));
      return;
    }

    // Replace "#include "*.cu"" with "include "*.dp.cpp""
    if (FileName.endswith(".cu")) {
      CharSourceRange InsertRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                                  /* IsTokenRange */ false);
      std::string NewFileName =
          "#include \"" + FileName.drop_back(strlen(".cu")).str() + ".dp.cpp\"";
      TransformSet.emplace_back(
          new ReplaceInclude(InsertRange, std::move(NewFileName)));
      return;
    }

    // To generate replacement of replacing "#include "*.c"" with "include
    // "*.c.dp.cpp"".
    if (NeedMigrate && FileName.endswith(".c")) {
      CharSourceRange InsertRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                                  /* IsTokenRange */ false);
      std::string NewFileName = "#include \"" + FileName.str() + ".dp.cpp\"";

      // For file path in preprocessing stage may be different with the one in
      // syntax analysis stage, here only file name is used as the key.
      const std::string Name = llvm::sys::path::filename(FileName).str();
      IncludeMapSet[Name].push_back(std::make_unique<ReplaceInclude>(
          InsertRange, std::move(NewFileName)));
    }
  }

  // Extra process thrust headers, map to PSTL mapping headers in runtime.
  // For multi thrust header files, only insert once for PSTL mapping header.
  if (IsAngled && (FileName.find("thrust/") != std::string::npos)) {
    if (!DplHeaderInserted) {
      std::string Replacement =
          std::string("<" + getCustomMainHelperFileName() + "/dpl_utils.hpp>");
      // CTST-2021:
      // The #include of oneapi/dpl/execution and oneapi/dpl/algorithm were
      // previously added here.  However, due to some unfortunate include
      // dependencies introduced with the PSTL/TBB headers from the
      // gcc-9.3.0 include files, those two headers must now be included
      // before the CL/sycl.hpp are included, so the FileInfo is set
      // to hold a boolean that'll indicate whether to insert them when
      // the #include CL/sycl.cpp is added later
      DplHeaderInserted = true;
      auto BeginLocInfo = DpctGlobalInfo::getLocInfo(FilenameRange.getBegin());
      auto FileInfo =
          DpctGlobalInfo::getInstance().insertFile(BeginLocInfo.first);
      FileInfo->setAddOneDplHeaders(true);
      TransformSet.emplace_back(
          new ReplaceInclude(FilenameRange, std::move(Replacement)));
      requestFeature(HelperFeatureEnum::DplUtils_non_local_include_dependency,
                     "");
    } else {
      // Replace the complete include directive with an empty string.
      TransformSet.emplace_back(new ReplaceInclude(
          CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                          /*IsTokenRange=*/false),
          ""));
      Updater.update(false);
    }
    return;
  }

  //  TODO: implement one of this for each source language.
  // If it's not an include from the SDK, leave it,
  // unless it's runtime header, in which case it will be replaced.
  // In other words, runtime header will be replaced regardless of where it's
  // coming from.
  if (!isChildOrSamePath(CudaPath, IncludePath) &&
      IncludePath.compare(0, 15, "/usr/local/cuda", 15)) {
    if (!(IsAngled && FileName.compare(StringRef("cuda_runtime.h")) == 0)) {
      return;
    }
  }

  // If CudaPath is in /usr/include,
  // for all the include files without following pattern, keep it
  if (!CudaPath.compare(0, 12, "/usr/include", 12)) {
    if (!FileName.startswith("cuda") && !FileName.startswith("cusolver") &&
        !FileName.startswith("cublas") && !FileName.startswith("cusparse") &&
        !FileName.startswith("curand")) {
      return;
    }
  }

  // Replace the complete include directive with an empty string.
  // Also remove the trailing spaces to end of the line.
  TransformSet.emplace_back(new ReplaceInclude(
      CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                      /*IsTokenRange=*/false),
      "", true));
  Updater.update(false);
}

void IncludesCallbacks::FileChanged(SourceLocation Loc, FileChangeReason Reason,
                                    SrcMgr::CharacteristicKind FileType,
                                    FileID PrevFID) {
  // Record the location when a file is entered
  if (Reason == clang::PPCallbacks::EnterFile) {
    DpctGlobalInfo::getInstance().setFileEnterLocation(Loc);

    std::string InRoot = ATM.InRoot;
    std::string InFile = SM.getFilename(Loc).str();
    bool IsInRoot = !isDirectory(InFile) && isChildOrSamePath(InRoot, InFile);

    if (!IsInRoot) {
      return;
    }

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
  for (const TextModification *TM : EmittedTransformations) {
    OS << Indent;
    TM->print(OS, getCompilerInstance().getASTContext(),
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
  for (const TextModification *TM : EmittedTransformations) {
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

void MigrationRule::emplaceTransformation(const char *RuleID,
                                          TextModification *TM) {
  ASTTraversalMetaInfo::getEmittedTransformations()[RuleID].emplace_back(TM);
  TransformSet->emplace_back(TM);
}

void IterationSpaceBuiltinRule::registerMatcher(MatchFinder &MF) {
  // TODO: check that threadIdx is not a local variable.
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
    NewName = DpctGlobalInfo::getItem(DRE) + ".get_local_range().get(";
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
    // available,  so dpct migrates it by 'threadIdx' matcher to identify the
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
    if (!Tok2Ptr.hasValue())
      return;

    const auto Tok2 = Tok2Ptr.getValue();
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

      if (DpctGlobalInfo::getAssumedNDRangeDim() == 1) {
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
      } else {
        if (FieldName == "x") {
          Dimension = 2;
        } else if (FieldName == "y")
          Dimension = 1;
        else if (FieldName == "z")
          Dimension = 0;
        else
          return;
      }

      Replacement += std::to_string(Dimension);
      Replacement += ")";

      emplaceTransformation(
          new ReplaceText(Begin, TyLen, std::move(Replacement)));
    }
    return;
  }

  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "memberExpr");
  const VarDecl *VD = getAssistNodeAsType<VarDecl>(Result, "varDecl", false);
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
    if (DpctGlobalInfo::getAssumedNDRangeDim() == 1) {
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
    } else {
      if (FieldName == "__fetch_builtin_x")
        Dimension = 2;
      else if (FieldName == "__fetch_builtin_y")
        Dimension = 1;
      else if (FieldName == "__fetch_builtin_z")
        Dimension = 0;
      else {
        llvm::dbgs() << "[" << getName()
                     << "] Unexpected field name: " << FieldName;
        return;
      }
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

REGISTER_RULE(IterationSpaceBuiltinRule)

void ErrorHandlingIfStmtRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      // Match if-statement that has no else and has a condition of either an
      // operator!= or a variable of type enum.
      ifStmt(unless(hasElse(anything())),
             hasCondition(
                 anyOf(binaryOperator(hasOperatorName("!=")).bind("op!="),
                       ignoringImpCasts(
                           declRefExpr(hasType(hasCanonicalType(enumType())))
                               .bind("var")))))
          .bind("errIf"),
      this);
  MF.addMatcher(
      // Match if-statement that has no else and has a condition of
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
#if 0
    //TODO: enable argument check
    for (const auto *S : CE->arguments()) {
      if (!isErrorHandlingSafeToRemove(S->IgnoreImplicit()))
        return false;
    }
#endif
      return true;
    }
#if 0
  //TODO: enable argument check
  else if (isa <DeclRefExpr>(S))
    return true;
  else if (isa<IntegerLiteral>(S))
    return true;
  else if (isa<StringLiteral>(S))
    return true;
#endif
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

REGISTER_RULE(ErrorHandlingIfStmtRule)

void ErrorHandlingHostAPIRule::registerMatcher(MatchFinder &MF) {
  auto isMigratedHostAPI = [&]() {
    return allOf(
        anyOf(returns(asString("cudaError_t")),
              returns(asString("cublasStatus_t")),
              returns(asString("nvgraphStatus_t")),
              returns(asString("cusparseStatus_t")),
              returns(asString("cusolverStatus_t")),
              returns(asString("cufftResult_t")),
              returns(asString("curandStatus_t"))),
        // cudaGetLastError returns cudaError_t but won't fail in the call
        unless(hasName("cudaGetLastError")),
        anyOf(unless(hasAttr(attr::CUDADevice)), hasAttr(attr::CUDAHost)));
  };

  // Match host api call in the condition session of flow control
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

  // Match host api call whose return value used inside flow control or return
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

  // Match host api call whose return value captured and used
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
  // if host api call in the condition session of flow control
  // or host api call whose return value used inside flow control or return
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

  // Check if the return value is saved in an variable,
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
  if (auto CMD = dyn_cast<CXXMethodDecl>(FD)) {
    if (CMD->getParent()->isLambda()) {
      IsLambda = true;
    }
  }

  std::string IndentStr = getIndent(FD->getBeginLoc(), SM).str();
  std::string InnerIndentStr = IndentStr + "  ";

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
      getNL() + IndentStr +
      std::string("catch (" + MapNames::getClNamespace(true) +
                  "exception const &exc) {") +
      getNL() + InnerIndentStr +
      std::string("std::cerr << exc.what() << \"Exception caught at file:\" << "
                  "__FILE__ << "
                  "\", line:\" << __LINE__ << std::endl;") +
      getNL() + InnerIndentStr + std::string("std::exit(1);") + getNL() +
      IndentStr + "}";
  if (IsLambda) {
    ReplaceStr += getNL() + IndentStr + "}";
  }
  emplaceTransformation(
      new InsertAfterStmt(FD->getBody(), std::move(ReplaceStr)));
}

REGISTER_RULE(ErrorHandlingHostAPIRule)

void AlignAttrsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(cxxRecordDecl(hasAttr(attr::Aligned)).bind("classDecl"), this);
}

void AlignAttrsRule::runRule(const MatchFinder::MatchResult &Result) {
  auto C = getNodeAsType<CXXRecordDecl>(Result, "classDecl");
  if (!C)
    return;
  auto &AV = C->getAttrs();

  for (auto A : AV) {
    if (A->getKind() == attr::Aligned) {
      auto SM = Result.SourceManager;
      auto ExpB = SM->getExpansionLoc(A->getLocation());
      if (!strncmp(SM->getCharacterData(ExpB), "__align__(", 10))
        emplaceTransformation(new ReplaceToken(ExpB, "__dpct_align__"));
      requestFeature(HelperFeatureEnum::Dpct_dpct_align_and_inline, ExpB);
    }
  }
}

REGISTER_RULE(AlignAttrsRule)

void FuncAttrsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(functionDecl(hasAttr(attr::AlwaysInline)).bind("funcDecl"),
                this);
}

void FuncAttrsRule::runRule(const MatchFinder::MatchResult &Result) {
  auto FD = getNodeAsType<FunctionDecl>(Result, "funcDecl");
  auto SM = Result.SourceManager;
  if (!FD)
    return;
  auto &FA = FD->getAttrs();
  for (auto A : FA) {
    if (A->getKind() == attr::AlwaysInline) {
      // directly used
      auto Loc =
          getDefinitionRange(A->getRange().getBegin(), A->getRange().getEnd())
              .getBegin();
      if (!strncmp(SM->getCharacterData(Loc), "__forceinline__", 15))
        emplaceTransformation(new ReplaceToken(Loc, "__dpct_inline__"));
      requestFeature(HelperFeatureEnum::Dpct_dpct_align_and_inline, Loc);
    }
  }
}

REGISTER_RULE(FuncAttrsRule)

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
    // TODO: investigate usage of __half and __half2 types and support it
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
    std::string InFile = dpct::DpctGlobalInfo::getSourceManager()
                             .getFilename(CalleeDecl->getLocation())
                             .str();
    bool InInstallPath = isChildOrSamePath(DpctInstallPath, InFile);
    bool InCudaPath = DpctGlobalInfo::isInCudaPath(CalleeDecl->getLocation());
    if (!(InInstallPath || InCudaPath))
      return;
  } else {
    return;
  };

  // TODO: 1. Investigate are there usages of atomic functions on local address
  //          space
  //       2. If item 1. shows atomic functions on local address space is
  //          significant, detect whether this atomic operation operates in
  //          global space or local space (currently, all in global space,
  //          see dpct_atomic.hpp for more details)
  const std::string AtomicFuncName = CE->getDirectCallee()->getName().str();
  if (MapNames::AtomicFuncNamesMap.find(AtomicFuncName) ==
      MapNames::AtomicFuncNamesMap.end())
    return;
  std::string ReplacedAtomicFuncName =
      MapNames::AtomicFuncNamesMap.at(AtomicFuncName);

  static const std::unordered_map<std::string, HelperFeatureEnum>
      FunctionNameToFeatureMap = {
          {"atomicAdd", HelperFeatureEnum::Atomic_atomic_fetch_add},
          {"atomicSub", HelperFeatureEnum::Atomic_atomic_fetch_sub},
          {"atomicAnd", HelperFeatureEnum::Atomic_atomic_fetch_and},
          {"atomicOr", HelperFeatureEnum::Atomic_atomic_fetch_or},
          {"atomicXor", HelperFeatureEnum::Atomic_atomic_fetch_xor},
          {"atomicMin", HelperFeatureEnum::Atomic_atomic_fetch_min},
          {"atomicMax", HelperFeatureEnum::Atomic_atomic_fetch_max},
          {"atomicExch", HelperFeatureEnum::Atomic_atomic_exchange},
          {"atomicCAS",
           HelperFeatureEnum::Atomic_atomic_compare_exchange_strong},
          {"atomicInc", HelperFeatureEnum::Atomic_atomic_fetch_compare_inc},
      };
  requestFeature(FunctionNameToFeatureMap.at(AtomicFuncName), CE);

  // Explicitly cast all arguments except first argument
  const Type *Arg0Type = CE->getArg(0)->getType().getTypePtrOrNull();
  // Atomic operation's first argument is always pointer type
  if (!Arg0Type || !Arg0Type->isPointerType()) {
    return;
  }
  const QualType PointeeType = Arg0Type->getPointeeType();

  std::string TypeName;
  bool IsTemplateType = false;
  if (auto *SubstedType = dyn_cast<SubstTemplateTypeParmType>(PointeeType)) {
    IsTemplateType = true;
    // Type is substituted in template initialization, use the template
    // parameter name
    if (!SubstedType->getReplacedParameter()->getIdentifier()) {
      return;
    }
    TypeName =
        SubstedType->getReplacedParameter()->getIdentifier()->getName().str();
  } else {
    TypeName = PointeeType.getAsString();
  }

  // add exceptions for atomic tranlastion:
  // eg. source code: atomicMin(double), don't migrate it, its user code.
  //     also: atomic_fetch_min<double> is not available in compute++.
  if ((TypeName == "double" && AtomicFuncName != "atomicAdd") ||
      (TypeName == "float" &&
       !(AtomicFuncName == "atomicAdd" || AtomicFuncName == "atomicExch"))) {

    return;
  }

  bool HasSharedAttr = false;
  bool NeedReport = false;
  getShareAttrRecursive(CE->getArg(0), HasSharedAttr, NeedReport);

  // Inline the code for ingeter types
  static std::unordered_map<std::string, std::string> AtomicMap = {
      {"atomicAdd", "fetch_add"}, {"atomicSub", "fetch_sub"},
      {"atomicAnd", "fetch_and"}, {"atomicOr", "fetch_or"},
      {"atomicXor", "fetch_xor"}, {"atomicMin", "fetch_min"},
      {"atomicMax", "fetch_max"},
  };

  auto IsMacro = CE->getBeginLoc().isMacroID();
  auto Iter = AtomicMap.find(AtomicFuncName);
  if (!IsMacro && !IsTemplateType && PointeeType->isIntegerType() &&
      Iter != AtomicMap.end()) {
    if (NeedReport)
      report(CE->getArg(0)->getBeginLoc(),
             Diagnostics::SHARE_MEMORY_ATTR_DEDUCE, false,
             getStmtSpelling(CE->getArg(0)),
             MapNames::getClNamespace() + "global_ptr",
             MapNames::getClNamespace() + "local_ptr");

    std::string ReplStr{MapNames::getClNamespace(true)};
    ReplStr += "atomic<";
    ReplStr += TypeName;
    if (HasSharedAttr) {
      ReplStr += ", ";
      ReplStr += MapNames::getClNamespace();
      ReplStr += "access::address_space::local_space";
    }
    ReplStr += ">(";
    ReplStr += MapNames::getClNamespace();
    if (HasSharedAttr)
      ReplStr += "local_ptr<";
    else
      ReplStr += "global_ptr<";
    ReplStr += TypeName;
    ReplStr += ">(";
    // Take care of __shared__ variables because their types are
    // changed to pointers
    bool Arg0NeedDeref = false;
    const Expr *Arg0RemoveCStyleCast = CE->getArg(0);
    if (const CStyleCastExpr *Arg0CSCE =
            dyn_cast<CStyleCastExpr>(CE->getArg(0))) {
      ReplStr += "(";
      ReplStr +=
          DpctGlobalInfo::getReplacedTypeName(Arg0CSCE->getTypeAsWritten());
      ReplStr += ")";
      Arg0RemoveCStyleCast = Arg0CSCE->getSubExpr();
    }

    auto *UO = dyn_cast<UnaryOperator>(Arg0RemoveCStyleCast);
    if (UO && UO->getOpcode() == clang::UO_AddrOf) {
      if (auto DRE = dyn_cast<DeclRefExpr>(UO->getSubExpr()->IgnoreImpCasts()))
        Arg0NeedDeref = IsTypeChangedToPointer(DRE);
    }
    // Deref the expression if it is the unary operator of a shared simple
    // variable
    if (Arg0NeedDeref) {
      std::ostringstream OS;
      printDerefOp(OS, Arg0RemoveCStyleCast);
      ReplStr += OS.str();
    } else {
      ArgumentAnalysis A(CE->getArg(0), false);
      A.analyze();
      ReplStr += A.getReplacedString();
    }
    ReplStr += ")).";
    ReplStr += Iter->second;
    ReplStr += "(";

    auto Arg1NeedDeref = false;
    if (auto DRE = dyn_cast<DeclRefExpr>(CE->getArg(1)->IgnoreImpCasts()))
      Arg1NeedDeref = IsTypeChangedToPointer(DRE);
    if (Arg1NeedDeref) {
      std::ostringstream OS;
      printDerefOp(OS, CE->getArg(1));
      ReplStr += OS.str();
    } else {
      ArgumentAnalysis A(CE->getArg(1), false);
      A.analyze();
      ReplStr += A.getReplacedString();
    }
    ReplStr += ")";

    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    return;
  }

  std::string SpaceName =
      MapNames::getClNamespace() + "access::address_space::local_space";
  std::string ReplAtomicFuncNameWithSpace =
      ReplacedAtomicFuncName + "<" + TypeName + ", " + SpaceName + ">";
  if (NeedReport)
    report(CE->getArg(0)->getBeginLoc(), Diagnostics::SHARE_MEMORY_ATTR_DEDUCE,
           false, getStmtSpelling(CE->getArg(0)), ReplacedAtomicFuncName,
           ReplAtomicFuncNameWithSpace);

  if (HasSharedAttr) {
    ReplacedAtomicFuncName = ReplAtomicFuncNameWithSpace;
  }

  emplaceTransformation(
      new ReplaceCalleeName(CE, std::move(ReplacedAtomicFuncName)));

  const unsigned NumArgs = CE->getNumArgs();
  for (unsigned i = 0; i < NumArgs; ++i) {
    const Expr *Arg = CE->getArg(i);
    if (auto *ImpCast = dyn_cast<ImplicitCastExpr>(Arg)) {
      if (ImpCast->getCastKind() != clang::CK_LValueToRValue) {
        if (i == 0) {
          if (dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts())) {
            emplaceTransformation(
                new InsertBeforeStmt(Arg, "(" + TypeName + "*)"));
          } else {
            insertAroundStmt(Arg, "(" + TypeName + "*)(", ")");
          }
        } else {
          if (dyn_cast<IntegerLiteral>(Arg->IgnoreImpCasts()) ||
              dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts())) {
            emplaceTransformation(
                new InsertBeforeStmt(Arg, "(" + TypeName + ")"));
          } else {
            insertAroundStmt(Arg, "(" + TypeName + ")(", ")");
          }
        }
      }
    }
  }
}

void AtomicFunctionRule::runRule(const MatchFinder::MatchResult &Result) {
  ReportUnsupportedAtomicFunc(
      getNodeAsType<CallExpr>(Result, "unsupportedAtomicFuncCall"));

  MigrateAtomicFunc(getNodeAsType<CallExpr>(Result, "supportedAtomicFuncCall"),
                    Result);
}

REGISTER_RULE(AtomicFunctionRule)

void ThrustFunctionRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> ThrustFuncNames(MapNames::ThrustFuncNamesMap.size());
  std::transform(
      MapNames::ThrustFuncNamesMap.begin(), MapNames::ThrustFuncNamesMap.end(),
      ThrustFuncNames.begin(),
      [](const std::pair<std::string, MapNames::ThrustFuncReplInfo> &p) {
        return p.first;
      });

  MF.addMatcher(callExpr(callee(functionDecl(
                             hasDeclContext(namespaceDecl(hasName("thrust"))))))
                    .bind("thrustFuncCall"),
                this);

  MF.addMatcher(
      unresolvedLookupExpr(hasAnyDeclaration(namedDecl(hasDeclContext(
                               namespaceDecl(hasName("thrust"))))),
                           hasParent(callExpr().bind("thrustApiCallExpr")))
          .bind("unresolvedThrustAPILookupExpr"),
      this);
}

TextModification *removeArg(const CallExpr *C, unsigned n,
                            const SourceManager &SM);

void ThrustFunctionRule::thrustFuncMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *CE,
    const UnresolvedLookupExpr *ULExpr) {

  auto &SM = DpctGlobalInfo::getSourceManager();

  // handle the a regular call expr
  std::string ThrustFuncName;
  if (ULExpr) {
    std::string Namespace;
    if (auto NNS = ULExpr->getQualifier()) {
      if (auto NS = NNS->getAsNamespace()) {
        Namespace = NS->getNameAsString();
      }
    }
    if (!Namespace.empty() && Namespace == "thrust")
      ThrustFuncName = ULExpr->getName().getAsString();
  } else {
    ThrustFuncName = CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  const unsigned NumArgs = CE->getNumArgs();
  auto QT = CE->getArg(0)->getType();
  LangOptions LO;
  std::string ArgT = QT.getAsString(PrintingPolicy(LO));

  auto ReplInfo = MapNames::ThrustFuncNamesMap.find(ThrustFuncName);

  // For the API migration defined in APINamesThrust.inc
  if (ReplInfo == MapNames::ThrustFuncNamesMap.end()) {
    dpct::ExprAnalysis EA;
    EA.analyze(CE);
    auto Range = getDefinitionRange(CE->getBeginLoc(), CE->getEndLoc());
    auto Len = Lexer::MeasureTokenLength(
        Range.getEnd(), SM, DpctGlobalInfo::getContext().getLangOpts());
    Len += SM.getDecomposedLoc(Range.getEnd()).second -
           SM.getDecomposedLoc(Range.getBegin()).second;
    auto ReplStr = EA.getReplacedString();
    emplaceTransformation(
        new ReplaceText(Range.getBegin(), Len, std::move(ReplStr)));
    return;
  }

  // For the API migration defined in APINamesMapThrust.inc
  auto HelperFeatureIter =
      MapNames::ThrustFuncNamesHelperFeaturesMap.find(ThrustFuncName);
  if (HelperFeatureIter != MapNames::ThrustFuncNamesHelperFeaturesMap.end()) {
    requestFeature(HelperFeatureIter->second, CE);
  }

  auto NewName = ReplInfo->second.ReplName;

  bool hasExecutionPolicy =
      ArgT.find("execution_policy_base") != std::string::npos;
  bool PolicyProcessed = false;

  if (ThrustFuncName == "transform" || ThrustFuncName == "copy_if") {
    if (NumArgs == 6) {
      hasExecutionPolicy = true;
    } else if (NumArgs == 5) {
      std::string FirstArgType = CE->getArg(0)->getType().getAsString();
      std::string SecondArgType = CE->getArg(1)->getType().getAsString();
      if (FirstArgType != SecondArgType)
        hasExecutionPolicy = true;
    }
  }

  if (ThrustFuncName == "sort") {
    auto ExprLoc = SM.getSpellingLoc(CE->getBeginLoc());
    if (SortULExpr.count(ExprLoc) != 0)
      return;
    else if (ULExpr) {
      SortULExpr.insert(ExprLoc);
    }
    if (NumArgs == 4) {
      hasExecutionPolicy = true;
    } else if (NumArgs == 3) {
      std::string FirstArgType = CE->getArg(0)->getType().getAsString();
      std::string SecondArgType = CE->getArg(1)->getType().getAsString();
      if (FirstArgType != SecondArgType)
        hasExecutionPolicy = true;
    }
  }
  // To migrate "thrust::cuda::par.on" that appears in CE' first arg to
  // "oneapi::dpl::execution::make_device_policy".
  const CallExpr *Call = nullptr;
  if (hasExecutionPolicy) {
    if (const auto *ICE = dyn_cast<ImplicitCastExpr>(CE->getArg(0))) {
      if (const auto *MT =
              dyn_cast<MaterializeTemporaryExpr>(ICE->getSubExpr())) {
        if (auto SubICE = dyn_cast<ImplicitCastExpr>(MT->getSubExpr())) {
          Call = dyn_cast<CXXMemberCallExpr>(SubICE->getSubExpr());
        }
      }
    } else if (const auto *SubCE = dyn_cast<CallExpr>(CE->getArg(0))) {
      Call = SubCE;
    } else {
      Call = dyn_cast<CXXMemberCallExpr>(CE->getArg(0));
    }
  }

  if (Call) {
    auto StreamArg = Call->getArg(0);
    std::ostringstream OS;
    if (const auto *ME = dyn_cast<MemberExpr>(Call->getCallee())) {
      auto BaseName =
          DpctGlobalInfo::getUnqualifiedTypeName(ME->getBase()->getType());
      if (BaseName == "thrust::cuda_cub::par_t") {
        OS << "oneapi::dpl::execution::make_device_policy(";
        printDerefOp(OS, StreamArg);
        OS << ")";
        emplaceTransformation(new ReplaceStmt(Call, OS.str()));
        PolicyProcessed = true;
      }
    }
  }

  // All the thrust APIs (such as thrust::copy_if, thrust::copy, thrust::fill,
  // thrust::count, thrust::equal) called in device function , should be
  // migrated to oneapi::dpl APIs without a policy on the DPC++ side
  if (auto FD = DpctGlobalInfo::getParentFunction(CE)) {
    if (FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) {
      if (ThrustFuncName == "sort") {
        report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
        return;
      } else if (hasExecutionPolicy) {
        emplaceTransformation(removeArg(CE, 0, *Result.SourceManager));
      }
    }
  }

  if (ThrustFuncName == "copy_if" &&
      (!hasExecutionPolicy && NumArgs == 5 || NumArgs > 5)) {
    NewName = MapNames::getDpctNamespace() + ThrustFuncName;
    requestFeature(HelperFeatureEnum::DplExtrasAlgorithm_copy_if, CE);

    if (ULExpr)
      emplaceTransformation(new ReplaceToken(
          ULExpr->getBeginLoc(), ULExpr->getEndLoc(), std::move(NewName)));
    else
      emplaceTransformation(new ReplaceCalleeName(CE, std::move(NewName)));
  } else if (ThrustFuncName == "make_zip_iterator") {
    // oneapi::dpl::make_zip_iterator expects the component iterators to be
    // passed directly instead of being wrapped in a tuple as
    // thrust::make_zip_iterator requires.
    std::string NewArg;
    if (auto CCE = dyn_cast<CXXConstructExpr>(CE->getArg(0)))
      if (const CallExpr *SubCE =
              dyn_cast<CallExpr>(CCE->getArg(0)->IgnoreImplicit())) {
        std::string Arg0 = getStmtSpelling(SubCE->getArg(0));
        std::string Arg1 = getStmtSpelling(SubCE->getArg(1));
        NewArg = Arg0 + ", " + Arg1;
      }

    if (NewArg.empty()) {
      std::string Arg0 = "std::get<0>(" + getStmtSpelling(CE->getArg(0)) + ")";
      std::string Arg1 = "std::get<1>(" + getStmtSpelling(CE->getArg(0)) + ")";
      NewArg = Arg0 + ", " + Arg1;
    }

    emplaceTransformation(removeArg(CE, 0, *Result.SourceManager));
    emplaceTransformation(
        new InsertAfterStmt(CE->getArg(0), std::move(NewArg)));

  } else if (ThrustFuncName == "binary_search" &&
             (NumArgs <= 4 || (NumArgs == 5 && hasExecutionPolicy))) {
    // Currently, we do not support migration of 4 of the 8 overloaded versions
    // of thrust::binary_search.  The ones we do not support are the ones
    // searching for a single value instead of a vector of values
    //
    // Supported parameter profiles:
    // 1. (policy, firstIt, lastIt, valueFirstIt, valueLastIt, resultIt)
    // 2. (firstIt, lastIt, valueFirstIt, valueLastIt, resultIt)
    // 3. (policy, firstIt, lastIt, valueFirstIt, valueLastIt, resultIt, comp)
    // 4. (firstIt, lastIt, valueFirstIt, valueLastIt, resultIt, comp)
    //
    // Not supported parameter profiles:
    // 1. (policy, firstIt, lastIt, value)
    // 2. (firstIt, lastIt, value)
    // 3. (policy, firstIt, lastIt, value, comp)
    // 4. (firstIt, lastIt, value, comp)
    //
    // The logic in the above if condition filters out the ones
    // currently not supported and issues a warning
    report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false);
    return;

  } else if (ThrustFuncName == "sort") {
    // Rule of thrust::sort migration
    //#. thrust api
    //   dpcpp api
    // 1. thurst::sort(policy, h_vec.begin(), h_vec.end())
    //   std::sort(oneapi::dpl::exection::par_unseq, h_vec.begin(), h_vec.end())
    //
    // 2. thrust::sort(h_vec.begin(), h_vec.end())
    //   std::sort(h_vec.begin(), h_vec.end())
    //
    // 3. thrust::sort(policy, d_vec.begin(), d_vec.end())
    //   oneapi::dpl::sort(make_device_policy(queue), d_vec.begin(),
    //   d_vec.end())
    //
    // 4. thrust::sort(d_vec.begin(), d_vec.end())
    //   oneapi::dpl::sort(make_device_policy(queue), d_vec.begin(),
    //   d_vec.end())
    //
    // When thrust::sort inside template function and is a UnresolvedLookupExpr,
    // we will map to oneapi::dpl::sort

    auto IteratorArg = CE->getArg(1);
    auto IteratorType = IteratorArg->getType().getAsString();
    if (ULExpr) {
      if (PolicyProcessed) {
        emplaceTransformation(new ReplaceToken(
            ULExpr->getBeginLoc(), ULExpr->getEndLoc(), std::move(NewName)));
        return;
      } else if (hasExecutionPolicy) {
        emplaceTransformation(removeArg(CE, 0, *Result.SourceManager));
      }
    } else if (IteratorType.find("device_ptr") == std::string::npos) {
      NewName = "std::sort";
      if (hasExecutionPolicy) {
        emplaceTransformation(new ReplaceStmt(
            CE->getArg(0), "oneapi::dpl::execution::par_unseq"));
      }
      emplaceTransformation(new ReplaceCalleeName(CE, std::move(NewName)));
      return;
    } else {
      if (PolicyProcessed) {
        emplaceTransformation(new ReplaceCalleeName(CE, std::move(NewName)));
        return;
      } else if (hasExecutionPolicy)
        emplaceTransformation(removeArg(CE, 0, *Result.SourceManager));
    }
  } else if (hasExecutionPolicy) {
    emplaceTransformation(new ReplaceCalleeName(CE, std::move(NewName)));
    return;
  }

  if (ThrustFuncName == "exclusive_scan") {
    DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), HT_Numeric);
    emplaceTransformation(new InsertText(CE->getEndLoc(), ", 0"));
  }

  if (ULExpr)
    emplaceTransformation(new ReplaceToken(
        ULExpr->getBeginLoc(), ULExpr->getEndLoc(), std::move(NewName)));
  else
    emplaceTransformation(new ReplaceCalleeName(CE, std::move(NewName)));
  if (CE->getNumArgs() <= 0)
    return;
  auto ExtraParam = ReplInfo->second.ExtraParam;
  if (!ExtraParam.empty()) {
    // This is a temporary fix until, the Intel(R) oneAPI DPC++ Compiler and
    // Intel(R) oneAPI DPC++ Library support creating a SYCL execution policy
    // without creating a unique one for every use
    if (ExtraParam == "oneapi::dpl::execution::sycl") {
      // If no policy is specified and raw pointers are used
      // a host execution policy must be specified to match the thrust
      // behavior
      if (CE->getArg(0)->getType()->isPointerType()) {
        ExtraParam = "oneapi::dpl::execution::seq";
      } else {
        if (isPlaceholderIdxDuplicated(CE))
          return;
        ExtraParam = makeDevicePolicy(CE);
      }
    }
    emplaceTransformation(
        new InsertBeforeStmt(CE->getArg(0), ExtraParam + ", "));
  }
}

void ThrustFunctionRule::runRule(const MatchFinder::MatchResult &Result) {

  if (const UnresolvedLookupExpr *ULExpr =
          getAssistNodeAsType<UnresolvedLookupExpr>(
              Result, "unresolvedThrustAPILookupExpr")) {
    const CallExpr *CE =
        getAssistNodeAsType<CallExpr>(Result, "thrustApiCallExpr");
    thrustFuncMigration(Result, CE, ULExpr);
  }

  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "thrustFuncCall")) {
    thrustFuncMigration(Result, CE);
  }
}

REGISTER_RULE(ThrustFunctionRule)

void ThrustCtorExprRule::registerMatcher(MatchFinder &MF) {

  auto hasAnyThrustRecord = []() {
    return cxxRecordDecl(hasName("complex"),
                         hasDeclContext(namespaceDecl(hasName("thrust"))));
  };
  auto hasFunctionalActor = []() {
    return cxxRecordDecl(hasName("thrust::detail::functional::actor"));
  };

  MF.addMatcher(
      cxxConstructExpr(hasType(hasAnyThrustRecord())).bind("thrustCtorExpr"),
      this);
  MF.addMatcher(cxxConstructExpr(hasType(hasFunctionalActor()))
                    .bind("thrustCtorPlaceHolder"),
                this);
}

void ThrustCtorExprRule::replacePlaceHolderExpr(const CXXConstructExpr *CE) {
  unsigned PlaceholderCount = 0;

  auto placeholderStr = [](unsigned Num) {
    return std::string("_") + std::to_string(Num);
  };

  // Walk the expression and replace all placeholder occurrences
  std::function<void(const Stmt *)> walk = [&](const Stmt *S) {
    if (auto DRE = dyn_cast<DeclRefExpr>(S)) {
      auto DREStr = getStmtSpelling(DRE);
      auto TypeStr = DRE->getType().getAsString();
      std::string PlaceHolderTypeStr =
          "const thrust::detail::functional::placeholder<";
      if (TypeStr.find(PlaceHolderTypeStr) == 0) {
        unsigned PlaceholderNum =
            (TypeStr[PlaceHolderTypeStr.length()] - '0') + 1;
        if (PlaceholderNum > PlaceholderCount)
          PlaceholderCount = PlaceholderNum;
        emplaceTransformation(
            new ReplaceStmt(DRE, placeholderStr(PlaceholderNum)));
      }
      return;
    }
    for (auto SI : S->children())
      walk(SI);
  };
  walk(CE);

  if (PlaceholderCount == 0)
    // No placeholders were found, so no replacement is necessary
    return;

  // Construct the lambda wrapper and insert around placeholder expression
  std::string LambdaPrefix = "[=](";
  for (unsigned i = 1; i <= PlaceholderCount; ++i) {
    if (i > 1)
      LambdaPrefix += ",";
    LambdaPrefix += "auto _" + std::to_string(i);
  }
  LambdaPrefix += "){return ";
  emplaceTransformation(new InsertBeforeStmt(CE, std::move(LambdaPrefix)));
  std::string LambdaPostfix = ";}";
  emplaceTransformation(new InsertAfterStmt(CE, std::move(LambdaPostfix)));
}

void ThrustCtorExprRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const CXXConstructExpr *CE =
          getNodeAsType<CXXConstructExpr>(Result, "thrustCtorExpr")) {
    // handle constructor expressions for thrust::complex
    std::string ExprStr = getStmtSpelling(CE);
    if (ExprStr.substr(0, 8) != "thrust::") {
      return;
    }
    auto P = ExprStr.find('<');
    if (P != std::string::npos) {
      std::string FuncName = ExprStr.substr(8, P - 8);
      auto ReplInfo = MapNames::ThrustFuncNamesMap.find(FuncName);
      if (ReplInfo == MapNames::ThrustFuncNamesMap.end()) {
        return;
      }
      std::string ReplName = ReplInfo->second.ReplName;
      if (ReplName == "std::complex") {
        DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(),
                                                   HT_Complex);
      }
      emplaceTransformation(
          new ReplaceText(CE->getBeginLoc(), P, std::move(ReplName)));
    }
  } else if (const CXXConstructExpr *CE = getNodeAsType<CXXConstructExpr>(
                 Result, "thrustCtorPlaceHolder")) {
    // handle constructor expressions with placeholders (_1, _2, etc)
    replacePlaceHolderExpr(CE);
  }
}

REGISTER_RULE(ThrustCtorExprRule)

// Rule for types replacements in var declarations and field declarations
void TypeInDeclRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      typeLoc(
          loc(qualType(hasDeclaration(namedDecl(anyOf(
              hasAnyName(
                  "cudaError", "curandStatus", "cublasStatus", "CUstream",
                  "CUstream_st", "thrust::complex", "thrust::device_vector",
                  "thrust::device_ptr", "thrust::host_vector", "cublasHandle_t",
                  "CUevent_st", "__half", "half", "__half2", "half2",
                  "cudaMemoryAdvise", "cudaError_enum", "cudaDeviceProp",
                  "cudaPitchedPtr", "thrust::counting_iterator",
                  "thrust::transform_iterator", "thrust::permutation_iterator",
                  "thrust::iterator_difference", "cusolverDnHandle_t",
                  "thrust::device_malloc_allocator", "thrust::divides",
                  "thrust::tuple", "thrust::maximum", "thrust::multiplies",
                  "thrust::plus", "cudaDataType_t", "cudaError_t", "CUresult",
                  "CUdevice", "cudaEvent_t", "cublasStatus_t", "cuComplex",
                  "cuDoubleComplex", "CUevent", "cublasFillMode_t",
                  "cublasDiagType_t", "cublasSideMode_t", "cublasOperation_t",
                  "cusolverStatus_t", "cusolverEigType_t", "cusolverEigMode_t",
                  "curandStatus_t", "cudaStream_t", "cusparseStatus_t",
                  "cusparseDiagType_t", "cusparseFillMode_t",
                  "cusparseIndexBase_t", "cusparseMatrixType_t",
                  "cusparseOperation_t", "cusparseMatDescr_t",
                  "cusparseHandle_t", "CUcontext", "cublasPointerMode_t",
                  "cusparsePointerMode_t", "cublasGemmAlgo_t",
                  "cusparseSolveAnalysisInfo_t", "cudaDataType",
                  "cublasDataType_t", "curandState_t", "curandState",
                  "curandStateXORWOW_t", "curandStatePhilox4_32_10_t",
                  "curandStateMRG32k3a_t", "thrust::minus", "thrust::negate",
                  "thrust::logical_or", "thrust::identity", "thrust::equal_to",
                  "thrust::less", "cudaSharedMemConfig", "curandGenerator_t",
                  "cufftHandle", "cufftReal", "cufftDoubleReal", "cufftComplex",
                  "cufftDoubleComplex", "cufftResult_t", "cufftResult",
                  "cufftType_t", "cufftType", "thrust::pair", "CUdeviceptr",
                  "cudaDeviceAttr", "CUmodule", "CUfunction", "cudaMemcpyKind",
                  "cudaComputeMode"),
              matchesName("cudnn.*|nccl.*")))))))
          .bind("cudaTypeDef"),
      this);
  MF.addMatcher(varDecl(hasTypeLoc(typeLoc(loc(templateSpecializationType(
                            hasAnyTemplateArgument(refersToType(hasDeclaration(
                                namedDecl(hasName("use_default"))))))))))
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

void TypeInDeclRule::processCudaStreamType(const DeclaratorDecl *DD,
                                           const SourceManager *SM,
                                           bool &SpecialCaseHappened) {
  Token Tok;
  Lexer::getRawToken(DD->getBeginLoc(), Tok, *SM, LangOptions());
  auto Tok2Ptr = Lexer::findNextToken(DD->getBeginLoc(), *SM, LangOptions());
  llvm::Optional<Token> TokAfterTypePtr;
  Token TokAfterType;
  // Distinguish between variable decls and just types (e.g. in function
  // signatures). If the next token of the tye is comma or r_paren, it is
  // just a name.
  // It matters for migration of cudaStream_t.
  bool IsTokenAfterTypeCommaRParenOrRef = false;

  if (Tok2Ptr.hasValue()) {
    auto Tok2 = Tok2Ptr.getValue();
    SourceLocation InsertLoc;
    auto PointerType = deducePointerType(DD, "CUstream_st");
    std::string TypeStr = Tok.getRawIdentifier().str();
    if (Tok.getKind() == tok::raw_identifier && TypeStr == "cudaStream_t") {
      SpecialCaseHappened = true;
      SrcAPIStaticsMap[TypeStr]++;
      // cudaStream_t const
      if (Tok2.getKind() == tok::raw_identifier &&
          Tok2.getRawIdentifier() == "const") {
        TokAfterTypePtr =
            Lexer::findNextToken(Tok2.getLocation(), *SM, LangOptions());
        if (TokAfterTypePtr.hasValue()) {
          TokAfterType = TokAfterTypePtr.getValue();
          if (TokAfterType.getKind() == tok::comma ||
              TokAfterType.getKind() == tok::r_paren ||
              TokAfterType.getKind() == tok::amp)
            IsTokenAfterTypeCommaRParenOrRef = true;
        }
        if (IsTokenAfterTypeCommaRParenOrRef) {
          emplaceTransformation(new ReplaceToken(Tok.getLocation(), ""));
          std::string T{MapNames::getClNamespace() + "queue "};
          T += PointerType;
          emplaceTransformation(
              new ReplaceToken(Tok2.getLocation(), std::move(T)));
        } else {
          emplaceTransformation(new ReplaceToken(Tok.getLocation(), ""));
          emplaceTransformation(new ReplaceToken(
              Tok2.getLocation(), MapNames::getClNamespace() + "queue"));
          InsertLoc = Tok2.getEndLoc().getLocWithOffset(1);
          emplaceTransformation(
              new InsertText(InsertLoc, std::move(PointerType)));
        }
      }
      // cudaStream_t
      else {
        TokAfterTypePtr =
            Lexer::findNextToken(Tok.getLocation(), *SM, LangOptions());
        if (TokAfterTypePtr.hasValue()) {
          TokAfterType = TokAfterTypePtr.getValue();
          if (TokAfterType.getKind() == tok::comma ||
              TokAfterType.getKind() == tok::r_paren ||
              TokAfterType.getKind() == tok::amp)
            IsTokenAfterTypeCommaRParenOrRef = true;
        }
        if (IsTokenAfterTypeCommaRParenOrRef) {
          std::string T{MapNames::getClNamespace() + "queue "};
          T += PointerType;
          emplaceTransformation(
              new ReplaceToken(Tok.getLocation(), std::move(T)));
        } else {
          emplaceTransformation(new ReplaceToken(
              Tok.getLocation(), MapNames::getClNamespace() + "queue"));
          InsertLoc = Tok.getEndLoc().getLocWithOffset(1);
          emplaceTransformation(
              new InsertText(InsertLoc, std::move(PointerType)));
        }
      }
    } else if (Tok.getKind() == tok::raw_identifier &&
               Tok.getRawIdentifier() == "const") {

      // const cudaStream_t
      TypeStr = Tok2.getRawIdentifier().str();
      if (Tok.getKind() == tok::raw_identifier && TypeStr == "cudaStream_t") {
        SpecialCaseHappened = true;
        SrcAPIStaticsMap[TypeStr]++;
        TokAfterTypePtr =
            Lexer::findNextToken(Tok2.getLocation(), *SM, LangOptions());
        if (TokAfterTypePtr.hasValue()) {
          TokAfterType = TokAfterTypePtr.getValue();
          if (TokAfterType.getKind() == tok::comma ||
              TokAfterType.getKind() == tok::r_paren ||
              TokAfterType.getKind() == tok::amp)
            IsTokenAfterTypeCommaRParenOrRef = true;
        }
        if (IsTokenAfterTypeCommaRParenOrRef) {
          emplaceTransformation(new ReplaceToken(Tok.getLocation(), ""));
          std::string T{MapNames::getClNamespace() + "queue "};
          T += PointerType;
          emplaceTransformation(
              new ReplaceToken(Tok2.getLocation(), std::move(T)));
        } else {
          emplaceTransformation(new ReplaceToken(Tok.getLocation(), ""));
          emplaceTransformation(new ReplaceToken(
              Tok2.getLocation(), MapNames::getClNamespace() + "queue"));
          InsertLoc = Tok2.getEndLoc().getLocWithOffset(1);
          emplaceTransformation(
              new InsertText(InsertLoc, std::move(PointerType)));
        }
      }
    }
  }
  auto SD = getSiblingDecls(DD);
  for (auto It = SD.begin(); It != SD.end(); ++It) {
    auto DD2 = *It;
    auto L2 = DD2->getLocation();
    auto P = SM->getCharacterData(L2);
    // Find the first non-space char after previous semicolon
    while (*P != ',')
      --P;
    ++P;
    while (isspace(*P))
      ++P;
    // Insert "*" or "*const" right before it
    auto InsertLoc = L2.getLocWithOffset(P - SM->getCharacterData(L2));
    auto PointerType = deducePointerType(DD2, "CUstream_st");
    emplaceTransformation(new InsertText(InsertLoc, std::move(PointerType)));
  }
}

void TypeInDeclRule::reportForNcclAndCudnn(const TypeLoc *TL,
                                           const SourceLocation BeginLoc) {
  auto QT = TL->getType();
  std::string TypeStrRemovePrefix = TL->getType().getAsString();

  auto IsNCCLMatched = TypeStrRemovePrefix.find("nccl") != std::string::npos;
  auto IsCUDNNMatched = TypeStrRemovePrefix.find("cudnn") != std::string::npos;
  if (IsNCCLMatched || IsCUDNNMatched) {
    auto TP = QT.getTypePtr();
    if (TP) {
      SourceLocation SL;
      if (getTypeDeclLocation(TP, SL)) {
        std::string FilePath =
            DpctGlobalInfo::getSourceManager().getFilename(SL).str();
        if (!DpctGlobalInfo::isInRoot(FilePath)) {
          if (IsNCCLMatched)
            report(BeginLoc, Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
                   "Intel(R) oneAPI Collective Communications Library");
          else if (IsCUDNNMatched)
            report(BeginLoc, Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
                   "Intel(R) oneAPI Deep Neural Network Library (oneDNN)");
        }
      }
    }
  }
}

bool TypeInDeclRule::replaceTemplateSpecialization(
    SourceManager *SM, LangOptions &LOpts, SourceLocation BeginLoc,
    const TemplateSpecializationTypeLoc TSL) {

  for (unsigned i = 0; i < TSL.getNumArgs(); ++i) {
    auto ArgLoc = TSL.getArgLoc(i);
    if (ArgLoc.getArgument().getKind() != TemplateArgument::Type)
      continue;
    auto UTL = ArgLoc.getTypeSourceInfo()->getTypeLoc().getUnqualifiedLoc();

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

      requestHelperFeatureForTypeNames(RealTypeNameStr, ETBeginLoc);

      std::string Replacement =
          MapNames::findReplacedName(MapNames::TypeNamesMap, RealTypeNameStr);

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
    Tok = Lexer::findNextToken(BeginLoc, *SM, LOpts).getValue();
    BeginLoc = Tok.getLocation();
  }
  auto LAngleLoc = TSL.getLAngleLoc();

  const char *Start = SM->getCharacterData(BeginLoc);
  const char *End = SM->getCharacterData(LAngleLoc);
  auto TyLen = End - Start;
  if (TyLen <= 0)
    return false;

  const std::string RealTypeNameStr(Start, TyLen);
  requestHelperFeatureForTypeNames(RealTypeNameStr, BeginLoc);
  std::string Replacement =
      MapNames::findReplacedName(MapNames::TypeNamesMap, RealTypeNameStr);
  if (!Replacement.empty()) {
    insertComplexHeader(BeginLoc, Replacement);
    if (RealTypeNameStr == "thrust::identity") {
      // CTST-2049: For thrust::identity the template type argument must be
      // removed as well for correct mapping to oneapi::dpl::identity
      auto RAngleLoc = TSL.getRAngleLoc();
      TyLen = SM->getCharacterData(RAngleLoc) - Start + 1;
    }
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
  if (NNTL.getTypeLocClass() == clang::TypeLoc::TemplateSpecialization) {
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
    requestHelperFeatureForTypeNames(NameToMap, TL->getBeginLoc());
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
      auto Arg1 = TSTL.getArgLoc(0).getTypeSourceInfo()->getTypeLoc();
      auto Arg2 = TSTL.getArgLoc(1).getTypeSourceInfo()->getTypeLoc();
      std::string Arg1Str = getNewTypeStr(&Arg1);
      std::string Arg2Str = getNewTypeStr(&Arg2);
      return NewBaseTypeStr + "<" + Arg2Str + ", " + Arg1Str + ">";
    }
    // Recurse down through the template arguments
    std::string NewTypeStr = NewBaseTypeStr + "<";
    for (unsigned i = 0; i < TSTL.getNumArgs(); ++i) {
      std::string ArgStr;
      if (TSTL.getArgLoc(i).getArgument().getKind() == TemplateArgument::Type) {
        auto ArgLoc = TSTL.getArgLoc(i).getTypeSourceInfo()->getTypeLoc();
        ArgStr = getNewTypeStr(&ArgLoc);
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

bool TypeInDeclRule::isDeviceRandomStateType(const TypeLoc *TL,
                                             const SourceLocation &SL) {
  std::string TypeStr = TL->getType().getAsString();

  if (MapNames::DeviceRandomGeneratorTypeMap.find(TypeStr) !=
      MapNames::DeviceRandomGeneratorTypeMap.end()) {
    if (TypeStr == "curandState_t" || TypeStr == "curandState" ||
        TypeStr == "curandStateXORWOW_t") {
      report(SL, Diagnostics::DIFFERENT_GENERATOR, false);
    }
    return true;
  } else {
    return false;
  }
}

void TypeInDeclRule::runRule(const MatchFinder::MatchResult &Result) {
  SourceManager *SM = Result.SourceManager;
  auto LOpts = Result.Context->getLangOpts();
  if (auto TL = getNodeAsType<TypeLoc>(Result, "cudaTypeDef")) {

    // if TL is the T in
    // template<typename T> void foo(T a);
    if (TL->getTypeLocClass() == clang::TypeLoc::SubstTemplateTypeParm ||
        TL->getBeginLoc().isInvalid()) {
      return;
    }

    auto TypeStr =
        DpctGlobalInfo::getTypeName(TL->getType().getUnqualifiedType());

    if (ProcessedTypeLocs.find(*TL) != ProcessedTypeLocs.end())
      return;

    // Try to migrate cudaSuccess to sycl::info::event_command_status if it is
    // used in cases like "cudaSuccess == cudaEventQuery()".
    if (EventAPICallRule::getEventQueryTraversal().startFromTypeLoc(*TL))
      return;

    // when the following code is not in inroot
    // #define MACRO_SHOULD_NOT_BE_MIGRATED (MatchedType)3
    // Even if MACRO_SHOULD_NOT_BE_MIGRATED used in inroot, DPCT should not
    // migrate MatchedType.
    if (!DpctGlobalInfo::isInRoot(SM->getSpellingLoc(TL->getBeginLoc())) &&
        isPartOfMacroDef(SM->getSpellingLoc(TL->getBeginLoc()),
                         SM->getSpellingLoc(TL->getBeginLoc()))) {
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

    if (isDeviceRandomStateType(TL, BeginLoc)) {
      auto P = MapNames::DeviceRandomGeneratorTypeMap.find(TypeStr);
      DpctGlobalInfo::getInstance().insertDeviceRandomStateTypeInfo(
          BeginLoc,
          Lexer::MeasureTokenLength(BeginLoc, *SM,
                                    DpctGlobalInfo::getContext().getLangOpts()),
          P->second);
      return;
    }

    if (TypeStr == "curandGenerator_t") {
      DpctGlobalInfo::getInstance().insertHostRandomEngineTypeInfo(
          BeginLoc,
          Lexer::MeasureTokenLength(
              BeginLoc, *SM, DpctGlobalInfo::getContext().getLangOpts()));
      return;
    }

    if (TypeStr == "cufftHandle") {
      DpctGlobalInfo::getInstance().insertFFTDescriptorTypeInfo(
          BeginLoc,
          Lexer::MeasureTokenLength(
              BeginLoc, *SM, DpctGlobalInfo::getContext().getLangOpts()));
      return;
    }

    reportForNcclAndCudnn(TL, BeginLoc);

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
        auto TSL = NTL.getUnqualifiedLoc().getAs<RecordTypeLoc>();

        const std::string TyName =
            dpct::DpctGlobalInfo::getTypeName(TSL.getType());
        std::string Replacement =
            MapNames::findReplacedName(MapNames::TypeNamesMap, TyName);
        requestHelperFeatureForTypeNames(TyName, BeginLoc);

        if (!Replacement.empty()) {
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
    }

    std::string Str =
        MapNames::findReplacedName(MapNames::TypeNamesMap, TypeStr);
    requestHelperFeatureForTypeNames(TypeStr, BeginLoc);
    // Add '#include <complex>' directive to the file only once
    if (TypeStr == "cuComplex" || TypeStr == "cuDoubleComplex") {
      DpctGlobalInfo::getInstance().insertHeader(BeginLoc, HT_Complex);
    }

    if (TypeStr.rfind("identity", 0) == 0) {
      emplaceTransformation(new ReplaceToken(
          TL->getBeginLoc().getLocWithOffset(Lexer::MeasureTokenLength(
              TL->getBeginLoc(), dpct::DpctGlobalInfo::getSourceManager(),
              dpct::DpctGlobalInfo::getContext().getLangOpts())),
          TL->getEndLoc(), ""));
    }

    const DeclStmt *DS = DpctGlobalInfo::findAncestor<DeclStmt>(TL);
    if (TypeStr == "cusparseMatDescr_t" && DS) {
      for (auto I : DS->decls()) {
        const VarDecl *VDI = dyn_cast<VarDecl>(I);
        if (VDI && VDI->hasInit()) {
          if (VDI->getInitStyle() == VarDecl::InitializationStyle::CInit) {
            const Expr *IE = VDI->getInit();
            // cusparseMatDescr_t descr = InitExpr ;
            //                         |          |
            //                       Begin       End
            auto End = SM->getExpansionRange(IE->getEndLoc()).getEnd();
            End = End.getLocWithOffset(Lexer::MeasureTokenLength(
                End, *SM, DpctGlobalInfo::getContext().getLangOpts()));
            SourceLocation Begin =
                SM->getExpansionRange(IE->getBeginLoc()).getBegin();

            auto C = SM->getCharacterData(Begin);
            int Offset = 0;
            while (*C != '=') {
              C--;
              Offset--;
            }
            Begin = Begin.getLocWithOffset(Offset);

            int Len = SM->getDecomposedLoc(End).second -
                      SM->getDecomposedLoc(Begin).second;
            assert(Len > 0);
            emplaceTransformation(new ReplaceText(Begin, Len, ""));
          }
        }
      }
    }

    const DeclaratorDecl *DD = nullptr;
    const VarDecl *VarD = DpctGlobalInfo::findAncestor<VarDecl>(TL);
    const FieldDecl *FieldD = DpctGlobalInfo::findAncestor<FieldDecl>(TL);
    const FunctionDecl *FD = DpctGlobalInfo::findAncestor<FunctionDecl>(TL);
    if (FD &&
        (FD->hasAttr<CUDADeviceAttr>() || FD->hasAttr<CUDAGlobalAttr>())) {
      if (TL->getType().getAsString().find("cublasHandle_t") !=
          std::string::npos)
        report(BeginLoc, Diagnostics::HANDLE_IN_DEVICE, false, TypeStr);
    }

    const Expr *Init = nullptr;

    if (VarD) {
      DD = VarD;
      if (VarD->hasInit())
        Init = VarD->getInit();
    } else if (FieldD) {
      DD = FieldD;
      if (FieldD->hasInClassInitializer())
        Init = FieldD->getInClassInitializer();
    }

    auto IsTypeInInitializer = [&]() -> bool {
      if (!Init)
        return false;
      if (TL->getBeginLoc() >= Init->getBeginLoc() &&
          TL->getEndLoc() <= Init->getEndLoc())
        return true;
      return false;
    };

    bool SpecialCaseHappened = false;
    if (DD && !IsTypeInInitializer()) {
      if (TL->getType().getAsString().find("cudaStream_t") !=
          std::string::npos) {
        processCudaStreamType(DD, SM, SpecialCaseHappened);
      }
    }
    if (!Str.empty() && !SpecialCaseHappened) {
      SrcAPIStaticsMap[TypeStr]++;

      /// Process code like:
      /// \code
      ///   vector.push_back(cudaStream_t());
      ///   cudaStream_t s = cudaStream_t();
      /// \endcode
      if (TL->getType().getAsString() == "cudaStream_t" ||
          TL->getType().getAsString() == "CUstream") {
        if (auto CSVIE =
                DpctGlobalInfo::findParent<CXXScalarValueInitExpr>(TL)) {
          emplaceTransformation(new ReplaceStmt(CSVIE, "nullptr"));
          return;
        }
      }

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

    auto TSTL = TL.getAs<TemplateSpecializationTypeLoc>();
    if (!TSTL)
      return;
    auto TST = dyn_cast<TemplateSpecializationType>(
        VD->getType().getUnqualifiedType());
    if (!TST)
      return;

    bool HasUnremovedPreviousArg = 0;
    for (unsigned i = 0; i < TST->getNumArgs(); i++) {
      if (!dpct::DpctGlobalInfo::getTypeName(TST->getArg(0).getAsType())
               .compare("thrust::use_default")) {
        auto ArgBeginLoc = TSTL.getArgLoc(i).getSourceRange().getBegin();
        auto ArgEndLoc = TSTL.getArgLoc(i).getSourceRange().getEnd();
        if (HasUnremovedPreviousArg && i < TST->getNumArgs() - 1) {
          ArgEndLoc = TSTL.getArgLoc(i - 1).getSourceRange().getBegin();
        }
        emplaceTransformation(new ReplaceToken(ArgBeginLoc, ArgEndLoc, ""));
      } else {
        HasUnremovedPreviousArg = 1;
      }
    }
  }
}

REGISTER_RULE(TypeInDeclRule)

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
}

void VectorTypeNamespaceRule::runRule(const MatchFinder::MatchResult &Result) {
  SourceManager *SM = Result.SourceManager;
  if (auto TL = getNodeAsType<TypeLoc>(Result, "vectorTypeTL")) {
    if (TL->getBeginLoc().isInvalid())
      return;

    // To skip user-defined type.
    if (const auto *ND = getNamedDecl(TL->getTypePtr())) {
      auto Loc = ND->getBeginLoc();
      auto Path = dpct::DpctGlobalInfo::getLocInfo(Loc).first;
      if (DpctGlobalInfo::isInRoot(Path, true))
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
      std::string Str =
          MapNames::findReplacedName(MapNames::TypeNamesMap, TypeStr);
      requestHelperFeatureForTypeNames(TypeStr, BeginLoc);
      if (!Str.empty()) {
        SrcAPIStaticsMap[TypeStr]++;
        emplaceTransformation(new ReplaceToken(BeginLoc, std::move(Str)));
      }
      if (*(TypeStr.end() - 1) == '1') {
        NeedRemoveVolatile = false;
      }
    }

    if (IsInScratchspace) {
      std::string TypeStr = TL->getType().getUnqualifiedType().getAsString();
      auto Begin = SM->getImmediateExpansionRange(TL->getBeginLoc()).getBegin();
      auto End = SM->getImmediateExpansionRange(TL->getEndLoc()).getEnd();
      if (*(TypeStr.end() - 1) == '1') {
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
      if (VD->getType().isVolatileQualified()) {
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
  if (auto CRD = getNodeAsType<CXXRecordDecl>(Result, "inheritanceType")) {
    for (auto ItBase = CRD->bases_begin(); ItBase != CRD->bases_end();
         ItBase++) {
      std::string TypeName = ItBase->getBaseTypeInfo()->getType().getAsString();
      if (MapNames::SupportedVectorTypes.find(TypeName) ==
          MapNames::SupportedVectorTypes.end())
        return;
      auto Begin = ItBase->getSourceRange().getBegin();
      auto End = ItBase->getSourceRange().getEnd();
      if (Begin.isInvalid()) {
        return;
      }
      if (*(TypeName.end() - 1) == '1') {
        if (Begin.isMacroID() &&
            (SM->isWrittenInScratchSpace(SM->getSpellingLoc(Begin)) ||
             SM->isWrittenInScratchSpace(SM->getSpellingLoc(End)))) {
          // Macro concatenate --> use immediateExpansion
          // Make (Begin, End) be the range of "##1"
          Begin = SM->getImmediateExpansionRange(Begin).getBegin();
          End = SM->getImmediateExpansionRange(End).getEnd();
          Begin = SM->getSpellingLoc(Begin);
          End = SM->getSpellingLoc(End);
          Begin = Begin.getLocWithOffset(Lexer::MeasureTokenLength(
              Begin, *SM, DpctGlobalInfo::getContext().getLangOpts()));
          End = End.getLocWithOffset(Lexer::MeasureTokenLength(
              End, *SM, DpctGlobalInfo::getContext().getLangOpts()));
          report(Begin, Comments::VECTYPE_INHERITATED, false);
        } else {
          // Make (Begin, End) be the range of "1"
          Begin = SM->getSpellingLoc(Begin);
          End = SM->getSpellingLoc(End);
          Begin = Begin.getLocWithOffset(
              Lexer::MeasureTokenLength(
                  Begin, *SM, DpctGlobalInfo::getContext().getLangOpts()) -
              1);
          End = End.getLocWithOffset(Lexer::MeasureTokenLength(
              End, *SM, DpctGlobalInfo::getContext().getLangOpts()));
        }
        auto Length = SM->getFileOffset(End) - SM->getFileOffset(Begin);
        return emplaceTransformation(new ReplaceText(Begin, Length, ""));
      }

      if (Begin.isInvalid())
        return;

      if (Begin.isMacroID()) {
        // Macro concatenate --> use immediateExpansion
        // Make Begin be the begin of "MACROARG##1"
        if (SM->isWrittenInScratchSpace(SM->getSpellingLoc(Begin))) {
          Begin = SM->getImmediateExpansionRange(Begin).getBegin();
        }
        Begin = SM->getSpellingLoc(Begin);
      }
      return emplaceTransformation(
          new InsertText(Begin, MapNames::getClNamespace()));
    }
  }
}

REGISTER_RULE(VectorTypeNamespaceRule)

void VectorTypeMemberAccessRule::registerMatcher(MatchFinder &MF) {
  auto memberAccess = [&]() {
    return hasObjectExpression(hasType(qualType(hasCanonicalType(
        recordType(hasDeclaration(cxxRecordDecl(vectorTypeName())))))));
  };

  // int2.x => int2.x()
  MF.addMatcher(
      memberExpr(allOf(memberAccess(), unless(hasParent(binaryOperator(allOf(
                                           hasLHS(memberExpr(memberAccess())),
                                           isAssignmentOperator()))))))
          .bind("VecMemberExpr"),
      this);

  // class A : int2{ void foo(){x = 3;}}
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(pointsTo(cxxRecordDecl(
                               hasAnyName(SUPPORTEDVECTORTYPENAMES))))))
                    .bind("DerivedVecMemberExpr"),
                this);

  // int2.x += xxx => int2.x() += xxx
  MF.addMatcher(
      binaryOperator(allOf(hasLHS(memberExpr(memberAccess())
                                      .bind("VecMemberExprAssignmentLHS")),
                           isAssignmentOperator()))
          .bind("VecMemberExprAssignment"),
      this);
}

void VectorTypeMemberAccessRule::renameMemberField(const MemberExpr *ME) {

  // To skip user-defined type.
  if (!ME || isTypeInRoot(ME->getBase()->getType().getTypePtr()))
    return;

  auto BaseTy = ME->getBase()->getType().getAsString();
  bool isPtr = false;
  // when BaseTy == "struct int1 *", remove " *"
  if (*(BaseTy.end() - 1) == '*') {
    BaseTy = BaseTy.erase(BaseTy.size() - 2, 2);
    isPtr = true;
  }
  auto &SM = DpctGlobalInfo::getSourceManager();
  if (*(BaseTy.end() - 1) == '1') {
    auto Begin = ME->getOperatorLoc();
    bool isImplicit = false;
    if (Begin.isInvalid()) {
      Begin = ME->getMemberLoc();
      isImplicit = true;
    }
    Begin = SM.getSpellingLoc(Begin);
    auto End =
        Lexer::getLocForEndOfToken(SM.getSpellingLoc(ME->getMemberLoc()), 0, SM,
                                   DpctGlobalInfo::getContext().getLangOpts());
    auto Length = SM.getFileOffset(End) - SM.getFileOffset(Begin);
    if (isPtr && isImplicit) {
      return emplaceTransformation(new ReplaceText(Begin, Length, "*this"));
    }
    if (isPtr) {
      auto BaseBegin = ME->getBeginLoc();
      emplaceTransformation(new InsertText(BaseBegin, "*"));
    }
    return emplaceTransformation(new ReplaceText(Begin, Length, ""));
  }
  std::string MemberName = ME->getMemberNameInfo().getAsString();
  if (MapNames::replaceName(MapNames::MemberNamesMap, MemberName))
    emplaceTransformation(
        new RenameFieldInMemberExpr(ME, std::move(MemberName)));
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

  if (auto ME = getNodeAsType<MemberExpr>(Result, "VecMemberExprAssignmentLHS"))
    renameMemberField(ME);
}

REGISTER_RULE(VectorTypeMemberAccessRule)

namespace clang {
namespace ast_matchers {

AST_MATCHER(FunctionDecl, overloadedVectorOperator) {
  if (!DpctGlobalInfo::isInRoot(Node.getBeginLoc()))
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

  // Helper function to get the scope of function declartion
  // Eg:
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
    const std::string Str = std::string(";") + getNL();
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

  const std::string OperatorName =
      BinaryOperator::getOpcodeStr(
          BinaryOperator::getOverloadedOpcode(CE->getOperator()))
          .str();

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

REGISTER_RULE(VectorTypeOperatorRule)

void VectorTypeCtorRule::registerMatcher(MatchFinder &MF) {

  // make_int2
  auto makeVectorFunc = [&]() {
    std::vector<std::string> MakeVectorFuncNames;
    for (const std::string &TypeName : MapNames::SupportedVectorTypes) {
      MakeVectorFuncNames.emplace_back("make_" + TypeName);
    }

    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(MakeVectorFuncNames));
  };

  // migrate utility for vector type: eg: make_int2
  MF.addMatcher(
      callExpr(callee(functionDecl(makeVectorFunc()))).bind("VecUtilFunc"),
      this);
}

std::string
VectorTypeCtorRule::getReplaceTypeName(const std::string &TypeName) {
  return std::string(
      MapNames::findReplacedName(MapNames::TypeNamesMap, TypeName));
}

// Determines which case of construction applies and creates replacements for
// the syntax. Returns the constructor node and a boolean indicating if a
// closed brace needs to be appended.
void VectorTypeCtorRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "VecUtilFunc")) {
    if (!CE->getDirectCallee())
      return;

    assert(CE->getDirectCallee()->getName().startswith("make_") &&
           "Found non make_<vector type> function");
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), getReplaceTypeName(CE->getType().getAsString())));
    return;
  }
}

REGISTER_RULE(VectorTypeCtorRule)

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
                    unless(hasAncestor(
                        cxxConstructExpr(hasType(namedDecl(hasName("dim3")))))))
                    .bind("dim3CtorDecl"),
                this);

  MF.addMatcher(
      cxxConstructExpr(hasType(namedDecl(hasName("dim3"))), argumentCountIs(3),
                       // skip fields in a struct.  The source loc is
                       // messed up (points to the start of the struct)
                       unless(hasAncestor(cxxRecordDecl())),
                       unless(hasParent(varDecl())),
                       unless(hasParent(exprWithCleanups())),
                       unless(hasAncestor(cxxConstructExpr(
                           hasType(namedDecl(hasName("dim3")))))))
          .bind("dim3CtorNoDecl"),
      this);

  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(anyOf(
                  namedDecl(hasAnyName("dim3", "cudaExtent", "cudaPos")),
                  typedefDecl(hasAnyName("dim3", "cudaExtent", "cudaPos")))))))
          .bind("dim3Type"),
      this);
}

ReplaceDim3Ctor *ReplaceDim3CtorRule::getReplaceDim3Modification(
    const MatchFinder::MatchResult &Result) {
  if (auto Ctor = getNodeAsType<CXXConstructExpr>(Result, "dim3CtorDecl")) {
    // dim3 a; or dim3 a(1);
    return new ReplaceDim3Ctor(Ctor, true /*isDecl*/);
  } else if (auto Ctor =
                 getNodeAsType<CXXConstructExpr>(Result, "dim3CtorNoDecl")) {
    // deflt = dim3(3);
    return new ReplaceDim3Ctor(Ctor, false /*isDecl*/);
  } else if (auto Ctor = getNodeAsType<CXXConstructExpr>(Result, "dim3Top")) {
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
    // add a transformation that will filter out all nested transformations
    emplaceTransformation(R->getEmpty());
    // all the nested transformations will be applied when R->getReplacement()
    // is called
    emplaceTransformation(R);
  }

  if (auto TL = getNodeAsType<TypeLoc>(Result, "dim3Type")) {
    if (TL->getBeginLoc().isInvalid())
      return;

    auto BeginLoc =
        getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc()).getBegin();
    SourceManager *SM = Result.SourceManager;

    // WA for concatinated macro token
    if (SM->isWrittenInScratchSpace(SM->getSpellingLoc(TL->getBeginLoc()))) {
      BeginLoc = SM->getExpansionLoc(TL->getBeginLoc());
    }

    Token Tok;
    auto LOpts = Result.Context->getLangOpts();
    Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
    if (Tok.isAnyIdentifier()) {

      if (TL->getTypeLocClass() == clang::TypeLoc::Elaborated) {
        // To handle case like "struct cudaExtent extent;"
        auto ETC = TL->getUnqualifiedLoc().getAs<ElaboratedTypeLoc>();
        auto NTL = ETC.getNamedTypeLoc();

        if (NTL.getTypeLocClass() == clang::TypeLoc::Record) {
          auto TSL = NTL.getUnqualifiedLoc().getAs<RecordTypeLoc>();

          const std::string TyName =
              dpct::DpctGlobalInfo::getTypeName(TSL.getType());
          std::string Str =
              MapNames::findReplacedName(MapNames::TypeNamesMap, TyName);
          requestHelperFeatureForTypeNames(TyName, BeginLoc);

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
      requestHelperFeatureForTypeNames(TypeName, BeginLoc);
      if (auto VD = DpctGlobalInfo::findAncestor<VarDecl>(TL)) {
        auto TypeStr = VD->getType().getAsString();
        if (VD->getKind() == Decl::Var &&
            (TypeStr == "dim3" || TypeStr == "struct cudaExtent" ||
             TypeStr == "struct cudaPos")) {
          std::string Replacement;
          std::string ReplacedType = "range";
          if (TypeStr == "dim3" || TypeStr == "struct cudaExtent") {
            ReplacedType = "range";
          } else if (TypeStr == "struct cudaPos") {
            ReplacedType = "id";
          }

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

REGISTER_RULE(ReplaceDim3CtorRule)

void Dim3MemberFieldsRule::FieldsRename(const MatchFinder::MatchResult &Result,
                                        std::string Str, const MemberExpr *ME) {
  auto SM = Result.SourceManager;

  SourceLocation MemberLoc, OptLoc;
  MemberLoc = SM->getSpellingLoc(ME->getMemberLoc());
  OptLoc = SM->getSpellingLoc(ME->getOperatorLoc());
  bool isArrow = ME->isArrow();
  if (isArrow)
    emplaceTransformation(new ReplaceText(OptLoc, 2, ""));
  else
    emplaceTransformation(new ReplaceText(OptLoc, 1, ""));

  auto Search =
      MapNames::Dim3MemberNamesMap.find(ME->getMemberNameInfo().getAsString());
  if (Search != MapNames::Dim3MemberNamesMap.end()) {
    std::string NewString = Search->second;
    emplaceTransformation(new ReplaceText(MemberLoc, 1, std::move(NewString)));
  }
}

// rule for dim3 types member fields replacements.
void Dim3MemberFieldsRule::registerMatcher(MatchFinder &MF) {
  // dim3->x/y/z => dim3->operator[](0)/(1)/(2)
  MF.addMatcher(
      memberExpr(
          has(implicitCastExpr(hasType(pointsTo(typedefDecl(hasName("dim3")))))
                  .bind("ImplCast")))
          .bind("Dim3MemberPointerExpr"),
      this);

  // dim3.x/y/z => dim3[0]/[1]/[2]
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(qualType(hasCanonicalType(
              recordType(hasDeclaration(cxxRecordDecl(hasName("dim3")))))))))
          .bind("Dim3MemberDotExpr"),
      this);
}

void Dim3MemberFieldsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "Dim3MemberPointerExpr")) {
    // E.g.
    // dim3 *pd3;
    // pd3->x;
    // will migrate to:
    // sycl::range<3> *pd3;
    // (*pd3)[0];
    auto Impl = getAssistNodeAsType<ImplicitCastExpr>(Result, "ImplCast");
    insertAroundStmt(Impl, "(*", ")");
    FieldsRename(Result, "->", ME);
  }

  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "Dim3MemberDotExpr")) {
    FieldsRename(Result, ".", ME);
  }
}

REGISTER_RULE(Dim3MemberFieldsRule)

void DevicePropVarRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(anyOf(
              hasType(qualType(hasCanonicalType(recordType(
                  hasDeclaration(cxxRecordDecl(hasName("cudaDeviceProp"))))))),
              hasType(
                  pointsTo(qualType(hasCanonicalType(recordType(hasDeclaration(
                      cxxRecordDecl(hasName("cudaDeviceProp")))))))))))
          .bind("DevicePropVar"),
      this);
}

void DevicePropVarRule::runRule(const MatchFinder::MatchResult &Result) {
  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "DevicePropVar");
  if (!ME)
    return;
  auto Parents = Result.Context->getParents(*ME);
  if (Parents.size() < 1)
    return;
  auto MemberName = ME->getMemberNameInfo().getAsString();

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
  } else if (MemberName == "pciDomainID" || MemberName == "pciBusID" ||
             MemberName == "pciDeviceID") {
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
  } else if (MemberName == "totalConstMem") {
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, "0");
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), "0"));
    return;
  } else if (MemberName == "textureAlignment") {
    requestFeature(HelperFeatureEnum::Device_get_current_device, ME);
    std::string Repl =
        MapNames::getDpctNamespace() + "get_current_device().get_info<" +
        MapNames::getClNamespace() + "info::device::mem_base_addr_align>()";
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, Repl);
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), std::move(Repl)));
    return;
  } else if (MemberName == "ECCEnabled") {
    requestFeature(HelperFeatureEnum::Device_get_current_device, ME);
    std::string Repl = MapNames::getDpctNamespace() +
                       "get_current_device().get_info<" +
                       MapNames::getClNamespace() +
                       "info::device::error_correction_support>()";
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), std::move(Repl)));
    return;
  }

  if (MemberName == "sharedMemPerBlock") {
    report(ME->getBeginLoc(), Diagnostics::LOCAL_MEM_SIZE, false);
  } else if (MemberName == "maxGridSize") {
    report(ME->getBeginLoc(), Diagnostics::MAX_GRID_SIZE, false);
  }

  auto Search = PropNamesMap.find(MemberName);
  if (Search == PropNamesMap.end()) {
    // TODO report migration error
    return;
  }
  if (Parents[0].get<clang::ImplicitCastExpr>()) {
    // migrate to get_XXX() eg. "b=a.minor" to "b=a.get_minor_version()"
    requestFeature(PropToGetFeatureMap.at(MemberName), ME);
    emplaceTransformation(
        new RenameFieldInMemberExpr(ME, "get_" + Search->second + "()"));
  } else if (auto *BO = Parents[0].get<clang::BinaryOperator>()) {
    // migrate to set_XXX() eg. "a.minor = 1" to "a.set_minor_version(1)"
    if (BO->getOpcode() == clang::BO_Assign) {
      requestFeature(PropToSetFeatureMap.at(MemberName), ME);
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, "set_" + Search->second));
      emplaceTransformation(new ReplaceText(BO->getOperatorLoc(), 1, "("));
      emplaceTransformation(new InsertAfterStmt(BO, ")"));
    }
  }
  if ((Search->second.compare(0, 13, "major_version") == 0) ||
      (Search->second.compare(0, 13, "minor_version") == 0)) {
    report(ME->getBeginLoc(), Comments::VERSION_COMMENT, false);
  }
  if (Search->second.compare(0, 10, "integrated") == 0) {
    report(ME->getBeginLoc(), Comments::NOT_SUPPORT_API_INTEGRATEDORNOT, false);
  }
}

REGISTER_RULE(DevicePropVarRule)

// Rule for enums constants.
void EnumConstantRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(hasType(enumDecl(hasAnyName(
                                "cudaComputeMode", "cudaMemcpyKind",
                                "cudaMemoryAdvise", "cudaDeviceAttr"))))))
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
      if (EnumTypeName == "cudaMemoryAdvise") {
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
    // TODO report migration error
    return;
  }

  emplaceTransformation(new ReplaceStmt(E, Search->second));
  requestHelperFeatureForEnumNames(EnumName, E);
}

REGISTER_RULE(EnumConstantRule)

void ErrorConstantsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(hasType(enumDecl(anyOf(
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
    SourceLocation OperatorLoc;
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

REGISTER_RULE(ErrorConstantsRule)

void ManualMigrateEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName("NCCL_.*"))))
                    .bind("NCCLConstants"),
                this);
  MF.addMatcher(declRefExpr(to(enumConstantDecl(matchesName("CUDNN_.*"))))
                    .bind("CUDNNConstants"),
                this);
}

void ManualMigrateEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "NCCLConstants")) {
    auto *ECD = cast<EnumConstantDecl>(DE->getDecl());
    std::string FilePath = DpctGlobalInfo::getSourceManager()
                               .getFilename(ECD->getBeginLoc())
                               .str();
    if (DpctGlobalInfo::isInRoot(FilePath)) {
      return;
    }
    report(dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
               DE->getBeginLoc()),
           Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
           "Intel(R) oneAPI Collective Communications Library");
  } else if (const DeclRefExpr *DE =
                 getNodeAsType<DeclRefExpr>(Result, "CUDNNConstants")) {
    auto *ECD = cast<EnumConstantDecl>(DE->getDecl());
    std::string FilePath = DpctGlobalInfo::getSourceManager()
                               .getFilename(ECD->getBeginLoc())
                               .str();
    if (DpctGlobalInfo::isInRoot(FilePath)) {
      return;
    }
    report(dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
               DE->getBeginLoc()),
           Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
           "Intel(R) oneAPI Deep Neural Network Library (oneDNN)");
  }
}

REGISTER_RULE(ManualMigrateEnumsRule)

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

  MF.addMatcher(declRefExpr(to(enumConstantDecl(
                                matchesName("(CUFFT_R2C|CUFFT_C2R|CUFFT_C2C|"
                                            "CUFFT_D2Z|CUFFT_Z2D|CUFFT_Z2Z)"))))
                    .bind("FFTTypeConstants"),
                this);
}

void FFTEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "FFTConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));
    return;
  }

  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "FFTTypeConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));

    auto Value = EC->getInitVal().getExtValue();
    DpctGlobalInfo::getFFTTypeSet().insert(getFFTTypeFromValue(Value));
    DpctGlobalInfo::getPrecAndDomPairSet().insert(
        getPrecAndDomainStrFromValue(Value));
    return;
  }
}

REGISTER_RULE(FFTEnumsRule)

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

REGISTER_RULE(BLASEnumsRule)

// Rule for RANDOM enums.
// Migrate RANDOM status values to corresponding int values
void RandomEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName("CURAND_STATUS.*"))))
          .bind("RANDOMStatusConstants"),
      this);
}

void RandomEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "RANDOMStatusConstants")) {
    auto *EC = cast<EnumConstantDecl>(DE->getDecl());
    emplaceTransformation(new ReplaceStmt(DE, toString(EC->getInitVal(), 10)));
  }
}

REGISTER_RULE(RandomEnumsRule)

// Rule for spBLAS enums.
// Migrate spBLAS status values to corresponding int values
// Other spBLAS named values are migrated to corresponding named values
void SPBLASEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName(
                      "(CUSPARSE_STATUS.*)|("
                      "CUSPARSE_POINTER_MODE.*)|(CUSPARSE_MATRIX_TYPE.*)"))))
          .bind("SPBLASStatusConstants"),
      this);
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName(
                      "(CUSPARSE_OPERATION.*)|(CUSPARSE_FILL_MODE.*)|(CUSPARSE_"
                      "DIAG_TYPE.*)|(CUSPARSE_INDEX_BASE.*)"))))
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

REGISTER_RULE(SPBLASEnumsRule)

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
  requestFeature(HelperFeatureEnum::BlasUtils_get_value, Expr);
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
        /*helper*/
        "cusparseCreateMatDescr", "cusparseDestroyMatDescr",
        "cusparseSetMatType", "cusparseGetMatType", "cusparseSetMatIndexBase",
        "cusparseGetMatIndexBase", "cusparseSetMatDiagType",
        "cusparseGetMatDiagType", "cusparseSetMatFillMode",
        "cusparseGetMatFillMode", "cusparseCreateSolveAnalysisInfo",
        "cusparseDestroySolveAnalysisInfo",
        /*level 2*/
        "cusparseScsrmv", "cusparseDcsrmv", "cusparseCcsrmv", "cusparseZcsrmv",
        "cusparseScsrsv_analysis", "cusparseDcsrsv_analysis",
        "cusparseCcsrsv_analysis", "cusparseZcsrsv_analysis",
        /*level 3*/
        "cusparseScsrmm", "cusparseDcsrmm", "cusparseCcsrmm", "cusparseZcsrmm");
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
  auto SL = SM.getExpansionLoc(CE->getBeginLoc());
  std::string Key =
      SM.getFilename(SL).str() + std::to_string(SM.getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  StringRef FuncNameRef(FuncName);

  LibraryMigrationFlags Flags;
  LibraryMigrationStrings ReplaceStrs;
  LibraryMigrationLocations Locations;
  initVars(CE, nullptr, nullptr, Flags, ReplaceStrs, Locations);
  Flags.IsAssigned = IsAssigned;

  std::string Msg = "the function call is redundant in DPC++.";
  if (FuncName == "cusparseCreate" || FuncName == "cusparseDestroy" ||
      FuncName == "cusparseSetStream" || FuncName == "cusparseGetStream") {
    Flags.NeedUseLambda = false;
    if (FuncName == "cusparseCreate") {
      std::string LHS = getDrefName(CE->getArg(0));
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
      ReplaceStrs.Repl =
          LHS + " = &{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
    } else if (FuncName == "cusparseDestroy") {
      dpct::ExprAnalysis EA(CE->getArg(0));
      ReplaceStrs.Repl = EA.getReplacedString() + " = nullptr";
    } else if (FuncName == "cusparseSetStream") {
      dpct::ExprAnalysis EA0(CE->getArg(0));
      dpct::ExprAnalysis EA1(CE->getArg(1));
      ReplaceStrs.Repl =
          EA0.getReplacedString() + " = " + EA1.getReplacedString();
    } else if (FuncName == "cusparseGetStream") {
      dpct::ExprAnalysis EA0(CE->getArg(0));
      std::string LHS = getDrefName(CE->getArg(1));
      ReplaceStrs.Repl = LHS + " = " + EA0.getReplacedString();
    }
  } else if (FuncName == "cusparseCreateMatDescr") {
    Flags.NeedUseLambda = false;
    std::string LHS = getDrefName(CE->getArg(0));
    ReplaceStrs.Repl = LHS + " = oneapi::mkl::index_base::zero";
  } else if (FuncName == "cusparseDestroyMatDescr" ||
             FuncName == "cusparseGetPointerMode" ||
             FuncName == "cusparseSetPointerMode" ||
             FuncName == "cusparseScsrsv_analysis" ||
             FuncName == "cusparseDcsrsv_analysis" ||
             FuncName == "cusparseCcsrsv_analysis" ||
             FuncName == "cusparseZcsrsv_analysis" ||
             FuncName == "cusparseCreateSolveAnalysisInfo" ||
             FuncName == "cusparseDestroySolveAnalysisInfo" ||
             FuncName == "cusparseSetMatType" ||
             FuncName == "cusparseGetMatType" ||
             FuncName == "cusparseSetMatDiagType" ||
             FuncName == "cusparseGetMatDiagType" ||
             FuncName == "cusparseSetMatFillMode" ||
             FuncName == "cusparseGetMatFillMode") {
    if (FuncName == "cusparseSetMatType") {
      Expr::EvalResult ER;
      if (CE->getArg(1)->EvaluateAsInt(ER, *Result.Context)) {
        int64_t Value = ER.Val.getInt().getExtValue();
        if (Value != 0) {
          DpctGlobalInfo::setSpBLASUnsupportedMatrixTypeFlag(true);
        }
      } else {
        DpctGlobalInfo::setSpBLASUnsupportedMatrixTypeFlag(true);
      }
    }

    if (IsAssigned) {
      report(Locations.PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED_0, false,
             FuncName, Msg);
      if (FuncName == "cusparseGetMatDiagType")
        emplaceTransformation(
            new ReplaceStmt(CE, false, "(oneapi::mkl::diag)0"));
      else if (FuncName == "cusparseGetMatFillMode")
        emplaceTransformation(
            new ReplaceStmt(CE, false, "(oneapi::mkl::uplo)0"));
      else
        emplaceTransformation(new ReplaceStmt(CE, false, "0"));
    } else {
      report(Locations.PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED, false,
             FuncName, Msg);
      emplaceTransformation(new ReplaceStmt(CE, false, ""));
    }
    return;
  } else if (FuncName == "cusparseSetMatIndexBase" ||
             FuncName == "cusparseGetMatIndexBase") {
    Flags.NeedUseLambda = false;
    ExprAnalysis EA0(CE->getArg(0));
    bool IsSet = FuncNameRef.startswith("cusparseSet");
    ExprAnalysis EA1;
    if (IsSet) {
      ReplaceStrs.Repl = EA0.getReplacedString() + " = ";
      EA1.analyze(CE->getArg(1));
      Expr::EvalResult ER;
      if (CE->getArg(1)->EvaluateAsInt(ER, *Result.Context)) {
        int64_t Value = ER.Val.getInt().getExtValue();
        if (Value == 0)
          ReplaceStrs.Repl = ReplaceStrs.Repl + "oneapi::mkl::index_base::zero";
        else
          ReplaceStrs.Repl = ReplaceStrs.Repl + "oneapi::mkl::index_base::one";
      } else {
        ReplaceStrs.Repl = ReplaceStrs.Repl + EA1.getReplacedString();
      }
    } else {
      ReplaceStrs.Repl = EA0.getReplacedString();
    }

    // Get API do not return status, so return directly.
    if (IsAssigned && IsSet) {
      insertAroundStmt(CE, "(", ", 0)");
      report(Locations.PrefixInsertLoc, Diagnostics::NOERROR_RETURN_COMMA_OP,
             true);
    }
    emplaceTransformation(new ReplaceStmt(CE, true, ReplaceStrs.Repl));
    return;
  } else if (FuncName == "cusparseScsrmv" || FuncName == "cusparseDcsrmv" ||
             FuncName == "cusparseCcsrmv" || FuncName == "cusparseZcsrmv") {
    std::string BufferType;
    if (FuncName == "cusparseScsrmv")
      BufferType = "float";
    else if (FuncName == "cusparseDcsrmv")
      BufferType = "double";
    else if (FuncName == "cusparseCcsrmv")
      BufferType = "std::complex<float>";
    else
      BufferType = "std::complex<double>";
    int ArgNum = CE->getNumArgs();
    std::vector<std::string> CallExprArguReplVec;
    for (int i = 0; i < ArgNum; ++i) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(i));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    std::string CSRValA, CSRRowPtrA, CSRColIndA, X, Y;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      auto ProcessBuffer = [&](const Expr *E, const std::string TypeStr) {
        std::string Decl;
        requestFeature(HelperFeatureEnum::Memory_get_buffer_T, E);
        std::string BufferName =
            getBufferNameAndDeclStr(E, TypeStr, ReplaceStrs.IndentStr, Decl);
        ReplaceStrs.PrefixInsertStr = ReplaceStrs.PrefixInsertStr + Decl;
        return BufferName;
      };
      CSRValA = ProcessBuffer(CE->getArg(7), BufferType);
      CSRRowPtrA = ProcessBuffer(CE->getArg(8), "int");
      CSRColIndA = ProcessBuffer(CE->getArg(9), "int");
      X = ProcessBuffer(CE->getArg(10), BufferType);
      Y = ProcessBuffer(CE->getArg(12), BufferType);
    } else {
      CSRValA = CallExprArguReplVec[7];
      CSRRowPtrA = CallExprArguReplVec[8];
      CSRColIndA = CallExprArguReplVec[9];
      X = CallExprArguReplVec[10];
      Y = CallExprArguReplVec[12];
    }

    std::string MatrixHandleName =
        "mat_handle_ct" +
        std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
    ReplaceStrs.PrefixInsertStr =
        ReplaceStrs.PrefixInsertStr + "oneapi::mkl::sparse::matrix_handle_t " +
        MatrixHandleName + ";" + getNL() + ReplaceStrs.IndentStr;
    ReplaceStrs.PrefixInsertStr = ReplaceStrs.PrefixInsertStr +
                                  "oneapi::mkl::sparse::init_matrix_handle(&" +
                                  MatrixHandleName + ");" + getNL() +
                                  ReplaceStrs.IndentStr;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      ReplaceStrs.PrefixInsertStr =
          ReplaceStrs.PrefixInsertStr + "oneapi::mkl::sparse::set_csr_data(" +
          MatrixHandleName + ", " + CallExprArguReplVec[2] + ", " +
          CallExprArguReplVec[3] + ", " + CallExprArguReplVec[6] + ", " +
          CSRRowPtrA + ", " + CSRColIndA + ", " + CSRValA + ");" + getNL() +
          ReplaceStrs.IndentStr;
    } else {
      if (FuncName == "cusparseScsrmv" || FuncName == "cusparseDcsrmv")
        ReplaceStrs.PrefixInsertStr =
            ReplaceStrs.PrefixInsertStr + "oneapi::mkl::sparse::set_csr_data(" +
            MatrixHandleName + ", " + CallExprArguReplVec[2] + ", " +
            CallExprArguReplVec[3] + ", " + CallExprArguReplVec[6] +
            ", const_cast<int*>(" + CSRRowPtrA + "), const_cast<int*>(" +
            CSRColIndA + "), const_cast<" + BufferType + "*>(" + CSRValA +
            "));" + getNL() + ReplaceStrs.IndentStr;
      else
        ReplaceStrs.PrefixInsertStr =
            ReplaceStrs.PrefixInsertStr + "oneapi::mkl::sparse::set_csr_data(" +
            MatrixHandleName + ", " + CallExprArguReplVec[2] + ", " +
            CallExprArguReplVec[3] + ", " + CallExprArguReplVec[6] +
            ", const_cast<int*>(" + CSRRowPtrA + "), const_cast<int*>(" +
            CSRColIndA + "), (" + BufferType + "*)" + CSRValA + ");" + getNL() +
            ReplaceStrs.IndentStr;
    }
    ReplaceStrs.SuffixInsertStr =
        ReplaceStrs.SuffixInsertStr + getNL() + ReplaceStrs.IndentStr +
        "oneapi::mkl::sparse::release_matrix_handle(&" + MatrixHandleName +
        ");";

    std::string TransStr;
    Expr::EvalResult ER;
    if (CE->getArg(1)->EvaluateAsInt(ER, *Result.Context)) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 0) {
        TransStr = "oneapi::mkl::transpose::nontrans";
      } else if (Value == 1) {
        TransStr = "oneapi::mkl::transpose::trans";
      } else {
        TransStr = "oneapi::mkl::transpose::conjtrans";
      }
    } else {
      const CStyleCastExpr *CSCE = nullptr;
      if (CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(1))) {
        ExprAnalysis EA(CSCE->getSubExpr());
        TransStr = MapNames::getDpctNamespace() + "get_transpose(" +
                   EA.getReplacedString() + ")";
        requestFeature(HelperFeatureEnum::BlasUtils_get_transpose, CE);

      } else {
        TransStr = CallExprArguReplVec[1];
      }
    }

    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      ReplaceStrs.Repl = "oneapi::mkl::sparse::gemv(*" +
                         CallExprArguReplVec[0] + ", " + TransStr + ", " +
                         getValueStr(CE->getArg(5), CallExprArguReplVec[5],
                                        CallExprArguReplVec[0], BufferType) +
                         ", " + MatrixHandleName + ", " + X + ", " +
                         getValueStr(CE->getArg(11), CallExprArguReplVec[11],
                                        CallExprArguReplVec[0], BufferType) +
                         ", " + Y + ")";
    } else {
      if (FuncName == "cusparseScsrmv" || FuncName == "cusparseDcsrmv")
        ReplaceStrs.Repl =
            "oneapi::mkl::sparse::gemv(*" + CallExprArguReplVec[0] + ", " +
            TransStr + ", " +
            getValueStr(CE->getArg(5), CallExprArguReplVec[5],
                           CallExprArguReplVec[0], BufferType) +
            ", " + MatrixHandleName + ", const_cast<" + BufferType + "*>(" + X +
            "), " +
            getValueStr(CE->getArg(11), CallExprArguReplVec[11],
                           CallExprArguReplVec[0], BufferType) +
            ", " + Y + ")";
      else
        ReplaceStrs.Repl =
            "oneapi::mkl::sparse::gemv(*" + CallExprArguReplVec[0] + ", " +
            TransStr + ", " +
            getValueStr(CE->getArg(5), CallExprArguReplVec[5],
                           CallExprArguReplVec[0], BufferType) +
            ", " + MatrixHandleName + ", (" + BufferType + "*)" + X + ", " +
            getValueStr(CE->getArg(11), CallExprArguReplVec[11],
                           CallExprArguReplVec[0], BufferType) +
            ", (" + BufferType + "*)" + Y + ")";
    }
  } else if (FuncName == "cusparseScsrmm" || FuncName == "cusparseDcsrmm" ||
             FuncName == "cusparseCcsrmm" || FuncName == "cusparseZcsrmm") {
    std::string BufferType;
    if (FuncName == "cusparseScsrmm")
      BufferType = "float";
    else if (FuncName == "cusparseDcsrmm")
      BufferType = "double";
    else if (FuncName == "cusparseCcsrmm")
      BufferType = "std::complex<float>";
    else
      BufferType = "std::complex<double>";
    int ArgNum = CE->getNumArgs();
    std::vector<std::string> CallExprArguReplVec;
    for (int i = 0; i < ArgNum; ++i) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(i));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    std::string CSRValA, CSRRowPtrA, CSRColIndA, B, C;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      auto ProcessBuffer = [&](const Expr *E, const std::string TypeStr) {
        std::string Decl;
        requestFeature(HelperFeatureEnum::Memory_get_buffer_T, E);
        std::string BufferName =
            getBufferNameAndDeclStr(E, TypeStr, ReplaceStrs.IndentStr, Decl);
        ReplaceStrs.PrefixInsertStr = ReplaceStrs.PrefixInsertStr + Decl;
        return BufferName;
      };
      CSRValA = ProcessBuffer(CE->getArg(8), BufferType);
      CSRRowPtrA = ProcessBuffer(CE->getArg(9), "int");
      CSRColIndA = ProcessBuffer(CE->getArg(10), "int");
      B = ProcessBuffer(CE->getArg(11), BufferType);
      C = ProcessBuffer(CE->getArg(14), BufferType);
    } else {
      CSRValA = CallExprArguReplVec[8];
      CSRRowPtrA = CallExprArguReplVec[9];
      CSRColIndA = CallExprArguReplVec[10];
      B = CallExprArguReplVec[11];
      C = CallExprArguReplVec[14];
    }

    std::string MatrixHandleName =
        "mat_handle_ct" +
        std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
    ReplaceStrs.PrefixInsertStr =
        ReplaceStrs.PrefixInsertStr + "oneapi::mkl::sparse::matrix_handle_t " +
        MatrixHandleName + ";" + getNL() + ReplaceStrs.IndentStr;
    ReplaceStrs.PrefixInsertStr = ReplaceStrs.PrefixInsertStr +
                                  "oneapi::mkl::sparse::init_matrix_handle(&" +
                                  MatrixHandleName + ");" + getNL() +
                                  ReplaceStrs.IndentStr;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      ReplaceStrs.PrefixInsertStr =
          ReplaceStrs.PrefixInsertStr + "oneapi::mkl::sparse::set_csr_data(" +
          MatrixHandleName + ", " + CallExprArguReplVec[2] + ", " +
          CallExprArguReplVec[4] + ", " + CallExprArguReplVec[7] + ", " +
          CSRRowPtrA + ", " + CSRColIndA + ", " + CSRValA + ");" + getNL() +
          ReplaceStrs.IndentStr;
    } else {
      if (FuncName == "cusparseScsrmm" || FuncName == "cusparseDcsrmm")
        ReplaceStrs.PrefixInsertStr =
            ReplaceStrs.PrefixInsertStr + "oneapi::mkl::sparse::set_csr_data(" +
            MatrixHandleName + ", " + CallExprArguReplVec[2] + ", " +
            CallExprArguReplVec[4] + ", " + CallExprArguReplVec[7] +
            ", const_cast<int*>(" + CSRRowPtrA + "), const_cast<int*>(" +
            CSRColIndA + "), const_cast<" + BufferType + "*>(" + CSRValA +
            "));" + getNL() + ReplaceStrs.IndentStr;
      else
        ReplaceStrs.PrefixInsertStr =
            ReplaceStrs.PrefixInsertStr + "oneapi::mkl::sparse::set_csr_data(" +
            MatrixHandleName + ", " + CallExprArguReplVec[2] + ", " +
            CallExprArguReplVec[4] + ", " + CallExprArguReplVec[7] +
            ", const_cast<int*>(" + CSRRowPtrA + "), const_cast<int*>(" +
            CSRColIndA + "), (" + BufferType + "*)" + CSRValA + ");" + getNL() +
            ReplaceStrs.IndentStr;
    }
    ReplaceStrs.SuffixInsertStr =
        ReplaceStrs.SuffixInsertStr + getNL() + ReplaceStrs.IndentStr +
        "oneapi::mkl::sparse::release_matrix_handle(&" + MatrixHandleName +
        ");";

    std::string TransStr;
    Expr::EvalResult ER;
    if (CE->getArg(1)->EvaluateAsInt(ER, *Result.Context)) {
      int64_t Value = ER.Val.getInt().getExtValue();
      if (Value == 0) {
        TransStr = "oneapi::mkl::transpose::nontrans";
      } else if (Value == 1) {
        TransStr = "oneapi::mkl::transpose::trans";
      } else {
        TransStr = "oneapi::mkl::transpose::conjtrans";
      }
    } else {
      const CStyleCastExpr *CSCE = nullptr;
      if (CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(1))) {
        ExprAnalysis EA(CSCE->getSubExpr());
        TransStr = MapNames::getDpctNamespace() + "get_transpose(" +
                   EA.getReplacedString() + ")";
        requestFeature(HelperFeatureEnum::BlasUtils_get_transpose, CE);
      } else {
        TransStr = CallExprArguReplVec[1];
      }
    }

    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      ReplaceStrs.Repl = "oneapi::mkl::sparse::gemm(*" +
                         CallExprArguReplVec[0] + ", " + TransStr + ", " +
                         getValueStr(CE->getArg(6), CallExprArguReplVec[6],
                                        CallExprArguReplVec[0], BufferType) +
                         ", " + MatrixHandleName + ", " + B + ", " +
                         CallExprArguReplVec[3] + ", " +
                         CallExprArguReplVec[12] + ", " +
                         getValueStr(CE->getArg(13), CallExprArguReplVec[13],
                                        CallExprArguReplVec[0], BufferType) +
                         ", " + C + ", " + CallExprArguReplVec[15] + ")";
    } else {
      if (FuncName == "cusparseScsrmm" || FuncName == "cusparseDcsrmm")
        ReplaceStrs.Repl =
            "oneapi::mkl::sparse::gemm(*" + CallExprArguReplVec[0] + ", " +
            TransStr + ", " +
            getValueStr(CE->getArg(6), CallExprArguReplVec[6],
                           CallExprArguReplVec[0], BufferType) +
            ", " + MatrixHandleName + ", const_cast<" + BufferType + "*>(" + B +
            "), " + CallExprArguReplVec[3] + ", " + CallExprArguReplVec[12] +
            ", " +
            getValueStr(CE->getArg(13), CallExprArguReplVec[13],
                           CallExprArguReplVec[0], BufferType) +
            ", " + C + ", " + CallExprArguReplVec[15] + ")";
      else
        ReplaceStrs.Repl =
            "oneapi::mkl::sparse::gemm(*" + CallExprArguReplVec[0] + ", " +
            TransStr + ", " +
            getValueStr(CE->getArg(6), CallExprArguReplVec[6],
                           CallExprArguReplVec[0], BufferType) +
            ", " + MatrixHandleName + ", (" + BufferType + "*)" + B + ", " +
            CallExprArguReplVec[3] + ", " + CallExprArguReplVec[12] + ", " +
            getValueStr(CE->getArg(13), CallExprArguReplVec[13],
                           CallExprArguReplVec[0], BufferType) +
            ", (" + BufferType + "*)" + C + ", " + CallExprArguReplVec[15] +
            ")";
    }
  }

  if (FuncNameRef.endswith("csrmv") || FuncNameRef.endswith("csrmm")) {
    if (Flags.NeedUseLambda && Flags.CanAvoidUsingLambda && !Flags.IsMacroArg) {
      DpctGlobalInfo::getInstance().insertSpBLASWarningLocOffset(
          Locations.OuterInsertLoc);
    } else {
      DpctGlobalInfo::getInstance().insertSpBLASWarningLocOffset(
          Locations.PrefixInsertLoc);
    }
  }

  addReplacementForLibraryAPI(Flags, ReplaceStrs, Locations, FuncName, CE);
}

REGISTER_RULE(SPBLASFunctionCallRule)

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
        "curandGenerateUniformDouble", "curandSetStream");
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
  auto SL = SM.getExpansionLoc(CE->getBeginLoc());
  std::string Key =
      SM.getFilename(SL).str() + std::to_string(SM.getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // TODO: For case like:
  //  #define CHECK_STATUS(x) fun(c)
  //  CHECK_STATUS(anAPICall());
  // Below code can distinguish this kind of function like macro, need refine to
  // cover more cases.
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
  auto SR = getScopeInsertRange(CE, FuncNameBegin, FuncCallEnd);
  SourceLocation PrefixInsertLoc = SR.getBegin(), SuffixInsertLoc = SR.getEnd();

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

  if (NeedUseLambda || IsMacroArg || IsInReturnStmt) {
    NeedUseLambda = true;
    SourceRange SR = getFunctionRange(CE);
    PrefixInsertLoc = SR.getBegin();
    SuffixInsertLoc = SR.getEnd();
    if (IsInReturnStmt) {
      CanAvoidUsingLambda = true;
      OriginStmtType = "return";
    }
  }

  std::string IndentStr = getIndent(PrefixInsertLoc, SM).str();
  std::string PrefixInsertStr;

  std::string Msg = "the function call is redundant in DPC++.";
  if (FuncName == "curandSetPseudoRandomGeneratorSeed" ||
      FuncName == "curandSetQuasiRandomGeneratorDimensions") {
    if (IsAssigned) {
      report(PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED_0, false, FuncName,
             Msg);
      emplaceTransformation(new ReplaceStmt(CE, false, "0"));
    } else {
      report(PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED, false, FuncName,
             Msg);
      emplaceTransformation(new ReplaceStmt(CE, false, ""));
    }
  }

  if (FuncName == "curandCreateGenerator") {
    bool IsDuplicated = true;
    auto REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    if (!REInfo) {
      DpctGlobalInfo::getInstance().insertRandomEngine(CE->getArg(0));
      REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
      IsDuplicated = false;
    }

    std::string EnumStr = ExprAnalysis::ref(CE->getArg(1));
    if (MapNames::RandomEngineTypeMap.find(EnumStr) ==
        MapNames::RandomEngineTypeMap.end()) {
      report(SM.getExpansionLoc(REInfo->getDeclaratorDeclTypeBeginLoc()),
             Diagnostics::UNMIGRATED_TYPE, false, "curandGenerator_t",
             "the migration depends on the second argument of "
             "curandCreateGenerator");
      report(PrefixInsertLoc, Diagnostics::NOT_SUPPORTED_PARAMETER, false,
             FuncName, "parameter " + EnumStr + " is unsupported");
      return;
    }
    REInfo->setUnsupportEngineFlag(false);

    if (EnumStr == "CURAND_RNG_PSEUDO_XORWOW" ||
        EnumStr == "CURAND_RNG_QUASI_SOBOL64" ||
        EnumStr == "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64") {
      report(SM.getExpansionLoc(REInfo->getDeclaratorDeclTypeBeginLoc()),
             Diagnostics::DIFFERENT_GENERATOR, false);
    } else if (EnumStr == "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32") {
      report(SM.getExpansionLoc(REInfo->getDeclaratorDeclTypeBeginLoc()),
             Diagnostics::DIFFERENT_BASIC_GENERATOR, false);
    }

    REInfo->setGeneratorName(getDrefName(CE->getArg(0)));

    std::string EngineType =
        MapNames::RandomEngineTypeMap.find(EnumStr)->second;
    if (IsDuplicated && (REInfo->getEngineType() != EngineType) &&
        REInfo->getIsRealCreate()) {
      REInfo->setEngineTypeReplacement("");
    } else {
      REInfo->setEngineTypeReplacement(EngineType);
    }
    REInfo->setIsRealCreate(true);
    DpctGlobalInfo::getHostRNGEngineTypeSet().insert(
        MapNames::RandomEngineTypeMap.find(EnumStr)->second);

    if (EnumStr == "CURAND_RNG_QUASI_DEFAULT" ||
        EnumStr == "CURAND_RNG_QUASI_SOBOL32" ||
        EnumStr == "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32" ||
        EnumStr == "CURAND_RNG_QUASI_SOBOL64" ||
        EnumStr == "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64")
      REInfo->setQuasiEngineFlag();

    REInfo->setCreateAPIInfo(FuncNameBegin, FuncCallEnd);

    if (isPlaceholderIdxDuplicated(CE->getArg(0)))
      return;
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    REInfo->setQueueStr("{{NEEDREPLACEQ" + std::to_string(Index) + "}}");
    buildTempVariableMap(Index, CE->getArg(0),
                         HelperFuncType::HFT_DefaultQueue);

    if (IsAssigned) {
      REInfo->setAssigned();
    }
  } else if (FuncName == "curandDestroyGenerator") {
    if (IsAssigned) {
      report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      insertAroundStmt(CE, "(", ", 0)");
    }
    emplaceTransformation(new ReplaceStmt(
        CE, false, ExprAnalysis::ref(CE->getArg(0)) + ".reset()"));
  } else if (FuncName == "curandSetPseudoRandomGeneratorSeed") {
    auto REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    if (!REInfo) {
      DpctGlobalInfo::getInstance().insertRandomEngine(CE->getArg(0));
      REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    }
    REInfo->setSeedExpr(CE->getArg(1));
  } else if (FuncName == "curandSetQuasiRandomGeneratorDimensions") {
    auto REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    if (!REInfo) {
      DpctGlobalInfo::getInstance().insertRandomEngine(CE->getArg(0));
      REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    }
    REInfo->setDimExpr(CE->getArg(1));
  } else if (MapNames::RandomGenerateFuncReplInfoMap.find(FuncName) !=
             MapNames::RandomGenerateFuncReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::RandomGenerateFuncReplInfoMap.find(FuncName);
    MapNames::RandomGenerateFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string BufferDecl;
    std::string BufferName;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      requestFeature(HelperFeatureEnum::Memory_get_buffer_T, CE);
      BufferName = getBufferNameAndDeclStr(
          CE->getArg(1), ReplInfo.BufferTypeInfo, IndentStr, BufferDecl);
    }

    SourceLocation DistrInsertLoc =
        SM.getExpansionLoc(CS->body_front()->getBeginLoc());
    std::string DistrIndentStr = getIndent(DistrInsertLoc, SM).str();
    std::string DistrName;
    if (FuncName == "curandGenerateLogNormal" ||
        FuncName == "curandGenerateLogNormalDouble") {
      ExprAnalysis EMean(CE->getArg(3)), EDev(CE->getArg(4));
      std::string DistrArg = EMean.getReplacedString() + ", " +
                             EDev.getReplacedString() + ", 0.0, 1.0";
      DistrName = DpctGlobalInfo::getInstance().insertHostRandomDistrInfo(
          DistrInsertLoc, ReplInfo.DistributeType, ReplInfo.ValueType, DistrArg,
          DistrIndentStr);
    } else if (FuncName == "curandGenerateNormal" ||
               FuncName == "curandGenerateNormalDouble") {
      ExprAnalysis EMean(CE->getArg(3)), EDev(CE->getArg(4));
      std::string DistrArg =
          EMean.getReplacedString() + ", " + EDev.getReplacedString();
      DistrName = DpctGlobalInfo::getInstance().insertHostRandomDistrInfo(
          DistrInsertLoc, ReplInfo.DistributeType, ReplInfo.ValueType, DistrArg,
          DistrIndentStr);
    } else if (FuncName == "curandGeneratePoisson") {
      ExprAnalysis ELambda(CE->getArg(3));
      DistrName = DpctGlobalInfo::getInstance().insertHostRandomDistrInfo(
          DistrInsertLoc, ReplInfo.DistributeType, ReplInfo.ValueType,
          ELambda.getReplacedString(), DistrIndentStr);
    } else {
      DistrName = DpctGlobalInfo::getInstance().insertHostRandomDistrInfo(
          DistrInsertLoc, ReplInfo.DistributeType, ReplInfo.ValueType, "",
          DistrIndentStr);
    }
    std::string Data;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
      auto TypePtr = CE->getArg(1)->getType().getTypePtr();
      if (!TypePtr || !TypePtr->isPointerType()) {
        Data =
            "(" + ReplInfo.ValueType + "*)" + ExprAnalysis::ref(CE->getArg(1));
      } else if (TypePtr->getPointeeType().getAsString() ==
                 ReplInfo.ValueType) {
        Data = ExprAnalysis::ref(CE->getArg(1));
      } else {
        Data =
            "(" + ReplInfo.ValueType + "*)" + ExprAnalysis::ref(CE->getArg(1));
      }
    } else {
      PrefixInsertStr = BufferDecl;
      Data = BufferName;
    }
    ArgumentAnalysis AA;
    AA.setCallSpelling(CE);
    AA.analyze(CE->getArg(2));
    auto ArgStr =
        AA.getRewritePrefix() + AA.getRewriteString() + AA.getRewritePostfix();

    std::string ReplStr;
    ReplStr = "oneapi::mkl::rng::generate(" + DistrName + ", " +
              getDrefName(CE->getArg(0)) + ", " + ArgStr + ", " + Data + ")";

    if (NeedUseLambda) {
      if (PrefixInsertStr.empty()) {
        // If there is one API call in the migrted code, it is unnecessary to
        // use a lambda expression
        NeedUseLambda = false;
      }
    }

    if (NeedUseLambda) {
      if (CanAvoidUsingLambda) {
        std::string InsertStr;
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
            !CanAvoidBrace)
          InsertStr = std::string("{") + getNL() + IndentStr + PrefixInsertStr +
                      ReplStr + ";" + getNL() + IndentStr + "}" + getNL() +
                      IndentStr;
        else
          InsertStr = PrefixInsertStr + ReplStr + ";" + getNL() + IndentStr;

        auto IT = new InsertText(OuterInsertLoc, std::move(InsertStr));
        IT->setBlockLevelFormatFlag();
        emplaceTransformation(std::move(IT));
        report(OuterInsertLoc, Diagnostics::CODE_LOGIC_CHANGED, true,
               OriginStmtType == "if" ? "an " + OriginStmtType
                                      : "a " + OriginStmtType);
        emplaceTransformation(new ReplaceStmt(CE, "0"));
      } else {
        if (IsAssigned) {
          report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_LAMBDA, false);
          insertAroundRange(
              PrefixInsertLoc, SuffixInsertLoc,
              std::string("[&](){") + getNL() + IndentStr + PrefixInsertStr,
              std::string(";") + getNL() + IndentStr + "return 0;" + getNL() +
                  IndentStr + std::string("}()"),
              true);
        } else {
          insertAroundRange(
              PrefixInsertLoc, SuffixInsertLoc,
              std::string("[&](){") + getNL() + IndentStr + PrefixInsertStr,
              std::string(";") + getNL() + IndentStr + std::string("}()"),
              true);
        }
        emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
      }
    } else {
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
          !CanAvoidBrace) {
        if (!PrefixInsertStr.empty()) {
          insertAroundRange(PrefixInsertLoc, SuffixInsertLoc,
                            std::string("{") + getNL() + IndentStr +
                                PrefixInsertStr,
                            getNL() + IndentStr + std::string("}"), true);
        }
      } else {
        emplaceTransformation(
            new InsertText(PrefixInsertLoc, std::move(PrefixInsertStr)));
      }
      if (IsAssigned) {
        insertAroundStmt(CE, "(", ", 0)");
        report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_COMMA_OP, true);
      }
      emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    }
  } else if (FuncName == "curandSetGeneratorOffset") {
    if (IsAssigned) {
      insertAroundStmt(CE, "(", ", 0)");
      report(PrefixInsertLoc, Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    std::string Repl =
        "oneapi::mkl::rng::skip_ahead(" + getDrefName(CE->getArg(0)) + ", ";
    ExprAnalysis EO;
    EO.analyze(CE->getArg(1));
    Repl = Repl + EO.getReplacedString() + ")";
    emplaceTransformation(new ReplaceStmt(CE, std::move(Repl)));
  } else if (FuncName == "curandSetStream") {
    auto REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
    if (!REInfo) {
      // Cannot find matched curandCreateGenerator, construct a fake
      // RandomEngineInfo
      DpctGlobalInfo::getInstance().insertRandomEngine(CE->getArg(0));
      REInfo = DpctGlobalInfo::getInstance().findRandomEngine(CE->getArg(0));
      REInfo->setEngineTypeReplacement("");
      REInfo->setIsRealCreate(false);
    }
    REInfo->setCreateAPIInfo(FuncNameBegin, FuncCallEnd,
                             getDrefName(CE->getArg(1)));
    if (IsAssigned) {
      REInfo->setAssigned();
    }
  }
}

REGISTER_RULE(RandomFunctionCallRule)

// Rule for device Random function calls.
void DeviceRandomFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName("curand_init", "curand_normal2", "curand_normal2_double",
                      "curand_normal_double", "curand_uniform");
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

    std::string Arg0Type = CE->getArg(0)->getType().getAsString();
    std::string Arg1Type = CE->getArg(1)->getType().getAsString();
    std::string Arg2Type = CE->getArg(2)->getType().getAsString();
    std::string DRefArg3Type;

    if (Arg0Type == "unsigned long long" && Arg1Type == "unsigned long long" &&
        Arg2Type == "unsigned long long" &&
        CE->getArg(3)->getType()->isPointerType()) {
      DRefArg3Type = CE->getArg(3)->getType()->getPointeeType().getAsString();
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

    ExprAnalysis EARNGSeed(CE->getArg(0));
    ExprAnalysis EARNGSubseq(CE->getArg(1));
    ExprAnalysis EARNGOffset(CE->getArg(2));

    auto IsLiteral = [=](const Expr *E) {
      if (dyn_cast<IntegerLiteral>(E->IgnoreCasts()) ||
          dyn_cast<FloatingLiteral>(E->IgnoreCasts()) ||
          dyn_cast<FixedPointLiteral>(E->IgnoreCasts())) {
        return true;
      }
      return false;
    };

    DpctGlobalInfo::getInstance().insertDeviceRandomInitAPIInfo(
        FuncNameBegin, FuncCallLength,
        MapNames::DeviceRandomGeneratorTypeMap.find(DRefArg3Type)->second,
        DRefArg3Type, EARNGSeed.getReplacedString(),
        EARNGSubseq.getReplacedString(), IsLiteral(CE->getArg(1)),
        EARNGOffset.getReplacedString(), IsLiteral(CE->getArg(2)),
        getDrefName(CE->getArg(3)), IndentStr);
  } else {
    const CompoundStmt *CS = findImmediateBlock(CE);
    if (!CS || !(CS->body_front()))
      return;

    SourceLocation DistrInsertLoc =
        SM.getExpansionLoc(CS->body_front()->getBeginLoc());
    std::string DistrIndentStr = getIndent(DistrInsertLoc, SM).str();
    std::string DrefedStateName = getDrefName(CE->getArg(0));

    if (FuncName == "curand_uniform") {
      DpctGlobalInfo::getDeviceRNGReturnNumSet().insert(1);
      DpctGlobalInfo::getInstance().insertDeviceRandomGenerateAPIInfo(
          FuncNameBegin, FuncCallLength, DistrInsertLoc,
          "oneapi::mkl::rng::device::uniform", "float", DistrIndentStr,
          DrefedStateName, IndentStr);
    } else if (FuncName == "curand_normal2") {
      DpctGlobalInfo::getDeviceRNGReturnNumSet().insert(2);
      DpctGlobalInfo::getInstance().insertDeviceRandomGenerateAPIInfo(
          FuncNameBegin, FuncCallLength, DistrInsertLoc,
          "oneapi::mkl::rng::device::gaussian", "float", DistrIndentStr,
          DrefedStateName, IndentStr);
    } else if (FuncName == "curand_normal2_double") {
      DpctGlobalInfo::getDeviceRNGReturnNumSet().insert(2);
      DpctGlobalInfo::getInstance().insertDeviceRandomGenerateAPIInfo(
          FuncNameBegin, FuncCallLength, DistrInsertLoc,
          "oneapi::mkl::rng::device::gaussian", "double", DistrIndentStr,
          DrefedStateName, IndentStr);
    } else if (FuncName == "curand_normal_double") {
      DpctGlobalInfo::getDeviceRNGReturnNumSet().insert(1);
      DpctGlobalInfo::getInstance().insertDeviceRandomGenerateAPIInfo(
          FuncNameBegin, FuncCallLength, DistrInsertLoc,
          "oneapi::mkl::rng::device::gaussian", "double", DistrIndentStr,
          DrefedStateName, IndentStr);
    }
  }
}

REGISTER_RULE(DeviceRandomFunctionCallRule)

void BLASFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "make_cuComplex", "make_cuDoubleComplex",
        /*Regular BLAS API*/
        /*Regular helper*/
        "cublasCreate_v2", "cublasDestroy_v2", "cublasSetVector",
        "cublasGetVector", "cublasSetVectorAsync", "cublasGetVectorAsync",
        "cublasSetMatrix", "cublasGetMatrix", "cublasSetMatrixAsync",
        "cublasGetMatrixAsync", "cublasSetStream_v2", "cublasGetStream_v2",
        "cublasGetPointerMode_v2", "cublasSetPointerMode_v2",
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
        "cublasCsrot_v2", "cublasZdrot_v2", "cublasSrotg_v2", "cublasDrotg_v2",
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
        "cublasSsyr_v2", "cublasDsyr_v2", "cublasSsyr2_v2", "cublasDsyr2_v2",
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
        /*Legacy API*/
        "cublasInit", "cublasShutdown", "cublasGetError",
        "cublasSetKernelStream",
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
        "cublasDrot", "cublasCsrot", "cublasZdrot", "cublasSrotg",
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
}

void BLASFunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
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

  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  const SourceManager *SM = Result.SourceManager;
  auto SL = SM->getExpansionLoc(CE->getBeginLoc());
  std::string Key = SM->getFilename(SL).str() +
                    std::to_string(SM->getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));

  SourceLocation FuncNameBegin(CE->getBeginLoc());
  SourceLocation FuncCallEnd(CE->getEndLoc());
  // There are some macroes like "#define API API_v2"
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
       FuncName == "cublasZtrsmBatched")) {
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
  if (MapNames::BatchedBLASFuncReplInfoMap.find(FuncName) !=
      MapNames::BatchedBLASFuncReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::BatchedBLASFuncReplInfoMap.find(FuncName);
    auto ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;
    BLASEnumInfo EnumInfo =
        BLASEnumInfo(ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
                     ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
    std::string BufferType = ReplInfo.BufferTypeInfo[0];

    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

    // update the replacement of four enmu arguments
    auto processEnumArgus = [&](const Expr *E, unsigned int Index,
                                std::string &Argu) {
      const CStyleCastExpr *CSCE = nullptr;
      if (CSCE = dyn_cast<CStyleCastExpr>(E)) {
        std::string CurrentArgumentRepl;
        processParamIntCastToBLASEnum(E, CSCE, Index, IndentStr, EnumInfo,
                                      PrefixInsertStr, CurrentArgumentRepl);
        Argu = CurrentArgumentRepl;
      }
    };

    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(i));
      CallExprArguReplVec.push_back(EA.getReplacedString());
      processEnumArgus(CE->getArg(i), i, CallExprArguReplVec[i]);
    }

    // update the replacement of buffers
    for (size_t i = 0; i < ReplInfo.BufferIndexInfo.size(); ++i) {
      int BufferIndex = ReplInfo.BufferIndexInfo[i];
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
        std::string BufferDecl;
        requestFeature(HelperFeatureEnum::Memory_get_buffer_T, CE);
        CallExprArguReplVec[BufferIndex] = getBufferNameAndDeclStr(
            CE->getArg(BufferIndex), BufferType, IndentStr, BufferDecl);
        PrefixInsertStr = PrefixInsertStr + BufferDecl;
      } else {
        if (BufferType == "std::complex<float>" ||
            BufferType == "std::complex<double>") {
          if (i == ReplInfo.BufferIndexInfo.size() - 1)
            CallExprArguReplVec[BufferIndex] =
                "(" + BufferType + "**)" + CallExprArguReplVec[BufferIndex];
          else
            CallExprArguReplVec[BufferIndex] = "(const " + BufferType + "**)" +
                                               CallExprArguReplVec[BufferIndex];
        }
      }
    }

    // update the replacement of scalar arguments
    for (size_t i = 0; i < ReplInfo.PointerIndexInfo.size(); ++i) {
      int ScalarIndex = ReplInfo.PointerIndexInfo[i];
      CallExprArguReplVec[ScalarIndex] = getValueStr(
          CE->getArg(ScalarIndex), CallExprArguReplVec[ScalarIndex],
          CallExprArguReplVec[0], BufferType);
      // requestFeature(HelperFeatureEnum::BlasUtils_get_value, CE);
    }

    // Declare temp variables for
    // m/n/k/lda/ldb/ldc/alpha/beta/transa/transb/groupsize These pointers are
    // accessed on host only and the value will be saved before MKL API returns,
    // so pass the addresses of the variables on stack directly.
    auto declareTempVars = [&](const std::string &TempVarType,
                               std::vector<std::string> Names,
                               std::vector<int> Indexes) {
      auto Num = Names.size();
      PrefixInsertStr = PrefixInsertStr + TempVarType;
      for (size_t i = 0; i < Num; i++) {
        std::string DeclName =
            Names[i] +
            std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
        PrefixInsertStr = PrefixInsertStr + " " + DeclName + " = " +
                          CallExprArguReplVec[Indexes[i]] + ",";
        CallExprArguReplVec[Indexes[i]] = "&" + DeclName;
      }
      PrefixInsertStr[PrefixInsertStr.size() - 1] = ';';
      PrefixInsertStr = PrefixInsertStr + getNL() + IndentStr;
    };

    if (FuncName == "cublasHgemmBatched" || FuncName == "cublasSgemmBatched" ||
        FuncName == "cublasDgemmBatched" || FuncName == "cublasCgemmBatched" ||
        FuncName == "cublasZgemmBatched") {
      if (CE->getArg(1)->IgnoreImplicit()->isLValue())
        CallExprArguReplVec[1] = "&" + CallExprArguReplVec[1];
      else
        declareTempVars({"oneapi::mkl::transpose"}, {"transpose_ct"}, {1});
      if (CE->getArg(2)->IgnoreImplicit()->isLValue())
        CallExprArguReplVec[2] = "&" + CallExprArguReplVec[2];
      else
        declareTempVars({"oneapi::mkl::transpose"}, {"transpose_ct"}, {2});

      declareTempVars("int64_t",
                      {"m_ct", "n_ct", "k_ct", "lda_ct", "ldb_ct", "ldc_ct",
                       "group_size_ct"},
                      {3, 4, 5, 8, 10, 13, 14});
      declareTempVars(BufferType, {"alpha_ct", "beta_ct"}, {6, 11});

      // insert the group_count to CallExprArguReplVec
      auto InsertIter = CallExprArguReplVec.begin();
      std::advance(InsertIter, 14);
      CallExprArguReplVec.insert(InsertIter, "1");
    } else {
      if (CE->getArg(1)->IgnoreImplicit()->isLValue())
        CallExprArguReplVec[1] = "&" + CallExprArguReplVec[1];
      else
        declareTempVars({"oneapi::mkl::side"}, {"side_ct"}, {1});
      if (CE->getArg(2)->IgnoreImplicit()->isLValue())
        CallExprArguReplVec[2] = "&" + CallExprArguReplVec[2];
      else
        declareTempVars({"oneapi::mkl::uplo"}, {"uplo_ct"}, {2});
      if (CE->getArg(3)->IgnoreImplicit()->isLValue())
        CallExprArguReplVec[3] = "&" + CallExprArguReplVec[3];
      else
        declareTempVars({"oneapi::mkl::transpose"}, {"transpose_ct"}, {3});
      if (CE->getArg(4)->IgnoreImplicit()->isLValue())
        CallExprArguReplVec[2] = "&" + CallExprArguReplVec[4];
      else
        declareTempVars({"oneapi::mkl::diag"}, {"diag_ct"}, {4});

      declareTempVars("int64_t",
                      {"m_ct", "n_ct", "lda_ct", "ldb_ct", "group_size_ct"},
                      {5, 6, 9, 11, 12});
      declareTempVars(BufferType, {"alpha_ct"}, {7});

      // insert the group_count to CallExprArguReplVec
      auto InsertIter = CallExprArguReplVec.begin();
      std::advance(InsertIter, 12);
      CallExprArguReplVec.insert(InsertIter, "1");
    }
    // add an empty event vector as the last argument
    CallExprArguReplVec.push_back("{}");
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
  } else if (FuncName == "cublasSsyrkx" || FuncName == "cublasDsyrkx") {
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

    // initialize the replacement of each arguments
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(i));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    // update the replacement of three buffer arguments
    std::string BufferType = FuncName == "cublasSsyrkx" ? "float" : "double";
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      requestFeature(HelperFeatureEnum::Memory_get_buffer_T, CE);
      std::string BufferDecl;
      CallExprArguReplVec[6] = getBufferNameAndDeclStr(
          CE->getArg(6), BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[8] = getBufferNameAndDeclStr(
          CE->getArg(8), BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[11] = getBufferNameAndDeclStr(
          CE->getArg(11), BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
    }

    // update the replacement of two scalar arguments
    CallExprArguReplVec[5] = getValueStr(CE->getArg(5), CallExprArguReplVec[5],
                       CallExprArguReplVec[0], BufferType);
    CallExprArguReplVec[10] = getValueStr(CE->getArg(10), CallExprArguReplVec[10],
                       CallExprArguReplVec[0], BufferType);

    // update the replacement of the fillmode enmu argument
    const CStyleCastExpr *CSCE = nullptr;
    if (CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(1))) {
      std::string CurrentArgumentRepl;
      processParamIntCastToBLASEnum(CE->getArg(1), CSCE, 1, IndentStr, EnumInfo,
                                    PrefixInsertStr, CurrentArgumentRepl);
      CallExprArguReplVec[1] = CurrentArgumentRepl;
      CSCE = nullptr;
    }

    // update the replacement of two transpose operation enmu arguments
    // original API: C = alpha*t(A, t_1)*t(t(B, t_1), transpose) + beta*C
    // migrated API: C = alpha*t(A, t_1)*t(B, t_2) + beta*C
    // So the migated API need insert another transpose state variable which is
    // decided by the first transpose state variable.
    // For float and double type, the transpose state variable can only be
    // transpose or non-transpose, so if t_1 == transpose, then t_2 should be
    // non-transpose and vice versa.

    // the first transpose operation enmu argument
    std::string TransTempVarName;
    if (CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(2))) {
      // It is a c-style cast expr, process with processParamIntCastToBLASEnum()
      // If the sub-expr has side-effect, a temp variable will be declared in
      // processParamIntCastToBLASEnum() function, if not, then we can use this
      // expr in the inserted argument directly.
      std::string CurrentArgumentRepl;
      TransTempVarName = processParamIntCastToBLASEnum(
          CE->getArg(2), CSCE, 2, IndentStr, EnumInfo, PrefixInsertStr,
          CurrentArgumentRepl);
      CallExprArguReplVec[2] = CurrentArgumentRepl;
      CSCE = nullptr;
    } else {
      // if it isn't c-style cast, then the expr type should be an enumeration.
      // because the another transpose state argument depends on this argument,
      // if this expr has side-effect, then we need declare a temp variable,
      // if not, then we can use this expr in the inserted argument directly.
      if (CE->getArg(2)->HasSideEffects(DpctGlobalInfo::getContext())) {
        TransTempVarName =
            getTempNameForExpr(CE->getArg(2), true, true) + "transpose_ct" +
            std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());

        PrefixInsertStr = PrefixInsertStr + "auto " + TransTempVarName + " = " +
                          CallExprArguReplVec[2] + ";" + getNL() + IndentStr;
        CallExprArguReplVec[2] = TransTempVarName;
      }
    }

    // The inserted transpose operation enmu argument (depends on the first one)
    // Save the replacement in the InsertStr first.
    std::string InsertStr;
    if (CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(2))) {
      // Case1: the first operation enmu argument is a c-style cast expr
      if (CE->getArg(2)->HasSideEffects(DpctGlobalInfo::getContext())) {
        // if the expr has side effect, use the temp variable name which returns
        // from processParamIntCastToBLASEnum()
        InsertStr = "(int)" + TransTempVarName +
                    "==0 ? oneapi::mkl::transpose::trans : "
                    "oneapi::mkl::transpose::nontrans";
      } else {
        // if the expr hasn't side effect, use the sub-expr to decide the
        // inserting argument
        std::string SubExprStr;
        bool IsTypeCastInMacro = false;
        if (CSCE->getSubExpr()->getBeginLoc().isMacroID() &&
            isOuterMostMacro(CSCE)) {
          // When type casting syntax is in a macro, analyze the entire CSCE
          // by ExprAnalysis. Then we need add a type cast to int before the
          // inserted argument.
          // Related to BLASFunctionCallRule::processParamIntCastToBLASEnum()
          IsTypeCastInMacro = true;
          ExprAnalysis SEA;
          SEA.analyze(CSCE);
          SubExprStr = SEA.getReplacedString();
        } else {
          SubExprStr = ExprAnalysis::ref(CSCE->getSubExpr());
        }
        Expr::EvalResult ER;
        if (CSCE->getSubExpr()->EvaluateAsInt(ER, *Result.Context) &&
            !CSCE->getSubExpr()->getBeginLoc().isMacroID()) {
          // if the sub-expr can be evaluated, then generate the corresponding
          // replacement directly.
          int64_t Value = ER.Val.getInt().getExtValue();
          if (Value == 0) {
            InsertStr = "oneapi::mkl::transpose::trans";
          } else if (Value == 1) {
            InsertStr = "oneapi::mkl::transpose::nontrans";
          } else {
            InsertStr = SubExprStr + "==0 ? oneapi::mkl::transpose::trans : "
                                     "oneapi::mkl::transpose::nontrans";
          }
        } else {
          // if the sub-expr cannot be evaluated, use the conditional operator
          SubExprStr = IsTypeCastInMacro ? "(int)" + SubExprStr : SubExprStr;
          InsertStr = SubExprStr + "==0 ? oneapi::mkl::transpose::trans : "
                                   "oneapi::mkl::transpose::nontrans";
        }
      }
    } else {
      // Case2: the first operation enmu argument isn't c-style cast expr, then
      // the expr type should be an enumeration.
      if (CE->getArg(2)->HasSideEffects(DpctGlobalInfo::getContext())) {
        // if the expr has side effect, use the temp variable name which
        // generated in the previous step.
        InsertStr = TransTempVarName + "==oneapi::mkl::transpose::nontrans ? "
                                       "oneapi::mkl::transpose::trans : "
                                       "oneapi::mkl::transpose::nontrans";
      } else {
        // The expr hasn't side effect, if the enumeration is literal, then
        // generate the corresponding replacement directly, if not, use the
        // conditional operator
        std::string TransStr = ExprAnalysis::ref(CE->getArg(2));
        auto TransPair = MapNames::BLASEnumsMap.find(TransStr);
        if (TransPair != MapNames::BLASEnumsMap.end()) {
          TransStr = TransPair->second;
        }
        if (TransStr == "oneapi::mkl::transpose::nontrans") {
          InsertStr = "oneapi::mkl::transpose::trans";
        } else if (TransStr == "oneapi::mkl::transpose::trans") {
          InsertStr = "oneapi::mkl::transpose::nontrans";
        } else {
          InsertStr = TransStr + "==oneapi::mkl::transpose::nontrans ? "
                                 "oneapi::mkl::transpose::trans : "
                                 "oneapi::mkl::transpose::nontrans";
        }
      }
    }
    // insert the InsertStr to CallExprArguReplVec
    auto InsertIter = CallExprArguReplVec.begin();
    std::advance(InsertIter, 3);
    CallExprArguReplVec.insert(InsertIter, InsertStr);
    // After this line, the old iterators of CallExprArguReplVec may be
    // invalidation. Need to use new iterators.

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
  } else if (FuncName == "cublasStrmm_v2" || FuncName == "cublasDtrmm_v2" ||
             FuncName == "cublasCtrmm_v2" || FuncName == "cublasZtrmm_v2") {
    std::string Replacement;
    BLASEnumInfo EnumInfo;
    std::string BufferType;
    if (FuncName == "cublasStrmm_v2" || FuncName == "cublasDtrmm_v2") {
      auto ReplInfoPair = MapNames::BLASFuncReplInfoMap.find(FuncName);
      auto ReplInfo = ReplInfoPair->second;
      Replacement = ReplInfo.ReplName;
      EnumInfo =
          BLASEnumInfo(ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
                       ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
      BufferType = ReplInfo.BufferTypeInfo[0];
    } else {
      auto ReplInfoPair = MapNames::BLASFuncComplexReplInfoMap.find(FuncName);
      auto ReplInfo = ReplInfoPair->second;
      Replacement = ReplInfo.ReplName;
      EnumInfo =
          BLASEnumInfo(ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
                       ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
      BufferType = ReplInfo.BufferTypeInfo[0];
    }

    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

    // initialize the replacement of each arguments
    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(i));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    // original API is like: C = A * B
    // migrated API is like: B = A * B
    // So first we need copy all arguments relative to B (including leading
    // dimension and data) to C, then remove all arguments relative to C in the
    // original call.
    // In order to call the memcpy API, if an expression will be used more than
    // twice and have side effect, it need be redecalred. In this API migration,
    // the ptrC, m, n and ldc may need redeclaration.

    // declare a temp variable for ptrC if it has side effect
    std::string PtrCName;
    if (CE->getArg(12)->HasSideEffects(DpctGlobalInfo::getContext())) {
      PtrCName = getTempNameForExpr(CE->getArg(12), true, true) + "ptr_ct" +
                 std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
      PrefixInsertStr = PrefixInsertStr + "auto " + PtrCName + " = " +
                        CallExprArguReplVec[12] + ";" + getNL() + IndentStr;
    } else {
      PtrCName = CallExprArguReplVec[12];
    }

    // declare temp variables for m, n and ldc if has side effect
    auto &Context = dpct::DpctGlobalInfo::getContext();
    std::string Arg5Repl, Arg6Repl, Arg13Repl;
    std::string TempArgsDecl;
    auto processTempVars = [&](const Expr *E, std::string Name,
                               std::string &ArgRepl) {
      if (E->HasSideEffects(Context)) {
        ArgRepl = getTempNameForExpr(E, true, true) + Name +
                  std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
        TempArgsDecl = TempArgsDecl + "auto " + ArgRepl + " = " +
                       ExprAnalysis::ref(E) + ";";
      } else {
        ArgRepl = ExprAnalysis::ref(E);
      }
    };
    processTempVars(CE->getArg(5), "m_ct", Arg5Repl);    // Arg5: m
    processTempVars(CE->getArg(6), "n_ct", Arg6Repl);    // Arg6: n
    processTempVars(CE->getArg(13), "ld_ct", Arg13Repl); // Arg13: ldc
    CallExprArguReplVec[5] = Arg5Repl;
    CallExprArguReplVec[6] = Arg6Repl;
    CallExprArguReplVec[13] = Arg13Repl;
    if (!TempArgsDecl.empty())
      PrefixInsertStr = PrefixInsertStr + TempArgsDecl + getNL() + IndentStr;

    requestFeature(HelperFeatureEnum::Util_matrix_mem_copy_T, CE);
    requestFeature(HelperFeatureEnum::Memory_memcpy_direction, CE);
    // generate the data memcpy API call
    PrefixInsertStr =
        PrefixInsertStr + MapNames::getDpctNamespace() + "matrix_mem_copy(" +
        PtrCName + ", " + CallExprArguReplVec[10] + ", " + Arg13Repl + ", " +
        CallExprArguReplVec[11] + ", " + Arg5Repl + ", " + Arg6Repl + ", " +
        MapNames::getDpctNamespace() + "device_to_device, *" +
        CallExprArguReplVec[0] + ");" + getNL() + IndentStr;

    // update the replacement of four enmu arguments
    auto processEnumArgus = [&](const Expr *E, unsigned int Index,
                                std::string &Argu) {
      const CStyleCastExpr *CSCE = nullptr;
      if (CSCE = dyn_cast<CStyleCastExpr>(E)) {
        std::string CurrentArgumentRepl;
        processParamIntCastToBLASEnum(E, CSCE, Index, IndentStr, EnumInfo,
                                      PrefixInsertStr, CurrentArgumentRepl);
        Argu = CurrentArgumentRepl;
      }
    };
    processEnumArgus(CE->getArg(1), 1, CallExprArguReplVec[1]);
    processEnumArgus(CE->getArg(2), 2, CallExprArguReplVec[2]);
    processEnumArgus(CE->getArg(3), 3, CallExprArguReplVec[3]);
    processEnumArgus(CE->getArg(4), 4, CallExprArguReplVec[4]);

    // update the replacement of two buffers
    // the second buffer is constrcuted with PtrCName
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      requestFeature(HelperFeatureEnum::Memory_get_buffer_T, CE);
      std::string BufferDecl;
      CallExprArguReplVec[8] = getBufferNameAndDeclStr(
          CE->getArg(8), BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[12] =
          getBufferNameAndDeclStr(PtrCName, BufferType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
    } else {
      if (FuncName == "cublasCtrmm_v2") {
        CallExprArguReplVec[8] =
            "(std::complex<float>*)" + CallExprArguReplVec[8];
        CallExprArguReplVec[12] = "(std::complex<float>*)" + PtrCName;
      } else if (FuncName == "cublasZtrmm_v2") {
        CallExprArguReplVec[8] =
            "(std::complex<double>*)" + CallExprArguReplVec[8];
        CallExprArguReplVec[12] = "(std::complex<double>*)" + PtrCName;
      } else {
        CallExprArguReplVec[12] = PtrCName;
      }
    }

    // requestFeature(HelperFeatureEnum::BlasUtils_get_value, CE);
    // update the replacement of a scalar argument
    CallExprArguReplVec[7] =
        getValueStr(CE->getArg(7), CallExprArguReplVec[7],
                       CallExprArguReplVec[0], BufferType);

    // Remove arguments ptrB and ldb from CallExprArguReplVec
    CallExprArguReplVec.erase(CallExprArguReplVec.begin() + 10,
                              CallExprArguReplVec.begin() + 12);
    // After this line, the old iterators of CallExprArguReplVec may be
    // invalidation. Need to use new iterators.

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
  } else if (FuncName == "cublasSgemmEx" || FuncName == "cublasCgemmEx") {
    std::string Replacement = "oneapi::mkl::blas::column_major::gemm";
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

    auto AType = CE->getArg(8);
    auto BType = CE->getArg(11);
    auto CType = CE->getArg(15);

    Expr::EvalResult ATypeER, BTypeER, CTypeER;
    bool CanATypeBeEval =
        AType->EvaluateAsInt(ATypeER, DpctGlobalInfo::getContext());
    bool CanBTypeBeEval =
        BType->EvaluateAsInt(BTypeER, DpctGlobalInfo::getContext());
    bool CanCTypeBeEval =
        CType->EvaluateAsInt(CTypeER, DpctGlobalInfo::getContext());

    int64_t ABTypeValue, CTypeValue;
    if (CanCTypeBeEval && (CanATypeBeEval || CanBTypeBeEval)) {
      CTypeValue = CTypeER.Val.getInt().getExtValue();
      if (CanATypeBeEval)
        ABTypeValue = ATypeER.Val.getInt().getExtValue();
      else
        ABTypeValue = BTypeER.Val.getInt().getExtValue();
    } else {
      report(FuncNameBegin, Diagnostics::UNSUPPORT_PARAM_COMBINATION, false,
             MapNames::ITFName.at(FuncName),
             "not all values of parameters could be evaluated in migration");
      return;
    }

    MapNames::BLASGemmExTypeInfo TypeInfo;
    std::string Key =
        std::to_string(ABTypeValue) + ":" + std::to_string(CTypeValue);
    if (MapNames::BLASTGemmExTypeInfoMap.find(Key) !=
        MapNames::BLASTGemmExTypeInfoMap.end()) {
      TypeInfo = MapNames::BLASTGemmExTypeInfoMap.find(Key)->second;
    } else {
      report(
          FuncNameBegin, Diagnostics::UNSUPPORT_PARAM_COMBINATION, false,
          MapNames::ITFName.at(FuncName),
          "the combination of matrix data type and scalar type is unsupported");
      return;
    }

    std::vector<int> ArgsIndex{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 16};
    // initialize the replacement of each arguments
    for (auto I : ArgsIndex) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(I));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    // update the replacement of four enmu arguments
    BLASEnumInfo EnumInfo = BLASEnumInfo({1, 2}, -1, -1, -1);
    auto processEnumArgus = [&](const Expr *E, unsigned int Index,
                                std::string &Argu) {
      const CStyleCastExpr *CSCE = nullptr;
      if (CSCE = dyn_cast<CStyleCastExpr>(E)) {
        std::string CurrentArgumentRepl;
        processParamIntCastToBLASEnum(E, CSCE, Index, IndentStr, EnumInfo,
                                      PrefixInsertStr, CurrentArgumentRepl);
        Argu = CurrentArgumentRepl;
      }
    };
    processEnumArgus(CE->getArg(1), 1, CallExprArguReplVec[1]);
    processEnumArgus(CE->getArg(2), 2, CallExprArguReplVec[2]);

    // update the replacement of three buffers
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      requestFeature(HelperFeatureEnum::Memory_get_buffer_T, CE);
      std::string BufferDecl;
      CallExprArguReplVec[7] = getBufferNameAndDeclStr(
          CE->getArg(7), TypeInfo.ABType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[9] = getBufferNameAndDeclStr(
          CE->getArg(10), TypeInfo.ABType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[12] = getBufferNameAndDeclStr(
          CE->getArg(14), TypeInfo.CType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
    } else {
      CallExprArguReplVec[7] =
          "(" + TypeInfo.ABType + "*)" + CallExprArguReplVec[7];
      CallExprArguReplVec[9] =
          "(" + TypeInfo.ABType + "*)" + CallExprArguReplVec[9];
      CallExprArguReplVec[12] =
          "(" + TypeInfo.CType + "*)" + CallExprArguReplVec[12];
    }

    // requestFeature(HelperFeatureEnum::BlasUtils_get_value, CE);
    // update the replacement of two scalar arguments
    CallExprArguReplVec[6] = getValueStr(
        CE->getArg(6), CallExprArguReplVec[6], CallExprArguReplVec[0],
        FuncName == "cublasCgemmEx" ? "std::complex<float>" : "");
    CallExprArguReplVec[11] = getValueStr(
        CE->getArg(13), CallExprArguReplVec[11], CallExprArguReplVec[0],
        FuncName == "cublasCgemmEx" ? "std::complex<float>" : "");
    if (Key == "2:2") {
      CallExprArguReplVec[6] = MapNames::getClNamespace() + "vec<float, 1>{" +
                               CallExprArguReplVec[6] + "}.convert<" +
                               MapNames::getClNamespace() + "half, " +
                               MapNames::getClNamespace() +
                               "rounding_mode::automatic>()[0]";
      CallExprArguReplVec[11] = MapNames::getClNamespace() + "vec<float, 1>{" +
                                CallExprArguReplVec[11] + "}.convert<" +
                                MapNames::getClNamespace() + "half, " +
                                MapNames::getClNamespace() +
                                "rounding_mode::automatic>()[0]";
    }

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
  } else if (FuncName == "cublasGemmEx") {
    std::string Replacement = "oneapi::mkl::blas::column_major::gemm";
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

    // clang-format off
    // MKL API does not have computeType and algo parameters.
    // computeType(alpha/beta)               AType/BType     CType IsSupportInMKL
    // CUDA_R_16F(2)/CUBLAS_COMPUTE_16F(64)  CUDA_R_16F(2)   CUDA_R_16F(2)   yes
    // CUDA_R_32I(10)                        CUDA_R_8I(3)    CUDA_R_32I(10)  no
    // CUDA_R_32F(0)/CUBLAS_COMPUTE_32F(68)  CUDA_R_16F(2)   CUDA_R_16F(2)   no (can cast alpha/beta to half)
    // CUDA_R_32F(0)                         CUDA_R_8I(3)    CUDA_R_32I(10)  no
    // CUDA_R_32F(0)/CUBLAS_COMPUTE_32F(68)  CUDA_R_16F(2)   CUDA_R_32F(0)   yes
    // CUDA_R_32F(0)/CUBLAS_COMPUTE_32F(68)  CUDA_R_32F(0)   CUDA_R_32F(0)   yes
    // CUDA_R_64F(1)/CUBLAS_COMPUTE_64F(70)  CUDA_R_64F(1)   CUDA_R_64F(1)   yes
    // CUDA_C_32F(4)                         CUDA_C_8I(7)    CUDA_C_32F(4)   no
    // CUDA_C_32F(4)                         CUDA_C_32F(4)   CUDA_C_32F(4)   yes
    // CUDA_C_64F(5)                         CUDA_C_64F(5)   CUDA_C_64F(5)   yes
    // clang-format on

    auto AType = CE->getArg(8);
    auto BType = CE->getArg(11);
    auto CType = CE->getArg(15);
    auto ComputeType = CE->getArg(17);

    Expr::EvalResult ATypeER, BTypeER, CTypeER, ComputeTypeER;
    bool CanATypeBeEval =
        AType->EvaluateAsInt(ATypeER, DpctGlobalInfo::getContext());
    bool CanBTypeBeEval =
        BType->EvaluateAsInt(BTypeER, DpctGlobalInfo::getContext());
    bool CanCTypeBeEval =
        CType->EvaluateAsInt(CTypeER, DpctGlobalInfo::getContext());
    bool CanComputeTypeBeEval =
        ComputeType->EvaluateAsInt(ComputeTypeER, DpctGlobalInfo::getContext());

    int64_t ABTypeValue, CTypeValue, ComputeTypeValue;
    if (CanCTypeBeEval && CanComputeTypeBeEval &&
        (CanATypeBeEval || CanBTypeBeEval)) {
      CTypeValue = CTypeER.Val.getInt().getExtValue();
      ComputeTypeValue = ComputeTypeER.Val.getInt().getExtValue();
      if (CanATypeBeEval)
        ABTypeValue = ATypeER.Val.getInt().getExtValue();
      else
        ABTypeValue = BTypeER.Val.getInt().getExtValue();
    } else {
      report(FuncNameBegin, Diagnostics::UNSUPPORT_PARAM_COMBINATION, false,
             MapNames::ITFName.at(FuncName),
             "not all values of parameters could be evaluated in migration");
      return;
    }

    MapNames::BLASGemmExTypeInfo TypeInfo;
    std::string Key = std::to_string(ComputeTypeValue) + ":" +
                      std::to_string(ABTypeValue) + ":" +
                      std::to_string(CTypeValue);
    if (MapNames::BLASGemmExTypeInfoMap.find(Key) !=
        MapNames::BLASGemmExTypeInfoMap.end()) {
      TypeInfo = MapNames::BLASGemmExTypeInfoMap.find(Key)->second;
    } else {
      report(
          FuncNameBegin, Diagnostics::UNSUPPORT_PARAM_COMBINATION, false,
          MapNames::ITFName.at(FuncName),
          "the combination of matrix data type and scalar type is unsupported");
      return;
    }

    std::vector<int> ArgsIndex{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 16};
    // initialize the replacement of each arguments
    for (auto I : ArgsIndex) {
      ExprAnalysis EA;
      EA.analyze(CE->getArg(I));
      CallExprArguReplVec.push_back(EA.getReplacedString());
    }

    // update the replacement of four enmu arguments
    BLASEnumInfo EnumInfo = BLASEnumInfo({1, 2}, -1, -1, -1);
    auto processEnumArgus = [&](const Expr *E, unsigned int Index,
                                std::string &Argu) {
      const CStyleCastExpr *CSCE = nullptr;
      if (CSCE = dyn_cast<CStyleCastExpr>(E)) {
        std::string CurrentArgumentRepl;
        processParamIntCastToBLASEnum(E, CSCE, Index, IndentStr, EnumInfo,
                                      PrefixInsertStr, CurrentArgumentRepl);
        Argu = CurrentArgumentRepl;
      }
    };
    processEnumArgus(CE->getArg(1), 1, CallExprArguReplVec[1]);
    processEnumArgus(CE->getArg(2), 2, CallExprArguReplVec[2]);

    // update the replacement of three buffers
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      requestFeature(HelperFeatureEnum::Memory_get_buffer_T, CE);
      std::string BufferDecl;
      CallExprArguReplVec[7] = getBufferNameAndDeclStr(
          CE->getArg(7), TypeInfo.ABType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[9] = getBufferNameAndDeclStr(
          CE->getArg(10), TypeInfo.ABType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
      CallExprArguReplVec[12] = getBufferNameAndDeclStr(
          CE->getArg(14), TypeInfo.CType, IndentStr, BufferDecl);
      PrefixInsertStr = PrefixInsertStr + BufferDecl;
    } else {
      CallExprArguReplVec[7] =
          "(" + TypeInfo.ABType + "*)" + CallExprArguReplVec[7];
      CallExprArguReplVec[9] =
          "(" + TypeInfo.ABType + "*)" + CallExprArguReplVec[9];
      CallExprArguReplVec[12] =
          "(" + TypeInfo.CType + "*)" + CallExprArguReplVec[12];
    }

    // update the replacement of two scalar arguments
    // requestFeature(HelperFeatureEnum::BlasUtils_get_value, CE);
    CallExprArguReplVec[6] = getValueStr(CE->getArg(6),
                                            "(" + TypeInfo.OriginScalarType +
                                                "*)" + CallExprArguReplVec[6],
                                            CallExprArguReplVec[0]);
    CallExprArguReplVec[11] = getValueStr(CE->getArg(11),
                                             "(" + TypeInfo.OriginScalarType +
                                                 "*)" + CallExprArguReplVec[11],
                                             CallExprArguReplVec[0]);
    if (Key == "0:2:2" || Key == "68:2:2") {
      CallExprArguReplVec[6] = MapNames::getClNamespace() + "vec<float, 1>{" +
                               CallExprArguReplVec[6] + "}.convert<" +
                               MapNames::getClNamespace() + "half, " +
                               MapNames::getClNamespace() +
                               "rounding_mode::automatic>()[0]";
      CallExprArguReplVec[11] = MapNames::getClNamespace() + "vec<float, 1>{" +
                                CallExprArguReplVec[11] + "}.convert<" +
                                MapNames::getClNamespace() + "half, " +
                                MapNames::getClNamespace() +
                                "rounding_mode::automatic>()[0]";
    }

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
            requestFeature(HelperFeatureEnum::Device_get_default_queue, CE);
            requestFeature(HelperFeatureEnum::Memory_dpct_memcpy, CE);
            std::string ResultTempPtr =
                "res_temp_ptr_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            std::string ResultTempHost =
                "res_temp_host_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "int64_t* " + ResultTempPtr +
                              " = " + MapNames::getClNamespace() +
                              "malloc_shared<int64_t>(" + "1, " +
                              MapNames::getDpctNamespace() +
                              "get_default_queue());" + getNL() + IndentStr;
            SuffixInsertStr =
                SuffixInsertStr + getNL() + IndentStr + "int " +
                ResultTempHost + " = (int)*" + ResultTempPtr + ";" + getNL() +
                IndentStr + MapNames::getDpctNamespace() + "dpct_memcpy(" +
                ExprAnalysis::ref(CE->getArg(i)) + ", &" + ResultTempHost +
                ", sizeof(int));" + getNL() + IndentStr +
                MapNames::getClNamespace() + "free(" + ResultTempPtr + ", " +
                MapNames::getDpctNamespace() + "get_default_queue());";
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
            requestFeature(HelperFeatureEnum::Memory_get_buffer_T, CE);
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
        CurrentArgumentRepl = getValueStr(
            CE->getArg(i), EA.getReplacedString(), CallExprArguReplVec[0]);
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
        // If there is one API call in the migrted code, it is unnecessary to
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
            requestFeature(HelperFeatureEnum::Device_get_default_queue, CE);
            requestFeature(HelperFeatureEnum::Memory_dpct_memcpy, CE);
            std::string ResultTempPtr =
                "res_temp_ptr_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            std::string ResultTempHost =
                "res_temp_host_ct" +
                std::to_string(DpctGlobalInfo::getSuffixIndexInRuleThenInc());
            PrefixInsertStr = PrefixInsertStr + "int64_t* " + ResultTempPtr +
                              " = " + MapNames::getClNamespace() +
                              "malloc_shared<int64_t>(" + "1, " +
                              MapNames::getDpctNamespace() +
                              "get_default_queue());" + getNL() + IndentStr;
            SuffixInsertStr =
                SuffixInsertStr + getNL() + IndentStr + "int " +
                ResultTempHost + " = (int)*" + ResultTempPtr + ";" + getNL() +
                IndentStr + MapNames::getDpctNamespace() + "dpct_memcpy(" +
                ExprAnalysis::ref(CE->getArg(i)) + ", &" + ResultTempHost +
                ", sizeof(int));" + getNL() + IndentStr +
                MapNames::getClNamespace() + "free(" + ResultTempPtr + ", " +
                MapNames::getDpctNamespace() + "get_default_queue());";
            CurrentArgumentRepl = ResultTempPtr;
          } else if (ReplInfo.BufferTypeInfo[IndexTemp] ==
                         "std::complex<float>" ||
                     ReplInfo.BufferTypeInfo[IndexTemp] ==
                         "std::complex<double>") {
            CurrentArgumentRepl = "(" + ReplInfo.BufferTypeInfo[IndexTemp] +
                                  "*)" + ExprAnalysis::ref(CE->getArg(i));
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
            requestFeature(HelperFeatureEnum::Memory_get_buffer_T, CE);
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
        // requestFeature(HelperFeatureEnum::BlasUtils_get_value, CE);
        CurrentArgumentRepl = getValueStr(
            CE->getArg(i), EA.getReplacedString(), CallExprArguReplVec[0],
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
        // If there is one API call in the migrted code, it is unnecessary to
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
    requestFeature(HelperFeatureEnum::Device_get_current_device, CE);
    requestFeature(HelperFeatureEnum::Device_device_ext_get_saved_queue, CE);
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
        if (VarType == "cuComplex") {
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
            CallExprReplStr = CallExprReplStr + ", (" +
                              ReplInfo.BufferTypeInfo[IndexTemp] + "*)" +
                              ParamsStrsVec[i];
          } else {
            CallExprReplStr = CallExprReplStr + ", " + ParamsStrsVec[i];
          }
        } else {
          requestFeature(HelperFeatureEnum::Memory_get_buffer_T, CE);
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
        requestFeature(HelperFeatureEnum::Device_get_default_queue, CE);
        PrefixInsertStr =
            PrefixInsertStr + ResultType + "* " + ResultTempPtr + " = " +
            MapNames::getClNamespace() + "malloc_shared<" + ResultType +
            ">(1, " + MapNames::getDpctNamespace() + "get_default_queue());" +
            getNL() + IndentStr + CallExprReplStr + ", " + ResultTempPtr +
            ").wait();" + getNL() + IndentStr;

        ReturnValueParamsStr =
            "(" + ResultTempPtr + "->real(), " + ResultTempPtr + "->imag())";

        if (NeedUseLambda) {
          PrefixInsertStr = PrefixInsertStr + ResultType + " " + ResultTempVal +
                            " = *" + ResultTempPtr + ";" + getNL() + IndentStr +
                            MapNames::getClNamespace() + "free(" +
                            ResultTempPtr + ", " +
                            MapNames::getDpctNamespace() +
                            "get_default_queue());" + getNL() + IndentStr;
          ReturnValueParamsStr =
              "(" + ResultTempVal + ".real(), " + ResultTempVal + ".imag())";
        } else {
          SuffixInsertStr =
              SuffixInsertStr + getNL() + IndentStr +
              MapNames::getClNamespace() + "free(" + ResultTempPtr + ", " +
              MapNames::getDpctNamespace() + "get_default_queue());";
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
  } else if (MapNames::BLASFuncWrapperReplInfoMap.find(FuncName) !=
             MapNames::BLASFuncWrapperReplInfoMap.end()) {
    auto ReplInfoPair = MapNames::BLASFuncWrapperReplInfoMap.find(FuncName);
    MapNames::BLASFuncReplInfo ReplInfo = ReplInfoPair->second;
    std::string Replacement = ReplInfo.ReplName;

    static const std::unordered_map<std::string, HelperFeatureEnum>
        FuncNameToFeatureMap = {
            {"cublasSgetrfBatched",
             HelperFeatureEnum::BlasUtils_getrf_batch_wrapper},
            {"cublasDgetrfBatched",
             HelperFeatureEnum::BlasUtils_getrf_batch_wrapper},
            {"cublasCgetrfBatched",
             HelperFeatureEnum::BlasUtils_getrf_batch_wrapper},
            {"cublasZgetrfBatched",
             HelperFeatureEnum::BlasUtils_getrf_batch_wrapper},
            {"cublasSgetrsBatched",
             HelperFeatureEnum::BlasUtils_getrs_batch_wrapper},
            {"cublasDgetrsBatched",
             HelperFeatureEnum::BlasUtils_getrs_batch_wrapper},
            {"cublasCgetrsBatched",
             HelperFeatureEnum::BlasUtils_getrs_batch_wrapper},
            {"cublasZgetrsBatched",
             HelperFeatureEnum::BlasUtils_getrs_batch_wrapper},
            {"cublasSgetriBatched",
             HelperFeatureEnum::BlasUtils_getri_batch_wrapper},
            {"cublasDgetriBatched",
             HelperFeatureEnum::BlasUtils_getri_batch_wrapper},
            {"cublasCgetriBatched",
             HelperFeatureEnum::BlasUtils_getri_batch_wrapper},
            {"cublasZgetriBatched",
             HelperFeatureEnum::BlasUtils_getri_batch_wrapper},
            {"cublasSgeqrfBatched",
             HelperFeatureEnum::BlasUtils_geqrf_batch_wrapper},
            {"cublasDgeqrfBatched",
             HelperFeatureEnum::BlasUtils_geqrf_batch_wrapper},
            {"cublasCgeqrfBatched",
             HelperFeatureEnum::BlasUtils_geqrf_batch_wrapper},
            {"cublasZgeqrfBatched",
             HelperFeatureEnum::BlasUtils_geqrf_batch_wrapper}};
    requestFeature(FuncNameToFeatureMap.at(FuncName), CE);

    BLASEnumInfo EnumInfo(
        ReplInfo.OperationIndexInfo, ReplInfo.FillModeIndexInfo,
        ReplInfo.SideModeIndexInfo, ReplInfo.DiagTypeIndexInfo);
    if (HasDeviceAttr) {
      report(FuncNameBegin, Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), Replacement);
      return;
    }

    StringRef FuncNameRef(FuncName);
    if (FuncNameRef.endswith("getrfBatched")) {
      report(FuncNameBegin, Diagnostics::DIFFERENT_LU_FACTORIZATION, false,
             getStmtSpelling(CE->getArg(4)), Replacement,
             MapNames::ITFName.at(FuncName));
    }

    int ArgNum = CE->getNumArgs();
    for (int i = 0; i < ArgNum; ++i) {
      const CStyleCastExpr *CSCE = nullptr;
      std::string CurrentArgumentRepl;
      if ((CSCE = dyn_cast<CStyleCastExpr>(CE->getArg(i)))) {
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
    CallExprReplStr = getFinalCallExprStr(Replacement) + CallExprReplStr;

    if (NeedUseLambda) {
      if (PrefixInsertStr == "") {
        // If there is one API call in the migrted code, it is unnecessary to
        // use a lambda expression
        NeedUseLambda = false;
      }
    }

    applyMigrationText(NeedUseLambda, IsMacroArg, CanAvoidBrace,
                       CanAvoidUsingLambda, OriginStmtType, IsAssigned,
                       OuterInsertLoc, PrefixInsertLoc, SuffixInsertLoc,
                       FuncNameBegin, FuncCallEnd, FuncCallLength, IndentStr,
                       PrefixInsertStr, SuffixInsertStr);
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
      requestFeature(HelperFeatureEnum::Device_device_ext_set_saved_queue, CE);
      Repl = "{{NEEDREPLACED" + std::to_string(Index) + "}}.set_saved_queue(" +
             EA.getReplacedString() + ")";
    } else {
      return;
    }

    if (SM->isMacroArgExpansion(CE->getBeginLoc()) &&
        SM->isMacroArgExpansion(CE->getEndLoc())) {
      if (IsAssigned) {
        report(SR.getBegin(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
        emplaceTransformation(
            new ReplaceText(SR.getBegin(), Len, "(" + Repl + ", 0)"));
      } else {
        emplaceTransformation(
            new ReplaceText(SR.getBegin(), Len, std::move(Repl)));
      }
    } else {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
        emplaceTransformation(new ReplaceStmt(CE, true, "(" + Repl + ", 0)"));
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
             FuncName == "cublasSetPointerMode_v2") {
    std::string Msg = "the function call is redundant in DPC++.";
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
             MapNames::ITFName.at(FuncName), "dpct::matrix_mem_copy");
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
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
               false, MapNames::ITFName.at(FuncName),
               "parameter " + ParamsStrsVec[3] +
                   " does not equal to parameter " + ParamsStrsVec[5]);
      } else if ((IncxStr == IncyStr) && (IncxStr != "1")) {
        // incx equals to incy, but does not equal to 1. Performance issue may
        // occur.
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
               false, MapNames::ITFName.at(FuncName),
               "parameter " + ParamsStrsVec[3] + " equals to parameter " +
                   ParamsStrsVec[5] + " but greater than 1");
      }
    } else {
      report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE, false,
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

    requestFeature(HelperFeatureEnum::Util_matrix_mem_copy, CE);
    requestFeature(HelperFeatureEnum::Memory_memcpy_direction, CE);
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
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      insertAroundStmt(CE, "(", ", 0)");
    }
  } else if (FuncName == "cublasSetMatrix" || FuncName == "cublasGetMatrix" ||
             FuncName == "cublasSetMatrixAsync" ||
             FuncName == "cublasGetMatrixAsync") {
    if (HasDeviceAttr) {
      report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
             MapNames::ITFName.at(FuncName), "dpct::matrix_mem_copy");
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
        report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
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
            report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE,
                   false, MapNames::ITFName.at(FuncName),
                   "parameter " + ParamsStrsVec[0] +
                       " is smaller than parameter " + ParamsStrsVec[4]);
          }
        } else {
          // rows cannot be evaluated. Performance issue may occur.
          report(
              CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE, false,
              MapNames::ITFName.at(FuncName),
              "parameter " + ParamsStrsVec[0] +
                  " could not be evaluated and may be smaller than parameter " +
                  ParamsStrsVec[4]);
        }
      }
    } else {
      report(CE->getBeginLoc(), Diagnostics::POTENTIAL_PERFORMACE_ISSUE, false,
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

    requestFeature(HelperFeatureEnum::Util_matrix_mem_copy, CE);
    requestFeature(HelperFeatureEnum::Memory_memcpy_direction, CE);
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
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      insertAroundStmt(CE, "(", ", 0)");
    }
  } else if (FuncName == "make_cuComplex" ||
             FuncName == "make_cuDoubleComplex") {
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

REGISTER_RULE(BLASFunctionCallRule)

// Rule for SOLVER enums.
// Migrate SOLVER status values to corresponding int values
// Other SOLVER named values are migrated to corresponding named values
void SOLVEREnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName("CUSOLVER_STATU.*"))))
          .bind("SOLVERStatusConstants"),
      this);
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(matchesName(
                      "(CUSOLVER_EIG_TYPE.*)|(CUSOLVER_EIG_MODE.*)"))))
          .bind("SLOVERNamedValueConstants"),
      this);
}

void SOLVEREnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "SOLVERStatusConstants")) {
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

REGISTER_RULE(SOLVEREnumsRule)

void SOLVERFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "cusolverDnCreate", "cusolverDnDestroy", "cusolverDnSpotrf_bufferSize",
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
        "cusolverDnCgesvd", "cusolverDnZgesvd", "cusolverDnSsyevd_bufferSize",
        "cusolverDnDsyevd_bufferSize", "cusolverDnSsyevd_bufferSize",
        "cusolverDnCheevd_bufferSize", "cusolverDnZheevd_bufferSize",
        "cusolverDnDsyevd", "cusolverDnSsyevd", "cusolverDnCheevd",
        "cusolverDnZheevd");
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
    AssignPrefix = "(";
    AssignPostfix = ", 0)";
  }

  if (HasDeviceAttr) {
    report(CE->getBeginLoc(), Diagnostics::FUNCTION_CALL_IN_DEVICE, false,
           MapNames::ITFName.at(FuncName), "dpct::dpct_memcpy");
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

      requestHelperFeatureForTypeNames(VarType, VD);

      auto Itr = MapNames::TypeNamesMap.find(VarType);
      if (Itr != MapNames::TypeNamesMap.end())
        VarType = Itr->second;
      PrefixBeforeScope = VarType + " " + VarName + ";" + getNL() + IndentStr +
                          PrefixBeforeScope;
      SourceLocation typeBegin =
          VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
      SourceLocation nameBegin = VD->getLocation();
      SourceLocation nameEnd = Lexer::getLocForEndOfToken(
          nameBegin, 0, *SM, Result.Context->getLangOpts());
      auto replLen =
          SM->getCharacterData(nameEnd) - SM->getCharacterData(typeBegin);
      emplaceTransformation(
          new ReplaceText(typeBegin, replLen, std::move(VarName)));
    } else {
      assert(0 && "Fail to get VarDecl information");
      return;
    }
  }

  if (MapNames::SOLVERFuncReplInfoMap.find(FuncName) !=
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
      // e.g. move arg#1 to arg#0
      // MyFunction(float* a, float* b);
      // ---> MyFunction(float* b, float*a);
      if (isReplIndex(i, ReplInfo.BufferIndexInfo, IndexTemp)) {
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
          requestFeature(HelperFeatureEnum::Memory_get_buffer_T, CE);
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
        requestFeature(HelperFeatureEnum::Memory_async_dpct_free, CE);
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
      report(StmtBegin, Diagnostics::NOERROR_RETURN_COMMA_OP, true);
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
      report(SM->getExpansionLoc(CE->getBeginLoc()),
             Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      emplaceTransformation(new ReplaceStmt(CE, true, "(" + Repl + ", 0)"));
    } else {
      emplaceTransformation(new ReplaceStmt(CE, true, Repl));
    }
  }
}

void SOLVERFunctionCallRule::getParameterEnd(
    const SourceLocation &ParameterEnd, SourceLocation &ParameterEndAfterComma,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  Optional<Token> TokSharedPtr;
  TokSharedPtr = Lexer::findNextToken(ParameterEnd, *(Result.SourceManager),
                                      LangOptions());
  Token TokComma = TokSharedPtr.getValue();
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

// TODO: Refactoring with BLASFunctionCallRule for removing duplication
std::string SOLVERFunctionCallRule::getBufferNameAndDeclStr(
    const Expr *Arg, const ASTContext &AC, const std::string &TypeAsStr,
    SourceLocation SL, std::string &BufferDecl, int DistinctionID) {

  std::string PointerName = ExprAnalysis::ref(Arg);
  std::string BufferTempName =
      getTempNameForExpr(Arg, true, true) + "buf_ct" +
      std::to_string(dpct::DpctGlobalInfo::getSuffixIndexInRuleThenInc());

  // TODO: reinterpret will copy more data
  requestFeature(HelperFeatureEnum::Memory_get_buffer_T, Arg);
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

REGISTER_RULE(SOLVERFunctionCallRule)

void FunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "cudaGetDeviceCount", "cudaGetDeviceProperties", "cudaDeviceReset",
        "cudaSetDevice", "cudaDeviceGetAttribute", "cudaDeviceGetP2PAttribute",
        "cudaDeviceGetPCIBusId", "cudaGetDevice", "cudaDeviceSetLimit",
        "cudaGetLastError", "cudaPeekAtLastError", "cudaDeviceSynchronize",
        "cudaThreadSynchronize", "cudaGetErrorString", "cudaGetErrorName",
        "cudaDeviceSetCacheConfig", "cudaDeviceGetCacheConfig", "clock",
        "cudaOccupancyMaxPotentialBlockSize", "cudaThreadSetLimit",
        "cudaFuncSetCacheConfig", "cudaThreadExit", "cudaDeviceGetLimit",
        "cudaDeviceSetSharedMemConfig", "cudaIpcCloseMemHandle",
        "cudaIpcGetEventHandle", "cudaIpcGetMemHandle",
        "cudaIpcOpenEventHandle", "cudaIpcOpenMemHandle", "cudaSetDeviceFlags",
        "cudaDeviceCanAccessPeer", "cudaDeviceDisablePeerAccess",
        "cudaDeviceEnablePeerAccess", "cudaDriverGetVersion",
        "cudaRuntimeGetVersion", "clock64", "__ldg",
        "cudaFuncSetSharedMemConfig", "cuFuncSetCacheConfig");
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
  // then we need to clear current attributename
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
  std::string Prefix, Suffix;
  if (IsAssigned) {
    Prefix = "(";
    Suffix = ", 0)";
  }

  if (FuncName == "cudaGetDeviceCount") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(
        new InsertBeforeStmt(CE, Prefix + ResultVarName + " = "));
    emplaceTransformation(
        new ReplaceStmt(CE, MapNames::getDpctNamespace() +
                                "dev_mgr::instance().device_count()" + Suffix));
    requestFeature(HelperFeatureEnum::Device_dev_mgr_device_count, CE);
  } else if (FuncName == "cudaGetDeviceProperties") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(
        new ReplaceStmt(CE->getCallee(), Prefix + MapNames::getDpctNamespace() +
                                             "dev_mgr::instance().get_device"));
    emplaceTransformation(new RemoveArg(CE, 0));
    emplaceTransformation(new InsertAfterStmt(
        CE, ".get_device_info(" + ResultVarName + ")" + Suffix));
    requestFeature(HelperFeatureEnum::Device_dev_mgr_get_device, CE);
    requestFeature(
        HelperFeatureEnum::Device_device_ext_get_device_info_return_info, CE);
  } else if (FuncName == "cudaDriverGetVersion" ||
             FuncName == "cudaRuntimeGetVersion") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(
        new InsertBeforeStmt(CE, Prefix + ResultVarName + " = "));

    std::string ReplStr =
        MapNames::getDpctNamespace() + "get_current_device().get_info<" +
        MapNames::getClNamespace() + "info::device::version>()";

    emplaceTransformation(new ReplaceStmt(CE, ReplStr + Suffix));
    report(CE->getBeginLoc(), Warnings::TYPE_MISMATCH, false);
    requestFeature(HelperFeatureEnum::Device_get_current_device, CE);
  } else if (FuncName == "cudaDeviceReset" || FuncName == "cudaThreadExit") {
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    if (isPlaceholderIdxDuplicated(CE))
      return;
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::HFT_CurrentDevice);
    emplaceTransformation(new ReplaceStmt(CE, Prefix + "{{NEEDREPLACED" +
                                                  std::to_string(Index) +
                                                  "}}.reset()" + Suffix));
    requestFeature(HelperFeatureEnum::Device_device_ext_reset, CE);
  } else if (FuncName == "cudaSetDevice") {
    DpctGlobalInfo::setDeviceChangedFlag(true);
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(
        CE->getCallee(), Prefix + MapNames::getDpctNamespace() +
                             "dev_mgr::instance().select_device"));
    if (IsAssigned)
      emplaceTransformation(new InsertAfterStmt(CE, ", 0)"));
    requestFeature(HelperFeatureEnum::Device_dev_mgr_select_device, CE);
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
               false, "recognized by the Intel(R) DPC++ Compatibility Tool");
        return;
      }
    }
    std::string ReplStr{ResultVarName};
    auto StmtStrArg2 = getStmtSpelling(CE->getArg(2));

    if (AttributeName == "cudaDevAttrComputeMode") {
      report(CE->getBeginLoc(), Diagnostics::COMPUTE_MODE, false);
      ReplStr += " = 1";
    } else {
      auto Search = EnumConstantRule::EnumNamesMap.find(AttributeName);
      if (Search == EnumConstantRule::EnumNamesMap.end()) {
        // TODO report migration error
        return;
      }
      requestHelperFeatureForEnumNames(AttributeName, CE);

      ReplStr += " = " + MapNames::getDpctNamespace() +
                 "dev_mgr::instance().get_device(";
      ReplStr += StmtStrArg2;
      ReplStr += ").";
      ReplStr += Search->second;
      ReplStr += "()";
      requestFeature(HelperFeatureEnum::Device_dev_mgr_get_device, CE);
    }
    if (IsAssigned)
      ReplStr = "(" + ReplStr + ", 0)";
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaDeviceGetP2PAttribute") {
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(new ReplaceStmt(CE, ResultVarName + " = 0"));
    report(CE->getBeginLoc(), Comments::NOTSUPPORTED, "P2P Access", false);
  } else if (FuncName == "cudaDeviceGetPCIBusId") {
    report(CE->getBeginLoc(), Comments::NOTSUPPORTED, "Get PCI BusId", false);
  } else if (FuncName == "cudaGetDevice") {
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(new InsertBeforeStmt(CE, ResultVarName + " = "));
    emplaceTransformation(
        new ReplaceStmt(CE, MapNames::getDpctNamespace() +
                                "dev_mgr::instance().current_device_id()"));
    requestFeature(HelperFeatureEnum::Device_dev_mgr_current_device_id, CE);
  } else if (FuncName == "cudaDeviceSynchronize" ||
             FuncName == "cudaThreadSynchronize") {
    if (isPlaceholderIdxDuplicated(CE))
      return;
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::HFT_CurrentDevice);
    std::string ReplStr =
        "{{NEEDREPLACED" + std::to_string(Index) + "}}.queues_wait_and_throw()";
    requestFeature(HelperFeatureEnum::Device_device_ext_queues_wait_and_throw,
                   CE);
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));

  } else if (FuncName == "cudaGetLastError" ||
             FuncName == "cudaPeekAtLastError") {
    if (IsAssigned) {
      report(CE->getBeginLoc(),
             Comments::TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0, false,
             MapNames::ITFName.at(FuncName));
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName),
             "the function call is redundant in DPC++.");
      emplaceTransformation(new ReplaceStmt(CE, true, ""));
    }
  } else if (FuncName == "cudaGetErrorString" ||
             FuncName == "cudaGetErrorName") {
    // Insert warning messages into the spelling locations in case
    // that these functions are contained in macro definitions
    auto Loc = Result.SourceManager->getSpellingLoc(CE->getBeginLoc());
    report(Loc, Comments::TRNA_WARNING_ERROR_HANDLING_API_COMMENTED, false,
           MapNames::ITFName.at(FuncName));
    emplaceTransformation(
        new InsertBeforeStmt(CE, "\"" + FuncName + " not supported\"/*"));
    emplaceTransformation(new InsertAfterStmt(CE, "*/"));
  } else if (FuncName == "clock" || FuncName == "clock64") {
    report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED_SYCL_UNDEF, false,
           FuncName);
    // Add '#include <time.h>' directive to the file only once
    auto Loc = CE->getBeginLoc();
    DpctGlobalInfo::getInstance().insertHeader(Loc, HT_Time);
  } else if (FuncName == "cudaDeviceSetLimit" ||
             FuncName == "cudaThreadSetLimit" ||
             FuncName == "cudaDeviceSetCacheConfig" ||
             FuncName == "cudaDeviceGetCacheConfig") {
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
  } else if (FuncName == "cudaOccupancyMaxPotentialBlockSize") {
    report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, false,
           MapNames::ITFName.at(FuncName));
  } else if (FuncName == "cudaDeviceGetLimit") {
    ExprAnalysis EA;
    EA.analyze(CE->getArg(0));
    auto Arg0Str = EA.getReplacedString();
    std::string ReplStr{"*"};
    ReplStr += Arg0Str;
    ReplStr += " = 0";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    report(CE->getBeginLoc(), Diagnostics::DEVICE_LIMIT_NOT_SUPPORTED, false);
  } else if (FuncName == "cudaDeviceSetSharedMemConfig" ||
             FuncName == "cudaFuncSetSharedMemConfig" ||
             FuncName == "cudaFuncSetCacheConfig" ||
             FuncName == "cuFuncSetCacheConfig") {
    std::string Msg = "DPC++ currently does not support configuring shared "
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
        "DPC++ currently does not support setting flags for devices.";
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
        "DPC++ currently does not support memory access across peer devices.";
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
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
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
  } else if (FuncName == "__ldg") {
    std::ostringstream OS;
    printDerefOp(OS, CE->getArg(0));
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
    report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false, FuncName,
           "there is no correspoinding API in DPC++.");
  } else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected function name: " << FuncName;
    return;
  }
}

REGISTER_RULE(FunctionCallRule)

EventAPICallRule *EventAPICallRule::CurrentRule = nullptr;
void EventAPICallRule::registerMatcher(MatchFinder &MF) {
  auto eventAPIName = [&]() {
    return hasAnyName(
        "cudaEventCreate", "cudaEventCreateWithFlags", "cudaEventDestroy",
        "cudaEventRecord", "cudaEventElapsedTime", "cudaEventSynchronize",
        "cudaEventQuery", "cuEventCreate", "cuEventRecord",
        "cuEventSynchronize", "cuEventQuery", "cuEventElapsedTime");
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
  MemberCallPrinter<const Expr *, StringRef> Printer(Call->getArg(0), false,
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

  for (auto R : Result) {
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

  if (FuncName == "cudaEventCreate" || FuncName == "cudaEventCreateWithFlags" ||
      FuncName == "cudaEventDestroy" || FuncName == "cuEventCreate") {
    auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(new ReplaceStmt(CE,
                                            /*IsProcessMacro*/ true, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(new ReplaceStmt(CE,
                                            /*IsProcessMacro*/ true, ""));
    }
  } else if (FuncName == "cudaEventQuery" || FuncName == "cuEventQuery") {
    if (getEventQueryTraversal().startFromQuery(CE))
      return;

    if (!isEventElapsedTimeFollowed(CE)) {
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
    std::string ReplStr = "(int)" + EA.getReplacedString() + ".get_info<" +
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
    bool NeedReport = false;
    std::string ReplStr{getStmtSpelling(CE->getArg(0))};
    ReplStr += ".wait_and_throw()";
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      NeedReport = true;
    }

    auto &Context = dpct::DpctGlobalInfo::getContext();
    const auto &TM = ReplaceStmt(CE, ReplStr);
    const auto R = TM.getReplacement(Context);
    DpctGlobalInfo::getInstance().insertEventSyncTypeInfo(R, NeedReport,
                                                          IsAssigned);
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

void EventAPICallRule::handleEventRecord(const CallExpr *CE,
                                         const MatchFinder::MatchResult &Result,
                                         bool IsAssigned) {
  report(CE->getBeginLoc(), Diagnostics::TIME_MEASUREMENT_FOUND, false);
  DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), HT_Chrono);
  std::ostringstream Repl;

  const ValueDecl *MD = getDecl(CE->getArg(0));
  if (!MD)
    return;
  // Insert the helper variable right after the event variables
  static std::set<std::pair<const Decl *, std::string>> DeclDupFilter;
  auto &SM = DpctGlobalInfo::getSourceManager();
  std::string InsertStr;
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
    emplaceTransformation(new InsertAfterDecl(MD, std::move(InsertStr)));
  }

  // Replace event recording with std::chrono timing
  Repl << getTimePointNameForEvent(CE->getArg(0), false)
       << " = std::chrono::steady_clock::now()";
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
        StmtStr = ArgName + " = " + Str;
      } else {
        std::string Str = StreamName + "->" + "ext_oneapi_submit_barrier()";
        StmtStr = ArgName + " = " + Str;
      }
      StmtStr = "(" + StmtStr + ", 0)";

      auto ReplWithSubmitBarrier =
          ReplaceStmt(CE, StmtStr).getReplacement(Context);
      auto ReplWithoutSubmitBarrier =
          ReplaceStmt(CE, "0").getReplacement(Context);
      DpctGlobalInfo::getInstance().insertTimeStubTypeInfo(
          ReplWithSubmitBarrier, ReplWithoutSubmitBarrier);
    }

    report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_ZERO, false);
    auto OuterStmt = findNearestNonExprNonDeclAncestorStmt(CE);
    Repl << "; ";
    if (IndentLoc.isMacroID())
      IndentLoc = SM.getExpansionLoc(IndentLoc);
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
      std::string ReplStr = ";";
      if (isInMacroDefinition(MD->getBeginLoc(), MD->getEndLoc())) {
        ReplStr += "\\";
      }
      if (IsDefaultStream) {
        if (isPlaceholderIdxDuplicated(CE))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);

        std::string Str = ArgName + " = {{NEEDREPLACEQ" +
                          std::to_string(Index) +
                          "}}.ext_oneapi_submit_barrier()";
        ReplStr += getNL();
        ReplStr += getIndent(IndentLoc, SM).str();
        ReplStr += Str;
      } else {
        std::string Str =
            ArgName + " = " + StreamName + "->ext_oneapi_submit_barrier()";
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

void EventAPICallRule::handleEventElapsedTime(bool IsAssigned) {
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
    Temp << "(" << Repl.str() << ", 0)";
    Repl = std::move(Temp);
    report(TimeElapsedCE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP,
           false);
  }
  const std::string Name =
      TimeElapsedCE->getCalleeDecl()->getAsFunction()->getNameAsString();
  emplaceTransformation(new ReplaceStmt(TimeElapsedCE, std::move(Repl.str())));
  handleTimeMeasurement();
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

  // Skip statements before RecordBeginLoc or after RecordEndLoc
  if (KCallLoc < RecordBeginLoc || KCallLoc > RecordEndLoc)
    return;

  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
    bool NeedWait = false;
    // In usmnone mode, if cudaThreadSynchronize apears after kernel call,
    // kernel wait is not needed.
    NeedWait = ThreadSyncLoc > KCallLoc;

    if (KCallLoc > RecordBeginLoc && !NeedWait) {
      if (IsKernelSync) {
        K->setEvent(ExprAnalysis::ref(EventExpr));
        K->setSync();
      } else {
        Queues2Wait.emplace_back(MapNames::getDpctNamespace() +
                                     "get_current_device()."
                                     "queues_wait_and_throw();",
                                 nullptr);
        requestFeature(HelperFeatureEnum::Device_get_current_device, KCall);
        requestFeature(
            HelperFeatureEnum::Device_device_ext_queues_wait_and_throw, KCall);
      }
    }
  }

  if (USMLevel == UsmLevel::UL_Restricted) {
    if (KCallLoc > RecordBeginLoc) {
      if (!IsKernelInLoopStmt && !IsKernelSync) {
        K->setEvent(ExprAnalysis::ref(EventExpr));
        Events2Wait.push_back(ExprAnalysis::ref(EventExpr) + ".wait();");
      } else if (IsKernelSync) {
        K->setEvent(ExprAnalysis::ref(EventExpr));
        K->setSync();
      } else {
        std::string WaitQueue = MapNames::getDpctNamespace() +
                                "get_current_device()."
                                "queues_wait_and_throw();";
        Events2Wait.push_back(WaitQueue);
        requestFeature(HelperFeatureEnum::Device_get_current_device, KCall);
        requestFeature(
            HelperFeatureEnum::Device_device_ext_queues_wait_and_throw, KCall);
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
          requestFeature(HelperFeatureEnum::Device_get_current_device, Call);
          requestFeature(
              HelperFeatureEnum::Device_device_ext_queues_wait_and_throw, Call);
        } else {
          auto ArgName = getStmtSpelling(StreamArg);
          Queues2Wait.emplace_back(ArgName + "->wait();", nullptr);
        }
      }
    }
  }
}

REGISTER_RULE(EventAPICallRule)

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
                      "cuStreamSynchronize", "cuStreamWaitEvent");
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

    if (isPlaceholderIdxDuplicated(CE))
      return;
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::HFT_CurrentDevice);
    ReplStr += " = " + getNewQueue(Index);
    requestFeature(HelperFeatureEnum::Device_device_ext_create_queue, CE);
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
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
    requestFeature(HelperFeatureEnum::Device_device_ext_destroy_queue, CE);
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
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
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
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
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
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
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
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
      ReplStr = StmtStr1 + ".wait()";
    } else {
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
      ReplStr = StmtStr1 + " = " + StmtStr0 + "ext_oneapi_submit_barrier({" +
                StmtStr1 + "})";
    }
    if (IsAssigned) {
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
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
      ReplStr = "(" + ReplStr + ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
    DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), HT_Future);
  } else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected function name: " << FuncName;
    return;
  }
}

REGISTER_RULE(StreamAPICallRule)

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
    if (!FD->isTemplateInstantiation())
      DpctGlobalInfo::getInstance().insertKernelCallExpr(KCall);

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
                     Diagnostics::SIZEOF_WARNING, false);
            }
          }
        }
      }
    }

    if (!FD)
      return;

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
      report(LaunchKernelCall->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
    }
  }
}

// Find and remove the semicolon after the kernel call
void KernelCallRule::removeTrailingSemicolon(
    const CallExpr *KCall,
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto &SM = (*Result.Context).getSourceManager();
  auto KELoc = KCall->getEndLoc();
  if (KELoc.isMacroID() && !isOuterMostMacro(KCall)) {
    KELoc = SM.getImmediateSpellingLoc(KELoc);
  }
  KELoc = SM.getExpansionRange(KELoc).getEnd();
  auto Tok = Lexer::findNextToken(KELoc, SM, LangOptions()).getValue();
  if (Tok.is(tok::TokenKind::semi))
    emplaceTransformation(new ReplaceToken(Tok.getLocation(), ""));
}

REGISTER_RULE(KernelCallRule)

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

  MF.addMatcher(cxxMemberCallExpr(hasDescendant(memberExpr(hasDescendant(
                                      implicitCastExpr().bind("impCastExpr")))))
                    .bind("cxxMemberCall"),
                this);
  MF.addMatcher(
      functionDecl(anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))
          .bind("deviceFuncDecl"),
      this);
}

void DeviceFunctionDeclRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {

  if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "deviceFuncDecl")) {
    if (FD->isTemplateInstantiation())
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

  if (auto CXXMCE =
          getAssistNodeAsType<CXXMemberCallExpr>(Result, "cxxMemberCall")) {
    if (auto ICE =
            getAssistNodeAsType<ImplicitCastExpr>(Result, "impCastExpr")) {
      auto Type = ICE->getType().getAsString();
      if (Type == "const class cooperative_groups::__v1::grid_group") {
        if (auto DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr())) {

          if (!DpctGlobalInfo::useNdRangeBarrier()) {
            auto Name = DRE->getNameInfo().getName().getAsString();
            report(CXXMCE->getBeginLoc(), Diagnostics::ND_RANGE_BARRIER, false,
                   Name);
            return;
          }

          std::string ReplStr = "dpct::experimental::nd_range_barrier(" +
                                DpctGlobalInfo::getItem(CXXMCE) + ", " +
                                DpctGlobalInfo::getSyncName() + ")";

          emplaceTransformation(new ReplaceStmt(CXXMCE, ReplStr));
          requestFeature(HelperFeatureEnum::Util_nd_range_barrier, CXXMCE);
        }
      }
    }
  }

  std::shared_ptr<DeviceFunctionInfo> FuncInfo;
  auto FD = getAssistNodeAsType<FunctionDecl>(Result, "funcDecl");
  if (!FD || (FD->hasAttr<CUDADeviceAttr>() && FD->hasAttr<CUDAHostAttr>() &&
              DpctGlobalInfo::getRunRound() == 1))
    return;
  if (FD->isVariadic()) {
    report(FD->getBeginLoc(), Warnings::DEVICE_VARIADIC_FUNCTION, false);
  }
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
  }

  if (auto CE = getAssistNodeAsType<CallExpr>(Result, "callExpr")) {
    FuncInfo->addCallee(CE);
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

  if (auto Var = getAssistNodeAsType<VarDecl>(Result, "varGrid")) {

    if (!Var->getInit())
      return;

    if (auto EWC = dyn_cast<ExprWithCleanups>(Var->getInit())) {
      auto CXXCExpr = dyn_cast<CXXConstructExpr>(EWC->getSubExpr());
      if (!CXXCExpr)
        return;

      auto IgnoreUSIS = CXXCExpr->IgnoreUnlessSpelledInSource();
      if (!IgnoreUSIS)
        return;

      auto CE = dyn_cast<CallExpr>(IgnoreUSIS);
      if (!CE)
        return;

      if (CE->getType().getAsString() !=
          "class cooperative_groups::__v1::grid_group")
        return;

      if (!DpctGlobalInfo::useNdRangeBarrier()) {
        auto Name = Var->getNameAsString();
        report(Var->getBeginLoc(), Diagnostics::ND_RANGE_BARRIER, false, Name);
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
                .getValue();
      End = Tok.getLocation();

      auto Length = SM.getFileOffset(End) - SM.getFileOffset(Begin);

      // Remove statement "cg::grid_group grid = cg::this_grid();"
      emplaceTransformation(new ReplaceText(Begin, Length, ""));
    }
  }
}

REGISTER_RULE(DeviceFunctionDeclRule)

void GlibcMemoryAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {

  MF.addMatcher(callExpr(hasParent(cStyleCastExpr().bind("CStyleCastExpr")),
                         callee(functionDecl(hasName("malloc"))))
                    .bind("mallocCallExpr"),
                this);

  MF.addMatcher(
      callExpr(callee(functionDecl(hasName("free")))).bind("FreeCallExpr"),
      this);
}

template <typename T> const T *GlibcMemoryAPIRule::getAncestor(const Stmt *CE) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*CE);
  while (Parents.size() == 1) {
    auto *Parent = Parents[0].get<T>();
    if (Parent) {
      return Parent;
    } else {
      Parents = Context.getParents(Parents[0]);
    }
  }
  return nullptr;
}

// This function is used to migrate malloc to new.
// \p ReplStmt is the statement to be processed
// \p DD is the variable assigned by \p CE,
// Take a case for example:
// "cudaEvent_t *kernelEvent = (cudaEvent_t *) malloc(*sizeof(cudaEvent_t));".
// ReplStmt stands for "(cudaEvent_t *) malloc(*sizeof(cudaEvent_t))",
// DD is variable "kernelEvent", and  CE is CallExpr "malloc".
void GlibcMemoryAPIRule::processMalloc(const Stmt *ReplStmt,
                                       const DeclaratorDecl *DD,
                                       const CallExpr *CE) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto &SM = DpctGlobalInfo::getSourceManager();
  std::string Repl;
  if (const BinaryOperator *BO = dyn_cast<BinaryOperator>(CE->getArg(0))) {
    if (BO->getOpcode() == BinaryOperatorKind::BO_Mul) {
      if (!isContainMacro(BO->getLHS()) &&
          isSameSizeofTypeWithTypeStr(BO->getLHS(),
                                      MapNames::getClNamespace() + "event")) {
        // case 1: sizeof(b) * a
        ArgumentAnalysis AA;
        AA.setCallSpelling(BO);
        AA.analyze(BO->getRHS());
        Repl = AA.getRewritePrefix() + AA.getRewriteString() +
               AA.getRewritePostfix();
      } else if (!isContainMacro(BO->getRHS()) &&
                 isSameSizeofTypeWithTypeStr(
                     BO->getRHS(), MapNames::getClNamespace() + "event")) {
        // case 2: a * sizeof(b)
        ArgumentAnalysis AA;
        AA.setCallSpelling(BO);
        AA.analyze(BO->getLHS());
        Repl = AA.getRewritePrefix() + AA.getRewriteString() +
               AA.getRewritePostfix();
      }
    }
  }

  if (Repl.empty()) {
    dpct::ExprAnalysis EA(CE->getArg(0));
    auto Str = EA.getReplacedString();
    std::string Size = "sizeof(" + MapNames::getClNamespace() + "event)";
    Str = "(" + Str + ")/";
    Repl = Str + Size;
  }
  std::string ReplStr =
      "new " + MapNames::getClNamespace() + "event[" + Repl + "]";
  auto R = ReplaceStmt(ReplStmt, ReplStr).getReplacement(Context);
  DpctGlobalInfo::getInstance().insertReplMalloc(
      R,
      DpctGlobalInfo::getLocInfo(SM.getExpansionLoc(DD->getBeginLoc())).second);
}

void GlibcMemoryAPIRule::processFree(const CallExpr *CE) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto &SM = DpctGlobalInfo::getSourceManager();
  std::string ReplStr = "delete [] " + getStmtSpelling(CE->getArg(0));

  auto Arg = CE->getArg(0);
  if (auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts())) {
    if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      // To process free called in C ordinary function.
      auto R = ReplaceStmt(CE, ReplStr).getReplacement(Context);
      DpctGlobalInfo::getInstance().insertReplFree(
          R, DpctGlobalInfo::getLocInfo(SM.getExpansionLoc(VD->getBeginLoc()))
                 .second);
    }
  } else if (auto ME = dyn_cast<MemberExpr>(Arg->IgnoreImpCasts())) {
    if (auto FD = dyn_cast<FieldDecl>(ME->getMemberDecl())) {
      // To process free called in member function.
      auto R = ReplaceStmt(CE, ReplStr).getReplacement(Context);
      DpctGlobalInfo::getInstance().insertReplFree(
          R, DpctGlobalInfo::getLocInfo(SM.getExpansionLoc(FD->getBeginLoc()))
                 .second);
    }
  }
}

void GlibcMemoryAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {

  if (auto CE = getAssistNodeAsType<CallExpr>(Result, "mallocCallExpr")) {
    if (auto CSCE =
            getAssistNodeAsType<CStyleCastExpr>(Result, "CStyleCastExpr")) {
      auto Type = CSCE->getType().getAsString();
      if (Type != "cudaEvent_t *")
        return;

      if (auto VD = getAncestor<VarDecl>(CE)) {
        // To process case like:
        // cudaEvent_t *kernelEvent = (cudaEvent_t *)
        // malloc(sizeof(cudaEvent_t));
        processMalloc(VD->getInit(), VD, CE);
      } else {
        // To process case like:
        // cudaEvent_t *kernelEvent;
        // kernelEvent = (cudaEvent_t *) malloc(sizeof(cudaEvent_t));
        auto BO = getAncestor<BinaryOperator>(CE);
        if (BO && BO->getOpcode() == BO_Assign) {
          if (auto DRE = dyn_cast<DeclRefExpr>(BO->getLHS())) {
            if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
              // To process malloc called in C ordinary function.
              processMalloc(CSCE, VD, CE);
            }
          } else if (auto ME = dyn_cast<MemberExpr>(BO->getLHS())) {
            if (auto FD = dyn_cast<FieldDecl>(ME->getMemberDecl())) {
              // To process malloc called in member function.
              processMalloc(CSCE, FD, CE);
            }
          }
        }
      }
    }
  }

  if (auto CE = getAssistNodeAsType<CallExpr>(Result, "FreeCallExpr")) {
    if (CE->getNumArgs() != 1)
      return;

    if (auto IIC = CE->getArg(0)->IgnoreImpCasts()) {
      auto Type = IIC->getType().getAsString();
      if (Type != "cudaEvent_t *")
        return;
    }
    processFree(CE);
  }
}

REGISTER_RULE(GlibcMemoryAPIRule)

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

void MemVarRule::processDeref(const Stmt *S, ASTContext &Context) {
  auto Parents = Context.getParents(*S);
  if (Parents.size()) {
    auto &Parent = Parents[0];
    if (auto ICE = Parent.get<ImplicitCastExpr>()) {
      processDeref(ICE, Context);
    } else if (auto ME = Parent.get<MemberExpr>()) {
      emplaceTransformation(new ReplaceToken(ME->getOperatorLoc(), "->"));
    } else if (auto CDSME = Parent.get<CXXDependentScopeMemberExpr>()) {
      emplaceTransformation(new ReplaceToken(CDSME->getOperatorLoc(), "->"));
    } else if (Parent.get<BinaryOperator>() || Parent.get<CallExpr>() ||
               Parent.get<CXXConstructExpr>() || Parent.get<ParenExpr>()) {
      emplaceTransformation(new InsertBeforeStmt(S, "*"));

    } else if (auto UO = Parent.get<UnaryOperator>()) {
      if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf) {
        emplaceTransformation(new ReplaceToken(UO->getOperatorLoc(), ""));
      } else {
        insertAroundStmt(S, "(*", ")");
      }
    } else {
      insertAroundStmt(S, "(*", ")");
    }
  }
}

void MemVarRule::previousHCurrentD(const VarDecl *VD, tooling::Replacement &R) {
  // 1. emit DPCT1055 warning
  // 2. add a new variable for host
  // 3. insert dpct::constant_memory and add the info from that replacment into
  //    current replacemrnt.
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
  //        replacement will be saved and it will not contain the addational
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
          VD, MemVarInfo::buildMemVarInfo(VD)->getDeclarationReplacement())) {
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
  std::string Warning =
      "The Intel\\(R\\) DPC\\+\\+ Compatibility Tool did not detect the "
      "variable "
      "[_a-zA-Z][_a-zA-Z0-9]+ used in device code. If this variable is also "
      "used in device code, you need to rewrite the code.";
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
    auto DS = Info->getDeclStmtOfVarType();
    SourceLocation InsertSL = SM.getExpansionLoc(DS->getEndLoc());
    InsertSL = InsertSL.getLocWithOffset(1); // this token is ;
    std::string InsertStr = getNL() + getIndent(InsertSL, SM).str() +
                            NewTypeName + "* " + Info->getName() + " = (" +
                            NewTypeName + "*)" + Info->getNameAppendSuffix() +
                            ";";
    emplaceTransformation(new InsertText(InsertSL, std::move(InsertStr)));
  } else if (auto DS = Info->getDeclStmtOfVarType()) {
    // remove var decl
    emplaceTransformation(ReplaceVarDecl::getVarDeclReplacement(
        MemVar, Info->getDeclarationReplacement()));

    Info->setLocalTypeName(Info->getType()->getBaseName());
    // add typecast for the __shared__ variable, since after migration the
    // __shared__ variable type will be uint8_t*
    SourceLocation InsertSL = SM.getExpansionLoc(DS->getEndLoc());
    InsertSL = InsertSL.getLocWithOffset(1); // this token is ;
    std::string InsertStr = getNL() + getIndent(InsertSL, SM).str() +
                            Info->getType()->getBaseName() + "* " +
                            Info->getName() + " = (" +
                            Info->getType()->getBaseName() + "*)" +
                            Info->getNameAppendSuffix() + ";";
    emplaceTransformation(new InsertText(InsertSL, std::move(InsertStr)));
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
    if (TM->getConstantFlag() == dpct::ConstantFlagType::Device &&
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
        if (R.second->getConstantFlag() == dpct::ConstantFlagType::Host &&
            R.second->getConstantOffset() == TM->getConstantOffset()) {
          // using flag and the offset of __constant__ to link
          // R(dcpt::constant_memery)  and R(reomving __constant__) from
          // previous exection previous is host, current is device:
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
      auto RVD = ReplaceVarDecl::getVarDeclReplacement(
          MemVar, Info->getDeclarationReplacement());
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
    if (TM->getConstantFlag() == dpct::ConstantFlagType::Host &&
        TM->getLineBeginOffset() == OffsetOfLineBegin) {
      // current __constant__ variable used in host, using OffsetOfLineBegin
      // link the R(reomving __constant__) and here

      // 1. check previous processed replacements, if found, do not check
      // info from yaml
      if (!FileInfo->getRepls())
        return false;
      auto &M = FileInfo->getRepls()->getReplMap();
      for (auto &R : M) {
        if (R.second->getConstantFlag() == dpct::ConstantFlagType::Device &&
            R.second->getConstantOffset() == TM->getConstantOffset()) {
          // using flag and the offset of __constant__ to link previous
          // exection previous is device, current is host:
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
            // R(reomving __constant__) from previous exection previous is
            // device, current is host:
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
      // The constant offset will be used in previousHCurrentD to distingush
      // uncecessary warnings.
      if (report(VD->getBeginLoc(), Diagnostics::HOST_CONSTANT, false,
                 VD->getNameAsString())) {
        TransformSet->back()->setConstantOffset(TM->getConstantOffset());
      }
    }
  }
  return false;
}

void MemVarRule::runRule(const MatchFinder::MatchResult &Result) {
  std::string CanonicalType;
  if (auto MemVar = getAssistNodeAsType<VarDecl>(Result, "var")) {
    if (isCubVar(MemVar)) {
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
          MemVar, Info->getDeclarationReplacement()));
    }
  }
  auto MemVarRef = getNodeAsType<DeclRefExpr>(Result, "used");
  auto Func = getAssistNodeAsType<FunctionDecl>(Result, "func");
  auto Decl = getAssistNodeAsType<VarDecl>(Result, "decl");
  DpctGlobalInfo &Global = DpctGlobalInfo::getInstance();
  if (MemVarRef && Func && Decl) {
    if (isCubVar(Decl)) {
      return;
    }
    auto VD = dyn_cast<VarDecl>(MemVarRef->getDecl());
    if (Func->isImplicit() ||
        Func->getTemplateSpecializationKind() == TSK_ImplicitInstantiation)
      return;
    if (VD == nullptr)
      return;

    auto Var = Global.findMemVarInfo(VD);
    if (Func->hasAttr<CUDAGlobalAttr>() ||
        (Func->hasAttr<CUDADeviceAttr>() && !Func->hasAttr<CUDAHostAttr>())) {
      if (Var)
        DeviceFunctionDecl::LinkRedecls(Func)->addVar(Var);
      if (!VD->getType()->isArrayType() && !VD->hasAttr<CUDAConstantAttr>()) {
        processDeref(MemVarRef, *Result.Context);
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

REGISTER_RULE(MemVarRule)

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
/// \param [in] E The expression need to be analyzed.
/// \param [in] Arg0Str The original string of the first argument of the malloc.
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
      requestFeature(HelperFeatureEnum::Memory_device_memory_get_ptr_q, E);
      Out << "*" << StreamStr;
    } else {
      requestFeature(HelperFeatureEnum::Memory_device_memory_get_ptr, E);
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
  // ReplType will be used as the template arguement in memory API.
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

  requestFeature(HelperFeatureEnum::Image_image_matrix, C);
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
      // Leverage CallExprRewritter to migrate the USM verison
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
      std::ostringstream OS;
      std::string Type;
      if (IsAssigned)
        OS << "(";
      printDerefOp(OS, C->getArg(0)->IgnoreCasts()->IgnoreParens(), &Type);
      if (Type != "NULL TYPE" && Type != "void *")
        OS << " = (" << getFinalCastTypeNameStr(Type) << ")";
      else
        OS << " = ";
      auto LocInfo = DpctGlobalInfo::getLocInfo(C->getBeginLoc());
      auto Action = [LocInfo, C, IsAssigned]() {
        requestFeature(HelperFeatureEnum::Memory_dpct_malloc, LocInfo.first);
        if (IsAssigned) {
          DiagnosticsUtils::report(LocInfo.first, LocInfo.second,
                                   Diagnostics::NOERROR_RETURN_COMMA_OP, true,
                                   false);
        }
      };
      auto Info = std::make_shared<PriorityReplInfo>();
      auto &Context = DpctGlobalInfo::getContext();
      Info->RelatedAction.emplace_back(Action);
      Info->Repls.emplace_back(
          InsertBeforeStmt(C, OS.str()).getReplacement(Context));
      Info->Repls.emplace_back(
          ReplaceCalleeName(C, MapNames::getDpctNamespace() + "dpct_malloc")
              .getReplacement(Context));
      if (auto TM = removeArg(C, 0, *Result.SourceManager))
        Info->Repls.push_back(TM->getReplacement(Context));
      if (IsAssigned) {
        Info->Repls.emplace_back(
            InsertAfterStmt(C, ", 0)").getReplacement(Context));
      }
      DpctGlobalInfo::addPriorityReplInfo(
          LocInfo.first + std::to_string(LocInfo.second), Info);
    }
  } else if (Name == "cudaHostAlloc" || Name == "cudaMallocHost" ||
             Name == "cuMemHostAlloc") {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  } else if (Name == "cudaMallocManaged") {
    if (USMLevel == UsmLevel::UL_Restricted) {
      buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
      mallocMigrationWithTransformation(
          *Result.SourceManager, C, Name,
          MapNames::getClNamespace() + "malloc_shared",
          "{{NEEDREPLACEQ" + std::to_string(Index) + "}}");
      emplaceTransformation(removeArg(C, 2, *Result.SourceManager));
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
        emplaceTransformation(new InsertBeforeStmt(C, "("));
      mallocMigrationWithTransformation(
          *Result.SourceManager, C, Name,
          MapNames::getClNamespace() + "malloc_device",
          "{{NEEDREPLACEQ" + std::to_string(Index) + "}}", true, 2);
      if (IsAssigned) {
        emplaceTransformation(new InsertAfterStmt(C, ", 0)"));
        report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      }
    } else {
      ExprAnalysis EA(C->getArg(2));
      EA.analyze();
      std::ostringstream OS;
      std::string Type;
      if (IsAssigned)
        OS << "(";
      printDerefOp(OS, C->getArg(2)->IgnoreCasts()->IgnoreParens(), &Type);
      if (Type != "NULL TYPE" && Type != "void *")
        OS << " = (" << Type << ")";
      else
        OS << " = ";

      emplaceTransformation(new InsertBeforeStmt(C, OS.str()));
      emplaceTransformation(new ReplaceCalleeName(
          C, MapNames::getDpctNamespace() + "dpct_malloc"));
      requestFeature(HelperFeatureEnum::Memory_dpct_malloc, C);
      if (IsAssigned) {
        emplaceTransformation(new InsertAfterStmt(C, ", 0)"));
        report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      }
    }
  } else if (Name == "cudaMallocPitch" || Name == "cudaMalloc3D") {
    std::ostringstream OS;
    std::string Type;
    if (IsAssigned)
      OS << "(";
    printDerefOp(OS, C->getArg(0)->IgnoreCasts()->IgnoreParens(), &Type);
    if (Name != "cudaMalloc3D" && Type != "NULL TYPE" && Type != "void *")
      OS << " = (" << Type << ")";
    else
      OS << " = ";

    requestFeature(HelperFeatureEnum::Memory_dpct_malloc_3d, C);
    requestFeature(HelperFeatureEnum::Memory_dpct_malloc_2d, C);
    emplaceTransformation(new InsertBeforeStmt(C, OS.str()));
    emplaceTransformation(
        new ReplaceCalleeName(C, MapNames::getDpctNamespace() + "dpct_malloc"));
    emplaceTransformation(removeArg(C, 0, *Result.SourceManager));
    std::ostringstream OS2;
    printDerefOp(OS2, C->getArg(1)->IgnoreCasts()->IgnoreParens());
    if (Name == "cudaMallocPitch") {
      emplaceTransformation(new ReplaceStmt(C->getArg(1), OS2.str()));
    }
    if (IsAssigned) {
      emplaceTransformation(new InsertAfterStmt(C, ", 0)"));
      report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
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
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  std::string ReplaceStr;
  // Detect if there is Async in the func name and crop the async substr
  std::string NameRef = Name;
  bool IsAsync = false;
  size_t AsyncLoc = NameRef.find("Async");
  if (AsyncLoc != std::string::npos) {
    IsAsync = true;
    NameRef = NameRef.substr(0, AsyncLoc);
    ReplaceStr = MapNames::getDpctNamespace() + "async_dpct_memcpy";
    requestFeature(HelperFeatureEnum::Memory_async_dpct_memcpy, C);
    requestFeature(HelperFeatureEnum::Memory_async_dpct_memcpy_2d, C);
    requestFeature(HelperFeatureEnum::Memory_async_dpct_memcpy_3d, C);
  } else {
    ReplaceStr = MapNames::getDpctNamespace() + "dpct_memcpy";
    requestFeature(HelperFeatureEnum::Memory_dpct_memcpy, C);
    requestFeature(HelperFeatureEnum::Memory_dpct_memcpy_2d, C);
    requestFeature(HelperFeatureEnum::Memory_dpct_memcpy_3d, C);
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
             NameRef.rfind("cuMemcpyDtoH", 0) == 0 ||
             NameRef.rfind("cuMemcpyHtoD", 0) == 0) {
    if (!NameRef.compare("cudaMemcpy")) {
      handleDirection(C, 3);
    }
    std::string AsyncQueue;
    size_t QueueIndex = NameRef.compare("cudaMemcpy") ? 3 : 4;
    if (C->getNumArgs() > QueueIndex &&
        !C->getArg(QueueIndex)->isDefaultArgument()) {
      if (!isPredefinedStreamHandle(C->getArg(QueueIndex)))
        AsyncQueue = ExprAnalysis::ref(C->getArg(QueueIndex));
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
        ReplaceStr = AsyncQueue + "->memcpy";
      }
    } else {
      if (!NameRef.compare("cudaMemcpy")) {
        handleAsync(C, 4, Result);
      } else {
        emplaceTransformation(
            new InsertAfterStmt(C->getArg(2), ", dpct::automatic"));
        handleAsync(C, 3, Result);
      }
    }
  }

  if (ULExpr)
    emplaceTransformation(new ReplaceToken(
        ULExpr->getBeginLoc(), ULExpr->getEndLoc(), std::move(ReplaceStr)));
  else
    emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
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

  std::string ReplaceStr;
  StringRef NameRef(Name);
  bool IsAsync = NameRef.endswith("Async");
  if (IsAsync) {
    NameRef = NameRef.drop_back(5 /* len of "Async" */);
    ReplaceStr = MapNames::getDpctNamespace() + "async_dpct_memcpy";
    requestFeature(HelperFeatureEnum::Memory_async_dpct_memcpy, C);
    requestFeature(HelperFeatureEnum::Memory_async_dpct_memcpy_2d, C);
    requestFeature(HelperFeatureEnum::Memory_async_dpct_memcpy_3d, C);
  } else {
    ReplaceStr = MapNames::getDpctNamespace() + "dpct_memcpy";
    requestFeature(HelperFeatureEnum::Memory_dpct_memcpy, C);
    requestFeature(HelperFeatureEnum::Memory_dpct_memcpy_2d, C);
    requestFeature(HelperFeatureEnum::Memory_dpct_memcpy_3d, C);
  }

  auto &SM = *Result.SourceManager;
  if (NameRef == "cudaMemcpy2DArrayToArray") {
    insertToPitchedData(C, 0);
    aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
    insertToPitchedData(C, 3);
    aggregate3DVectorClassCtor(C, "id", 4, "0", SM);
    aggregate3DVectorClassCtor(C, "range", 6, "1", SM);
    emplaceTransformation(removeArg(C, 8, SM));
  } else if (NameRef == "cudaMemcpy2DFromArray") {
    handleAsync(C, 8, Result);
    emplaceTransformation(removeArg(C, 7, *Result.SourceManager));
    aggregatePitchedData(C, 0, 1, SM);
    insertZeroOffset(C, 2);
    insertToPitchedData(C, 2);
    aggregate3DVectorClassCtor(C, "id", 3, "0", SM);
    aggregate3DVectorClassCtor(C, "range", 5, "1", SM);
  } else if (NameRef == "cudaMemcpy2DToArray") {
    handleAsync(C, 8, Result);
    emplaceTransformation(removeArg(C, 7, *Result.SourceManager));
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
    emplaceTransformation(removeArg(C, 7, SM));
  } else if (NameRef == "cudaMemcpyFromArray") {
    handleAsync(C, 6, Result);
    emplaceTransformation(removeArg(C, 5, SM));
    aggregatePitchedData(C, 0, 4, SM, true);
    insertZeroOffset(C, 1);
    insertToPitchedData(C, 1);
    aggregate3DVectorClassCtor(C, "id", 2, "0", SM);
    aggregate3DVectorClassCtor(C, "range", 4, "1", SM, 1);
  } else if (NameRef == "cudaMemcpyToArray") {
    handleAsync(C, 6, Result);
    emplaceTransformation(removeArg(C, 5, SM));
    insertToPitchedData(C, 0);
    aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
    aggregatePitchedData(C, 3, 4, SM, true);
    insertZeroOffset(C, 4);
    aggregate3DVectorClassCtor(C, "range", 4, "1", SM, 1);
  }

  if (ULExpr)
    emplaceTransformation(new ReplaceToken(
        ULExpr->getBeginLoc(), ULExpr->getEndLoc(), std::move(ReplaceStr)));
  else
    emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
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
      requestHelperFeatureForEnumNames(DirectionName, C);
      Direction = nullptr;
      DirectionName = Search->second;
    }
  }

  DpctGlobalInfo &Global = DpctGlobalInfo::getInstance();
  auto MallocInfo = Global.findCudaMalloc(C->getArg(1));
  auto VD = CudaMallocInfo::getDecl(C->getArg(0));
  if (MallocInfo && VD) {
    if (auto Var = Global.findMemVarInfo(VD)) {
      requestFeature(HelperFeatureEnum::Memory_device_memory_assign, C);
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
      requestFeature(HelperFeatureEnum::Memory_dpct_memcpy, C);
      ReplaceStr = MapNames::getDpctNamespace() + "dpct_memcpy";
    }
  } else {
    if (C->getNumArgs() == 6 && !C->getArg(5)->isDefaultArgument()) {
      if (!isPredefinedStreamHandle(C->getArg(5))) {
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
      requestFeature(HelperFeatureEnum::Memory_async_dpct_memcpy, C);
      ReplaceStr = MapNames::getDpctNamespace() + "async_dpct_memcpy";
    }
  }

  if (ULExpr) {
    emplaceTransformation(new ReplaceToken(
        ULExpr->getBeginLoc(), ULExpr->getEndLoc(), std::move(ReplaceStr)));
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
      Repl << MapNames::getClNamespace() + "free(" << ArgStr
           << ", {{NEEDREPLACEQ" + std::to_string(Index) + "}})";
      emplaceTransformation(new ReplaceStmt(C, std::move(Repl.str())));
    } else {
      requestFeature(HelperFeatureEnum::Memory_dpct_free, C);
      emplaceTransformation(
          new ReplaceCalleeName(C, MapNames::getDpctNamespace() + "dpct_free"));
    }
  } else if (Name == "cudaFreeHost" || Name == "cuMemFreeHost") {
    if (USMLevel == UsmLevel::UL_Restricted) {
      ExprAnalysis EA;
      EA.analyze(C->getArg(0));
      std::ostringstream Repl;
      buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
      Repl << MapNames::getClNamespace() + "free(" << EA.getReplacedString()
           << ", {{NEEDREPLACEQ" + std::to_string(Index) + "}})";
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

  std::string ReplaceStr;
  StringRef NameRef(Name);
  bool IsAsync = NameRef.endswith("Async");
  if (IsAsync) {
    NameRef = NameRef.drop_back(5 /* len of "Async" */);
    ReplaceStr = MapNames::getDpctNamespace() + "async_dpct_memset";
    requestFeature(HelperFeatureEnum::Memory_async_dpct_memset, C);
    requestFeature(HelperFeatureEnum::Memory_async_dpct_memset_2d, C);
    requestFeature(HelperFeatureEnum::Memory_async_dpct_memset_3d, C);
  } else {
    ReplaceStr = MapNames::getDpctNamespace() + "dpct_memset";
    requestFeature(HelperFeatureEnum::Memory_dpct_memset, C);
    requestFeature(HelperFeatureEnum::Memory_dpct_memset_2d, C);
    requestFeature(HelperFeatureEnum::Memory_dpct_memset_3d, C);
  }

  if (NameRef == "cudaMemset2D") {
    handleAsync(C, 5, Result);
  } else if (NameRef == "cudaMemset3D") {
    handleAsync(C, 3, Result);
  } else if (NameRef == "cudaMemset") {
    std::string AsyncQueue;
    if (C->getNumArgs() > 3 && !C->getArg(3)->isDefaultArgument()) {
      if (!isPredefinedStreamHandle(C->getArg(3)))
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

  requestFeature(HelperFeatureEnum::Memory_device_memory_get_size, C);
  Replacement = getDrefName(C->getArg(0)) + " = " + StmtStrArg1 + ".get_size()";
  emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
}

void MemoryMigrationRule::prefetchMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
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
      if (!isPredefinedStreamHandle(C->getArg(3)))
        StmtStrArg3 = ExprAnalysis::ref(C->getArg(3));
    } else {
      StmtStrArg3 = "0";
    }

    // In clang "define NULL __null"
    if (StmtStrArg3 == "0" || StmtStrArg3 == "") {
      requestFeature(HelperFeatureEnum::Device_dev_mgr_get_device, C);
      requestFeature(HelperFeatureEnum::Device_device_ext_default_queue, C);
      Replacement = MapNames::getDpctNamespace() +
                    "dev_mgr::instance().get_device(" + StmtStrArg2 +
                    ").default_queue().prefetch(" + StmtStrArg0 + "," +
                    StmtStrArg1 + ")";
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
    report(C->getBeginLoc(), Diagnostics::NOTSUPPORTED, false,
           "cudaMemPrefetchAsync");
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

  if (Name == "cudaHostGetDevicePointer") {
    if (USMLevel == UsmLevel::UL_Restricted) {
      std::ostringstream Repl;
      ExprAnalysis EA;
      EA.analyze(C->getArg(0));
      auto Arg0Str = EA.getReplacedString();
      EA.analyze(C->getArg(1));
      auto Arg1Str = EA.getReplacedString();
      Repl << "*(" << Arg0Str << ") = " << Arg1Str;
      emplaceTransformation(new ReplaceStmt(C, std::move(Repl.str())));
    } else {
      report(C->getBeginLoc(), Diagnostics::NOTSUPPORTED, false,
             MapNames::ITFName.at(Name));
    }
  } else if (Name == "cudaHostRegister" || Name == "cudaHostUnregister") {
    auto Msg = MapNames::RemovedAPIWarningMessage.find(Name);
    if (IsAssigned) {
      report(C->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(Name), Msg->second);
      emplaceTransformation(new ReplaceStmt(C, "0"));
    } else {
      report(C->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(Name), Msg->second);
      emplaceTransformation(new ReplaceStmt(C, ""));
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
    requestFeature(HelperFeatureEnum::Image_image_matrix_get_channel, C);
  } else if (Name == "cuMemGetInfo_v2" || Name == "cudaMemGetInfo") {
    auto &SM = DpctGlobalInfo::getSourceManager();
    std::ostringstream OS;
    if (IsAssigned)
      OS << "(";
    auto SecArg = C->getArg(1);
    printDerefOp(OS, SecArg);
    OS << " = " << MapNames::getDpctNamespace()
       << "get_current_device().get_device_info()"
          ".get_global_mem_size()";
    requestFeature(HelperFeatureEnum::Device_get_current_device, C);
    requestFeature(
        HelperFeatureEnum::Device_device_ext_get_device_info_return_info, C);
    requestFeature(HelperFeatureEnum::Device_device_info_get_global_mem_size,
                   C);
    if (IsAssigned) {
      OS << ", 0)";
      report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
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
  requestFeature(HelperFeatureEnum::Image_image_matrix_get_channel, C);
  requestFeature(HelperFeatureEnum::Image_image_matrix_get_range, C);
  requestFeature(HelperFeatureEnum::Image_image_matrix_get_range_T, C);
}

void MemoryMigrationRule::cudaHostGetFlags(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::ostringstream OS;
  printDerefOp(OS, C->getArg(0));
  OS << " = 0";
  emplaceTransformation(new ReplaceStmt(C, OS.str()));
}

void MemoryMigrationRule::cudaMemAdvise(const MatchFinder::MatchResult &Result,
                                        const CallExpr *C,
                                        const UnresolvedLookupExpr *ULExpr,
                                        bool IsAssigned) {
  // Do nothing if USM is disabled
  if (USMLevel == UsmLevel::UL_None) {
    report(C->getBeginLoc(), Diagnostics::NOTSUPPORTED, false, "cudaMemAdvise");
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
    OS << MapNames::getDpctNamespace() +
              "cpu_device().default_queue().mem_advise("
       << Arg0Str << ", " << Arg1Str << ", " << Arg2Str << ")";
    emplaceTransformation(new ReplaceStmt(C, OS.str()));
    requestFeature(HelperFeatureEnum::Device_cpu_device, C);
    requestFeature(HelperFeatureEnum::Device_device_ext_default_queue, C);
    return;
  }
  OS << MapNames::getDpctNamespace() + "get_device(" << Arg3Str
     << ").default_queue().mem_advise(" << Arg0Str << ", " << Arg1Str << ", "
     << Arg2Str << ")";
  emplaceTransformation(new ReplaceStmt(C, OS.str()));
  requestFeature(HelperFeatureEnum::Device_get_device, C);
  requestFeature(HelperFeatureEnum::Device_device_ext_default_queue, C);
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
        "cudaMemAdvise", "cudaGetChannelDesc", "cuMemHostAlloc",
        "cuMemFreeHost", "cuMemGetInfo_v2", "cuMemAlloc_v2", "cuMemcpyHtoD_v2",
        "cuMemcpyDtoH_v2", "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2",
        "cuMemcpy2D_v2", "cuMemcpy2DAsync_v2", "cuMemcpy3D_v2",
        "cudaMemGetInfo");
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
    if (IsAssigned && Name.compare("cudaHostRegister") &&
        Name.compare("cudaHostUnregister") && Name.compare("cudaMemAdvise") &&
        Name.compare("cudaArrayGetInfo") && Name.compare("cudaMalloc") &&
        Name.compare("cudaMallocPitch") && Name.compare("cudaMalloc3D") &&
        Name.compare("cublasAlloc") && Name.compare("cuMemGetInfo_v2") &&
        Name.compare("cudaHostAlloc") && Name.compare("cudaMallocHost") &&
        Name.compare("cuMemHostAlloc") && Name.compare("cudaMemGetInfo")) {
      report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      insertAroundStmt(C, "(", ", 0)");
    } else if (IsAssigned && !Name.compare("cudaMemAdvise") &&
               USMLevel != UsmLevel::UL_None) {
      report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      insertAroundStmt(C, "(", ", 0)");
    } else if (IsAssigned && !Name.compare("cudaArrayGetInfo")) {
      report(C->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
      std::string IndentStr =
          getIndent(C->getBeginLoc(), *Result.SourceManager).str();
      IndentStr += "  ";
      std::string PreStr{"([&](){"};
      PreStr += getNL();
      PreStr += IndentStr;
      std::string PostStr{";"};
      PostStr += getNL();
      PostStr += IndentStr;
      PostStr += "}(), 0)";
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
  requestFeature(HelperFeatureEnum::Memory_device_memory_get_ptr, C);
  emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
}

MemoryMigrationRule::MemoryMigrationRule() {
  SetRuleProperty(RT_ApplyToCudaFile | RT_ApplyToCppFile);
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
          {"cudaMallocManaged", &MemoryMigrationRule::mallocMigration},
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
          {"cudaFree", &MemoryMigrationRule::freeMigration},
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
          {"cudaGetSymbolAddress",
           &MemoryMigrationRule::getSymbolAddressMigration},
          {"cudaGetSymbolSize", &MemoryMigrationRule::getSymbolSizeMigration},
          {"cudaHostGetDevicePointer", &MemoryMigrationRule::miscMigration},
          {"cudaHostRegister", &MemoryMigrationRule::miscMigration},
          {"cudaHostUnregister", &MemoryMigrationRule::miscMigration},
          {"cudaMemPrefetchAsync", &MemoryMigrationRule::prefetchMigration},
          {"cudaArrayGetInfo", &MemoryMigrationRule::cudaArrayGetInfo},
          {"cudaHostGetFlags", &MemoryMigrationRule::cudaHostGetFlags},
          {"cudaMemAdvise", &MemoryMigrationRule::cudaMemAdvise},
          {"cudaGetChannelDesc", &MemoryMigrationRule::miscMigration},
          {"cuMemHostAlloc", &MemoryMigrationRule::mallocMigration},
          {"cuMemGetInfo_v2", &MemoryMigrationRule::miscMigration},
          {"cudaMemGetInfo", &MemoryMigrationRule::miscMigration}};

  for (auto &P : Dispatcher)
    MigrationDispatcher[P.first] =
        std::bind(P.second, this, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4);
}

/// Convert a raw pointer argument and a pitch argument to a dpct::pitched_data
/// constructor. If \p ExcludeSizeArg is true, the argument represent the pitch
/// size will not be included in the constructor.
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
  requestFeature(HelperFeatureEnum::Memory_pitched_data, C);
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
          requestHelperFeatureForEnumNames(Enum->getName().str(), C);
        }
      }
    }
  }
}

void MemoryMigrationRule::handleAsync(const CallExpr *C, unsigned i,
                                      const MatchFinder::MatchResult &Result) {
  if (C->getNumArgs() > i && !C->getArg(i)->isDefaultArgument()) {
    auto StreamExpr = C->getArg(i)->IgnoreImplicitAsWritten();
    emplaceTransformation(new InsertBeforeStmt(StreamExpr, "*"));
    if (auto IL = dyn_cast<IntegerLiteral>(StreamExpr)) {
      if (IL->getValue().getZExtValue() == 0) {
        emplaceTransformation(removeArg(C, i, *Result.SourceManager));
        return;
      } else {
        emplaceTransformation(new InsertBeforeStmt(
            StreamExpr, "(" + MapNames::getClNamespace() + "queue *)"));
      }
    } else if (isPredefinedStreamHandle(StreamExpr)) {
      emplaceTransformation(removeArg(C, i, *Result.SourceManager));
      return;
    } else if (!isa<DeclRefExpr>(StreamExpr)) {
      insertAroundStmt(StreamExpr, "(", ")");
    }
  }
}

REGISTER_RULE(MemoryMigrationRule)

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
  requestFeature(HelperFeatureEnum::Memory_pitched_data, VD);
  emplaceParamDecl(VD, getCtadType("id"), true, "0", "from_pos", "to_pos");
  emplaceParamDecl(VD, getCtadType("range"), true, "1", "size");
  if (hasDirection) {
    emplaceParamDecl(VD, MapNames::getDpctNamespace() + "memcpy_direction",
                     false, "0", "direction");
    requestFeature(HelperFeatureEnum::Memory_memcpy_direction, VD);
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
    requestFeature(PitchMemberToFeature.at(Member), E);
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
          requestFeature(HelperFeatureEnum::Image_image_matrix_to_pitched_data,
                         BO);
          emplaceTransformation(
              new InsertAfterStmt(BO->getRHS(), "->to_pitched_data()"));
        } else if (QualName == "CUarray_st") {
          requestFeature(HelperFeatureEnum::Image_image_matrix_to_pitched_data,
                         BO);
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
        requestFeature(HelperFeatureEnum::Memory_pitched_data, CE);
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
              {"pitch", HelperFeatureEnum::Memory_pitched_data_set_pitch},
              {"ptr", HelperFeatureEnum::Memory_pitched_data_set_data_ptr},
              {"xsize", HelperFeatureEnum::Memory_pitched_data_set_x},
              {"ysize", HelperFeatureEnum::Memory_pitched_data_set_y}};
      static const std::unordered_map<std::string, HelperFeatureEnum>
          PitchMemberNameToGetFeatureMap = {
              {"pitch", HelperFeatureEnum::Memory_pitched_data_get_pitch},
              {"ptr", HelperFeatureEnum::Memory_pitched_data_get_data_ptr},
              {"xsize", HelperFeatureEnum::Memory_pitched_data_get_x},
              {"ysize", HelperFeatureEnum::Memory_pitched_data_get_y}};
      if (auto BO = DpctGlobalInfo::findParent<BinaryOperator>(M)) {
        if (BO->getOpcode() == BO_Assign) {
          requestFeature(PitchMemberNameToSetFeatureMap.at(MemberName.str()),
                         BO);
          emplaceTransformation(ReplaceMemberAssignAsSetMethod(BO, M, Replace));
          return;
        }
      }
      emplaceTransformation(new ReplaceToken(
          M->getMemberLoc(), buildString("get_", Replace, "()")));
      requestFeature(PitchMemberNameToGetFeatureMap.at(MemberName.str()), M);
    }
  }
}

REGISTER_RULE(MemoryDataTypeRule)

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

REGISTER_RULE(UnnamedTypesRule)

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

REGISTER_RULE(CMemoryAPIRule)

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

REGISTER_RULE(GuessIndentWidthRule)

void MathFunctionsRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> MathFunctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_OPERATOR(APINAME, OPKIND) APINAME,
#define ENTRY_TYPECAST(APINAME) APINAME,
#define ENTRY_UNSUPPORTED(APINAME) APINAME,
#include "APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_NO_REWRITE
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
  };

  MF.addMatcher(
      callExpr(callee(functionDecl(
                   internal::Matcher<NamedDecl>(
                       new internal::HasNameMatcher(MathFunctions)),
                   anyOf(unless(hasDeclContext(namespaceDecl(anything()))),
                         hasDeclContext(namespaceDecl(hasName("std")))))),
               unless(hasAncestor(
                   cxxConstructExpr(hasType(typedefDecl(hasName("dim3")))))))
          .bind("math"),
      this);
}

void MathFunctionsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto CE = getNodeAsType<CallExpr>(Result, "math")) {
    ExprAnalysis EA(CE);
    EA.applyAllSubExprRepl();

    auto FD = CE->getDirectCallee();
    if (FD) {
      std::string Name = FD->getNameInfo().getName().getAsString();
      if (Name == "__brev") {
        requestFeature(HelperFeatureEnum::Util_reverse_bits, CE);
      } else if (Name == "__vmaxs4" || Name == "__vmaxu2") {
        requestFeature(HelperFeatureEnum::Util_vectorized_max, CE);
      } else if (Name == "__vminu2") {
        requestFeature(HelperFeatureEnum::Util_vectorized_min, CE);
      } else if (Name == "__vcmpgtu2") {
        requestFeature(HelperFeatureEnum::Util_vectorized_isgreater_T, CE);
        requestFeature(HelperFeatureEnum::Util_vectorized_isgreater_unsigned,
                       CE);
      } else if (Name == "__byte_perm") {
        requestFeature(HelperFeatureEnum::Util_byte_level_permute, CE);
      }
    }
  }
}

REGISTER_RULE(MathFunctionsRule)

void WarpFunctionsRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> WarpFunctions = {
      "__shfl_up_sync", "__shfl_down_sync", "__shfl_sync", "__shfl_up",
      "__shfl_down",    "__shfl",           "__shfl_xor",  "__shfl_xor_sync",
      "__all",          "__all_sync",       "__any",       "__any_sync",
      "__ballot",       "__ballot_sync",    "__activemask"};

  MF.addMatcher(callExpr(callee(functionDecl(internal::Matcher<NamedDecl>(
                             new internal::HasNameMatcher(WarpFunctions)))),
                         hasAncestor(functionDecl().bind("ancestor")))
                    .bind("warp"),
                this);
}

void WarpFunctionsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto CE = getNodeAsType<CallExpr>(Result, "warp")) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}
REGISTER_RULE(WarpFunctionsRule)

void SyncThreadsRule::registerMatcher(MatchFinder &MF) {
  auto SyncAPI = [&]() {
    return hasAnyName("__syncthreads", "this_thread_block", "sync",
                      "__threadfence_block", "__threadfence",
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

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  if (FuncName == "__syncthreads" || FuncName == "sync") {
    if (auto CXXCE = dyn_cast<CXXMemberCallExpr>(CE)) {
      if (auto ME = dyn_cast<MemberExpr>(CXXCE->getCallee())) {
        if (auto ICE = dyn_cast<ImplicitCastExpr>(ME->getBase())) {
          auto Type = ICE->getType().getAsString();
          if (Type == "const class cooperative_groups::__v1::grid_group") {
            return;
          }
        }
      }
    }
    report(CE->getBeginLoc(), Diagnostics::BARRIER_PERFORMANCE_TUNNING, true);
    std::string Replacement = DpctGlobalInfo::getItem(CE) + ".barrier()";
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
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
    std::string ReplStr = CLNS + "ext::oneapi::atomic_fence(" + CLNS +
                          "ext::oneapi::memory_order::acq_rel, " + CLNS +
                          "ext::oneapi::memory_scope::work_group" + ")";
    report(CE->getBeginLoc(), Diagnostics::MEMORY_ORDER_PERFORMANCE_TUNNING,
           true);
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "__threadfence") {
    std::string CLNS = MapNames::getClNamespace();
    std::string ReplStr = CLNS + "ext::oneapi::atomic_fence(" + CLNS +
                          "ext::oneapi::memory_order::acq_rel, " + CLNS +
                          "ext::oneapi::memory_scope::device" + ")";
    report(CE->getBeginLoc(), Diagnostics::MEMORY_ORDER_PERFORMANCE_TUNNING,
           true);
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "__threadfence_system") {
    std::string CLNS = MapNames::getClNamespace();
    std::string ReplStr = CLNS + "ext::oneapi::atomic_fence(" + CLNS +
                          "ext::oneapi::memory_order::acq_rel, " + CLNS +
                          "ext::oneapi::memory_scope::system" + ")";
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
    report(CE->getBeginLoc(), Diagnostics::BARRIER_PERFORMANCE_TUNNING, true);
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "__syncwarp") {
    std::string ReplStr;
    ReplStr = DpctGlobalInfo::getItem(CE) + ".barrier()";
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  }
}

REGISTER_RULE(SyncThreadsRule)

void KernelFunctionInfoRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      varDecl(hasType(recordDecl(hasName("cudaFuncAttributes")))).bind("decl"),
      this);
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName("cudaFuncGetAttributes"))))
          .bind("call"),
      this);
  MF.addMatcher(callExpr(callee(functionDecl(hasAnyName("cuFuncGetAttribute"))))
                    .bind("callFuncGetAttribute"),
                this);
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(
                               recordDecl(hasName("cudaFuncAttributes")))))
                    .bind("member"),
                this);
}

void KernelFunctionInfoRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto V = getNodeAsType<VarDecl>(Result, "decl")) {
    emplaceTransformation(new ReplaceTypeInDecl(
        V, MapNames::getDpctNamespace() + "kernel_function_info"));
    requestFeature(HelperFeatureEnum::Kernel_kernel_function_info, V);
  } else if (auto C = getNodeAsType<CallExpr>(Result, "call")) {
    requestFeature(HelperFeatureEnum::Kernel_get_kernel_function_info, C);
    emplaceTransformation(
        new ReplaceToken(C->getBeginLoc(), "(" + MapNames::getDpctNamespace() +
                                               "get_kernel_function_info"));
    emplaceTransformation(new InsertAfterStmt(C, ", 0)"));
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

REGISTER_RULE(KernelFunctionInfoRule)

// split api name
std::vector<std::vector<std::string>>
RecognizeAPINameRule::splitAPIName(std::vector<std::string> &AllAPINames) {
  std::vector<std::vector<std::string>> Result;
  std::vector<std::string> FuncNames, FuncNamesHasNS, FuncNamespaces,
      MemFuncNames, ObjNames, MemFuncNamesHasNS, ObjNamesHasNS, ObjNamespaces;
  for (auto &APIName : AllAPINames) {
    size_t ScopeResolutionOpPos = APIName.find("::");
    size_t DotPos = APIName.find(".");
    if (DotPos == std::string::npos) {
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
            APIName.substr(ScopeResolutionOpPos + std::string("::").length()));
      }
    } else {
      // 3. ObjectName.FunctionName
      if (ScopeResolutionOpPos == std::string::npos) {
        if (std::find(ObjNames.begin(), ObjNames.end(),
                      APIName.substr(0, DotPos + std::string(".").length())) ==
            ObjNames.end()) {
          ObjNames.emplace_back(
              APIName.substr(0, DotPos + std::string(".").length()));
        }
        MemFuncNames.emplace_back(
            APIName.substr(DotPos + std::string(".").length()));
      } else {
        // 4. Namespace::ObjectName.FunctionName
        if (std::find(ObjNamespaces.begin(), ObjNamespaces.end(),
                      APIName.substr(0, ScopeResolutionOpPos)) ==
            ObjNamespaces.end()) {
          ObjNamespaces.emplace_back(APIName.substr(0, ScopeResolutionOpPos));
        }
        if (std::find(ObjNamesHasNS.begin(), ObjNamesHasNS.end(),
                      APIName.substr(ScopeResolutionOpPos +
                                         std::string("::").length(),
                                     DotPos - ScopeResolutionOpPos -
                                         std::string("::").length())) ==
            ObjNamesHasNS.end()) {
          ObjNamesHasNS.emplace_back(APIName.substr(
              ScopeResolutionOpPos + std::string("::").length(),
              DotPos - ScopeResolutionOpPos - std::string("::").length()));
        }
        MemFuncNamesHasNS.emplace_back(
            APIName.substr(DotPos + std::string(".").length()));
      }
    }
  }
  return {FuncNames, FuncNamesHasNS,    FuncNamespaces, MemFuncNames,
          ObjNames,  MemFuncNamesHasNS, ObjNamesHasNS,  ObjNamespaces};
}

void RecognizeAPINameRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> AllAPINames = MigrationStatistics::GetAllAPINames();
  // AllAPIComponent[0] : FuncNames
  // AllAPIComponent[1] : FuncNamesHasNS
  // AllAPIComponent[2] : FuncNamespaces
  // AllAPIComponent[3] : MemFuncNames
  // AllAPIComponent[4] : ObjNames
  // AllAPIComponent[5] : MemFuncNamesHasNS
  // AllAPIComponent[6] : ObjNamesHasNS
  // AllAPIComponent[7] : ObjNamespaces
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
    MF.addMatcher(
        callExpr(
            allOf(callee(functionDecl(matchesName("(nccl.*)|(cudnn.*)"))),
                  unless(callee(functionDecl(internal::Matcher<NamedDecl>(
                      new internal::HasNameMatcher(AllAPIComponent[0]))))),
                  unless(hasAncestor(cudaKernelCallExpr())),
                  unless(callee(hasDeclContext(namedDecl(hasName("std")))))))
            .bind("ManualMigrateAPI"),
        this);
  }

  if (!AllAPIComponent[1].empty() && !AllAPIComponent[2].empty()) {
    MF.addMatcher(
        callExpr(
            callee(functionDecl(allOf(
                namedDecl(internal::Matcher<NamedDecl>(
                    new internal::HasNameMatcher(AllAPIComponent[1]))),
                hasDeclContext(
                    namespaceDecl(namedDecl(internal::Matcher<NamedDecl>(
                        new internal::HasNameMatcher(AllAPIComponent[2])))))))))
            .bind("APINamesHasNSUsed"),
        this);
  }

  if (!AllAPIComponent[3].empty() && !AllAPIComponent[4].empty()) {
    MF.addMatcher(
        cxxMemberCallExpr(
            on(hasType(hasCanonicalType(
                qualType(hasDeclaration(namedDecl(internal::Matcher<NamedDecl>(
                    new internal::HasNameMatcher(AllAPIComponent[4])))))))),
            callee(cxxMethodDecl(namedDecl(internal::Matcher<NamedDecl>(
                new internal::HasNameMatcher(AllAPIComponent[3]))))))
            .bind("MFAPINamesUsed"),
        this);
  }

  if (!AllAPIComponent[5].empty() && !AllAPIComponent[6].empty() &&
      !AllAPIComponent[7].empty()) {
    MF.addMatcher(
        cxxMemberCallExpr(
            on(hasType(hasCanonicalType(qualType(hasDeclaration(allOf(
                namedDecl(internal::Matcher<NamedDecl>(
                    new internal::HasNameMatcher(AllAPIComponent[6]))),
                hasDeclContext(namespaceDecl(namedDecl(
                    internal::Matcher<NamedDecl>(new internal::HasNameMatcher(
                        AllAPIComponent[7]))))))))))),
            callee(cxxMethodDecl(namedDecl(internal::Matcher<NamedDecl>(
                new internal::HasNameMatcher(AllAPIComponent[5]))))))
            .bind("MFAPINamesHasNSUsed"),
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

void RecognizeAPINameRule::processMemberFuncCall(const CXXMemberCallExpr *MC) {
  // 1. assemble api name
  // 2. emit warning for unmigrated api
  QualType ObjType = MC->getObjectType().getCanonicalType();
  if (isTypeInRoot(ObjType.getTypePtr()) || !MC->getMethodDecl()) {
    return;
  }
  std::string ObjNameSpace, ObjName;
  auto ObjDecl = getNamedDecl(ObjType.getTypePtr());
  if (const auto *NSD = dyn_cast<NamespaceDecl>(ObjDecl->getDeclContext())) {
    if (!NSD->isInlineNamespace()) {
      ObjNameSpace = NSD->getName().str() + "::";
    }
  }
  if (!MC->getMethodDecl())
    return;
  ObjName = ObjNameSpace + ObjDecl->getNameAsString() + ".";
  std::string FuncName = MC->getMethodDecl()->getNameAsString();
  std::string APIName = ObjName + FuncName;

  SrcAPIStaticsMap[getFunctionSignature(MC->getMethodDecl(), ObjName)]++;

  if (!MigrationStatistics::IsMigrated(APIName)) {
    GAnalytics(getFunctionSignature(MC->getMethodDecl(), ObjName));
    const SourceManager &SM = DpctGlobalInfo::getSourceManager();
    const SourceLocation FileLoc = SM.getFileLoc(MC->getBeginLoc());

    std::string SLStr = FileLoc.printToString(SM);

    std::size_t PosCol = SLStr.rfind(':');
    std::size_t PosRow = SLStr.rfind(':', PosCol - 1);
    std::string FileName = SLStr.substr(0, PosRow);
    LOCStaticsMap[FileName][2]++;

    auto Iter = MapNames::ITFName.find(APIName.c_str());
    if (Iter != MapNames::ITFName.end()) {
      report(MC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             Iter->second);
    } else {
      report(MC->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, APIName);
    }
  }
}

void RecognizeAPINameRule::processFuncCall(const CallExpr *CE,
                                           bool HaveKeywordInAPIName) {
  std::string Namespace;
  const NamedDecl *ND = dyn_cast<NamedDecl>(CE->getCalleeDecl());
  if (ND) {
    const auto *NSD = dyn_cast<NamespaceDecl>(ND->getDeclContext());
    if (NSD && !NSD->isInlineNamespace()) {
      Namespace = NSD->getName().str();
    }
  }

  std::string APIName = CE->getCalleeDecl()->getAsFunction()->getNameAsString();

  if (!Namespace.empty() && (Namespace == "thrust" || Namespace == "cub")) {
    APIName = Namespace + "::" + APIName;
  }

  SrcAPIStaticsMap[getFunctionSignature(CE->getCalleeDecl()->getAsFunction(),
                                        "")]++;
  if (APIName.size() >= 4 && APIName.substr(0, 4) == "nccl") {
    auto D = CE->getCalleeDecl();
    if (D) {
      auto FilePath = DpctGlobalInfo::getSourceManager()
                          .getFilename(D->getBeginLoc())
                          .str();
      if (DpctGlobalInfo::isInRoot(FilePath)) {
        return;
      }
    }
    report(CE->getBeginLoc(), Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
           "Intel(R) oneAPI Collective Communications Library");
  } else if (APIName.size() >= 5 && APIName.substr(0, 5) == "cudnn") {
    auto D = CE->getCalleeDecl();
    if (D) {
      auto FilePath = DpctGlobalInfo::getSourceManager()
                          .getFilename(D->getBeginLoc())
                          .str();
      if (DpctGlobalInfo::isInRoot(FilePath)) {
        return;
      }
    }
    report(CE->getBeginLoc(), Diagnostics::MANUAL_MIGRATION_LIBRARY, false,
           "Intel(R) oneAPI Deep Neural Network Library (oneDNN)");
  } else if (HaveKeywordInAPIName) {
    // In the AST matcher, it will match function call whose name contains
    // keyword. If the keyword is at name begin, code will go in to previous two
    // branch. If code goes here, we treat the API is user-defined, just return.
    return;
  } else if (!MigrationStatistics::IsMigrated(APIName)) {
    GAnalytics(getFunctionSignature(CE->getCalleeDecl()->getAsFunction(), ""));
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
  const CXXMemberCallExpr *MC = nullptr;
  if ((CE = getNodeAsType<CallExpr>(Result, "APINamesUsed")) ||
      (CE = getNodeAsType<CallExpr>(Result, "APINamesHasNSUsed"))) {
    processFuncCall(CE, false);
  } else if (CE = getNodeAsType<CallExpr>(Result, "ManualMigrateAPI")) {
    processFuncCall(CE, true);
  } else if ((MC =
                  getNodeAsType<CXXMemberCallExpr>(Result, "MFAPINamesUsed")) ||
             (MC = getNodeAsType<CXXMemberCallExpr>(Result,
                                                    "MFAPINamesHasNSUsed"))) {
    processMemberFuncCall(MC);
  }
  return;
}

REGISTER_RULE(RecognizeAPINameRule)

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

REGISTER_RULE(RecognizeTypeRule)

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
    requestFeature(HelperFeatureEnum::Image_image_data_set_data, AssignPtrExpr);
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
    requestFeature(HelperFeatureEnum::Image_image_data_set_data, AssignPtrExpr);
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
    requestFeature(HelperFeatureEnum::Image_image_data_set_data,
                   AssignArrayExpr);
    // Remove all the assign expr
    removeRange(BORange);
    removeRange(AssignArrayRange);

    emplaceTransformation(new InsertText(LastPos, std::move(InsertStr)));
  }
}

REGISTER_RULE(TextureMemberSetRule)

void TextureRule::registerMatcher(MatchFinder &MF) {
  auto DeclMatcher = varDecl(hasType(templateSpecializationType(
      hasDeclaration(classTemplateSpecializationDecl(hasName("texture"))))));

  auto DeclMatcherUTF = varDecl(hasType(templateSpecializationType()));
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
      typeLoc(loc(qualType(hasDeclaration(namedDecl(hasAnyName(
                  "cudaChannelFormatDesc", "cudaChannelFormatKind",
                  "cudaTextureDesc", "cudaResourceDesc", "cudaResourceType",
                  "cudaTextureAddressMode", "cudaTextureFilterMode",
                  "cudaArray", "cudaArray_t", "CUarray_st", "CUarray",
                  "CUarray_format", "CUarray_format_enum", "CUdeviceptr",
                  "CUresourcetype", "CUresourcetype_enum", "CUaddress_mode",
                  "CUaddress_mode_enum", "CUfilter_mode", "CUfilter_mode_enum",
                  "CUDA_RESOURCE_DESC", "CUDA_TEXTURE_DESC", "CUtexref"))))))
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
    report(ME->getBeginLoc(), Diagnostics::NOTSUPPORTED, false, Field);
    return;
  }

  if (AssignedBO) {
    StringRef MethodName;
    auto AssignedValue = getMemberAssignedValue(AssignedBO, Field, MethodName);
    if (MethodName.empty()) {
      requestFeature(HelperFeatureEnum::Image_sampling_info_set_addressing_mode,
                     AssignedBO);
      requestFeature(HelperFeatureEnum::Image_sampling_info_set_filtering_mode,
                     AssignedBO);
      requestFeature(
          HelperFeatureEnum::
              Image_sampling_info_set_coordinate_normalization_mode_enum,
          AssignedBO);

      requestFeature(
          HelperFeatureEnum::Image_image_wrapper_base_set_addressing_mode,
          AssignedBO);
      requestFeature(
          HelperFeatureEnum::Image_image_wrapper_base_set_filtering_mode,
          AssignedBO);
      requestFeature(
          HelperFeatureEnum::
              Image_image_wrapper_base_set_coordinate_normalization_mode_enum,
          AssignedBO);
    } else {
      if (SamplingInfoToSetFeatureMap.count(MethodName.str())) {
        requestFeature(SamplingInfoToSetFeatureMap.at(MethodName.str()),
                       AssignedBO);
      }
      if (ImageWrapperBaseToSetFeatureMap.count(MethodName.str())) {
        requestFeature(ImageWrapperBaseToSetFeatureMap.at(MethodName.str()),
                       AssignedBO);
      }
    }
    emplaceTransformation(ReplaceMemberAssignAsSetMethod(
        AssignedBO, ME, MethodName, AssignedValue));
  } else {
    if (ReplField == "coordinate_normalization_mode") {
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, "is_coordinate_normalized()"));
      requestFeature(
          HelperFeatureEnum::Image_image_wrapper_base_is_coordinate_normalized,
          ME);
      requestFeature(
          HelperFeatureEnum::Image_sampling_info_is_coordinate_normalized, ME);
    } else {
      emplaceTransformation(new RenameFieldInMemberExpr(
          ME, buildString("get_", ReplField, "()")));
      if (SamplingInfoToGetFeatureMap.count(ReplField)) {
        requestFeature(SamplingInfoToGetFeatureMap.at(ReplField), ME);
      }
      if (ImageWrapperBaseToGetFeatureMap.count(ReplField)) {
        requestFeature(ImageWrapperBaseToGetFeatureMap.at(ReplField), ME);
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

// Return the binary operator if E is the lhs of an assign experssion, otherwise
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
      auto ArgNum = TST->getNumArgs();

      if (!isa<ParmVarDecl>(VD) || ArgNum != 3)
        return;

      auto Arg2 = TST->getArg(2);
      if (getStmtSpelling(Arg2.getAsExpr()) == "cudaReadModeNormalizedFloat")
        report(VD->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_NORM_READ_MODE,
               true);

      processTexVarDeclInDevice(VD);
    }
  } else if (auto VD = getAssistNodeAsType<VarDecl>(Result, "texDecl")) {
    auto TST = VD->getType()->getAs<TemplateSpecializationType>();
    if (!TST)
      return;

    auto ArgNum = TST->getNumArgs();

    if (ArgNum == 3) {
      auto Arg2 = TST->getArg(2);
      if (getStmtSpelling(Arg2.getAsExpr()) == "cudaReadModeNormalizedFloat")
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
          requestFeature(HelperFeatureEnum::Image_image_data_set_data_type, BO);
          emplaceTransformation(
              ReplaceMemberAssignAsSetMethod(BO, ME, "data_type"));
        } else {
          requestFeature(HelperFeatureEnum::Image_image_data_get_data_type, ME);
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
              {"x", HelperFeatureEnum::Image_image_channel_get_channel_size},
              {"y", HelperFeatureEnum::Image_image_channel_get_channel_size},
              {"z", HelperFeatureEnum::Image_image_channel_get_channel_size},
              {"w", HelperFeatureEnum::Image_image_channel_get_channel_size},
              {"f",
               HelperFeatureEnum::Image_image_channel_get_channel_data_type}};
      static const std::unordered_map<std::string, HelperFeatureEnum>
          MethodNameToSetFeatureMap = {
              {"x", HelperFeatureEnum::Image_image_channel_set_channel_size},
              {"y", HelperFeatureEnum::Image_image_channel_set_channel_size},
              {"z", HelperFeatureEnum::Image_image_channel_set_channel_size},
              {"w", HelperFeatureEnum::Image_image_channel_set_channel_size},
              {"f",
               HelperFeatureEnum::Image_image_channel_set_channel_data_type}};
      static std::map<std::string, std::string> ExtraArgMap = {
          {"x", "1"}, {"y", "2"}, {"z", "3"}, {"w", "4"}, {"f", ""}};
      std::string MemberName = ME->getMemberNameInfo().getAsString();
      if (auto BO = getParentAsAssignedBO(ME, *Result.Context)) {
        if (MemberName == "f")
          requestFeature(
              HelperFeatureEnum::Image_image_wrapper_base_set_channel_data_type,
              ME);
        else
          requestFeature(
              HelperFeatureEnum::Image_image_wrapper_base_set_channel_size, ME);
        requestFeature(MethodNameToSetFeatureMap.at(MemberName), BO);
        emplaceTransformation(ReplaceMemberAssignAsSetMethod(
            BO, ME, MethodNameMap[MemberName], "", ExtraArgMap[MemberName]));
      } else {
        if (MemberName == "f")
          requestFeature(
              HelperFeatureEnum::Image_image_wrapper_base_get_channel_data_type,
              ME);
        else
          requestFeature(
              HelperFeatureEnum::Image_image_wrapper_base_get_channel_size, ME);
        requestFeature(MethodNameToGetFeatureMap.at(MemberName), ME);
        emplaceTransformation(new RenameFieldInMemberExpr(
            ME, buildString("get_", MethodNameMap[MemberName], "()")));
      }
    } else {
      replaceTextureMember(ME, *Result.Context, *Result.SourceManager);
    }
  } else if (auto TL = getNodeAsType<TypeLoc>(Result, "texType")) {
    const std::string &ReplType = MapNames::findReplacedName(
        MapNames::TypeNamesMap,
        DpctGlobalInfo::getUnqualifiedTypeName(TL->getType(), *Result.Context));

    requestHelperFeatureForTypeNames(
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
      if (MethodName.empty()) {
        requestFeature(
            HelperFeatureEnum::Image_image_wrapper_base_set_addressing_mode,
            CE);
        requestFeature(
            HelperFeatureEnum::Image_image_wrapper_base_set_filtering_mode, CE);
        requestFeature(
            HelperFeatureEnum::
                Image_image_wrapper_base_set_coordinate_normalization_mode_enum,
            CE);
      } else {
        requestFeature(ImageWrapperBaseToSetFeatureMap.at(MethodName.str()),
                       CE);
      }
      std::shared_ptr<CallExprRewriter> Rewriter =
          std::make_shared<AssignableRewriter>(
              CE, std::make_shared<PrinterRewriter<MemberCallPrinter<
                      const Expr *, RenameWithSuffix, StringRef>>>(
                      CE, Name, CE->getArg(0), true,
                      RenameWithSuffix("set", MethodName), Value));
      Optional<std::string> Result = Rewriter->rewrite();
      if (Result.hasValue())
        emplaceTransformation(
            new ReplaceStmt(CE, true, std::move(Result).getValue()));
      return;
    }
    if (Name == "cudaCreateChannelDesc") {
      auto Callee =
          dyn_cast<DeclRefExpr>(CE->getCallee()->IgnoreImplicitAsWritten());
      if (Callee) {
        auto TemArg = Callee->template_arguments();
        if (TemArg.size() != 0) {
          auto ChnType = TemArg[0].getArgument().getAsType().getAsString();
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
      requestHelperFeatureForEnumNames(EnumName, ECD);
      if (MapNames::replaceName(EnumConstantRule::EnumNamesMap, EnumName)) {
        emplaceTransformation(new ReplaceStmt(DRE, EnumName));
      } else {
        report(DRE->getBeginLoc(), Diagnostics::NOTSUPPORTED, false, EnumName);
      }
    }
  } else if (auto TL = getNodeAsType<TypeLoc>(Result, "texObj")) {
    if (auto FD = DpctGlobalInfo::getParentFunction(TL)) {
      if (!FD->hasAttr<CUDAGlobalAttr>() && !FD->hasAttr<CUDADeviceAttr>()) {
        emplaceTransformation(new ReplaceToken(
            TL->getBeginLoc(), TL->getEndLoc(),
            MapNames::getDpctNamespace() + "image_wrapper_base_p"));
        requestFeature(HelperFeatureEnum::Image_image_wrapper_base_p_alias,
                       TL->getBeginLoc());
      }
    } else if (auto VD = DpctGlobalInfo::findAncestor<VarDecl>(TL)) {
      if (VD->hasGlobalStorage()) {
        emplaceTransformation(new ReplaceToken(
            TL->getBeginLoc(), TL->getEndLoc(),
            MapNames::getDpctNamespace() + "image_wrapper_base_p"));
        requestFeature(HelperFeatureEnum::Image_image_wrapper_base_p_alias,
                       TL->getBeginLoc());
      }
    }
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
    report(ME->getBeginLoc(), Diagnostics::NOTSUPPORTED, false,
           ME->getMemberDecl()->getName());
  }

  if (FieldName == "channel") {
    if (removeExtraMemberAccess(TopMember))
      return;
  }

  if (AssignedBO) {
    static const std::unordered_map<std::string, HelperFeatureEnum>
        ResourceTypeNameToSetFeature = {
            {"devPtr", HelperFeatureEnum::Image_image_data_set_data_ptr},
            {"desc", HelperFeatureEnum::Image_image_data_set_channel},
            {"array", HelperFeatureEnum::Image_image_data_set_data_ptr},
            {"width", HelperFeatureEnum::Image_image_data_set_x},
            {"height", HelperFeatureEnum::Image_image_data_set_y},
            {"pitchInBytes", HelperFeatureEnum::Image_image_data_set_pitch},
            {"sizeInBytes", HelperFeatureEnum::Image_image_data_set_x},
            {"hArray", HelperFeatureEnum::Image_image_data_set_data_ptr},
            {"format", HelperFeatureEnum::Image_image_data_set_channel_type},
            {"numChannels",
             HelperFeatureEnum::Image_image_data_set_channel_num}};
    requestFeature(ResourceTypeNameToSetFeature.at(
                       TopMember->getMemberNameInfo().getAsString()),
                   ME);
    emplaceTransformation(
        ReplaceMemberAssignAsSetMethod(AssignedBO, TopMember, FieldName));
  } else {
    auto MemberName = TopMember->getMemberDecl()->getName();
    if (MemberName == "array" || MemberName == "hArray") {
      emplaceTransformation(new InsertBeforeStmt(
          TopMember, "(" + MapNames::getDpctNamespace() + "image_matrix_p)"));
      requestFeature(HelperFeatureEnum::Image_image_matrix_p_alias, TopMember);
    }
    static const std::unordered_map<std::string, HelperFeatureEnum>
        ResourceTypeNameToGetFeature = {
            {"devPtr", HelperFeatureEnum::Image_image_data_get_data_ptr},
            {"desc", HelperFeatureEnum::Image_image_data_get_channel},
            {"array", HelperFeatureEnum::Image_image_data_get_data_ptr},
            {"width", HelperFeatureEnum::Image_image_data_get_x},
            {"height", HelperFeatureEnum::Image_image_data_get_y},
            {"pitchInBytes", HelperFeatureEnum::Image_image_data_get_pitch},
            {"sizeInBytes", HelperFeatureEnum::Image_image_data_get_x},
            {"hArray", HelperFeatureEnum::Image_image_data_get_data_ptr},
            {"format", HelperFeatureEnum::Image_image_data_get_channel_type},
            {"numChannels",
             HelperFeatureEnum::Image_image_data_get_channel_num}};
    requestFeature(ResourceTypeNameToGetFeature.at(
                       TopMember->getMemberNameInfo().getAsString()),
                   ME);
    emplaceTransformation(new RenameFieldInMemberExpr(
        TopMember, buildString("get_", FieldName, "()")));
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
    unsigned LastIndex;
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
  for (auto R : Result) {
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
    requestFeature(
        HelperFeatureEnum::
            Image_image_wrapper_base_set_addressing_mode_filtering_mode_coordinate_normalization_mode,
        BO);
    requestFeature(
        HelperFeatureEnum::
            Image_image_wrapper_base_set_addressing_mode_filtering_mode_is_normalized,
        BO);
    requestFeature(
        HelperFeatureEnum::
            Image_sampling_info_set_addressing_mode_filtering_mode_coordinate_normalization_mode,
        BO);
    requestFeature(
        HelperFeatureEnum::
            Image_sampling_info_set_addressing_mode_filtering_mode_is_normalized,
        BO);
    return true;
  } else {
    return false;
  }
}

REGISTER_RULE(TextureRule)

void CXXNewExprRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(cxxNewExpr().bind("newExpr"), this);
}

void CXXNewExprRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto CNE = getAssistNodeAsType<CXXNewExpr>(Result, "newExpr")) {
    // E.g. new cudaEvent_t *()
    Token Tok;
    auto LOpts = Result.Context->getLangOpts();
    SourceManager *SM = Result.SourceManager;
    auto BeginLoc =
        CNE->getAllocatedTypeSourceInfo()->getTypeLoc().getBeginLoc();
    Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
    if (Tok.isAnyIdentifier()) {
      std::string Str = MapNames::findReplacedName(
          MapNames::TypeNamesMap, Tok.getRawIdentifier().str());

      requestHelperFeatureForTypeNames(Tok.getRawIdentifier().str(), BeginLoc);

      SourceManager &SM = DpctGlobalInfo::getSourceManager();
      BeginLoc = SM.getExpansionLoc(BeginLoc);
      if (!Str.empty()) {
        emplaceTransformation(new ReplaceToken(BeginLoc, std::move(Str)));
        return;
      }
    }

    // E.g. #define NEW_STREAM new cudaStream_t
    //      stream = NEW_STREAM;
    auto TypeName = CNE->getAllocatedType().getAsString();
    auto ReplName = std::string(
        MapNames::findReplacedName(MapNames::TypeNamesMap, TypeName));

    requestHelperFeatureForTypeNames(TypeName, BeginLoc);

    if (!ReplName.empty()) {
      auto BeginLoc =
          CNE->getAllocatedTypeSourceInfo()->getTypeLoc().getBeginLoc();
      emplaceTransformation(new ReplaceToken(BeginLoc, std::move(ReplName)));
    }
  }
}

REGISTER_RULE(CXXNewExprRule)

void NamespaceRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(usingDirectiveDecl().bind("usingDirective"), this);
  MF.addMatcher(namespaceAliasDecl().bind("namespaceAlias"), this);
  MF.addMatcher(usingDecl().bind("using"), this);
}

void NamespaceRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto UDD =
          getAssistNodeAsType<UsingDirectiveDecl>(Result, "usingDirective")) {
    std::string Namespace = UDD->getNominatedNamespace()->getNameAsString();
    if (Namespace == "cooperative_groups" || Namespace == "placeholders")
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
        for (auto child : TS->decls()) {
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
    auto Iter = MapNames::MathRewriterMap.find(UD->getNameAsString());
    if (Iter != MapNames::MathRewriterMap.end()) {
      DpctGlobalInfo::getInstance().insertHeader(UD->getBeginLoc(), HT_Math);
      std::string Repl{"using "};
      Repl += Iter->second;
      auto NextTok = Lexer::findNextToken(
          End, SM, DpctGlobalInfo::getContext().getLangOpts());
      if (!NextTok.hasValue() || !NextTok.getValue().is(tok::semi)) {
        Repl += ";";
      }
      emplaceTransformation(new ReplaceText(Beg, Len, std::move(Repl)));
    }
  }
}

REGISTER_RULE(NamespaceRule)

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
      Tok = Lexer::findNextToken(Tok.getLocation(), *SM, LOpts).getValue();
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

REGISTER_RULE(RemoveBaseClassRule)

void ThrustVarRule::registerMatcher(MatchFinder &MF) {
  auto hasPolicyName = [&]() { return hasAnyName("seq", "host", "device"); };

  MF.addMatcher(declRefExpr(to(varDecl(hasPolicyName()).bind("varDecl")))
                    .bind("declRefExpr"),
                this);
}

void ThrustVarRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto DRE = getNodeAsType<DeclRefExpr>(Result, "declRefExpr")) {
    auto VD = getAssistNodeAsType<VarDecl>(Result, "varDecl", false);

    if (DRE->hasQualifier()) {

      auto Namespace = DRE->getQualifierLoc()
                           .getNestedNameSpecifier()
                           ->getAsNamespace()
                           ->getNameAsString();

      if (Namespace != "thrust")
        return;

      const std::string ThrustVarName = Namespace + "::" + VD->getName().str();

      std::string Replacement =
          MapNames::findReplacedName(MapNames::TypeNamesMap, ThrustVarName);

      requestHelperFeatureForTypeNames(ThrustVarName, DRE);

      if (Replacement == "oneapi::dpl::execution::dpcpp_default")
        Replacement = makeDevicePolicy(DRE);

      if (!Replacement.empty()) {
        emplaceTransformation(new ReplaceToken(
            DRE->getBeginLoc(), DRE->getEndLoc(), std::move(Replacement)));
      }
    }
  }
}

REGISTER_RULE(ThrustVarRule)

void PreDefinedStreamHandleRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(integerLiteral(equals(0)).bind("stream"), this);
  MF.addMatcher(parenExpr(has(cStyleCastExpr(has(
                              integerLiteral(anyOf(equals(1), equals(2)))))))
                    .bind("stream"),
                this);
}

void PreDefinedStreamHandleRule::runRule(
    const MatchFinder::MatchResult &Result) {
  if (auto E = getNodeAsType<Expr>(Result, "stream")) {
    std::string Str = getStmtSpelling(E);
    if (Str == "cudaStreamDefault" || Str == "cudaStreamLegacy" ||
        Str == "cudaStreamPerThread") {
      auto &SM = DpctGlobalInfo::getSourceManager();
      auto Begin = getStmtExpansionSourceRange(E).getBegin();
      unsigned int Length = Lexer::MeasureTokenLength(
          Begin, SM, DpctGlobalInfo::getContext().getLangOpts());
      if (isPlaceholderIdxDuplicated(E))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, E, HelperFuncType::HFT_DefaultQueue);
      emplaceTransformation(new ReplaceText(
          Begin, Length, "&{{NEEDREPLACEQ" + std::to_string(Index) + "}}"));
    }
  }
}

REGISTER_RULE(PreDefinedStreamHandleRule)

void AsmRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      asmStmt(hasAncestor(functionDecl(
                  anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))
          .bind("asm"),
      this);
}

void AsmRule::runRule(const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto E = getNodeAsType<Stmt>(Result, "asm")) {
    report(E->getBeginLoc(), Diagnostics::DEVICE_ASM, true);
  }
  return;
}

REGISTER_RULE(AsmRule)

// Rule for FFT function calls.
void FFTFunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "cufftPlan1d", "cufftPlan2d", "cufftPlan3d", "cufftPlanMany",
        "cufftMakePlan1d", "cufftMakePlan2d", "cufftMakePlan3d",
        "cufftMakePlanMany", "cufftMakePlanMany64", "cufftExecC2C",
        "cufftExecR2C", "cufftExecC2R", "cufftExecZ2Z", "cufftExecZ2D",
        "cufftExecD2Z", "cufftCreate", "cufftDestroy", "cufftSetStream");
  };
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), parentStmt()))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               unless(parentStmt())))
                    .bind("FunctionCallUsed"),
                this);

  // Currently, only exec functions support function pointer migration
  auto execFunctionName = [&]() {
    return hasAnyName("cufftExecC2C", "cufftExecR2C", "cufftExecC2R",
                      "cufftExecZ2Z", "cufftExecZ2D", "cufftExecD2Z");
  };
  MF.addMatcher(varDecl(hasInitializer(unaryOperator(
                            hasOperatorName("&"),
                            hasUnaryOperand(declRefExpr(hasDeclaration(
                                functionDecl(execFunctionName())))))))
                    .bind("FunctionPointerDecl"),
                this);
  MF.addMatcher(binaryOperator(hasOperatorName("="), hasLHS(declRefExpr()),
                               hasRHS(unaryOperator(
                                   hasOperatorName("&"),
                                   hasUnaryOperand(declRefExpr(hasDeclaration(
                                       functionDecl(execFunctionName())))))))
                    .bind("FunctionPointerAssignment"),
                this);
}

void FFTFunctionCallRule::prepareFEAInfo(std::string IndentStr,
                                         std::string FuncName,
                                         std::string FuncPtrName,
                                         LibraryMigrationLocations Locations,
                                         LibraryMigrationFlags Flags,
                                         SourceLocation SL) {
  dpct::FFTFunctionCallBuilder FFCB(nullptr, IndentStr, FuncName, FuncPtrName,
                                    Locations, Flags);
  if (FuncName == "cufftExecC2C" || FuncName == "cufftExecZ2Z" ||
      FuncName == "cufftExecC2R" || FuncName == "cufftExecR2C" ||
      FuncName == "cufftExecZ2D" || FuncName == "cufftExecD2Z") {
    FFCB.updateExecCallExpr();
    FFTExecAPIInfo FEAInfo;
    FFCB.updateFFTExecAPIInfo(FEAInfo);
    FEAInfo.HandleDeclFileAndOffset = "";
    FEAInfo.QueueIndex = -1;
    FEAInfo.CompoundStmtBeginOffset = 0;
    FEAInfo.PlanHandleDeclBeginOffset = 0;
    FEAInfo.ExecAPIBeginOffset = 0;
    DpctGlobalInfo::getInstance().insertFFTExecAPIInfo(SL, FEAInfo);
  }
}

void FFTFunctionCallRule::processFunctionPointer(const VarDecl *VD) {
  std::string FuncName;
  std::string FuncPtrName;
  auto &SM = DpctGlobalInfo::getSourceManager();
  const UnaryOperator *UO = dyn_cast<UnaryOperator>(VD->getInit());
  if (!UO)
    return;
  const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(UO->getSubExpr());
  if (!DRE)
    return;
  const FunctionDecl *FD = dyn_cast<FunctionDecl>(DRE->getDecl());
  if (!FD)
    return;
  FuncName = FD->getNameAsString();
  auto SL = SM.getExpansionLoc(VD->getBeginLoc());
  std::string Key =
      SM.getFilename(SL).str() + std::to_string(SM.getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));
  FuncPtrName = VD->getNameAsString();
  if (VD->getStorageDuration() == SD_Static)
    FuncPtrName = "static " + FuncPtrName;

  LibraryMigrationFlags Flags;
  Flags.IsAssigned = false;
  Flags.IsFunctionPointer = true;
  Flags.IsFunctionPointerAssignment = false;
  LibraryMigrationStrings ReplaceStrs;
  LibraryMigrationLocations Locations;
  initVars(nullptr, VD, nullptr, Flags, ReplaceStrs, Locations);

  prepareFEAInfo(ReplaceStrs.IndentStr, FuncName, FuncPtrName, Locations, Flags,
                 SL);
}

void FFTFunctionCallRule::processFunctionPointerAssignment(
    const BinaryOperator *BO) {
  std::string FuncName;
  std::string FuncPtrName;
  auto &SM = DpctGlobalInfo::getSourceManager();
  const UnaryOperator *UO = dyn_cast<UnaryOperator>(BO->getRHS());
  if (!UO)
    return;
  const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(UO->getSubExpr());
  if (!DRE)
    return;
  const FunctionDecl *FD = dyn_cast<FunctionDecl>(DRE->getDecl());
  if (!FD)
    return;
  FuncName = FD->getNameAsString();
  auto SL = SM.getExpansionLoc(BO->getBeginLoc());
  std::string Key =
      SM.getFilename(SL).str() + std::to_string(SM.getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));

  LibraryMigrationFlags Flags;
  Flags.IsAssigned = false;
  Flags.IsFunctionPointer = false;
  Flags.IsFunctionPointerAssignment = true;
  LibraryMigrationStrings ReplaceStrs;
  LibraryMigrationLocations Locations;
  initVars(nullptr, nullptr, BO, Flags, ReplaceStrs, Locations);

  prepareFEAInfo(ReplaceStrs.IndentStr, FuncName, FuncPtrName, Locations, Flags,
                 SL);
}

void FFTFunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  const VarDecl *VD = getNodeAsType<VarDecl>(Result, "FunctionPointerDecl");
  const BinaryOperator *BO =
      getNodeAsType<BinaryOperator>(Result, "FunctionPointerAssignment");

  if (!CE) {
    CE = getNodeAsType<CallExpr>(Result, "FunctionCallUsed");
    if (CE) {
      IsAssigned = true;
    } else if (VD) {
      processFunctionPointer(VD);
      return;
    } else if (BO) {
      processFunctionPointerAssignment(BO);
      return;
    } else {
      return;
    }
  }

  std::string FuncName;
  std::string FuncPtrName;

  auto &SM = DpctGlobalInfo::getSourceManager();

  if (!CE->getDirectCallee())
    return;
  auto SL = SM.getExpansionLoc(CE->getBeginLoc());
  std::string Key =
      SM.getFilename(SL).str() + std::to_string(SM.getDecomposedLoc(SL).second);
  DpctGlobalInfo::updateInitSuffixIndexInRule(
      DpctGlobalInfo::getSuffixIndexInitValue(Key));
  FuncName = CE->getDirectCallee()->getNameInfo().getName().getAsString();

  StringRef FuncNameRef(FuncName);

  LibraryMigrationFlags Flags;
  Flags.IsAssigned = IsAssigned;
  Flags.IsFunctionPointer = false;
  Flags.IsFunctionPointerAssignment = false;
  LibraryMigrationStrings ReplaceStrs;
  LibraryMigrationLocations Locations;
  initVars(CE, VD, BO, Flags, ReplaceStrs, Locations);

  dpct::FFTFunctionCallBuilder FFCB(CE, ReplaceStrs.IndentStr, FuncName,
                                    FuncPtrName, Locations, Flags);
  if (FuncName == "cufftSetStream") {
    const DeclaratorDecl *DD = getHandleVar(CE->getArg(0));
    if (!DD)
      return;

    SourceLocation SL = SM.getExpansionLoc(DD->getBeginLoc());
    std::string HandleInfoKey =
        DpctGlobalInfo::getLocInfo(SL).first + ":" +
        std::to_string(DpctGlobalInfo::getLocInfo(SL).second);

    std::string StreamStr = getDrefName(CE->getArg(1));

    if (IsAssigned) {
      emplaceTransformation(new ReplaceStmt(CE, false, "0"));
    } else {
      emplaceTransformation(new ReplaceStmt(CE, false, ""));
    }

    const CompoundStmt *CS =
        findTheOuterMostCompoundStmtUntilMeetControlFlowNodes(CE);
    if (!CS)
      return;
    SourceLocation CompoundStmtBeginSL = SM.getExpansionLoc(CS->getBeginLoc());
    SourceLocation PlanHandleDeclBeginSL =
        SM.getExpansionLoc(DD->getBeginLoc());
    SourceLocation SetStreamAPIBeginSL = SM.getExpansionLoc(CE->getBeginLoc());
    DpctGlobalInfo::getInstance().updateFFTSetStreamAPIInfoMap(
        CompoundStmtBeginSL, PlanHandleDeclBeginSL, SetStreamAPIBeginSL,
        StreamStr);
    return;
  } else if (FuncName == "cufftCreate" || FuncName == "cufftDestroy") {
    if (IsAssigned) {
      report(Locations.PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED_0, false,
             FuncName, "the function call is redundant in DPC++.");
      emplaceTransformation(new ReplaceStmt(CE, false, "0"));
    } else {
      report(Locations.PrefixInsertLoc, Diagnostics::FUNC_CALL_REMOVED, false,
             FuncName, "the function call is redundant in DPC++.");
      emplaceTransformation(new ReplaceStmt(CE, false, ""));
    }
    return;
  } else if (FuncName == "cufftPlan1d" || FuncName == "cufftMakePlan1d" ||
             FuncName == "cufftPlan2d" || FuncName == "cufftMakePlan2d" ||
             FuncName == "cufftPlan3d" || FuncName == "cufftMakePlan3d" ||
             FuncName == "cufftPlanMany" || FuncName == "cufftMakePlanMany" ||
             FuncName == "cufftMakePlanMany64") {

    const DeclaratorDecl *DD = getHandleVar(CE->getArg(0));
    if (!DD)
      return;
    SourceLocation SL = SM.getExpansionLoc(DD->getBeginLoc());
    std::string HandleDeclFileAndOffset =
        DpctGlobalInfo::getLocInfo(SL).first + ":" +
        std::to_string(DpctGlobalInfo::getLocInfo(SL).second);

    FFTPlanAPIInfo FPAInfo;
    FFCB.updateFFTPlanAPIInfo(FPAInfo, Flags);
    FFCB.updateFFTHandleInfoFromPlan(HandleDeclFileAndOffset);
    replacementLocation(Locations, Flags, FPAInfo.ReplaceOffset,
                        FPAInfo.ReplaceLen, FPAInfo.InsertOffsets,
                        FPAInfo.FilePath);
    FPAInfo.HandleDeclFileAndOffset = HandleDeclFileAndOffset;
    if (FuncNameRef.startswith("cufftMake")) {
      FPAInfo.UnsupportedArg =
          ExprAnalysis::ref(CE->getArg(CE->getNumArgs() - 1));
    }

    DpctGlobalInfo::getInstance().insertFFTPlanAPIInfo(
        SM.getExpansionLoc(CE->getBeginLoc()), FPAInfo);
    return;
  } else if (FuncName == "cufftExecC2C" || FuncName == "cufftExecZ2Z" ||
             FuncName == "cufftExecC2R" || FuncName == "cufftExecR2C" ||
             FuncName == "cufftExecZ2D" || FuncName == "cufftExecD2Z") {
    std::string FFTHandleInfoKey;
    int Index = -1;
    unsigned int CompoundStmtBeginOffset = 0;
    unsigned int PlanHandleDeclBeginOffset = 0;
    unsigned int ExecAPIBeginOffset = 0;

    const DeclaratorDecl *DD = getHandleVar(CE->getArg(0));
    if (!DD)
      return;
    if (isPlaceholderIdxDuplicated(CE))
      return;

    Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);

    SourceLocation SL = SM.getExpansionLoc(DD->getBeginLoc());
    FFTHandleInfoKey = DpctGlobalInfo::getLocInfo(SL).first + ":" +
                       std::to_string(DpctGlobalInfo::getLocInfo(SL).second);

    FFCB.updateExecCallExpr(FFTHandleInfoKey);

    const CompoundStmt *CS =
        findTheOuterMostCompoundStmtUntilMeetControlFlowNodes(CE);
    if (CS) {
      CompoundStmtBeginOffset =
          DpctGlobalInfo::getLocInfo(SM.getExpansionLoc(CS->getBeginLoc()))
              .second;
      PlanHandleDeclBeginOffset =
          DpctGlobalInfo::getLocInfo(SM.getExpansionLoc(DD->getBeginLoc()))
              .second;
      ExecAPIBeginOffset =
          DpctGlobalInfo::getLocInfo(SM.getExpansionLoc(CE->getBeginLoc()))
              .second;
    }

    SourceLocation TypeBegin;
    int TypeLength = 0;
    if (FFCB.moveDeclOutOfBracesIfNeeds(Flags, TypeBegin, TypeLength)) {
      emplaceTransformation(new ReplaceText(TypeBegin, TypeLength, ""));
    }

    FFTExecAPIInfo FEAInfo;
    FFCB.updateFFTExecAPIInfo(FEAInfo);
    FEAInfo.HandleDeclFileAndOffset = FFTHandleInfoKey;
    FEAInfo.QueueIndex = Index;
    FEAInfo.CompoundStmtBeginOffset = CompoundStmtBeginOffset;
    FEAInfo.PlanHandleDeclBeginOffset = PlanHandleDeclBeginOffset;
    FEAInfo.ExecAPIBeginOffset = ExecAPIBeginOffset;

    // If previous stat is setStream(plan, s), then using "s" as queue and
    // not emit warning.
    std::string DefiniteStream;
    if (isPreviousStmtRelatedSetStream(CE, Index, DefiniteStream)) {
      FEAInfo.DefiniteStream = DefiniteStream;
    }

    DpctGlobalInfo::getInstance().insertFFTExecAPIInfo(
        SM.getExpansionLoc(CE->getBeginLoc()), FEAInfo);
    return;
  }
}

REGISTER_RULE(FFTFunctionCallRule)

void DriverModuleAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto DriverModuleAPI = [&]() {
    return hasAnyName("cuModuleLoad", "cuModuleLoadData", "cuModuleUnload",
                      "cuModuleGetFunction", "cuLaunchKernel",
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
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "callUsed"))) {
      return;
    }
    IsAssigned = true;
  }
  (void)IsAssigned;

  std::string APIName = "";
  if (auto DC = CE->getDirectCallee()) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    APIName = DC->getNameAsString();
    DpctGlobalInfo::getInstance().insertHeader(
        SM.getExpansionLoc(CE->getBeginLoc()), HT_DL);
  } else {
    return;
  }

  if (APIName == "cuModuleLoad" || APIName == "cuModuleLoadData") {
    report(CE->getBeginLoc(), Diagnostics::MODULE_FILENAME_MANUAL_FIX, false);
  }

  ExprAnalysis EA;
  EA.analyze(CE);
  emplaceTransformation(EA.getReplacement());
}

REGISTER_RULE(DriverModuleAPIRule)

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
      OS << "(";
    auto FirArg = CE->getArg(0)->IgnoreImplicitAsWritten();
    auto SecArg = CE->getArg(1)->IgnoreImplicitAsWritten();

    ExprAnalysis SecEA(SecArg);
    SecEA.analyze();
    std::string Rep;
    printDerefOp(OS, FirArg);
    OS << " = " << SecEA.getReplacedString();
    if (IsAssigned) {
      OS << ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDeviceGetName") {
    if (IsAssigned)
      OS << "(";
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
    requestFeature(HelperFeatureEnum::Device_dev_mgr_get_device, CE);
    if (IsAssigned) {
      OS << ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
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
    std::string ThrRep;
    ExprAnalysis EA(ThrArg);
    EA.analyze();
    ThrRep = EA.getReplacedString();
    std::string common_str = " = " + MapNames::getDpctNamespace() +
                             "dev_mgr::instance().get_device(";
    std::string major_api = ").get_major_version()";
    std::string minor_api = ").get_minor_version()";
    requestFeature(HelperFeatureEnum::Device_dev_mgr_get_device, CE);
    requestFeature(HelperFeatureEnum::Device_device_ext_get_major_version, CE);
    requestFeature(HelperFeatureEnum::Device_device_ext_get_minor_version, CE);
    if (IsAssigned) {
      OS << Indent << "  ";
      printDerefOp(OS, FirArg);
      OS << common_str << ThrRep << major_api << ";" << getNL();
      OS << Indent << "  ";
      printDerefOp(OS, SecArg);
      OS << common_str << ThrRep << minor_api << ";" << getNL();
      OS << Indent << "  "
         << "return 0;" << getNL();
    } else {
      printDerefOp(OS, FirArg);
      OS << common_str << ThrRep << major_api << ";" << getNL() << Indent;
      printDerefOp(OS, SecArg);
      OS << common_str << ThrRep << minor_api;
    }
    if (IsAssigned) {
      OS << Indent << "}()";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_LAMBDA, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDriverGetVersion") {
    if (IsAssigned)
      OS << "(";
    auto FirArg = CE->getArg(0)->IgnoreImplicitAsWritten();
    printDerefOp(OS, FirArg);
    OS << " = " << MapNames::getDpctNamespace()
       << "get_current_device()"
          ".get_info<" +
              MapNames::getClNamespace() + "info::device::version>()";
    requestFeature(HelperFeatureEnum::Device_get_current_device, CE);
    if (IsAssigned) {
      OS << ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    report(CE->getBeginLoc(), Diagnostics::TYPE_MISMATCH, false);
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDeviceGetCount") {
    if (IsAssigned)
      OS << "(";
    auto Arg = CE->getArg(0)->IgnoreImplicitAsWritten();
    printDerefOp(OS, Arg);
    OS << " = "
       << MapNames::getDpctNamespace() + "dev_mgr::instance().device_count()";
    requestFeature(HelperFeatureEnum::Device_dev_mgr_device_count, CE);
    if (IsAssigned) {
      OS << ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDeviceGetAttribute") {
    if (IsAssigned)
      OS << "(";
    auto FirArg = CE->getArg(0)->IgnoreImplicitAsWritten();
    auto SecArg = CE->getArg(1);
    auto ThrArg = CE->getArg(2)->IgnoreImplicitAsWritten();

    std::string AttributeName;
    std::string DevStr;
    std::string SYCLCallName;
    if (auto DRE = dyn_cast<DeclRefExpr>(SecArg)) {
      AttributeName = DRE->getNameInfo().getAsString();
      auto Search = EnumConstantRule::EnumNamesMap.find(AttributeName);
      if (Search == EnumConstantRule::EnumNamesMap.end()) {
        report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
        return;
      }
      requestHelperFeatureForEnumNames(AttributeName, CE);
      SYCLCallName = Search->second;
    } else {
      report(CE->getBeginLoc(), Diagnostics::UNPROCESSED_DEVICE_ATTRIBUTE,
             false, "recognized by the Intel(R) DPC++ Compatibility Tool");
      return;
    }
    printDerefOp(OS, FirArg);
    ExprAnalysis EA(ThrArg);
    EA.analyze();
    DevStr = EA.getReplacedString();
    OS << " = " << MapNames::getDpctNamespace()
       << "dev_mgr::instance().get_device(" << DevStr << ")." << SYCLCallName
       << "()";
    requestFeature(HelperFeatureEnum::Device_dev_mgr_get_device, CE);
    if (IsAssigned) {
      OS << ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  }
}

REGISTER_RULE(DriverDeviceAPIRule)

void DriverContextAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto contextAPI = [&]() {
    return hasAnyName("cuInit", "cuCtxCreate_v2", "cuCtxSetCurrent",
                      "cuCtxGetCurrent", "cuCtxSynchronize", "cuCtxDestroy_v2");
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
  if (IsAssigned) {
    OS << "(";
  }
  if (APIName == "cuInit") {
    std::string Msg = "the function call is redundant in DPC++.";
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
  } else if (APIName == "cuCtxCreate_v2" || APIName == "cuCtxDestroy_v2") {
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

    if (APIName == "cuCtxDestroy_v2") {
      std::string Msg = "the function call is redundant in DPC++.";
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               APIName, Msg);
        emplaceTransformation(replaceText(CallBegin, CallEnd, "0", SM));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               APIName, Msg);
        CallEnd = CallEnd.getLocWithOffset(1);
        emplaceTransformation(replaceText(CallBegin, CallEnd, "", SM));
      }
      return;
    }

    auto CtxArg = CE->getArg(0)->IgnoreImplicitAsWritten();
    auto DevArg = CE->getArg(2)->IgnoreImplicitAsWritten();
    ExprAnalysis EA(DevArg);
    EA.analyze();
    printDerefOp(OS, CtxArg);
    OS << " = " << EA.getReplacedString();
    if (IsAssigned) {
      OS << ", 0)";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
    }

    emplaceTransformation(replaceText(CallBegin, CallEnd, OS.str(), SM));
    return;
  } else if (APIName == "cuCtxSetCurrent") {
    auto Arg = CE->getArg(0)->IgnoreImplicitAsWritten();
    ExprAnalysis EA(Arg);
    EA.analyze();
    OS << MapNames::getDpctNamespace() + "dev_mgr::instance().select_device("
       << EA.getReplacedString() << ")";
    requestFeature(HelperFeatureEnum::Device_dev_mgr_select_device, CE);
  } else if (APIName == "cuCtxGetCurrent") {
    auto Arg = CE->getArg(0)->IgnoreImplicitAsWritten();
    printDerefOp(OS, Arg);
    OS << " = "
       << MapNames::getDpctNamespace() +
              "dev_mgr::instance().current_device_id()";
    requestFeature(HelperFeatureEnum::Device_dev_mgr_current_device_id, CE);
  } else if (APIName == "cuCtxSynchronize") {
    OS << MapNames::getDpctNamespace() +
              "get_current_device().queues_wait_and_throw()";
    requestFeature(HelperFeatureEnum::Device_get_current_device, CE);
    requestFeature(HelperFeatureEnum::Device_device_ext_queues_wait_and_throw,
                   CE);
  }
  if (IsAssigned) {
    OS << ", 0)";
    report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_COMMA_OP, false);
  }
  emplaceTransformation(new ReplaceStmt(CE, OS.str()));
}

REGISTER_RULE(DriverContextAPIRule)

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
  // process __host__ __device__ function definition except overloaded operator
  if (FD && !FD->isOverloadedOperator() &&
      FD->getTemplateSpecializationKind() ==
          TemplateSpecializationKind::TSK_Undeclared) {
    auto NameInfo = FD->getNameInfo();
    /// TODO: add support for macro
    if (NameInfo.getBeginLoc().isMacroID())
      return;
    auto BeginLoc = SM.getExpansionLoc(FD->getBeginLoc());
    HostDeviceFuncInfo HDFI;
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
    if (T.hasValue() && T.getValue().is(tok::TokenKind::semi)) {
      End = Global.getLocInfo(T.getValue().getLocation());
    }
    auto FileInfo = DpctGlobalInfo::getInstance().insertFile(Beg.first);
    std::string &FileContent = FileInfo->getFileContent();
    auto NameLocInfo = Global.getLocInfo(NameInfo.getBeginLoc());
    HDFI.FuncStartOffset = Beg.second;
    HDFI.FuncEndOffset = End.second;
    HDFI.FuncNameOffset = NameLocInfo.second + NameInfo.getAsString().length();
    HDFI.FuncContentCache =
        FileContent.substr(Beg.second, End.second - Beg.second + 1);
    if (!FD->isThisDeclarationADefinition()) {
      Global.insertHostDeviceFuncDeclInfo(
          DNG.getName(FD), std::make_pair(NameLocInfo.first, HDFI));
      return;
    }
    bool NeedInsert = false;
    for (auto &Info : Global.getCudaArchPPInfoMap()[FileInfo->getFilePath()]) {
      if ((Info.first > Beg.second) && (Info.first < End.second)) {
        Info.second.isInHDFunc = true;
        NeedInsert = true;
      }
    }
    if (NeedInsert)
      Global.insertHostDeviceFuncDefInfo(
          DNG.getName(FD), std::make_pair(NameLocInfo.first, HDFI));
  } // address __host__ __device__ function call
  else if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "callExpr")) {
    /// TODO: add support for macro
    if (CE->getBeginLoc().isMacroID())
      return;
    if (auto *PF = DpctGlobalInfo::getParentFunction(CE)) {
      if (PF->hasAttr<CUDADeviceAttr>() || PF->hasAttr<CUDAGlobalAttr>())
        return;
    }
    const FunctionDecl *DC = CE->getDirectCallee();
    if (DC) {
      unsigned int Offset = DC->getNameAsString().length();
      std::string Name(DNG.getName(DC));
      if (DC->isTemplateInstantiation()) {
        if (auto DFT = DC->getPrimaryTemplate()) {
          const FunctionDecl *TFD = DFT->getTemplatedDecl();
          if (TFD)
            Name = DNG.getName(TFD);
        }
      }
      auto LocInfo = Global.getLocInfo(CE->getBeginLoc());
      Global.insertHostDeviceFuncCallInfo(
          std::move(Name),
          std::make_pair(LocInfo.first, LocInfo.second + Offset));
    }
  }
}
REGISTER_RULE(CudaArchMacroRule)

void CubRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(hasAnyName(
                  "WarpScan", "WarpReduce", "BlockScan", "BlockReduce"))))))
          .bind("TypeLoc"),
      this);

  MF.addMatcher(
      typedefDecl(
          hasType(hasCanonicalType(qualType(hasDeclaration(namedDecl(hasAnyName(
              "WarpScan", "WarpReduce", "BlockScan", "BlockReduce")))))))
          .bind("TypeDefDecl"),
      this);

  MF.addMatcher(
      declStmt(
          has(varDecl(anyOf(
              hasType(hasCanonicalType(qualType(
                  hasDeclaration(namedDecl(hasAnyName("TempStorage")))))),
              hasType(arrayType(hasElementType(hasCanonicalType(qualType(
                  hasDeclaration(namedDecl(hasAnyName("TempStorage"))))))))))))
          .bind("DeclStmt"),
      this);

  MF.addMatcher(cxxMemberCallExpr(has(memberExpr(member(hasAnyName(
                                      "InclusiveSum", "ExclusiveSum",
                                      "InclusiveScan", "ExclusiveScan",
                                      "Reduce", "Sum", "Broadcast", "Scan")))))
                    .bind("MemberCall"),
                this);

  MF.addMatcher(callExpr(allOf(callee(functionDecl(hasAnyName(
                                   "ShuffleIndex", "ThreadLoad", "ThreadStore",
                                   "Sum", "Min", "Max", "Reduce"))),
                               parentStmt()))
                    .bind("FuncCall"),
                this);

  MF.addMatcher(callExpr(allOf(callee(functionDecl(
                                   hasAnyName("Sum", "Min", "Max", "Reduce",
                                              "ThreadLoad", "ShuffleIndex"))),
                               unless(parentStmt())))
                    .bind("FuncCallUsed"),
                this);
}

std::string CubRule::getOpRepl(const Expr *Operator) {
  std::string OpRepl;
  if (!Operator) {
    return MapNames::getClNamespace() + "ext::oneapi::plus<>()";
  }
  if (auto Op = dyn_cast<CXXConstructExpr>(Operator)) {
    auto CtorArg = Op->getArg(0)->IgnoreImplicitAsWritten();
    if (auto DRE = dyn_cast<DeclRefExpr>(CtorArg)) {
      auto D = DRE->getDecl();
      if (!D)
        return OpRepl;
      std::string OpType = D->getType().getCanonicalType().getAsString();
      if (OpType == "struct cub::Sum" || OpType == "struct cub::Max" ||
          OpType == "struct cub::Min") {
        ExprAnalysis EA(Operator);
        OpRepl = EA.getReplacedString();
      }
    } else if (auto CXXTempObj = dyn_cast<CXXTemporaryObjectExpr>(CtorArg)) {
      std::string OpType =
          CXXTempObj->getType().getCanonicalType().getAsString();
      if (OpType == "struct cub::Sum") {
        OpRepl = MapNames::getClNamespace() + "ext::oneapi::plus<>()";
      } else if (OpType == "struct cub::Max") {
        OpRepl = MapNames::getClNamespace() + "ext::oneapi::maximum<>()";
      } else if (OpType == "struct cub::Min") {
        OpRepl = MapNames::getClNamespace() + "ext::oneapi::minimum<>()";
      }
    }
  }
  return OpRepl;
}
void CubRule::processCubDeclStmt(const DeclStmt *DS) {
  std::string Repl;
  for (auto Decl : DS->decls()) {
    auto VDecl = dyn_cast<VarDecl>(Decl);
    if (!VDecl)
      return;
    std::string VarType =
        VDecl->getTypeSourceInfo()->getType().getCanonicalType().getAsString();
    std::string VarName = VDecl->getNameAsString();

    auto MatcherScope = DpctGlobalInfo::findAncestor<CompoundStmt>(Decl);
    if (!isCubVar(VDecl)) {
      return;
    }
    // always remove TempStorage variable declaration
    emplaceTransformation(new ReplaceStmt(DS, ""));

    // process TempStorage used in class constructor
    auto TempVarMatcher = compoundStmt(forEachDescendant(
        declRefExpr(to(varDecl(hasName(VarName)))).bind("TempVar")));
    auto MatchResult = ast_matchers::match(TempVarMatcher, *MatcherScope,
                                           DpctGlobalInfo::getContext());
    for (auto &Element : MatchResult) {
      auto DRE = Element.getNodeAs<DeclRefExpr>("TempVar");
      auto ObjDecl = DpctGlobalInfo::findAncestor<VarDecl>(DRE);
      if (!ObjDecl || !DRE) {
        continue;
      }
      auto ArrayOfDRE = DpctGlobalInfo::findAncestor<ArraySubscriptExpr>(DRE);
      auto ObjCanonicalType = ObjDecl->getType().getCanonicalType();
      std::string ObjTypeStr = ObjCanonicalType.getAsString();
      if (isTypeInRoot(ObjCanonicalType.getTypePtr())) {
        continue;
      } else if (ObjTypeStr.find("class cub::WarpScan") == 0 ||
                 ObjTypeStr.find("class cub::WarpReduce") == 0) {
        Repl = DpctGlobalInfo::getSubGroup(DRE);
      } else if (ObjTypeStr.find("class cub::BlockScan") == 0 ||
                 ObjTypeStr.find("class cub::BlockReduce") == 0) {
        Repl = DpctGlobalInfo::getGroup(DRE);
      } else {
        continue;
      }
      if (ArrayOfDRE) {
        emplaceTransformation(new ReplaceStmt(ArrayOfDRE, Repl));
      } else {
        emplaceTransformation(new ReplaceStmt(DRE, Repl));
      }
    }
  }
}
void CubRule::processCubTypeDef(const TypedefDecl *TD) {
  auto CanonicalType = TD->getUnderlyingType().getCanonicalType();
  std::string CanonicalTypeStr = CanonicalType.getAsString();
  std::string TypeName = TD->getNameAsString();
  if (isTypeInRoot(CanonicalType.getTypePtr()) ||
      CanonicalTypeStr.find("class cub::") != 0) {
    return;
  }
  auto &Context = clang::dpct::DpctGlobalInfo::getContext();
  auto &SM = clang::dpct::DpctGlobalInfo::getSourceManager();
  auto MyMatcher = compoundStmt(forEachDescendant(
      typeLoc(loc(qualType(hasDeclaration(typedefDecl(hasName(TypeName))))))
          .bind("typeLoc")));
  auto MatcherScope = DpctGlobalInfo::findAncestor<CompoundStmt>(TD);
  if (!MatcherScope)
    return;
  auto TypeLocMatchResult =
      ast_matchers::match(MyMatcher, *MatcherScope, Context);
  bool DeleteFlag = true;
  // Currently, typedef decl can be deteled in following cases
  for (auto &Element : TypeLocMatchResult) {
    if (auto TL = Element.getNodeAs<TypeLoc>("typeLoc")) {
      // 1.Used in TempStorage variable declaration
      if (auto AncestorVD = DpctGlobalInfo::findAncestor<VarDecl>(TL)) {
        auto VarType = AncestorVD->getType().getCanonicalType();
        std::string VarTypeStr =
            AncestorVD->getType().getCanonicalType().getAsString();
        if (isTypeInRoot(VarType.getTypePtr()) ||
            !(VarTypeStr.find("TempStorage") != std::string::npos &&
              VarTypeStr.find("struct cub::") == 0)) {
          DeleteFlag = false;
          break;
        }
      } // 2.Used in temporary class constructor
      else if (auto AncestorMTE =
                   DpctGlobalInfo::findAncestor<MaterializeTemporaryExpr>(TL)) {
        auto MC = DpctGlobalInfo::findAncestor<CXXMemberCallExpr>(AncestorMTE);
        if (MC) {
          auto ObjType = MC->getObjectType().getCanonicalType();
          std::string ObjTypeStr = ObjType.getAsString();
          if (isTypeInRoot(ObjType.getTypePtr()) ||
              !(ObjTypeStr.find("class cub::WarpScan") == 0 ||
                ObjTypeStr.find("class cub::WarpReduce") == 0 ||
                ObjTypeStr.find("class cub::BlockScan") == 0 ||
                ObjTypeStr.find("class cub::BlockReduce") == 0)) {
            DeleteFlag = false;
            break;
          }
        }
      } // 3.Used in self typedef decl
      else if (auto AncestorTD =
                   DpctGlobalInfo::findAncestor<TypedefDecl>(TL)) {
        if (AncestorTD != TD) {
          DeleteFlag = false;
          break;
        }
      } else {
        DeleteFlag = false;
        break;
      }
    }
  }
  if (DeleteFlag) {
    emplaceTransformation(new ReplaceDecl(TD, ""));
  } else {
    auto BeginLoc =
        SM.getExpansionLoc(TD->getTypeSourceInfo()->getTypeLoc().getBeginLoc());
    auto EndLoc =
        SM.getExpansionLoc(TD->getTypeSourceInfo()->getTypeLoc().getEndLoc());
    if (CanonicalTypeStr.find("Warp") != std::string::npos) {
      emplaceTransformation(replaceText(
          BeginLoc, EndLoc.getLocWithOffset(1),
          MapNames::getClNamespace() + "ext::oneapi::sub_group", SM));
    } else if (CanonicalTypeStr.find("Block") != std::string::npos) {
      auto DeviceFuncDecl = DpctGlobalInfo::findAncestor<FunctionDecl>(TD);
      if (DeviceFuncDecl && (DeviceFuncDecl->hasAttr<CUDADeviceAttr>() ||
                             DeviceFuncDecl->hasAttr<CUDAGlobalAttr>())) {
        if (auto DI = DeviceFunctionDecl::LinkRedecls(DeviceFuncDecl)) {
          auto &Map = DpctGlobalInfo::getInstance().getCubPlaceholderIndexMap();
          Map.insert({PlaceholderIndex, DI});
          emplaceTransformation(
              replaceText(BeginLoc, EndLoc.getLocWithOffset(1),
                          MapNames::getClNamespace() + "group<{{NEEDREPLACEC" +
                              std::to_string(PlaceholderIndex++) + "}}>",
                          SM));
        }
      }
    }
  }
}

// function to remove a temp storage vardecl
void CubRule::removeVarDecl(const VarDecl *VD) {
  static std::unordered_map<std::string, std::vector<bool>> DeclStmtBitMap;
  if (!VD) {
    return;
  }
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &Context = DpctGlobalInfo::getContext();
  if (auto DS = Context.getParents(*VD)[0].get<DeclStmt>()) {
    auto LocInfo = DpctGlobalInfo::getLocInfo(DS->getBeginLoc());
    std::string Key = LocInfo.first + std::to_string(LocInfo.second);
    auto Decls = DS->decls();
    unsigned int DeclNum = DS->decls().end() - DS->decls().begin();
    // if this declstmt has one sub decl, then we just need to remove whole
    // declstmt simply.
    if (DeclNum == 1) {
      emplaceTransformation(new ReplaceStmt(DS, ""));
      return;
    }
    if (!DeclStmtBitMap.count(Key)) {
      DeclStmtBitMap[Key] =
          std::vector<bool>(DS->decls().end() - DS->decls().begin(), true);
    }
    auto NameLength = VD->getNameAsString().length();
    auto DeclBegPtr = SM.getCharacterData(VD->getBeginLoc());
    for (auto decl_itr = Decls.begin(); decl_itr != Decls.end(); decl_itr++) {
      if (auto SubDecl = dyn_cast<VarDecl>(*decl_itr)) {
        if (SubDecl == VD) {
          int InitLength = 0;
          if (SubDecl->hasInit()) {
            ExprAnalysis InitEA(SubDecl->getInit());
            InitLength = InitEA.getReplacedString().length();
          }
          /// Example1(for non first decl):              Example2(for first
          /// decl): Step 1: Init Beg and End                   Step 1: Init Beg
          /// and End int *a = nullptr, b = 100;                 int *a =
          /// nullptr, b = 100;
          ///                   ^     ^                       ^         ^
          ///                  Beg   End                     Beg       End
          ///
          /// Step 2: Adjust the Beg to previous comma   Stemp2: Adjust the Beg
          /// to the
          ///                                            begin of
          ///                                            prt-declarator, the End
          ///                                            to the behind comma.
          /// int **a = nullptr, b = 100;                int **a = nullptr, b =
          /// 100;
          ///                  ^       ^                     ^            ^
          ///                 Beg     End                   Beg          End
          ///
          /// Step 3: Remove code from Beg to End
          SourceLocation Beg = SM.getExpansionLoc(SubDecl->getEndLoc());
          SourceLocation End =
              SubDecl->hasInit()
                  ? SM.getExpansionLoc(SubDecl->getInit()->getBeginLoc())
                        .getLocWithOffset(InitLength - 1)
                  : SubDecl->getEndLoc().getLocWithOffset(NameLength - 1);
          if (decl_itr != Decls.begin()) {
            auto BegPtr = SM.getCharacterData(Beg);
            auto CommaPtr = BegPtr;
            while (CommaPtr && (CommaPtr > DeclBegPtr) && *(CommaPtr) != ',') {
              CommaPtr--;
            };
            if (CommaPtr == DeclBegPtr) {
              return;
            } else {
              if (decl_itr - Decls.begin() == 1 && !DeclStmtBitMap[Key][0]) {
                CommaPtr++;
              }
              Beg = Beg.getLocWithOffset(CommaPtr - BegPtr);
            }
          } else {
            QualType TypeTemp = VD->getType();
            while (TypeTemp->isPointerType()) {
              auto BegPtr = SM.getCharacterData(Beg);
              auto StarPtr = BegPtr;
              while (StarPtr && (StarPtr > DeclBegPtr) &&
                     (*(StarPtr) != '*' || StarPtr == BegPtr)) {
                StarPtr--;
              };
              if (StarPtr == DeclBegPtr) {
                return;
              }
              Beg = Beg.getLocWithOffset(StarPtr - BegPtr);
              TypeTemp = TypeTemp->getPointeeType();
            };
            auto tok = Lexer::findNextToken(End, SM, Context.getLangOpts());
            if (tok.hasValue() && tok.getValue().is(tok::comma)) {
              End = tok.getValue().getLocation();
            } else {
              return;
            }
          }
          DeclStmtBitMap[Key][decl_itr - Decls.begin()] = false;
          // if all sub decls are removed, we need to remove this declstmt
          if (std::find(DeclStmtBitMap[Key].begin(), DeclStmtBitMap[Key].end(),
                        true) == DeclStmtBitMap[Key].end()) {
            auto DeclPRInfo = std::make_shared<PriorityReplInfo>();
            DeclPRInfo->Repls.emplace_back(
                (new ReplaceStmt(DS, ""))->getReplacement(Context));
            DeclPRInfo->Priority = 1;
            DpctGlobalInfo::addPriorityReplInfo(Key, DeclPRInfo);
          } else {
            auto SubDeclPRInfo = std::make_shared<PriorityReplInfo>();
            SubDeclPRInfo->Repls.emplace_back(
                replaceText(Beg, End.getLocWithOffset(1), "", SM)
                    ->getReplacement(Context));
            DpctGlobalInfo::addPriorityReplInfo(Key, SubDeclPRInfo);
          }
          break;
        }
      }
    }
  }
}

/// Pesudo code:
/// loop_1 {
///   ...
///   tempstorage = nullptr;
///   ...
///   loop_j {
///     ...
///     loop_N {
///       func(tempstorage, ...);
///       tempstorage = ...
///     }
///   }
/// }
/// The callexpr is redundant if following two conditions are meet:
/// (1) No modified reference between tempstorage initialization and callexpr.
/// (2) No modified reference in loop_j or deeper loop.
/// The redundant callexpr can be remove safely.
bool CubRule::isRedundantCallExpr(const CallExpr *CE) {
  auto FuncArgs = CE->getArgs();
  auto TempStorage = FuncArgs[0]->IgnoreImplicitAsWritten();
  SourceLocation InitLoc;
  std::vector<const DeclRefExpr *> TempStorageMatchResult;
  std::vector<const DeclRefExpr *> TempStorageSizeMatchResult;
  auto isNullptrOrZero = [](const Expr *E) {
    if (!E) {
      return false;
    } else if (E->getStmtClass() == Stmt::StmtClass::GNUNullExprClass ||
               E->getStmtClass() ==
                   Stmt::StmtClass::CXXNullPtrLiteralExprClass) {
      return true;
    } else if (E->isEvaluatable(DpctGlobalInfo::getContext())) {
      Expr::EvalResult Result;
      E->EvaluateAsRValue(Result, DpctGlobalInfo::getContext());
      if (Result.Val.isInt() && Result.Val.getInt() == 0) {
        return true;
      }
    }
    return false;
  };
  auto DRE = dyn_cast<DeclRefExpr>(TempStorage);
  if (!DRE) {
    return false;
  }
  auto VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!VD) {
    return false;
  }

  const Expr *Init = nullptr;
  SourceLocation SearchEndLoc =
      DpctGlobalInfo::getSourceManager().getExpansionLoc(DRE->getBeginLoc());
  SourceLocation LastModifiedLoc;
  std::vector<const Stmt *> DRELoopList;
  std::vector<const Stmt *> CELoopList;
  findAllVarRef(DRE, TempStorageMatchResult);
  findLoop(CE, CELoopList);
  if (VD->hasInit()) {
    // tempstorage = nullptr/NULL/0/...
    if (VD->getInitStyle() == VarDecl::InitializationStyle::CInit) {
      Init = VD->getInit()->IgnoreImplicitAsWritten();
      if (isNullptrOrZero(Init)) {
        InitLoc = DpctGlobalInfo::getSourceManager().getExpansionLoc(
            VD->getBeginLoc());
      }
      // tempstorage = { nullptr/NULL/0/... }
    } else if (VD->getInitStyle() == VarDecl::InitializationStyle::ListInit) {
      if (auto InitList = dyn_cast<InitListExpr>(VD->getInit())) {
        Init = InitList->getInit(0)->IgnoreImplicitAsWritten();
        if (isNullptrOrZero(Init)) {
          InitLoc = DpctGlobalInfo::getSourceManager().getExpansionLoc(
              VD->getBeginLoc());
        }
      }
    }
  }
  for (auto &Element : TempStorageMatchResult) {
    if (Element == DRE) {
      continue;
    }
    SourceLocation CurLoc = DpctGlobalInfo::getSourceManager().getExpansionLoc(
        Element->getBeginLoc());
    bool IsModified = isModifiedRef(Element);
    bool IsAssignedWithNull = false;
    if (IsModified) {
      if (auto BO = DpctGlobalInfo::findAncestor<BinaryOperator>(Element)) {
        if (BO->getLHS() == Element &&
            isNullptrOrZero(BO->getRHS()->IgnoreImplicitAsWritten())) {
          IsAssignedWithNull = true;
        }
      }
    }
    if (IsAssignedWithNull && (CurLoc < SearchEndLoc) &&
        (InitLoc.isInvalid() || CurLoc > InitLoc)) {
      InitLoc = CurLoc;
      Init = Element;
    }
    if (IsModified && !IsAssignedWithNull) {
      if (CurLoc < SearchEndLoc) {
        LastModifiedLoc = CurLoc;
      } else {
        findLoop(Element, DRELoopList);
      }
    }
  }
  bool IsSafeToRemoveCallExpr = true;
  if (!CELoopList.empty()) {
    int CELoopListSize = CELoopList.size();
    for (int i = 0; i < CELoopListSize; i++) {
      if (DpctGlobalInfo::isAncestor(CELoopList[i], Init)) {
        break;
      } else {
        if (!DRELoopList.empty() &&
            std::find(DRELoopList.begin(), DRELoopList.end(), CELoopList[i]) !=
                DRELoopList.end()) {
          IsSafeToRemoveCallExpr = false;
          break;
        }
      }
    }
  }
  if (!InitLoc.isInvalid() &&
      (LastModifiedLoc.isInvalid() || InitLoc > LastModifiedLoc) &&
      IsSafeToRemoveCallExpr) {
    return true;
  }
  return false;
}

// Analyze temp_storage and temp_storage_size argument to determing
// whether these two argument and related decl or cudaMalloc can be
// removed.
// If the d_temp_storage and temp_storage_bytes only used in
// Reduce/Min/Max/Sum and cudaMalloc, then we can remove related decl
// and cudaMalloc*.
void CubRule::removeRedundantTempVar(const CallExpr *CE) {
  auto FuncArgs = CE->getArgs();
  auto TempStorage = FuncArgs[0]->IgnoreImplicitAsWritten();
  auto TempStorageSize = FuncArgs[1]->IgnoreImplicitAsWritten();
  SourceLocation InitLoc;
  std::vector<const DeclRefExpr *> TempStorageMatchResult;
  std::vector<const DeclRefExpr *> TempStorageSizeMatchResult;
  std::vector<const CallExpr *> TempStorageRelatedMalloc;
  std::vector<const CallExpr *> TempStorageSizeRelatedMalloc;
  bool IsSafeToRemoveTempStorage = true;
  bool IsSafeToRemoveTempStorageSize = true;
  auto TempVarAnalysis = [](const DeclRefExpr *DRE, bool &IsSafeToRemove,
                            std::vector<const CallExpr *> &RelatedMalloc) {
    if (auto CE = dpct::DpctGlobalInfo::findAncestor<CallExpr>(DRE)) {
      if (auto FuncDecl = CE->getDirectCallee()) {
        std::string FuncName = FuncDecl->getNameAsString();
        if (FuncName == "Reduce" || FuncName == "Min" || FuncName == "Max" ||
            FuncName == "Sum") {
          const DeclContext *FuncDeclContext = FuncDecl->getDeclContext();
          if (auto CXXRD = dyn_cast<CXXRecordDecl>(FuncDeclContext)) {
            if (CXXRD->getNameAsString() == "DeviceSegmentedReduce") {
              return;
            }
          }
        } else if (FuncName == "cudaMalloc" || FuncName == "cuMemAlloc_v2") {
          RelatedMalloc.push_back(CE);
          return;
        }
      }
    };
    IsSafeToRemove = false;
    return;
  };
  if (auto DRE = dyn_cast<DeclRefExpr>(TempStorage)) {
    if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      findAllVarRef(DRE, TempStorageMatchResult);
      for (auto &Element : TempStorageMatchResult) {
        if (Element == DRE) {
          continue;
        }
        if (IsSafeToRemoveTempStorage) {
          TempVarAnalysis(Element, IsSafeToRemoveTempStorage,
                          TempStorageRelatedMalloc);
        } else {
          break;
        }
      }
      if (IsSafeToRemoveTempStorage) {
        removeVarDecl(VD);
        for (auto Itr = TempStorageRelatedMalloc.begin();
             Itr != TempStorageRelatedMalloc.end();) {
          bool IsUsed = false;
          if (!isExprUsed(*Itr, IsUsed)) {
            Itr = TempStorageRelatedMalloc.erase(Itr);
            continue;
          }
          auto LocInfo = DpctGlobalInfo::getLocInfo((*Itr)->getBeginLoc());
          auto Info = std::make_shared<PriorityReplInfo>();
          Info->Priority = 1;
          if (IsUsed) {
            Info->Repls.emplace_back(ReplaceStmt(*Itr, "0").getReplacement(
                DpctGlobalInfo::getContext()));
          } else {
            Info->Repls.emplace_back(ReplaceStmt(*Itr, "").getReplacement(
                DpctGlobalInfo::getContext()));
          }
          DpctGlobalInfo::addPriorityReplInfo(
              LocInfo.first + std::to_string(LocInfo.second), Info);
          Itr++;
        }
      } else {
        return;
      }
    }
  }
  if (auto DRE = dyn_cast<DeclRefExpr>(TempStorageSize)) {
    if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      findAllVarRef(DRE, TempStorageSizeMatchResult);
      for (auto &Element : TempStorageSizeMatchResult) {
        if (Element == DRE) {
          continue;
        }
        if (IsSafeToRemoveTempStorageSize) {
          TempVarAnalysis(Element, IsSafeToRemoveTempStorageSize,
                          TempStorageSizeRelatedMalloc);
        } else {
          break;
        }
      }
      for (auto Element : TempStorageSizeRelatedMalloc) {
        if (std::find(TempStorageRelatedMalloc.begin(),
                      TempStorageRelatedMalloc.end(),
                      Element) == TempStorageRelatedMalloc.end()) {
          IsSafeToRemoveTempStorageSize = false;
        }
      }
      if (IsSafeToRemoveTempStorageSize) {
        removeVarDecl(VD);
      }
    }
  }
}

void CubRule::processDeviceLevelFuncCall(const CallExpr *CE,
                                         bool FuncCallUsed) {
  auto DC = CE->getDirectCallee();
  std::string FuncName = DC->getNameAsString();
  if (auto FD = DpctGlobalInfo::getParentFunction(CE)) {
    if (FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) {
      report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
      return;
    }
  }

  // If some parameter is temporary object, we need to skip
  // ExpreWithCleanups Node to determine whether return value is used
  auto &Context = DpctGlobalInfo::getContext();
  if (auto EWC = Context.getParents(*CE)[0].get<ExprWithCleanups>()) {
    bool OldFuncCallUsed = FuncCallUsed;
    if (!isExprUsed(EWC, FuncCallUsed)) {
      FuncCallUsed = OldFuncCallUsed;
    }
  }
  if (isRedundantCallExpr(CE)) {
    if (FuncCallUsed) {
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
    return;
  }
  // generate callexpr replacement
  auto FuncArgs = CE->getArgs();
  std::string Repl, ParamList, OpRepl, InitRepl, QueueRepl, DataType,
      GROUPSIZE_Default = "128";
  ParamAssembler CubParamAs(ParamList);
  ExprAnalysis InputEA(FuncArgs[2]);
  ExprAnalysis OutputEA(FuncArgs[3]);
  ExprAnalysis SegmentNumEA(FuncArgs[4]);
  ExprAnalysis OffsetBegEA(FuncArgs[5]);
  ExprAnalysis OffsetEndEA(FuncArgs[6]);
  if (DC->getParamDecl(2)->getType()->isPointerType()) {
    DataType = DC->getParamDecl(2)
                   ->getType()
                   ->getPointeeType()
                   .getUnqualifiedType()
                   .getCanonicalType()
                   .getAsString();
  } else {
    return;
  }
  if (FuncName == "Reduce") {
    ExprAnalysis InitEA(FuncArgs[8]);
    InitRepl = InitEA.getReplacedString();
    OpRepl = getOpRepl(FuncArgs[7]);
    if (OpRepl.empty()) {
      report(CE->getBeginLoc(), Diagnostics::UNSUPPORTED_BINARY_OPERATION,
             false);
      OpRepl = "dpct_placeholder";
    }
  } else if (FuncName == "Sum") {
    InitRepl = "0";
    OpRepl = MapNames::getClNamespace() + "ext::oneapi::plus<>()";
  } else if (FuncName == "Min") {
    DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(),
                                               HT_STD_Numeric_Limits);
    InitRepl = "std::numeric_limits<" + DataType + ">::max()";
    OpRepl = MapNames::getClNamespace() + "ext::oneapi::minimum<>()";
  } else if (FuncName == "Max") {
    DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(),
                                               HT_STD_Numeric_Limits);
    InitRepl = "std::numeric_limits<" + DataType + ">::lowest()";
    OpRepl = MapNames::getClNamespace() + "ext::oneapi::maximum<>()";
  }
  if ((FuncName == "Reduce" && FuncArgs[9]->isDefaultArgument()) ||
      (FuncName != "Reduce" && FuncArgs[7]->isDefaultArgument())) {
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
    QueueRepl = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
  } else {
    ExprAnalysis StreamEA(FuncArgs[FuncName == "Reduce" ? 9 : 7]);
    QueueRepl = "*(" + StreamEA.getReplacedString() + ")";
  }

  CubParamAs << QueueRepl << InputEA.getReplacedString()
             << OutputEA.getReplacedString() << SegmentNumEA.getReplacedString()
             << ("(unsigned int *)(" + OffsetBegEA.getReplacedString() + ")")
             << ("(unsigned int *)(" + OffsetEndEA.getReplacedString() + ")")
             << OpRepl << InitRepl;
  if (FuncCallUsed) {
    Repl = "(" + MapNames::getDpctNamespace() + "device::segmented_reduce<" +
           GROUPSIZE_Default + ">(" + ParamList + "), 0)";
  } else {
    Repl = MapNames::getDpctNamespace() + "device::segmented_reduce<" +
           GROUPSIZE_Default + ">(" + ParamList + ")";
  }
  report(CE->getBeginLoc(), Diagnostics::REDUCE_PERFORMANCE_TUNE, false);
  emplaceTransformation(new ReplaceStmt(CE, Repl));
  removeRedundantTempVar(CE);
  requestFeature(HelperFeatureEnum::DplExtrasDpcppExtensions_segmented_reduce,
                 DpctGlobalInfo::getLocInfo(CE->getBeginLoc()).first);
  DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), HT_DPL_Utils);
}

void CubRule::processThreadLevelFuncCall(const CallExpr *CE,
                                         bool FuncCallUsed) {
  std::string Repl;
  auto DC = CE->getDirectCallee();
  std::string FuncName = DC->getNameAsString();
  if (FuncName == "ThreadLoad") {
    auto FuncArgs = CE->getArgs();
    const Expr *InData = FuncArgs[0];
    ExprAnalysis InEA(InData);
    Repl = "*(" + InEA.getReplacedString() + ")";
    emplaceTransformation(new ReplaceStmt(CE, Repl));
  } else if (FuncName == "ThreadStore") {
    auto FuncArgs = CE->getArgs();
    const Expr *OutputIterator = FuncArgs[0];
    const Expr *Value = FuncArgs[1];
    ExprAnalysis ItrEA(OutputIterator);
    ExprAnalysis ValueEA(Value);
    Repl =
        "*(" + ItrEA.getReplacedString() + ") = " + ValueEA.getReplacedString();
    emplaceTransformation(new ReplaceStmt(CE, Repl));
  }
}

void CubRule::processWarpLevelFuncCall(const CallExpr *CE, bool FuncCallUsed) {
  std::string Repl;
  size_t WarpSize = 32;
  auto DC = CE->getDirectCallee();
  std::string FuncName = DC->getNameAsString();
  if (FuncName == "ShuffleIndex") {
    auto TA = DC->getTemplateSpecializationArgs();
    if (!TA)
      return;
    WarpSize = TA->get(0).getAsIntegral().getExtValue();
    std::string ValueType =
        TA->get(1).getAsType().getUnqualifiedType().getAsString();
    auto MemberMask = CE->getArg(2);
    auto Mask = dyn_cast<IntegerLiteral>(MemberMask);
    if (Mask && Mask->getValue().getZExtValue() == 0xffffffff) {
      const Expr *Value = CE->getArg(0);
      const Expr *Lane = CE->getArg(1);
      ExprAnalysis ValueEA(Value);
      ExprAnalysis LaneEA(Lane);
      auto DeviceFuncDecl = getImmediateOuterFuncDecl(CE);
      Repl = DpctGlobalInfo::getSubGroup(CE, DeviceFuncDecl) + ".shuffle(" +
             ValueEA.getReplacedString() + ", " + LaneEA.getReplacedString() +
             ")";
      emplaceTransformation(new ReplaceStmt(CE, Repl));
      if (DeviceFuncDecl) {
        auto DI = DeviceFunctionDecl::LinkRedecls(DeviceFuncDecl);
        if (DI) {
          DI->addSubGroupSizeRequest(WarpSize, CE->getBeginLoc(), "shuffle");
        }
      }
    } else {
      report(CE->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
    }
  }
}

void CubRule::processCubFuncCall(const CallExpr *CE, bool FuncCallUsed) {
  std::string Repl;
  if (auto DC = CE->getDirectCallee()) {
    const DeclContext *MaybeFirstNS = DC->getDeclContext();
    // Namespace::Object.Function()
    if (auto CXXRD = dyn_cast<CXXRecordDecl>(MaybeFirstNS)) {
      if (CXXRD->getNameAsString() != "DeviceSegmentedReduce") {
        return;
      }
      MaybeFirstNS = CXXRD->getDeclContext();
    }
    if (auto ND = dyn_cast<NamespaceDecl>(MaybeFirstNS)) {
      if (ND->getNameAsString() != "cub") {
        return;
      }
    } else {
      return;
    }
    std::string FuncName = DC->getNameAsString();
    if (FuncName == "ShuffleIndex") {
      processWarpLevelFuncCall(CE, FuncCallUsed);
    } else if (FuncName == "ThreadLoad" || FuncName == "ThreadStore") {
      processThreadLevelFuncCall(CE, FuncCallUsed);
    } else if (FuncName == "Reduce" || FuncName == "Min" || FuncName == "Max" ||
               FuncName == "Sum") {
      processDeviceLevelFuncCall(CE, FuncCallUsed);
    }
  }
}

void CubRule::processBlockLevelMemberCall(const CXXMemberCallExpr *BlockMC) {
  if (!BlockMC || !BlockMC->getMethodDecl()) {
    return;
  }
  std::string Repl, NewFuncName, ParamList, InitRepl, OpRepl, Indent,
      GroupOrWorkitem, AggregateOrCallback;
  ParamAssembler CubParamAs(ParamList);
  std::string FuncName = BlockMC->getMethodDecl()->getNameAsString();
  std::string ValueType;
  int NumArgs = BlockMC->getNumArgs();
  auto FuncArgs = BlockMC->getArgs();
  auto MD = BlockMC->getMethodDecl()->getParent();
  if (auto CTS = dyn_cast<ClassTemplateSpecializationDecl>(MD)) {
    auto &TA = CTS->getTemplateArgs();
    ValueType = TA[0].getAsType().getUnqualifiedType().getAsString();
  }
  Indent = getIndent(BlockMC->getBeginLoc(), DpctGlobalInfo::getSourceManager())
               .str();
  if (BlockMC->getObjectType()->getTypeClass() ==
      clang::Type::TypeClass::SubstTemplateTypeParm) {
    auto DRE =
        dyn_cast_or_null<DeclRefExpr>(BlockMC->getImplicitObjectArgument());
    if (DRE) {
      GroupOrWorkitem = DRE->getNameInfo().getAsString();
    }
  }
  if (GroupOrWorkitem.empty()) {
    GroupOrWorkitem = DpctGlobalInfo::getGroup(BlockMC);
  }
  if (FuncName == "InclusiveSum" || FuncName == "ExclusiveSum" ||
      FuncName == "InclusiveScan" || FuncName == "ExclusiveScan") {
    const Expr *InData = FuncArgs[0];
    const Expr *OutData = FuncArgs[1];
    ExprAnalysis InEA(InData);
    ExprAnalysis OutEA(OutData);
    bool IsReferenceOutput = false;
    if (FuncName == "ExclusiveScan") {
      if (NumArgs == 4) {
        if (BlockMC->getMethodDecl()
                ->getParamDecl(0)
                ->getType()
                ->isLValueReferenceType()) {
          if (BlockMC->getMethodDecl()->getPrimaryTemplate() &&
              BlockMC->getMethodDecl()
                      ->getPrimaryTemplate()
                      ->getTemplateParameters()
                      ->size() == 2) {
            ExprAnalysis InitEA(FuncArgs[2]);
            InitRepl = InitEA.getReplacedString();
            GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
            OpRepl = getOpRepl(FuncArgs[3]);
            NewFuncName =
                MapNames::getDpctNamespace() + "group::exclusive_scan";
            requestFeature(
                HelperFeatureEnum::DplExtrasDpcppExtensions_exclusive_scan,
                BlockMC);
            DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                       HT_DPL_Utils);
            IsReferenceOutput = true;
          } else {
            report(BlockMC->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
            return;
          }
        } else {
          if (BlockMC->getMethodDecl()
                  ->getParamDecl(0)
                  ->getType()
                  .getAsString() == BlockMC->getMethodDecl()
                                        ->getParamDecl(2)
                                        ->getType()
                                        .getAsString()) {
            ExprAnalysis InitEA(FuncArgs[2]);
            InitRepl = InitEA.getReplacedString();
            OpRepl = getOpRepl(FuncArgs[3]);
            NewFuncName =
                MapNames::getClNamespace() + "exclusive_scan_over_group";
          } else {
            ExprAnalysis AggregateOrCallbackEA(FuncArgs[3]);
            GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
            AggregateOrCallback = AggregateOrCallbackEA.getReplacedString();
            OpRepl = getOpRepl(FuncArgs[2]);
            NewFuncName =
                MapNames::getDpctNamespace() + "group::exclusive_scan";
            requestFeature(
                HelperFeatureEnum::DplExtrasDpcppExtensions_exclusive_scan,
                BlockMC);
            DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                       HT_DPL_Utils);
          }
        }
      } else if (NumArgs == 5) {
        if (!BlockMC->getMethodDecl()
                 ->getParamDecl(0)
                 ->getType()
                 ->isLValueReferenceType()) {
          GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
          ExprAnalysis InitEA(FuncArgs[2]);
          ExprAnalysis AggregateOrCallbackEA(FuncArgs[4]);
          InitRepl = InitEA.getReplacedString();
          OpRepl = getOpRepl(FuncArgs[3]);
          AggregateOrCallback = AggregateOrCallbackEA.getReplacedString();
          NewFuncName = MapNames::getDpctNamespace() + "group::exclusive_scan";
          requestFeature(
              HelperFeatureEnum::DplExtrasDpcppExtensions_exclusive_scan,
              BlockMC);
          DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                     HT_DPL_Utils);
        } else {
          report(BlockMC->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
          return;
        }
      }
    } else if (FuncName == "InclusiveScan") {
      if (NumArgs == 3) {
        if (BlockMC->getMethodDecl()
                ->getParamDecl(0)
                ->getType()
                ->isLValueReferenceType()) {
          GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
          OpRepl = getOpRepl(FuncArgs[2]);
          NewFuncName = MapNames::getDpctNamespace() + "group::inclusive_scan";
          requestFeature(
              HelperFeatureEnum::DplExtrasDpcppExtensions_inclusive_scan,
              BlockMC);
          DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                     HT_DPL_Utils);
          IsReferenceOutput = true;
        } else {
          OpRepl = getOpRepl(FuncArgs[2]);
          NewFuncName =
              MapNames::getClNamespace() + "inclusive_scan_over_group";
        }
      } else if (NumArgs == 4) {
        if (BlockMC->getMethodDecl()
                ->getParamDecl(0)
                ->getType()
                ->isLValueReferenceType()) {
          report(BlockMC->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
          return;
        }
        GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
        OpRepl = getOpRepl(FuncArgs[2]);
        ExprAnalysis AggregateOrCallbackEA(FuncArgs[3]);
        AggregateOrCallback = AggregateOrCallbackEA.getReplacedString();
        NewFuncName = MapNames::getDpctNamespace() + "group::inclusive_scan";
        requestFeature(
            HelperFeatureEnum::DplExtrasDpcppExtensions_exclusive_scan,
            BlockMC);
        DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                   HT_DPL_Utils);
      }
    } else if (FuncName == "ExclusiveSum") {
      if (NumArgs == 2) {
        OpRepl = getOpRepl(nullptr);
        InitRepl = "0";
        if (BlockMC->getMethodDecl()
                ->getParamDecl(0)
                ->getType()
                ->isLValueReferenceType()) {
          NewFuncName = MapNames::getDpctNamespace() + "group::exclusive_scan";
          requestFeature(
              HelperFeatureEnum::DplExtrasDpcppExtensions_exclusive_scan,
              BlockMC);
          DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                     HT_DPL_Utils);
          GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
          IsReferenceOutput = true;
        } else {
          NewFuncName =
              MapNames::getClNamespace() + "exclusive_scan_over_group";
        }
      } else if (NumArgs == 3) {
        if (!BlockMC->getMethodDecl()
                 ->getParamDecl(0)
                 ->getType()
                 ->isLValueReferenceType()) {
          GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
          OpRepl = getOpRepl(nullptr);
          if (BlockMC->getMethodDecl()
                  ->getParamDecl(1)
                  ->getType()
                  .getAsString() == BlockMC->getMethodDecl()
                                        ->getParamDecl(2)
                                        ->getType()
                                        .getAsString()) {
            InitRepl = "0";
          }
          ExprAnalysis AggregateOrCallbackEA(FuncArgs[2]);
          AggregateOrCallback = AggregateOrCallbackEA.getReplacedString();
          NewFuncName = MapNames::getDpctNamespace() + "group::exclusive_scan";
          requestFeature(
              HelperFeatureEnum::DplExtrasDpcppExtensions_exclusive_scan,
              BlockMC);
          DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                     HT_DPL_Utils);
        } else {
          report(BlockMC->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
          return;
        }
      }
    } else if (FuncName == "InclusiveSum") {
      if (NumArgs == 2) {
        if (BlockMC->getMethodDecl()
                ->getParamDecl(0)
                ->getType()
                ->isLValueReferenceType()) {
          GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
          OpRepl = getOpRepl(nullptr);
          NewFuncName = MapNames::getDpctNamespace() + "group::inclusive_scan";
          requestFeature(
              HelperFeatureEnum::DplExtrasDpcppExtensions_inclusive_scan,
              BlockMC);
          DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                     HT_DPL_Utils);
          IsReferenceOutput = true;
        } else {
          OpRepl = getOpRepl(nullptr);
          NewFuncName =
              MapNames::getClNamespace() + "inclusive_scan_over_group";
        }
      } else if (NumArgs == 3) {
        if (BlockMC->getMethodDecl()
                ->getParamDecl(0)
                ->getType()
                ->isLValueReferenceType()) {
          report(BlockMC->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
          return;
        }
        GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
        OpRepl = getOpRepl(nullptr);
        ExprAnalysis AggregateOrCallbackEA(FuncArgs[2]);
        AggregateOrCallback = AggregateOrCallbackEA.getReplacedString();
        NewFuncName = MapNames::getDpctNamespace() + "group::inclusive_scan";
        requestFeature(
            HelperFeatureEnum::DplExtrasDpcppExtensions_inclusive_scan,
            BlockMC);
        DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                   HT_DPL_Utils);
      }
    } else {
      report(BlockMC->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
      return;
    }
    if (IsReferenceOutput) {
      CubParamAs << GroupOrWorkitem << InEA.getReplacedString()
                 << OutEA.getReplacedString() << InitRepl << OpRepl
                 << AggregateOrCallback;
      Repl = NewFuncName + "(" + ParamList + ")";
    } else {
      CubParamAs << GroupOrWorkitem << InEA.getReplacedString() << InitRepl
                 << OpRepl << AggregateOrCallback;
      Repl = OutEA.getReplacedString() + " = " + NewFuncName + "(" + ParamList +
             ")";
    }
    emplaceTransformation(new ReplaceStmt(BlockMC, Repl));
  } else if (FuncName == "Sum" || FuncName == "Reduce") {
    if (BlockMC->getMethodDecl()
            ->getParamDecl(0)
            ->getType()
            ->isLValueReferenceType()) {
      GroupOrWorkitem = DpctGlobalInfo::getItem(BlockMC);
      NewFuncName = MapNames::getDpctNamespace() + "group::reduce";
      requestFeature(HelperFeatureEnum::DplExtrasDpcppExtensions_reduce,
                     BlockMC);
      DpctGlobalInfo::getInstance().insertHeader(BlockMC->getBeginLoc(),
                                                 HT_DPL_Utils);
    } else {
      NewFuncName = MapNames::getClNamespace() + "reduce_over_group";
    }
    const Expr *InData = FuncArgs[0];
    ExprAnalysis InEA(InData);
    if (FuncName == "Reduce" && NumArgs == 2) {
      OpRepl = getOpRepl(FuncArgs[1]);
    } else if (FuncName == "Sum" && NumArgs == 1) {
      OpRepl = getOpRepl(nullptr);
    } else {
      report(BlockMC->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
      return;
    }
    CubParamAs << GroupOrWorkitem << InEA.getReplacedString() << OpRepl;
    Repl = NewFuncName + "(" + ParamList + ")";
    emplaceTransformation(new ReplaceStmt(BlockMC, Repl));
  }
}

void CubRule::processWarpLevelMemberCall(const CXXMemberCallExpr *WarpMC) {
  if (!WarpMC || !WarpMC->getMethodDecl()) {
    return;
  }
  size_t WarpSize = 32;
  std::string Repl, NewFuncName, ParamList, InitRepl, OpRepl, Indent,
      GroupOrWorkitem, AggregateOrCallback;
  ParamAssembler CubParamAs(ParamList);
  std::string FuncName = WarpMC->getMethodDecl()->getNameAsString();
  std::string ValueType;
  int NumArgs = WarpMC->getNumArgs();
  auto MD = WarpMC->getMethodDecl()->getParent();
  if (auto CTS = dyn_cast<ClassTemplateSpecializationDecl>(MD)) {
    auto &TA = CTS->getTemplateArgs();
    ValueType = TA[0].getAsType().getUnqualifiedType().getAsString();
    WarpSize = TA[1].getAsIntegral().getExtValue();
  }
  Indent = getIndent(WarpMC->getBeginLoc(), DpctGlobalInfo::getSourceManager())
               .str();
  auto FD = DpctGlobalInfo::getParentFunction(WarpMC);
  if (WarpMC->getObjectType()->getTypeClass() ==
      clang::Type::TypeClass::SubstTemplateTypeParm) {
    auto DRE =
        dyn_cast_or_null<DeclRefExpr>(WarpMC->getImplicitObjectArgument());
    if (DRE) {
      GroupOrWorkitem = DRE->getNameInfo().getAsString();
    }
  }
  if (GroupOrWorkitem.empty()) {
    GroupOrWorkitem = DpctGlobalInfo::getSubGroup(WarpMC, FD);
  }
  if (FuncName == "InclusiveSum" || FuncName == "ExclusiveSum" ||
      FuncName == "InclusiveScan" || FuncName == "ExclusiveScan") {
    auto FuncArgs = WarpMC->getArgs();
    const Expr *InData = FuncArgs[0];
    const Expr *OutData = FuncArgs[1];
    if (FuncName == "ExclusiveScan") {
      if (NumArgs == 3) {
        OpRepl = getOpRepl(FuncArgs[2]);
      } else if (NumArgs == 4 &&
                 WarpMC->getMethodDecl()
                         ->getParamDecl(0)
                         ->getType()
                         .getCanonicalType()
                         .getAsString() == WarpMC->getMethodDecl()
                                               ->getParamDecl(2)
                                               ->getType()
                                               .getCanonicalType()
                                               .getAsString()) {
        ExprAnalysis InitEA(FuncArgs[2]);
        InitRepl = ", " + InitEA.getReplacedString();
        OpRepl = getOpRepl(FuncArgs[3]);
      } else {
        report(WarpMC->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
        return;
      }
      NewFuncName = "exclusive_scan_over_group";
    } else if (FuncName == "InclusiveScan" && NumArgs == 3) {
      OpRepl = getOpRepl(FuncArgs[2]);
      NewFuncName = "inclusive_scan_over_group";
    } else if (FuncName == "ExclusiveSum" && NumArgs == 2) {
      OpRepl = getOpRepl(nullptr);
      NewFuncName = "exclusive_scan_over_group";
    } else if (FuncName == "InclusiveSum" && NumArgs == 2) {
      OpRepl = getOpRepl(nullptr);
      NewFuncName = "inclusive_scan_over_group";
    } else {
      report(WarpMC->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
      return;
    }
    ExprAnalysis InEA(InData);
    ExprAnalysis OutEA(OutData);
    Repl = OutEA.getReplacedString() + " = " + MapNames::getClNamespace() +
           NewFuncName + "(" + GroupOrWorkitem + ", " +
           InEA.getReplacedString() + InitRepl + ", " + OpRepl + ")";
    emplaceTransformation(new ReplaceStmt(WarpMC, Repl));
  } else if (FuncName == "Broadcast") {
    auto FuncArgs = WarpMC->getArgs();
    const Expr *InData = FuncArgs[0];
    const Expr *SrcLane = FuncArgs[1];
    ExprAnalysis InEA(InData);
    ExprAnalysis SrcLaneEA(SrcLane);
    Repl = MapNames::getClNamespace() + "group_broadcast(" +
           DpctGlobalInfo::getSubGroup(WarpMC) + ", " +
           InEA.getReplacedString() + ", " + SrcLaneEA.getReplacedString() +
           ")";
    NewFuncName = "group_broadcast";
    emplaceTransformation(new ReplaceStmt(WarpMC, Repl));
  } else if (FuncName == "Reduce" || FuncName == "Sum") {
    auto FuncArgs = WarpMC->getArgs();
    const Expr *InData = FuncArgs[0];
    ExprAnalysis InEA(InData);
    if (FuncName == "Reduce" && NumArgs == 2) {
      OpRepl = getOpRepl(FuncArgs[1]);
    } else if (FuncName == "Sum" && NumArgs == 1) {
      OpRepl = getOpRepl(nullptr);
    } else {
      report(WarpMC->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
      return;
    }
    NewFuncName = "reduce_over_group";
    Repl = MapNames::getClNamespace() + "reduce_over_group(" + GroupOrWorkitem +
           ", " + InEA.getReplacedString() + ", " + OpRepl + ")";
    emplaceTransformation(new ReplaceStmt(WarpMC, Repl));
  }
  if (auto FuncInfo = DeviceFunctionDecl::LinkRedecls(FD)) {
    FuncInfo->addSubGroupSizeRequest(WarpSize, WarpMC->getBeginLoc(),
                                     NewFuncName);
  }
}

void CubRule::processCubMemberCall(const CXXMemberCallExpr *MC) {
  auto ObjType = MC->getObjectType().getCanonicalType();
  std::string ObjTypeStr = ObjType.getAsString();

  if (isTypeInRoot(ObjType.getTypePtr())) {
    return;
  } else if (ObjTypeStr.find("class cub::WarpScan") == 0 ||
             ObjTypeStr.find("class cub::WarpReduce") == 0) {
    processWarpLevelMemberCall(MC);
  } else if (ObjTypeStr.find("class cub::BlockScan") == 0 ||
             ObjTypeStr.find("class cub::BlockReduce") == 0) {
    processBlockLevelMemberCall(MC);
  } else {
    report(MC->getBeginLoc(), Diagnostics::NOTSUPPORTED, false);
    return;
  }
}

void CubRule::processTypeLoc(const TypeLoc *TL) {
  auto TD = DpctGlobalInfo::findAncestor<TypedefDecl>(TL);
  if (TD || isTypeInRoot(TL->getType().getCanonicalType().getTypePtr()))
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto Range = getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc());
  auto BeginLoc = Range.getBegin();
  auto EndLoc = Range.getEnd();
  std::string TypeName = TL->getType().getCanonicalType().getAsString();
  if (TypeName.find("class cub::WarpScan") == 0 ||
      TypeName.find("class cub::WarpReduce") == 0) {
    emplaceTransformation(
        replaceText(BeginLoc, EndLoc.getLocWithOffset(1),
                    MapNames::getClNamespace() + "ext::oneapi::sub_group", SM));
  } else if (TypeName.find("class cub::BlockScan") == 0 ||
             TypeName.find("class cub::BlockReduce") == 0) {
    auto DeviceFuncDecl = DpctGlobalInfo::findAncestor<FunctionDecl>(TL);
    if (DeviceFuncDecl && (DeviceFuncDecl->hasAttr<CUDADeviceAttr>() ||
                           DeviceFuncDecl->hasAttr<CUDAGlobalAttr>())) {
      if (auto DI = DeviceFunctionDecl::LinkRedecls(DeviceFuncDecl)) {
        auto &Map = DpctGlobalInfo::getInstance().getCubPlaceholderIndexMap();
        Map.insert({PlaceholderIndex, DI});
        emplaceTransformation(
            replaceText(BeginLoc, EndLoc.getLocWithOffset(1),
                        MapNames::getClNamespace() + "group<{{NEEDREPLACEC" +
                            std::to_string(PlaceholderIndex++) + "}}>",
                        SM));
      }
    }
  }
}

int CubRule::PlaceholderIndex = 1;

void CubRule::runRule(const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CXXMemberCallExpr *MC =
          getNodeAsType<CXXMemberCallExpr>(Result, "MemberCall")) {
    processCubMemberCall(MC);
  } else if (const DeclStmt *DS = getNodeAsType<DeclStmt>(Result, "DeclStmt")) {
    processCubDeclStmt(DS);
  } else if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FuncCall")) {
    processCubFuncCall(CE);
  } else if (const CallExpr *CE =
                 getNodeAsType<CallExpr>(Result, "FuncCallUsed")) {
    processCubFuncCall(CE, true);
  } else if (const TypedefDecl *TD =
                 getNodeAsType<TypedefDecl>(Result, "TypeDefDecl")) {
    processCubTypeDef(TD);
  } else if (auto TL = getNodeAsType<TypeLoc>(Result, "cudaTypeDef")) {
    processTypeLoc(TL);
  }
}
REGISTER_RULE(CubRule)

void ASTTraversalManager::matchAST(ASTContext &Context, TransformSetTy &TS,
                                   StmtStringMap &SSM) {
  this->Context = &Context;
  for (auto &I : Storage) {
    I->registerMatcher(Matchers);
    if (auto TR = dyn_cast<MigrationRule>(&*I)) {
      TR->TM = this;
      TR->setTransformSet(TS);
    }
  }

  StaticsInfo::printMigrationRules(Storage);

  Matchers.matchAST(Context);

  StaticsInfo::printMatchedRules(Storage);
  CHECKPOINT_ASTMATCHER_RUN_EXIT();
}

void ASTTraversalManager::emplaceAllRules(int SourceFileFlag) {
  std::vector<std::vector<std::string>> Rules;

  for (auto &F : ASTTraversalMetaInfo::getConstructorTable()) {

    auto RuleObj = (MigrationRule *)F.second();
    CommonRuleProperty RuleProperty = RuleObj->GetRuleProperty();

    auto RType = RuleProperty.RType;
    auto RulesDependon = RuleProperty.RulesDependon;

    if (RType & SourceFileFlag) {
      std::string CurrentRuleName = ASTTraversalMetaInfo::getName(F.first);
      if (DpctGlobalInfo::getRunRound() == 1 &&
          CurrentRuleName == "CudaArchMacroRule")
        continue;
      std::vector<std::string> Vec;
      Vec.push_back(CurrentRuleName);
      for (auto const &RuleName : RulesDependon) {
        Vec.push_back(RuleName);
      }
      Rules.push_back(Vec);
    }
  }

  std::vector<std::string> SortedRules = ruleTopoSort(Rules);

  for (std::vector<std::string>::reverse_iterator it = SortedRules.rbegin();
       it != SortedRules.rend(); it++) {
    auto *ID = ASTTraversalMetaInfo::getID(*it);
    if (!ID) {
      llvm::errs() << "[ERROR] Rule\"" << *it << "\" not found\n";
      dpctExit(MigrationError);
    }
    emplaceMigrationRule(ID);
  }
}

const CompilerInstance &MigrationRule::getCompilerInstance() { return TM->CI; }
