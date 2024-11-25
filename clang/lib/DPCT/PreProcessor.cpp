//===--------------- PreProcessor.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PreProcessor.h"
#include "AnalysisInfo.h"
#include "Diagnostics/Diagnostics.h"
#include "FileGenerator/GenFiles.h"
#include "RulesLang/MapNamesLang.h"
#include "RulesLangLib/MapNamesLangLib.h"
#include "TextModification.h"
#include "Utility.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Basic/Cuda.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/DPCT/DpctOptions.h"
#include "clang/Lex/MacroArgs.h"
#include <string>
#include <tuple>
#include <utility>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::dpct;
using namespace clang::tooling;

extern DpctOption<opt, bool> ProcessAll;
namespace clang {
namespace dpct {



std::shared_ptr<clang::dpct::ReplaceToken>
generateReplacement(SourceLocation SL, MacroMigrationRule &Rule) {
  requestFeature(Rule.HelperFeature);
  for (auto ItHeader = Rule.Includes.begin(); ItHeader != Rule.Includes.end();
       ItHeader++) {
    DpctGlobalInfo::getInstance().insertHeader(SL, *ItHeader);
  }
  return std::make_shared<ReplaceToken>(SL, std::move(Rule.Out));
}

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
bool IncludesCallbacks::ReplaceCuMacro(const Token &MacroNameTok,
                                       MacroInfo *MI) {
  bool IsInAnalysisScope = isInAnalysisScope(MacroNameTok.getLocation());
  if (!IsInAnalysisScope) {
    return false;
  }
  if (!MacroNameTok.getIdentifierInfo()) {
    return false;
  }
  std::string MacroName = MacroNameTok.getIdentifierInfo()->getName().str();
  auto Iter = MapNames::MacroRuleMap.find(MacroName);
  if (Iter != MapNames::MacroRuleMap.end()) {
    auto Repl = generateReplacement(MacroNameTok.getLocation(), Iter->second);
    if (MacroName == "__CUDA_ARCH__") {
      if (DpctGlobalInfo::getInstance().getContext().getLangOpts().CUDA) {
        insertCudaArchRepl(Repl->getReplacement(DpctGlobalInfo::getContext()));
        return true;
      }
      return false;
    }
    if (MacroName == "__CUDACC__") {
      if (MacroNameTok.getIdentifierInfo()->hasMacroDefinition()) {
        Repl->setSYCLHeaderNeeded(false);
      } else {
        return false;
      }
    } else if (MacroName == "CUDART_VERSION" ||
               MacroName == "__CUDART_API_VERSION" ||
               MacroName == "CUDA_VERSION") {
      // These two macros are defined by CUDA header file
      auto LocInfo = DpctGlobalInfo::getLocInfo(MacroNameTok.getLocation());
      auto Ver = clang::getCudaVersionPair(DpctGlobalInfo::getSDKVersion());
      DpctGlobalInfo::getInstance()
          .insertFile(LocInfo.first)
          ->setRTVersionValue(
              std::to_string(Ver.first * 1000 + Ver.second * 10));
    } else if (MacroName == "NCCL_VERSION_CODE" && MI) {
      auto LocInfo = DpctGlobalInfo::getLocInfo(MacroNameTok.getLocation());
      DpctGlobalInfo::getInstance()
          .insertFile(LocInfo.first)
          ->setCCLVerValue(Lexer::getSourceText(
                               CharSourceRange::getCharRange(
                                   MI->getReplacementToken(0).getLocation(),
                                   Lexer::getLocForEndOfToken(
                                       MI->getReplacementToken(0).getLocation(),
                                       0, SM, LangOptions())),
                               SM, LangOptions())
                               .str());
    }
    if (DpctGlobalInfo::getContext().getLangOpts().CUDA) {
      // These two macros are defined by CUDA compiler
      if (MacroName == "__CUDACC_VER_MAJOR__") {
        auto LocInfo = DpctGlobalInfo::getLocInfo(MacroNameTok.getLocation());
        auto Ver = clang::getCudaVersionPair(DpctGlobalInfo::getSDKVersion());
        DpctGlobalInfo::getInstance()
            .insertFile(LocInfo.first)
            ->setMajorVersionValue(std::to_string(Ver.first));
      }
      if (MacroName == "__CUDACC_VER_MINOR__") {
        auto LocInfo = DpctGlobalInfo::getLocInfo(MacroNameTok.getLocation());
        auto Ver = clang::getCudaVersionPair(DpctGlobalInfo::getSDKVersion());
        DpctGlobalInfo::getInstance()
            .insertFile(LocInfo.first)
            ->setMinorVersionValue(std::to_string(Ver.second));
      }
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

    // The "__noinline__" macro is re-defined and it is used in
    // "__attribute__()", do not migrate it.
    if ((GetSourceFileType(
             DpctGlobalInfo::getInstance().getMainFile()->getFilePath()) ==
         SPT_CppSource) &&
        (II->getName() == "__noinline__")) {
      continue;
    }

    auto ItRule = MapNames::MacroRuleMap.find(II->getName().str());
    if (ItRule != MapNames::MacroRuleMap.end()) {
      TransformSet.emplace_back(
          generateReplacement(Iter->getLocation(), ItRule->second));
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

    if (MapNamesLang::AtomicFuncNamesMap.find(II->getName().str()) !=
        MapNamesLang::AtomicFuncNamesMap.end()) {
      std::string HashStr =
          getHashStrFromLoc(MI->getReplacementToken(0).getLocation());
      DpctGlobalInfo::getInstance().insertAtomicInfo(
          HashStr, MacroNameTok.getLocation(), II->getName().str());
    } else if (MacroNameTok.getLocation().isValid() &&
               MacroNameTok.getIdentifierInfo() &&
               MapNamesLang::VectorTypeMigratedTypeSizeMap.find(
                   MacroNameTok.getIdentifierInfo()->getName().str()) !=
                   MapNamesLang::VectorTypeMigratedTypeSizeMap.end()) {
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

    dpct::DpctGlobalInfo::getExpansionRangeBeginMap()[getCombinedStrFromLoc(
        DefRange.getBegin())] =
        std::make_pair(DpctGlobalInfo::getLocInfo(
                           MI->getReplacementToken(0).getLocation()),
                       DpctGlobalInfo::getLocInfo(MI->getDefinitionEndLoc()));
    if (dpct::DpctGlobalInfo::getMacroDefines().find(HashKey) ==
        dpct::DpctGlobalInfo::getMacroDefines().end()) {
      // Record all processed macro definition
      dpct::DpctGlobalInfo::getMacroDefines()[HashKey] = true;
      size_t i;
      // Record all tokens in the macro definition
      for (i = 0; i < MI->getNumTokens(); i++) {
        std::shared_ptr<dpct::DpctGlobalInfo::MacroExpansionRecord> R =
            std::make_shared<dpct::DpctGlobalInfo::MacroExpansionRecord>(
                MacroNameTok.getIdentifierInfo(), MI, Range, IsInAnalysisScope,
                i);
        dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord()
            [getCombinedStrFromLoc(MI->getReplacementToken(i).getLocation())] =
                R;
      }
      if (Args && IsInAnalysisScope) {
        for (unsigned int i = 0; i < Args->getNumMacroArguments(); ++i) {
          std::shared_ptr<dpct::DpctGlobalInfo::MacroArgRecord> R =
              std::make_shared<dpct::DpctGlobalInfo::MacroArgRecord>(MI, i);
          auto str =
              getCombinedStrFromLoc(Args->getUnexpArgument(i)->getLocation());
          auto &Global = DpctGlobalInfo::getInstance();
          dpct::DpctGlobalInfo::getMacroArgRecordMap()
              [Global.getMainFile()->getFilePath().getPath().str() +
               getCombinedStrFromLoc(
                   Args->getUnexpArgument(i)->getLocation())] = R;
        }
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
  // line for finding this replacement in MemVarAnalysisRule. Since the variable
  // name is difficult to get here, the warning is also emitted in
  // MemVarAnalysisRule.
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

  if (ReplaceCuMacro(MacroNameTok, MI)) {
    return;
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
        auto &Map = DpctGlobalInfo::getConstantReplProcessedFlagMap();
        Map[TM] = false;
      }
    } else {
      TransformSet.emplace_back(TM);
    }
  }

  if (Name == "NCCL_VERSION") {
    std::vector<std::string> MA(3);
    auto calCclCompatVersion = [=](int arg1, int arg2, int arg3) {
      return std::to_string(((arg1) <= 2 && (arg2) <= 8)
                                ? (arg1)*1000 + (arg2)*100 + (arg3)
                                : (arg1)*10000 + (arg2)*100 + (arg3));
    };
    for (unsigned int i = 0; i < Args->getNumMacroArguments(); ++i) {
      MA[i] =
          Lexer::getSourceText(CharSourceRange::getCharRange(
                                   Args->getUnexpArgument(i)->getLocation(),
                                   Lexer::getLocForEndOfToken(
                                       Args->getUnexpArgument(i)->getLocation(),
                                       0, SM, LangOptions())),
                               SM, LangOptions())
              .str();
    }
    auto Length = Lexer::MeasureTokenLength(
        Range.getEnd(), SM, dpct::DpctGlobalInfo::getContext().getLangOpts());
    Length += SM.getDecomposedLoc(Range.getEnd()).second -
              SM.getDecomposedLoc(Range.getBegin()).second;
    TransformSet.emplace_back(new ReplaceText(
        Range.getBegin(), Length,
        std::move(calCclCompatVersion(std::stoi(MA[0]), std::stoi(MA[1]),
                                      std::stoi(MA[2])))));
  }

  if (Name == "NCCL_MAJOR" || Name == "NCCL_MINOR") {
    TransformSet.emplace_back(new ReplaceToken(
        Range.getBegin(),
        std::move(Lexer::getSourceText(
                      CharSourceRange::getCharRange(
                          MI->getReplacementToken(0).getLocation(),
                          Lexer::getLocForEndOfToken(
                              MI->getReplacementToken(0).getLocation(), 0, SM,
                              LangOptions())),
                      SM, LangOptions())
                      .str())));
  }

  if (TKind == tok::identifier && Name == "CUDART_CB") {
#ifdef _WIN32
    TransformSet.emplace_back(
        new ReplaceText(Range.getBegin(), 9, "__stdcall"));
#else
    TransformSet.emplace_back(removeMacroInvocationAndTrailingSpaces(Range));
#endif
  }

  auto Iter = MapNamesLang::HostAllocSet.find(Name.str());
  if (TKind == tok::identifier && Iter != MapNamesLang::HostAllocSet.end()) {
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
                    .getCudaArchPPInfoMap()[clang::tooling::UnifiedPath(SM.getFilename(Loc))];
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
  if (MacroName == "__CUDA_ARCH__") {
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
  if (MacroName == "__CUDA_ARCH__") {
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
  const std::string E(BP, Size);
  for (auto &MacroRule : MapNames::MacroRuleMap) {
    size_t Pos = 0;
    std::string MacroName = MacroRule.first;

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

      auto Repl = generateReplacement(IB, MacroRule.second);
      if (MacroName == "__CUDA_ARCH__" &&
          DpctGlobalInfo::getInstance().getContext().getLangOpts().CUDA) {
        insertCudaArchRepl(Repl->getReplacement(DpctGlobalInfo::getContext()));
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
  return !IsAngled ||
         !MapNames::isInSet(MapNamesLangLib::ThrustFileExcludeSet, Name);
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

    clang::tooling::UnifiedPath InFile = SM.getFilename(Loc).str();
    if (IsFileInCmd || ProcessAll ||
        GetSourceFileType(InFile.getCanonicalPath()) & SPT_CudaSource) {
      IncludeFileMap[DpctGlobalInfo::removeSymlinks(
          SM.getFileManager(), InFile.getCanonicalPath().str())] = false;
    }
    IsFileInCmd = false;

    loadYAMLIntoFileInfo(InFile.getCanonicalPath());
  }
}

} // namespace dpct
} // namespace clang