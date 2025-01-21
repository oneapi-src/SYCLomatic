//===--------------- AnalysisInfo.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "Diagnostics/Diagnostics.h"
#include "MigrationReport/Statics.h"
#include "RuleInfra/ExprAnalysis.h"
#include "RuleInfra/MapNames.h"
#include "RulesLang/MapNamesLang.h"
#include "RulesMathLib/MapNamesRandom.h"
#include "TextModification.h"
#include "Utility.h"

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include <algorithm>
#include <deque>
#include <optional>
#include <string>
#define TYPELOC_CAST(Target) static_cast<const Target &>(TL)

llvm::StringRef getReplacedName(const clang::NamedDecl *D) {
  auto Iter = clang::dpct::MapNames::TypeNamesMap.find(D->getQualifiedNameAsString(false));
  if (Iter != clang::dpct::MapNames::TypeNamesMap.end()) {
    auto Range = clang::dpct::getDefinitionRange(D->getBeginLoc(), D->getEndLoc());
    for (auto ItHeader = Iter->second->Includes.begin();
         ItHeader != Iter->second->Includes.end(); ItHeader++) {
      clang::dpct::DpctGlobalInfo::getInstance().insertHeader(Range.getBegin(),
                                                              *ItHeader);
    }
    return Iter->second->NewName;
  }
  return llvm::StringRef();
}

namespace clang {
extern std::function<bool(SourceLocation)> IsInAnalysisScopeFunc;
extern std::function<unsigned int()> GetRunRound;
extern std::function<void(SourceLocation, unsigned)> RecordTokenSplit;
namespace dpct {
///// global variable definition /////
std::vector<std::pair<HeaderType, std::string>> HeaderSpellings;
static const std::string RegexPrefix = "{{NEEDREPLACE", RegexSuffix = "}}";

///// global function definition /////
void initHeaderSpellings() {
  HeaderSpellings = {
#define HEADER(Name, Spelling) {HT_##Name, Spelling},
#include "RulesInclude/HeaderTypes.inc"
  };
}
const std::string &getDefaultString(HelperFuncType HFT) {
  const static std::string NullString;
  switch (HFT) {
  case clang::dpct::HelperFuncType::HFT_DefaultQueue: {
    const static std::string DefaultQueue =
        DpctGlobalInfo::useNoQueueDevice()
            ? DpctGlobalInfo::getGlobalQueueName()
            : DpctGlobalInfo::getDefaultQueueFreeFuncCall();
    return DefaultQueue;
  }
  case clang::dpct::HelperFuncType::HFT_DefaultQueuePtr: {
    const static std::string DefaultQueue =
        DpctGlobalInfo::useNoQueueDevice()
            ? DpctGlobalInfo::getGlobalQueueName()
            : (DpctGlobalInfo::useSYCLCompat()
                   ? buildString(MapNames::getDpctNamespace() +
                                 "get_current_device().default_queue()")
                   : buildString(
                         "&", DpctGlobalInfo::getDefaultQueueFreeFuncCall()));
    return DefaultQueue;
  }
  case clang::dpct::HelperFuncType::HFT_CurrentDevice: {
    const static std::string DefaultDevice =
        DpctGlobalInfo::useNoQueueDevice()
            ? DpctGlobalInfo::getGlobalDeviceName()
            : MapNames::getDpctNamespace() + "get_current_device()";
    return DefaultDevice;
  }
  case clang::dpct::HelperFuncType::HFT_InitValue: {
    return NullString;
  }
  }
  clang::dpct::DpctDebugs()
      << "[HelperFuncType] Unexpected value: "
      << static_cast<std::underlying_type_t<HelperFuncType>>(HFT) << "\n";
  assert(0);
  return NullString;
}
std::string getStringForRegexDefaultQueueAndDevice(HelperFuncType HFT,
                                                   int Index) {
  if (HFT == HelperFuncType::HFT_DefaultQueue ||
      HFT == HelperFuncType::HFT_DefaultQueuePtr ||
      HFT == HelperFuncType::HFT_CurrentDevice) {
    if (DpctGlobalInfo::getDeviceChangedFlag() ||
        !DpctGlobalInfo::getUsingDRYPattern()) {
      return getDefaultString(HFT);
    }

    auto HelperFuncReplInfoIter =
        DpctGlobalInfo::getHelperFuncReplInfoMap().find(Index);
    if (HelperFuncReplInfoIter ==
        DpctGlobalInfo::getHelperFuncReplInfoMap().end()) {
      return getDefaultString(HFT);
    }

    std::string CounterKey =
        HelperFuncReplInfoIter->second.DeclLocFile.getCanonicalPath().str() +
        ":" + std::to_string(HelperFuncReplInfoIter->second.DeclLocOffset);

    auto TempVariableDeclCounterIter =
        DpctGlobalInfo::getTempVariableDeclCounterMap().find(CounterKey);
    if (TempVariableDeclCounterIter ==
        DpctGlobalInfo::getTempVariableDeclCounterMap().end()) {
      return getDefaultString(HFT);
    }

    return TempVariableDeclCounterIter->second
        .PlaceholderStr[static_cast<int>(HFT)];
  }
  return "";
}
template <class T>
void removeDuplicateVar(GlobalMap<T> &VarMap,
                        std::unordered_set<std::string> &VarNames) {
  auto Itr = VarMap.begin();
  while (Itr != VarMap.end()) {
    if (VarNames.find(Itr->second->getName()) == VarNames.end()) {
      VarNames.insert(Itr->second->getName());
      ++Itr;
    } else {
      Itr = VarMap.erase(Itr);
    }
  }
}
template <class CallT>
bool deduceTemplateArguments(const CallT *C, const FunctionTemplateDecl *FTD,
                             std::vector<TemplateArgumentInfo> &TAIList) {
  if (!FTD)
    return false;

  if (!DpctGlobalInfo::isInAnalysisScope(FTD->getBeginLoc()))
    return false;
  auto &TemplateParmsList = *FTD->getTemplateParameters();
  if (TAIList.size() == TemplateParmsList.size())
    return true;
  if (TAIList.size() > TemplateParmsList.size())
    return false;

  TAIList.resize(TemplateParmsList.size());

  auto ArgItr = C->arg_begin();
  auto ParmItr = FTD->getTemplatedDecl()->param_begin();
  while (ArgItr != C->arg_end() &&
         ParmItr != FTD->getTemplatedDecl()->param_end()) {
    deduceTemplateArgument(TAIList, *ArgItr, *ParmItr);
    ++ArgItr;
    ++ParmItr;
  }
  for (size_t i = 0; i < TAIList.size(); ++i) {
    auto &Arg = TAIList[i];
    if (!Arg.isNull())
      continue;
    auto TemplateParm = TemplateParmsList.getParam(i);
    if (auto TTPD = dyn_cast<TemplateTypeParmDecl>(TemplateParm)) {
      if (TTPD->hasDefaultArgument()) {
        if (auto TSI = TTPD->getDefaultArgument().getTypeSourceInfo()) {
          Arg.setAsType(TSI->getTypeLoc());
        }
      }
    } else if (auto NTTPD = dyn_cast<NonTypeTemplateParmDecl>(TemplateParm)) {
      if (NTTPD->hasDefaultArgument()) {
        if (auto TSI = NTTPD->getDefaultArgument().getTypeSourceInfo()) {
          Arg.setAsType(TSI->getTypeLoc());
        }
      }
    }
  }
  return false;
}

template <class CallT>
bool deduceTemplateArguments(const CallT *C, const FunctionDecl *FD,
                             std::vector<TemplateArgumentInfo> &TAIList) {
  if (FD)
    return deduceTemplateArguments(C, FD->getPrimaryTemplate(), TAIList);
  return false;
}

template <class CallT>
bool deduceTemplateArguments(const CallT *C, const NamedDecl *ND,
                             std::vector<TemplateArgumentInfo> &TAIList) {
  if (!ND)
    return false;
  if (auto FTD = dyn_cast<FunctionTemplateDecl>(ND)) {
    return deduceTemplateArguments(C, FTD, TAIList);
  } else if (auto FD = dyn_cast<FunctionDecl>(ND)) {
    return deduceTemplateArguments(C, FD, TAIList);
  } else if (auto UD = dyn_cast<UsingShadowDecl>(ND)) {
    return deduceTemplateArguments(C, UD->getUnderlyingDecl(), TAIList);
  }
  return false;
}
SourceLocation getActualInsertLocation(SourceLocation InsertLoc,
                                       const SourceManager &SM,
                                       const LangOptions &LO) {
  do {
    if (InsertLoc.isFileID())
      return InsertLoc;

    if (SM.isAtEndOfImmediateMacroExpansion(InsertLoc.getLocWithOffset(
            Lexer::MeasureTokenLength(SM.getSpellingLoc(InsertLoc), SM, LO)))) {
      // If InsertLoc is at the end of macro definition, continue to find
      // immediate expansion. example: #define BBB int bbb #define CALL foo(int
      // aaa, BBB) The insert location should be at the end of BBB instead of
      // the end of bbb.
      InsertLoc = SM.getImmediateExpansionRange(InsertLoc).getBegin();
    } else if (SM.isMacroArgExpansion(InsertLoc)) {
      // If is macro argument, continue to find if argument is macro or written
      // code.
      // example:
      // #define BBB int b, int c = 0
      // #define CALL(x) foo(int aaa, x)
      // CALL(BBB)
      InsertLoc = SM.getImmediateSpellingLoc(InsertLoc);
    } else {
      // Else return insert location directly,
      return InsertLoc;
    }
  } while (true);

  return InsertLoc;
}
template <class TargetType>
std::shared_ptr<TargetType> makeTextureObjectInfo(const ValueDecl *D,
                                                  bool IsKernelCall) {
  if (IsKernelCall) {
    if (auto VD = dyn_cast<VarDecl>(D)) {
      return std::make_shared<TargetType>(VD);
    }
  } else if (const auto *PVD = dyn_cast<ParmVarDecl>(D);
             PVD && PVD->getTypeSourceInfo()) {
    return std::make_shared<TargetType>(PVD);
  }
  return std::shared_ptr<TargetType>();
}
bool isModuleFunction(const FunctionDecl *FD) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  return FD->getLanguageLinkage() == CLanguageLinkage &&
         FD->hasAttr<CUDAGlobalAttr>() &&
         DpctGlobalInfo::getModuleFiles().find(
             DpctGlobalInfo::getLocInfo(SM.getExpansionLoc(FD->getBeginLoc()))
                 .first) != DpctGlobalInfo::getModuleFiles().end();
}
void processTypeLoc(const TypeLoc &TL, ExprAnalysis &EA,
                    const SourceManager &SM) {
  EA.analyze(TL);
  if (EA.hasReplacement()) {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(SM, &TL, EA.getReplacedString(),
                                         nullptr));
  }
  EA.applyAllSubExprRepl();
}
HelperFuncCatalog getQueueKind() {
  if (DpctGlobalInfo::useSYCLCompat()) {
    return HelperFuncCatalog::GetDefaultQueue;
  }
  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
    return HelperFuncCatalog::GetInOrderQueue;
  }
  return HelperFuncCatalog::GetOutOfOrderQueue;
}

///// class FreeQueriesInfo /////
class FreeQueriesInfo {
public:
  enum FreeQueriesKind {
    NdItem = 0,
    Group,
    SubGroup,
    End,
  };
  static constexpr char FreeQueriesRegexCh = 'F';

private:
  static constexpr unsigned KindBits = 4;
  static constexpr unsigned KindMask = (1 << KindBits) - 1;
  static constexpr unsigned MacroShiftBits = KindBits;
  static constexpr unsigned MacroMask = 1 << MacroShiftBits;
  static constexpr unsigned IndexShiftBits = MacroShiftBits + 1;

private:
  struct FreeQueriesNames {
    std::string NonFreeQueriesName;
    std::string FreeQueriesFuncName;
    std::string ExtraVariableName;
  };
  struct MacroInfo {
    clang::tooling::UnifiedPath FilePath;
    unsigned Offset;
    unsigned Dimension = 0;
    std::vector<unsigned> Infos;
  };
  static std::vector<std::shared_ptr<FreeQueriesInfo>> InfoList;
  static std::vector<std::shared_ptr<MacroInfo>> MacroInfos;

  clang::tooling::UnifiedPath FilePath;
  unsigned ExtraDeclLoc = 0;
  unsigned Counter[FreeQueriesKind::End] = {0};
  std::string Indent;
  std::string NL;
  std::shared_ptr<DeviceFunctionInfo> FuncInfo;
  unsigned Dimension = 3;
  std::set<unsigned> Refs;
  unsigned Idx = 0;

  static const FreeQueriesNames &getNames(FreeQueriesKind);
  static std::shared_ptr<FreeQueriesInfo> getInfo(const FunctionDecl *);
  static void printFreeQueriesFunctionName(llvm::raw_ostream &OS,
                                           FreeQueriesKind K,
                                           unsigned Dimension) {
    OS << getNames(K).FreeQueriesFuncName;
    if (K != FreeQueriesKind::SubGroup) {
      OS << '<';
      if (Dimension) {
        OS << Dimension;
      } else {
        OS << "dpct_placeholder /* Fix the dimension manually */";
      }
      OS << '>';
    }
    OS << "()";
  }
  static FreeQueriesKind getKind(unsigned Num) {
    return static_cast<FreeQueriesKind>(Num & KindMask);
  }
  static unsigned getIndex(unsigned Num) { return Num >> IndexShiftBits; }
  static bool isMacro(unsigned Num) { return Num & MacroMask; }
  static unsigned getRegexNum(unsigned Idx, bool IsMacro,
                              FreeQueriesKind Kind) {
    return static_cast<unsigned>((Idx << IndexShiftBits) |
                                 (IsMacro * MacroMask) | (Kind & KindMask));
  }

  void emplaceExtraDecl();
  void printImmediateText(llvm::raw_ostream &, SourceLocation, FreeQueriesKind);
  std::string getReplaceString(FreeQueriesKind K);

public:
  static void reset() {
    InfoList.clear();
    MacroInfos.clear();
  }
  template <class Node>
  static void printImmediateText(llvm::raw_ostream &, const Node *,
                                 const FunctionDecl *, FreeQueriesKind);
  static void buildInfo() {
    for (auto &Info : InfoList)
      Info->emplaceExtraDecl();
    for (auto &Info : MacroInfos) {
      Info->Dimension = InfoList[Info->Infos.front()]->Dimension;
      for (auto Idx : Info->Infos) {
        if (Info->Dimension != InfoList[Idx]->Dimension) {
          Info->Dimension = 0;
          DiagnosticsUtils::report(Info->FilePath, Info->Offset,
                                   Diagnostics::FREE_QUERIES_DIMENSION, true,
                                   false);
          break;
        }
      }
    }
  }
  static std::string getReplaceString(unsigned Num);

  FreeQueriesInfo() = default;
};
///// class RnnBackwardFuncInfoBuilder /////
class RnnBackwardFuncInfoBuilder {
  std::vector<RnnBackwardFuncInfo> &RBFuncInfo;
  std::vector<RnnBackwardFuncInfo> ValidBackwardDataFuncInfo;
  std::vector<RnnBackwardFuncInfo> ValidBackwardWeightFuncInfo;
  std::vector<std::shared_ptr<ExtReplacement>> Repls;
  using InfoIter = std::vector<RnnBackwardFuncInfo>::iterator;

public:
  RnnBackwardFuncInfoBuilder(std::vector<RnnBackwardFuncInfo> &Infos)
      : RBFuncInfo(Infos){};
  // This function check if the RNN function input referenced between
  // backwarddata and backwardweight functiona call.
  bool isInputNotChanged(InfoIter Data, InfoIter Weight) {
    for (auto &RnnInput : Data->RnnInputDeclLoc) {
      auto &RnnInputRefs =
          DpctGlobalInfo::getRnnInputMap()[RnnInput][Data->FilePath];
      for (auto &RnnInputRef : RnnInputRefs) {
        if ((RnnInputRef > (Data->Offset + Data->Length - 1)) &&
            RnnInputRef < Weight->Offset) {
          return false;
        }
      }
    }
    return true;
  }
  // This function check if the backwarddata and backwardweight function
  // call have same input.
  bool isInputSame(InfoIter Data, InfoIter Weight) {
    for (unsigned InputIndex = 0; InputIndex < 3; InputIndex++) {
      if (Data->RnnInputDeclLoc[InputIndex] !=
          Weight->RnnInputDeclLoc[InputIndex]) {
        return false;
      }
    }
    return true;
  }
  // This function check if the backwarddata and backwardweight function in
  // the same scope and backwardweight called after backwarddata.
  // For example, function will return ture for pattern in following pseudo
  // code:
  //   if(...) {
  //     backwarddata(...);
  //     ..
  //     backwardweight(...);
  //   }
  bool isValidScopeAndOrder(InfoIter Data, InfoIter Weight) {
    return !((Data->CompoundLoc != Weight->CompoundLoc) &&
             (Data->Offset >= Weight->Offset));
  }
  void build() {
    if (RBFuncInfo.empty()) {
      return;
    }
    for (auto &Info : RBFuncInfo) {
      if (Info.isDataGradient) {
        ValidBackwardDataFuncInfo.emplace_back(Info);
      } else {
        ValidBackwardWeightFuncInfo.emplace_back(Info);
      }
    }
    std::vector<int> WeightPairdFlag(ValidBackwardWeightFuncInfo.size(), 0);
    auto DataBegin = ValidBackwardDataFuncInfo.begin();
    auto DataEnd = ValidBackwardDataFuncInfo.end();
    auto WeightBegin = ValidBackwardWeightFuncInfo.begin();
    auto WeightEnd = ValidBackwardWeightFuncInfo.end();
    for (auto DataIter = DataBegin; DataIter != DataEnd; DataIter++) {
      bool DataPaired = false;
      for (auto WeightIter = WeightBegin; WeightIter != WeightEnd;
           WeightIter++) {
        if (isInputNotChanged(DataIter, WeightIter) &&
            isInputSame(DataIter, WeightIter) &&
            isValidScopeAndOrder(DataIter, WeightIter)) {
          DataPaired = true;
          WeightPairdFlag[WeightIter - WeightBegin] = 1;
          auto Repl = generateReplacement(DataIter, WeightIter);
          Repls.insert(Repls.end(), Repl.begin(), Repl.end());
          break;
        }
      }
      if (!DataPaired) {
        DiagnosticsUtils::report(DataIter->FilePath, DataIter->Offset,
                                 Diagnostics::API_NOT_MIGRATED, true, false,
                                 "cudnnRNNBackwardData_v8");
      }
    }
    for (auto WeightIter = WeightBegin; WeightIter != WeightEnd; WeightIter++) {
      if (!WeightPairdFlag[WeightIter - WeightBegin]) {
        DiagnosticsUtils::report(WeightIter->FilePath, WeightIter->Offset,
                                 Diagnostics::API_NOT_MIGRATED, true, false,
                                 "cudnnRNNBackwardWeights_v8");
      }
    }
  }
  std::vector<std::shared_ptr<ExtReplacement>> getReplacement() {
    return Repls;
  }
  std::vector<std::shared_ptr<ExtReplacement>>
  generateReplacement(InfoIter Data, InfoIter Weight) {
    std::vector<std::shared_ptr<ExtReplacement>> Repls;
    std::ostringstream DataRepl, WeightRepl;
    RnnBackwardFuncInfo &DataFuncInfo = *Data;
    RnnBackwardFuncInfo &WeightFuncInfo = *Weight;
    requestFeature(HelperFeatureEnum::device_ext);
    Diagnostics WarningType;
    if (WeightFuncInfo.isAssigned) {
      WarningType = Diagnostics::FUNC_CALL_REMOVED_0;
      WeightRepl << "0";
    } else {
      WarningType = Diagnostics::FUNC_CALL_REMOVED;
    }
    DiagnosticsUtils::report(
        WeightFuncInfo.FilePath, WeightFuncInfo.Offset, WarningType, true,
        false, "cudnnRNNBackwardWeights_v8",
        "this call and cudnnRNNBackwardData_v8 are migrated to a single "
        "function call async_rnn_backward");

    if (DataFuncInfo.isAssigned) {
      DataRepl << MapNames::getCheckErrorMacroName() << "(";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    DataRepl << DataFuncInfo.FuncArgs[0] << ".async_rnn_backward("
             << DataFuncInfo.FuncArgs[1];
    // Combine 21 args from backwarddata and 2 args from backwardweight
    // into args of async_rnn_backward.
    for (unsigned int index = 3; index <= 21; index++) {
      DataRepl << ", " << DataFuncInfo.FuncArgs[index];
      if (index == 6) {
        DataRepl << ", " << WeightFuncInfo.FuncArgs[0];
      } else if (index == 17) {
        DataRepl << ", " << WeightFuncInfo.FuncArgs[1];
      }
    }
    if (DataFuncInfo.isAssigned) {
      DataRepl << "))";
    } else {
      DataRepl << ")";
    }
    Repls.emplace_back(std::make_shared<ExtReplacement>(
        DataFuncInfo.FilePath, DataFuncInfo.Offset, DataFuncInfo.Length,
        DataRepl.str(), nullptr));
    Repls.emplace_back(std::make_shared<ExtReplacement>(
        WeightFuncInfo.FilePath, WeightFuncInfo.Offset, WeightFuncInfo.Length,
        WeightRepl.str(), nullptr));

    return Repls;
  }
};
///// class EventSyncTypeInfo /////
void EventSyncTypeInfo::buildInfo(clang::tooling::UnifiedPath FilePath,
                                  unsigned int Offset) {
  if (NeedReport)
    DiagnosticsUtils::report(FilePath, Offset,
                             Diagnostics::NOERROR_RETURN_COMMA_OP, true, false);

  if (IsAssigned && ReplText.empty()) {
    ReplText = "0";
  }

  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, Offset, Length, ReplText, nullptr));
}
///// class TimeStubTypeInfo /////
void TimeStubTypeInfo::buildInfo(clang::tooling::UnifiedPath FilePath,
                                 unsigned int Offset, bool isReplTxtWithSB) {
  if (isReplTxtWithSB)
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, Offset, Length, StrWithSB,
                                         nullptr));
  else
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, Offset, Length, StrWithoutSB,
                                         nullptr));
}
///// class BuiltinVarInfo /////
void BuiltinVarInfo::buildInfo(clang::tooling::UnifiedPath FilePath,
                               unsigned int Offset, unsigned int ID) {
  std::string R = Repl + std::to_string(ID) + ")";
  DpctGlobalInfo::getInstance().addReplacement(
      std::make_shared<ExtReplacement>(FilePath, Offset, Len, R, nullptr));
}

///// class ParameterStream /////
ParameterStream &ParameterStream::operator<<(const std::string &InputParamStr) {
  if (InputParamStr.size() == 0) {
    return *this;
  }

  if (!FormatInformation.EnableFormat) {
    // append the string directly
    Str = Str + InputParamStr;
    return *this;
  }

  if (FormatInformation.IsAllParamsOneLine) {
    // all parameters are in one line
    Str = Str + ", " + InputParamStr;
    return *this;
  }

  if (FormatInformation.IsEachParamNL) {
    // each parameter is in a single line
    Str = Str + "," + getNL() + FormatInformation.NewLineIndentStr +
          InputParamStr;
    return *this;
  }

  // parameters will be inserted in one line unless the line length > column
  // limit.
  if (FormatInformation.CurrentLength + 2 + (int)InputParamStr.size() <=
      ColumnLimit) {
    Str = Str + ", " + InputParamStr;
    FormatInformation.CurrentLength =
        FormatInformation.CurrentLength + 2 + InputParamStr.size();
    return *this;
  } else {
    Str = Str + std::string(",") + getNL() +
          FormatInformation.NewLineIndentStr + InputParamStr;
    FormatInformation.CurrentLength =
        FormatInformation.NewLineIndentLength + InputParamStr.size();
    return *this;
  }
}
ParameterStream &ParameterStream::operator<<(int InputInt) {
  return *this << std::to_string(InputInt);
}
///// class DpctFileInfo /////
void DpctFileInfo::buildReplacements() {
  if (!isInAnalysisScope())
    return;

  if (FilePath.getCanonicalPath().empty())
    return;
  // Traverse all the global variables stored one by one to check if its name
  // is same with normal global variable's name in host side, if the one is
  // found, postfix "_ct" is added to this __constant__ symbol's name.
  std::unordered_map<unsigned int, std::string> ReplUpdated;
  for (const auto &Entry : MemVarMap) {
    if (Entry.second->isIgnore() || !Entry.second->isConstant() ||
        Entry.second->isUseDeviceGlobal())
      continue;

    auto Name = Entry.second->getName();
    auto &GlobalVarNameSet = dpct::DpctGlobalInfo::getGlobalVarNameSet();
    if (GlobalVarNameSet.find(Name) != end(GlobalVarNameSet)) {
      Entry.second->setName(Name + "_ct");
    }

    std::string Repl = Entry.second->getDeclarationReplacement(nullptr);
    auto FilePath = Entry.second->getFilePath();
    auto Offset = Entry.second->getNewConstVarOffset();
    auto Length = Entry.second->getNewConstVarLength();

    auto &ReplText = ReplUpdated[Offset];
    if (!ReplText.empty()) {
      ReplText += getNL() + Repl;
    } else {
      ReplText = Repl;
    }

    auto R = std::make_shared<ExtReplacement>(FilePath, Offset, Length,
                                              ReplText, nullptr);

    addReplacement(R);
  }

  for (auto &Kernel : KernelMap)
    Kernel.second->addReplacements();

  for (auto &BuiltinVar : BuiltinVarInfoMap) {
    auto Ptr = MemVarMap::getHeadWithoutPathCompression(
        &(BuiltinVar.second.DFI->getVarMap()));
    if (DpctGlobalInfo::getAssumedNDRangeDim() == 1 && Ptr) {
      unsigned int ID = (Ptr->Dim == 1) ? 0 : 2;
      BuiltinVar.second.buildInfo(FilePath, BuiltinVar.first, ID);
    } else {
      BuiltinVar.second.buildInfo(FilePath, BuiltinVar.first, 2);
    }
  }

  for (auto &AtomicInfo : AtomicMap) {
    if (std::get<2>(AtomicInfo.second))
      DiagnosticsUtils::report(getFilePath(), std::get<0>(AtomicInfo.second),
                               Diagnostics::API_NOT_OCCURRED_IN_AST, true, true,
                               std::get<1>(AtomicInfo.second));
  }

  for (auto &DescInfo : EventSyncTypeMap) {
    DescInfo.second.buildInfo(FilePath, DescInfo.first);
  }

  const auto &TimeStubBounds = getTimeStubBounds();
  if (TimeStubBounds.empty()) {
    for (auto &DescInfo : TimeStubTypeMap) {
      DescInfo.second.buildInfo(FilePath, DescInfo.first,
                                /*bool isReplTxtWithSB*/ true);
    }
  } else {
    for (auto &DescInfo : TimeStubTypeMap) {
      bool isReplTxtWithSB = isReplTxtWithSubmitBarrier(DescInfo.first);
      DescInfo.second.buildInfo(FilePath, DescInfo.first, isReplTxtWithSB);
    }
  }

  buildRnnBackwardFuncInfo();

  // insert header file of user defined rules
  std::string InsertHeaderStr;
  llvm::raw_string_ostream HeaderOS(InsertHeaderStr);
  if (!InsertedHeaders.empty()) {
    HeaderOS << getNL();
  }
  for (auto &HeaderStr : InsertedHeaders) {
    if (HeaderStr[0] != '<' && HeaderStr[0] != '"') {
      HeaderStr = "\"" + HeaderStr + "\"";
    }
    HeaderOS << "#include " << HeaderStr << getNL();
  }
  HeaderOS.flush();
  insertHeader(std::move(InsertHeaderStr), LastIncludeOffset);

  std::string InsertHeaderStrCUDA;
  llvm::raw_string_ostream HeaderOSCUDA(InsertHeaderStrCUDA);

  for (auto &HeaderStr : InsertedHeadersCUDA) {
    if (HeaderStr[0] != '<' && HeaderStr[0] != '"') {
      HeaderStr = "\"" + HeaderStr + "\"";
    }
    HeaderOSCUDA << getNL() << "#include " << HeaderStr;
  }
  HeaderOSCUDA.flush();
  insertHeader(std::move(InsertHeaderStrCUDA), LastIncludeOffset, IP_Left,
               RT_CUDAWithCodePin);

  FreeQueriesInfo::buildInfo();

  // This loop need to be put at the end of DpctFileInfo::buildReplacements.
  // In addReplacement() the insertHeader() may be invoked, so the size of
  // vector IncludeDirectiveInsertions may increase.
  // So here cannot use for loop like "for(auto e : vec)" since the iterator may
  // be invalid due to the allocation of new storage.
  for (size_t I = 0, End = IncludeDirectiveInsertions.size(); I < End; I++) {
    auto IncludeDirective = IncludeDirectiveInsertions[I];
    bool IsInExternC = false;
    unsigned int NewInsertLocation = 0;
    for (auto &ExternCRange : ExternCRanges) {
      if (IncludeDirective->getOffset() >= ExternCRange.first &&
          IncludeDirective->getOffset() <= ExternCRange.second) {
        IsInExternC = true;
        NewInsertLocation = ExternCRange.first;
        break;
      }
    }
    if (IsInExternC) {
      IncludeDirective->setOffset(NewInsertLocation);
    }
    addReplacement(IncludeDirective);
    // Update the End since the size may be changed.
    End = IncludeDirectiveInsertions.size();
  }
}
void DpctFileInfo::setKernelCallDim() {
  for (auto &Kernel : KernelMap)
    Kernel.second->setKernelCallDim();
}
void DpctFileInfo::setKernelDim() {
  for (auto &DeviceFunc : FuncMap) {
    auto Info = DeviceFunc.second->getFuncInfo();
    if (Info->isKernel() && !Info->isKernelInvoked()) {
      Info->getVarMap().Dim = 3;
    }
  }
}
void DpctFileInfo::buildUnionFindSet() {
  for (auto &Kernel : KernelMap)
    Kernel.second->buildUnionFindSet();
}
void DpctFileInfo::buildUnionFindSetForUncalledFunc() {
  for (auto &DeviceFunc : FuncMap) {
    auto Info = DeviceFunc.second->getFuncInfo();
    Info->buildInfo();
    constructUnionFindSetRecursively(Info);
  }
}
void DpctFileInfo::buildKernelInfo() {
  for (auto &Kernel : KernelMap)
    Kernel.second->buildInfo();

  for (auto &D : FuncMap) {
    if (auto I = D.second->getFuncInfo())
      I->buildInfo();
  }
}
void DpctFileInfo::buildRnnBackwardFuncInfo() {
  RnnBackwardFuncInfoBuilder Builder(RBFuncInfo);
  Builder.build();
  for (auto &Repl : Builder.getReplacement()) {
    addReplacement(Repl);
  }
}
void DpctFileInfo::postProcess() {
  if (!isInAnalysisScope())
    return;
  for (auto &D : FuncMap)
    D.second->emplaceReplacement();
  if (!ReplsSYCL->empty()) {
    ReplsSYCL->postProcess();
    if (DpctGlobalInfo::getRunRound() == 0) {
      auto &CacheEntry =
          DpctGlobalInfo::getInstance().getFileReplCache()[FilePath];
      CacheEntry.first = ReplsCUDA;
      CacheEntry.second = ReplsSYCL;
    }
  }
}
void DpctFileInfo::emplaceReplacements(
    std::map<clang::tooling::UnifiedPath, tooling::Replacements> &ReplSet) {
  if (!ReplsSYCL->empty())
    ReplsSYCL->emplaceIntoReplSet(ReplSet[FilePath]);
}
void DpctFileInfo::addReplacement(std::shared_ptr<ExtReplacement> Repl) {
  if (Repl->getLength() == 0 && Repl->getReplacementText().empty())
    return;
  if (Repl->IsForCodePin)
    ReplsCUDA->addReplacement(Repl);
  else
    ReplsSYCL->addReplacement(Repl);
}
bool DpctFileInfo::isInAnalysisScope() {
  return DpctGlobalInfo::isInAnalysisScope(FilePath);
}
void DpctFileInfo::setFileEnterOffset(unsigned Offset) {
  auto MF = DpctGlobalInfo::getInstance().getMainFile();
  if (!HasInclusionDirectiveSet.count(MF)) {
    FirstIncludeOffset[MF] = Offset;
    LastIncludeOffset = Offset;
  }
}
void DpctFileInfo::setFirstIncludeOffset(unsigned Offset) {
  auto MF = DpctGlobalInfo::getInstance().getMainFile();
  if (!HasInclusionDirectiveSet.count(MF)) {
    FirstIncludeOffset[MF] = Offset;
    LastIncludeOffset = Offset;
    HasInclusionDirectiveSet.insert(std::move(MF));
  }
}
void DpctFileInfo::concatHeader(llvm::raw_string_ostream &OS) {}
template <class FirstT, class... Args>
void DpctFileInfo::concatHeader(llvm::raw_string_ostream &OS, FirstT &&First,
                                Args &&...Arguments) {
  appendString(OS, "#include ", std::forward<FirstT>(First), getNL());
  concatHeader(OS, std::forward<Args>(Arguments)...);
}
std::optional<HeaderType> DpctFileInfo::findHeaderType(StringRef Header) {
  auto Pos = llvm::find_if(
      HeaderSpellings, [=](const std::pair<HeaderType, StringRef> &p) -> bool {
        return p.second == Header;
      });
  if (Pos == std::end(HeaderSpellings))
    return std::nullopt;
  return Pos->first;
}
StringRef DpctFileInfo::getHeaderSpelling(HeaderType Value) {
  if (Value < NUM_HEADERS)
    return HeaderSpellings[Value].second;

  // Only assertion in debug
  assert(false && "unknown HeaderType");
  return "";
}
void DpctFileInfo::insertHeader(HeaderType Type, unsigned Offset,
                                ReplacementType IsForCodePin) {
  if (Type == HT_DPL_Algorithm || Type == HT_DPL_Execution || Type == HT_SYCL) {
    if (auto MF = DpctGlobalInfo::getInstance().getMainFile())
      if (this != MF.get() && FirstIncludeOffset.count(MF)) {
        DpctGlobalInfo::getInstance().getMainFile()->insertHeader(
            Type, FirstIncludeOffset.at(MF));
      }
  }
  if (HeaderInsertedBitMap[Type])
    return;
  HeaderInsertedBitMap[Type] = true;
  std::string ReplStr;
  llvm::raw_string_ostream OS(ReplStr);
  std::string MigratedMacroDefinitionStr;
  llvm::raw_string_ostream MigratedMacroDefinitionOS(
      MigratedMacroDefinitionStr);

  switch (Type) {
  // The #include of <oneapi/dpl/execution> and <oneapi/dpl/algorithm> were
  // previously added here.  However, due to some unfortunate include
  // dependencies introduced with the PSTL/TBB headers from the gcc-9.3.0
  // include files, those two headers must now be included before the
  // <sycl/sycl.hpp> are included, so the FileInfo is set to hold a boolean
  // that'll indicate whether to insert them when the #include <sycl/sycl.cpp>
  // is added later
  case HT_DPL_Algorithm:
  case HT_DPL_Execution:
    concatHeader(OS, getHeaderSpelling(Type));
    if (auto Iter = FirstIncludeOffset.find(
            DpctGlobalInfo::getInstance().getMainFile());
        Iter != FirstIncludeOffset.end())
      insertHeader(OS.str(), Iter->second, InsertPosition::IP_AlwaysLeft);
    return;
  case HT_SYCL:
    // Add the label for profiling macro "DPCT_PROFILING_ENABLED", which will be
    // replaced by "#define DPCT_PROFILING_ENABLED" or not in the post
    // replacement.
    OS << "{{NEEDREPLACEP0}}";

    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
      OS << "#define DPCT_USM_LEVEL_NONE" << getNL();
    concatHeader(OS, getHeaderSpelling(Type));
    if (DpctGlobalInfo::useSYCLCompat()) {
      concatHeader(OS, getHeaderSpelling(HT_COMPAT_SYCLcompat));
      HeaderInsertedBitMap[HT_COMPAT_SYCLcompat] = true;
    } else {
      concatHeader(OS, getHeaderSpelling(HT_DPCT_Dpct));
      HeaderInsertedBitMap[HT_DPCT_Dpct] = true;
    }
    DpctGlobalInfo::printUsingNamespace(OS);
    if (DpctGlobalInfo::useNoQueueDevice()) {
      static bool Flag = true;
      auto SourceFileType = GetSourceFileType(getFilePath());
      if (Flag && (SourceFileType == SPT_CudaSource ||
                   SourceFileType == SPT_CppSource)) {
        OS << MapNames::getClNamespace() << "device "
           << DpctGlobalInfo::getGlobalDeviceName()
           << "(sycl::default_selector_v);" << getNL();
        // Now the UsmLevel must not be UL_None here.
        OS << MapNames::getClNamespace() << "queue "
           << DpctGlobalInfo::getGlobalQueueName() << "("
           << DpctGlobalInfo::getGlobalDeviceName() << ", "
           << MapNames::getClNamespace() << "property_list{"
           << MapNames::getClNamespace() << "property::queue::in_order()";

        // replaced to insert "property::queue::enable_profiling()" or not
        // in the post replacement.
        OS << "{{NEEDREPLACEI0}}";
        OS << "});" << getNL();
        Flag = false;
      } else {
        OS << "extern " << MapNames::getClNamespace() << "device "
           << DpctGlobalInfo::getGlobalDeviceName() << ";" << getNL();
        // Now the UsmLevel must not be UL_None here.
        OS << "extern " << MapNames::getClNamespace() << "queue "
           << DpctGlobalInfo::getGlobalQueueName() << ";" << getNL();
      }
    }
    if (auto Iter = FirstIncludeOffset.find(
            DpctGlobalInfo::getInstance().getMainFile());
        Iter != FirstIncludeOffset.end())
      insertHeader(OS.str(), Iter->second, InsertPosition::IP_Left);
    if (!RTVersionValue.empty())
      MigratedMacroDefinitionOS << "#define DPCT_COMPAT_RT_VERSION "
                                << RTVersionValue << getNL();
    if (!MajorVersionValue.empty())
      MigratedMacroDefinitionOS << "#define DPCT_COMPAT_RT_MAJOR_VERSION "
                                << MajorVersionValue << getNL();
    if (!MinorVersionValue.empty())
      MigratedMacroDefinitionOS << "#define DPCT_COMPAT_RT_MINOR_VERSION "
                                << MinorVersionValue << getNL();
    if (!CCLVerValue.empty())
      MigratedMacroDefinitionOS << "#define DPCT_COMPAT_CCL_VERSION "
                                << CCLVerValue << getNL();
    insertHeader(MigratedMacroDefinitionOS.str(), FileBeginOffset,
                 InsertPosition::IP_AlwaysLeft);
    for (const auto &File :
         DpctGlobalInfo::getCustomHelperFunctionAddtionalIncludes()) {
      if (auto Iter = FirstIncludeOffset.find(
              DpctGlobalInfo::getInstance().getMainFile());
          Iter != FirstIncludeOffset.end())
        if (!File.empty() && File[0] == '<')
          insertHeader("#include " + File + getNL(), Iter->second,
                       InsertPosition::IP_Right);
        else
          insertHeader("#include \"" + File + "\"" + getNL(), Iter->second,
                       InsertPosition::IP_Right);
    }
    return;

  // Because <dpct/dpl_utils.hpp> includes <oneapi/dpl/execution> and
  // <oneapi/dpl/algorithm>, so we have to make sure that
  // <oneapi/dpl/execution> and <oneapi/dpl/algorithm> are inserted before
  // <sycl/sycl.hpp>
  // e.g.
  // #include <sycl/sycl.hpp>
  // #include <dpct/dpct.hpp>
  // #include <dpct/dpl_utils.hpp>
  // ...
  // This will cause compilation error due to onedpl header dependence
  // The order we expect is:
  // e.g.
  // #include <oneapi/dpl/execution>
  // #include <oneapi/dpl/algorithm>
  // #include <sycl/sycl.hpp>
  // #include <dpct/dpct.hpp>
  // #include <dpct/dpl_utils.hpp>
  //
  // We will insert <oneapi/dpl/execution> and <oneapi/dpl/algorithm> at the
  // begining of the main file
  case HT_DPCT_DPL_Utils:
    insertHeader(HT_DPL_Execution);
    insertHeader(HT_DPL_Algorithm);
    break;
  case HT_DPCT_CodePin_CUDA:
  case HT_DPCT_CodePin_SYCL: {
    OS << getNL();
    concatHeader(OS, getHeaderSpelling(Type));
    std::string CurrentFilePath =
        llvm::sys::path::convert_to_slash(getFilePath().getCanonicalPath());
    auto InRootPath = llvm::sys::path::convert_to_slash(
        DpctGlobalInfo::getInRoot().getCanonicalPath());
    size_t FilePathCount =
        std::count_if(CurrentFilePath.begin(), CurrentFilePath.end(),
                      [](char c) { return c == '/'; });
    size_t InRootPathCount = std::count_if(InRootPath.begin(), InRootPath.end(),
                                           [](char c) { return c == '/'; });
    std::string SchemaRelativePath = "\"";
    assert(FilePathCount >= InRootPathCount &&
           "The processed file should be under --in-root folder.");
    for (size_t i = 1; i < FilePathCount - InRootPathCount; i++) {
      SchemaRelativePath += "../";
    }
    SchemaRelativePath += "codepin_autogen_util.hpp\"";
    concatHeader(OS, SchemaRelativePath);
    if (auto Iter = FirstIncludeOffset.find(
            DpctGlobalInfo::getInstance().getMainFile());
        Iter != FirstIncludeOffset.end())
      insertHeader(OS.str(), Iter->second, InsertPosition::IP_Right,
                   IsForCodePin);
    return;
  } break;
  default:
    break;
  }

  if (FirstIncludeOffset.count(DpctGlobalInfo::getInstance().getMainFile()) &&
      Offset !=
          FirstIncludeOffset.at(DpctGlobalInfo::getInstance().getMainFile()))
    OS << getNL();
  concatHeader(OS, getHeaderSpelling(Type));
  return insertHeader(OS.str(), LastIncludeOffset, InsertPosition::IP_Right);
}
void DpctFileInfo::insertHeader(HeaderType Type, ReplacementType IsForCodePin) {
  switch (Type) {
#define HEADER(Name, Spelling)                                                 \
  case HT_##Name:                                                              \
    return insertHeader(HT_##Name, LastIncludeOffset, IsForCodePin);
#include "RulesInclude/HeaderTypes.inc"
  default:
    return;
  }
}
const DpctFileInfo::SourceLineInfo &
DpctFileInfo::getLineInfo(unsigned LineNumber) {
  if (!LineNumber || LineNumber > Lines.size()) {
    llvm::dbgs() << "[DpctFileInfo::getLineInfo] illegal line number "
                 << LineNumber;
    static SourceLineInfo InvalidLine;
    return InvalidLine;
  }
  return Lines[--LineNumber];
}
void DpctFileInfo::setLineRange(ExtReplacements::SourceLineRange &LineRange,
                                std::shared_ptr<ExtReplacement> Repl) {
  unsigned Begin = Repl->getOffset();
  unsigned End = Begin + Repl->getLength();

  // Update original code range embedded in the migrated code
  auto &Map = getFuncDeclRangeMap();
  for (auto &Entry : Map) {
    for (auto &Range : Entry.second) {
      if (Begin >= Range.first && End <= Range.second) {
        Begin = Range.first;
        End = Range.second;
      }
    }
  }

  auto &BeginLine = getLineInfoFromOffset(Begin);
  auto &EndLine = getLineInfoFromOffset(End);
  LineRange.SrcBeginLine = BeginLine.Number;
  LineRange.SrcBeginOffset = BeginLine.Offset;
  if (EndLine.Offset == End)
    LineRange.SrcEndLine = EndLine.Number - 1;
  else
    LineRange.SrcEndLine = EndLine.Number;
}
void DpctFileInfo::insertIncludedFilesInfo(std::shared_ptr<DpctFileInfo> Info) {
  auto Iter = IncludedFilesInfoSet.find(Info);
  if (Iter == IncludedFilesInfoSet.end()) {
    IncludedFilesInfoSet.insert(Info);
  }
}

bool DpctFileInfo::isReplTxtWithSubmitBarrier(unsigned Offset) {
  bool ReplTxtWithSB = true;
  for (const auto &Entry : TimeStubBounds) {
    size_t Begin = Entry.first;
    size_t End = Entry.second;
    if (Offset >= Begin && Offset <= End) {
      ReplTxtWithSB = false;
      break;
    }
  }
  return ReplTxtWithSB;
}
// TODO: implement one of this for each source language.
bool DpctFileInfo::isInCudaPath() {
  return DpctGlobalInfo::isInCudaPath(FilePath);
}
void DpctFileInfo::buildLinesInfo() {
  if (FilePath.getCanonicalPath().empty())
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();

  llvm::Expected<FileEntryRef> Result =
      SM.getFileManager().getFileRef(FilePath.getCanonicalPath());

  if (auto E = Result.takeError())
    return;

  auto FID = SM.getOrCreateFileID(*Result, SrcMgr::C_User);
  auto &Content = SM.getSLocEntry(FID).getFile().getContentCache();
  if (!Content.SourceLineCache) {
    bool Invalid;
    SM.getLineNumber(FID, 0, &Invalid);
    if (Invalid)
      return;
  }
  auto RawBuffer =
      Content.getBufferOrNone(SM.getDiagnostics(), SM.getFileManager())
          .value_or(llvm::MemoryBufferRef())
          .getBuffer();
  if (RawBuffer.empty())
    return;
  FileContentCache = RawBuffer.str();
  FileSize = RawBuffer.size();
  auto LineCache = Content.SourceLineCache.getLines();
  auto NumLines = Content.SourceLineCache.size();
  StringRef CacheBuffer(FileContentCache);
  for (unsigned L = 1; L < NumLines; ++L)
    Lines.emplace_back(L, LineCache, CacheBuffer);
  Lines.emplace_back(NumLines, LineCache[NumLines - 1], FileSize, CacheBuffer);
}
const DpctFileInfo::SourceLineInfo &
DpctFileInfo::getLineInfoFromOffset(unsigned Offset) {
  return *(std::upper_bound(Lines.begin(), Lines.end(), Offset,
                            [](unsigned Offset, const SourceLineInfo &Line) {
                              return Line.Offset > Offset;
                            }) -
           1);
}
///// class DpctGlobalInfo /////
DpctGlobalInfo::MacroDefRecord::MacroDefRecord(SourceLocation NTL, bool IIAS)
    : IsInAnalysisScope(IIAS) {
  auto LocInfo = DpctGlobalInfo::getLocInfo(NTL);
  FilePath = LocInfo.first;
  Offset = LocInfo.second;
}

DpctGlobalInfo::MacroArgRecord::MacroArgRecord(const MacroInfo *MI,
                                               int ArgIndex)
    : ArgIndex(ArgIndex) {
  ArgName = MI->params()[ArgIndex]->getName().str();
  for (auto Tok : MI->tokens()) {
    auto II = Tok.getIdentifierInfo();
    if (II && (II == MI->params()[ArgIndex])) {
      ArgLoc = Tok.getLocation();
      break;
    }
  }
}

DpctGlobalInfo::MacroExpansionRecord::MacroExpansionRecord(
    IdentifierInfo *ID, const MacroInfo *MI, SourceRange Range,
    bool IsInAnalysisScope, int TokenIndex) {
  auto LocInfoBegin =
      DpctGlobalInfo::getLocInfo(MI->getReplacementToken(0).getLocation());
  auto LocInfoEnd = DpctGlobalInfo::getLocInfo(
      MI->getReplacementToken(MI->getNumTokens() - 1).getLocation());
  Name = ID->getName().str();
  NumTokens = MI->getNumTokens();
  FilePath = LocInfoBegin.first;
  ReplaceTokenBeginOffset = LocInfoBegin.second;
  ReplaceTokenEndOffset = LocInfoEnd.second;
  this->Range = Range;
  this->IsInAnalysisScope = IsInAnalysisScope;
  this->IsFunctionLike = MI->getNumParams() > 0;
  this->TokenIndex = TokenIndex;
}
std::string DpctGlobalInfo::removeSymlinks(clang::FileManager &FM,
                                           std::string FilePathStr) {
  // Get rid of symlinks
  SmallString<4096> NoSymlinks = StringRef("");
  auto Dir =
      FM.getOptionalDirectoryRef(llvm::sys::path::parent_path(FilePathStr));
  if (Dir) {
    StringRef DirName = FM.getCanonicalName(*Dir);
    StringRef FileName = llvm::sys::path::filename(FilePathStr);
    llvm::sys::path::append(NoSymlinks, DirName, FileName);
  }
  return NoSymlinks.str().str();
}
bool DpctGlobalInfo::isInRoot(const clang::tooling::UnifiedPath &FilePath) {
  if (isChildPath(InRoot, FilePath)) {
    return !isExcluded(FilePath);
  } else {
    return false;
  }
}
bool DpctGlobalInfo::isExcluded(const clang::tooling::UnifiedPath &FilePath) {
  static std::map<std::string, bool> Cache;
  if (FilePath.getPath().empty() || DpctGlobalInfo::getExcludePath().empty()) {
    return false;
  }
  if (FilePath.getCanonicalPath().empty()) {
    return false;
  }
  if (Cache.count(FilePath.getCanonicalPath().str())) {
    return Cache[FilePath.getCanonicalPath().str()];
  }
  for (auto &Path : DpctGlobalInfo::getExcludePath()) {
    if (isChildOrSamePath(Path.first, FilePath)) {
      Cache[FilePath.getCanonicalPath().str()] = true;
      return true;
    }
  }
  Cache[FilePath.getCanonicalPath().str()] = false;
  return false;
}
// TODO: implement one of this for each source language.
bool DpctGlobalInfo::isInCudaPath(SourceLocation SL) {
  return isInCudaPath(getSourceManager()
                          .getFilename(getSourceManager().getExpansionLoc(SL))
                          .str());
}
void DpctGlobalInfo::setSYCLFileExtension(SYCLFileExtensionEnum Extension) {
  switch (Extension) {
  case SYCLFileExtensionEnum::DP_CPP:
    SYCLSourceExtension = ".dp.cpp";
    SYCLHeaderExtension = ".dp.hpp";
    break;
  case SYCLFileExtensionEnum::SYCL_CPP:
    SYCLSourceExtension = ".sycl.cpp";
    SYCLHeaderExtension = ".sycl.hpp";
    break;
  case SYCLFileExtensionEnum::CPP:
    SYCLSourceExtension = ".cpp";
    SYCLHeaderExtension = ".hpp";
    break;
  }
}

void DpctGlobalInfo::printItem(llvm::raw_ostream &OS, const Stmt *S,
                               const FunctionDecl *FD) {
  FreeQueriesInfo::printImmediateText(OS, S, FD,
                                      FreeQueriesInfo::FreeQueriesKind::NdItem);
}
std::string DpctGlobalInfo::getItem(const Stmt *S, const FunctionDecl *FD) {
  return buildStringFromPrinter(DpctGlobalInfo::printItem, S, FD);
}
void DpctGlobalInfo::registerNDItemUser(const Stmt *S, const FunctionDecl *FD) {
  getItem(S, FD);
}
void DpctGlobalInfo::printGroup(llvm::raw_ostream &OS, const Stmt *S,
                                const FunctionDecl *FD) {
  FreeQueriesInfo::printImmediateText(OS, S, FD,
                                      FreeQueriesInfo::FreeQueriesKind::Group);
}
std::string DpctGlobalInfo::getGroup(const Stmt *S, const FunctionDecl *FD) {
  return buildStringFromPrinter(DpctGlobalInfo::printGroup, S, FD);
}
void DpctGlobalInfo::printSubGroup(llvm::raw_ostream &OS, const Stmt *S,
                                   const FunctionDecl *FD) {
  FreeQueriesInfo::printImmediateText(
      OS, S, FD, FreeQueriesInfo::FreeQueriesKind::SubGroup);
}
std::string DpctGlobalInfo::getSubGroup(const Stmt *S, const FunctionDecl *FD) {
  return buildStringFromPrinter(DpctGlobalInfo::printSubGroup, S, FD);
}
std::string DpctGlobalInfo::getDefaultQueue(const Stmt *S) {
  auto Idx = getPlaceholderIdx(S);
  if (!Idx) {
    Idx = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Idx, S, HelperFuncType::HFT_DefaultQueue);
  }

  return buildString(RegexPrefix, 'Q', Idx, RegexSuffix);
}
const std::string &DpctGlobalInfo::getDefaultQueueFreeFuncCall() {
  static const std::string DefaultQueueFreeFuncCall = [&]() {
    if (auto Iter = MapNames::CustomHelperFunctionMap.find(getQueueKind());
        Iter != MapNames::CustomHelperFunctionMap.end()) {
      return Iter->second;
    }
    return MapNames::getDpctNamespace() + "get_" +
           getDefaultQueueMemFuncName() + "()";
  }();
  return DefaultQueueFreeFuncCall;
}
const std::string &DpctGlobalInfo::getDefaultQueueMemFuncName() {
  static const std::string DefaultQueueMemFuncName = [&]() {
    if (DpctGlobalInfo::useSYCLCompat())
      return "default_queue";
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
      return "out_of_order_queue";
    return "in_order_queue";
  }();
  return DefaultQueueMemFuncName;
}
void DpctGlobalInfo::setContext(ASTContext &C) {
  Context = &C;
  SM = &(Context->getSourceManager());
  FM = &(SM->getFileManager());
  Context->getParentMapContext().setTraversalKind(TK_AsIs);
}
void DpctGlobalInfo::insertKCIndentWidth(unsigned int W) {
  auto Iter = KCIndentWidthMap.find(W);
  if (Iter != KCIndentWidthMap.end())
    Iter->second++;
  else
    KCIndentWidthMap.insert(std::make_pair(W, 1));
}
unsigned int DpctGlobalInfo::getKCIndentWidth() {
  if (KCIndentWidthMap.empty())
    return DpctGlobalInfo::getCodeFormatStyle().IndentWidth;

  std::multimap<unsigned int, unsigned int, std::greater<unsigned int>>
      OccuranceIndentWidthMap;
  for (const auto &I : KCIndentWidthMap)
    OccuranceIndentWidthMap.insert(std::make_pair(I.second, I.first));

  return OccuranceIndentWidthMap.begin()->second;
}
void DpctGlobalInfo::setExcludePath(std::vector<std::string> ExcludePathVec) {
  if (ExcludePathVec.empty()) {
    return;
  }
  std::set<std::string> ProcessedPath;
  for (auto Itr = ExcludePathVec.begin(); Itr != ExcludePathVec.end(); Itr++) {
    if ((*Itr).empty()) {
      continue;
    }
    clang::tooling::UnifiedPath PathBuf = *Itr;
    if (PathBuf.getCanonicalPath().empty()) {
      clang::dpct::PrintMsg("Note: Path " + PathBuf.getPath().str() +
                            " is invalid and will be ignored by option "
                            "--in-root-exclude.\n");
      continue;
    }
    if (ProcessedPath.count(*Itr)) {
      continue;
    }
    ProcessedPath.insert(*Itr);
    bool IsDirectory;
    if ((IsDirectory = llvm::sys::fs::is_directory(*Itr)) ||
        llvm::sys::fs::is_regular_file(*Itr) ||
        llvm::sys::fs::is_symlink_file(*Itr)) {
      if (!isChildOrSamePath(InRoot, *Itr)) {
        clang::dpct::PrintMsg("Note: Path " + PathBuf.getCanonicalPath().str() +
                              " is not in --in-root directory and will be "
                              "ignored by --in-root-exclude.\n");
      } else {
        bool IsNeedInsert = true;
        for (auto EP_Itr = ExcludePath.begin(); EP_Itr != ExcludePath.end();) {
          if ((EP_Itr->first == *Itr) ||
              (EP_Itr->second && isChildOrSamePath(EP_Itr->first, *Itr))) {
            // 1. If current path is child or same path of previous path,
            //    then we skip it.
            IsNeedInsert = false;
            break;
          } else if (IsDirectory && isChildOrSamePath(*Itr, EP_Itr->first)) {
            // 2. If previous path is child of current path, then
            //    we delete previous path.
            EP_Itr = ExcludePath.erase(EP_Itr);
          } else {
            EP_Itr++;
          }
        }
        if (IsNeedInsert) {
          ExcludePath.insert({*Itr, IsDirectory});
        }
      }
    } else {
      clang::dpct::PrintMsg("Note: Path " + PathBuf.getCanonicalPath().str() +
                            " is invalid and will be ignored by option "
                            "--in-root-exclude.\n");
    }
  }
}
int DpctGlobalInfo::getSuffixIndexInitValue(std::string FileNameAndOffset) {
  auto Res = LocationInitIndexMap.find(FileNameAndOffset);
  if (Res == LocationInitIndexMap.end()) {
    LocationInitIndexMap.insert(
        std::make_pair(FileNameAndOffset, CurrentMaxIndex + 1));
    return CurrentMaxIndex + 1;
  } else {
    return Res->second;
  }
}

int DpctGlobalInfo::getSuffixIndexInRuleThenInc() {
  int Res = CurrentIndexInRule;
  if (CurrentMaxIndex < Res)
    CurrentMaxIndex = Res;
  CurrentIndexInRule++;
  return Res;
}
int DpctGlobalInfo::getSuffixIndexGlobalThenInc() {
  int Res = CurrentMaxIndex;
  CurrentMaxIndex++;
  return Res;
}
std::string DpctGlobalInfo::getStringForRegexReplacement(StringRef MatchedStr) {
  unsigned Index = 0;
  char Method = MatchedStr[RegexPrefix.length()];
  bool HasError =
      MatchedStr.substr(RegexPrefix.length() + 1).consumeInteger(10, Index);
  assert(!HasError && "Must consume an integer");
  (void)HasError;
  // D: device, used for pretty code
  // Q: queue, used for pretty code
  // R: range dim, used for built-in variables (threadIdx.x,...) migration
  // G: range dim, used for cg::thread_block migration
  // C: range dim, used for cub block migration
  // F: free queries function migration, such as this_work_item::get_nd_item,
  // this_work_item::get_work_group, this_work_item::get_sub_group.
  // E: extension, used for c source file migration
  // P: profiling enable or disable for time measurement.
  // Z: queue pointer.
  switch (Method) {
  case 'R':
    if (DpctGlobalInfo::getAssumedNDRangeDim() == 1) {
      if (auto DFI = getCudaKernelDimDFI(Index)) {
        auto Ptr =
            MemVarMap::getHeadWithoutPathCompression(&(DFI->getVarMap()));
        if (Ptr && Ptr->Dim == 1) {
          return "0";
        }
      }
    }
    return "2";
  case 'G':
    if (DpctGlobalInfo::getAssumedNDRangeDim() == 1) {
      if (auto DFI = getCudaKernelDimDFI(Index)) {
        auto Ptr =
            MemVarMap::getHeadWithoutPathCompression(&(DFI->getVarMap()));
        if (Ptr && Ptr->Dim == 1) {
          return "1";
        }
      }
    }
    return "3";
  case 'C':
    if (DpctGlobalInfo::getAssumedNDRangeDim() == 1) {
      return std::to_string(DpctGlobalInfo::getInstance()
                                .getCubPlaceholderIndexMap()[Index]
                                ->getVarMap()
                                .getHeadNodeDim());
    }
    return "3";
  case 'D':
    return getStringForRegexDefaultQueueAndDevice(
        HelperFuncType::HFT_CurrentDevice, Index);
  case 'Q':
    return getStringForRegexDefaultQueueAndDevice(
        HelperFuncType::HFT_DefaultQueue, Index);
  case 'Z':
    return getStringForRegexDefaultQueueAndDevice(
        HelperFuncType::HFT_DefaultQueuePtr, Index);
  case 'E': {
    auto &Vec = DpctGlobalInfo::getInstance().getCSourceFileInfo();
    return Vec[Index]->hasCUDASyntax()
               ? ("c" + DpctGlobalInfo::getSYCLSourceExtension())
               : "c";
  }
  case 'P': {
    std::string ReplStr;
    if (DpctGlobalInfo::getEnablepProfilingFlag())
      ReplStr = (DpctGlobalInfo::useSYCLCompat()
                     ? std::string("#define SYCLCOMPAT_PROFILING_ENABLED")
                     : std::string("#define DPCT_PROFILING_ENABLED")) +
                getNL();

    return ReplStr;
  }
  case 'I': {
    std::string ReplStr;
    if (DpctGlobalInfo::getEnablepProfilingFlag())
      ReplStr = ", " + MapNames::getClNamespace() +
                "property::queue::enable_profiling()";

    return ReplStr;
  }
  case FreeQueriesInfo::FreeQueriesRegexCh:
    return FreeQueriesInfo::getReplaceString(Index);
  default:
    clang::dpct::DpctDebugs() << "[char] Unexpected value: " << Method << "\n";
    assert(0);
    return MatchedStr.str();
  }
}
std::optional<clang::tooling::UnifiedPath>
DpctGlobalInfo::getAbsolutePath(FileID ID) {
  assert(SM && "SourceManager must be initialized");
  if (auto FileEntryRef = SM->getFileEntryRefForID(ID))
    return getAbsolutePath(*FileEntryRef);
  return std::nullopt;
}
std::optional<clang::tooling::UnifiedPath>
DpctGlobalInfo::getAbsolutePath(FileEntryRef File) {
  if (auto RealPath = File.getFileEntry().tryGetRealPathName();
      !RealPath.empty())
    return clang::tooling::UnifiedPath(RealPath);

  llvm::SmallString<512> FilePathAbs(File.getName());
  SM->getFileManager().makeAbsolutePath(FilePathAbs);
  return clang::tooling::UnifiedPath(FilePathAbs);
}
std::pair<clang::tooling::UnifiedPath, unsigned>
DpctGlobalInfo::getLocInfo(SourceLocation Loc, bool *IsInvalid) {
  if (SM->isMacroArgExpansion(Loc)) {
    Loc = SM->getImmediateSpellingLoc(Loc);
  }
  auto LocInfo = SM->getDecomposedLoc(SM->getExpansionLoc(Loc));
  auto AbsPath = getAbsolutePath(LocInfo.first);
  if (AbsPath)
    return std::make_pair(AbsPath.value(), LocInfo.second);
  if (IsInvalid)
    *IsInvalid = true;
  return std::make_pair(clang::tooling::UnifiedPath(), 0);
}
std::string DpctGlobalInfo::getTypeName(QualType QT, const ASTContext &Context,
                                        bool SuppressScope) {
  if (auto ET = QT->getAs<ElaboratedType>()) {
    if (ET->getQualifier())
      QT = Context.getElaboratedType(ElaboratedTypeKeyword::None,
                                     ET->getQualifier(), ET->getNamedType(),
                                     ET->getOwnedTagDecl());
    else
      QT = ET->getNamedType();
  }
  auto TT = QT->getAs<TypedefType>();
  if (TT && SuppressScope) {
    return TT->getDecl()->getNameAsString();
  }
  auto PP = Context.getPrintingPolicy();
  PP.SuppressScope = SuppressScope;
  PP.SuppressTagKeyword = true;
  return QT.getAsString(PP);
}
std::string DpctGlobalInfo::getReplacedTypeName(QualType QT,
                                                const ASTContext &Context,
                                                bool SuppressScope) {
  if (!QT.isNull())
    if (const auto *AT = dyn_cast<AutoType>(QT.getTypePtr())) {
      QT = AT->getDeducedType();
      if (QT.isNull()) {
        return "";
      }
    }
  std::string MigratedTypeStr;
  setGetReplacedNamePtr(&getReplacedName);
  llvm::raw_string_ostream OS(MigratedTypeStr);
  clang::PrintingPolicy PP =
      clang::PrintingPolicy(DpctGlobalInfo::getContext().getLangOpts());
  PP.SuppressScope = SuppressScope;
  QT.print(OS, PP);
  OS.flush();
  setGetReplacedNamePtr(nullptr);
  return getFinalCastTypeNameStr(MigratedTypeStr);
}
std::string DpctGlobalInfo::getOriginalTypeName(QualType QT) {
  std::string OriginalTypeStr;
  llvm::raw_string_ostream OS(OriginalTypeStr);
  clang::PrintingPolicy PP =
      clang::PrintingPolicy(DpctGlobalInfo::getContext().getLangOpts());
  QT.print(OS, PP);
  OS.flush();
  return OriginalTypeStr;
}
std::shared_ptr<DeviceFunctionDecl> DpctGlobalInfo::insertDeviceFunctionDecl(
    const FunctionDecl *Specialization, const FunctionTypeLoc &FTL,
    const ParsedAttributes &Attrs, const TemplateArgumentListInfo &TAList) {
  auto LocInfo = getLocInfo(FTL);
  return insertFile(LocInfo.first)
      ->insertNode<ExplicitInstantiationDecl, DeviceFunctionDecl>(
          LocInfo.second, FTL, Attrs, Specialization, TAList);
}

void DpctGlobalInfo::buildKernelInfo() {
  for (auto &File : FileMap)
    File.second->buildKernelInfo();

  // Construct a union-find set for all the instances of MemVarMap in
  // DeviceFunctionInfo. During the traversal of the call-graph, do union
  // operation if caller and callee both need item variable, then after the
  // traversal, all MemVarMap instance which need item are divided into
  // some groups. Among different groups, there is no call relationship. If
  // kernel-call is 3D, then set its head's dim to 3D. When generating
  // replacements, find current nodes' head to decide to use which dim.

  // Below 4 for-loop cannot be merged.
  // The later loop depends on the info generated by the previous loop.
  // Now we consider two links: the call-chain and the macro spelling loc
  // link Since the macro spelling loc may link a global func from a device
  // func, we cannot merge set dim into the second loop. Because global func
  // is the first level function in the buildUnionFindSet(), if it is
  // visited from previous device func, there is no chance to propagate its
  // correct dim value (there is no upper level func call to global func and
  // then it will be skipped).
  for (auto &File : FileMap)
    File.second->setKernelCallDim();
  for (auto &File : FileMap)
    File.second->setKernelDim();
  for (auto &File : FileMap)
    File.second->buildUnionFindSet();
  for (auto &File : FileMap)
    File.second->buildUnionFindSetForUncalledFunc();
}
void DpctGlobalInfo::buildReplacements() {
  // add PriorityRepl into ReplMap and execute related action, e.g.,
  // request feature or emit warning.
  for (auto &ReplInfo : PriorityReplInfoMap) {
    for (auto &Repl : ReplInfo.second->Repls) {
      addReplacement(Repl);
    }
    for (auto &Action : ReplInfo.second->RelatedAction) {
      Action();
    }
  }

  for (auto &File : FileMap)
    File.second->buildReplacements();

  // All cases of replacing placeholders:
  // dev_count  queue_count  dev_decl            queue_decl
  // 0          1            /                   get_default_queue
  // 1          0            get_current_device  /
  // 1          1            get_current_device  get_default_queue
  // 2          1            dev_ct1             get_default_queue
  // 1          2            dev_ct1             q_ct1
  // >=2        >=2          dev_ct1             q_ct1
  bool NeedDpctHelpFunc = DpctGlobalInfo::needDpctDeviceExt() ||
                          TempVariableDeclCounterMap.size() > 1 ||
                          DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None;
  unsigned int IndentLen = 2;
  if (getGuessIndentWidthMatcherFlag())
    IndentLen = getIndentWidth();
  std::string IndentStr = std::string(IndentLen, ' ');
  std::string DevDeclStr = getNL() + IndentStr;
  llvm::raw_string_ostream DevDecl(DevDeclStr);
  std::string QDeclStr =
      getNL() + IndentStr + MapNames::getClNamespace() + "queue ";
  llvm::raw_string_ostream QDecl(QDeclStr);
  if (NeedDpctHelpFunc) {
    DevDecl << MapNames::getDpctNamespace()
            << "device_ext &dev_ct1 = " << MapNames::getDpctNamespace()
            << "get_current_device();";
    QDecl << "&q_ct1 = ";
    if (DpctGlobalInfo::useSYCLCompat())
      QDecl << '*';
    QDecl << "dev_ct1." << DpctGlobalInfo::getDefaultQueueMemFuncName()
          << "();";
  } else {
    DevDecl << MapNames::getClNamespace() + "device dev_ct1;";
    // Now the UsmLevel must not be UL_None here.
    QDecl << "q_ct1(dev_ct1, " << MapNames::getClNamespace() << "property_list{"
          << MapNames::getClNamespace() << "property::queue::in_order()";

    // replaced to insert of "property::queue::enable_profiling()" or not in
    // the post replacement.
    QDecl << "{{NEEDREPLACEI0}}";
    QDecl << "});";
  }

  for (auto &Counter : TempVariableDeclCounterMap) {
    if (DpctGlobalInfo::useNoQueueDevice()) {
      Counter.second.PlaceholderStr[1] = DpctGlobalInfo::getGlobalQueueName();
      Counter.second.PlaceholderStr[2] = DpctGlobalInfo::getGlobalDeviceName();
      Counter.second.PlaceholderStr[3] = "&" + DpctGlobalInfo::getGlobalQueueName();
      // Need not insert q_ct1 and dev_ct1 declrations and request feature.
      continue;
    }
    const auto ColonPos = Counter.first.find_last_of(':');
    const auto DeclLocFile = Counter.first.substr(0, ColonPos);
    const auto DeclLocOffset = std::stoi(Counter.first.substr(ColonPos + 1));
    if (!getDeviceChangedFlag() && getUsingDRYPattern()) {
      if (Counter.second.CurrentDeviceCounter > 1 ||
          Counter.second.DefaultQueueCounter > 1) {
        Counter.second.PlaceholderStr[2] = "dev_ct1";
        getInstance().addReplacement(std::make_shared<ExtReplacement>(
            DeclLocFile, DeclLocOffset, 0, DevDecl.str(), nullptr));
        if (Counter.second.DefaultQueueCounter > 1 || !NeedDpctHelpFunc) {
          Counter.second.PlaceholderStr[1] = "q_ct1";
          Counter.second.PlaceholderStr[3] = "&q_ct1";
          getInstance().addReplacement(std::make_shared<ExtReplacement>(
              DeclLocFile, DeclLocOffset, 0, QDecl.str(), nullptr));
        }
      }
    }
    if (Counter.second.CurrentDeviceCounter > 0 ||
        Counter.second.DefaultQueueCounter > 1)
      requestFeature(HelperFeatureEnum::device_ext);
    if (Counter.second.DefaultQueueCounter > 0)
      requestFeature(HelperFeatureEnum::device_ext);
  }
}
void DpctGlobalInfo::processCudaArchMacro() {
  // process __CUDA_ARCH__ macro
  auto &ReplMap = DpctGlobalInfo::getInstance().getCudaArchMacroReplMap();
  // process __CUDA_ARCH__ macro of directive condition in generated host code:
  // if __CUDA_ARCH__ > 800      -->  if !DPCT_COMPATIBILITY_TEMP
  // if defined(__CUDA_ARCH__)   -->  if !defined(DPCT_COMPATIBILITY_TEMP)
  // if !defined(__CUDA_ARCH__)  -->  if defined(DPCT_COMPATIBILITY_TEMP)
  auto processIfMacro = [&](std::shared_ptr<ExtReplacement> Repl,
                            DirectiveInfo DI) {
    std::string FilePath = Repl->getFilePath().str();
    auto &CudaArchDefinedMap =
        DpctGlobalInfo::getInstance().getCudaArchDefinedMap()[FilePath];
    if (CudaArchDefinedMap.count((*Repl).getOffset())) {
      unsigned int ExclamationOffset =
          CudaArchDefinedMap[(*Repl).getOffset()] - DI.ConditionLoc - 1;
      if (ExclamationOffset <= (DI.Condition.length() - 1) &&
          DI.Condition[ExclamationOffset] == '!') {
        addReplacement(std::make_shared<ExtReplacement>(
            FilePath, CudaArchDefinedMap[(*Repl).getOffset()] - 1, 1, "",
            nullptr));
      } else {
        addReplacement(std::make_shared<ExtReplacement>(
            FilePath, CudaArchDefinedMap[(*Repl).getOffset()], 0, "!",
            nullptr));
      }
    } else {
      if (useSYCLCompat())
        (*Repl).setReplacementText("!SYCLCOMPAT_COMPATIBILITY_TEMP");
      else
        (*Repl).setReplacementText("!DPCT_COMPATIBILITY_TEMP");
    }
  };

  for (auto Iter = ReplMap.begin(); Iter != ReplMap.end();) {
    auto Repl = Iter->second;
    unsigned CudaArchOffset = Repl->getOffset();
    std::string FilePath = Repl->getFilePath().str();
    auto &CudaArchPPInfosMap =
        DpctGlobalInfo::getInstance().getCudaArchPPInfoMap()[FilePath];
    bool DirectiveReserved = true;
    for (auto Iterator = CudaArchPPInfosMap.begin();
         Iterator != CudaArchPPInfosMap.end(); Iterator++) {
      auto Info = Iterator->second;
      if (!Info.isInHDFunc)
        continue;
      unsigned Pos_a = 0, Len_a = 0, Pos_b = 0, Len_b = 0,
               Round = DpctGlobalInfo::getRunRound();
      if (CudaArchOffset >= Info.IfInfo.ConditionLoc &&
          CudaArchOffset <=
              Info.IfInfo.ConditionLoc + Info.IfInfo.Condition.length()) {
        if (Info.ElInfo.size() == 0) {
          if (Info.ElseInfo.DirectiveLoc == 0) {
            //  Remove unnecessary condition branch, as code is absolutely dead
            //  or active Origin Code:
            //  ...
            //  #ifdef __CUDA_ARCH__ / #if defined(__CUDA_ARCH__) / #if
            //  __CUDA_ARCH__ / #ifndef __CUDA_ARCH__ / #if
            //  !defined(__CUDA_ARCH__)
            //    host_code/device code;
            //  #endif
            //  ...
            //
            //  After Migration:
            //  Round = 0 for device code, final migration code:
            //    ...
            //    empty/device code;
            //    ...
            //  Round = 1 for host code, final migration code:
            //    ...
            //    host_code/empty;
            //    ...
            if ((Info.DT == IfType::IT_Ifdef && Round == 1) ||
                (Info.DT == IfType::IT_Ifndef && Round == 0) ||
                (Info.DT == IfType::IT_If && Round == 1 &&
                 (Info.IfInfo.Condition == "defined(__CUDA_ARCH__)" ||
                  Info.IfInfo.Condition == "__CUDA_ARCH__")) ||
                (Info.DT == IfType::IT_If && Round == 0 &&
                 Info.IfInfo.Condition == "!defined(__CUDA_ARCH__)")) {
              Pos_a = Info.IfInfo.NumberSignLoc;
              if (Pos_a != UINT_MAX) {
                Len_a =
                    Info.EndInfo.DirectiveLoc - Pos_a + 5 /*length of endif*/;
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_a, Len_a, "", nullptr));
                DirectiveReserved = false;
              }
            } else if ((Info.DT == IfType::IT_Ifdef && Round == 0) ||
                       (Info.DT == IfType::IT_Ifndef && Round == 1) ||
                       (Info.DT == IfType::IT_If && Round == 0 &&
                        (Info.IfInfo.Condition == "defined(__CUDA_ARCH__)" ||
                         Info.IfInfo.Condition == "__CUDA_ARCH__")) ||
                       (Info.DT == IfType::IT_If && Round == 1 &&
                        Info.IfInfo.Condition == "!defined(__CUDA_ARCH__)")) {
              Pos_a = Info.IfInfo.NumberSignLoc;
              Pos_b = Info.EndInfo.NumberSignLoc;
              if (Pos_a != UINT_MAX && Pos_b != UINT_MAX) {
                Len_a = Info.IfInfo.ConditionLoc +
                        Info.IfInfo.Condition.length() - Pos_a;
                Len_b =
                    Info.EndInfo.DirectiveLoc - Pos_b + 5 /*length of endif*/;
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_a, Len_a, "", nullptr));
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_b, Len_b, "", nullptr));
                DirectiveReserved = false;
              }
            }
          } else {
            //  Remove conditional branch, as code is absolutely dead or active
            //  Origin Code:
            //  ...
            //  #ifdef __CUDA_ARCH__ / #if defined(__CUDA_ARCH__) / #if
            //  __CUDA_ARCH__ / #ifndef __CUDA_ARCH / #if
            //  !defined(__CUDA_ARCH__)
            //    host_code/device_code;
            //  #else
            //    device_code/host_code;
            //  #endif
            //  ...
            //
            //  After Migration:
            //  Round = 0 for device code, final migration code:
            //    ...
            //    device_code;
            //    ...
            //  Round = 1 for host code, final migration code:
            //    ...
            //    host_code;
            //    ...
            if ((Info.DT == IfType::IT_Ifdef && Round == 1) ||
                (Info.DT == IfType::IT_Ifndef && Round == 0) ||
                (Info.DT == IfType::IT_If && Round == 1 &&
                 (Info.IfInfo.Condition == "defined(__CUDA_ARCH__)" ||
                  Info.IfInfo.Condition == "__CUDA_ARCH__")) ||
                (Info.DT == IfType::IT_If && Round == 0 &&
                 Info.IfInfo.Condition == "!defined(__CUDA_ARCH__)")) {
              Pos_a = Info.IfInfo.NumberSignLoc;
              Pos_b = Info.EndInfo.NumberSignLoc;
              if (Pos_a != UINT_MAX && Pos_b != UINT_MAX) {
                Len_a =
                    Info.ElseInfo.DirectiveLoc - Pos_a + 4 /*length of else*/;
                Len_b =
                    Info.EndInfo.DirectiveLoc - Pos_b + 5 /*length of endif*/;
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_a, Len_a, "", nullptr));
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_b, Len_b, "", nullptr));
                DirectiveReserved = false;
              }
            } else if ((Info.DT == IfType::IT_Ifdef && Round == 0) ||
                       (Info.DT == IfType::IT_Ifndef && Round == 1) ||
                       (Info.DT == IfType::IT_If && Round == 0 &&
                        (Info.IfInfo.Condition == "defined(__CUDA_ARCH__)" ||
                         Info.IfInfo.Condition == "__CUDA_ARCH__")) ||
                       (Info.DT == IfType::IT_If && Round == 1 &&
                        Info.IfInfo.Condition == "!defined(__CUDA_ARCH__)")) {
              Pos_a = Info.IfInfo.NumberSignLoc;
              Pos_b = Info.ElseInfo.NumberSignLoc;
              if (Pos_a != UINT_MAX && Pos_b != UINT_MAX) {
                Len_a = Info.IfInfo.ConditionLoc +
                        Info.IfInfo.Condition.length() - Pos_a;
                Len_b =
                    Info.EndInfo.DirectiveLoc - Pos_b + 5 /*length of endif*/;
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_a, Len_a, "", nullptr));
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_b, Len_b, "", nullptr));
                DirectiveReserved = false;
              }
            }
          }
        }
        //  if directive in which __CUDA_ARCH__ inside was reserved, then we
        //  need process this directive for generated host code:
        //  ifndef__CUDA_ARCH__ --> ifdef DPCT_COMPATIBILITY_TEMP
        //  ifdef __CUDA_ARCH__ --> ifndef DPCT_COMPATIBILITY_TEMP
        if (DirectiveReserved && Round == 1) {
          if (Info.DT == IfType::IT_Ifdef) {
            Pos_a = Info.IfInfo.DirectiveLoc;
            Len_a = 5 /*length of ifdef*/;
            addReplacement(std::make_shared<ExtReplacement>(
                FilePath, Pos_a, Len_a, "ifndef", nullptr));
          } else if (Info.DT == IfType::IT_Ifndef) {
            Pos_a = Info.IfInfo.DirectiveLoc;
            Len_a = 6 /*length of ifndef*/;
            addReplacement(std::make_shared<ExtReplacement>(
                FilePath, Pos_a, Len_a, "ifdef", nullptr));
          } else if (Info.DT == IfType::IT_If) {
            processIfMacro(Repl, Info.IfInfo);
          }
        }
        break;
      } else {
        //  Info.ElInfo.size() == 0
        if (Round == 0)
          continue;
        for (auto &ElifInfoPair : Info.ElInfo) {
          auto &ElifInfo = ElifInfoPair.second;
          if (CudaArchOffset >= ElifInfo.ConditionLoc &&
              CudaArchOffset <=
                  ElifInfo.ConditionLoc + ElifInfo.Condition.length()) {
            processIfMacro(Repl, ElifInfo);
            break;
          }
        }
      }
    }
    if (DirectiveReserved) {
      addReplacement(Repl);
      Iter = ReplMap.erase(Iter);
    } else {
      Iter++;
    }
  }
}

void DpctGlobalInfo::generateHostCode(tooling::Replacements &ProcessedReplList,
                                      HostDeviceFuncLocInfo &Info, unsigned ID) {
  std::vector<std::shared_ptr<ExtReplacement>> ExtraRepl;

  unsigned int Pos, Len;
  std::string OriginText = Info.FuncContentCache;
  StringRef SR(OriginText);
  llvm::RewriteBuffer RB;
  RB.Initialize(SR.begin(), SR.end());
  for (const auto &R : ProcessedReplList) {
    unsigned ROffset = R.getOffset();
    if (ROffset >= Info.FuncStartOffset && ROffset <= Info.FuncEndOffset) {
      Pos = ROffset - Info.FuncStartOffset;
      Len = R.getLength();
      RB.ReplaceText(Pos, Len, R.getReplacementText());
    }
  }
  Pos = Info.FuncNameOffset - Info.FuncStartOffset;
  Len = 0;
  RB.ReplaceText(Pos, Len, "_host_ct" + std::to_string(ID));
  std::string DefResult;
  llvm::raw_string_ostream DefStream(DefResult);
  RB.write(DefStream);
  std::string NewFuncBody = DefStream.str();
  auto R = std::make_shared<ExtReplacement>(
      Info.FilePath, Info.FuncEndOffset + 1, 0, getNL() + NewFuncBody, nullptr);
  ExtraRepl.emplace_back(R);

  for (auto &R : ExtraRepl) {
    auto &FileReplCache = DpctGlobalInfo::getFileReplCache();
    FileReplCache[R->getFilePath().str()].second->addReplacement(R);
  }
  return;
}
void DpctGlobalInfo::postProcess() {
  auto &MSMap = DpctGlobalInfo::getMainSourceFileMap();
  bool isFirstPass = !DpctGlobalInfo::getRunRound();
  processCudaArchMacro();
  for (auto &Element : HostDeviceFuncInfoMap) {
    auto &Info = Element.second;
    if (Info.isDefInserted) {
      Info.needGenerateHostCode = true;
      if (Info.PostFixId == -1) {
        Info.PostFixId = HostDeviceFuncInfo::MaxId++;
      }
      for (auto &E : Info.LocInfos) {
        auto &LocInfo = E.second;
        if (isFirstPass) {
          auto &MSFiles = MSMap[LocInfo.FilePath];
          for (auto &File : MSFiles) {
            ReProcessFile.emplace(File);
          }
        }
        if (LocInfo.Type == HDFuncInfoType::HDFI_Call && !LocInfo.Processed) {
          if (LocInfo.CalledByHostDeviceFunction && isFirstPass) {
            LocInfo.Processed = true;
            continue;
          }
          LocInfo.Processed = true;
          auto R = std::make_shared<ExtReplacement>(
              LocInfo.FilePath, LocInfo.FuncEndOffset, 0,
              "_host_ct" + std::to_string(Info.PostFixId), nullptr);
          addReplacement(R);
          if (!isFirstPass) {
            auto &FileReplCache = DpctGlobalInfo::getFileReplCache();
            FileReplCache[R->getFilePath().str()].second->addReplacement(R);
          }
        }
      }
    }
  }
  if (!ReProcessFile.empty() && isFirstPass) {
    DpctGlobalInfo::setNeedRunAgain(true);
  }
  for (const auto &R : IncludeMapSet) {
    if (auto F = findFile(R.first)) {
      if (!F->getReplsSYCL()->empty()) {
        addReplacement(R.second);
      }
    }
  }
  for (auto &File : FileMap) {
    auto &S = File.second->getConstantMacroTMSet();
    auto &Map = DpctGlobalInfo::getConstantReplProcessedFlagMap();
    for (auto &E : S) {
      if (!Map[E]) {
        addReplacement(E->getReplacement(DpctGlobalInfo::getContext()));
      }
    }
    File.second->postProcess();
  }
  if (!isFirstPass) {
    for (auto &Element : HostDeviceFuncInfoMap) {
      auto &Info = Element.second;
      if (Info.needGenerateHostCode) {
        for (auto &E : Info.LocInfos) {
          auto &LocInfo = E.second;
          if (LocInfo.Type == HDFuncInfoType::HDFI_Call) {
            continue;
          }
          tooling::Replacements ReplLists;
          FileMap[LocInfo.FilePath]->getReplsSYCL()->emplaceIntoReplSet(
              ReplLists);
          generateHostCode(ReplLists, LocInfo, Info.PostFixId);
        }
      }
    }
  }
}
void DpctGlobalInfo::emplaceReplacements(ReplTy &ReplSetsCUDA /*out*/,
                                         ReplTy &ReplSetsSYCL /*out*/) {
  if (DpctGlobalInfo::isNeedRunAgain())
    return;
  for (auto &FileRepl : FileReplCache) {
    FileRepl.second.first->emplaceIntoReplSet(
        ReplSetsCUDA[FileRepl.first.getCanonicalPath().str()]);
    FileRepl.second.second->emplaceIntoReplSet(
        ReplSetsSYCL[FileRepl.first.getCanonicalPath().str()]);
  }
}
std::shared_ptr<KernelCallExpr>
DpctGlobalInfo::buildLaunchKernelInfo(const CallExpr *LaunchKernelCall,
                                      bool IsAssigned) {
  auto DefRange = getDefinitionRange(LaunchKernelCall->getBeginLoc(),
                                     LaunchKernelCall->getEndLoc());
  auto LocInfo = getLocInfo(DefRange.getBegin());
  auto FileInfo = insertFile(LocInfo.first);
  if (FileInfo->findNode<KernelCallExpr>(LocInfo.second))
    return std::shared_ptr<KernelCallExpr>();

  auto KernelInfo = KernelCallExpr::buildFromCudaLaunchKernel(
      LocInfo, LaunchKernelCall, IsAssigned);
  if (KernelInfo) {
    FileInfo->insertNode(LocInfo.second, KernelInfo);
  } else {
    auto FuncName = LaunchKernelCall->getDirectCallee()
                        ->getNameInfo()
                        .getName()
                        .getAsString();
    DiagnosticsUtils::report(LocInfo.first, LocInfo.second,
                             Diagnostics::API_NOT_MIGRATED, true, false,
                             FuncName);
  }

  return KernelInfo;
}
void DpctGlobalInfo::insertCudaMalloc(const CallExpr *CE) {
  if (auto MallocVar = CudaMallocInfo::getMallocVar(CE->getArg(0)))
    insertCudaMallocInfo(MallocVar)->setSizeExpr(CE->getArg(1));
}
void DpctGlobalInfo::insertCublasAlloc(const CallExpr *CE) {
  if (auto MallocVar = CudaMallocInfo::getMallocVar(CE->getArg(2)))
    insertCudaMallocInfo(MallocVar)->setSizeExpr(CE->getArg(0), CE->getArg(1));
}
std::shared_ptr<CudaMallocInfo> DpctGlobalInfo::findCudaMalloc(const Expr *E) {
  if (auto Src = CudaMallocInfo::getMallocVar(E))
    return findCudaMallocInfo(Src);
  return std::shared_ptr<CudaMallocInfo>();
}
void DpctGlobalInfo::insertReplInfoFromYAMLToFileInfo(
    const clang::tooling::UnifiedPath &FilePath,
    std::shared_ptr<tooling::TranslationUnitReplacements> TUR) {
  auto FileInfo = insertFile(FilePath);
  if (FileInfo->PreviousTUReplFromYAML == nullptr)
    FileInfo->PreviousTUReplFromYAML = TUR;
}
std::shared_ptr<tooling::TranslationUnitReplacements>
DpctGlobalInfo::getReplInfoFromYAMLSavedInFileInfo(
    clang::tooling::UnifiedPath FilePath) {
  auto FileInfo = findObject(FileMap, FilePath);
  if (FileInfo)
    return FileInfo->PreviousTUReplFromYAML;
  else
    return nullptr;
}
void DpctGlobalInfo::insertEventSyncTypeInfo(
    const std::shared_ptr<clang::dpct::ExtReplacement> Repl, bool NeedReport,
    bool IsAssigned) {
  std::string FilePath = Repl->getFilePath().str();
  unsigned int Offset = Repl->getOffset();
  unsigned int Length = Repl->getLength();
  const std::string ReplText = Repl->getReplacementText().str();
  auto FileInfo = insertFile(FilePath);
  auto &M = FileInfo->getEventSyncTypeMap();
  auto Iter = M.find(Offset);
  if (Iter == M.end()) {
    M.insert(std::make_pair(
        Offset, EventSyncTypeInfo(Length, ReplText, NeedReport, IsAssigned)));
  } else {
    Iter->second.IsAssigned = IsAssigned;
  }
}
void DpctGlobalInfo::updateEventSyncTypeInfo(
    const std::shared_ptr<clang::dpct::ExtReplacement> Repl) {
  std::string FilePath = Repl->getFilePath().str();
  unsigned int Offset = Repl->getOffset();
  unsigned int Length = Repl->getLength();
  const std::string ReplText = Repl->getReplacementText().str();
  auto FileInfo = insertFile(FilePath);
  auto &M = FileInfo->getEventSyncTypeMap();
  auto Iter = M.find(Offset);
  if (Iter != M.end()) {
    Iter->second.ReplText = ReplText;
    Iter->second.NeedReport = false;
  } else {
    M.insert(std::make_pair(Offset,
                            EventSyncTypeInfo(Length, ReplText, false, false)));
  }
}
void DpctGlobalInfo::insertTimeStubTypeInfo(
    const std::shared_ptr<clang::dpct::ExtReplacement> ReplWithSB,
    const std::shared_ptr<clang::dpct::ExtReplacement> ReplWithoutSB) {
  std::string FilePath = ReplWithSB->getFilePath().str();
  unsigned int Offset = ReplWithSB->getOffset();
  unsigned int Length = ReplWithSB->getLength();
  std::string StrWithSubmitBarrier = ReplWithSB->getReplacementText().str();
  std::string StrWithoutSubmitBarrier =
      ReplWithoutSB->getReplacementText().str();
  auto FileInfo = insertFile(FilePath);
  auto &M = FileInfo->getTimeStubTypeMap();
  M.insert(std::make_pair(Offset, TimeStubTypeInfo(Length, StrWithSubmitBarrier,
                                                   StrWithoutSubmitBarrier)));
}
void DpctGlobalInfo::updateTimeStubTypeInfo(SourceLocation BeginLoc,
                                            SourceLocation EndLoc) {
  auto LocInfo = getLocInfo(BeginLoc);
  auto FileInfo = insertFile(LocInfo.first);
  size_t Begin = getLocInfo(BeginLoc).second;
  size_t End = getLocInfo(EndLoc).second;
  auto &TimeStubBounds = FileInfo->getTimeStubBounds();
  TimeStubBounds.push_back(std::make_pair(Begin, End));
}
void DpctGlobalInfo::insertBuiltinVarInfo(
    SourceLocation SL, unsigned int Len, std::string Repl,
    std::shared_ptr<DeviceFunctionInfo> DFI) {
  auto LocInfo = getLocInfo(SL);
  auto FileInfo = insertFile(LocInfo.first);
  auto &M = FileInfo->getBuiltinVarInfoMap();
  auto Iter = M.find(LocInfo.second);
  if (Iter == M.end()) {
    BuiltinVarInfo BVI(Len, Repl, DFI);
    M.insert(std::make_pair(LocInfo.second, BVI));
  }
}
void DpctGlobalInfo::insertSpBLASWarningLocOffset(SourceLocation SL) {
  auto LocInfo = getLocInfo(SL);
  auto FileInfo = insertFile(LocInfo.first);
  FileInfo->getSpBLASSet().insert(LocInfo.second);
}
std::shared_ptr<TextModification>
DpctGlobalInfo::findConstantMacroTMInfo(SourceLocation SL) {
  auto LocInfo = getLocInfo(SL);
  auto FileInfo = insertFile(LocInfo.first);
  auto &S = FileInfo->getConstantMacroTMSet();
  for (const auto &TM : S) {
    if (TM->getConstantOffset() == LocInfo.second) {
      return TM;
    }
  }
  return nullptr;
}
void DpctGlobalInfo::insertConstantMacroTMInfo(
    SourceLocation SL, std::shared_ptr<TextModification> TM) {
  auto LocInfo = getLocInfo(SL);
  auto FileInfo = insertFile(LocInfo.first);
  TM->setConstantOffset(LocInfo.second);
  auto &S = FileInfo->getConstantMacroTMSet();
  S.insert(TM);
}
void DpctGlobalInfo::insertAtomicInfo(std::string HashStr, SourceLocation SL,
                                      std::string FuncName) {
  auto LocInfo = getLocInfo(SL);
  auto FileInfo = insertFile(LocInfo.first);
  auto &M = FileInfo->getAtomicMap();
  if (M.find(HashStr) == M.end()) {
    M.insert(std::make_pair(HashStr,
                            std::make_tuple(LocInfo.second, FuncName, true)));
  }
}
void DpctGlobalInfo::removeAtomicInfo(std::string HashStr) {
  for (auto &File : FileMap) {
    auto &M = File.second->getAtomicMap();
    auto Iter = M.find(HashStr);
    if (Iter != M.end()) {
      std::get<2>(Iter->second) = false;
      return;
    }
  }
}
void DpctGlobalInfo::setFileEnterLocation(SourceLocation Loc) {
  auto LocInfo = getLocInfo(Loc);
  insertFile(LocInfo.first)->setFileEnterOffset(LocInfo.second);
}
void DpctGlobalInfo::setFirstIncludeLocation(SourceLocation Loc) {
  auto LocInfo = getLocInfo(Loc);
  insertFile(LocInfo.first)->setFirstIncludeOffset(LocInfo.second);
}
void DpctGlobalInfo::setLastIncludeLocation(SourceLocation Loc) {
  auto LocInfo = getLocInfo(Loc);
  insertFile(LocInfo.first)->setLastIncludeOffset(LocInfo.second);
}
void DpctGlobalInfo::setMathHeaderInserted(SourceLocation Loc, bool B) {
  auto LocInfo = getLocInfo(Loc);
  insertFile(LocInfo.first)->setMathHeaderInserted(B);
}
void DpctGlobalInfo::setAlgorithmHeaderInserted(SourceLocation Loc, bool B) {
  auto LocInfo = getLocInfo(Loc);
  insertFile(LocInfo.first)->setAlgorithmHeaderInserted(B);
}
void DpctGlobalInfo::setTimeHeaderInserted(SourceLocation Loc, bool B) {
  auto LocInfo = getLocInfo(Loc);
  insertFile(LocInfo.first)->setTimeHeaderInserted(B);
}
void DpctGlobalInfo::insertHeader(SourceLocation Loc, HeaderType Type,
                                  ReplacementType IsForCodePin) {
  auto LocInfo = getLocInfo(Loc);
  insertFile(LocInfo.first)->insertHeader(Type, IsForCodePin);
}
void DpctGlobalInfo::insertHeader(SourceLocation Loc, std::string HeaderName) {
  auto LocInfo = getLocInfo(Loc);
  insertFile(LocInfo.first)->insertCustomizedHeader(std::move(HeaderName));
}
void DpctGlobalInfo::removeVarNameInGlobalVarNameSet(
    const std::string &VarName) {
  auto Iter = getGlobalVarNameSet().find(VarName);
  if (Iter != getGlobalVarNameSet().end()) {
    getGlobalVarNameSet().erase(Iter);
  }
}
int DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc() {
  int Res = HelperFuncReplInfoIndex;
  HelperFuncReplInfoIndex++;
  return Res;
}
void DpctGlobalInfo::recordIncludingRelationship(
    const clang::tooling::UnifiedPath &CurrentFileName,
    const clang::tooling::UnifiedPath &IncludedFileName) {
  auto CurrentFileInfo = this->insertFile(CurrentFileName);
  auto IncludedFileInfo = this->insertFile(IncludedFileName);
  CurrentFileInfo->insertIncludedFilesInfo(IncludedFileInfo);
}
unsigned int DpctGlobalInfo::getCudaKernelDimDFIIndexThenInc() {
  unsigned int Res = CudaKernelDimDFIIndex;
  ++CudaKernelDimDFIIndex;
  return Res;
}
void DpctGlobalInfo::insertCudaKernelDimDFIMap(
    unsigned int Index, std::shared_ptr<DeviceFunctionInfo> Ptr) {
  CudaKernelDimDFIMap.insert(std::make_pair(Index, Ptr));
}
std::shared_ptr<DeviceFunctionInfo>
DpctGlobalInfo::getCudaKernelDimDFI(unsigned int Index) {
  auto Iter = CudaKernelDimDFIMap.find(Index);
  if (Iter != CudaKernelDimDFIMap.end())
    return Iter->second;
  return nullptr;
}
void DpctGlobalInfo::resetInfo() {
  FileMap.clear();
  PrecAndDomPairSet.clear();
  KCIndentWidthMap.clear();
  LocationInitIndexMap.clear();
  ExpansionRangeToMacroRecord.clear();
  EndifLocationOfIfdef.clear();
  ConditionalCompilationLoc.clear();
  MacroTokenToMacroDefineLoc.clear();
  FunctionCallInMacroMigrateRecord.clear();
  EndOfEmptyMacros.clear();
  BeginOfEmptyMacros.clear();
  FileRelpsMap.clear();
  MsfInfoMap.clear();
  MacroDefines.clear();
  CAPPInfoMap.clear();
  CurrentMaxIndex = 0;
  CurrentIndexInRule = 0;
  IncludingFileSet.clear();
  FileSetInCompilationDB.clear();
  GlobalVarNameSet.clear();
  HasFoundDeviceChanged = false;
  HelperFuncReplInfoMap.clear();
  HelperFuncReplInfoIndex = 1;
  TempVariableDeclCounterMap.clear();
  TempVariableHandledMap.clear();
  UsingDRYPattern = true;
  NeedRunAgain = false;
  SpellingLocToDFIsMapForAssumeNDRange.clear();
  DFIToSpellingLocsMapForAssumeNDRange.clear();
  FreeQueriesInfo::reset();
  CustomHelperFunctionAddtionalIncludes.clear();
}
void DpctGlobalInfo::updateSpellingLocDFIMaps(
    SourceLocation SL, std::shared_ptr<DeviceFunctionInfo> DFI) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  std::string Loc = getCombinedStrFromLoc(SM.getSpellingLoc(SL));

  auto IterOfL2D = SpellingLocToDFIsMapForAssumeNDRange.find(Loc);
  if (IterOfL2D == SpellingLocToDFIsMapForAssumeNDRange.end()) {
    std::unordered_set<std::shared_ptr<DeviceFunctionInfo>> Set;
    Set.insert(DFI);
    SpellingLocToDFIsMapForAssumeNDRange.insert(std::make_pair(Loc, Set));
  } else {
    IterOfL2D->second.insert(DFI);
  }

  auto IterOfD2L = DFIToSpellingLocsMapForAssumeNDRange.find(DFI);
  if (IterOfD2L == DFIToSpellingLocsMapForAssumeNDRange.end()) {
    std::unordered_set<std::string> Set;
    Set.insert(Loc);
    DFIToSpellingLocsMapForAssumeNDRange.insert(std::make_pair(DFI, Set));
  } else {
    IterOfD2L->second.insert(Loc);
  }
}
std::unordered_set<std::shared_ptr<DeviceFunctionInfo>>
DpctGlobalInfo::getDFIVecRelatedFromSpellingLoc(
    std::shared_ptr<DeviceFunctionInfo> DFI) {
  std::unordered_set<std::shared_ptr<DeviceFunctionInfo>> Res;
  auto IterOfD2L = DFIToSpellingLocsMapForAssumeNDRange.find(DFI);
  if (IterOfD2L == DFIToSpellingLocsMapForAssumeNDRange.end()) {
    return Res;
  }

  for (const auto &SpellingLoc : IterOfD2L->second) {
    auto IterOfL2D = SpellingLocToDFIsMapForAssumeNDRange.find(SpellingLoc);
    if (IterOfL2D != SpellingLocToDFIsMapForAssumeNDRange.end()) {
      Res.insert(IterOfL2D->second.begin(), IterOfL2D->second.end());
    }
  }
  return Res;
}
void DpctGlobalInfo::addPriorityReplInfo(
    std::string Key, std::shared_ptr<PriorityReplInfo> Info) {
  if (PriorityReplInfoMap.count(Key)) {
    if (PriorityReplInfoMap[Key]->Priority == Info->Priority) {
      PriorityReplInfoMap[Key]->Repls.insert(
          PriorityReplInfoMap[Key]->Repls.end(), Info->Repls.begin(),
          Info->Repls.end());
      PriorityReplInfoMap[Key]->RelatedAction.insert(
          PriorityReplInfoMap[Key]->RelatedAction.end(),
          Info->RelatedAction.begin(), Info->RelatedAction.end());
    } else if (PriorityReplInfoMap[Key]->Priority < Info->Priority) {
      PriorityReplInfoMap[Key] = Info;
    }
  } else {
    PriorityReplInfoMap[Key] = Info;
  }
}
std::tuple<unsigned int, std::string, SourceRange>
    DpctGlobalInfo::LastMacroRecord =
        std::make_tuple<unsigned int, std::string, SourceRange>(0, "",
                                                                SourceRange());
DpctGlobalInfo::DpctGlobalInfo() {
  IsInAnalysisScopeFunc = DpctGlobalInfo::checkInAnalysisScope;
  GetRunRound = DpctGlobalInfo::getRunRound;
  RecordTokenSplit = DpctGlobalInfo::recordTokenSplit;
  tooling::SetGetRunRound(DpctGlobalInfo::getRunRound);
  tooling::SetReProcessFile(DpctGlobalInfo::ReProcessFile);
  tooling::SetIsExcludePathHandler(DpctGlobalInfo::isExcluded);
}
void DpctGlobalInfo::recordTokenSplit(SourceLocation SL, unsigned Len) {
  auto It = getExpansionRangeToMacroRecord().find(
      getCombinedStrFromLoc(SM->getSpellingLoc(SL)));
  if (It != getExpansionRangeToMacroRecord().end()) {
    dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord()
        [getCombinedStrFromLoc(SM->getSpellingLoc(SL).getLocWithOffset(Len))] =
            It->second;
  }
}
/// This variable saved the info of previous migration from the
/// MainSourceFiles.yaml file. This variable is valid after
/// canContinueMigration() is called.
std::shared_ptr<clang::tooling::TranslationUnitReplacements>
    DpctGlobalInfo::MainSourceYamlTUR =
        std::make_shared<clang::tooling::TranslationUnitReplacements>();
clang::tooling::UnifiedPath DpctGlobalInfo::InRoot;
clang::tooling::UnifiedPath DpctGlobalInfo::OutRoot;
std::vector<clang::tooling::UnifiedPath> DpctGlobalInfo::AnalysisScope;
std::unordered_set<std::string> DpctGlobalInfo::ChangeExtensions = {};
std::string DpctGlobalInfo::SYCLSourceExtension = std::string();
std::string DpctGlobalInfo::SYCLHeaderExtension = std::string();
// TODO: implement one of this for each source language.
clang::tooling::UnifiedPath DpctGlobalInfo::CudaPath;
std::string DpctGlobalInfo::RuleFile = std::string();
UsmLevel DpctGlobalInfo::UsmLvl = UsmLevel::UL_None;
unsigned DpctGlobalInfo::BuildScriptType = 0;
clang::CudaVersion DpctGlobalInfo::SDKVersion = clang::CudaVersion::UNKNOWN;
bool DpctGlobalInfo::NeedDpctDeviceExt = false;
bool DpctGlobalInfo::IsIncMigration = true;
bool DpctGlobalInfo::IsQueryAPIMapping = false;
unsigned int DpctGlobalInfo::AssumedNDRangeDim = 3;
std::unordered_set<std::string> DpctGlobalInfo::PrecAndDomPairSet;
format::FormatRange DpctGlobalInfo::FmtRng = format::FormatRange::none;
DPCTFormatStyle DpctGlobalInfo::FmtST = DPCTFormatStyle::FS_LLVM;
bool DpctGlobalInfo::EnableCtad = false;
bool DpctGlobalInfo::EnableCodePin = false;
bool DpctGlobalInfo::IsMLKHeaderUsed = false;
bool DpctGlobalInfo::GenBuildScript = false;
bool DpctGlobalInfo::MigrateBuildScriptOnly = false;
bool DpctGlobalInfo::EnableComments = false;
bool DpctGlobalInfo::TempEnableDPCTNamespace = false;
ASTContext *DpctGlobalInfo::Context = nullptr;
SourceManager *DpctGlobalInfo::SM = nullptr;
FileManager *DpctGlobalInfo::FM = nullptr;
bool DpctGlobalInfo::KeepOriginCode = false;
bool DpctGlobalInfo::SyclNamedLambda = false;
bool DpctGlobalInfo::GuessIndentWidthMatcherFlag = false;
unsigned int DpctGlobalInfo::IndentWidth = 0;
std::map<unsigned int, unsigned int> DpctGlobalInfo::KCIndentWidthMap;
std::unordered_map<std::string, int> DpctGlobalInfo::LocationInitIndexMap;
std::unordered_map<
    std::string,
    std::pair<std::pair<clang::tooling::UnifiedPath /*begin file name*/,
                        unsigned int /*begin offset*/>,
              std::pair<clang::tooling::UnifiedPath /*end file name*/,
                        unsigned int /*end offset*/>>>
    DpctGlobalInfo::ExpansionRangeBeginMap;
bool DpctGlobalInfo::CheckUnicodeSecurityFlag = false;
bool DpctGlobalInfo::EnablepProfilingFlag = false;
std::map<std::string, std::shared_ptr<DpctGlobalInfo::MacroExpansionRecord>>
    DpctGlobalInfo::ExpansionRangeToMacroRecord;
std::unordered_map<std::string, std::shared_ptr<DpctGlobalInfo::MacroArgRecord>>
    DpctGlobalInfo::MacroArgRecordMap;
std::map<std::string, SourceLocation> DpctGlobalInfo::EndifLocationOfIfdef;
std::vector<std::pair<clang::tooling::UnifiedPath, size_t>>
    DpctGlobalInfo::ConditionalCompilationLoc;
std::map<std::string, std::shared_ptr<DpctGlobalInfo::MacroDefRecord>>
    DpctGlobalInfo::MacroTokenToMacroDefineLoc;
std::map<std::string, std::string>
    DpctGlobalInfo::FunctionCallInMacroMigrateRecord;
std::map<std::string, SourceLocation> DpctGlobalInfo::EndOfEmptyMacros;
std::map<std::string, unsigned int> DpctGlobalInfo::BeginOfEmptyMacros;
std::unordered_map<std::string, std::vector<clang::tooling::Replacement>>
    DpctGlobalInfo::FileRelpsMap;
std::unordered_map<std::string, clang::tooling::MainSourceFileInfo>
    DpctGlobalInfo::MsfInfoMap;
const std::string DpctGlobalInfo::YamlFileName = "MainSourceFiles.yaml";
std::map<std::string, bool> DpctGlobalInfo::MacroDefines;
int DpctGlobalInfo::CurrentMaxIndex = 0;
int DpctGlobalInfo::CurrentIndexInRule = 0;
std::set<clang::tooling::UnifiedPath> DpctGlobalInfo::IncludingFileSet;
std::set<std::string> DpctGlobalInfo::FileSetInCompilationDB;
std::set<std::string> DpctGlobalInfo::GlobalVarNameSet;
clang::format::FormatStyle DpctGlobalInfo::CodeFormatStyle;
bool DpctGlobalInfo::HasFoundDeviceChanged = false;
std::unordered_map<int, DpctGlobalInfo::HelperFuncReplInfo>
    DpctGlobalInfo::HelperFuncReplInfoMap;
int DpctGlobalInfo::HelperFuncReplInfoIndex = 1;
std::unordered_map<std::string, DpctGlobalInfo::TempVariableDeclCounter>
    DpctGlobalInfo::TempVariableDeclCounterMap;
std::unordered_map<std::string, int> DpctGlobalInfo::TempVariableHandledMap;
bool DpctGlobalInfo::UsingDRYPattern = true;
unsigned int DpctGlobalInfo::CudaKernelDimDFIIndex = 1;
std::unordered_map<unsigned int, std::shared_ptr<DeviceFunctionInfo>>
    DpctGlobalInfo::CudaKernelDimDFIMap;
CudaArchPPMap DpctGlobalInfo::CAPPInfoMap;
HDFuncInfoMap DpctGlobalInfo::HostDeviceFuncInfoMap;
// __CUDA_ARCH__ Offset -> defined(...) Offset
CudaArchDefMap DpctGlobalInfo::CudaArchDefinedMap;
std::unordered_map<std::string, std::shared_ptr<ExtReplacement>>
    DpctGlobalInfo::CudaArchMacroRepl;
std::unordered_map<clang::tooling::UnifiedPath,
                   std::pair<std::shared_ptr<ExtReplacements>,
                             std::shared_ptr<ExtReplacements>>>
    DpctGlobalInfo::FileReplCache;
std::set<clang::tooling::UnifiedPath> DpctGlobalInfo::ReProcessFile;
bool DpctGlobalInfo::NeedRunAgain = false;
unsigned int DpctGlobalInfo::RunRound = 0;
std::set<clang::tooling::UnifiedPath> DpctGlobalInfo::ModuleFiles;
std::unordered_map<std::string,
                   std::unordered_set<std::shared_ptr<DeviceFunctionInfo>>>
    DpctGlobalInfo::SpellingLocToDFIsMapForAssumeNDRange;
std::unordered_map<std::shared_ptr<DeviceFunctionInfo>,
                   std::unordered_set<std::string>>
    DpctGlobalInfo::DFIToSpellingLocsMapForAssumeNDRange;
unsigned DpctGlobalInfo::ExtensionDEFlag = static_cast<unsigned>(-1);
unsigned DpctGlobalInfo::ExtensionDDFlag = 0;
unsigned DpctGlobalInfo::ExperimentalFlag = 0;
unsigned DpctGlobalInfo::HelperFuncPreferenceFlag = 0;
bool DpctGlobalInfo::AnalysisModeFlag = false;
bool DpctGlobalInfo::UseSYCLCompatFlag = false;
bool DpctGlobalInfo::CVersionCUDALaunchUsedFlag = false;
unsigned int DpctGlobalInfo::ColorOption = 1;
std::unordered_map<int, std::shared_ptr<DeviceFunctionInfo>>
    DpctGlobalInfo::CubPlaceholderIndexMap;
std::vector<std::shared_ptr<DpctFileInfo>> DpctGlobalInfo::CSourceFileInfo;
bool DpctGlobalInfo::OptimizeMigrationFlag = false;
std::unordered_map<std::string, std::shared_ptr<PriorityReplInfo>>
    DpctGlobalInfo::PriorityReplInfoMap;
std::unordered_map<std::string, bool> DpctGlobalInfo::ExcludePath = {};
std::map<std::string, clang::tooling::OptionInfo> DpctGlobalInfo::CurrentOptMap;
std::unordered_map<std::string, std::unordered_map<clang::tooling::UnifiedPath,
                                                   std::vector<unsigned>>>
    DpctGlobalInfo::RnnInputMap;
std::unordered_map<clang::tooling::UnifiedPath,
                   std::vector<clang::tooling::UnifiedPath>>
    DpctGlobalInfo::MainSourceFileMap;
std::unordered_map<std::string, bool> DpctGlobalInfo::MallocHostInfoMap;
std::map<std::shared_ptr<TextModification>, bool>
    DpctGlobalInfo::ConstantReplProcessedFlagMap;
IncludeMapSetTy DpctGlobalInfo::IncludeMapSet;
std::vector<std::pair<std::string, VarInfoForCodePin>>
    DpctGlobalInfo::CodePinTypeInfoMap;
std::vector<std::pair<std::string, VarInfoForCodePin>>
    DpctGlobalInfo::CodePinTemplateTypeInfoMap;
std::vector<std::pair<std::string, std::vector<std::string>>>
    DpctGlobalInfo::CodePinTypeDepsVec;
std::vector<std::pair<std::string, std::vector<std::string>>>
    DpctGlobalInfo::CodePinDumpFuncDepsVec;
std::unordered_set<std::string> DpctGlobalInfo::NeedParenAPISet = {};
std::unordered_set<std::string>
    DpctGlobalInfo::CustomHelperFunctionAddtionalIncludes = {};
///// class DpctNameGenerator /////
void DpctNameGenerator::printName(const FunctionDecl *FD,
                                  llvm::raw_ostream &OS) {
  if (G.writeName(FD, OS)) {
    FD->printQualifiedName(OS, PP);
    OS << "@";
    FD->getType().print(OS, PP);
  }
}
DpctNameGenerator::DpctNameGenerator(ASTContext &Ctx)
    : G(Ctx), PP(Ctx.getPrintingPolicy()) {
  PP.PrintCanonicalTypes = true;
}
std::string DpctNameGenerator::getName(const FunctionDecl *D) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  printName(D, OS);
  return OS.str();
}
///// class SizeInfo /////
SizeInfo::SizeInfo(std::shared_ptr<TemplateDependentStringInfo> TDSI)
    : TDSI(TDSI) {}
const std::string &SizeInfo::getSize() {
  if (TDSI)
    return TDSI->getSourceString();
  return Size;
}
void SizeInfo::setTemplateList(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  if (TDSI)
    TDSI = TDSI->applyTemplateArguments(TemplateList);
}
///// class CtTypeInfo /////
#define TYPE_CAST(Target) dyn_cast<Target>(T)
std::string getTypedefOrUsingTypeName(QualType QT) {
  const Type *T = QT.getTypePtr();
  switch (T->getTypeClass()) {
  case Type::TypeClass::IncompleteArray:
    return getTypedefOrUsingTypeName(
        TYPE_CAST(IncompleteArrayType)->getElementType());
  case Type::TypeClass::ConstantArray:
    return getTypedefOrUsingTypeName(
        TYPE_CAST(ConstantArrayType)->getElementType());
  case Type::TypeClass::Pointer:
    return getTypedefOrUsingTypeName(TYPE_CAST(PointerType)->getPointeeType());
  case Type::TypeClass::Elaborated:
    return getTypedefOrUsingTypeName(TYPE_CAST(ElaboratedType)->desugar());
  case Type::TypeClass::Typedef: {
    const TypedefNameDecl *TND = TYPE_CAST(TypedefType)->getDecl();
    if (isUserDefinedDecl(TND)) {
      Decl::Kind K = TND->getDeclContext()->getDeclKind();
      if (K != Decl::Kind::TranslationUnit && K != Decl::Kind::Namespace)
        return TND->getNameAsString();
    }
    return "";
  }
  case Type::TypeClass::Using: {
    const UsingShadowDecl *USD = TYPE_CAST(clang::UsingType)->getFoundDecl();
    if (isUserDefinedDecl(USD)) {
      Decl::Kind K = USD->getDeclContext()->getDeclKind();
      if (K != Decl::Kind::TranslationUnit && K != Decl::Kind::Namespace)
        return USD->getNameAsString();
    }
    return "";
  }
  default:
    return "";
  }
}
#undef TYPE_CAST

CtTypeInfo::CtTypeInfo() {
  PointerLevel = 0;
  IsReference = 0;
  IsTemplate = 0;
  TemplateDependentMacro = 0;
  IsArray = 0;
  ContainSizeofType = 0;
  IsConstantQualified = 0;
}
CtTypeInfo::CtTypeInfo(const TypeLoc &TL, bool NeedSizeFold) : CtTypeInfo() {
  setTypeInfo(TL, NeedSizeFold);
}
CtTypeInfo::CtTypeInfo(const VarDecl *D, bool NeedSizeFold) : CtTypeInfo() {
  if (D && D->getTypeSourceInfo()) {
    auto TL = D->getTypeSourceInfo()->getTypeLoc();
    IsConstantQualified = D->hasAttr<CUDAConstantAttr>();
    setTypeInfo(TL, NeedSizeFold);
    if (TL.getTypeLocClass() == TypeLoc::IncompleteArray) {
      if (auto CAT = dyn_cast<ConstantArrayType>(D->getType())) {
        Range[0] = std::to_string(CAT->getSize().getZExtValue());
      }
    }
    if (D->hasAttr<CUDASharedAttr>()) {
      std::string TN = getTypedefOrUsingTypeName(D->getType());
      const FunctionDecl *FD = DpctGlobalInfo::findAncestor<FunctionDecl>(D);
      if (!TN.empty() && FD) {
        SharedVarInfo.TypeName = TN;
        SharedVarInfo.DefinitionFuncName = FD->getNameAsString();
      }
    }
  }
}
std::string CtTypeInfo::getRangeArgument(const std::string &MemSize,
                                         bool MustArguments) {
  std::string Arg = "(";
  for (unsigned i = 0; i < Range.size(); ++i) {
    auto Size = Range[i].getSize();
    if (Size.empty()) {
      if (MemSize.empty()) {
        Arg += "1, ";
      } else {
        Arg += MemSize;
        Arg += ", ";
      }
      for (unsigned tmp = i + 1; tmp < Range.size(); ++tmp)
        Arg += "1, ";
      break;
    } else
      Arg += Size;
    Arg += ", ";
  }
  return (Arg.size() == 1) ? (MustArguments ? (Arg + ")") : "")
                           : Arg.replace(Arg.size() - 2, 2, ")");
}
void CtTypeInfo::adjustAsMemType() {
  setPointerAsArray();
  removeQualifier();
}
std::shared_ptr<CtTypeInfo> CtTypeInfo::applyTemplateArguments(
    const std::vector<TemplateArgumentInfo> &TA) {
  auto NewType = std::make_shared<CtTypeInfo>(*this);
  if (TDSI)
    NewType->TDSI = TDSI->applyTemplateArguments(TA);
  for (auto &R : NewType->Range)
    R.setTemplateList(TA);
  NewType->BaseName.clear();
  NewType->updateName();
  return NewType;
}
void CtTypeInfo::setTypeInfo(const TypeLoc &TL, bool NeedSizeFold) {
  switch (TL.getTypeLocClass()) {
  case TypeLoc::Qualified:
    BaseName = TL.getType().getLocalQualifiers().getAsString(
        DpctGlobalInfo::getContext().getPrintingPolicy());
    return setTypeInfo(TYPELOC_CAST(QualifiedTypeLoc).getUnqualifiedLoc(),
                       NeedSizeFold);
  case TypeLoc::ConstantArray:
    IsArray = true;
    return setArrayInfo(TYPELOC_CAST(ConstantArrayTypeLoc), NeedSizeFold);
  case TypeLoc::DependentSizedArray:
    return setArrayInfo(TYPELOC_CAST(DependentSizedArrayTypeLoc), NeedSizeFold);
  case TypeLoc::IncompleteArray:
    return setArrayInfo(TYPELOC_CAST(IncompleteArrayTypeLoc), NeedSizeFold);
  case TypeLoc::Pointer:
    ++PointerLevel;
    return setTypeInfo(TYPELOC_CAST(PointerTypeLoc).getPointeeLoc());
  case TypeLoc::LValueReference:
  case TypeLoc::RValueReference:
    IsReference = true;
    return setTypeInfo(TYPELOC_CAST(ReferenceTypeLoc).getPointeeLoc());
  case TypeLoc::Elaborated: {
    const TypeLoc &NamedTypeLoc =
        TYPELOC_CAST(ElaboratedTypeLoc).getNamedTypeLoc();
    if (const auto TTL = NamedTypeLoc.getAs<TypedefTypeLoc>()) {
      if (setTypedefInfo(TTL, NeedSizeFold))
        return;
    }
    break;
  }
  case TypeLoc::Typedef: {
    if (setTypedefInfo(TYPELOC_CAST(TypedefTypeLoc), NeedSizeFold))
      return;
    break;
  }
  default:
    break;
  }
  setName(TL);
}
std::string CtTypeInfo::getFoldedArraySize(const ConstantArrayTypeLoc &TL) {
  const auto *const SizeExpr = TL.getSizeExpr();

  auto IsContainMacro =
      isContainMacro(SizeExpr) || !TL.getSizeExpr()->getBeginLoc().isFileID();

  auto DREMatcher = ast_matchers::findAll(ast_matchers::declRefExpr());
  auto DREMatchedResults =
      ast_matchers::match(DREMatcher, *SizeExpr, DpctGlobalInfo::getContext());
  bool IsContainDRE = !DREMatchedResults.empty();

  bool IsContainSizeOfUserDefinedType = false;
  auto SOMatcher = ast_matchers::findAll(
      ast_matchers::unaryExprOrTypeTraitExpr(ast_matchers::ofKind(UETT_SizeOf))
          .bind("so"));
  auto SOMatchedResults =
      ast_matchers::match(SOMatcher, *SizeExpr, DpctGlobalInfo::getContext());
  for (const auto &Res : SOMatchedResults) {
    const auto *UETT = Res.getNodeAs<UnaryExprOrTypeTraitExpr>("so");
    if (UETT->isArgumentType()) {
      const auto *const RD =
          UETT->getArgumentType().getCanonicalType()->getAsRecordDecl();
      if (MapNamesLang::SupportedVectorTypes.count(RD->getNameAsString()) ==
          0) {
        IsContainSizeOfUserDefinedType = true;
        break;
      }
    }
  }

  // We need not fold the size expression in these cases.
  if (!IsContainMacro && !IsContainDRE && !IsContainSizeOfUserDefinedType) {
    return getUnfoldedArraySize(TL);
  }

  auto TLRange = getDefinitionRange(TL.getBeginLoc(), TL.getEndLoc());
  auto SizeExprRange = getRangeInRange(SizeExpr->getSourceRange(),
                                       TLRange.getBegin(), TLRange.getEnd());
  auto SizeExprBegin = SizeExprRange.first;
  auto SizeExprEnd = SizeExprRange.second;
  auto &SM = DpctGlobalInfo::getSourceManager();
  size_t Length =
      SM.getCharacterData(SizeExprEnd) - SM.getCharacterData(SizeExprBegin);
  auto DL = SM.getDecomposedLoc(SizeExprBegin);
  auto OriginalStr =
      std::string(SM.getBufferData(DL.first).substr(DL.second, Length));

  // When it is a literal in macro, we also need not fold.
  auto LiteralStr = toString(TL.getTypePtr()->getSize(), 10, false, false);
  if (OriginalStr == LiteralStr) {
    return getUnfoldedArraySize(TL);
  }

  ArraySizeOriginExprs.push_back(std::move(OriginalStr));
  return buildString(LiteralStr, "/*", ArraySizeOriginExprs.back(), "*/");
}
std::string CtTypeInfo::getUnfoldedArraySize(const ConstantArrayTypeLoc &TL) {
  ContainSizeofType = containSizeOfType(TL.getSizeExpr());
  ExprAnalysis A;
  A.analyze(TL.getSizeExpr());
  return A.getReplacedString();
}
bool CtTypeInfo::setTypedefInfo(const TypedefTypeLoc &TL, bool NeedSizeFold) {
  const TypedefNameDecl *TND = TL.getTypedefNameDecl();
  if (!TND)
    return false;
  if (!TND->getTypeSourceInfo())
    return false;
  const TypeLoc TypedefTpyeDeclLoc = TND->getTypeSourceInfo()->getTypeLoc();
  ConstantArrayTypeLoc CATL;
  if (DpctGlobalInfo::isInAnalysisScope(TypedefTpyeDeclLoc.getBeginLoc()) &&
      (CATL = TypedefTpyeDeclLoc.getAs<ConstantArrayTypeLoc>())) {
    setArrayInfo(CATL, NeedSizeFold);
    return true;
  }
  return false;
}
void CtTypeInfo::setArrayInfo(const ConstantArrayTypeLoc &TL,
                              bool NeedSizeFold) {
  ContainSizeofType = containSizeOfType(TL.getSizeExpr());
  if (NeedSizeFold) {
    Range.emplace_back(getFoldedArraySize(TL));
  } else {
    Range.emplace_back(getUnfoldedArraySize(TL));
  }
  setTypeInfo(TL.getElementLoc(), NeedSizeFold);
}
void CtTypeInfo::setArrayInfo(const DependentSizedArrayTypeLoc &TL,
                              bool NeedSizeFold) {
  ContainSizeofType = containSizeOfType(TL.getSizeExpr());
  ExprAnalysis EA;
  EA.analyze(TL.getSizeExpr());
  auto TDSI = EA.getTemplateDependentStringInfo();
  if (TDSI->containsTemplateDependentMacro())
    TemplateDependentMacro = true;
  Range.emplace_back(EA.getTemplateDependentStringInfo());
  setTypeInfo(TL.getElementLoc(), NeedSizeFold);
}
void CtTypeInfo::setArrayInfo(const IncompleteArrayTypeLoc &TL,
                              bool NeedSizeFold) {
  Range.emplace_back();
  setTypeInfo(TL.getElementLoc(), NeedSizeFold);
}
void CtTypeInfo::setName(const TypeLoc &TL) {
  ExprAnalysis EA;
  EA.analyze(TL);
  TDSI = EA.getTemplateDependentStringInfo();
  auto SetFromTL = EA.getHelperFeatureSet();
  HelperFeatureSet.insert(SetFromTL.begin(), SetFromTL.end());

  IsTemplate = TL.getTypePtr()->isDependentType();
  updateName();
}
void CtTypeInfo::updateName() {
  BaseNameWithoutQualifiers = TDSI->getSourceString();
  auto SetFromTTDSI = TDSI->getHelperFeatureSet();
  HelperFeatureSet.insert(SetFromTTDSI.begin(), SetFromTTDSI.end());

  if (isPointer()) {
    BaseNameWithoutQualifiers += ' ';
    BaseNameWithoutQualifiers.append(PointerLevel, '*');
  }

  if (BaseName.empty())
    BaseName = BaseNameWithoutQualifiers;
  else {
    BaseName = buildString(BaseName, " ", BaseNameWithoutQualifiers);
  }
}
void CtTypeInfo::setPointerAsArray() {
  if (isPointer()) {
    --PointerLevel;
    Range.emplace_back();
    updateName();
  }
}
///// class VarInfo /////
void VarInfo::applyTemplateArguments(
    const std::vector<TemplateArgumentInfo> &TAList) {
  Ty = Ty->applyTemplateArguments(TAList);
}
void VarInfo::requestFeatureForSet(const clang::tooling::UnifiedPath &Path) {
  if (Ty) {
    for (const auto &Item : Ty->getHelperFeatureSet()) {
      requestFeature(Item);
    }
  }
}
///// class MemVarInfo /////
std::shared_ptr<MemVarInfo> MemVarInfo::buildMemVarInfo(const VarDecl *Var) {
  if (auto Func = DpctGlobalInfo::findAncestor<FunctionDecl>(Var)) {
    if (Func->getTemplateSpecializationKind() ==
            TSK_ExplicitInstantiationDefinition ||
        Func->getTemplateSpecializationKind() == TSK_ImplicitInstantiation)
      return std::shared_ptr<MemVarInfo>();
    auto LocInfo = DpctGlobalInfo::getLocInfo(Var);
    auto VI = std::make_shared<MemVarInfo>(LocInfo.second, LocInfo.first, Var);
    if (!DpctGlobalInfo::useGroupLocalMemory() || !VI->isShared() ||
        VI->isExtern())
      if (auto DFI = DeviceFunctionDecl::LinkRedecls(Func))
        DFI->addVar(VI);
    return VI;
  }
  return DpctGlobalInfo::getInstance().insertMemVarInfo(Var);
}

// This function, `migrateToDeviceGlobal`, migrates a CUDA `__device__` or
// `__constant__` variable declaration to the SYCL device global equivalent. The
// migration process involves four key steps. The function handles various
// transformations as follows:
//
// 1. Remove any array brackets following the variable name.
//    - It identifies the array type using TypeLoc and removes the brackets
//    while preserving the dimensions for later use.
//    - If the array size comes from a macro argument, it maps the macro
//    argument correctly using the `MacroArgRecord`.
// 2. Process the initialization expression.
//    - If the initialization style is C-style (with an equal sign), it remove
//    the equal sign and adds braces around scalar initializers to use
//    initializer list in SYCL.
// 3. Replace the variable type.
//    - Replace the origin type with
//    `sycl::ext::oneapi::experimental::device_global`
//      and the correct base type and dimensions.
//    - It manages macro arguments to correctly replace the base type when
//    required.
// 4. Insert the `static` specifier if the variable is declared globally and
//    does not already have the `static` storage class.
//
// Example1 (Specifier __device__ will be removed in preprocessor callbacks):
// Origin code:
// __device__ int var_a[3] = {1, 2, 3};
//
// As follow list the result after each key step listed in previous:
// 1. int var_a = {1, 2, 3};
// 2. int var_a {1, 2, 3};
// 3. sycl::ext::oneapi::experimental::device_global<int[3]> var_a {1, 2, 3};
// 4. static sycl::ext::oneapi::experimental::device_global<int[3]> var_a {1, 2,
// 3};
//
// Example2 (Specifier __device__ will be removed in preprocessor callbacks):
// Origin code:
// #define VAR(type, name, size) static __device__ type name[size];
// VAR(int, a, 3)
//
// As follow list the result after each key step listed in previous:
// 1. #define VAR(type, name, size) static type name;
//    VAR(int, a, 3)
// 2. #define VAR(type, name, init) static type name;
//    VAR(int, a, 3)
// 3. #define VAR(type, name, init) static
//    sycl::ext::oneapi::experimental::device_global<type[size]> name;
//    VAR(int, a, 3)
// 4. #define VAR(type, name, init) static
//    sycl::ext::oneapi::experimental::device_global<type[size]> name;
//    VAR(int, a, 3)
void MemVarInfo::migrateToDeviceGlobal(const VarDecl *MemVar) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &Ctx = DpctGlobalInfo::getContext();
  auto &MacroArgMap = DpctGlobalInfo::getMacroArgRecordMap();
  auto TSI = MemVar->getTypeSourceInfo();
  auto OriginTL = TSI->getTypeLoc();
  auto TL = OriginTL;
  auto BegLoc = MemVar->getBeginLoc();
  if (BegLoc.isMacroID()) {
    BegLoc = SM.getExpansionLoc(BegLoc);
  }
  auto LocInfo = DpctGlobalInfo::getLocInfo(BegLoc);
  std::string Dims;
  bool IsArray = OriginTL.getType()->isArrayType();
  // Step 1
  while (auto ATL = TL.getAs<clang::ArrayTypeLoc>()) {
    auto BRange = ATL.getBracketsRange();
    BRange = getDefinitionRange(BRange.getBegin(), BRange.getEnd());
    auto RT =
        ReplaceText(SM.getSpellingLoc(BRange.getBegin()),
                    SM.getSpellingLoc(BRange.getEnd()).getLocWithOffset(1), "");
    DpctGlobalInfo::getInstance().addReplacement(RT.getReplacement(Ctx));
    Dims += "[";
    std::string SizeStr;
    if (clang::Expr *SE = ATL.getSizeExpr()) {
      auto SizeLoc = SE->getBeginLoc();
      if (SM.isMacroArgExpansion(SizeLoc)) {
        auto Iter =
            MacroArgMap.find(DpctGlobalInfo::getInstance()
                                 .getMainFile()
                                 ->getFilePath()
                                 .getPath()
                                 .str() +
                             getCombinedStrFromLoc(SM.getSpellingLoc(SizeLoc)));
        if (Iter != MacroArgMap.end()) {
          SizeStr = Iter->second->ArgName;
        }
      }
      if (SizeStr.empty()) {
        SizeStr = ExprAnalysis::ref(SE);
      }
    }
    Dims += SizeStr + "]";
    TL = ATL.getElementLoc();
  }
  // Step 2
  if (MemVar->hasInit()) {
    if ((MemVar->getInitStyle() == VarDecl::InitializationStyle::CInit)) {
      DiagnosticsUtils::report(LocInfo.first, LocInfo.second,
                               Diagnostics::DEVICE_GLOBAL_INIT, true, false);
      if (!dyn_cast<InitListExpr>(
              MemVar->getInit()->IgnoreImplicitAsWritten())) {
        auto IBS = InsertBeforeStmt(MemVar->getInit(), "{");
        auto IAS = InsertAfterStmt(MemVar->getInit(), "}");
        DpctGlobalInfo::getInstance().addReplacement(IBS.getReplacement(Ctx));
        DpctGlobalInfo::getInstance().addReplacement(IAS.getReplacement(Ctx));
      }
      auto NextTok = Lexer::findNextToken(
          IsArray ? SM.getSpellingLoc(OriginTL.getEndLoc())
                  : SM.getSpellingLoc(MemVar->getLocation()),
          SM, DpctGlobalInfo::getContext().getLangOpts());
      if (NextTok.has_value() && NextTok.value().is(tok::equal)) {
        auto RTok = ReplaceToken(NextTok.value().getLocation(), "");
        DpctGlobalInfo::getInstance().addReplacement(RTok.getReplacement(Ctx));
      }
    }
  }
  // Step 3
  std::string BaseTypeStr;
  SourceLocation TypeReplLoc;
  size_t TypeReplLen = 0;
  if (SM.isMacroArgExpansion(OriginTL.getBeginLoc())) {
    auto Iter = MacroArgMap.find(
        DpctGlobalInfo::getInstance()
            .getMainFile()
            ->getFilePath()
            .getPath()
            .str() +
        getCombinedStrFromLoc(SM.getSpellingLoc(OriginTL.getBeginLoc())));
    if (Iter != MacroArgMap.end()) {
      BaseTypeStr = Iter->second->ArgName;
      TypeReplLoc = Iter->second->ArgLoc;
      TypeReplLen = BaseTypeStr.size();
    }
  }
  if (BaseTypeStr.empty()) {
    BaseTypeStr = getType()->getBaseNameWithoutQualifiers();
    TypeReplLoc = TL.getBeginLoc();
    TypeReplLen =
        SM.getFileOffset(TL.getEndLoc()) - SM.getFileOffset(TL.getBeginLoc()) +
        Lexer::MeasureTokenLength(TL.getEndLoc(), SM,
                                  DpctGlobalInfo::getContext().getLangOpts());
  }
  std::string TypeStr = MapNames::getClNamespace() +
                        "ext::oneapi::experimental::device_global<" +
                        BaseTypeStr + Dims + ">";
  auto RT = ReplaceText(TypeReplLoc, TypeReplLen, std::move(TypeStr));
  DpctGlobalInfo::getInstance().addReplacement(RT.getReplacement(Ctx));
  // Step 4
  if (MemVar->getStorageClass() != SC_Static && getScope() == Global) {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(LocInfo.first, LocInfo.second, 0,
                                         "static ", nullptr));
  }
}

MemVarInfo::VarAttrKind MemVarInfo::getAddressAttr(const VarDecl *VD) {
  if (VD->hasAttrs())
    return getAddressAttr(VD->getAttrs());
  return Host;
}
MemVarInfo::MemVarInfo(unsigned Offset,
                       const clang::tooling::UnifiedPath &FilePath,
                       const VarDecl *Var)
    : VarInfo(Offset, FilePath, Var,
              !(DpctGlobalInfo::useGroupLocalMemory() &&
                getAddressAttr(Var) == Shared &&
                Var->getStorageClass() != SC_Extern) &&
                  isLexicallyInLocalScope(Var)),
      Attr(getAddressAttr(Var)),
      Scope(isLexicallyInLocalScope(Var)
                ? (Var->getStorageClass() == SC_Extern ? Extern : Local)
                : Global),
      PointerAsArray(false) {
  if (isTreatPointerAsArray()) {
    Attr = Device;
    getType()->adjustAsMemType();
    PointerAsArray = true;
  }
  if (Var->hasInit())
    setInitList(Var->getInit(), Var);
  if (Var->getStorageClass() == SC_Static || getScope() == Global) {
    IsStatic = true;
  }

  if (auto Func = Var->getParentFunctionOrMethod()) {
    auto VT = Var->getType();
    DeclOfVarType = VT->getAsCXXRecordDecl();
    if (!DeclOfVarType) {
      if (const clang::ArrayType *AT = VT->getAsArrayTypeUnsafe()) {
        auto ElementType = AT->getElementType();
        DeclOfVarType = ElementType->getAsCXXRecordDecl();
      }
    }
    if (DeclOfVarType) {
      auto F = DeclOfVarType->getParentFunctionOrMethod();
      if (F && (F == Func)) {
        IsTypeDeclaredLocal = true;

        auto getParentDeclStmt = [&](const Decl *D) -> const DeclStmt * {
          auto P = getParentStmt(D);
          if (!P)
            return nullptr;
          auto DS = dyn_cast<DeclStmt>(P);
          if (!DS)
            return nullptr;
          return DS;
        };

        auto DS1 = getParentDeclStmt(Var);
        auto DS2 = getParentDeclStmt(DeclOfVarType);
        if (DS1 && DS2 && DS1 == DS2) {
          IsAnonymousType = true;
          DeclStmtOfVarType = DS2;
          const auto LocInfo = DpctGlobalInfo::getLocInfo(
              getDefinitionRange(DS2->getBeginLoc(), DS2->getEndLoc())
                  .getBegin());
          const auto LocStr = LocInfo.first.getCanonicalPath().str() + ":" +
                              std::to_string(LocInfo.second);
          auto Iter = AnonymousTypeDeclStmtMap.find(LocStr);
          if (Iter != AnonymousTypeDeclStmtMap.end()) {
            LocalTypeName = "type_ct" + std::to_string(Iter->second);
          } else {
            LocalTypeName =
                "type_ct" + std::to_string(AnonymousTypeDeclStmtMap.size() + 1);
            AnonymousTypeDeclStmtMap.insert(
                std::make_pair(LocStr, AnonymousTypeDeclStmtMap.size() + 1));
          }
        } else if (DS2) {
          DeclStmtOfVarType = DS2;
        }
      }
    }
  }
  if (getType()->getDimension() == 0 && !isTypeDeclaredLocal()) {
    if (Attr == Constant)
      AccMode = Value;
    else
      AccMode = Reference;
  } else if (getType()->getDimension() <= 1) {
    AccMode = Pointer;
  } else if (isShared() && isLocal()) {
    AccMode = PointerToArray;
  } else {
    AccMode = Accessor;
  }

  newConstVarInit(Var);
}
void MemVarInfo::newConstVarInit(const VarDecl *Var) {
  CharSourceRange SR(DpctGlobalInfo::getSourceManager().getExpansionRange(
      Var->getSourceRange()));
  auto BeginLoc = SR.getBegin();
  SourceManager &SM = DpctGlobalInfo::getSourceManager();
  size_t repLength = 0;
  auto Buffer = SM.getCharacterData(BeginLoc);
  auto Data = Buffer[repLength];
  while (Data != ';')
    Data = Buffer[++repLength];
  NewConstVarLength = ++repLength;
  NewConstVarOffset = DpctGlobalInfo::getLocInfo(BeginLoc).second;
}
std::string MemVarInfo::getDeclarationReplacement(const VarDecl *VD) {
  switch (Scope) {
  case clang::dpct::MemVarInfo::Local:
    if (isShared() && DpctGlobalInfo::useGroupLocalMemory() && VD) {

      auto FD = dyn_cast<FunctionDecl>(VD->getDeclContext());
      if (FD && FD->hasAttr<CUDADeviceAttr>())
        DiagnosticsUtils::report(getFilePath(), getOffset(),
                                 Diagnostics::GROUP_LOCAL_MEMORY, true, false);

      std::string Ret;
      llvm::raw_string_ostream OS(Ret);
      OS << "auto &" << getName() << " = "
         << "*" << MapNames::getClNamespace()
         << "ext::oneapi::group_local_memory_for_overwrite<"
         << getType()->getBaseName();
      for (auto &ArraySize : getType()->getRange()) {
        OS << "[" << ArraySize.getSize() << "]";
      }
      OS << ">(";
      FreeQueriesInfo::printImmediateText(
          OS, VD, nullptr, FreeQueriesInfo::FreeQueriesKind::Group);
      OS << "); ";
      return OS.str();
    }
    return "";
  case clang::dpct::MemVarInfo::Extern:
    if (isShared() && getType()->getDimension() > 1) {
      // For case like:
      // extern __shared__ int shad_mem[][2][3];
      // int p = shad_mem[0][0][2];
      // will be migrated to:
      // auto shad_mem = (int(*)[2][3])dpct_local;
      std::string Dimension;
      size_t Index = 0;
      for (auto &Entry : getType()->getRange()) {
        Index++;
        if (Index == 1)
          continue;
        Dimension = Dimension + "[" + Entry.getSize() + "]";
      }
      return buildString("auto ", getName(), " = (", getType()->getBaseName(),
                         "(*)", Dimension, ")", ExternVariableName, ";");
    }

    return buildString("auto ", getName(), " = (", getType()->getBaseName(),
                       " *)", ExternVariableName, ";");
  case clang::dpct::MemVarInfo::Global: {
    if (isShared())
      return "";
    if ((getAttr() == MemVarInfo::VarAttrKind::Constant) &&
        !isUseHelperFunc() && !isUseDeviceGlobal()) {
      std::string Dims;
      const static std::string NullString;
      for (auto &Dim : getType()->getRange()) {
        Dims = Dims + "[" + Dim.getSize() + "]";
      }
      return buildString(isStatic() ? "static " : "", getMemoryType(), " ",
                         getConstVarName() + Dims,
                         PointerAsArray ? "" : getInitArguments(NullString),
                         ";");
    }
    return getMemoryDecl();
  }
  }
  clang::dpct::DpctDebugs()
      << "[MemVarInfo::VarAttrKind] Unexpected value: " << Scope << "\n";
  assert(0);
  return "";
}
std::string MemVarInfo::getInitStmt(StringRef QueueString) {
  if (QueueString.empty())
    return getConstVarName() + ".init();";
  return buildString(getConstVarName(), ".init(", QueueString, ");");
}
std::string MemVarInfo::getMemoryDecl(const std::string &MemSize) {
  return buildString(isStatic() ? "static " : "", getMemoryType(), " ",
                     getConstVarName(),
                     PointerAsArray ? "" : getInitArguments(MemSize), ";");
}
std::string MemVarInfo::getMemoryDecl() {
  const static std::string NullString;
  return getMemoryDecl(NullString);
}
std::string MemVarInfo::getExternGlobalVarDecl() {
  return buildString("extern ", getMemoryType(), " ", getConstVarName(), ";");
}
void MemVarInfo::appendAccessorOrPointerDecl(const std::string &ExternMemSize,
                                             bool ExternEmitWarning,
                                             StmtList &AccList,
                                             StmtList &PtrList, LocInfo LI) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  if (isShared()) {
    OS << getSyclAccessorType(LI);
    OS << " " << getAccessorName() << "(";
    if (getType()->getDimension() && AccMode != PointerToArray)
      OS << getRangeClass() << getType()->getRangeArgument(ExternMemSize, false)
         << ", ";
    OS << "cgh)";
    OS << ";";
    StmtWithWarning AccDecl(OS.str());
    for (const auto &OriginExpr : getType()->getArraySizeOriginExprs()) {
      DiagnosticsUtils::report(getFilePath(), getOffset(),
                               Diagnostics::MACRO_EXPR_REPLACED, false, false,
                               OriginExpr);
      AccDecl.Warnings.push_back(
          DiagnosticsUtils::getWarningTextAndUpdateUniqueID(
              Diagnostics::MACRO_EXPR_REPLACED, OriginExpr));
    }
    if ((isExtern() && ExternEmitWarning) || getType()->containSizeofType()) {
      DiagnosticsUtils::report(getFilePath(), getOffset(),
                               Diagnostics::SIZEOF_WARNING, false, false,
                               "local memory",
                               "Check that the allocated memory size in the "
                               "migrated code is correct");
      AccDecl.Warnings.push_back(
          DiagnosticsUtils::getWarningTextAndUpdateUniqueID(
              Diagnostics::SIZEOF_WARNING, "local memory",
              "Check that the allocated memory size in the migrated code is "
              "correct"));
    }
    if (getType()->getDimension() > 3) {
      if (DiagnosticsUtils::report(getFilePath(), getOffset(),
                                   Diagnostics::EXCEED_MAX_DIMENSION, false,
                                   false)) {
        AccDecl.Warnings.push_back(
            DiagnosticsUtils::getWarningTextAndUpdateUniqueID(
                Diagnostics::EXCEED_MAX_DIMENSION));
      }
    }
    AccList.emplace_back(std::move(AccDecl));
  } else if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted &&
             AccMode != Accessor) {
    requestFeature(HelperFeatureEnum::device_ext);
    PtrList.emplace_back(buildString("auto ", getPtrName(), " = ",
                                     getConstVarName(), ".get_ptr();"));
  } else {
    requestFeature(HelperFeatureEnum::device_ext);
    AccList.emplace_back(buildString("auto ", getAccessorName(), " = ",
                                     getConstVarName(), ".get_access(cgh);"));
  }
}
std::string MemVarInfo::getRangeClass() {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  return DpctGlobalInfo::printCtadClass(OS,
                                        MapNames::getClNamespace() + "range",
                                        getType()->getDimension())
      .str();
}
std::string MemVarInfo::getRangeDecl(const std::string &MemSize) {
  return buildString(getRangeClass(), " ", getRangeName(),
                     getType()->getRangeArgument(MemSize, false), ";");
}
ParameterStream &MemVarInfo::getFuncDecl(ParameterStream &PS) {
  if (AccMode == PointerToArray) {
    PS << getType()->getBaseName() << " ";
    PS << getArgName();
    auto Range = getType()->getRange();
    for (size_t i = 0; i < Range.size(); i++) {
      PS << "[" << Range[i].getSize() << "]";
    }
    return PS;
  }
  if (AccMode == Value) {
    PS << getAccessorDataType(true, true) << " ";
  } else if (AccMode == Pointer) {
    PS << getAccessorDataType(true, true);
    if (!getType()->isPointer())
      PS << " ";
    PS << "*";
  } else if (AccMode == Reference) {
    PS << getAccessorDataType(true, true);
    if (!getType()->isPointer())
      PS << " ";
    PS << "&";
  } else if (AccMode == Accessor && isExtern() && isShared() &&
             getType()->getDimension() > 1) {
    PS << getAccessorDataType();
    PS << " *";
  } else {
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None || isShared())
      PS << getSyclAccessorType() << " ";
    else
      PS << getDpctAccessorType() << " ";
  }
  return PS << getArgName();
}
ParameterStream &MemVarInfo::getFuncArg(ParameterStream &PS) {
  return PS << getArgName();
}
ParameterStream &MemVarInfo::getKernelArg(ParameterStream &PS) {
  if (isShared() || DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
    if (AccMode == Pointer) {
      if (!getType()->isWritten())
        PS << "(" << getAccessorDataType(false, true) << " *)";
      PS << getAccessorName() << ".";
      if (getType()->isTemplate())
        PS << "template ";
      PS << "get_multi_ptr<" << MapNames::getClNamespace()
         << "access::decorated::no>().get()";
    } else if (AccMode == PointerToArray) {
      if (getType()->isWritten()) {
        PS << getAccessorName();
      } else {
        std::string CastType = getType()->getBaseName();
        auto Range = getType()->getRange();
        CastType = CastType + " (*)";
        for (size_t i = 1; i < Range.size(); i++) {
          CastType = CastType + "[" + Range[i].getSize() + "]";
        }
        PS << "reinterpret_cast<" << CastType << ">(" << getAccessorName()
           << ".template get_multi_ptr<" << MapNames::getClNamespace()
           << "access::decorated::no>().get())";
      }
    } else {
      PS << getAccessorName();
    }
  } else {
    if (AccMode == Accessor) {
      PS << getAccessorName();
    } else {
      if (AccMode == Value || AccMode == Reference) {
        PS << "*";
      }
      PS << getPtrName();
    }
  }
  return PS;
}
std::string MemVarInfo::getAccessorDataType(bool IsTypeUsedInDevFunDecl,
                                            bool NeedCheckExtraConstQualifier) {
  if (isExtern()) {
    return "uint8_t";
  } else if (isTypeDeclaredLocal()) {
    if (IsTypeUsedInDevFunDecl) {
      return "uint8_t";
    } else {
      // used in accessor decl
      return "uint8_t[sizeof(" + LocalTypeName + ")]";
    }
  }

  std::string Ret = getType()->getBaseName();
  if ((!getType()->isArray() && !getType()->isPointer()) ||
      isTreatPointerAsArray())
    return Ret;
  if (NeedCheckExtraConstQualifier && getType()->isConstantQualified()) {
    return Ret + " const";
  }
  return Ret;
}
MemVarInfo::VarAttrKind MemVarInfo::getAddressAttr(const AttrVec &Attrs) {
  VarAttrKind Attr = Host;
  for (auto VarAttr : Attrs) {
    auto Kind = VarAttr->getKind();
    if (Kind == attr::HIPManaged)
      return Managed;
    if (Kind == attr::CUDAConstant)
      return Constant;
    if (Kind == attr::CUDAShared)
      return Shared;
    if (Kind == attr::CUDADevice)
      Attr = Device;
  }
  return Attr;
}
void MemVarInfo::setInitList(const Expr *E, const VarDecl *V) {
  if (auto Ctor = dyn_cast<CXXConstructExpr>(E)) {
    if (!Ctor->getNumArgs() || Ctor->getArg(0)->isDefaultArgument())
      return;
  }
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto Beg = E->getBeginLoc();
  auto End = E->getEndLoc();
  if (Beg.isMacroID() && End.isMacroID()) {
    if (SM.getExpansionLoc(Beg) != SM.getExpansionLoc(End)) {
      if (auto IL = dyn_cast<InitListExpr>(E->IgnoreImplicitAsWritten())) {
        std::string Result;
        size_t InitsNum = IL->getNumInits();
        for (unsigned i = 0; i < InitsNum; ++i) {
          const Expr *IE = IL->getInit(i);
          Result += getStmtSpelling(IE);
          if (i != InitsNum - 1) {
            Result += ", ";
          }
        }
        InitList = "{" + Result + "}";
        return;
      }
    }
  }
  InitList = getStmtSpelling(E, V->getSourceRange());
}
std::string MemVarInfo::getMemoryType() {
  static std::string DeviceGlobalMemory =
      MapNames::getClNamespace() + "ext::oneapi::experimental::device_global";
  switch (Attr) {
  case clang::dpct::MemVarInfo::Device: {
    requestFeature(HelperFeatureEnum::device_ext);
    static std::string DeviceMemory =
        MapNames::getDpctNamespace() + "global_memory";
    if (isUseDeviceGlobal()) {
      return getMemoryType(DeviceGlobalMemory, getType());
    }
    return getMemoryType(DeviceMemory, getType());
  }
  case clang::dpct::MemVarInfo::Constant: {
    requestFeature(HelperFeatureEnum::device_ext);
    std::string ConstantMemory =
        MapNames::getDpctNamespace() + "constant_memory";
    if (isUseHelperFunc()) {
      return getMemoryType(ConstantMemory, getType());
    } else if (isUseDeviceGlobal()) {
      return getMemoryType(DeviceGlobalMemory, getType());
    }
    return getMemoryType("const ", getType());
  }
  case clang::dpct::MemVarInfo::Shared: {
    static std::string SharedMemory =
        MapNames::getDpctNamespace() + "local_memory";
    static std::string ExternSharedMemory =
        MapNames::getDpctNamespace() + "extern_local_memory";
    if (isExtern())
      return ExternSharedMemory;
    return getMemoryType(SharedMemory, getType());
  }
  case clang::dpct::MemVarInfo::Managed: {

    requestFeature(HelperFeatureEnum::device_ext);

    static std::string ManagedMemory =
        MapNames::getDpctNamespace() + "shared_memory";

    return getMemoryType(ManagedMemory, getType());
  }
  default:
    llvm::dbgs() << "[MemVarInfo::getMemoryType] Unexpected attribute.";
    return "";
  }
}
std::string MemVarInfo::getMemoryType(const std::string &MemoryType,
                                      std::shared_ptr<CtTypeInfo> VarType) {
  if (isUseHelperFunc()) {
    return buildString(MemoryType, "<", VarType->getBaseName(), ", ",
                       VarType->getDimension(), ">");
  } else if (isUseDeviceGlobal()) {
    std::string Dims;
    for (auto &D : VarType->getRange()) {
      Dims = Dims + "[" + D.getSize() + "]";
    }
    return buildString(MemoryType, "<", VarType->getBaseName(), Dims, ">");
  } else {
    return buildString(MemoryType, VarType->getBaseNameWithoutQualifiers());
  }
}
std::string MemVarInfo::getInitArguments(const std::string &MemSize,
                                         bool MustArguments) {
  if (isUseHelperFunc()) {
    if (InitList.empty())
      return getType()->getRangeArgument(MemSize, MustArguments);
    if (getType()->getDimension())
      return buildString("(", getRangeClass(),
                         getType()->getRangeArgument(MemSize, true),
                         ", " + InitList, ")");
    return buildString("(", InitList, ")");
  } else if (isUseDeviceGlobal()) {
    return InitList;
  } else {
    return InitList.empty() ? "" : buildString(" = ", InitList);
  }
}
const std::string &MemVarInfo::getMemoryAttr() {
  requestFeature(HelperFeatureEnum::device_ext);
  switch (Attr) {
  case clang::dpct::MemVarInfo::Device: {
    static std::string DeviceMemory =
        MapNames::getDpctNamespace() +
        (DpctGlobalInfo::useSYCLCompat() ? "memory_region::global" : "global");
    return DeviceMemory;
  }
  case clang::dpct::MemVarInfo::Constant: {
    static std::string ConstantMemory =
        MapNames::getDpctNamespace() + (DpctGlobalInfo::useSYCLCompat()
                                            ? "memory_region::constant"
                                            : "constant");
    return ConstantMemory;
  }
  case clang::dpct::MemVarInfo::Shared: {
    static std::string SharedMemory =
        MapNames::getDpctNamespace() +
        (DpctGlobalInfo::useSYCLCompat() ? "memory_region::local" : "local");
    return SharedMemory;
  }
  case clang::dpct::MemVarInfo::Managed: {
    static std::string ManagedMemory =
        MapNames::getDpctNamespace() +
        (DpctGlobalInfo::useSYCLCompat() ? "memory_region::shared" : "shared");
    return ManagedMemory;
  }
  default:
    llvm::dbgs() << "[MemVarInfo::getMemoryAttr] Unexpected attribute.";
    static std::string NullString;
    return NullString;
  }
}
std::string MemVarInfo::getSyclAccessorType(LocInfo LI) {
  std::string Ret;
  llvm::raw_string_ostream OS(Ret);
  if (getAttr() == MemVarInfo::VarAttrKind::Shared) {
    if (!getType()->SharedVarInfo.TypeName.empty() &&
        !LI.first.getCanonicalPath().empty() && LI.second) {
      DiagnosticsUtils::report(LI.first.getCanonicalPath().str(), LI.second,
                               Warnings::MOVE_TYPE_DEFINITION_KERNEL_FUNC, true,
                               false, getType()->SharedVarInfo.TypeName,
                               getType()->SharedVarInfo.DefinitionFuncName);
    }
    OS << MapNames::getClNamespace() << "local_accessor<";
    if (AccMode == PointerToArray) {
      OS << getType()->getBaseName();
      auto Range = getType()->getRange();
      for (size_t i = 0; i < Range.size(); i++) {
        OS << "[" << Range[i].getSize() << "]";
      }
      OS << ", 0>";
    } else {
      OS << getAccessorDataType() << ", ";
      OS << getType()->getDimension() << ">";
    }
  } else {
    OS << MapNames::getClNamespace() << "accessor<";
    OS << getAccessorDataType() << ", ";
    OS << getType()->getDimension() << ", ";

    OS << MapNames::getClNamespace() << "access_mode::";
    if (getAttr() == MemVarInfo::VarAttrKind::Constant)
      OS << "read";
    else
      OS << "read_write";
    OS << ", ";

    OS << MapNames::getClNamespace() << "access::target::";
    switch (getAttr()) {
    case VarAttrKind::Constant:
    case VarAttrKind::Device:
    case VarAttrKind::Managed:
      OS << "device";
      break;
    default:
      break;
    }

    OS << ">";
  }
  return OS.str();
}
std::string MemVarInfo::getDpctAccessorType() {
  requestFeature(HelperFeatureEnum::device_ext);
  auto Type = getType();
  return buildString(MapNames::getDpctNamespace(true), "accessor<",
                     getAccessorDataType(), ", ", getMemoryAttr(), ", ",
                     Type->getDimension(), ">");
}
std::string MemVarInfo::getArgName() {
  if (isExtern())
    return ExternVariableName;
  else if (isTypeDeclaredLocal())
    return getNameAppendSuffix();
  return getName();
}
const std::string MemVarInfo::ExternVariableName = "dpct_local";
std::unordered_map<std::string, int> MemVarInfo::AnonymousTypeDeclStmtMap;
///// class TextureTypeInfo /////
TextureTypeInfo::TextureTypeInfo(std::string &&DataType, int TexType) {
  setDataTypeAndTexType(std::move(DataType), TexType);
}
void TextureTypeInfo::setDataTypeAndTexType(std::string &&Type, int TexType) {
  DataType = std::move(Type);
  IsArray = TexType & 0xF0;
  Dimension = TexType & 0x0F;
  // The DataType won't use dpct helper feature
  MapNames::replaceName(MapNames::TypeNamesMap, DataType);
}
void TextureTypeInfo::prepareForImage() {
  if (IsArray)
    ++Dimension;
}
void TextureTypeInfo::endForImage() {
  if (IsArray)
    --Dimension;
}

ParameterStream &TextureTypeInfo::printType(ParameterStream &PS,
                                            const std::string &TemplateName) {
  PS << TemplateName << "<" << DataType << ", " << Dimension;
  if (IsArray)
    PS << ", true";
  PS << ">";
  return PS;
}
///// class TextureInfo /////
TextureInfo::TextureInfo(unsigned Offset,
                         const clang::tooling::UnifiedPath &FilePath,
                         StringRef Name)
    : FilePath(FilePath), Offset(Offset), Name(Name) {
  NewVarName = Name.str();
  for (auto &C : NewVarName) {
    if ((!isDigit(C)) && (!isLetter(C)) && (C != '_'))
      C = '_';
  }
  if (NewVarName.size() > 1 && NewVarName[NewVarName.size() - 1] == '_')
    NewVarName.pop_back();
}
TextureInfo::TextureInfo(const VarDecl *VD)
    : TextureInfo(DpctGlobalInfo::getLocInfo(
                      VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
                  VD->getName()) {}
TextureInfo::TextureInfo(const VarDecl *VD, std::string Subscript)
    : TextureInfo(DpctGlobalInfo::getLocInfo(
                      VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
                  VD->getName().str() + "[" + Subscript + "]") {}
TextureInfo::TextureInfo(
    std::pair<clang::tooling::UnifiedPath, unsigned> LocInfo, StringRef Name)
    : TextureInfo(LocInfo.second, LocInfo.first.getCanonicalPath(), Name) {}
ParameterStream &TextureInfo::getDecl(ParameterStream &PS,
                                      const std::string &TemplateDeclName) {
  return Type->printType(PS, MapNames::getDpctNamespace() + TemplateDeclName)
         << " " << Name;
}
TextureInfo::TextureInfo(unsigned Offset,
                         const clang::tooling::UnifiedPath &FilePath,
                         const VarDecl *VD)
    : TextureInfo(Offset, FilePath, VD->getName()) {
  if (auto D = dyn_cast_or_null<ClassTemplateSpecializationDecl>(
          VD->getType()->getAsCXXRecordDecl())) {
    auto &TemplateList = D->getTemplateInstantiationArgs();
    auto DataTy = TemplateList[0].getAsType();
    if (auto ET = dyn_cast<ElaboratedType>(DataTy))
      DataTy = ET->getNamedType();
    setType(DpctGlobalInfo::getUnqualifiedTypeName(DataTy),
            TemplateList[1].getAsIntegral().getExtValue());
  } else {
    auto TST = VD->getType()->getAs<TemplateSpecializationType>();
    if (TST) {
      auto Args = TST->template_arguments();
      auto Arg0 = Args[0];
      auto Arg1 = Args[1];
      if (Arg1.getKind() == clang::TemplateArgument::Expression) {
        auto DataTy = Arg0.getAsType();
        if (auto ET = dyn_cast<ElaboratedType>(DataTy))
          DataTy = ET->getNamedType();
        Expr::EvalResult ER;
        if (!Arg1.getAsExpr()->isValueDependent() &&
            Arg1.getAsExpr()->EvaluateAsInt(ER, DpctGlobalInfo::getContext())) {
          int64_t Value = ER.Val.getInt().getExtValue();
          setType(DpctGlobalInfo::getUnqualifiedTypeName(DataTy), Value);
        }
      }
    }
  }
}
void TextureInfo::setType(std::string &&DataType, int TexType) {
  setType(std::make_shared<TextureTypeInfo>(std::move(DataType), TexType));
}
void TextureInfo::setType(std::shared_ptr<TextureTypeInfo> TypeInfo) {
  if (TypeInfo)
    Type = TypeInfo;
}
std::string TextureInfo::getHostDeclString() {
  ParameterStream PS;
  Type->prepareForImage();
  requestFeature(HelperFeatureEnum::device_ext);
  getDecl(PS, DpctGlobalInfo::useExtBindlessImages()
                  ? "experimental::bindless_image_wrapper"
                  : "image_wrapper")
      << ";";
  Type->endForImage();
  return PS.Str;
}
std::string TextureInfo::getSamplerDecl() {
  requestFeature(HelperFeatureEnum::device_ext);
  return buildString("auto ", NewVarName, "_smpl = ", Name, ".get_sampler();");
}
std::string TextureInfo::getAccessorDecl(const std::string &QueueStr) {
  requestFeature(HelperFeatureEnum::device_ext);
  std::string Ret;
  llvm::raw_string_ostream OS(Ret);
  OS << "auto " << NewVarName << "_acc = " << Name << ".get_access(cgh";
  printQueueStr(OS, QueueStr);
  OS << ");";
  return Ret;
}
std::string TextureInfo::InitDecl(const std::string &QueueStr) {
  ParameterStream PS;
  PS << Name << ".create_image(" << QueueStr << ");";
  return PS.Str;
}
void TextureInfo::addDecl(StmtList &InitList, StmtList &AccessorList,
                          StmtList &SamplerList, const std::string &QueueStr) {
  if (DpctGlobalInfo::useExtBindlessImages()) {
    AccessorList.emplace_back("auto " + NewVarName + "_handle = " + Name +
                              ".get_handle();");
    return;
  }
  InitList.emplace_back(InitDecl(QueueStr));
  AccessorList.emplace_back(getAccessorDecl(QueueStr));
  SamplerList.emplace_back(getSamplerDecl());
}
ParameterStream &TextureInfo::getFuncDecl(ParameterStream &PS) {
  if (DpctGlobalInfo::useExtBindlessImages()) {
    PS << MapNames::getClNamespace()
       << "ext::oneapi::experimental::sampled_image_handle " << Name;
    return PS;
  }
  requestFeature(HelperFeatureEnum::device_ext);
  return getDecl(PS, "image_accessor_ext");
}
ParameterStream &TextureInfo::getFuncArg(ParameterStream &PS) {
  return PS << Name;
}
ParameterStream &TextureInfo::getKernelArg(ParameterStream &OS) {
  requestFeature(HelperFeatureEnum::device_ext);
  if (DpctGlobalInfo::useExtBindlessImages()) {
    OS << NewVarName << "_handle";
    return OS;
  }
  getType()->printType(OS, MapNames::getDpctNamespace() + "image_accessor_ext");
  OS << "(" << NewVarName << "_smpl, " << NewVarName << "_acc)";
  return OS;
}
///// class TextureObjectInfo /////
std::string TextureObjectInfo::getAccessorDecl(const std::string &QueueString) {
  ParameterStream PS;
  PS << "auto " << NewVarName << "_acc = static_cast<";
  getType()->printType(PS, MapNames::getDpctNamespace() + "image_wrapper")
      << " *>(" << Name << ")->get_access(cgh";
  printQueueStr(PS, QueueString);
  PS << ");";
  requestFeature(HelperFeatureEnum::device_ext);
  return PS.Str;
}
std::string TextureObjectInfo::InitDecl(const std::string &QueueStr) {
  ParameterStream PS;
  PS << "static_cast<";
  getType()->printType(PS, MapNames::getDpctNamespace() + "image_wrapper")
      << " *>(" << Name << ")->create_image(" << QueueStr << ");";
  return PS.Str;
}
std::string TextureObjectInfo::getSamplerDecl() {
  requestFeature(HelperFeatureEnum::device_ext);
  return buildString("auto ", NewVarName, "_smpl = ", Name, "->get_sampler();");
}
std::string TextureObjectInfo::getParamDeclType() {
  requestFeature(HelperFeatureEnum::device_ext);
  ParameterStream PS;
  Type->printType(PS, MapNames::getDpctNamespace() + "image_accessor_ext");
  return PS.Str;
}
void TextureObjectInfo::merge(std::shared_ptr<TextureObjectInfo> Target) {
  if (Target)
    setType(Target->getType());
}
void TextureObjectInfo::addParamDeclReplacement() {
  if (Type) {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, Offset, ReplaceTypeLength,
                                         getParamDeclType(), nullptr));
  }
}

// class TempStorageVarInfo //
void TempStorageVarInfo::addAccessorDecl(StmtList &AccessorList,
                                         StringRef LocalSize) const {
  std::string Accessor;
  llvm::raw_string_ostream OS(Accessor);
  switch (Kind) {
  case BlockReduce:
    OS << MapNames::getClNamespace() << "local_accessor<std::byte, 1> " << Name
       << "_acc(";
    DpctGlobalInfo::printCtadClass(OS, MapNames::getClNamespace() + "range", 1);
    OS << '(' << LocalSize << ".size() * sizeof("
       << ValueType->getSourceString() << ')' << ')';
    break;
  case BlockShuffle:
  case BlockRadixSort:
    OS << MapNames::getClNamespace() << "local_accessor<uint8_t, 1> " << Name
       << "_acc(";
    OS << TmpMemSizeCalFn << '(' << LocalSize << ".size()" << ')';
    break;
  }

  OS << ", cgh);";
  AccessorList.emplace_back(Accessor);
}
void TempStorageVarInfo::applyTemplateArguments(
    const std::vector<TemplateArgumentInfo> &TAList) {
  ValueType = ValueType->applyTemplateArguments(TAList);
}
ParameterStream &TempStorageVarInfo::getFuncDecl(ParameterStream &PS) {
  switch (Kind) {
  case BlockReduce:
    PS << MapNames::getClNamespace() << "local_accessor<std::byte, 1> ";
    break;
  case BlockShuffle:
  case BlockRadixSort:
    PS << "uint8_t *";
    break;
  }
  return PS << Name;
}
ParameterStream &TempStorageVarInfo::getFuncArg(ParameterStream &PS) {
  return PS << Name;
}
ParameterStream &TempStorageVarInfo::getKernelArg(ParameterStream &PS) {
  if (Kind == BlockReduce)
    return PS << Name << "_acc";
  return PS << "&" << Name << "_acc[0]";
}
///// class CudaLaunchTextureObjectInfo /////
std::string
CudaLaunchTextureObjectInfo::getAccessorDecl(const std::string &QueueString) {
  requestFeature(HelperFeatureEnum::device_ext);
  ParameterStream PS;
  PS << "auto " << Name << "_acc = static_cast<";
  getType()->printType(PS, MapNames::getDpctNamespace() + "image_wrapper")
      << " *>(" << ArgStr << ")->get_access(cgh";
  printQueueStr(PS, QueueString);
  PS << ");";
  return PS.Str;
}
std::string CudaLaunchTextureObjectInfo::getSamplerDecl() {
  requestFeature(HelperFeatureEnum::device_ext);
  return buildString("auto ", Name, "_smpl = (", ArgStr, ")->get_sampler();");
}
///// class MemberTextureObjectInfo /////
MemberTextureObjectInfo::NewVarNameRAII::NewVarNameRAII(
    MemberTextureObjectInfo *M)
    : OldName(std::move(M->Name)), Member(M) {
  Member->Name = buildString(M->BaseName, '.', M->MemberName);
}
std::shared_ptr<MemberTextureObjectInfo>
MemberTextureObjectInfo::create(const MemberExpr *ME) {
  auto LocInfo = DpctGlobalInfo::getLocInfo(ME);
  auto Ret =
      std::shared_ptr<MemberTextureObjectInfo>(new MemberTextureObjectInfo(
          LocInfo.second, LocInfo.first, getTempNameForExpr(ME, false, false)));
  Ret->MemberName = ME->getMemberDecl()->getNameAsString();
  return Ret;
}
void MemberTextureObjectInfo::addDecl(StmtList &InitList,
                                      StmtList &AccessorList,
                                      StmtList &SamplerList,
                                      const std::string &QueueStr) {
  NewVarNameRAII RAII(this);
  TextureObjectInfo::addDecl(InitList, AccessorList, SamplerList, QueueStr);
}
///// class StructureTextureObjectInfo /////
StructureTextureObjectInfo::StructureTextureObjectInfo(const ParmVarDecl *PVD)
    : TextureObjectInfo(PVD) {
  ContainsVirtualPointer =
      checkPointerInStructRecursively(getRecordDecl(PVD->getType()));
  setType("", 0);
}
StructureTextureObjectInfo::StructureTextureObjectInfo(const VarDecl *VD)
    : TextureObjectInfo(VD) {
  ContainsVirtualPointer =
      checkPointerInStructRecursively(getRecordDecl(VD->getType()));
  setType("", 0);
}
std::shared_ptr<StructureTextureObjectInfo>
StructureTextureObjectInfo::create(const CXXThisExpr *This) {
  auto RD = getRecordDecl(This->getType());
  if (!RD)
    return nullptr;

  auto LocInfo = DpctGlobalInfo::getLocInfo(RD);

  auto Ret = std::shared_ptr<StructureTextureObjectInfo>(
      new StructureTextureObjectInfo(LocInfo.second, LocInfo.first,
                                     RD->getName()));
  Ret->ContainsVirtualPointer = checkPointerInStructRecursively(RD);
  Ret->IsBase = true;
  Ret->setType("", 0);
  return Ret;
}
std::shared_ptr<MemberTextureObjectInfo>
StructureTextureObjectInfo::addMember(const MemberExpr *ME) {
  auto Member = MemberTextureObjectInfo::create(ME);
  return Members.emplace(Member->getMemberName().str(), Member).first->second;
}
void StructureTextureObjectInfo::addDecl(StmtList &InitList,
                                         StmtList &AccessorList,
                                         StmtList &SamplerList,
                                         const std::string &Queue) {
  for (const auto &M : Members) {
    M.second->setBaseName(Name);
  }
}
void StructureTextureObjectInfo::merge(
    std::shared_ptr<StructureTextureObjectInfo> Target) {
  if (!Target)
    return;

  dpct::merge(Members, Target->Members);
}
void StructureTextureObjectInfo::merge(
    std::shared_ptr<TextureObjectInfo> Target) {
  merge(std::dynamic_pointer_cast<StructureTextureObjectInfo>(Target));
}
ParameterStream &StructureTextureObjectInfo::getKernelArg(ParameterStream &OS) {
  OS << Name;
  return OS;
}
///// class TemplateArgumentInfo /////
TemplateArgumentInfo::TemplateArgumentInfo(const TemplateArgumentLoc &TAL,
                                           SourceRange Range)
    : Kind(TAL.getArgument().getKind()) {
  setArgFromExprAnalysis(TAL,
                         getDefinitionRange(Range.getBegin(), Range.getEnd()));
}
TemplateArgumentInfo::TemplateArgumentInfo(std::string &&Str)
    : Kind(TemplateArgument::Null) {
  setArgStr(std::move(Str));
}
std::shared_ptr<const TemplateDependentStringInfo>
TemplateArgumentInfo::getDependentStringInfo() const {
  if (isNull()) {
    static std::shared_ptr<TemplateDependentStringInfo> Placeholder =
        std::make_shared<TemplateDependentStringInfo>(
            "dpct_placeholder/*Fix the type mannually*/");
    return Placeholder;
  }
  return DependentStr;
}
void TemplateArgumentInfo::setAsType(QualType QT) {
  if (isPlaceholderType(QT))
    return;
  setArgStr(DpctGlobalInfo::getReplacedTypeName(QT));
  Kind = TemplateArgument::Type;
}
void TemplateArgumentInfo::setAsType(const TypeLoc &TL) {
  setArgFromExprAnalysis(TL);
  Kind = TemplateArgument::Type;
}
void TemplateArgumentInfo::setAsType(std::string TS) {
  setArgStr(std::move(TS));
  Kind = TemplateArgument::Type;
}
void TemplateArgumentInfo::setAsNonType(const llvm::APInt &Int) {
  setArgStr(toString(Int, 10, true, false));
  Kind = TemplateArgument::Integral;
}
void TemplateArgumentInfo::setAsNonType(const Expr *E) {
  setArgFromExprAnalysis(E);
  Kind = TemplateArgument::Expression;
}
bool TemplateArgumentInfo::isPlaceholderType(QualType QT) {
  if (auto BT = QT->getAs<BuiltinType>()) {
    if (BT->isPlaceholderType() || BT->isDependentType())
      return true;
  }
  return false;
}
template <class T>
void TemplateArgumentInfo::setArgFromExprAnalysis(const T &Arg,
                                                  SourceRange ParentRange) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto Range = getArgSourceRange(Arg);
  auto Begin = Range.getBegin();
  auto End = Range.getEnd();
  if (Begin.isMacroID() && End.isMacroID()) {
    size_t Length;
    if (ParentRange.isValid()) {
      auto RR =
          getRangeInRange(Range, ParentRange.getBegin(), ParentRange.getEnd());
      Begin = RR.first;
      End = RR.second;
      Length = SM.getCharacterData(End) - SM.getCharacterData(Begin);
    } else {
      auto RR = getDefinitionRange(Range.getBegin(), Range.getEnd());
      Begin = RR.getBegin();
      End = RR.getEnd();
      Length = SM.getCharacterData(End) - SM.getCharacterData(Begin) +
               Lexer::MeasureTokenLength(
                   End, SM, DpctGlobalInfo::getContext().getLangOpts());
    }
    std::string Result = std::string(SM.getCharacterData(Begin), Length);
    setArgStr(std::move(Result));
  } else {
    ExprAnalysis EA;
    EA.analyze(Arg);
    DependentStr = EA.getTemplateDependentStringInfo();
  }
}
///// class MemVarMap /////
void MemVarMap::addTexture(std::shared_ptr<TextureInfo> Tex) {
  TextureMap.insert(std::make_pair(Tex->getOffset(), Tex));
}
void MemVarMap::addCUBTempStorage(std::shared_ptr<TempStorageVarInfo> Tmp) {
  TempStorageMap.insert(std::make_pair(Tmp->getOffset(), Tmp));
}
void MemVarMap::addVar(std::shared_ptr<MemVarInfo> Var) {
  auto Attr = Var->getAttr();
  if (Var->isGlobal() && (Attr == MemVarInfo::VarAttrKind::Device ||
                          Attr == MemVarInfo::VarAttrKind::Managed)) {
    setGlobalMemAcc(true);
  }
  getMap(Var->getScope())
      .insert(MemVarInfoMap::value_type(Var->getOffset(), Var));
}
void MemVarMap::merge(const MemVarMap &OtherMap) {
  static std::vector<TemplateArgumentInfo> NullTemplates;
  return merge(OtherMap, NullTemplates);
}
void MemVarMap::merge(const MemVarMap &VarMap,
                      const std::vector<TemplateArgumentInfo> &TemplateArgs) {
  setItem(hasItem() || VarMap.hasItem());
  setStream(hasStream() || VarMap.hasStream());
  setSync(hasSync() || VarMap.hasSync());
  setBF64(hasBF64() || VarMap.hasBF64());
  setBF16(hasBF16() || VarMap.hasBF16());
  setGlobalMemAcc(hasGlobalMemAcc() || VarMap.hasGlobalMemAcc());
  merge(LocalVarMap, VarMap.LocalVarMap, TemplateArgs);
  merge(GlobalVarMap, VarMap.GlobalVarMap, TemplateArgs);
  merge(ExternVarMap, VarMap.ExternVarMap, TemplateArgs);
  merge(TempStorageMap, VarMap.TempStorageMap, TemplateArgs);
  dpct::merge(TextureMap, VarMap.TextureMap);
}
int MemVarMap::calculateExtraArgsSize() const {
  int Size = 0;
  if (hasStream())
    Size += MapNamesLang::KernelArgTypeSizeMap.at(KernelArgType::KAT_Stream);

  Size = Size + calculateExtraArgsSize(LocalVarMap) +
         calculateExtraArgsSize(GlobalVarMap) +
         calculateExtraArgsSize(ExternVarMap);
  Size = Size + TextureMap.size() * MapNamesLang::KernelArgTypeSizeMap.at(
                                        KernelArgType::KAT_Texture);

  return Size;
}
template <MemVarMap::CallOrDecl COD>
inline std::string
MemVarMap::getArgumentsOrParameters(int PreParams, int PostParams, LocInfo LI,
                                    FormatInfo FormatInformation) const {
  ParameterStream PS;
  if (PreParams != 0)
    PS << ", ";
  if (hasItem())
    getItem<COD>(PS) << ", ";
  if (hasStream())
    getStream<COD>(PS) << ", ";
  if (hasSync())
    getSync<COD>(PS) << ", ";
  if (!ExternVarMap.empty())
    GetArgOrParam<MemVarInfo, COD>()(PS, ExternVarMap.begin()->second) << ", ";
  getArgumentsOrParametersFromMap<MemVarInfo, COD>(PS, GlobalVarMap, LI);
  getArgumentsOrParametersFromMap<MemVarInfo, COD>(PS, LocalVarMap, LI);
  getArgumentsOrParametersFromoTextureInfoMap<COD>(PS, TextureMap);
  getArgumentsOrParametersFromMap<TempStorageVarInfo, COD>(PS, TempStorageMap);
  std::string Result = PS.Str;
  return (Result.empty() || PostParams != 0) && PreParams == 0
             ? Result
             : Result.erase(Result.size() - 2, 2);
}
template <>
std::string MemVarMap::getArgumentsOrParameters<MemVarMap::DeclParameter>(
    int PreParams, int PostParams, LocInfo LI,
    FormatInfo FormatInformation) const {
  ParameterStream PS;
  if (DpctGlobalInfo::getFormatRange() != clang::format::FormatRange::none) {
    PS = ParameterStream(FormatInformation,
                         DpctGlobalInfo::getCodeFormatStyle().ColumnLimit);
  } else {
    PS = ParameterStream(FormatInformation, 80);
  }
  getArgumentsOrParametersForDecl(PS, PreParams, PostParams, LI);
  std::string Result = PS.Str;

  if (Result.empty())
    return Result;

  // Remove pre splitter
  unsigned int RemoveLength = 0;
  if (FormatInformation.IsFirstArg) {
    if (FormatInformation.IsAllParamsOneLine) {
      // comma and space
      RemoveLength = 2;
    } else {
      // calculate length from the first character "," to the next none space
      // character
      RemoveLength = 1;
      while (RemoveLength < Result.size()) {
        if (!isspace(Result[RemoveLength]))
          break;
        RemoveLength++;
      }
    }
    Result = Result.substr(RemoveLength, Result.size() - RemoveLength);
  }

  // Add post splitter
  RemoveLength = 0;
  if (PostParams != 0 && PreParams == 0) {
    Result = Result + ", ";
  }

  return Result;
}
std::string MemVarMap::getExtraCallArguments(bool HasPreParam,
                                             bool HasPostParam) const {
  return getArgumentsOrParameters<CallArgument>(HasPreParam, HasPostParam);
}
void MemVarMap::requestFeatureForAllVarMaps(
    const clang::tooling::UnifiedPath &Path) const {
  for (const auto &Item : LocalVarMap) {
    Item.second->requestFeatureForSet(Path);
  }
  for (const auto &Item : GlobalVarMap) {
    Item.second->requestFeatureForSet(Path);
  }
  for (const auto &Item : ExternVarMap) {
    Item.second->requestFeatureForSet(Path);
  }
}
std::string MemVarMap::getExtraDeclParam(bool HasPreParam, bool HasPostParam,
                                         LocInfo LI,
                                         FormatInfo FormatInformation) const {
  return getArgumentsOrParameters<DeclParameter>(HasPreParam, HasPostParam, LI,
                                                 FormatInformation);
}
std::string
MemVarMap::getKernelArguments(bool HasPreParam, bool HasPostParam,
                              const clang::tooling::UnifiedPath &Path) const {
  requestFeatureForAllVarMaps(Path);
  return getArgumentsOrParameters<KernelArgument>(HasPreParam, HasPostParam);
}
const MemVarInfoMap &MemVarMap::getMap(MemVarInfo::VarScope Scope) const {
  return const_cast<MemVarMap *>(this)->getMap(Scope);
}
const GlobalMap<TextureInfo> &MemVarMap::getTextureMap() const {
  return TextureMap;
}
const GlobalMap<TempStorageVarInfo> &MemVarMap::getTempStorageMap() const {
  return TempStorageMap;
}
void MemVarMap::removeDuplicateVar() {
  std::unordered_set<std::string> VarNames{getItemName(),
                                           DpctGlobalInfo::getStreamName()};
  dpct::removeDuplicateVar(GlobalVarMap, VarNames);
  dpct::removeDuplicateVar(LocalVarMap, VarNames);
  dpct::removeDuplicateVar(ExternVarMap, VarNames);
  dpct::removeDuplicateVar(TextureMap, VarNames);
  dpct::removeDuplicateVar(TempStorageMap, VarNames);
}
MemVarInfoMap &MemVarMap::getMap(MemVarInfo::VarScope Scope) {
  switch (Scope) {
  case clang::dpct::MemVarInfo::Local:
    return LocalVarMap;
  case clang::dpct::MemVarInfo::Extern:
    return ExternVarMap;
  case clang::dpct::MemVarInfo::Global:
    return GlobalVarMap;
  }
  clang::dpct::DpctDebugs()
      << "[MemVarInfo::VarScope] Unexpected value: " << Scope << "\n";
  assert(0);
  static MemVarInfoMap InvalidMap;
  return InvalidMap;
}
bool MemVarMap::isSameAs(const MemVarMap &Other) const {
  if (HasItem != Other.HasItem)
    return false;
  if (HasStream != Other.HasStream)
    return false;
  if (HasSync != Other.HasSync)
    return false;

#define COMPARE_MAP(MAP)                                                       \
  {                                                                            \
    if (MAP.size() != Other.MAP.size())                                        \
      return false;                                                            \
    if (!std::equal(MAP.begin(), MAP.end(), Other.MAP.begin()))                \
      return false;                                                            \
  }
  COMPARE_MAP(LocalVarMap);
  COMPARE_MAP(GlobalVarMap);
  COMPARE_MAP(ExternVarMap);
  COMPARE_MAP(TextureMap);
#undef COMPARE_MAP
  return true;
}
const MemVarMap *
MemVarMap::getHeadWithoutPathCompression(const MemVarMap *CurNode) {
  if (!CurNode)
    return nullptr;
  const MemVarMap *Head = nullptr;
  while (true) {
    if (CurNode->Parent == CurNode) {
      Head = CurNode;
      break;
    }
    CurNode = CurNode->Parent;
  }
  return Head;
}
MemVarMap *MemVarMap::getHead(MemVarMap *CurNode) {
  if (!CurNode)
    return nullptr;
  MemVarMap *Head =
      const_cast<MemVarMap *>(getHeadWithoutPathCompression(CurNode));
  if (!Head)
    return nullptr;
  while (CurNode != Head) {
    MemVarMap *Temp = CurNode->Parent;
    CurNode->Parent = Head;
    CurNode = Temp;
  }
  return Head;
}
unsigned int MemVarMap::getHeadNodeDim() const {
  auto Ptr = getHeadWithoutPathCompression(this);
  if (Ptr)
    return Ptr->Dim;
  else
    return 3;
}
int MemVarMap::calculateExtraArgsSize(const MemVarInfoMap &Map) const {
  int Size = 0;
  for (auto &VarInfoPair : Map) {
    auto D = VarInfoPair.second->getType()->getDimension();
    Size += MapNamesLang::getArrayTypeSize(D);
  }
  return Size;
}
template <class T, MemVarMap::CallOrDecl COD>
void MemVarMap::getArgumentsOrParametersFromMap(ParameterStream &PS,
                                                const GlobalMap<T> &VarMap,
                                                LocInfo LI) {
  for (const auto &VI : VarMap) {
    if constexpr (!std::is_same_v<T, TempStorageVarInfo>) {
      if (!VI.second->isUseHelperFunc()) {
        continue;
      }
      if (!VI.second->getType()->SharedVarInfo.TypeName.empty() &&
          !LI.first.getCanonicalPath().empty() && LI.second) {
        DiagnosticsUtils::report(
            LI.first.getCanonicalPath().str(), LI.second,
            Warnings::MOVE_TYPE_DEFINITION_DEVICE_FUNC, true, false,
            VI.second->getType()->SharedVarInfo.TypeName,
            VI.second->getType()->SharedVarInfo.DefinitionFuncName);
      }
    }
    if (PS.FormatInformation.EnableFormat) {
      ParameterStream TPS;
      GetArgOrParam<T, COD>()(TPS, VI.second);
      PS << TPS.Str;
    } else {
      GetArgOrParam<T, COD>()(PS, VI.second) << ", ";
    }
  }
}

template <MemVarMap::CallOrDecl COD>
void MemVarMap::getArgumentsOrParametersFromoTextureInfoMap(
    ParameterStream &PS, const GlobalMap<TextureInfo> &VarMap) {
  for (const auto &VI : VarMap) {
    if (PS.FormatInformation.EnableFormat) {
      ParameterStream TPS;
      GetArgOrParam<TextureInfo, COD>()(TPS, VI.second);
      PS << TPS.Str;
    } else {
      GetArgOrParam<TextureInfo, COD>()(PS, VI.second) << ", ";
    }
  }
}

void MemVarMap::getArgumentsOrParametersForDecl(ParameterStream &PS,
                                                int PreParams, int PostParams,
                                                LocInfo LI) const {
  if (hasItem()) {
    getItem<MemVarMap::DeclParameter>(PS);
  }

  if (hasStream()) {
    getStream<MemVarMap::DeclParameter>(PS);
  }

  if (hasSync()) {
    getSync<MemVarMap::DeclParameter>(PS);
  }

  if (!ExternVarMap.empty()) {
    ParameterStream TPS;
    GetArgOrParam<MemVarInfo, MemVarMap::DeclParameter>()(
        TPS, ExternVarMap.begin()->second);
    PS << TPS.Str;
  }

  getArgumentsOrParametersFromMap<MemVarInfo, MemVarMap::DeclParameter>(
      PS, GlobalVarMap, LI);
  getArgumentsOrParametersFromMap<MemVarInfo, MemVarMap::DeclParameter>(
      PS, LocalVarMap, LI);
  getArgumentsOrParametersFromoTextureInfoMap<MemVarMap::DeclParameter>(
      PS, TextureMap);
  getArgumentsOrParametersFromMap<TempStorageVarInfo, MemVarMap::DeclParameter>(
      PS, TempStorageMap);
}
///// class CallFunctionExpr /////
void CallFunctionExpr::buildCallExprInfo(const CXXConstructExpr *Ctor) {
  if (!Ctor)
    return;
  if (Ctor->getParenOrBraceRange().isInvalid())
    return;

  buildTextureObjectArgsInfo(Ctor);

  auto CtorDecl = Ctor->getConstructor();
  Name = getName(CtorDecl);
  setFuncInfo(DeviceFunctionDecl::LinkRedecls(CtorDecl));
  IsAllTemplateArgsSpecified =
      deduceTemplateArguments(Ctor, CtorDecl, TemplateArgs);

  SourceLocation InsertLocation;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto Info = getFuncInfo();
  if (Info) {
    if (Info->NonDefaultParamNum) {
      if (Ctor->getNumArgs() >= Info->NonDefaultParamNum) {
        InsertLocation =
            Ctor->getArg(Info->NonDefaultParamNum - 1)->getEndLoc();
      } else {
        ExtraArgLoc = 0;
        return;
      }
    } else {
      InsertLocation = Ctor->getParenOrBraceRange().getBegin();
    }
  }
  ExtraArgLoc = SM.getFileOffset(Lexer::getLocForEndOfToken(
      getActualInsertLocation(InsertLocation, SM,
                              DpctGlobalInfo::getContext().getLangOpts()),
      0, SM, DpctGlobalInfo::getContext().getLangOpts()));
}

void CallFunctionExpr::buildCallExprInfo(const CallExpr *CE) {
  if (!CE)
    return;
  CallFuncExprOffset = DpctGlobalInfo::getLocInfo(CE->getBeginLoc()).second;
  buildCalleeInfo(CE->getCallee()->IgnoreParenImpCasts(), CE->getNumArgs());
  buildTextureObjectArgsInfo(CE);
  bool HasImplicitArg = false;
  if (auto FD = CE->getDirectCallee()) {
    IsAllTemplateArgsSpecified = deduceTemplateArguments(CE, FD, TemplateArgs);
    HasImplicitArg = isa<CXXOperatorCallExpr>(CE) && isa<CXXMethodDecl>(FD);
  } else if (auto Unresolved = dyn_cast<UnresolvedLookupExpr>(
                 CE->getCallee()->IgnoreImplicitAsWritten())) {
    for (const auto &D : Unresolved->decls()) {
      IsAllTemplateArgsSpecified = deduceTemplateArguments(CE, D, TemplateArgs);
      if (IsAllTemplateArgsSpecified)
        break;
    }
  } else if (isa<CXXDependentScopeMemberExpr>(
                 CE->getCallee()->IgnoreImplicitAsWritten())) {
    // Un-instantiate member call. Cannot analyze related method declaration.
    return;
  }

  if (HasImplicitArg) {
    HasArgs = CE->getNumArgs() == 1;
  } else {
    HasArgs = CE->getNumArgs();
  }
  auto Info = getFuncInfo();
  if (Info) {
    if ((Info->getOverloadedOperatorKind() !=
         OverloadedOperatorKind::OO_None) &&
        (Info->getOverloadedOperatorKind() !=
         OverloadedOperatorKind::OO_Call)) {
      return;
    }
    if (Info->ParamsNum == 0) {
      ExtraArgLoc =
          DpctGlobalInfo::getSourceManager().getFileOffset(CE->getRParenLoc());
    } else if (Info->NonDefaultParamNum == 0) {
      // if all params have default value
      if (CE->getNumArgs()) {
        ExtraArgLoc = DpctGlobalInfo::getSourceManager().getFileOffset(
            CE->getArg(HasImplicitArg ? 1 : 0)->getBeginLoc());
      } else {
        ExtraArgLoc = DpctGlobalInfo::getSourceManager().getFileOffset(
            CE->getRParenLoc());
      }
    } else {
      // if some params have default value, set ExtraArgLoc to the location
      // before the comma
      if (CE->getNumArgs() > Info->NonDefaultParamNum - 1) {
        auto &SM = DpctGlobalInfo::getSourceManager();
        auto CERange = getDefinitionRange(CE->getBeginLoc(), CE->getEndLoc());
        auto TempLoc = Lexer::getLocForEndOfToken(
            CERange.getEnd(), 0, SM,
            DpctGlobalInfo::getContext().getLangOpts());
        auto PairRange = getRangeInRange(
            CE->getArg(Info->NonDefaultParamNum - 1 + HasImplicitArg),
            CERange.getBegin(), TempLoc);
        auto RealEnd = PairRange.second;
        auto IT = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
            getCombinedStrFromLoc(RealEnd));
        if (IT !=
                dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
            IT->second->TokenIndex == IT->second->NumTokens) {
          RealEnd =
              SM.getImmediateExpansionRange(
                    CE->getArg(Info->NonDefaultParamNum - 1 + HasImplicitArg)
                        ->getEndLoc())
                  .getEnd();
          RealEnd = Lexer::getLocForEndOfToken(
              RealEnd, 0, SM, DpctGlobalInfo::getContext().getLangOpts());
          IT = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
              getCombinedStrFromLoc(RealEnd));
        }
        while (
            IT !=
                dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
            RealEnd.isMacroID() &&
            IT->second->TokenIndex == IT->second->NumTokens) {
          RealEnd = SM.getImmediateExpansionRange(RealEnd).getEnd();
          RealEnd = Lexer::getLocForEndOfToken(
              RealEnd, 0, SM, DpctGlobalInfo::getContext().getLangOpts());
          IT = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
              getCombinedStrFromLoc(RealEnd));
        }

        ExtraArgLoc = DpctGlobalInfo::getSourceManager().getFileOffset(RealEnd);
      } else {
        ExtraArgLoc = 0;
      }
    }
  }
}
void CallFunctionExpr::emplaceReplacement() {
  buildInfo();
  if (IsADLEnable)
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, CallFuncExprOffset, 0,
                                         "::", nullptr));
  if (ExtraArgLoc)
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, ExtraArgLoc, 0,
                                         getExtraArguments(), nullptr));
}
bool CallFunctionExpr::hasWrittenTemplateArgs() {
  for (auto &Arg : TemplateArgs)
    if (!Arg.isNull() && Arg.isWritten())
      return true;
  return false;
}
std::string CallFunctionExpr::getTemplateArguments(bool &IsNeedWarning,
                                                   bool WrittenArgsOnly,
                                                   bool WithScalarWrapped) {
  IsNeedWarning = false;
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  for (auto &TA : TemplateArgs) {
    if ((TA.isNull() || !TA.isWritten()) && WrittenArgsOnly)
      continue;
    std::string Str = TA.getString();
    if (TA.isNull() && !Str.empty()) {
      IsNeedWarning = true;
    }
    if (WithScalarWrapped && (!TA.isType() && !TA.isNull())) {
      appendString(OS, "dpct_kernel_scalar<", Str, ">, ");
      requestFeature(HelperFeatureEnum::device_ext);
    } else {
      // This code path is used to process code like:
      // my_kernel<<<1, 1>>>([=] __device__(int idx) { idx++; });
      // When generating kernel name for "my_kernel", the type of this lambda
      // expr is "lambda at FilePath:Row:Col", which will cause compiling
      // failure. Current solution: use the location's hash value as its type.
      StringRef StrRef(Str);
      if (StrRef.starts_with("(lambda at")) {
        Str = "class lambda_" + getHashAsString(Str).substr(0, 6);
      }
      appendString(OS, Str, ", ");
    }
  }
  OS.flush();
  return (Result.empty()) ? Result : Result.erase(Result.size() - 2);
}
std::string CallFunctionExpr::getExtraArguments() {
  auto Info = getFuncInfo();
  if (!Info)
    return "";
  return getVarMap().getExtraCallArguments(
      Info->NonDefaultParamNum, Info->ParamsNum - Info->NonDefaultParamNum);
}
std::shared_ptr<TextureObjectInfo> CallFunctionExpr::addTextureObjectArgInfo(
    unsigned ArgIdx, std::shared_ptr<TextureObjectInfo> Info) {
  auto &Obj = TextureObjectList[ArgIdx];
  if (!Obj)
    Obj = Info;
  return Obj;
}
std::shared_ptr<TextureObjectInfo> CallFunctionExpr::addTextureObjectArg(
    unsigned ArgIdx, const DeclRefExpr *TexRef, bool isKernelCall) {
  std::shared_ptr<TextureObjectInfo> Info;
  if (TextureObjectInfo::isTextureObject(TexRef)) {
    Info = makeTextureObjectInfo<TextureObjectInfo>(TexRef->getDecl(),
                                                    isKernelCall);
  } else if (TexRef->getType()->isRecordType()) {
    Info = makeTextureObjectInfo<StructureTextureObjectInfo>(TexRef->getDecl(),
                                                             isKernelCall);
  }
  if (Info)
    return addTextureObjectArgInfo(ArgIdx, Info);
  return Info;
}
std::shared_ptr<TextureObjectInfo>
CallFunctionExpr::addStructureTextureObjectArg(unsigned ArgIdx,
                                               const MemberExpr *TexRef,
                                               bool isKernelCall) {
  if (auto DRE = dyn_cast<DeclRefExpr>(TexRef->getBase())) {
    if (auto Info = std::dynamic_pointer_cast<StructureTextureObjectInfo>(
            addTextureObjectArg(ArgIdx, DRE, isKernelCall))) {
      return Info->addMember(TexRef);
    }
  } else if (auto This = dyn_cast<CXXThisExpr>(TexRef->getBase())) {
    auto ThisObj = StructureTextureObjectInfo::create(This);
    if (ThisObj) {
      BaseTextureObject = std::move(ThisObj);
      return BaseTextureObject->addMember(TexRef);
    }
  }
  return {};
}
std::shared_ptr<TextureObjectInfo> CallFunctionExpr::addTextureObjectArg(
    unsigned ArgIdx, const ArraySubscriptExpr *TexRef, bool isKernelCall) {
  if (TextureObjectInfo::isTextureObject(TexRef)) {
    if (auto Base =
            dyn_cast<DeclRefExpr>(TexRef->getBase()->IgnoreImpCasts())) {
      if (isKernelCall) {
        if (auto VD = dyn_cast<VarDecl>(Base->getDecl())) {
          return addTextureObjectArgInfo(
              ArgIdx, std::make_shared<TextureObjectInfo>(
                          VD, ExprAnalysis::ref(TexRef->getIdx())));
        }
      } else if (auto PVD = dyn_cast<ParmVarDecl>(Base->getDecl())) {
        return addTextureObjectArgInfo(
            ArgIdx, std::make_shared<TextureObjectInfo>(
                        PVD, ExprAnalysis::ref(TexRef->getIdx())));
      }
    }
  }
  return std::shared_ptr<TextureObjectInfo>();
}
void CallFunctionExpr::setFuncInfo(std::shared_ptr<DeviceFunctionInfo> Info) {
  if (!Info) {
    return;
  }
  if (std::find(FuncInfo.begin(), FuncInfo.end(), Info) == FuncInfo.end()) {
    FuncInfo.push_back(Info);
  }
}
void CallFunctionExpr::buildCalleeInfo(const Expr *Callee,
                                       std::optional<unsigned int> NumArgs) {
  if (auto CallDecl =
          dyn_cast_or_null<FunctionDecl>(Callee->getReferencedDeclOfCallee())) {
    Name = getNameWithNamespace(CallDecl, Callee);
    if (auto FTD = CallDecl->getPrimaryTemplate()) {
      if (FTD->getTemplateParameters()->hasParameterPack())
        return;
    }
    setFuncInfo(DeviceFunctionDecl::LinkRedecls(CallDecl));
    if (auto DRE = dyn_cast<DeclRefExpr>(Callee)) {
      buildTemplateArguments(DRE->template_arguments(),
                             Callee->getSourceRange());
      auto ParentFunc = DpctGlobalInfo::getParentFunction(Callee);
      if (ParentFunc &&
          isa<TranslationUnitDecl>(ParentFunc->getDeclContext())) {
        return;
      }
      if (!isa<TranslationUnitDecl>(CallDecl->getDeclContext()) ||
          !DpctGlobalInfo::isInAnalysisScope(CallDecl->getBeginLoc()) ||
          DRE->getQualifier() || CallDecl->isOverloadedOperator())
        return;
      for (unsigned i = 0; i < NumArgs; i++) {
        auto Type = CallDecl->getParamDecl(i)
                        ->getOriginalType()
                        .getCanonicalType()
                        ->getUnqualifiedDesugaredType();
        while (Type && Type->isAnyPointerType()) {
          Type = Type->getPointeeType().getTypePtrOrNull();
        }

        if (Type->getAsRecordDecl() &&
            DpctGlobalInfo::isInCudaPath(
                Type->getAsRecordDecl()->getLocation())) {
          IsADLEnable = true;
          break;
        }
      }
    }
  } else if (auto Unresolved = dyn_cast<UnresolvedLookupExpr>(Callee)) {
    Name = "";
    if (Unresolved->getQualifier())
      Name = getNestedNameSpecifierString(Unresolved->getQualifier());
    Name += Unresolved->getName().getAsString();
    setFuncInfo(DeviceFunctionDecl::LinkUnresolved(Unresolved, NumArgs));
    buildTemplateArguments(Unresolved->template_arguments(),
                           Callee->getSourceRange());
  } else if (auto DependentScope =
                 dyn_cast<CXXDependentScopeMemberExpr>(Callee)) {
    Name = DependentScope->getMember().getAsString();
    buildTemplateArguments(DependentScope->template_arguments(),
                           Callee->getSourceRange());
  } else if (auto DSDRE = dyn_cast<DependentScopeDeclRefExpr>(Callee)) {
    Name = DSDRE->getDeclName().getAsString();
    buildTemplateArgumentsFromTypeLoc(DSDRE->getQualifierLoc().getTypeLoc());
  } else if (auto DRE = dyn_cast<DeclRefExpr>(Callee->IgnoreImpCasts())) {
    Name = DRE->getNameInfo().getAsString();
  } else {
    Name = "(" + ExprAnalysis::ref(Callee) + ")";
  }
}
std::string CallFunctionExpr::getName(const NamedDecl *D) {
  if (auto ID = D->getIdentifier())
    return ID->getName().str();
  return "";
}
void CallFunctionExpr::buildTemplateArguments(
    const llvm::ArrayRef<TemplateArgumentLoc> &ArgsList, SourceRange Range) {
  if (TemplateArgs.empty())
    for (auto &Arg : ArgsList)
      TemplateArgs.emplace_back(Arg, Range);
}
void CallFunctionExpr::buildTemplateArgumentsFromTypeLoc(const TypeLoc &TL) {
  if (!TL)
    return;
  switch (TL.getTypeLocClass()) {
  /// e.g. X<T>;
  case TypeLoc::TemplateSpecialization:
    return buildTemplateArgumentsFromSpecializationType(
        TYPELOC_CAST(TemplateSpecializationTypeLoc));
  /// e.g.: X<T1>::template Y<T2>
  case TypeLoc::DependentTemplateSpecialization:
    return buildTemplateArgumentsFromSpecializationType(
        TYPELOC_CAST(DependentTemplateSpecializationTypeLoc));
  default:
    break;
  }
}
template <class TyLoc>
void CallFunctionExpr::buildTemplateArgumentsFromSpecializationType(
    const TyLoc &TL) {
  for (size_t i = 0; i < TL.getNumArgs(); ++i) {
    TemplateArgs.emplace_back(TL.getArgLoc(i), TL.getSourceRange());
  }
}
/// This function gets the \p FD name with the necessary qualified namespace at
/// \p Callee position.
/// Algorithm:
/// 1. record all NamespaceDecl nodes of the ancestors \p FD and \p Callee, get
/// two namespace sequences. E.g.,
///   decl: aaa,bbb,ccc; callee: aaa,eee;
/// 2. Remove the longest continuous common subsequence
/// 3. the rest sequence of \p FD is the namespace sequence
std::string CallFunctionExpr::getNameWithNamespace(const FunctionDecl *FD,
                                                   const Expr *Callee) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto getNamespaceSeq =
      [&](DynTypedNodeList Parents) -> std::deque<std::string> {
    std::deque<std::string> Seq;
    while (Parents.size() > 0) {
      auto *Parent = Parents[0].get<NamespaceDecl>();
      if (Parent) {
        Seq.push_front(Parent->getNameAsString());
      }
      Parents = Context.getParents(Parents[0]);
    }
    return Seq;
  };

  std::deque<std::string> FDNamespaceSeq =
      getNamespaceSeq(Context.getParents(*FD));
  std::deque<std::string> CalleeNamespaceSeq =
      getNamespaceSeq(Context.getParents(*Callee));

  auto FDIter = FDNamespaceSeq.begin();
  for (const auto &CalleeNamespace : CalleeNamespaceSeq) {
    if (FDNamespaceSeq.empty())
      break;

    if (CalleeNamespace == *FDIter) {
      FDIter++;
      FDNamespaceSeq.pop_front();
    } else {
      break;
    }
  }

  std::string Result;
  for (const auto &I : FDNamespaceSeq) {
    // If I is empty, it means this namespace is an unnamed namespace. So its
    // members have internal linkage. So just remove it.
    if (I.empty())
      continue;
    Result = Result + I + "::";
  }

  return Result + getName(FD);
}
void CallFunctionExpr::buildTextureObjectArgsInfo(const CallExpr *CE) {
  buildTextureObjectArgsInfo<CallExpr>(CE);
  if (DpctGlobalInfo::useExtBindlessImages() || DpctGlobalInfo::useSYCLCompat())
    return;
  if (auto ME = dyn_cast<MemberExpr>(CE->getCallee()->IgnoreImpCasts())) {
    if (auto DRE = dyn_cast<DeclRefExpr>(ME->getBase()->IgnoreImpCasts())) {
      auto BaseObject = makeTextureObjectInfo<StructureTextureObjectInfo>(
          DRE->getDecl(), CE->getStmtClass() == Stmt::CUDAKernelCallExprClass);
      if (BaseObject)
        BaseTextureObject = std::move(BaseObject);
    }
  }
}
template <class CallT>
void CallFunctionExpr::buildTextureObjectArgsInfo(const CallT *C) {
  auto Args = C->arguments();
  auto IsKernel = C->getStmtClass() == Stmt::CUDAKernelCallExprClass;
  auto ArgsNum = std::distance(Args.begin(), Args.end());
  unsigned Idx = 0;
  TextureObjectList.resize(ArgsNum);
  if (DpctGlobalInfo::useExtBindlessImages() ||
      DpctGlobalInfo::useSYCLCompat()) {
    // Need return after resize, ortherwise will cause array out of bound.
    return;
  }
  for (auto ArgItr = Args.begin(); ArgItr != Args.end(); Idx++, ArgItr++) {
    const Expr *Arg = (*ArgItr)->IgnoreImpCasts();
    if (auto Ctor = dyn_cast<CXXConstructExpr>(Arg)) {
      if (Ctor->getConstructor()->isCopyOrMoveConstructor()) {
        Arg = Ctor->getArg(0);
      }
    }

    if (const auto *ICE = dyn_cast<ImplicitCastExpr>(Arg)) {
      if (ICE->getCastKind() == CK_DerivedToBase) {
        continue;
      }
    }
    if (auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts()))
      addTextureObjectArg(Idx, DRE, IsKernel);
    else if (auto ASE = dyn_cast<ArraySubscriptExpr>(Arg->IgnoreImpCasts()))
      addTextureObjectArg(Idx, ASE, IsKernel);
  }
}
void CallFunctionExpr::mergeTextureObjectInfo(
    std::shared_ptr<DeviceFunctionInfo> Info) {
  if (BaseTextureObject)
    BaseTextureObject->merge(Info->getBaseTextureObject());
  for (unsigned Idx = 0; Idx < TextureObjectList.size(); ++Idx) {
    if (auto &Obj = TextureObjectList[Idx]) {
      Obj->merge(Info->getTextureObject(Idx));
    }
  }
}
///// class DeviceFunctionDecl /////
DeviceFunctionDecl::DeviceFunctionDecl(
    unsigned Offset, const clang::tooling::UnifiedPath &FilePathIn,
    const FunctionDecl *FD)
    : Offset(Offset), OffsetForAttr(Offset), FilePath(FilePathIn),
      ParamsNum(FD->param_size()), ReplaceOffset(0), ReplaceLength(0),
      NonDefaultParamNum(FD->getMostRecentDecl()->getMinRequiredArguments()),
      FuncInfo(getFuncInfo(FD)) {
  if (FD->isFunctionTemplateSpecialization()) {
    SourceRange ReturnTypeRange = FD->getReturnTypeSourceRange();
    OffsetForAttr =
        DpctGlobalInfo::getLocInfo(ReturnTypeRange.getBegin()).second;
  }
  if (!FuncInfo) {
    FuncInfo = std::make_shared<DeviceFunctionInfo>(
        FD->param_size(), NonDefaultParamNum, getFunctionName(FD));
  }
  if (!FilePath.getCanonicalPath().empty()) {
    SourceProcessType FileType = GetSourceFileType(FilePath);
    if (!(FileType & SPT_CudaHeader) && !(FileType & SPT_CppHeader) &&
        FD->isThisDeclarationADefinition()) {
      FuncInfo->setDefinitionFilePath(FilePath);
    }
  }

  static AttrVec NullAttrs;
  buildReplaceLocInfo(
      FD->getTypeSourceInfo()->getTypeLoc().getAs<FunctionTypeLoc>(),
      FD->hasAttrs() ? FD->getAttrs() : NullAttrs);
  buildTextureObjectParamsInfo(FD->parameters());
  if (FD->hasAttr<CUDAGlobalAttr>()) {
    collectInfoForWrapper(FD);
  }
}
DeviceFunctionDecl::DeviceFunctionDecl(
    unsigned Offset, const clang::tooling::UnifiedPath &FilePathIn,
    const FunctionTypeLoc &FTL, const ParsedAttributes &Attrs,
    const FunctionDecl *Specialization)
    : Offset(Offset), OffsetForAttr(Offset), FilePath(FilePathIn),
      ParamsNum(Specialization->getNumParams()), ReplaceOffset(0),
      ReplaceLength(0),
      NonDefaultParamNum(
          Specialization->getMostRecentDecl()->getMinRequiredArguments()),
      FuncInfo(getFuncInfo(Specialization)) {
  IsDefFilePathNeeded = false;

  buildReplaceLocInfo(FTL, Attrs);
  buildTextureObjectParamsInfo(FTL.getParams());
  if (Specialization->hasAttr<CUDAGlobalAttr>()) {
    collectInfoForWrapper(Specialization);
  }
}
std::shared_ptr<DeviceFunctionInfo>
DeviceFunctionDecl::LinkUnresolved(const UnresolvedLookupExpr *ULE,
                                   std::optional<unsigned int> NumArgs) {
  std::vector<NamedDecl *> List;
  for (auto *D : ULE->decls()) {
    if (NumArgs) {
      const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
      if (!FD) {
        if (const FunctionTemplateDecl *FTD =
                dyn_cast<FunctionTemplateDecl>(D)) {
          FD = FTD->getTemplatedDecl();
        }
      }
      if (FD) {
        if (NumArgs.value() >= FD->getMinRequiredArguments() &&
            NumArgs.value() <= FD->getNumParams()) {
          List.push_back(D);
        }
      }
    } else {
      List.push_back(D);
    }
  }
  return LinkDeclRange(List, getFunctionName(ULE));
}
std::shared_ptr<DeviceFunctionInfo>
DeviceFunctionDecl::LinkRedecls(const FunctionDecl *FD) {
  if (auto D = DpctGlobalInfo::getInstance().findDeviceFunctionDecl(FD))
    return D->getFuncInfo();
  if (auto FTD = FD->getPrimaryTemplate())
    return LinkTemplateDecl(FTD);
  else if (FTD = FD->getDescribedFunctionTemplate())
    return LinkTemplateDecl(FTD);
  else if (auto Decl = FD->getInstantiatedFromMemberFunction())
    FD = Decl;
  return LinkDeclRange(FD->redecls(), getFunctionName(FD));
}
std::shared_ptr<DeviceFunctionInfo>
DeviceFunctionDecl::LinkTemplateDecl(const FunctionTemplateDecl *FTD) {
  return LinkDeclRange(FTD->redecls(), getFunctionName(FTD));
}
std::shared_ptr<DeviceFunctionInfo>
DeviceFunctionDecl::LinkExplicitInstantiation(
    const FunctionDecl *Specialization, const FunctionTypeLoc &FTL,
    const ParsedAttributes &Attrs, const TemplateArgumentListInfo &TAList) {
  auto Info = LinkRedecls(Specialization);
  if (Info) {
    auto D = DpctGlobalInfo::getInstance().insertDeviceFunctionDecl(
        Specialization, FTL, Attrs, TAList);
    D->setFuncInfo(Info);
  }
  return Info;
}
void DeviceFunctionDecl::emplaceReplacement() {
  auto Repl = std::make_shared<ExtReplacement>(
      FilePath, ReplaceOffset, ReplaceLength,
      getExtraParameters(std::make_pair(FilePath, Offset)), nullptr);
  Repl->setNotFormatFlag();
  DpctGlobalInfo::getInstance().addReplacement(Repl);

  if (FuncInfo->IsSyclExternMacroNeeded()) {
    std::string StrRepl = "SYCL_EXTERNAL ";
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, OffsetForAttr, 0, StrRepl,
                                         nullptr));
  }

  if (FuncInfo->IsAlwaysInlineDevFunc()) {
    std::string StrRepl = "inline ";
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, OffsetForAttr, 0, StrRepl,
                                         nullptr));
  }
  if (FuncInfo->IsForceInlineDevFunc()) {
    std::string StrRepl = DpctGlobalInfo::useSYCLCompat()
                              ? "__syclcompat_inline__ "
                              : "__dpct_inline__ ";
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, OffsetForAttr, 0, StrRepl,
                                         nullptr));
  }

  for (auto &Obj : TextureObjectList) {
    if (Obj) {
      Obj->merge(FuncInfo->getTextureObject((Obj->getParamIdx())));
      if (DpctGlobalInfo::useExtBindlessImages()) {
        continue;
      }
      if (!Obj->getType()) {
        // Type dpct_placeholder
        Obj->setType("dpct_placeholder/*Fix the type manually*/", 1);
        DiagnosticsUtils::report(Obj->getFilePath(), Obj->getOffset(),
                                 Diagnostics::UNDEDUCED_TYPE, true, false,
                                 "image_accessor_ext");
      }
      Obj->addParamDeclReplacement();
    }
  }
  if (FuncInfo->getDeviceFunctionInfoForWrapper()) {
    insertWrapper();
  }
}
void DeviceFunctionDecl::LinkDecl(const FunctionDecl *FD, DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  if (!DpctGlobalInfo::isInAnalysisScope(FD->getBeginLoc()))
    return;
  if (!FD->hasAttr<CUDADeviceAttr>() && !FD->hasAttr<CUDAGlobalAttr>())
    return;

  // Ignore explicit instantiation definition, as the decl in AST has wrong
  // location info. And it is processed in
  // DPCTConsumer::HandleCXXExplicitFunctionInstantiation
  if (FD->getTemplateSpecializationKind() ==
      TSK_ExplicitInstantiationDefinition)
    return;
  if (FD->isImplicit() ||
      (FD->getTemplateSpecializationKind() == TSK_ImplicitInstantiation &&
       FD->getPrimaryTemplate())) {
    auto &FuncInfo = getFuncInfo(FD);
    if (Info) {
      if (FuncInfo)
        Info->merge(FuncInfo);
      FuncInfo = Info;
    } else if (FuncInfo) {
      Info = FuncInfo;
    } else {
      Info = std::make_shared<DeviceFunctionInfo>(
          FD->param_size(), FD->getMostRecentDecl()->getMinRequiredArguments(),
          getFunctionName(FD));
      FuncInfo = Info;
    }
    return;
  }
  std::shared_ptr<DeviceFunctionDecl> D;

  D = DpctGlobalInfo::getInstance().insertDeviceFunctionDecl(FD);

  if (Info) {
    if (auto FuncInfo = D->getFuncInfo())
      Info->merge(FuncInfo);
    D->setFuncInfo(Info);
  } else if (auto FuncInfo = D->getFuncInfo())
    Info = FuncInfo;
  else
    List.push_back(D);

  if (Info && isModuleFunction(FD) && FD->hasAttr<CUDAGlobalAttr>()) {
    Info->collectInfoForWrapper(FD);
    Info->setModuleUsed();
  }
}
void DeviceFunctionDecl::LinkDecl(const NamedDecl *ND, DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  switch (ND->getKind()) {
  case Decl::CXXMethod:
  case Decl::Function:
    return LinkRedecls(static_cast<const FunctionDecl *>(ND), List, Info);
  case Decl::FunctionTemplate:
    return LinkDecl(static_cast<const FunctionTemplateDecl *>(ND), List, Info);
  case Decl::UsingShadow:
    return LinkDecl(
        static_cast<const UsingShadowDecl *>(ND)->getUnderlyingDecl(), List,
        Info);
    break;
  default:
    DpctDiags() << "[DeviceFunctionDecl::LinkDecl] Unexpected decl type: "
                << ND->getDeclKindName() << "\n";
    return;
  }
}
void DeviceFunctionDecl::LinkDecl(const FunctionTemplateDecl *FTD,
                                  DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  LinkDecl(FTD->getAsFunction(), List, Info);
  LinkDeclRange(FTD->specializations(), List, Info);
}
void DeviceFunctionDecl::LinkRedecls(
    const FunctionDecl *FD, DeclList &List,
    std::shared_ptr<DeviceFunctionInfo> &Info) {
  LinkDeclRange(FD->redecls(), List, Info);
}
void DeviceFunctionDecl::setFuncInfo(std::shared_ptr<DeviceFunctionInfo> Info) {
  if (FuncInfo.get() == Info.get())
    return;
  FuncInfo = Info;
  if (IsDefFilePathNeeded)
    FuncInfo->setDefinitionFilePath(FilePath);
}
const FormatInfo &DeviceFunctionDecl::getFormatInfo() {
  return FormatInformation;
}
void DeviceFunctionDecl::buildTextureObjectParamsInfo(
    const ArrayRef<ParmVarDecl *> &Parms) {
  TextureObjectList.assign(Parms.size(), std::shared_ptr<TextureObjectInfo>());
  if (DpctGlobalInfo::useSYCLCompat())
    return;
  for (unsigned Idx = 0; Idx < Parms.size(); ++Idx) {
    auto Param = Parms[Idx];
    std::string ParamName =
        DpctGlobalInfo::getUnqualifiedTypeName(Param->getType());
    if (ParamName == "cudaTextureObject_t" ||
        ParamName == "cudaSurfaceObject_t" || ParamName == "CUsurfObject") {
      TextureObjectList[Idx] = std::make_shared<TextureObjectInfo>(Param);
    }
  }
}
std::string DeviceFunctionDecl::getExtraParameters(LocInfo LI) {
  std::string Result =
      FuncInfo->getExtraParameters(FilePath, LI, FormatInformation);
  if (!Result.empty() && IsReplaceFollowedByPP) {
    Result += getNL();
  }
  return Result;
}
std::shared_ptr<DeviceFunctionInfo> &
DeviceFunctionDecl::getFuncInfo(const FunctionDecl *FD) {
  DpctNameGenerator G;
  std::string Key;
  // For static functions or functions in anonymous namespace,
  // need to add filepath as prefix to differentiate them.
  if (FD->isStatic() || FD->isInAnonymousNamespace()) {
    auto LocInfo = DpctGlobalInfo::getLocInfo(FD);
    Key = LocInfo.first.getCanonicalPath().str() + G.getName(FD);
  } else {
    Key = G.getName(FD);
  }
  return FuncInfoMap[Key];
}
std::unordered_map<std::string, std::shared_ptr<DeviceFunctionInfo>>
    DeviceFunctionDecl::FuncInfoMap;
///// class ExplicitInstantiationDecl /////
void ExplicitInstantiationDecl::processFunctionTypeLoc(
    const FunctionTypeLoc &FTL) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  ExprAnalysis EA;
  processTypeLoc(FTL.getReturnLoc(), EA, SM);
  for (const auto &Parm : FTL.getParams()) {
    processTypeLoc(Parm->getTypeSourceInfo()->getTypeLoc(), EA, SM);
  }
}

void ExplicitInstantiationDecl::processTemplateArgumentList(
    const TemplateArgumentListInfo &TAList) {
  ExprAnalysis EA;
  for (const clang::TemplateArgumentLoc &ArgLoc : TAList.arguments()) {
    EA.analyze(ArgLoc);
    if (EA.hasReplacement())
      DpctGlobalInfo::getInstance().addReplacement(
          EA.getReplacement()->getReplacement(DpctGlobalInfo::getContext()));
  }
}
void ExplicitInstantiationDecl::initTemplateArgumentList(
    const TemplateArgumentListInfo &TAList,
    const FunctionDecl *Specialization) {
  if (Specialization->getTemplateSpecializationArgs() == nullptr)
    return;
  for (auto &Arg : Specialization->getTemplateSpecializationArgs()->asArray()) {
    TemplateArgumentInfo TA;
    switch (Arg.getKind()) {
    case TemplateArgument::Integral:
      TA.setAsNonType(Arg.getAsIntegral());
      break;
    case TemplateArgument::Expression:
      TA.setAsNonType(Arg.getAsExpr());
      break;
    case TemplateArgument::Type:
      TA.setAsType(Arg.getAsType());
      break;
    default:
      break;
    }
    InstantiationArgs.emplace_back(std::move(TA));
  }
}
std::string ExplicitInstantiationDecl::getExtraParameters(LocInfo LI) {
  return getFuncInfo()->getExtraParameters(FilePath, InstantiationArgs, LI,
                                           getFormatInfo());
}
///// class KernelPrinter /////
class KernelPrinter {
  const std::string NL;
  std::string Indent;
  llvm::raw_string_ostream &Stream;

  void incIndent() { Indent += "  "; }
  void decIndent() { Indent.erase(Indent.length() - 2, 2); }

public:
  class Block {
    KernelPrinter &Printer;
    bool WithBrackets;

  public:
    Block(KernelPrinter &Printer, bool WithBrackets)
        : Printer(Printer), WithBrackets(WithBrackets) {
      if (WithBrackets)
        Printer.line("{");
      Printer.incIndent();
    }
    ~Block() {
      Printer.decIndent();
      if (WithBrackets)
        Printer.line("}");
    }
  };

public:
  KernelPrinter(const std::string &NL, const std::string &Indent,
                llvm::raw_string_ostream &OS)
      : NL(NL), Indent(Indent), Stream(OS) {}
  std::unique_ptr<Block> block(bool WithBrackets = false) {
    return std::make_unique<Block>(*this, WithBrackets);
  }
  template <class T> KernelPrinter &operator<<(const T &S) {
    Stream << S;
    return *this;
  }
  template <class... Args> KernelPrinter &line(Args &&...Arguments) {
    appendString(Stream, Indent, std::forward<Args>(Arguments)..., NL);
    return *this;
  }
  KernelPrinter &operator<<(const StmtList &Stmts) {
    for (auto &S : Stmts) {
      if (S.StmtStr.empty())
        continue;
      if (!S.Warnings.empty()) {
        for (auto &Warning : S.Warnings) {
          line("/*");
          line(Warning);
          line("*/");
        }
      }
      line(S.StmtStr);
    }
    return *this;
  }
  KernelPrinter &indent() { return (*this) << Indent; }
  KernelPrinter &newLine() { return (*this) << NL; }
  std::string str() {
    auto Result = Stream.str();
    return Result.substr(Indent.length(),
                         Result.length() - Indent.length() - NL.length());
  }
};

void DeviceFunctionDecl::insertWrapper() {
  auto NL = std::string(getNL());
  std::string WrapperStr = "";
  llvm::raw_string_ostream OS(WrapperStr);
  KernelPrinter Printer(NL, "", OS);
  Printer.newLine();
  Printer.newLine();
  auto InfoForWrapper = FuncInfo->getDeviceFunctionInfoForWrapper();
  bool ModuleUsed = FuncInfo->isModuleUsed();
  auto &TParamsInfo = InfoForWrapper->TemplateParametersInfo;
  auto &ParamsInfo = InfoForWrapper->ParametersInfo;
  std::string FuncName = FuncInfo->getFunctionName();
  if (ModuleUsed) {
    Printer.line("extern \"C\" {");
  }
  {
    auto FunctionBlock = Printer.block();
    Printer.indent();
    requestFeature(HelperFeatureEnum::device_ext);
    // 1.Generate wrapper signature
    if (ModuleUsed) {
      Printer << "DPCT_EXPORT void " << FuncName << "_wrapper("
              << MapNames::getClNamespace() << "queue &queue, const "
              << MapNames::getClNamespace()
              << "nd_range<3> &nr, unsigned int localMemSize, void "
                 "**kernelParams, void **extra)";
    } else {
      Printer.line("// Auto generated SYCL kernel wrapper used to migration "
                   "kernel function pointer.");
      if (!TParamsInfo.empty()) {
        Printer << "template<";
        for (size_t i = 0; i < TParamsInfo.size(); i++) {
          Printer << (i == 0 ? "" : " ,") << TParamsInfo[i].first << " "
                  << TParamsInfo[i].second
                  << TemplateParameterDefaultValueMap[i];
        }
        Printer << ">";
        Printer.newLine();
      }
      Printer << "void " << FuncName << "_wrapper(";
      for (size_t i = 0; i < ParamsInfo.size(); i++) {
        Printer << (i == 0 ? "" : " ,") << ParamsInfo[i].first << " "
                << ParamsInfo[i].second << ParameterDefaultValueMap[i];
      }
      Printer << ")";
    }
    // 2.Generate wrapper body
    if (HasBody) {
      if (ModuleUsed) {
        auto for_each_parameter = [&](auto F) {
          auto it = ParamsInfo.begin();
          for (int i = 0; it != ParamsInfo.end(); ++it, ++i) {
            F(i, it->second);
          }
        };
        Printer << " {";
        {
          auto BodyBlock = Printer.block();
          Printer.newLine();
          auto DefaultParamNum = ParamsNum - NonDefaultParamNum;
          Printer.line(llvm::formatv(
              "// {0} non-default parameters, {1} default parameters",
              NonDefaultParamNum, DefaultParamNum));
          Printer.line(
              llvm::formatv("{0}args_selector<{1}, {2}, decltype({3})> "
                            "selector(kernelParams, extra);",
                            MapNames::getDpctNamespace(), NonDefaultParamNum,
                            DefaultParamNum, FuncName));
          for_each_parameter([&](auto &&i, auto &&p) {
            Printer.line("auto& " + p + " = selector.get<" + std::to_string(i) +
                         ">();");
          });
          (InfoForWrapper->KernelForWrapper)->buildInfo();
          Printer.line((InfoForWrapper->KernelForWrapper)->getReplacement());
        }
        Printer.line("}");
      } else {
        Printer << " {";
        {
          auto BodyBlock = Printer.block();
          Printer.newLine();
          Printer.line(MapNames::getClNamespace() + "queue queue = *" +
                       MapNames::getDpctNamespace() + "kernel_launcher::_que;");
          Printer.line(
              "unsigned int localMemSize = " + MapNames::getDpctNamespace() +
              "kernel_launcher::_local_mem_size;");
          Printer.line(MapNames::getClNamespace() + "nd_range<3> nr = " +
                       MapNames::getDpctNamespace() + "kernel_launcher::_nr;");
          Printer.newLine();
          (InfoForWrapper->KernelForWrapper)->buildInfo();
          Printer.line((InfoForWrapper->KernelForWrapper)->getReplacement());
        }
        Printer.line("}");
      }
    } else {
      Printer << ";";
      Printer.newLine();
    }
  }
  if (ModuleUsed) {
    Printer << "}";
  }

  auto Repl = std::make_shared<ExtReplacement>(FilePath, DeclEnd, 0, WrapperStr,
                                               nullptr);
  Repl->setBlockLevelFormatFlag();
  DpctGlobalInfo::getInstance().addReplacement(std::move(Repl));
}
void DeviceFunctionDecl::collectInfoForWrapper(const FunctionDecl *FD) {
  if ((FD->getTemplatedKind() != FunctionDecl::TemplatedKind::TK_NonTemplate) &&
      (FD->getTemplatedKind() !=
       FunctionDecl::TemplatedKind::TK_FunctionTemplate)) {
    return;
  }
  const FunctionDecl *Def;
  HasBody = FD->hasBody(Def);
  if (HasBody && FD != Def) {
    HasBody = false;
  }

  if (auto FTD = FD->getDescribedFunctionTemplate()) {
    if (auto TemplateParmsList = FTD->getTemplateParameters()) {
      for (size_t i = 0; i < TemplateParmsList->size(); ++i) {
        auto TemplateParm = TemplateParmsList->getParam(i);
        if (auto TTPD = dyn_cast<TemplateTypeParmDecl>(TemplateParm)) {
          if (TTPD->hasDefaultArgument() &&
              !TTPD->defaultArgumentWasInherited()) {
            ExprAnalysis EA;
            EA.analyze(
                TTPD->getDefaultArgument().getTypeSourceInfo()->getTypeLoc());
            TemplateParameterDefaultValueMap[i] =
                " = " + EA.getReplacedString();
          }
        } else if (auto NTTPD =
                       dyn_cast<NonTypeTemplateParmDecl>(TemplateParm)) {
          if (NTTPD->hasDefaultArgument() &&
              !NTTPD->defaultArgumentWasInherited()) {
            TemplateParameterDefaultValueMap[i] =
                " = " + ExprAnalysis::ref(
                            NTTPD->getDefaultArgument().getSourceExpression());
          }
        }
      }
    }
  }
  for (size_t i = 0; i < FD->param_size(); i++) {
    auto PDecl = FD->getParamDecl(i);
    if (!PDecl->hasInheritedDefaultArg()) {
      if (PDecl->hasUninstantiatedDefaultArg()) {
        ParameterDefaultValueMap[i] =
            " = " + ExprAnalysis::ref(PDecl->getUninstantiatedDefaultArg());
      } else if (PDecl->hasDefaultArg()) {
        ParameterDefaultValueMap[i] =
            " = " + ExprAnalysis::ref(PDecl->getDefaultArg());
      }
    }
  }

  // FD has relatively large range, which is likely to be straddle,
  // getDefinitionRange may not work as good as getExpansionRange
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto EndLoc =
      SM.getSpellingLoc(SM.getExpansionRange(FD->getEndLoc()).getEnd());
  auto LastTokenLen = Lexer::MeasureTokenLength(
      EndLoc, SM, dpct::DpctGlobalInfo::getContext().getLangOpts());
  EndLoc = EndLoc.getLocWithOffset(LastTokenLen);
  if (!HasBody) {
    LastTokenLen = Lexer::MeasureTokenLength(
        EndLoc, SM, dpct::DpctGlobalInfo::getContext().getLangOpts());
    EndLoc = EndLoc.getLocWithOffset(LastTokenLen);
  }
  DeclEnd = SM.getFileOffset(EndLoc);
}

///// class DeviceFunctionInfo /////
DeviceFunctionInfo::DeviceFunctionInfo(size_t ParamsNum,
                                       size_t NonDefaultParamNum,
                                       std::string FunctionName)
    : ParamsNum(ParamsNum), NonDefaultParamNum(NonDefaultParamNum),
      IsBuilt(false),
      TextureObjectList(ParamsNum, std::shared_ptr<TextureObjectInfo>()),
      FunctionName(FunctionName), IsLambda(false) {
  ParametersProps.resize(ParamsNum);
}

void DeviceFunctionInfo::collectInfoForWrapper(const FunctionDecl *FD) {
  if (!DFInfoForWrapper) {
    DFInfoForWrapper = std::make_shared<DeviceFunctionInfoForWrapper>();
    auto LocInfo = DpctGlobalInfo::getLocInfo(FD->getBeginLoc());
    auto &TemplateParametersInfo = DFInfoForWrapper->TemplateParametersInfo;
    auto &ParametersInfo = DFInfoForWrapper->ParametersInfo;

    auto &Context = dpct::DpctGlobalInfo::getContext();
    auto Parents = Context.getParents(*FD);
    if (Parents.size()) {
      if (auto FTD = Parents[0].get<FunctionTemplateDecl>()) {
        FD = FTD->getTemplatedDecl();
        if (auto TemplateParmsList = FTD->getTemplateParameters()) {
          for (size_t i = 0; i < TemplateParmsList->size(); ++i) {
            auto TemplateParm = TemplateParmsList->getParam(i);
            if (auto TTPD = dyn_cast<TemplateTypeParmDecl>(TemplateParm)) {
              TemplateParametersInfo.push_back(
                  {std::string(TTPD->wasDeclaredWithTypename() ? "typename"
                                                               : "class") +
                       std::string(TTPD->isParameterPack() ? "... " : ""),
                   TTPD->getNameAsString()});
            } else if (auto NTTPD =
                           dyn_cast<NonTypeTemplateParmDecl>(TemplateParm)) {
              std::string DefVal;
              ExprAnalysis EA;
              EA.analyze(NTTPD->getTypeSourceInfo()->getTypeLoc());
              TemplateParametersInfo.push_back(
                  {EA.getReplacedString(),
                   NTTPD->getNameAsString()});
            }
          }
        }
      }
    }
    DFInfoForWrapper->KernelForWrapper =
        KernelCallExpr::buildForWrapper(LocInfo.first, FD);
    std::string TemplateArgsStr;
    for (size_t i = 0; i < TemplateParametersInfo.size(); i++) {
      TemplateArgsStr +=
          (i == 0 ? "" : ", ") + TemplateParametersInfo[i].second;
    }
    if (!TemplateArgsStr.empty()) {
      DFInfoForWrapper->KernelForWrapper->setTemplateArgsStrForWrapper(
          std::move(TemplateArgsStr));
    }
    for (auto It = FD->param_begin(); It != FD->param_end(); It++) {
      ParametersInfo.push_back(
          {DpctGlobalInfo::getReplacedTypeName((*It)->getType()),
           (*It)->getNameAsString()});
    }
  }
}

std::shared_ptr<CallFunctionExpr>
DeviceFunctionInfo::findCallee(const CallExpr *C) {
  auto CallLocInfo = DpctGlobalInfo::getLocInfo(C);
  return findObject(CallExprMap, CallLocInfo.second);
}
std::shared_ptr<TextureObjectInfo>
DeviceFunctionInfo::getTextureObject(unsigned Idx) {
  if (Idx < TextureObjectList.size())
    return TextureObjectList[Idx];
  return {};
}
void DeviceFunctionInfo::buildInfo() {
  if (isBuilt()) {
    VarMap.removeDuplicateVar();
    return;
  }
  setBuilt();
  auto &Map = VarMap.getMap(clang::dpct::MemVarInfo::Global);
  for (auto It = Map.begin(); It != Map.end();) {
    auto &Info = It->second;
    if (!Info->getUsedBySymbolAPIFlag()) {
      if (DpctGlobalInfo::isOptimizeMigration() && Info->isConstant()) {
        Info->setUseHelperFuncFlag(false);
      }
      if (DpctGlobalInfo::useExpDeviceGlobal() &&
          (Info->isConstant() || Info->isDevice())) {
        Info->setUseHelperFuncFlag(false);
        Info->setUseDeviceGlobalFlag(true);
      }
    }
    if (!Info->isUseHelperFunc()) {
      It = Map.erase(It);
    } else {
      ++It;
    }
  }
  for (auto &Call : CallExprMap) {
    Call.second->emplaceReplacement();
    VarMap.merge(Call.second->getVarMap());
    mergeCalledTexObj(Call.second->getBaseTextureObjectInfo(),
                      Call.second->getTextureObjectList());
  }
  VarMap.removeDuplicateVar();
}
std::string
DeviceFunctionInfo::getExtraParameters(const clang::tooling::UnifiedPath &Path,
                                       LocInfo LI,
                                       FormatInfo FormatInformation) {
  buildInfo();
  VarMap.requestFeatureForAllVarMaps(Path);
  return VarMap.getExtraDeclParam(NonDefaultParamNum,
                                  ParamsNum - NonDefaultParamNum, LI,
                                  FormatInformation);
}
std::string DeviceFunctionInfo::getExtraParameters(
    const clang::tooling::UnifiedPath &Path,
    const std::vector<TemplateArgumentInfo> &TAList, LocInfo LI,
    FormatInfo FormatInformation) {
  MemVarMap TmpVarMap;
  buildInfo();
  TmpVarMap.merge(VarMap, TAList);
  TmpVarMap.requestFeatureForAllVarMaps(Path);
  return TmpVarMap.getExtraDeclParam(NonDefaultParamNum,
                                     ParamsNum - NonDefaultParamNum, LI,
                                     FormatInformation);
}
void DeviceFunctionInfo::merge(std::shared_ptr<DeviceFunctionInfo> Other) {
  if (this == Other.get())
    return;
  VarMap.merge(Other->getVarMap());
  dpct::merge(CallExprMap, Other->CallExprMap);
  if (BaseObjectTexture)
    BaseObjectTexture->merge(Other->BaseObjectTexture);
  else
    BaseObjectTexture = Other->BaseObjectTexture;
  mergeTextureObjectList(Other->TextureObjectList);
}
void DeviceFunctionInfo::addSubGroupSizeRequest(unsigned int Size,
                                                SourceLocation Loc,
                                                std::string APIName,
                                                std::string VarName) {
  if (Size == 0 || Loc.isInvalid())
    return;
  auto LocInfo = DpctGlobalInfo::getLocInfo(Loc);
  RequiredSubGroupSize.push_back(
      std::make_tuple(Size, LocInfo.first, LocInfo.second, APIName, VarName));
}
bool DeviceFunctionInfo::isParameterReferenced(unsigned int Index) {
  if (Index >= ParametersProps.size())
    return true;
  return ParametersProps[Index].IsReferenced;
}
void DeviceFunctionInfo::setParameterReferencedStatus(unsigned int Index,
                                                      bool IsReferenced) {
  if (Index >= ParametersProps.size())
    return;
  ParametersProps[Index].IsReferenced =
      ParametersProps[Index].IsReferenced || IsReferenced;
}
void DeviceFunctionInfo::mergeCalledTexObj(
    std::shared_ptr<StructureTextureObjectInfo> BaseObj,
    const std::vector<std::shared_ptr<TextureObjectInfo>> &TexObjList) {
  if (BaseObj) {
    if (BaseObj->isBase()) {
      if (BaseObjectTexture)
        BaseObjectTexture->merge(BaseObj);
      else
        BaseObjectTexture = BaseObj;
    } else if (BaseObj->getParamIdx() < TextureObjectList.size()) {
      auto &Parm = TextureObjectList[BaseObj->getParamIdx()];
      if (Parm)
        Parm->merge(BaseObj);
      else
        Parm = BaseObj;
    }
  }
  for (auto &Obj : TexObjList) {
    if (!Obj)
      continue;
    if (Obj->getParamIdx() >= TextureObjectList.size())
      continue;
    if (auto &Parm = TextureObjectList[Obj->getParamIdx()]) {
      Parm->merge(Obj);
    } else {
      TextureObjectList[Obj->getParamIdx()] = Obj;
    }
  }
}
void DeviceFunctionInfo::mergeTextureObjectList(
    const std::vector<std::shared_ptr<TextureObjectInfo>> &Other) {
  auto SelfItr = TextureObjectList.begin();
  auto BranchItr = Other.begin();
  while ((SelfItr != TextureObjectList.end()) && (BranchItr != Other.end())) {
    if (!(*SelfItr))
      *SelfItr = *BranchItr;
    ++SelfItr;
    ++BranchItr;
  }
  TextureObjectList.insert(SelfItr, BranchItr, Other.end());
}
///// class KernelCallExpr /////
KernelCallExpr::ArgInfo::ArgInfo(const ParmVarDecl *PVD,
                                 KernelArgumentAnalysis &Analysis,
                                 const Expr *Arg, bool Used, int Index,
                                 KernelCallExpr *BASE)
    : IsPointer(false), IsRedeclareRequired(false),
      IsUsedAsLvalueAfterMalloc(Used), Index(Index) {
  if (PVD &&
      PVD->getType()->getTypeClass() == Type::TypeClass::SubstTemplateTypeParm)
    IsDependentType = true;
  if (isa<InitListExpr>(Arg)) {
    HasImplicitConversion = true;
  } else if (const auto *CCE = dyn_cast<CXXConstructExpr>(Arg)) {
    HasImplicitConversion = true;
    if (CCE->getNumArgs() == 1) {
      Arg = CCE->getArg(0);
    }
  }
#ifdef _WIN32
  // This code path is for ConstructorConversion on Windows since its AST is
  // different from the one on Linux.
  if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(Arg)) {
    Arg = MTE->getSubExpr();
  }
#endif
  if (const auto *ICE = dyn_cast<ImplicitCastExpr>(Arg)) {
    auto CK = ICE->getCastKind();
    if (CK != CK_LValueToRValue) {
      HasImplicitConversion = true;
    }
    if (CK == CK_ConstructorConversion || CK == CK_UserDefinedConversion ||
        CK == CK_DerivedToBase) {
      IsRedeclareRequired = true;
    }
  }
  Analysis.analyze(Arg);
  ArgString = Analysis.getReplacedString();
  TryGetBuffer = Analysis.TryGetBuffer;
  IsRedeclareRequired |= Analysis.IsRedeclareRequired;
  IsPointer = Analysis.IsPointer;
  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
    IsDoublePointer = Analysis.IsDoublePointer;
  }

  if (IsPointer) {
    QualType PointerType;
    if (Arg->getType().getTypePtr()->getTypeClass() ==
        Type::TypeClass::Decayed) {
      PointerType = Arg->getType().getCanonicalType();
    } else {
      PointerType = Arg->getType();
    }
    TypeString = DpctGlobalInfo::getReplacedTypeName(PointerType);
    ArgSize = MapNamesLang::KernelArgTypeSizeMap.at(KernelArgType::KAT_Default);

    // Currently, all the device RNG state structs are passed to kernel by
    // pointer. So we check the pointee type, if it is in the type map, we
    // replace the TypeString with the MKL generator type.
    std::string PointeeTypeStr =
        Arg->getType()->getPointeeType().getUnqualifiedType().getAsString();
    auto Iter =
        MapNamesRandom::DeviceRandomGeneratorTypeMap.find(PointeeTypeStr);
    if (Iter != MapNamesRandom::DeviceRandomGeneratorTypeMap.end()) {
      // Here the "*" is not added in the TypeString, the "*" will be added
      // in function buildKernelArgsStmt
      TypeString = Iter->second;
      IsDeviceRandomGeneratorType = true;
    }
  } else {
    auto QT = Arg->getType();
    QT = QT.getUnqualifiedType();
    auto Iter =
        MapNamesLang::VectorTypeMigratedTypeSizeMap.find(QT.getAsString());
    if (Iter != MapNamesLang::VectorTypeMigratedTypeSizeMap.end())
      ArgSize = Iter->second;
    else
      ArgSize =
          MapNamesLang::KernelArgTypeSizeMap.at(KernelArgType::KAT_Default);
    if (PVD) {
      TypeString = DpctGlobalInfo::getReplacedTypeName(PVD->getType());
    }
  }
  if (IsRedeclareRequired || IsPointer || BASE->IsInMacroDefine) {
    IdString = getTempNameForExpr(Arg, false, true, BASE->IsInMacroDefine,
                                  Analysis.CallSpellingBegin,
                                  Analysis.CallSpellingEnd);
  }
}
KernelCallExpr::ArgInfo::ArgInfo(const ParmVarDecl *PVD,
                                 const std::string &ArgsArrayName,
                                 KernelCallExpr *Kernel)
    : IsPointer(PVD->getType()->isPointerType()), IsRedeclareRequired(true),
      IsUsedAsLvalueAfterMalloc(true),
      TryGetBuffer(DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
                   IsPointer),
      TypeString(DpctGlobalInfo::getReplacedTypeName(PVD->getType())),
      IdString(PVD->getName().str() + "_"),
      Index(PVD->getFunctionScopeIndex()) {
  // For parameter declaration 'float *a' with index = 2 and args array's
  // name is 'args', the arg string will be '*(float **)args[2]'.
  llvm::raw_string_ostream OS(ArgString);
  // Get pointer type of the parameter declaration's type, e.g. 'float **'.
  auto CastPointerType =
      DpctGlobalInfo::getContext().getPointerType(PVD->getType());
  // Print '*(float **)'.
  OS << "*(" << DpctGlobalInfo::getReplacedTypeName(CastPointerType) << ")";
  // Print args array subscript.
  OS << ArgsArrayName << "[" << Index << "]";

  if (TextureObjectInfo::isTextureObject(PVD)) {
    IsRedeclareRequired = false;
    Texture = std::make_shared<CudaLaunchTextureObjectInfo>(PVD, OS.str());
    Kernel->addTextureObjectArgInfo(Index, Texture);
  }
}
KernelCallExpr::ArgInfo::ArgInfo(const ParmVarDecl *PVD, KernelCallExpr *Kernel)
    : IsPointer(PVD->getType()->isPointerType()), IsRedeclareRequired(true),
      IsUsedAsLvalueAfterMalloc(true),
      TryGetBuffer(DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
                   IsPointer),
      TypeString(DpctGlobalInfo::getReplacedTypeName(PVD->getType())),
      IdString(PVD->getName().str() + "_"),
      Index(PVD->getFunctionScopeIndex()) {
  auto ArgName = PVD->getNameAsString();
  ArgString = ArgName;

  if (TextureObjectInfo::isTextureObject(PVD)) {
    Texture = std::make_shared<CudaLaunchTextureObjectInfo>(PVD, ArgName);
    Kernel->addTextureObjectArgInfo(Index, Texture);
  }
  IsRedeclareRequired = false;
}
KernelCallExpr::ArgInfo::ArgInfo(std::shared_ptr<TextureObjectInfo> Obj,
                                 KernelCallExpr *BASE, std::string ArgStr)
    : IsUsedAsLvalueAfterMalloc(false), Texture(Obj) {
  IsPointer = false;
  IsRedeclareRequired = false;
  TypeString = "";
  Index = 0;
  if (auto S = std::dynamic_pointer_cast<StructureTextureObjectInfo>(Obj)) {
    IsDoublePointer = S->containsVirtualPointer();
  }
  ArgString = std::move(ArgStr);
  IdString = ArgString + "_";
  ArgSize = MapNamesLang::KernelArgTypeSizeMap.at(KernelArgType::KAT_Texture);
}
const std::string &KernelCallExpr::ArgInfo::getArgString() const {
  return ArgString;
}
const std::string &KernelCallExpr::ArgInfo::getTypeString() const {
  return TypeString;
}
void KernelCallExpr::print(KernelPrinter &Printer) {
  std::unique_ptr<KernelPrinter::Block> Block;
  if (NeedLambda)
    Block = std::move(Printer.block(true));
  else if (!OuterStmts.empty())
    Block = std::move(Printer.block(NeedBraces));
  OuterStmts.print(Printer);
  printSubmit(Printer);
  if (NeedDefaultRetValue)
    Printer.line("return 0;");
  Block.reset();
  if (!getEvent().empty() && isSync())
    Printer.line(getEvent(), "->wait();");
}
void KernelCallExpr::printSubmit(KernelPrinter &Printer) {
  std::string SubGroupSizeWarning;
  auto DeviceFuncInfo = getFuncInfo();
  struct {
    bool isFirstRef = true;
    bool isEvaluated = true;
    unsigned int Size = 0;
    std::string SizeStr;
  } RequiredSubGroupSize;
  if (DeviceFuncInfo) {
    std::deque<std::shared_ptr<DeviceFunctionInfo>> ProcessRequireQueue;
    std::set<std::shared_ptr<DeviceFunctionInfo>> ProcessedSet;
    ProcessRequireQueue.push_back(DeviceFuncInfo);
    ProcessedSet.insert(DeviceFuncInfo);
    // New function name, LocInfo
    std::vector<std::pair<std::string,
                          std::pair<clang::tooling::UnifiedPath, unsigned>>>
        ShflFunctions;
    while (!ProcessRequireQueue.empty()) {
      auto SGSize = ProcessRequireQueue.front()->getSubGroupSize();
      for (auto &Element : SGSize) {
        std::string NewAPIName = std::get<3>(Element);
        unsigned int Size = std::get<0>(Element);
        if (NewAPIName ==
                (MapNames::getDpctNamespace() + "shift_sub_group_right") ||
            NewAPIName ==
                (MapNames::getDpctNamespace() + "shift_sub_group_left") ||
            NewAPIName ==
                (MapNames::getDpctNamespace() + "select_from_sub_group") ||
            NewAPIName ==
                (MapNames::getDpctNamespace() + "permute_sub_group_by_xor")) {
          ShflFunctions.push_back(
              {NewAPIName, {std::get<1>(Element), std::get<2>(Element)}});
        }
        if (RequiredSubGroupSize.isFirstRef) {
          RequiredSubGroupSize.isFirstRef = false;
          if (Size == UINT_MAX) {
            RequiredSubGroupSize.isEvaluated = false;
            RequiredSubGroupSize.SizeStr = std::get<4>(Element);
            ExecutionConfig.SubGroupSize =
                " [[intel::reqd_sub_group_size(dpct_placeholder)]]";
            SubGroupSizeWarning =
                DiagnosticsUtils::getWarningTextAndUpdateUniqueID(
                    Diagnostics::SUBGROUP_SIZE_NOT_EVALUATED,
                    std::get<4>(Element));
          } else {
            RequiredSubGroupSize.Size = Size;
            ExecutionConfig.SubGroupSize =
                " [[intel::reqd_sub_group_size(" + std::to_string(Size) + ")]]";
          }
        } else {
          bool isNeedEmitWarning = true;
          std::string ConflictSize;
          if (RequiredSubGroupSize.isEvaluated) {
            if (Size == UINT_MAX) {
              ConflictSize = "\'" + std::get<4>(Element) + "\'";
            } else if (RequiredSubGroupSize.Size != Size) {
              ConflictSize = std::to_string(Size);
            } else {
              isNeedEmitWarning = false;
            }
          } else {
            if (Size != UINT_MAX) {
              ConflictSize = std::to_string(Size);
            } else if (RequiredSubGroupSize.SizeStr != std::get<4>(Element)) {
              ConflictSize = "\'" + std::get<4>(Element) + "\'";
            } else {
              isNeedEmitWarning = false;
            }
          }
          if (isNeedEmitWarning) {
            DiagnosticsUtils::report(std::get<1>(Element), std::get<2>(Element),
                                     Diagnostics::SUBGROUP_SIZE_CONFLICT, true,
                                     false, NewAPIName, ConflictSize);
          }
        }
      }
      for (auto &Element : ProcessRequireQueue.front()->getCallExprMap()) {
        auto Child = Element.second->getFuncInfo();
        if (Child && ProcessedSet.find(Child) == ProcessedSet.end()) {
          ProcessRequireQueue.push_back(Element.second->getFuncInfo());
          ProcessedSet.insert(Child);
        }
      }
      ProcessRequireQueue.pop_front();
    }
    if (RequiredSubGroupSize.Size != 0 &&
        (SizeOfHighestDimension == 0 ||
         SizeOfHighestDimension < RequiredSubGroupSize.Size)) {
      for (auto &E : ShflFunctions) {
        DiagnosticsUtils::report(E.second.first, E.second.second,
                                 Diagnostics::UNSAFE_WORKGROUP_SIZE, true,
                                 false, RequiredSubGroupSize.Size, E.first,
                                 RequiredSubGroupSize.Size);
      }
    }
  }
  Printer.indent();
  if (!SubGroupSizeWarning.empty()) {
    Printer << "/*" << getNL();
    Printer.indent();
    Printer << SubGroupSizeWarning << getNL();
    Printer.indent();
    Printer << "*/" << getNL();
    Printer.indent();
  }
  if (!getEvent().empty()) {
    Printer << "*" << getEvent() << " = ";
  }

  printStreamBase(Printer);
  if (isDefaultStream()) {
    SubmitStmts.DefaultStreamFlag = true;
  }
  if (DpctGlobalInfo::useExpInOrderQueueEvents() &&
      (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted)) {
    SubmitStmts.ImplicitSyncFlag = true;
  }
  if (SubmitStmts.empty()) {
    printParallelFor(Printer, false);
  } else {
    (Printer << "submit(").newLine();
    printSubmitLambda(Printer);
  }
}
void KernelCallExpr::printSubmitLambda(KernelPrinter &Printer) {
  auto Lamda = Printer.block();
  Printer.line("[&](" + MapNames::getClNamespace() + "handler &cgh) {");
  {
    auto Body = Printer.block();
    SubmitStmts.print(Printer);
    printParallelFor(Printer, true);
  }
  if (getVarMap().hasSync())
    Printer.line("}).wait();");
  else
    Printer.line("});");
}
void KernelCallExpr::printParallelFor(KernelPrinter &Printer, bool IsInSubmit) {
  std::string TemplateArgsStr;
  if (DpctGlobalInfo::isSyclNamedLambda() && hasTemplateArgs()) {
    if (IsForWrapper) {
      TemplateArgsStr = TemplateArgsStrForWrapper;
    } else {
      bool IsNeedWarning = false;
      TemplateArgsStr = getTemplateArguments(IsNeedWarning, false, true);
      if (!TemplateArgsStr.empty() && IsNeedWarning) {
        printWarningMessage(Printer, Diagnostics::UNDEDUCED_TYPE,
                            "dpct_kernel_name");
      }
    }
  }
  if (IsInSubmit) {
    Printer.indent() << "cgh.";
  }
  if (!SubmitStmts.NdRangeList.empty() && DpctGlobalInfo::isCommentsEnabled())
    Printer.line("// run the kernel within defined ND range");
  Printer << "parallel_for";
  if (DpctGlobalInfo::isSyclNamedLambda()) {
    Printer << "<dpct_kernel_name<class " << getName() << "_"
            << LocInfo.LocHash;
    if (hasTemplateArgs())
      Printer << ", " << TemplateArgsStr;
    Printer << ">>";
    requestFeature(HelperFeatureEnum::device_ext);
  }
  (Printer << "(").newLine();
  auto B = Printer.block();
  static std::string CanIgnoreRangeStr3D =
      DpctGlobalInfo::getCtadClass(MapNames::getClNamespace() + "range", 3) +
      "(1, 1, 1)";
  static std::string CanIgnoreRangeStr1D =
      DpctGlobalInfo::getCtadClass(MapNames::getClNamespace() + "range", 1) +
      "(1)";
  if (ExecutionConfig.NdRange != "") {
    Printer.line(ExecutionConfig.NdRange + ",");
    if (!ExecutionConfig.Properties.empty()) {
      Printer << ExecutionConfig.Properties << ", ";
    }
    Printer.line("[=](", MapNames::getClNamespace(), "nd_item<3> ",
                 getItemName(), ")", ExecutionConfig.SubGroupSize, " {");
  } else if (DpctGlobalInfo::getAssumedNDRangeDim() == 1 && getFuncInfo() &&
             MemVarMap::getHeadWithoutPathCompression(
                 &(getFuncInfo()->getVarMap())) &&
             MemVarMap::getHeadWithoutPathCompression(
                 &(getFuncInfo()->getVarMap()))
                     ->Dim == 1) {
    DpctGlobalInfo::printCtadClass(Printer.indent(),
                                   MapNames::getClNamespace() + "nd_range", 1)
        << "(";
    if (ExecutionConfig.GroupSizeFor1D == CanIgnoreRangeStr1D) {
      Printer << ExecutionConfig.LocalSizeFor1D;
    } else if (ExecutionConfig.LocalSizeFor1D == CanIgnoreRangeStr1D) {
      Printer << ExecutionConfig.GroupSizeFor1D;
    } else {
      Printer << ExecutionConfig.GroupSizeFor1D << " * "
              << ExecutionConfig.LocalSizeFor1D;
    }
    Printer << ", ";
    Printer << ExecutionConfig.LocalSizeFor1D;
    (Printer << "), ").newLine();
    if (!ExecutionConfig.Properties.empty()) {
      Printer << ExecutionConfig.Properties << ", ";
    }
    Printer.line("[=](" + MapNames::getClNamespace() + "nd_item<1> ",
                 getItemName(), ")", ExecutionConfig.SubGroupSize, " {");
  } else {
    Printer.indent();
    Printer << MapNames::getClNamespace() + "nd_range<3>(";
    if (ExecutionConfig.GroupSize == CanIgnoreRangeStr3D) {
      Printer << ExecutionConfig.LocalSize;
    } else if (ExecutionConfig.LocalSize == CanIgnoreRangeStr3D) {
      Printer << ExecutionConfig.GroupSize;
    } else {
      Printer << ExecutionConfig.GroupSize << " * "
              << ExecutionConfig.LocalSize;
    }
    Printer << ", ";
    Printer << ExecutionConfig.LocalSize;
    (Printer << "), ").newLine();
    if (!ExecutionConfig.Properties.empty()) {
      Printer << ExecutionConfig.Properties << ", ";
    }
    Printer.line("[=](" + MapNames::getClNamespace() + "nd_item<3> ",
                 getItemName(), ")", ExecutionConfig.SubGroupSize, " {");
  }

  if (getVarMap().hasSync()) {
    std::string SyncParamDecl;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
      SyncParamDecl = "auto atm_" + DpctGlobalInfo::getSyncName() + " = " +
                      MapNames::getClNamespace() + "atomic_ref<unsigned int, " +
                      MapNames::getClNamespace() + "memory_order::seq_cst, " +
                      MapNames::getClNamespace() + "memory_scope::device, " +
                      MapNames::getClNamespace() +
                      "access::address_space::global_space>(" +
                      DpctGlobalInfo::getSyncName() + "[0]);";

    } else {
      SyncParamDecl = "auto atm_" + DpctGlobalInfo::getSyncName() + " = " +
                      MapNames::getClNamespace() + "atomic_ref<unsigned int, " +
                      MapNames::getClNamespace() + "memory_order::seq_cst, " +
                      MapNames::getClNamespace() + "memory_scope::device, " +
                      MapNames::getClNamespace() +
                      "access::address_space::global_space>(*(unsigned int "
                      "*)&" +
                      DpctGlobalInfo::getSyncName() + "[0]);";
    }
    KernelStmts.emplace_back(SyncParamDecl);
  }
  printKernel(Printer);

  if (getVarMap().hasSync() && !IsInSubmit)
    Printer.line("}).wait();");
  else
    Printer.line("});");
}
void KernelCallExpr::printKernel(KernelPrinter &Printer) {
  auto B = Printer.block();
  for (auto &S : KernelStmts) {
    Printer.line(S.StmtStr);
  }
  std::string TemplateArgsStr;
  if (IsForWrapper) {
    if (!TemplateArgsStrForWrapper.empty()) {
      TemplateArgsStr = "<" + TemplateArgsStrForWrapper + ">";
    }
  } else if (hasWrittenTemplateArgs()) {
    bool IsNeedWarning = false;
    TemplateArgsStr =
        buildString("<", getTemplateArguments(IsNeedWarning), ">");
    if (!TemplateArgsStr.empty() && IsNeedWarning) {
      printWarningMessage(Printer, Diagnostics::UNDEDUCED_TYPE,
                          "dpct_kernel_name");
    }
  }
  Printer.indent() << getName() << TemplateArgsStr << "(" << KernelArgs << ");";
  Printer.newLine();
}
template <typename IDTy, typename... Ts>
void KernelCallExpr::printWarningMessage(KernelPrinter &Printer, IDTy MsgID,
                                         Ts &&...Vals) {
  Printer.indent();
  Printer << "/*" << getNL();
  Printer.indent();
  Printer << DiagnosticsUtils::getWarningTextAndUpdateUniqueID(
                 MsgID, std::forward<Ts>(Vals)...)
          << getNL();
  Printer.indent();
  Printer << "*/" << getNL();
}
template <class T> void KernelCallExpr::printStreamBase(T &Printer) {
  if (ExecutionConfig.Stream[0] == '*' || ExecutionConfig.Stream[0] == '&') {
    Printer << "(" << ExecutionConfig.Stream << ")";
  } else {
    Printer << ExecutionConfig.Stream;
  }
  if (isQueuePtr())
    Printer << "->";
  else
    Printer << ".";
}
KernelCallExpr::KernelCallExpr(unsigned Offset,
                               const clang::tooling::UnifiedPath &FilePath,
                               const CUDAKernelCallExpr *KernelCall)
    : CallFunctionExpr(Offset, FilePath, KernelCall), IsSync(false) {
  setIsInMacroDefine(KernelCall);
  setNeedAddLambda(KernelCall);
  buildCallExprInfo(KernelCall);
  buildArgsInfo(KernelCall);
  buildKernelInfo(KernelCall);
}
void KernelCallExpr::addAccessorDecl() {
  auto &VM = getVarMap();
  if (VM.hasExternShared()) {
    addAccessorDecl(VM.getMap(MemVarInfo::Extern).begin()->second);
  }
  addAccessorDecl(MemVarInfo::Local);
  addAccessorDecl(MemVarInfo::Global);
  for (auto &Tex : getTextureObjectList()) {
    if (Tex) {
      if (!Tex->getType()) {
        // Type dpct_placeholder
        Tex->setType("dpct_placeholder/*Fix the type manually*/", 1);
        DiagnosticsUtils::report(getFilePath(), getOffset(),
                                 Diagnostics::UNDEDUCED_TYPE, true, false,
                                 "image_accessor_ext");
      }
      Tex->addDecl(OuterStmts.InitList, SubmitStmts.TextureList,
                   SubmitStmts.SamplerList, getQueueStr());
    }
  }
  for (auto &Tex : VM.getTextureMap()) {
    Tex.second->addDecl(OuterStmts.InitList, SubmitStmts.TextureList,
                        SubmitStmts.SamplerList, getQueueStr());
  }
  for (auto &Tmp : VM.getTempStorageMap()) {
    Tmp.second->addAccessorDecl(SubmitStmts.AccessorList,
                                ExecutionConfig.LocalSize);
  }
}
void KernelCallExpr::buildInfo() {
  CallFunctionExpr::buildInfo();
  TotalArgsSize =
      getVarMap().calculateExtraArgsSize() + calculateOriginArgsSize();
}
void KernelCallExpr::setKernelCallDim() {
  if (auto Ptr = getFuncInfo()) {
    Ptr->setKernelInvoked();
    Ptr->KernelCallBlockDim = std::max(Ptr->KernelCallBlockDim, BlockDim);
    if (GridDim == 1 && BlockDim == 1) {
      if (auto HeadPtr = MemVarMap::getHead(&(Ptr->getVarMap()))) {
        Ptr->getVarMap().Dim = std::max((unsigned int)1, HeadPtr->Dim);
      } else {
        Ptr->getVarMap().Dim = 1;
      }
    } else {
      Ptr->getVarMap().Dim = 3;
    }
  }
}
void KernelCallExpr::buildUnionFindSet() {
  if (auto Ptr = getFuncInfo()) {
    constructUnionFindSetRecursively(Ptr);
  }
}
void KernelCallExpr::addReplacements() {
  if (TotalArgsSize > MapNamesLang::KernelArgTypeSizeMap.at(
                          KernelArgType::KAT_MaxParameterSize))
    DiagnosticsUtils::report(getFilePath(), getOffset(),
                             Diagnostics::EXCEED_MAX_PARAMETER_SIZE, true,
                             false);
  auto R = std::make_shared<ExtReplacement>(getFilePath(), getOffset(), 0,
                                            getReplacement(), nullptr);
  R->setBlockLevelFormatFlag();
  DpctGlobalInfo::getInstance().addReplacement(R);
}
std::string KernelCallExpr::getExtraArguments() {
  if (!getFuncInfo()) {
    return "";
  }

  return getVarMap().getKernelArguments(getFuncInfo()->NonDefaultParamNum,
                                        getFuncInfo()->ParamsNum -
                                            getFuncInfo()->NonDefaultParamNum,
                                        getFilePath());
}
const std::vector<KernelCallExpr::ArgInfo> &KernelCallExpr::getArgsInfo() {
  return ArgsInfo;
}
int KernelCallExpr::calculateOriginArgsSize() const {
  int Size = 0;
  for (auto &ArgInfo : ArgsInfo) {
    Size += ArgInfo.ArgSize;
  }
  return Size;
}
std::string KernelCallExpr::getReplacement() {
  addPropertiesStmt();
  addDevCapCheckStmt();
  addAccessorDecl();
  addStreamDecl();
  buildKernelArgsStmt();

  std::string Result;
  llvm::raw_string_ostream OS(Result);
  KernelPrinter Printer(LocInfo.NL, LocInfo.Indent, OS);
  print(Printer);
  auto ResultStr = Printer.str();
  if (NeedLambda) {
    ResultStr = "[&]()" + Printer.str() + "()";
  }
  return ResultStr;
}
std::shared_ptr<KernelCallExpr> KernelCallExpr::buildFromCudaLaunchKernel(
    const std::pair<clang::tooling::UnifiedPath, unsigned> &LocInfo,
    const CallExpr *CE, bool IsAssigned) {
  auto LaunchFD = CE->getDirectCallee();
  if (!LaunchFD || (LaunchFD->getName() != "cudaLaunchKernel" &&
                    LaunchFD->getName() != "cudaLaunchCooperativeKernel")) {
    return std::shared_ptr<KernelCallExpr>();
  }
  auto Kernel = std::shared_ptr<KernelCallExpr>(
      new KernelCallExpr(LocInfo.second, LocInfo.first));
  // Call the lambda function with default return value.
  if (IsAssigned) {
    Kernel->setNeedDefaultRet();
    Kernel->setNeedAddLambda();
  }
  Kernel->buildLocationInfo(CE);
  Kernel->buildExecutionConfig(
      ArrayRef<const Expr *>{CE->getArg(1), CE->getArg(2), CE->getArg(4),
                             CE->getArg(5)},
      CE);
  Kernel->buildNeedBracesInfo(CE);
  const FunctionDecl *FD = nullptr;
  if (auto Callee = getAddressedRef(CE->getArg(0), true, &FD)) {
    Kernel->buildCalleeInfo(Callee, std::nullopt);
    auto FuncInfo = Kernel->getFuncInfo();
    if (FD && FuncInfo) {
      auto ArgsArray = ExprAnalysis::ref(CE->getArg(3));
      if (!isa<DeclRefExpr>(CE->getArg(3)->IgnoreImplicitAsWritten())) {
        ArgsArray = "(" + ArgsArray + ")";
      }
      Kernel->resizeTextureObjectList(FD->getNumParams());
      for (auto &Parm : FD->parameters()) {
        Kernel->ArgsInfo.emplace_back(Parm, ArgsArray, Kernel.get());
        if (!isDeviceCopyable(Parm->getType(), nullptr)) {
          DiagnosticsUtils::report(
              LocInfo.first, LocInfo.second,
              Diagnostics::NOT_DEVICE_COPYABLE_ADD_SPECIALIZATION, true, true,
              DpctGlobalInfo::getOriginalTypeName(Parm->getType()));
        }
      }
    }
  } else {
    Kernel->buildCalleeInfo(CE->getArg(0), std::nullopt);
    DiagnosticsUtils::report(LocInfo.first, LocInfo.second,
                             Diagnostics::UNDEDUCED_KERNEL_FUNCTION_POINTER,
                             true, false);
  }
  return Kernel;
}
std::shared_ptr<KernelCallExpr>
KernelCallExpr::buildForWrapper(clang::tooling::UnifiedPath FilePath,
                                const FunctionDecl *FD) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto Kernel =
      std::shared_ptr<KernelCallExpr>(new KernelCallExpr(0, FilePath));
  Kernel->IsForWrapper = true;
  Kernel->Name = FD->getNameAsString();
  Kernel->setFuncInfo(DeviceFunctionDecl::LinkRedecls(FD));
  Kernel->ExecutionConfig.Config[0] = "";
  Kernel->ExecutionConfig.Config[1] = "";
  Kernel->ExecutionConfig.Config[2] = "localMemSize";
  Kernel->ExecutionConfig.Config[3] = "queue";
  Kernel->ExecutionConfig.Config[4] = "nr";
  Kernel->ExecutionConfig.IsDefaultStream = false;
  Kernel->ExecutionConfig.IsQueuePtr = false;
  Kernel->NeedBraces = false;
  Kernel->getFuncInfo()->getVarMap().Dim = 3;
  Kernel->resizeTextureObjectList(FD->getNumParams());
  for (auto &Parm : FD->parameters()) {
    Kernel->ArgsInfo.emplace_back(Parm, Kernel.get());
  }
  Kernel->LocInfo.NL = getNL();
  Kernel->LocInfo.Indent = getIndent(FD->getBeginLoc(), SM).str() + "    ";
  return Kernel;
}
void KernelCallExpr::buildArgsInfo(const CallExpr *CE) {
  KernelArgumentAnalysis Analysis(IsInMacroDefine);
  auto KCallSpellingRange =
      getTheLastCompleteImmediateRange(CE->getBeginLoc(), CE->getEndLoc());
  Analysis.setCallSpelling(KCallSpellingRange.first, KCallSpellingRange.second);
  auto &TexList = getTextureObjectList();
  const auto *FD = CE->getDirectCallee();
  const auto *FTD = FD ? FD->getPrimaryTemplate() : nullptr;
  for (unsigned Idx = 0; Idx < CE->getNumArgs(); ++Idx) {
    auto Arg = CE->getArg(Idx);
    auto CallDefRange = getDefinitionRange(CE->getBeginLoc(), CE->getEndLoc());
    auto ArgString = getStringInRange(
        Arg->getSourceRange(), CallDefRange.getBegin(), CallDefRange.getEnd());
    if (auto Obj = TexList[Idx]) {
      ArgsInfo.emplace_back(Obj, this, ArgString);
    } else {
      bool Used = true;
      if (auto *ArgDRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts()))
        Used = isArgUsedAsLvalueUntil(ArgDRE, CE);
      ArgsInfo.emplace_back(FD ? FD->parameters()[Idx] : nullptr, Analysis, Arg,
                            Used, Idx, this);
      if (FTD && FTD->getTemplatedDecl()
                     ->parameters()[Idx]
                     ->getType()
                     ->isDependentType())
        ArgsInfo.back().IsDependentType = true;
    }
  }
}
std::string KernelCallExpr::getQueueStr() const {
  if (isDefaultStream())
    return "";
  std::string Ret;
  if (isQueuePtr())
    Ret = "*";
  return Ret += ExecutionConfig.Stream;
}
void KernelCallExpr::buildKernelInfo(const CUDAKernelCallExpr *KernelCall) {
  buildLocationInfo(KernelCall);
  if (auto Config = KernelCall->getConfig())
    buildExecutionConfig(Config->arguments(), KernelCall);
  buildNeedBracesInfo(KernelCall);
}
void KernelCallExpr::setIsInMacroDefine(const CUDAKernelCallExpr *KernelCall) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  // Check if the whole kernel call is in macro arg
  auto CallBegin = KernelCall->getBeginLoc();
  auto CallEnd = KernelCall->getEndLoc();

  auto Range =
      getDefinitionRange(KernelCall->getBeginLoc(), KernelCall->getEndLoc());
  auto ItMatch = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      getCombinedStrFromLoc(Range.getBegin()));
  if (ItMatch != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
    IsInMacroDefine = true;
    return;
  }

  if (SM.isMacroArgExpansion(CallBegin) && SM.isMacroArgExpansion(CallEnd) &&
      isLocInSameMacroArg(CallBegin, CallEnd)) {
    IsInMacroDefine = false;
    return;
  }

  auto CalleeSpelling = KernelCall->getCallee()->getBeginLoc();
  if (SM.isMacroArgExpansion(CalleeSpelling)) {
    CalleeSpelling = SM.getImmediateExpansionRange(CalleeSpelling).getBegin();
  }
  CalleeSpelling = SM.getSpellingLoc(CalleeSpelling);

  ItMatch = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      getCombinedStrFromLoc(CalleeSpelling));
  if (ItMatch != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
    IsInMacroDefine = true;
  }
}
// Check if the kernel call is in a ParenExpr
void KernelCallExpr::setNeedAddLambda(const CUDAKernelCallExpr *KernelCall) {
  if (dyn_cast<ParenExpr>(getParentStmt(KernelCall))) {
    NeedLambda = true;
  }
}
void KernelCallExpr::setNeedAddLambda() { NeedLambda = true; }
void KernelCallExpr::setNeedDefaultRet() { NeedDefaultRetValue = true; }
void KernelCallExpr::buildNeedBracesInfo(const CallExpr *KernelCall) {
  NeedBraces = true;
  auto &Context = dpct::DpctGlobalInfo::getContext();
  // if parent is CompoundStmt, then find if it has more than 1 children.
  // else if parent is ExprWithCleanups, then do further check.
  // else it must be case like:  if/for/while() kernel-call, pair of
  // braces are needed.
  auto Parents = Context.getParents(*KernelCall);
  while (Parents.size() == 1) {
    if (auto *Parent = Parents[0].get<CompoundStmt>()) {
      NeedBraces = (Parent->size() > 1);
      return;
    } else if (Parents[0].get<ExprWithCleanups>()) {
      // treat ExprWithCleanups same as CUDAKernelCallExpr when they show
      // up together
      Parents = Context.getParents(Parents[0]);
    } else {
      return;
    }
  }
}
void KernelCallExpr::buildLocationInfo(const CallExpr *KernelCall) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  SourceLocation Begin = KernelCall->getBeginLoc();
  LocInfo.NL = getNL();
  LocInfo.Indent = getIndent(Begin, SM).str();
  LocInfo.LocHash = getHashAsString(Begin.printToString(SM)).substr(0, 6);
  if (IsInMacroDefine) {
    LocInfo.NL = "\\" + LocInfo.NL;
  }
}
template <class ArgsRange>
void KernelCallExpr::buildExecutionConfig(const ArgsRange &ConfigArgs,
                                          const CallExpr *KernelCall) {
  bool NeedTypeCast = true;
  int Idx = 0;
  auto KCallSpellingRange = getTheLastCompleteImmediateRange(
      KernelCall->getBeginLoc(), KernelCall->getEndLoc());
  for (auto Arg : ConfigArgs) {
    KernelConfigAnalysis A(IsInMacroDefine);
    A.setCallSpelling(KCallSpellingRange.first, KCallSpellingRange.second);
    A.analyze(Arg, Idx, Idx < 2);
    ExecutionConfig.Config[Idx] = A.getReplacedString();
    if (Idx == 0) {
      ExecutionConfig.GroupDirectRef = A.isDirectRef();
      if (DpctGlobalInfo::useSYCLCompat() && A.isDim3Var())
        ExecutionConfig.Config[Idx] =
            "static_cast<" + MapNames::getClNamespace() + "range<3>>(" +
            ExecutionConfig.Config[Idx] + ")";
    } else if (Idx == 1) {
      ExecutionConfig.LocalDirectRef = A.isDirectRef();
      if (DpctGlobalInfo::useSYCLCompat() && A.isDim3Var())
        ExecutionConfig.Config[Idx] =
            "static_cast<" + MapNames::getClNamespace() + "range<3>>(" +
            ExecutionConfig.Config[Idx] + ")";
      // Using another analysis because previous analysis may return directly
      // when in macro is true.
      // Here set the argument of KFA as false, so it will not return directly.
      KernelConfigAnalysis KFA(false);
      KFA.setCallSpelling(KCallSpellingRange.first, KCallSpellingRange.second);
      KFA.analyze(Arg, 1, true);
      if (KFA.isNeedEmitWGSizeWarning())
        DiagnosticsUtils::report(getFilePath(), getOffset(),
                                 Diagnostics::EXCEED_MAX_WORKGROUP_SIZE, true,
                                 false);
      SizeOfHighestDimension = KFA.getSizeOfHighestDimension();
    } else if (Idx == 3) {
      llvm::SmallVector<clang::ast_matchers::BoundNodes, 1U> DREResults;
      DREResults = findDREInScope(Arg);
      for (auto &Result : DREResults) {
        const DeclRefExpr *MatchedDRE =
            Result.getNodeAs<DeclRefExpr>("VarReference");
        if (!MatchedDRE)
          continue;
        auto Type = MatchedDRE->getDecl()->getType().getAsString();

        if (Type.find("cudaStream_t") != std::string::npos ||
            dyn_cast_or_null<CXXDependentScopeMemberExpr>(
                getParentStmt(MatchedDRE)))
          NeedTypeCast = false;
      }
    }
    ++Idx;
  }

  Idx = 0;
  for (auto Arg : ConfigArgs) {
    if (Idx > 1)
      break;
    KernelConfigAnalysis AnalysisTry1D(IsInMacroDefine);
    AnalysisTry1D.IsTryToUseOneDimension = true;
    AnalysisTry1D.analyze(Arg, Idx, Idx < 2);
    if (Idx == 0) {
      GridDim = AnalysisTry1D.Dim;
      ExecutionConfig.GroupSizeFor1D = AnalysisTry1D.getReplacedString();
      if (DpctGlobalInfo::useSYCLCompat() && AnalysisTry1D.isDim3Var())
        ExecutionConfig.GroupSizeFor1D =
            "static_cast<" + MapNames::getClNamespace() + "range<1>>(" +
            ExecutionConfig.GroupSizeFor1D + ")";
    } else if (Idx == 1) {
      BlockDim = AnalysisTry1D.Dim;
      ExecutionConfig.LocalSizeFor1D = AnalysisTry1D.getReplacedString();
      if (DpctGlobalInfo::useSYCLCompat() && AnalysisTry1D.isDim3Var())
        ExecutionConfig.LocalSizeFor1D =
            "static_cast<" + MapNames::getClNamespace() + "range<1>>(" +
            ExecutionConfig.LocalSizeFor1D + ")";
    }
    ++Idx;
  }

  if (ExecutionConfig.Stream == "0") {
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    ExecutionConfig.Stream = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
    ExecutionConfig.IsQueuePtr = false;
    buildTempVariableMap(Index, *ConfigArgs.begin(),
                         HelperFuncType::HFT_DefaultQueue);
  } else if (NeedTypeCast) {
    ExecutionConfig.Stream =
        buildString("((sycl::queue*)(", ExecutionConfig.Stream, "))");
  }
}
void KernelCallExpr::removeExtraIndent() {
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      getFilePath(), getOffset() - LocInfo.Indent.length(),
      LocInfo.Indent.length(), "", nullptr));
}

void KernelCallExpr::addPropertiesStmt() {
  if (DpctGlobalInfo::useRootGroup()) {
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    OS << "auto exp_props = "
          "sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::"
          "experimental::use_root_sync};";
    ExecutionConfig.Properties = "exp_props";
    OuterStmts.OthersList.emplace_back(Str);
  }
}
void KernelCallExpr::addDevCapCheckStmt() {
  llvm::SmallVector<std::string> AspectList;
  if (getVarMap().hasBF64()) {
    AspectList.push_back(MapNames::getClNamespace() + "aspect::fp64");
  }
  if (getVarMap().hasBF16()) {
    AspectList.push_back(MapNames::getClNamespace() + "aspect::fp16");
  }
  if (!AspectList.empty()) {
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    OS << MapNames::getDpctNamespace() << "has_capability_or_fail(";
    if (auto Iter = MapNames::CustomHelperFunctionMap.find(getQueueKind());
        Iter != MapNames::CustomHelperFunctionMap.end()) {
      OS << Iter->second << ".";
    } else {
      requestFeature(HelperFeatureEnum::device_ext);
      printStreamBase(OS);
    }
    OS << "get_device(), ";
    OS << "{" << AspectList.front();
    for (size_t i = 1; i < AspectList.size(); ++i) {
      OS << ", " << AspectList[i];
    }
    OS << "});";
    OuterStmts.OthersList.emplace_back(OS.str());
  }
}
void KernelCallExpr::addAccessorDecl(MemVarInfo::VarScope Scope) {
  for (auto &VI : getVarMap().getMap(Scope)) {
    addAccessorDecl(VI.second);
  }
}
void KernelCallExpr::addAccessorDecl(std::shared_ptr<MemVarInfo> VI) {
  if (!VI->isUseHelperFunc()) {
    return;
  }
  if (!VI->isShared()) {
    requestFeature(HelperFeatureEnum::device_ext);
    OuterStmts.InitList.emplace_back(VI->getInitStmt(getQueueStr()));
    if (VI->isLocal()) {
      SubmitStmts.MemoryList.emplace_back(
          VI->getMemoryDecl(ExecutionConfig.ExternMemSize));
    } else if (getFilePath() != VI->getFilePath() &&
               !isIncludedFile(getFilePath(), VI->getFilePath())) {
      // Global variable definition and global variable reference are not in the
      // same file, and are not a share variable, insert extern variable
      // declaration.
      OuterStmts.ExternList.emplace_back(VI->getExternGlobalVarDecl());
    }
  }
  VI->appendAccessorOrPointerDecl(ExecutionConfig.ExternMemSize,
                                  EmitSizeofWarning, SubmitStmts.AccessorList,
                                  SubmitStmts.PtrList,
                                  std::make_pair(getFilePath(), getOffset()));
  if (VI->isTypeDeclaredLocal()) {
    if (DiagnosticsUtils::report(getFilePath(), getOffset(),
                                 Diagnostics::TYPE_IN_FUNCTION, false, false,
                                 VI->getName(), VI->getLocalTypeName())) {
      if (!SubmitStmts.AccessorList.empty()) {
        SubmitStmts.AccessorList.back().Warnings.push_back(
            DiagnosticsUtils::getWarningTextAndUpdateUniqueID(
                Diagnostics::TYPE_IN_FUNCTION, VI->getName(),
                VI->getLocalTypeName()));
      }
    }
  }
}
void KernelCallExpr::addStreamDecl() {
  if (getVarMap().hasStream())
    SubmitStmts.StreamList.emplace_back(
        buildString(MapNames::getClNamespace() + "stream ",
                    DpctGlobalInfo::getStreamName(), "(64 * 1024, 80, cgh);"));
  if (getVarMap().hasSync()) {
    auto DefaultQueue = DpctGlobalInfo::getDefaultQueueFreeFuncCall();
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      OuterStmts.OthersList.emplace_back(
          buildString(MapNames::getDpctNamespace(), "global_memory<",
                      MapNames::getDpctNamespace(), "byte_t, 1> d_",
                      DpctGlobalInfo::getSyncName(), "(4);"));

      OuterStmts.OthersList.emplace_back(buildString(
          "d_", DpctGlobalInfo::getSyncName(), ".init(", DefaultQueue, ");"));

      SubmitStmts.SyncList.emplace_back(
          buildString("auto ", DpctGlobalInfo::getSyncName(), " = ",
                      MapNames::getDpctNamespace(), "get_access(d_",
                      DpctGlobalInfo::getSyncName(), ".get_ptr(), cgh);"));

      OuterStmts.OthersList.emplace_back(buildString(
          MapNames::getDpctNamespace(), "dpct_memset(d_",
          DpctGlobalInfo::getSyncName(), ".get_ptr(), 0, sizeof(int));"));

      requestFeature(HelperFeatureEnum::device_ext);
    } else {
      OuterStmts.OthersList.emplace_back(buildString(
          MapNames::getDpctNamespace(), "global_memory<unsigned int, 0> d_",
          DpctGlobalInfo::getSyncName(), "(0);"));
      OuterStmts.OthersList.emplace_back(buildString(
          "unsigned *", DpctGlobalInfo::getSyncName(), " = d_",
          DpctGlobalInfo::getSyncName(), ".get_ptr(", DefaultQueue, ");"));

      OuterStmts.OthersList.emplace_back(
          buildString(DefaultQueue, ".memset(", DpctGlobalInfo::getSyncName(),
                      ", 0, sizeof(int)).wait();"));

      requestFeature(HelperFeatureEnum::device_ext);
    }
  }
}
void KernelCallExpr::buildKernelArgsStmt() {
  size_t ArgCounter = 0;
  KernelArgs = "";
  for (auto &Arg : getArgsInfo()) {
    // if current arg is the first arg with default value, insert extra args
    // before current arg
    if (getFuncInfo()) {
      if (ArgCounter == getFuncInfo()->NonDefaultParamNum) {
        KernelArgs += getExtraArguments();
      }
    }
    if (ArgCounter != 0)
      KernelArgs += ", ";
    if (Arg.IsDoublePointer &&
        DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      DiagnosticsUtils::report(getFilePath(), getOffset(),
                               Diagnostics::VIRTUAL_POINTER, true, false,
                               Arg.getArgString());
    }

    if (Arg.TryGetBuffer) {
      auto BufferName = Arg.getIdStringWithSuffix("buf");
      // If Arg is used as lvalue after its most recent memory allocation,
      // offsets are necessary; otherwise, offsets are not necessary.

      std::string TypeStr = Arg.getTypeString();
      if (Arg.IsDeviceRandomGeneratorType) {
        TypeStr = TypeStr + " *";
      }
      if (Arg.IsDependentType) {
        TypeStr = "decltype(" + Arg.getArgString() + ")";
      }

      if (DpctGlobalInfo::isOptimizeMigration() && getFuncInfo() &&
          !(getFuncInfo()->isParameterReferenced(ArgCounter))) {
        // Typecast can be removed only when it is a template function and
        // all template arguments are specified explicitly.
        if (IsAllTemplateArgsSpecified)
          KernelArgs += buildString("nullptr");
        else
          KernelArgs += buildString("(", TypeStr, ")nullptr");
      } else {
        if (Arg.IsUsedAsLvalueAfterMalloc) {
          requestFeature(HelperFeatureEnum::device_ext);
          SubmitStmts.AccessorList.emplace_back(buildString(
              MapNames::getDpctNamespace() + "access_wrapper ",
              Arg.getIdStringWithSuffix("acc"), "(", Arg.getArgString(),
              Arg.IsDefinedOnDevice ? ".get_ptr()" : "", ", cgh);"));
          KernelArgs += buildString(Arg.getIdStringWithSuffix("acc"),
                                    ".get_raw_pointer()");
        } else {
          requestFeature(HelperFeatureEnum::device_ext);
          SubmitStmts.AccessorList.emplace_back(buildString(
              "auto ", Arg.getIdStringWithSuffix("acc"),
              " = " + MapNames::getDpctNamespace() + "get_access(",
              Arg.getArgString(), Arg.IsDefinedOnDevice ? ".get_ptr()" : "",
              ", cgh);"));
          KernelArgs +=
              buildString("&", Arg.getIdStringWithSuffix("acc"), "[0]");
        }
      }
    } else if (Arg.IsRedeclareRequired || IsInMacroDefine) {
      std::string TypeStr = "auto";
      if (Arg.HasImplicitConversion && !Arg.getTypeString().empty() &&
          !Arg.IsDependentType) {
        TypeStr = Arg.getTypeString();
      }
      SubmitStmts.CommandGroupList.emplace_back(
          buildString(TypeStr, " ", Arg.getIdStringWithIndex(), " = ",
                      Arg.getArgString(), ";"));
      KernelArgs += Arg.getIdStringWithIndex();
    } else if (Arg.Texture && !DpctGlobalInfo::useExtBindlessImages()) {
      ParameterStream OS;
      Arg.Texture->getKernelArg(OS);
      KernelArgs += OS.Str;
    } else {
      KernelArgs += Arg.getArgString();
    }
    ArgCounter += 1;
  }

  // if all params have no default value, insert extra args in the end of params
  if (getFuncInfo()) {
    if (ArgCounter == getFuncInfo()->NonDefaultParamNum) {
      KernelArgs = KernelArgs + getExtraArguments();
    }
  }

  if (KernelArgs.empty()) {
    KernelArgs += getExtraArguments();
  }
}
KernelPrinter &KernelCallExpr::SubmitStmtsList::print(KernelPrinter &Printer) {
  printList(Printer, StreamList);
  printList(Printer, SyncList);
  printList(Printer, MemoryList);
  printList(Printer, RangeList, "ranges used for accessors to device memory");
  printList(Printer, PtrList, "pointers to device memory");
  printList(Printer, AccessorList, "accessors to device memory");
  printList(Printer, TextureList, "accessors to image objects");
  printList(Printer, SamplerList, "sampler of image objects");
  printList(Printer, NdRangeList,
            "ranges to define ND iteration space for the kernel");
  printList(Printer, CommandGroupList, "helper variables defined");
  if (ImplicitSyncFlag) {
    if (DefaultStreamFlag) {
      Printer.line("cgh.depends_on(dpct::get_current_device().get_in_order_"
                   "queues_last_events());");
    } else {
      Printer.line("cgh.depends_on(dpct::get_default_queue().ext_oneapi_get_"
                   "last_event());");
    }
    Printer.newLine();
  }
  return Printer;
}
bool KernelCallExpr::SubmitStmtsList::empty() const noexcept {
  return !ImplicitSyncFlag && CommandGroupList.empty() && NdRangeList.empty() &&
         AccessorList.empty() && PtrList.empty() && MemoryList.empty() &&
         RangeList.empty() && TextureList.empty() && SamplerList.empty() &&
         StreamList.empty() && SyncList.empty();
}
KernelPrinter &KernelCallExpr::SubmitStmtsList::printList(
    KernelPrinter &Printer, const StmtList &List, StringRef Comments) {
  if (List.empty())
    return Printer;
  if (!Comments.empty() && DpctGlobalInfo::isCommentsEnabled())
    Printer.line("// ", Comments);
  Printer << List;
  return Printer.newLine();
}
KernelPrinter &KernelCallExpr::OuterStmtsList::print(KernelPrinter &Printer) {
  printList(Printer, ExternList);
  printList(Printer, InitList, "init global memory");
  printList(Printer, OthersList);
  return Printer;
}
bool KernelCallExpr::OuterStmtsList::empty() const noexcept {
  return InitList.empty() && ExternList.empty() && OthersList.empty();
}
KernelPrinter &KernelCallExpr::OuterStmtsList::printList(KernelPrinter &Printer,
                                                         const StmtList &List,
                                                         StringRef Comments) {
  if (List.empty())
    return Printer;
  if (!Comments.empty() && DpctGlobalInfo::isCommentsEnabled())
    Printer.line("// ", Comments);
  Printer << List;
  return Printer.newLine();
}
///// class CudaMallocInfo /////
const VarDecl *CudaMallocInfo::getMallocVar(const Expr *Arg) {
  if (auto UO = dyn_cast<UnaryOperator>(Arg->IgnoreImpCasts())) {
    if (UO->getOpcode() == UO_AddrOf) {
      return getDecl(UO->getSubExpr());
    }
  }
  return nullptr;
}
const VarDecl *CudaMallocInfo::getDecl(const Expr *E) {
  if (auto DeclRef = dyn_cast<DeclRefExpr>(E->IgnoreImpCasts()))
    return dyn_cast<VarDecl>(DeclRef->getDecl());
  return nullptr;
}
void CudaMallocInfo::setSizeExpr(const Expr *SizeExpression) {
  ArgumentAnalysis A(SizeExpression, false);
  A.analyze();
  Size = A.getReplacedString();
}
void CudaMallocInfo::setSizeExpr(const Expr *N, const Expr *ElemSize) {
  ArgumentAnalysis AN(N, false);
  ArgumentAnalysis AElemSize(ElemSize, false);
  AN.analyze();
  AElemSize.analyze();
  Size = "(" + AN.getReplacedString() + ")*(" + AElemSize.getReplacedString() +
         ")";
}
std::string CudaMallocInfo::getAssignArgs(const std::string &TypeName) {
  return Name + ", " + Size;
}

///// end /////
int HostDeviceFuncInfo::MaxId = 0;

const int TextureObjectInfo::ReplaceTypeLength = strlen("cudaTextureObject_t");

#define TYPE_CAST(qual_type, type) dyn_cast<type>(qual_type)
#define ARG_TYPE_CAST(type) TYPE_CAST(ArgType, type)
#define PARM_TYPE_CAST(type) TYPE_CAST(ParmType, type)

template <class T>
void setTypeTemplateArgument(std::vector<TemplateArgumentInfo> &TAILis,
                             unsigned Idx, T Ty) {
  if (Idx < TAILis.size()) {
    auto &TA = TAILis[Idx];
    if (TA.isNull())
      TA.setAsType(Ty);
  }
}
template <class T>
void setNonTypeTemplateArgument(std::vector<TemplateArgumentInfo> &TAILis,
                                unsigned Idx, T Ty) {
  if (Idx < TAILis.size()) {
    auto &TA = TAILis[Idx];
    if (TA.isNull())
      TA.setAsNonType(Ty);
  }
}

bool getInnerType(QualType &Ty, TypeLoc &TL) {
  if (auto TypedefTy = dyn_cast<TypedefType>(Ty)) {
    if (!TemplateArgumentInfo::isPlaceholderType(TypedefTy->desugar())) {
      Ty = TypedefTy->desugar();
      TL = TypedefTy->getDecl()->getTypeSourceInfo()->getTypeLoc();
      return true;
    }
  } else if (auto ElaboratedTy = dyn_cast<ElaboratedType>(Ty)) {
    Ty = ElaboratedTy->getNamedType();
    if (TL)
      TL = TYPELOC_CAST(ElaboratedTypeLoc).getNamedTypeLoc();
    return true;
  }
  return false;
}

void deduceTemplateArgumentFromType(std::vector<TemplateArgumentInfo> &TAIList,
                                    QualType ParmType, QualType ArgType,
                                    TypeLoc TL = TypeLoc());

template <class NonTypeValueT>
void deduceNonTypeTemplateArgument(std::vector<TemplateArgumentInfo> &TAIList,
                                   const Expr *Parm,
                                   const NonTypeValueT &Value) {
  Parm = Parm->IgnoreImplicitAsWritten();
  if (auto DRE = dyn_cast<DeclRefExpr>(Parm)) {
    if (auto NTTPD = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl())) {
      setNonTypeTemplateArgument(TAIList, NTTPD->getIndex(), Value);
    }
  } else if (auto C = dyn_cast<ConstantExpr>(Parm)) {
    deduceNonTypeTemplateArgument(TAIList, C->getSubExpr(), Value);
  } else if (auto S = dyn_cast<SubstNonTypeTemplateParmExpr>(Parm)) {
    deduceNonTypeTemplateArgument(TAIList, S->getReplacement(), Value);
  }
}

void deduceTemplateArgumentFromTemplateArgs(
    std::vector<TemplateArgumentInfo> &TAIList, const TemplateArgument &Parm,
    const TemplateArgument &Arg,
    const TemplateArgumentLoc &ArgLoc = TemplateArgumentLoc()) {
  switch (Parm.getKind()) {
  case TemplateArgument::Expression:
    switch (Arg.getKind()) {
    case TemplateArgument::Expression:
      deduceNonTypeTemplateArgument(TAIList, Parm.getAsExpr(), Arg.getAsExpr());
      return;
    case TemplateArgument::Integral:
      if (ArgLoc.getArgument().isNull())
        deduceNonTypeTemplateArgument(TAIList, Parm.getAsExpr(),
                                      Arg.getAsIntegral());
      else
        deduceNonTypeTemplateArgument(TAIList, Parm.getAsExpr(),
                                      ArgLoc.getSourceExpression());
      break;
    default:
      break;
    }
    break;
  case TemplateArgument::Type:
    switch (Arg.getKind()) {
    case TemplateArgument::Expression:
      deduceTemplateArgumentFromType(TAIList, Parm.getAsType(),
                                     Arg.getAsExpr()->getType());
      return;
    case TemplateArgument::Type:
      if (ArgLoc.getArgument().isNull()) {
        deduceTemplateArgumentFromType(TAIList, Parm.getAsType(),
                                       Arg.getAsType());
      } else {
        deduceTemplateArgumentFromType(
            TAIList, Parm.getAsType(), ArgLoc.getTypeSourceInfo()->getType(),
            ArgLoc.getTypeSourceInfo()->getTypeLoc());
      }
      break;
    default:
      // Currently dpct does not collect enough information
      // to deduce from other kinds of template arguments.
      // Stop the deduction.
      return;
    }
    break;
  default:
    break;
  }
}

bool compareTemplateName(std::string N1, TemplateName N2) {
  std::string NameStr;
  llvm::raw_string_ostream OS(NameStr);
  PrintFullTemplateName(OS, DpctGlobalInfo::getContext().getPrintingPolicy(),
                        N2);
  OS.flush();
  return N1.compare(NameStr);
}

bool compareTemplateName(TemplateName N1, TemplateName N2) {
  std::string NameStr;
  llvm::raw_string_ostream OS(NameStr);
  PrintFullTemplateName(OS, DpctGlobalInfo::getContext().getPrintingPolicy(),
                        N1);
  OS.flush();
  return compareTemplateName(NameStr, N2);
}

void deduceTemplateArgumentFromTemplateSpecialization(
    std::vector<TemplateArgumentInfo> &TAIList, QualType ParmType,
    QualType ArgType, TypeLoc TL = TypeLoc()) {
  auto ParmTST = dyn_cast<TemplateSpecializationType>(ParmType);
  auto ParmArgs = ParmTST->template_arguments();
  switch (ArgType->getTypeClass()) {
  case Type::Record:
    if (auto CTSD = dyn_cast<ClassTemplateSpecializationDecl>(
            ARG_TYPE_CAST(RecordType)->getDecl())) {
      if (compareTemplateName(CTSD->getName().data(),
                              ParmTST->getTemplateName())) {
        // If the names of 2 template classes are different
        // DPCT should stop the deduction.
        return;
      }
      const auto *TA = CTSD->getTemplateArgsAsWritten();
      if (TA && TA->getTemplateArgs()
                        ->getTypeSourceInfo()
                        ->getType()
                        ->getTypeClass() == Type::TemplateSpecialization) {
        auto TL = TA->getTemplateArgs()->getTypeSourceInfo()->getTypeLoc();
        auto &TSTL = TYPELOC_CAST(TemplateSpecializationTypeLoc);
        for (unsigned i = 0; i < TSTL.getNumArgs(); ++i) {
          deduceTemplateArgumentFromTemplateArgs(
              TAIList, ParmArgs[i], TSTL.getArgLoc(i).getArgument(),
              TSTL.getArgLoc(i));
        }
      }
    }
    break;
  case Type::TemplateSpecialization: {
    // To support following alias template cases:
    // template<size_t N>
    // using new_type = old_type<size_t, N>
    // Since new_type(the ArgType) takes 1 arg and old_type(the ParmTST)
    // takes 2 args, need to get the alias type of ArgType and recursively
    // call deduceTemplateArgumentFromType
    auto TST = ARG_TYPE_CAST(TemplateSpecializationType);
    if (TST->isTypeAlias()) {
      deduceTemplateArgumentFromType(TAIList, ParmType, TST->getAliasedType());
    } else if (compareTemplateName(TST->getTemplateName(),
                                   ParmTST->getTemplateName())) {
      // If the name of 2 template classes are different
      // DPCT should stop the deduction.
      return;
    } else {
      if (TL) {
        auto &TSTL = TYPELOC_CAST(TemplateSpecializationTypeLoc);
        unsigned i;
        // Parm uses template parameter pack, return
        if (TSTL.getNumArgs() > ParmArgs.size()) {
          return;
        }
        for (i = 0; i < TSTL.getNumArgs(); ++i) {
          deduceTemplateArgumentFromTemplateArgs(
              TAIList, ParmArgs[i], TSTL.getArgLoc(i).getArgument(),
              TSTL.getArgLoc(i));
        }
      } else {
        auto Args = TST->template_arguments();
        // Parm uses template parameter pack, return
        if (Args.size() > ParmArgs.size()) {
          return;
        }
        for (unsigned i = 0; i < Args.size(); ++i) {
          deduceTemplateArgumentFromTemplateArgs(TAIList, ParmArgs[i], Args[i]);
        }
      }
    }
    break;
  }
  default:
    break;
  }
}

TypeLoc getPointeeTypeLoc(TypeLoc TL) {
  if (!TL)
    return TL;
  switch (TL.getTypeLocClass()) {
  case TypeLoc::ConstantArray:
  case TypeLoc::DependentSizedArray:
  case TypeLoc::IncompleteArray:
    return TYPELOC_CAST(ArrayTypeLoc).getElementLoc();
  case TypeLoc::Pointer:
    return TYPELOC_CAST(PointerTypeLoc).getPointeeLoc();
  default:
    return TypeLoc();
  }
}

void deduceTemplateArgumentFromArrayElement(
    std::vector<TemplateArgumentInfo> &TAIList, QualType ParmType,
    QualType ArgType, TypeLoc TL = TypeLoc()) {
  const ArrayType *ParmArray = PARM_TYPE_CAST(ArrayType);
  const ArrayType *ArgArray = ARG_TYPE_CAST(ArrayType);
  if (!ParmArray || !ArgArray) {
    return;
  }
  deduceTemplateArgumentFromType(TAIList, ParmArray->getElementType(),
                                 ArgArray->getElementType(),
                                 getPointeeTypeLoc(TL));
}

void deduceTemplateArgumentFromType(std::vector<TemplateArgumentInfo> &TAIList,
                                    QualType ParmType, QualType ArgType,
                                    TypeLoc TL) {
  ParmType = ParmType.getCanonicalType();
  if (!ParmType->isDependentType())
    return;

  if (TL) {
    TL = TL.getUnqualifiedLoc();
    if (TL.getTypePtr()->getTypeClass() != ArgType->getTypeClass() ||
        TL.getTypeLocClass() == TypeLoc::SubstTemplateTypeParm)
      TL = TypeLoc();
  }

  switch (ParmType->getTypeClass()) {
  case Type::TemplateTypeParm:
    if (TL) {
      setTypeTemplateArgument(
          TAIList, PARM_TYPE_CAST(TemplateTypeParmType)->getIndex(), TL);
    } else {
      ArgType.removeLocalFastQualifiers(ParmType.getCVRQualifiers());
      setTypeTemplateArgument(
          TAIList, PARM_TYPE_CAST(TemplateTypeParmType)->getIndex(), ArgType);
    }
    break;
  case Type::TemplateSpecialization:
    deduceTemplateArgumentFromTemplateSpecialization(TAIList, ParmType, ArgType,
                                                     TL);
    break;
  case Type::Pointer:
    if (auto ArgPointer = ARG_TYPE_CAST(PointerType)) {
      deduceTemplateArgumentFromType(TAIList, ParmType->getPointeeType(),
                                     ArgPointer->getPointeeType(),
                                     getPointeeTypeLoc(TL));
    } else if (auto ArgArray = ARG_TYPE_CAST(ArrayType)) {
      deduceTemplateArgumentFromType(TAIList, ParmType->getPointeeType(),
                                     ArgArray->getElementType(),
                                     getPointeeTypeLoc(TL));
    } else if (auto DecayedArg = ARG_TYPE_CAST(DecayedType)) {
      deduceTemplateArgumentFromType(TAIList, ParmType,
                                     DecayedArg->getDecayedType(), TL);
    }
    break;
  case Type::LValueReference: {
    auto ParmPointeeType =
        PARM_TYPE_CAST(LValueReferenceType)->getPointeeTypeAsWritten();
    if (auto LVRT = ARG_TYPE_CAST(LValueReferenceType)) {
      deduceTemplateArgumentFromType(
          TAIList, ParmPointeeType, LVRT->getPointeeTypeAsWritten(),
          TL ? TYPELOC_CAST(LValueReferenceTypeLoc).getPointeeLoc() : TL);
    } else if (ParmPointeeType.getQualifiers().hasConst()) {
      deduceTemplateArgumentFromType(TAIList, ParmPointeeType, ArgType, TL);
    }
    break;
  }
  case Type::RValueReference:
    if (auto RVRT = ARG_TYPE_CAST(RValueReferenceType)) {
      deduceTemplateArgumentFromType(
          TAIList,
          PARM_TYPE_CAST(RValueReferenceType)->getPointeeTypeAsWritten(),
          RVRT->getPointeeTypeAsWritten(),
          TL ? TYPELOC_CAST(RValueReferenceTypeLoc).getPointeeLoc() : TL);
    }
    break;
  case Type::ConstantArray: {
    auto ArgConstArray = ARG_TYPE_CAST(ConstantArrayType);
    auto ParmConstArray = PARM_TYPE_CAST(ConstantArrayType);
    if (ArgConstArray && ParmConstArray &&
        ArgConstArray->getSize() == ParmConstArray->getSize()) {
      deduceTemplateArgumentFromArrayElement(TAIList, ParmType, ArgType, TL);
    }
    break;
  }
  case Type::IncompleteArray:
    deduceTemplateArgumentFromArrayElement(TAIList, ParmType, ArgType, TL);
    break;
  case Type::DependentSizedArray: {
    deduceTemplateArgumentFromArrayElement(TAIList, ParmType, ArgType, TL);
    auto ParmSizeExpr = PARM_TYPE_CAST(DependentSizedArrayType)->getSizeExpr();
    if (TL && TL.getTypePtr()->isArrayType()) {
      deduceNonTypeTemplateArgument(TAIList, ParmSizeExpr,
                                    TYPELOC_CAST(ArrayTypeLoc).getSizeExpr());
    } else if (auto DSAT = ARG_TYPE_CAST(DependentSizedArrayType)) {
      deduceNonTypeTemplateArgument(TAIList, ParmSizeExpr, DSAT->getSizeExpr());
    } else if (auto CAT = ARG_TYPE_CAST(ConstantArrayType)) {
      deduceNonTypeTemplateArgument(TAIList, ParmSizeExpr, CAT->getSize());
    }
    break;
  }
  default:
    break;
  }

  if (getInnerType(ArgType, TL)) {
    deduceTemplateArgumentFromType(TAIList, ParmType, ArgType, TL);
  }
}

void deduceTemplateArgument(std::vector<TemplateArgumentInfo> &TAIList,
                            const Expr *Arg, const ParmVarDecl *PVD) {
  auto ArgType = Arg->getType();
  auto ParmType = PVD->getType();

  TypeLoc TL;
  if (auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImplicitAsWritten())) {
    if (auto DD = dyn_cast<DeclaratorDecl>(DRE->getDecl()))
      TL = DD->getTypeSourceInfo()->getTypeLoc();
  } else if (const auto *CMCE =
                 dyn_cast<CXXMemberCallExpr>(Arg->IgnoreImplicitAsWritten())) {
    if (const auto *MD = CMCE->getMethodDecl()) {
      QualType ReturnType = MD->getReturnType();
      if (const auto *PtrType = ReturnType->getAs<PointerType>()) {
        QualType PointeeType = PtrType->getPointeeType();
        if (const auto *SubstType =
                PointeeType->getAs<SubstTemplateTypeParmType>()) {
          const auto Index = SubstType->getIndex();
          if (const auto *Callee = dyn_cast<MemberExpr>(CMCE->getCallee())) {
            if (Index < Callee->getNumTemplateArgs())
              ArgType = DpctGlobalInfo::getContext().getPointerType(
                  Callee->getTemplateArgs()[Index]
                      .getTypeSourceInfo()
                      ->getType());
          }
        }
      }
    }
  }
  deduceTemplateArgumentFromType(TAIList, ParmType, ArgType, TL);
}

std::shared_ptr<DeviceFunctionInfo> CallFunctionExpr::getFuncInfo() {
  if (FuncInfo.empty()) {
    return std::shared_ptr<DeviceFunctionInfo>();
  }
  return FuncInfo.front();
}

void CallFunctionExpr::buildInfo() {
  for (auto &Info : FuncInfo) {
    const clang::tooling::UnifiedPath &DefFilePath =
        Info->getDefinitionFilePath();
    // SYCL_EXTERNAL macro is not needed if the device function is lambda
    // expression, becuase 'sycl_device' attribute cannot be applied or will be
    // ignored.
    //
    // e.g.,
    // [] (T a, T b ) -> SYCL_EXTERNAL T { return a * b; }
    // [] (T a, T b ) SYCL_EXTERNAL { return a * b; }
    //
    // Intel(R) oneAPI DPC++ Compiler emits warning of ignoring SYCL_EXTERNAL in
    // the first example and emits error when compiling the second example.
    //
    // TODO: Need to revisit the condition to add SYCL_EXTERNAL macro if issues
    // are observed in the future.
    if (!DefFilePath.getCanonicalPath().empty() &&
        DefFilePath != getFilePath() &&
        !isIncludedFile(getFilePath(), DefFilePath) && !Info->isLambda()) {
      Info->setNeedSyclExternMacro();
    }

    if (DpctGlobalInfo::isOptimizeMigration() && !Info->isInlined() &&
        !Info->IsSyclExternMacroNeeded()) {
      if (Info->isKernel())
        Info->setForceInlineDevFunc();
      else
        Info->setAlwaysInlineDevFunc();
    }

    Info->buildInfo();
  }
  size_t FuncInfoSize = FuncInfo.size();
  if (FuncInfoSize) {
    VarMap.merge(FuncInfo.front()->getVarMap(), TemplateArgs);
    mergeTextureObjectInfo(FuncInfo.front());
  }
  for (size_t i = 0; i < FuncInfoSize; i++) {
    for (size_t j = i + 1; j < FuncInfoSize; j++) {
      if (!FuncInfo[i]->getVarMap().isSameAs(FuncInfo[j]->getVarMap())) {
        DiagnosticsUtils::report(getFilePath(), getOffset(),
                                 Warnings::DEVICE_CALL_DIFFERENT, true, false,
                                 FuncInfo[i]->getFunctionName());
        return;
      }
    }
  }
}

bool isInSameLine(SourceLocation First, SourceLocation Second,
                  const SourceManager &SM) {
  bool Invalid = false;
  return isInSameLine(SM.getExpansionLoc(First), SM.getExpansionLoc(Second),
                        SM, Invalid) &&
         !Invalid;
}

unsigned calculateCudaAttrLength(const AttributeCommonInfo &A,
                                 SourceLocation AlignLocation,
                                 const SourceManager &SM,
                                 const LangOptions &LO) {
  std::string Expected;
  switch (A.getParsedKind()) {
  case AttributeCommonInfo::AT_CUDAGlobal:
    Expected = "__global__";
    break;
  case AttributeCommonInfo::AT_CUDADevice:
    Expected = "__device__";
    break;
  case AttributeCommonInfo::AT_CUDAHost:
    Expected = "__host__";
    break;
  default:
    return 0;
  }

  auto Begin = SM.getExpansionLoc(A.getRange().getBegin());
  if (!isInSameLine(Begin, AlignLocation, SM))
    return 0;
  auto Length = Lexer::MeasureTokenLength(Begin, SM, LO);
  if (Expected.compare(0, std::string::npos, SM.getCharacterData(Begin),
                       Length))
    return 0;
  return getLenIncludingTrailingSpaces(
      SourceRange(Begin, Begin.getLocWithOffset(Length)), SM);
}

template <class IteratorT>
unsigned calculateCudaAttrLength(IteratorT AttrBegin, IteratorT AttrEnd,
                                 SourceLocation AlignLoc,
                                 const SourceManager &SM,
                                 const LangOptions &LO) {
  unsigned Length = 0;

  if (SM.isMacroArgExpansion(AlignLoc))
    return 0;
  AlignLoc = SM.getExpansionLoc(AlignLoc);

  std::for_each(AttrBegin, AttrEnd, [&](const AttributeCommonInfo &A) {
    Length += calculateCudaAttrLength(A, AlignLoc, SM, LO);
  });

  return Length;
}

unsigned calculateCudaAttrLength(const ParsedAttributes &Attrs,
                                 SourceLocation AlignLoc,
                                 const SourceManager &SM,
                                 const LangOptions &LO) {
  return calculateCudaAttrLength(Attrs.begin(), Attrs.end(), AlignLoc, SM, LO);
}

unsigned calculateCudaAttrLength(const AttrVec &Attrs, SourceLocation AlignLoc,
                                 const SourceManager &SM,
                                 const LangOptions &LO) {
  struct AttrIterator
      : llvm::iterator_adaptor_base<AttrIterator, AttrVec::const_iterator,
                                    std::random_access_iterator_tag, Attr> {
    AttrIterator(AttrVec::const_iterator I) : iterator_adaptor_base(I) {}

    reference operator*() const { return **I; }
    friend class ParsedAttributesView;
  };
  return calculateCudaAttrLength(AttrIterator(Attrs.begin()),
                                 AttrIterator(Attrs.end()), AlignLoc, SM, LO);
}

bool isEachParamEachLine(const ArrayRef<ParmVarDecl *> Parms,
                         SourceManager &SM) {
  if (Parms.size() < 2)
    return false;
  auto Itr = Parms.begin();
  auto NextItr = Itr;
  while (++NextItr != Parms.end()) {
    if (isInSameLine((*Itr)->getBeginLoc(), (*NextItr)->getBeginLoc(), SM))
      return false;
    Itr = NextItr;
  }
  return true;
}

// PARAMETER INSERT LOCATION RULES:
// 1. Origin parameters number <= 1
//    Do not add new line until longer than 80. The new line begin is aligned
//    with the end location of "("
// 2. Origin parameters number > 1
//    2.1 If each parameter is in a single line:
//           Each added parameter is in a single line.
//           The new line begin is aligned with the last parameter's line
//           begin
//    2.2 There are 2 parameters in one line:
//           Do not add new line until longer than 80.
//           The new line begin is aligned with the last parameter's line
//           begin
template <class AttrsT>
FormatInfo buildFormatInfo(const FunctionTypeLoc &FTL,
                           SourceLocation InsertLocation, const AttrsT &Attrs,
                           SourceManager &SM, const LangOptions &LO) {
  SourceLocation AlignLocation;
  FormatInfo Format;
  Format.EnableFormat = true;

  bool CurrentSameLineWithAlign = false;
  Format.IsAllParamsOneLine = false;
  Format.CurrentLength = SM.getExpansionColumnNumber(InsertLocation);

  if (FTL.getNumParams()) {
    Format.IsEachParamNL = isEachParamEachLine(FTL.getParams(), SM);
    auto FirstParmLoc = SM.getExpansionLoc(FTL.getParam(0)->getBeginLoc());
    if (CurrentSameLineWithAlign =
            isInSameLine(FirstParmLoc, InsertLocation, SM)) {
      AlignLocation = FirstParmLoc;
    } else {
      Format.NewLineIndentStr = getIndent(InsertLocation, SM).str();
      Format.NewLineIndentLength = Format.NewLineIndentStr.length();
      return Format;
    }
  } else {
    Format.IsEachParamNL = false;
    AlignLocation = SM.getExpansionLoc(FTL.getLParenLoc()).getLocWithOffset(1);
    CurrentSameLineWithAlign = isInSameLine(AlignLocation, InsertLocation, SM);
  }

  auto CudaAttrLength = calculateCudaAttrLength(Attrs, AlignLocation, SM, LO);
  Format.NewLineIndentLength =
      SM.getExpansionColumnNumber(AlignLocation) - CudaAttrLength - 1;
  Format.NewLineIndentStr.assign(Format.NewLineIndentLength, ' ');
  if (CurrentSameLineWithAlign)
    Format.CurrentLength -= CudaAttrLength;

  return Format;
}

template <class AttrsT>
void DeviceFunctionDecl::buildReplaceLocInfo(const FunctionTypeLoc &FTL,
                                             const AttrsT &Attrs) {
  if (!FTL)
    return;

  SourceLocation InsertLocation;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &LO = DpctGlobalInfo::getContext().getLangOpts();
  if (NonDefaultParamNum) {
    InsertLocation = FTL.getParam(NonDefaultParamNum - 1)->getEndLoc();
  } else {
    InsertLocation = FTL.getLParenLoc();
  }

  InsertLocation = getActualInsertLocation(InsertLocation, SM, LO);
  if (InsertLocation.isMacroID()) {
    InsertLocation = Lexer::getLocForEndOfToken(
        SM.getSpellingLoc(InsertLocation), 0, SM, LO);
    FormatInformation.EnableFormat = true;
    FormatInformation.IsAllParamsOneLine = true;
  } else {
    InsertLocation = Lexer::getLocForEndOfToken(InsertLocation, 0, SM, LO);
    FormatInformation = buildFormatInfo(FTL, InsertLocation, Attrs, SM, LO);
  }
  FormatInformation.IsFirstArg = (NonDefaultParamNum == 0);

  // Skipping #ifdef #endif pair
  Token TokOfHash;
  if (!Lexer::getRawToken(InsertLocation, TokOfHash, SM, LO, true)) {
    auto ItIf = DpctGlobalInfo::getEndifLocationOfIfdef().find(
        getHashStrFromLoc(TokOfHash.getEndLoc()));
    while (ItIf != DpctGlobalInfo::getEndifLocationOfIfdef().end()) {
      InsertLocation = Lexer::getLocForEndOfToken(ItIf->second, 0, SM, LO);
      InsertLocation = Lexer::GetBeginningOfToken(
          Lexer::findNextToken(InsertLocation, SM, LO)->getLocation(), SM, LO);
      if (Lexer::getRawToken(InsertLocation, TokOfHash, SM, LO, true))
        break;
      ItIf = DpctGlobalInfo::getEndifLocationOfIfdef().find(
          getHashStrFromLoc(TokOfHash.getEndLoc()));
    }
  }

  // Skip whitespace, e.g. void foo(        void) {}
  //                                        |
  //                                      need get here
  if (!Lexer::getRawToken(InsertLocation, TokOfHash, SM, LO, true)) {
    InsertLocation = TokOfHash.getLocation();
  }

  Token PPTok;
  if (!Lexer::getRawToken(InsertLocation, PPTok, SM, LO, true) &&
      PPTok.is(tok::hash)) {
    IsReplaceFollowedByPP = true;
  }

  ReplaceOffset = SM.getFileOffset(InsertLocation);
  if (FTL.getNumParams() == 0) {
    Token Tok;
    if (!Lexer::getRawToken(InsertLocation, Tok, SM, LO, true) &&
        Tok.is(tok::raw_identifier) && Tok.getRawIdentifier() == "void") {
      ReplaceLength = Tok.getLength();
    }
  }
}

bool isInAnalysisScope(SourceLocation SL) {
  return DpctGlobalInfo::isInAnalysisScope(SL);
}

std::vector<std::shared_ptr<FreeQueriesInfo>> FreeQueriesInfo::InfoList;
std::vector<std::shared_ptr<FreeQueriesInfo::MacroInfo>>
    FreeQueriesInfo::MacroInfos;

const FreeQueriesInfo::FreeQueriesNames &
FreeQueriesInfo::getNames(FreeQueriesKind K) {
  static FreeQueriesNames Names[FreeQueriesInfo::FreeQueriesKind::End] = {
      {getItemName(),
       MapNames::getClNamespace() + "ext::oneapi::this_work_item::get_nd_item",
       getItemName()},
      {getItemName() + ".get_group()",
       MapNames::getClNamespace() + "ext::oneapi::this_work_item::get_work_group",
       "group" + getCTFixedSuffix()},
      {getItemName() + ".get_sub_group()",
       MapNames::getClNamespace() + "ext::oneapi::this_work_item::get_sub_group",
       "sub_group" + getCTFixedSuffix()},
  };
  return Names[K];
}

std::shared_ptr<FreeQueriesInfo>
FreeQueriesInfo::getInfo(const FunctionDecl *FD) {
  if (!FD)
    return std::shared_ptr<FreeQueriesInfo>();

  if (!FD->doesThisDeclarationHaveABody())
    return std::shared_ptr<FreeQueriesInfo>();

  if (auto CS = dyn_cast_or_null<CompoundStmt>(FD->getBody())) {
    if (CS->body_empty())
      return std::shared_ptr<FreeQueriesInfo>();

    auto ExtraDeclLoc = CS->body_front()->getBeginLoc();
    auto LocInfo = DpctGlobalInfo::getLocInfo(ExtraDeclLoc);
    auto Iter = std::find_if(InfoList.begin(), InfoList.end(),
                             [&](const std::shared_ptr<FreeQueriesInfo> &Info) {
                               return Info->FilePath == LocInfo.first &&
                                      Info->ExtraDeclLoc == LocInfo.second;
                             });
    if (Iter != InfoList.end())
      return *Iter;

    auto Info = std::make_shared<FreeQueriesInfo>();
    Info->FilePath = std::move(LocInfo.first);
    Info->ExtraDeclLoc = LocInfo.second;
    Info->Idx = InfoList.size();
    Info->FuncInfo = DeviceFunctionDecl::LinkRedecls(FD);
    Info->Indent =
        getIndent(ExtraDeclLoc, DpctGlobalInfo::getSourceManager()).str();
    Info->NL = getNL();
    InfoList.push_back(Info);
    return Info;
  }

  return std::shared_ptr<FreeQueriesInfo>();
}

template <class Node>
void FreeQueriesInfo::printImmediateText(llvm::raw_ostream &OS, const Node *S,
                                         const FunctionDecl *FD,
                                         FreeQueriesKind K) {
#ifdef DPCT_DEBUG_BUILD
  assert(K != FreeQueriesKind::End && "Unexpected FreeQueriesKind::End");
#endif // DPCT_DEBUG_BUILD

  if (!FD) {
    FD = DpctGlobalInfo::getParentFunction(S);
  }

  if (DpctGlobalInfo::useFreeQueries()) {
    if (auto Info = getInfo(FD)) {
      return Info->printImmediateText(OS, S->getBeginLoc(), K);
    }

#ifdef DPCT_DEBUG_BUILD
    llvm::errs() << "Can not get FreeQueriesInfo for this FunctionDecl\n";
    assert(0);
#endif // DPCT_DEBUG_BUILD

  } else {
    if (auto DFI = DeviceFunctionDecl::LinkRedecls(FD))
      DFI->setItem();
    OS << getNames(K).NonFreeQueriesName;
  }
}

/// Generate regex replacement as placeholder.
void FreeQueriesInfo::printImmediateText(llvm::raw_ostream &OS,
                                         SourceLocation SL, FreeQueriesKind K) {
  unsigned Index = Idx;
  auto IsMacro = SL.isMacroID();
  if (IsMacro && K != SubGroup) {
    auto MacroLoc = DpctGlobalInfo::getLocInfo(
        DpctGlobalInfo::getSourceManager().getSpellingLoc(SL));
    auto Iter = std::find_if(MacroInfos.begin(), MacroInfos.end(),
                             [&](std::shared_ptr<MacroInfo> Info) -> bool {
                               return (MacroLoc.first == Info->FilePath) &&
                                      (MacroLoc.second == Info->Offset);
                             });
    if (Iter == MacroInfos.end()) {
      MacroInfos.push_back(std::make_shared<MacroInfo>());
      Iter = --MacroInfos.end();
      (*Iter)->FilePath = MacroLoc.first;
      (*Iter)->Offset = MacroLoc.second;
    }
    (*Iter)->Infos.push_back(Idx);
    Index = Iter - MacroInfos.begin();
  } else {
    auto SLocInfo = DpctGlobalInfo::getLocInfo(SL);
    if (SLocInfo.first != FilePath)
      return;

    if (Refs.insert(SLocInfo.second).second) {
      ++Counter[K];
    }
  }

  OS << RegexPrefix << FreeQueriesRegexCh << getRegexNum(Index, IsMacro, K)
     << RegexSuffix;
  return;
}

/// Generate temporary variable declaration when reference counter > 2.
/// Declaration example:
/// auto item_ct1 = this_nd_item<3>();
void FreeQueriesInfo::emplaceExtraDecl() {
  std::string Ret;
  llvm::raw_string_ostream OS(Ret);
  if (DpctGlobalInfo::getAssumedNDRangeDim() == 1 && FuncInfo) {
    if (auto VarMapHead =
            MemVarMap::getHeadWithoutPathCompression(&FuncInfo->getVarMap())) {
      Dimension = VarMapHead->Dim;
    }
  }
  if (Counter[FreeQueriesKind::NdItem] > 1) {
    auto &KindNames =
        getNames(static_cast<FreeQueriesKind>(FreeQueriesKind::NdItem));
    OS << "auto " << KindNames.ExtraVariableName << " = ";
    printFreeQueriesFunctionName(
        OS, static_cast<FreeQueriesKind>(FreeQueriesKind::NdItem), Dimension);
    OS << ';' << NL << Indent;
  }
  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, ExtraDeclLoc, 0, OS.str(), nullptr));
}

std::string FreeQueriesInfo::getReplaceString(unsigned Num) {
  auto Index = getIndex(Num);
  auto Kind = getKind(Num);
  bool IsMacro = isMacro(Num);
  if (IsMacro) {
    if (Index < MacroInfos.size()) {
      return buildStringFromPrinter(printFreeQueriesFunctionName, Kind,
                                    MacroInfos[Index]->Dimension);
    }
#ifdef DPCT_DEBUG_BUILD
    llvm::errs() << "FreeQueriesInfo index[" << Index
                 << "]is larger than list size[" << InfoList.size() << "]\n";
    assert(0);
#endif // DPCT_DEBUG_BUILD
  }
  if (Index < InfoList.size())
    return InfoList[Index]->getReplaceString(getKind(Num));
#ifdef DPCT_DEBUG_BUILD
  llvm::errs() << "FreeQueriesInfo index[" << Index
               << "]is larger than list size[" << InfoList.size() << "]\n";
  assert(0);
#endif // DPCT_DEBUG_BUILD
  return "";
}

std::string FreeQueriesInfo::getReplaceString(FreeQueriesKind K) {
  if (K != FreeQueriesKind::NdItem || Counter[K] < 2)
    return buildStringFromPrinter(printFreeQueriesFunctionName, K, Dimension);
  else
    return getNames(K).ExtraVariableName;
}

} // namespace dpct
} // namespace clang
