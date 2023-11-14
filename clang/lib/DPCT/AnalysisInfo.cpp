//===--------------- AnalysisInfo.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "Diagnostics.h"
#include "ExprAnalysis.h"
#include "Statics.h"
#include "Utility.h"

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include <algorithm>
#include <deque>
#include <fstream>
#include <optional>

#define TYPELOC_CAST(Target) static_cast<const Target &>(TL)

llvm::StringRef getReplacedName(const clang::NamedDecl *D) {
  auto Iter = MapNames::TypeNamesMap.find(D->getQualifiedNameAsString(false));
  if (Iter != MapNames::TypeNamesMap.end()) {
    auto Range = getDefinitionRange(D->getBeginLoc(), D->getEndLoc());
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
int HostDeviceFuncInfo::MaxId = 0;
std::string DpctGlobalInfo::InRoot = std::string();
std::string DpctGlobalInfo::OutRoot = std::string();
std::string DpctGlobalInfo::AnalysisScope = std::string();
std::unordered_set<std::string> DpctGlobalInfo::ChangeExtensions = {};
// TODO: implement one of this for each source language.
std::string DpctGlobalInfo::CudaPath = std::string();
std::string DpctGlobalInfo::RuleFile = std::string();
UsmLevel DpctGlobalInfo::UsmLvl = UsmLevel::UL_None;
clang::CudaVersion DpctGlobalInfo::SDKVersion = clang::CudaVersion::UNKNOWN;
bool DpctGlobalInfo::NeedDpctDeviceExt = false;
bool DpctGlobalInfo::IsIncMigration = true;
bool DpctGlobalInfo::IsQueryAPIMapping = false;
unsigned int DpctGlobalInfo::AssumedNDRangeDim = 3;
std::unordered_set<std::string> DpctGlobalInfo::PrecAndDomPairSet;
format::FormatRange DpctGlobalInfo::FmtRng = format::FormatRange::none;
DPCTFormatStyle DpctGlobalInfo::FmtST = DPCTFormatStyle::FS_LLVM;
std::set<ExplicitNamespace> DpctGlobalInfo::ExplicitNamespaceSet;
bool DpctGlobalInfo::EnableCtad = false;
bool DpctGlobalInfo::GenBuildScript = false;
bool DpctGlobalInfo::EnableComments = false;
bool DpctGlobalInfo::TempEnableDPCTNamespace = false;
bool DpctGlobalInfo::IsMLKHeaderUsed = false;
ASTContext *DpctGlobalInfo::Context = nullptr;
SourceManager *DpctGlobalInfo::SM = nullptr;
FileManager *DpctGlobalInfo::FM = nullptr;
bool DpctGlobalInfo::KeepOriginCode = false;
bool DpctGlobalInfo::SyclNamedLambda = false;
bool DpctGlobalInfo::CheckUnicodeSecurityFlag = false;
std::unordered_map<std::string, SourceRange> DpctGlobalInfo::ExpansionRangeBeginMap;
bool DpctGlobalInfo::EnablepProfilingFlag = false;
std::map<std::string, std::shared_ptr<DpctGlobalInfo::MacroExpansionRecord>>
    DpctGlobalInfo::ExpansionRangeToMacroRecord;
std::tuple<unsigned int, std::string, SourceRange>
    DpctGlobalInfo::LastMacroRecord =
        std::make_tuple<unsigned int, std::string, SourceRange>(0, "",
                                                                SourceRange());
std::map<std::string, SourceLocation> DpctGlobalInfo::EndifLocationOfIfdef;
std::vector<std::pair<std::string, size_t>>
    DpctGlobalInfo::ConditionalCompilationLoc;
std::map<std::string, std::shared_ptr<DpctGlobalInfo::MacroDefRecord>>
    DpctGlobalInfo::MacroTokenToMacroDefineLoc;
std::map<std::string, std::string>
    DpctGlobalInfo::FunctionCallInMacroMigrateRecord;
std::map<std::string, SourceLocation> DpctGlobalInfo::EndOfEmptyMacros;
std::map<std::string, unsigned int> DpctGlobalInfo::BeginOfEmptyMacros;
std::map<std::string, bool> DpctGlobalInfo::MacroDefines;
std::set<std::string> DpctGlobalInfo::IncludingFileSet;
std::set<std::string> DpctGlobalInfo::FileSetInCompiationDB;
std::unordered_map<std::string, std::vector<clang::tooling::Replacement>>
    DpctGlobalInfo::FileRelpsMap;
std::unordered_map<std::string, std::string> DpctGlobalInfo::DigestMap;
const std::string DpctGlobalInfo::YamlFileName = "MainSourceFiles.yaml";
std::set<std::string> DpctGlobalInfo::GlobalVarNameSet;
const std::string MemVarInfo::ExternVariableName = "dpct_local";
std::unordered_map<const DeclStmt *, int> MemVarInfo::AnonymousTypeDeclStmtMap;
const int TextureObjectInfo::ReplaceTypeLength = strlen("cudaTextureObject_t");
bool DpctGlobalInfo::GuessIndentWidthMatcherFlag = false;
unsigned int DpctGlobalInfo::IndentWidth = 0;
std::map<unsigned int, unsigned int> DpctGlobalInfo::KCIndentWidthMap;
std::unordered_map<std::string, int> DpctGlobalInfo::LocationInitIndexMap;
int DpctGlobalInfo::CurrentMaxIndex = 0;
int DpctGlobalInfo::CurrentIndexInRule = 0;
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
unsigned int DpctGlobalInfo::RunRound = 0;
bool DpctGlobalInfo::NeedRunAgain = false;
std::set<std::string> DpctGlobalInfo::ModuleFiles;
bool DpctGlobalInfo::OptimizeMigrationFlag = false;

std::unordered_map<std::string, std::shared_ptr<DeviceFunctionInfo>>
    DeviceFunctionDecl::FuncInfoMap;
CudaArchPPMap DpctGlobalInfo::CAPPInfoMap;
HDFuncInfoMap DpctGlobalInfo::HostDeviceFuncInfoMap;
// __CUDA_ARCH__ Offset -> defined(...) Offset
CudaArchDefMap DpctGlobalInfo::CudaArchDefinedMap;
std::unordered_map<std::string, std::shared_ptr<ExtReplacement>>
    DpctGlobalInfo::CudaArchMacroRepl;
std::unordered_map<std::string, std::shared_ptr<ExtReplacements>>
    DpctGlobalInfo::FileReplCache;
std::set<std::string> DpctGlobalInfo::ReProcessFile;
std::set<std::string> DpctGlobalInfo::ProcessedFile;
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
unsigned int DpctGlobalInfo::ColorOption = 1;
std::unordered_map<int, std::shared_ptr<DeviceFunctionInfo>>
    DpctGlobalInfo::CubPlaceholderIndexMap;
std::unordered_map<std::string, std::shared_ptr<PriorityReplInfo>>
    DpctGlobalInfo::PriorityReplInfoMap;
std::unordered_map<std::string, bool> DpctGlobalInfo::ExcludePath = {};
std::map<std::string, clang::tooling::OptionInfo> DpctGlobalInfo::CurrentOptMap;
std::unordered_map<std::string,
                   std::unordered_map<std::string, std::vector<unsigned>>>
    DpctGlobalInfo::RnnInputMap;
std::unordered_map<std::string, std::vector<std::string>>
    DpctGlobalInfo::MainSourceFileMap;
std::unordered_map<std::string, bool>
    DpctGlobalInfo::MallocHostInfoMap;
/// This variable saved the info of previous migration from the
/// MainSourceFiles.yaml file. This variable is valid after
/// canContinueMigration() is called.
std::shared_ptr<clang::tooling::TranslationUnitReplacements>
    DpctGlobalInfo::MainSourceYamlTUR =
        std::make_shared<clang::tooling::TranslationUnitReplacements>();

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
    std::string FilePath;
    unsigned Offset;
    unsigned Dimension = 0;
    std::vector<unsigned> Infos;
  };
  static std::vector<std::shared_ptr<FreeQueriesInfo>> InfoList;
  static std::vector<std::shared_ptr<MacroInfo>> MacroInfos;

  std::string FilePath;
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
  DigestMap.clear();
  MacroDefines.clear();
  CurrentMaxIndex = 0;
  CurrentIndexInRule = 0;
  IncludingFileSet.clear();
  FileSetInCompiationDB.clear();
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
}

DpctGlobalInfo::DpctGlobalInfo() {
  IsInAnalysisScopeFunc = DpctGlobalInfo::checkInAnalysisScope;
  GetRunRound = DpctGlobalInfo::getRunRound;
  RecordTokenSplit = DpctGlobalInfo::recordTokenSplit;
  tooling::SetGetRunRound(DpctGlobalInfo::getRunRound);
  tooling::SetReProcessFile(DpctGlobalInfo::ReProcessFile);
  tooling::SetIsExcludePathHandler(DpctGlobalInfo::isExcluded);
}

std::shared_ptr<KernelCallExpr>
DpctGlobalInfo::buildLaunchKernelInfo(const CallExpr *LaunchKernelCall) {
  auto LocInfo = getLocInfo(LaunchKernelCall->getBeginLoc());
  auto FileInfo = insertFile(LocInfo.first);
  if (FileInfo->findNode<KernelCallExpr>(LocInfo.second))
    return std::shared_ptr<KernelCallExpr>();

  auto KernelInfo =
      KernelCallExpr::buildFromCudaLaunchKernel(LocInfo, LaunchKernelCall);
  if (KernelInfo) {
    FileInfo->insertNode(LocInfo.second, KernelInfo);
  }

  return KernelInfo;
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
    QDecl << "&q_ct1 = dev_ct1." << DpctGlobalInfo::getDeviceQueueName()
          << "();";
  } else {
    DevDecl << MapNames::getClNamespace() + "device dev_ct1;";
    // Now the UsmLevel must not be UL_None here.
    QDecl << "q_ct1(dev_ct1, " << MapNames::getClNamespace() << "property_list{"
          << MapNames::getClNamespace() << "property::queue::in_order()";
    if (DpctGlobalInfo::getEnablepProfilingFlag()) {
      QDecl << ", " << MapNames::getClNamespace()
            << "property::queue::enable_profiling()";
    }
    QDecl << "});";
  }

  for (auto &Counter : TempVariableDeclCounterMap) {
    if (DpctGlobalInfo::useNoQueueDevice()) {
      Counter.second.PlaceholderStr[1] = DpctGlobalInfo::getGlobalQueueName();
      Counter.second.PlaceholderStr[2] = DpctGlobalInfo::getGlobalDeviceName();
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

void DpctGlobalInfo::processCudaArchMacro(){
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
        DpctGlobalInfo::getInstance()
            .getCudaArchDefinedMap()[FilePath];
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
      (*Repl).setReplacementText("!DPCT_COMPATIBILITY_TEMP");
    }
  };

  for (auto Iter = ReplMap.begin(); Iter != ReplMap.end();) {
    auto Repl = Iter->second;
    unsigned CudaArchOffset = Repl->getOffset();
    std::string FilePath = Repl->getFilePath().str();
    auto &CudaArchPPInfosMap =
        DpctGlobalInfo::getInstance()
            .getCudaArchPPInfoMap()[FilePath];
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

void DpctGlobalInfo::postProcess() {
  auto &MSMap = DpctGlobalInfo::getMainSourceFileMap();
  bool isFirstPass = !DpctGlobalInfo::getRunRound();
  processCudaArchMacro();
  for (auto &Element : HostDeviceFuncInfoMap) {
    auto &Info = Element.second;
    if (Info.isCalledInHost && Info.isDefInserted) {
      Info.needGenerateHostCode = true;
      if (Info.PostFixId == -1) {
        Info.PostFixId = HostDeviceFuncInfo::MaxId++;
      }
      for (auto &E : Info.LocInfos) {
        auto &LocInfo = E.second;
        if (isFirstPass) {
          auto &MSFiles = MSMap[LocInfo.FilePath];
          for (auto &File : MSFiles) {
            if (ProcessedFile.count(File))
              ReProcessFile.emplace(File);
          }
        }
        if (LocInfo.Type == HDFuncInfoType::HDFI_Call &&
          !LocInfo.Processed) {
          if(LocInfo.CalledByHostDeviceFunction && isFirstPass) {
            LocInfo.Processed = true;
            continue;
          }
          LocInfo.Processed = true;
          auto R = std::make_shared<ExtReplacement>(
              LocInfo.FilePath, LocInfo.FuncEndOffset, 0,
              "_host_ct" + std::to_string(Info.PostFixId), nullptr);
          addReplacement(R);
        }
      }
    }
  }
  if (!ReProcessFile.empty() && isFirstPass) {
    DpctGlobalInfo::setNeedRunAgain(true);
  }
  for (auto &File : FileMap) {
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
          auto &ReplLists =
              FileMap[LocInfo.FilePath]->getRepls()->getReplMap();
          generateHostCode(ReplLists, LocInfo, Info.PostFixId);
        }
      }
    }
  }
}

void DpctGlobalInfo::generateHostCode(
    std::multimap<unsigned int, std::shared_ptr<clang::dpct::ExtReplacement>>
        &ProcessedReplList,
    HostDeviceFuncLocInfo Info, unsigned ID) {
  std::vector<std::shared_ptr<ExtReplacement>> ExtraRepl;

  unsigned int Pos, Len;
  std::string OriginText = Info.FuncContentCache;
  StringRef SR(OriginText);
  RewriteBuffer RB;
  RB.Initialize(SR.begin(), SR.end());
  for (auto &Element : ProcessedReplList) {
    auto R = Element.second;
    unsigned ROffset = R->getOffset();
    if (ROffset >= Info.FuncStartOffset && ROffset <= Info.FuncEndOffset) {
      Pos = ROffset - Info.FuncStartOffset;
      Len = R->getLength();
      RB.ReplaceText(Pos, Len, R->getReplacementText());
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
    FileReplCache[R->getFilePath().str()]->addReplacement(R);
  }
  return;
}

bool DpctFileInfo::isInAnalysisScope() { return DpctGlobalInfo::isInAnalysisScope(FilePath); }

// TODO: implement one of this for each source language.
bool DpctFileInfo::isInCudaPath() {
  return DpctGlobalInfo::isInCudaPath(FilePath);
}

void DpctFileInfo::buildLinesInfo() {
  if (FilePath.empty())
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();

  llvm::Expected<FileEntryRef> Result =
      SM.getFileManager().getFileRef(FilePath);

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
  
  for (auto &D : FuncMap){
    if(auto I = D.second->getFuncInfo())
      I->buildInfo();
  }
}
void DpctFileInfo::postProcess() {
  if (!isInAnalysisScope())
    return;
  for (auto &D : FuncMap)
    D.second->emplaceReplacement();
  if (!Repls->empty()) {
    Repls->postProcess();
    if (DpctGlobalInfo::getRunRound() == 0) {
      DpctGlobalInfo::getInstance().cacheFileRepl(FilePath, Repls);
    }
  }
}

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
      DataRepl << "DPCT_CHECK_ERROR(";
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

void DpctFileInfo::buildRnnBackwardFuncInfo() {
  RnnBackwardFuncInfoBuilder Builder(RBFuncInfo);
  Builder.build();
  for(auto &Repl : Builder.getReplacement()) {
    addReplacement(Repl);
  }
}

void DpctFileInfo::buildReplacements() {
  if (!isInAnalysisScope())
    return;

  if (FilePath.empty())
    return;
  // Traverse all the global variables stored one by one to check if its name
  // is same with normal global variable's name in host side, if the one is
  // found, postfix "_ct" is added to this __constant__ symbol's name.
  std::unordered_map<unsigned int, std::string> ReplUpdated;
  for (const auto &Entry : MemVarMap) {
    if (Entry.second->isIgnore())
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
                               Diagnostics::API_NOT_OCCURRED_IN_AST, true,
                               true, std::get<1>(AtomicInfo.second));
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

void DpctFileInfo::emplaceReplacements(ReplTy &ReplSet) {
  if (!Repls->empty())
    Repls->emplaceIntoReplSet(ReplSet[FilePath]);
}

std::vector<std::pair<HeaderType, std::string>> HeaderSpellings;

void initHeaderSpellings() {
  HeaderSpellings = {
#define HEADER(Name, Spelling) {HT_##Name, Spelling},
#include "HeaderTypes.inc"
  };
}

StringRef DpctFileInfo::getHeaderSpelling(HeaderType Value) {
  if (Value < NUM_HEADERS)
    return HeaderSpellings[Value].second;

  // Only assertion in debug
  assert(false && "unknown HeaderType");
  return "";
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

void DpctFileInfo::insertHeader(HeaderType Type, unsigned Offset) {
  if (Type == HT_DPL_Algorithm || Type == HT_DPL_Execution ||
      Type == HT_DPCT_DNNL_Utils) {
    if (this != DpctGlobalInfo::getInstance().getMainFile().get())
      DpctGlobalInfo::getInstance().getMainFile()->insertHeader(
          Type, FirstIncludeOffset);
  }
  if (HeaderInsertedBitMap[Type])
    return;
  HeaderInsertedBitMap[Type] = true;
  std::string ReplStr;
  llvm::raw_string_ostream OS(ReplStr);

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
  case HT_DPCT_DNNL_Utils:
    concatHeader(OS, getHeaderSpelling(Type));
    return insertHeader(OS.str(), FirstIncludeOffset,
                        InsertPosition::IP_AlwaysLeft);
  case HT_SYCL:
    if(DpctGlobalInfo::getEnablepProfilingFlag())
      OS << "#define DPCT_PROFILING_ENABLED" << getNL();
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
      OS << "#define DPCT_USM_LEVEL_NONE" << getNL();
    if (!RTVersionValue.empty())
      OS << "#define DPCT_COMPAT_RT_VERSION " << RTVersionValue << getNL();
    concatHeader(OS, getHeaderSpelling(Type));
    concatHeader(OS, getHeaderSpelling(HT_DPCT_Dpct));
    HeaderInsertedBitMap[HT_DPCT_Dpct] = true;
    if (!DpctGlobalInfo::getExplicitNamespaceSet().count(
            ExplicitNamespace::EN_DPCT) ||
        DpctGlobalInfo::isDPCTNamespaceTempEnabled()) {
      OS << "using namespace dpct;" << getNL();
    }
    if (!DpctGlobalInfo::getExplicitNamespaceSet().count(
            ExplicitNamespace::EN_SYCL) &&
        !DpctGlobalInfo::getExplicitNamespaceSet().count(
            ExplicitNamespace::EN_CL)) {
      OS << "using namespace sycl;" << getNL();
    }
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
        if (DpctGlobalInfo::getEnablepProfilingFlag()) {
          OS << ", " << MapNames::getClNamespace()
             << "property::queue::enable_profiling()";
        }
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
    return insertHeader(OS.str(), FirstIncludeOffset, InsertPosition::IP_Left);

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
  case HT_MKL_RNG:
    insertHeader(HT_MKL_Mkl);
    break;
  default:
    break;
  }

  if (Offset != FirstIncludeOffset)
    OS << getNL();
  concatHeader(OS, getHeaderSpelling(Type));
  return insertHeader(OS.str(), LastIncludeOffset, InsertPosition::IP_Right);
}

void DpctFileInfo::insertHeader(HeaderType Type) {
  switch (Type) {
#define HEADER(Name, Spelling)                                                           \
  case HT_##Name:                                                                 \
    return insertHeader(HT_##Name, LastIncludeOffset);
#include "HeaderTypes.inc"
  default:
    return;
  }
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

std::optional<std::string>
DpctGlobalInfo::getAbsolutePath(const FileEntry &File) {
  if (auto RealPath = File.tryGetRealPathName(); !RealPath.empty())
    return RealPath.str();

  llvm::SmallString<512> FilePathAbs(File.getName());
  SM->getFileManager().makeAbsolutePath(FilePathAbs);
  llvm::sys::path::native(FilePathAbs);
  // Need to remove dot to keep the file path
  // added by ASTMatcher and added by
  // AnalysisInfo::getLocInfo() consistent.
  llvm::sys::path::remove_dots(FilePathAbs, true);
  return (std::string)FilePathAbs;
}
std::optional<std::string> DpctGlobalInfo::getAbsolutePath(FileID ID) {
  assert(SM && "SourceManager must be initialized");
  if (const auto *FileEntry = SM->getFileEntryForID(ID))
    return getAbsolutePath(*FileEntry);
  return std::nullopt;
}

int KernelCallExpr::calculateOriginArgsSize() const {
  int Size = 0;
  for (auto &ArgInfo : ArgsInfo) {
    Size += ArgInfo.ArgSize;
  }
  return Size;
}

template <class ArgsRange>
void KernelCallExpr::buildExecutionConfig(
    const ArgsRange &ConfigArgs, const CallExpr *KernelCall) {
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
    } else if (Idx == 1) {
      ExecutionConfig.LocalDirectRef = A.isDirectRef();
      // Using another analysis because previous analysis may return directly
      // when in macro is true.
      // Here set the argument of KFA as false, so it will not return directly.
      KernelConfigAnalysis KFA(false);
      KFA.setCallSpelling(KCallSpellingRange.first, KCallSpellingRange.second);
      KFA.analyze(Arg, 1, true);
      if (KFA.isNeedEmitWGSizeWarning())
        DiagnosticsUtils::report(getFilePath(), getBegin(),
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
    } else if (Idx == 1) {
      BlockDim = AnalysisTry1D.Dim;
      ExecutionConfig.LocalSizeFor1D = AnalysisTry1D.getReplacedString();
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

void KernelCallExpr::buildKernelInfo(const CUDAKernelCallExpr *KernelCall) {
  buildLocationInfo(KernelCall);
  buildExecutionConfig(KernelCall->getConfig()->arguments(), KernelCall);
  buildNeedBracesInfo(KernelCall);
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

void KernelCallExpr::addDevCapCheckStmt() {
  llvm::SmallVector<std::string> AspectList;
  if (getVarMap().hasBF64()) {
    AspectList.push_back(MapNames::getClNamespace() + "aspect::fp64");
  }
  if (getVarMap().hasBF16()) {
    AspectList.push_back(MapNames::getClNamespace() + "aspect::fp16");
  }
  if (!AspectList.empty()) {
    requestFeature(HelperFeatureEnum::device_ext);
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    OS << MapNames::getDpctNamespace() << "has_capability_or_fail(";
    printStreamBase(OS);
    OS << "get_device(), {" << AspectList.front();
    for (size_t i = 1; i < AspectList.size(); ++i) {
      OS << ", " << AspectList[i];
    }
    OS << "});";
    OuterStmts.emplace_back(OS.str());
  }
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
        DiagnosticsUtils::report(getFilePath(), getBegin(),
                                 Diagnostics::UNDEDUCED_TYPE, true, false,
                                 "image_accessor_ext");
      }
      Tex->addDecl(SubmitStmtsList.TextureList, SubmitStmtsList.SamplerList,
                   getQueueStr());
    }
  }
  for (auto &Tex : VM.getTextureMap()) {
    Tex.second->addDecl(SubmitStmtsList.TextureList,
                        SubmitStmtsList.SamplerList, getQueueStr());
  }
}

void KernelCallExpr::addAccessorDecl(MemVarInfo::VarScope Scope) {
  for (auto &VI : getVarMap().getMap(Scope)) {
    addAccessorDecl(VI.second);
  }
}

void KernelCallExpr::addAccessorDecl(std::shared_ptr<MemVarInfo> VI) {
  if (!VI->isShared()) {
    requestFeature(HelperFeatureEnum::device_ext);
    SubmitStmtsList.InitList.emplace_back(VI->getInitStmt(getQueueStr()));
    if (VI->isLocal()) {
      SubmitStmtsList.MemoryList.emplace_back(
          VI->getMemoryDecl(ExecutionConfig.ExternMemSize));
    } else if (getFilePath() != VI->getFilePath() &&
               !isIncludedFile(getFilePath(), VI->getFilePath())) {
      // Global variable definition and global variable reference are not in the
      // same file, and are not a share variable, insert extern variable
      // declaration.
      SubmitStmtsList.ExternList.emplace_back(VI->getExternGlobalVarDecl());
    }
  }
  VI->appendAccessorOrPointerDecl(
      ExecutionConfig.ExternMemSize, EmitSizeofWarning,
      SubmitStmtsList.AccessorList, SubmitStmtsList.PtrList);
  if (VI->isTypeDeclaredLocal()) {
    if (DiagnosticsUtils::report(getFilePath(), getBegin(),
                                 Diagnostics::TYPE_IN_FUNCTION, false, false,
                                 VI->getName(), VI->getLocalTypeName())) {
      if (!SubmitStmtsList.AccessorList.empty()) {
        SubmitStmtsList.AccessorList.back().Warnings.push_back(
            DiagnosticsUtils::getWarningTextAndUpdateUniqueID(
                Diagnostics::TYPE_IN_FUNCTION, VI->getName(),
                VI->getLocalTypeName()));
      }
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
      DiagnosticsUtils::report(getFilePath(), getBegin(),
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
          SubmitStmtsList.AccessorList.emplace_back(buildString(
              MapNames::getDpctNamespace() + "access_wrapper<", TypeStr, "> ",
              Arg.getIdStringWithSuffix("acc"), "(", Arg.getArgString(),
              Arg.IsDefinedOnDevice ? ".get_ptr()" : "", ", cgh);"));
          KernelArgs += buildString(Arg.getIdStringWithSuffix("acc"),
                                    ".get_raw_pointer()");
        } else {
          requestFeature(HelperFeatureEnum::device_ext);
          SubmitStmtsList.AccessorList.emplace_back(buildString(
              "auto ", Arg.getIdStringWithSuffix("acc"),
              " = " + MapNames::getDpctNamespace() + "get_access(",
              Arg.getArgString(), Arg.IsDefinedOnDevice ? ".get_ptr()" : "",
              ", cgh);"));
          KernelArgs += buildString("(", TypeStr, ")(&",
                                    Arg.getIdStringWithSuffix("acc"), "[0])");
        }
      }
    } else if (Arg.IsRedeclareRequired || IsInMacroDefine) {
      std::string TypeStr =
          Arg.getTypeString().empty()
              ? "auto"
              : (Arg.IsDeviceRandomGeneratorType ? Arg.getTypeString() + " *"
                                                 : Arg.getTypeString());
      SubmitStmtsList.CommandGroupList.emplace_back(
          buildString(TypeStr, " ", Arg.getIdStringWithIndex(), " = ",
                      Arg.getArgString(), ";"));
      KernelArgs += Arg.getIdStringWithIndex();
    } else if (Arg.Texture) {
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

void KernelCallExpr::print(KernelPrinter &Printer) {
  std::unique_ptr<KernelPrinter::Block> Block;
  if (!OuterStmts.empty()) {
    if (NeedBraces)
      Block = std::move(Printer.block(true));
    else
      Block = std::move(Printer.block(false));
    for (auto &S : OuterStmts)
      Printer.line(S.StmtStr);
  }
  if (NeedLambda) {
    Block = std::move(Printer.block(true));
  }
  printSubmit(Printer);
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
    std::vector<std::pair<std::string, std::pair<std::string, unsigned>>>
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
  if (SubmitStmtsList.empty()) {
    printParallelFor(Printer, false);
  } else {
    (Printer << "submit(").newLine();
    printSubmitLamda(Printer);
  }
}

void KernelCallExpr::printSubmitLamda(KernelPrinter &Printer) {
  auto Lamda = Printer.block();
  Printer.line("[&](" + MapNames::getClNamespace() + "handler &cgh) {");
  {
    auto Body = Printer.block();
    SubmitStmtsList.print(Printer);
    printParallelFor(Printer, true);
  }
  if (getVarMap().hasSync())
    Printer.line("}).wait();");
  else
    Printer.line("});");
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

void KernelCallExpr::printParallelFor(KernelPrinter &Printer, bool IsInSubmit) {
  std::string TemplateArgsStr;
  if (DpctGlobalInfo::isSyclNamedLambda() && hasTemplateArgs()) {
    bool IsNeedWarning = false;
    TemplateArgsStr = getTemplateArguments(IsNeedWarning, false, true);
    if (!TemplateArgsStr.empty() && IsNeedWarning) {
      printWarningMessage(Printer, Diagnostics::UNDEDUCED_TYPE,
                          "dpct_kernel_name");
    }
  }
  if (IsInSubmit) {
    Printer.indent() << "cgh.";
  }
  if (!SubmitStmtsList.NdRangeList.empty() &&
      DpctGlobalInfo::isCommentsEnabled())
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
    Printer.line("[=](" + MapNames::getClNamespace() + "nd_item<1> ",
                 getItemName(), ")", ExecutionConfig.SubGroupSize, " {");
  } else {
    DpctGlobalInfo::printCtadClass(Printer.indent(),
                                   MapNames::getClNamespace() + "nd_range", 3)
        << "(";
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
  if (hasWrittenTemplateArgs()) {
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

std::string KernelCallExpr::getReplacement() {
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

inline std::string CallFunctionExpr::getExtraArguments() {
  if (!FuncInfo)
    return "";
  return getVarMap().getExtraCallArguments(FuncInfo->NonDefaultParamNum,
                                           FuncInfo->ParamsNum -
                                               FuncInfo->NonDefaultParamNum);
}

const DeclRefExpr *getAddressedRef(const Expr *E) {
  E = E->IgnoreImplicitAsWritten();
  if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
    if (DRE->getDecl()->getKind() == Decl::Function) {
      return DRE;
    }
  } else if (auto Paren = dyn_cast<ParenExpr>(E)) {
    return getAddressedRef(Paren->getSubExpr());
  } else if (auto Cast = dyn_cast<CastExpr>(E)) {
    return getAddressedRef(Cast->getSubExprAsWritten());
  } else if (auto UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == UO_AddrOf) {
      return getAddressedRef(UO->getSubExpr());
    }
  }
  return nullptr;
}

std::shared_ptr<KernelCallExpr> KernelCallExpr::buildFromCudaLaunchKernel(
    const std::pair<std::string, unsigned> &LocInfo, const CallExpr *CE) {
  auto LaunchFD = CE->getDirectCallee();
  if (!LaunchFD || (LaunchFD->getName() != "cudaLaunchKernel" &&
                    LaunchFD->getName() != "cudaLaunchCooperativeKernel")) {
    return std::shared_ptr<KernelCallExpr>();
  }
  if (auto Callee = getAddressedRef(CE->getArg(0))) {
    auto Kernel = std::shared_ptr<KernelCallExpr>(
        new KernelCallExpr(LocInfo.second, LocInfo.first));
    Kernel->buildCalleeInfo(Callee);
    Kernel->buildLocationInfo(CE);
    Kernel->buildExecutionConfig(ArrayRef<const Expr *>{
        CE->getArg(1), CE->getArg(2), CE->getArg(4), CE->getArg(5)}, CE);
    Kernel->buildNeedBracesInfo(CE);
    auto FD =
        dyn_cast_or_null<FunctionDecl>(Callee->getReferencedDeclOfCallee());
    auto FuncInfo = Kernel->getFuncInfo();
    if (FD && FuncInfo) {
      auto ArgsArray = ExprAnalysis::ref(CE->getArg(3));
      if (!isa<DeclRefExpr>(CE->getArg(3)->IgnoreImplicitAsWritten())) {
        ArgsArray = "(" + ArgsArray + ")";
      }
      Kernel->resizeTextureObjectList(FD->getNumParams());
      for (auto &Parm : FD->parameters()) {
        Kernel->ArgsInfo.emplace_back(Parm, ArgsArray, Kernel.get());
      }
    }
    return Kernel;
  }
  return std::shared_ptr<KernelCallExpr>();
}

std::shared_ptr<KernelCallExpr>
KernelCallExpr::buildForWrapper(std::string FilePath, const FunctionDecl *FD,
                                std::shared_ptr<DeviceFunctionInfo> FuncInfo) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto Kernel =
      std::shared_ptr<KernelCallExpr>(new KernelCallExpr(0, FilePath));
  Kernel->Name = FD->getNameAsString();
  Kernel->setFuncInfo(FuncInfo);
  Kernel->ExecutionConfig.Config[0] = "";
  Kernel->ExecutionConfig.Config[1] = "";
  Kernel->ExecutionConfig.Config[2] = "localMemSize";
  Kernel->ExecutionConfig.Config[3] = "queue";
  Kernel->ExecutionConfig.Config[4] = "nr";
  Kernel->ExecutionConfig.IsDefaultStream = false;
  Kernel->ExecutionConfig.IsQueuePtr = false;
  Kernel->NeedBraces = false;
  Kernel->getFuncInfo()->getVarMap().Dim = 3;
  for (auto &Parm : FD->parameters()) {
    Kernel->ArgsInfo.emplace_back(Parm, Kernel.get());
  }
  Kernel->LocInfo.NL = getNL();
  Kernel->LocInfo.Indent = getIndent(FD->getBeginLoc(), SM).str() + "    ";
  return Kernel;
}

void KernelCallExpr::setKernelCallDim() {
  if (auto Ptr = getFuncInfo()) {
    Ptr->setKernelInvoked();
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

void KernelCallExpr::buildInfo() {
  CallFunctionExpr::buildInfo();
  TotalArgsSize =
      getVarMap().calculateExtraArgsSize() + calculateOriginArgsSize();
}

void KernelCallExpr::addReplacements() {
  if (TotalArgsSize >
      MapNames::KernelArgTypeSizeMap.at(KernelArgType::KAT_MaxParameterSize))
    DiagnosticsUtils::report(getFilePath(), getBegin(),
                             Diagnostics::EXCEED_MAX_PARAMETER_SIZE, true,
                             false);
  auto R = std::make_shared<ExtReplacement>(getFilePath(), getBegin(), 0,
                                            getReplacement(), nullptr);
  R->setBlockLevelFormatFlag();
  DpctGlobalInfo::getInstance().addReplacement(R);
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

void KernelCallExpr::setIsInMacroDefine(const CUDAKernelCallExpr *KernelCall) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  // Check if the whole kernel call is in macro arg
  auto CallBegin = KernelCall->getBeginLoc();
  auto CallEnd = KernelCall->getEndLoc();

  auto Range = getDefinitionRange(KernelCall->getBeginLoc(), KernelCall->getEndLoc());
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

#define TYPE_CAST(qual_type, type) dyn_cast<type>(qual_type)
#define ARG_TYPE_CAST(type) TYPE_CAST(ArgType, type)
#define PARM_TYPE_CAST(type) TYPE_CAST(ParmType, type)

bool TemplateArgumentInfo::isPlaceholderType(QualType QT) {
  if (auto BT = QT->getAs<BuiltinType>()) {
    if (BT->isPlaceholderType() || BT->isDependentType())
      return true;
  }
  return false;
}

template <class T>
void setTypeTemplateArgument(std::vector<TemplateArgumentInfo> &TAILis,
                             unsigned Idx, T Ty) {
  auto &TA = TAILis[Idx];
  if (TA.isNull())
    TA.setAsType(Ty);
}
template <class T>
void setNonTypeTemplateArgument(std::vector<TemplateArgumentInfo> &TAILis,
                                unsigned Idx, T Ty) {
  auto &TA = TAILis[Idx];
  if (TA.isNull())
    TA.setAsNonType(Ty);
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
  N2.print(OS, DpctGlobalInfo::getContext().getPrintingPolicy(),
           TemplateName::Qualified::Fully);
  OS.flush();
  return N1.compare(NameStr);
}

bool compareTemplateName(TemplateName N1, TemplateName N2) {
  std::string NameStr;
  llvm::raw_string_ostream OS(NameStr);
  N1.print(OS, DpctGlobalInfo::getContext().getPrintingPolicy(),
           TemplateName::Qualified::Fully);
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
      if (CTSD->getTypeAsWritten() &&
          CTSD->getTypeAsWritten()->getType()->getTypeClass() ==
              Type::TemplateSpecialization) {
        auto TL = CTSD->getTypeAsWritten()->getTypeLoc();
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
    if (ArgConstArray &&
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
  }
  deduceTemplateArgumentFromType(TAIList, ParmType, ArgType, TL);
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
      if (TTPD->hasDefaultArgument())
        Arg.setAsType(TTPD->getDefaultArgumentInfo()->getTypeLoc());
    } else if (auto NTTPD = dyn_cast<NonTypeTemplateParmDecl>(TemplateParm)) {
      if (NTTPD->hasDefaultArgument())
        Arg.setAsNonType(NTTPD->getDefaultArgument());
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

void CallFunctionExpr::setFuncInfo(std::shared_ptr<DeviceFunctionInfo> Info) {
  if (FuncInfo && Info && (FuncInfo != Info)) {
    if (!FuncInfo->getVarMap().isSameAs(Info->getVarMap())) {
      DiagnosticsUtils::report(getFilePath(), getBegin(),
                               Warnings::DEVICE_CALL_DIFFERENT, true, false,
                               FuncInfo->getFunctionName());
    }
  }
  FuncInfo = Info;
}

void CallFunctionExpr::buildCalleeInfo(const Expr *Callee) {
  if (auto CallDecl =
          dyn_cast_or_null<FunctionDecl>(Callee->getReferencedDeclOfCallee())) {
    Name = getNameWithNamespace(CallDecl, Callee);
    if (auto FTD = CallDecl->getPrimaryTemplate()) {
      if (FTD->getTemplateParameters()->hasParameterPack())
        return;
    }
    setFuncInfo(DeviceFunctionDecl::LinkRedecls(CallDecl));
    if (auto DRE = dyn_cast<DeclRefExpr>(Callee)) {
      buildTemplateArguments(DRE->template_arguments(), Callee->getSourceRange());
    }
  } else if (auto Unresolved = dyn_cast<UnresolvedLookupExpr>(Callee)) {
    Name = "";
    if(Unresolved->getQualifier())
      Name = getNestedNameSpecifierString(Unresolved->getQualifier());
    Name += Unresolved->getName().getAsString();
    setFuncInfo(DeviceFunctionDecl::LinkUnresolved(Unresolved));
    buildTemplateArguments(Unresolved->template_arguments(), Callee->getSourceRange());
  } else if (auto DependentScope =
                 dyn_cast<CXXDependentScopeMemberExpr>(Callee)) {
    Name = DependentScope->getMember().getAsString();
    buildTemplateArguments(DependentScope->template_arguments(), Callee->getSourceRange());
  } else if (auto DSDRE = dyn_cast<DependentScopeDeclRefExpr>(Callee)) {
    Name = DSDRE->getDeclName().getAsString();
    buildTemplateArgumentsFromTypeLoc(DSDRE->getQualifierLoc().getTypeLoc());
  }
}
SourceLocation getActualInsertLocation(SourceLocation InsertLoc,
                                       const SourceManager &SM,
                                       const LangOptions &LO);

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
  if (FuncInfo) {
    if (FuncInfo->NonDefaultParamNum) {
      if (Ctor->getNumArgs() >= FuncInfo->NonDefaultParamNum) {
        InsertLocation =
            Ctor->getArg(FuncInfo->NonDefaultParamNum - 1)->getEndLoc();
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
  buildCalleeInfo(CE->getCallee()->IgnoreParenImpCasts());
  buildTextureObjectArgsInfo(CE);
  bool HasImplicitArg = false;
  if (auto FD = CE->getDirectCallee()) {
    IsAllTemplateArgsSpecified = deduceTemplateArguments(CE, FD, TemplateArgs);
    HasImplicitArg = isa<CXXOperatorCallExpr>(CE) && isa<CXXMethodDecl>(FD);
  } else if (auto Unresolved = dyn_cast<UnresolvedLookupExpr>(
                 CE->getCallee()->IgnoreImplicitAsWritten())) {
    if (Unresolved->getNumDecls())
      IsAllTemplateArgsSpecified = deduceTemplateArguments(
          CE, Unresolved->decls_begin().getDecl(), TemplateArgs);
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

  if (FuncInfo) {
    if (FuncInfo->ParamsNum == 0) {
      ExtraArgLoc =
          DpctGlobalInfo::getSourceManager().getFileOffset(CE->getRParenLoc());
    } else if (FuncInfo->NonDefaultParamNum == 0) {
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
      if (CE->getNumArgs() > FuncInfo->NonDefaultParamNum - 1) {
        auto &SM = DpctGlobalInfo::getSourceManager();
        auto CERange = getDefinitionRange(CE->getBeginLoc(), CE->getEndLoc());
        auto TempLoc = Lexer::getLocForEndOfToken(CERange.getEnd(),  0, SM, DpctGlobalInfo::getContext().getLangOpts());
        auto PairRange = getRangeInRange(CE->getArg(FuncInfo->NonDefaultParamNum - 1 + HasImplicitArg), CERange.getBegin(), TempLoc);
        auto RealEnd = PairRange.second;
        auto IT = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
            getCombinedStrFromLoc(RealEnd));
        if (IT !=
                dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end() &&
            IT->second->TokenIndex == IT->second->NumTokens) {
          RealEnd = SM.getImmediateExpansionRange(
                          CE->getArg(FuncInfo->NonDefaultParamNum - 1 +
                                     HasImplicitArg)
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

        ExtraArgLoc =
            DpctGlobalInfo::getSourceManager().getFileOffset(RealEnd);
      } else {
        ExtraArgLoc = 0;
      }
    }
  }

}

template <class TargetType>
std::shared_ptr<TargetType> makeTextureObjectInfo(const ValueDecl *D,
                                                  bool IsKernelCall) {
  if (IsKernelCall) {
    if (auto VD = dyn_cast<VarDecl>(D)) {
      return std::make_shared<TargetType>(VD);
    }
  } else if (auto PVD = dyn_cast<ParmVarDecl>(D)) {
    return std::make_shared<TargetType>(PVD);
  }
  return std::shared_ptr<TargetType>();
}

void CallFunctionExpr::buildTextureObjectArgsInfo(const CallExpr *CE) {
  if (auto ME = dyn_cast<MemberExpr>(CE->getCallee()->IgnoreImpCasts())) {
    if (auto DRE = dyn_cast<DeclRefExpr>(ME->getBase()->IgnoreImpCasts())) {
      auto BaseObject = makeTextureObjectInfo<StructureTextureObjectInfo>(
          DRE->getDecl(), CE->getStmtClass() == Stmt::CUDAKernelCallExprClass);
      if (BaseObject)
        BaseTextureObject = std::move(BaseObject);
    }
  }
  buildTextureObjectArgsInfo<CallExpr>(CE);
}

std::shared_ptr<TextureObjectInfo> CallFunctionExpr::addTextureObjectArg(
    unsigned ArgIdx, const DeclRefExpr *TexRef, bool isKernelCall) {
  std::shared_ptr<TextureObjectInfo> Info;
  if (TextureObjectInfo::isTextureObject(TexRef)) {
    Info = makeTextureObjectInfo<TextureObjectInfo>(TexRef->getDecl(), isKernelCall);
  } else if (TexRef->getType()->isRecordType()) {
    Info = makeTextureObjectInfo<StructureTextureObjectInfo>(TexRef->getDecl(), isKernelCall);
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

void CallFunctionExpr::mergeTextureObjectInfo() {
  if (BaseTextureObject)
    BaseTextureObject->merge(FuncInfo->getBaseTextureObject());
  for (unsigned Idx = 0; Idx < TextureObjectList.size(); ++Idx) {
    if (auto &Obj = TextureObjectList[Idx]) {
      Obj->merge(FuncInfo->getTextureObject(Idx));
    }
  }
}

std::string CallFunctionExpr::getName(const NamedDecl *D) {
  if (auto ID = D->getIdentifier())
    return ID->getName().str();
  return "";
}

void CallFunctionExpr::buildInfo() {
  if (!FuncInfo)
    return;

  const std::string &DefFilePath = FuncInfo->getDefinitionFilePath();
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
  if (!DefFilePath.empty() && DefFilePath != getFilePath() &&
      !isIncludedFile(getFilePath(), DefFilePath) && !FuncInfo->isLambda()) {
    FuncInfo->setNeedSyclExternMacro();
  }

  if (DpctGlobalInfo::isOptimizeMigration() && !FuncInfo->isInlined() &&
      !FuncInfo->IsSyclExternMacroNeeded()) {
    if (FuncInfo->isKernel())
      FuncInfo->setForceInlineDevFunc();
    else
      FuncInfo->setAlwaysInlineDevFunc();
  }

  FuncInfo->buildInfo();
  VarMap.merge(FuncInfo->getVarMap(), TemplateArgs);
  mergeTextureObjectInfo();
}

void CallFunctionExpr::emplaceReplacement() {
  buildInfo();

  if (ExtraArgLoc)
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, ExtraArgLoc, 0,
                                         getExtraArguments(), nullptr));
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
    if(TA.isNull() && !Str.empty()) {
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
      if (StrRef.startswith("(lambda at")) {
        Str = "class lambda_" + getHashAsString(Str).substr(0, 6);
      }
      appendString(OS, Str, ", ");
    }
  }
  OS.flush();
  return (Result.empty()) ? Result : Result.erase(Result.size() - 2);
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

void processTypeLoc(const TypeLoc &TL, ExprAnalysis &EA,
                    const SourceManager &SM) {
  EA.analyze(TL);
  if (EA.hasReplacement()) {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(SM, &TL, EA.getReplacedString(),
                                         nullptr));
  }
}

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

void DeviceFunctionInfo::mergeTextureObjectList(
    const std::vector<std::shared_ptr<TextureObjectInfo>> &Other) {
  auto SelfItr = TextureObjectList.begin();
  auto BranchItr = Other.begin();
  while ((SelfItr != TextureObjectList.end()) &&
         (BranchItr != Other.end())) {
    if (!(*SelfItr))
      *SelfItr = *BranchItr;
    ++SelfItr;
    ++BranchItr;
  }
  TextureObjectList.insert(SelfItr, BranchItr, Other.end());
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
    if(Obj->getParamIdx() >= TextureObjectList.size())
      continue;
    if (auto &Parm = TextureObjectList[Obj->getParamIdx()]) {
      Parm->merge(Obj);
    } else {
      TextureObjectList[Obj->getParamIdx()] = Obj;
    }
  }
}

void DeviceFunctionInfo::buildInfo() {
  if (isBuilt())
    return;
  setBuilt();
  for (auto &Call : CallExprMap) {
    Call.second->emplaceReplacement();
    VarMap.merge(Call.second->getVarMap());
    mergeCalledTexObj(Call.second->getBaseTextureObjectInfo(),
                      Call.second->getTextureObjectList());
  }
  VarMap.removeDuplicateVar();
}

std::string DeviceFunctionDecl::getExtraParameters() {
  std::string Result =
      FuncInfo->getExtraParameters(FilePath, FormatInformation);
  if (!Result.empty() && IsReplaceFollowedByPP) {
    Result += getNL();
  }
  return Result;
}

std::string ExplicitInstantiationDecl::getExtraParameters() {
  return getFuncInfo()->getExtraParameters(FilePath, InstantiationArgs,
                                           getFormatInfo());
}

inline void DeviceFunctionDeclInModule::insertWrapper() {
  auto NL = std::string(getNL());
  std::string WrapperStr = "";
  llvm::raw_string_ostream OS(WrapperStr);
  KernelPrinter Printer(NL, "", OS);
  Printer.newLine();
  Printer.newLine();
  Printer.line("extern \"C\" {");
  {
    auto FunctionBlock = Printer.block();
    Printer.indent();
    requestFeature(HelperFeatureEnum::device_ext);
    Printer << "DPCT_EXPORT void " << FuncName << "_wrapper(" << MapNames::getClNamespace()
            << "queue &queue, const " << MapNames::getClNamespace()
            << "nd_range<3> &nr, unsigned int localMemSize, void "
               "**kernelParams, void **extra)";
    if (HasBody) {
      auto for_each_parameter = [&](auto F) {
	auto it = getParametersInfo().begin();
	for (int i = 0;
	     it != getParametersInfo().end();
	     ++it, ++i) {
	  F(i, it->second);
	}
      };

      Printer << " {";
      {
        auto BodyBlock = Printer.block();
        Printer.newLine();
	auto DefaultParamNum = ParamsNum-NonDefaultParamNum;
	Printer.line(llvm::formatv(
           "// {0} non-default parameters, {1} default parameters",
           NonDefaultParamNum, DefaultParamNum));
	Printer.line(llvm::formatv(
           "{0}args_selector<{1}, {2}, decltype({3})> selector(kernelParams, extra);",
           MapNames::getDpctNamespace(),
           NonDefaultParamNum, DefaultParamNum, FuncName));
	for_each_parameter([&](auto&& i, auto&& p) {
	  Printer.line("auto& " + p
		       + " = selector.get<"
		       + std::to_string(i) + ">();");
	});

        Kernel->buildInfo();
        Printer.line(Kernel->getReplacement());
      }
      Printer.line("}");
    } else {
      Printer << ";";
      Printer.newLine();
    }
  }

  Printer << "}";

  auto Repl = std::make_shared<ExtReplacement>(FilePath, DeclEnd, 0, WrapperStr,
                                               nullptr);
  Repl->setBlockLevelFormatFlag();
  DpctGlobalInfo::getInstance().addReplacement(Repl);
}

inline void DeviceFunctionDecl::emplaceReplacement() {
  auto Repl = std::make_shared<ExtReplacement>(
      FilePath, ReplaceOffset, ReplaceLength, getExtraParameters(), nullptr);
  Repl->setNotFormatFlag();
  DpctGlobalInfo::getInstance().addReplacement(Repl);

  if (FuncInfo->IsSyclExternMacroNeeded()) {
    std::string StrRepl = "SYCL_EXTERNAL ";
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, Offset, 0, StrRepl,
                                         nullptr));
  }

  if (FuncInfo->IsAlwaysInlineDevFunc()) {
    std::string StrRepl = "inline ";
    DpctGlobalInfo::getInstance().addReplacement(
      std::make_shared<ExtReplacement>(FilePath, Offset, 0, StrRepl, nullptr));
  }
  if (FuncInfo->IsForceInlineDevFunc()) {
    std::string StrRepl = "__dpct_inline__ ";
    DpctGlobalInfo::getInstance().addReplacement(
      std::make_shared<ExtReplacement>(FilePath, Offset, 0, StrRepl, nullptr));
  }

  for (auto &Obj : TextureObjectList) {
    if (Obj) {
      Obj->merge(FuncInfo->getTextureObject((Obj->getParamIdx())));
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
}

inline void DeviceFunctionDeclInModule::emplaceReplacement() {
  DeviceFunctionDecl::emplaceReplacement();
  insertWrapper();
}

void DeviceFunctionDeclInModule::buildParameterInfo(const FunctionDecl *FD) {
  for (auto It = FD->param_begin(); It != FD->param_end(); It++) {
    ParametersInfo.push_back(std::pair<std::string, std::string>(
        (*It)->getOriginalType().getAsString(), (*It)->getNameAsString()));
  }
}

void DeviceFunctionDeclInModule::buildWrapperInfo(const FunctionDecl *FD) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  const FunctionDecl *Def;
  HasBody = FD->hasBody(Def);
  if (HasBody && FD != Def) {
    HasBody = false;
  }

  FuncName = FD->getNameAsString();
  // FD has relatively large range, which is likely to be straddle,
  // getDefinitionRange may not work as good as getExpansionRange
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

void DeviceFunctionDeclInModule::buildCallInfo(const FunctionDecl *FD) {
  Kernel = KernelCallExpr::buildForWrapper(FilePath, FD, getFuncInfo(FD));
}

bool isModuleFunction(const FunctionDecl *FD) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  return
    FD->getLanguageLinkage() == CLanguageLinkage
    && FD->hasAttr<CUDAGlobalAttr>()
    && DpctGlobalInfo::getModuleFiles().find(
          DpctGlobalInfo::getLocInfo(SM.getExpansionLoc(FD->getBeginLoc()))
              .first) != DpctGlobalInfo::getModuleFiles().end();
}

DeviceFunctionDecl::DeviceFunctionDecl(unsigned Offset,
                                       const std::string &FilePathIn,
                                       const FunctionDecl *FD)
    : Offset(Offset), FilePath(FilePathIn), ParamsNum(FD->param_size()),
      ReplaceOffset(0), ReplaceLength(0),
      NonDefaultParamNum(FD->getMostRecentDecl()->getMinRequiredArguments()),
      FuncInfo(getFuncInfo(FD)) {
  if (!FuncInfo) {
    FuncInfo = std::make_shared<DeviceFunctionInfo>(
        FD->param_size(), NonDefaultParamNum, getFunctionName(FD));
  }
  if (!FilePath.empty()) {
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
}

DeviceFunctionDecl::DeviceFunctionDecl(unsigned Offset,
                                       const std::string &FilePathIn,
                                       const FunctionTypeLoc &FTL,
                                       const ParsedAttributes &Attrs,
                                       const FunctionDecl *Specialization)
    : Offset(Offset), FilePath(FilePathIn),
      ParamsNum(Specialization->getNumParams()), ReplaceOffset(0),
      ReplaceLength(0),
      NonDefaultParamNum(
          Specialization->getMostRecentDecl()->getMinRequiredArguments()),
      FuncInfo(getFuncInfo(Specialization)) {
  IsDefFilePathNeeded = false;

  buildReplaceLocInfo(FTL, Attrs);
  buildTextureObjectParamsInfo(FTL.getParams());
}

bool isInSameLine(SourceLocation First, SourceLocation Second,
                  const SourceManager &SM) {
  bool Invalid = false;
  return ::isInSameLine(SM.getExpansionLoc(First), SM.getExpansionLoc(Second),
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

void DeviceFunctionDecl::setFuncInfo(std::shared_ptr<DeviceFunctionInfo> Info) {
  if (FuncInfo.get() == Info.get())
    return;
  FuncInfo = Info;
  if (IsDefFilePathNeeded)
    FuncInfo->setDefinitionFilePath(FilePath);
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
  if (isModuleFunction(FD)) {
    D = DpctGlobalInfo::getInstance().insertDeviceFunctionDeclInModule(FD);
  } else {
    D = DpctGlobalInfo::getInstance().insertDeviceFunctionDecl(FD);
  }
  if (Info) {
    if (auto FuncInfo = D->getFuncInfo())
      Info->merge(FuncInfo);
    D->setFuncInfo(Info);
  } else if (auto FuncInfo = D->getFuncInfo())
    Info = FuncInfo;
  else
    List.push_back(D);
}

void DeviceFunctionDecl::LinkRedecls(
    const FunctionDecl *FD, DeclList &List,
    std::shared_ptr<DeviceFunctionInfo> &Info) {
  LinkDeclRange(FD->redecls(), List, Info);
}

void DeviceFunctionDecl::LinkDecl(const FunctionTemplateDecl *FTD,
                                  DeclList &List,
                                  std::shared_ptr<DeviceFunctionInfo> &Info) {
  LinkDecl(FTD->getAsFunction(), List, Info);
  LinkDeclRange(FTD->specializations(), List, Info);
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

MemVarInfo::MemVarInfo(unsigned Offset, const std::string &FilePath,
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
  if (Var->getStorageClass() == SC_Static ||
      getAddressAttr(Var)==Constant) {
    IsStatic = true;
  }

  if (auto Func = Var->getParentFunctionOrMethod()) {
    if (DeclOfVarType = Var->getType()->getAsCXXRecordDecl()) {
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
          auto Iter = AnonymousTypeDeclStmtMap.find(DS2);
          if (Iter != AnonymousTypeDeclStmtMap.end()) {
            LocalTypeName = "type_ct" + std::to_string(Iter->second);
          } else {
            LocalTypeName =
                "type_ct" + std::to_string(AnonymousTypeDeclStmtMap.size() + 1);
            AnonymousTypeDeclStmtMap.insert(
                std::make_pair(DS2, AnonymousTypeDeclStmtMap.size() + 1));
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
  } else {
    AccMode = Accessor;
  }

  newConstVarInit(Var);
}

std::shared_ptr<DeviceFunctionInfo> &
DeviceFunctionDecl::getFuncInfo(const FunctionDecl *FD) {
  DpctNameGenerator G;
  std::string Key;
  // For static functions or functions in anonymous namespace,
  // need to add filepath as prefix to differentiate them.
  if (FD->isStatic() || FD->isInAnonymousNamespace()) {
    auto LocInfo = DpctGlobalInfo::getLocInfo(FD);
    Key = LocInfo.first + G.getName(FD);
  } else {
    Key = G.getName(FD);
  }
  return FuncInfoMap[Key];
}

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

std::string MemVarInfo::getMemoryType() {
  switch (Attr) {
  case clang::dpct::MemVarInfo::Device: {
    requestFeature(HelperFeatureEnum::device_ext);
    static std::string DeviceMemory =
        MapNames::getDpctNamespace() + "global_memory";
    return getMemoryType(DeviceMemory, getType());
  }
  case clang::dpct::MemVarInfo::Constant: {
    requestFeature(HelperFeatureEnum::device_ext);
    static std::string ConstantMemory =
        MapNames::getDpctNamespace() + "constant_memory";
    return getMemoryType(ConstantMemory, getType());
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

const std::string &MemVarInfo::getMemoryAttr() {
  requestFeature(HelperFeatureEnum::device_ext);
  switch (Attr) {
  case clang::dpct::MemVarInfo::Device: {
    static std::string DeviceMemory = MapNames::getDpctNamespace() + "global";
    return DeviceMemory;
  }
  case clang::dpct::MemVarInfo::Constant: {
    static std::string ConstantMemory =
        MapNames::getDpctNamespace() + "constant";
    return ConstantMemory;
  }
  case clang::dpct::MemVarInfo::Shared: {
    static std::string SharedMemory = MapNames::getDpctNamespace() + "local";
    return SharedMemory;
  }
  case clang::dpct::MemVarInfo::Managed: {
    static std::string ManagedMemory = MapNames::getDpctNamespace() + "shared";
    return ManagedMemory;
  }
  default:
    llvm::dbgs() << "[MemVarInfo::getMemoryAttr] Unexpected attribute.";
    static std::string NullString;
    return NullString;
  }
}

std::string MemVarInfo::getDeclarationReplacement(const VarDecl *VD) {
  switch (Scope) {
  case clang::dpct::MemVarInfo::Local:
    if (DpctGlobalInfo::useGroupLocalMemory() && VD) {

      auto FD = dyn_cast<FunctionDecl>(VD->getDeclContext());
      if (FD && FD->hasAttr<CUDADeviceAttr>())
        DiagnosticsUtils::report(getFilePath(), getOffset(),
                                 Diagnostics::GROUP_LOCAL_MEMORY, true, false);

      std::string Ret;
      llvm::raw_string_ostream OS(Ret);
      OS << "auto &" << getName() << " = "
         << "*" << MapNames::getClNamespace()
         << "ext::oneapi::group_local_memory_for_overwrite<" << getType()->getBaseName();
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
    return getMemoryDecl();
  }
  }
  clang::dpct::DpctDebugs()
      << "[MemVarInfo::VarAttrKind] Unexpected value: " << Scope << "\n";
  assert(0);
  return "";
}

std::string MemVarInfo::getSyclAccessorType() {
  std::string Ret;
  llvm::raw_string_ostream OS(Ret);
  if (getAttr() == MemVarInfo::VarAttrKind::Shared) {
    OS << MapNames::getClNamespace() << "local_accessor<";
    OS << getAccessorDataType() << ", ";
    OS << getType()->getDimension() << ">";
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
void MemVarInfo::appendAccessorOrPointerDecl(const std::string &ExternMemSize,
                                             bool ExternEmitWarning,
                                             StmtList &AccList,
                                             StmtList &PtrList) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  if (isShared()) {
    OS << getSyclAccessorType();
    OS << " " << getAccessorName() << "(";
    if (getType()->getDimension())
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
                               Diagnostics::SIZEOF_WARNING, false, false, "local memory");
      AccDecl.Warnings.push_back(
          DiagnosticsUtils::getWarningTextAndUpdateUniqueID(
              Diagnostics::SIZEOF_WARNING, "local memory"));
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
void MemVarMap::removeDuplicateVar() {
  std::unordered_set<std::string> VarNames{getItemName(),
                                           DpctGlobalInfo::getStreamName()};
  dpct::removeDuplicateVar(GlobalVarMap, VarNames);
  dpct::removeDuplicateVar(LocalVarMap, VarNames);
  dpct::removeDuplicateVar(ExternVarMap, VarNames);
  dpct::removeDuplicateVar(TextureMap, VarNames);
}

std::string MemVarMap::getExtraCallArguments(bool HasPreParam,
                                             bool HasPostParam) const {
  return getArgumentsOrParameters<CallArgument>(HasPreParam, HasPostParam);
}
std::string MemVarMap::getExtraDeclParam(bool HasPreParam, bool HasPostParam,
                                         FormatInfo FormatInformation) const {
  return getArgumentsOrParameters<DeclParameter>(HasPreParam, HasPostParam,
                                                 FormatInformation);
}
std::string MemVarMap::getKernelArguments(bool HasPreParam, bool HasPostParam,
                                          const std::string &Path) const {
  requestFeatureForAllVarMaps(Path);
  return getArgumentsOrParameters<KernelArgument>(HasPreParam, HasPostParam);
}
bool MemVarMap::isSameAs(const MemVarMap& Other) const {
  if (HasItem != Other.HasItem)
    return false;
  if (HasStream != Other.HasStream)
    return false;
  if (HasSync != Other.HasSync)
    return false;

  #define COMPARE_MAP(MAP)                                                     \
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

CtTypeInfo::CtTypeInfo(const TypeLoc &TL, bool NeedSizeFold)
    : PointerLevel(0), IsTemplate(false) {
  setTypeInfo(TL, NeedSizeFold);
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

void CtTypeInfo::setArrayInfo(const IncompleteArrayTypeLoc &TL,
                              bool NeedSizeFold) {
  Range.emplace_back();
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
      if (MapNames::SupportedVectorTypes.count(RD->getNameAsString()) == 0) {
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

void SizeInfo::setTemplateList(
    const std::vector<TemplateArgumentInfo> &TemplateList) {
  if (TDSI)
    TDSI = TDSI->applyTemplateArguments(TemplateList);
}

void TimeStubTypeInfo::buildInfo(std::string FilePath, unsigned int Offset,
                                 bool isReplTxtWithSB) {
  if (isReplTxtWithSB)
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, Offset, Length, StrWithSB,
                                         nullptr));
  else
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, Offset, Length, StrWithoutSB,
                                         nullptr));
}

void EventSyncTypeInfo::buildInfo(std::string FilePath, unsigned int Offset) {
  if (NeedReport)
    DiagnosticsUtils::report(FilePath, Offset,
                             Diagnostics::NOERROR_RETURN_COMMA_OP, true, false);

  if (IsAssigned && ReplText.empty()) {
    ReplText = "0";
  }

  DpctGlobalInfo::getInstance().addReplacement(std::make_shared<ExtReplacement>(
      FilePath, Offset, Length, ReplText, nullptr));
}

void BuiltinVarInfo::buildInfo(std::string FilePath, unsigned int Offset,
                               unsigned int ID) {
  std::string R = Repl + std::to_string(ID) + ")";
  DpctGlobalInfo::getInstance().addReplacement(
      std::make_shared<ExtReplacement>(FilePath, Offset, Len, R, nullptr));
}

bool isInAnalysisScope(SourceLocation SL) { return DpctGlobalInfo::isInAnalysisScope(SL); }

std::vector<std::shared_ptr<FreeQueriesInfo>> FreeQueriesInfo::InfoList;
std::vector<std::shared_ptr<FreeQueriesInfo::MacroInfo>>
    FreeQueriesInfo::MacroInfos;

const FreeQueriesInfo::FreeQueriesNames &
FreeQueriesInfo::getNames(FreeQueriesKind K) {
  static FreeQueriesNames Names[FreeQueriesInfo::FreeQueriesKind::End] = {
      {getItemName(),
       MapNames::getClNamespace() + "ext::oneapi::experimental::this_nd_item",
       getItemName()},
      {getItemName() + ".get_group()",
       MapNames::getClNamespace() + "ext::oneapi::experimental::this_group",
       "group" + getCTFixedSuffix()},
      {getItemName() + ".get_sub_group()",
       MapNames::getClNamespace() + "ext::oneapi::experimental::this_sub_group",
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

static const std::string RegexPrefix = "{{NEEDREPLACE", RegexSuffix = "}}";

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

template <class F, class... Ts>
std::string buildStringFromPrinter(F Func, Ts &&...Args) {
  std::string Ret;
  llvm::raw_string_ostream OS(Ret);
  Func(OS, std::forward<Ts>(Args)...);
  return OS.str();
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

std::string getStringForRegexDefaultQueueAndDevice(HelperFuncType HFT,
                                                   int Index);

std::string DpctGlobalInfo::getStringForRegexReplacement(StringRef MatchedStr) {
  unsigned Index = 0;
  char Method = MatchedStr[RegexPrefix.length()];
  bool HasError =
      MatchedStr.substr(RegexPrefix.length() + 1).consumeInteger(10, Index);
  assert(!HasError && "Must consume an integer");
  (void) HasError;
  // D: device, used for pretty code
  // Q: queue, used for pretty code
  // R: range dim, used for built-in variables (threadIdx.x,...) migration
  // G: range dim, used for cg::thread_block migration
  // C: range dim, used for cub block migration
  // F: free queries function migration, such as this_nd_item, this_group,
  //    this_sub_group.
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
  case FreeQueriesInfo::FreeQueriesRegexCh:
    return FreeQueriesInfo::getReplaceString(Index);
  default:
    clang::dpct::DpctDebugs() << "[char] Unexpected value: " << Method << "\n";
    assert(0);
    return MatchedStr.str();
  }
}

const std::string &getDefaultString(HelperFuncType HFT) {
  const static std::string NullString;
  switch (HFT) {
  case clang::dpct::HelperFuncType::HFT_DefaultQueue: {
    const static std::string DefaultQueue =
        DpctGlobalInfo::useNoQueueDevice()
            ? DpctGlobalInfo::getGlobalQueueName()
            : buildString(MapNames::getDpctNamespace() + "get_" +
                          DpctGlobalInfo::getDeviceQueueName() + "()");
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
        HelperFuncReplInfoIter->second.DeclLocFile + ":" +
        std::to_string(HelperFuncReplInfoIter->second.DeclLocOffset);

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

const std::string &DpctGlobalInfo::getDeviceQueueName() {
  static const std::string DeviceQueue = [&]() {
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
      return "out_of_order_queue";
    else
      return "in_order_queue";
  }();
  return DeviceQueue;
}

std::string DpctGlobalInfo::getDefaultQueue(const Stmt *S) {
  auto Idx = getPlaceholderIdx(S);
  if (!Idx) {
    Idx = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Idx, S, HelperFuncType::HFT_DefaultQueue);
  }

  return buildString(RegexPrefix, 'Q', Idx, RegexSuffix);
}

void StructureTextureObjectInfo::merge(
    std::shared_ptr<StructureTextureObjectInfo> Target) {
  if (!Target)
    return;

  dpct::merge(Members, Target->Members);
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

} // namespace dpct
} // namespace clang
