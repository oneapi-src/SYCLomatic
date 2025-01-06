//===--------------- AnalysisInfo.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_ANALYSIS_INFO_H
#define DPCT_ANALYSIS_INFO_H

#include "ErrorHandle/Error.h"
#include "RuleInfra/ExprAnalysis.h"
#include "ExtReplacements.h"
#include "RulesInclude/InclusionHeaders.h"
#include "UserDefinedRules/UserDefinedRules.h"
#include "FileGenerator/GenFiles.h"
#include "MigrationReport/Statics.h"
#include "TextModification.h"
#include "Utility.h"
#include "CommandOption/ValidateArguments.h"
#include <bitset>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/ParentMapContext.h"

#include "clang/Basic/Cuda.h"

#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"

llvm::StringRef getReplacedName(const clang::NamedDecl *D);
void setGetReplacedNamePtr(llvm::StringRef (*Ptr)(const clang::NamedDecl *D));

namespace clang {
namespace dpct {

using format::FormatRange;
using LocInfo = std::pair<tooling::UnifiedPath, unsigned int>;
template <class F, class... Ts>
std::string buildStringFromPrinter(F Func, Ts &&...Args) {
  std::string Ret;
  llvm::raw_string_ostream OS(Ret);
  Func(OS, std::forward<Ts>(Args)...);
  return OS.str();
}

enum class HelperFuncType : int {
  HFT_InitValue = 0,
  HFT_DefaultQueue = 1,
  HFT_CurrentDevice = 2,
  HFT_DefaultQueuePtr = 3
};

enum class KernelArgType : int {
  KAT_Stream = 0,
  KAT_Texture,
  KAT_Accessor1D,
  KAT_Accessor2D,
  KAT_Accessor3D,
  KAT_Array1D,
  KAT_Array2D,
  KAT_Array3D,
  KAT_Default,
  KAT_MaxParameterSize
};
// This struct defines a set of Repls with priority.
// The priority is designated by an unsigned number, the
// higher the number, the higher the priority.
struct PriorityReplInfo {
  std::vector<std::shared_ptr<ExtReplacement>> Repls;
  std::vector<std::function<void(void)>> RelatedAction;
  unsigned int Priority = 0;
};

class CudaMallocInfo;
class TextureInfo;
class KernelCallExpr;
class DeviceFunctionInfo;
class CallFunctionExpr;
class DeviceFunctionDecl;
class MemVarInfo;
class VarInfo;
class ExplicitInstantiationDecl;
class KernelPrinter;

struct EventSyncTypeInfo {
  EventSyncTypeInfo(unsigned int Length, std::string ReplText, bool NeedReport,
                    bool IsAssigned)
      : Length(Length), ReplText(ReplText), NeedReport(NeedReport),
        IsAssigned(IsAssigned) {}
  void buildInfo(clang::tooling::UnifiedPath FilePath, unsigned int Offset);

  unsigned int Length;
  std::string ReplText;
  bool NeedReport = false;
  bool IsAssigned = false;
};

struct TimeStubTypeInfo {
  TimeStubTypeInfo(unsigned int Length, std::string StrWithSB,
                   std::string StrWithoutSB)
      : Length(Length), StrWithSB(StrWithSB), StrWithoutSB(StrWithoutSB) {}

  void buildInfo(clang::tooling::UnifiedPath FilePath, unsigned int Offset,
                 bool isReplTxtWithSB);

  unsigned int Length;
  std::string StrWithSB;
  std::string StrWithoutSB;
};

struct BuiltinVarInfo {
  BuiltinVarInfo(unsigned int Len, std::string Repl,
                 std::shared_ptr<DeviceFunctionInfo> DFI)
      : Len(Len), Repl(Repl), DFI(DFI) {}
  void buildInfo(clang::tooling::UnifiedPath FilePath, unsigned int Offset,
                 unsigned int Dim);

  unsigned int Len = 0;
  std::string Repl;
  std::shared_ptr<DeviceFunctionInfo> DFI = nullptr;
};

struct FormatInfo {
  FormatInfo() : EnableFormat(false), IsAllParamsOneLine(true) {}
  bool EnableFormat;
  bool IsAllParamsOneLine;
  bool IsEachParamNL = false;
  int CurrentLength = 0;
  int NewLineIndentLength = 0;
  std::string NewLineIndentStr;
  bool IsFirstArg = false;
};

enum HDFuncInfoType { HDFI_Def, HDFI_Decl, HDFI_Call };

struct HostDeviceFuncLocInfo {
  clang::tooling::UnifiedPath FilePath;
  std::string FuncContentCache;
  unsigned FuncStartOffset = 0;
  unsigned FuncEndOffset = 0;
  unsigned FuncNameOffset = 0;
  bool Processed = false;
  bool CalledByHostDeviceFunction = false;
  HDFuncInfoType Type;
};

struct HostDeviceFuncInfo {
  std::unordered_map<std::string, HostDeviceFuncLocInfo> LocInfos;
  bool isDefInserted = false;
  bool needGenerateHostCode = false;
  int PostFixId = -1;
  static int MaxId;
};

enum IfType { IT_Unknow, IT_If, IT_Ifdef, IT_Ifndef, IT_Elif };

struct DirectiveInfo {
  unsigned NumberSignLoc = 0;
  unsigned DirectiveLoc = 0;
  unsigned ConditionLoc = 0;
  std::string Condition;
};

struct CudaArchPPInfo {
  IfType DT = IfType::IT_Unknow;
  DirectiveInfo IfInfo;
  DirectiveInfo ElseInfo;
  DirectiveInfo EndInfo;
  std::unordered_map<unsigned, DirectiveInfo> ElInfo;
  bool isInHDFunc = false;
};
// Field: The member field of user defined class type.
// Base: The base class of user defined class type.
// Alias: The alias name of user defined class type.
// The enum is using to clarify the different user defined type when
// migrate the codepin with user defined class.
enum class CodePinVarInfoType { Field, Base, Alias };
struct MemberOrBaseInfoForCodePin {
  bool UserDefinedTypeFlag = false;
  int PointerDepth = 0;
  bool IsBaseMember = false;
  std::vector<int> Dims;
  std::string TypeNameInCuda;
  std::string TypeNameInSycl;
  std::string MemberName;
  std::string CodePinMemberName;
};

struct VarInfoForCodePin {
  bool TemplateFlag = false;
  bool TopTypeFlag = false;
  bool IsValid = false;
  bool IsTypeDef = false;
  std::string OrgTypeName;
  std::string HashKey;
  std::string VarRecordType;
  std::string VarName;
  std::string VarNameWithoutScopeAndTemplateArgs;
  std::string TemplateInstArgs;
  std::vector<std::string> Namespaces;
  std::vector<std::string> TemplateArgs;
  std::vector<MemberOrBaseInfoForCodePin> Bases;
  std::vector<MemberOrBaseInfoForCodePin> Members;
};

struct MemcpyOrderAnalysisInfo {
  MemcpyOrderAnalysisInfo(
      std::vector<std::pair<const Stmt *, MemcpyOrderAnalysisNodeKind>>
          MemcpyOrderVec,
      std::vector<unsigned int> DREOffsetVec)
      : MemcpyOrderVec(MemcpyOrderVec), DREOffsetVec(DREOffsetVec) {}
  MemcpyOrderAnalysisInfo() : MemcpyOrderVec({}), DREOffsetVec({}) {}

  std::vector<std::pair<const Stmt *, MemcpyOrderAnalysisNodeKind>>
      MemcpyOrderVec;
  std::vector<unsigned int> DREOffsetVec;
};

struct RnnBackwardFuncInfo {
  clang::tooling::UnifiedPath FilePath;
  unsigned int Offset;
  unsigned int Length;
  bool isAssigned;
  bool isDataGradient;
  std::string CompoundLoc;
  std::vector<std::string> RnnInputDeclLoc;
  std::vector<std::string> FuncArgs;
};

struct DeviceFunctionInfoForWrapper {
  std::vector<std::pair<std::string, std::string>> ParametersInfo;
  std::vector<std::pair<std::string, std::string>> TemplateParametersInfo;
  std::shared_ptr<KernelCallExpr> KernelForWrapper;
};

// <function name, Info>
using HDFuncInfoMap = std::unordered_map<std::string, HostDeviceFuncInfo>;
// <file path, <Offset, Info>>
using CudaArchPPMap =
    std::unordered_map<clang::tooling::UnifiedPath,
                       std::unordered_map<unsigned int, CudaArchPPInfo>>;
using CudaArchDefMap =
    std::unordered_map<std::string,
                       std::unordered_map<unsigned int, unsigned int>>;
class ParameterStream {
public:
  ParameterStream() { FormatInformation = FormatInfo(); }
  ParameterStream(FormatInfo FormatInformation, int ColumnLimit)
      : FormatInformation(FormatInformation), ColumnLimit(ColumnLimit) {}

  ParameterStream &operator<<(const std::string &InputParamStr);
  ParameterStream &operator<<(int InputInt);

  std::string Str = "";
  FormatInfo FormatInformation;
  int ColumnLimit = 80;
};

struct StmtWithWarning {
  StmtWithWarning(std::string Str, std::vector<std::string> Warnings = {})
      : StmtStr(Str), Warnings(Warnings) {}

  std::string StmtStr;
  std::vector<std::string> Warnings;
};

using StmtList = std::vector<StmtWithWarning>;

template <class T> using GlobalMap = std::map<unsigned, std::shared_ptr<T>>;
using MemVarInfoMap = GlobalMap<MemVarInfo>;

template <class T> inline void merge(T &Master, const T &Branch) {
  Master.insert(Branch.begin(), Branch.end());
}

inline void appendString(llvm::raw_string_ostream &OS) {}
template <class FirstArgT, class... ArgsT>
inline void appendString(llvm::raw_string_ostream &OS, FirstArgT &&First,
                         ArgsT &&...Args) {
  OS << std::forward<FirstArgT>(First);
  appendString(OS, std::forward<ArgsT>(Args)...);
}

template <class... Arguments>
inline std::string buildString(Arguments &&...Args) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  appendString(OS, std::forward<Arguments>(Args)...);
  return OS.str();
}

template <class MapType>
inline typename MapType::mapped_type
findObject(const MapType &Map, const typename MapType::key_type &Key) {
  auto Itr = Map.find(Key);
  if (Itr == Map.end())
    return typename MapType::mapped_type();
  return Itr->second;
}

template <class MapType,
          class ObjectType = typename MapType::mapped_type::element_type,
          class... Args>
inline typename MapType::mapped_type
insertObject(MapType &Map, const typename MapType::key_type &Key,
             Args &&...InitArgs) {
  auto &Obj = Map[Key];
  if (!Obj)
    Obj = std::make_shared<ObjectType>(Key, std::forward<Args>(InitArgs)...);
  return Obj;
}

void initHeaderSpellings();

enum UsingType {
  UT_Queue_P,
};

// clang-format off
//
//                                   DpctGlobalInfo
//                                         |
//              --------------------------------------------------------
//              |                          |                           |
//         DpctFileInfo               DpctFileInfo                ... (other info)
//                                         |
//             ------------------------------------------------------------------------------------
//             |                           |                          |                           |
//    MemVarInfo                   DeviceFunctionDecl           KernelCallExpr             CudaMallocInfo
//  (Global Variable)                      |            (inherit from CallFunctionExpr)
//                                 DeviceFunctionInfo
//                                         |
//                           ----------------------------
//                           |                          |
//                    CallFunctionExpr              MemVarInfo
//                 (Call Expr in Function)    (Defined in Function)
//                           |
//                  DeviceFunctionInfo
//                     (Callee Info)
//
// clang-format on

// Store analysis info (eg. memory variable info, kernel function info,
// replacements and so on) of each file
class DpctFileInfo {
public:
  DpctFileInfo(const clang::tooling::UnifiedPath &FilePathIn)
      : ReplsSYCL(std::make_shared<ExtReplacements>(FilePathIn)),
        ReplsCUDA(std::make_shared<ExtReplacements>(FilePathIn)),
        FilePath(FilePathIn) {
    buildLinesInfo();
  }
  template <class Obj> std::shared_ptr<Obj> findNode(unsigned Offset) {
    return findObject(getMap<Obj>(), Offset);
  }
  template <class Obj, class Node>
  std::shared_ptr<Obj> insertNode(unsigned Offset, const Node *N) {
    return insertObject(getMap<Obj>(), Offset, FilePath, N);
  }
  template <class Obj, class MappedT, class... Args>
  std::shared_ptr<MappedT> insertNode(unsigned Offset, Args &&...Arguments) {
    return insertObject<GlobalMap<MappedT>, Obj>(
        getMap<MappedT>(), Offset, FilePath, std::forward<Args>(Arguments)...);
  }
  template <class Obj>
  std::shared_ptr<Obj> insertNode(unsigned Offset,
                                  std::shared_ptr<Obj> Object) {
    return getMap<Obj>().insert(std::make_pair(Offset, Object)).first->second;
  }
  const clang::tooling::UnifiedPath &getFilePath() { return FilePath; }

  // Build kernel and device function declaration replacements and store them.
  void buildReplacements();
  void setKernelCallDim();
  void setKernelDim();
  void buildUnionFindSet();
  void buildUnionFindSetForUncalledFunc();
  void buildKernelInfo();
  void buildRnnBackwardFuncInfo();
  void postProcess();

  // Emplace stored replacements into replacement set.
  void emplaceReplacements(std::map<clang::tooling::UnifiedPath,
                                    tooling::Replacements> &ReplSet /*out*/);
  void addReplacement(std::shared_ptr<ExtReplacement> Repl);
  bool isInAnalysisScope();
  std::shared_ptr<ExtReplacements> getReplsSYCL() { return ReplsSYCL; }
  std::shared_ptr<ExtReplacements> getReplsCUDA() { return ReplsCUDA; }
  size_t getFileSize() const { return FileSize; }
  std::string &getFileContent() { return FileContentCache; }

  // Header inclusion directive insertion functions
  void setFileEnterOffset(unsigned Offset);
  void setFirstIncludeOffset(unsigned Offset);
  void setLastIncludeOffset(unsigned Offset) { LastIncludeOffset = Offset; }
  void setHeaderInserted(HeaderType Header) {
    HeaderInsertedBitMap[Header] = true;
  }
  void setMathHeaderInserted(bool B = true) {
    HeaderInsertedBitMap[HeaderType::HT_Math] = B;
  }
  void setAlgorithmHeaderInserted(bool B = true) {
    HeaderInsertedBitMap[HeaderType::HT_Algorithm] = B;
  }
  void setTimeHeaderInserted(bool B = true) {
    HeaderInsertedBitMap[HeaderType::HT_Time] = B;
  }

  void concatHeader(llvm::raw_string_ostream &OS);
  template <class FirstT, class... Args>
  void concatHeader(llvm::raw_string_ostream &OS, FirstT &&First,
                    Args &&...Arguments);

  std::optional<HeaderType> findHeaderType(StringRef Header);
  StringRef getHeaderSpelling(HeaderType Type);

  // Insert one or more header inclusion directives at a specified offset
  template <typename ReplacementT>
  void insertHeader(ReplacementT &&Repl, unsigned Offset,
                    InsertPosition InsertPos = IP_Left,
                    ReplacementType IsForCodePin = RT_ForSYCLMigration) {
    auto R = std::make_shared<ExtReplacement>(
        FilePath, Offset, 0, std::forward<ReplacementT>(Repl), nullptr);
    R->setSYCLHeaderNeeded(false);
    R->setInsertPosition(InsertPos);
    R->IsForCodePin = IsForCodePin;
    IncludeDirectiveInsertions.push_back(R);
  }

  template <typename ReplacementT>
  void insertCustomizedHeader(ReplacementT &&Repl) {
    if (auto Type = findHeaderType(Repl))
      return insertHeader(Type.value());
    if (std::find(InsertedHeaders.begin(), InsertedHeaders.end(), Repl) ==
        InsertedHeaders.end()) {
      InsertedHeaders.push_back(Repl);
    }
  }

  void insertHeader(HeaderType Type, unsigned Offset,
                    ReplacementType IsForCodePin = RT_ForSYCLMigration);
  void insertHeader(HeaderType Type,
                    ReplacementType IsForCodePin = RT_ForSYCLMigration);

  // Record line info in file.
  struct SourceLineInfo {
    SourceLineInfo() : SourceLineInfo(-1, -1, -1, StringRef()) {}
    SourceLineInfo(unsigned LineNumber, unsigned Offset, unsigned End,
                   StringRef Buffer)
        : Number(LineNumber), Offset(Offset), Length(End - Offset),
          Line(Buffer.substr(Offset, Length)) {}
    SourceLineInfo(unsigned LineNumber, ArrayRef<unsigned> LineCache,
                   StringRef Buffer)
        : SourceLineInfo(LineNumber, LineCache[LineNumber - 1],
                         LineCache[LineNumber], Buffer) {}

    // Line number.
    const unsigned Number;
    // Offset at the begin of line.
    const unsigned Offset;
    // Length of the line.
    const unsigned Length;
    // String of the line, ref to FileContentCache.
    StringRef Line;
  };

  const SourceLineInfo &getLineInfo(unsigned LineNumber);
  StringRef getLineString(unsigned LineNumber) {
    return getLineInfo(LineNumber).Line;
  }

  // Get line number by offset
  unsigned getLineNumber(unsigned Offset) {
    return getLineInfoFromOffset(Offset).Number;
  }
  // Set line range info of replacement
  void setLineRange(ExtReplacements::SourceLineRange &LineRange,
                    std::shared_ptr<ExtReplacement> Repl);
  void insertIncludedFilesInfo(std::shared_ptr<DpctFileInfo> Info);

  std::map<const CompoundStmt *, MemcpyOrderAnalysisInfo> &
  getMemcpyOrderAnalysisResultMap() {
    return MemcpyOrderAnalysisResultMap;
  }
  std::map<std::string, std::vector<std::pair<unsigned int, unsigned int>>> &
  getFuncDeclRangeMap() {
    return FuncDeclRangeMap;
  }
  std::map<unsigned int, EventSyncTypeInfo> &getEventSyncTypeMap() {
    return EventSyncTypeMap;
  }
  std::map<unsigned int, TimeStubTypeInfo> &getTimeStubTypeMap() {
    return TimeStubTypeMap;
  }
  std::map<unsigned int, BuiltinVarInfo> &getBuiltinVarInfoMap() {
    return BuiltinVarInfoMap;
  }
  std::unordered_set<std::shared_ptr<DpctFileInfo>> &getIncludedFilesInfoSet() {
    return IncludedFilesInfoSet;
  }
  std::set<unsigned int> &getSpBLASSet() { return SpBLASSet; }
  std::unordered_set<std::shared_ptr<TextModification>> &
  getConstantMacroTMSet() {
    return ConstantMacroTMSet;
  }
  std::vector<tooling::Replacement> &getReplacements() {
    return PreviousTUReplFromYAML->Replacements;
  }
  std::unordered_map<std::string, std::tuple<unsigned int, std::string, bool>> &
  getAtomicMap() {
    return AtomicMap;
  }
  void setAddOneDplHeaders(bool Value) { AddOneDplHeaders = Value; }
  std::vector<std::pair<unsigned int, unsigned int>> &getTimeStubBounds() {
    return TimeStubBounds;
  }
  std::vector<std::pair<unsigned int, unsigned int>> &getExternCRanges() {
    return ExternCRanges;
  }
  std::vector<RnnBackwardFuncInfo> &getRnnBackwardFuncInfo() {
    return RBFuncInfo;
  }
  void setRTVersionValue(std::string Value) { RTVersionValue = Value; }
  std::string getRTVersionValue() { return RTVersionValue; }
  void setMajorVersionValue(std::string Value) { MajorVersionValue = Value; }
  std::string getMajorVersionValue() { return MajorVersionValue; }
  void setMinorVersionValue(std::string Value) { MinorVersionValue = Value; }
  std::string getMinorVersionValue() { return MinorVersionValue; }

  void setCCLVerValue(std::string Value) { CCLVerValue = Value; }
  std::string getCCLVerValue() { return CCLVerValue; }
  bool hasCUDASyntax() { return HeaderInsertedBitMap[HeaderType::HT_SYCL]; }

  std::shared_ptr<tooling::TranslationUnitReplacements> PreviousTUReplFromYAML =
      nullptr;

private:
  std::vector<std::pair<unsigned int, unsigned int>> TimeStubBounds;
  std::unordered_set<std::shared_ptr<DpctFileInfo>> IncludedFilesInfoSet;

  template <class Obj> GlobalMap<Obj> &getMap() {
    llvm::dbgs() << "[DpctFileInfo::getMap] Unknow map type";
    static GlobalMap<Obj> NullMap;
    return NullMap;
  }

  bool isReplTxtWithSubmitBarrier(unsigned Offset);

  // TODO: implement one of this for each source language.
  bool isInCudaPath();

  void buildLinesInfo();
  const SourceLineInfo &getLineInfoFromOffset(unsigned Offset);

  std::map<const CompoundStmt *, MemcpyOrderAnalysisInfo>
      MemcpyOrderAnalysisResultMap;

  std::map<std::string /*Function name*/,
           std::vector<
               std::pair<unsigned int /*Begin location of function signature*/,
                         unsigned int /*End location of function signature*/>>>
      FuncDeclRangeMap;

  std::map<unsigned int, EventSyncTypeInfo> EventSyncTypeMap;
  std::map<unsigned int, TimeStubTypeInfo> TimeStubTypeMap;
  std::map<unsigned int, BuiltinVarInfo> BuiltinVarInfoMap;
  GlobalMap<MemVarInfo> MemVarMap;
  GlobalMap<DeviceFunctionDecl> FuncMap;
  GlobalMap<KernelCallExpr> KernelMap;
  GlobalMap<CudaMallocInfo> CudaMallocMap;
  GlobalMap<TextureInfo> TextureMap;
  std::set<unsigned int> SpBLASSet;
  std::unordered_set<std::shared_ptr<TextModification>> ConstantMacroTMSet;
  std::unordered_map<std::string, std::tuple<unsigned int, std::string, bool>>
      AtomicMap;
  std::shared_ptr<ExtReplacements> ReplsSYCL;
  std::shared_ptr<ExtReplacements> ReplsCUDA;
  size_t FileSize = 0;
  std::vector<SourceLineInfo> Lines;

  clang::tooling::UnifiedPath FilePath;
  std::string FileContentCache;

  // Save the FirstIncludeOffset in each MainFile
  std::map<std::shared_ptr<DpctFileInfo> /*MainFile*/, unsigned>
      FirstIncludeOffset;
  unsigned LastIncludeOffset = 0;
  const unsigned FileBeginOffset = 0;
  // Save the status whether FirstIncludeOffset is set by setFirstIncludeOffset
  // for each MainFile
  std::set<std::shared_ptr<DpctFileInfo> /*MainFile*/> HasInclusionDirectiveSet;
  std::vector<std::string> InsertedHeaders;
  std::vector<std::string> InsertedHeadersCUDA;
  std::bitset<32> HeaderInsertedBitMap;
  std::bitset<32> UsingInsertedBitMap;
  bool AddOneDplHeaders = false;
  std::vector<std::shared_ptr<ExtReplacement>> IncludeDirectiveInsertions;
  std::vector<std::pair<unsigned int, unsigned int>> ExternCRanges;
  std::vector<RnnBackwardFuncInfo> RBFuncInfo;
  std::string RTVersionValue = "";
  std::string MajorVersionValue = "";
  std::string MinorVersionValue = "";
  std::string CCLVerValue = "";
};
template <> inline GlobalMap<MemVarInfo> &DpctFileInfo::getMap() {
  return MemVarMap;
}
template <> inline GlobalMap<DeviceFunctionDecl> &DpctFileInfo::getMap() {
  return FuncMap;
}
template <> inline GlobalMap<KernelCallExpr> &DpctFileInfo::getMap() {
  return KernelMap;
}
template <> inline GlobalMap<CudaMallocInfo> &DpctFileInfo::getMap() {
  return CudaMallocMap;
}
template <> inline GlobalMap<TextureInfo> &DpctFileInfo::getMap() {
  return TextureMap;
}

class DpctGlobalInfo {
public:
  static DpctGlobalInfo &getInstance() {
    static DpctGlobalInfo Info;
    return Info;
  }

  class MacroDefRecord {
  public:
    clang::tooling::UnifiedPath FilePath;
    unsigned Offset;
    bool IsInAnalysisScope;
    MacroDefRecord(SourceLocation NTL, bool IIAS);
  };
  // This class is used to store information about macro arguments in a
  // macro definition. For example, consider the macro definition:
  // "#define CALL(x, y) x(y)".
  // - For the first argument "x", the member ArgName will be "x", ArgLoc will
  // be the source location of the token "x" in the macro definition, and
  // ArgIndex will be 0.
  // - For the second argument "y", the member ArgName will be "y", ArgLoc will
  // be the source location of the token "y" in the macro definition, and
  // ArgIndex will be 1.
  class MacroArgRecord {
  public:
    std::string ArgName;
    SourceLocation ArgLoc;
    int ArgIndex;
    MacroArgRecord(const MacroInfo *MI, int ArgIndex);
  };

  class MacroExpansionRecord {
  public:
    std::string Name;
    int NumTokens;
    clang::tooling::UnifiedPath FilePath;
    unsigned ReplaceTokenBeginOffset;
    unsigned ReplaceTokenEndOffset;
    SourceRange Range;
    bool IsInAnalysisScope;
    bool IsFunctionLike;
    int TokenIndex;
    MacroExpansionRecord(IdentifierInfo *ID, const MacroInfo *MI,
                         SourceRange Range, bool IsInAnalysisScope,
                         int TokenIndex);
  };

  struct HelperFuncReplInfo {
    HelperFuncReplInfo(const clang::tooling::UnifiedPath DeclLocFile =
                           clang::tooling::UnifiedPath(),
                       unsigned int DeclLocOffset = 0,
                       bool IsLocationValid = false)
        : DeclLocFile(DeclLocFile), DeclLocOffset(DeclLocOffset),
          IsLocationValid(IsLocationValid) {}
    clang::tooling::UnifiedPath DeclLocFile;
    unsigned int DeclLocOffset = 0;
    bool IsLocationValid = false;
  };

  struct TempVariableDeclCounter {
    TempVariableDeclCounter(int DefaultQueueCounter = 0,
                            int CurrentDeviceCounter = 0)
        : DefaultQueueCounter(DefaultQueueCounter),
          CurrentDeviceCounter(CurrentDeviceCounter),
          PlaceholderStr{
              "", DpctGlobalInfo::getDefaultQueueFreeFuncCall(),
              MapNames::getDpctNamespace() + "get_current_device()",
              (DpctGlobalInfo::useSYCLCompat()
                   ? buildString(MapNames::getDpctNamespace() +
                                 "get_current_device().default_queue()")
                   : buildString(
                         "&", DpctGlobalInfo::getDefaultQueueFreeFuncCall()))} {
    }
    int DefaultQueueCounter = 0;
    int CurrentDeviceCounter = 0;
    std::string PlaceholderStr[4];
  };

  static std::string removeSymlinks(clang::FileManager &FM,
                                    std::string FilePathStr);
  static bool isInRoot(SourceLocation SL) {
    return isInRoot(DpctGlobalInfo::getLocInfo(SL).first);
  }
  static bool isInRoot(const clang::tooling::UnifiedPath &FilePath);
  static bool isInAnalysisScope(SourceLocation SL) {
    return isInAnalysisScope(DpctGlobalInfo::getLocInfo(SL).first);
  }
  static bool isInAnalysisScope(const clang::tooling::UnifiedPath &FilePath) {
    return std::any_of(AnalysisScope.begin(), AnalysisScope.end(),
                       [&](const clang::tooling::UnifiedPath &P) {
                         return isChildPath(P, FilePath);
                       });
  }
  static bool isExcluded(const clang::tooling::UnifiedPath &FilePath);
  // TODO: implement one of this for each source language.
  static bool isInCudaPath(SourceLocation SL);
  // TODO: implement one of this for each source language.
  static bool isInCudaPath(clang::tooling::UnifiedPath FilePath) {
    return isChildPath(CudaPath, FilePath);
  }

  static void setInRoot(const clang::tooling::UnifiedPath &InRootPath) {
    InRoot = InRootPath;
  }
  static const clang::tooling::UnifiedPath &getInRoot() { return InRoot; }
  static void setOutRoot(const clang::tooling::UnifiedPath &OutRootPath) {
    OutRoot = OutRootPath;
  }
  static const clang::tooling::UnifiedPath &getOutRoot() { return OutRoot; }
  static void setAnalysisScope(
      const std::vector<clang::tooling::UnifiedPath> &InputAnalysisScope) {
    AnalysisScope = InputAnalysisScope;
  }
  static const std::vector<clang::tooling::UnifiedPath> &getAnalysisScope() {
    return AnalysisScope;
  }
  static void addChangeExtensions(const std::string &Extension) {
    assert(!Extension.empty());
    ChangeExtensions.insert(Extension);
  }
  static const std::unordered_set<std::string> &getChangeExtensions() {
    return ChangeExtensions;
  }
  static const std::string &getSYCLSourceExtension() {
    return SYCLSourceExtension;
  }
  static const std::string &getSYCLHeaderExtension() {
    return SYCLHeaderExtension;
  }
  static void setSYCLFileExtension(SYCLFileExtensionEnum Extension);
  // TODO: implement one of this for each source language.
  static void setCudaPath(const clang::tooling::UnifiedPath &InputCudaPath) {
    CudaPath = InputCudaPath;
  }
  // TODO: implement one of this for each source language.
  static const clang::tooling::UnifiedPath &getCudaPath() { return CudaPath; }
  static const std::string getCudaVersion() {
    return clang::CudaVersionToString(SDKVersion);
  }

  static void printItem(llvm::raw_ostream &, const Stmt *,
                        const FunctionDecl *FD = nullptr);
  static std::string getItem(const Stmt *, const FunctionDecl *FD = nullptr);
  static void registerNDItemUser(const Stmt *,
                                 const FunctionDecl *FD = nullptr);
  static void printGroup(llvm::raw_ostream &, const Stmt *,
                         const FunctionDecl *FD = nullptr);
  static std::string getGroup(const Stmt *, const FunctionDecl *FD = nullptr);
  static void printSubGroup(llvm::raw_ostream &, const Stmt *,
                            const FunctionDecl *FD = nullptr);
  static std::string getSubGroup(const Stmt *,
                                 const FunctionDecl *FD = nullptr);
  static std::string getDefaultQueue(const Stmt *);
  static const std::string &getDefaultQueueFreeFuncCall();
  static const std::string &getDefaultQueueMemFuncName();
  static const std::string &getStreamName() {
    const static std::string StreamName = "stream" + getCTFixedSuffix();
    return StreamName;
  }
  static const std::string &getSyncName() {
    const static std::string SyncName = "sync" + getCTFixedSuffix();
    return SyncName;
  }
  static const std::string &getInRootHash() {
    const static std::string Hash = getHashAsString(getInRoot()).substr(0, 6);
    return Hash;
  }
  static void setContext(ASTContext &C);
  static void setRuleFile(const std::string &Path) { RuleFile = Path; }
  static ASTContext &getContext() {
    assert(Context);
    return *Context;
  }
  static SourceManager &getSourceManager() {
    assert(SM);
    return *SM;
  }
  static FileManager &getFileManager() {
    assert(FM);
    return *FM;
  }
  static bool isKeepOriginCode() { return KeepOriginCode; }
  static void setKeepOriginCode(bool KOC) { KeepOriginCode = KOC; }
  static bool isSyclNamedLambda() { return SyclNamedLambda; }
  static void setSyclNamedLambda(bool SNL) { SyclNamedLambda = SNL; }
  static void setCheckUnicodeSecurityFlag(bool CUS) {
    CheckUnicodeSecurityFlag = CUS;
  }
  static bool getCheckUnicodeSecurityFlag() { return CheckUnicodeSecurityFlag; }
  static void setEnablepProfilingFlag(bool EP) { EnablepProfilingFlag = EP; }
  static bool getEnablepProfilingFlag() { return EnablepProfilingFlag; }
  static bool getGuessIndentWidthMatcherFlag() {
    return GuessIndentWidthMatcherFlag;
  }
  static void setGuessIndentWidthMatcherFlag(bool Flag) {
    GuessIndentWidthMatcherFlag = Flag;
  }
  static void setIndentWidth(unsigned int W) { IndentWidth = W; }
  static unsigned int getIndentWidth() { return IndentWidth; }
  static void insertKCIndentWidth(unsigned int W);
  static unsigned int getKCIndentWidth();
  static UsmLevel getUsmLevel() { return UsmLvl; }
  static void setUsmLevel(UsmLevel UL) { UsmLvl = UL; }
  static unsigned getBuildScript() { return BuildScriptType; }
  static void setBuildScript(unsigned BS_type) { BuildScriptType = BS_type; }
  template <BuildScriptKind BS_kind> static bool getMigrateBuildScriptType() {
    return BuildScriptType & (1 << static_cast<unsigned>(BS_kind));
  }
  static bool migrateCMakeScripts() {
    return getMigrateBuildScriptType<BuildScriptKind::BS_CMake>();
  }
  static bool migratePythonScripts() {
    return getMigrateBuildScriptType<BuildScriptKind::BS_Python>();
  }
  static clang::CudaVersion getSDKVersion() { return SDKVersion; }
  static void setSDKVersion(clang::CudaVersion V) { SDKVersion = V; }
  static bool isIncMigration() { return IsIncMigration; }
  static void setIsIncMigration(bool Flag) { IsIncMigration = Flag; }
  static bool isQueryAPIMapping() { return IsQueryAPIMapping; }
  static void setIsQueryAPIMapping(bool Flag) { IsQueryAPIMapping = Flag; }
  static bool needDpctDeviceExt() { return NeedDpctDeviceExt; }
  static void setNeedDpctDeviceExt() { NeedDpctDeviceExt = true; }
  static unsigned int getAssumedNDRangeDim() { return AssumedNDRangeDim; }
  static void setAssumedNDRangeDim(unsigned int Dim) {
    AssumedNDRangeDim = Dim;
  }
  static bool getUsingExtensionDE(DPCPPExtensionsDefaultEnabled Ext) {
    return ExtensionDEFlag & (1 << static_cast<unsigned>(Ext));
  }
  static void setExtensionDEFlag(unsigned Flag) {
    // The bits in Flag was reversed, so we need to check whether the ExtDE_All
    // bit of Flag is 0. That means disable all default enabled extensions,
    // otherwise disable the extensions represented by the 0 bit
    if (Flag &
        (1 << static_cast<unsigned>(DPCPPExtensionsDefaultEnabled::ExtDE_All)))
      ExtensionDEFlag = Flag;
    else
      ExtensionDEFlag = 0;
  }
  static unsigned getExtensionDEFlag() { return ExtensionDEFlag; }
  static bool getUsingExtensionDD(DPCPPExtensionsDefaultDisabled Ext) {
    return ExtensionDDFlag & (1 << static_cast<unsigned>(Ext));
  }
  static void setExtensionDDFlag(unsigned Flag) {
    // If the ExtDD_All bit is 1, enable all default disabled extensions.
    if (Flag &
        (1 << static_cast<unsigned>(DPCPPExtensionsDefaultDisabled::ExtDD_All)))
      ExtensionDDFlag = static_cast<unsigned>(-1);
    else
      ExtensionDDFlag = Flag;
  }
  static unsigned getExtensionDDFlag() { return ExtensionDDFlag; }
  template <ExperimentalFeatures Exp> static bool getUsingExperimental() {
    return ExperimentalFlag & (1 << static_cast<unsigned>(Exp));
  }
  static void setExperimentalFlag(unsigned Flag) {
    // If the ExtDD_All bit is 1, enable all default disabled experimental
    // features.
    if (Flag & (1 << static_cast<unsigned>(ExperimentalFeatures::Exp_All)))
      ExperimentalFlag = static_cast<unsigned>(-1);
    else
      ExperimentalFlag = Flag;
  }
  static unsigned getExperimentalFlag() { return ExperimentalFlag; }
  static bool getHelperFuncPreference(HelperFuncPreference HFP) {
    return HelperFuncPreferenceFlag & (1 << static_cast<unsigned>(HFP));
  }
  static void setHelperFuncPreferenceFlag(unsigned Flag) {
    HelperFuncPreferenceFlag = Flag;
  }
  static unsigned getHelperFuncPreferenceFlag() {
    return HelperFuncPreferenceFlag;
  }
  static bool isAnalysisModeEnabled() { return AnalysisModeFlag; }
  static void enableAnalysisMode() { AnalysisModeFlag = true; }
  static format::FormatRange getFormatRange() { return FmtRng; }
  static void setFormatRange(format::FormatRange FR) { FmtRng = FR; }
  static DPCTFormatStyle getFormatStyle() { return FmtST; }
  static void setFormatStyle(DPCTFormatStyle FS) { FmtST = FS; }
  // Processing the folder or file by following rules:
  // Rule1: For {child path, parent path}, only parent path will be kept.
  // Rule2: Ignore invalid path.
  // Rule3: If path is not in --in-root, then ignore it.
  static void setExcludePath(std::vector<std::string> ExcludePathVec);
  static std::unordered_map<std::string, bool> getExcludePath() {
    return ExcludePath;
  }
  static bool isCtadEnabled() { return EnableCtad; }
  static void setCtadEnabled(bool Enable) { EnableCtad = Enable; }
  static bool isCodePinEnabled() { return EnableCodePin; }
  static void setCodePinEnabled(bool Enable = false) { EnableCodePin = Enable; }
  static bool isGenBuildScript() { return GenBuildScript; }
  static void setGenBuildScriptEnabled(bool Enable = true) {
    GenBuildScript = Enable;
  }
  static bool IsMigrateBuildScriptOnlyEnabled() {
    return MigrateBuildScriptOnly;
  }
  static void setMigrateBuildScriptOnlyEnabled(bool Enable = true) {
    MigrateBuildScriptOnly = Enable;
  }
  static bool isCommentsEnabled() { return EnableComments; }
  static void setCommentsEnabled(bool Enable = true) {
    EnableComments = Enable;
  }
  static bool isDPCTNamespaceTempEnabled() { return TempEnableDPCTNamespace; }
  static void setDPCTNamespaceTempEnabled() { TempEnableDPCTNamespace = true; }
  static std::unordered_set<std::string> &getPrecAndDomPairSet() {
    return PrecAndDomPairSet;
  }
  static bool isMKLHeaderUsed() { return IsMLKHeaderUsed; }
  static void setMKLHeaderUsed(bool Used = true) { IsMLKHeaderUsed = Used; }
  static int getSuffixIndexInitValue(std::string FileNameAndOffset);
  static void updateInitSuffixIndexInRule(int InitVal) {
    CurrentIndexInRule = InitVal;
  }
  static int getSuffixIndexInRuleThenInc();
  static int getSuffixIndexGlobalThenInc();
  static const std::string &getGlobalQueueName() {
    const static std::string Q = "q_ct1";
    return Q;
  }
  static const std::string &getGlobalDeviceName() {
    const static std::string D = "dev_ct1";
    return D;
  }
  static std::string getStringForRegexReplacement(StringRef);
  static void setCodeFormatStyle(const clang::format::FormatStyle &Style) {
    CodeFormatStyle = Style;
  }
  static clang::format::FormatStyle getCodeFormatStyle() {
    return CodeFormatStyle;
  }
  static bool IsVarUsedByRuntimeSymbolAPI(std::shared_ptr<MemVarInfo> Info);

private:
  template <class T, class T1, class... Ts>
  static constexpr bool IsSameAsAnyTy = (std::is_same_v<T, Ts> || ...);

  template <typename T>
  static constexpr bool IsNonPtrNode =
      IsSameAsAnyTy<T, TemplateArgument, TemplateArgumentLoc,
                    NestedNameSpecifierLoc, QualType, TypeLoc, ObjCProtocolLoc>;

public:
  template <class TargetTy, class NodeTy>
  static inline std::conditional_t<IsNonPtrNode<TargetTy>,
                                   std::optional<TargetTy>, const TargetTy *>
  findAncestor(const NodeTy *N,
               const std::function<bool(const DynTypedNode &)> &Condition) {
    if (LLVM_LIKELY(N)) {
      auto &Context = getContext();
      clang::DynTypedNodeList Parents = Context.getParents(*N);
      while (!Parents.empty()) {
        auto &Cur = Parents[0];
        if (Condition(Cur)) {
          if constexpr (IsNonPtrNode<TargetTy>)
            return *Cur.get<TargetTy>();
          else
            return Cur.get<TargetTy>();
        }
        Parents = Context.getParents(Cur);
      }
    }
    if constexpr (IsNonPtrNode<TargetTy>)
      return std::nullopt;
    else
      return nullptr;
  }

  template <class NodeTy>
  static inline bool checkSpecificBO(const NodeTy *Node,
                                     const BinaryOperator *BO) {
    return findAncestor<BinaryOperator>(
        Node, [&](const DynTypedNode &Cur) -> bool {
          return Cur.get<BinaryOperator>() == BO;
        });
  }

  template <class TargetTy, class NodeTy>
  static auto findAncestor(const NodeTy *Node) {
    return findAncestor<TargetTy>(Node, [&](const DynTypedNode &Cur) -> bool {
      return Cur.get<TargetTy>();
    });
  }
  template <class TargetTy, class NodeTy, class... SkipNodeTy>
  static auto findParent(const NodeTy *Node) {
    return findAncestor<TargetTy>(Node, [](const DynTypedNode &Cur) -> bool {
      if ((... || Cur.get<SkipNodeTy>())) {
        return false;
      }
      return true;
    });
  }

  template <typename TargetTy, typename NodeTy>
  static bool isAncestor(TargetTy *AncestorNode, NodeTy *Node) {
    return findAncestor<TargetTy>(Node, [&](const DynTypedNode &Cur) -> bool {
      return Cur.get<TargetTy>() == AncestorNode;
    });
  }
  template <class NodeTy>
  inline static const clang::FunctionDecl *
  getParentFunction(const NodeTy *Node) {
    return findAncestor<clang::FunctionDecl>(Node);
  }
  template <class TargetTy, class NodeTy>
  static inline const clang::Expr *
  getChildExprOfTargetAncestor(const NodeTy *N) {
    if (!N)
      return nullptr;

    auto &Context = clang::dpct::DpctGlobalInfo::getContext();
    clang::DynTypedNode PreviousNode = clang::DynTypedNode::create(*N);
    clang::DynTypedNodeList Parents = Context.getParents(*N);
    while (!Parents.empty()) {
      auto &Cur = Parents[0];
      if (Cur.get<TargetTy>())
        return PreviousNode.get<clang::Expr>();
      PreviousNode = Cur;
      Parents = Context.getParents(Cur);
    }

    return nullptr;
  }

  template <class StreamTy, class... Args>
  static inline StreamTy &
  printCtadClass(StreamTy &Stream, size_t CanNotDeducedArgsNum,
                 StringRef ClassName, Args &&...Arguments) {
    Stream << ClassName;
    if (!DpctGlobalInfo::isCtadEnabled()) {
      printArguments(Stream << "<", std::forward<Args>(Arguments)...) << ">";
    } else if (CanNotDeducedArgsNum) {
      printPartialArguments(Stream << "<", CanNotDeducedArgsNum,
                            std::forward<Args>(Arguments)...)
          << ">";
    }
    return Stream;
  }
  template <class StreamTy, class... Args>
  static inline StreamTy &printCtadClass(StreamTy &Stream, StringRef ClassName,
                                         Args &&...Arguments) {
    return printCtadClass(Stream, 0, ClassName,
                          std::forward<Args>(Arguments)...);
  }
  template <class... Args>
  static inline std::string getCtadClass(Args &&...Arguments) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    return printCtadClass(OS, std::forward<Args>(Arguments)...).str();
  }
  template <class T>
  static inline std::pair<clang::tooling::UnifiedPath, unsigned>
  getLocInfo(const T *N, bool *IsInvalid = nullptr /* out */) {
    return getLocInfo(getLocation(N), IsInvalid);
  }
  static std::pair<clang::tooling::UnifiedPath, unsigned>
  getLocInfo(const TypeLoc &TL, bool *IsInvalid = nullptr /*out*/) {
    return getLocInfo(TL.getBeginLoc(), IsInvalid);
  }
  // Return the absolute path of \p ID
  static std::optional<clang::tooling::UnifiedPath> getAbsolutePath(FileID ID);
  // Return the absolute path of \p File
  static std::optional<clang::tooling::UnifiedPath>
  getAbsolutePath(FileEntryRef File);
  static std::pair<clang::tooling::UnifiedPath, unsigned>
  getLocInfo(SourceLocation Loc, bool *IsInvalid = nullptr /* out */);
  static std::string getTypeName(QualType QT, const ASTContext &Context,
                                 bool SuppressScope = false);
  static std::string getTypeName(QualType QT, bool SuppressScope = false) {
    return getTypeName(QT, DpctGlobalInfo::getContext(), SuppressScope);
  }
  static std::string getUnqualifiedTypeName(QualType QT,
                                            const ASTContext &Context) {
    return getTypeName(QT.getUnqualifiedType(), Context, false);
  }
  static std::string getUnqualifiedTypeName(QualType QT) {
    return getUnqualifiedTypeName(QT, DpctGlobalInfo::getContext());
  }
  static std::string getUnqualifiedAndUnScopeTypeName(QualType QT) {
    return getTypeName(QT.getUnqualifiedType(), DpctGlobalInfo::getContext(),
                       true);
  }
  /// This function will return the replaced type name with qualifiers.
  /// Currently, since clang do not support get the order of original
  /// qualifiers, this function will follow the behavior of
  /// clang::QualType.print(), in other words, the behavior is that the
  /// qualifiers(const, volatile...) will occur before the simple type(int,
  /// bool...) regardless its order in origin code.
  /// \param [in] QT The input qualified type which need migration.
  /// \param [in] Context The AST context.
  /// \param [in] SuppressScope Suppresses printing of scope specifiers.
  /// \return The replaced type name string with qualifiers.
  static std::string getReplacedTypeName(QualType QT, const ASTContext &Context,
                                         bool SuppressScope = false);
  static std::string getReplacedTypeName(QualType QT,
                                         bool SuppressScope = false) {
    return getReplacedTypeName(QT, DpctGlobalInfo::getContext(), SuppressScope);
  }
  /// This function will return the original type name with qualifiers.
  /// The order of original qualifiers will follow the behavior of
  /// clang::QualType.print() regardless its order in origin code.
  /// \param [in] QT The input qualified type.
  /// \return The type name string with qualifiers.
  static std::string getOriginalTypeName(QualType QT);

#define GLOBAL_TYPE(TYPE, NODE_TYPE)                                           \
  std::shared_ptr<TYPE> find##TYPE(const NODE_TYPE *Node) {                    \
    return findNode<TYPE>(Node);                                               \
  }                                                                            \
  std::shared_ptr<TYPE> insert##TYPE(const NODE_TYPE *Node) {                  \
    return insertNode<TYPE>(Node);                                             \
  }

  GLOBAL_TYPE(MemVarInfo, VarDecl)
  GLOBAL_TYPE(DeviceFunctionDecl, FunctionDecl)
  GLOBAL_TYPE(KernelCallExpr, CUDAKernelCallExpr)
  GLOBAL_TYPE(CudaMallocInfo, VarDecl)
  GLOBAL_TYPE(TextureInfo, VarDecl)
#undef GLOBAL_TYPE

  std::shared_ptr<DeviceFunctionDecl> insertDeviceFunctionDecl(
      const FunctionDecl *Specialization, const FunctionTypeLoc &FTL,
      const ParsedAttributes &Attrs, const TemplateArgumentListInfo &TAList);

  // Build kernel and device function declaration replacements and store
  // them.
  void buildKernelInfo();
  void buildReplacements();
  void processCudaArchMacro();
  void generateHostCode(tooling::Replacements &ProcessedReplList,
                        HostDeviceFuncLocInfo &Info, unsigned ID);
  void postProcess();
  void cacheFileRepl(clang::tooling::UnifiedPath FilePath,
                     std::pair<std::shared_ptr<ExtReplacements>,
                               std::shared_ptr<ExtReplacements>>
                         Repl) {
    FileReplCache[FilePath] = Repl;
  }
  // Emplace stored replacements into replacement set.
  void emplaceReplacements(ReplTy &ReplSetsCUDA /*out*/,
                           ReplTy &ReplSetsSYCL /*out*/);
  std::shared_ptr<KernelCallExpr>
  buildLaunchKernelInfo(const CallExpr *, bool IsAssigned = false);
  void insertCudaMalloc(const CallExpr *CE);
  void insertCublasAlloc(const CallExpr *CE);
  std::shared_ptr<CudaMallocInfo> findCudaMalloc(const Expr *CE);
  void addReplacement(std::shared_ptr<ExtReplacement> Repl) {
    insertFile(Repl->getFilePath().str())->addReplacement(Repl);
  }
  CudaArchPPMap &getCudaArchPPInfoMap() { return CAPPInfoMap; }
  HDFuncInfoMap &getHostDeviceFuncInfoMap() { return HostDeviceFuncInfoMap; }
  std::unordered_map<std::string, std::shared_ptr<ExtReplacement>> &
  getCudaArchMacroReplMap() {
    return CudaArchMacroRepl;
  }
  CudaArchDefMap &getCudaArchDefinedMap() { return CudaArchDefinedMap; }
  void insertReplInfoFromYAMLToFileInfo(
      const clang::tooling::UnifiedPath &FilePath,
      std::shared_ptr<tooling::TranslationUnitReplacements> TUR);
  std::shared_ptr<tooling::TranslationUnitReplacements>
  getReplInfoFromYAMLSavedInFileInfo(clang::tooling::UnifiedPath FilePath);
  void insertEventSyncTypeInfo(
      const std::shared_ptr<clang::dpct::ExtReplacement> Repl,
      bool NeedReport = false, bool IsAssigned = false);
  void updateEventSyncTypeInfo(
      const std::shared_ptr<clang::dpct::ExtReplacement> Repl);
  void insertTimeStubTypeInfo(
      const std::shared_ptr<clang::dpct::ExtReplacement> ReplWithSB,
      const std::shared_ptr<clang::dpct::ExtReplacement> ReplWithoutSB);
  void updateTimeStubTypeInfo(SourceLocation BeginLoc, SourceLocation EndLoc);
  void insertBuiltinVarInfo(SourceLocation SL, unsigned int Len,
                            std::string Repl,
                            std::shared_ptr<DeviceFunctionInfo> DFI);
  void insertSpBLASWarningLocOffset(SourceLocation SL);
  std::shared_ptr<TextModification> findConstantMacroTMInfo(SourceLocation SL);
  void insertConstantMacroTMInfo(SourceLocation SL,
                                 std::shared_ptr<TextModification> TM);
  void insertAtomicInfo(std::string HashStr, SourceLocation SL,
                        std::string FuncName);
  void removeAtomicInfo(std::string HashStr);
  void setFileEnterLocation(SourceLocation Loc);
  void setFirstIncludeLocation(SourceLocation Loc);
  void setLastIncludeLocation(SourceLocation Loc);
  void setMathHeaderInserted(SourceLocation Loc, bool B);
  void setAlgorithmHeaderInserted(SourceLocation Loc, bool B);
  void setTimeHeaderInserted(SourceLocation Loc, bool B);
  void insertHeader(SourceLocation Loc, HeaderType Type,
                    ReplacementType IsForCodePin = RT_ForSYCLMigration);
  void insertHeader(SourceLocation Loc, std::string HeaderName);
  static std::unordered_map<
      std::string,
      std::pair<std::pair<clang::tooling::UnifiedPath /*begin file name*/,
                          unsigned int /*begin offset*/>,
                std::pair<clang::tooling::UnifiedPath /*end file name*/,
                          unsigned int /*end offset*/>>> &
  getExpansionRangeBeginMap() {
    return ExpansionRangeBeginMap;
  }
  static std::unordered_map<std::string, std::shared_ptr<MacroArgRecord>> &
  getMacroArgRecordMap() {
    return MacroArgRecordMap;
  }
  static std::map<std::string, std::shared_ptr<MacroExpansionRecord>> &
  getExpansionRangeToMacroRecord() {
    return ExpansionRangeToMacroRecord;
  }
  static std::map<std::string,
                  std::shared_ptr<DpctGlobalInfo::MacroDefRecord>> &
  getMacroTokenToMacroDefineLoc() {
    return MacroTokenToMacroDefineLoc;
  }
  static std::map<std::string, std::string> &
  getFunctionCallInMacroMigrateRecord() {
    return FunctionCallInMacroMigrateRecord;
  }
  static std::map<std::string, SourceLocation> &getEndifLocationOfIfdef() {
    return EndifLocationOfIfdef;
  }
  static std::vector<std::pair<clang::tooling::UnifiedPath, size_t>> &
  getConditionalCompilationLoc() {
    return ConditionalCompilationLoc;
  }
  static std::map<std::string, unsigned int> &getBeginOfEmptyMacros() {
    return BeginOfEmptyMacros;
  }
  static std::map<std::string, SourceLocation> &getEndOfEmptyMacros() {
    return EndOfEmptyMacros;
  }
  static std::map<std::string, bool> &getMacroDefines() { return MacroDefines; }
  static std::set<clang::tooling::UnifiedPath> &getIncludingFileSet() {
    return IncludingFileSet;
  }
  static std::set<std::string> &getFileSetInCompilationDB() {
    return FileSetInCompilationDB;
  }
  static std::unordered_map<std::string,
                            std::vector<clang::tooling::Replacement>> &
  getFileRelpsMap() {
    return FileRelpsMap;
  }
  static std::unordered_map<std::string, clang::tooling::MainSourceFileInfo> &
  getMsfInfoMap() {
    return MsfInfoMap;
  }
  static std::string getYamlFileName() { return YamlFileName; }
  static std::set<std::string> &getGlobalVarNameSet() {
    return GlobalVarNameSet;
  }
  static void removeVarNameInGlobalVarNameSet(const std::string &VarName);
  static bool getDeviceChangedFlag() { return HasFoundDeviceChanged; }
  static void setDeviceChangedFlag(bool Flag) { HasFoundDeviceChanged = Flag; }
  static std::unordered_map<int, HelperFuncReplInfo> &
  getHelperFuncReplInfoMap() {
    return HelperFuncReplInfoMap;
  }
  static int getHelperFuncReplInfoIndexThenInc();
  static std::unordered_map<std::string, TempVariableDeclCounter> &
  getTempVariableDeclCounterMap() {
    return TempVariableDeclCounterMap;
  }
  // Key: string: file:offset for a replacement.
  // Value: int: index of the placeholder in a replacement.
  static std::unordered_map<std::string, int> &getTempVariableHandledMap() {
    return TempVariableHandledMap;
  }
  static bool getUsingDRYPattern() { return UsingDRYPattern; }
  static void setUsingDRYPattern(bool Flag) { UsingDRYPattern = Flag; }
  static bool useNdRangeBarrier() {
    return getUsingExperimental<ExperimentalFeatures::Exp_NdRangeBarrier>();
  }
  static bool useRootGroup() {
    return getUsingExperimental<ExperimentalFeatures::Exp_RootGroup>();
  }
  static bool useFreeQueries() {
    return getUsingExperimental<ExperimentalFeatures::Exp_FreeQueries>();
  }
  static bool useGroupLocalMemory() {
    return getUsingExperimental<ExperimentalFeatures::Exp_GroupSharedMemory>();
  }
  static bool useLogicalGroup() {
    return getUsingExperimental<ExperimentalFeatures::Exp_LogicalGroup>();
  }
  static bool useUserDefineReductions() {
    return getUsingExperimental<
        ExperimentalFeatures::Exp_UserDefineReductions>();
  }
  static bool useMaskedSubGroupFunction() {
    return getUsingExperimental<
        ExperimentalFeatures::Exp_MaskedSubGroupFunction>();
  }
  static bool useExtDPLAPI() {
    return getUsingExperimental<ExperimentalFeatures::Exp_DPLExperimentalAPI>();
  }
  static bool useOccupancyCalculation() {
    return getUsingExperimental<
        ExperimentalFeatures::Exp_OccupancyCalculation>();
  }
  static bool useExtJointMatrix() {
    return getUsingExperimental<ExperimentalFeatures::Exp_Matrix>();
  }
  static bool useExtBFloat16Math() {
    return getUsingExperimental<ExperimentalFeatures::Exp_BFloat16Math>();
  }
  static bool useExtBindlessImages() {
    return getUsingExperimental<ExperimentalFeatures::Exp_BindlessImages>();
  }
  static bool useExtGraph() {
    return getUsingExperimental<ExperimentalFeatures::Exp_Graph>();
  }
  static bool useExpNonUniformGroups() {
    return getUsingExperimental<ExperimentalFeatures::Exp_NonUniformGroups>();
  }
  static bool useExpDeviceGlobal() {
    return getUsingExperimental<ExperimentalFeatures::Exp_DeviceGlobal>();
  }
  static bool useExpVirtualMemory() {
    return getUsingExperimental<ExperimentalFeatures::Exp_VirtualMemory>();
  }
  static bool useExpInOrderQueueEvents() {
    return getUsingExperimental<ExperimentalFeatures::Exp_InOrderQueueEvents>();
  }
  static bool useExpNonStandardSYCLBuiltins() {
    return getUsingExperimental<
        ExperimentalFeatures::Exp_NonStandardSYCLBuiltins>();
  }
  static bool useExtPrefetch() {
    return getUsingExperimental<ExperimentalFeatures::Exp_Prefetch>();
  }
  static bool useNoQueueDevice() {
    return getHelperFuncPreference(HelperFuncPreference::NoQueueDevice);
  }
  static void setCVersionCUDALaunchUsed() { CVersionCUDALaunchUsedFlag = true; }
  static bool isCVersionCUDALaunchUsed() { return CVersionCUDALaunchUsedFlag; }
  static void setUseSYCLCompat(bool Flag = true) { UseSYCLCompatFlag = Flag; }
  static bool useSYCLCompat() { return UseSYCLCompatFlag; }
  static bool useEnqueueBarrier() {
    return getUsingExtensionDE(
        DPCPPExtensionsDefaultEnabled::ExtDE_EnqueueBarrier);
  }
  static bool useQueueEmpty() {
    return getUsingExtensionDE(DPCPPExtensionsDefaultEnabled::ExtDE_QueueEmpty);
  }
  static bool useCAndCXXStandardLibrariesExt() {
    return getUsingExtensionDD(
        DPCPPExtensionsDefaultDisabled::ExtDD_CCXXStandardLibrary);
  }
  static bool useIntelDeviceMath() {
    return getUsingExtensionDD(
        DPCPPExtensionsDefaultDisabled::ExtDD_IntelDeviceMath);
  }
  static bool usePeerAccess() {
    return getUsingExtensionDE(DPCPPExtensionsDefaultEnabled::ExtDE_PeerAccess);
  }
  static bool useAssert() {
    return getUsingExtensionDE(DPCPPExtensionsDefaultEnabled::ExtDE_Assert);
  }
  static bool useDeviceInfo() {
    return getUsingExtensionDE(DPCPPExtensionsDefaultEnabled::ExtDE_DeviceInfo);
  }
  static bool useBFloat16() {
    return getUsingExtensionDE(DPCPPExtensionsDefaultEnabled::ExtDE_BFloat16);
  }
  static std::unordered_set<std::string> &
  getCustomHelperFunctionAddtionalIncludes() {
    return CustomHelperFunctionAddtionalIncludes;
  }
  std::shared_ptr<DpctFileInfo>
  insertFile(const clang::tooling::UnifiedPath &FilePath) {
    return insertObject(FileMap, FilePath);
  }
  std::shared_ptr<DpctFileInfo>
  findFile(const clang::tooling::UnifiedPath &FilePath) {
    return findObject(FileMap, FilePath);
  }
  std::shared_ptr<DpctFileInfo> getMainFile() const { return MainFile; }
  void setMainFile(std::shared_ptr<DpctFileInfo> Main) { MainFile = Main; }
  void recordIncludingRelationship(
      const clang::tooling::UnifiedPath &CurrentFileName,
      const clang::tooling::UnifiedPath &IncludedFileName);
  static unsigned int getCudaKernelDimDFIIndexThenInc();
  static void
  insertCudaKernelDimDFIMap(unsigned int Index,
                            std::shared_ptr<DeviceFunctionInfo> Ptr);
  static std::shared_ptr<DeviceFunctionInfo>
  getCudaKernelDimDFI(unsigned int Index);
  static std::set<clang::tooling::UnifiedPath> &getModuleFiles() {
    return ModuleFiles;
  }
  static void setRunRound(unsigned int Round) { RunRound = Round; }
  static unsigned int getRunRound() { return RunRound; }
  static void setNeedRunAgain(bool NRA) { NeedRunAgain = NRA; }
  static bool isNeedRunAgain() { return NeedRunAgain; }
  static std::unordered_map<clang::tooling::UnifiedPath,
                            std::pair<std::shared_ptr<ExtReplacements>,
                                      std::shared_ptr<ExtReplacements>>> &
  getFileReplCache() {
    return FileReplCache;
  }
  void resetInfo();
  static void updateSpellingLocDFIMaps(SourceLocation SL,
                                       std::shared_ptr<DeviceFunctionInfo> DFI);
  static std::unordered_set<std::shared_ptr<DeviceFunctionInfo>>
  getDFIVecRelatedFromSpellingLoc(std::shared_ptr<DeviceFunctionInfo> DFI);
  static unsigned int getColorOption() { return ColorOption; }
  static void setColorOption(unsigned Color) { ColorOption = Color; }
  std::unordered_map<int, std::shared_ptr<DeviceFunctionInfo>> &
  getCubPlaceholderIndexMap() {
    return CubPlaceholderIndexMap;
  }
  std::vector<std::shared_ptr<DpctFileInfo>> &getCSourceFileInfo() {
    return CSourceFileInfo;
  }
  static std::unordered_map<std::string, std::shared_ptr<PriorityReplInfo>> &
  getPriorityReplInfoMap() {
    return PriorityReplInfoMap;
  }
  // For PriorityRelpInfo with same key, the Info with low priority will
  // be filtered and the Info with same priority will be merged.
  static void addPriorityReplInfo(std::string Key,
                                  std::shared_ptr<PriorityReplInfo> Info);
  static void setOptimizeMigrationFlag(bool Flag) {
    OptimizeMigrationFlag = Flag;
  }
  static bool isOptimizeMigration() { return OptimizeMigrationFlag; }
  static std::map<std::string, clang::tooling::OptionInfo> &getCurrentOptMap() {
    return CurrentOptMap;
  }
  static void setMainSourceYamlTUR(
      std::shared_ptr<clang::tooling::TranslationUnitReplacements> Ptr) {
    MainSourceYamlTUR = Ptr;
  }
  static std::shared_ptr<clang::tooling::TranslationUnitReplacements>
  getMainSourceYamlTUR() {
    return MainSourceYamlTUR;
  }
  static std::unordered_map<
      std::string,
      std::unordered_map<clang::tooling::UnifiedPath, std::vector<unsigned>>> &
  getRnnInputMap() {
    return RnnInputMap;
  }
  static std::unordered_map<clang::tooling::UnifiedPath,
                            std::vector<clang::tooling::UnifiedPath>> &
  getMainSourceFileMap() {
    return MainSourceFileMap;
  }
  static std::unordered_map<std::string, bool> &getMallocHostInfoMap() {
    return MallocHostInfoMap;
  }
  static std::map<std::shared_ptr<TextModification>, bool> &
  getConstantReplProcessedFlagMap() {
    return ConstantReplProcessedFlagMap;
  }
  static IncludeMapSetTy &getIncludeMapSet() { return IncludeMapSet; }
  static auto &getCodePinTypeInfoVec() { return CodePinTypeInfoMap; }
  static auto &getCodePinTemplateTypeInfoVec() {
    return CodePinTemplateTypeInfoMap;
  }
  static auto &getCodePinTypeDepsVec() { return CodePinTypeDepsVec; }
  static auto &getCodePinDumpFuncDepsVec() { return CodePinDumpFuncDepsVec; }
  static void setNeedParenAPI(const std::string &Name) {
    NeedParenAPISet.insert(Name);
  }
  static bool isNeedParenAPI(const std::string &Name) {
    return NeedParenAPISet.count(Name);
  }
  static void printUsingNamespace(llvm::raw_ostream &);
  // #tokens, name of the second token, SourceRange of a macro
  static std::tuple<unsigned int, std::string, SourceRange> LastMacroRecord;

private:
  DpctGlobalInfo();

  DpctGlobalInfo(const DpctGlobalInfo &) = delete;
  DpctGlobalInfo(DpctGlobalInfo &&) = delete;
  DpctGlobalInfo &operator=(const DpctGlobalInfo &) = delete;
  DpctGlobalInfo &operator=(DpctGlobalInfo &&) = delete;

  // Wrapper of isInAnalysisScope for std::function usage.
  static bool checkInAnalysisScope(SourceLocation SL) {
    return isInAnalysisScope(SL);
  }

  // Record token split when it's in macro
  static void recordTokenSplit(SourceLocation SL, unsigned Len);

  // Find stored info by its corresponding AST node.
  // VarDecl=>MemVarInfo
  // FunctionDecl=>DeviceFunctionDecl
  // CUDAKernelCallExpr=>KernelCallExpr
  // VarDecl=>CudaMallocInfo
  template <class Info, class Node>
  inline std::shared_ptr<Info> findNode(const Node *N) {
    if (!N)
      return std::shared_ptr<Info>();
    auto LocInfo = getLocInfo(N);
    if (isInAnalysisScope(LocInfo.first))
      return insertFile(LocInfo.first)->template findNode<Info>(LocInfo.second);
    return std::shared_ptr<Info>();
  }
  // Insert info if it doesn't exist.
  // The info will be used in Global.buildReplacements().
  // The key is the location of the Node.
  // The correction of the key is guaranteed by getLocation().
  template <class Info, class Node>
  inline std::shared_ptr<Info> insertNode(const Node *N) {
    auto LocInfo = getLocInfo(N);
    return insertFile(LocInfo.first)
        ->template insertNode<Info>(LocInfo.second, N);
  }

  template <class T> static inline SourceLocation getLocation(const T *N) {
    return N->getBeginLoc();
  }
  static SourceLocation getLocation(const VarDecl *VD) {
    return getDefinitionRange(VD->getLocation(), VD->getLocation()).getBegin();
  }
  static SourceLocation getLocation(const FunctionDecl *FD) {
    return FD->getBeginLoc();
  }
  static SourceLocation getLocation(const FieldDecl *FD) {
    return FD->getLocation();
  }
  static SourceLocation getLocation(const CallExpr *CE) {
    return getDefinitionRange(CE->getBeginLoc(), CE->getEndLoc()).getEnd();
  }
  // The result will be also stored in KernelCallExpr.BeginLoc
  static SourceLocation getLocation(const CUDAKernelCallExpr *CKC) {
    return getTheLastCompleteImmediateRange(CKC->getBeginLoc(),
                                            CKC->getEndLoc())
        .first;
  }

  std::shared_ptr<DpctFileInfo> MainFile = nullptr;
  std::unordered_map<clang::tooling::UnifiedPath, std::shared_ptr<DpctFileInfo>>
      FileMap;
  static std::shared_ptr<clang::tooling::TranslationUnitReplacements>
      MainSourceYamlTUR;
  static clang::tooling::UnifiedPath InRoot;
  static clang::tooling::UnifiedPath OutRoot;
  static std::vector<clang::tooling::UnifiedPath> AnalysisScope;
  static std::unordered_set<std::string> ChangeExtensions;
  static std::string SYCLSourceExtension;
  static std::string SYCLHeaderExtension;
  // TODO: implement one of this for each source language.
  static clang::tooling::UnifiedPath CudaPath;
  static std::string RuleFile;
  static UsmLevel UsmLvl;
  static unsigned BuildScriptType;
  static clang::CudaVersion SDKVersion;
  static bool NeedDpctDeviceExt;
  static bool IsIncMigration;
  static bool IsQueryAPIMapping;
  static unsigned int AssumedNDRangeDim;
  static std::unordered_set<std::string> PrecAndDomPairSet;
  static format::FormatRange FmtRng;
  static DPCTFormatStyle FmtST;
  static bool EnableCtad;
  static bool EnableCodePin;
  static bool IsMLKHeaderUsed;
  static bool GenBuildScript;
  static bool MigrateBuildScriptOnly;
  static bool EnableComments;

  // This variable is only set true when option "--report-type=stats" or option
  // " --report-type=all" is specified to get the migration status report, while
  // dpct namespace is not enabled.
  static bool TempEnableDPCTNamespace;
  static ASTContext *Context;
  static SourceManager *SM;
  static FileManager *FM;
  static bool KeepOriginCode;
  static bool SyclNamedLambda;
  static bool GuessIndentWidthMatcherFlag;
  static unsigned int IndentWidth;
  static std::map<unsigned int, unsigned int> KCIndentWidthMap;
  static std::unordered_map<std::string, int> LocationInitIndexMap;
  static std::unordered_map<
      std::string,
      std::pair<std::pair<clang::tooling::UnifiedPath /*begin file name*/,
                          unsigned int /*begin offset*/>,
                std::pair<clang::tooling::UnifiedPath /*end file name*/,
                          unsigned int /*end offset*/>>>
      ExpansionRangeBeginMap;
  static bool CheckUnicodeSecurityFlag;
  static bool EnablepProfilingFlag;
  static std::map<std::string,
                  std::shared_ptr<DpctGlobalInfo::MacroExpansionRecord>>
      ExpansionRangeToMacroRecord;
  // key: The hash string of the location of function-like macro argument
  // value: Function-like macro argument information
  static std::unordered_map<std::string,
                            std::shared_ptr<DpctGlobalInfo::MacroArgRecord>>
      MacroArgRecordMap;
  static std::map<std::string, SourceLocation> EndifLocationOfIfdef;
  static std::vector<std::pair<clang::tooling::UnifiedPath, size_t>>
      ConditionalCompilationLoc;
  static std::map<std::string, std::shared_ptr<DpctGlobalInfo::MacroDefRecord>>
      MacroTokenToMacroDefineLoc;
  static std::map<std::string, std::string> FunctionCallInMacroMigrateRecord;
  // key: The hash string of the first non-empty token after the end location of
  // macro expansion
  // value: begin location of macro expansion
  static std::map<std::string, SourceLocation> EndOfEmptyMacros;
  // key: The hash string of the begin location of the macro expansion
  // value: The end location of the macro expansion
  static std::map<std::string, unsigned int> BeginOfEmptyMacros;
  static std::unordered_map<std::string,
                            std::vector<clang::tooling::Replacement>>
      FileRelpsMap;
  static std::unordered_map<std::string, clang::tooling::MainSourceFileInfo>
      MsfInfoMap;
  static const std::string YamlFileName;
  static std::map<std::string, bool> MacroDefines;
  static int CurrentMaxIndex;
  static int CurrentIndexInRule;
  static std::set<clang::tooling::UnifiedPath> IncludingFileSet;
  static std::set<std::string> FileSetInCompilationDB;
  static std::set<std::string> GlobalVarNameSet;
  static clang::format::FormatStyle CodeFormatStyle;
  static bool HasFoundDeviceChanged;
  static std::unordered_map<int, HelperFuncReplInfo> HelperFuncReplInfoMap;
  static int HelperFuncReplInfoIndex;
  static std::unordered_map<std::string, TempVariableDeclCounter>
      TempVariableDeclCounterMap;
  static std::unordered_map<std::string, int> TempVariableHandledMap;
  static bool UsingDRYPattern;
  static unsigned int CudaKernelDimDFIIndex;
  static std::unordered_map<unsigned int, std::shared_ptr<DeviceFunctionInfo>>
      CudaKernelDimDFIMap;
  static CudaArchPPMap CAPPInfoMap;
  static HDFuncInfoMap HostDeviceFuncInfoMap;
  static CudaArchDefMap CudaArchDefinedMap;
  static std::unordered_map<std::string, std::shared_ptr<ExtReplacement>>
      CudaArchMacroRepl;
  static std::unordered_map<clang::tooling::UnifiedPath,
                            std::pair<std::shared_ptr<ExtReplacements>,
                                      std::shared_ptr<ExtReplacements>>>
      FileReplCache;
  static std::set<clang::tooling::UnifiedPath> ReProcessFile;
  static bool NeedRunAgain;
  static unsigned int RunRound;
  static std::set<clang::tooling::UnifiedPath> ModuleFiles;
  static std::unordered_map<
      std::string, std::unordered_set<std::shared_ptr<DeviceFunctionInfo>>>
      SpellingLocToDFIsMapForAssumeNDRange;
  static std::unordered_map<std::shared_ptr<DeviceFunctionInfo>,
                            std::unordered_set<std::string>>
      DFIToSpellingLocsMapForAssumeNDRange;
  static unsigned ExtensionDEFlag;
  static unsigned ExtensionDDFlag;
  static unsigned ExperimentalFlag;
  static unsigned HelperFuncPreferenceFlag;
  static bool AnalysisModeFlag;
  static bool UseSYCLCompatFlag;
  static bool CVersionCUDALaunchUsedFlag;
  static unsigned int ColorOption;
  static std::unordered_map<int, std::shared_ptr<DeviceFunctionInfo>>
      CubPlaceholderIndexMap;
  static std::vector<std::shared_ptr<DpctFileInfo>> CSourceFileInfo;
  static bool OptimizeMigrationFlag;
  static std::unordered_map<std::string, std::shared_ptr<PriorityReplInfo>>
      PriorityReplInfoMap;
  static std::unordered_map<std::string, bool> ExcludePath;
  static std::map<std::string, clang::tooling::OptionInfo> CurrentOptMap;
  static std::unordered_map<
      std::string,
      std::unordered_map<clang::tooling::UnifiedPath, std::vector<unsigned>>>
      RnnInputMap;
  static std::unordered_map<clang::tooling::UnifiedPath,
                            std::vector<clang::tooling::UnifiedPath>>
      MainSourceFileMap;
  static std::unordered_map<std::string, bool> MallocHostInfoMap;
  /// The key of this map is repl for specifier "__const__" and the value
  /// "true" means this repl has been processed.
  static std::map<std::shared_ptr<TextModification>, bool>
      ConstantReplProcessedFlagMap;
  static IncludeMapSetTy IncludeMapSet;
  static std::vector<std::pair<std::string, VarInfoForCodePin>>
      CodePinTypeInfoMap;
  static std::vector<std::pair<std::string, VarInfoForCodePin>>
      CodePinTemplateTypeInfoMap;
  static std::vector<std::pair<std::string, std::vector<std::string>>>
      CodePinTypeDepsVec;
  static std::vector<std::pair<std::string, std::vector<std::string>>>
      CodePinDumpFuncDepsVec;
  static std::unordered_set<std::string> NeedParenAPISet;
  static std::unordered_set<std::string> CustomHelperFunctionAddtionalIncludes;
};

/// Generate mangle name of FunctionDecl as key of DeviceFunctionInfo.
/// For template dependent FunctionDecl, generate name with pattern
/// "QuailifiedName@FunctionType".
/// e.g.: template<class T> void test(T *int)
/// -> test@void (type-parameter-0-1 *)
class DpctNameGenerator {
  ASTNameGenerator G;
  PrintingPolicy PP;
  void printName(const FunctionDecl *FD, llvm::raw_ostream &OS);

public:
  DpctNameGenerator() : DpctNameGenerator(DpctGlobalInfo::getContext()) {}
  explicit DpctNameGenerator(ASTContext &Ctx);
  std::string getName(const FunctionDecl *D);
};

class TemplateArgumentInfo;

// Store array size string. Like below:
// a[10]: Store "10" as size;
// a[]: Store "" as empty size;
// a[SIZE]: Store as a TemplateDependentStringInfo while "SIZE" is a template
// parameter;
class SizeInfo {
  std::string Size;
  std::shared_ptr<TemplateDependentStringInfo> TDSI;

public:
  SizeInfo() = default;
  SizeInfo(std::string Size) : Size(std::move(Size)) {}
  SizeInfo(std::shared_ptr<TemplateDependentStringInfo> TDSI);
  const std::string &getSize();
  // Get actual size string according to template arguments list;
  void setTemplateList(const std::vector<TemplateArgumentInfo> &TemplateList);
};
// CtTypeInfo is basic class with info of element type, range, template info all
// get from type.
class CtTypeInfo {
public:
  struct {
    std::string TypeName;
    std::string DefinitionFuncName;
  } SharedVarInfo;
  // If NeedSizeFold is true, array size will be folded, but original expression
  // will follow as comments. If NeedSizeFold is false, original size expression
  // will be the size string.
  CtTypeInfo();
  CtTypeInfo(const TypeLoc &TL, bool NeedSizeFold = false);
  CtTypeInfo(const VarDecl *D, bool NeedSizeFold = false);
  const std::string &getBaseName() { return BaseName; }
  const std::string &getBaseNameWithoutQualifiers() {
    return BaseNameWithoutQualifiers;
  }
  size_t getDimension() { return Range.size(); }
  std::vector<SizeInfo> &getRange() { return Range; }
  // when there is no arguments, parameter MustArguments determine whether
  // parens will exist. Null string will be returned when MustArguments is
  // false, otherwise "()" will be returned.
  std::string getRangeArgument(const std::string &MemSize, bool MustArguments);
  inline bool isTemplate() const { return IsTemplate; }
  inline bool isPointer() const { return PointerLevel; }
  inline bool isArray() const { return IsArray; }
  inline bool isReference() const { return IsReference; }
  inline void adjustAsMemType();
  // Get instantiated type name with given template arguments.
  // e.g. X<T>, with T = int, result type will be X<int>.
  std::shared_ptr<CtTypeInfo>
  applyTemplateArguments(const std::vector<TemplateArgumentInfo> &TA);
  bool isWritten() const {
    return !TDSI || !isTemplate() || TDSI->isDependOnWritten();
  }
  std::set<HelperFeatureEnum> getHelperFeatureSet() { return HelperFeatureSet; }
  bool containSizeofType() { return ContainSizeofType; }
  std::vector<std::string> getArraySizeOriginExprs() {
    return ArraySizeOriginExprs;
  }
  bool containsTemplateDependentMacro() const { return TemplateDependentMacro; }
  bool isConstantQualified() const { return IsConstantQualified; }

private:
  // For ConstantArrayType, size in generated code is folded as an integer.
  // If \p NeedSizeFold is true, original size expression will be appended as
  // comments.
  void setTypeInfo(const TypeLoc &TL, bool NeedSizeFold = false);

  // Get folded array size with original size expression following as comments.
  // e.g.,
  // #define SIZE 24
  // dpct::global_memory<int, 1>(24 /* SIZE */);
  // Exception for particular case:
  // __device__ int a[24];
  // will be migrated to:
  // dpct::global_memory<int, 1> a(24);
  inline std::string getFoldedArraySize(const ConstantArrayTypeLoc &TL);

  // Get original array size expression.
  std::string getUnfoldedArraySize(const ConstantArrayTypeLoc &TL);

  bool setTypedefInfo(const TypedefTypeLoc &TL, bool NeedSizeFold);

  // Typically C++ array with constant size.
  // e.g.: __device__ int a[20];
  // If \p NeedSizeFold is true, original size expression will be appended as
  // comments.
  // e.g.,
  // #define SIZE 24
  // dpct::global_memory<int, 1>(24 /* SIZE */);
  void setArrayInfo(const ConstantArrayTypeLoc &TL, bool NeedFoldSize);

  // Typically C++ array with template dependent size.
  // e.g.: template<size_t S>
  // ...
  // __device__ int a[S];
  void setArrayInfo(const DependentSizedArrayTypeLoc &TL, bool NeedSizeFold);

  // IncompleteArray is an array defined without size.
  // e.g.: extern __shared__ int a[];
  void setArrayInfo(const IncompleteArrayTypeLoc &TL, bool NeedSizeFold);
  void setName(const TypeLoc &TL);
  void updateName();
  void setPointerAsArray();
  void removeQualifier() { BaseName = BaseNameWithoutQualifiers; }

private:
  unsigned PointerLevel : 16;
  unsigned IsReference : 1;
  unsigned IsTemplate : 1;
  unsigned TemplateDependentMacro : 1;
  unsigned IsArray : 1;
  unsigned ContainSizeofType : 1;
  unsigned IsConstantQualified : 1;
  std::string BaseName;
  std::string BaseNameWithoutQualifiers;
  std::vector<SizeInfo> Range;
  std::vector<std::string> ArraySizeOriginExprs{};
  std::set<HelperFeatureEnum> HelperFeatureSet;
  std::shared_ptr<TemplateDependentStringInfo> TDSI;
};

// variable info includes name, type and location.
class VarInfo {
public:
  VarInfo(unsigned Offset, const clang::tooling::UnifiedPath &FilePathIn,
          const VarDecl *Var, bool NeedFoldSize = false)
      : FilePath(FilePathIn), Offset(Offset), Name(Var->getName()),
        Ty(std::make_shared<CtTypeInfo>(Var, NeedFoldSize)) {}
  const clang::tooling::UnifiedPath &getFilePath() { return FilePath; }
  unsigned getOffset() { return Offset; }
  const std::string &getName() { return Name; }
  const std::string getNameAppendSuffix() { return Name + "_ct1"; }
  std::shared_ptr<CtTypeInfo> &getType() { return Ty; }
  std::string getDerefName() {
    return buildString(getName(), "_deref_", DpctGlobalInfo::getInRootHash());
  }
  void applyTemplateArguments(const std::vector<TemplateArgumentInfo> &TAList);
  void requestFeatureForSet(const clang::tooling::UnifiedPath &Path);

private:
  const clang::tooling::UnifiedPath FilePath;
  unsigned Offset;
  std::string Name;
  std::shared_ptr<CtTypeInfo> Ty;
};

// memory variable info includes basic variable info and memory attributes.
class MemVarInfo : public VarInfo {
public:
  enum VarAttrKind {
    Device = 0,
    Constant,
    Shared,
    Host,
    Managed,
  };
  enum VarScope { Local = 0, Extern, Global };

  static std::shared_ptr<MemVarInfo> buildMemVarInfo(const VarDecl *Var);
  static VarAttrKind getAddressAttr(const VarDecl *VD);

  MemVarInfo(unsigned Offset, const clang::tooling::UnifiedPath &FilePath,
             const VarDecl *Var);

  VarAttrKind getAttr() { return Attr; }
  VarScope getScope() { return Scope; }
  bool isGlobal() { return Scope == Global; }
  bool isExtern() { return Scope == Extern; }
  bool isLocal() { return Scope == Local; }
  bool isShared() { return Attr == Shared; }
  bool isConstant() { return Attr == Constant; }
  bool isDevice() { return Attr == Device; }
  bool isManaged() { return Attr == Managed; }
  bool isTypeDeclaredLocal() { return IsTypeDeclaredLocal; }
  bool isAnonymousType() { return IsAnonymousType; }
  const CXXRecordDecl *getDeclOfVarType() { return DeclOfVarType; }
  const DeclStmt *getDeclStmtOfVarType() { return DeclStmtOfVarType; }
  void setLocalTypeName(std::string T) { LocalTypeName = T; }
  std::string getLocalTypeName() { return LocalTypeName; }
  void setIgnoreFlag(bool Flag) { IsIgnored = Flag; }
  bool isIgnore() { return IsIgnored; }
  bool isStatic() { return IsStatic; }
  void setName(std::string NewName) { NewConstVarName = NewName; }
  unsigned int getNewConstVarOffset() { return NewConstVarOffset; }
  unsigned int getNewConstVarLength() { return NewConstVarLength; }
  const std::string getConstVarName() {
    return NewConstVarName.empty() ? getArgName() : NewConstVarName;
  }
  // Initialize offset and length for __constant__ variable that needs to be
  // renamed.
  void newConstVarInit(const VarDecl *Var);
  std::string getDeclarationReplacement(const VarDecl *);
  std::string getInitStmt() { return getInitStmt(""); }
  std::string getInitStmt(StringRef QueueString);
  std::string getMemoryDecl(const std::string &MemSize);
  std::string getMemoryDecl();
  std::string getExternGlobalVarDecl();
  void appendAccessorOrPointerDecl(const std::string &ExternMemSize,
                                   bool ExternEmitWarning, StmtList &AccList,
                                   StmtList &PtrList, LocInfo LI);
  std::string getRangeClass();
  std::string getRangeDecl(const std::string &MemSize);
  ParameterStream &getFuncDecl(ParameterStream &PS);
  ParameterStream &getFuncArg(ParameterStream &PS);
  ParameterStream &getKernelArg(ParameterStream &PS);
  std::string getAccessorDataType(bool IsTypeUsedInDevFunDecl = false,
                                  bool NeedCheckExtraConstQualifier = false);
  void setUsedBySymbolAPIFlag(bool Flag) { UsedBySymbolAPIFlag = Flag; }
  bool getUsedBySymbolAPIFlag() { return UsedBySymbolAPIFlag; }
  void setUseHelperFuncFlag(bool Flag) { UseHelperFuncFlag = Flag; }
  bool isUseHelperFunc() { return UseHelperFuncFlag; }
  void setUseDeviceGlobalFlag(bool Flag) { UseDeviceGlobalFlag = Flag; }
  bool isUseDeviceGlobal() { return UseDeviceGlobalFlag; }
  void migrateToDeviceGlobal(const VarDecl *MemVar);

private:
  bool isTreatPointerAsArray() {
    return getType()->isPointer() && getScope() == Global &&
           DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None;
  }
  static VarAttrKind getAddressAttr(const AttrVec &Attrs);
  void setInitList(const Expr *E, const VarDecl *V);

  std::string getMemoryType();
  std::string getMemoryType(const std::string &MemoryType,
                            std::shared_ptr<CtTypeInfo> VarType);
  std::string getInitArguments(const std::string &MemSize,
                               bool MustArguments = false);
  const std::string &getMemoryAttr();
  std::string getSyclAccessorType(LocInfo LI = LocInfo());
  std::string getDpctAccessorType();
  std::string getNameWithSuffix(StringRef Suffix) {
    return buildString(getArgName(), "_", Suffix, getCTFixedSuffix());
  }
  std::string getAccessorName() { return getNameWithSuffix("acc"); }
  std::string getPtrName() { return getNameWithSuffix("ptr"); }
  std::string getRangeName() { return getNameWithSuffix("range"); }
  std::string getArgName();

private:
  // Passing by accessor, value or pointer when invoking kernel.
  // Constant scalar variables are passed by value while other 0/1D variables
  // defined on device memory are passed by pointer in device function calls.
  // The rest are passed by accessor.
  enum DpctAccessMode { Value, Pointer, Accessor, Reference, PointerToArray };

private:
  VarAttrKind Attr;
  VarScope Scope;
  DpctAccessMode AccMode;
  bool PointerAsArray;
  std::string InitList;
  bool IsIgnored = false;
  bool IsStatic = false;

  static const std::string ExternVariableName;

  // To store the new name for __constant__ variable's name that needs to be
  // renamed.
  std::string NewConstVarName;

  // To store the offset and length for __constant__ variable's name
  // that needs to be renamed.
  unsigned int NewConstVarOffset;
  unsigned int NewConstVarLength;

  bool IsTypeDeclaredLocal = false;
  bool IsAnonymousType = false;
  const CXXRecordDecl *DeclOfVarType = nullptr;
  const DeclStmt *DeclStmtOfVarType = nullptr;
  std::string LocalTypeName = "";

  static std::unordered_map<std::string, int> AnonymousTypeDeclStmtMap;
  bool UsedBySymbolAPIFlag = false;
  bool UseHelperFuncFlag = true;
  bool UseDeviceGlobalFlag = false;
};

class TextureTypeInfo {
  std::string DataType;
  int Dimension;
  bool IsArray;

public:
  TextureTypeInfo(std::string &&DataType, int TexType);
  void setDataTypeAndTexType(std::string &&Type, int TexType);
  void prepareForImage();
  void endForImage();
  std::string getDataType() { return DataType; }
  ParameterStream &printType(ParameterStream &PS,
                             const std::string &TemplateName);
};

class TextureInfo {
protected:
  const clang::tooling::UnifiedPath FilePath;
  const unsigned Offset;
  std::string Name;       // original expression str
  std::string NewVarName; // name of new variable which tool

  std::shared_ptr<TextureTypeInfo> Type;

protected:
  TextureInfo(unsigned Offset, const clang::tooling::UnifiedPath &FilePath,
              StringRef Name);
  TextureInfo(const VarDecl *VD);
  TextureInfo(const VarDecl *VD, std::string Subscript);
  TextureInfo(std::pair<clang::tooling::UnifiedPath, unsigned> LocInfo,
              StringRef Name);
  ParameterStream &getDecl(ParameterStream &PS,
                           const std::string &TemplateDeclName);
  template <class StreamT>
  static void printQueueStr(StreamT &OS, const std::string &Queue) {
    if (Queue.empty())
      return;
    OS << ", " << Queue;
  }

public:
  TextureInfo(unsigned Offset, const clang::tooling::UnifiedPath &FilePath,
              const VarDecl *VD);
  virtual ~TextureInfo() = default;
  void setType(std::string &&DataType, int TexType);
  void setType(std::shared_ptr<TextureTypeInfo> TypeInfo);
  std::shared_ptr<TextureTypeInfo> getType() const { return Type; }
  virtual std::string getHostDeclString();
  virtual std::string getSamplerDecl();
  virtual std::string getAccessorDecl(const std::string &QueueStr);
  virtual std::string InitDecl(const std::string &QueueStr);
  virtual void addDecl(StmtList &InitList, StmtList &AccessorList,
                       StmtList &SamplerList, const std::string &QueueStr);
  ParameterStream &getFuncDecl(ParameterStream &PS);
  ParameterStream &getFuncArg(ParameterStream &PS);
  virtual ParameterStream &getKernelArg(ParameterStream &OS);
  const std::string &getName() { return Name; }
  unsigned getOffset() { return Offset; }
  clang::tooling::UnifiedPath getFilePath() { return FilePath; }
  bool isUseHelperFunc() { return true; }
};

// texture object info can be used for CUDA texture and surface objects.
class TextureObjectInfo : public TextureInfo {
  static const int ReplaceTypeLength;

  // If it is a parameter in the function, it is the parameter index, either it
  // is 0.
  unsigned ParamIdx;

  TextureObjectInfo(const VarDecl *VD, unsigned ParamIdx)
      : TextureInfo(VD), ParamIdx(ParamIdx) {}
  TextureObjectInfo(const VarDecl *VD, std::string Subscript, unsigned ParamIdx)
      : TextureInfo(VD, Subscript), ParamIdx(ParamIdx) {}

protected:
  TextureObjectInfo(unsigned Offset,
                    const clang::tooling::UnifiedPath &FilePath, StringRef Name)
      : TextureInfo(Offset, FilePath, Name), ParamIdx(0) {}

public:
  TextureObjectInfo(const ParmVarDecl *PVD)
      : TextureObjectInfo(PVD, PVD->getFunctionScopeIndex()) {}
  TextureObjectInfo(const VarDecl *VD) : TextureObjectInfo(VD, 0) {}

  TextureObjectInfo(const ParmVarDecl *PVD, std::string Subscript)
      : TextureObjectInfo(PVD, Subscript, PVD->getFunctionScopeIndex()) {}
  TextureObjectInfo(const VarDecl *VD, std::string Subscript)
      : TextureObjectInfo(VD, Subscript, 0) {}

  virtual ~TextureObjectInfo() = default;
  std::string getAccessorDecl(const std::string &QueueString) override;
  std::string InitDecl(const std::string &QueueStr) override;
  std::string getSamplerDecl() override;
  inline unsigned getParamIdx() const { return ParamIdx; }
  std::string getParamDeclType();
  virtual void merge(std::shared_ptr<TextureObjectInfo> Target);
  virtual void addParamDeclReplacement();

  template <class Node> static inline bool isTextureObject(const Node *E) {
    if (E)
      return DpctGlobalInfo::getUnqualifiedTypeName(E->getType()) ==
             "cudaTextureObject_t";
    return false;
  }
};

class CudaLaunchTextureObjectInfo : public TextureObjectInfo {
  std::string ArgStr;

public:
  CudaLaunchTextureObjectInfo(const ParmVarDecl *PVD, const std::string &ArgStr)
      : TextureObjectInfo(static_cast<const VarDecl *>(PVD)), ArgStr(ArgStr) {}
  std::string getAccessorDecl(const std::string &QueueString) override;
  std::string getSamplerDecl() override;
};

class MemberTextureObjectInfo : public TextureObjectInfo {
  StringRef BaseName;
  std::string MemberName;

  class NewVarNameRAII {
    std::string OldName;
    MemberTextureObjectInfo *Member;

  public:
    NewVarNameRAII(MemberTextureObjectInfo *M);
    ~NewVarNameRAII() { Member->Name = std::move(OldName); }
  };

  MemberTextureObjectInfo(unsigned Offset,
                          const clang::tooling::UnifiedPath &FilePath,
                          StringRef Name)
      : TextureObjectInfo(Offset, FilePath, Name) {}

public:
  static std::shared_ptr<MemberTextureObjectInfo> create(const MemberExpr *ME);
  void addDecl(StmtList &InitList, StmtList &AccessorList,
               StmtList &SamplerList, const std::string &QueueStr) override;
  void setBaseName(StringRef Name) { BaseName = Name; }
  StringRef getMemberName() { return MemberName; }
};

class StructureTextureObjectInfo : public TextureObjectInfo {
  std::unordered_map<std::string, std::shared_ptr<MemberTextureObjectInfo>>
      Members;
  bool ContainsVirtualPointer;
  bool IsBase = false;

  StructureTextureObjectInfo(unsigned Offset,
                             const clang::tooling::UnifiedPath &FilePath,
                             StringRef Name)
      : TextureObjectInfo(Offset, FilePath, Name) {}

public:
  StructureTextureObjectInfo(const ParmVarDecl *PVD);
  StructureTextureObjectInfo(const VarDecl *VD);
  static std::shared_ptr<StructureTextureObjectInfo>
  create(const CXXThisExpr *This);
  bool isBase() const { return IsBase; }
  bool containsVirtualPointer() const { return ContainsVirtualPointer; }
  std::shared_ptr<MemberTextureObjectInfo> addMember(const MemberExpr *ME);
  void addDecl(StmtList &InitList, StmtList &AccessorList,
               StmtList &SamplerList, const std::string &Queue) override;
  void addParamDeclReplacement() override { return; };
  void merge(std::shared_ptr<StructureTextureObjectInfo> Target);
  void merge(std::shared_ptr<TextureObjectInfo> Target) override;
  ParameterStream &getKernelArg(ParameterStream &OS) override;
};

class TemplateArgumentInfo {
public:
  explicit TemplateArgumentInfo(const TemplateArgumentLoc &TAL,
                                SourceRange Range);
  explicit TemplateArgumentInfo(std::string &&Str);
  TemplateArgumentInfo() : Kind(TemplateArgument::Null), IsWritten(false) {}

  bool isWritten() const { return IsWritten; }
  bool isNull() const { return !DependentStr; }
  bool isType() const { return Kind == TemplateArgument::Type; }
  const std::string &getString() const {
    return getDependentStringInfo()->getSourceString();
  }
  std::shared_ptr<const TemplateDependentStringInfo>
  getDependentStringInfo() const;
  void setAsType(QualType QT);
  void setAsType(const TypeLoc &TL);
  void setAsType(std::string TS);
  void setAsNonType(const llvm::APInt &Int);
  void setAsNonType(const Expr *E);

  static bool isPlaceholderType(clang::QualType QT);

private:
  template <class T>
  void setArgFromExprAnalysis(const T &Arg,
                              SourceRange ParentRange = SourceRange());

  template <class T> SourceRange getArgSourceRange(const T &Arg) {
    return Arg.getSourceRange();
  }

  template <class T> SourceRange getArgSourceRange(const T *Arg) {
    return Arg->getSourceRange();
  }

  void setArgStr(std::string &&Str) {
    DependentStr =
        std::make_shared<TemplateDependentStringInfo>(std::move(Str));
  }
  std::shared_ptr<TemplateDependentStringInfo> DependentStr;
  TemplateArgument::ArgKind Kind;
  bool IsWritten = true;
};

class TempStorageVarInfo {
public:
  enum APIKind {
    BlockReduce,
    BlockRadixSort,
  };

private:
  unsigned Offset;
  APIKind Kind;
  std::string Name;
  std::string TmpMemSizeCalFn;
  std::shared_ptr<TemplateDependentStringInfo> ValueType;

public:
  TempStorageVarInfo(unsigned Off, APIKind Kind, StringRef Name,
                     std::string TmpMemSizeCalFn,
                     std::shared_ptr<TemplateDependentStringInfo> ValT)
      : Offset(Off), Kind(Kind), Name(Name.str()),
        TmpMemSizeCalFn(TmpMemSizeCalFn), ValueType(ValT) {}
  const std::string &getName() const { return Name; }
  unsigned getOffset() const { return Offset; }
  void addAccessorDecl(StmtList &AccessorList, StringRef LocalSize) const;
  void applyTemplateArguments(const std::vector<TemplateArgumentInfo> &TA);
  ParameterStream &getFuncDecl(ParameterStream &PS);
  ParameterStream &getFuncArg(ParameterStream &PS);
  ParameterStream &getKernelArg(ParameterStream &PS);
};

// memory variable map includes memory variable used in __global__/__device__
// function and call expression.
class MemVarMap {
public:
  MemVarMap()
      : HasItem(false), HasStream(false), HasSync(false), HasBF64(false),
        HasBF16(false), HasGlobalMemAcc(false) {}
  unsigned int Dim = 1;
  /// This member is only used to construct the union-find set.
  MemVarMap *Parent = this;
  bool hasItem() const { return HasItem; }
  bool hasStream() const { return HasStream; }
  bool hasSync() const { return HasSync; }
  bool hasBF64() const { return HasBF64; }
  bool hasBF16() const { return HasBF16; }
  bool hasGlobalMemAcc() const { return HasGlobalMemAcc; }
  bool hasExternShared() const { return !ExternVarMap.empty(); }
  void setItem(bool Has = true) { HasItem = Has; }
  void setStream(bool Has = true) { HasStream = Has; }
  void setSync(bool Has = true) { HasSync = Has; }
  void setBF64(bool Has = true) { HasBF64 = Has; }
  void setBF16(bool Has = true) { HasBF16 = Has; }
  void setGlobalMemAcc(bool Has = true) { HasGlobalMemAcc = Has; }
  void addCUBTempStorage(std::shared_ptr<TempStorageVarInfo> Tmp);
  void addTexture(std::shared_ptr<TextureInfo> Tex);
  void addVar(std::shared_ptr<MemVarInfo> Var);
  void merge(const MemVarMap &OtherMap);
  void merge(const MemVarMap &VarMap,
             const std::vector<TemplateArgumentInfo> &TemplateArgs);
  int calculateExtraArgsSize() const;

  enum CallOrDecl {
    CallArgument = 0,
    KernelArgument,
    DeclParameter,
  };

private:
  template <CallOrDecl COD>
  std::string
  getArgumentsOrParameters(int PreParams, int PostParams,
                           LocInfo LI = LocInfo(),
                           FormatInfo FormatInformation = FormatInfo()) const;

public:
  std::string getExtraCallArguments(bool HasPreParam, bool HasPostParam) const;
  void
  requestFeatureForAllVarMaps(const clang::tooling::UnifiedPath &Path) const;

  // When adding the ExtraParam with new line, the second argument should be
  // true, and the third argument is the string of indent, which will occur
  // before each ExtraParam.
  std::string
  getExtraDeclParam(bool HasPreParam, bool HasPostParam, LocInfo LI,
                    FormatInfo FormatInformation = FormatInfo()) const;
  std::string getKernelArguments(bool HasPreParam, bool HasPostParam,
                                 const clang::tooling::UnifiedPath &Path) const;
  const MemVarInfoMap &getMap(MemVarInfo::VarScope Scope) const;
  const GlobalMap<TextureInfo> &getTextureMap() const;
  const GlobalMap<TempStorageVarInfo> &getTempStorageMap() const;
  void removeDuplicateVar();

  MemVarInfoMap &getMap(MemVarInfo::VarScope Scope);
  bool isSameAs(const MemVarMap &Other) const;

  static const MemVarMap *
  getHeadWithoutPathCompression(const MemVarMap *CurNode);
  static MemVarMap *getHead(MemVarMap *CurNode);
  unsigned int getHeadNodeDim() const;

private:
  template <class VarT>
  static void merge(GlobalMap<VarT> &Master, const GlobalMap<VarT> &Branch,
                    const std::vector<TemplateArgumentInfo> &TemplateArgs) {
    if (TemplateArgs.empty())
      return dpct::merge(Master, Branch);
    for (auto &VarInfoPair : Branch)
      Master
          .insert(std::make_pair(VarInfoPair.first,
                                 std::make_shared<VarT>(*VarInfoPair.second)))
          .first->second->applyTemplateArguments(TemplateArgs);
  }
  int calculateExtraArgsSize(const MemVarInfoMap &Map) const;

  template <CallOrDecl COD>
  inline ParameterStream &getItem(ParameterStream &PS) const {
    return PS << getItemName();
  }

  template <CallOrDecl COD>
  inline ParameterStream &getStream(ParameterStream &PS) const {
    return PS << DpctGlobalInfo::getStreamName();
  }

  template <CallOrDecl COD>
  inline ParameterStream &getSync(ParameterStream &PS) const {
    return PS << buildString("atm_", DpctGlobalInfo::getSyncName());
  }

  template <class T, CallOrDecl COD>
  static void getArgumentsOrParametersFromMap(ParameterStream &PS,
                                              const GlobalMap<T> &VarMap,
                                              LocInfo LI = LocInfo());
  template <CallOrDecl COD>
  static void getArgumentsOrParametersFromoTextureInfoMap(
      ParameterStream &PS, const GlobalMap<TextureInfo> &VarMap);
  template <class T, CallOrDecl COD> struct GetArgOrParam;
  template <class T> struct GetArgOrParam<T, DeclParameter> {
    ParameterStream &operator()(ParameterStream &PS, std::shared_ptr<T> V) {
      return V->getFuncDecl(PS);
    }
  };
  template <class T> struct GetArgOrParam<T, CallArgument> {
    ParameterStream &operator()(ParameterStream &PS, std::shared_ptr<T> V) {
      return V->getFuncArg(PS);
    }
  };
  template <class T> struct GetArgOrParam<T, KernelArgument> {
    ParameterStream &operator()(ParameterStream &PS, std::shared_ptr<T> V) {
      return V->getKernelArg(PS);
    }
  };
  void getArgumentsOrParametersForDecl(ParameterStream &PS, int PreParams,
                                       int PostParams, LocInfo LI) const;

  bool HasItem, HasStream, HasSync, HasBF64, HasBF16, HasGlobalMemAcc;
  MemVarInfoMap LocalVarMap;
  MemVarInfoMap GlobalVarMap;
  MemVarInfoMap ExternVarMap;
  GlobalMap<TextureInfo> TextureMap;
  GlobalMap<TempStorageVarInfo> TempStorageMap;
};

template <>
inline ParameterStream &
MemVarMap::getItem<MemVarMap::DeclParameter>(ParameterStream &PS) const {
  std::string NDItem = "nd_item<3>";
  if (DpctGlobalInfo::getAssumedNDRangeDim() == 1 &&
      MemVarMap::getHeadWithoutPathCompression(this) &&
      MemVarMap::getHeadWithoutPathCompression(this)->Dim == 1) {
    NDItem = "nd_item<1>";
  }

  std::string ItemParamDecl =
      "const " + MapNames::getClNamespace() + NDItem + " &" + getItemName();
  return PS << ItemParamDecl;
}

template <>
inline ParameterStream &
MemVarMap::getStream<MemVarMap::DeclParameter>(ParameterStream &PS) const {
  static std::string StreamParamDecl = "const " + MapNames::getClNamespace() +
                                       "stream &" +
                                       DpctGlobalInfo::getStreamName();
  return PS << StreamParamDecl;
}

template <>
inline ParameterStream &
MemVarMap::getSync<MemVarMap::DeclParameter>(ParameterStream &PS) const {
  static std::string SyncParamDecl =
      MapNames::getClNamespace() + "atomic_ref<unsigned int, " +
      MapNames::getClNamespace() + "memory_order::seq_cst, " +
      MapNames::getClNamespace() + "memory_scope::device, " +
      MapNames::getClNamespace() + "access::address_space::global_space> &" +
      DpctGlobalInfo::getSyncName();
  return PS << SyncParamDecl;
}

// call function expression includes location, name, arguments num, template
// arguments and all function decls related to this call, also merges memory
// variable info of all related function decls.
class CallFunctionExpr {
public:
  template <class T>
  CallFunctionExpr(unsigned Offset,
                   const clang::tooling::UnifiedPath &FilePathIn, const T &C)
      : FilePath(FilePathIn), Offset(Offset), CallFuncExprOffset(Offset) {}

  void buildCallExprInfo(const CXXConstructExpr *Ctor);
  void buildCallExprInfo(const CallExpr *CE);

  const MemVarMap &getVarMap() { return VarMap; }
  const std::vector<std::shared_ptr<TextureObjectInfo>> &
  getTextureObjectList() {
    return TextureObjectList;
  }
  std::shared_ptr<StructureTextureObjectInfo> getBaseTextureObjectInfo() const {
    return BaseTextureObject;
  }

  void emplaceReplacement();
  unsigned getExtraArgLoc() { return ExtraArgLoc; }
  bool hasArgs() { return HasArgs; }
  bool hasTemplateArgs() { return !TemplateArgs.empty(); }
  bool hasWrittenTemplateArgs();
  const std::string &getName() { return Name; }

  std::string getTemplateArguments(bool &IsNeedWarning,
                                   bool WrittenArgsOnly = true,
                                   bool WithScalarWrapped = false);

  virtual std::string getExtraArguments();

  void setHasSideEffects(bool Val = true) {
    CallGroupFunctionInControlFlow = Val;
  }
  bool hasSideEffects() const { return CallGroupFunctionInControlFlow; }

  std::shared_ptr<TextureObjectInfo>
  addTextureObjectArgInfo(unsigned ArgIdx,
                          std::shared_ptr<TextureObjectInfo> Info);
  virtual std::shared_ptr<TextureObjectInfo>
  addTextureObjectArg(unsigned ArgIdx, const DeclRefExpr *TexRef,
                      bool isKernelCall = false);
  virtual std::shared_ptr<TextureObjectInfo>
  addStructureTextureObjectArg(unsigned ArgIdx, const MemberExpr *TexRef,
                               bool isKernelCall = false);
  virtual std::shared_ptr<TextureObjectInfo>
  addTextureObjectArg(unsigned ArgIdx, const ArraySubscriptExpr *TexRef,
                      bool isKernelCall = false);
  std::shared_ptr<DeviceFunctionInfo> getFuncInfo();
  bool IsAllTemplateArgsSpecified = false;

  virtual ~CallFunctionExpr() = default;

protected:
  void setFuncInfo(std::shared_ptr<DeviceFunctionInfo>);
  std::string Name;
  unsigned getOffset() { return Offset; }
  const clang::tooling::UnifiedPath &getFilePath() { return FilePath; }
  void buildInfo();
  void buildCalleeInfo(const Expr *Callee, std::optional<unsigned int> NumArgs);
  void resizeTextureObjectList(size_t Size) { TextureObjectList.resize(Size); }

private:
  static std::string getName(const NamedDecl *D);
  void
  buildTemplateArguments(const llvm::ArrayRef<TemplateArgumentLoc> &ArgsList,
                         SourceRange Range);

  void buildTemplateArgumentsFromTypeLoc(const TypeLoc &TL);
  template <class TyLoc>
  void buildTemplateArgumentsFromSpecializationType(const TyLoc &TL);

  std::string getNameWithNamespace(const FunctionDecl *FD, const Expr *Callee);

  void buildTextureObjectArgsInfo(const CallExpr *CE);

  template <class CallT> void buildTextureObjectArgsInfo(const CallT *C);
  void mergeTextureObjectInfo(std::shared_ptr<DeviceFunctionInfo> Info);

  const clang::tooling::UnifiedPath FilePath;
  unsigned Offset = 0;
  unsigned CallFuncExprOffset = 0;
  unsigned ExtraArgLoc = 0;
  std::vector<std::shared_ptr<DeviceFunctionInfo>> FuncInfo;
  std::vector<TemplateArgumentInfo> TemplateArgs;

  // <ParameterIndex, ParameterName>
  std::vector<std::pair<int, std::string>> ParmRefArgs;
  MemVarMap VarMap;
  bool HasArgs = false;
  bool IsADLEnable = false;
  bool CallGroupFunctionInControlFlow = false;
  std::vector<std::shared_ptr<TextureObjectInfo>> TextureObjectList;
  std::shared_ptr<StructureTextureObjectInfo> BaseTextureObject;
};

// device function declaration info includes location, name, and related
// DeviceFunctionInfo
class DeviceFunctionDecl {
public:
  DeviceFunctionDecl(unsigned Offset,
                     const clang::tooling::UnifiedPath &FilePathIn,
                     const FunctionDecl *FD);
  DeviceFunctionDecl(unsigned Offset,
                     const clang::tooling::UnifiedPath &FilePathIn,
                     const FunctionTypeLoc &FTL, const ParsedAttributes &Attrs,
                     const FunctionDecl *Specialization);
  static std::shared_ptr<DeviceFunctionInfo>
  LinkUnresolved(const UnresolvedLookupExpr *ULE,
                 std::optional<unsigned int> NumArgs);
  static std::shared_ptr<DeviceFunctionInfo>
  LinkRedecls(const FunctionDecl *FD);
  static std::shared_ptr<DeviceFunctionInfo>
  LinkTemplateDecl(const FunctionTemplateDecl *FTD);
  static std::shared_ptr<DeviceFunctionInfo> LinkExplicitInstantiation(
      const FunctionDecl *Specialization, const FunctionTypeLoc &FTL,
      const ParsedAttributes &Attrs, const TemplateArgumentListInfo &TAList);
  std::shared_ptr<DeviceFunctionInfo> getFuncInfo() const { return FuncInfo; }

  virtual void emplaceReplacement();
  static void reset() { FuncInfoMap.clear(); }

  using DeclList = std::vector<std::shared_ptr<DeviceFunctionDecl>>;

  static void LinkDecl(const FunctionDecl *FD, DeclList &List,
                       std::shared_ptr<DeviceFunctionInfo> &Info);
  static void LinkDecl(const NamedDecl *ND, DeclList &List,
                       std::shared_ptr<DeviceFunctionInfo> &Info);
  static void LinkDecl(const FunctionTemplateDecl *FTD, DeclList &List,
                       std::shared_ptr<DeviceFunctionInfo> &Info);
  static void LinkRedecls(const FunctionDecl *ND, DeclList &List,
                          std::shared_ptr<DeviceFunctionInfo> &Info);

  template <class IteratorRange>
  static std::shared_ptr<DeviceFunctionInfo>
  LinkDeclRange(IteratorRange &&Range, const std::string &FunctionName) {
    std::shared_ptr<DeviceFunctionInfo> Info;
    DeclList List;
    LinkDeclRange(std::move(Range), List, Info);
    if (List.empty())
      return Info;
    if (!Info)
      Info = std::make_shared<DeviceFunctionInfo>(
          List[0]->ParamsNum, List[0]->NonDefaultParamNum, FunctionName);
    for (auto &D : List)
      D->setFuncInfo(Info);
    return Info;
  }

  template <class IteratorRange>
  static void LinkDeclRange(IteratorRange &&Range, DeclList &List,
                            std::shared_ptr<DeviceFunctionInfo> &Info) {
    for (auto D : Range)
      LinkDecl(D, List, Info);
  }
  void setFuncInfo(std::shared_ptr<DeviceFunctionInfo> Info);
  void insertWrapper();
  void collectInfoForWrapper(const FunctionDecl *FD);
  virtual ~DeviceFunctionDecl() = default;

protected:
  const FormatInfo &getFormatInfo();
  void buildTextureObjectParamsInfo(const ArrayRef<ParmVarDecl *> &Parms);

  template <class AttrsT>
  void buildReplaceLocInfo(const FunctionTypeLoc &FTL, const AttrsT &Attrs);

  virtual std::string getExtraParameters(LocInfo LI);

  unsigned Offset;
  unsigned OffsetForAttr;
  const clang::tooling::UnifiedPath FilePath;
  unsigned ParamsNum;
  unsigned ReplaceOffset;
  unsigned ReplaceLength;
  bool IsReplaceFollowedByPP = false;
  unsigned NonDefaultParamNum;
  bool IsDefFilePathNeeded = false;
  std::vector<std::shared_ptr<TextureObjectInfo>> TextureObjectList;
  FormatInfo FormatInformation;
  bool HasBody = false;
  size_t DeclEnd = 0;
  std::map<int, std::string> TemplateParameterDefaultValueMap;
  std::map<int, std::string> ParameterDefaultValueMap;
  static std::shared_ptr<DeviceFunctionInfo> &getFuncInfo(const FunctionDecl *);
  static std::unordered_map<std::string, std::shared_ptr<DeviceFunctionInfo>>
      FuncInfoMap;

private:
  std::shared_ptr<DeviceFunctionInfo> &FuncInfo;
};

class ExplicitInstantiationDecl : public DeviceFunctionDecl {
  std::vector<TemplateArgumentInfo> InstantiationArgs;

public:
  ExplicitInstantiationDecl(unsigned Offset,
                            const clang::tooling::UnifiedPath &FilePathIn,
                            const FunctionTypeLoc &FTL,
                            const ParsedAttributes &Attrs,
                            const FunctionDecl *Specialization,
                            const TemplateArgumentListInfo &TAList)
      : DeviceFunctionDecl(Offset, FilePathIn, FTL, Attrs, Specialization) {
    initTemplateArgumentList(TAList, Specialization);
  }
  static void processFunctionTypeLoc(const FunctionTypeLoc &);
  static void processTemplateArgumentList(const TemplateArgumentListInfo &);

private:
  void initTemplateArgumentList(const TemplateArgumentListInfo &TAList,
                                const FunctionDecl *Specialization);
  std::string getExtraParameters(LocInfo LI) override;
};

// device function info includes parameters num, memory variable and call
// expression in the function.
class DeviceFunctionInfo {
  struct ParameterProps {
    bool IsReferenced = false;
  };

public:
  DeviceFunctionInfo(size_t ParamsNum, size_t NonDefaultParamNum,
                     std::string FunctionName);

  bool ConstructGraphVisited = false;
  unsigned int KernelCallBlockDim = 1;

  std::shared_ptr<CallFunctionExpr> findCallee(const CallExpr *C);
  template <class CallT>
  inline std::shared_ptr<CallFunctionExpr> addCallee(const CallT *C) {
    auto CallLocInfo = DpctGlobalInfo::getLocInfo(C);
    auto Call =
        insertObject(CallExprMap, CallLocInfo.second, CallLocInfo.first, C);
    Call->buildCallExprInfo(C);
    return Call;
  }
  void addVar(std::shared_ptr<MemVarInfo> Var) { VarMap.addVar(Var); }
  void setItem() { VarMap.setItem(); }
  void setStream() { VarMap.setStream(); }
  void setSync() { VarMap.setSync(); }
  void setBF64() { VarMap.setBF64(); }
  void setBF16() { VarMap.setBF16(); }
  void setGlobalMemAcc() { VarMap.setGlobalMemAcc(); }
  void addTexture(std::shared_ptr<TextureInfo> Tex) { VarMap.addTexture(Tex); }
  MemVarMap &getVarMap() { return VarMap; }
  std::shared_ptr<TextureObjectInfo> getTextureObject(unsigned Idx);
  std::shared_ptr<StructureTextureObjectInfo> getBaseTextureObject() const {
    return BaseObjectTexture;
  }
  void setCallGroupFunctionInControlFlow(bool Val = true) {
    CallGroupFunctionInControlFlow = Val;
  }
  bool hasCallGroupFunctionInControlFlow() const {
    return CallGroupFunctionInControlFlow;
  }
  void setHasSideEffectsAnalyzed(bool Val = true) {
    HasCheckedCallGroupFunctionInControlFlow = Val;
  }
  bool hasSideEffectsAnalyzed() const {
    return HasCheckedCallGroupFunctionInControlFlow;
  }

  void buildInfo();
  bool hasParams() { return ParamsNum != 0; }
  bool isBuilt() { return IsBuilt; }
  void setBuilt() { IsBuilt = true; }
  bool isLambda() { return IsLambda; }
  void setLambda() { IsLambda = true; }
  bool isInlined() { return IsInlined; }
  void setInlined() { IsInlined = true; }
  bool isKernel() { return IsKernel; }
  void setKernel() { IsKernel = true; }
  bool isKernelInvoked() { return IsKernelInvoked; }
  void setKernelInvoked() { IsKernelInvoked = true; }
  std::string getExtraParameters(const clang::tooling::UnifiedPath &Path,
                                 LocInfo LI,
                                 FormatInfo FormatInformation = FormatInfo());
  std::string
  getExtraParameters(const clang::tooling::UnifiedPath &Path,
                     const std::vector<TemplateArgumentInfo> &TAList,
                     LocInfo LI, FormatInfo FormatInformation = FormatInfo());
  void setDefinitionFilePath(const clang::tooling::UnifiedPath &Path) {
    DefinitionFilePath = Path;
  }
  const clang::tooling::UnifiedPath &getDefinitionFilePath() {
    return DefinitionFilePath;
  }

  void setNeedSyclExternMacro() { NeedSyclExternMacro = true; }
  bool IsSyclExternMacroNeeded() { return NeedSyclExternMacro; }
  void setAlwaysInlineDevFunc() { AlwaysInlineDevFunc = true; }
  bool IsAlwaysInlineDevFunc() { return AlwaysInlineDevFunc; }
  void setForceInlineDevFunc() { ForceInlineDevFunc = true; }
  bool IsForceInlineDevFunc() { return ForceInlineDevFunc; }
  void setOverloadedOperatorKind(OverloadedOperatorKind Kind) {
    OO_Kind = Kind;
  }
  OverloadedOperatorKind getOverloadedOperatorKind() { return OO_Kind; }
  void merge(std::shared_ptr<DeviceFunctionInfo> Other);
  size_t ParamsNum;
  size_t NonDefaultParamNum;
  GlobalMap<CallFunctionExpr> &getCallExprMap() { return CallExprMap; }
  void addSubGroupSizeRequest(unsigned int Size, SourceLocation Loc,
                              std::string APIName, std::string VarName = "");
  std::vector<std::tuple<unsigned int, clang::tooling::UnifiedPath,
                         unsigned int, std::string, std::string>> &
  getSubGroupSize() {
    return RequiredSubGroupSize;
  }
  bool isParameterReferenced(unsigned int Index);
  void setParameterReferencedStatus(unsigned int Index, bool IsReferenced);
  std::string getFunctionName() { return FunctionName; }
  void collectInfoForWrapper(const FunctionDecl *FD);
  void setModuleUsed() { ModuleUsed = true; }
  bool isModuleUsed() { return ModuleUsed; }
  std::shared_ptr<DeviceFunctionInfoForWrapper>
  getDeviceFunctionInfoForWrapper() {
    return DFInfoForWrapper;
  }

private:
  void mergeCalledTexObj(
      std::shared_ptr<StructureTextureObjectInfo> BaseObj,
      const std::vector<std::shared_ptr<TextureObjectInfo>> &TexObjList);

  void mergeTextureObjectList(
      const std::vector<std::shared_ptr<TextureObjectInfo>> &Other);

  bool IsBuilt;
  clang::tooling::UnifiedPath DefinitionFilePath;
  bool NeedSyclExternMacro = false;
  bool AlwaysInlineDevFunc = false;
  bool ForceInlineDevFunc = false;
  // subgroup size, filepath, offset, API name, var name
  std::vector<std::tuple<unsigned int, clang::tooling::UnifiedPath,
                         unsigned int, std::string, std::string>>
      RequiredSubGroupSize;
  GlobalMap<CallFunctionExpr> CallExprMap;
  MemVarMap VarMap;

  std::shared_ptr<StructureTextureObjectInfo> BaseObjectTexture;
  std::vector<std::shared_ptr<TextureObjectInfo>> TextureObjectList;
  std::vector<ParameterProps> ParametersProps;
  std::string FunctionName;
  bool IsInlined = false;
  bool IsLambda;
  bool IsKernel = false;
  bool IsKernelInvoked = false;
  bool CallGroupFunctionInControlFlow = false;
  bool HasCheckedCallGroupFunctionInControlFlow = false;
  OverloadedOperatorKind OO_Kind = OverloadedOperatorKind::OO_None;
  bool ModuleUsed = false;
  std::shared_ptr<DeviceFunctionInfoForWrapper> DFInfoForWrapper = nullptr;
};

class KernelCallExpr : public CallFunctionExpr {
public:
  bool IsInMacroDefine = false;
  bool NeedLambda = false;
  bool IsForWrapper = false;
  bool NeedDefaultRetValue = false;

private:
  struct ArgInfo {
    ArgInfo(const ParmVarDecl *PVD, KernelArgumentAnalysis &Analysis,
            const Expr *Arg, bool Used, int Index, KernelCallExpr *BASE);
    ArgInfo(const ParmVarDecl *PVD, const std::string &ArgsArrayName,
            KernelCallExpr *Kernel);
    ArgInfo(const ParmVarDecl *PVD, KernelCallExpr *Kernel);
    ArgInfo(std::shared_ptr<TextureObjectInfo> Obj, KernelCallExpr *BASE,
            std::string ArgStr);
    inline const std::string &getArgString() const;
    inline const std::string &getTypeString() const;
    inline std::string getIdStringWithIndex() const {
      return buildString(IdString, "ct", Index);
    }
    inline std::string getIdStringWithSuffix(const std::string &Suffix) const {
      return buildString(IdString, Suffix, "_ct", Index);
    }
    bool IsPointer;
    // If the pointer is used as lvalue after its most recent memory allocation
    bool IsRedeclareRequired;
    bool IsUsedAsLvalueAfterMalloc;
    bool IsDefinedOnDevice = false;
    bool TryGetBuffer = false;
    std::string ArgString;
    std::string TypeString;
    std::string IdString;
    int Index;
    int ArgSize = 0;
    bool IsDeviceRandomGeneratorType = false;
    bool HasImplicitConversion = false;
    bool IsDoublePointer = false;
    bool IsDependentType = false;

    std::shared_ptr<TextureObjectInfo> Texture;
  };

  void print(KernelPrinter &Printer);
  void printSubmit(KernelPrinter &Printer);
  void printSubmitLambda(KernelPrinter &Printer);
  void printParallelFor(KernelPrinter &Printer, bool IsInSubmit);
  void printKernel(KernelPrinter &Printer);
  template <typename IDTy, typename... Ts>
  void printWarningMessage(KernelPrinter &Printer, IDTy MsgID, Ts &&...Vals);
  template <class T> void printStreamBase(T &Printer);

public:
  KernelCallExpr(unsigned Offset, const clang::tooling::UnifiedPath &FilePath,
                 const CUDAKernelCallExpr *KernelCall);

  void addAccessorDecl();
  void buildInfo();
  void setKernelCallDim();
  void buildUnionFindSet();
  void addReplacements();
  std::string getExtraArguments() override;

  const std::vector<ArgInfo> &getArgsInfo();
  int calculateOriginArgsSize() const;

  std::string getReplacement();

  void setEvent(const std::string &E) { Event = E; }
  const std::string &getEvent() { return Event; }
  void setSync(bool Sync = true) { IsSync = Sync; }
  bool isSync() { return IsSync; }

  static std::shared_ptr<KernelCallExpr> buildFromCudaLaunchKernel(
      const std::pair<clang::tooling::UnifiedPath, unsigned> &LocInfo,
      const CallExpr *, bool IsAssigned = false);
  static std::shared_ptr<KernelCallExpr>
  buildForWrapper(clang::tooling::UnifiedPath, const FunctionDecl *);
  void setTemplateArgsStrForWrapper(std::string Str) {
    TemplateArgsStrForWrapper = std::move(Str);
  }
  unsigned int GridDim = 3;
  unsigned int BlockDim = 3;
  void setEmitSizeofWarningFlag(bool Flag) { EmitSizeofWarning = Flag; }

private:
  KernelCallExpr(unsigned Offset, const clang::tooling::UnifiedPath &FilePath)
      : CallFunctionExpr(Offset, FilePath, nullptr), IsSync(false) {}
  void buildArgsInfoFromArgsArray(const FunctionDecl *FD,
                                  const Expr *ArgsArray) {}
  void buildArgsInfo(const CallExpr *CE);
  bool isDefaultStream() const {
    return StringRef(ExecutionConfig.Stream).starts_with("{{NEEDREPLACEQ") ||
           ExecutionConfig.IsDefaultStream;
  }
  bool isQueuePtr() const { return ExecutionConfig.IsQueuePtr; }
  std::string getQueueStr() const;

  void buildKernelInfo(const CUDAKernelCallExpr *KernelCall);
  void setIsInMacroDefine(const CUDAKernelCallExpr *KernelCall);
  void setNeedAddLambda(const CUDAKernelCallExpr *KernelCall);
  void setNeedAddLambda();
  void setNeedDefaultRet();
  void buildNeedBracesInfo(const CallExpr *KernelCall);
  void buildLocationInfo(const CallExpr *KernelCall);
  template <class ArgsRange>
  void buildExecutionConfig(const ArgsRange &ConfigArgs,
                            const CallExpr *KernelCall);

  void removeExtraIndent();
  void addDevCapCheckStmt();
  void addPropertiesStmt();
  void addAccessorDecl(MemVarInfo::VarScope Scope);
  void addAccessorDecl(std::shared_ptr<MemVarInfo> VI);
  void addStreamDecl();

  void buildKernelArgsStmt();

  struct {
    std::string LocHash;
    std::string NL;
    std::string Indent;
  } LocInfo;
  // true, if migrated SYCL code block need extra { }
  bool NeedBraces = true;
  struct {
    std::string Config[6] = {"", "", "", "0", "", ""};
    std::string &GroupSize = Config[0];
    std::string &LocalSize = Config[1];
    std::string &ExternMemSize = Config[2];
    std::string &Stream = Config[3];
    bool LocalDirectRef = false, GroupDirectRef = false;
    std::string GroupSizeFor1D = "";
    std::string LocalSizeFor1D = "";
    std::string &NdRange = Config[4];
    std::string Properties = "";
    std::string &SubGroupSize = Config[5];
    bool IsDefaultStream = false;
    bool IsQueuePtr = true;
  } ExecutionConfig;

  std::vector<ArgInfo> ArgsInfo;

  std::string Event;
  bool IsSync;

  class SubmitStmtsList {
  public:
    StmtList StreamList;
    StmtList SyncList;
    StmtList RangeList;
    StmtList MemoryList;
    StmtList PtrList;
    StmtList AccessorList;
    StmtList TextureList;
    StmtList SamplerList;
    StmtList NdRangeList;
    StmtList CommandGroupList;
    bool ImplicitSyncFlag = false;
    bool DefaultStreamFlag = false;
    KernelPrinter &print(KernelPrinter &Printer);
    bool empty() const noexcept;

  private:
    KernelPrinter &printList(KernelPrinter &Printer, const StmtList &List,
                             StringRef Comments = "");
  };
  SubmitStmtsList SubmitStmts;

  class OuterStmtsList {
  public:
    StmtList ExternList;
    StmtList InitList;
    StmtList OthersList;

    KernelPrinter &print(KernelPrinter &Printer);
    bool empty() const noexcept;

  private:
    KernelPrinter &printList(KernelPrinter &Printer, const StmtList &List,
                             StringRef Comments = "");
  };
  OuterStmtsList OuterStmts;
  StmtList KernelStmts;
  std::string KernelArgs;
  std::string TemplateArgsStrForWrapper;
  int TotalArgsSize = 0;
  bool EmitSizeofWarning = false;
  unsigned int SizeOfHighestDimension = 0;
};

class CudaMallocInfo {
public:
  CudaMallocInfo(unsigned Offset, const clang::tooling::UnifiedPath &FilePath,
                 const VarDecl *VD)
      : Name(VD->getName().str()) {}

  static const VarDecl *getMallocVar(const Expr *Arg);
  static const VarDecl *getDecl(const Expr *E);
  void setSizeExpr(const Expr *SizeExpression);
  void setSizeExpr(const Expr *N, const Expr *ElemSize);
  std::string getAssignArgs(const std::string &TypeName);

private:
  std::string Size;
  std::string Name;
};

/// Find the innermost FunctionDecl's child node (CompoundStmt node) where \S
/// is located. If there is no CompoundStmt of FunctionDecl out of \S, return
/// nullptr.
/// Caller should make sure that /S is not nullptr.
template <typename T>
inline const clang::CompoundStmt *findInnerMostBlock(const T *S) {
  auto &Context = DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  std::vector<DynTypedNode> AncestorNodes;
  while (Parents.size() >= 1) {
    AncestorNodes.push_back(Parents[0]);
    Parents = Context.getParents(Parents[0]);
  }

  for (unsigned int i = 0; i < AncestorNodes.size(); ++i) {
    if (auto CS = AncestorNodes[i].get<CompoundStmt>()) {
      if (i + 1 < AncestorNodes.size() &&
          (AncestorNodes[i + 1].get<FunctionDecl>() ||
           AncestorNodes[i + 1].get<CXXMethodDecl>() ||
           AncestorNodes[i + 1].get<CXXConstructorDecl>() ||
           AncestorNodes[i + 1].get<CXXDestructorDecl>())) {
        return CS;
      }
    }
  }
  return nullptr;
}

template <typename T>
inline DpctGlobalInfo::HelperFuncReplInfo
generateHelperFuncReplInfo(const T *S) {
  DpctGlobalInfo::HelperFuncReplInfo Info;
  if (!S) {
    Info.IsLocationValid = false;
    return Info;
  }

  auto CS = findInnerMostBlock(S);
  if (!CS) {
    Info.IsLocationValid = false;
    return Info;
  }

  auto EndOfLBrace = CS->getLBracLoc().getLocWithOffset(1);
  if (EndOfLBrace.isMacroID()) {
    Info.IsLocationValid = false;
    return Info;
  }

  Info.IsLocationValid = true;
  std::tie(Info.DeclLocFile, Info.DeclLocOffset) =
      DpctGlobalInfo::getLocInfo(EndOfLBrace);
  return Info;
}

/// If it is not duplicated, return 0.
/// If it is duplicated, return the correct Index which is >= 1.
template <typename T> int getPlaceholderIdx(const T *S) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  SourceLocation Loc = S->getBeginLoc();
  Loc = SM.getExpansionLoc(Loc);

  auto LocInfo = DpctGlobalInfo::getLocInfo(Loc);
  std::string Key = LocInfo.first.getCanonicalPath().str() + ":" +
                    std::to_string(LocInfo.second);
  auto Iter = DpctGlobalInfo::getTempVariableHandledMap().find(Key);
  if (Iter != DpctGlobalInfo::getTempVariableHandledMap().end()) {
    return Iter->second;
  } else {
    return 0;
  }
}

/// return true: update success
/// return false: key already there, map is not changed.
template <typename T> bool UpdatePlaceholderIdxMap(const T *S, int Index) {
  auto Range = getDefinitionRange(S->getBeginLoc(), S->getEndLoc());
  SourceLocation Loc = Range.getBegin();
  auto LocInfo = DpctGlobalInfo::getLocInfo(Loc);
  std::string Key = LocInfo.first.getCanonicalPath().str() + ":" +
                    std::to_string(LocInfo.second);
  auto Iter = DpctGlobalInfo::getTempVariableHandledMap().find(Key);
  if (Iter != DpctGlobalInfo::getTempVariableHandledMap().end()) {
    return true;
  } else {
    DpctGlobalInfo::getTempVariableHandledMap().insert(
        std::make_pair(Key, Index));
    return false;
  }
}

template <typename T> int isPlaceholderIdxDuplicated(const T *S) {
  if (getPlaceholderIdx(S) == 0)
    return false;
  else
    return true;
}

// There are 3 maps are used to record related information:
// unordered_map<int, HelperFuncReplInfo> HelperFuncReplInfoMap,
// unordered_map<string, TempVariableDeclCounter> TempVariableDeclCounterMap and
// unordered_map<string, int> TempVariableHandledMap.
//
// 1. HelperFuncReplInfoMap's key is the Index of each placeholder, its value is
// a HelperFuncReplInfo struct which saved the declaration insert location of
// this placeholder and a boolean represent whether this location is valid.
// 2. TempVariableDeclCounterMap's key is the declaration insert location, it's
// value is a TempVariableDeclCounter which counts how many device declaration
// and queue declaration need be inserted here respectively.
// 3. TempVariableHandledMap's key is the begin location of the declaration or
// statement of each placeholder. This map is to avoid one placeholder to be
// counted more than once. Its value is Index.
//
// The rule of inserting declaration:
// If pair (m, n) means device counter value is n and queue counter value is n,
// using (0,0), (0,1), (1,0), (1,1), (>=2,0), (0,>=2), (>=2,1), (1,>=2) and
// (>=2,>=2) can construct a graph.
// Then there are 5 edges will need insert declaration:
// (1,0) to (>=2,0) and (1,1) to (>=2,1) need add device declaration
// (0,1) to (0,>=2) and (1,1) to (1,>=2) need add both declaration
// (>=2,1) to (>=2,>=2) need add queue declaration
template <typename T>
inline void buildTempVariableMap(int Index, const T *S, HelperFuncType HFT) {
  if (UpdatePlaceholderIdxMap(S, Index)) {
    return;
  }

  DpctGlobalInfo::HelperFuncReplInfo HFInfo = generateHelperFuncReplInfo(S);

  if (!HFInfo.IsLocationValid)
    return;

  DpctGlobalInfo::getHelperFuncReplInfoMap().insert(
      std::make_pair(Index, HFInfo));
  std::string KeyForDeclCounter = HFInfo.DeclLocFile.getCanonicalPath().str() +
                                  ":" + std::to_string(HFInfo.DeclLocOffset);

  auto &Counter =
      DpctGlobalInfo::getTempVariableDeclCounterMap()[KeyForDeclCounter];
  switch (HFT) {
  case HelperFuncType::HFT_DefaultQueue:
  case HelperFuncType::HFT_DefaultQueuePtr:
    ++Counter.DefaultQueueCounter;
    break;
  case HelperFuncType::HFT_CurrentDevice:
    ++Counter.CurrentDeviceCounter;
    break;
  default:
    break;
  }
}

} // namespace dpct
} // namespace clang

#endif
