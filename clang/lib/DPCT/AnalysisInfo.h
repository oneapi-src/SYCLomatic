//===--------------- AnalysisInfo.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_ANALYSIS_INFO_H
#define DPCT_ANALYSIS_INFO_H

#include "Error.h"
#include "ExprAnalysis.h"
#include "ExtReplacements.h"
#include "InclusionHeaders.h"
#include "LibraryAPIMigration.h"
#include "Rules.h"
#include "SaveNewFiles.h"
#include "Statics.h"
#include "Utility.h"
#include "ValidateArguments.h"
#include <bitset>
#include <optional>
#include <unordered_set>
#include <vector>

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
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
enum class HelperFuncType : int {
  HFT_InitValue = 0,
  HFT_DefaultQueue = 1,
  HFT_CurrentDevice = 2
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
class DeviceFunctionDeclInModule;
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
  bool isCalledInHost = false;
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

using ReplTy = std::map<std::string, tooling::Replacements>;

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
//   Global Variable)                      |            (inherit from CallFunctionExpr)
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
      : Repls(std::make_shared<ExtReplacements>(FilePathIn)),
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
  const clang::tooling::UnifiedPath &getFilePath();

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
  std::shared_ptr<ExtReplacements> getRepls();
  size_t getFileSize() const;
  std::string &getFileContent();

  // Header inclusion directive insertion functions
  void setFileEnterOffset(unsigned Offset);
  void setFirstIncludeOffset(unsigned Offset);
  void setLastIncludeOffset(unsigned Offset);
  void setHeaderInserted(HeaderType Header);
  void setMathHeaderInserted(bool B = true);
  void setAlgorithmHeaderInserted(bool B = true);
  void setTimeHeaderInserted(bool B = true);

  void concatHeader(llvm::raw_string_ostream &OS) {}
  template <class FirstT, class... Args>
  void concatHeader(llvm::raw_string_ostream &OS, FirstT &&First,
                    Args &&...Arguments) {
    appendString(OS, "#include ", std::forward<FirstT>(First), getNL());
    concatHeader(OS, std::forward<Args>(Arguments)...);
  }

  std::optional<HeaderType> findHeaderType(StringRef Header);
  StringRef getHeaderSpelling(HeaderType Type);

  // Insert one or more header inclusion directives at a specified offset
  template <typename ReplacementT>
  void insertHeader(ReplacementT &&Repl, unsigned Offset,
                    InsertPosition InsertPos = IP_Left) {
    auto R = std::make_shared<ExtReplacement>(
        FilePath, Offset, 0, std::forward<ReplacementT>(Repl), nullptr);
    R->setSYCLHeaderNeeded(false);
    R->setInsertPosition(InsertPos);
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

  void insertHeader(HeaderType Type, unsigned Offset);
  void insertHeader(HeaderType Type);

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
  StringRef getLineString(unsigned LineNumber);

  // Get line number by offset
  unsigned getLineNumber(unsigned Offset);
  // Set line range info of replacement
  void setLineRange(ExtReplacements::SourceLineRange &LineRange,
                    std::shared_ptr<ExtReplacement> Repl);
  void insertIncludedFilesInfo(std::shared_ptr<DpctFileInfo> Info);

  std::map<const CompoundStmt *, MemcpyOrderAnalysisInfo> &
  getMemcpyOrderAnalysisResultMap();
  std::map<std::string, std::vector<std::pair<unsigned int, unsigned int>>> &
  getFuncDeclRangeMap();
  std::map<unsigned int, EventSyncTypeInfo> &getEventSyncTypeMap();
  std::map<unsigned int, TimeStubTypeInfo> &getTimeStubTypeMap();
  std::map<unsigned int, BuiltinVarInfo> &getBuiltinVarInfoMap();
  std::unordered_set<std::shared_ptr<DpctFileInfo>> &getIncludedFilesInfoSet();
  std::set<unsigned int> &getSpBLASSet();
  std::unordered_set<std::shared_ptr<TextModification>> &
  getConstantMacroTMSet();
  std::vector<tooling::Replacement> &getReplacements();
  std::unordered_map<std::string, std::tuple<unsigned int, std::string, bool>> &
  getAtomicMap();
  void setAddOneDplHeaders(bool Value);
  std::vector<std::pair<unsigned int, unsigned int>> &getTimeStubBounds();
  std::vector<std::pair<unsigned int, unsigned int>> &getExternCRanges();
  std::vector<RnnBackwardFuncInfo> &getRnnBackwardFuncInfo();
  void setRTVersionValue(std::string Value);
  std::string getRTVersionValue();
  void setCCLVerValue(std::string Value);
  std::string getCCLVerValue();

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
  std::shared_ptr<ExtReplacements> Repls;
  size_t FileSize = 0;
  std::vector<SourceLineInfo> Lines;

  clang::tooling::UnifiedPath FilePath;
  std::string FileContentCache;

  unsigned FirstIncludeOffset = 0;
  unsigned LastIncludeOffset = 0;
  bool HasInclusionDirective = false;
  std::vector<std::string> InsertedHeaders;
  std::bitset<32> HeaderInsertedBitMap;
  std::bitset<32> UsingInsertedBitMap;
  bool AddOneDplHeaders = false;
  std::vector<std::shared_ptr<ExtReplacement>> IncludeDirectiveInsertions;
  std::vector<std::pair<unsigned int, unsigned int>> ExternCRanges;
  std::vector<RnnBackwardFuncInfo> RBFuncInfo;
  std::string RTVersionValue = "";
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
              "",
              buildString(MapNames::getDpctNamespace(), "get_",
                          DpctGlobalInfo::getDeviceQueueName(), "()"),
              MapNames::getDpctNamespace() + "get_current_device()"} {}
    int DefaultQueueCounter = 0;
    int CurrentDeviceCounter = 0;
    std::string PlaceholderStr[3];
  };

  static std::string removeSymlinks(clang::FileManager &FM,
                                    std::string FilePathStr);
  static bool isInRoot(SourceLocation SL);
  static bool isInRoot(clang::tooling::UnifiedPath FilePath);
  static bool isInAnalysisScope(SourceLocation SL);
  static bool isInAnalysisScope(clang::tooling::UnifiedPath FilePath);
  static bool isExcluded(const clang::tooling::UnifiedPath &FilePath);
  // TODO: implement one of this for each source language.
  static bool isInCudaPath(SourceLocation SL);
  // TODO: implement one of this for each source language.
  static bool isInCudaPath(clang::tooling::UnifiedPath FilePath);

  static void setInRoot(const clang::tooling::UnifiedPath &InRootPath);
  static const clang::tooling::UnifiedPath &getInRoot();
  static void setOutRoot(const clang::tooling::UnifiedPath &OutRootPath);
  static const clang::tooling::UnifiedPath &getOutRoot();
  static void
  setAnalysisScope(const clang::tooling::UnifiedPath &InputAnalysisScope);
  static const clang::tooling::UnifiedPath &getAnalysisScope();
  static void addChangeExtensions(const std::string &Extension);
  static const std::unordered_set<std::string> &getChangeExtensions();
  // TODO: implement one of this for each source language.
  static void setCudaPath(const clang::tooling::UnifiedPath &InputCudaPath);
  // TODO: implement one of this for each source language.
  static const clang::tooling::UnifiedPath &getCudaPath();
  static const std::string getCudaVersion();

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
  static const std::string &getDeviceQueueName();
  static const std::string &getStreamName();
  static const std::string &getSyncName();
  static const std::string &getInRootHash();
  static void setContext(ASTContext &C);
  static void setRuleFile(const std::string &Path);
  static ASTContext &getContext();
  static SourceManager &getSourceManager();
  static FileManager &getFileManager();
  static bool isKeepOriginCode();
  static void setKeepOriginCode(bool KOC = true);
  static bool isSyclNamedLambda();
  static void setSyclNamedLambda(bool SNL = true);
  static void setCheckUnicodeSecurityFlag(bool CUS);
  static bool getCheckUnicodeSecurityFlag();
  static void setEnablepProfilingFlag(bool EP);
  static bool getEnablepProfilingFlag();
  static bool getGuessIndentWidthMatcherFlag();
  static void setGuessIndentWidthMatcherFlag(bool Flag = true);
  static void setIndentWidth(unsigned int W);
  static unsigned int getIndentWidth();
  static void insertKCIndentWidth(unsigned int W);
  static unsigned int getKCIndentWidth();
  static UsmLevel getUsmLevel();
  static void setUsmLevel(UsmLevel UL);
  static clang::CudaVersion getSDKVersion();
  static void setSDKVersion(clang::CudaVersion V);
  static bool isIncMigration();
  static void setIsIncMigration(bool Flag);
  static bool isQueryAPIMapping();
  static void setIsQueryAPIMapping(bool Flag);
  static bool needDpctDeviceExt();
  static void setNeedDpctDeviceExt();
  static unsigned int getAssumedNDRangeDim();
  static void setAssumedNDRangeDim(unsigned int Dim);
  static bool getUsingExtensionDE(DPCPPExtensionsDefaultEnabled Ext);
  static void setExtensionDEFlag(unsigned Flag);
  static unsigned getExtensionDEFlag();
  static bool getUsingExtensionDD(DPCPPExtensionsDefaultDisabled Ext);
  static void setExtensionDDFlag(unsigned Flag);
  static unsigned getExtensionDDFlag();
  template <ExperimentalFeatures Exp> static bool getUsingExperimental() {
    return ExperimentalFlag & (1 << static_cast<unsigned>(Exp));
  }
  static void setExperimentalFlag(unsigned Flag);
  static unsigned getExperimentalFlag();
  static bool getHelperFuncPreference(HelperFuncPreference HFP);
  static void setHelperFuncPreferenceFlag(unsigned Flag);
  static unsigned getHelperFuncPreferenceFlag();
  static format::FormatRange getFormatRange();
  static void setFormatRange(format::FormatRange FR);
  static DPCTFormatStyle getFormatStyle();
  static void setFormatStyle(DPCTFormatStyle FS);
  // Processing the folder or file by following rules:
  // Rule1: For {child path, parent path}, only parent path will be kept.
  // Rule2: Ignore invalid path.
  // Rule3: If path is not in --in-root, then ignore it.
  static void setExcludePath(std::vector<std::string> ExcludePathVec);
  static std::unordered_map<std::string, bool> getExcludePath();
  static std::set<ExplicitNamespace> getExplicitNamespaceSet();
  static void
  setExplicitNamespace(std::vector<ExplicitNamespace> NamespacesVec);
  static bool isCtadEnabled();
  static void setCtadEnabled(bool Enable = true);
  static bool isGenBuildScript();
  static void setGenBuildScriptEnabled(bool Enable = true);
  static bool IsMigrateCmakeScriptEnabled();
  static void setMigrateCmakeScriptEnabled(bool Enable = true);
  static bool IsMigrateCmakeScriptOnlyEnabled();
  static void setMigrateCmakeScriptOnlyEnabled(bool Enable = true);
  static bool isCommentsEnabled();
  static void setCommentsEnabled(bool Enable = true);
  static bool isDPCTNamespaceTempEnabled();
  static void setDPCTNamespaceTempEnabled();
  static std::unordered_set<std::string> &getPrecAndDomPairSet();
  static bool isMKLHeaderUsed();
  static void setMKLHeaderUsed(bool Used = true);
  static int getSuffixIndexInitValue(std::string FileNameAndOffset);
  static void updateInitSuffixIndexInRule(int InitVal);
  static int getSuffixIndexInRuleThenInc();
  static int getSuffixIndexGlobalThenInc();
  static const std::string &getGlobalQueueName();
  static const std::string &getGlobalDeviceName();
  static std::string getStringForRegexReplacement(StringRef);
  static void
  setCodeFormatStyle(const clang::format::FormatStyle &Style);
  static clang::format::FormatStyle getCodeFormatStyle();

  template <class TargetTy, class NodeTy>
  static inline const TargetTy *
  findAncestor(const NodeTy *N,
               const std::function<bool(const DynTypedNode &)> &Condition) {
    if (!N)
      return nullptr;

    auto &Context = getContext();
    clang::DynTypedNodeList Parents = Context.getParents(*N);
    while (!Parents.empty()) {
      auto &Cur = Parents[0];
      if (Condition(Cur))
        return Cur.get<TargetTy>();
      Parents = Context.getParents(Cur);
    }

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
  static const TargetTy *findAncestor(const NodeTy *Node) {
    return findAncestor<TargetTy>(Node, [&](const DynTypedNode &Cur) -> bool {
      return Cur.get<TargetTy>();
    });
  }
  template <class TargetTy, class NodeTy>
  static const TargetTy *findParent(const NodeTy *Node) {
    return findAncestor<TargetTy>(
        Node, [](const DynTypedNode &Cur) -> bool { return true; });
  }

  template <typename TargetTy, typename NodeTy>
  static bool isAncestor(TargetTy *AncestorNode, NodeTy *Node) {
    return findAncestor<TargetTy>(Node, [&](const DynTypedNode &Cur) -> bool {
      if (Cur.get<TargetTy>() == AncestorNode) {
        return true;
      } else {
        return false;
      };
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
  getLocInfo(const TypeLoc &TL, bool *IsInvalid = nullptr /*out*/);
  // Return the absolute path of \p ID
  static std::optional<clang::tooling::UnifiedPath> getAbsolutePath(FileID ID);
  // Return the absolute path of \p File
  static std::optional<clang::tooling::UnifiedPath>
  getAbsolutePath(const FileEntry &File);
  static std::pair<clang::tooling::UnifiedPath, unsigned>
  getLocInfo(SourceLocation Loc, bool *IsInvalid = nullptr /* out */);
  static std::string getTypeName(QualType QT,
                                        const ASTContext &Context);
  static std::string getTypeName(QualType QT);
  static std::string getUnqualifiedTypeName(QualType QT,
                                            const ASTContext &Context);
  static std::string getUnqualifiedTypeName(QualType QT);
  /// This function will return the replaced type name with qualifiers.
  /// Currently, since clang do not support get the order of original
  /// qualifiers, this function will follow the behavior of
  /// clang::QualType.print(), in other words, the behavior is that the
  /// qualifiers(const, volatile...) will occur before the simple type(int,
  /// bool...) regardless its order in origin code.
  /// \param [in] QT The input qualified type which need migration.
  /// \param [in] Context The AST context.
  /// \return The replaced type name string with qualifiers.
  static std::string getReplacedTypeName(QualType QT,
                                                const ASTContext &Context);
  static std::string getReplacedTypeName(QualType QT);
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
  std::shared_ptr<DeviceFunctionDecl>
  insertDeviceFunctionDeclInModule(const FunctionDecl *FD);

  // Build kernel and device function declaration replacements and store
  // them.
  void buildKernelInfo();
  void buildReplacements();
  void processCudaArchMacro();
  void generateHostCode(
      std::multimap<unsigned int, std::shared_ptr<clang::dpct::ExtReplacement>>
          &ProcessedReplList,
      HostDeviceFuncLocInfo Info, unsigned ID);
  void postProcess();
  void cacheFileRepl(clang::tooling::UnifiedPath FilePath,
                     std::shared_ptr<ExtReplacements> Repl);
  // Emplace stored replacements into replacement set.
  void emplaceReplacements(ReplTy &ReplSets /*out*/);
  std::shared_ptr<KernelCallExpr> buildLaunchKernelInfo(const CallExpr *);
  void insertCudaMalloc(const CallExpr *CE);
  void insertCublasAlloc(const CallExpr *CE);
  std::shared_ptr<CudaMallocInfo> findCudaMalloc(const Expr *CE);
  void addReplacement(std::shared_ptr<ExtReplacement> Repl);
  CudaArchPPMap &getCudaArchPPInfoMap();
  HDFuncInfoMap &getHostDeviceFuncInfoMap();
  std::unordered_map<std::string, std::shared_ptr<ExtReplacement>> &
  getCudaArchMacroReplMap();
  CudaArchDefMap &getCudaArchDefinedMap();
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
  void insertHeader(SourceLocation Loc, HeaderType Type);
  void insertHeader(SourceLocation Loc, std::string HeaderName);
  static std::unordered_map<
      std::string,
      std::pair<std::pair<clang::tooling::UnifiedPath /*begin file name*/,
                          unsigned int /*begin offset*/>,
                std::pair<clang::tooling::UnifiedPath /*end file name*/,
                          unsigned int /*end offset*/>>> &
  getExpansionRangeBeginMap();
  static std::map<std::string, std::shared_ptr<MacroExpansionRecord>> &
  getExpansionRangeToMacroRecord();
  static std::map<std::string,
                  std::shared_ptr<DpctGlobalInfo::MacroDefRecord>> &
  getMacroTokenToMacroDefineLoc();
  static std::map<std::string, std::string> &
  getFunctionCallInMacroMigrateRecord();
  static std::map<std::string, SourceLocation> &getEndifLocationOfIfdef();
  static std::vector<std::pair<clang::tooling::UnifiedPath, size_t>> &
  getConditionalCompilationLoc();
  static std::map<std::string, unsigned int> &getBeginOfEmptyMacros();
  static std::map<std::string, SourceLocation> &getEndOfEmptyMacros();
  static std::map<std::string, bool> &getMacroDefines();
  static std::set<clang::tooling::UnifiedPath> &getIncludingFileSet();
  static std::set<std::string> &getFileSetInCompiationDB();
  static std::unordered_map<std::string,
                            std::vector<clang::tooling::Replacement>> &
  getFileRelpsMap();
  static std::unordered_map<std::string, std::string> &getDigestMap();
  static std::string getYamlFileName();
  static std::set<std::string> &getGlobalVarNameSet();
  static void removeVarNameInGlobalVarNameSet(const std::string &VarName);
  static bool getDeviceChangedFlag();
  static void setDeviceChangedFlag(bool Flag);
  static std::unordered_map<int, HelperFuncReplInfo> &
  getHelperFuncReplInfoMap();
  static int getHelperFuncReplInfoIndexThenInc();
  static std::unordered_map<std::string, TempVariableDeclCounter> &
  getTempVariableDeclCounterMap();
  // Key: string: file:offset for a replacement.
  // Value: int: index of the placeholder in a replacement.
  static std::unordered_map<std::string, int> &getTempVariableHandledMap();
  static bool getUsingDRYPattern();
  static void setUsingDRYPattern(bool Flag);
  static bool useNdRangeBarrier();
  static bool useFreeQueries();
  static bool useGroupLocalMemory();
  static bool useLogicalGroup();
  static bool useUserDefineReductions();
  static bool useMaskedSubGroupFunction();
  static bool useExtDPLAPI();
  static bool useOccupancyCalculation();
  static bool useExtJointMatrix();
  static bool useExtBFloat16Math();
  static bool useNoQueueDevice();
  static bool useEnqueueBarrier();
  static bool useCAndCXXStandardLibrariesExt();
  static bool useIntelDeviceMath();
  static bool useDeviceInfo();
  static bool useBFloat16();
  std::shared_ptr<DpctFileInfo>
  insertFile(const clang::tooling::UnifiedPath &FilePath);
  std::shared_ptr<DpctFileInfo> getMainFile() const;
  void setMainFile(std::shared_ptr<DpctFileInfo> Main);
  void recordIncludingRelationship(
      const clang::tooling::UnifiedPath &CurrentFileName,
      const clang::tooling::UnifiedPath &IncludedFileName);
  static unsigned int getCudaKernelDimDFIIndexThenInc();
  static void
  insertCudaKernelDimDFIMap(unsigned int Index,
                            std::shared_ptr<DeviceFunctionInfo> Ptr);
  static std::shared_ptr<DeviceFunctionInfo>
  getCudaKernelDimDFI(unsigned int Index);
  static std::set<clang::tooling::UnifiedPath> &getModuleFiles();
  static void setRunRound(unsigned int Round);
  static unsigned int getRunRound();
  static void setNeedRunAgain(bool NRA);
  static bool isNeedRunAgain();
  static std::unordered_map<clang::tooling::UnifiedPath,
                            std::shared_ptr<ExtReplacements>> &
  getFileReplCache();
  void resetInfo();
  static void updateSpellingLocDFIMaps(SourceLocation SL,
                                       std::shared_ptr<DeviceFunctionInfo> DFI);
  static std::unordered_set<std::shared_ptr<DeviceFunctionInfo>>
  getDFIVecRelatedFromSpellingLoc(std::shared_ptr<DeviceFunctionInfo> DFI);
  static unsigned int getColorOption();
  static void setColorOption(unsigned Color);
  std::unordered_map<int, std::shared_ptr<DeviceFunctionInfo>> &
  getCubPlaceholderIndexMap();
  static std::unordered_map<std::string, std::shared_ptr<PriorityReplInfo>> &
  getPriorityReplInfoMap();
  // For PriorityRelpInfo with same key, the Info with low priority will
  // be filtered and the Info with same priority will be merged.
  static void
  addPriorityReplInfo(std::string Key, std::shared_ptr<PriorityReplInfo> Info);
  static void setOptimizeMigrationFlag(bool Flag);
  static bool isOptimizeMigration();
  static std::map<std::string, clang::tooling::OptionInfo> &
  getCurrentOptMap();
  static void setMainSourceYamlTUR(
      std::shared_ptr<clang::tooling::TranslationUnitReplacements> Ptr);
  static std::shared_ptr<clang::tooling::TranslationUnitReplacements>
  getMainSourceYamlTUR();
  static std::unordered_map<
      std::string,
      std::unordered_map<clang::tooling::UnifiedPath, std::vector<unsigned>>> &
  getRnnInputMap();
  static std::unordered_map<clang::tooling::UnifiedPath,
                                   std::vector<clang::tooling::UnifiedPath>> &
  getMainSourceFileMap();
  static std::unordered_map<std::string, bool> &getMallocHostInfoMap();
  static std::map<std::shared_ptr<TextModification>, bool> &
  getConstantReplProcessedFlagMap();
  static std::set<std::string> &getVarUsedByRuntimeSymbolAPISet();
  static void setNeedParenAPI(const std::string &Name);
  static bool isNeedParenAPI(const std::string &Name);
  // #tokens, name of the second token, SourceRange of a macro
  static std::tuple<unsigned int, std::string, SourceRange> LastMacroRecord;
private:
  DpctGlobalInfo();

  DpctGlobalInfo(const DpctGlobalInfo &) = delete;
  DpctGlobalInfo(DpctGlobalInfo &&) = delete;
  DpctGlobalInfo &operator=(const DpctGlobalInfo &) = delete;
  DpctGlobalInfo &operator=(DpctGlobalInfo &&) = delete;

  // Wrapper of isInAnalysisScope for std::function usage.
  static bool checkInAnalysisScope(SourceLocation SL);

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
  static SourceLocation getLocation(const VarDecl *VD);
  static SourceLocation getLocation(const FunctionDecl *FD);
  static SourceLocation getLocation(const FieldDecl *FD);
  static SourceLocation getLocation(const CallExpr *CE);
  // The result will be also stored in KernelCallExpr.BeginLoc
  static SourceLocation getLocation(const CUDAKernelCallExpr *CKC);




  std::shared_ptr<DpctFileInfo> MainFile = nullptr;
  std::unordered_map<clang::tooling::UnifiedPath, std::shared_ptr<DpctFileInfo>>
      FileMap;
  static std::shared_ptr<clang::tooling::TranslationUnitReplacements>
      MainSourceYamlTUR;
  static clang::tooling::UnifiedPath InRoot;
  static clang::tooling::UnifiedPath OutRoot;
  static clang::tooling::UnifiedPath AnalysisScope;
  static std::unordered_set<std::string> ChangeExtensions;
  // TODO: implement one of this for each source language.
  static clang::tooling::UnifiedPath CudaPath;
  static std::string RuleFile;
  static UsmLevel UsmLvl;
  static clang::CudaVersion SDKVersion;
  static bool NeedDpctDeviceExt;
  static bool IsIncMigration;
  static bool IsQueryAPIMapping;
  static unsigned int AssumedNDRangeDim;
  static std::unordered_set<std::string> PrecAndDomPairSet;
  static format::FormatRange FmtRng;
  static DPCTFormatStyle FmtST;
  static bool EnableCtad;
  static bool IsMLKHeaderUsed;
  static bool GenBuildScript;
  static bool MigrateCmakeScript;
  static bool MigrateCmakeScriptOnly;
  static bool EnableComments;
  static std::set<ExplicitNamespace> ExplicitNamespaceSet;

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
  static std::unordered_map<std::string, std::string> DigestMap;
  static const std::string YamlFileName;
  static std::map<std::string, bool> MacroDefines;
  static int CurrentMaxIndex;
  static int CurrentIndexInRule;
  static std::set<clang::tooling::UnifiedPath> IncludingFileSet;
  static std::set<std::string> FileSetInCompiationDB;
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
                            std::shared_ptr<ExtReplacements>>
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
  static unsigned int ColorOption;
  static std::unordered_map<int, std::shared_ptr<DeviceFunctionInfo>>
      CubPlaceholderIndexMap;
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
  static std::set<std::string> VarUsedByRuntimeSymbolAPISet;
  static std::unordered_set<std::string> NeedParenAPISet;
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
  // If NeedSizeFold is true, array size will be folded, but original expression
  // will follow as comments. If NeedSizeFold is false, original size expression
  // will be the size string.
  CtTypeInfo(const TypeLoc &TL, bool NeedSizeFold = false);
  CtTypeInfo(const VarDecl *D, bool NeedSizeFold = false);
  const std::string &getBaseName();
  size_t getDimension();
  std::vector<SizeInfo> &getRange();
  // when there is no arguments, parameter MustArguments determine whether
  // parens will exist. Null string will be returned when MustArguments is
  // false, otherwise "()" will be returned.
  std::string getRangeArgument(const std::string &MemSize, bool MustArguments);
  inline bool isTemplate() const;
  inline bool isPointer() const;
  inline bool isArray() const;
  inline bool isReference() const;
  inline void adjustAsMemType();
  // Get instantiated type name with given template arguments.
  // e.g. X<T>, with T = int, result type will be X<int>.
  std::shared_ptr<CtTypeInfo>
  applyTemplateArguments(const std::vector<TemplateArgumentInfo> &TA);
  bool isWritten() const;
  std::set<HelperFeatureEnum> getHelperFeatureSet();
  bool containSizeofType();
  std::vector<std::string> getArraySizeOriginExprs();
  bool containsTemplateDependentMacro() const;
  bool isConstantQualified() const;

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
  void removeQualifier();

private:
  std::string BaseName;
  std::string BaseNameWithoutQualifiers;
  std::vector<SizeInfo> Range;
  unsigned PointerLevel;
  bool IsReference;
  bool IsTemplate;
  bool TemplateDependentMacro = false;
  bool IsArray = false;

  std::shared_ptr<TemplateDependentStringInfo> TDSI;
  std::set<HelperFeatureEnum> HelperFeatureSet;
  bool ContainSizeofType = false;
  std::vector<std::string> ArraySizeOriginExprs{};
  bool IsConstantQualified = false;
};

// variable info includes name, type and location.
class VarInfo {
public:
  VarInfo(unsigned Offset, const clang::tooling::UnifiedPath &FilePathIn,
          const VarDecl *Var, bool NeedFoldSize = false)
      : FilePath(FilePathIn), Offset(Offset), Name(Var->getName()),
        Ty(std::make_shared<CtTypeInfo>(Var, NeedFoldSize)) {}
  const clang::tooling::UnifiedPath &getFilePath();
  unsigned getOffset();
  const std::string &getName();
  const std::string getNameAppendSuffix();
  std::shared_ptr<CtTypeInfo> &getType();
  std::string getDerefName();
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

  VarAttrKind getAttr();
  VarScope getScope();
  bool isGlobal();
  bool isExtern();
  bool isLocal();
  bool isShared();
  bool isTypeDeclaredLocal();
  bool isAnonymousType();
  const CXXRecordDecl *getDeclOfVarType();
  const DeclStmt *getDeclStmtOfVarType();
  void setLocalTypeName(std::string T);
  std::string getLocalTypeName();
  void setIgnoreFlag(bool Flag);
  bool isIgnore();
  bool isStatic();
  void setName(std::string NewName);
  unsigned int getNewConstVarOffset();
  unsigned int getNewConstVarLength();
  const std::string getConstVarName();
  // Initialize offset and length for __constant__ variable that needs to be
  // renamed.
  void newConstVarInit(const VarDecl *Var);
  std::string getDeclarationReplacement(const VarDecl *);
  std::string getInitStmt();
  std::string getInitStmt(StringRef QueueString);
  std::string getMemoryDecl(const std::string &MemSize);
  std::string getMemoryDecl();
  std::string getExternGlobalVarDecl();
  void appendAccessorOrPointerDecl(const std::string &ExternMemSize,
                                   bool ExternEmitWarning, StmtList &AccList,
                                   StmtList &PtrList);
  std::string getRangeClass();
  std::string getRangeDecl(const std::string &MemSize);
  ParameterStream &getFuncDecl(ParameterStream &PS);
  ParameterStream &getFuncArg(ParameterStream &PS);
  ParameterStream &getKernelArg(ParameterStream &PS);
  std::string getAccessorDataType(bool IsTypeUsedInDevFunDecl = false,
                                  bool NeedCheckExtraConstQualifier = false);
  void setUseHelperFuncFlag(bool Flag);
  bool isUseHelperFunc();

private:
  bool isTreatPointerAsArray();
  static VarAttrKind getAddressAttr(const AttrVec &Attrs);
  void setInitList(const Expr *E, const VarDecl *V);

  std::string getMemoryType();
  std::string getMemoryType(const std::string &MemoryType,
                                   std::shared_ptr<CtTypeInfo> VarType);
  std::string getInitArguments(const std::string &MemSize,
                               bool MustArguments = false);
  const std::string &getMemoryAttr();
  std::string getSyclAccessorType();
  std::string getDpctAccessorType();
  std::string getNameWithSuffix(StringRef Suffix);
  std::string getAccessorName();
  std::string getPtrName();
  std::string getRangeName();
  std::string getArgName();

private:
  // Passing by accessor, value or pointer when invoking kernel.
  // Constant scalar variables are passed by value while other 0/1D variables
  // defined on device memory are passed by pointer in device function calls.
  // The rest are passed by accessor.
  enum DpctAccessMode {
    Value,
    Pointer,
    Accessor,
    Reference,
  };

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

  static std::unordered_map<const DeclStmt *, int> AnonymousTypeDeclStmtMap;
  bool UseHelperFuncFlag = true;
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
  std::string getDataType();
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
  std::shared_ptr<TextureTypeInfo> getType() const;
  virtual std::string getHostDeclString();
  virtual std::string getSamplerDecl();
  virtual std::string getAccessorDecl(const std::string &QueueStr);
  virtual void addDecl(StmtList &AccessorList, StmtList &SamplerList,
                       const std::string &QueueStr);
  ParameterStream &getFuncDecl(ParameterStream &PS);
  ParameterStream &getFuncArg(ParameterStream &PS);
  virtual ParameterStream &getKernelArg(ParameterStream &OS);
  const std::string &getName();
  unsigned getOffset();
  clang::tooling::UnifiedPath getFilePath();
  bool isUseHelperFunc();
};

// texture handle info
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
  std::string getSamplerDecl() override;
  inline unsigned getParamIdx() const;
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
  void addDecl(StmtList &AccessorList, StmtList &SamplerList,
               const std::string &QueueStr) override;
  void setBaseName(StringRef Name);
  StringRef getMemberName();
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
  bool isBase() const;
  bool containsVirtualPointer() const;
  std::shared_ptr<MemberTextureObjectInfo> addMember(const MemberExpr *ME);
  void addDecl(StmtList &AccessorList, StmtList &SamplerList,
               const std::string &Queue) override;
  void addParamDeclReplacement() override;
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

  bool isWritten() const;
  bool isNull() const;
  bool isType() const;
  const std::string &getString() const;
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
                              SourceRange ParentRange = SourceRange()) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    auto Range = getArgSourceRange(Arg);
    auto Begin = Range.getBegin();
    auto End = Range.getEnd();
    if (Begin.isMacroID() && SM.isMacroArgExpansion(Begin) && End.isMacroID() &&
        SM.isMacroArgExpansion(End)) {

      size_t Length;
      if (ParentRange.isValid()) {
        auto RR = getRangeInRange(Range, ParentRange.getBegin(),
                                  ParentRange.getEnd());
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

  template <class T> SourceRange getArgSourceRange(const T &Arg) {
    return Arg.getSourceRange();
  }

  template <class T> SourceRange getArgSourceRange(const T *Arg) {
    return Arg->getSourceRange();
  }

  void setArgStr(std::string &&Str);
  std::shared_ptr<TemplateDependentStringInfo> DependentStr;
  TemplateArgument::ArgKind Kind;
  bool IsWritten = true;
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
  bool hasItem() const;
  bool hasStream() const;
  bool hasSync() const;
  bool hasBF64() const;
  bool hasBF16() const;
  bool hasGlobalMemAcc() const;
  bool hasExternShared() const;
  void setItem(bool Has = true);
  void setStream(bool Has = true);
  void setSync(bool Has = true);
  void setBF64(bool Has = true);
  void setBF16(bool Has = true);
  void setGlobalMemAcc(bool Has = true);
  void addTexture(std::shared_ptr<TextureInfo> Tex);
  void addVar(std::shared_ptr<MemVarInfo> Var);
  void merge(const MemVarMap &OtherMap);
  void merge(const MemVarMap &VarMap,
             const std::vector<TemplateArgumentInfo> &TemplateArgs);
  int calculateExtraArgsSize() const;
  std::string getExtraCallArguments(bool HasPreParam, bool HasPostParam) const;
  void
  requestFeatureForAllVarMaps(const clang::tooling::UnifiedPath &Path) const;

  // When adding the ExtraParam with new line, the second argument should be
  // true, and the third argument is the string of indent, which will occur
  // before each ExtraParam.
  std::string
  getExtraDeclParam(bool HasPreParam, bool HasPostParam,
                    FormatInfo FormatInformation = FormatInfo()) const;
  std::string getKernelArguments(bool HasPreParam, bool HasPostParam,
                                 const clang::tooling::UnifiedPath &Path) const;
  const MemVarInfoMap &getMap(MemVarInfo::VarScope Scope) const;
  const GlobalMap<TextureInfo> &getTextureMap() const;
  void removeDuplicateVar();

  MemVarInfoMap &getMap(MemVarInfo::VarScope Scope);
  bool isSameAs(const MemVarMap &Other) const;

  enum CallOrDecl {
    CallArgument = 0,
    KernelArgument,
    DeclParameter,
  };

  static const MemVarMap *
  getHeadWithoutPathCompression(const MemVarMap *CurNode);
  static MemVarMap *getHead(MemVarMap *CurNode);
  unsigned int getHeadNodeDim() const;

private:
  static void merge(MemVarInfoMap &Master, const MemVarInfoMap &Branch,
                    const std::vector<TemplateArgumentInfo> &TemplateArgs);
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

  template <CallOrDecl COD>
  inline std::string
  getArgumentsOrParameters(int PreParams, int PostParams,
                           FormatInfo FormatInformation = FormatInfo()) const {
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
      GetArgOrParam<MemVarInfo, COD>()(PS, ExternVarMap.begin()->second)
          << ", ";
    getArgumentsOrParametersFromMap<MemVarInfo, COD>(PS, GlobalVarMap);
    getArgumentsOrParametersFromMap<MemVarInfo, COD>(PS, LocalVarMap);
    getArgumentsOrParametersFromMap<TextureInfo, COD>(PS, TextureMap);

    std::string Result = PS.Str;
    return (Result.empty() || PostParams != 0) && PreParams == 0
               ? Result
               : Result.erase(Result.size() - 2, 2);
  }

  template <class T, CallOrDecl COD>
  static void getArgumentsOrParametersFromMap(ParameterStream &PS,
                                              const GlobalMap<T> &VarMap) {
    for (const auto &VI : VarMap) {
      if (!VI.second->isUseHelperFunc()) {
        continue;
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
  void getArgumentsOrParametersForDecl(ParameterStream &PS,
                                              int PreParams,
                                              int PostParams) const;

  bool HasItem, HasStream, HasSync, HasBF64, HasBF16, HasGlobalMemAcc;
  MemVarInfoMap LocalVarMap;
  MemVarInfoMap GlobalVarMap;
  MemVarInfoMap ExternVarMap;
  GlobalMap<TextureInfo> TextureMap;
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

template <>
inline std::string
MemVarMap::getArgumentsOrParameters<MemVarMap::DeclParameter>(
    int PreParams, int PostParams, FormatInfo FormatInformation) const {

  ParameterStream PS;
  if (DpctGlobalInfo::getFormatRange() != clang::format::FormatRange::none) {
    PS = ParameterStream(FormatInformation,
                         DpctGlobalInfo::getCodeFormatStyle().ColumnLimit);
  } else {
    PS = ParameterStream(FormatInformation, 80);
  }
  getArgumentsOrParametersForDecl(PS, PreParams, PostParams);
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

// call function expression includes location, name, arguments num, template
// arguments and all function decls related to this call, also merges memory
// variable info of all related function decls.
class CallFunctionExpr {
public:
  template <class T>
  CallFunctionExpr(unsigned Offset,
                   const clang::tooling::UnifiedPath &FilePathIn, const T &C)
      : FilePath(FilePathIn), BeginLoc(Offset) {}

  void buildCallExprInfo(const CXXConstructExpr *Ctor);
  void buildCallExprInfo(const CallExpr *CE);

  const MemVarMap &getVarMap();
  const std::vector<std::shared_ptr<TextureObjectInfo>> &
  getTextureObjectList();
  std::shared_ptr<StructureTextureObjectInfo> getBaseTextureObjectInfo() const;

  void emplaceReplacement();
  unsigned getExtraArgLoc();
  bool hasArgs();
  bool hasTemplateArgs();
  bool hasWrittenTemplateArgs();
  const std::string &getName();

  std::string getTemplateArguments(bool &IsNeedWarning,
                                   bool WrittenArgsOnly = true,
                                   bool WithScalarWrapped = false);

  virtual std::string getExtraArguments();

  void setHasSideEffects(bool Val = true);
  bool hasSideEffects() const;

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
  unsigned getBegin();
  const clang::tooling::UnifiedPath &getFilePath();
  void buildInfo();
  void buildCalleeInfo(const Expr *Callee);
  void resizeTextureObjectList(size_t Size);

private:
  static std::string getName(const NamedDecl *D);
  void
  buildTemplateArguments(const llvm::ArrayRef<TemplateArgumentLoc> &ArgsList,
                         SourceRange Range);

  void buildTemplateArgumentsFromTypeLoc(const TypeLoc &TL);
  template <class TyLoc>
  void buildTemplateArgumentsFromSpecializationType(const TyLoc &TL) {
    for (size_t i = 0; i < TL.getNumArgs(); ++i) {
      TemplateArgs.emplace_back(TL.getArgLoc(i), TL.getSourceRange());
    }
  }

  std::string getNameWithNamespace(const FunctionDecl *FD, const Expr *Callee);

  void buildTextureObjectArgsInfo(const CallExpr *CE);

  template <class CallT> void buildTextureObjectArgsInfo(const CallT *C) {
    auto Args = C->arguments();
    auto IsKernel = C->getStmtClass() == Stmt::CUDAKernelCallExprClass;
    auto ArgsNum = std::distance(Args.begin(), Args.end());
    auto ArgItr = Args.begin();
    unsigned Idx = 0;
    TextureObjectList.resize(ArgsNum);
    while (ArgItr != Args.end()) {
      const Expr *Arg = (*ArgItr)->IgnoreImpCasts();
      if (auto Ctor = dyn_cast<CXXConstructExpr>(Arg)) {
        if (Ctor->getConstructor()->isCopyOrMoveConstructor()) {
          Arg = Ctor->getArg(0);
        }
      }
      if (auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts()))
        addTextureObjectArg(Idx, DRE, IsKernel);
      else if (auto ASE = dyn_cast<ArraySubscriptExpr>(Arg->IgnoreImpCasts()))
        addTextureObjectArg(Idx, ASE, IsKernel);
      Idx++;
      ArgItr++;
    }
  }
  void mergeTextureObjectInfo();

  const clang::tooling::UnifiedPath FilePath;
  unsigned BeginLoc = 0;
  unsigned ExtraArgLoc = 0;
  std::shared_ptr<DeviceFunctionInfo> FuncInfo;
  std::vector<TemplateArgumentInfo> TemplateArgs;

  // <ParameterIndex, ParameterName>
  std::vector<std::pair<int, std::string>> ParmRefArgs;
  MemVarMap VarMap;
  bool HasArgs = false;
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
  LinkUnresolved(const UnresolvedLookupExpr *ULE);
  static std::shared_ptr<DeviceFunctionInfo>
  LinkRedecls(const FunctionDecl *FD);
  static std::shared_ptr<DeviceFunctionInfo>
  LinkTemplateDecl(const FunctionTemplateDecl *FTD);
  static std::shared_ptr<DeviceFunctionInfo> LinkExplicitInstantiation(
      const FunctionDecl *Specialization, const FunctionTypeLoc &FTL,
      const ParsedAttributes &Attrs, const TemplateArgumentListInfo &TAList);
  std::shared_ptr<DeviceFunctionInfo> getFuncInfo() const;

  virtual void emplaceReplacement();
  static void reset();

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

  virtual ~DeviceFunctionDecl() = default;

protected:
  const FormatInfo &getFormatInfo();
  void buildTextureObjectParamsInfo(const ArrayRef<ParmVarDecl *> &Parms);

  template <class AttrsT>
  void buildReplaceLocInfo(const FunctionTypeLoc &FTL, const AttrsT &Attrs);

  virtual std::string getExtraParameters();

  unsigned Offset;
  const clang::tooling::UnifiedPath FilePath;
  unsigned ParamsNum;
  unsigned ReplaceOffset;
  unsigned ReplaceLength;
  bool IsReplaceFollowedByPP = false;
  unsigned NonDefaultParamNum;
  bool IsDefFilePathNeeded = false;
  std::vector<std::shared_ptr<TextureObjectInfo>> TextureObjectList;
  FormatInfo FormatInformation;

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
  std::string getExtraParameters() override;
};

class DeviceFunctionDeclInModule : public DeviceFunctionDecl {
  void insertWrapper();
  bool HasBody = false;
  size_t DeclEnd;
  std::string FuncName;
  std::vector<std::pair<std::string, std::string>> ParametersInfo;
  std::shared_ptr<KernelCallExpr> Kernel;
  void buildParameterInfo(const FunctionDecl *FD);
  void buildWrapperInfo(const FunctionDecl *FD);
  void buildCallInfo(const FunctionDecl *FD);
  std::vector<std::pair<std::string, std::string>> &getParametersInfo();

public:
  DeviceFunctionDeclInModule(unsigned Offset,
                             const clang::tooling::UnifiedPath &FilePathIn,
                             const FunctionTypeLoc &FTL,
                             const ParsedAttributes &Attrs,
                             const FunctionDecl *FD);
  DeviceFunctionDeclInModule(unsigned Offset,
                             const clang::tooling::UnifiedPath &FilePathIn,
                             const FunctionDecl *FD);
  void emplaceReplacement() override;
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
  void addVar(std::shared_ptr<MemVarInfo> Var);
  void setItem();
  void setStream();
  void setSync();
  void setBF64();
  void setBF16();
  void setGlobalMemAcc();
  void addTexture(std::shared_ptr<TextureInfo> Tex);
  MemVarMap &getVarMap();
  std::shared_ptr<TextureObjectInfo> getTextureObject(unsigned Idx);
  std::shared_ptr<StructureTextureObjectInfo> getBaseTextureObject() const;
  void setCallGroupFunctionInControlFlow(bool Val = true);
  bool hasCallGroupFunctionInControlFlow() const;
  void setHasSideEffectsAnalyzed(bool Val = true);
  bool hasSideEffectsAnalyzed() const;

  void buildInfo();
  bool hasParams();
  bool isBuilt();
  void setBuilt();
  bool isLambda();
  void setLambda();
  bool isInlined();
  void setInlined();
  bool isKernel();
  void setKernel();
  bool isKernelInvoked();
  void setKernelInvoked();
  std::string
  getExtraParameters(const clang::tooling::UnifiedPath &Path,
                     FormatInfo FormatInformation = FormatInfo());
  std::string
  getExtraParameters(const clang::tooling::UnifiedPath &Path,
                     const std::vector<TemplateArgumentInfo> &TAList,
                     FormatInfo FormatInformation = FormatInfo());
  void setDefinitionFilePath(const clang::tooling::UnifiedPath &Path);
  const clang::tooling::UnifiedPath &getDefinitionFilePath();



  void setNeedSyclExternMacro() { NeedSyclExternMacro = true; }
  bool IsSyclExternMacroNeeded() { return NeedSyclExternMacro; }
  void setAlwaysInlineDevFunc() { AlwaysInlineDevFunc = true; }
  bool IsAlwaysInlineDevFunc() { return AlwaysInlineDevFunc; }
  void setForceInlineDevFunc() { ForceInlineDevFunc = true; }
  bool IsForceInlineDevFunc() { return ForceInlineDevFunc; }
  void merge(std::shared_ptr<DeviceFunctionInfo> Other);
  size_t ParamsNum;
  size_t NonDefaultParamNum;
  GlobalMap<CallFunctionExpr> &getCallExprMap();
  void addSubGroupSizeRequest(unsigned int Size, SourceLocation Loc,
                              std::string APIName, std::string VarName = "");
  std::vector<std::tuple<unsigned int, clang::tooling::UnifiedPath,
                         unsigned int, std::string, std::string>> &
  getSubGroupSize();
  bool isParameterReferenced(unsigned int Index);
  void setParameterReferencedStatus(unsigned int Index, bool IsReferenced);
  std::string getFunctionName();

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
};

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

class KernelCallExpr : public CallFunctionExpr {
public:
  bool IsInMacroDefine = false;
  bool NeedLambda = false;

private:
  struct ArgInfo {
    ArgInfo(const ParmVarDecl *PVD, KernelArgumentAnalysis &Analysis,
            const Expr *Arg, bool Used, int Index, KernelCallExpr *BASE)
        : IsPointer(false), IsRedeclareRequired(false),
          IsUsedAsLvalueAfterMalloc(Used), Index(Index) {
      Analysis.analyze(Arg);
      ArgString = Analysis.getReplacedString();
      TryGetBuffer = Analysis.TryGetBuffer;
      IsRedeclareRequired = Analysis.IsRedeclareRequired;
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
        ArgSize = MapNames::KernelArgTypeSizeMap.at(KernelArgType::KAT_Default);

        // Currently, all the device RNG state structs are passed to kernel by
        // pointer. So we check the pointee type, if it is in the type map, we
        // replace the TypeString with the MKL generator type.
        std::string PointeeTypeStr =
            Arg->getType()->getPointeeType().getUnqualifiedType().getAsString();
        auto Iter = MapNames::DeviceRandomGeneratorTypeMap.find(PointeeTypeStr);
        if (Iter != MapNames::DeviceRandomGeneratorTypeMap.end()) {
          // Here the "*" is not added in the TypeString, the "*" will be added
          // in function buildKernelArgsStmt
          TypeString = Iter->second;
          IsDeviceRandomGeneratorType = true;
        }
      } else {
        auto QT = Arg->getType();
        QT = QT.getUnqualifiedType();
        auto Iter =
            MapNames::VectorTypeMigratedTypeSizeMap.find(QT.getAsString());
        if (Iter != MapNames::VectorTypeMigratedTypeSizeMap.end())
          ArgSize = Iter->second;
        else
          ArgSize =
              MapNames::KernelArgTypeSizeMap.at(KernelArgType::KAT_Default);
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

    ArgInfo(const ParmVarDecl *PVD, const std::string &ArgsArrayName,
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

    ArgInfo(const ParmVarDecl *PVD, KernelCallExpr *Kernel)
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

    ArgInfo(std::shared_ptr<TextureObjectInfo> Obj, KernelCallExpr *BASE)
        : IsUsedAsLvalueAfterMalloc(false), Texture(Obj) {
      IsPointer = false;
      IsRedeclareRequired = false;
      TypeString = "";
      Index = 0;
      if (auto S = std::dynamic_pointer_cast<StructureTextureObjectInfo>(Obj)) {
        IsDoublePointer = S->containsVirtualPointer();
      }
      ArgString = Obj->getName();
      IdString = ArgString + "_";
      ArgSize = MapNames::KernelArgTypeSizeMap.at(KernelArgType::KAT_Texture);
    }

    inline const std::string &getArgString() const { return ArgString; }
    inline const std::string &getTypeString() const { return TypeString; }
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
    bool IsDoublePointer = false;

    std::shared_ptr<TextureObjectInfo> Texture;
  };

  void print(KernelPrinter &Printer);
  void printSubmit(KernelPrinter &Printer);
  void printSubmitLamda(KernelPrinter &Printer);
  void printParallelFor(KernelPrinter &Printer, bool IsInSubmit);
  void printKernel(KernelPrinter &Printer);
  template <typename IDTy, typename... Ts>
  void printWarningMessage(KernelPrinter &Printer, IDTy MsgID, Ts &&...Vals);
  template <class T> void printStreamBase(T &Printer);

public:
  KernelCallExpr(unsigned Offset, const clang::tooling::UnifiedPath &FilePath,
                 const CUDAKernelCallExpr *KernelCall)
      : CallFunctionExpr(Offset, FilePath, KernelCall), IsSync(false) {
    setIsInMacroDefine(KernelCall);
    setNeedAddLambda(KernelCall);
    buildCallExprInfo(KernelCall);
    buildArgsInfo(KernelCall);
    buildKernelInfo(KernelCall);
  }

  void addAccessorDecl();
  void buildInfo();
  void setKernelCallDim();
  void buildUnionFindSet();
  void addReplacements();
  inline std::string getExtraArguments() override {
    if (!getFuncInfo()) {
      return "";
    }

    return getVarMap().getKernelArguments(getFuncInfo()->NonDefaultParamNum,
                                          getFuncInfo()->ParamsNum -
                                              getFuncInfo()->NonDefaultParamNum,
                                          getFilePath());
  }

  inline const std::vector<ArgInfo> &getArgsInfo() { return ArgsInfo; }
  int calculateOriginArgsSize() const;

  std::string getReplacement();

  inline void setEvent(const std::string &E) { Event = E; }
  inline const std::string &getEvent() { return Event; }

  inline void setSync(bool Sync = true) { IsSync = Sync; }
  inline bool isSync() { return IsSync; }

  static std::shared_ptr<KernelCallExpr> buildFromCudaLaunchKernel(
      const std::pair<clang::tooling::UnifiedPath, unsigned> &LocInfo,
      const CallExpr *);
  static std::shared_ptr<KernelCallExpr>
  buildForWrapper(clang::tooling::UnifiedPath, const FunctionDecl *,
                  std::shared_ptr<DeviceFunctionInfo>);
  unsigned int GridDim = 3;
  unsigned int BlockDim = 3;
  void setEmitSizeofWarningFlag(bool Flag) { EmitSizeofWarning = Flag; }

private:
  KernelCallExpr(unsigned Offset, const clang::tooling::UnifiedPath &FilePath)
      : CallFunctionExpr(Offset, FilePath, nullptr), IsSync(false) {}
  void buildArgsInfoFromArgsArray(const FunctionDecl *FD,
                                  const Expr *ArgsArray) {}
  void buildArgsInfo(const CallExpr *CE) {
    KernelArgumentAnalysis Analysis(IsInMacroDefine);
    auto KCallSpellingRange =
        getTheLastCompleteImmediateRange(CE->getBeginLoc(), CE->getEndLoc());
    Analysis.setCallSpelling(KCallSpellingRange.first,
                             KCallSpellingRange.second);
    auto &TexList = getTextureObjectList();

    for (unsigned Idx = 0; Idx < CE->getNumArgs(); ++Idx) {
      if (auto Obj = TexList[Idx]) {
        ArgsInfo.emplace_back(Obj, this);
      } else {
        auto Arg = CE->getArg(Idx);
        bool Used = true;
        if (auto *ArgDRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts()))
          Used = isArgUsedAsLvalueUntil(ArgDRE, CE);
        const auto FD = CE->getDirectCallee();
        ArgsInfo.emplace_back(FD ? FD->parameters()[Idx] : nullptr, Analysis,
                              Arg, Used, Idx, this);
      }
    }
  }
  bool isDefaultStream() const {
    return StringRef(ExecutionConfig.Stream).startswith("{{NEEDREPLACEQ") ||
           ExecutionConfig.IsDefaultStream;
  }

  bool isQueuePtr() const { return ExecutionConfig.IsQueuePtr; }

  std::string getQueueStr() const {
    if (isDefaultStream())
      return "";
    std::string Ret;
    if (isQueuePtr())
      Ret = "*";
    return Ret += ExecutionConfig.Stream;
  }

  void buildKernelInfo(const CUDAKernelCallExpr *KernelCall);
  void setIsInMacroDefine(const CUDAKernelCallExpr *KernelCall);
  void setNeedAddLambda(const CUDAKernelCallExpr *KernelCall);
  void buildNeedBracesInfo(const CallExpr *KernelCall);
  void buildLocationInfo(const CallExpr *KernelCall);
  template <class ArgsRange>
  void buildExecutionConfig(const ArgsRange &ConfigArgs,
                            const CallExpr *KernelCall);

  void removeExtraIndent() {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(getFilePath(),
                                         getBegin() - LocInfo.Indent.length(),
                                         LocInfo.Indent.length(), "", nullptr));
  }
  void addDevCapCheckStmt();
  void addAccessorDecl(MemVarInfo::VarScope Scope);
  void addAccessorDecl(std::shared_ptr<MemVarInfo> VI);
  void addStreamDecl() {
    if (getVarMap().hasStream())
      SubmitStmtsList.StreamList.emplace_back(buildString(
          MapNames::getClNamespace() + "stream ",
          DpctGlobalInfo::getStreamName(), "(64 * 1024, 80, cgh);"));
    if (getVarMap().hasSync()) {
      auto DefaultQueue =
          buildString(MapNames::getDpctNamespace(), "get_",
                      DpctGlobalInfo::getDeviceQueueName(), "()");
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
        OuterStmts.OthersList.emplace_back(
            buildString(MapNames::getDpctNamespace(), "global_memory<",
                        MapNames::getDpctNamespace(), "byte_t, 1> d_",
                        DpctGlobalInfo::getSyncName(), "(4);"));

        OuterStmts.OthersList.emplace_back(buildString(
            "d_", DpctGlobalInfo::getSyncName(), ".init(", DefaultQueue, ");"));

        SubmitStmtsList.SyncList.emplace_back(
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
    std::string &SubGroupSize = Config[5];
    bool IsDefaultStream = false;
    bool IsQueuePtr = true;
  } ExecutionConfig;

  std::vector<ArgInfo> ArgsInfo;

  std::string Event;
  bool IsSync;

  class {
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

    inline KernelPrinter &print(KernelPrinter &Printer) {
      printList(Printer, StreamList);
      printList(Printer, SyncList);
      printList(Printer, MemoryList);
      printList(Printer, RangeList,
                "ranges used for accessors to device memory");
      printList(Printer, PtrList, "pointers to device memory");
      printList(Printer, AccessorList, "accessors to device memory");
      printList(Printer, TextureList, "accessors to image objects");
      printList(Printer, SamplerList, "sampler of image objects");
      printList(Printer, NdRangeList,
                "ranges to define ND iteration space for the kernel");
      printList(Printer, CommandGroupList, "helper variables defined");
      return Printer;
    }

    bool empty() const noexcept {
      return CommandGroupList.empty() && NdRangeList.empty() &&
             AccessorList.empty() && PtrList.empty() && MemoryList.empty() &&
             RangeList.empty() && TextureList.empty() && SamplerList.empty() &&
             StreamList.empty() && SyncList.empty();
    }

  private:
    KernelPrinter &printList(KernelPrinter &Printer, const StmtList &List,
                             StringRef Comments = "") {
      if (List.empty())
        return Printer;
      if (!Comments.empty() && DpctGlobalInfo::isCommentsEnabled())
        Printer.line("// ", Comments);
      Printer << List;
      return Printer.newLine();
    }
  } SubmitStmtsList;

  class {
  public:
    StmtList ExternList;
    StmtList InitList;
    StmtList OthersList;

    inline KernelPrinter &print(KernelPrinter &Printer) {
      printList(Printer, ExternList);
      printList(Printer, InitList, "init global memory");
      printList(Printer, OthersList);
      return Printer;
    }

    bool empty() const noexcept {
      return InitList.empty() && ExternList.empty() && OthersList.empty();
    }

  private:
    KernelPrinter &printList(KernelPrinter &Printer, const StmtList &List,
                             StringRef Comments = "") {
      if (List.empty())
        return Printer;
      if (!Comments.empty() && DpctGlobalInfo::isCommentsEnabled())
        Printer.line("// ", Comments);
      Printer << List;
      return Printer.newLine();
    }
  } OuterStmts;
  StmtList KernelStmts;
  std::string KernelArgs;
  int TotalArgsSize = 0;
  bool EmitSizeofWarning = false;
  unsigned int SizeOfHighestDimension = 0;
};

class CudaMallocInfo {
public:
  CudaMallocInfo(unsigned Offset, const clang::tooling::UnifiedPath &FilePath,
                 const VarDecl *VD)
      : Name(VD->getName().str()) {}

  static const VarDecl *getMallocVar(const Expr *Arg) {
    if (auto UO = dyn_cast<UnaryOperator>(Arg->IgnoreImpCasts())) {
      if (UO->getOpcode() == UO_AddrOf) {
        return getDecl(UO->getSubExpr());
      }
    }
    return nullptr;
  }
  static const VarDecl *getDecl(const Expr *E) {
    if (auto DeclRef = dyn_cast<DeclRefExpr>(E->IgnoreImpCasts()))
      return dyn_cast<VarDecl>(DeclRef->getDecl());
    return nullptr;
  }

  void setSizeExpr(const Expr *SizeExpression) {
    ArgumentAnalysis A(SizeExpression, false);
    A.analyze();
    Size = A.getReplacedString();
  }
  void setSizeExpr(const Expr *N, const Expr *ElemSize) {
    ArgumentAnalysis AN(N, false);
    ArgumentAnalysis AElemSize(ElemSize, false);
    AN.analyze();
    AElemSize.analyze();
    Size = "(" + AN.getReplacedString() + ")*(" +
           AElemSize.getReplacedString() + ")";
  }

  std::string getAssignArgs(const std::string &TypeName) {
    return Name + ", " + Size;
  }

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

  if (DpctGlobalInfo::getTempVariableDeclCounterMap().count(
          KeyForDeclCounter) == 0) {
    DpctGlobalInfo::getTempVariableDeclCounterMap().insert(
        {KeyForDeclCounter, {}});
  }
  auto Iter =
      DpctGlobalInfo::getTempVariableDeclCounterMap().find(KeyForDeclCounter);
  switch (HFT) {
  case HelperFuncType::HFT_DefaultQueue:
    ++Iter->second.DefaultQueueCounter;
    break;
  case HelperFuncType::HFT_CurrentDevice:
    ++Iter->second.CurrentDeviceCounter;
    break;
  default:
    break;
  }
}

} // namespace dpct
} // namespace clang

#endif
