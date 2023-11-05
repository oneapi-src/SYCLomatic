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
void setGetReplacedNamePtr(
    llvm::StringRef (*Ptr)(const clang::NamedDecl *D));

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
  void buildInfo(std::string FilePath, unsigned int Offset);

  unsigned int Length;
  std::string ReplText;
  bool NeedReport = false;
  bool IsAssigned = false;
};

struct TimeStubTypeInfo {
  TimeStubTypeInfo(unsigned int Length, std::string StrWithSB,
                   std::string StrWithoutSB)
      : Length(Length), StrWithSB(StrWithSB), StrWithoutSB(StrWithoutSB) {}

  void buildInfo(std::string FilePath, unsigned int Offset,
                 bool isReplTxtWithSB);

  unsigned int Length;
  std::string StrWithSB;
  std::string StrWithoutSB;
};

struct BuiltinVarInfo {
  BuiltinVarInfo(unsigned int Len, std::string Repl,
                 std::shared_ptr<DeviceFunctionInfo> DFI)
      : Len(Len), Repl(Repl), DFI(DFI) {}
  void buildInfo(std::string FilePath, unsigned int Offset, unsigned int Dim);

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

enum HDFuncInfoType{HDFI_Def, HDFI_Decl, HDFI_Call};

struct HostDeviceFuncLocInfo {
  std::string FilePath;
  std::string FuncContentCache;
  unsigned FuncStartOffset = 0;
  unsigned FuncEndOffset = 0;
  unsigned FuncNameOffset = 0;
  bool Processed = false;
  bool CalledByHostDeviceFunction = false;
  HDFuncInfoType Type;
};

struct HostDeviceFuncInfo {
  std::unordered_map<std::string, HostDeviceFuncLocInfo>
  LocInfos;
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
  std::string FilePath;
  unsigned int Offset;
  unsigned int Length;
  bool isAssigned;
  bool isDataGradient;
  std::string CompoundLoc;
  std::vector<std::string> RnnInputDeclLoc;
  std::vector<std::string> FuncArgs;
};

// <function name, Info>
using HDFuncInfoMap =
    std::unordered_map<std::string, HostDeviceFuncInfo>;
// <file path, <Offset, Info>>
using CudaArchPPMap =
    std::unordered_map<std::string,
                       std::unordered_map<unsigned int, CudaArchPPInfo>>;
using CudaArchDefMap =
    std::unordered_map<std::string,
                       std::unordered_map<unsigned int, unsigned int>>;
class ParameterStream {
public:
  ParameterStream() { FormatInformation = FormatInfo(); }
  ParameterStream(FormatInfo FormatInformation, int ColumnLimit)
      : FormatInformation(FormatInformation), ColumnLimit(ColumnLimit) {}

  ParameterStream &operator<<(const std::string &InputParamStr) {
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
  ParameterStream &operator<<(int InputInt) {
    return *this << std::to_string(InputInt);
  }

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

//                             DpctGlobalInfo
//                                         |
//              --------------------------------------
//              |                          |                           |
//    DpctFileInfo       DpctFileInfo     ... (other info)
//                            |
//           -----------------------------------------------------
//           |                           |                         | |
//  MemVarInfo  DeviceFunctionDecl  KernelCallExpr  CudaMallocInfo
// Global Variable)                |   (inherit from CallFunctionExpr)
//                           DeviceFunctionInfo
//                                          |
//                        --------------------------
//                        |                                     |
//            CallFunctionExpr              MemVarInfo
//       (Call Expr in Function)   (Defined in Function)
//                        |
//          DeviceFunctionInfo
//               (Callee Info)

// Store analysis info (eg. memory variable info, kernel function info,
// replacements and so on) of each file
class DpctFileInfo {
public:
  DpctFileInfo(const std::string &FilePathIn)
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
  inline const std::string &getFilePath() { return FilePath; }

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
  void emplaceReplacements(ReplTy &ReplSet /*out*/);

  inline void addReplacement(std::shared_ptr<ExtReplacement> Repl) {
    if (Repl->getLength() == 0 && Repl->getReplacementText().empty())
      return;
    Repls->addReplacement(Repl);
  }
  bool isInAnalysisScope();
  std::shared_ptr<ExtReplacements> getRepls() { return Repls; }

  size_t getFileSize() const { return FileSize; }

  std::string &getFileContent() { return FileContentCache; }

  // Header inclusion directive insertion functions
  void setFileEnterOffset(unsigned Offset) {
    if (!HasInclusionDirective) {
      FirstIncludeOffset = Offset;
      LastIncludeOffset = Offset;
    }
  }

  void setFirstIncludeOffset(unsigned Offset) {
    if (!HasInclusionDirective) {
      FirstIncludeOffset = Offset;
      LastIncludeOffset = Offset;
      HasInclusionDirective = true;
    }
  }

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

  inline const SourceLineInfo &getLineInfo(unsigned LineNumber) {
    if (!LineNumber || LineNumber > Lines.size()) {
      llvm::dbgs() << "[DpctFileInfo::getLineInfo] illegal line number "
                   << LineNumber;
      static SourceLineInfo InvalidLine;
      return InvalidLine;
    }
    return Lines[--LineNumber];
  }
  StringRef getLineString(unsigned LineNumber) {
    return getLineInfo(LineNumber).Line;
  }

  // Get line number by offset
  inline unsigned getLineNumber(unsigned Offset) {
    return getLineInfoFromOffset(Offset).Number;
  }
  // Set line range info of replacement
  void setLineRange(ExtReplacements::SourceLineRange &LineRange,
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
  void insertIncludedFilesInfo(std::shared_ptr<DpctFileInfo> Info) {
    auto Iter = IncludedFilesInfoSet.find(Info);
    if (Iter == IncludedFilesInfoSet.end()) {
      IncludedFilesInfoSet.insert(Info);
    }
  }

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

  std::shared_ptr<tooling::TranslationUnitReplacements> PreviousTUReplFromYAML =
      nullptr;
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
  inline const SourceLineInfo &getLineInfoFromOffset(unsigned Offset) {
    return *(std::upper_bound(Lines.begin(), Lines.end(), Offset,
                              [](unsigned Offset, const SourceLineInfo &Line) {
                                return Line.Offset > Offset;
                              }) -
             1);
  }

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

  std::string FilePath;
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
    std::string FilePath;
    unsigned Offset;
    bool IsInAnalysisScope;
    MacroDefRecord(SourceLocation NTL, bool IIAS) : IsInAnalysisScope(IIAS) {
      auto LocInfo = DpctGlobalInfo::getLocInfo(NTL);
      FilePath = LocInfo.first;
      Offset = LocInfo.second;
    }
  };

  class MacroExpansionRecord {
  public:
    std::string Name;
    int NumTokens;
    std::string FilePath;
    unsigned ReplaceTokenBeginOffset;
    unsigned ReplaceTokenEndOffset;
    SourceRange Range;
    bool IsInAnalysisScope;
    bool IsFunctionLike;
    int TokenIndex;
    MacroExpansionRecord(IdentifierInfo *ID, const MacroInfo *MI,
                         SourceRange Range, bool IsInAnalysisScope, int TokenIndex) {
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
  };

  struct HelperFuncReplInfo {
    HelperFuncReplInfo(const std::string DeclLocFile = "",
                       unsigned int DeclLocOffset = 0,
                       bool IsLocationValid = false)
        : DeclLocFile(DeclLocFile), DeclLocOffset(DeclLocOffset),
          IsLocationValid(IsLocationValid) {}
    std::string DeclLocFile;
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

  inline static bool isInRoot(SourceLocation SL) {
    return isInRoot(DpctGlobalInfo::getLocInfo(SL).first);
  }
  static bool isInRoot(const std::string &FilePath,
                       bool IsChildRelative = true) {
    if (IsChildRelative) {
      std::string Path(FilePath);
      makeCanonical(Path);
      if (isChildPath(InRoot, Path)) {
        return !isExcluded(Path);
      } else {
        return false;
      }
    } else {
      if (isChildPath(InRoot, FilePath, IsChildRelative)) {
        return !isExcluded(FilePath, IsChildRelative);
      } else {
        return false;
      }
    }
  }
  inline static bool isInAnalysisScope(SourceLocation SL) {
    return isInAnalysisScope(DpctGlobalInfo::getLocInfo(SL).first);
  }
  // Input Arg needs to be an absolute file path
  static bool isInAnalysisScope(const std::string &AbsFilePath,
                                bool IsChildRelative = true) {
    return isChildPath(AnalysisScope, AbsFilePath, IsChildRelative);
  }

  static bool isExcluded(const std::string &FilePath, bool IsRelative = true) {
    static std::map<std::string, bool> Cache;
    if (FilePath.empty() || DpctGlobalInfo::getExcludePath().empty()) {
      return false;
    }
    std::string CanonicalPath = FilePath;
    if (IsRelative) {
      if (!makeCanonical(CanonicalPath)) {
        return false;
      }
    }
    if (Cache.count(CanonicalPath)) {
      return Cache[CanonicalPath];
    }
    for (auto &Path : DpctGlobalInfo::getExcludePath()) {
      if (isChildOrSamePath(Path.first, CanonicalPath)) {
        Cache[CanonicalPath] = true;
        return true;
      }
    }
    Cache[CanonicalPath] = false;
    return false;
  }
  // TODO: implement one of this for each source language.
  inline static bool isInCudaPath(SourceLocation SL) {
    return isInCudaPath(getSourceManager()
                            .getFilename(getSourceManager().getExpansionLoc(SL))
                            .str());
  }
  // TODO: implement one of this for each source language.
  static bool isInCudaPath(const std::string &FilePath) {
    std::string Path = FilePath;
    makeCanonical(Path);
    return isChildPath(CudaPath, Path);
  }
  static void setInRoot(const std::string &InRootPath) { InRoot = InRootPath; }
  static const std::string &getInRoot() {
    assert(!InRoot.empty());
    return InRoot;
  }
  static void setOutRoot(const std::string &OutRootPath) {
    OutRoot = OutRootPath;
  }
  static const std::string &getOutRoot() {
    assert(!OutRoot.empty());
    return OutRoot;
  }
  static void setAnalysisScope(const std::string &InputAnalysisScope) {
    assert(!InputAnalysisScope.empty());
    AnalysisScope = InputAnalysisScope;
  }
  static const std::string &getAnalysisScope() {
    assert(!AnalysisScope.empty());
    return AnalysisScope;
  }
  static inline void addChangeExtensions(const std::string &Extension) {
    assert(!Extension.empty());
    ChangeExtensions.insert(Extension);
  }
  static inline const std::unordered_set<std::string> &getChangeExtensions() {
    return ChangeExtensions;
  }
  // TODO: implement one of this for each source language.
  static void setCudaPath(const std::string &InputCudaPath) {
    CudaPath = InputCudaPath;
  }
  // TODO: implement one of this for each source language.
  static const std::string &getCudaPath() {
    assert(!CudaPath.empty());
    return CudaPath;
  }

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
  static const std::string &getDeviceQueueName();
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
  static void setContext(ASTContext &C) {
    Context = &C;
    SM = &(Context->getSourceManager());
    FM = &(SM->getFileManager());
    Context->getParentMapContext().setTraversalKind(TK_AsIs);
  }
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
  inline static bool isKeepOriginCode() { return KeepOriginCode; }
  inline static void setKeepOriginCode(bool KOC = true) {
    KeepOriginCode = KOC;
  }
  inline static bool isSyclNamedLambda() { return SyclNamedLambda; }
  inline static void setSyclNamedLambda(bool SNL = true) {
    SyclNamedLambda = SNL;
  }
  inline static void setCheckUnicodeSecurityFlag(bool CUS) {
    CheckUnicodeSecurityFlag = CUS;
  }
  inline static bool getCheckUnicodeSecurityFlag() {
    return CheckUnicodeSecurityFlag;
  }
  inline static void setEnablepProfilingFlag(bool EP) {
    EnablepProfilingFlag = EP;
  }
  inline static bool getEnablepProfilingFlag() {
    return EnablepProfilingFlag;
  }
  inline static bool getGuessIndentWidthMatcherFlag() {
    return GuessIndentWidthMatcherFlag;
  }
  inline static void setGuessIndentWidthMatcherFlag(bool Flag = true) {
    GuessIndentWidthMatcherFlag = Flag;
  }
  inline static void setIndentWidth(unsigned int W) { IndentWidth = W; }
  inline static unsigned int getIndentWidth() { return IndentWidth; }
  inline static void insertKCIndentWidth(unsigned int W) {
    auto Iter = KCIndentWidthMap.find(W);
    if (Iter != KCIndentWidthMap.end())
      Iter->second++;
    else
      KCIndentWidthMap.insert(std::make_pair(W, 1));
  }
  inline static unsigned int getKCIndentWidth() {
    if (KCIndentWidthMap.empty())
      return DpctGlobalInfo::getCodeFormatStyle().IndentWidth;

    std::multimap<unsigned int, unsigned int, std::greater<unsigned int>>
        OccuranceIndentWidthMap;
    for (const auto &I : KCIndentWidthMap)
      OccuranceIndentWidthMap.insert(std::make_pair(I.second, I.first));

    return OccuranceIndentWidthMap.begin()->second;
  }
  inline static UsmLevel getUsmLevel() { return UsmLvl; }
  inline static void setUsmLevel(UsmLevel UL) { UsmLvl = UL; }
  inline static clang::CudaVersion getSDKVersion() { return SDKVersion; }
  inline static void setSDKVersion(clang::CudaVersion V) { SDKVersion = V; }
  inline static bool isIncMigration() { return IsIncMigration; }
  inline static void setIsIncMigration(bool Flag) { IsIncMigration = Flag; }
  inline static bool isQueryAPIMapping() { return IsQueryAPIMapping; }
  inline static void setIsQueryAPIMapping(bool Flag) {
    IsQueryAPIMapping = Flag;
  }
  inline static bool needDpctDeviceExt() { return NeedDpctDeviceExt; }
  inline static void setNeedDpctDeviceExt() { NeedDpctDeviceExt = true; }
  inline static unsigned int getAssumedNDRangeDim() {
    return AssumedNDRangeDim;
  }
  inline static void setAssumedNDRangeDim(unsigned int Dim) {
    AssumedNDRangeDim = Dim;
  }

  inline static bool getUsingExtensionDE(DPCPPExtensionsDefaultEnabled Ext) {
    return ExtensionDEFlag & (1 << static_cast<unsigned>(Ext));
  }
  inline static void setExtensionDEFlag(unsigned Flag) { ExtensionDEFlag = Flag; }
  inline static unsigned getExtensionDEFlag() { return ExtensionDEFlag; }

  inline static bool getUsingExtensionDD(DPCPPExtensionsDefaultDisabled Ext) {
    return ExtensionDDFlag & (1 << static_cast<unsigned>(Ext));
  }
  inline static void setExtensionDDFlag(unsigned Flag) { ExtensionDDFlag = Flag; }
  inline static unsigned getExtensionDDFlag() { return ExtensionDDFlag; }


  template <ExperimentalFeatures Exp> static bool getUsingExperimental() {
    return ExperimentalFlag & (1 << static_cast<unsigned>(Exp));
  }
  static void setExperimentalFlag(unsigned Flag) { ExperimentalFlag = Flag; }
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

  inline static format::FormatRange getFormatRange() { return FmtRng; }
  inline static void setFormatRange(format::FormatRange FR) { FmtRng = FR; }
  inline static DPCTFormatStyle getFormatStyle() { return FmtST; }
  inline static void setFormatStyle(DPCTFormatStyle FS) { FmtST = FS; }
  // Processing the folder or file by following rules:
  // Rule1: For {child path, parent path}, only parent path will be kept.
  // Rule2: Ignore invalid path.
  // Rule3: If path is not in --in-root, then ignore it.
  inline static void setExcludePath(std::vector<std::string> ExcludePathVec) {
    if (ExcludePathVec.empty()) {
      return;
    }
    std::set<std::string> ProcessedPath;
    for (auto Itr = ExcludePathVec.begin(); Itr != ExcludePathVec.end();
         Itr++) {
      if ((*Itr).empty()) {
        continue;
      }
      std::string PathBuf = *Itr;
      if (!makeCanonical(*Itr)) {
        clang::dpct::PrintMsg("Note: Path " + PathBuf +
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
          clang::dpct::PrintMsg("Note: Path " + PathBuf +
                                " is not in --in-root directory and will be "
                                "ignored by --in-root-exclude.\n");
        } else {
          bool IsNeedInsert = true;
          for (auto EP_Itr = ExcludePath.begin();
               EP_Itr != ExcludePath.end();) {
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
        clang::dpct::PrintMsg("Note: Path " + PathBuf +
                              " is invalid and will be ignored by option "
                              "--in-root-exclude.\n");
      }
    }
  }
  inline static std::unordered_map<std::string, bool> getExcludePath() {
    return ExcludePath;
  }
  inline static std::set<ExplicitNamespace> getExplicitNamespaceSet() {
    return ExplicitNamespaceSet;
  }
  inline static void
  setExplicitNamespace(std::vector<ExplicitNamespace> NamespacesVec) {
    size_t NamespaceVecSize = NamespacesVec.size();
    if (!NamespaceVecSize || NamespaceVecSize > 2) {
      ShowStatus(MigrationErrorInvalidExplicitNamespace);
      dpctExit(MigrationErrorInvalidExplicitNamespace);
    }
    for (auto &Namespace : NamespacesVec) {
      // 1. Ensure option none is alone
      bool Check1 =
          (Namespace == ExplicitNamespace::EN_None && NamespaceVecSize == 2);
      // 2. Ensure option cl, sycl, sycl-math only enabled one
      bool Check2 =
          ((Namespace == ExplicitNamespace::EN_CL ||
            Namespace == ExplicitNamespace::EN_SYCL ||
            Namespace == ExplicitNamespace::EN_SYCL_Math) &&
           (ExplicitNamespaceSet.size() == 1 &&
            ExplicitNamespaceSet.count(ExplicitNamespace::EN_DPCT) == 0));
      // 3. Check whether option dpct duplicated
      bool Check3 =
          (Namespace == ExplicitNamespace::EN_DPCT &&
           ExplicitNamespaceSet.count(ExplicitNamespace::EN_DPCT) == 1);
      if (Check1 || Check2 || Check3) {
        ShowStatus(MigrationErrorInvalidExplicitNamespace);
        dpctExit(MigrationErrorInvalidExplicitNamespace);
      } else {
        ExplicitNamespaceSet.insert(Namespace);
      }
    }
  }
  inline static bool isCtadEnabled() { return EnableCtad; }
  inline static void setCtadEnabled(bool Enable = true) { EnableCtad = Enable; }
  inline static bool isGenBuildScript() { return GenBuildScript; }
  inline static void setGenBuildScriptEnabled(bool Enable = true) {
    GenBuildScript = Enable;
  }
  inline static bool isCommentsEnabled() { return EnableComments; }
  inline static void setCommentsEnabled(bool Enable = true) {
    EnableComments = Enable;
  }

  inline static bool isDPCTNamespaceTempEnabled() {
    return TempEnableDPCTNamespace;
  }
  inline static void setDPCTNamespaceTempEnabled() {
    TempEnableDPCTNamespace = true;
  }

  inline static std::unordered_set<std::string> &getPrecAndDomPairSet() {
    return PrecAndDomPairSet;
  }

  inline static bool isMKLHeaderUsed() { return IsMLKHeaderUsed; }
  inline static void setMKLHeaderUsed(bool Used = true) {
    IsMLKHeaderUsed = Used;
  }

  inline static int getSuffixIndexInitValue(std::string FileNameAndOffset) {
    auto Res = LocationInitIndexMap.find(FileNameAndOffset);
    if (Res == LocationInitIndexMap.end()) {
      LocationInitIndexMap.insert(
          std::make_pair(FileNameAndOffset, CurrentMaxIndex + 1));
      return CurrentMaxIndex + 1;
    } else {
      return Res->second;
    }
  }

  inline static void updateInitSuffixIndexInRule(int InitVal) {
    CurrentIndexInRule = InitVal;
  }
  inline static int getSuffixIndexInRuleThenInc() {
    int Res = CurrentIndexInRule;
    if (CurrentMaxIndex < Res)
      CurrentMaxIndex = Res;
    CurrentIndexInRule++;
    return Res;
  }
  inline static int getSuffixIndexGlobalThenInc() {
    int Res = CurrentMaxIndex;
    CurrentMaxIndex++;
    return Res;
  }
  inline static const std::string &getGlobalQueueName() {
    const static std::string Q = "q_ct1";
    return Q;
  }
  inline static const std::string &getGlobalDeviceName() {
    const static std::string D = "dev_ct1";
    return D;
  }

  static std::string getStringForRegexReplacement(StringRef);

  inline static void setCodeFormatStyle(const clang::format::FormatStyle &Style) {
    CodeFormatStyle = Style;
  }
  inline static clang::format::FormatStyle getCodeFormatStyle() {
    return CodeFormatStyle;
  }

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
  static inline std::pair<std::string, unsigned>
  getLocInfo(const T *N, bool *IsInvalid = nullptr /* out */) {
    return getLocInfo(getLocation(N), IsInvalid);
  }

  static std::pair<std::string, unsigned>
  getLocInfo(const TypeLoc &TL, bool *IsInvalid = nullptr /*out*/) {
    return getLocInfo(TL.getBeginLoc(), IsInvalid);
  }

  // Return the absolute path of \p ID
  static std::optional<std::string> getAbsolutePath(FileID ID);
  // Return the absolute path of \p File
  static std::optional<std::string> getAbsolutePath(const FileEntry &File);

  static inline std::pair<std::string, unsigned>
  getLocInfo(SourceLocation Loc, bool *IsInvalid = nullptr /* out */) {
    if (SM->isMacroArgExpansion(Loc)) {
      Loc = SM->getImmediateSpellingLoc(Loc);
    }
    auto LocInfo = SM->getDecomposedLoc(SM->getExpansionLoc(Loc));
    auto AbsPath = getAbsolutePath(LocInfo.first);
    if (AbsPath)
      return std::make_pair(AbsPath.value(), LocInfo.second);
    if (IsInvalid)
      *IsInvalid = true;
    return std::make_pair("", 0);
  }

  static inline std::string getTypeName(QualType QT,
                                        const ASTContext &Context) {
    if (auto ET = QT->getAs<ElaboratedType>()) {
      if (ET->getQualifier())
        QT = Context.getElaboratedType(ETK_None, ET->getQualifier(),
                              ET->getNamedType(),
                              ET->getOwnedTagDecl());
      else
        QT = ET->getNamedType();
    }
    auto PP = Context.getPrintingPolicy();
    PP.SuppressTagKeyword  = true;
    return QT.getAsString(PP);
  }
  static inline std::string getTypeName(QualType QT) {
    return getTypeName(QT, DpctGlobalInfo::getContext());
  }
  static inline std::string getUnqualifiedTypeName(QualType QT,
                                                   const ASTContext &Context) {
    return getTypeName(QT.getUnqualifiedType(), Context);
  }
  static inline std::string getUnqualifiedTypeName(QualType QT) {
    return getUnqualifiedTypeName(QT, DpctGlobalInfo::getContext());
  }

  /// This function will return the replaced type name with qualifiers.
  /// Currently, since clang do not support get the order of original
  /// qualifiers, this function will follow the behavior of
  /// clang::QualType.print(), in other words, the behavior is that the
  /// qualifiers(const, volatile...) will occur before the simple type(int,
  /// bool...) regardless its order in origin code.
  /// \param [in] QT The input qualified type which need migration.
  /// \param [in] Context The AST context.
  /// \return The replaced type name string with qualifiers.
  static inline std::string getReplacedTypeName(QualType QT,
                                                const ASTContext &Context) {
    if (!QT.isNull())
      if (const auto *AT = dyn_cast<AutoType>(QT.getTypePtr())) {
        QT = AT->getDeducedType();
        if(QT.isNull()) {
          return "";
        }
      }
    std::string MigratedTypeStr;
    setGetReplacedNamePtr(&getReplacedName);
    llvm::raw_string_ostream OS(MigratedTypeStr);
    clang::PrintingPolicy PP =
        clang::PrintingPolicy(DpctGlobalInfo::getContext().getLangOpts());
    QT.print(OS, PP);
    OS.flush();
    setGetReplacedNamePtr(nullptr);
    return getFinalCastTypeNameStr(MigratedTypeStr);
  }
  static inline std::string getReplacedTypeName(QualType QT) {
    return getReplacedTypeName(QT, DpctGlobalInfo::getContext());
  }
  /// This function will return the original type name with qualifiers.
  /// The order of original qualifiers will follow the behavior of
  /// clang::QualType.print() regardless its order in origin code.
  /// \param [in] QT The input qualified type.
  /// \return The type name string with qualifiers.
  static inline std::string getOriginalTypeName(QualType QT) {
    std::string OriginalTypeStr;
    llvm::raw_string_ostream OS(OriginalTypeStr);
    clang::PrintingPolicy PP =
        clang::PrintingPolicy(DpctGlobalInfo::getContext().getLangOpts());
    QT.print(OS, PP);
    OS.flush();
    return OriginalTypeStr;
  }
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
      const ParsedAttributes &Attrs, const TemplateArgumentListInfo &TAList) {
    auto LocInfo = getLocInfo(FTL);
    return insertFile(LocInfo.first)
        ->insertNode<ExplicitInstantiationDecl, DeviceFunctionDecl>(
            LocInfo.second, FTL, Attrs, Specialization, TAList);
  }

  std::shared_ptr<DeviceFunctionDecl>
  insertDeviceFunctionDeclInModule(const FunctionDecl *FD) {
    auto LocInfo = getLocInfo(FD);
    return insertFile(LocInfo.first)
        ->insertNode<DeviceFunctionDeclInModule, DeviceFunctionDecl>(
            LocInfo.second, FD);
  }

  // Build kernel and device function declaration replacements and store
  // them.
  void buildKernelInfo() {
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

  void buildReplacements();
  std::set<std::string> &getProcessedFile() {
    return ProcessedFile;
  }
  void processCudaArchMacro();
  void generateHostCode(
      std::multimap<unsigned int, std::shared_ptr<clang::dpct::ExtReplacement>>
          &ProcessedReplList,
      HostDeviceFuncLocInfo Info, unsigned ID);
  void postProcess();
  void cacheFileRepl(std::string FilePath,
                     std::shared_ptr<ExtReplacements> Repl) {
    FileReplCache[FilePath] = Repl;
  }
  // Emplace stored replacements into replacement set.
  void emplaceReplacements(ReplTy &ReplSets /*out*/) {
    if (DpctGlobalInfo::isNeedRunAgain())
      return;
    for (auto &FileRepl : FileReplCache) {
      FileRepl.second->emplaceIntoReplSet(ReplSets[FileRepl.first]);
    }
  }
  std::shared_ptr<KernelCallExpr> buildLaunchKernelInfo(const CallExpr *);

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
      std::string FilePath,
      std::shared_ptr<tooling::TranslationUnitReplacements> TUR) {
    auto FileInfo = insertFile(FilePath);
    if (FileInfo->PreviousTUReplFromYAML == nullptr)
      FileInfo->PreviousTUReplFromYAML = TUR;
  }
  std::shared_ptr<tooling::TranslationUnitReplacements>
  getReplInfoFromYAMLSavedInFileInfo(std::string FilePath) {
    auto FileInfo = findObject(FileMap, FilePath);
    if (FileInfo)
      return FileInfo->PreviousTUReplFromYAML;
    else
      return nullptr;
  }

  void insertEventSyncTypeInfo(
      const std::shared_ptr<clang::dpct::ExtReplacement> Repl,
      bool NeedReport = false, bool IsAssigned = false) {
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

  void updateEventSyncTypeInfo(
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
      M.insert(std::make_pair(
          Offset, EventSyncTypeInfo(Length, ReplText, false, false)));
    }
  }

  void insertTimeStubTypeInfo(
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
    M.insert(
        std::make_pair(Offset, TimeStubTypeInfo(Length, StrWithSubmitBarrier,
                                                StrWithoutSubmitBarrier)));
  }

  void updateTimeStubTypeInfo(SourceLocation BeginLoc, SourceLocation EndLoc) {

    auto LocInfo = getLocInfo(BeginLoc);
    auto FileInfo = insertFile(LocInfo.first);

    size_t Begin = getLocInfo(BeginLoc).second;
    size_t End = getLocInfo(EndLoc).second;
    auto &TimeStubBounds = FileInfo->getTimeStubBounds();
    TimeStubBounds.push_back(std::make_pair(Begin, End));
  }

  void insertBuiltinVarInfo(SourceLocation SL, unsigned int Len,
                            std::string Repl,
                            std::shared_ptr<DeviceFunctionInfo> DFI);

  void insertSpBLASWarningLocOffset(SourceLocation SL) {
    auto LocInfo = getLocInfo(SL);
    auto FileInfo = insertFile(LocInfo.first);
    FileInfo->getSpBLASSet().insert(LocInfo.second);
  }

  std::shared_ptr<TextModification> findConstantMacroTMInfo(SourceLocation SL) {
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

  void insertConstantMacroTMInfo(SourceLocation SL,
                                 std::shared_ptr<TextModification> TM) {
    auto LocInfo = getLocInfo(SL);
    auto FileInfo = insertFile(LocInfo.first);
    TM->setConstantOffset(LocInfo.second);
    auto &S = FileInfo->getConstantMacroTMSet();
    S.insert(TM);
  }

  void insertAtomicInfo(std::string HashStr, SourceLocation SL,
                        std::string FuncName) {
    auto LocInfo = getLocInfo(SL);
    auto FileInfo = insertFile(LocInfo.first);
    auto &M = FileInfo->getAtomicMap();
    if (M.find(HashStr) == M.end()) {
      M.insert(std::make_pair(HashStr,
                              std::make_tuple(LocInfo.second, FuncName, true)));
    }
  }

  void removeAtomicInfo(std::string HashStr) {
    for (auto &File : FileMap) {
      auto &M = File.second->getAtomicMap();
      auto Iter = M.find(HashStr);
      if (Iter != M.end()) {
        std::get<2>(Iter->second) = false;
        return;
      }
    }
  }

  void setFileEnterLocation(SourceLocation Loc) {
    auto LocInfo = getLocInfo(Loc);
    insertFile(LocInfo.first)->setFileEnterOffset(LocInfo.second);
  }

  void setFirstIncludeLocation(SourceLocation Loc) {
    auto LocInfo = getLocInfo(Loc);
    insertFile(LocInfo.first)->setFirstIncludeOffset(LocInfo.second);
  }

  void setLastIncludeLocation(SourceLocation Loc) {
    auto LocInfo = getLocInfo(Loc);
    insertFile(LocInfo.first)->setLastIncludeOffset(LocInfo.second);
  }

  void setMathHeaderInserted(SourceLocation Loc, bool B) {
    auto LocInfo = getLocInfo(Loc);
    insertFile(LocInfo.first)->setMathHeaderInserted(B);
  }

  void setAlgorithmHeaderInserted(SourceLocation Loc, bool B) {
    auto LocInfo = getLocInfo(Loc);
    insertFile(LocInfo.first)->setAlgorithmHeaderInserted(B);
  }

  void setTimeHeaderInserted(SourceLocation Loc, bool B) {
    auto LocInfo = getLocInfo(Loc);
    insertFile(LocInfo.first)->setTimeHeaderInserted(B);
  }

  void insertHeader(SourceLocation Loc, HeaderType Type) {
    auto LocInfo = getLocInfo(Loc);
    insertFile(LocInfo.first)->insertHeader(Type);
  }

  void insertHeader(SourceLocation Loc, std::string HeaderName) {
    auto LocInfo = getLocInfo(Loc);
    insertFile(LocInfo.first)->insertCustomizedHeader(std::move(HeaderName));
  }

  static std::unordered_map<std::string, SourceRange> &getExpansionRangeBeginMap() {
    return ExpansionRangeBeginMap;
  }

  static std::map<std::string, std::shared_ptr<MacroExpansionRecord>> &
  getExpansionRangeToMacroRecord() {
    return ExpansionRangeToMacroRecord;
  }

  static std::map<std::string, std::shared_ptr<DpctGlobalInfo::MacroDefRecord>>
      &getMacroTokenToMacroDefineLoc() {
    return MacroTokenToMacroDefineLoc;
  }

  static std::map<std::string, std::string> &
  getFunctionCallInMacroMigrateRecord() {
    return FunctionCallInMacroMigrateRecord;
  }

  static std::map<std::string, SourceLocation> &getEndifLocationOfIfdef() {
    return EndifLocationOfIfdef;
  }

  static std::vector<std::pair<std::string, size_t>> &
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
  static std::set<std::string> &getIncludingFileSet() {
    return IncludingFileSet;
  }
  static std::set<std::string> &getFileSetInCompiationDB() {
    return FileSetInCompiationDB;
  }
  static std::unordered_map<std::string,
                            std::vector<clang::tooling::Replacement>> &
  getFileRelpsMap() {
    return FileRelpsMap;
  }
  static std::unordered_map<std::string, std::string> &getDigestMap() {
    return DigestMap;
  }
  static std::string getYamlFileName() { return YamlFileName; }

  static std::set<std::string> &getGlobalVarNameSet() {
    return GlobalVarNameSet;
  }
  static void removeVarNameInGlobalVarNameSet(const std::string &VarName) {
    auto Iter = getGlobalVarNameSet().find(VarName);
    if (Iter != getGlobalVarNameSet().end()) {
      getGlobalVarNameSet().erase(Iter);
    }
  }
  static bool getDeviceChangedFlag() { return HasFoundDeviceChanged; }
  static void setDeviceChangedFlag(bool Flag) { HasFoundDeviceChanged = Flag; }
  static std::unordered_map<int, HelperFuncReplInfo> &
  getHelperFuncReplInfoMap() {
    return HelperFuncReplInfoMap;
  }
  static int getHelperFuncReplInfoIndexThenInc() {
    int Res = HelperFuncReplInfoIndex;
    HelperFuncReplInfoIndex++;
    return Res;
  }
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
    return getUsingExperimental<ExperimentalFeatures::Exp_UserDefineReductions>();
  }
  static bool useMaskedSubGroupFunction() {
    return getUsingExperimental<ExperimentalFeatures::Exp_MaskedSubGroupFunction>();
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
  static bool useNoQueueDevice() {
    return getHelperFuncPreference(HelperFuncPreference::NoQueueDevice);
  }
  static bool useEnqueueBarrier() {
    return getUsingExtensionDE(DPCPPExtensionsDefaultEnabled::ExtDE_EnqueueBarrier);
  }
  static bool useCAndCXXStandardLibrariesExt() {
    return getUsingExtensionDD(DPCPPExtensionsDefaultDisabled::ExtDD_CCXXStandardLibrary);
  }
  static bool useIntelDeviceMath() {
    return getUsingExtensionDD(DPCPPExtensionsDefaultDisabled::ExtDD_IntelDeviceMath);
  }

  static bool useDeviceInfo() {
    return getUsingExtensionDE(DPCPPExtensionsDefaultEnabled::ExtDE_DeviceInfo);
  }

  static bool useBFloat16() {
    return getUsingExtensionDE(DPCPPExtensionsDefaultEnabled::ExtDE_BFloat16);
  }

  inline std::shared_ptr<DpctFileInfo> insertFile(const std::string &FilePath) {
    return insertObject(FileMap, FilePath);
  }

  inline std::shared_ptr<DpctFileInfo> getMainFile() const {
    return MainFile;
  }

  inline void setMainFile(std::shared_ptr<DpctFileInfo> Main) {
    MainFile = Main;
  }

  inline void recordIncludingRelationship(const std::string &CurrentFileName,
                                          const std::string &IncludedFileName) {
    auto CurrentFileInfo = this->insertFile(CurrentFileName);
    auto IncludedFileInfo = this->insertFile(IncludedFileName);
    CurrentFileInfo->insertIncludedFilesInfo(IncludedFileInfo);
  }

  static unsigned int getCudaKernelDimDFIIndexThenInc() {
    unsigned int Res = CudaKernelDimDFIIndex;
    ++CudaKernelDimDFIIndex;
    return Res;
  }
  static void
  insertCudaKernelDimDFIMap(unsigned int Index,
                            std::shared_ptr<DeviceFunctionInfo> Ptr) {
    CudaKernelDimDFIMap.insert(std::make_pair(Index, Ptr));
  }
  static std::shared_ptr<DeviceFunctionInfo>
  getCudaKernelDimDFI(unsigned int Index) {
    auto Iter = CudaKernelDimDFIMap.find(Index);
    if (Iter != CudaKernelDimDFIMap.end())
      return Iter->second;
    return nullptr;
  }

  static std::set<std::string> &getModuleFiles() { return ModuleFiles; }

  // #tokens, name of the second token, SourceRange of a macro
  static std::tuple<unsigned int, std::string, SourceRange> LastMacroRecord;

  static void setRunRound(unsigned int Round) { RunRound = Round; }
  static unsigned int getRunRound() { return RunRound; }
  static void setNeedRunAgain(bool NRA) { NeedRunAgain = NRA; }
  static bool isNeedRunAgain() { return NeedRunAgain; }
  static std::unordered_map<std::string, std::shared_ptr<ExtReplacements>> &
  getFileReplCache() {
    return FileReplCache;
  }
  void resetInfo();
  static inline void
  updateSpellingLocDFIMaps(SourceLocation SL,
                           std::shared_ptr<DeviceFunctionInfo> DFI) {
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

  static inline std::unordered_set<std::shared_ptr<DeviceFunctionInfo>>
  getDFIVecRelatedFromSpellingLoc(std::shared_ptr<DeviceFunctionInfo> DFI) {
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
  static unsigned int getColorOption() { return ColorOption; }
  static void setColorOption(unsigned Color)  { ColorOption = Color; }
  std::unordered_map<int, std::shared_ptr<DeviceFunctionInfo>> &
  getCubPlaceholderIndexMap() {
    return CubPlaceholderIndexMap;
  }
  static inline std::unordered_map<std::string,
                                   std::shared_ptr<PriorityReplInfo>> &
  getPriorityReplInfoMap() {
    return PriorityReplInfoMap;
  }
  // For PriorityRelpInfo with same key, the Info with low priority will
  // be filtered and the Info with same priority will be merged.
  static inline void
  addPriorityReplInfo(std::string Key, std::shared_ptr<PriorityReplInfo> Info) {
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

  static void setOptimizeMigrationFlag(bool Flag) {
    OptimizeMigrationFlag = Flag;
  }
  static bool isOptimizeMigration() { return OptimizeMigrationFlag; }

  static inline std::map<std::string, clang::tooling::OptionInfo> &
  getCurrentOptMap() {
    return CurrentOptMap;
  }
  static inline void setMainSourceYamlTUR(
      std::shared_ptr<clang::tooling::TranslationUnitReplacements> Ptr) {
    MainSourceYamlTUR = Ptr;
  }
  static inline std::shared_ptr<clang::tooling::TranslationUnitReplacements>
  getMainSourceYamlTUR() {
    return MainSourceYamlTUR;
  }
  static inline std::unordered_map<
      std::string, std::unordered_map<std::string, std::vector<unsigned>>> &
  getRnnInputMap() {
    return RnnInputMap;
  }
  static inline std::unordered_map<std::string, std::vector<std::string>> &
  getMainSourceFileMap(){
    return MainSourceFileMap;
  };
  static inline std::unordered_map<std::string, bool> &
  getMallocHostInfoMap(){
    return MallocHostInfoMap;
  };
  static inline std::map<std::shared_ptr<TextModification>, bool> &
  getConstantReplInsertFlagMap() {
    return ConstantReplInsertFlagMap;
  }
  static inline std::set<std::string> &getVarUsedByRuntimeSymbolAPISet() {
    return VarUsedByRuntimeSymbolAPISet;
  }

private:
  DpctGlobalInfo();

  DpctGlobalInfo(const DpctGlobalInfo &) = delete;
  DpctGlobalInfo(DpctGlobalInfo &&) = delete;
  DpctGlobalInfo &operator=(const DpctGlobalInfo &) = delete;
  DpctGlobalInfo &operator=(DpctGlobalInfo &&) = delete;

  // Wrapper of isInAnalysisScope for std::function usage.
  static bool checkInAnalysisScope(SourceLocation SL) { return isInAnalysisScope(SL); }

  // Record token split when it's in macro
  static void recordTokenSplit(SourceLocation SL, unsigned Len) {
    auto It = getExpansionRangeToMacroRecord().find(
        getCombinedStrFromLoc(SM->getSpellingLoc(SL)));
    if (It != getExpansionRangeToMacroRecord().end()) {
      dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord()
          [getCombinedStrFromLoc(
              SM->getSpellingLoc(SL).getLocWithOffset(Len))] = It->second;
    }
  }

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
  static inline SourceLocation getLocation(const VarDecl *VD) {
    return VD->getLocation();
  }
  static inline SourceLocation getLocation(const FunctionDecl *FD) {
    return FD->getBeginLoc();
  }
  static inline SourceLocation getLocation(const FieldDecl *FD) {
    return FD->getLocation();
  }
  static inline SourceLocation getLocation(const CallExpr *CE) {
    return CE->getEndLoc();
  }
  // The result will be also stored in KernelCallExpr.BeginLoc
  static inline SourceLocation getLocation(const CUDAKernelCallExpr *CKC) {
    return getTheLastCompleteImmediateRange(CKC->getBeginLoc(), CKC->getEndLoc()).first;
  }
  std::shared_ptr<DpctFileInfo> MainFile = nullptr;
  std::unordered_map<std::string, std::shared_ptr<DpctFileInfo>> FileMap;
  static std::shared_ptr<clang::tooling::TranslationUnitReplacements>
      MainSourceYamlTUR;
  static std::string InRoot;
  static std::string OutRoot;
  static std::string AnalysisScope;
  static std::unordered_set<std::string> ChangeExtensions;
  // TODO: implement one of this for each source language.
  static std::string CudaPath;
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
  static bool EnableComments;
  static std::string ClNamespace;
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
  static std::unordered_map<std::string, SourceRange> ExpansionRangeBeginMap;
  static bool CheckUnicodeSecurityFlag;
  static bool EnablepProfilingFlag;
  static std::map<std::string,
                  std::shared_ptr<DpctGlobalInfo::MacroExpansionRecord>>
      ExpansionRangeToMacroRecord;
  static std::map<std::string, SourceLocation> EndifLocationOfIfdef;
  static std::vector<std::pair<std::string, size_t>> ConditionalCompilationLoc;
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
  static std::set<std::string> IncludingFileSet;
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
  static bool UsingThisItem;
  static unsigned int CudaKernelDimDFIIndex;
  static std::unordered_map<unsigned int, std::shared_ptr<DeviceFunctionInfo>>
      CudaKernelDimDFIMap;
  static CudaArchPPMap CAPPInfoMap;
  static HDFuncInfoMap HostDeviceFuncInfoMap;
  static CudaArchDefMap CudaArchDefinedMap;
  static std::unordered_map<std::string, std::shared_ptr<ExtReplacement>>
      CudaArchMacroRepl;
  static std::unordered_map<std::string, std::shared_ptr<ExtReplacements>>
      FileReplCache;
  static std::set<std::string> ReProcessFile;
  static std::set<std::string> ProcessedFile;
  static bool NeedRunAgain;
  static unsigned int RunRound;
  static std::set<std::string> ModuleFiles;
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
      std::string, std::unordered_map<std::string, std::vector<unsigned>>>
      RnnInputMap;
  static std::unordered_map<std::string, std::vector<std::string>>
      MainSourceFileMap;
  static std::unordered_map<std::string, bool> MallocHostInfoMap;
  static std::map<std::shared_ptr<TextModification>, bool>
      ConstantReplInsertFlagMap;
  static std::set<std::string> VarUsedByRuntimeSymbolAPISet;
};

/// Generate mangle name of FunctionDecl as key of DeviceFunctionInfo.
/// For template dependent FunctionDecl, generate name with pattern
/// "QuailifiedName@FunctionType".
/// e.g.: template<class T> void test(T *int)
/// -> test@void (type-parameter-0-1 *)
class DpctNameGenerator {
  ASTNameGenerator G;
  PrintingPolicy PP;

  void printName(const FunctionDecl *FD, llvm::raw_ostream &OS) {
    if (G.writeName(FD, OS)) {
      FD->printQualifiedName(OS, PP);
      OS << "@";
      FD->getType().print(OS, PP);
    }
  }

public:
  DpctNameGenerator() : DpctNameGenerator(DpctGlobalInfo::getContext()) {}
  explicit DpctNameGenerator(ASTContext &Ctx)
      : G(Ctx), PP(Ctx.getPrintingPolicy()) {
    PP.PrintCanonicalTypes = true;
  }
  std::string getName(const FunctionDecl *D) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    printName(D, OS);
    return OS.str();
  }
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
  SizeInfo(std::shared_ptr<TemplateDependentStringInfo> TDSI) : TDSI(TDSI) {}
  const std::string &getSize() {
    if (TDSI)
      return TDSI->getSourceString();
    return Size;
  }
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
  CtTypeInfo(const VarDecl *D, bool NeedSizeFold = false)
      : PointerLevel(0), IsReference(false), IsTemplate(false) {
    if (D && D->getTypeSourceInfo()) {
      auto TL = D->getTypeSourceInfo()->getTypeLoc();
      IsConstantQualified = D->hasAttr<CUDAConstantAttr>();
      setTypeInfo(TL, NeedSizeFold);
      if (TL.getTypeLocClass() ==
          TypeLoc::IncompleteArray) {
        if (auto CAT = dyn_cast<ConstantArrayType>(D->getType())) {
          Range[0] = std::to_string(CAT->getSize().getZExtValue());
        }
      }
    }
  }

  inline const std::string &getBaseName() { return BaseName; }

  inline size_t getDimension() { return Range.size(); }
  inline std::vector<SizeInfo> &getRange() { return Range; }
  // when there is no arguments, parameter MustArguments determine whether
  // parens will exist. Null string will be returned when MustArguments is
  // false, otherwise "()" will be returned.
  std::string getRangeArgument(const std::string &MemSize, bool MustArguments);

  inline bool isTemplate() const { return IsTemplate; }
  inline bool isPointer() const { return PointerLevel; }
  inline bool isArray() const { return IsArray; }
  inline bool isReference() const { return IsReference; }
  inline void adjustAsMemType() {
    setPointerAsArray();
    removeQualifier();
  }

  // Get instantiated type name with given template arguments.
  // e.g. X<T>, with T = int, result type will be X<int>.
  std::shared_ptr<CtTypeInfo>
  applyTemplateArguments(const std::vector<TemplateArgumentInfo> &TA);

  bool isWritten() const {
    return !TDSI || !isTemplate() || TDSI->isDependOnWritten();
  }
  std::set<HelperFeatureEnum> getHelperFeatureSet() { return HelperFeatureSet; }
  inline bool containSizeofType() { return ContainSizeofType; }
  inline std::vector<std::string> getArraySizeOriginExprs() { return ArraySizeOriginExprs; }

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

  void setPointerAsArray() {
    if (isPointer()) {
      --PointerLevel;
      Range.emplace_back();
      updateName();
    }
  }
  inline void removeQualifier() { BaseName = BaseNameWithoutQualifiers; }

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
  VarInfo(unsigned Offset, const std::string &FilePathIn,
          const VarDecl *Var, bool NeedFoldSize = false)
      : FilePath(FilePathIn), Offset(Offset), Name(Var->getName()),
        Ty(std::make_shared<CtTypeInfo>(Var, NeedFoldSize)) {}

  inline const std::string &getFilePath() { return FilePath; }
  inline unsigned getOffset() { return Offset; }
  inline const std::string &getName() { return Name; }
  inline const std::string getNameAppendSuffix() { return Name + "_ct1"; }
  inline std::shared_ptr<CtTypeInfo> &getType() { return Ty; }

  inline std::string getDerefName() {
    return buildString(getName(), "_deref_", DpctGlobalInfo::getInRootHash());
  }

  inline void
  applyTemplateArguments(const std::vector<TemplateArgumentInfo> &TAList) {
    Ty = Ty->applyTemplateArguments(TAList);
  }
  inline void requestFeatureForSet(const std::string &Path) {
    if (Ty) {
      for (const auto &Item : Ty->getHelperFeatureSet()) {
        requestFeature(Item);
      }
    }
  }

private:
  const std::string FilePath;
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
  static VarAttrKind getAddressAttr(const VarDecl *VD) {
    if (VD->hasAttrs())
      return getAddressAttr(VD->getAttrs());
    return Host;
  }

  MemVarInfo(unsigned Offset, const std::string &FilePath, const VarDecl *Var);

  VarAttrKind getAttr() { return Attr; }
  VarScope getScope() { return Scope; }
  bool isGlobal() { return Scope == Global; }
  bool isExtern() { return Scope == Extern; }
  bool isLocal() { return Scope == Local; }
  bool isShared() { return Attr == Shared; }
  bool isTypeDeclaredLocal() { return IsTypeDeclaredLocal; }
  bool isAnonymousType() { return IsAnonymousType; }
  const CXXRecordDecl *getDeclOfVarType() { return DeclOfVarType; }
  const DeclStmt *getDeclStmtOfVarType() { return DeclStmtOfVarType; }
  void setLocalTypeName(std::string T) { LocalTypeName = T; }
  std::string getLocalTypeName() { return LocalTypeName; }
  void setIgnoreFlag(bool Flag) { IsIgnored = Flag; }
  bool isIgnore() { return IsIgnored; }
  bool isStatic() { return IsStatic; }

  inline void setName(std::string NewName) { NewConstVarName = NewName; }

  inline unsigned int getNewConstVarOffset() { return NewConstVarOffset; }
  inline unsigned int getNewConstVarLength() { return NewConstVarLength; }

  inline const std::string getConstVarName() {
    return NewConstVarName.empty() ? getArgName() : NewConstVarName;
  }

  // Initialize offset and length for __constant__ variable that needs to be
  // renamed.
  void newConstVarInit(const VarDecl *Var) {
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

  std::string getDeclarationReplacement(const VarDecl *);

  std::string getInitStmt() { return getInitStmt(""); }
  std::string getInitStmt(StringRef QueueString) {
    if (QueueString.empty())
      return getConstVarName() + ".init();";
    return buildString(getConstVarName(), ".init(", QueueString, ");");
  }

  inline std::string getMemoryDecl(const std::string &MemSize) {
    return buildString(isStatic() ? "static " : "", getMemoryType(), " ",
                       getConstVarName(),
                       PointerAsArray ? "" : getInitArguments(MemSize), ";");
  }
  std::string getMemoryDecl() {
    const static std::string NullString;
    return getMemoryDecl(NullString);
  }

  std::string getExternGlobalVarDecl() {
    return buildString("extern ", getMemoryType(), " ", getConstVarName(), ";");
  }

  void appendAccessorOrPointerDecl(const std::string &ExternMemSize,
                                   bool ExternEmitWarning, StmtList &AccList,
                                   StmtList &PtrList);

  inline std::string getRangeClass() {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    return DpctGlobalInfo::printCtadClass(OS,
                                          MapNames::getClNamespace() + "range",
                                          getType()->getDimension())
        .str();
  }
  std::string getRangeDecl(const std::string &MemSize) {
    return buildString(getRangeClass(), " ", getRangeName(),
                       getType()->getRangeArgument(MemSize, false), ";");
  }
  ParameterStream &getFuncDecl(ParameterStream &PS) {
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
  ParameterStream &getFuncArg(ParameterStream &PS) {
    return PS << getArgName();
  }
  ParameterStream &getKernelArg(ParameterStream &PS) {
    if (isShared() || DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      if (AccMode == Pointer) {
        if (!getType()->isWritten())
          PS << "(" << getAccessorDataType(false, true) << " *)";
        PS << getAccessorName() << ".get_pointer()";
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
  std::string getAccessorDataType(bool IsTypeUsedInDevFunDecl = false,
                                  bool NeedCheckExtraConstQualifier = false) {
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
  void setNotUseDpctHelperTypeFlag(bool Flag) {
    NotUseDpctHelperTypeFlag = Flag;
  };
  bool isNotUseDpctHelperType() { return NotUseDpctHelperTypeFlag; }

private:
  bool isTreatPointerAsArray() {
    return getType()->isPointer() && getScope() == Global &&
           DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None;
  }

  static VarAttrKind getAddressAttr(const AttrVec &Attrs);

  void setInitList(const Expr *E, const VarDecl *V) {
    if (auto Ctor = dyn_cast<CXXConstructExpr>(E)) {
      if (!Ctor->getNumArgs() || Ctor->getArg(0)->isDefaultArgument())
        return;
    }
    InitList = getStmtSpelling(E, V->getSourceRange());
  }

  std::string getMemoryType();
  inline std::string getMemoryType(const std::string &MemoryType,
                                   std::shared_ptr<CtTypeInfo> VarType);
  std::string getInitArguments(const std::string &MemSize,
                               bool MustArguments = false);
  const std::string &getMemoryAttr();
  std::string getSyclAccessorType();
  std::string getDpctAccessorType() {
    requestFeature(HelperFeatureEnum::device_ext);
    auto Type = getType();
    return buildString(MapNames::getDpctNamespace(true), "accessor<",
                       getAccessorDataType(), ", ", getMemoryAttr(), ", ",
                       Type->getDimension(), ">");
  }
  inline std::string getNameWithSuffix(StringRef Suffix) {
    return buildString(getArgName(), "_", Suffix, getCTFixedSuffix());
  }
  inline std::string getAccessorName() { return getNameWithSuffix("acc"); }
  inline std::string getPtrName() { return getNameWithSuffix("ptr"); }
  inline std::string getRangeName() { return getNameWithSuffix("range"); }
  std::string getArgName() {
    if (isExtern())
      return ExternVariableName;
    else if (isTypeDeclaredLocal())
      return getNameAppendSuffix();
    return getName();
  }

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
  bool NotUseDpctHelperTypeFlag = false;
};

class TextureTypeInfo {
  std::string DataType;
  int Dimension;
  bool IsArray;

public:
  TextureTypeInfo(std::string &&DataType, int TexType) {
    setDataTypeAndTexType(std::move(DataType), TexType);
  }

  void setDataTypeAndTexType(std::string &&Type, int TexType) {
    DataType = std::move(Type);
    IsArray = TexType & 0xF0;
    Dimension = TexType & 0x0F;
    // The DataType won't use dpct helper feature
    MapNames::replaceName(MapNames::TypeNamesMap, DataType);
  }

  void prepareForImage() {
    if (IsArray)
      ++Dimension;
  }
  void endForImage() {
    if (IsArray)
      --Dimension;
  }
  std::string getDataType() { return DataType; }
  ParameterStream &printType(ParameterStream &PS,
                             const std::string &TemplateName) {
    PS << TemplateName << "<" << DataType << ", " << Dimension;
    if (IsArray)
      PS << ", true";
    PS << ">";
    return PS;
  }
};

class TextureInfo {
protected:
  const std::string FilePath;
  const unsigned Offset;
  std::string Name; // original expression str
  std::string NewVarName; // name of new variable which tool

  std::shared_ptr<TextureTypeInfo> Type;

protected:
  TextureInfo(unsigned Offset, const std::string &FilePath, StringRef Name)
      : FilePath(FilePath), Offset(Offset), Name(Name) {
    NewVarName = Name.str();
    for (auto &C : NewVarName) {
      if ((!isDigit(C)) && (!isLetter(C)) && (C != '_'))
        C = '_';
    }
    if (NewVarName.size() > 1 && NewVarName[NewVarName.size() - 1] == '_')
      NewVarName.pop_back();
  }
  TextureInfo(const VarDecl *VD)
      : TextureInfo(DpctGlobalInfo::getLocInfo(
                        VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
                    VD->getName()) {}
  TextureInfo(const VarDecl *VD, std::string Subscript)
      : TextureInfo(DpctGlobalInfo::getLocInfo(
                        VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
                    VD->getName().str() + "[" + Subscript + "]") {}
  TextureInfo(std::pair<StringRef, unsigned> LocInfo, StringRef Name)
      : TextureInfo(LocInfo.second, LocInfo.first.str(), Name) {}

  ParameterStream &getDecl(ParameterStream &PS,
                           const std::string &TemplateDeclName) {
    return Type->printType(PS, MapNames::getDpctNamespace() + TemplateDeclName)
           << " " << Name;
  }

  template <class StreamT>
  static void printQueueStr(StreamT &OS, const std::string &Queue) {
    if (Queue.empty())
      return;
    OS << ", " << Queue;
  }

public:
  TextureInfo(unsigned Offset, const std::string &FilePath, const VarDecl *VD)
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
              Arg1.getAsExpr()->EvaluateAsInt(ER,
                                              DpctGlobalInfo::getContext())) {
            int64_t Value = ER.Val.getInt().getExtValue();
            setType(DpctGlobalInfo::getUnqualifiedTypeName(DataTy), Value);
          }
        }
      }
    }
  }

  virtual ~TextureInfo() = default;
  void setType(std::string &&DataType, int TexType) {
    setType(std::make_shared<TextureTypeInfo>(std::move(DataType), TexType));
  }
  inline void setType(std::shared_ptr<TextureTypeInfo> TypeInfo) {
    if (TypeInfo)
      Type = TypeInfo;
  }

  inline std::shared_ptr<TextureTypeInfo> getType() const { return Type; }

  virtual std::string getHostDeclString() {
    ParameterStream PS;
    Type->prepareForImage();
    requestFeature(HelperFeatureEnum::device_ext);

    getDecl(PS, "image_wrapper") << ";";
    Type->endForImage();
    return PS.Str;
  }

  virtual std::string getSamplerDecl() {
    requestFeature(HelperFeatureEnum::device_ext);
    return buildString("auto ", NewVarName, "_smpl = ", Name,
                       ".get_sampler();");
  }
  virtual std::string getAccessorDecl(const std::string &QueueStr) {
    requestFeature(HelperFeatureEnum::device_ext);
    std::string Ret;
    llvm::raw_string_ostream OS(Ret);
    OS << "auto " << NewVarName << "_acc = " << Name << ".get_access(cgh";
    printQueueStr(OS, QueueStr);
    OS << ");";
    return Ret;
  }
  virtual void addDecl(StmtList &AccessorList, StmtList &SamplerList, const std::string&QueueStr) {
    AccessorList.emplace_back(getAccessorDecl(QueueStr));
    SamplerList.emplace_back(getSamplerDecl());
  }

  inline ParameterStream &getFuncDecl(ParameterStream &PS) {
    requestFeature(HelperFeatureEnum::device_ext);
    return getDecl(PS, "image_accessor_ext");
  }
  inline ParameterStream &getFuncArg(ParameterStream &PS) { return PS << Name; }
  virtual ParameterStream &getKernelArg(ParameterStream &OS) {
    requestFeature(HelperFeatureEnum::device_ext);
    getType()->printType(OS,
                         MapNames::getDpctNamespace() + "image_accessor_ext");
    OS << "(" << NewVarName << "_smpl, " << NewVarName << "_acc)";
    return OS;
  }
  inline const std::string &getName() { return Name; }

  inline unsigned getOffset() { return Offset; }
  inline std::string getFilePath() { return FilePath; }
  inline bool isNotUseDpctHelperType() { return false; }
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
  TextureObjectInfo(unsigned Offset, const std::string &FilePath,
                    StringRef Name)
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
  std::string getAccessorDecl(const std::string &QueueString) override {
    ParameterStream PS;

    PS << "auto " << NewVarName << "_acc = static_cast<";
    getType()->printType(PS, MapNames::getDpctNamespace() + "image_wrapper")
        << " *>(" << Name << ")->get_access(cgh";
    printQueueStr(PS, QueueString);
    PS << ");";
    requestFeature(HelperFeatureEnum::device_ext);
    return PS.Str;
  }
  std::string getSamplerDecl() override {
    requestFeature(HelperFeatureEnum::device_ext);
    return buildString("auto ", NewVarName, "_smpl = ", Name,
                       "->get_sampler();");
  }
  inline unsigned getParamIdx() const { return ParamIdx; }

  std::string getParamDeclType() {
    requestFeature(HelperFeatureEnum::device_ext);
    ParameterStream PS;
    Type->printType(PS, MapNames::getDpctNamespace() + "image_accessor_ext");
    return PS.Str;
  }

  virtual void merge(std::shared_ptr<TextureObjectInfo> Target) {
    if (Target)
      setType(Target->getType());
  }

  virtual void addParamDeclReplacement() {
    if (Type) {
      DpctGlobalInfo::getInstance().addReplacement(
          std::make_shared<ExtReplacement>(FilePath, Offset, ReplaceTypeLength,
                                           getParamDeclType(), nullptr));
    }
  }

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
  std::string getAccessorDecl(const std::string &QueueString) override {
    requestFeature(HelperFeatureEnum::device_ext);
    ParameterStream PS;
    PS << "auto " << Name << "_acc = static_cast<";
    getType()->printType(PS, MapNames::getDpctNamespace() + "image_wrapper")
        << " *>(" << ArgStr << ")->get_access(cgh";
    printQueueStr(PS, QueueString);
    PS << ");";
    return PS.Str;
  }
  std::string getSamplerDecl() override {
    requestFeature(HelperFeatureEnum::device_ext);
    return buildString("auto ", Name, "_smpl = (", ArgStr, ")->get_sampler();");
  }
};

class MemberTextureObjectInfo : public TextureObjectInfo {
  StringRef BaseName;
  std::string MemberName;

  class NewVarNameRAII {
    std::string OldName;
    MemberTextureObjectInfo *Member;

  public:
    NewVarNameRAII(MemberTextureObjectInfo *M)
        : OldName(std::move(M->Name)), Member(M) {
      Member->Name = buildString(M->BaseName, '.', M->MemberName);
    }
    ~NewVarNameRAII() { Member->Name = std::move(OldName); }
  };

  MemberTextureObjectInfo(unsigned Offset, const std::string& FilePath,
    StringRef Name)
    : TextureObjectInfo(Offset, FilePath, Name) {}

public:
  static std::shared_ptr<MemberTextureObjectInfo> create(const MemberExpr* ME) {
    auto LocInfo = DpctGlobalInfo::getLocInfo(ME);
    auto Ret = std::shared_ptr<MemberTextureObjectInfo>(
        new MemberTextureObjectInfo(LocInfo.second, LocInfo.first,
                                    getTempNameForExpr(ME, false, false)));
    Ret->MemberName = ME->getMemberDecl()->getNameAsString();
    return Ret;
  }
  void addDecl(StmtList &AccessorList, StmtList &SamplerList,
               const std::string &QueueStr) override {
    NewVarNameRAII RAII(this);
    TextureObjectInfo::addDecl(AccessorList, SamplerList, QueueStr);
  }
  void setBaseName(StringRef Name) { BaseName = Name; }
  StringRef getMemberName() { return MemberName; }
};

class StructureTextureObjectInfo : public TextureObjectInfo {
  std::unordered_map<std::string, std::shared_ptr<MemberTextureObjectInfo>>
      Members;
  bool ContainsVirtualPointer;
  bool IsBase = false;

  StructureTextureObjectInfo(unsigned Offset, const std::string &FilePath,
                             StringRef Name)
      : TextureObjectInfo(Offset, FilePath, Name) {}

public:
  StructureTextureObjectInfo(const ParmVarDecl *PVD) : TextureObjectInfo(PVD) {
    ContainsVirtualPointer =
        checkPointerInStructRecursively(getRecordDecl(PVD->getType()));
    setType("", 0);
  }
  StructureTextureObjectInfo(const VarDecl *VD) : TextureObjectInfo(VD) {
    ContainsVirtualPointer =
        checkPointerInStructRecursively(getRecordDecl(VD->getType()));
    setType("", 0);
  }
  static std::shared_ptr<StructureTextureObjectInfo>
  create(const CXXThisExpr *This);
  bool isBase() const { return IsBase; }
  bool containsVirtualPointer() const { return ContainsVirtualPointer; }
  std::shared_ptr<MemberTextureObjectInfo> addMember(const MemberExpr *ME) {
    auto Member = MemberTextureObjectInfo::create(ME);
    return Members.emplace(Member->getMemberName().str(), Member).first->second;
  }
  void addDecl(StmtList &AccessorList, StmtList &SamplerList,
               const std::string &Queue) override {
    for (const auto &M : Members) {
      M.second->setBaseName(Name);
    }
  }
  void addParamDeclReplacement() override { return; }
  void merge(std::shared_ptr<StructureTextureObjectInfo> Target);
  void merge(std::shared_ptr<TextureObjectInfo> Target) override {
    merge(std::dynamic_pointer_cast<StructureTextureObjectInfo>(Target));
  }
  ParameterStream &getKernelArg(ParameterStream &OS) override {
    OS << Name;
    return OS;
  }
};

class TemplateArgumentInfo {
public:
  explicit TemplateArgumentInfo(const TemplateArgumentLoc &TAL,
                                SourceRange Range)
      : Kind(TAL.getArgument().getKind()) {
    setArgFromExprAnalysis(
        TAL, getDefinitionRange(Range.getBegin(), Range.getEnd()));
  }

  explicit TemplateArgumentInfo(std::string &&Str)
      : Kind(TemplateArgument::Null) {
    setArgStr(std::move(Str));
  }
  TemplateArgumentInfo() : Kind(TemplateArgument::Null), IsWritten(false) {}

  inline bool isWritten() const { return IsWritten; }
  inline bool isNull() const { return !DependentStr; }
  inline bool isType() const { return Kind == TemplateArgument::Type; }
  inline const std::string &getString() const {
    return getDependentStringInfo()->getSourceString();
  }
  inline std::shared_ptr<const TemplateDependentStringInfo>
  getDependentStringInfo() const {
    if (isNull()) {
      static std::shared_ptr<TemplateDependentStringInfo> Placeholder =
          std::make_shared<TemplateDependentStringInfo>(
              "dpct_placeholder/*Fix the type mannually*/");
      return Placeholder;
    }
    return DependentStr;
  }
  void setAsType(QualType QT) {
    if (isPlaceholderType(QT))
      return;
    setArgStr(DpctGlobalInfo::getReplacedTypeName(QT));
    Kind = TemplateArgument::Type;
  }
  void setAsType(const TypeLoc &TL) {
    setArgFromExprAnalysis(TL);
    Kind = TemplateArgument::Type;
  }
  void setAsType(std::string TS) {
    setArgStr(std::move(TS));
    Kind = TemplateArgument::Type;
  }
  void setAsNonType(const llvm::APInt &Int) {
    setArgStr(toString(Int, 10, true, false));
    Kind = TemplateArgument::Integral;
  }
  void setAsNonType(const Expr *E) {
    setArgFromExprAnalysis(E);
    Kind = TemplateArgument::Expression;
  }

  static bool isPlaceholderType(clang::QualType QT);

private:
  template <class T>
  void setArgFromExprAnalysis(const T &Arg, SourceRange ParentRange = SourceRange()) {
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

  void setArgStr(std::string &&Str) {
    DependentStr =
        std::make_shared<TemplateDependentStringInfo>(std::move(Str));
  }
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
  bool hasItem() const { return HasItem; }
  bool hasStream() const { return HasStream; }
  bool hasSync() const { return HasSync; }
  bool hasBF64() const { return HasBF64; }
  bool hasBF16() const { return HasBF16; }
  bool hasGlobalMemAcc() const { return HasGlobalMemAcc; }
  bool hasExternShared() const { return !ExternVarMap.empty(); }
  inline void setItem(bool Has = true) { HasItem = Has; }
  inline void setStream(bool Has = true) { HasStream = Has; }
  inline void setSync(bool Has = true) { HasSync = Has; }
  inline void setBF64(bool Has = true) { HasBF64 = Has; }
  inline void setBF16(bool Has = true) { HasBF16 = Has; }
  inline void setGlobalMemAcc(bool Has = true) { HasGlobalMemAcc = Has; }
  inline void addTexture(std::shared_ptr<TextureInfo> Tex) {
    TextureMap.insert(std::make_pair(Tex->getOffset(), Tex));
  }
  void addVar(std::shared_ptr<MemVarInfo> Var) {
    auto Attr = Var->getAttr();
    if (Var->isGlobal() && (Attr == MemVarInfo::VarAttrKind::Device ||
                            Attr == MemVarInfo::VarAttrKind::Managed)) {
      setGlobalMemAcc(true);
    }
    getMap(Var->getScope())
        .insert(MemVarInfoMap::value_type(Var->getOffset(), Var));
  }
  inline void merge(const MemVarMap &OtherMap) {
    static std::vector<TemplateArgumentInfo> NullTemplates;
    return merge(OtherMap, NullTemplates);
  }
  void merge(const MemVarMap &VarMap,
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
    dpct::merge(TextureMap, VarMap.TextureMap);
  }
  int calculateExtraArgsSize() const {
    int Size = 0;
    if (hasStream())
      Size += MapNames::KernelArgTypeSizeMap.at(KernelArgType::KAT_Stream);

    Size = Size + calculateExtraArgsSize(LocalVarMap) +
           calculateExtraArgsSize(GlobalVarMap) +
           calculateExtraArgsSize(ExternVarMap);
    Size = Size + TextureMap.size() * MapNames::KernelArgTypeSizeMap.at(
                                          KernelArgType::KAT_Texture);

    return Size;
  }
  std::string getExtraCallArguments(bool HasPreParam, bool HasPostParam) const;
  void requestFeatureForAllVarMaps(const std::string &Path) const {
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

  // When adding the ExtraParam with new line, the second argument should be
  // true, and the third argument is the string of indent, which will occur
  // before each ExtraParam.
  std::string
  getExtraDeclParam(bool HasPreParam, bool HasPostParam,
                    FormatInfo FormatInformation = FormatInfo()) const;
  std::string getKernelArguments(bool HasPreParam, bool HasPostParam,
                                 const std::string &Path) const;

  const MemVarInfoMap &getMap(MemVarInfo::VarScope Scope) const {
    return const_cast<MemVarMap *>(this)->getMap(Scope);
  }
  const GlobalMap<TextureInfo> &getTextureMap() const { return TextureMap; }
  void removeDuplicateVar();

  MemVarInfoMap &getMap(MemVarInfo::VarScope Scope) {
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
  bool isSameAs(const MemVarMap &Other) const;

  enum CallOrDecl {
    CallArgument = 0,
    KernelArgument,
    DeclParameter,
  };

  static const MemVarMap *
  getHeadWithoutPathCompression(const MemVarMap *CurNode) {
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

  static MemVarMap *getHead(MemVarMap *CurNode) {
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

  unsigned int getHeadNodeDim() const {
    auto Ptr = getHeadWithoutPathCompression(this);
    if (Ptr)
      return Ptr->Dim;
    else
      return 3;
  }

private:
  static void merge(MemVarInfoMap &Master, const MemVarInfoMap &Branch,
                    const std::vector<TemplateArgumentInfo> &TemplateArgs) {
    if (TemplateArgs.empty())
      return dpct::merge(Master, Branch);
    for (auto &VarInfoPair : Branch)
      Master
          .insert(
              std::make_pair(VarInfoPair.first,
                             std::make_shared<MemVarInfo>(*VarInfoPair.second)))
          .first->second->applyTemplateArguments(TemplateArgs);
  }
  int calculateExtraArgsSize(const MemVarInfoMap &Map) const {
    int Size = 0;
    for (auto &VarInfoPair : Map) {
      auto D = VarInfoPair.second->getType()->getDimension();
      Size += MapNames::getArrayTypeSize(D);
    }
    return Size;
  }

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
      if (VI.second->isNotUseDpctHelperType()) {
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
  inline void getArgumentsOrParametersForDecl(ParameterStream &PS,
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

inline void MemVarMap::getArgumentsOrParametersForDecl(ParameterStream &PS,
                                                       int PreParams,
                                                       int PostParams) const {
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
      PS, GlobalVarMap);
  getArgumentsOrParametersFromMap<MemVarInfo, MemVarMap::DeclParameter>(
      PS, LocalVarMap);
  getArgumentsOrParametersFromMap<TextureInfo, MemVarMap::DeclParameter>(
      PS, TextureMap);
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
  CallFunctionExpr(unsigned Offset, const std::string &FilePathIn, const T &C)
      : FilePath(FilePathIn), BeginLoc(Offset) {}

  void buildCallExprInfo(const CXXConstructExpr *Ctor);
  void buildCallExprInfo(const CallExpr *CE);

  inline const MemVarMap &getVarMap() { return VarMap; }
  inline const std::vector<std::shared_ptr<TextureObjectInfo>> &
  getTextureObjectList() {
    return TextureObjectList;
  }
  std::shared_ptr<StructureTextureObjectInfo> getBaseTextureObjectInfo() const {
    return BaseTextureObject;
  }

  void emplaceReplacement();
  unsigned getExtraArgLoc() { return ExtraArgLoc; }
  inline bool hasArgs() { return HasArgs; }
  inline bool hasTemplateArgs() { return !TemplateArgs.empty(); }
  inline bool hasWrittenTemplateArgs() {
    for (auto &Arg : TemplateArgs)
      if (!Arg.isNull() && Arg.isWritten())
        return true;
    return false;
  }
  inline const std::string &getName() { return Name; }

  std::string getTemplateArguments(bool &IsNeedWarning,
                                   bool WrittenArgsOnly = true,
                                   bool WithScalarWrapped = false);

  virtual std::string getExtraArguments();

  inline void setHasSideEffects(bool Val = true) {
    CallGroupFunctionInControlFlow = Val;
  }
  inline bool hasSideEffects() const { return CallGroupFunctionInControlFlow; }

  std::shared_ptr<TextureObjectInfo>
  addTextureObjectArgInfo(unsigned ArgIdx,
                          std::shared_ptr<TextureObjectInfo> Info) {
    auto &Obj = TextureObjectList[ArgIdx];
    if (!Obj)
      Obj = Info;
    return Obj;
  }
  virtual std::shared_ptr<TextureObjectInfo>
  addTextureObjectArg(unsigned ArgIdx, const DeclRefExpr *TexRef,
                      bool isKernelCall = false);
  virtual std::shared_ptr<TextureObjectInfo>
  addStructureTextureObjectArg(unsigned ArgIdx, const MemberExpr *TexRef,
                               bool isKernelCall = false);
  virtual std::shared_ptr<TextureObjectInfo>
  addTextureObjectArg(unsigned ArgIdx, const ArraySubscriptExpr *TexRef,
                      bool isKernelCall = false);
  std::shared_ptr<DeviceFunctionInfo> getFuncInfo() { return FuncInfo; }
  bool IsAllTemplateArgsSpecified = false;

  virtual ~CallFunctionExpr() = default;

protected:
  void setFuncInfo(std::shared_ptr<DeviceFunctionInfo>);
  std::string Name;
  inline unsigned getBegin() { return BeginLoc; }
  inline const std::string &getFilePath() { return FilePath; }
  void buildInfo();
  void buildCalleeInfo(const Expr *Callee);
  void resizeTextureObjectList(size_t Size) { TextureObjectList.resize(Size); }

private:
  static std::string getName(const NamedDecl *D);
  void
  buildTemplateArguments(const llvm::ArrayRef<TemplateArgumentLoc> &ArgsList,
                         SourceRange Range) {
    if (TemplateArgs.empty())
      for (auto &Arg : ArgsList)
        TemplateArgs.emplace_back(Arg, Range);
  }

  void buildTemplateArgumentsFromTypeLoc(const TypeLoc &TL);
  template <class TyLoc>
  void buildTemplateArgumentsFromSpecializationType(const TyLoc &TL) {
    for (size_t i = 0; i < TL.getNumArgs(); ++i) {
      TemplateArgs.emplace_back(TL.getArgLoc(i), TL.getSourceRange() );
    }
  }

  std::string getNameWithNamespace(const FunctionDecl *FD, const Expr *Callee);

  void buildTextureObjectArgsInfo(const CallExpr* CE);

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
      else if (auto ASE =
                   dyn_cast<ArraySubscriptExpr>(Arg->IgnoreImpCasts()))
        addTextureObjectArg(Idx, ASE, IsKernel);
      Idx++;
      ArgItr++;
    }
  }
  void mergeTextureObjectInfo();

  const std::string FilePath;
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
  DeviceFunctionDecl(unsigned Offset, const std::string &FilePathIn,
                     const FunctionDecl *FD);
  DeviceFunctionDecl(unsigned Offset, const std::string &FilePathIn,
                     const FunctionTypeLoc &FTL, const ParsedAttributes &Attrs,
                     const FunctionDecl *Specialization);
  inline static std::shared_ptr<DeviceFunctionInfo>
  LinkUnresolved(const UnresolvedLookupExpr *ULE) {
    return LinkDeclRange(ULE->decls(), getFunctionName(ULE));
  }
  inline static std::shared_ptr<DeviceFunctionInfo>
  LinkRedecls(const FunctionDecl *FD) {
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
  inline static std::shared_ptr<DeviceFunctionInfo>
  LinkTemplateDecl(const FunctionTemplateDecl *FTD) {
    return LinkDeclRange(FTD->redecls(), getFunctionName(FTD));
  }
  inline static std::shared_ptr<DeviceFunctionInfo> LinkExplicitInstantiation(
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

  inline std::shared_ptr<DeviceFunctionInfo> getFuncInfo() const {
    return FuncInfo;
  }

  virtual void emplaceReplacement();
  static void reset() { FuncInfoMap.clear(); };

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
  void buildReplaceLocInfo(const FunctionDecl *FD);

  virtual ~DeviceFunctionDecl() = default;

protected:
  const FormatInfo &getFormatInfo() { return FormatInformation; }
  void buildTextureObjectParamsInfo(const ArrayRef<ParmVarDecl *> &Parms) {
    TextureObjectList.assign(Parms.size(),
                             std::shared_ptr<TextureObjectInfo>());
    for (unsigned Idx = 0; Idx < Parms.size(); ++Idx) {
      auto Param = Parms[Idx];
      if (DpctGlobalInfo::getUnqualifiedTypeName(Param->getType()) ==
          "cudaTextureObject_t")
        TextureObjectList[Idx] = std::make_shared<TextureObjectInfo>(Param);
    }
  }

  template <class AttrsT>
  void buildReplaceLocInfo(const FunctionTypeLoc &FTL, const AttrsT &Attrs);

  virtual std::string getExtraParameters();

  unsigned Offset;
  const std::string FilePath;
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
  ExplicitInstantiationDecl(unsigned Offset, const std::string &FilePathIn,
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
  std::vector<std::pair<std::string, std::string>> &getParametersInfo() {
    return ParametersInfo;
  }

public:
  DeviceFunctionDeclInModule(unsigned Offset, const std::string &FilePathIn,
                             const FunctionTypeLoc &FTL,
                             const ParsedAttributes &Attrs,
                             const FunctionDecl *FD)
      : DeviceFunctionDecl(Offset, FilePathIn, FTL, Attrs, FD) {
    buildParameterInfo(FD);
    buildWrapperInfo(FD);
    buildCallInfo(FD);
  }
  DeviceFunctionDeclInModule(unsigned Offset, const std::string &FilePathIn,
                             const FunctionDecl *FD)
      : DeviceFunctionDecl(Offset, FilePathIn, FD) {
    buildParameterInfo(FD);
    buildWrapperInfo(FD);
    buildCallInfo(FD);
  }
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
                     std::string FunctionName)
      : ParamsNum(ParamsNum), NonDefaultParamNum(NonDefaultParamNum),
        IsBuilt(false),
        TextureObjectList(ParamsNum, std::shared_ptr<TextureObjectInfo>()),
        FunctionName(FunctionName), IsLambda(false) {
    ParametersProps.resize(ParamsNum);
  }

  bool ConstructGraphVisited = false;

  std::shared_ptr<CallFunctionExpr> findCallee(const CallExpr *C) {
    auto CallLocInfo = DpctGlobalInfo::getLocInfo(C);
    return findObject(CallExprMap, CallLocInfo.second);
  }
  template <class CallT>
  inline std::shared_ptr<CallFunctionExpr> addCallee(const CallT *C) {
    auto CallLocInfo = DpctGlobalInfo::getLocInfo(C);
    auto Call =
        insertObject(CallExprMap, CallLocInfo.second, CallLocInfo.first, C);
    Call->buildCallExprInfo(C);
    return Call;
  }
  inline void addVar(std::shared_ptr<MemVarInfo> Var) { VarMap.addVar(Var); }
  inline void setItem() { VarMap.setItem(); }
  inline void setStream() { VarMap.setStream(); }
  inline void setSync() { VarMap.setSync(); }
  inline void setBF64() { VarMap.setBF64(); }
  inline void setBF16() { VarMap.setBF16(); }
  inline void setGlobalMemAcc() { VarMap.setGlobalMemAcc(); }
  inline void addTexture(std::shared_ptr<TextureInfo> Tex) {
    VarMap.addTexture(Tex);
  }
  inline MemVarMap &getVarMap() { return VarMap; }
  inline std::shared_ptr<TextureObjectInfo> getTextureObject(unsigned Idx) {
    if (Idx < TextureObjectList.size())
      return TextureObjectList[Idx];
    return {};
  }
  std::shared_ptr<StructureTextureObjectInfo> getBaseTextureObject() const {
    return BaseObjectTexture;
  }

  inline void setCallGroupFunctionInControlFlow(bool Val = true) {
    CallGroupFunctionInControlFlow = Val;
  }
  inline bool hasCallGroupFunctionInControlFlow() const {
    return CallGroupFunctionInControlFlow;
  }

  inline void setHasSideEffectsAnalyzed(bool Val = true) {
    HasCheckedCallGroupFunctionInControlFlow = Val;
  }
  inline bool hasSideEffectsAnalyzed() const {
    return HasCheckedCallGroupFunctionInControlFlow;
  }

  void buildInfo();
  inline bool hasParams() { return ParamsNum != 0; }

  inline bool isBuilt() { return IsBuilt; }
  inline void setBuilt() { IsBuilt = true; }

  inline bool isLambda() { return IsLambda; }
  inline void setLambda() { IsLambda = true; }

  inline bool isInlined() { return IsInlined; }
  inline void setInlined() { IsInlined = true; }

  inline bool isKernel() { return IsKernel; }
  inline void setKernel() { IsKernel = true; }

  inline bool isKernelInvoked() { return IsKernelInvoked; }
  inline void setKernelInvoked() { IsKernelInvoked = true; }

  inline std::string
  getExtraParameters(const std::string &Path,
                     FormatInfo FormatInformation = FormatInfo()) {
    buildInfo();
    VarMap.requestFeatureForAllVarMaps(Path);
    return VarMap.getExtraDeclParam(
        NonDefaultParamNum, ParamsNum - NonDefaultParamNum, FormatInformation);
  }
  std::string
  getExtraParameters(const std::string &Path,
                     const std::vector<TemplateArgumentInfo> &TAList,
                     FormatInfo FormatInformation = FormatInfo()) {
    MemVarMap TmpVarMap;
    buildInfo();
    TmpVarMap.merge(VarMap, TAList);
    TmpVarMap.requestFeatureForAllVarMaps(Path);
    return TmpVarMap.getExtraDeclParam(
        NonDefaultParamNum, ParamsNum - NonDefaultParamNum, FormatInformation);
  }

  void setDefinitionFilePath(const std::string &Path) {
    DefinitionFilePath = Path;
  }
  const std::string &getDefinitionFilePath() { return DefinitionFilePath; }
  void setNeedSyclExternMacro() { NeedSyclExternMacro = true; }
  bool IsSyclExternMacroNeeded() { return NeedSyclExternMacro; }
  void setAlwaysInlineDevFunc() { AlwaysInlineDevFunc = true; }
  bool IsAlwaysInlineDevFunc() { return AlwaysInlineDevFunc; }
  void setForceInlineDevFunc() { ForceInlineDevFunc = true; }
  bool IsForceInlineDevFunc() { return ForceInlineDevFunc; }
  void merge(std::shared_ptr<DeviceFunctionInfo> Other);
  size_t ParamsNum;
  size_t NonDefaultParamNum;
  GlobalMap<CallFunctionExpr> &getCallExprMap() { return CallExprMap; }
  void addSubGroupSizeRequest(unsigned int Size, SourceLocation Loc,
                              std::string APIName, std::string VarName = "") {
    if (Size == 0 || Loc.isInvalid())
      return;
    auto LocInfo = DpctGlobalInfo::getLocInfo(Loc);
    RequiredSubGroupSize.push_back(
        std::make_tuple(Size, LocInfo.first, LocInfo.second, APIName, VarName));
  }
  std::vector<std::tuple<unsigned int, std::string, unsigned int, std::string,
                         std::string>> &
  getSubGroupSize() {
    return RequiredSubGroupSize;
  }
  bool isParameterReferenced(unsigned int Index) {
    if (Index >= ParametersProps.size())
      return true;
    return ParametersProps[Index].IsReferenced;
  }
  void setParameterReferencedStatus(unsigned int Index, bool IsReferenced) {
    if (Index >= ParametersProps.size())
      return;
    ParametersProps[Index].IsReferenced =
        ParametersProps[Index].IsReferenced || IsReferenced;
  }
  std::string getFunctionName() { return FunctionName; }

private:
  void mergeCalledTexObj(
      std::shared_ptr<StructureTextureObjectInfo> BaseObj,
      const std::vector<std::shared_ptr<TextureObjectInfo>> &TexObjList);

  void mergeTextureObjectList(
      const std::vector<std::shared_ptr<TextureObjectInfo>> &Other);

  bool IsBuilt;
  std::string DefinitionFilePath;
  bool NeedSyclExternMacro = false;
  bool AlwaysInlineDevFunc = false;
  bool ForceInlineDevFunc = false;
  // subgroup size, filepath, offset, API name, var name
  std::vector<std::tuple<unsigned int, std::string, unsigned int, std::string,
                         std::string>>
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
  KernelCallExpr(unsigned Offset, const std::string &FilePath,
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

  static std::shared_ptr<KernelCallExpr>
  buildFromCudaLaunchKernel(const std::pair<std::string, unsigned> &LocInfo,
                            const CallExpr *);
  static std::shared_ptr<KernelCallExpr>
  buildForWrapper(std::string, const FunctionDecl *,
                  std::shared_ptr<DeviceFunctionInfo>);
  unsigned int GridDim = 3;
  unsigned int BlockDim = 3;
  void setEmitSizeofWarningFlag(bool Flag) { EmitSizeofWarning = Flag; }

private:
  KernelCallExpr(unsigned Offset, const std::string &FilePath)
      : CallFunctionExpr(Offset, FilePath, nullptr), IsSync(false) {}
  void buildArgsInfoFromArgsArray(const FunctionDecl *FD,
                                  const Expr *ArgsArray) {}
  void buildArgsInfo(const CallExpr *CE) {
    KernelArgumentAnalysis Analysis(IsInMacroDefine);
    auto KCallSpellingRange =
        getTheLastCompleteImmediateRange(CE->getBeginLoc(), CE->getEndLoc());
    Analysis.setCallSpelling(KCallSpellingRange.first, KCallSpellingRange.second);
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
        OuterStmts.emplace_back(
            buildString(MapNames::getDpctNamespace(), "global_memory<",
                        MapNames::getDpctNamespace(), "byte_t, 1> d_",
                        DpctGlobalInfo::getSyncName(), "(4);"));

        OuterStmts.emplace_back(buildString("d_", DpctGlobalInfo::getSyncName(),
                                            ".init(", DefaultQueue, ");"));

        SubmitStmtsList.SyncList.emplace_back(
            buildString("auto ", DpctGlobalInfo::getSyncName(), " = ",
                        MapNames::getDpctNamespace(), "get_access(d_",
                        DpctGlobalInfo::getSyncName(), ".get_ptr(), cgh);"));

        OuterStmts.emplace_back(buildString(
            MapNames::getDpctNamespace(), "dpct_memset(d_",
            DpctGlobalInfo::getSyncName(), ".get_ptr(), 0, sizeof(int));\n"));

        requestFeature(HelperFeatureEnum::device_ext);
      } else {
        OuterStmts.emplace_back(buildString(
            MapNames::getDpctNamespace(), "global_memory<unsigned int, 0> d_",
            DpctGlobalInfo::getSyncName(), "(0);"));
        OuterStmts.emplace_back(buildString(
            "unsigned *", DpctGlobalInfo::getSyncName(), " = d_",
            DpctGlobalInfo::getSyncName(), ".get_ptr(", DefaultQueue, ");"));

        OuterStmts.emplace_back(buildString(DefaultQueue, ".memset(",
                                            DpctGlobalInfo::getSyncName(),
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
    StmtList InitList;
    StmtList ExternList;
    StmtList PtrList;
    StmtList AccessorList;
    StmtList TextureList;
    StmtList SamplerList;
    StmtList NdRangeList;
    StmtList CommandGroupList;

    inline KernelPrinter &print(KernelPrinter &Printer) {
      printList(Printer, StreamList);
      printList(Printer, SyncList);
      printList(Printer, ExternList);
      printList(Printer, MemoryList);
      printList(Printer, InitList, "init global memory");
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
             AccessorList.empty() && PtrList.empty() && InitList.empty() &&
             ExternList.empty() && MemoryList.empty() && RangeList.empty() &&
             TextureList.empty() && SamplerList.empty() && StreamList.empty() &&
             SyncList.empty();
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

  StmtList OuterStmts;
  StmtList KernelStmts;
  std::string KernelArgs;
  int TotalArgsSize = 0;
  bool EmitSizeofWarning = false;
  unsigned int SizeOfHighestDimension = 0;
};

class CudaMallocInfo {
public:
  CudaMallocInfo(unsigned Offset, const std::string &FilePath,
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
  std::string Key = LocInfo.first + ":" + std::to_string(LocInfo.second);
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
  std::string Key = LocInfo.first + ":" + std::to_string(LocInfo.second);
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
  std::string KeyForDeclCounter =
      HFInfo.DeclLocFile + ":" + std::to_string(HFInfo.DeclLocOffset);

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
