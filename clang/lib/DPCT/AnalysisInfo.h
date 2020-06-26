//===--- AnalysisInfo.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_ANALYSIS_INFO_H
#define DPCT_ANALYSIS_INFO_H

#include "Debug.h"
#include "ExprAnalysis.h"
#include "ExtReplacements.h"
#include "Utility.h"
#include "ValidateArguments.h"
#include <bitset>
#include <unordered_set>

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"

void setTypeNamesMapPtr(const std::map<std::string, std::string> *Ptr);

namespace clang {
namespace dpct {

enum class HelperFuncType : int {
  InitValue = 0,
  DefaultQueue = 1,
  CurrentDevice = 2
};

enum class KernelArgType : int {
  Stream = 0,
  Texture,
  Accessor1D,
  Accessor2D,
  Accessor3D,
  Array1D,
  Array2D,
  Array3D,
  Default,
  MaxParameterSize
};

class CudaMallocInfo;
class RandomEngineInfo;
class TextureInfo;
class KernelCallExpr;
class DeviceFunctionInfo;
class CallFunctionExpr;
class DeviceFunctionDecl;
class MemVarInfo;
class VarInfo;

// Below four structs are all used for device RNG library API migration.
// In the origin code, the returned random number vector size is decided when
// the generate API is called.
// In MKL side, the vec_size is need when generator is declared.
// So we collect these info and then build replacement.

// This struct saves the generator's type.
struct DeviceRandomStateTypeInfo {
  DeviceRandomStateTypeInfo(unsigned int Length, std::string GeneratorType)
      : Length(Length), GeneratorType(GeneratorType) {}
  void buildInfo(std::string FilePath, unsigned int Offset);

  unsigned int Length;
  std::string GeneratorType;
};

// This struct saves all arguments info of the init API.
struct DeviceRandomInitAPIInfo {
  DeviceRandomInitAPIInfo(unsigned int Length, std::string GeneratorType,
                          std::string RNGSeed, std::string RNGSubseq,
                          std::string RNGOffset, std::string RNGStateName,
                          std::string IndentStr)
      : Length(Length), GeneratorType(GeneratorType), RNGSeed(RNGSeed),
        RNGSubseq(RNGSubseq), RNGOffset(RNGOffset), RNGStateName(RNGStateName),
        IndentStr(IndentStr) {}
  void buildInfo(std::string FilePath, unsigned int Offset);

  unsigned int Length;
  std::string GeneratorType;
  std::string RNGSeed;
  std::string RNGSubseq;
  std::string RNGOffset;
  std::string RNGStateName;
  std::string IndentStr;
};

// This struct saves all argument info of the generate API.
struct DeviceRandomGenerateAPIInfo {
  DeviceRandomGenerateAPIInfo(unsigned int Length, unsigned int DistrDeclOffset,
                              std::string DistrType, std::string ValueType,
                              std::string DistrIndentStr,
                              std::string RNGStateName, std::string IndentStr)
      : Length(Length), DistrDeclOffset(DistrDeclOffset), DistrType(DistrType),
        ValueType(ValueType), DistrIndentStr(DistrIndentStr),
        RNGStateName(RNGStateName), IndentStr(IndentStr) {}
  void buildInfo(std::string FilePath, unsigned int Offset);

  unsigned int Length;
  unsigned int DistrDeclOffset;
  std::string DistrType;
  std::string ValueType;
  std::string DistrIndentStr;
  std::string RNGStateName;
  std::string IndentStr;
  std::string DistrName;
};

// This struct saves the info for building the definition of distr variables.
struct DeviceRandomDistrInfo {
  DeviceRandomDistrInfo(std::string DistrType, std::string ValueType,
                        std::string DistrName, std::string IndentStr)
      : DistrType(DistrType), ValueType(ValueType), DistrName(DistrName),
        IndentStr(IndentStr) {}
  void buildInfo(std::string FilePath, unsigned int Offset);

  std::string DistrType;
  std::string ValueType;
  std::string DistrName;
  std::string IndentStr;
};

struct FormatInfo {
  FormatInfo() : EnableFormat(false), IsAllParamsOneLine(true) {}
  bool EnableFormat;
  bool IsAllParamsOneLine;
  bool IsEachParamNL;
  int CurrentLength;
  int NewLineIndentLength;
  std::string NewLineIndentStr;
};

class ParameterStream {
public:
  ParameterStream() { FormatInformation = FormatInfo(); }
  ParameterStream(FormatInfo FormatInformation, int ColumnLimit)
      : FormatInformation(FormatInformation), ColumnLimit(ColumnLimit) {}

  ParameterStream &operator<<(const std::string &InputParamStr) {
    if (InputParamStr.size()==0) {
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

    // parameters will be inserted in one line unless the line length > column limit.
    if (FormatInformation.CurrentLength + 2 + (int)InputParamStr.size() <=
        ColumnLimit) {
      Str = Str + ", " + InputParamStr;
      FormatInformation.CurrentLength = FormatInformation.CurrentLength + 2 +
                                        InputParamStr.size();
      return *this;
    } else {
      Str = Str + std::string(",") + getNL() +
                                   FormatInformation.NewLineIndentStr +
                                   InputParamStr;
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
  int ColumnLimit;
};

template <class T> using GlobalMap = std::map<unsigned, std::shared_ptr<T>>;
using MemVarInfoMap = GlobalMap<MemVarInfo>;

using ReplTy = std::map<std::string, tooling::Replacements>;

template <class T> inline void merge(T &Master, const T &Branch) {
  Master.insert(Branch.begin(), Branch.end());
}

template <class... Arguments>
inline void appendString(llvm::raw_string_ostream &OS, Arguments &&... Args) {
  std::initializer_list<int>{(OS << std::forward<Arguments>(Args), 0)...};
}

template <class... Arguments>
inline std::string buildString(Arguments &&... Args) {
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

template <class MapType, class... Args>
inline typename MapType::mapped_type
insertObject(MapType &Map, const typename MapType::key_type &Key,
             Args... InitArgs) {
  auto &Obj = Map[Key];
  if (!Obj)
    Obj = std::make_shared<typename MapType::mapped_type::element_type>(
        Key, InitArgs...);
  return Obj;
}

enum HeaderType {
  SYCL = 0,
  Math,
  Algorithm,
  Time,
  Complex,
  Future,
  MKL_BLAS_Solver,
  MKL_RNG,
  MKL_RNG_DEVICE,
  MKL_SPBLAS,
};

enum UsingType {
  QUEUE_P,
};

//                             DpctGlobalInfo
//                                         |
//              --------------------------------------
//              |                          |                           |
//    DpctFileInfo       DpctFileInfo     ...
//              |
//           -----------------------------------------------------
//           |                           |                         | |
//  MemVarInfo  DeviceFunctionDecl  KernelCallExpr  CudaMallocInfo
// Global Variable)                |   (inherite from CallFunctionExpr)
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
      : Repls(this), FilePath(FilePathIn) {
    buildLinesInfo();
  }
  template <class Obj> std::shared_ptr<Obj> findNode(unsigned Offset) {
    return findObject(getMap<Obj>(), Offset);
  }
  template <class Obj, class Node>
  std::shared_ptr<Obj> insertNode(unsigned Offset, const Node *N) {
    return insertObject(getMap<Obj>(), Offset, FilePath, N);
  }
  template <class Obj>
  std::shared_ptr<Obj> insertNode(unsigned Offset,
                                  std::shared_ptr<Obj> Object) {
    return getMap<Obj>().insert(std::make_pair(Offset, Object)).first->second;
  }
  inline const std::string &getFilePath() { return FilePath; }

  // Build kernel and device function declaration replacements and store them.
  void buildReplacements();

  // Emplace stored replacements into replacement set.
  void emplaceReplacements(ReplTy &ReplSet /*out*/);

  inline void addReplacement(std::shared_ptr<ExtReplacement> Repl) {
    insertHeader(SYCL);
    Repls.addReplacement(Repl);
  }

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

  void setMathHeaderInserted(bool B = true) {
    HeaderInsertedBitMap[HeaderType::Math] = B;
  }

  void setAlgorithmHeaderInserted(bool B = true) {
    HeaderInsertedBitMap[HeaderType::Algorithm] = B;
  }

 void setTimeHeaderInserted(bool B = true) {
    HeaderInsertedBitMap[HeaderType::Time] = B;
  }

  template <class... Args>
  void concatHeader(llvm::raw_string_ostream &OS, Args... Arguments) {
    std::initializer_list<int>{
        (appendString(OS, "#include ", std::move(std::forward<Args>(Arguments)),
                      getNL()),
         0)...};
  }

  // Insert one or more header inclusion directives at a specified offset
  void insertHeader(std::string &&Repl, unsigned Offset) {
    addReplacement(
        std::make_shared<ExtReplacement>(FilePath, Offset, 0, Repl, nullptr));
  }

  // Insert one or more header inclusion directives at first or last inclusion
  // locations
  template <typename... T>
  void insertHeader(HeaderType Type, unsigned Offset, T... Args);

  void insertHeader(HeaderType Type) {
    switch (Type) {
    case SYCL:
      return insertHeader(HeaderType::SYCL, FirstIncludeOffset, "<CL/sycl.hpp>",
                          "<dpct/dpct.hpp>");
    case Math:
      return insertHeader(HeaderType::Math, LastIncludeOffset, "<cmath>");
    case Algorithm:
      return insertHeader(HeaderType::Algorithm, LastIncludeOffset,
                          "<algorithm>");
    case Complex:
      return insertHeader(HeaderType::Complex, LastIncludeOffset, "<complex>");
    case Future:
      return insertHeader(HeaderType::Future, LastIncludeOffset, "<future>");
    case Time:
      return insertHeader(HeaderType::Time, LastIncludeOffset, "<time.h>");
    case MKL_BLAS_Solver:
      return insertHeader(HeaderType::MKL_BLAS_Solver, LastIncludeOffset,
                          "<mkl_blas_sycl.hpp>", "<mkl_lapack_sycl.hpp>",
                          "<mkl_sycl_types.hpp>", "<dpct/blas_utils.hpp>");
    case MKL_RNG:
      return insertHeader(HeaderType::MKL_RNG, LastIncludeOffset,
                          "<mkl_rng_sycl.hpp>");
    case MKL_RNG_DEVICE:
      return insertHeader(HeaderType::MKL_RNG, LastIncludeOffset,
                          "<mkl_rng_sycl_device.hpp>");
    case MKL_SPBLAS:
      return insertHeader(HeaderType::MKL_SPBLAS, LastIncludeOffset,
                          "<mkl_spblas_sycl.hpp>", "<dpct/blas_utils.hpp>");
    }
  }

  // Record line info in file.
  struct SourceLineInfo {
    SourceLineInfo() : SourceLineInfo(-1, -1, -1, nullptr) {}
    SourceLineInfo(unsigned LineNumber, unsigned Offset, unsigned End,
                   const char *Buffer)
        : Number(LineNumber), Offset(Offset), Length(End - Offset),
          Line(Buffer ? std::string(Buffer + Offset, Length) : "") {}
    SourceLineInfo(unsigned LineNumber, unsigned *LineCache, const char *Buffer)
        : SourceLineInfo(LineNumber, LineCache[LineNumber - 1],
                         LineCache[LineNumber], Buffer) {}

    // Line number.
    const unsigned Number;
    // Offset at the begin of line.
    const unsigned Offset;
    // Length of the line.
    const unsigned Length;
    // String of the line, only available when -keep-original-code is on.
    const std::string Line;
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
  inline const std::string &getLineString(unsigned LineNumber) {
    return getLineInfo(LineNumber).Line;
  }

  // Get line number by offset
  inline unsigned getLineNumber(unsigned Offset) {
    return getLineInfoFromOffset(Offset).Number;
  }
  // Set line range info of replacement
  void setLineRange(ExtReplacements::SourceLineRange &LineRange,
                    std::shared_ptr<ExtReplacement> Repl) {
    unsigned Begin = Repl->getOffset(), End = Begin + Repl->getLength();
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

  // The key of below three maps are the offset of the replacement.
  std::map<unsigned int, DeviceRandomStateTypeInfo> &
  getDeviceRandomStateTypeMap() {
    return DeviceRandomStateTypeMap;
  }
  std::map<unsigned int, DeviceRandomInitAPIInfo> &getDeviceRandomInitAPIMap() {
    return DeviceRandomInitAPIMap;
  }
  std::map<unsigned int, DeviceRandomGenerateAPIInfo> &
  getDeviceRandomGenerateAPIMap() {
    return DeviceRandomGenerateAPIMap;
  }
  // Since multi generate API can share one distr variable, so this function
  // merges different distr variables if possible.
  void buildDeviceDistrDeclInfo() {
    std::unordered_map<std::string, std::string> NameMap;
    std::string Key;
    std::string Name;
    int ID = 1;
    for (auto &Info : DeviceRandomGenerateAPIMap) {
      Key = std::to_string(Info.second.DistrDeclOffset) + ":" +
            Info.second.DistrType + ":" + Info.second.ValueType;
      auto Iter = NameMap.find(Key);
      if (Iter == NameMap.end()) {
        Name = "distr_ct" + std::to_string(ID);
        NameMap.insert(std::make_pair(Key, Name));
        Info.second.DistrName = Name;
        DeviceRandomDistrDeclMap.insert(std::make_pair(
            Info.second.DistrDeclOffset,
            DeviceRandomDistrInfo(Info.second.DistrType, Info.second.ValueType,
                                  Name, Info.second.DistrIndentStr)));
        ID++;
      } else {
        Info.second.DistrName = Iter->second;
      }
    }
  }

  std::unordered_set<std::shared_ptr<DpctFileInfo>> &getIncludedFilesInfoSet() {
    return IncludedFilesInfoSet;
  }
  std::set<unsigned int> &getSpBLASSet() {
  return SpBLASSet;
}

private:
  std::unordered_set<std::shared_ptr<DpctFileInfo>> IncludedFilesInfoSet;

  template <class Obj> GlobalMap<Obj> &getMap() {
    llvm::dbgs() << "[DpctFileInfo::getMap] Unknow map type";
    static GlobalMap<Obj> NullMap;
    return NullMap;
  }

  bool isInRoot();
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

  std::map<unsigned int, DeviceRandomStateTypeInfo> DeviceRandomStateTypeMap;
  std::map<unsigned int, DeviceRandomInitAPIInfo> DeviceRandomInitAPIMap;
  std::map<unsigned int, DeviceRandomGenerateAPIInfo>
      DeviceRandomGenerateAPIMap;
  std::map<unsigned int, DeviceRandomDistrInfo> DeviceRandomDistrDeclMap;
  GlobalMap<MemVarInfo> MemVarMap;
  GlobalMap<DeviceFunctionDecl> FuncMap;
  GlobalMap<KernelCallExpr> KernelMap;
  GlobalMap<CudaMallocInfo> CudaMallocMap;
  GlobalMap<RandomEngineInfo> RandomEngineMap;
  GlobalMap<TextureInfo> TextureMap;
  std::set<unsigned int> SpBLASSet;

  ExtReplacements Repls;
  std::vector<SourceLineInfo> Lines;

  std::string FilePath;

  unsigned FirstIncludeOffset = 0;
  unsigned LastIncludeOffset = 0;
  bool HasInclusionDirective = false;

  std::bitset<32> HeaderInsertedBitMap;
  std::bitset<32> UsingInsertedBitMap;
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
template <> inline GlobalMap<RandomEngineInfo> &DpctFileInfo::getMap() {
  return RandomEngineMap;
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
    SourceLocation NameTokenLoc;
    bool IsInRoot;
    MacroDefRecord(SourceLocation NTL, bool IIR)
        : NameTokenLoc(NTL), IsInRoot(IIR) {}
  };

  class MacroExpansionRecord {
  public:
    std::string Name;
    int NumTokens;
    SourceLocation ReplaceTokenBegin;
    SourceLocation ReplaceTokenEnd;
    SourceRange Range;
    bool IsInRoot;
    bool IsFunctionLike;
    int TokenIndex;
    MacroExpansionRecord(IdentifierInfo *ID, const MacroInfo *MI,
                         SourceRange Range, bool IsInRoot, int TokenIndex) {
      Name = ID->getName().str();
      NumTokens = MI->getNumTokens();
      ReplaceTokenBegin = MI->getReplacementToken(0).getLocation();
      ReplaceTokenEnd =
          MI->getReplacementToken(MI->getNumTokens() - 1).getLocation();
      this->Range = Range;
      this->IsInRoot = IsInRoot;
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
          CurrentDeviceCounter(CurrentDeviceCounter) {}
    int DefaultQueueCounter = 0;
    int CurrentDeviceCounter = 0;
  };

  static std::string removeSymlinks(clang::FileManager &FM,
                                    std::string FilePathStr) {
    // Get rid of symlinks
    SmallString<4096> NoSymlinks = StringRef("");
    auto Dir = FM.getDirectory(
        llvm::sys::path::parent_path(FilePathStr));
    if (Dir) {
      StringRef DirName = FM.getCanonicalName(*Dir);
      StringRef FileName = llvm::sys::path::filename(FilePathStr);
      llvm::sys::path::append(NoSymlinks, DirName, FileName);
    }
    return NoSymlinks.str().str();
  }

  inline static bool isInRoot(SourceLocation SL) {
    return isInRoot(getSourceManager()
                        .getFilename(getSourceManager().getExpansionLoc(SL))
                        .str());
  }
  static bool isInRoot(const std::string &FilePath, bool IsFilePathAbs = true) {
    if (IsFilePathAbs) {
      std::string Path = removeSymlinks(getFileManager(), FilePath);
      makeCanonical(Path);
      return isChildPath(InRoot, Path);
    } else {
      return isChildPath(InRoot, FilePath, IsFilePathAbs);
    }
  }
  inline static bool replaceMacroName(SourceLocation SL) {
    auto &SM = getSourceManager();
    std::string Path = SM.getFilename(SM.getExpansionLoc(SL)).str();
    if (isInCudaPath(Path)) {
      return true;
    }
    makeCanonical(Path);
    StringRef Filename = llvm::sys::path::filename(Path);
    // The above condition is not always sufficient for the following
    // specific header files
    if (Filename == "cublas_api.h" || Filename == "cublas.h" ||
        Filename == "cublasLt.h" || Filename == "cublas_v2.h" ||
        Filename == "cublasXt.h" || Filename == "nvblas.h") {
      return true;
    }
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
  // TODO: implement one of this for each source language.
  static void setCudaPath(const std::string &InputCudaPath) {
    CudaPath = InputCudaPath;
  }
  // TODO: implement one of this for each source language.
  static const std::string &getCudaPath() {
    assert(!CudaPath.empty());
    return CudaPath;
  }
  static const std::string &getItemName() {
    const static std::string ItemName = "item" + getCTFixedSuffix();
    return ItemName;
  }
  static const std::string &getStreamName() {
    const static std::string StreamName = "stream" + getCTFixedSuffix();
    return StreamName;
  }
  static const std::string &getInRootHash() {
    const static std::string Hash = getHashAsString(getInRoot()).substr(0, 6);
    return Hash;
  }
  static void setCompilerInstance(CompilerInstance &C) {
    CI = &C;
    setContext(C.getASTContext());
  }
  static void setContext(ASTContext &C) {
    Context = &C;
    SM = &(Context->getSourceManager());
    FM = &(SM->getFileManager());
    Context->getParentMapContext().setTraversalKind(TK_AsIs);
  }
  static CompilerInstance &getCompilerInstance() {
    assert(CI);
    return *CI;
  }
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
  inline static bool getGuessIndentWidthMatcherFlag() {
    return GuessIndentWidthMatcherFlag;
  }
  inline static void setGuessIndentWidthMatcherFlag(bool Flag = true) {
    GuessIndentWidthMatcherFlag = Flag;
  }
  inline static void setIndentWidth(unsigned int W) { IndentWidth = W; }
  inline static unsigned int getIndentWidth() { return IndentWidth; }
  inline static UsmLevel getUsmLevel() { return UsmLvl; }
  inline static void setUsmLevel(UsmLevel UL) { UsmLvl = UL; }
  inline static format::FormatRange getFormatRange() { return FmtRng; }
  inline static void setFormatRange(format::FormatRange FR) { FmtRng = FR; }
  inline static DPCTFormatStyle getFormatStyle() { return FmtST; }
  inline static void setFormatStyle(DPCTFormatStyle FS) { FmtST = FS; }
  inline static bool isCtadEnabled() { return EnableCtad; }
  inline static void setCtadEnabled(bool Enable = true) { EnableCtad = Enable; }
  inline static bool isCommentsEnabled() { return EnableComments; }
  inline static void setCommentsEnabled(bool Enable = true) {
    EnableComments = Enable;
  }

  // This set collects all the different vector size of the return value of the
  // generate API. If the size of this set is 1, than we can use this vec_size
  // in all generator types. Otherwise, a placeholder will be inserted.
  inline static std::unordered_set<int> &getDeviceRNGReturnNumSet() {
    return DeviceRNGReturnNumSet;
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
    CurrentMaxIndex = Res;
    CurrentIndexInRule++;
    return Res;
  }

  inline static void setCodeFormatStyle(clang::format::FormatStyle Style) {
    CodeFormatStyle = Style;
  }
  inline static clang::format::FormatStyle getCodeFormatStyle() {
    return CodeFormatStyle;
  }

  template <class TargetTy, class NodeTy>
  static inline const TargetTy *
  findAncestor(const NodeTy *N,
               const std::function<bool(const ast_type_traits::DynTypedNode &)>
                   &Condition) {
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
        Node, [&](const ast_type_traits::DynTypedNode &Cur) -> bool {
          return Cur.get<BinaryOperator>() == BO;
        });
  }

  template <class TargetTy, class NodeTy>
  static const TargetTy *findAncestor(const NodeTy *Node) {
    return findAncestor<TargetTy>(
        Node, [&](const ast_type_traits::DynTypedNode &Cur) -> bool {
          return Cur.get<TargetTy>();
        });
  }
  template <class NodeTy>
  inline static const clang::FunctionDecl *
  getParentFunction(const NodeTy *Node) {
    return findAncestor<clang::FunctionDecl>(Node);
  }
  template <class StreamTy, class... Args>
  static inline StreamTy &
  printCtadClass(StreamTy &Stream, size_t CanNotDeducedArgsNum,
                 StringRef ClassName, Args &&... Arguments) {
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
                                         Args &&... Arguments) {
    return printCtadClass(Stream, 0, ClassName,
                          std::forward<Args>(Arguments)...);
  }
  template <class... Args>
  static inline std::string getCtadClass(Args &&... Arguments) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    return printCtadClass(OS, std::forward<Args>(Arguments)...).str();
  }
  template <class T>
  static inline std::pair<std::string, unsigned>
  getLocInfo(const T *N, bool *IsInvalid = nullptr /* out */) {
    return getLocInfo(getLocation(N), IsInvalid);
  }

  static inline std::pair<std::string, unsigned>
  getLocInfo(SourceLocation Loc, bool *IsInvalid = nullptr /* out */) {
    auto LocInfo =
        SM->getDecomposedLoc(getSourceManager().getExpansionLoc(Loc));
    if (auto FileEntry = SM->getFileEntryForID(LocInfo.first)) {
      // To avoid potential path inconsist issue,
      // using tryGetRealPathName while applicable.
      std::string FileName;
      if (!FileEntry->tryGetRealPathName().empty()) {
        FileName = FileEntry->tryGetRealPathName().str();
      }
      else {
        llvm::SmallString<512> FilePathAbs(FileEntry->getName());
        getSourceManager().getFileManager().makeAbsolutePath(FilePathAbs);
        llvm::sys::path::native(FilePathAbs);
        // Need to remove dot to keep the file path
        // added by ASTMatcher and added by
        // AnalysisInfo::getLocInfo() consistent.
        llvm::sys::path::remove_dots(FilePathAbs, true);
        FileName = std::string(FilePathAbs.str());
      }
      return std::make_pair(FileName, LocInfo.second);
    }
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
    return QT.getAsString(Context.getPrintingPolicy());
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

  static inline void
  getReplacedTypeNameRecursive(QualType QT, const ASTContext &Context,
                               std::string &MigratedTypeStr) {
    if (!QT.hasQualifiers() && !QT.getTypePtr()->isPointerType()) {
      std::string TypeName = QT.getAsString(Context.getPrintingPolicy());
      MapNames::replaceName(MapNames::TypeNamesMap, TypeName);
      MigratedTypeStr = buildString(TypeName, MigratedTypeStr);
      return;
    }
    if (!QT.getQualifiers().isEmptyWhenPrinted(Context.getPrintingPolicy())) {
      MigratedTypeStr = buildString(
          " ", QT.getQualifiers().getAsString(Context.getPrintingPolicy()),
          MigratedTypeStr);
    }
    QualType RemovedQ = QT.getUnqualifiedType();
    QualType RemovedP = RemovedQ;
    if (RemovedQ.getTypePtr()->isPointerType()) {
      MigratedTypeStr = buildString(" *", MigratedTypeStr);
      RemovedP = RemovedQ.getTypePtr()->getPointeeType();
    }
    getReplacedTypeNameRecursive(RemovedP, Context, MigratedTypeStr);
    return;
  }

  /// This function will return the replaced type name with qualifiers.
  /// Currently, since clang do not support get the order of original
  /// qualifiers, this function will follow the behavior of
  /// clang::QualType.print(), in other words, the behavior is that the
  /// qualifiers(const, volatile...) will occur before the simple type(int,
  /// bool...) regardless its order in origin code. \param [in] QT The input
  /// qualified type which need migration. \param [in] Context The AST context.
  /// \return The replaced type name string with qualifiers.
  static inline std::string getReplacedTypeName(QualType QT,
                                                const ASTContext &Context) {
    std::string MigratedTypeStr;
    setTypeNamesMapPtr(&MapNames::TypeNamesMap);
    llvm::raw_string_ostream OS(MigratedTypeStr);
    clang::PrintingPolicy PP =
        clang::PrintingPolicy(DpctGlobalInfo::getContext().getLangOpts());
    QT.print(OS, PP);
    OS.flush();
    setTypeNamesMapPtr(nullptr);
    return MigratedTypeStr;
  }
  static inline std::string getReplacedTypeName(QualType QT) {
    return getReplacedTypeName(QT, DpctGlobalInfo::getContext());
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
  GLOBAL_TYPE(RandomEngineInfo, DeclaratorDecl)
  GLOBAL_TYPE(TextureInfo, VarDecl)
#undef GLOBAL_TYPE

  // Build kernel and device function declaration replacements and store
  // them.
  void buildReplacements() {
    for (auto &File : FileMap)
      File.second->buildReplacements();
  }

  // Emplace stored replacements into replacement set.
  void emplaceReplacements(ReplTy &ReplSets /*out*/) {
    for (auto &File : FileMap)
      File.second->emplaceReplacements(ReplSets);
  }

  std::shared_ptr<KernelCallExpr> buildLaunchKernelInfo(const CallExpr *);

  void insertCudaMalloc(const CallExpr *CE);
  void insertCublasAlloc(const CallExpr *CE);
  std::shared_ptr<CudaMallocInfo> findCudaMalloc(const Expr *CE);
  void addReplacement(std::shared_ptr<ExtReplacement> Repl) {
    insertFile(Repl->getFilePath().str())->addReplacement(Repl);
  }
  void insertDeviceRandomStateTypeInfo(SourceLocation SL, unsigned int Length,
                                       std::string GeneratorType) {
    auto LocInfo = getLocInfo(SL);
    auto FileInfo = insertFile(LocInfo.first);
    auto &M = FileInfo->getDeviceRandomStateTypeMap();
    if (M.find(LocInfo.second) == M.end()) {
      M.insert(std::make_pair(
          LocInfo.second, DeviceRandomStateTypeInfo(Length, GeneratorType)));
    }
  }
  void insertDeviceRandomInitAPIInfo(SourceLocation SL, unsigned int Length,
                                     std::string GeneratorType,
                                     std::string RNGSeed, std::string RNGSubseq,
                                     std::string RNGOffset,
                                     std::string StateName,
                                     std::string IndentStr) {
    auto LocInfo = getLocInfo(SL);
    auto FileInfo = insertFile(LocInfo.first);
    auto &M = FileInfo->getDeviceRandomInitAPIMap();
    if (M.find(LocInfo.second) == M.end()) {
      M.insert(std::make_pair(
          LocInfo.second,
          DeviceRandomInitAPIInfo(Length, GeneratorType, RNGSeed, RNGSubseq,
                                  RNGOffset, StateName, IndentStr)));
    }
  }
  void insertDeviceRandomGenerateAPIInfo(
      SourceLocation SL, unsigned int Length, SourceLocation DistrInsetLoc,
      std::string DistrType, std::string ValueType, std::string DistrIndentStr,
      std::string RNGStateName, std::string IndentStr) {
    auto LocInfo = getLocInfo(SL);
    auto DistrInsetLocInfo = getLocInfo(DistrInsetLoc);
    auto FileInfo = insertFile(LocInfo.first);
    auto &M = FileInfo->getDeviceRandomGenerateAPIMap();
    if (M.find(LocInfo.second) == M.end()) {
      M.insert(std::make_pair(
          LocInfo.second,
          DeviceRandomGenerateAPIInfo(Length, DistrInsetLocInfo.second,
                                      DistrType, ValueType, DistrIndentStr,
                                      RNGStateName, IndentStr)));
    }
  }

  void insertRandomEngine(const Expr *E);
  std::shared_ptr<RandomEngineInfo> findRandomEngine(const Expr *E);

  void insertSpBLASWarningLocOffset(SourceLocation SL) {
    auto LocInfo = getLocInfo(SL);
    auto FileInfo = insertFile(LocInfo.first);
    FileInfo->getSpBLASSet().insert(LocInfo.second);
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

  static std::map<const char *, std::shared_ptr<MacroExpansionRecord>> &
  getExpansionRangeToMacroRecord() {
    return ExpansionRangeToMacroRecord;
  }

  static std::map<const char *, std::shared_ptr<DpctGlobalInfo::MacroDefRecord>>
      &getMacroTokenToMacroDefineLoc() {
    return MacroTokenToMacroDefineLoc;
  }

  static std::map<std::string, SourceLocation> &getBeginOfEmptyMacros() {
    return BeginOfEmptyMacros;
  }
  static std::map<std::string, SourceLocation> &getEndOfEmptyMacros() {
    return EndOfEmptyMacros;
  }
  static std::map<MacroInfo *, bool> &getMacroDefines() { return MacroDefines; }
  static std::set<std::string> &getIncludingFileSet() { return IncludingFileSet; }
  static std::set<std::string> &getFileSetInCompiationDB() { return FileSetInCompiationDB; }
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
      getTempVariableDeclCounterMap(){
    return TempVariableDeclCounterMap;
  }
  static std::unordered_set<std::string> &getTempVariableHandledSet() {
    return TempVariableHandledSet;
  }
  static bool getUsingDRYPattern() { return UsingDRYPattern; }
  static void setUsingDRYPattern(bool Flag) { UsingDRYPattern = Flag; }

  static bool getSpBLASUnsupportedMatrixTypeFlag() {
    return SpBLASUnsupportedMatrixTypeFlag;
  }
  static void setSpBLASUnsupportedMatrixTypeFlag(bool Flag) {
    SpBLASUnsupportedMatrixTypeFlag = Flag;
  }

  inline std::shared_ptr<DpctFileInfo> insertFile(const std::string &FilePath) {
    return insertObject(FileMap, FilePath);
  }

  inline void recordIncludingRelationship(const std::string &CurrentFileName,
                                          const std::string &IncludedFileName) {
    auto CurrentFileInfo = this->insertFile(CurrentFileName);
    auto IncludedFileInfo = this->insertFile(IncludedFileName);
    CurrentFileInfo->insertIncludedFilesInfo(IncludedFileInfo);
  }

private:
  DpctGlobalInfo();

  DpctGlobalInfo(const DpctGlobalInfo &) = delete;
  DpctGlobalInfo(DpctGlobalInfo &&) = delete;
  DpctGlobalInfo &operator=(const DpctGlobalInfo &) = delete;
  DpctGlobalInfo &operator=(DpctGlobalInfo &&) = delete;

  // Wrapper of isInRoot for std::function usage.
  static bool checkInRoot(SourceLocation SL) { return isInRoot(SL); }

  // Find stored info by its corresponding AST node.
  // VarDecl=>MemVarInfo
  // FunctionDecl=>DeviceFunctionDecl
  // CUDAKernelCallExpr=>KernelCallExpr
  // VarDecl=>CudaMallocInfo
  // DeclaratorDecl=>RandomEngineInfo
  template <class Info, class Node>
  inline std::shared_ptr<Info> findNode(const Node *N) {
    if (!N)
      return std::shared_ptr<Info>();
    auto LocInfo = getLocInfo(N);
    if (isInRoot(LocInfo.first))
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
    return SM->getExpansionLoc(FD->getBeginLoc());
  }
  static inline SourceLocation getLocation(const FieldDecl *FD) {
    return FD->getLocation();
  }
  static inline SourceLocation getLocation(const CallExpr *CE) {
    return CE->getEndLoc();
  }
  // The result will be also stored in KernelCallExpr.BeginLoc
  static inline SourceLocation getLocation(const CUDAKernelCallExpr *CKC) {
    // if the BeginLoc of CKC is in macro define, use getImmediateSpellingLoc.
    auto It = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
        SM->getCharacterData(SM->getSpellingLoc(CKC->getBeginLoc())));
    if (CKC->getBeginLoc().isMacroID() &&
        It != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
      return SM->getImmediateSpellingLoc(CKC->getBeginLoc());
    }
    // if the BeginLoc of CKC is in macro arg expansion, use
    // getImmediateExpansionRange.
    if (CKC->getBeginLoc().isMacroID() &&
        SM->isMacroArgExpansion(CKC->getBeginLoc())) {
      return SM->getImmediateSpellingLoc(
          SM->getImmediateExpansionRange(CKC->getBeginLoc()).getBegin());
    }
    return CKC->getBeginLoc();
  }

  std::unordered_map<std::string, std::shared_ptr<DpctFileInfo>> FileMap;

  static std::string InRoot;
  // TODO: implement one of this for each source language.
  static std::string CudaPath;
  static UsmLevel UsmLvl;
  static std::unordered_set<int> DeviceRNGReturnNumSet;
  static format::FormatRange FmtRng;
  static DPCTFormatStyle FmtST;
  static bool EnableCtad;
  static bool EnableComments;
  static std::string ClNamespace;
  static CompilerInstance *CI;
  static ASTContext *Context;
  static SourceManager *SM;
  static FileManager   *FM;
  static bool KeepOriginCode;
  static bool SyclNamedLambda;
  static bool GuessIndentWidthMatcherFlag;
  static unsigned int IndentWidth;
  static std::unordered_map<std::string, int> LocationInitIndexMap;
  static std::map<const char *,
                  std::shared_ptr<DpctGlobalInfo::MacroExpansionRecord>>
      ExpansionRangeToMacroRecord;
  static std::map<const char *, std::shared_ptr<DpctGlobalInfo::MacroDefRecord>>
      MacroTokenToMacroDefineLoc;
  // key: The hash string of the first non-empty token after the end location of
  // macro expansion
  // value: begin location of macro expansion
  static std::map<std::string, SourceLocation> EndOfEmptyMacros;
  // key: The hash string of the begin location of the macro expansion
  // value: The end location of the macro expansion
  static std::map<std::string, SourceLocation> BeginOfEmptyMacros;
  static std::map<MacroInfo *, bool> MacroDefines;
  static int CurrentMaxIndex;
  static int CurrentIndexInRule;
  static std::set<std::string> IncludingFileSet;
  static std::set<std::string> FileSetInCompiationDB;
  static clang::format::FormatStyle CodeFormatStyle;
  static bool HasFoundDeviceChanged;
  static std::unordered_map<int, HelperFuncReplInfo> HelperFuncReplInfoMap;
  static int HelperFuncReplInfoIndex;
  static std::unordered_map<std::string, TempVariableDeclCounter>
      TempVariableDeclCounterMap;
  static std::unordered_set<std::string> TempVariableHandledSet;
  static bool UsingDRYPattern;
  static bool SpBLASUnsupportedMatrixTypeFlag;
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
  // If NeedSizeFold is true, array size will be folded, but orginal expression
  // will follow as comments. If NeedSizeFold is false, original size expression
  // will be the size string.
  CtTypeInfo(const TypeLoc &TL, bool NeedSizeFold = false,
             bool IsShared = false);

  inline const std::string &getBaseName() { return BaseName; }

  inline size_t getDimension() { return Range.size(); }

  const std::string &getTemplateSpecializationName() {
    if (isTemplate() && TDSI)
      return TDSI->getSourceString();
    return getBaseName();
  }

  // when there is no arguments, parameter MustArguments determine whether
  // parens will exist. Null string will be returned when MustArguments is
  // false, otherwise "()" will be returned.
  std::string getRangeArgument(const std::string &MemSize, bool MustArguments);

  inline bool isTemplate() const { return IsTemplate; }
  inline bool isPointer() const { return IsPointer; }
  inline bool isReference() const { return IsReference; }
  inline void adjustAsMemType() {
    setPointerAsArray();
    removeQualifier();
  }

  /// Get instantiated type name with given template arguments.
  /// e.g. X<T>, with T = int, result type will be X<int>.
  std::shared_ptr<CtTypeInfo>
  applyTemplateArguments(const std::vector<TemplateArgumentInfo> &TA);

  bool isWritten() const {
    return !TDSI || !isTemplate() || TDSI->isDependOnWritten();
  }

private:
  CtTypeInfo() : IsPointer(false), IsTemplate(false) {}

  /// For ConstantArrayType, size in generated code is folded as an integer.
  /// If \p NeedSizeFold is true, original size expression will followed as
  /// comments.
  void setTypeInfo(const TypeLoc &TL, bool NeedSizeFold = false);

  /// Get folded array size with original size expression following as comments.
  /// e.g.,
  /// #define SIZE 24
  /// dpct::device_memory<int, 1>(24 /* SIZE */);
  /// Exception for particular case:
  /// __device__ int a[24];
  /// will be migrated to:
  /// dpct::device_memory<int, 1> a(24);
  inline std::string getFoldedArraySize(const ConstantArrayTypeLoc &TL) {
    if (TL.getSizeExpr()->getStmtClass() == Stmt::IntegerLiteralClass &&
        TL.getSizeExpr()->getBeginLoc().isFileID())
      return TL.getTypePtr()->getSize().toString(10, false);
    return buildString(TL.getTypePtr()->getSize().toString(10, false), "/*",
                       getStmtSpelling(TL.getSizeExpr()), "*/");
  }

  // Get original array size expression.
  std::string getUnfoldedArraySize(const ConstantArrayTypeLoc &TL);

  /// Typically C++ array with constant size.
  /// e.g.: __device__ int a[20];
  /// If \p NeedSizeFold is true, original size expression will followed as
  /// comments.
  /// e.g.,
  /// #define SIZE 24
  /// dpct::device_memory<int, 1>(24 /* SIZE */);
  void setArrayInfo(const ConstantArrayTypeLoc &TL, bool NeedFoldSize);

  /// Typically C++ array with template depedent size.
  /// e.g.: template<size_t S>
  /// ...
  /// __device__ int a[S];
  void setArrayInfo(const DependentSizedArrayTypeLoc &TL, bool NeedSizeFold);

  /// IncompleteArray is an array defined without size.
  /// e.g.: extern __shared__ int a[];
  void setArrayInfo(const IncompleteArrayTypeLoc &TL, bool NeedSizeFold);
  void setName(const TypeLoc &TL);

  void setPointerAsArray() {
    if (isPointer()) {
      IsPointer = false;
      Range.emplace_back();
    }
  }
  inline void removeQualifier() { BaseName = BaseNameWithoutQualifiers; }

private:
  std::string BaseName;
  std::string BaseNameWithoutQualifiers;
  std::vector<SizeInfo> Range;
  bool IsPointer;
  bool IsReference;
  bool IsTemplate;
  bool IsShared = false;

  std::shared_ptr<TemplateDependentStringInfo> TDSI;
};

// variable info includes name, type and location.
class VarInfo {
public:
  VarInfo(unsigned Offset, const std::string &FilePathIn, const VarDecl *Var)
      : VarInfo(FilePathIn, Offset, Var) {}
  VarInfo(unsigned Offset, const std::string &FilePathIn,
          const FieldDecl *FieldVar, std::string MemberExprString)
      : VarInfo(FilePathIn, Offset, FieldVar, MemberExprString) {}

  inline const std::string &getFilePath() { return FilePath; }
  inline unsigned getOffset() { return Offset; }
  inline const std::string &getRefString() { return RefString; }
  inline const std::string &getName() { return Name; }
  inline std::shared_ptr<CtTypeInfo> &getType() { return Ty; }

  inline std::string getDerefName() {
    return buildString(getName(), "_deref_", DpctGlobalInfo::getInRootHash());
  }

  inline void
  applyTemplateArguments(const std::vector<TemplateArgumentInfo> &TAList) {
    Ty = Ty->applyTemplateArguments(TAList);
  }

protected:
  VarInfo(const std::string &FilePath, unsigned Offset,
          const DeclaratorDecl *DD, const std::string &RefStringIn = "")
      : FilePath(FilePath), Offset(Offset), Name(DD->getName()),
        RefString(RefStringIn.empty() ? Name : RefStringIn),
        Ty(std::make_shared<CtTypeInfo>(
            DD->getTypeSourceInfo()->getTypeLoc())) {}
  inline void setType(std::shared_ptr<CtTypeInfo> T) { Ty = T; }

private:
  const std::string FilePath;
  unsigned Offset;
  std::string Name;
  std::string RefString;
  std::shared_ptr<CtTypeInfo> Ty;
};

// memory variable info includs basic variable info and memory attributes.
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

  MemVarInfo(unsigned Offset, const std::string &FilePath, const VarDecl *Var)
      : VarInfo(Offset, FilePath, Var), Attr(getAddressAttr(Var)),
        Scope((Var->isInLocalScope())
                  ? (Var->getStorageClass() == SC_Extern ? Extern : Local)
                  : Global),
        PointerAsArray(false) {
    setType(std::make_shared<CtTypeInfo>(Var->getTypeSourceInfo()->getTypeLoc(),
                                         isLocal(), isShared()));
    if (getType()->isPointer()) {
      Attr = Device;
      getType()->adjustAsMemType();
      PointerAsArray = true;
    }
    if (Var->hasInit())
      setInitList(Var->getInit());
    if (getType()->getDimension() == 0 && Attr == Constant) {
      AccMode = Value;
    } else if (getType()->getDimension() <= 1) {
      AccMode = Pointer;
    } else {
      AccMode = Accessor;
    }
  }

  VarAttrKind getAttr() { return Attr; }
  VarScope getScope() { return Scope; }
  bool isGlobal() { return Scope == Global; }
  bool isExtern() { return Scope == Extern; }
  bool isLocal() { return Scope == Local; }
  bool isShared() { return Attr == Shared; }

  std::string getDeclarationReplacement();
  inline std::string getMemoryDecl(const std::string &MemSize) {
    return buildString(getMemoryType(), " ", getArgName(),
                       PointerAsArray ? "" : getInitArguments(MemSize), ";");
  }
  std::string getMemoryDecl() {
    const static std::string NullString;
    return getMemoryDecl(NullString);
  }

  std::string getExternGlobalVarDecl() {
    return buildString("extern ", getMemoryType(), " ", getArgName(), ";");
  }

  void appendAccessorOrPointerDecl(const std::string &ExternMemSize,
                                   std::vector<std::string> &AccList,
                                   std::vector<std::string> &PtrList) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    if (isShared()) {
      auto Dimension = getType()->getDimension();
      OS << MapNames::getClNamespace() + "::accessor<"
         << getAccessorDataType(true) << ", " << Dimension
         << ", " + MapNames::getClNamespace() + "::access::mode::read_write, " +
                MapNames::getClNamespace() + "::access::target::local> "
         << getAccessorName() << "(";
      if (Dimension > 1) {
        OS << getRangeName() << ", ";
      } else if (Dimension == 1) {
        OS << getRangeClass()
           << getType()->getRangeArgument(ExternMemSize, false) << ", ";
      }
      OS << "cgh);";
      AccList.push_back(std::move(OS.str()));
    } else if (DpctGlobalInfo::getUsmLevel() == UsmLevel::restricted &&
               AccMode != Accessor) {
      PtrList.push_back(buildString("auto ", getPtrName(), " = ", getArgName(),
                                    ".get_ptr();"));
    } else {
      AccList.push_back(buildString("auto ", getAccessorName(), " = ",
                                    getArgName(), ".get_access(cgh);"));
    }
  }
  inline std::string getRangeClass() {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    return DpctGlobalInfo::printCtadClass(
               OS, MapNames::getClNamespace() + "::range",
               getType()->getDimension())
        .str();
  }
  std::string getRangeDecl(const std::string &MemSize) {
    return buildString(getRangeClass(), " ", getRangeName(),
                       getType()->getRangeArgument(MemSize, false), ";");
  }
  ParameterStream &getFuncDecl(ParameterStream &PS) {

    if (AccMode == Value) {
      PS << getAccessorDataType(true) << " ";
    } else if (AccMode == Pointer) {
      PS << getAccessorDataType(true) << " *";
    } else {
      PS << getDpctAccessorType(true) << " ";
    }
    return PS << getArgName();
  }
  ParameterStream &getFuncArg(ParameterStream &PS) {
    return PS << getArgName();
  }
  ParameterStream &getKernelArg(ParameterStream &PS) {
    if (isShared() || DpctGlobalInfo::getUsmLevel() == UsmLevel::none) {
      if (AccMode == Accessor) {
        PS << getDpctAccessorType(true) << "(";
        PS << getAccessorName();
        if (isShared()) {
          PS << ", " << getRangeName();
        }
        PS << ")";
      } else if (AccMode == Pointer) {
        if (!getType()->isWritten())
          PS << "(" << getAccessorDataType(true) << " *)";
        PS << getAccessorName() << ".get_pointer()";
      } else {
        PS << getAccessorName();
      }
    } else {
      if (AccMode == Accessor) {
        PS << getAccessorName();
      } else {
        if (AccMode == Value) {
          PS << "*";
        }
        PS << getPtrName();
      }
    }
    return PS;
  }

private:
  static VarAttrKind getAddressAttr(const AttrVec &Attrs);

  void setInitList(const Expr *E) {
    if (auto Ctor = dyn_cast<CXXConstructExpr>(E)) {
      if (!Ctor->getNumArgs() || Ctor->getArg(0)->isDefaultArgument())
        return;
    }
    InitList = getStmtSpelling(E);
  }

  std::string getMemoryType();
  inline std::string getMemoryType(const std::string &MemoryType,
                                   std::shared_ptr<CtTypeInfo> VarType) {
    return buildString(MemoryType, "<",
                       VarType->getTemplateSpecializationName(), ", ",
                       VarType->getDimension(), ">");
  }
  std::string getInitArguments(const std::string &MemSize,
                               bool MustArguments = false) {
    if (InitList.empty())
      return getType()->getRangeArgument(MemSize, MustArguments);
    if (getType()->getDimension())
      return buildString("(", getRangeClass(),
                         getType()->getRangeArgument(MemSize, true),
                         ", " + InitList, ")");
    return buildString("(", InitList, ")");
  }
  const std::string &getMemoryAttr();
  std::string getAccessorDataType(bool UsingTemplateName) {
    if (isExtern()) {
      return "uint8_t";
    } else if (UsingTemplateName) {
      return getType()->getTemplateSpecializationName();
    } else {
      return getType()->getBaseName();
    }
  }
  std::string getDpctAccessorType(bool UsingTemplateName) {
    auto Type = getType();
    return buildString("dpct::accessor<",
                       getAccessorDataType(UsingTemplateName), ", ",
                       getMemoryAttr(), ", ", Type->getDimension(), ">");
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
    return getName();
  }

private:
  /// Passing by dpct::accessor, value or pointer when invoking kernel.
  /// Constant scalar variables are passed by value while other 0/1D variables
  /// defined on device memory are passed by pointer in device function calls.
  /// The rest are passed by dpct::accessor.
  enum DpctAccessMode {
    Value,
    Pointer,
    Accessor,
  };

private:
  VarAttrKind Attr;
  VarScope Scope;
  DpctAccessMode AccMode;
  bool PointerAsArray;
  std::string InitList;

  static const std::string ExternVariableName;
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

  ParameterStream &printType(ParameterStream &PS,
                               const std::string &TemplateName) {
    PS << TemplateName << "<" << DataType << ", " << Dimension;
    if (IsArray)
      PS << ", true";
    PS << ">";
    return PS;
  }
};

// Texture info.
class TextureInfo {
protected:
  const std::string FilePath;
  const unsigned Offset;
  const std::string Name;

  std::shared_ptr<TextureTypeInfo> Type;

protected:
  TextureInfo(unsigned Offset, const std::string &FilePath, StringRef Name)
      : FilePath(FilePath), Offset(Offset), Name(Name) {}
  TextureInfo(const VarDecl *VD)
      : TextureInfo(DpctGlobalInfo::getLocInfo(
                        VD->getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
                    VD->getName()) {}
  TextureInfo(std::pair<StringRef, unsigned> LocInfo, StringRef Name)
      : TextureInfo(LocInfo.second, LocInfo.first.str(), Name) {}

  ParameterStream &getDecl(ParameterStream &PS,
                             const std::string &TemplateDeclName) {
    return Type->printType(PS, "dpct::" + TemplateDeclName) << " " << Name;
  }

public:
  TextureInfo(unsigned Offset, const std::string &FilePath, const VarDecl *VD)
      : TextureInfo(Offset, FilePath, VD->getName()) {
    if (auto D = dyn_cast<ClassTemplateSpecializationDecl>(
            VD->getType()->getAsCXXRecordDecl())) {
      auto &TemplateList = D->getTemplateInstantiationArgs();
      auto DataTy = TemplateList[0].getAsType();
      if (auto ET = dyn_cast<ElaboratedType>(DataTy))
        DataTy = ET->getNamedType();
      setType(DpctGlobalInfo::getUnqualifiedTypeName(DataTy),
              TemplateList[1].getAsIntegral().getExtValue());
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
    getDecl(PS, "image") << ";";
    Type->endForImage();
    return PS.Str;
  }

  virtual std::string getSamplerDecl() {
    return buildString("auto ", Name, "_smpl = ", Name, ".get_sampler();");
  }
  virtual std::string getAccessorDecl() {
    return buildString("auto ", Name, "_acc = ", Name, ".get_access(cgh);");
  }

  inline ParameterStream &getFuncDecl(ParameterStream &PS) {
    return getDecl(PS, "image_accessor");
  }
  inline ParameterStream &getFuncArg(ParameterStream &PS) {
    return PS << Name;
  }
  inline ParameterStream &getKernelArg(ParameterStream &OS) {
    getType()->printType(OS, "dpct::image_accessor");
    OS << "(" << Name << "_smpl, " << Name << "_acc)";
    return OS;
  }
  inline const std::string &getName() { return Name; }

  inline unsigned getOffset() { return Offset; }
};

// texture handle info
class TextureObjectInfo : public TextureInfo {
  static const int ReplaceTypeLength;

  /// If it is a parameter in the function, it is the parameter index,either it
  /// is 0.
  unsigned ParamIdx;

  TextureObjectInfo(const VarDecl *VD, unsigned ParamIdx)
      : TextureInfo(VD), ParamIdx(ParamIdx) {}

public:
  TextureObjectInfo(const ParmVarDecl *PVD)
      : TextureObjectInfo(PVD, PVD->getFunctionScopeIndex()) {}
  TextureObjectInfo(const VarDecl *VD) : TextureObjectInfo(VD, 0) {}
  virtual ~TextureObjectInfo() = default;
  std::string getAccessorDecl() override {
    ParameterStream PS;
    PS << "auto " << Name << "_acc = static_cast<";
    getType()->printType(PS, "dpct::image")
        << " *>(" << Name << ")->get_access(cgh);";
    return PS.Str;
  }
  std::string getSamplerDecl() override {
    return buildString("auto ", Name, "_smpl = ", Name, "->get_sampler();");
  }
  inline unsigned getParamIdx() const { return ParamIdx; }

  std::string getParamDeclType() {
    ParameterStream PS;
    Type->printType(PS, "dpct::image_accessor");
    return PS.Str;
  }

  void addParamDeclReplacement() {
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
  std::string getAccessorDecl() override {
    ParameterStream PS;
    PS << "auto " << Name << "_acc = static_cast<";
    getType()->printType(PS, "dpct::image")
        << " *>(" << ArgStr << ")->get_access(cgh);";
    return PS.Str;
  }
  std::string getSamplerDecl() override {
    return buildString("auto ", Name, "_smpl = ", ArgStr, "->get_sampler();");
  }
};

class TemplateArgumentInfo {
public:
  explicit TemplateArgumentInfo(const TemplateArgumentLoc &TAL)
      : Kind(TAL.getArgument().getKind()) {
    setArgFromExprAnalysis(TAL);
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
              "PlaceHolder/*Fix the type mannually*/");
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
  void setAsNonType(const llvm::APInt &Int) {
    setArgStr(Int.toString(10, true));
    Kind = TemplateArgument::Integral;
  }
  void setAsNonType(const Expr *E) {
    setArgFromExprAnalysis(E);
    Kind = TemplateArgument::Expression;
  }

  static bool isPlaceholderType(clang::QualType QT);

private:
  template <class T> void setArgFromExprAnalysis(const T &Arg) {
    ExprAnalysis EA;
    EA.analyze(Arg);
    DependentStr = EA.getTemplateDependentStringInfo();
  }
  void setArgStr(std::string &&Str) {
    DependentStr =
        std::make_shared<TemplateDependentStringInfo>(std::move(Str));
  }
  std::shared_ptr<TemplateDependentStringInfo> DependentStr;
  TemplateArgument::ArgKind Kind;
  bool IsWritten = true;
  bool IsTemplateDependent = false;
};

// memory variable map includes memory variable used in __global__/__device__
// function and call expression.
class MemVarMap {
public:
  MemVarMap() : HasItem(false), HasStream(false) {}

  bool hasItem() const { return HasItem; }
  bool hasStream() const { return HasStream; }
  bool hasExternShared() const { return !ExternVarMap.empty(); }
  inline void setItem(bool Has = true) { HasItem = Has; }
  inline void setStream(bool Has = true) { HasStream = Has; }
  inline void addTexture(std::shared_ptr<TextureInfo> Tex) {
    TextureMap.insert(std::make_pair(Tex->getOffset(), Tex));
  }
  void addVar(std::shared_ptr<MemVarInfo> Var) {
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
    merge(LocalVarMap, VarMap.LocalVarMap, TemplateArgs);
    merge(GlobalVarMap, VarMap.GlobalVarMap, TemplateArgs);
    merge(ExternVarMap, VarMap.ExternVarMap, TemplateArgs);
    dpct::merge(TextureMap, VarMap.TextureMap);
  }
  int calculateExtraArgsSize() const {
    int Size = 0;
    if (hasStream())
      Size += MapNames::KernelArgTypeSizeMap.at(KernelArgType::Stream);

    Size = Size + calculateExtraArgsSize(LocalVarMap) +
           calculateExtraArgsSize(GlobalVarMap) +
           calculateExtraArgsSize(ExternVarMap);
    Size = Size + TextureMap.size() *
                      MapNames::KernelArgTypeSizeMap.at(KernelArgType::Texture);

    return Size;
  }
  std::string getExtraCallArguments(bool HasPreParam, bool HasPostParam) const;

  // If want adding the ExtraParam with new line, the second argument should be
  // true, and the third argument is the string of indent, which will occur
  // before each ExtraParam.
  std::string
  getExtraDeclParam(bool HasPreParam, bool HasPostParam,
                    FormatInfo FormatInformation = FormatInfo()) const;
  std::string getKernelArguments(bool HasPreParam, bool HasPostParam) const;

  const MemVarInfoMap &getMap(MemVarInfo::VarScope Scope) const {
    return const_cast<MemVarMap *>(this)->getMap(Scope);
  }
  const GlobalMap<TextureInfo> &getTextureMap() const { return TextureMap; }

  MemVarInfoMap &getMap(MemVarInfo::VarScope Scope) {
    switch (Scope) {
    case clang::dpct::MemVarInfo::Local:
      return LocalVarMap;
    case clang::dpct::MemVarInfo::Extern:
      return ExternVarMap;
    case clang::dpct::MemVarInfo::Global:
      return GlobalVarMap;
    default:
      llvm::dbgs() << "[MemVarMap::getMap] Unknow variable scope.";
      static MemVarInfoMap InvalidMap;
      return InvalidMap;
    }
  }

  enum CallOrDecl {
    CallArgument = 0,
    KernelArgument,
    DeclParameter,
  };

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
    return PS << DpctGlobalInfo::getItemName();
  }

  template <CallOrDecl COD>
  inline ParameterStream &getStream(ParameterStream &PS) const {
    return PS << DpctGlobalInfo::getStreamName();
  }

  template <CallOrDecl COD>
  inline std::string getArgumentsOrParameters(int PreParams, int PostParams,
                           FormatInfo FormatInformation = FormatInfo()) const {
    ParameterStream PS;
    if (PreParams != 0)
      PS << ", ";
    if (hasItem())
      getItem<COD>(PS) << ", ";
    if (hasStream())
      getStream<COD>(PS) << ", ";
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
    for (auto VI : VarMap) {
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
  inline void
  getArgumentsOrParametersForDecl(ParameterStream &PS, int PreParams,
                                  int PostParams) const;

  bool HasItem, HasStream;
  MemVarInfoMap LocalVarMap;
  MemVarInfoMap GlobalVarMap;
  MemVarInfoMap ExternVarMap;
  GlobalMap<TextureInfo> TextureMap;
};

template <>
inline ParameterStream &
MemVarMap::getItem<MemVarMap::DeclParameter>(ParameterStream &PS) const {
  static std::string ItemParamDecl = MapNames::getClNamespace() +
                                     "::nd_item<3> " +
                                     DpctGlobalInfo::getItemName();
  return PS << ItemParamDecl;
}

template <>
inline ParameterStream &
MemVarMap::getStream<MemVarMap::DeclParameter>(ParameterStream &PS) const {
  static std::string StreamParamDecl = MapNames::getClNamespace() +
                                       "::stream " +
                                       DpctGlobalInfo::getStreamName();
  return PS << StreamParamDecl;
}

inline void MemVarMap::getArgumentsOrParametersForDecl(
    ParameterStream &PS, int PreParams, int PostParams) const {
  if (hasItem()) {
    getItem<MemVarMap::DeclParameter>(PS);
  }

  if (hasStream()) {
    getStream<MemVarMap::DeclParameter>(PS);
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

  // Remove pre spiliter
  unsigned int RemoveLength = 0;
  if (PreParams == 0) {
    if (FormatInformation.IsAllParamsOneLine) {
      // comma and space
      RemoveLength = 2;
    } else {
      // calculate length from the first charactor "," to the next nospace
      // charactor
      RemoveLength = 1;
      while (RemoveLength < Result.size()) {
        if (!isspace(Result[RemoveLength]))
          break;
        RemoveLength++;
      }
    }
    Result = Result.substr(RemoveLength, Result.size() - RemoveLength);
  }

  // Add post spiliter
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
  CallFunctionExpr(unsigned Offset, const std::string &FilePathIn,
    const CallExpr *CE);

  void buildCallExprInfo(const CallExpr *CE);

  inline const MemVarMap &getVarMap() { return VarMap; }
  inline const std::vector<std::shared_ptr<TextureObjectInfo>> &
  getTextureObjectList() {
    return TextureObjectList;
  }

  void emplaceReplacement();
  inline bool hasArgs() { return HasArgs; }
  inline bool hasWrittenTemplateArgs() {
    for (auto &Arg : TemplateArgs)
      if (!Arg.isNull() && Arg.isWritten())
        return true;
    return false;
  }
  inline const std::string &getName() { return Name; }

  std::string getTemplateArguments(bool WithScalarWrapped = false);

  inline virtual std::string getExtraArguments();

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

protected:
  inline unsigned getBegin() { return BeginLoc; }
  inline const std::string &getFilePath() { return FilePath; }
  void buildInfo();
  std::shared_ptr<DeviceFunctionInfo> getFuncInfo() {
    return FuncInfo;
  }
  void buildCalleeInfo(const Expr *Callee);
  void resizeTextureObjectList(size_t Size) { TextureObjectList.resize(Size); }

private:
  static std::string getName(const NamedDecl *D);
  void
  buildTemplateArguments(const llvm::ArrayRef<TemplateArgumentLoc> &ArgsList) {
    if (TemplateArgs.empty())
      for (auto &Arg : ArgsList)
        TemplateArgs.emplace_back(Arg);
  }

  void buildTemplateArgumentsFromTypeLoc(const TypeLoc &TL);
  template <class TyLoc> void buildTemplateArgumentsFromSpecializationType(const TyLoc &TL) {
    for (size_t i = 0; i < TL.getNumArgs(); ++i) {
      TemplateArgs.emplace_back(TL.getArgLoc(i));
    }
  }

  void buildTextureObjectArgsInfo(const CallExpr *CE) {
    for (unsigned Idx = 0; Idx < CE->getNumArgs(); ++Idx) {
      addTextureObjectArg(
          Idx, dyn_cast<DeclRefExpr>(CE->getArg(Idx)->IgnoreImpCasts()),
          CE->getStmtClass() == Stmt::CUDAKernelCallExprClass);
    }
  }
  void mergeTextureObjectTypeInfo();

private:
  const std::string FilePath;
  unsigned BeginLoc;
  unsigned ExtraArgLoc;
  std::string Name;
  std::vector<TemplateArgumentInfo> TemplateArgs;
  std::shared_ptr<DeviceFunctionInfo> FuncInfo;
  MemVarMap VarMap;
  bool HasArgs;
  std::vector<std::shared_ptr<TextureObjectInfo>> TextureObjectList;
};

// device function declaration info includes location, name, and related
// DeviceFunctionInfo
class DeviceFunctionDecl {
public:
  DeviceFunctionDecl(unsigned Offset, const std::string &FilePathIn,
                     const FunctionDecl *FD)
      : Offset(Offset), FilePath(FilePathIn), ParamsNum(FD->param_size()),
        ReplaceOffset(0), ReplaceLength(0) {
    if (!FilePath.empty()) {
      SourceProcessType FileType = GetSourceFileType(FilePath);
      if ((FileType & TypeCudaHeader || FileType & TypeCppHeader) &&
          FD->isThisDeclarationADefinition()) {
        IsDefFilePathNeeded = false;
      } else {
        IsDefFilePathNeeded = FD->isThisDeclarationADefinition();
      }
    }
    IsStatic = FD->getStorageClass() == SC_Static;
    buildReplaceLocInfo(FD);
    buildTextureObjectParamsInfo(FD);
  }

  bool inline isStatic() { return IsStatic; }
  inline static std::shared_ptr<DeviceFunctionInfo>
  LinkUnresolved(const UnresolvedLookupExpr *ULE) {
    return LinkDeclRange(ULE->decls());
  }
  inline static std::shared_ptr<DeviceFunctionInfo>
  LinkRedecls(const FunctionDecl *FD) {
    if (auto D = DpctGlobalInfo::getInstance().findDeviceFunctionDecl(FD))
      return D->getFuncInfo();
    return LinkDeclRange(FD->redecls());
  }
  inline static std::shared_ptr<DeviceFunctionInfo>
  LinkTemplateDecl(const FunctionTemplateDecl *FTD) {
    return LinkDeclRange(FTD->specializations());
  }

  inline std::shared_ptr<DeviceFunctionInfo> getFuncInfo() const {
    return FuncInfo;
  }
  void emplaceReplacement();

private:
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
  LinkDeclRange(IteratorRange &&Range) {
    std::shared_ptr<DeviceFunctionInfo> Info;
    DeclList List;
    LinkDeclRange(std::move(Range), List, Info);
    if (List.empty())
      return Info;
    if (!Info)
      Info = std::make_shared<DeviceFunctionInfo>(
          List[0]->ParamsNum, List[0]->isStatic(), List[0]->NonDefaultParamNum);
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

private:
  void buildTextureObjectParamsInfo(const FunctionDecl *FD) {
    TextureObjectList.assign(FD->getNumParams(),
                             std::shared_ptr<TextureObjectInfo>());
    for (unsigned Idx = 0; Idx < FD->getNumParams(); ++Idx) {
      auto Param = FD->getParamDecl(Idx);
      if (DpctGlobalInfo::getUnqualifiedTypeName(Param->getType()) ==
          "cudaTextureObject_t")
        TextureObjectList[Idx] = std::make_shared<TextureObjectInfo>(Param);
    }
  }
  template <class T>
  inline void
  calculateRemoveLength(const FunctionDecl *FD, const std::string &AttrStr,
                        unsigned int &NeedRemoveLength,
                        const SourceLocation &BeginParamLoc,
                        const SourceManager &SM, const LangOptions &LO) {
    if (const Attr *A = FD->getAttr<T>()) {
      if (SM.isMacroArgExpansion(A->getRange().getBegin()) &&
          SM.isMacroArgExpansion(BeginParamLoc))
        return;
      auto Begin = SM.getExpansionLoc(A->getRange().getBegin());
      auto BeginParamLocExpand = SM.getExpansionLoc(BeginParamLoc);
      auto Length = Lexer::MeasureTokenLength(Begin, SM, LO);
      auto C = SM.getCharacterData(Begin);
      std::string Str = std::string(C, C + Length);
      bool InValidFlag = false;
      if (Str == AttrStr &&
          isInSameLine(Begin, BeginParamLocExpand, SM, InValidFlag) &&
          !InValidFlag) {
        NeedRemoveLength =
            NeedRemoveLength +
            getLenIncludingTrailingSpaces(SourceRange(Begin, Begin), SM);
      }
    }
  }

  unsigned Offset;
  const std::string FilePath;
  unsigned ParamsNum;
  unsigned ReplaceOffset;
  unsigned ReplaceLength;
  unsigned NonDefaultParamNum;
  bool IsDefFilePathNeeded;
  std::shared_ptr<DeviceFunctionInfo> FuncInfo;
  bool IsStatic;

  std::vector<std::shared_ptr<TextureObjectInfo>> TextureObjectList;
  FormatInfo FormatInformation;
};

// device function info includes parameters num, memory variable and call
// expression in the function.
class DeviceFunctionInfo {
public:
  DeviceFunctionInfo(size_t ParamsNum, bool IsStatic, size_t NonDefaultParamNum)
      : ParamsNum(ParamsNum), NonDefaultParamNum(NonDefaultParamNum),
        IsBuilt(false), IsStatic(IsStatic),
        TextureObjectTypeList(ParamsNum, std::shared_ptr<TextureTypeInfo>()) {}

  inline std::shared_ptr<CallFunctionExpr> addCallee(const CallExpr *CE) {
    auto CallLocInfo = DpctGlobalInfo::getLocInfo(CE);
    auto C =
        insertObject(CallExprMap, CallLocInfo.second, CallLocInfo.first, CE);
    C->buildCallExprInfo(CE);
    return C;
  }
  inline void addVar(std::shared_ptr<MemVarInfo> Var) { VarMap.addVar(Var); }
  inline void setItem() { VarMap.setItem(); }
  inline void setStream() { VarMap.setStream(); }
  inline void addTexture(std::shared_ptr<TextureInfo> Tex) {
    VarMap.addTexture(Tex);
  }
  inline const MemVarMap &getVarMap() { return VarMap; }
  inline std::shared_ptr<TextureTypeInfo> getTextureTypeInfo(unsigned Idx) {
    if (Idx < TextureObjectTypeList.size())
      return TextureObjectTypeList[Idx];
    return std::shared_ptr<TextureTypeInfo>();
  }

  void buildInfo();
  inline bool hasParams() { return ParamsNum != 0; }

  inline bool isBuilt() { return IsBuilt; }
  inline void setBuilt() { IsBuilt = true; }

  inline std::string
  getExtraParameters(FormatInfo FormatInformation = FormatInfo()) {
    buildInfo();
    return VarMap.getExtraDeclParam(
        NonDefaultParamNum, ParamsNum - NonDefaultParamNum, FormatInformation);
  }

  void setDefinitionFilePath(const std::string &Path) {
    DefinitionFilePath = Path;
  }
  const std::string &getDefinitionFilePath() { return DefinitionFilePath; }
  void setNeedSyclExternMacro() {
    if (!IsStatic)
      NeedSyclExternMacro = true;
  }
  bool IsSyclExternMacroNeeded() { return NeedSyclExternMacro; }
  void inline setStatic(bool Static = true) { IsStatic = Static; }
  void merge(std::shared_ptr<DeviceFunctionInfo> Other);
  size_t ParamsNum;
  size_t NonDefaultParamNum;

private:
  void mergeCalledTexObj(
      const std::vector<std::shared_ptr<TextureObjectInfo>> &TexObjList);

  void mergeTextureTypeList(
      const std::vector<std::shared_ptr<TextureTypeInfo>> &Other);

  bool IsBuilt;
  std::string DefinitionFilePath;
  bool NeedSyclExternMacro = false;
  bool IsStatic;

  GlobalMap<CallFunctionExpr> CallExprMap;
  MemVarMap VarMap;

  std::vector<std::shared_ptr<TextureTypeInfo>> TextureObjectTypeList;
};

// kernel call info is specific CallFunctionExpr, which include info of kernel
// call.
class KernelCallExpr : public CallFunctionExpr {
public:
  bool IsInMacroDefine = false;
private:
  using StmtList = std::vector<std::string>;

  struct ArgInfo {
    ArgInfo(KernelArgumentAnalysis &Analysis, const Expr *Arg, bool Used,
            int Index, KernelCallExpr* BASE)
        : IsPointer(false), IsRedeclareRequired(false),
          IsUsedAsLvalueAfterMalloc(Used), Index(Index) {
      Analysis.analyze(Arg);
      ArgString = Analysis.getReplacedString();
      if (DpctGlobalInfo::getUsmLevel() == UsmLevel::none)
        IsPointer = Analysis.IsPointer;
      IsRedeclareRequired = Analysis.IsRedeclareRequired;
      IsDefinedOnDevice = Analysis.IsDefinedOnDevice;
      IsKernelParamPtr = Analysis.IsKernelParamPtr;

      if (IsPointer) {
        QualType PointerType;
        if (Arg->getType().getTypePtr()->getTypeClass() ==
            Type::TypeClass::Decayed) {
          PointerType = Arg->getType().getCanonicalType();
        } else {
          PointerType = Arg->getType();
        }
        TypeString = DpctGlobalInfo::getReplacedTypeName(PointerType);
        ArgSize = MapNames::KernelArgTypeSizeMap.at(KernelArgType::Default);

        // Currently, all the device RNG state struct are passed to kernel by
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
          ArgSize = MapNames::KernelArgTypeSizeMap.at(KernelArgType::Default);
      }

      if (IsRedeclareRequired || IsPointer || BASE->IsInMacroDefine) {
        IdString = getTempNameForExpr(Arg, false, true, BASE->IsInMacroDefine);
      }
    }

    ArgInfo(const ParmVarDecl *PVD, const std::string &ArgsArrayName,
            KernelCallExpr *Kernel)
        : IsPointer(DpctGlobalInfo::getUsmLevel() == UsmLevel::none &&
                    PVD->getType()->isPointerType()),
          IsRedeclareRequired(true), IsUsedAsLvalueAfterMalloc(true),
          TypeString(DpctGlobalInfo::getReplacedTypeName(PVD->getType())),
          IdString(PVD->getName().str() + "_"),
          Index(PVD->getFunctionScopeIndex()) {
      /// For parameter declaration 'float *a' with index = 2 and args array's
      /// name is 'args', the arg string will be '*(float **)args[2]'.
      std::ostringstream OS;
      /// Get pointer type of the parameter declaration's type, e.g. 'float **'.
      auto CastPointerType =
          DpctGlobalInfo::getContext().getPointerType(PVD->getType());
      /// Print '*(float **)'.
      OS << "*(" << DpctGlobalInfo::getReplacedTypeName(CastPointerType) << ")";
      /// Print args array subscript.
      OS << ArgsArrayName << "[" << Index << "]";
      ArgString = OS.str();

      if (TextureObjectInfo::isTextureObject(PVD)) {
        IsRedeclareRequired = false;
        Texture = std::make_shared<CudaLaunchTextureObjectInfo>(PVD, ArgString);
        Kernel->addTextureObjectArgInfo(Index, Texture);
      }
    }

    ArgInfo(std::shared_ptr<TextureObjectInfo> Obj, KernelCallExpr *BASE)
        : Texture(Obj) {
      IsPointer = false;
      IsRedeclareRequired = false;
      TypeString = "";
      Index = 0;
      ArgSize = MapNames::KernelArgTypeSizeMap.at(KernelArgType::Texture);
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
    bool IsKernelParamPtr = false;
    std::string ArgString;
    std::string TypeString;
    std::string IdString;
    int Index;
    int ArgSize = 0;
    bool IsDeviceRandomGeneratorType = false;

    std::shared_ptr<TextureObjectInfo> Texture;
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
    template <class... Args> KernelPrinter &line(Args &&... Arguments) {
      appendString(Stream, Indent, std::forward<Args>(Arguments)..., NL);
      return *this;
    }
    KernelPrinter &operator<<(const StmtList &Stmts) {
      for (auto &S : Stmts)
        line(S);
      return *this;
    }
    KernelPrinter &indent() { return (*this) << Indent; }
    KernelPrinter &newLine() { return (*this) << NL; }
    std::string str() {
      auto Result = Stream.str();
      return Result.substr(Indent.length(),
        Result.length() - Indent.length() - NL.length());;
    }
  };

  void print(KernelPrinter &Printer);
  void printSubmit(KernelPrinter &Printer);
  void printSubmitLamda(KernelPrinter &Printer);
  void printParallelFor(KernelPrinter &Printer);
  void printKernel(KernelPrinter &Printer);

public:
  KernelCallExpr(unsigned Offset, const std::string &FilePath,
                 const CUDAKernelCallExpr *KernelCall)
      : CallFunctionExpr(Offset, FilePath, KernelCall), IsSync(false) {
    setIsInMacroDefine(KernelCall);
    buildCallExprInfo(KernelCall);
    buildArgsInfo(KernelCall);
    buildKernelInfo(KernelCall);
  }

  void addAccessorDecl();
  void buildInfo();
  inline std::string getExtraArguments() override {
    if (!getFuncInfo()) {
      return "";
    }
    return getVarMap().getKernelArguments(
        getFuncInfo()->NonDefaultParamNum,
        getFuncInfo()->ParamsNum - getFuncInfo()->NonDefaultParamNum);
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

private:
  KernelCallExpr(unsigned Offset, const std::string &FilePath)
      : CallFunctionExpr(Offset, FilePath, nullptr), IsSync(false) {}

  void buildArgsInfoFromArgsArray(const FunctionDecl *FD,
                                  const Expr *ArgsArray) {}
  void buildArgsInfo(const CallExpr *CE) {
    KernelArgumentAnalysis Analysis(IsInMacroDefine);
    auto &TexList = getTextureObjectList();

    for (unsigned Idx = 0; Idx < CE->getNumArgs(); ++Idx) {
      if (auto Obj = TexList[Idx]) {
        ArgsInfo.emplace_back(Obj, this);
      } else {
        auto Arg = CE->getArg(Idx);
        bool Used = true;
        if (auto *ArgDRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts()))
          Used = isArgUsedAsLvalueUntil(ArgDRE, CE);
        ArgsInfo.emplace_back(Analysis, Arg, Used, Idx, this);
      }
    }
  }
  bool isIncludedFile(const std::string &CurrentFile,
                      const std::string &CheckingFile);
  void buildKernelInfo(const CUDAKernelCallExpr *KernelCall);
  void setIsInMacroDefine(const CUDAKernelCallExpr *KernelCall);
  void buildNeedBracesInfo(const CallExpr *KernelCall);
  void buildLocationInfo(const CallExpr*KernelCall);
  template <class ArgsRange>
  void buildExecutionConfig(const ArgsRange &ConfigArgs) {
    int Idx = 0;
    bool LocalReversed = false, GroupReversed = false;
    for (auto Arg : ConfigArgs) {
      KernelConfigAnalysis A(IsInMacroDefine);
      A.analyze(Arg, Idx < 2);
      ExecutionConfig.Config[Idx] = A.getReplacedString();
      if (Idx == 0) {
        GroupReversed = A.reversed();
        ExecutionConfig.GroupDirectRef = A.isDirectRef();
      } else if (Idx == 1) {
        LocalReversed = A.reversed();
        ExecutionConfig.LocalDirectRef = A.isDirectRef();
      }
      ++Idx;
    }
    ExecutionConfig.DeclLocalRange =
        !LocalReversed && !ExecutionConfig.LocalDirectRef;
    ExecutionConfig.DeclGroupRange =
        LocalReversed && !GroupReversed && !ExecutionConfig.GroupDirectRef;
    ExecutionConfig.DeclGlobalRange = !LocalReversed && !GroupReversed;

    if (ExecutionConfig.Stream == "0") {
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      QueueStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
      buildTempVariableMap(Index, *ConfigArgs.begin(),
                           HelperFuncType::DefaultQueue);
    }
  }

  void removeExtraIndent() {
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(getFilePath(),
                                         getBegin() - LocInfo.Indent.length(),
                                         LocInfo.Indent.length(), "", nullptr));
  }
  void addAccessorDecl(MemVarInfo::VarScope Scope);
  void addAccessorDecl(std::shared_ptr<MemVarInfo> VI);
  void addStreamDecl() {
    if (getVarMap().hasStream())
      SubmitStmtsList.StreamList.emplace_back(buildString(
          MapNames::getClNamespace() + "::stream ",
          DpctGlobalInfo::getStreamName(), "(64 * 1024, 80, cgh);"));
  }
  void addNdRangeDecl() {
    if (ExecutionConfig.DeclGlobalRange) {
      SubmitStmtsList.NdRangeList.emplace_back(
          buildString("auto dpct_global_range = ", ExecutionConfig.GroupSize,
                      " * ", ExecutionConfig.LocalSize, ";"));
    }
    if (ExecutionConfig.DeclGroupRange) {
      SubmitStmtsList.NdRangeList.emplace_back(buildString(
          "auto dpct_group_range = ", ExecutionConfig.GroupSize, ";"));
    }
    if (ExecutionConfig.DeclLocalRange) {
      SubmitStmtsList.NdRangeList.emplace_back(buildString(
          "auto dpct_local_range = ", ExecutionConfig.LocalSize, ";"));
    }
  }

  void buildKernelArgsStmt();
  void printReverseRange(KernelPrinter &Printer, const std::string &RangeName) {
    DpctGlobalInfo::printCtadClass(Printer,
                                   MapNames::getClNamespace() + "::range", 3)
        << "(" << RangeName << ".get(2), " << RangeName << ".get(1), "
        << RangeName << ".get(0))";
  }
  void printKernelRange(KernelPrinter &Printer, const std::string &RangeStr,
                        const std::string &DeclName, bool DeclRange,
                        bool DirectRef) {
    if (DeclRange) {
      printReverseRange(Printer, DeclName);
    } else if (DirectRef) {
      printReverseRange(Printer, RangeStr);
    } else {
      Printer << RangeStr;
    }
  }

  struct {
    std::string LocHash;
    std::string NL;
    std::string Indent;
  } LocInfo;
  // true, if migrated DPC++ code block need extra { }
  bool NeedBraces = true;
  struct {
    std::string Config[4] = {"", "", "", "0"};
    std::string &GroupSize = Config[0];
    std::string &LocalSize = Config[1];
    std::string &ExternMemSize = Config[2];
    std::string &Stream = Config[3];
    bool DeclGlobalRange = false, DeclLocalRange = false,
         DeclGroupRange = false;
    bool LocalDirectRef = false, GroupDirectRef = false;
  } ExecutionConfig;

  std::string Event;
  bool IsSync;
  std::vector<ArgInfo> ArgsInfo;

  class {
  public:
    StmtList StreamList;
    StmtList RangeList;
    StmtList MemoryList;
    StmtList ExternList;
    StmtList PtrList;
    StmtList AccessorList;
    StmtList TextureList;
    StmtList SamplerList;
    StmtList NdRangeList;
    StmtList CommandGroupList;

    inline KernelPrinter &print(KernelPrinter &Printer) {
      printList(Printer, StreamList);
      printList(Printer, ExternList);
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
  std::string QueueStr;
  int TotalArgsSize = 0;
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

class RandomEngineInfo {
public:
  RandomEngineInfo(unsigned Offset, const std::string &FilePath,
                   const DeclaratorDecl *DD)
      : SeedExpr("0"), DimExpr("1"), IsQuasiEngine(false), IsClassMember(false),
        NeedPrint(true) {
    if (dyn_cast<FieldDecl>(DD))
      IsClassMember = true;
    else
      IsClassMember = false;

    DeclaratorDeclName = DD->getNameAsString();
    DeclFilePath =
        DpctGlobalInfo::getSourceManager().getFilename(DD->getBeginLoc()).str();

    DeclaratorDeclBeginOffset = DpctGlobalInfo::getSourceManager()
                                    .getDecomposedLoc(DD->getBeginLoc())
                                    .second;
    DeclaratorDeclEndOffset = DpctGlobalInfo::getSourceManager()
                                  .getDecomposedLoc(DD->getEndLoc())
                                  .second;
  }
  // Seed is an unsigned long long type value in origin code, if it is not set,
  // use 0 as default.
  // The legal value of Dim in origin code is 1 to 20000, so if it is not set,
  // use 1 as default.
  static const DeclaratorDecl *getHandleVar(const Expr *Arg) {
    const DeclaratorDecl *D = nullptr;
    if (auto UO = dyn_cast<UnaryOperator>(Arg->IgnoreImpCasts())) {
      if (UO->getOpcode() == UO_AddrOf) {
        D = getDecl(UO->getSubExpr());
      }
    } else {
      D = getDecl(Arg);
    }
    return D;
  }
  static const DeclaratorDecl *getDecl(const Expr *E) {
    if (auto DeclRef = dyn_cast<DeclRefExpr>(E->IgnoreImpCasts())) {
      if (dyn_cast<VarDecl>(DeclRef->getDecl()))
        return dyn_cast<DeclaratorDecl>(DeclRef->getDecl());
    } else if (auto Member = dyn_cast<MemberExpr>(E->IgnoreImpCasts())) {
      if (dyn_cast<FieldDecl>(Member->getMemberDecl()))
        return dyn_cast<DeclaratorDecl>(Member->getMemberDecl());
    }
    return nullptr;
  }

  void setEngineTypeReplacement(std::string EngineType) {
    TypeReplacement = EngineType;
  }
  void setSeedExpr(const Expr *Seed) {
    ArgumentAnalysis AS(Seed, false);
    AS.analyze();
    SeedExpr = AS.getReplacedString();
  }
  void setDimExpr(const Expr *Dim) {
    ArgumentAnalysis AD(Dim, false);
    AD.analyze();
    DimExpr = AD.getReplacedString();
  }
  std::string getSeedExpr() { return SeedExpr; }
  std::string getDimExpr() { return DimExpr; }

  void setCreateCallFilePath(std::string Path) { CreateCallFilePath = Path; }
  void setTypeBeginOffset(unsigned int Offset) { TypeBeginOffset = Offset; }
  void setTypeLength(unsigned int Len) { TypeLength = Len; }
  void setCreateAPIBegin(unsigned int Offset) { CreateAPIBegin = Offset; }
  void setCreateAPILength(unsigned int Len) { CreateAPILength = Len; }

  void setTypeReplacement(std::string Repl) { TypeReplacement = Repl; }
  void setQuasiEngineFlag() { IsQuasiEngine = true; }

  void setIdentifierEndOffset(unsigned int Offset) {
    IdentifierEndOffset = Offset;
  }
  void buildInfo();
  bool isClassMember() { return IsClassMember; }
  std::string getDeclaratorDeclName() { return DeclaratorDeclName; }
  SourceLocation getDeclaratorDeclBeginLoc() {
    auto &SM = DpctGlobalInfo::getSourceManager();
    auto FE = SM.getFileManager().getFile(DeclFilePath);
    if (std::error_code ec = FE.getError())
      return SourceLocation();
    auto FID = SM.getOrCreateFileID(FE.get(), SrcMgr::C_User);
    return SM.getComposedLoc(FID, DeclaratorDeclBeginOffset);
  }
  SourceLocation getDeclaratorDeclEndLoc() {
    auto &SM = DpctGlobalInfo::getSourceManager();
    auto FE = SM.getFileManager().getFile(DeclFilePath);
    if (std::error_code ec = FE.getError())
      return SourceLocation();
    auto FID = SM.getOrCreateFileID(FE.get(), SrcMgr::C_User);
    return SM.getComposedLoc(FID, DeclaratorDeclEndOffset);
  }
  void setNeedPrint(bool Flag){ NeedPrint = Flag; }
  void setQueueStr(std::string Q) { QueueStr = Q; }

private:
  std::string SeedExpr;     // Replaced Seed variable string
  std::string DimExpr;      // Replaced Dimension variable string
  bool IsQuasiEngine; // If origin code used a quasirandom number generator,
                      // this flag need be set as true.
  std::string DeclFilePath; // Where the curandGenerator_t handle is declared.
  std::string
      CreateCallFilePath; // Where the curandCreateGenerator API is called.
  unsigned int
      TypeBeginOffset; // The offset of the begin of curandGenerator_t handle.
  unsigned int TypeLength; // The length of the curandGenerator_t handle type.
  unsigned int
      CreateAPIBegin; // The offset of the begin of curandCreateGenerator API.
  unsigned int
      CreateAPILength; // The length of the begin of curandCreateGenerator API.
  unsigned int IdentifierEndOffset; // The offset at the end of
                                    // curandGenerator_t handle declaration.
  std::string TypeReplacement;      // The replcaement string of the type of
                                    // curandGenerator_t handle.
  bool IsClassMember;               // Whether curandGenerator_t handle is a
                                    // class member.
  std::string DeclaratorDeclName;   // Name of declarator declaration.
  unsigned int DeclaratorDeclBeginOffset;
  unsigned int DeclaratorDeclEndOffset;
  bool NeedPrint;
  std::string QueueStr;
};

template <class... T>
void DpctFileInfo::insertHeader(HeaderType Type, unsigned Offset, T... Args) {
  if (!HeaderInsertedBitMap[Type]) {
    HeaderInsertedBitMap[Type] = true;
    std::string ReplStr;
    llvm::raw_string_ostream RSO(ReplStr);
    // Start a new line if we're not inserting at the first inclusion offset
    if (Offset != FirstIncludeOffset) {
      RSO << getNL();
    } else {
      if ((DpctGlobalInfo::getUsmLevel() == UsmLevel::none) && (Type == SYCL)) {
        RSO << "#define DPCT_USM_LEVEL_NONE" << getNL();
      }
      if (DpctGlobalInfo::isSyclNamedLambda() && (Type == SYCL)) {
        RSO << "#define DPCT_NAMED_LAMBDA" << getNL();
      }
    }
    concatHeader(RSO, std::forward<T>(Args)...);
    insertHeader(std::move(RSO.str()), Offset);
  }
}

/// Find the innermost FunctionDecl's child node(CompoundStmt node) where \S is
/// located. If there is no CompoundStmt of FunctionDecl out of \S, return
/// nullptr.
/// Caller should make sure that /S is not nullptr.
template<typename T>
inline const clang::CompoundStmt *findInnerMostBlock(const T *S) {
  auto &Context = DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*S);
  std::vector<ast_type_traits::DynTypedNode> AncestorNodes;
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

template<typename T>
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
  Info.DeclLocFile =
      DpctGlobalInfo::getSourceManager().getFilename(EndOfLBrace).str();
  Info.DeclLocOffset = DpctGlobalInfo::getSourceManager()
                           .getDecomposedLoc(EndOfLBrace)
                           .second;
  return Info;
}

template <typename T>
bool checkWhetherIsDuplicate(const T *S, bool UpdateSet = true) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  SourceLocation Loc = S->getBeginLoc();
  Loc = SM.getExpansionLoc(Loc);

  std::string Key = SM.getFilename(Loc).str() + ":" +
                    std::to_string(SM.getDecomposedLoc(Loc).second);
  auto Iter = DpctGlobalInfo::getTempVariableHandledSet().find(Key);
  if (Iter != DpctGlobalInfo::getTempVariableHandledSet().end()) {
    return true;
  } else {
    if (UpdateSet) {
      DpctGlobalInfo::getTempVariableHandledSet().insert(Key);
    }
    return false;
  }
}

// There are 2 maps and 1 set are used to record related information:
// unordered_map<int, HelperFuncReplInfo> HelperFuncReplInfoMap,
// unordered_map<string, TempVariableDeclCounter> TempVariableDeclCounterMap and
// unordered_set<string> TempVariableHandledSet.
//
// 1. HelperFuncReplInfoMap's key is the Index of each placeholder, its value is
// a HelperFuncReplInfo struct which saved the decalaration insert location of
// this placeholder and a boolean represent whether this location is valid.
// 2. TempVariableDeclCounterMap's key is the decalaration insert location, it's
// value is a TempVariableDeclCounter which counts how many device declaration
// and queue declaration need be inserted here respectively.
// 3. TempVariableHandledSet's key is the begin location of the decalaration or
// statement of ecah placeholder. This set is to avoid one placeholder to be
// counted more than once.
//
// The rule of inserting declaration:
// If pair (m,n) means device counter value is n and queue counter value is n,
// using (0,0), (0,1), (1,0), (1,1), (>=2,0), (0,>=2), (>=2,1), (1,>=2) and
// (>=2,>=2) can construct a graph.
// Then there are 5 edges will need insert declaration:
// (1,0) to (>=2,0) and (1,1) to (>=2,1) need add device declaration
// (0,1) to (0,>=2) and (1,1) to (1,>=2) need add both declaration
// (>=2,1) to (>=2,>=2) need add queue declaration
template <typename T>
inline void buildTempVariableMap(int Index, const T *S,
                                 HelperFuncType HFT) {
  if (checkWhetherIsDuplicate(S)) {
    return;
  }

  DpctGlobalInfo::HelperFuncReplInfo HFInfo =
      generateHelperFuncReplInfo(S);

  if (!HFInfo.IsLocationValid)
    return;

  DpctGlobalInfo::getHelperFuncReplInfoMap().insert(
      std::make_pair(Index, HFInfo));
  std::string KeyForDeclCounter =
      HFInfo.DeclLocFile + ":" + std::to_string(HFInfo.DeclLocOffset);

  auto Iter =
      DpctGlobalInfo::getTempVariableDeclCounterMap().find(KeyForDeclCounter);
  if (Iter != DpctGlobalInfo::getTempVariableDeclCounterMap().end()) {
    unsigned int IndentLen = 2;
    if (clang::dpct::DpctGlobalInfo::getGuessIndentWidthMatcherFlag())
      IndentLen = clang::dpct::DpctGlobalInfo::getIndentWidth();
    std::string IndentStr = std::string(IndentLen, ' ');

    std::string DevDecl =
        getNL() + IndentStr +
        "dpct::device_ext &dev_ct1 = dpct::get_current_device();";
    std::string QDecl = getNL() + IndentStr + MapNames::getClNamespace() +
                        "::queue &q_ct1 = dev_ct1.default_queue();";
    if (HFT == HelperFuncType::DefaultQueue) {
      if (Iter->second.DefaultQueueCounter == 1) {
        if (Iter->second.CurrentDeviceCounter <= 1) {
          if (DpctGlobalInfo::getUsingDRYPattern() &&
              !DpctGlobalInfo::getDeviceChangedFlag())
            DpctGlobalInfo::getInstance().addReplacement(
                std::make_shared<ExtReplacement>(HFInfo.DeclLocFile,
                                                 HFInfo.DeclLocOffset, 0,
                                                 DevDecl, nullptr));
        }
        if (DpctGlobalInfo::getUsingDRYPattern() &&
            !DpctGlobalInfo::getDeviceChangedFlag())
          DpctGlobalInfo::getInstance().addReplacement(
              std::make_shared<ExtReplacement>(
                  HFInfo.DeclLocFile, HFInfo.DeclLocOffset, 0, QDecl, nullptr));
      }
      Iter->second.DefaultQueueCounter = Iter->second.DefaultQueueCounter + 1;
    } else if (HFT == HelperFuncType::CurrentDevice) {
      if (Iter->second.CurrentDeviceCounter == 1 &&
          Iter->second.DefaultQueueCounter <= 1) {
        if (DpctGlobalInfo::getUsingDRYPattern() &&
            !DpctGlobalInfo::getDeviceChangedFlag())
          DpctGlobalInfo::getInstance().addReplacement(
              std::make_shared<ExtReplacement>(HFInfo.DeclLocFile,
                                               HFInfo.DeclLocOffset, 0, DevDecl,
                                               nullptr));
      }
      Iter->second.CurrentDeviceCounter = Iter->second.CurrentDeviceCounter + 1;
    }
  } else {
    DpctGlobalInfo::TempVariableDeclCounter Counter(0, 0);
    if (HFT == HelperFuncType::DefaultQueue) {
      Counter.DefaultQueueCounter = Counter.DefaultQueueCounter + 1;
    } else if (HFT == HelperFuncType::CurrentDevice) {
      Counter.CurrentDeviceCounter = Counter.CurrentDeviceCounter + 1;
    }
    DpctGlobalInfo::getTempVariableDeclCounterMap().insert(
        std::make_pair(KeyForDeclCounter, Counter));
  }
}

} // namespace dpct
} // namespace clang

#endif
