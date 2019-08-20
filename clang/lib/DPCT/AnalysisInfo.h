//===--- AnalysisInfo.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
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
#include "Diagnostics.h"
#include "ExprAnalysis.h"
#include "ExtReplacements.h"
#include <bitset>

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

namespace clang {
namespace dpct {

class CudaMallocInfo;
class TextureInfo;
class KernelCallExpr;
class DeviceFunctionInfo;
class CallFunctionExpr;
class DeviceFunctionDecl;
class MemVarInfo;
class VarInfo;

template <class T> using GlobalMap = std::map<unsigned, std::shared_ptr<T>>;
using MemVarInfoMap = GlobalMap<MemVarInfo>;

using ReplTy = std::map<std::string, tooling::Replacements>;

inline void appendString(llvm::raw_string_ostream &OS) {}
template <class T, class... Arguments>
inline void appendString(llvm::raw_string_ostream &OS, const T &S,
                         Arguments &&... Args) {
  OS << S;
  appendString(OS, std::forward<Arguments>(Args)...);
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
  Time,
  Complex,
  Future,
  MKL,
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
  inline const std::string &getFilePath() { return FilePath; }
  inline bool hasReplacements() { return isInRoot() && !Repls.empty(); }

  // Build kernel and device function declaration replacements and store them.
  void buildReplacements();

  // Emplace stored replacements into replacement set.
  void emplaceReplacements(tooling::Replacements &ReplSet /*out*/);

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
      HasInclusionDirective = true;
    }
  }

  void setLastIncludeOffset(unsigned Offset) {
    LastIncludeOffset = Offset;
  }

  void setMathHeaderInserted(bool B = true) {
    HeaderInsertedBitMap[HeaderType::Math] = B;
  }

  // Concat several header inclusion directives to one
  inline void concatHeader(llvm::raw_string_ostream &OS) {}

  template <class... Args>
  void concatHeader(llvm::raw_string_ostream &OS, std::string &&FirstHeaderName,
                    Args... Arguments) {
    appendString(OS, "#include ", std::move(FirstHeaderName), getNL());
    concatHeader(OS, std::forward<Args>(Arguments)...);
  }

  // Insert one or more header inclusion directives at a specified offset
  void insertHeader(std::string &&Repl, unsigned Offset) {
    addReplacement(
        std::make_shared<ExtReplacement>(FilePath, Offset, 0, Repl, nullptr));
  }

  // Insert one or more header inclusion directives at first or last inclusion
  // locations
  template <typename... T>
  void insertHeader(HeaderType Type, unsigned Offset, T... Args) {
    if (!HeaderInsertedBitMap[Type]) {
      HeaderInsertedBitMap[Type] = true;
      std::string ReplStr;
      llvm::raw_string_ostream RSO(ReplStr);
      if (Offset == LastIncludeOffset)
        RSO << getNL();
      concatHeader(RSO, std::forward<T>(Args)...);
      insertHeader(std::move(RSO.str()), Offset);
    }
  }

  void insertHeader(HeaderType Type) {
    switch (Type) {
    case SYCL:
      return insertHeader(HeaderType::SYCL, FirstIncludeOffset, "<CL/sycl.hpp>",
                           "<dpct/dpct.hpp>");
    case Math:
      return insertHeader(HeaderType::Math, LastIncludeOffset, "<cmath>");
    case Complex:
      return insertHeader(HeaderType::Complex, LastIncludeOffset, "<complex>");
    case Future:
      return insertHeader(HeaderType::Future, LastIncludeOffset, "<future>");
    case Time:
      return insertHeader(HeaderType::Time, LastIncludeOffset, "<time.h>");
    case MKL:
      return insertHeader(HeaderType::MKL, LastIncludeOffset, "<mkl_blas_sycl.hpp>",
                          "<mkl_lapack_sycl.hpp>", "<sycl_types.hpp>",
                          "<dpct/blas_utils.hpp>");
    }
  }

  // Record line info in file.
  struct SourceLineInfo {
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
    if (!LineNumber || LineNumber > Lines.size())
      dpct_unreachable("illegal line number " + std::to_string(LineNumber));
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

private:
  template <class Obj> GlobalMap<Obj> &getMap() {
    dpct_unreachable("unknow map type");
  }

  bool isInRoot();
  bool isInCudaPath();

  void buildLinesInfo();
  inline const SourceLineInfo &getLineInfoFromOffset(unsigned Offset) {
    return *(std::upper_bound(Lines.begin(), Lines.end(), Offset,
                              [](unsigned Offset, const SourceLineInfo &Line) {
                                return Line.Offset > Offset;
                              }) -
             1);
  }

  GlobalMap<MemVarInfo> MemVarMap;
  GlobalMap<DeviceFunctionDecl> FuncMap;
  GlobalMap<KernelCallExpr> KernelMap;
  GlobalMap<CudaMallocInfo> CudaMallocMap;
  GlobalMap<TextureInfo> TextureMap;

  ExtReplacements Repls;
  std::vector<SourceLineInfo> Lines;

  std::string FilePath;

  unsigned FirstIncludeOffset = 0;
  unsigned LastIncludeOffset = 0;
  bool HasInclusionDirective = false;

  std::bitset<32> HeaderInsertedBitMap;
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

  inline static bool isInRoot(SourceLocation SL) {
    return isInRoot(
        getSourceManager().getFilename(getSourceManager().getExpansionLoc(SL)));
  }
  static bool isInRoot(const std::string &FilePath) {
    std::string Path = FilePath;
    makeCanonical(Path);
    return isChildPath(InRoot, Path);
  }
  inline static bool isInCudaPath(SourceLocation SL) {
    return isInCudaPath(
        getSourceManager().getFilename(getSourceManager().getExpansionLoc(SL)));
  }
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
  static void setCudaPath(const std::string &InputCudaPath) {
    CudaPath = InputCudaPath;
  }
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
  inline static bool isKeepOriginCode() { return KeepOriginCode; }
  inline static void setKeepOriginCode(bool KOC = true) {
    KeepOriginCode = KOC;
  }

  template <class T>
  static inline std::pair<llvm::StringRef, unsigned>
  getLocInfo(const T *N, bool *IsInvalid = nullptr /* out */) {
    return getLocInfo(getLocation(N), IsInvalid);
  }

  static inline std::pair<llvm::StringRef, unsigned>
  getLocInfo(SourceLocation Loc, bool *IsInvalid = nullptr /* out */) {
    auto LocInfo =
        SM->getDecomposedLoc(getSourceManager().getExpansionLoc(Loc));
    if (auto FileEntry = SM->getFileEntryForID(LocInfo.first)) {
      return std::make_pair(FileEntry->getName(), LocInfo.second);
    }
    if (IsInvalid)
      *IsInvalid = true;
    return std::make_pair(StringRef(), 0);
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

  // Build kernel and device function declaration replacements and store them.
  void buildReplacements() {
    for (auto &File : FileMap)
      File.second->buildReplacements();
  }

  // Emplace stored replacements into replacement set.
  void emplaceReplacements(ReplTy &ReplSets /*out*/) {
    for (auto &File : FileMap)
      if (File.second->hasReplacements())
        File.second->emplaceReplacements(ReplSets[File.first]);
  }

  void insertCudaMalloc(const CallExpr *CE);
  void insertCublasAlloc(const CallExpr *CE);
  std::shared_ptr<CudaMallocInfo> findCudaMalloc(const Expr *CE);
  void addReplacement(std::shared_ptr<ExtReplacement> Repl) {
    insertFile(Repl->getFilePath())->addReplacement(Repl);
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

  void insertHeader(SourceLocation Loc, HeaderType Type) {
    auto LocInfo = getLocInfo(Loc);
    insertFile(LocInfo.first)->insertHeader(Type);
  }

private:
  DpctGlobalInfo() = default;

  DpctGlobalInfo(const DpctGlobalInfo &) = delete;
  DpctGlobalInfo(DpctGlobalInfo &&) = delete;
  DpctGlobalInfo &operator=(const DpctGlobalInfo &) = delete;
  DpctGlobalInfo &operator=(DpctGlobalInfo &&) = delete;

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
    if (isInRoot(LocInfo.first))
      return insertFile(LocInfo.first)->template findNode<Info>(LocInfo.second);
    return std::shared_ptr<Info>();
  }
  // Insert info if it doesn't exist.
  template <class Info, class Node>
  inline std::shared_ptr<Info> insertNode(const Node *N) {
    auto LocInfo = getLocInfo(N);
    return insertFile(LocInfo.first)
        ->template insertNode<Info>(LocInfo.second, N);
  }

  inline std::shared_ptr<DpctFileInfo>
  insertFile(const std::string &FilePath) {
    return insertObject(FileMap, FilePath);
  }
  template <class T> static inline SourceLocation getLocation(const T *N) {
    return N->getBeginLoc();
  }
  static inline SourceLocation getLocation(const VarDecl *VD) {
    return VD->getLocation();
  }
  static inline SourceLocation getLocation(const FunctionDecl *FD) {
    return FD->getLocation();
  }
  static inline SourceLocation getLocation(const FieldDecl *FD) {
    return FD->getLocation();
  }

  std::unordered_map<std::string, std::shared_ptr<DpctFileInfo>> FileMap;

  static std::string InRoot;
  static std::string CudaPath;
  static CompilerInstance *CI;
  static ASTContext *Context;
  static SourceManager *SM;
  static bool KeepOriginCode;
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
  const std::string &getSize() { return Size; }
  // Get actual size string according to template arguments list;
  void setTemplateList(const std::vector<TemplateArgumentInfo> &TemplateList);
};
// CtTypeInfo is basic class with info of element type, range, template info all
// get from type.
class CtTypeInfo {
public:
  // Array size will be folded, if exist.
  CtTypeInfo(const QualType &Ty);
  // If NeedSizeFold is true, array size will be folded, but orginal expression
  // will follow as comments. If NeedSizeFold is false, original size expression
  // will be the size string.
  CtTypeInfo(const TypeLoc &TL, bool NeedSizeFold = false);

  inline const std::string &getBaseName() { return BaseName; }

  inline size_t getDimension() { return Range.size(); }

  const std::string &getTemplateSpecializationName() {
    if (isTemplate() && TemplateType)
      return TemplateType->getTemplateSpecializationName();
    return getBaseName();
  }

  inline const std::string &getOrginalBaseType() { return OrginalBaseType; }
  // when there is no arguments, parameter MustArguments determine whether
  // parens will exist. Null string will be returned when MustArguments is
  // false, otherwise "()" will be returned.
  std::string getRangeArgument(const std::string &MemSize, bool MustArguments);

  inline bool isTemplate() { return IsTemplate; }
  inline bool isPointer() { return IsPointer; }
  inline bool isReference() { return IsReference; }
  inline void adjustAsMemType() {
    setPointerAsArray();
    removeQualifier();
  }
  void setTemplateType(const std::vector<TemplateArgumentInfo> &TA);

private:
  CtTypeInfo()
      : IsPointer(false), IsTemplate(false), TemplateIndex(0),
        TemplateList(nullptr) {}
  void setTypeInfo(const TypeLoc &TL, bool NeedSizeFold);
  void setTypeInfo(QualType Ty);

  // Get folded array size with original size expression following as comments.
  // For e.g.,
  // #define SIZE 24
  // dpct::shared_memory<int, 1>(24 /* SIZE */);
  inline std::string getFoldedArraySize(const ConstantArrayTypeLoc &TL) {
    return getFoldedArraySize(TL.getTypePtr()) + "/*" +
           getStmtSpelling(TL.getSizeExpr(), DpctGlobalInfo::getContext()) +
           "*/";
  }

  // Get folded array size only.
  inline std::string getFoldedArraySize(const ConstantArrayType *Ty) {
    return Ty->getSize().toString(10, false);
  }

  // Get original array size expression.
  std::string getUnfoldedArraySize(const ConstantArrayTypeLoc &TL);

  void setArrayInfo(QualType &Type);
  void setTemplateInfo(QualType &Type);
  void setPointerInfo(QualType &Type);
  void setReferenceInfo(QualType &Type);
  void setName(QualType &Type);

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
  std::string OrginalBaseType;
  std::vector<SizeInfo> Range;
  bool IsPointer;
  bool IsReference;
  bool IsTemplate;
  unsigned TemplateIndex;
  std::shared_ptr<CtTypeInfo> TemplateType;
  const std::vector<TemplateArgumentInfo> *TemplateList;
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
  };
  enum VarScope { Local = 0, Extern, Global };

  static std::shared_ptr<MemVarInfo> buildMemVarInfo(const VarDecl *Var);

  MemVarInfo(unsigned Offset, const std::string &FilePath, const VarDecl *Var)
      : VarInfo(Offset, FilePath, Var), Attr(getAttr(Var->getAttrs())),
        Scope((Var->isLexicallyWithinFunctionOrMethod())
                  ? (Var->isExternallyDeclarable() ? Extern : Local)
                  : Global),
        PointerAsArray(false) {
    setType(std::make_shared<CtTypeInfo>(Var->getTypeSourceInfo()->getTypeLoc(),
                                         isLocal()));
    if (getType()->isPointer()) {
      Attr = Device;
      getType()->adjustAsMemType();
      PointerAsArray = true;
    }
    if (Var->hasInit())
      setInitList(Var->getInit());
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

  std::string getAccessorDecl() {
    return buildString("auto ", getAccessorName(), " = ", getArgName(),
                       ".get_access(cgh);");
  }
  std::string getRangeDecl() {
    return buildString("auto ", getRangeName(), " = ", getArgName(),
                       ".get_range();");
  }
  llvm::raw_ostream &getFuncDecl(llvm::raw_ostream &OS) {
    return OS << getDpctAccessorType(false) << " " << getArgName();
  }
  llvm::raw_ostream &getFuncArg(llvm::raw_ostream &OS) {
    return OS << getArgName();
  }
  llvm::raw_ostream &getKernelArg(llvm::raw_ostream &OS) {
    OS << getDpctAccessorType(true) << "(" << getAccessorName();
    if (isShared())
      OS << ", " << getRangeName();
    return OS << ")";
  }

private:
  static VarAttrKind getAttr(const AttrVec &Attrs);

  void setInitList(const Expr *E) {
    if (auto Ctor = dyn_cast<CXXConstructExpr>(E)) {
      if (!Ctor->getNumArgs() || Ctor->getArg(0)->isDefaultArgument())
        return;
    }
    InitList = getStmtSpelling(E, DpctGlobalInfo::getContext());
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
    return buildString("(dpct::dpct_range<", getType()->getDimension(), ">",
                       getType()->getRangeArgument(MemSize, true),
                       ", " + InitList, ")");
  }
  const std::string &getMemoryAttr();

  std::string getDpctAccessorType(bool UsingTemplateName) {
    if (isExtern()) {
      return "dpct::dpct_accessor<dpct::byte_t, dpct::shared, 1>";
    } else {
      auto Type = getType();
      return buildString(
          "dpct::dpct_accessor<",
          (UsingTemplateName ? Type->getTemplateSpecializationName()
                             : Type->getBaseName()),
          ", ", getMemoryAttr(), ", ", Type->getDimension(), ">");
    }
  }
  std::string getAccessorName() {
    return buildString(getArgName(), "_acc" + getCTFixedSuffix());
  }
  std::string getRangeName() {
    return buildString(getArgName(), "_range" + getCTFixedSuffix());
  }
  std::string getArgName() {
    if (isExtern())
      return ExternVariableName;
    return getName();
  }

private:
  VarAttrKind Attr;
  VarScope Scope;
  bool PointerAsArray;
  std::string InitList;

  static const std::string ExternVariableName;
};

class TextureInfo {
  std::string DataType;
  int Dimension;

  const std::string FilePath;
  const unsigned Offset;
  const std::string Name;

  llvm::raw_ostream &getDecl(llvm::raw_ostream &OS, StringRef TemplateType) {
    return OS << "dpct::" << TemplateType << "<" << DataType << ", "
              << Dimension << "> " << Name;
  }

public:
  TextureInfo(unsigned Offset, const std::string &FilePath, const VarDecl *VD)
      : FilePath(FilePath), Offset(Offset), Name(VD->getName()) {
    if (auto D = dyn_cast<ClassTemplateSpecializationDecl>(
            VD->getType()->getAsCXXRecordDecl())) {
      auto &TemplateList = D->getTemplateInstantiationArgs();
      auto DataTy = TemplateList[0].getAsType();
      if (auto ET = dyn_cast<ElaboratedType>(DataTy))
        DataTy = ET->getNamedType();
      DataType = DataTy.getUnqualifiedType().getAsString(
          DpctGlobalInfo::getContext().getLangOpts());
      MapNames::replaceName(MapNames::TypeNamesMap, DataType);
      Dimension = TemplateList[1].getAsIntegral().getExtValue();
    }
  }
  inline llvm::raw_ostream &getFuncDecl(llvm::raw_ostream &OS) {
    return getDecl(OS, "dpct_image_accessor");
  }
  inline llvm::raw_ostream &getFuncArg(llvm::raw_ostream &OS) {
    return OS << Name;
  }
  inline llvm::raw_ostream &getKernelArg(llvm::raw_ostream &OS) {
    return OS << Name << "_acc";
  }

  std::string getDeclReplacement() {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    getDecl(OS, "dpct_image") << ";";
    return OS.str();
  }

  std::string getAccessorDecl() {
    return buildString("auto ", Name, "_acc = ", Name, ".get_access(cgh);");
  }

  inline unsigned getOffset() { return Offset; }
  inline const std::string &getName() { return Name; }
};

class TemplateArgumentInfo {
public:
  enum TemplateKind {
    Type = 0,
    String,
  };

  TemplateArgumentInfo(const QualType &QT) : Kind(Type) {
    Ty.LocalDecl = !QT->isElaboratedTypeSpecifier() &&
                   QT->hasUnnamedOrLocalType() &&
                   QT->getAsTagDecl()->getDeclContext()->isFunctionOrMethod();
    Ty.T = std::make_shared<CtTypeInfo>(QT);
    Str = Ty.T->getBaseName();
  }
  TemplateArgumentInfo(const Expr *Expr)
      : Kind(String),
        Str(getStmtSpelling(Expr, DpctGlobalInfo::getContext())) {}

  bool isType() { return Kind == Type; }
  std::shared_ptr<CtTypeInfo> getAsType() const {
    assert(Kind == Type);
    return Ty.T;
  }
  const std::string &getAsString() const { return Str; }

private:
  TemplateKind Kind;
  struct {
    bool LocalDecl;
    std::shared_ptr<CtTypeInfo> T;
  } Ty;
  std::string Str;
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
    merge(TextureMap, VarMap.TextureMap);
  }
  std::string getCallArguments(bool HasArgs) const {
    return getArgumentsOrParameters<CallArgument>(HasArgs);
  }
  std::string getDeclParam(bool HasParams) const {
    return getArgumentsOrParameters<DeclParameter>(HasParams);
  }
  std::string getKernelArguments(bool HasArgs) const {
    return getArgumentsOrParameters<KernelArgument>(HasArgs);
  }

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
      llvm_unreachable("unknow variable scope");
    }
  }

private:
  static void merge(MemVarInfoMap &Master, const MemVarInfoMap &Branch,
                    const std::vector<TemplateArgumentInfo> &TemplateArgs) {
    if (TemplateArgs.empty())
      return merge(Master, Branch);
    for (auto &VarInfoPair : Branch)
      Master.insert(VarInfoPair)
          .first->second->getType()
          ->setTemplateType(TemplateArgs);
  }
  template <class T>
  static void merge(GlobalMap<T> &Master, const GlobalMap<T> &Branch) {
    for (auto &VarInfoPair : Branch)
      Master.insert(VarInfoPair);
  }

  enum CallOrDecl {
    CallArgument = 0,
    KernelArgument,
    DeclParameter,
  };

  template <CallOrDecl COD>
  inline llvm::raw_ostream &getItem(llvm::raw_ostream &OS) const {
    return OS << DpctGlobalInfo::getItemName();
  }

  template <CallOrDecl COD>
  inline llvm::raw_ostream &getStream(llvm::raw_ostream &OS) const {
    return OS << DpctGlobalInfo::getStreamName();
  }

  template <CallOrDecl COD>
  std::string getArgumentsOrParameters(bool HasData) const {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    if (HasData)
      OS << ", ";
    if (hasItem())
      getItem<COD>(OS) << ", ";
    if (hasStream())
      getStream<COD>(OS) << ", ";
    if (!ExternVarMap.empty())
      GetArgOrParam<MemVarInfo, COD>()(OS, ExternVarMap.begin()->second)
          << ", ";
    getArgumentsOrParametersFromMap<MemVarInfo, COD>(OS, GlobalVarMap);
    getArgumentsOrParametersFromMap<MemVarInfo, COD>(OS, LocalVarMap);
    getArgumentsOrParametersFromMap<TextureInfo, COD>(OS, TextureMap);
    OS.flush();
    return Result.empty() ? Result : Result.erase(Result.size() - 2, 2);
  }

  template <class T, CallOrDecl COD>
  static void getArgumentsOrParametersFromMap(llvm::raw_ostream &OS,
                                              const GlobalMap<T> &VarMap) {
    for (auto VI : VarMap)
      GetArgOrParam<T, COD>()(OS, VI.second) << ", ";
  }

  template <class T, CallOrDecl COD> struct GetArgOrParam;
  template <class T> struct GetArgOrParam<T, DeclParameter> {
    llvm::raw_ostream &operator()(llvm::raw_ostream &OS, std::shared_ptr<T> V) {
      return V->getFuncDecl(OS);
    }
  };
  template <class T> struct GetArgOrParam<T, CallArgument> {
    llvm::raw_ostream &operator()(llvm::raw_ostream &OS, std::shared_ptr<T> V) {
      return V->getFuncArg(OS);
    }
  };
  template <class T> struct GetArgOrParam<T, KernelArgument> {
    llvm::raw_ostream &operator()(llvm::raw_ostream &OS, std::shared_ptr<T> V) {
      return V->getKernelArg(OS);
    }
  };

  bool HasItem, HasStream;
  MemVarInfoMap LocalVarMap;
  MemVarInfoMap GlobalVarMap;
  MemVarInfoMap ExternVarMap;
  GlobalMap<TextureInfo> TextureMap;
};

template <>
inline llvm::raw_ostream &
MemVarMap::getItem<MemVarMap::DeclParameter>(llvm::raw_ostream &OS) const {
  static std::string ItemParamDecl =
      "cl::sycl::nd_item<3> " + DpctGlobalInfo::getItemName();
  return OS << ItemParamDecl;
}

template <>
inline llvm::raw_ostream &
MemVarMap::getStream<MemVarMap::DeclParameter>(llvm::raw_ostream &OS) const {
  static std::string StreamParamDecl =
      "cl::sycl::stream " + DpctGlobalInfo::getStreamName();
  return OS << StreamParamDecl;
}

class ArgumentsInfo {
public:
  ArgumentsInfo() = default;

  inline bool empty() { return ArgInfos.empty(); }

  void buildArgsInfo(const CallExpr *CE) {
    KernelArgumentAnalysis Analysis;
    for (auto Arg : CE->arguments()) {
      KernelArgumentAnalysis Analysis;
      Analysis.analyze(Arg);
      ArgInfos.emplace_back(ArgInfo(Analysis, Arg));
    }
  }

private:
  struct ArgInfo {
    ArgInfo(KernelArgumentAnalysis &Analysis, const Expr* arg) {
      ArgString = Analysis.getReplacedString();
      isPointer = Analysis.isPointer;
      isRedeclareRequired = Analysis.isRedeclareRequired;
      TypeString = arg->getType().getAsString(
          DpctGlobalInfo::getContext().getPrintingPolicy());
    }

    const StringRef getArgString() {
      return ArgString;
    }

    std::string getTypeString() {
      return TypeString;
    }
    bool isPointer;
    bool isRedeclareRequired;
    std::string ArgString;
    std::string TypeString;
  };

  std::vector<ArgInfo> ArgInfos;

public:
  std::vector<ArgInfo>& getArgInfos() {
    return ArgInfos;
  }
};

// call function expression includes location, name, arguments num, template
// arguments and all function decls related to this call, also merges memory
// variable info of all related function decls.
class CallFunctionExpr {
public:
  CallFunctionExpr(unsigned Offset, const std::string &FilePathIn,
                   const CallExpr *CE)
      : FilePath(FilePathIn), BeginLoc(Offset),
        RParenLoc(DpctGlobalInfo::getSourceManager().getFileOffset(
            CE->getRParenLoc())) {}

  void buildCallExprInfo(const CallExpr *CE);

  inline const MemVarMap &getVarMap() { return VarMap; }

  void emplaceReplacement();
  inline bool hasArgs() { return HasArgs; }
  inline bool hasTemplateArgs() { return !TemplateArgs.empty(); }
  inline const std::string &getName() { return Name; }

  std::string getTemplateArguments(bool WithScalarWrapped = false);

  inline virtual std::string getExtraArguments() {
    return getVarMap().getCallArguments(hasArgs());
  }

protected:
  inline unsigned getBegin() { return BeginLoc; }
  inline const std::string &getFilePath() { return FilePath; }
  void buildInfo();

private:
  static std::string getName(const NamedDecl *D);
  void
  buildTemplateArguments(const llvm::ArrayRef<TemplateArgumentLoc> &ArgsList) {
    for (auto &Arg : ArgsList)
      addTemplateType(Arg);
  }
  void addTemplateType(const TemplateArgumentLoc &TA);

private:
  const std::string FilePath;
  unsigned BeginLoc;
  unsigned RParenLoc;
  std::string Name;
  std::vector<TemplateArgumentInfo> TemplateArgs;
  std::shared_ptr<DeviceFunctionInfo> FuncInfo;
  MemVarMap VarMap;
  bool HasArgs;
};

// device function declaration info includes location, name, and related
// DeviceFunctionInfo
class DeviceFunctionDecl {
public:
  DeviceFunctionDecl(unsigned Offset, const std::string &FilePathIn,
                     const FunctionDecl *FD)
      : Offset(Offset), FilePath(FilePathIn), ParamsNum(FD->param_size()),
        ReplaceOffset(0), ReplaceLength(0) {
    buildReplaceLocInfo(FD);
  }

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
      Info = std::make_shared<DeviceFunctionInfo>(List[0]->ParamsNum);
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

  inline void setFuncInfo(std::shared_ptr<DeviceFunctionInfo> Info) {
    FuncInfo = Info;
  }

  void buildReplaceLocInfo(const FunctionDecl *FD);

private:
  unsigned Offset;
  const std::string FilePath;
  unsigned ParamsNum;
  unsigned ReplaceOffset;
  unsigned ReplaceLength;
  std::shared_ptr<DeviceFunctionInfo> FuncInfo;
};

// device function info includes parameters num, memory variable and call
// expression in the function.
class DeviceFunctionInfo {
public:
  DeviceFunctionInfo(size_t ParamsNum) : IsBuilt(false), ParamsNum(ParamsNum) {}
  DeviceFunctionInfo(const FunctionDecl *Func)
      : DeviceFunctionInfo(Func->param_size()) {}

  inline void addCallee(const CallExpr *CE) {
    auto CallLocInfo = DpctGlobalInfo::getLocInfo(CE);
    insertObject(CallExprMap, CallLocInfo.second, CallLocInfo.first, CE)
        ->buildCallExprInfo(CE);
  }
  inline void addVar(std::shared_ptr<MemVarInfo> Var) { VarMap.addVar(Var); }
  inline void setItem() { VarMap.setItem(); }
  inline void setStream() { VarMap.setStream(); }
  inline void addTexture(std::shared_ptr<TextureInfo> Tex) {
    VarMap.addTexture(Tex);
  }
  inline const MemVarMap &getVarMap() { return VarMap; }

  void buildInfo();
  inline bool hasParams() { return ParamsNum != 0; }

  inline bool isBuilt() { return IsBuilt; }
  inline void setBuilt() { IsBuilt = true; }

  inline const std::string &getParameters() {
    if (!isBuilt())
      buildInfo();
    return Params;
  }

  void merge(std::shared_ptr<DeviceFunctionInfo> Other);

private:
  void mergeCallMap(const GlobalMap<CallFunctionExpr> &Other);

  bool IsBuilt;
  size_t ParamsNum;
  std::string Params;

  GlobalMap<CallFunctionExpr> CallExprMap;
  MemVarMap VarMap;
};

// kernel call info is specific CallFunctionExpr, which include info of kernel
// call.
class KernelCallExpr : public CallFunctionExpr {

  class FormatStmtBlock {
    const std::string &NL;
    std::string &Indent;
    llvm::raw_string_ostream &StmtStream;

  public:
    FormatStmtBlock(const std::string &NL, std::string &Indent,
                    llvm::raw_string_ostream &Stmts)
        : NL(NL), Indent(Indent), StmtStream(Stmts) {
      Indent += "  ";
    }
    FormatStmtBlock(const FormatStmtBlock &Parent)
        : FormatStmtBlock(Parent.NL, Parent.Indent, Parent.StmtStream) {}
    ~FormatStmtBlock() { Indent.erase(Indent.size() - 2); }
    template <class... Arguments> inline void pushStmt(Arguments &&... Args) {
      appendString(StmtStream, Indent, std::forward<Arguments>(Args)..., NL);
    }
  };

public:
  KernelCallExpr(unsigned Offset, const std::string &FilePath,
                 const CUDAKernelCallExpr *KernelCall)
      : CallFunctionExpr(Offset, FilePath, KernelCall), IsSync(false) {
    buildCallExprInfo(KernelCall);
    ArgsInfo.buildArgsInfo(KernelCall);
    buildKernelInfo(KernelCall);
  }

  void getAccessorDecl(FormatStmtBlock &Block);
  void buildInfo();
  inline std::string getExtraArguments() override {
    return getVarMap().getKernelArguments(hasArgs());
  }

  ArgumentsInfo& getArgsInfo() {
    return ArgsInfo;
  }

  std::string getReplacement();

  inline void setEvent(const std::string &E) { Event = E; }
  inline std::string getEvent() { return Event; }

  inline void setSync(bool Sync = true) { IsSync = Sync; }
  inline bool isSync() { return IsSync; }

private:
  void buildKernelInfo(const CUDAKernelCallExpr *KernelCall);
  void buildExecutionConfig(const CUDAKernelCallExpr *KernelCall);
  std::string analysisExcutionConfig(const Expr *Config);

  void getAccessorDecl(FormatStmtBlock &Block, MemVarInfo::VarScope Scope);
  void getAccessorDecl(FormatStmtBlock &Block, std::shared_ptr<MemVarInfo> VI);
  void getStreamDecl(FormatStmtBlock &Block);

  using StmtList = std::vector<std::string>;
  void buildKernelPointerArgsStmt(StmtList &BufferAndOffsets,
                                  StmtList &Accessors, StmtList &Redecls);
  void buildKernelPointerArgBufferAndOffsetStmt(const std::string &RefName,
                                                const std::string &ArgName,
                                                StmtList &Buffers);
  void buildKernelPointerArgAccessorStmt(const std::string &ArgName,
                                         StmtList &Accessors);
  void buildKernelPointerArgRedeclStmt(const std::string &ArgName,
                                       const std::string &TypeName,
                                       StmtList &Redecls);

  struct {
    std::string LocHash;
    std::string NL;
    std::string Indent;
  } LocInfo;
  struct {
    std::string NDSize;
    std::string WGSize;
    std::string ExternMemSize;
    std::string Stream;
  } ExecutionConfig;

  std::string Event;
  bool IsSync;
  ArgumentsInfo ArgsInfo;
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
    ArgumentAnalysis A(SizeExpression);
    A.analyze();
    Size = A.getReplacedString();
  }
  void setSizeExpr(const Expr *N, const Expr *ElemSize) {
    ArgumentAnalysis AN(N);
    ArgumentAnalysis AElemSize(ElemSize);
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

} // namespace dpct
} // namespace clang

#endif // CU2SYCL_ANALYSIS_INFO_H
