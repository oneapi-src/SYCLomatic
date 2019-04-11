//===--- AnalysisInfo.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_ANALYSIS_INFO_H
#define SYCLCT_ANALYSIS_INFO_H

#include "Debug.h"
#include "ExprAnalysis.h"
#include "ExtReplacements.h"

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

namespace clang {
namespace syclct {

class CudaMallocInfo;
class KernelCallExpr;
class CallFunctionExpr;
class DeviceFunctionDecl;
class MemVarInfo;
class VarInfo;

template <class T> using GlobalMap = std::map<unsigned, std::shared_ptr<T>>;
using MemVarInfoMap = GlobalMap<MemVarInfo>;

using ReplTy = std::map<std::string, tooling::Replacements>;

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

//                             SyclctGlobalInfo
//                                         |
//              --------------------------------------
//              |                          |                           |
//    SyclctFileInfo       SyclctFileInfo     ...
//              |
//           ----------------------------------------------------------
//           |                           |                         | |
//  MemVarInfo  DeviceFunctionDecl  KernelCallExpr  CudaMallocInfo
// Global Variable)                |   (inheriance from CallFunctionExpr)
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
class SyclctFileInfo {
public:
  SyclctFileInfo(const std::string &FilePathIn)
      : Repls(FilePath), FilePath(FilePathIn) {}
  template <class Obj> std::shared_ptr<Obj> findNode(unsigned Offset) {
    return findObject(getMap<Obj>(), Offset);
  }
  template <class Obj, class Node>
  std::shared_ptr<Obj> insertNode(unsigned Offset, const Node *N) {
    return insertObject(getMap<Obj>(), Offset, FilePath, N);
  }
  inline const std::string &getFilePath() { return FilePath; }

  // Build kernel and device function declaration replacements and store them.
  void buildReplacements();

  // Emplace stored replacements into replacement set.
  void emplaceReplacements(tooling::Replacements &ReplSet /*out*/);

  inline void addReplacement(std::shared_ptr<ExtReplacement> Repl) {
    Repls.addReplacement(Repl);
  }

private:
  template <class Obj> GlobalMap<Obj> &getMap() {
    syclct_unreachable("unknow map type");
  }
  void clear() {
    MemVarMap.clear();
    FuncMap.clear();
    KernelMap.clear();
    CudaMallocMap.clear();
  }

  GlobalMap<MemVarInfo> MemVarMap;
  GlobalMap<DeviceFunctionDecl> FuncMap;
  GlobalMap<KernelCallExpr> KernelMap;
  GlobalMap<CudaMallocInfo> CudaMallocMap;

  ExtReplacements Repls;

  std::string FilePath;
};
template <> inline GlobalMap<MemVarInfo> &SyclctFileInfo::getMap() {
  return MemVarMap;
}
template <> inline GlobalMap<DeviceFunctionDecl> &SyclctFileInfo::getMap() {
  return FuncMap;
}
template <> inline GlobalMap<KernelCallExpr> &SyclctFileInfo::getMap() {
  return KernelMap;
}
template <> inline GlobalMap<CudaMallocInfo> &SyclctFileInfo::getMap() {
  return CudaMallocMap;
}

class SyclctGlobalInfo {
public:
  static SyclctGlobalInfo &getInstance() {
    static SyclctGlobalInfo Info;
    return Info;
  }

  inline static bool isInRoot(SourceLocation SL) {
    return isInRoot(
        getSourceManager().getFilename(getSourceManager().getExpansionLoc(SL)));
  }
  static bool isInRoot(std::string &&FilePath) {
    makeCanonical(FilePath);
    return isChildPath(InRoot, FilePath);
  }
  static void setInRoot(const std::string &InRootPath) { InRoot = InRootPath; }
  static const std::string &getInRoot() {
    assert(!InRoot.empty());
    return InRoot;
  }
  static const std::string &getItemName() {
    const static std::string ItemName = "item_" + getInRootHash();
    return ItemName;
  }
  static const std::string &getInRootHash() {
    const static std::string Hash = getHashAsString(getInRoot()).substr(0, 6);
    return Hash;
  }
  static void setContext(ASTContext &C) {
    Context = &C;
    SM = &(Context->getSourceManager());
  }
  static ASTContext &getContext() {
    assert(Context);
    return *Context;
  }
  static SourceManager &getSourceManager() {
    assert(SM);
    return *SM;
  }

  template <class T>
  static inline std::pair<llvm::StringRef, unsigned> getLocInfo(const T *N) {
    auto Loc = getLocation(N);
    if (SM->isMacroArgExpansion(Loc))
      return getFilePathInfo(SM->getSpellingLoc(Loc));
    return getFilePathInfo(SM->getExpansionLoc(Loc));
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
#undef GLOBAL_TYPE

  // Build kernel and device function declaration replacements and store them.
  void buildReplacements() {
    for (auto &File : FileMap)
      File.second->buildReplacements();
  }

  // Emplace stored replacements into replacement set.
  void emplaceReplacements(ReplTy &ReplSets /*out*/) {
    for (auto &File : FileMap)
      File.second->emplaceReplacements(ReplSets[File.first]);
  }

  void insertCudaMalloc(const CallExpr *CE);
  std::shared_ptr<CudaMallocInfo> findCudaMalloc(const Expr *CE);
  void addReplacement(std::shared_ptr<ExtReplacement> Repl) {
    insertFile(Repl->getFilePath())->addReplacement(Repl);
  }

private:
  SyclctGlobalInfo() = default;

  SyclctGlobalInfo(const SyclctGlobalInfo &) = delete;
  SyclctGlobalInfo(SyclctGlobalInfo &&) = delete;
  SyclctGlobalInfo &operator=(const SyclctGlobalInfo &) = delete;
  SyclctGlobalInfo &operator=(SyclctGlobalInfo &&) = delete;

  // Find stored info by its corresponding AST node.
  // VarDecl=>MemVarInfo
  // FunctionDecl=>DeviceFunctionDecl
  // CUDAKernelCallExpr=>KernelCallExpr
  // VarDecl=>CudaMallocInfo
  template <class Info, class Node>
  inline std::shared_ptr<Info> findNode(const Node *N) {
    auto LocInfo = getLocInfo(N);
    if (isInRoot(LocInfo.first))
      return insertFile(LocInfo.first)->findNode<Info>(LocInfo.second);
    return false;
  }
  // Insert info if it doesn't exist.
  template <class Info, class Node>
  inline std::shared_ptr<Info> insertNode(const Node *N) {
    auto LocInfo = getLocInfo(N);
    return insertFile(LocInfo.first)->insertNode<Info>(LocInfo.second, N);
  }

  inline std::shared_ptr<SyclctFileInfo>
  insertFile(const std::string &FilePath) {
    return insertObject(FileMap, FilePath);
  }
  static std::pair<llvm::StringRef, unsigned> inline getFilePathInfo(
      const SourceLocation &SL) {
    auto LocInfo = SM->getDecomposedLoc(SL);
    return std::pair<llvm::StringRef, unsigned>(
        SM->getFileEntryForID(LocInfo.first)->getName(), LocInfo.second);
  }
  template <class T> static inline SourceLocation getLocation(const T *N) {
    return N->getBeginLoc();
  }
  static inline SourceLocation getLocation(const VarDecl *VD) {
    return VD->getLocation();
  }

  std::unordered_map<std::string, std::shared_ptr<SyclctFileInfo>> FileMap;

  static std::string InRoot;
  static ASTContext *Context;
  static SourceManager *SM;
};

class TemplateArgumentInfo;

// TypeInfo is basic class with info of element type, range, template info all
// get from type.
class TypeInfo {
public:
  TypeInfo(QualType Type);

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
  inline void adjustAsMemType() {
    setPointerAsArray();
    removeQualifier();
  }
  void setTemplateType(const std::vector<TemplateArgumentInfo> &TA);

private:
  void setArrayInfo(QualType &Type);
  void setTemplateInfo(QualType &Type);
  void setPointerInfo(QualType &Type);
  void setName(QualType &Type);
  void setPointerAsArray() {
    if (isPointer()) {
      IsPointer = false;
      Range.push_back(0);
    }
  }
  inline void removeQualifier() { BaseName = BaseNameWithoutQualifiers; }

private:
  std::string BaseName;
  std::string BaseNameWithoutQualifiers;
  std::string OrginalBaseType;
  std::vector<const ArrayType *> Range;
  bool IsPointer;
  bool IsTemplate;
  unsigned TemplateIndex;
  std::shared_ptr<TypeInfo> TemplateType;
  const std::vector<TemplateArgumentInfo> *TemplateList;
};

// variable info includes name, type and location.
class VarInfo {
public:
  VarInfo(unsigned Offset, const std::string &FilePathIn, const VarDecl *Var)
      : FilePath(FilePathIn), Offset(Offset), Name(Var->getName().str()),
        Type(std::make_shared<TypeInfo>(Var->getType())) {}

  inline const std::string &getFilePath() { return FilePath; }
  inline unsigned getOffset() { return Offset; }
  inline const std::string &getName() { return Name; }
  inline std::shared_ptr<TypeInfo> &getType() { return Type; }

private:
  const std::string FilePath;
  unsigned Offset;
  std::string Name;
  std::shared_ptr<TypeInfo> Type;
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
  std::string getMemoryDecl(const std::string &MemSize) {
    return getMemoryType() + " " + getArgName() +
           (PointerAsArray ? "" : getInitArguments(MemSize)) + ";";
  }
  std::string getMemoryDecl() {
    const static std::string NullString;
    return getMemoryDecl(NullString);
  }
  std::string getAccessorDecl(const std::string &MemSize) {
    std::string MemoryVar;
    if (isExtern())
      MemoryVar = getMemoryType() + getInitArguments(MemSize, true);
    else
      MemoryVar = getArgName();
    return "auto " + getAccessorName() + " = " + MemoryVar +
           ".get_access(cgh);";
  }
  std::string getFuncDecl() {
    return getSyclctAccessorType(false) + " " + getArgName();
  }
  std::string getFuncArg() { return getArgName(); }
  std::string getKernelArg() {
    return getSyclctAccessorType(true) + "(" + getAccessorName() + ")";
  }

private:
  static VarAttrKind getAttr(const AttrVec &Attrs);

  void setInitList(const Expr *E) {
    if (auto Ctor = dyn_cast<CXXConstructExpr>(E)) {
      if (!Ctor->getNumArgs() || Ctor->getArg(0)->isDefaultArgument())
        return;
    }
    InitList = getStmtSpelling(E, SyclctGlobalInfo::getContext());
  }

  std::string getMemoryType();
  inline std::string getMemoryType(const std::string &MemoryType,
                                   std::shared_ptr<TypeInfo> VarType) {
    return MemoryType + "<" + VarType->getTemplateSpecializationName() + ", " +
           std::to_string(VarType->getDimension()) + ">";
  }
  std::string getInitArguments(const std::string &MemSize,
                               bool MustArguments = false) {
    if (InitList.empty())
      return getType()->getRangeArgument(MemSize, MustArguments);
    return "(syclct::syclct_range<" +
           std::to_string(getType()->getDimension()) + ">" +
           getType()->getRangeArgument(MemSize, true) + ", " + InitList + ")";
  }
  const std::string &getMemoryAttr();

  std::string getSyclctAccessorType(bool UsingTemplateName) {
    if (isExtern()) {
      static std::string ExternType =
          "syclct::syclct_accessor<syclct::byte_t, syclct::shared, 1>";
      return ExternType;
    } else {
      auto Type = getType();
      return "syclct::syclct_accessor<" +
             (UsingTemplateName ? Type->getTemplateSpecializationName()
                                : Type->getBaseName()) +
             ", " + getMemoryAttr() + ", " +
             std::to_string(Type->getDimension()) + ">";
    }
  }
  std::string getAccessorName() { return getArgName() + AccessorSuffix; }
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

  static const std::string AccessorSuffix;
  static const std::string ExternVariableName;
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
    Ty.T = std::make_shared<TypeInfo>(QT);
    Str = Ty.T->getBaseName();
  }
  TemplateArgumentInfo(const Expr *Expr)
      : Kind(String),
        Str(getStmtSpelling(Expr, SyclctGlobalInfo::getContext())) {}

  bool isType() { return Kind == Type; }
  std::shared_ptr<TypeInfo> getAsType() const {
    assert(Kind == Type);
    return Ty.T;
  }
  const std::string &getAsString() const { return Str; }

private:
  TemplateKind Kind;
  struct {
    bool LocalDecl;
    std::shared_ptr<TypeInfo> T;
  } Ty;
  std::string Str;
};

// memory variable map includes memory variable used in __global__/__device__
// function and call expression.
class MemVarMap {
public:
  MemVarMap() : Item(false) {}

  bool hasItem() const { return Item; }
  bool hasExternShared() const { return !ExternVarMap.empty(); }
  void setItem(bool hasItem = true) { Item = hasItem; }
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
    merge(LocalVarMap, VarMap.LocalVarMap, TemplateArgs);
    merge(GlobalVarMap, VarMap.GlobalVarMap, TemplateArgs);
    merge(ExternVarMap, VarMap.ExternVarMap, TemplateArgs);
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

  MemVarInfoMap &getMap(MemVarInfo::VarScope Scope) {
    switch (Scope) {
    case clang::syclct::MemVarInfo::Local:
      return LocalVarMap;
    case clang::syclct::MemVarInfo::Extern:
      return ExternVarMap;
    case clang::syclct::MemVarInfo::Global:
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
  static void merge(MemVarInfoMap &Master, const MemVarInfoMap &Branch) {
    for (auto &VarInfoPair : Branch)
      Master.insert(VarInfoPair);
  }

  enum CallOrDecl {
    CallArgument = 0,
    KernelArgument,
    DeclParameter,
  };

  template <CallOrDecl COD> inline const std::string &getItemName() const {
    return SyclctGlobalInfo::getItemName();
  }

  template <CallOrDecl COD>
  std::string getArgumentsOrParameters(bool HasData) const {
    std::string Result;
    if (HasData)
      Result = ", ";
    if (hasItem())
      Result += getItemName<COD>() + ", ";
    if (!ExternVarMap.empty())
      Result +=
          getArgumentOrParameter<COD>(ExternVarMap.begin()->second) + ", ";
    Result += getArgumentsOrParametersFromMap<COD>(GlobalVarMap);
    Result += getArgumentsOrParametersFromMap<COD>(LocalVarMap);
    return Result.empty() ? Result : Result.erase(Result.size() - 2, 2);
  }

  template <CallOrDecl COD>
  static std::string
  getArgumentsOrParametersFromMap(const MemVarInfoMap &VarMap) {
    std::string Result;
    for (auto &VI : VarMap)
      Result += getArgumentOrParameter<COD>(VI.second) + ", ";
    return Result;
  }

  template <CallOrDecl COD>
  static std::string getArgumentOrParameter(std::shared_ptr<MemVarInfo> VI) {
    llvm_unreachable("not call or decl");
  }

  bool Item;
  MemVarInfoMap LocalVarMap;
  MemVarInfoMap GlobalVarMap;
  MemVarInfoMap ExternVarMap;
};
template <>
inline std::string MemVarMap::getArgumentOrParameter<MemVarMap::DeclParameter>(
    std::shared_ptr<MemVarInfo> VI) {
  return VI->getFuncDecl();
}
template <>
inline std::string MemVarMap::getArgumentOrParameter<MemVarMap::CallArgument>(
    std::shared_ptr<MemVarInfo> VI) {
  return VI->getFuncArg();
}
template <>
inline std::string MemVarMap::getArgumentOrParameter<MemVarMap::KernelArgument>(
    std::shared_ptr<MemVarInfo> VI) {
  return VI->getKernelArg();
}

template <>
inline const std::string &
MemVarMap::getItemName<MemVarMap::DeclParameter>() const {
  static std::string ItemName =
      "cl::sycl::nd_item<3> " + SyclctGlobalInfo::getItemName();
  return ItemName;
}

class ArgumentsInfo {
public:
  ArgumentsInfo(const CallExpr *C);

  inline bool empty() { return Arguments.empty(); }
  inline const std::vector<std::shared_ptr<VarInfo>> &getKernelPointerArgs() {
    return KernelArgs;
  }
  std::string getArguments() {
    std::string Result;
    for (auto &Arg : Arguments)
      Result += (Arg.getArgString() + ", ").str();
    return Result.empty() ? Result : Result.erase(Result.size() - 2, 2);
  }
  void emplaceReplacements() {
    auto &Global = SyclctGlobalInfo::getInstance();
    for (auto &Arg : Arguments)
      if (Arg.Repl)
        // TODO: Output debug info.
        Global.addReplacement(Arg.Repl);
  }

private:
  void buildArgsInfo(const CallExpr::const_arg_range &Args,
                     ArgumentAnalysis &Analysis) {
    for (auto Arg : Args) {
      Analysis.analysis(Arg);
      Arguments.emplace_back(Analysis);
    }
  }

  struct ArgInfo {
    ArgInfo(ArgumentAnalysis &Analysis) {
      auto TM = Analysis.getReplacement();
      if (TM)
        Repl = TM->getReplacement(SyclctGlobalInfo::getContext());
      else
        Arg = Analysis.getReplacedString();
    }
    const StringRef getArgString() {
      if (Repl)
        return Repl->getReplacementText();
      return Arg;
    }
    std::string Arg;
    std::shared_ptr<ExtReplacement> Repl;
  };

  std::vector<ArgInfo> Arguments;
  std::vector<std::shared_ptr<VarInfo>> KernelArgs;
};

class DeviceFunctionInfo;
// call function expression includes location, name, arguments num, template
// arguments and all function decls related to this call, also merges memory
// variable info of all related function decls.
class CallFunctionExpr {
public:
  CallFunctionExpr(unsigned Offset, const std::string &FilePathIn,
                   const CallExpr *CE)
      : FilePath(FilePathIn), BeginLoc(Offset),
        RParenLoc(SyclctGlobalInfo::getSourceManager().getFileOffset(
            CE->getRParenLoc())),
        Args(CE) {}

  void buildCallExprInfo(const CallExpr *CE);
  inline const MemVarMap &getVarMap() { return VarMap; }

  void emplaceReplacement();
  inline bool hasArgs() { return !Args.empty(); }
  inline bool hasTemplateArgs() { return !TemplateArgs.empty(); }
  inline const std::string &getName() { return Name; }

  std::string getTemplateArguments(bool WithScalarWrapped = false);

  inline virtual std::string getExtraArguments() {
    return getVarMap().getCallArguments(hasArgs());
  }
  inline std::string getOriginArguments() { return Args.getArguments(); }
  inline std::string getArguments() {
    return getOriginArguments() + getExtraArguments();
  }

protected:
  inline const std::vector<std::shared_ptr<VarInfo>> &
  getKernelPointerVarList() {
    return Args.getKernelPointerArgs();
  }
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
  ArgumentsInfo Args;
  std::vector<TemplateArgumentInfo> TemplateArgs;
  std::shared_ptr<DeviceFunctionInfo> FuncInfo;
  MemVarMap VarMap;
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
    if (auto D = SyclctGlobalInfo::getInstance().findDeviceFunctionDecl(FD))
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
    auto CallLocInfo = SyclctGlobalInfo::getLocInfo(CE);
    insertObject(CallExprMap, CallLocInfo.second, CallLocInfo.first, CE)
        ->buildCallExprInfo(CE);
  }
  inline void addVar(std::shared_ptr<MemVarInfo> Var) { VarMap.addVar(Var); }
  inline void setItem() { VarMap.setItem(); }
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
    std::string &Stmts;

  public:
    FormatStmtBlock(const std::string &NL, std::string &Indent,
                    std::string &Stmts)
        : NL(NL), Indent(Indent), Stmts(Stmts) {
      Indent += "  ";
    }
    FormatStmtBlock(const FormatStmtBlock &Parent)
        : FormatStmtBlock(Parent.NL, Parent.Indent, Parent.Stmts) {}
    ~FormatStmtBlock() { Indent.erase(Indent.size() - 2); }
    inline void pushStmt(const std::string &S) { Stmts += Indent + S + NL; }
  };

public:
  KernelCallExpr(unsigned Offset, const std::string &FilePath,
                 const CUDAKernelCallExpr *KernelCall)
      : CallFunctionExpr(Offset, FilePath, KernelCall), IsSync(false) {
    buildCallExprInfo(KernelCall);
    buildKernelInfo(KernelCall);
  }

  void getAccessorDecl(FormatStmtBlock &Block);
  void buildInfo();
  inline std::string getExtraArguments() override {
    return getVarMap().getKernelArguments(hasArgs());
  }

  std::string getReplacement();

  inline void setSync(bool Sync = true) { IsSync = Sync; }
  inline bool isSync() { return IsSync; }

private:
  void buildKernelInfo(const CUDAKernelCallExpr *KernelCall);
  void buildExecutionConfig(const CUDAKernelCallExpr *KernelCall);
  std::string analysisExcutionConfig(const Expr *Config);

  void getAccessorDecl(FormatStmtBlock &Block, MemVarInfo::VarScope Scope);

  using StmtList = std::vector<std::string>;
  void buildKernelPointerArgsStmt(StmtList &BufferAndOffsets,
                                  StmtList &Accessors, StmtList &Redecls);
  void buildKernelPointerArgBufferAndOffsetStmt(const std::string &ArgName,
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

  bool IsSync;
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
    A.analysis();
    Size = A.getReplacedString();
  }

  std::string getAssignArgs(const std::string &TypeName) {
    return Name + ", " + Size;
  }

private:
  std::string Size;
  std::string Name;
};

} // namespace syclct
} // namespace clang

#endif // CU2SYCL_ANALYSIS_INFO_H
