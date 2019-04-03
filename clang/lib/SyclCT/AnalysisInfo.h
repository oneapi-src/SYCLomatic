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

#include "ExprAnalysis.h"
#include "TextModification.h"
#include "Utility.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"
#include <map>
#include <string>

namespace clang {
namespace syclct {

class CudaMallocInfo;
class KernelCallExpr;
class CallFunctionExpr;
class DeviceFunctionInfo;
class MemVarInfo;
class VarInfo;

template <class T> using GlobalMap = std::map<unsigned, std::shared_ptr<T>>;
using MemVarInfoMap = GlobalMap<MemVarInfo>;

template <class Node> inline unsigned getLocationId(const Node *N) {
  return N->getBeginLoc().getRawEncoding();
}
template <> inline unsigned getLocationId<VarDecl>(const VarDecl *N) {
  return N->getLocation().getRawEncoding();
}

template <class Obj, class Node>
std::shared_ptr<Obj> findNode(const Node *N, GlobalMap<Obj> &Map) {
  auto Itr = Map.find(getLocationId(N));
  if (Itr == Map.end())
    return std::shared_ptr<Obj>();
  return Itr->second;
}
template <class Obj, class Node>
std::shared_ptr<Obj> registerNode(const Node *N, GlobalMap<Obj> &Map) {
  if (auto Result = findNode(N, Map))
    return Result;
  return Map
      .insert(typename GlobalMap<Obj>::value_type(getLocationId(N),
                                                  std::make_shared<Obj>(N)))
      .first->second;
}

class SyclctGlobalInfo {
public:
  static SyclctGlobalInfo &getInstance() {
    static SyclctGlobalInfo Info;
    return Info;
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
    getInstance().clear();
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
  static bool isInRoot(const SourceLocation &SL) {
    std::string FilePath =
        getSourceManager().getFilename(getSourceManager().getExpansionLoc(SL));
    makeCanonical(FilePath);
    return isChildPath(InRoot, FilePath);
  }

#define GLOBAL_TYPE(TYPE, NODE_TYPE)                                           \
  std::shared_ptr<TYPE> find##TYPE(const NODE_TYPE *Node) {                    \
    return findObject<TYPE>(Node);                                             \
  }                                                                            \
  std::shared_ptr<TYPE> register##TYPE(const NODE_TYPE *Node) {                \
    return registerObject<TYPE>(Node);                                         \
  }

  GLOBAL_TYPE(MemVarInfo, VarDecl)
  GLOBAL_TYPE(DeviceFunctionInfo, FunctionDecl)
  GLOBAL_TYPE(KernelCallExpr, CUDAKernelCallExpr)
  GLOBAL_TYPE(CudaMallocInfo, VarDecl)
#undef GLOBAL_TYPE

  template <class T> GlobalMap<T> &getMap() {
    llvm_unreachable("unexpected type of map");
  }

  void emplaceKernelAndDeviceReplacement(TransformSetTy &TS);

  void registerCudaMalloc(const CallExpr *CE);
  std::shared_ptr<CudaMallocInfo> findCudaMalloc(const Expr *CE);

private:
  SyclctGlobalInfo() = default;

  SyclctGlobalInfo(const SyclctGlobalInfo &rhs) = delete;
  SyclctGlobalInfo(SyclctGlobalInfo &&rhs) = delete;
  SyclctGlobalInfo &operator=(const SyclctGlobalInfo &rhs) = delete;
  SyclctGlobalInfo &operator=(SyclctGlobalInfo &&rhs) = delete;

  template <class Obj, class Node>
  std::shared_ptr<Obj> findObject(const Node *N) {
    return findNode(N, getMap<Obj>());
  }
  template <class Obj, class Node>
  std::shared_ptr<Obj> registerObject(const Node *N) {
    return registerNode(N, getMap<Obj>());
  }
  void clear() {
    MemVarMap.clear();
    FuncMap.clear();
    KernelMap.clear();
    CudaMallocMap.clear();
  }

  static std::string InRoot;
  static ASTContext *Context;
  static SourceManager *SM;

  GlobalMap<MemVarInfo> MemVarMap;
  GlobalMap<DeviceFunctionInfo> FuncMap;
  GlobalMap<KernelCallExpr> KernelMap;
  GlobalMap<CudaMallocInfo> CudaMallocMap;
};
template <> inline GlobalMap<MemVarInfo> &SyclctGlobalInfo::getMap() {
  return MemVarMap;
}
template <> inline GlobalMap<DeviceFunctionInfo> &SyclctGlobalInfo::getMap() {
  return FuncMap;
}
template <> inline GlobalMap<KernelCallExpr> &SyclctGlobalInfo::getMap() {
  return KernelMap;
}
template <> inline GlobalMap<CudaMallocInfo> &SyclctGlobalInfo::getMap() {
  return CudaMallocMap;
}

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
  VarInfo(const VarDecl *Var)
      : Loc(getLocationId(Var)), Name(Var->getName().str()),
        Type(std::make_shared<TypeInfo>(Var->getType())) {}

  inline unsigned getLoc() { return Loc; }
  inline const std::string &getName() { return Name; }
  inline std::shared_ptr<TypeInfo> &getType() { return Type; }

private:
  unsigned Loc;
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

  MemVarInfo(const VarDecl *Var)
      : VarInfo(Var), Attr(getAttr(Var->getAttrs())),
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

  bool hasItem() { return Item; }
  bool hasExternShared() { return !ExternVarMap.empty(); }
  void setItem(bool hasItem = true) { Item = hasItem; }
  void addVar(std::shared_ptr<MemVarInfo> Var) {
    getMap(Var->getScope())
        .insert(MemVarInfoMap::value_type(Var->getLoc(), Var));
  }
  void merge(std::shared_ptr<MemVarMap> VarMap,
             const std::vector<TemplateArgumentInfo> &TemplateArgs) {
    if (VarMap) {
      setItem(hasItem() || VarMap->hasItem());
      merge(LocalVarMap, VarMap->LocalVarMap, TemplateArgs);
      merge(GlobalVarMap, VarMap->GlobalVarMap, TemplateArgs);
      merge(ExternVarMap, VarMap->ExternVarMap, TemplateArgs);
    }
  }
  std::string getCallArguments(bool HasArgs) {
    return getArgumentsOrParameters<CallArgument>(HasArgs);
  }
  std::string getDeclParam(bool HasParams) {
    return getArgumentsOrParameters<DeclParameter>(HasParams);
  }
  std::string getKernelArguments(bool HasArgs) {
    return getArgumentsOrParameters<KernelArgument>(HasArgs);
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

  template <CallOrDecl COD> inline const std::string &getItemName() {
    return SyclctGlobalInfo::getItemName();
  }

  template <CallOrDecl COD> std::string getArgumentsOrParameters(bool HasData) {
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
inline const std::string &MemVarMap::getItemName<MemVarMap::DeclParameter>() {
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
      Result += Arg.Arg + ", ";
    return Result.empty() ? Result : Result.erase(Result.size() - 2, 2);
  }
  void emplaceReplacements(TransformSetTy &TS) {
    for (auto &Arg : Arguments)
      if (Arg.Repl)
        TS.emplace_back(Arg.Repl);
  }

private:
  void buildArgsInfo(const CallExpr::const_arg_range &Args,
                     ExprAnalysis &Analysis) {
    for (auto Arg : Args) {
      Analysis.analysis(Arg);
      auto TM = Analysis.getReplacement();
      Arguments.emplace_back(Analysis.getReplacedString(), TM);
    }
  }

  struct ArgInfo {
    ArgInfo(const std::string &Arg, TextModification *TM)
        : Arg(Arg), Repl(TM) {}
    ArgInfo(const std::string &Arg) : ArgInfo(Arg, nullptr) {}
    std::string Arg;
    TextModification *Repl;
  };

  std::vector<ArgInfo> Arguments;
  std::vector<std::shared_ptr<VarInfo>> KernelArgs;
};

// call function expression includes location, name, arguments num, template
// arguments and all function decls related to this call, also merges memory
// variable info of all related function decls.
class CallFunctionExpr {
public:
  CallFunctionExpr(const CallExpr *CE)
      : Loc(getLocationId(CE)), RParenLoc(CE->getRParenLoc()), Args(CE),
        VarMap(std::make_shared<MemVarMap>()) {
    buildCallExprInfo(CE);
  }

  std::shared_ptr<MemVarMap> getVarMap() { return VarMap; }

  void buildInfo(TransformSetTy &TS);
  inline bool hasArgs() { return !Args.empty(); }
  inline bool hasTemplateArgs() { return !TemplateArgs.empty(); }
  inline const std::string &getName() { return Name; }

  std::string getTemplateArguments(bool WithScalarWrapped = false) {
    const static std::string ScalarWrapperPrefix = "syclct_kernel_scalar<",
                             ScalarWrapperSuffix = ">, ";
    std::string Result;
    for (auto &TA : TemplateArgs) {
      if (WithScalarWrapped && !TA.isType())
        Result += ScalarWrapperPrefix + TA.getAsString() + ScalarWrapperSuffix;
      else
        Result += TA.getAsString() + ", ";
    }
    return (Result.empty()) ? Result : Result.erase(Result.size() - 2);
  }

  virtual std::string getExtraArguments() {
    return VarMap->getCallArguments(hasArgs());
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

private:
  static std::string getName(const NamedDecl *D);
  void buildCallExprInfo(const CallExpr *CE);
  void
  buildTemplateArguments(const llvm::ArrayRef<TemplateArgumentLoc> &ArgsList) {
    for (auto &Arg : ArgsList)
      addTemplateType(Arg);
  }
  void addTemplateType(const TemplateArgumentLoc &TA);
  void addTemplateFunctionDecl(const FunctionTemplateDecl *FTD);
  void addTemplateClassDecl(const ClassTemplateDecl *CTD);
  void addTemplateDecl(const TemplateDecl *TD);
  void addRecordDecl(const CXXRecordDecl *D);
  void addNamedDecl(const NamedDecl *ND);
  void addFunctionDecl(const FunctionDecl *FD);
  void addFunctionDecl(std::shared_ptr<DeviceFunctionInfo> FuncDecl);

  unsigned Loc;
  SourceLocation RParenLoc;
  std::string Name;
  ArgumentsInfo Args;
  std::vector<TemplateArgumentInfo> TemplateArgs;
  GlobalMap<DeviceFunctionInfo> FuncDeclMap;
  std::shared_ptr<MemVarMap> VarMap;
};

// device function info includes location,name,parameters num, memory variable
// and call expression in the function.
class DeviceFunctionInfo {
public:
  DeviceFunctionInfo(const FunctionDecl *Func)
      : Built(false), FuncDecl(Func), Loc(getLocationId(Func)),
        ParamsNum(Func->getNumParams()), VarMap(std::make_shared<MemVarMap>()) {
    computeParenLoc();
  }
  unsigned getLoc() { return Loc; }
  void addCallee(const CallExpr *CE) { registerNode(CE, CallExprMap); }
  void addVar(std::shared_ptr<MemVarInfo> Var) {
    return getVarMap()->addVar(Var);
  }
  void setItem() { getVarMap()->setItem(); }
  std::shared_ptr<MemVarMap> getVarMap() {
    if (!VarMap)
      VarMap = std::make_shared<MemVarMap>();
    return VarMap;
  }
  void setVarMap(std::shared_ptr<MemVarMap> MVM) { VarMap = MVM; }

  void buildInfo(TransformSetTy &TS);
  bool hasParams() { return ParamsNum != 0; }

  TextModification *getTextModification();

  bool hasBuilt() { return Built; }
  void setBuilt() { Built = true; }

private:
  void computeParenLoc();
  std::string getParameters() { return VarMap->getDeclParam(hasParams()); }

  // make sure buildInfo(TransformSetTy &TS) only run once
  bool Built;

  const FunctionDecl *FuncDecl;
  unsigned Loc;
  std::string Name;
  size_t ParamsNum;
  GlobalMap<CallFunctionExpr> CallExprMap;
  std::shared_ptr<MemVarMap> VarMap;
  SourceLocation LParenLoc;
  SourceLocation RParenLoc;
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
  KernelCallExpr(const CUDAKernelCallExpr *KernelCall)
      : CallFunctionExpr((const CallExpr *)KernelCall), IsSync(false) {
    buildKernelInfo(KernelCall);
  }

  void getAccessorDecl(FormatStmtBlock &Block);
  inline std::string getExtraArguments() override {
    return getVarMap()->getKernelArguments(hasArgs());
  }

  std::string getReplacement();
  inline SourceLocation getBeginLoc() { return LocInfo.BeginLoc; }

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
    SourceLocation BeginLoc;
    std::string LocHash;
    std::string NL;
    std::string Indent;
  } LocInfo;
  struct {
    std::string NDSize;
    std::string WGSize;
    std::string ExternMemSize;
  } ExecutionConfig;

  bool IsSync;
};

class CudaMallocInfo {
public:
  CudaMallocInfo(const VarDecl *VD) : Name(VD->getName().str()) {}

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
