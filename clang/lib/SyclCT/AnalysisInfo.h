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

#include "TextModification.h"
#include "Utility.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"
#include <map>
#include <string>

namespace clang {
namespace syclct {

class KernelCallExpr;
class CallFunctionExpr;
class DeviceFunctionInfo;
class MemVarInfo;
class VarInfo;

template <class T> using GlobalMap = std::map<unsigned, std::shared_ptr<T>>;
using MemVarInfoMap = GlobalMap<MemVarInfo>;

template <class Obj, class Node>
std::shared_ptr<Obj> findNode(const Node *N, GlobalMap<Obj> &Map) {
  auto Itr = Map.find(N->getBeginLoc().getRawEncoding());
  if (Itr == Map.end())
    return std::shared_ptr<Obj>();
  return Itr->second;
}
template <class Obj, class Node>
std::shared_ptr<Obj> registerNode(const Node *N, GlobalMap<Obj> &Map) {
  if (auto Result = findNode(N, Map))
    return Result;
  return Map
      .insert(typename GlobalMap<Obj>::value_type(
          N->getBeginLoc().getRawEncoding(), std::make_shared<Obj>(N)))
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
  static void setContext(const ASTContext &C) {
    Context = &C;
    SM = &(Context->getSourceManager());
  }
  static const ASTContext &getContext() {
    assert(Context);
    return *Context;
  }
  static const SourceManager &getSourceManager() {
    assert(SM);
    return *SM;
  }
  static bool isInRoot(SourceLocation SL) {
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
#undef GLOBAL_TYPE

  template <class T> GlobalMap<T> &getMap() {
    llvm_unreachable("unexpected type of map");
  }
  void emplaceKernelAndDeviceReplacement(TransformSetTy &TS,
                                         StmtStringMap &SSM);

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

  static std::string InRoot;
  static const ASTContext *Context;
  static const SourceManager *SM;

  GlobalMap<MemVarInfo> MemVarMap;
  GlobalMap<DeviceFunctionInfo> FuncMap;
  GlobalMap<KernelCallExpr> KernelMap;
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

// TypeInfo is basic class with info of element type, range, template info all
// get from type.
class TemplateArgumentInfo;
class TypeInfo {
public:
  TypeInfo(const QualType &Type)
      : Type(Type), Template(false), TemplateIndex(0) {
    setArrayInfo();
    setTemplateInfo();
    setName();
  }
  const std::string &getName() { return Name; }
  size_t getDimension() { return Range.size(); }
  const std::vector<size_t> &getRange() { return Range; }

  std::string getAsTemplateArguments() {
    return "<" + getActualTypeName() + ", " + std::to_string(Range.size()) +
           ">";
  }

  std::string getActualTypeName() {
    if (isTemplate() && TemplateType)
      return TemplateType->getActualTypeName();
    return getName();
  }

  std::string getSyclctRange(const std::string &MemSize) {
    return "syclct::syclct_range<" + std::to_string(getDimension()) + ">" +
           getRangeArgument(MemSize);
  }

  bool isTemplate() { return Template; }
  void setTemplateType(const std::vector<TemplateArgumentInfo> &TA);

private:
  void setArrayInfo() {
    while (Type->isArrayType()) {
      if (auto ArrayType = dyn_cast<ConstantArrayType>(Type))
        Range.push_back((ArrayType)->getSize().getZExtValue());
      else
        Range.push_back(0);
      Type = Type->getAsArrayTypeUnsafe()->getElementType();
    }
  }
  void setTemplateInfo() {
    if (auto TemplateType =
            dyn_cast<TemplateTypeParmType>(Type->getCanonicalTypeInternal())) {
      Template = true;
      TemplateIndex = TemplateType->getIndex();
    }
  }
  void setName() {
    Name = Type.getAsString(
        SyclctGlobalInfo::getInstance().getContext().getPrintingPolicy());
    if (!isTemplate()) {
      auto Itr = MapNames::TypeNamesMap.find(Name);
      if (Itr != MapNames::TypeNamesMap.end())
        Name = Itr->second;
    }
  }
  std::string getRangeArgument(const std::string &MemSize) {
    std::string Arg = "(";
    for (auto R : Range)
      if (R)
        Arg += std::to_string(R) + ", ";
      else if (!MemSize.empty())
        Arg += MemSize + ", ";
      else
        llvm_unreachable("array size should not be zero");
    return (Arg.size() == 1) ? (Arg + ")")
                             : Arg.replace(Arg.size() - 2, 2, ")");
  }

private:
  QualType Type;
  std::string Name;
  std::vector<size_t> Range;
  bool Template;
  unsigned TemplateIndex;
  std::shared_ptr<TypeInfo> TemplateType;
};

// variable info includes name, type and location.
class VarInfo {
public:
  VarInfo(const VarDecl *Var)
      : Loc(Var->getBeginLoc().getRawEncoding()), Name(Var->getName().str()),
        Type(std::make_shared<TypeInfo>(Var->getType())) {}

  unsigned getLoc() { return Loc; }
  const std::string &getName() { return Name; }
  std::shared_ptr<TypeInfo> &getType() { return Type; }

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
        Scope((Var->isDefinedOutsideFunctionOrMethod())
                  ? ((Attr == Shared) ? Extern : Global)
                  : (Var->isExternallyDeclarable() ? Extern : Local)) {}

  VarAttrKind getAttr() { return Attr; }
  VarScope getScope() { return Scope; }
  bool isGlobal() { return Scope == Global; }
  bool isExtern() { return Scope == Extern; }
  bool isLocal() { return Scope == Local; }
  bool isShared() { return Attr == Shared; }

  std::string getDeclarationReplacement();
  std::string getMemoryDecl(const std::string &MemSize) {
    return getMemoryType() + " " + getArgName() + "(" +
           getInitArguments(MemSize) + ");";
  }
  std::string getMemoryDecl() {
    const static std::string NullString;
    return getMemoryDecl(NullString);
  }
  std::string getAccessorDecl() {
    return "auto " + getAccessorName() + " = " + getArgName() +
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

  std::string getMemoryType();
  std::string getInitArguments(const std::string &MemSize) {
    std::string ExtraString;
    if (isShared())
      ExtraString = ", cgh";
    return getType()->getSyclctRange(MemSize) + ExtraString;
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
             (UsingTemplateName ? Type->getActualTypeName() : Type->getName()) +
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

  static const std::string AccessorSuffix;
  static const std::string ExternVariableName;
};

class TemplateArgumentInfo {
public:
  enum TemplateKind {
    Type = 0,
    String,
  };

  TemplateArgumentInfo(const QualType &T) : Kind(Type) {
    TT.LocalDecl = !T->isElaboratedTypeSpecifier() &&
                   T->hasUnnamedOrLocalType() &&
                   T->getAsTagDecl()->getDeclContext()->isFunctionOrMethod();
    TT.TI = std::make_shared<TypeInfo>(T);
  }
  TemplateArgumentInfo(const Expr *Expr)
      : Kind(String), S(getStmtSpelling(Expr, SyclctGlobalInfo::getContext())) {
  }
  TemplateArgumentInfo(const llvm::APSInt &I)
      : Kind(String), S(I.toString(10)) {}

  std::shared_ptr<TypeInfo> getAsType() const {
    assert(Kind == Type);
    return TT.TI;
  }
  const std::string &getAsExprString() const {
    assert(Kind == String);
    return S;
  }
  std::string getAsCallArgument() const {
    switch (Kind) {
    case clang::syclct::TemplateArgumentInfo::Type:
      return (TT.LocalDecl ? "class " : "") + TT.TI->getName();
    case clang::syclct::TemplateArgumentInfo::String:
      return S;
    default:
      llvm_unreachable("unknow template type");
    }
  }

private:
  TemplateKind Kind;
  struct TemplateType {
    bool LocalDecl;
    std::shared_ptr<TypeInfo> TI;
  } TT;
  std::string S;
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

// call function expression includes location, name, arguments num, template
// arguments and all function decls related to this call, also merges memory
// variable info of all related function decls.
class CallFunctionExpr {
public:
  CallFunctionExpr(const CallExpr *CE)
      : Loc(CE->getBeginLoc().getRawEncoding()), RParenLoc(CE->getRParenLoc()),
        ArgsNum(CE->getNumArgs()), VarMap(std::make_shared<MemVarMap>()) {
    buildCallExprInfo(CE);
  }

  std::shared_ptr<MemVarMap> getVarMap() { return VarMap; }

  void buildInfo(TransformSetTy &TS);
  bool hasArgs() { return ArgsNum != 0; }
  const std::string &getName() { return Name; }

  std::string getTemplateArguments() {
    std::string Result = "<";
    for (auto &TA : TemplateArgs)
      Result += TA.getAsCallArgument() + ", ";
    return (Result.size() == 1) ? ""
                                : Result.replace(Result.size() - 2, 2, ">");
  }

  std::string getArguments() { return VarMap->getCallArguments(hasArgs()); }

private:
  void buildCallExprInfo(const CallExpr *CE);
  static std::string getName(const NamedDecl *D);
  void addTemplateType(const TemplateArgument &TA);
  void getTemplateArguments(const ArrayRef<TemplateArgumentLoc> &TemplateArgs);
  void getTemplateSpecializationInfo(const FunctionDecl *FD);
  void addTemplateFunctionDecl(const FunctionTemplateDecl *FTD);
  void addTemplateClassDecl(const ClassTemplateDecl *CTD);
  void addTemplateDecl(const TemplateDecl *TD);
  void addRecordDecl(const CXXRecordDecl *D);
  void addNamedDecl(const NamedDecl *ND);
  void addFunctionDecl(const FunctionDecl *FD);
  void addFunctionDecl(std::shared_ptr<DeviceFunctionInfo> FuncDecl);

  unsigned Loc;
  SourceLocation RParenLoc;
  size_t ArgsNum;
  std::string Name;
  std::vector<TemplateArgumentInfo> TemplateArgs;
  GlobalMap<DeviceFunctionInfo> FuncDeclMap;
  std::shared_ptr<MemVarMap> VarMap;
};

// device function info includes location,name,parameters num, memory variable
// and call expression in the function.
class DeviceFunctionInfo {
public:
  DeviceFunctionInfo(const FunctionDecl *Func)
      : Built(false), Loc(Func->getBeginLoc().getRawEncoding()),
        RParenLoc(getRParenLoc(Func)), ParamsNum(Func->getNumParams()),
        VarMap(std::make_shared<MemVarMap>()) {}
  unsigned getLoc() { return Loc; }
  void addCallee(const CallExpr *CE) { registerNode(CE, CallExprMap); }
  void addVar(std::shared_ptr<MemVarInfo> Var) {
    if (!VarMap)
      VarMap = std::make_shared<MemVarMap>();
    return VarMap->addVar(Var);
  }
  void setItem(){ getVarMap()->setItem(); }
  std::shared_ptr<MemVarMap> getVarMap() { return VarMap; }
  void setVarMap(std::shared_ptr<MemVarMap> MVM) { VarMap = MVM; }

  void buildInfo(TransformSetTy &TS);
  bool hasParams() { return ParamsNum != 0; }

  TextModification *getTextModification() {
    return new InsertText(RParenLoc, getParameters());
  }
  bool hasBuilt() { return Built; }
  void setBuilt() { Built = true; }

private:
  static SourceLocation getRParenLoc(const FunctionDecl *FD);
  std::string getParameters() { return VarMap->getDeclParam(hasParams()); }

  // make sure buildInfo(TransformSetTy &TS) only run once
  bool Built;

  unsigned Loc;
  SourceLocation RParenLoc;
  std::string Name;
  size_t ParamsNum;
  GlobalMap<CallFunctionExpr> CallExprMap;
  std::shared_ptr<MemVarMap> VarMap;
};

// kernel call info is specific CallFunctionExpr, which include info of kernel
// call.
class KernelCallExpr : public CallFunctionExpr {
public:
  KernelCallExpr(const CUDAKernelCallExpr *KernelCall)
      : CallFunctionExpr((const CallExpr *)KernelCall), KernelCall(KernelCall),
        ExternMemSize(getExternMemSize(KernelCall)) {}

  std::string getAccessorDecl(const std::string &Indent, const std::string &NL);
  std::string getArguments() {
    return getVarMap()->getKernelArguments(hasArgs());
  }

  const CUDAKernelCallExpr *getCallExpr() { return KernelCall; }

private:
  static std::string getExternMemSize(const CUDAKernelCallExpr *KernelCall);

  std::string getAccessorDecl(MemVarInfo::VarScope Scope,
                              const std::string &Indent,
                              const std::string &NL) {
    assert(Scope != MemVarInfo::Extern);
    std::string Result;
    for (auto VI : getVarMap()->getMap(Scope)) {
      if (Scope == MemVarInfo::Local) {
        Result += Indent + VI.second->getMemoryDecl() + NL;
      }
      Result += Indent + VI.second->getAccessorDecl() + NL;
    }
    return Result;
  }

  std::string getExternDecl(std::shared_ptr<MemVarInfo> VI) {
    assert(!ExternMemSize.empty());
    return VI->getMemoryDecl(ExternMemSize);
  }

  const CUDAKernelCallExpr *KernelCall;
  const std::string ExternMemSize;
};

} // namespace syclct
} // namespace clang

#endif // CU2SYCL_ANALYSIS_INFO_H
