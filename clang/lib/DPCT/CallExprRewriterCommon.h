//===--------------- CallExprRewriterCommon.h -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_CALL_EXPR_REWRITER_COMMON_H
#define DPCT_CALL_EXPR_REWRITER_COMMON_H

#include "ASTTraversal.h"
#include "AnalysisInfo.h"
#include "BLASAPIMigration.h"
#include "CallExprRewriter.h"
#include "Config.h"
#include "ExprAnalysis.h"
#include "MapNames.h"
#include "Utility.h"
#include "ToolChains/Cuda.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include <cstdarg>

extern std::string DpctInstallPath; // Installation directory for this tool

using namespace clang::ast_matchers;
namespace clang {
namespace dpct {

/// Returns true if E contains one of the forms:
/// (blockDim/blockIdx/threadIdx/gridDim).(x/y/z) or warpSize
inline bool isContainTargetSpecialExpr(const Expr *E) {
  if (containIterationSpaceBuiltinVar(E) || containBuiltinWarpSize(E))
    return true;
  return false;
}

inline bool isArgMigratedToAccessor(const CallExpr *Call, unsigned Index) {
  if (auto DRE = dyn_cast<DeclRefExpr>(Call->getArg(Index)->IgnoreImpCasts())) {
    if (!DRE->getDecl()->hasAttrs())
      return false;
    for (auto A : DRE->getDecl()->getAttrs()) {
      auto K = A->getKind();
      if (K == attr::CUDAConstant || K == attr::CUDADevice ||
          K == attr::CUDAShared)
        return true;
    }
  }
  return false;
}

inline std::string getTypecastName(const CallExpr *Call) {
  auto Arg0TypeName = Call->getArg(0)->getType().getAsString();
  auto Arg1TypeName = Call->getArg(1)->getType().getAsString();
  auto RetTypeName = Call->getType().getAsString();
  bool B0 = isArgMigratedToAccessor(Call, 0);
  bool B1 = isArgMigratedToAccessor(Call, 1);
  if (B0 && !B1)
    return Arg1TypeName;
  if (!B0 && B1)
    return Arg0TypeName;
  if (B0 && B1)
    return RetTypeName;
  return {};
}

inline const Expr *getAddrOfedExpr(const Expr *E) {
  E = E->IgnoreImplicitAsWritten()->IgnoreParens();
  if (auto UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == clang::UO_Deref) {
      return UO->getSubExpr()->IgnoreImplicitAsWritten()->IgnoreParens();
    }
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (COCE->getOperator() == clang::OO_Star && COCE->getNumArgs() == 1) {
      return COCE->getArg(0)->IgnoreImplicitAsWritten()->IgnoreParens();
    }
  }
  return nullptr;
}

// In AST, &SubExpr could be recognized as UnaryOperator or CXXOperatorCallExpr.
// To get the SubExpr from the original Expr, both cases need to be handled.
inline const Expr *getDereferencedExpr(const Expr *E) {
  E = E->IgnoreImplicitAsWritten()->IgnoreParens();
  if (auto UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == clang::UO_AddrOf) {
      return UO->getSubExpr()->IgnoreImplicitAsWritten()->IgnoreParens();
    }
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (COCE->getOperator() == clang::OO_Amp && COCE->getNumArgs() == 1) {
      return COCE->getArg(0)->IgnoreImplicitAsWritten()->IgnoreParens();
    }
  }
  return nullptr;
}

class DerefStreamExpr {
  const Expr *E;

  template <class StreamT> void printDefaultQueue(StreamT &Stream) const {
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, E, HelperFuncType::HFT_DefaultQueue);
    Stream << "{{NEEDREPLACEQ" << Index << "}}";
  }

public:
  template <class StreamT>
  void printArg(StreamT &Stream, ArgumentAnalysis &A) const {
    if (isDefaultStream(E))
      printDefaultQueue(Stream);
    else
      DerefExpr(E).printArg(Stream, A);
  }
  template <class StreamT> void printMemberBase(StreamT &Stream) const {
    if (isDefaultStream(E)) {
      printDefaultQueue(Stream);
      Stream << ".";
    } else {
      DerefExpr(E).printMemberBase(Stream);
    }
  }

  template <class StreamT> void print(StreamT &Stream) const {
    if (isDefaultStream(E))
      printDefaultQueue(Stream);
    else
      DerefExpr(E).print(Stream);
  }

  DerefStreamExpr(const Expr *Expression) : E(Expression) {}
};

template <class SubExprT> class CastIfNotSameExprPrinter {
  std::string TypeInfo;
  SubExprT SubExpr;

public:
  CastIfNotSameExprPrinter(std::string &&T, SubExprT &&S)
      : TypeInfo(std::forward<std::string>(T)),
        SubExpr(std::forward<SubExprT>(S)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    const Expr *InputArg = SubExpr->IgnoreImpCasts();
    clang::QualType ArgType = InputArg->getType().getCanonicalType();
    ArgType.removeLocalFastQualifiers(clang::Qualifiers::CVRMask);
    if (ArgType.getAsString() != TypeInfo) {
      Stream << "(" << TypeInfo << ")";
    }
    dpct::print(Stream, SubExpr);
  }
};

class CastIfSpecialExpr {
  const Expr *Arg;
  const CallExpr *CE;

public:
  CastIfSpecialExpr(const Expr *Arg, const CallExpr *CE) : Arg(Arg), CE(CE) {}
  template <class StreamT> void print(StreamT &Stream) const {
    if (isContainTargetSpecialExpr(Arg)) {
      clang::QualType ArgType = Arg->getType().getCanonicalType();
      ArgType.removeLocalFastQualifiers(clang::Qualifiers::CVRMask);
      Stream << "(" << ArgType.getAsString() << ")";
      if (!dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts()) &&
          !dyn_cast<IntegerLiteral>(Arg->IgnoreImpCasts()) &&
          !dyn_cast<ParenExpr>(Arg->IgnoreImpCasts()) &&
          !dyn_cast<PseudoObjectExpr>(Arg->IgnoreImpCasts())) {
        Stream << "(";
        dpct::print(Stream, std::make_pair(CE, Arg));
        Stream << ")";
        return;
      }
    }
    dpct::print(Stream, std::make_pair(CE, Arg));
  }
};

template <class SubExprT> class DerefCastIfNeedExprPrinter {
  std::string TypeInfo;
  SubExprT SubExpr;

public:
  DerefCastIfNeedExprPrinter(std::string &&T, SubExprT &&S)
      : TypeInfo(std::forward<std::string>(T)),
        SubExpr(std::forward<SubExprT>(S)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    const Expr *InputArg = SubExpr->IgnoreImpCasts();
    const Expr *DerefInputArg = getDereferencedExpr(InputArg);
    if (DerefInputArg) {
      dpct::print(Stream, DerefInputArg);
    } else {
      clang::QualType ArgType = InputArg->getType().getCanonicalType();
      ArgType.removeLocalFastQualifiers(clang::Qualifiers::CVRMask);
      Stream << "*";
      if (ArgType.getAsString() != TypeInfo) {
        Stream << "(" << TypeInfo << ")";
      }
      dpct::print(Stream, SubExpr);
    }
  }
};

template <class SubExprT> class DoublePointerConstCastExprPrinter {
  std::string TypeInfo;
  SubExprT SubExpr;
  bool DoesBaseValueNeedConst;
  bool DoesFirstLevelPointerNeedConst;

public:
  DoublePointerConstCastExprPrinter(std::string &&T, SubExprT &&S,
                                    bool DoesBaseValueNeedConst,
                                    bool DoesFirstLevelPointerNeedConst)
      : TypeInfo(std::forward<std::string>(T)),
        SubExpr(std::forward<SubExprT>(S)),
        DoesBaseValueNeedConst(DoesBaseValueNeedConst),
        DoesFirstLevelPointerNeedConst(DoesFirstLevelPointerNeedConst) {}
  template <class StreamT> void print(StreamT &Stream) const {
    if (!checkConstQualifierInDoublePointerType(
            SubExpr, DoesBaseValueNeedConst, DoesFirstLevelPointerNeedConst)) {
      std::string CastType = TypeInfo + " " +
                             (DoesBaseValueNeedConst ? "const *" : "*") +
                             (DoesFirstLevelPointerNeedConst ? "const *" : "*");
      Stream << "const_cast<" << CastType << ">(";
      dpct::print(Stream, SubExpr);
      Stream << ")";
    } else {
      dpct::print(Stream, SubExpr);
    }
  }
};

inline std::function<std::string(const CallExpr *)>
makeCharPtrCreator() {
  return [=](const CallExpr *C) -> std::string {
    return "char *";
  };
}

inline std::function<DerefStreamExpr(const CallExpr *)>
makeDerefStreamExprCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> DerefStreamExpr {
    return DerefStreamExpr(C->getArg(Idx));
  };
}

inline std::function<AddrOfExpr(const CallExpr *)>
makeAddrOfExprCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> AddrOfExpr {
    return AddrOfExpr(C->getArg(Idx));
  };
}

inline std::function<DerefExpr(const CallExpr *)> makeDerefExprCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> DerefExpr {
    return DerefExpr(C->getArg(Idx));
  };
}

inline std::function<DerefExpr(const CallExpr *)> makeDerefExprCreator(
    std::function<std::pair<const CallExpr *, const Expr *>(const CallExpr *)>
        F) {
  return [=](const CallExpr *C) -> DerefExpr {
    return DerefExpr(F(C).second, F(C).first);
  };
}

inline std::function<BLASEnumExpr(const CallExpr *)>
makeBLASEnumCallArgCreator(unsigned Idx, BLASEnumExpr::BLASEnumType BET) {
  return [=](const CallExpr *C) -> BLASEnumExpr {
    return BLASEnumExpr::create(C->getArg(Idx), BET);
  };
}

inline std::function<const Expr *(const CallExpr *)> makeCallArgCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> const Expr * { return C->getArg(Idx); };
}

inline std::function<const StringRef(const CallExpr *)>
makeCallArgCreator(std::string Str) {
  return [=](const CallExpr *C) -> const StringRef { return StringRef(Str); };
}

inline std::function<bool(const CallExpr *)> makeBooleanCreator(bool B) {
  return [=](const CallExpr *C) -> bool { return B; };
}

inline std::function<std::pair<const CallExpr *, const Expr *>(const CallExpr *)>
makeCallArgCreatorWithCall(unsigned Idx) {
  return [=](const CallExpr *C) -> std::pair<const CallExpr *, const Expr *> {
    return std::pair<const CallExpr *, const Expr *>(C, C->getArg(Idx));
  };
}

inline const Expr *removeCStyleCast(const Expr *E) {
  if (auto CSCE = dyn_cast<ExplicitCastExpr>(E->IgnoreImplicitAsWritten())) {
    return CSCE->getSubExpr()->IgnoreImplicitAsWritten();
  } else {
    return E;
  }
}

// Prepare the arg for deref by removing the CStyleCast
// Should be used when the cast information is not relevant.
// e.g. migrating cudaMallocHost((void**)ptr, size) to
// *ptr = sycl::malloc_host<float>(size, q_ct1);
inline std::function<std::pair<const CallExpr *, const Expr *>(const CallExpr *)>
makeDerefArgCreatorWithCall(unsigned Idx) {
  return [=](const CallExpr *C) -> std::pair<const CallExpr *, const Expr *> {
    return std::pair<const CallExpr *, const Expr *>(
        C, removeCStyleCast(C->getArg(Idx)));
  };
}

template <class T1, class T2>
inline std::function<std::pair<T1, T2>(const CallExpr *)>
makeCombinedArg(std::function<T1(const CallExpr *)> Part1,
                std::function<T2(const CallExpr *)> Part2) {
  return [=](const CallExpr *C) -> std::pair<T1, T2> {
    return std::make_pair(Part1(C), Part2(C));
  };
}

inline std::function<std::vector<RenameWithSuffix>(const CallExpr *)>
makeStructDismantler(unsigned Idx, const std::vector<std::string> &Suffixes) {
  return [=](const CallExpr *C) -> std::vector<RenameWithSuffix> {
    std::vector<RenameWithSuffix> Ret;
    if (auto DRE = dyn_cast_or_null<DeclRefExpr>(
            getDereferencedExpr(C->getArg(Idx)))) {
      Ret.reserve(Suffixes.size());
      auto Origin = DRE->getDecl()->getName();
      std::transform(Suffixes.begin(), Suffixes.end(), std::back_inserter(Ret),
                     [&](StringRef Suffix) -> RenameWithSuffix {
                       return RenameWithSuffix(Origin, Suffix);
                     });
    }
    return Ret;
  };
}

inline std::function<std::string(const CallExpr *)>
makeExtendStr(unsigned Idx, const std::string Suffix) {
  return [=](const CallExpr *C) -> std::string {
    ArgumentAnalysis AA;
    AA.setCallSpelling(C);
    AA.analyze(C->getArg(Idx));
    std::string S = "(std::string(" + AA.getRewriteString() + ") + \"" +
                    Suffix + "\").c_str()";
    return S;
  };
}

inline std::string registerAndGetQueueStr(const CallExpr *C, std::string Prefix) {
  int Index = getPlaceholderIdx(C);
  if (Index == 0) {
    Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
  }
  buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
  return Prefix + "{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
}

inline std::function<std::string(const CallExpr *)> makeQueueStr() {
  return [=](const CallExpr *C) -> std::string {
    return registerAndGetQueueStr(C, "");
  };
}

inline std::function<std::string(const CallExpr *)> makeQueuePtrStr() {
  return [=](const CallExpr *C) -> std::string {
    return registerAndGetQueueStr(C, "&");
  };
}

inline std::function<std::string(const CallExpr *)> makeDeviceStr() {
  return [=](const CallExpr *C) -> std::string {
    int Index = getPlaceholderIdx(C);
    if (Index == 0) {
      Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    }

    buildTempVariableMap(Index, C, HelperFuncType::HFT_CurrentDevice);
    std::string S = "{{NEEDREPLACED" + std::to_string(Index) + "}}";
    return S;
  };
}

template <class BaseT, class... CallArgsT>
using MemberCallPrinterCreator =
    PrinterCreator<MemberCallPrinter<BaseT, StringRef, CallArgsT...>,
                   std::function<BaseT(const CallExpr *)>, bool, std::string,
                   std::function<CallArgsT(const CallExpr *)>...>;

template <class BaseT, class... CallArgsT>
inline std::function<
    MemberCallPrinter<BaseT, StringRef, CallArgsT...>(const CallExpr *)>
makeMemberCallCreator(std::function<BaseT(const CallExpr *)> BaseFunc,
                      bool IsArrow, std::string Member,
                      std::function<CallArgsT(const CallExpr *)>... Args) {
  return MemberCallPrinterCreator<BaseT, CallArgsT...>(BaseFunc, IsArrow,
                                                       Member, Args...);
}

template <class BaseT, class MemberT>
inline std::function<
    MemberCallPrinter<BaseT, MemberT>(const CallExpr *)>
makeMemberCallCreator(std::function<BaseT(const CallExpr *)> BaseFunc,
                      bool IsArrow,
                      std::function<MemberT(const CallExpr *)> Member) {

  return PrinterCreator<MemberCallPrinter<BaseT, MemberT>,
    std::function<BaseT(const CallExpr *)>, bool,
    std::function<MemberT(const CallExpr *)>>(BaseFunc, IsArrow,
                                              Member);
}


template <class... StmtT>
inline std::function<
    LambdaPrinter<StmtT...>(const CallExpr *)>
makeLambdaCreator(bool IsCaptureRef,
                      std::function<StmtT(const CallExpr *)>... Stmts) {
  return PrinterCreator<LambdaPrinter<StmtT...>, bool,
                        std::function<StmtT(const CallExpr *)>...>(
                        IsCaptureRef, Stmts...);
}

inline std::vector<TemplateArgumentInfo>
getTemplateArgsList(const CallExpr *C) {
  ArrayRef<TemplateArgumentLoc> TemplateArgsList;
  std::vector<TemplateArgumentInfo> Ret;
  auto Callee = C->getCallee()->IgnoreImplicitAsWritten();
  if (auto DRE = dyn_cast<DeclRefExpr>(Callee)) {
    TemplateArgsList = DRE->template_arguments();
  } else if (auto ULE = dyn_cast<UnresolvedLookupExpr>(Callee)) {
    TemplateArgsList = ULE->template_arguments();
  }
  for (const auto &Arg : TemplateArgsList) {
    Ret.emplace_back(Arg, C->getSourceRange());
  }
  return Ret;
}

inline std::function<TemplatedNamePrinter<
    StringRef, std::vector<TemplateArgumentInfo>>(const CallExpr *)>
makeTemplatedCalleeCreator(std::string CalleeName,
                           std::vector<size_t> Indexes) {
  return PrinterCreator<
      TemplatedNamePrinter<StringRef, std::vector<TemplateArgumentInfo>>,
      std::string,
      std::function<std::vector<TemplateArgumentInfo>(const CallExpr *)>>(
      CalleeName, [=](const CallExpr *C) -> std::vector<TemplateArgumentInfo> {
        std::vector<TemplateArgumentInfo> Ret;
        auto List = getTemplateArgsList(C);
        for (auto Idx : Indexes) {
          if (Idx < List.size()) {
            Ret.emplace_back(List[Idx]);
          }
        }
        return Ret;
      });
}

inline std::function<TemplateArgumentInfo(const CallExpr *)>
makeCallArgCreatorFromTemplateArg(unsigned Idx) {
  return [=](const CallExpr *CE) -> TemplateArgumentInfo {
    return getTemplateArgsList(CE)[Idx];
  };
}

template <class First>
inline void setTemplateArgumentInfo(const CallExpr *C,
                             std::vector<TemplateArgumentInfo> &Vec,
                             std::function<First(const CallExpr *)> F) {
  TemplateArgumentInfo TAI;
  TAI.setAsType(F(C));
  Vec.emplace_back(TAI);
}

template <class First, class... ArgsT>
inline void setTemplateArgumentInfo(const CallExpr *C,
                             std::vector<TemplateArgumentInfo> &Vec,
                             std::function<First(const CallExpr *)> F,
                             ArgsT... Args) {
  TemplateArgumentInfo TAI;
  TAI.setAsType(F(C));
  Vec.emplace_back(TAI);
  setTemplateArgumentInfo(C, Vec, Args...);
}

template <class... TemplateArgsT>
inline std::function<
    TemplatedNamePrinter<StringRef, TemplateArgsT...>(const CallExpr *)>
makeTemplatedName(StringRef TemplatedName,
                  std::function<TemplateArgsT(const CallExpr *)>... Args) {
  return PrinterCreator<TemplatedNamePrinter<StringRef, TemplateArgsT...>,
                        StringRef,
                        std::function<TemplateArgsT(const CallExpr *)>...>(
      TemplatedName, std::move(Args)...);
}

template <class... CallArgsT>
inline std::function<TemplatedNamePrinter<
    StringRef, std::vector<TemplateArgumentInfo>>(const CallExpr *)>
makeTemplatedCalleeWithArgsCreator(
    std::string Callee, std::function<CallArgsT(const CallExpr *)>... Args) {
  return PrinterCreator<
      TemplatedNamePrinter<StringRef, std::vector<TemplateArgumentInfo>>,
      std::string,
      std::function<std::vector<TemplateArgumentInfo>(const CallExpr *)>>(
      Callee, [=](const CallExpr *C) -> std::vector<TemplateArgumentInfo> {
        std::vector<TemplateArgumentInfo> Ret;
        setTemplateArgumentInfo(C, Ret, Args...);
        return Ret;
      });
}

template <BinaryOperatorKind Op, class LValueT, class RValueT>
inline std::function<BinaryOperatorPrinter<Op, LValueT, RValueT>(const CallExpr *)>
makeBinaryOperatorCreator(std::function<LValueT(const CallExpr *)> L,
                          std::function<RValueT(const CallExpr *)> R) {
  return PrinterCreator<BinaryOperatorPrinter<Op, LValueT, RValueT>,
                        std::function<LValueT(const CallExpr *)>,
                        std::function<RValueT(const CallExpr *)>>(std::move(L),
                                                                  std::move(R));
}

template <class CalleeT, class... CallArgsT>
inline std::function<CallExprPrinter<CalleeT, CallArgsT...>(const CallExpr *)>
makeCallExprCreator(std::function<CalleeT(const CallExpr *)> Callee,
                    std::function<CallArgsT(const CallExpr *)>... Args) {
  return PrinterCreator<CallExprPrinter<CalleeT, CallArgsT...>,
                        std::function<CalleeT(const CallExpr *)>,
                        std::function<CallArgsT(const CallExpr *)>...>(Callee,
                                                                       Args...);
}

template <class... CallArgsT>
inline std::function<CallExprPrinter<StringRef, CallArgsT...>(const CallExpr *)>
makeCallExprCreator(std::string Callee,
                    std::function<CallArgsT(const CallExpr *)>... Args) {
  return PrinterCreator<CallExprPrinter<StringRef, CallArgsT...>, std::string,
                        std::function<CallArgsT(const CallExpr *)>...>(Callee,
                                                                       Args...);
}

template < class BaseT, class ArgValueT>
inline std::function<
    ArraySubscriptExprPrinter<BaseT, ArgValueT>(const CallExpr *)>
makeArraySubscriptExprCreator(std::function<BaseT(const CallExpr *)> E,
                          std::function<ArgValueT(const CallExpr *)> I) {
  return PrinterCreator<ArraySubscriptExprPrinter<BaseT, ArgValueT>,
                        std::function<BaseT(const CallExpr *)>,
                        std::function<ArgValueT(const CallExpr *)>>(std::move(E),
                                                                  std::move(I));
}

inline std::function<std::string(const CallExpr *)>
makeFuncNameFromDevAttrCreator(unsigned idx) {
  return [=](const CallExpr *CE) -> std::string {
    auto Arg = CE->getArg(idx)->IgnoreImplicitAsWritten();
    if (auto DRE = dyn_cast<DeclRefExpr>(Arg)) {
      auto ArgName = DRE->getNameInfo().getAsString();
      auto Search = EnumConstantRule::EnumNamesMap.find(ArgName);
      if (Search != EnumConstantRule::EnumNamesMap.end()) {
        requestHelperFeatureForEnumNames(ArgName);
        return Search->second->NewName;
      }
    }
    return "";
  };
}
inline std::function<std::string(const CallExpr *)> getWorkGroupDim(unsigned index) {
  return [=](const CallExpr *C) {
    if (!dyn_cast<DeclRefExpr>(C->getArg(index)->IgnoreImplicitAsWritten()))
      return "";
    auto Arg = dyn_cast<DeclRefExpr>(C->getArg(index)->
                IgnoreImplicitAsWritten())->getNameInfo().getAsString();
    if (Arg == "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X")
      return "0";
    else if (Arg == "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y") {
      return "1";
    } else if (Arg == "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z") {
      return "2";
    }
    return "";
  };
}

inline std::function<std::string(const CallExpr *)> makeLiteral(std::string Str) {
  return [=](const CallExpr *) { return Str; };
}

inline std::function<bool(const CallExpr *)> SinCosPerfPred(){
  return [=](const CallExpr *) { return true; };
}

inline std::function<std::string(const CallExpr *)>
makeArgWithAddressSpaceCast(int ArgIdx, std::string Type) {
  return [=](const CallExpr *C) -> std::string {
    const Expr *E = C->getArg(ArgIdx);
    if (!E) {
      return "";
    }
    ExprAnalysis EA(E);
    std::string Result =
        MapNames::getClNamespace() + "address_space_cast<" +
        MapNames::getClNamespace() +
        "access::address_space::" + getAddressSpace(C, ArgIdx) + ", " +
        MapNames::getClNamespace() + "access::decorated::yes" + ", " +
        Type + ">(" + EA.getReplacedString() + ")";
    return Result;
  };
}

template <class BaseT, class MemberT>
inline std::function<MemberExprPrinter<BaseT, MemberT>(const CallExpr *)>
makeMemberExprCreator(std::function<BaseT(const CallExpr *)> Base, bool IsArrow,
                      std::function<MemberT(const CallExpr *)> Member) {
  return PrinterCreator<MemberExprPrinter<BaseT, MemberT>,
                        std::function<BaseT(const CallExpr *)>, bool,
                        std::function<MemberT(const CallExpr *)>>(Base, IsArrow,
                                                                  Member);
}

template <class BaseT, class MemberT>
inline std::function<StaticMemberExprPrinter<BaseT, MemberT>(const CallExpr *)>
makeStaticMemberExprCreator(std::function<BaseT(const CallExpr *)> Base,
                            std::function<MemberT(const CallExpr *)> Member) {
  return PrinterCreator<StaticMemberExprPrinter<BaseT, MemberT>,
                        std::function<BaseT(const CallExpr *)>,
                        std::function<MemberT(const CallExpr *)>>(Base, Member);
}

template <class TypeInfoT, class SubExprT>
inline std::function<CastExprPrinter<TypeInfoT, SubExprT>(const CallExpr *)>
makeCastExprCreator(std::function<TypeInfoT(const CallExpr *)> TypeInfo,
                    std::function<SubExprT(const CallExpr *)> Sub,
                    bool ExtraParen = false) {
  return PrinterCreator<CastExprPrinter<TypeInfoT, SubExprT>,
                        std::function<TypeInfoT(const CallExpr *)>,
                        std::function<SubExprT(const CallExpr *)>, bool>(
      TypeInfo, Sub, ExtraParen);
}

template <class SubExprT>
inline std::function<CastIfNotSameExprPrinter<SubExprT>(const CallExpr *)>
makeCastIfNotSameExprCreator(
    std::function<std::string(const CallExpr *)> TypeInfo,
    std::function<SubExprT(const CallExpr *)> Sub) {
  return PrinterCreator<CastIfNotSameExprPrinter<SubExprT>,
                        std::function<std::string(const CallExpr *)>,
                        std::function<SubExprT(const CallExpr *)>>(TypeInfo,
                                                                   Sub);
}

inline std::function<CastIfSpecialExpr(const CallExpr *)>
CastIfSpecialExprCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> CastIfSpecialExpr {
    return CastIfSpecialExpr(C->getArg(Idx), C);
  };
}

template <class SubExprT>
inline std::function<DerefCastIfNeedExprPrinter<SubExprT>(const CallExpr *)>
makeDerefCastIfNeedExprCreator(
    std::function<std::string(const CallExpr *)> TypeInfo,
    std::function<SubExprT(const CallExpr *)> Sub) {
  return PrinterCreator<DerefCastIfNeedExprPrinter<SubExprT>,
                        std::function<std::string(const CallExpr *)>,
                        std::function<SubExprT(const CallExpr *)>>(TypeInfo,
                                                                   Sub);
}

template <class SubExprT>
inline std::function<DoublePointerConstCastExprPrinter<SubExprT>(const CallExpr *)>
makeDoublePointerConstCastExprCreator(
    std::function<std::string(const CallExpr *)> TypeInfo,
    std::function<SubExprT(const CallExpr *)> Sub,
    std::function<bool(const CallExpr *)> DoesBaseValueNeedConst,
    std::function<bool(const CallExpr *)> DoesFirstLevelPointerNeedConst) {
  return PrinterCreator<DoublePointerConstCastExprPrinter<SubExprT>,
                        std::function<std::string(const CallExpr *)>,
                        std::function<SubExprT(const CallExpr *)>,
                        std::function<bool(const CallExpr *)>,
                        std::function<bool(const CallExpr *)>>(
      TypeInfo, Sub, DoesBaseValueNeedConst, DoesFirstLevelPointerNeedConst);
}

template <class... ArgsT>
inline std::function<NewExprPrinter<ArgsT...>(const CallExpr *)>
makeNewExprCreator(std::string TypeName,
                   std::function<ArgsT(const CallExpr *)>... Args) {
  return PrinterCreator<NewExprPrinter<ArgsT...>, std::string,
                        std::function<ArgsT(const CallExpr *)>...>(TypeName,
                                                                   Args...);
}

template <class SubExprT>
inline std::function<TypenameExprPrinter<SubExprT>(const CallExpr *)>
makeTypenameExprCreator(
                   std::function<SubExprT(const CallExpr *)> SubExpr) {
  return PrinterCreator<TypenameExprPrinter<SubExprT>,
                        std::function<SubExprT(const CallExpr *)>>(SubExpr);
}

template <class SubExprT>
inline std::function<ZeroInitializerPrinter<SubExprT>(const CallExpr *)>
makeZeroInitializerCreator(std::function<SubExprT(const CallExpr *)> SubExpr) {
  return PrinterCreator<ZeroInitializerPrinter<SubExprT>,
                        std::function<SubExprT(const CallExpr *)>>(SubExpr);
}

inline bool isCallAssigned(const CallExpr *C) { return isAssigned(C); }

inline bool isCallInRetStmt(const CallExpr *C) { return isInRetStmt(C); }

inline bool isCallAssignedOrInRetStmt(const CallExpr *C) {
  return isInRetStmt(C) || isAssigned(C);
}

template <unsigned int Idx>
inline unsigned int getSizeFromCallArg(const CallExpr *C, std::string &Var) {
  // Default sub group size align with cuda warp size
  if (Idx == UINT_MAX) {
    return 32;
  }
  auto SizeExpr = C->getArg(Idx);
  Expr::EvalResult Result;
  if (!SizeExpr->isValueDependent() &&
      SizeExpr->EvaluateAsInt(Result, DpctGlobalInfo::getContext())) {
    return Result.Val.getInt().getZExtValue();
  } else {
    ExprAnalysis EA(SizeExpr);
    Var = EA.getReplacedString();
    return UINT_MAX;
  }
}

/// If the input \p QT is a pointer type or an array type, this function will
/// return the deref-ed type. Otherwise an empty QualType object will be
/// returned. The caller needs to check if the return value is null using
/// isNull().
inline QualType DerefQualType(QualType QT) {
  QualType DerefQT;
  if (QT->isPointerType()) {
    DerefQT = QT->getPointeeType();
  } else if (QT->isArrayType()) {
    DerefQT = dyn_cast<ArrayType>(QT.getTypePtr())->getElementType();
  }
  return DerefQT;
}

// Get the replaced type of a function call argument
// For example, foo(x) where x is an int2, this function will return sycl::int2
inline std::function<std::string(const CallExpr *C)> getReplacedType(size_t Idx) {
  return [=](const CallExpr *C) -> std::string {
    if (Idx >= C->getNumArgs())
      return "";
    return DpctGlobalInfo::getReplacedTypeName(C->getArg(Idx)->getType());
  };
}

// Get the derefed type name of an arg while getDereferencedExpr is get the
// derefed expr.
inline std::function<std::string(const CallExpr *C)> getDerefedType(size_t Idx) {
  return [=](const CallExpr *C) -> std::string {
    if (Idx >= C->getNumArgs())
      return "";
    auto TE = removeCStyleCast(C->getArg(Idx));
    // Deref by removing the "&" of &SubExpr
    auto DE = getDereferencedExpr(TE);
    bool NeedDeref = true;
    // If getDereferencedExpr returns value, DE is the derefed TE.
    if (DE) {
      NeedDeref = false;
      TE = DE;
    }
    QualType DerefQT = TE->getType();
    while (const auto *ET = dyn_cast<ElaboratedType>(DerefQT)) {
      DerefQT = ET->getNamedType();
      if (const auto *TDT = dyn_cast<TypedefType>(DerefQT)) {
        auto *TDecl = TDT->getDecl();
        if (dpct::DpctGlobalInfo::isInCudaPath(TDecl->getLocation()))
          break;
        DerefQT = TDecl->getUnderlyingType();
      }
    }
    std::string TypeStr = DpctGlobalInfo::getReplacedTypeName(DerefQT);
    if (TypeStr == "<dependent type>" || TypeStr.empty()) {
      if (NeedDeref) {
        return "typename std::remove_pointer<decltype(" +
               ExprAnalysis::ref(TE) + ")>::type";
      } else {
        return "typename std::remove_reference<decltype(" +
               ExprAnalysis::ref(TE) + ")>::type";
      }
    }
    if (NeedDeref) {
      DerefQT = DerefQualType(DerefQT);
      if (DerefQT.isNull())
        return "";
      return DpctGlobalInfo::getReplacedTypeName(DerefQT);
    }
    return TypeStr;
  };
}

inline std::function<std::string(const CallExpr *)> getTemplateArg(size_t Idx) {
  return [=](const CallExpr *C) -> std::string {
    std::string TemplateArgStr = "";
    if (auto *Callee = dyn_cast<DeclRefExpr>(C->getCallee()->IgnoreParenImpCasts())) {
      auto TAL = Callee->template_arguments();
      if (TAL.size() <= Idx) {
        return TemplateArgStr;
      }
      const TemplateArgument &TA = TAL[Idx].getArgument();
      TemplateArgumentInfo TAI;
      switch (TA.getKind()) {
      case TemplateArgument::Integral:
        TAI.setAsNonType(TA.getAsIntegral());
        break;
      case TemplateArgument::Expression:
        TAI.setAsNonType(TA.getAsExpr());
        break;
      case TemplateArgument::Type:
        TAI.setAsType(TA.getAsType());
        break;
      default:
        break;
      }
      TemplateArgStr = TAI.getString();
    }
    return TemplateArgStr;
  };
}

// Can only be used if CheckCanUseTemplateMalloc is true.
inline std::function<std::string(const CallExpr *C)> getDoubleDerefedType(size_t Idx) {
  return [=](const CallExpr *C) -> std::string {
    if (Idx >= C->getNumArgs())
      return "";

    // Remove CStyleCast if any
    auto TE = removeCStyleCast(C->getArg(Idx));

    // Deref twice
    QualType DerefQT = TE->getType();
    DerefQT = DerefQualType(DerefQT);
    if (DerefQT.isNull())
      return "";
    DerefQT = DerefQualType(DerefQT);
    if (DerefQT.isNull())
      return "";
    std::string ReplType = DpctGlobalInfo::getReplacedTypeName(DerefQT);

    return ReplType;
  };
}

// Remove sizeof(T) if using template version.
// Can only be used if CheckCanUseTemplateMalloc is true.
inline std::function<std::string(const CallExpr *C)> getSizeForMalloc(size_t PtrIdx,
                                                               size_t SizeIdx) {
  return [=](const CallExpr *C) -> std::string {
    auto AllocatedExpr = C->getArg(PtrIdx);
    auto SizeExpr = C->getArg(SizeIdx);
    const Expr *AE = nullptr;
    if (auto CSCE = dyn_cast<CStyleCastExpr>(AllocatedExpr)) {
      AE = CSCE->getSubExpr()->IgnoreImplicitAsWritten();
    } else {
      AE = AllocatedExpr;
    }

    ArgumentAnalysis AA;
    AA.setCallSpelling(C);
    AA.analyze(SizeExpr);
    std::string OrginalStr =
        AA.getRewritePrefix() + AA.getRewriteString() + AA.getRewritePostfix();

    // Deref twice
    QualType DerefQT = AE->getType();
    DerefQT = DerefQualType(DerefQT);
    if (DerefQT.isNull())
      return OrginalStr;
    DerefQT = DerefQualType(DerefQT);
    if (DerefQT.isNull())
      return OrginalStr;

    std::string TypeStr = DpctGlobalInfo::getReplacedTypeName(DerefQT);

    auto BO = dyn_cast<BinaryOperator>(SizeExpr);
    if (BO && BO->getOpcode() == BinaryOperatorKind::BO_Mul) {
      std::string Repl;
      if (!isContainMacro(BO->getLHS()) &&
          isSameSizeofTypeWithTypeStr(BO->getLHS(), TypeStr)) {
        // case 1: sizeof(b) * a
        ArgumentAnalysis AASize;
        AASize.setCallSpelling(C);
        AASize.analyze(BO->getRHS());
        Repl = AASize.getRewritePrefix() + AASize.getRewriteString() +
               AASize.getRewritePostfix();
        return Repl;
      } else if (!isContainMacro(BO->getRHS()) &&
                 isSameSizeofTypeWithTypeStr(BO->getRHS(), TypeStr)) {
        // case 2: a * sizeof(b)
        ArgumentAnalysis AASize;
        AASize.setCallSpelling(C);
        AASize.analyze(BO->getLHS());
        Repl = AASize.getRewritePrefix() + AASize.getRewriteString() +
               AASize.getRewritePostfix();
        return Repl;
      } else {
        return OrginalStr;
      }
    } else {
      // case 3: sizeof(b)
      if (!isContainMacro(SizeExpr) &&
          isSameSizeofTypeWithTypeStr(SizeExpr, TypeStr)) {
        return "1";
      }
    }

    return OrginalStr;
  };
}

inline std::function<bool(const CallExpr *C)> checkIsUSM() {
  return [](const CallExpr *C) -> bool {
    return DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted;
  };
}

inline std::function<bool(const CallExpr *C)> checkIsUseNoQueueDevice() {
  return [](const CallExpr *C) -> bool {
    return DpctGlobalInfo::useNoQueueDevice();
  };
}

inline std::function<bool(const CallExpr *C)> checkArgSpelling(size_t index,
                                                        std::string str) {
  return [=](const CallExpr *C) -> bool {
    return getStmtSpelling(C->getArg(index)) == str;
  };
}

inline std::function<bool(const CallExpr *C)> checkIsCallExprOnly() {
  return [=](const CallExpr *C) -> bool {
    auto parentStmt = getParentStmt(C);
    if (parentStmt != nullptr && (dyn_cast<CompoundStmt>(parentStmt) ||
                          dyn_cast<ExprWithCleanups>(parentStmt)))
      return true;
    return false;
    };
}

inline std::function<bool(const CallExpr *C)> checkIsGetWorkGroupDim(size_t index) {
  return [=](const CallExpr *C) -> bool {
    if (getStmtSpelling(C->getArg(index)).
          find("CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_") != std::string::npos) {
      return true;
    }
    return false;
    };
}

inline std::function<bool(const CallExpr *C)> checkIsArgIntegerLiteral(size_t index) {
  return [=](const CallExpr *C) -> bool {
    auto Arg2Expr = C->getArg(index);
    if (auto NamedCaster = dyn_cast<ExplicitCastExpr>(Arg2Expr)) {
      if (NamedCaster->getTypeAsWritten()->isIntegerType()) {
        Arg2Expr = NamedCaster->getSubExpr();
      }
    }
    return Arg2Expr->getStmtClass() == Stmt::IntegerLiteralClass;
  };
}

template <size_t Idx>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createFactoryWithSubGroupSizeRequest(
    std::string NewFuncName,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Inner) {
  return std::make_pair(
      std::move(Inner.first),
      std::make_shared<RewriterFactoryWithSubGroupSize>(
          getSizeFromCallArg<Idx>, std::move(NewFuncName), Inner.second));
}

template <size_t Idx, class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createFactoryWithSubGroupSizeRequest(
    std::string NewFuncName,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Inner,
    T) {
  return createFactoryWithSubGroupSizeRequest<Idx>(std::move(NewFuncName),
                                                   std::move(Inner));
}

template <class... StmtPrinters>
inline std::shared_ptr<CallExprRewriterFactoryBase>
createMultiStmtsRewriterFactory(
    const std::string &SourceName, bool CheckAssigned, bool CheckInRetStmt,
    bool UseDdpctCheckError, bool ExtraParen,
    std::function<StmtPrinters(const CallExpr *)> &&...Creators) {
  return std::make_shared<ConditionalRewriterFactory>(
      (CheckAssigned && CheckInRetStmt)
          ? isCallAssignedOrInRetStmt
          : (CheckAssigned
                 ? isCallAssigned
                 : (CheckInRetStmt ? isCallInRetStmt
                                   : [](const CallExpr *C) { return true; })),
      std::make_shared<AssignableRewriterFactory>(
          std::make_shared<CallExprRewriterFactory<
              PrinterRewriter<CommaExprPrinter<StmtPrinters...>>,
              std::function<StmtPrinters(const CallExpr *)>...>>(SourceName,
                                                                 Creators...),
          CheckAssigned, CheckInRetStmt, UseDdpctCheckError, ExtraParen),
      std::make_shared<CallExprRewriterFactory<
          PrinterRewriter<MultiStmtsPrinter<StmtPrinters...>>,
          std::function<StmtPrinters(const CallExpr *)>...>>(SourceName,
                                                             Creators...));
}

/// Create UnaryOpRewriterFactory with given arguments.
/// \p SourceName the source callee name of original call expr.
/// \p ArgValueCreator use to get argument value from original call expr.
template <UnaryOperatorKind UO, class ArgValue>
inline std::shared_ptr<CallExprRewriterFactoryBase> createUnaryOpRewriterFactory(
    const std::string &SourceName,
    std::function<ArgValue(const CallExpr *)> &&ArgValueCreator) {
  return std::make_shared<
      CallExprRewriterFactory<UnaryOpRewriter<UO, ArgValue>,
                              std::function<ArgValue(const CallExpr *)>>>(
      SourceName,
      std::forward<std::function<ArgValue(const CallExpr *)>>(ArgValueCreator));
}

/// Create BinaryOpRewriterFactory with given arguments.
/// \p SourceName the source callee name of original call expr.
/// \p LValueCreator use to get lhs from original call expr.
/// \p RValueCreator use to get rhs from original call expr.
template <BinaryOperatorKind BO, class LValue, class RValue>
inline std::shared_ptr<CallExprRewriterFactoryBase> createBinaryOpRewriterFactory(
    const std::string &SourceName,
    std::function<LValue(const CallExpr *)> &&LValueCreator,
    std::function<RValue(const CallExpr *)> &&RValueCreator) {
  return std::make_shared<
      CallExprRewriterFactory<BinaryOpRewriter<BO, LValue, RValue>,
                              std::function<LValue(const CallExpr *)>,
                              std::function<RValue(const CallExpr *)>>>(
      SourceName,
      std::forward<std::function<LValue(const CallExpr *)>>(LValueCreator),
      std::forward<std::function<RValue(const CallExpr *)>>(RValueCreator));
}


template <class BaseT, class MemberT>
inline std::shared_ptr<CallExprRewriterFactoryBase> createMemberExprRewriterFactory(
    const std::string &SourceName,
    std::function<BaseT(const CallExpr *)> &&BaseCreator, bool IsArrow,
    std::function<MemberT(const CallExpr *)> &&MemberCreator) {
  return std::make_shared<
      CallExprRewriterFactory<MemberExprRewriter<BaseT, MemberT>,
                              std::function<BaseT(const CallExpr *)>, bool,
                              std::function<MemberT(const CallExpr *)>>>(
      SourceName,
      std::forward<std::function<BaseT(const CallExpr *)>>(BaseCreator),
      IsArrow,
      std::forward<std::function<MemberT(const CallExpr *)>>(MemberCreator));
}

inline std::shared_ptr<CallExprRewriterFactoryBase> createIfElseRewriterFactory(
    const std::string &SourceName,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        PredCreator,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        IfCreator,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        ElseCreator,
    int i) {
  return std::make_shared<CallExprRewriterFactory<
      IfElseRewriter, std::shared_ptr<CallExprRewriterFactoryBase>,
      std::shared_ptr<CallExprRewriterFactoryBase>,
      std::shared_ptr<CallExprRewriterFactoryBase>>>(
      SourceName, PredCreator.second, IfCreator.second, ElseCreator.second);
}

/// Create CallExprRewriterFactory with given arguments.
/// \p SourceName the source callee name of original call expr.
/// \p ArgsCreator use to get call args from original call expr.
template <class CalleeT, class... CallArgsT>
inline std::shared_ptr<CallExprRewriterFactoryBase> createCallExprRewriterFactory(
    const std::string &SourceName,
    std::function<CallExprPrinter<CalleeT, CallArgsT...>(const CallExpr *)>
        Args) {
  return std::make_shared<CallExprRewriterFactory<
      SimpleCallExprRewriter<CalleeT, CallArgsT...>,
      std::function<CallExprPrinter<CalleeT, CallArgsT...>(const CallExpr *)>>>(
      SourceName, Args);
}

/// Create TemplatedCallExprRewriterFactory with given arguments.
/// \p SourceName the source callee name of original call expr.
/// \p CalleeCreator use to get templated callee from original call expr.
/// \p ArgsCreator use to get call args from original call expr.
template <class... ArgsT>
inline std::shared_ptr<CallExprRewriterFactoryBase>
createTemplatedCallExprRewriterFactory(
    const std::string &SourceName,
    std::function<TemplatedNamePrinter<
        StringRef, std::vector<TemplateArgumentInfo>>(const CallExpr *)>
        CalleeCreator,
    std::function<ArgsT(const CallExpr *)>... ArgsCreator) {
  return std::make_shared<CallExprRewriterFactory<
      TemplatedCallExprRewriter<ArgsT...>,
      std::function<TemplatedNamePrinter<
          StringRef, std::vector<TemplateArgumentInfo>>(const CallExpr *)>,
      std::function<ArgsT(const CallExpr *)>...>>(
      SourceName,
      std::forward<std::function<TemplatedNamePrinter<
          StringRef, std::vector<TemplateArgumentInfo>>(const CallExpr *)>>(
          CalleeCreator),
      std::forward<std::function<ArgsT(const CallExpr *)>>(ArgsCreator)...);
}

/// Create MemberCallExprRewriterFactory with given arguments.
/// \p SourceName the source callee name of original call expr.
/// \p BaseCreator use to get base expr from original call expr.
/// \p IsArrow the member operator is arrow or dot as default.
/// \p ArgsCreator use to get call args from original call expr.
template <class BaseT, class... ArgsT>
inline std::shared_ptr<CallExprRewriterFactoryBase>
createMemberCallExprRewriterFactory(
    const std::string &SourceName,
    std::function<BaseT(const CallExpr *)> BaseCreator, bool IsArrow,
    std::string MemberName,
    std::function<ArgsT(const CallExpr *)>... ArgsCreator) {
  return std::make_shared<CallExprRewriterFactory<
      MemberCallExprRewriter<BaseT, ArgsT...>,
      std::function<BaseT(const CallExpr *)>, bool, std::string,
      std::function<ArgsT(const CallExpr *)>...>>(
      SourceName,
      std::forward<std::function<BaseT(const CallExpr *)>>(BaseCreator),
      IsArrow, MemberName,
      std::forward<std::function<ArgsT(const CallExpr *)>>(ArgsCreator)...);
}

template <class BaseT, class... ArgsT>
inline std::shared_ptr<CallExprRewriterFactoryBase>
createMemberCallExprRewriterFactory(
    const std::string &SourceName, BaseT BaseCreator, bool IsArrow,
    std::string MemberName,
    std::function<ArgsT(const CallExpr *)>... ArgsCreator) {
  return std::make_shared<CallExprRewriterFactory<
      MemberCallExprRewriter<BaseT, ArgsT...>, BaseT, bool, std::string,
      std::function<ArgsT(const CallExpr *)>...>>(
      SourceName, BaseCreator, IsArrow, MemberName,
      std::forward<std::function<ArgsT(const CallExpr *)>>(ArgsCreator)...);
}

template <class BaseT, class ArgsT>
inline std::shared_ptr<CallExprRewriterFactoryBase>
createArraySubscriptExprRewriterFactory(
    const std::string &SourceName,
    std::function<BaseT(const CallExpr *)> BaseCreator,
    std::function<ArgsT(const CallExpr *)> ArgsCreator) {
  return std::make_shared<
      CallExprRewriterFactory<ArraySubscriptRewriter<BaseT, ArgsT>,
                              std::function<BaseT(const CallExpr *)>,
                              std::function<ArgsT(const CallExpr *)>>>(
      SourceName, BaseCreator,
      std::forward<std::function<ArgsT(const CallExpr *)>>(ArgsCreator));
}

template <class... ArgsT>
inline std::shared_ptr<CallExprRewriterFactoryBase> createReportWarningRewriterFactory(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        Factory,
    const std::string &FuncName, Diagnostics MsgId, ArgsT... ArgsCreator) {
  return std::make_shared<ReportWarningRewriterFactory<ArgsT...>>(
      Factory.second, FuncName, MsgId, ArgsCreator...);
}

template <class ArgT>
inline std::shared_ptr<CallExprRewriterFactoryBase>
createDeleterCallExprRewriterFactory(
    const std::string &SourceName,
    std::function<ArgT(const CallExpr *)> &&ArgCreator) {
  return std::make_shared<CallExprRewriterFactory<
      DeleterCallExprRewriter<ArgT>, std::function<ArgT(const CallExpr *)>>>(
      SourceName,
      std::forward<std::function<ArgT(const CallExpr *)>>(ArgCreator));
}

template <class ArgT>
inline std::shared_ptr<CallExprRewriterFactoryBase> createToStringExprRewriterFactory(
    const std::string &SourceName,
    std::function<ArgT(const CallExpr *)> &&ArgCreator) {
  return std::make_shared<CallExprRewriterFactory<
      ToStringExprRewriter<ArgT>, std::function<ArgT(const CallExpr *)>>>(
      SourceName,
      std::forward<std::function<ArgT(const CallExpr *)>>(ArgCreator));
}

inline std::shared_ptr<CallExprRewriterFactoryBase>
createRemoveAPIRewriterFactory(const std::string &SourceName,
                               std::string Message = "") {
  return std::make_shared<
      CallExprRewriterFactory<RemoveAPIRewriter, std::string>>(SourceName,
                                                               Message);
}

/// Create AssignableRewriterFactory key-value pair with inner key-value.
/// If the call expr's return value is used, will insert around "(" and ", 0)".
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createAssignableFactory(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input) {
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<AssignableRewriterFactory>(Input.second));
}
/// Create AssignableRewriterFactory key-value pair with inner key-value.
/// If the call expr's return value is used, will insert around "(" and ", 0)".
template <class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createAssignableFactory(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input,
    T) {
  return createAssignableFactory(std::move(Input));
}

inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createAssignableFactoryWithExtraParen(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input) {
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(Input.first), std::make_shared<AssignableRewriterFactory>(
                                  Input.second, true, false, true, true));
}

template <class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createAssignableFactoryWithExtraParen(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input,
    T) {
  return createAssignableFactoryWithExtraParen(std::move(Input));
}

inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createInsertAroundFactory(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input,
    std::string &&Prefix, std::string &&Suffix) {
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<InsertAroundRewriterFactory>(
          Input.second, std::move(Prefix), std::move(Suffix)));
}

/// Create RewriterFactoryWithFeatureRequest key-value pair with inner
/// key-value. Will call requestFeature when used to create CallExprRewriter.
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input) {
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<RewriterFactoryWithFeatureRequest>(Feature,
                                                          Input.second));
}
/// Create RewriterFactoryWithFeatureRequest key-value pair with inner
/// key-value. Will call requestFeature when used to create CallExprRewriter.
template <class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input,
    T) {
  return createFeatureRequestFactory(Feature, std::move(Input));
}


/// Create RewriterFactoryWithHeaderFile key-value pair with inner
/// key-value. Will call insertHeader when used to create CallExprRewriter.
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createInsertHeaderFactory(
    HeaderType Header,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input) {
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<RewriterFactoryWithHeaderFile>(Header,
                                                          Input.second));
}
/// Create RewriterFactoryWithHeaderFile key-value pair with inner
/// key-value. Will call insertHeader when used to create CallExprRewriter.
template <class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createInsertHeaderFactory(
    HeaderType Header,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Input,
    T) {
  return createInsertHeaderFactory(Header, std::move(Input));
}

/// Create ConditonalRewriterFactory key-value pair with two key-value
/// candidates and predicate.
/// If predicate result is true, \p First will be used, else \p Second will be
/// used.
/// Also check the key of \p First and \p Second is same in debug build.
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createConditionalFactory(
    std::function<bool(const CallExpr *)> Pred,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&First,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Second) {
#ifdef DPCT_DEBUG_BUILD
  if (First.first != Second.first) {
    llvm::errs() << "Condtional factory has different name: [" << First.first
                 << "] : [" << Second.first << "]\n";
    assert(0 && "Condtional factory has different name");
  }
#endif // DPCT_DEBUG_BUILD
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(First.first), std::make_shared<ConditionalRewriterFactory>(
                                  Pred, First.second, Second.second));
}

/// Create ConditonalRewriterFactory key-value pair with two key-value
/// candidates and predicate.
/// If predicate result is true, \p First will be used, else \p Second will be
/// used.
/// Also check the key of \p First and \p Second is same in debug build.
template <class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createConditionalFactory(
    std::function<bool(const CallExpr *)> Pred,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&First,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Second,
    T) {
  return createConditionalFactory(std::move(Pred), std::move(First),
                                  std::move(Second));
}

/// Create MathSpecificElseEmuRewriterFactory key-value pair with one
/// key-value candidates and predicate. If predicate result is true, \p First
/// will be used.
template <class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createMathSpecificElseEmuRewriterFactory(
    std::function<bool(const CallExpr *)> Pred,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&First,
    T) {
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      std::move(First.first),
      std::make_shared<MathSpecificElseEmuRewriterFactory>(Pred, First.second));
}

template <typename T>
inline std::pair<std::string, CaseRewriterFactory::CaseT>
createCase(
    CaseRewriterFactory::PredT pred,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>&& entry, T) {
  return {std::move(entry.first),
          {std::move(pred), std::move(entry.second)}};
}

template <class... Ts>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createCaseRewriterFactory(
    std::pair<std::string, CaseRewriterFactory::CaseT> first,
    Ts... rest) {
  return {std::move(first.first),
          std::make_shared<CaseRewriterFactory>(
               std::move(first.second),
               std::move(rest.second)...)};
}

inline std::function<bool(const CallExpr *)> makePointerChecker(unsigned Idx) {
  return [=](const CallExpr *C) -> bool {
    return C->getArg(Idx)->getType()->isPointerType();
  };
}

/// Create rewriter factory for migration of cudaBindTexture APIs.
/// \p StartIdx is represent the first available argument's index.
/// For cudaBindTexture and cudaBindTexture2D, it is 1.
/// For  cudaBindTextureToArray, it is 0.
/// The first predicate will check the \p StartIdx 'th argument whether is
/// pointer. If it is true, the call expr will be migrated to member call expr.
/// e.g.: cudaBindTexture(0, &tex, data, &desc, size) -> tex.attach(data, size,
/// desc) with template arguments: <1, 2>. Else will check the second predicate:
/// If \p Start + 2 'th argument's type whether is cudaChannelFormatDesc.
/// If it is true, e.g.: cudaBindTexture(0, tex, data, desc, size) ->
/// tex.attach(data, size, desc).
/// Else, e.g.: cudaBindTexture(0, tex, data, size) ->tex.attach(data, size,
/// desc).
template <size_t StartIdx, size_t... Idx>
inline std::shared_ptr<CallExprRewriterFactoryBase>
createBindTextureRewriterFactory(const std::string &Source) {
  std::function<bool(const CallExpr *)> TypeChecker =
      [=](const CallExpr *C) -> bool {
    if (C->getNumArgs() > StartIdx + 2)
      return DpctGlobalInfo::getUnqualifiedTypeName(
                 C->getArg(StartIdx + 2)->getType()) == "cudaChannelFormatDesc";
    return false;
  };

  return std::make_shared<ConditionalRewriterFactory>(
      makePointerChecker(StartIdx + 0),
      createMemberCallExprRewriterFactory(
          Source, makeDerefExprCreator(StartIdx + 0), true, "attach",
          makeCallArgCreator(StartIdx + 1),
          makeCallArgCreator(StartIdx + Idx + 1)...,
          makeDerefExprCreator(StartIdx + 2)),
      std::make_shared<ConditionalRewriterFactory>(
          TypeChecker,
          createMemberCallExprRewriterFactory(
              Source, makeCallArgCreatorWithCall(StartIdx + 0), false, "attach",
              makeCallArgCreatorWithCall(StartIdx + 1),
              makeCallArgCreatorWithCall(StartIdx + Idx + 1)...,
              makeCallArgCreatorWithCall(StartIdx + 2)),
          createMemberCallExprRewriterFactory(
              Source, makeCallArgCreatorWithCall(StartIdx + 0), false, "attach",
              makeCallArgCreatorWithCall(StartIdx + 1),
              makeCallArgCreatorWithCall(StartIdx + Idx)...)));
}

template <size_t... Idx>
class TextureReadRewriterFactory : public CallExprRewriterFactoryBase {
  std::string Source;
  int TexType;

  template <class BaseT>
  std::shared_ptr<CallExprRewriter>
  createRewriter(const CallExpr *C, bool RetAssign, BaseT Base) const {
    const static std::string MemberName = "read";
    using ReaderPrinter = decltype(makeMemberCallCreator(
        std::declval<std::function<BaseT(const CallExpr *)>>(), false,
        MemberName, makeCallArgCreatorWithCall(Idx)...)(C));
    if (RetAssign) {
      return std::make_shared<PrinterRewriter<
          BinaryOperatorPrinter<BO_Assign, DerefExpr, ReaderPrinter>>>(
          C, Source, DerefExpr(C->getArg(0), C),
          ReaderPrinter(std::move(Base), false, MemberName,
                        std::make_pair(C, C->getArg(Idx + 1))...));
    }
    return std::make_shared<PrinterRewriter<ReaderPrinter>>(
        C, Source, Base, false, MemberName,
        std::make_pair(C, C->getArg(Idx))...);
  }

public:
  TextureReadRewriterFactory(std::string Name, int Tex)
      : Source(std::move(Name)), TexType(Tex) {}
  std::shared_ptr<CallExprRewriter>
  create(const CallExpr *Call) const override {
    const Expr *SourceExpr = Call->getArg(0);
    unsigned SourceIdx = 0;
    QualType TargetType = Call->getType();
    StringRef SourceName;
    bool RetAssign = false;
    if (SourceExpr->getType()->isPointerType()) {
      TargetType = SourceExpr->getType()->getPointeeType();
      SourceExpr = Call->getArg(1);
      SourceIdx = 1;
      RetAssign = true;
      if (auto UO = dyn_cast<UnaryOperator>(SourceExpr)) {
        if (UO->getOpcode() == UnaryOperator::Opcode::UO_AddrOf) {
          SourceExpr = UO->getSubExpr();
        }
      }
    }
    SourceExpr = SourceExpr->IgnoreImpCasts();
    if (auto FD = DpctGlobalInfo::getParentFunction(Call)) {
      auto FuncInfo = DeviceFunctionDecl::LinkRedecls(FD);
      if (FuncInfo) {
        auto CallInfo = FuncInfo->addCallee(Call);
        if (auto ME = dyn_cast<MemberExpr>(SourceExpr)) {
          auto MemberInfo =
              CallInfo->addStructureTextureObjectArg(SourceIdx, ME, false);
          if (MemberInfo) {
            FuncInfo->addTexture(MemberInfo);
            MemberInfo->setType(
                DpctGlobalInfo::getUnqualifiedTypeName(TargetType), TexType);
            SourceName = MemberInfo->getName();
            return createRewriter(Call, RetAssign, SourceName);
          }
        } else if (auto DRE = dyn_cast<DeclRefExpr>(SourceExpr)) {
          auto TexInfo = CallInfo->addTextureObjectArg(SourceIdx, DRE, false);
          if (TexInfo) {
            TexInfo->setType(DpctGlobalInfo::getUnqualifiedTypeName(TargetType),
                             TexType);
          }
        }
      }
    }

    return createRewriter(Call, RetAssign,
                          std::make_pair(Call, Call->getArg(RetAssign & 0x01)));
  }
};

/// Create rewriter factory for texture reader APIs.
/// Predicate: check the first arg if is pointer and set texture info with
/// corresponding data. Migrate the call expr to an assign expr if Pred result
/// is true; e.g.: tex1D(&u, tex, 1.0f) -> u = tex.read(1.0f) Migrate the call
/// expr to an assign expr if Pred result is false; e.g.: tex1D(tex, 1.0f) ->
/// tex.read(1.0f) The template arguments is the member call arguments' index in
/// original call expr.
template <size_t... Idx>
inline std::shared_ptr<CallExprRewriterFactoryBase>
createTextureReaderRewriterFactory(const std::string &Source, int TextureType) {
  return std::make_shared<TextureReadRewriterFactory<Idx...>>(Source,
                                                              TextureType);
}

template <class... MsgArgs>
inline std::shared_ptr<CallExprRewriterFactoryBase>
createUnsupportRewriterFactory(const std::string &Source, Diagnostics MsgID,
                               MsgArgs &&...Args) {
  return std::make_shared<UnsupportFunctionRewriterFactory<MsgArgs...>>(
      Source, MsgID, std::forward<MsgArgs>(Args)...);
}

template <class ArgT>
inline std::shared_ptr<CallExprRewriterFactoryBase>
createDerefExprRewriterFactory(
    const std::string &SourceName,
    std::function<ArgT(const CallExpr *)> &&ArgCreator) {
  return std::make_shared<CallExprRewriterFactory<
      DerefExprRewriter<ArgT>, std::function<ArgT(const CallExpr *)>>>(
      SourceName,
      std::forward<std::function<ArgT(const CallExpr *)>>(ArgCreator));
}

class CheckWarning1073 {
  unsigned Idx;

public:
  CheckWarning1073(unsigned I) : Idx(I) {}
  bool operator()(const CallExpr *C) {
    auto DerefE = getDereferencedExpr(C->getArg(Idx));
    return DerefE && isa<DeclRefExpr>(DerefE);
  }
};

// sycl has 2 overloading of malloc_device
// 1. sycl::malloc_device(Addr, Size)
// 2. sycl::malloc_device<type>(Addr, Size)
// DPCT will use the template version if following constraints hold:
// 1. The Addr can be derefed twice. The derefed type is type_1
// 2. The Size argument contains sizeof(type_2)
// 3. type_1 and type_2 are the same
// 4. The Size argument does not contain macro
class CheckCanUseTemplateMalloc {
  unsigned AddrArgIdx;
  unsigned SizeArgIdx;

public:
  CheckCanUseTemplateMalloc(unsigned AddrIdx, unsigned SizeIdx)
      : AddrArgIdx(AddrIdx), SizeArgIdx(SizeIdx) {}
  bool operator()(const CallExpr *C) {
    if (C->getNumArgs() <= AddrArgIdx)
      return false;
    auto AllocatedExpr = C->getArg(AddrArgIdx);
    const Expr *AE = nullptr;
    if (auto CSCE = dyn_cast<CStyleCastExpr>(AllocatedExpr)) {
      AE = CSCE->getSubExpr()->IgnoreImplicitAsWritten();
    } else {
      AE = AllocatedExpr;
    }

    // Try to deref twice to avoid the type is an unresolved template
    QualType DerefQT = AE->getType();
    DerefQT = DerefQualType(DerefQT);
    if (DerefQT.isNull())
      return false;
    DerefQT = DerefQualType(DerefQT);
    if (DerefQT.isNull())
      return false;

    if (C->getNumArgs() <= SizeArgIdx)
      return false;
    auto SizeExpr = C->getArg(SizeArgIdx);

    std::string TypeStr = DpctGlobalInfo::getReplacedTypeName(DerefQT);
    auto BO = dyn_cast<BinaryOperator>(SizeExpr);
    if (BO && BO->getOpcode() == BinaryOperatorKind::BO_Mul) {
      std::string Repl;
      if (!isContainMacro(BO->getLHS()) &&
          isSameSizeofTypeWithTypeStr(BO->getLHS(), TypeStr)) {
        // case 1: sizeof(b) * a
        return true;
      } else if (!isContainMacro(BO->getRHS()) &&
                 isSameSizeofTypeWithTypeStr(BO->getRHS(), TypeStr)) {
        // case 2: a * sizeof(b)
        return true;
      }
      return false;
    } else {
      // case 3: sizeof(b)
      if (!isContainMacro(SizeExpr) &&
          isSameSizeofTypeWithTypeStr(SizeExpr, TypeStr)) {
        return true;
      }
    }

    return false;
  }
};

class CheckCanUseCLibraryMallocOrFree {
  unsigned AddrArgIdx;
  bool isFree;
public:
  CheckCanUseCLibraryMallocOrFree(unsigned AddrIdx, bool isFree)
      : AddrArgIdx(AddrIdx), isFree(isFree) {}
  bool operator()(const CallExpr *C) {
    if (!DpctGlobalInfo::isOptimizeMigration()) {
      return false;
    }
    if (C->getNumArgs() <= AddrArgIdx)
      return false;
    auto AllocatedExpr = C->getArg(AddrArgIdx);
    const Expr *AE = nullptr;
    if (auto CSCE = dyn_cast<CStyleCastExpr>(
            AllocatedExpr->IgnoreImplicitAsWritten())) {
      AE = CSCE->getSubExpr()->IgnoreImplicitAsWritten();
    } else {
      AE = AllocatedExpr->IgnoreImplicitAsWritten();
    }
    const DeclRefExpr* DRE = nullptr;
    if (isFree) {
      DRE = dyn_cast<DeclRefExpr>(AE);
    } else if (auto UO = dyn_cast<UnaryOperator>(AE)) {
      if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf) {
        DRE = dyn_cast<DeclRefExpr>(UO->getSubExpr());
      }
    }
    if (!DRE) {
      return false;
    }
    auto DREDecl = DRE->getDecl();
    if (!DREDecl || !isa_and_nonnull<FunctionDecl>(DREDecl->getDeclContext()) ||
        !DREDecl->getType()->isPointerType()) {
      return false;
    }
    // If the pointer is only accessed on host side, then it's safe to replace
    // sycl::malloc_host with c library malloc.
    return isPointerHostAccessOnly(DREDecl);
  }
};

template <typename Compare = std::equal_to<>> class CheckArgCount {
  unsigned Count;
  Compare Comp;
  bool IncludeDefaultArg;

public:
  CheckArgCount(unsigned I, Compare Comp = Compare(),
                bool IncludeDefaultArg = true)
      : Count(I), Comp(Comp), IncludeDefaultArg(IncludeDefaultArg) {}
  bool operator()(const CallExpr *C) {
    unsigned DefaultArgNum = 0;
    llvm::ArrayRef<const Expr *> Args(C->getArgs(), C->getNumArgs());
    if (!IncludeDefaultArg) {
      DefaultArgNum =
          std::count_if(Args.begin(), Args.end(), [](const Expr *Arg) -> bool {
            return Arg->isDefaultArgument();
          });
    }
    return Comp(C->getNumArgs() - DefaultArgNum, Count);
  }
};

template <typename T>
CheckArgCount(unsigned I, T Comp, bool IncludeDefaultArg) -> CheckArgCount<T>;

class CheckBaseType {
  std::string TypeName;

public:
  CheckBaseType(std::string Name) : TypeName(Name) {}
  bool operator()(const CallExpr *C) {
    auto BaseType = getBaseTypeStr(C);
    if (BaseType.empty())
      return false;
    return TypeName == BaseType;
  }
};

inline auto UseNDRangeBarrier = [](const CallExpr *C) -> bool {
  return DpctGlobalInfo::useNdRangeBarrier();
};
inline auto UseLogicalGroup = [](const CallExpr *C) -> bool {
  return DpctGlobalInfo::useLogicalGroup();
};

class CheckDerefedTypeBeforeCast {
  unsigned Idx;
  std::string TypeName;

public:
  CheckDerefedTypeBeforeCast(unsigned I, std::string Name)
      : Idx(I), TypeName(Name) {}
  bool operator()(const CallExpr *C) {
    if (C->getNumArgs() > Idx) {
      std::ostringstream OS;
      std::string Type;
      printDerefOp(OS, C->getArg(Idx)->IgnoreCasts()->IgnoreParens(), &Type);
      if (Type != TypeName) {
        return false;
      }
    }
    return true;
  }
};

class CheckIsMaskedSubGroupFunctionEnabled {
public:
  CheckIsMaskedSubGroupFunctionEnabled() {}
  bool operator()(const CallExpr *C) {
    return DpctGlobalInfo::useMaskedSubGroupFunction();
  }
};

class CheckArgIsConstantIntWithValue {
  int value;
  int index;

public:
  CheckArgIsConstantIntWithValue(int idx, int val) : value(val), index(idx) {}
  bool operator()(const CallExpr *C) {
    auto Arg = C->getArg(index);
    Expr::EvalResult Result;
    if (!Arg->isValueDependent() &&
        Arg->EvaluateAsInt(Result, DpctGlobalInfo::getContext()) &&
        Result.Val.getInt().getSExtValue() == value) {
      return true;
    }
    return false;
  }
};

class CheckArgIsConstantIntWithUnsignedValue {
  unsigned int value;
  int index;

public:
  CheckArgIsConstantIntWithUnsignedValue(int idx, unsigned int val)
      : value(val), index(idx) {}
  bool operator()(const CallExpr *C) {
    auto Arg = C->getArg(index);
    Expr::EvalResult Result;
    if (!Arg->isValueDependent() &&
        Arg->EvaluateAsInt(Result, DpctGlobalInfo::getContext()) &&
        Result.Val.getInt().getZExtValue() == value) {
      return true;
    }
    return false;
  }
};

class CheckArgIsDefaultCudaStream {
  unsigned ArgIndex;

public:
  CheckArgIsDefaultCudaStream(unsigned ArgIndex) : ArgIndex(ArgIndex) {}
  bool operator()(const CallExpr *C) const {
    return isDefaultStream(C->getArg(ArgIndex));
  }
};

class CheckIsPtr {
  unsigned Idx;

public:
  CheckIsPtr(unsigned I) : Idx(I) {}
  // Normally, we will deref the ptr after we know it's a ptr,
  // so this check should return false in cases like template.
  bool operator()(const CallExpr *C) {
    if (C->getNumArgs() > Idx) {
      if (!C->getDirectCallee())
        return false;
      if (!C->getDirectCallee()->getParamDecl(Idx))
        return false;
      return C->getDirectCallee()
          ->getParamDecl(Idx)
          ->getType()
          ->isPointerType();
    }
    return false;
  }
};

template <class F, class S> class CheckAnd {
  F Fir;
  S Sec;

public:
  CheckAnd(const F &Fir, const S &Sec) : Fir(Fir), Sec(Sec) {}
  bool operator()(const CallExpr *C) { return Fir(C) && Sec(C); }
};

template <class F, class... Args> class CheckOr {
  F Fir;
  CheckOr<Args...> Sec;

public:
  CheckOr(const F &Fir, const Args &... args) : Fir(Fir), Sec(args...) {}
  bool operator()(const CallExpr *C) { return Fir(C) || Sec(C); }
};

template <class F> class CheckOr<F> {
  F Fir;

public:
  CheckOr(const F &Fir) : Fir(Fir) {}
  bool operator()(const CallExpr *C) { return Fir(C); }
};

template <class F, class... Args>
CheckOr<F, Args...> makeCheckOr(const F &Fir, const Args &... args) {
  return CheckOr<F, Args...>(Fir, args...);
}

template <class F, class S>
CheckAnd<F, S> makeCheckAnd(const F &Fir, const S &Sec) {
  return CheckAnd<F, S>(Fir, Sec);
}

template <class T> class CheckNot {
  T Expr;

public:
  CheckNot(T Expr) : Expr(Expr) {}
  bool operator()(const CallExpr *C) { return !Expr(C); }
};

template <class T> CheckNot<T> makeCheckNot(T Expr) {
  return CheckNot<T>(Expr);
}

class IsPolicyArgType {
  unsigned Idx;

public:
  IsPolicyArgType(unsigned I) : Idx(I) {}
  bool operator()(const CallExpr *C) {
    if (C->getNumArgs() <= Idx)
      return false;

    std::string ArgType = C->getArg(Idx)->getType().getCanonicalType().getUnqualifiedType().getAsString();
    if (// Explicitly known policy types
        // thrust::device
        ArgType=="struct thrust::detail::execution_policy_base<struct thrust::cuda_cub::par_t>"             ||
        ArgType=="struct thrust::cuda_cub::par_t"                                                           ||
        // thrust::host
        ArgType=="struct thrust::detail::execution_policy_base<struct thrust::system::cpp::detail::par_t>"  ||
        ArgType=="struct thrust::system::cpp::detail::par_t"                                                ||
        // thrust::seq
        ArgType=="struct thrust::detail::execution_policy_base<struct thrust::detail::seq_t>"               ||
        ArgType=="struct thrust::detail::seq_t"                                                             ||
        // cudaStream_t stream;
        // thrust::cuda::par.on(stream)
        ArgType=="struct thrust::detail::execution_policy_base<struct thrust::cuda_cub::execute_on_stream>" ||
        ArgType=="struct thrust::cuda_cub::execute_on_stream"                                               ||
        // class MyAlloctor {};
        // template<typename T>
        // void foo() {
        //   cudaStream_t stream;
        //   MyAlloctor thrust_allocator;
        //   auto policy = thrust::cuda::par(thrust_allocator).on(stream);
        //   ...
        //  }
        // Here ArgType for policy is "struct thrust::detail::execute_with_allocator<class MyAlloctor &, thrust::cuda_cub::execute_on_stream_base>"
        ArgType.find("struct thrust::detail::execute_with_allocator") != std::string::npos)
      return true;

    if (// Templated policy types.  If we see a templated type assume it is a policy if it is not the same type as the next argument type
        // FIXME, this check is a hack.  It would be better if we analyzed the templated type rather than comparing it against the
        // next argument.

        (//template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
         //__global__ void copy_if_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Predicate pred, Iterator3 result2) {
         //  *result2 = thrust::copy_if(exec, first, last, result1, pred);
         //
         // For the above code, exec will have type type-parameter-0-0
         ArgType=="type-parameter-0-0" ||

         //template <typename InputType, typename OutputType>
         //void myfunction(const std::shared_ptr<const Container<InputType>> &inImageData,
         //                int *dev_a, int *dev_b) {
         //  thrust::transform(thrust::cuda::par.on(inImageData->getStream()), dev_a, dev_a + 10, dev_b, my_math());
         //
         // For the above code, thrust::cuda::par.on(inImageData->getStream()) will have type <dependent type>
         ArgType=="<dependent type>") &&

        (Idx+1) < C->getNumArgs() &&
        C->getArg(Idx+0)->getType().getCanonicalType().getUnqualifiedType() !=
        C->getArg(Idx+1)->getType().getCanonicalType().getUnqualifiedType())
      return true;

    return false;
  }
};

class HasDirectCallee {
public:
  HasDirectCallee() {}
  bool operator()(const CallExpr *C) {
    const FunctionDecl *FD = C->getDirectCallee();
    return FD;
  }
};

template<class Attr> class IsDirectCalleeHasAttribute {
public:
  IsDirectCalleeHasAttribute() {}
  bool operator()(const CallExpr *C) {
    const FunctionDecl *FD = C->getDirectCallee();
    if (!FD)
      return false;
    return FD->hasAttr<Attr>();
  }
};

template <class Attr> class IsContextCallHasAttribute {
public:
  IsContextCallHasAttribute() {}
  bool operator()(const CallExpr *C) {
    const FunctionDecl *ContextFD = getImmediateOuterFuncDecl(C);
    if (!ContextFD)
      return false;
    return ContextFD->hasAttr<Attr>();
  }
};

inline std::function<std::string(const CallExpr *)> MemberExprBase() {
  return [=](const CallExpr *C) -> std::string {
    auto ME = dyn_cast<MemberExpr>(C->getCallee()->IgnoreImpCasts());
    if (!ME)
      return "";
    auto Base = ME->getBase()->IgnoreImpCasts();
    if (!Base)
      return "";
    return ExprAnalysis::ref(Base);
  };
}

class NeedExtraParens {
  unsigned Idx;
public:
  NeedExtraParens(unsigned I) : Idx(I) {}
  bool operator()(const CallExpr *C) { return needExtraParens(C->getArg(Idx)); }
};

class IsParameterIntegerType {
  unsigned Idx;
public:
  IsParameterIntegerType(unsigned Idx) : Idx(Idx) {}
  bool operator()(const CallExpr *C) {
    return C->getArg(Idx)->getType()->isIntegerType();
  }
};

class IsArgumentIntegerType {
  unsigned Idx;
public:
  IsArgumentIntegerType(unsigned Idx) : Idx(Idx) {}
  bool operator()(const CallExpr *C) {
    return C->getArg(Idx)->IgnoreImpCasts()->getType()->isIntegerType();
  }
};

namespace math {
class IsDefinedInCUDA {
public:
  IsDefinedInCUDA() {}
  bool operator()(const CallExpr *C) {
    auto FD = C->getDirectCallee();
    if (!FD)
      return false;
    return isFromCUDA(FD);
  }
};
} // namespace math
} // namespace dpct
} // namespace clang

#define ASSIGNABLE_FACTORY(x) createAssignableFactory(x 0),
#define ASSIGNABLE_FACTORY_WITH_PAREN(x)                                       \
  createAssignableFactoryWithExtraParen(x 0),
#define INSERT_AROUND_FACTORY(x, PREFIX, SUFFIX)                               \
  createInsertAroundFactory(x PREFIX, SUFFIX),
#define FEATURE_REQUEST_FACTORY(FEATURE, x)                                    \
  createFeatureRequestFactory(FEATURE, x 0),
#define HEADER_INSERT_FACTORY(HEADER, x) createInsertHeaderFactory(HEADER, x 0),
#define SUBGROUPSIZE_FACTORY(IDX, NEWFUNCNAME, x)                              \
  createFactoryWithSubGroupSizeRequest<IDX>(NEWFUNCNAME, x 0),
#define STREAM(x) makeDerefStreamExprCreator(x)
#define ADDROF(x) makeAddrOfExprCreator(x)
#define DEREF(x) makeDerefExprCreator(x)
#define DEREF_CAST_IF_NEED(T, S) makeDerefCastIfNeedExprCreator(T, S)
#define STRUCT_DISMANTLE(idx, ...) makeStructDismantler(idx, {__VA_ARGS__})
#define ARG(x) makeCallArgCreator(x)
#define ARG_WC(x) makeDerefArgCreatorWithCall(x)
#define TEMPLATE_ARG(x) makeCallArgCreatorFromTemplateArg(x)
#define BOOL(x) makeBooleanCreator(x)
#define BLAS_ENUM_ARG(x, BLAS_ENUM_TYPE)                                       \
  makeBLASEnumCallArgCreator(x, BLAS_ENUM_TYPE)
#define EXTENDSTR(idx, str) makeExtendStr(idx, str)
#define QUEUESTR makeQueueStr()
#define QUEUEPTRSTR makeQueuePtrStr()
#define BO(Op, L, R) makeBinaryOperatorCreator<Op>(L, R)
#define MEMBER_CALL(...) makeMemberCallCreator(__VA_ARGS__)
#define MEMBER_EXPR(...) makeMemberExprCreator(__VA_ARGS__)
#define STATIC_MEMBER_EXPR(...) makeStaticMemberExprCreator(__VA_ARGS__)
#define LAMBDA(...) makeLambdaCreator(__VA_ARGS__)
#define CALL(...) makeCallExprCreator(__VA_ARGS__)
#define ARRAY_SUBSCRIPT(e, i) makeArraySubscriptExprCreator(e, i)
#define CAST(T, S) makeCastExprCreator(T, S)
#define CAST_IF_NOT_SAME(T, S) makeCastIfNotSameExprCreator(T, S)
#define CAST_IF_SPECIAL(Idx) CastIfSpecialExprCreator(Idx)
#define DOUBLE_POINTER_CONST_CAST(BASE_VALUE_TYPE, EXPR,                       \
                                  DOES_BASE_VALUE_NEED_CONST,                  \
                                  DOES_FIRST_LEVEL_POINTER_NEED_CONST)         \
  makeDoublePointerConstCastExprCreator(BASE_VALUE_TYPE, EXPR,                 \
                                        DOES_BASE_VALUE_NEED_CONST,            \
                                        DOES_FIRST_LEVEL_POINTER_NEED_CONST)
#define NEW(...) makeNewExprCreator(__VA_ARGS__)
#define TYPENAME(SUBEXPR) makeTypenameExprCreator(SUBEXPR)
#define ZERO_INITIALIZER(SUBEXPR) makeZeroInitializerCreator(SUBEXPR)
#define SUBGROUP                                                               \
  std::function<SubGroupPrinter(const CallExpr *)>(SubGroupPrinter::create)
#define NDITEM std::function<ItemPrinter(const CallExpr *)>(ItemPrinter::create)
#define GROUP                                                                  \
  std::function<GroupPrinter(const CallExpr *)>(GroupPrinter::create)
#define POINTER_CHECKER(x) makePointerChecker(x)
#define LITERAL(x) makeLiteral(x)
#define TEMPLATED_NAME(Name, ...) makeTemplatedName(Name, __VA_ARGS__)
#define TEMPLATED_CALLEE(FuncName, ...)                                        \
  makeTemplatedCalleeCreator(FuncName, {__VA_ARGS__})
#define TEMPLATED_CALLEE_WITH_ARGS(FuncName, ...)                              \
  makeTemplatedCalleeWithArgsCreator(FuncName, __VA_ARGS__)
#define CASE(Pred, Entry) \
  createCase(Pred, Entry 0)
#define OTHERWISE(Entry) \
  createCase(CaseRewriterFactory::true_pred, Entry 0)

#define CONDITIONAL_FACTORY_ENTRY(Pred, First, Second)                         \
  createConditionalFactory(Pred, First Second 0),
#define IFELSE_FACTORY_ENTRY(FuncName, Pred, IfBlock, ElseBlock)               \
  {FuncName, createIfElseRewriterFactory(FuncName, Pred IfBlock ElseBlock 0)},
#define TEMPLATED_CALL_FACTORY_ENTRY(FuncName, ...)                            \
  {FuncName, createTemplatedCallExprRewriterFactory(FuncName, __VA_ARGS__)},
#define ASSIGN_FACTORY_ENTRY(FuncName, L, R)                                   \
  {FuncName, createBinaryOpRewriterFactory<BinaryOperatorKind::BO_Assign>(      \
                 FuncName, L, R)},
#define BINARY_OP_FACTORY_ENTRY(FuncName, OP, L, R)                            \
  {FuncName, createBinaryOpRewriterFactory<OP>(FuncName, L, R)},
#define UNARY_OP_FACTORY_ENTRY(FuncName, OP, Arg)                            \
  {FuncName, createUnaryOpRewriterFactory<OP>(FuncName, Arg)},
#define MEM_EXPR_ENTRY(FuncName, B, IsArrow, M)                                \
  {FuncName, createMemberExprRewriterFactory(FuncName, B, IsArrow, M)},
#define CALL_FACTORY_ENTRY(FuncName, C)                                        \
  {FuncName, createCallExprRewriterFactory(FuncName, C)},
#define MEMBER_CALL_FACTORY_ENTRY(FuncName, ...)                               \
  {FuncName, createMemberCallExprRewriterFactory(FuncName, __VA_ARGS__)},
#define ARRAYSUBSCRIPT_EXPR_FACTORY_ENTRY(FuncName, ...)                       \
  {FuncName, createArraySubscriptExprRewriterFactory(FuncName, __VA_ARGS__)},
#define DELETER_FACTORY_ENTRY(FuncName, Arg)                                   \
  {FuncName, createDeleterCallExprRewriterFactory(FuncName, Arg)},
#define UNSUPPORT_FACTORY_ENTRY(FuncName, MsgID, ...)                          \
  {FuncName, createUnsupportRewriterFactory(FuncName, MsgID, __VA_ARGS__)},
#define MULTI_STMTS_FACTORY_ENTRY(FuncName, ...)                               \
  {FuncName, createMultiStmtsRewriterFactory(FuncName, __VA_ARGS__)},
#define WARNING_FACTORY_ENTRY(FuncName, Factory, ...)                          \
  {FuncName, createReportWarningRewriterFactory(Factory FuncName, __VA_ARGS__)},
#define TOSTRING_FACTORY_ENTRY(FuncName, ...)                                  \
  {FuncName, createToStringExprRewriterFactory(FuncName, __VA_ARGS__)},
#define REMOVE_API_FACTORY_ENTRY(FuncName)                                     \
  {FuncName, createRemoveAPIRewriterFactory(FuncName)},
#define REMOVE_API_FACTORY_ENTRY_WITH_MSG(FuncName, Msg)                       \
  {FuncName, createRemoveAPIRewriterFactory(FuncName, Msg)},
#define CASE_FACTORY_ENTRY(...) \
  createCaseRewriterFactory(__VA_ARGS__),
#define DEREF_FACTORY_ENTRY(FuncName, E)                                       \
  {FuncName, createDerefExprRewriterFactory(FuncName, E)},

#endif // DPCT_CALL_EXPR_REWRITER_COMMON_H
