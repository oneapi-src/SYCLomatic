//===--------------- CallExprRewriterThrust.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"

namespace clang {
namespace dpct {

class CheckThrustArgType {
  unsigned Idx;
  std::string TypeName;

public:
  CheckThrustArgType(unsigned I, std::string Name) : Idx(I), TypeName(Name) {}
  bool operator()(const CallExpr *C) {
    std::string ArgType;
    unsigned NumArgs = C->getNumArgs();
    if (Idx < NumArgs) {
      // template<typename T>
      // void testfunc() {
      //  thrust::host_vector<T> V(1);
      //  thrust::stable_sort(V.begin(),V.end(),thrust::not2(thrust::greater_equal()));
      // }
      // void foo() {
      //   testfunc<int>();
      // }
      // For the code above argument "V.begin()" has type
      // "thrust::host_vector<T>" in AST.
      if (auto Call = dyn_cast<CallExpr>(C->getArg(Idx))) {
        if (auto CDSME =
                dyn_cast<CXXDependentScopeMemberExpr>(Call->getCallee())) {
          if (auto DRE = dyn_cast<DeclRefExpr>(CDSME->getBase())) {
            ArgType = DRE->getType().getAsString();
            if (ArgType.find("thrust::host_vector") != std::string::npos)
              return false;
          }
        }
      }

      ArgType = C->getArg(Idx)
                    ->getType()
                    .getCanonicalType()
                    .getUnqualifiedType()
                    .getAsString();
    }
    // template <class T>
    // void foo_host(){
    //  ...
    //  thrust::remove_copy_if(A.begin(), A.end(), R.begin(), pred);
    // }
    // For the code above argument "A.begin()" has type <dependent type> in AST,
    // we follow currrent solution assuming it a device iterator.
    if (ArgType == "<dependent type>")
      return true;

    // template <class T>
    // void foo() {
    //   greater_than_zero pred;
    //   thrust::device_vector<T> A(4);
    //   thrust::replace_if(A.begin(), A.end(), pred, 0);
    // }
    // For the code above argument "A.begin()" has type
    // "thrust::device_vector<T>"" in AST.
    if (ArgType.find("thrust::device_vector") != std::string::npos)
      return true;

    return ArgType.find(TypeName) != std::string::npos;
  }
};

inline std::function<ThrustFunctor(const CallExpr *)>
makeThrustFunctorArgCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> ThrustFunctor {
    return ThrustFunctor(C->getArg(Idx));
  };
}

std::function<bool(const CallExpr *)>
checkEnableExtDPLAPI() {
  return [=](const CallExpr *) -> bool {
    return DpctGlobalInfo::useExtDPLAPI();
  };
}

inline std::function<std::string(const CallExpr *)>
makeMappedThrustPolicyEnum(unsigned Idx) {
  auto getBaseType = [](QualType QT) -> std::string {
    auto PP = DpctGlobalInfo::getContext().getPrintingPolicy();
    PP.PrintCanonicalTypes = true;
    return QT.getUnqualifiedType().getAsString(PP);
  };
  auto getMehtodName = [](const ValueDecl *VD) -> std::string {
    if (!VD)
      return "";
    if (VD->getIdentifier()) {
      return VD->getNameAsString();
    }
    return "";
  };

  return [=](const CallExpr *C) -> std::string {
    auto E = C->getArg(Idx);
    E = E->IgnoreImpCasts();

    // To migrate "thrust::cuda::par.on" that appears in CE' first arg to
    // "oneapi::dpl::execution::make_device_policy" in senario:
    //   template<typename Itr>
    // void foo(Itr Beg, Itr End){
    //   cudaStream_t s1;
    //   ...
    //   thrust::sort(thrust::cuda::par.on(s1), Beg, End);
    // }
    const CallExpr *Call = nullptr;

    if (const auto *ICE = dyn_cast<ImplicitCastExpr>(C->getArg(0))) {
      if (const auto *MT =
              dyn_cast<MaterializeTemporaryExpr>(ICE->getSubExpr())) {
        if (auto SubICE = dyn_cast<ImplicitCastExpr>(MT->getSubExpr())) {
          Call = dyn_cast<CXXMemberCallExpr>(SubICE->getSubExpr());
        }
      }
    } else if (const auto *SubCE = dyn_cast<CallExpr>(C->getArg(0))) {
      Call = SubCE;
    } else {
      Call = dyn_cast<CXXMemberCallExpr>(C->getArg(0));
    }

    if (Call) {
      std::ostringstream OS;
      if (const auto *ME = dyn_cast<MemberExpr>(Call->getCallee())) {
        auto BaseName =
            DpctGlobalInfo::getUnqualifiedTypeName(ME->getBase()->getType());
        if (BaseName == "thrust::cuda_cub::par_t") {
          OS << "oneapi::dpl::execution::make_device_policy(";
          printDerefOp(OS, Call->getArg(0));
          OS << ")";
          return OS.str();
        }
      }
    }

    if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
      std::string EnumName = DRE->getNameInfo().getName().getAsString();
      if (EnumName == "device" || EnumName == "par") {
        return "oneapi::dpl::execution::make_device_policy(" +
               makeQueueStr()(C) + ")";
      } else if (EnumName == "seq" || EnumName == "host") {
        return "oneapi::dpl::execution::seq";
      } else {
        return EnumName;
      }
    } else if (auto MTE = dyn_cast<MaterializeTemporaryExpr>(E)) {
      if (auto CMCE = dyn_cast_or_null<CXXMemberCallExpr>(
              MTE->getSubExpr()->IgnoreImpCasts())) {
        auto BaseType = getBaseType(CMCE->getObjectType());
        auto MethodName = getMehtodName(CMCE->getMethodDecl());
        if (BaseType == "thrust::cuda_cub::par_t" && MethodName == "on") {
          return "oneapi::dpl::execution::make_device_policy(" +
                 getDrefName(CMCE->getArg(0)) + ")";
        }
      }
    }

    return "oneapi::dpl::execution::make_device_policy(" + makeQueueStr()(C) +
           ")";
  };
}

enum class PolicyState : bool { HasPolicy = true, NoPolicy = false };

struct ThrustOverload {
  int argCnt;
  PolicyState hasPolicy;
  int ptrCnt;
  std::string migratedFunc;
  HelperFeatureEnum feature;
};

std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
thrustFactory(const std::string &thrustFunc,
              std::vector<ThrustOverload> overloads);

inline std::function<std::vector<const clang::Expr *>(const CallExpr *)>
makeCallArgVectorCreator(unsigned idx, int number) {
  return [=](const CallExpr *C) -> std::vector<const clang::Expr *> {
    std::vector<const clang::Expr *> args{};

    for (auto i = 0; i < number; i++)
      args.push_back(C->getArg(idx + i));
    return args;
  };
}

inline auto makeTemplatedCallArgCreator(unsigned idx) {
  return makeCallExprCreator(
      TEMPLATED_CALLEE_WITH_ARGS(
          MapNames::getDpctNamespace() + "device_pointer", getDerefedType(idx)),
      ARG(idx));
}

inline std::function<std::vector<CallExprPrinter<
    TemplatedNamePrinter<StringRef, std::vector<TemplateArgumentInfo>>,
    const Expr *>>(const CallExpr *)>
makeTemplatedCallArgVectorCreator(unsigned idx, int number) {
  using callExprCreatorType = decltype(makeTemplatedCallArgCreator(idx));

  std::vector<callExprCreatorType> callExprs{};

  for (auto i = 0; i < number; i++) {
    callExprs.push_back(makeTemplatedCallArgCreator(idx + i));
  }

  using eltType = std::invoke_result_t<callExprCreatorType, const CallExpr *>;

  return [=, callExprs = std::move(callExprs)](
             const CallExpr *C) -> std::vector<eltType> {
    std::vector<eltType> args;
    for (auto i = 0; i < number; i++)
      args.push_back(callExprs[i](C));
    return args;
  };
}

auto createSequentialPolicyCallExprRewriterFactory(
    const std::string &thrustFunc, const std::string &syclFunc, int argCnt,
    PolicyState hasPolicy) {

  int argStart = 0;

  if (static_cast<bool>(hasPolicy)) {
    // Skip policy argument
    argCnt--;
    argStart = 1;
  }

  return createCallExprRewriterFactory(
      thrustFunc,
      makeCallExprCreator(syclFunc, ARG("oneapi::dpl::execution::seq"),
                          makeCallArgVectorCreator(argStart, argCnt)));
}

auto createMappedPolicyCallExprRewriterFactory(const std::string &thrustFunc,
                                               const std::string &syclFunc,
                                               int argCnt) {

  // Skip policy argument
  int argStart = 1;
  argCnt--;

  auto mappedPolicy = makeMappedThrustPolicyEnum(0);

  return createCallExprRewriterFactory(
      thrustFunc,
      makeCallExprCreator(syclFunc, mappedPolicy,
                          makeCallArgVectorCreator(argStart, argCnt)));
}

auto createDevicePolicyCallExprRewriterFactory(const std::string &thrustFunc,
                                               const std::string &syclFunc,
                                               int argCnt, int templatedCnt,
                                               PolicyState hasPolicy) {

  auto makeDevicePolicy = makeCallExprCreator(
      "oneapi::dpl::execution::make_device_policy", QUEUESTR);

  int argStart = 0;

  if (static_cast<bool>(hasPolicy)) {
    // Skip policy argument
    argCnt--;
    argStart = 1;
  }

  return createCallExprRewriterFactory(
      thrustFunc, makeCallExprCreator(
                      syclFunc, makeDevicePolicy,
                      makeTemplatedCallArgVectorCreator(argStart, templatedCnt),
                      makeCallArgVectorCreator(argStart + templatedCnt,
                                               argCnt - templatedCnt)));
}

std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
thrustOverloadFactory(const std::string &thrustFunc,
                      const ThrustOverload &overload) {
  auto not_usm = createIfElseRewriterFactory(
      thrustFunc,
      createFeatureRequestFactory(
          HelperFeatureEnum::Memory_is_device_ptr,
          {thrustFunc,
           createCallExprRewriterFactory(
               thrustFunc,
               makeCallExprCreator(
                   MapNames::getDpctNamespace() + "is_device_ptr",
                   ARG(static_cast<bool>(overload.hasPolicy) ? 1 : 0)))}),
      createFeatureRequestFactory(
          HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          {thrustFunc, createDevicePolicyCallExprRewriterFactory(
                           thrustFunc, overload.migratedFunc, overload.argCnt,
                           overload.ptrCnt, overload.hasPolicy)}),
      {thrustFunc, createSequentialPolicyCallExprRewriterFactory(
                       thrustFunc, overload.migratedFunc, overload.argCnt,
                       overload.hasPolicy)},
      0);

  auto usm =
      (static_cast<bool>(overload.hasPolicy)
           ? std::pair{thrustFunc,
                       createMappedPolicyCallExprRewriterFactory(
                           thrustFunc, overload.migratedFunc, overload.argCnt)}
           : createConditionalFactory(
                 CheckThrustArgType(static_cast<bool>(overload.hasPolicy) ? 1
                                                                          : 0,
                                    "thrust::device_ptr"),
                 {thrustFunc, createDevicePolicyCallExprRewriterFactory(
                                  thrustFunc, overload.migratedFunc,
                                  overload.argCnt, 0, overload.hasPolicy)},
                 {thrustFunc, createSequentialPolicyCallExprRewriterFactory(
                                  thrustFunc, overload.migratedFunc,
                                  overload.argCnt, overload.hasPolicy)}));

  return createFeatureRequestFactory(
      overload.feature,
      createConditionalFactory(
          makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
          {thrustFunc, not_usm}, std::move(usm)));
}

std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
thrustFactory(const std::string &thrustFunc,
              std::vector<ThrustOverload> overloads) {

  auto u =
      std::pair{thrustFunc, createUnsupportRewriterFactory(
                                thrustFunc, Diagnostics::OVERLOAD_UNSUPPORTED,
                                makeCallArgCreator(thrustFunc))};

  for (auto it = overloads.rbegin(); it != overloads.rend(); it++) {
    u = (static_cast<bool>(it->hasPolicy)
             ? createConditionalFactory(
                   makeCheckAnd(CheckArgCount(it->argCnt), IsPolicyArgType(0)),
                   thrustOverloadFactory(thrustFunc, *it), std::move(u))
             : createConditionalFactory(
                   makeCheckAnd(CheckArgCount(it->argCnt),
                                makeCheckNot(IsPolicyArgType(0))),
                   thrustOverloadFactory(thrustFunc, *it), std::move(u)));
  }

  return u;
}

#define THRUST_FUNCTOR(x) makeThrustFunctorArgCreator(x)
#define REWRITER_FACTORY_ENTRY(FuncName, RewriterFactory, ...)                 \
  {FuncName, std::make_shared<RewriterFactory>(FuncName, __VA_ARGS__)},
#define FUNC_NAME_FACTORY_ENTRY(FuncName, RewriterName)                        \
  REWRITER_FACTORY_ENTRY(FuncName, FuncCallExprRewriterFactory, RewriterName)
// clang-format off
void CallExprRewriterFactoryBase::initRewriterMapThrust() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesThrust.inc"
#define ENTRY_HOST(from, to, policy)
#define ENTRY_DEVICE(SOURCEAPINAME, TARGETAPINAME, EXTR)                       \
  FUNC_NAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_BOTH(SOURCEAPINAME, TARGETAPINAME, EXTR)                         \
  FUNC_NAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#include "APINamesMapThrust.inc"
#undef ENTRY_HOST
#undef ENTRY_DEVICE
#undef ENTRY_BOTH
      }));
}
// clang-format

} // namespace dpct
} // namespace clang
