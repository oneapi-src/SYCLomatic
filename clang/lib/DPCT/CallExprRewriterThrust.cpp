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

inline std::function<ThrustFunctor(const CallExpr *)>
makeThrustFunctorArgCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> ThrustFunctor {
    return ThrustFunctor(C->getArg(Idx));
  };
}

inline std::function<std::string(const CallExpr *)>
makeMappedThrustPolicyEnum(unsigned Idx) {
  auto getBaseType = [](QualType QT) -> std::string {
    auto PP = DpctGlobalInfo::getContext().getPrintingPolicy();
    PP.PrintCanonicalTypes = true;
    return QT.getUnqualifiedType().getAsString(PP);
  };
  auto getMehtodName = [](const ValueDecl* VD) -> std::string {
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
    if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
      std::string EnumName = DRE->getNameInfo().getName().getAsString();
      if (EnumName == "device") {
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
        if (BaseType == "thrust::cuda_cub::par_t" &&
            MethodName == "on") {
          return "oneapi::dpl::execution::make_device_policy(" +
                 getDrefName(CMCE->getArg(0)) + ")";
        }
      }
    } else if (auto CE = dyn_cast<CallExpr>(E)) {
      if (auto ME = dyn_cast_or_null<MemberExpr>(CE->getCallee())) {
        auto BaseType = getBaseType(ME->getBase()->getType());
        auto MethodName = getMehtodName(ME->getMemberDecl());
        if (BaseType == "thrust::cuda_cub::par_t" &&
            MethodName == "on") {
          return "oneapi::dpl::execution::make_device_policy(" +
                 getDrefName(CE->getArg(0)) + ")";
        }
      }
    }
    return "oneapi::dpl::execution::make_device_policy(" + makeQueueStr()(C) +
           ")";
  };
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
