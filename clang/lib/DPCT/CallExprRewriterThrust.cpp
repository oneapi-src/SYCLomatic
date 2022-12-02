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
// thrust::replace_if
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(6),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::replace_if(policy, ptr, ptr, stc, pred, val);
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_replace_if,
      IFELSE_FACTORY_ENTRY(
        "thrust::replace_if",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::replace_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::replace_if",
                            CALL(MapNames::getDpctNamespace() + "replace_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  ARG(4), ARG(5)))),
        CALL_FACTORY_ENTRY("thrust::replace_if",
                          CALL(MapNames::getDpctNamespace() + "replace_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))),
    //Handling case: thrust::replace_if(thrust::device, device.begin(), device.end(), stencil.begin(), pred, 0);
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_replace_if,
        CALL_FACTORY_ENTRY("thrust::replace_if",
                           CALL(MapNames::getDpctNamespace() + "replace_if",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(4),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::replace_if(ptr, ptr, pred, val);
      IFELSE_FACTORY_ENTRY(
        "thrust::replace_if",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::replace_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::replace_if",
                            CALL("oneapi::dpl::replace_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  ARG(2), ARG(3)))),
        CALL_FACTORY_ENTRY("thrust::replace_if",
                           CALL("oneapi::dpl::replace_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3)))),
      //Handling case: thrust::replace_if(device.begin(), device.end(), pred, 0);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::replace_if",
                          CALL("oneapi::dpl::replace_if",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::replace_if",
                          CALL("oneapi::dpl::replace_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::replace_if(policy, ptr, ptr, pred, val);
        IFELSE_FACTORY_ENTRY(
          "thrust::replace_if",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::replace_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::replace_if",
                              CALL("oneapi::dpl::replace_if",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    ARG(3), ARG(4)))),
          CALL_FACTORY_ENTRY("thrust::replace_if",
                             CALL("oneapi::dpl::replace_if",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(1), ARG(2), ARG(3), ARG(4)))),
        //Handling case: thrust::replace_if(ptr, ptr, stc, pred, val);
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_replace_if,
          IFELSE_FACTORY_ENTRY(
            "thrust::replace_if",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
              CALL_FACTORY_ENTRY("thrust::replace_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
              CALL_FACTORY_ENTRY("thrust::replace_if",
                                CALL(MapNames::getDpctNamespace() + "replace_if",
                                      CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                      ARG(3), ARG(4)))),
            CALL_FACTORY_ENTRY("thrust::replace_if",
                              CALL(MapNames::getDpctNamespace() + "replace_if",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))))),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::replace_if(policy, device.begin(), device.end(), pred, 0);
        CALL_FACTORY_ENTRY("thrust::replace_if",
                          CALL("oneapi::dpl::replace_if",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3), ARG(4))),
        //Handling case: thrust::replace_if(device.begin(), device.end(), stencil.begin(), pred, 0);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_replace_if,
            CALL_FACTORY_ENTRY("thrust::replace_if",
                              CALL(MapNames::getDpctNamespace() + "replace_if",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_replace_if,
            CALL_FACTORY_ENTRY("thrust::replace_if",
                              CALL(MapNames::getDpctNamespace() + "replace_if",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))))))
    )
  )
)

// thrust::remove_if
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::remove_if(policy, ptr, ptr, stc, pred);
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_if,
      IFELSE_FACTORY_ENTRY(
        "thrust::remove_if",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::remove_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::remove_if",
                              CALL(MapNames::getDpctNamespace() + "remove_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  ARG(4)))),
        CALL_FACTORY_ENTRY("thrust::remove_if",
                          CALL(MapNames::getDpctNamespace() + "remove_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4))))),
    //Handling case: thrust::remove_if(thrust::device, device.begin(), device.end(), stencil.begin(), pred);
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_if,
      CALL_FACTORY_ENTRY("thrust::remove_if",
                        CALL(MapNames::getDpctNamespace() + "remove_if",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4))))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(3),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::remove_if(ptr, ptr, pred);
        IFELSE_FACTORY_ENTRY(
          "thrust::remove_if",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::remove_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::remove_if",
                              CALL("oneapi::dpl::remove_if",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    ARG(2)))),
          CALL_FACTORY_ENTRY("thrust::remove_if",
                             CALL("oneapi::dpl::remove_if",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2)))),
      //Handling case: thrust::remove_if(device.begin(), device.end(), pred);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::remove_if",
                             CALL("oneapi::dpl::remove_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  ARG(0), ARG(1), ARG(2))),
          CALL_FACTORY_ENTRY("thrust::remove_if",
                             CALL("oneapi::dpl::remove_if",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      CONDITIONAL_FACTORY_ENTRY(
      CompareArgType(0, 1),
      //Handling case: thrust::remove_if(policy, ptr, ptr, pred);
      IFELSE_FACTORY_ENTRY(
        "thrust::remove_if",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::remove_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::remove_if",
                            CALL("oneapi::dpl::remove_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  ARG(3)))),
        CALL_FACTORY_ENTRY("thrust::remove_if",
                           CALL("oneapi::dpl::remove_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3)))),
      //Handling case: thrust::remove_if(ptr, ptr, stc, pred);
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_if,
        IFELSE_FACTORY_ENTRY(
          "thrust::remove_if",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::remove_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::remove_if",
                              CALL(MapNames::getDpctNamespace() + "remove_if",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    ARG(3)))),
          CALL_FACTORY_ENTRY("thrust::remove_if",
                            CALL(MapNames::getDpctNamespace() + "remove_if",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2), ARG(3)))))),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::remove_if(thrust::device, device.begin(), device.end(), pred);
        CALL_FACTORY_ENTRY("thrust::remove_if",
                          CALL("oneapi::dpl::remove_if",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3))),
        //Handling case: thrust::remove_if(device.begin(), device.end(), stencil.begin(), pred);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_if,
            CALL_FACTORY_ENTRY("thrust::remove_if",
                              CALL(MapNames::getDpctNamespace() + "remove_if",
                                   CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                   ARG(0), ARG(1), ARG(2), ARG(3)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_if,
            CALL_FACTORY_ENTRY("thrust::remove_if",
                              CALL(MapNames::getDpctNamespace() + "remove_if",
                                   ARG("oneapi::dpl::execution::seq"),
                                   ARG(0), ARG(1), ARG(2), ARG(3))))))
    )
  )
)

// thrust::remove_copy_if
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(6),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::remove_copy_if(policy, ptr, ptr, stc, ptr, pred);
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if,
      IFELSE_FACTORY_ENTRY(
        "thrust::remove_copy_if",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::remove_copy_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                            CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4)),
                                  ARG(5)))),
        CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                          CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))),
    //Handling case: thrust::remove_copy_if(thrust::device, device.begin(), device.end(), stencil.begin(), result, pred);
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if,
      CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                        CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(4),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::remove_copy_if(ptr, ptr, ptr, pred);
      IFELSE_FACTORY_ENTRY(
        "thrust::remove_copy_if",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::remove_copy_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                            CALL("oneapi::dpl::remove_copy_if",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  ARG(3)))),
        CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                           CALL("oneapi::dpl::remove_copy_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3)))),
      //Handling case: thrust::remove_copy_if(device.begin(), device.end(), result, pred);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                           CALL("oneapi::dpl::remove_copy_if",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                           CALL("oneapi::dpl::remove_copy_if",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::remove_copy_if(policy, ptr, ptr, ptr, pred);
        IFELSE_FACTORY_ENTRY(
          "thrust::remove_copy_if",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::remove_copy_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                              CALL("oneapi::dpl::remove_copy_if",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                    ARG(4)))),
          CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                             CALL("oneapi::dpl::remove_copy_if",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(1), ARG(2), ARG(3), ARG(4)))),
        //Handling case: thrust::remove_copy_if(ptr, ptr, stc, ptr, pred);
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if,
          IFELSE_FACTORY_ENTRY(
            "thrust::remove_copy_if",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
              CALL_FACTORY_ENTRY("thrust::remove_copy_if", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
              CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                                CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                      CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(3)),
                                      ARG(4)))),
            CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                              CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))))),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::remove_copy_if(thrust::device, device.begin(), device.end(), result, pred);
        CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                          CALL("oneapi::dpl::remove_copy_if",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3), ARG(4))),
        //Handling case: thrust::remove_copy_if(device.begin(), device.end(), stencil.begin(), result, pred);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if,
            CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                              CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                   CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                   ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_remove_copy_if,
            CALL_FACTORY_ENTRY("thrust::remove_copy_if",
                              CALL(MapNames::getDpctNamespace() + "remove_copy_if",
                                   ARG("oneapi::dpl::execution::seq"),
                                   ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))))))
    )
  )
)

// thrust::not1
CALL_FACTORY_ENTRY("thrust::not1", CALL("oneapi::dpl::not1", ARG(0)))

// thrust::any_of
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(4),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    IFELSE_FACTORY_ENTRY(
      "thrust::any_of",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::any_of", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::any_of",
                            CALL("oneapi::dpl::any_of",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                ARG(3)))),
      CALL_FACTORY_ENTRY("thrust::any_of",
                          CALL("oneapi::dpl::any_of",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(4)))),
    CALL_FACTORY_ENTRY("thrust::any_of",
                      CALL("oneapi::dpl::any_of",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3)))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    IFELSE_FACTORY_ENTRY(
      "thrust::any_of",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::any_of", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::any_of",
                            CALL("oneapi::dpl::any_of",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                ARG(2)))),
      CALL_FACTORY_ENTRY("thrust::any_of",
                          CALL("oneapi::dpl::any_of",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2)))),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      CALL_FACTORY_ENTRY("thrust::any_of", CALL("oneapi::dpl::any_of",
                         CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                         ARG(0), ARG(1), ARG(2))),
      CALL_FACTORY_ENTRY("thrust::any_of", CALL("oneapi::dpl::any_of", ARG("oneapi::dpl::execution::seq"),
                         ARG(0), ARG(1), ARG(2))))
  )
)

// thrust::replace
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::replace(policy, ptr, ptr, val, val);
    IFELSE_FACTORY_ENTRY(
      "thrust::replace",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::replace", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::replace",
                            CALL("oneapi::dpl::replace",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                ARG(3), ARG(4)))),
      CALL_FACTORY_ENTRY("thrust::replace",
                          CALL("oneapi::dpl::replace",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(4)))),
    CALL_FACTORY_ENTRY("thrust::replace",
                       CALL("oneapi::dpl::replace",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4)))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::replace(ptr, ptr, val, val);
    IFELSE_FACTORY_ENTRY(
      "thrust::replace",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::replace", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::replace",
                            CALL("oneapi::dpl::replace",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                ARG(2), ARG(3)))),
      CALL_FACTORY_ENTRY("thrust::replace",
                          CALL("oneapi::dpl::replace",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(3)))),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      CALL_FACTORY_ENTRY("thrust::replace", CALL("oneapi::dpl::replace",
                         CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                         ARG(0), ARG(1), ARG(2), ARG(3))),
      CALL_FACTORY_ENTRY("thrust::replace", CALL("oneapi::dpl::replace",
                         ARG("oneapi::dpl::execution::seq"),
                         ARG(0), ARG(1), ARG(2), ARG(3))))
  )
)

// thrust::adjacent_difference
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::adjacent_difference(policy, ptr, ptr, ptr, op);
    IFELSE_FACTORY_ENTRY(
      "thrust::adjacent_difference",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::adjacent_difference", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                          CALL("oneapi::dpl::adjacent_difference",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                ARG(4)))),
      CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                         CALL("oneapi::dpl::adjacent_difference",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(4)))),
    //Handling case: thrust::adjacent_difference(policy, device.begin(), device.end(), result, op);
    CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                      CALL("oneapi::dpl::adjacent_difference",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4)))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(3),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::adjacent_difference(ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::adjacent_difference",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::adjacent_difference", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                            CALL("oneapi::dpl::adjacent_difference",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2))))),
        CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                           CALL("oneapi::dpl::adjacent_difference",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2)))),
      //Handling case: thrust::adjacent_difference(host.begin(), host.end(), result);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                          CALL("oneapi::dpl::adjacent_difference",
                               CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                               ARG(0), ARG(1), ARG(2))),
        CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                          CALL("oneapi::dpl::adjacent_difference",
                               ARG("oneapi::dpl::execution::seq"),
                               ARG(0), ARG(1), ARG(2))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::adjacent_difference(policy, ptr, ptr, ptr);
        IFELSE_FACTORY_ENTRY(
          "thrust::adjacent_difference",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::adjacent_difference", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                              CALL("oneapi::dpl::adjacent_difference",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3))))),
          CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                             CALL("oneapi::dpl::adjacent_difference",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(1), ARG(2), ARG(3)))),
        //Handling case: thrust::adjacent_difference(ptr, ptr, ptr, op);
        IFELSE_FACTORY_ENTRY(
          "thrust::adjacent_difference",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::adjacent_difference", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                              CALL("oneapi::dpl::adjacent_difference",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    ARG(3)))),
          CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                             CALL("oneapi::dpl::adjacent_difference",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2), ARG(3))))),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::adjacent_difference(policy, device.begin(), device.end(), result);
        CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                          CALL("oneapi::dpl::adjacent_difference",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3))),
        //Handling case: thrust::adjacent_difference(host.begin(), host.end(), result, op);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                            CALL("oneapi::dpl::adjacent_difference",
                                 CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                 ARG(0), ARG(1), ARG(2), ARG(3))),
          CALL_FACTORY_ENTRY("thrust::adjacent_difference",
                            CALL("oneapi::dpl::adjacent_difference",
                                 ARG("oneapi::dpl::execution::seq"),
                                 ARG(0), ARG(1), ARG(2), ARG(3)))))
    )
  )
)

// thrust::inclusive_scan
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  CONDITIONAL_FACTORY_ENTRY(
    makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
    //Handling case: thrust::inclusive_scan(policy, ptr, ptr, ptr, op);
    IFELSE_FACTORY_ENTRY(
      "thrust::inclusive_scan",
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
        CALL_FACTORY_ENTRY("thrust::inclusive_scan", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
      FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
        CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                          CALL("oneapi::dpl::inclusive_scan",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                ARG(4)))),
      CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                         CALL("oneapi::dpl::inclusive_scan",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(1), ARG(2), ARG(3), ARG(4)))),
    //Handling case: thrust::inclusive_scan(policy, device.begin(), device.end(), result, op);
    CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                      CALL("oneapi::dpl::inclusive_scan",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4)))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(3),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::inclusive_scan(ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::inclusive_scan",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::inclusive_scan", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                            CALL("oneapi::dpl::inclusive_scan",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2))))),
        CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                           CALL("oneapi::dpl::inclusive_scan",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2)))),
      //Handling case: thrust::inclusive_scan(host.begin(), host.end(), result);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                           CALL("oneapi::dpl::inclusive_scan",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2))),
        CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                           CALL("oneapi::dpl::inclusive_scan",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::inclusive_scan(policy, ptr, ptr, ptr);
        IFELSE_FACTORY_ENTRY(
          "thrust::inclusive_scan",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::inclusive_scan", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                              CALL("oneapi::dpl::inclusive_scan",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3))))),
          CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                             CALL("oneapi::dpl::inclusive_scan",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(1), ARG(2), ARG(3)))),
        //Handling case: thrust::inclusive_scan(ptr, ptr, ptr, op);
        IFELSE_FACTORY_ENTRY(
          "thrust::inclusive_scan",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::inclusive_scan", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                              CALL("oneapi::dpl::inclusive_scan",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    ARG(3)))),
          CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                             CALL("oneapi::dpl::inclusive_scan",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2), ARG(3))))),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::inclusive_scan(policy, device.begin(), device.end(), result);
        CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                          CALL("oneapi::dpl::inclusive_scan",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3))),
        //Handling case: thrust::inclusive_scan(host.begin(), host.end(), result, op);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                             CALL("oneapi::dpl::inclusive_scan",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  ARG(0), ARG(1), ARG(2), ARG(3))),
          CALL_FACTORY_ENTRY("thrust::inclusive_scan",
                             CALL("oneapi::dpl::inclusive_scan",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2), ARG(3)))))
    )
  )
)

// thrust::gather
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_gather,
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(5),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      // Handling case: thrust::gather(policy, ptr, ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::gather",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::gather", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::gather",
                              CALL(MapNames::getDpctNamespace() + "gather",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4))))),
        CALL_FACTORY_ENTRY("thrust::gather",
                            CALL(MapNames::getDpctNamespace() + "gather",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4)))),
      // Handling case: thrust::gather(thrust::device, d_map.begin(), d_map.end(), d_values.begin(),d_output.begin());
      CALL_FACTORY_ENTRY("thrust::gather",
                        CALL(MapNames::getDpctNamespace() + "gather",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4)))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      // Handling case: thrust::gather(ptr, ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::gather",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::gather", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::gather",
                              CALL(MapNames::getDpctNamespace() + "gather",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3))))),
        CALL_FACTORY_ENTRY("thrust::gather",
                            CALL(MapNames::getDpctNamespace() + "gather",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3)))),
      // Handling case: thrust::gather(h_map.begin(), h_map.end(), h_values.begin(),h_output.begin());
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::gather", CALL(MapNames::getDpctNamespace() + "gather",
                           CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                           ARG(0), ARG(1), ARG(2), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::gather", CALL(MapNames::getDpctNamespace() + "gather", ARG("oneapi::dpl::execution::seq"),
                          ARG(0), ARG(1), ARG(2), ARG(3))))
    )
  )
)

// thrust::scatter
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_scatter,
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(5),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      // Handling case: thrust::scatter(policy, ptr, ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::scatter",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::scatter", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::scatter",
                              CALL(MapNames::getDpctNamespace() + "scatter",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4))))),
        CALL_FACTORY_ENTRY("thrust::scatter",
                            CALL(MapNames::getDpctNamespace() + "scatter",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4)))),
      // Handling case: thrust::scatter(policy, d_values.begin(), d_values.end(), d_map.begin(), d_output.begin());
      CALL_FACTORY_ENTRY("thrust::scatter",
                        CALL(MapNames::getDpctNamespace() + "scatter",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4)))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      // Handling case: thrust::scatter(ptr, ptr, ptr, ptr);
      IFELSE_FACTORY_ENTRY(
        "thrust::scatter",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::scatter", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::scatter",
                              CALL(MapNames::getDpctNamespace() + "scatter",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3))))),
        CALL_FACTORY_ENTRY("thrust::scatter",
                            CALL(MapNames::getDpctNamespace() + "scatter",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3)))),
      // Handling case: thrust::scatter(d_values.begin(), d_values.end(), d_map.begin(), d_output.begin());
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::scatter", CALL(MapNames::getDpctNamespace() + "scatter",
                           CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                           ARG(0), ARG(1), ARG(2), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::scatter", CALL(MapNames::getDpctNamespace() + "scatter", ARG("oneapi::dpl::execution::seq"),
                           ARG(0), ARG(1), ARG(2), ARG(3))))
    )
  )
)

// thrust::unique_by_key_copy
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_unique_copy,
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(7),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::unique_by_key_copy(policy, ptr, ptr, ptr, ptr, ptr, pred);
      IFELSE_FACTORY_ENTRY(
        "thrust::unique_by_key_copy",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::unique_by_key_copy", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                            CALL(MapNames::getDpctNamespace() + "unique_copy",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(5)), ARG(5)),
                                  ARG(6)))),
        CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                          CALL(MapNames::getDpctNamespace() + "unique_copy",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))),
      // Handling case: thrust::unique_by_key_copy(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
      CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                        CALL(MapNames::getDpctNamespace() + "unique_copy",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6)))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgCount(5),
      CONDITIONAL_FACTORY_ENTRY(
        makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
        //Handling case: thrust::unique_by_key_copy(ptr, ptr, ptr, ptr, ptr);
        IFELSE_FACTORY_ENTRY(
          "thrust::unique_by_key_copy",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                              CALL(MapNames::getDpctNamespace() + "unique_copy",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4))))),
          CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                            CALL(MapNames::getDpctNamespace() + "unique_copy",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))),
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          // Handling case: thrust::unique_by_key_copy(h_keys.begin(), h_keys.end(), h_values.begin(), h_output_keys.begin(), h_output_values.begin());
          CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                             CALL(MapNames::getDpctNamespace() + "unique_copy",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))),
          CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                             CALL(MapNames::getDpctNamespace() + "unique_copy",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))))
      ),
      CONDITIONAL_FACTORY_ENTRY(
        makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
        CONDITIONAL_FACTORY_ENTRY(
          CompareArgType(0, 1),
          //Handling case: thrust::unique_by_key_copy(policy, ptr, ptr, ptr, ptr, ptr);
          IFELSE_FACTORY_ENTRY(
            "thrust::unique_by_key_copy",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
              CALL_FACTORY_ENTRY("thrust::unique_by_key_copy", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
              CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                                CALL(MapNames::getDpctNamespace() + "unique_copy",
                                      CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(5)), ARG(5))))),
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                              CALL(MapNames::getDpctNamespace() + "unique_copy",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(1), ARG(2), ARG(3), ARG(4), ARG(5)))),
          //Handling case: thrust::unique_by_key_copy(ptr, ptr, ptr, ptr, ptr, pred);
          IFELSE_FACTORY_ENTRY(
            "thrust::unique_by_key_copy",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
              CALL_FACTORY_ENTRY("thrust::unique_by_key_copy", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
              CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                                CALL(MapNames::getDpctNamespace() + "unique_copy",
                                      CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(4)), ARG(4)),
                                      ARG(5)))),
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                              CALL(MapNames::getDpctNamespace() + "unique_copy",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))),
        CONDITIONAL_FACTORY_ENTRY(
          CompareArgType(0, 1),
          // Handling case: thrust::unique_by_key_copy(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(),d_output_keys.begin(), d_output_values.begin());
          CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                             CALL(MapNames::getDpctNamespace() + "unique_copy",
                                  makeMappedThrustPolicyEnum(0),
                                  ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
          CONDITIONAL_FACTORY_ENTRY(
            CheckArgType(1, "thrust::device_ptr"),
            // Handling case: thrust::unique_by_key_copy(d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                               CALL(MapNames::getDpctNamespace() + "unique_copy",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                               CALL(MapNames::getDpctNamespace() + "unique_copy",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5)))))
      )
    )
  )
)

// thrust::transform_reduce
CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(5),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(0, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY(
            "thrust::transform_reduce",
            CALL("std::transform_reduce",
                 CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                 ARG(0), ARG(1), ARG(3), ARG(4), ARG(2))),
        CALL_FACTORY_ENTRY("thrust::transform_reduce",
                           CALL("std::transform_reduce",
                                ARG("oneapi::dpl::execution::seq"), ARG(0),
                                ARG(1), ARG(3), ARG(4), ARG(2)))),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(0, "thrust::device"),
        CALL_FACTORY_ENTRY(
            "thrust::transform_reduce",
            CALL("std::transform_reduce",
                 CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                 ARG(1), ARG(2), ARG(4), ARG(5), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::transform_reduce",
                           CALL("std::transform_reduce",
                                ARG("oneapi::dpl::execution::seq"), ARG(1),
                                ARG(2), ARG(4), ARG(5), ARG(3)))))


// thrust::swap
CALL_FACTORY_ENTRY("thrust::swap",  CALL("std::swap",  ARG(0), ARG(1)))

// thrust::make_pair
CALL_FACTORY_ENTRY("thrust::make_pair",  CALL("std::make_pair",  ARG(0), ARG(1)))

// thrust::make_pair
CALL_FACTORY_ENTRY("thrust::make_pair",  CALL("std::make_pair",  ARG(0), ARG(1)))

// thrust::stable_sort_by_key
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_stable_sort,
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(5),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      // Handling case: thrust::stable_sort_by_key(policy, ptr, ptr, ptr, comp)
      IFELSE_FACTORY_ENTRY(
        "thrust::stable_sort_by_key",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::stable_sort_by_key", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                            CALL(MapNames::getDpctNamespace() + "stable_sort",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  ARG(4)))),
        CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                          CALL(MapNames::getDpctNamespace() + "stable_sort",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4)))),
      // Handling case: thrust::stable_sort_by_key(thrust::device,keys_first, keys_last, values_first, comp)
      CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                        CALL(MapNames::getDpctNamespace() + "stable_sort",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4)))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgCount(3),
      CONDITIONAL_FACTORY_ENTRY(
        makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
        // Handling case: thrust::stable_sort_by_key(ptr, ptr, ptr)
        IFELSE_FACTORY_ENTRY(
          "thrust::stable_sort_by_key",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::stable_sort_by_key", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                              CALL(MapNames::getDpctNamespace() + "stable_sort",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2))))),
          CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                            CALL(MapNames::getDpctNamespace() + "stable_sort",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2)))),
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          // Handling case: thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
          CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                             CALL(MapNames::getDpctNamespace() + "stable_sort",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  ARG(0), ARG(1), ARG(2))),
          CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                             CALL(MapNames::getDpctNamespace() + "stable_sort",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2))))
      ),
      CONDITIONAL_FACTORY_ENTRY(
        makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
        CONDITIONAL_FACTORY_ENTRY(
          CompareArgType(0, 1),
          // Handling case: thrust::stable_sort_by_key(policy, ptr, ptr, ptr);
          IFELSE_FACTORY_ENTRY(
            "thrust::stable_sort_by_key",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
              CALL_FACTORY_ENTRY("thrust::stable_sort_by_key", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
              CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                                CALL(MapNames::getDpctNamespace() + "stable_sort",
                                      CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3))))),
            CALL_FACTORY_ENTRY("thrust::unique_by_key_copy",
                              CALL(MapNames::getDpctNamespace() + "stable_sort",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(1), ARG(2), ARG(3)))),
          //Handling case: thrust::stable_sort_by_key(ptr, ptr, ptr,pred);
          IFELSE_FACTORY_ENTRY(
            "thrust::stable_sort_by_key",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
              CALL_FACTORY_ENTRY("thrust::stable_sort_by_key", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
              CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                                CALL(MapNames::getDpctNamespace() + "stable_sort",
                                      CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                      ARG(3)))),
            CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                              CALL(MapNames::getDpctNamespace() + "stable_sort",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3))))),
        CONDITIONAL_FACTORY_ENTRY(
          CompareArgType(0, 1),
          // Handling case: thrust::stable_sort_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin());
          CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                             CALL(MapNames::getDpctNamespace() + "stable_sort",
                                  makeMappedThrustPolicyEnum(0),
                                  ARG(1), ARG(2), ARG(3))),
          CONDITIONAL_FACTORY_ENTRY(
            CheckArgType(1, "thrust::device_ptr"),
            //  Handling case: thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(),binary_pred);
            CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                               CALL(MapNames::getDpctNamespace() + "stable_sort",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    ARG(0), ARG(1), ARG(2), ARG(3))),
            CALL_FACTORY_ENTRY("thrust::stable_sort_by_key",
                               CALL(MapNames::getDpctNamespace() + "stable_sort",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3)))))
      )
    )
  )
)

// thrust::find
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(4),
  //Handling case: thrust::find(policy, device.begin(), device.end(), value);
  CALL_FACTORY_ENTRY("thrust::find",
                      CALL("oneapi::dpl::find",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3))),
  //Handling case: thrust::find(host.begin(), host.end(), value);
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgType(1, "thrust::device_ptr"),
    CALL_FACTORY_ENTRY("thrust::find",
                       CALL("oneapi::dpl::find",
                            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                            ARG(0), ARG(1), ARG(2))),
    CALL_FACTORY_ENTRY("thrust::find",
                       CALL("oneapi::dpl::find",
                            ARG("oneapi::dpl::execution::seq"),
                            ARG(0), ARG(1), ARG(2))))
)

// thrust::sort_by_key
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_sort,
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  //Handling case: thrust::sort_by_key(policy, device.begin(), device.end(), device.begin(), comp);
  CALL_FACTORY_ENTRY("thrust::sort_by_key",
                      CALL(MapNames::getDpctNamespace() + "sort",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4))),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(3),
    //Handling case: thrust::sort_by_key(host.begin(), host.end(), host.begin());
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(1, "thrust::device_ptr"),
      CALL_FACTORY_ENTRY("thrust::sort_by_key",
                       CALL(MapNames::getDpctNamespace() + "sort",
                            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                            ARG(0), ARG(1), ARG(2))),
      CALL_FACTORY_ENTRY("thrust::sort_by_key",
                       CALL(MapNames::getDpctNamespace() + "sort",
                            ARG("oneapi::dpl::execution::seq"),
                            ARG(0), ARG(1), ARG(2)))),
    CONDITIONAL_FACTORY_ENTRY(
      CompareArgType(0, 1),
      //Handling case: thrust::sort_by_key(policy, device.begin(), device.end(), device.begin());
      CALL_FACTORY_ENTRY("thrust::sort_by_key",
                      CALL(MapNames::getDpctNamespace() + "sort",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3))),
      //Handling case: thrust::sort_by_key(device.begin(), device.end(), device.begin(), comp);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::sort_by_key",
                       CALL(MapNames::getDpctNamespace() + "sort",
                            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                            ARG(0), ARG(1), ARG(2), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::sort_by_key",
                       CALL(MapNames::getDpctNamespace() + "sort",
                            ARG("oneapi::dpl::execution::seq"),
                            ARG(0), ARG(1), ARG(2), ARG(3))))))
))


// thrust::inner_product
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasNumeric_inner_product,
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(5),
  //Handling case: thrust::inner_product(policy, device.begin(), device.end(), device.begin(), init);
  CALL_FACTORY_ENTRY("thrust::inner_product",
                      CALL(MapNames::getDpctNamespace() + "inner_product",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4))),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(7),
    //Handling case: thrust::inner_product(policy, device.begin(), device.end(), device.begin(), init, binary_op1, binary_op2);
    CALL_FACTORY_ENTRY("thrust::inner_product",
                      CALL(MapNames::getDpctNamespace() + "inner_product",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgCount(4),
      //Handling case: thrust::inner_product(device.begin(), device.end(), device.begin(), init);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::inner_product",
                       CALL(MapNames::getDpctNamespace() + "inner_product",
                            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                            ARG(0), ARG(1), ARG(2), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::inner_product",
                       CALL(MapNames::getDpctNamespace() + "inner_product",
                            ARG("oneapi::dpl::execution::seq"),
                            ARG(0), ARG(1), ARG(2), ARG(3)))),
      //Handling case: thrust::inner_product(device.begin(), device.end(), device.begin(), init, binary_op1, binary_op2);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::inner_product",
                       CALL(MapNames::getDpctNamespace() + "inner_product",
                            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
        CALL_FACTORY_ENTRY("thrust::inner_product",
                       CALL(MapNames::getDpctNamespace() + "inner_product",
                            ARG("oneapi::dpl::execution::seq"),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))))
))

// thrust::reduce_by_key
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(8),
  //Handling case: thrust::reduce_by_key(policy, device.begin(), device.end(), device.begin(), device.end(), device.begin(), binary_pred, binary_op);
  CALL_FACTORY_ENTRY("thrust::reduce_by_key",
                      CALL("oneapi::dpl::reduce_by_segment",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7))),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(7),
    CONDITIONAL_FACTORY_ENTRY(
      CompareArgType(0, 1),
      //Handling case: thrust::reduce_by_key(policy, device.begin(), device.end(), device.begin(), device.end(), device.begin(), binary_pred);
      CALL_FACTORY_ENTRY("thrust::reduce_by_key",
                      CALL("oneapi::dpl::reduce_by_segment",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))),
      //Handling case: thrust::reduce_by_key(device.begin(), device.end(), device.begin(), device.end(), device.begin(), binary_pred, binary_op);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::reduce_by_key",
                       CALL("oneapi::dpl::reduce_by_segment",
                            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))),
        CALL_FACTORY_ENTRY("thrust::reduce_by_key",
                       CALL("oneapi::dpl::reduce_by_segment",
                            ARG("oneapi::dpl::execution::seq"),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgCount(6),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::reduce_by_key(policy, device.begin(), device.end(), device.begin(), device.end(), device.begin());
        CALL_FACTORY_ENTRY("thrust::reduce_by_key",
                        CALL("oneapi::dpl::reduce_by_segment",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
        //Handling case: thrust::reduce_by_key(device.begin(), device.end(), device.begin(), device.end(), device.begin(), binary_pred);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::reduce_by_key",
                         CALL("oneapi::dpl::reduce_by_segment",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
          CALL_FACTORY_ENTRY("thrust::reduce_by_key",
                         CALL("oneapi::dpl::reduce_by_segment",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))
      ),
      //Handling case: thrust::reduce_by_key(device.begin(), device.end(), device.begin(), device.end(), device.begin());
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::reduce_by_key",
                       CALL("oneapi::dpl::reduce_by_segment",
                            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))),
        CALL_FACTORY_ENTRY("thrust::reduce_by_key",
                       CALL("oneapi::dpl::reduce_by_segment",
                            ARG("oneapi::dpl::execution::seq"),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))))))
)

// thrust::for_each
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(4),
  //Handling case: thrust::for_each(policy, device.begin(), device.end(), f);
  CALL_FACTORY_ENTRY("thrust::for_each",
                      CALL("std::for_each",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), THRUST_FUNCTOR(3))),
  //Handling case: thrust::for_each(host.begin(), host.end(), f);
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgType(0, "thrust::device_ptr"),
    CALL_FACTORY_ENTRY("thrust::for_each",
                       CALL("std::for_each",
                            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                            ARG(0), ARG(1), THRUST_FUNCTOR(2))),
    CALL_FACTORY_ENTRY("thrust::for_each",
                       CALL("std::for_each",
                            ARG("oneapi::dpl::execution::seq"),
                            ARG(0), ARG(1), THRUST_FUNCTOR(2))))
)

// thrust::transform
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(6),
  //Handling case: thrust::transform(policy, device.begin(), device.end(), device.end(), device.end(), f);
  CALL_FACTORY_ENTRY("thrust::transform",
                      CALL("std::transform",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4), THRUST_FUNCTOR(5))),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(4),
    //Handling case: thrust::transform(host.begin(), host.end(), host.end(), f);
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(0, "thrust::device_ptr"),
      CALL_FACTORY_ENTRY("thrust::transform",
                         CALL("std::transform",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), THRUST_FUNCTOR(3))),
      CALL_FACTORY_ENTRY("thrust::transform",
                         CALL("std::transform",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), THRUST_FUNCTOR(3)))),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::transform(policy, device.begin(), device.end(), device.begin(), f);
        CALL_FACTORY_ENTRY("thrust::transform",
                        CALL("std::transform",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), THRUST_FUNCTOR(4))),
        //Handling case: thrust::transform(device.begin(), device.end(), device.begin(), device.end(), f);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::transform",
                         CALL("std::transform",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), ARG(3), THRUST_FUNCTOR(4))),
          CALL_FACTORY_ENTRY("thrust::transform",
                         CALL("std::transform",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(3), THRUST_FUNCTOR(4))))
      ))
)

// thrust::copy_if
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(6),
  //Handling case: thrust::copy_if(policy, device.begin(), device.end(), device.end(), device.end(), f);
  FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_copy_if,
  CALL_FACTORY_ENTRY("thrust::copy_if",
                      CALL(MapNames::getDpctNamespace() + "copy_if",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4), THRUST_FUNCTOR(5)))),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(4),
    //Handling case: thrust::copy_if(host.begin(), host.end(), host.end(), f);
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgType(0, "thrust::device_ptr"),
      CALL_FACTORY_ENTRY("thrust::copy_if",
                         CALL("std::copy_if",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), THRUST_FUNCTOR(3))),
      CALL_FACTORY_ENTRY("thrust::copy_if",
                         CALL("std::copy_if",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), THRUST_FUNCTOR(3)))),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::copy_if(policy, device.begin(), device.end(), device.begin(), f);
        CALL_FACTORY_ENTRY("thrust::copy_if",
                        CALL("std::copy_if",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), THRUST_FUNCTOR(4))),
        //Handling case: thrust::copy_if(device.begin(), device.end(), device.begin(), device.end(), f);
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_copy_if,
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::copy_if",
                         CALL(MapNames::getDpctNamespace() + "copy_if",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), ARG(3), THRUST_FUNCTOR(4))),
          CALL_FACTORY_ENTRY("thrust::copy_if",
                         CALL(MapNames::getDpctNamespace() + "copy_if",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(3), THRUST_FUNCTOR(4)))))
      ))
)

// thrust::make_zip_iterator
CALL_FACTORY_ENTRY("thrust::make_zip_iterator",  CALL("oneapi::dpl::make_zip_iterator",  ARG(0)))
// thrust::inclusive_scan_by_key
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(7),
  //Handling case: thrust::inclusive_scan_by_key(policy, device.begin(), device.end(), device.begin(), result, binary_pred, binary_op);
  CALL_FACTORY_ENTRY("thrust::inclusive_scan_by_key",
                      CALL("oneapi::dpl::inclusive_scan_by_segment",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(6),
    CONDITIONAL_FACTORY_ENTRY(
      CompareArgType(0, 1),
      //Handling case: thrust::inclusive_scan_by_key(policy, device.begin(), device.end(), device.begin(), result, binary_pred);
      CALL_FACTORY_ENTRY("thrust::inclusive_scan_by_key",
                      CALL("oneapi::dpl::inclusive_scan_by_segment",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
      //Handling case: thrust::inclusive_scan_by_key(device.begin(), device.end(), device.begin(), result, binary_pred, binary_op);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::inclusive_scan_by_key",
                       CALL("oneapi::dpl::inclusive_scan_by_segment",
                            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
        CALL_FACTORY_ENTRY("thrust::inclusive_scan_by_key",
                       CALL("oneapi::dpl::inclusive_scan_by_segment",
                            ARG("oneapi::dpl::execution::seq"),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgCount(5),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        //Handling case: thrust::inclusive_scan_by_key(policy, device.begin(), device.end(), device.begin(), result);
        CALL_FACTORY_ENTRY("thrust::inclusive_scan_by_key",
                        CALL("oneapi::dpl::inclusive_scan_by_segment",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4))),
        //Handling case: thrust::inclusive_scan_by_key(device.begin(), device.end(), device.begin(), result, binary_pred);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::inclusive_scan_by_key",
                         CALL("oneapi::dpl::inclusive_scan_by_segment",
                              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                              ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))),
          CALL_FACTORY_ENTRY("thrust::inclusive_scan_by_key",
                         CALL("oneapi::dpl::inclusive_scan_by_segment",
                              ARG("oneapi::dpl::execution::seq"),
                              ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))))
      ),
      //Handling case: thrust::inclusive_scan_by_key(device.begin(), device.end(), device.begin(), result);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::inclusive_scan_by_key",
                       CALL("oneapi::dpl::inclusive_scan_by_segment",
                            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                            ARG(0), ARG(1), ARG(2), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::inclusive_scan_by_key",
                       CALL("oneapi::dpl::inclusive_scan_by_segment",
                            ARG("oneapi::dpl::execution::seq"),
                            ARG(0), ARG(1), ARG(2), ARG(3))))))
)


// thrust::exclusive_scan_by_key
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(8),
  //Handling case: thrust::exclusive_scan_by_key(policy, device.begin(), device.end(), device.begin(), result, init, binary_pred, binary_op);
  CALL_FACTORY_ENTRY("thrust::exclusive_scan_by_key",
                      CALL("oneapi::dpl::exclusive_scan_by_segment",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7))),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(7),
    CONDITIONAL_FACTORY_ENTRY(
      CompareArgType(0, 1),
      //Handling case: thrust::exclusive_scan_by_key(policy, device.begin(), device.end(), device.begin(), result, init, binary_pred);
      CALL_FACTORY_ENTRY("thrust::exclusive_scan_by_key",
                      CALL("oneapi::dpl::exclusive_scan_by_segment",
                            makeMappedThrustPolicyEnum(0),
                            ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))),
      //Handling case: thrust::exclusive_scan_by_key(device.begin(), device.end(), device.begin(), result, init, binary_pred, binary_op);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::exclusive_scan_by_key",
                       CALL("oneapi::dpl::exclusive_scan_by_segment",
                            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))),
        CALL_FACTORY_ENTRY("thrust::exclusive_scan_by_key",
                       CALL("oneapi::dpl::exclusive_scan_by_segment",
                            ARG("oneapi::dpl::execution::seq"),
                            ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6))))
    ),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgCount(6),
        CONDITIONAL_FACTORY_ENTRY(
          CompareArgType(0, 1),
          //Handling case: thrust::exclusive_scan_by_key(policy, device.begin(), device.end(), device.begin(), result, init);
          CALL_FACTORY_ENTRY("thrust::exclusive_scan_by_key",
                          CALL("oneapi::dpl::exclusive_scan_by_segment",
                                makeMappedThrustPolicyEnum(0),
                                ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
          //Handling case: thrust::exclusive_scan_by_key(device.begin(), device.end(), device.begin(), result, init, binary_pred);
          CONDITIONAL_FACTORY_ENTRY(
            CheckArgType(1, "thrust::device_ptr"),
            CALL_FACTORY_ENTRY("thrust::exclusive_scan_by_key",
                           CALL("oneapi::dpl::exclusive_scan_by_segment",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))),
            CALL_FACTORY_ENTRY("thrust::exclusive_scan_by_key",
                           CALL("oneapi::dpl::exclusive_scan_by_segment",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5))))
        ),
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgCount(5),
          CONDITIONAL_FACTORY_ENTRY(
            CompareArgType(0, 1),
            //Handling case: thrust::exclusive_scan_by_key(policy, device.begin(), device.end(), device.begin(), result);
            CALL_FACTORY_ENTRY("thrust::exclusive_scan_by_key",
                            CALL("oneapi::dpl::exclusive_scan_by_segment",
                                  makeMappedThrustPolicyEnum(0),
                                  ARG(1), ARG(2), ARG(3), ARG(4))),
            //Handling case: thrust::exclusive_scan_by_key(device.begin(), device.end(), device.begin(), result, init);
            CONDITIONAL_FACTORY_ENTRY(
              CheckArgType(1, "thrust::device_ptr"),
              CALL_FACTORY_ENTRY("thrust::exclusive_scan_by_key",
                             CALL("oneapi::dpl::exclusive_scan_by_segment",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))),
              CALL_FACTORY_ENTRY("thrust::exclusive_scan_by_key",
                             CALL("oneapi::dpl::exclusive_scan_by_segment",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))))
          ),
          //Handling case: thrust::exclusive_scan_by_key(device.begin(), device.end(), device.begin(), result);
          CONDITIONAL_FACTORY_ENTRY(
            CheckArgType(1, "thrust::device_ptr"),
            CALL_FACTORY_ENTRY("thrust::exclusive_scan_by_key",
                           CALL("oneapi::dpl::exclusive_scan_by_segment",
                                CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                ARG(0), ARG(1), ARG(2), ARG(3))),
            CALL_FACTORY_ENTRY("thrust::exclusive_scan_by_key",
                           CALL("oneapi::dpl::exclusive_scan_by_segment",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2), ARG(3)))))
        )
    )
)

// thrust::make_reverse_iterator
CALL_FACTORY_ENTRY("thrust::make_reverse_iterator",  CALL("oneapi::dpl::make_reverse_iterator",  ARG(0)))

// thrust::exclusive_scan
CONDITIONAL_FACTORY_ENTRY(
  CheckArgCount(3),
  // thrust::exclusive_scan(data, data + 6, data)
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgType(1, "thrust::device_ptr"),
    CALL_FACTORY_ENTRY("thrust::exclusive_scan",
      CALL("std::exclusive_scan",
        CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
        ARG(0), ARG(1), ARG(2), ARG("0"))),
    CALL_FACTORY_ENTRY("thrust::exclusive_scan",
      CALL("std::exclusive_scan",
        ARG("oneapi::dpl::execution::seq"), ARG(0), ARG(1), ARG(2), ARG("0")))
  ),
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(4),
    CONDITIONAL_FACTORY_ENTRY(
      CompareArgType(0, 1),
      // thrust::exclusive_scan(thrust::host, data, data + 6, data);
      CALL_FACTORY_ENTRY("thrust::exclusive_scan",
        CALL("std::exclusive_scan",
          makeMappedThrustPolicyEnum(0), ARG(1), ARG(2), ARG(3), ARG("0"))),
      // thrust::exclusive_scan(data, data + 6, data, 4);
      CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "thrust::device_ptr"),
        CALL_FACTORY_ENTRY("thrust::exclusive_scan",
          CALL("std::exclusive_scan",
            CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
            ARG(0), ARG(1), ARG(2), ARG(3))),
        CALL_FACTORY_ENTRY("thrust::exclusive_scan",
          CALL("std::exclusive_scan",
            ARG("oneapi::dpl::execution::seq"), ARG(0), ARG(1), ARG(2), ARG(3)))
      )
    ),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgCount(5),
      CONDITIONAL_FACTORY_ENTRY(
        CompareArgType(0, 1),
        // thrust::exclusive_scan(thrust::host, data, data + 6, data, 4);
        CALL_FACTORY_ENTRY("thrust::exclusive_scan",
          CALL("std::exclusive_scan",
            makeMappedThrustPolicyEnum(0), ARG(1), ARG(2), ARG(3), ARG(4))),
        // thrust::exclusive_scan(data, data + 10, data, 1, binary_op);
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          CALL_FACTORY_ENTRY("thrust::exclusive_scan",
            CALL("std::exclusive_scan",
              CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
              ARG(0), ARG(1), ARG(2), ARG(3), ARG(4))),
          CALL_FACTORY_ENTRY("thrust::exclusive_scan",
            CALL("std::exclusive_scan",
              ARG("oneapi::dpl::execution::seq"), ARG(0), ARG(1), ARG(2), ARG(3), ARG(4)))
        )
      ),
      // thrust::exclusive_scan(thrust::host, data, data + 10, data, 1, binary_op);
      CALL_FACTORY_ENTRY("thrust::exclusive_scan",
        CALL("std::exclusive_scan",
          makeMappedThrustPolicyEnum(0), ARG(1), ARG(2), ARG(3), ARG(4), ARG(5)))
    )
  )
)

// thrust::get
CALL_FACTORY_ENTRY(
    "thrust::get",
    CALL(TEMPLATED_CALLEE("std::get", std::vector<size_t>(1, 0)), ARG(0)))

// thrust::unique_by_key
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasAlgorithm_unique_copy,
  CONDITIONAL_FACTORY_ENTRY(
    CheckArgCount(5),
    CONDITIONAL_FACTORY_ENTRY(
      makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
      //Handling case: thrust::unique_by_key(policy, ptr, ptr,  ptr, pred);
      IFELSE_FACTORY_ENTRY(
        "thrust::unique_by_key",
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
          CALL_FACTORY_ENTRY("thrust::unique_by_key", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
          CALL_FACTORY_ENTRY("thrust::unique_by_key",
                            CALL("dpct::unique",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                  CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3)),
                                  ARG(4)))),
        CALL_FACTORY_ENTRY("thrust::unique_by_key",
                          CALL("dpct::unique",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(1), ARG(2), ARG(3), ARG(4)))),
      // Handling case: thrust::unique_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin(), d_output_keys.begin(), d_output_values.begin(), binary_pred);
      CALL_FACTORY_ENTRY("thrust::unique_by_key",
                        CALL("dpct::unique",
                              makeMappedThrustPolicyEnum(0),
                              ARG(1), ARG(2), ARG(3), ARG(4)))
    ),
    CONDITIONAL_FACTORY_ENTRY(
      CheckArgCount(3),
      CONDITIONAL_FACTORY_ENTRY(
        makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
        //Handling case: thrust::unique_by_key(ptr, ptr, ptr);
        IFELSE_FACTORY_ENTRY(
          "thrust::unique_by_key",
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
            CALL_FACTORY_ENTRY("thrust::unique_by_key", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
          FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
            CALL_FACTORY_ENTRY("thrust::unique_by_key",
                              CALL("dpct::unique",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                    CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2))))),
          CALL_FACTORY_ENTRY("thrust::unique_by_key",
                            CALL("dpct::unique",
                                  ARG("oneapi::dpl::execution::seq"),
                                  ARG(0), ARG(1), ARG(2)))),
        CONDITIONAL_FACTORY_ENTRY(
          CheckArgType(1, "thrust::device_ptr"),
          // Handling case: thrust::unique_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
          CALL_FACTORY_ENTRY("thrust::unique_by_key",
                             CALL("dpct::unique",
                                  CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                  ARG(0), ARG(1), ARG(2))),
          CALL_FACTORY_ENTRY("thrust::unique_by_key",
                             CALL("dpct::unique",
                                ARG("oneapi::dpl::execution::seq"),
                                ARG(0), ARG(1), ARG(2))))
      ),
      CONDITIONAL_FACTORY_ENTRY(
        makeCheckAnd(CheckIsPtr(1), makeCheckNot(checkIsUSM())),
        CONDITIONAL_FACTORY_ENTRY(
          CompareArgType(0, 1),
          //Handling case: thrust::unique_by_key(policy, ptr, ptr, ptr);
          IFELSE_FACTORY_ENTRY(
            "thrust::unique_by_key",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
              CALL_FACTORY_ENTRY("thrust::unique_by_key", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
              CALL_FACTORY_ENTRY("thrust::unique_by_key",
                                CALL("dpct::unique",
                                      CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(3)), ARG(3))))),
            CALL_FACTORY_ENTRY("thrust::unique_by_key",
                              CALL("dpct::unique",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(1), ARG(2), ARG(3)))),
          //Handling case: thrust::unique_by_key(ptr, ptr, ptr, pred);
          IFELSE_FACTORY_ENTRY(
            "thrust::unique_by_key",
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_is_device_ptr,
              CALL_FACTORY_ENTRY("thrust::unique_by_key", CALL(MapNames::getDpctNamespace() + "is_device_ptr", ARG(1)))),
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasMemory_device_pointer_forward_decl,
              CALL_FACTORY_ENTRY("thrust::unique_by_key",
                                CALL("dpct::unique",
                                      CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(0)), ARG(0)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(1)), ARG(1)),
                                      CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() + "device_pointer", getDerefedType(2)), ARG(2)),
                                      ARG(3)))),
            CALL_FACTORY_ENTRY("thrust::unique_by_key",
                              CALL("dpct::unique",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3))))),
        CONDITIONAL_FACTORY_ENTRY(
          CompareArgType(0, 1),
          // Handling case: thrust::unique_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin());
          CALL_FACTORY_ENTRY("thrust::unique_by_key",
                             CALL("dpct::unique",
                                  makeMappedThrustPolicyEnum(0),
                                  ARG(1), ARG(2), ARG(3))),
          CONDITIONAL_FACTORY_ENTRY(
            CheckArgType(1, "thrust::device_ptr"),
            // Handling case: thrust::unique_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), binary_pred);
            CALL_FACTORY_ENTRY("thrust::unique_by_key",
                               CALL("dpct::unique",
                                    CALL("oneapi::dpl::execution::make_device_policy", QUEUESTR),
                                    ARG(0), ARG(1), ARG(2), ARG(3))),
            CALL_FACTORY_ENTRY("thrust::unique_by_key",
                               CALL("dpct::unique",
                                    ARG("oneapi::dpl::execution::seq"),
                                    ARG(0), ARG(1), ARG(2), ARG(3)))))
      )
    )
  )
)

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
// clang-format on

} // namespace dpct
} // namespace clang
