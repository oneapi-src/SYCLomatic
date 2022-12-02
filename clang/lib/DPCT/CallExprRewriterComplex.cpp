//===--------------- CallExprRewriterComplex.cpp --------------------------===//
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

// clang-format off
void CallExprRewriterFactoryBase::initRewriterMapComplex() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
CALL_FACTORY_ENTRY("make_cuComplex", CALL(MapNames::getClNamespace() + "float2",
                                          makeCallArgCreatorWithCall(0),
                                          makeCallArgCreatorWithCall(1)))
CALL_FACTORY_ENTRY("make_cuFloatComplex",
                   CALL(MapNames::getClNamespace() + "float2",
                        makeCallArgCreatorWithCall(0),
                        makeCallArgCreatorWithCall(1)))
CALL_FACTORY_ENTRY("make_cuDoubleComplex",
                   CALL(MapNames::getClNamespace() + "double2",
                        makeCallArgCreatorWithCall(0),
                        makeCallArgCreatorWithCall(1)))
CALL_FACTORY_ENTRY("__saturatef",
                   CALL(MapNames::getClNamespace() + "clamp<float>",
                        makeCallArgCreatorWithCall(0),
                        makeCallArgCreator("0.0f"),
                        makeCallArgCreator("1.0f")))
MEMBER_CALL_FACTORY_ENTRY("cuCreal", makeDerefArgCreatorWithCall(0), false, "x")
MEMBER_CALL_FACTORY_ENTRY("cuCimag", makeDerefArgCreatorWithCall(0), false, "y")
MEMBER_CALL_FACTORY_ENTRY("cuCrealf", makeDerefArgCreatorWithCall(0), false,
                          "x")
MEMBER_CALL_FACTORY_ENTRY("cuCimagf", makeDerefArgCreatorWithCall(0), false,
                          "y")
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_cabs,
                        CALL_FACTORY_ENTRY("cuCabs",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "cabs<double>",
                                                makeCallArgCreatorWithCall(0))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_conj,
                        CALL_FACTORY_ENTRY("cuConj",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "conj<double>",
                                                makeCallArgCreatorWithCall(0))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_cabs,
                        CALL_FACTORY_ENTRY("cuCabsf",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "cabs<float>",
                                                makeCallArgCreatorWithCall(0))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_conj,
                        CALL_FACTORY_ENTRY("cuConjf",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "conj<float>",
                                                makeCallArgCreatorWithCall(0))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_cmul,
                        CALL_FACTORY_ENTRY("cuCmul",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "cmul<double>",
                                                makeCallArgCreatorWithCall(0),
                                                makeCallArgCreatorWithCall(1))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_cdiv,
                        CALL_FACTORY_ENTRY("cuCdiv",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "cdiv<double>",
                                                makeCallArgCreatorWithCall(0),
                                                makeCallArgCreatorWithCall(1))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_cmul,
                        CALL_FACTORY_ENTRY("cuCmulf",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "cmul<float>",
                                                makeCallArgCreatorWithCall(0),
                                                makeCallArgCreatorWithCall(1))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_cdiv,
                        CALL_FACTORY_ENTRY("cuCdivf",
                                           CALL(MapNames::getDpctNamespace() +
                                                    "cdiv<float>",
                                                makeCallArgCreatorWithCall(0),
                                                makeCallArgCreatorWithCall(1))))
BINARY_OP_FACTORY_ENTRY("cuCadd", BinaryOperatorKind::BO_Add,
                        makeCallArgCreatorWithCall(0),
                        makeCallArgCreatorWithCall(1))
BINARY_OP_FACTORY_ENTRY("cuCsub", BinaryOperatorKind::BO_Sub,
                        makeCallArgCreatorWithCall(0),
                        makeCallArgCreatorWithCall(1))
BINARY_OP_FACTORY_ENTRY("cuCaddf", BinaryOperatorKind::BO_Add,
                        makeCallArgCreatorWithCall(0),
                        makeCallArgCreatorWithCall(1))
BINARY_OP_FACTORY_ENTRY("cuCsubf", BinaryOperatorKind::BO_Sub,
                        makeCallArgCreatorWithCall(0),
                        makeCallArgCreatorWithCall(1))

      }));
}
// clang-format on

} // namespace dpct
} // namespace clang
