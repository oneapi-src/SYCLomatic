//===---APINamesComplex.inc--------------------------------------------===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

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
MEMBER_CALL_FACTORY_ENTRY("cuCreal", makeDerefArgCreatorWithCall(0), false, "x")
MEMBER_CALL_FACTORY_ENTRY("cuCimag", makeDerefArgCreatorWithCall(0), false, "y")
MEMBER_CALL_FACTORY_ENTRY("cuCrealf", makeDerefArgCreatorWithCall(0), false,
                          "x")
MEMBER_CALL_FACTORY_ENTRY("cuCimagf", makeDerefArgCreatorWithCall(0), false,
                          "y")
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_cabs,
                        CALL_FACTORY_ENTRY("cuCabs",
                                           CALL(MapNames::getC2SNamespace() +
                                                    "cabs<double>",
                                                makeCallArgCreatorWithCall(0))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_conj,
                        CALL_FACTORY_ENTRY("cuConj",
                                           CALL(MapNames::getC2SNamespace() +
                                                    "conj<double>",
                                                makeCallArgCreatorWithCall(0))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_cabs,
                        CALL_FACTORY_ENTRY("cuCabsf",
                                           CALL(MapNames::getC2SNamespace() +
                                                    "cabs<float>",
                                                makeCallArgCreatorWithCall(0))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_conj,
                        CALL_FACTORY_ENTRY("cuConjf",
                                           CALL(MapNames::getC2SNamespace() +
                                                    "conj<float>",
                                                makeCallArgCreatorWithCall(0))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_cmul,
                        CALL_FACTORY_ENTRY("cuCmul",
                                           CALL(MapNames::getC2SNamespace() +
                                                    "cmul<double>",
                                                makeCallArgCreatorWithCall(0),
                                                makeCallArgCreatorWithCall(1))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_cdiv,
                        CALL_FACTORY_ENTRY("cuCdiv",
                                           CALL(MapNames::getC2SNamespace() +
                                                    "cdiv<double>",
                                                makeCallArgCreatorWithCall(0),
                                                makeCallArgCreatorWithCall(1))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_cmul,
                        CALL_FACTORY_ENTRY("cuCmulf",
                                           CALL(MapNames::getC2SNamespace() +
                                                    "cmul<float>",
                                                makeCallArgCreatorWithCall(0),
                                                makeCallArgCreatorWithCall(1))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Util_cdiv,
                        CALL_FACTORY_ENTRY("cuCdivf",
                                           CALL(MapNames::getC2SNamespace() +
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
