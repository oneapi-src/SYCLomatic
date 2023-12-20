//===--------------- APINamesTexture.inc ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
CONDITIONAL_FACTORY_ENTRY(
    [](const CallExpr *C) -> bool { return C->getNumArgs(); },
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                            ENTRY_RENAMED("cudaCreateChannelDesc",
                                          MapNames::getDpctNamespace() +
                                              "image_channel")),
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        ENTRY_TEMPLATED("cudaCreateChannelDesc",
                        TEMPLATED_CALLEE(MapNames::getDpctNamespace() +
                                             "image_channel::create",
                                         std::vector<size_t>(1, 0)))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ENTRY_RENAMED("cudaCreateChannelDescHalf",
                                      MapNames::getDpctNamespace() +
                                          "image_channel::create<" +
                                          MapNames::getClNamespace() +
                                          "half>"))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ENTRY_RENAMED("cudaCreateChannelDescHalf1",
                                      MapNames::getDpctNamespace() +
                                          "image_channel::create<" +
                                          MapNames::getClNamespace() +
                                          "half>"))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ENTRY_RENAMED("cudaCreateChannelDescHalf2",
                                      MapNames::getDpctNamespace() +
                                          "image_channel::create<" +
                                          MapNames::getClNamespace() +
                                          "half2>"))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ENTRY_RENAMED("cudaCreateChannelDescHalf4",
                                      MapNames::getDpctNamespace() +
                                          "image_channel::create<" +
                                          MapNames::getClNamespace() +
                                          "half4>"))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                            POINTER_CHECKER(0),
                            MEMBER_CALL_FACTORY_ENTRY("cudaUnbindTexture",
                                                      DEREF(0), true, "detach"),
                            MEMBER_CALL_FACTORY_ENTRY(
                                "cudaUnbindTexture", ARG(0), false, "detach"))))

ASSIGNABLE_FACTORY(DELETER_FACTORY_ENTRY("cudaDestroyTextureObject", ARG(0)))
ASSIGNABLE_FACTORY(DELETER_FACTORY_ENTRY("cuTexObjectDestroy", ARG(0)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cudaGetTextureObjectResourceDesc", DEREF(0),
                            MEMBER_CALL(ARG(1), true, "get_data"))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(
        ASSIGN_FACTORY_ENTRY("cudaGetTextureObjectTextureDesc", DEREF(0),
                             MEMBER_CALL(ARG(1), true, "get_sampling_info"))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cuTexObjectGetResourceDesc", DEREF(0),
                            MEMBER_CALL(ARG(1), true, "get_data"))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(
        ASSIGN_FACTORY_ENTRY("cuTexObjectGetTextureDesc", DEREF(0),
                             MEMBER_CALL(ARG(1), true, "get_sampling_info"))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ENTRY_TEXTURE("tex1D", 0x01, 1))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ENTRY_TEXTURE("tex2D", 0x02, 1, 2))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ENTRY_TEXTURE("tex3D", 0x03, 1, 2, 3))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        WARNING_FACTORY_ENTRY("tex1Dfetch",
                            ENTRY_TEXTURE("tex1Dfetch", 0x01, 1),
                        Diagnostics::TEX_FETCH))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ENTRY_TEXTURE("tex1DLayered", 0xF1, 2, 1))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ENTRY_TEXTURE("tex2DLayered", 0xF2, 3, 1, 2))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ASSIGNABLE_FACTORY(ENTRY_BIND("cudaBindTexture",
                                                      1 /*start arg idx*/,
                                                      2 /*size in bytes*/)))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(ENTRY_BIND("cudaBindTexture2D", 1 /*start arg idx*/,
                                  2 /*size in dimension x*/,
                                  3 /*size in dimension y*/, 4 /*pitch*/)))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ASSIGNABLE_FACTORY(ENTRY_BIND("cudaBindTextureToArray",
                                                      0 /*start arg idx*/)))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cudaCreateTextureObject", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                     "create_image_wrapper",
                                                 DEREF(1), DEREF(2)))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuTexObjectCreate", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                     "create_image_wrapper",
                                                 DEREF(1), DEREF(2)))))

ENTRY_UNSUPPORTED("cudaGetTextureObjectResourceViewDesc",
                  Diagnostics::API_NOT_MIGRATED)

CONDITIONAL_FACTORY_ENTRY(
    CheckWarning1073(1),
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::device_ext,
        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
            "cuArrayCreate_v2", DEREF(0),
            NEW(MapNames::getDpctNamespace() + "image_matrix",
                STRUCT_DISMANTLE(1, "channel_type_ct1", "channel_num_ct1",
                                 "x_ct1", "y_ct1"))))),
    UNSUPPORT_FACTORY_ENTRY("cuArrayCreate_v2",
                            Diagnostics::CANNOT_CAPUTURE_AGUMENTS, ARG(1)))
ASSIGNABLE_FACTORY(DELETER_FACTORY_ENTRY("cuArrayDestroy", ARG(0)))
ENTRY_UNSUPPORTED("cuTexObjectGetResourceViewDesc", Diagnostics::API_NOT_MIGRATED)
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
            "cuTexRefSetArray", ARG(0), true, "attach",
            CALL(MapNames::getDpctNamespace() + "image_data", ARG(1)))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cuTexRefSetAddressMode",
                                                 ARG(0), true, "set", ARG(2))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cuTexRefSetFilterMode",
                                                 ARG(0), true, "set", ARG(1))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cuTexRefGetAddressMode", DEREF(0),
        MEMBER_CALL(ARG(1), true, "get_addressing_mode"))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuTexRefGetFilterMode", DEREF(0),
                                            MEMBER_CALL(ARG(1), true,
                                                        "get_filtering_mode"))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cuTexRefGetFlags", DEREF(0),
        BO(BinaryOperatorKind::BO_Shl,
           MEMBER_CALL(ARG(1), true, "is_coordinate_normalized"), ARG("1")))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cuTexRefSetAddress_v2", ARG(1), true, "attach",
                            ARG(2), ARG(3))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
                        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
                            "cuTexRefSetAddress2D_v3", ARG(0), true, "attach",
                            ARG(2), STRUCT_DISMANTLE(1, "x_ct1", "y_ct1"),
                            ARG(3))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    MULTI_STMTS_FACTORY_ENTRY(
        "cuTexRefSetFormat", true, false, true, true,
        MEMBER_CALL(ARG(0), true, "set_channel_type", ARG(1)),
        MEMBER_CALL(ARG(0), true, "set_channel_num", ARG(2))))
