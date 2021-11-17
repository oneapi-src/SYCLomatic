//===---APINamesTexture.inc ---------------------------------------------===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
CONDITIONAL_FACTORY_ENTRY(
    [](const CallExpr *C) -> bool { return C->getNumArgs(); },
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_channel,
                            ENTRY_RENAMED("cudaCreateChannelDesc",
                                          MapNames::getDpctNamespace() +
                                              "image_channel")),
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Image_image_channel_create,
        ENTRY_TEMPLATED("cudaCreateChannelDesc",
                        TEMPLATED_CALLEE(MapNames::getDpctNamespace() +
                                             "image_channel::create",
                                         {0}))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_channel_create,
                        ENTRY_RENAMED("cudaCreateChannelDescHalf",
                                      MapNames::getDpctNamespace() +
                                          "image_channel::create<" +
                                          MapNames::getClNamespace() +
                                          "half>"))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_wrapper,
                        ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
                            POINTER_CHECKER(0),
                            MEMBER_CALL_FACTORY_ENTRY("cudaUnbindTexture",
                                                      DEREF(0), true, "detach"),
                            MEMBER_CALL_FACTORY_ENTRY(
                                "cudaUnbindTexture", ARG(0), false, "detach"))))

ASSIGNABLE_FACTORY(DELETER_FACTORY_ENTRY("cudaDestroyTextureObject", ARG(0)))
ASSIGNABLE_FACTORY(DELETER_FACTORY_ENTRY("cuTexObjectDestroy", ARG(0)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_wrapper_base_get_data,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cudaGetTextureObjectResourceDesc", DEREF(0),
                            MEMBER_CALL(ARG(1), true, "get_data"))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Image_image_wrapper_base_get_sampling_info,
    ASSIGNABLE_FACTORY(
        ASSIGN_FACTORY_ENTRY("cudaGetTextureObjectTextureDesc", DEREF(0),
                             MEMBER_CALL(ARG(1), true, "get_sampling_info"))))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_wrapper_base_get_data,
                        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
                            "cuTexObjectGetResourceDesc", DEREF(0),
                            MEMBER_CALL(ARG(1), true, "get_data"))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Image_image_wrapper_base_get_sampling_info,
    ASSIGNABLE_FACTORY(
        ASSIGN_FACTORY_ENTRY("cuTexObjectGetTextureDesc", DEREF(0),
                             MEMBER_CALL(ARG(1), true, "get_sampling_info"))))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_accessor_ext,
                        ENTRY_TEXTURE("tex1D", 0x01, 1))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_accessor_ext,
                        ENTRY_TEXTURE("tex2D", 0x02, 1, 2))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_accessor_ext,
                        ENTRY_TEXTURE("tex3D", 0x03, 1, 2, 3))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_accessor_ext,
                        ENTRY_TEXTURE("tex1Dfetch", 0x01, 1))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_accessor_ext,
                        ENTRY_TEXTURE("tex1DLayered", 0xF1, 2, 1))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_accessor_ext,
                        ENTRY_TEXTURE("tex2DLayered", 0xF2, 3, 1, 2))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_wrapper_attach,
                        ASSIGNABLE_FACTORY(ENTRY_BIND("cudaBindTexture",
                                                      1 /*start arg idx*/,
                                                      2 /*size in bytes*/)))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Image_image_wrapper_attach,
    ASSIGNABLE_FACTORY(ENTRY_BIND("cudaBindTexture2D", 1 /*start arg idx*/,
                                  2 /*size in dimension x*/,
                                  3 /*size in dimension y*/, 4 /*pitch*/)))
FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Image_image_wrapper_attach,
                        ASSIGNABLE_FACTORY(ENTRY_BIND("cudaBindTextureToArray",
                                                      0 /*start arg idx*/)))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Image_create_image_wrapper,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cudaCreateTextureObject", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                     "create_image_wrapper",
                                                 DEREF(1), DEREF(2)))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Image_create_image_wrapper,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cuTexObjectCreate", DEREF(0),
                                            CALL(MapNames::getDpctNamespace() +
                                                     "create_image_wrapper",
                                                 DEREF(1), DEREF(2)))))

ENTRY_UNSUPPORTED("cudaGetTextureObjectResourceViewDesc",
                  Diagnostics::NOTSUPPORTED)

CONDITIONAL_FACTORY_ENTRY(
    CheckWarning1073(1),
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Image_image_matrix,
        ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
            "cuArrayCreate_v2", DEREF(0),
            NEW(MapNames::getDpctNamespace() + "image_matrix",
                STRUCT_DISMANTLE(1, "channel_type_ct1", "channel_num_ct1",
                                 "x_ct1", "y_ct1"))))),
    UNSUPPORT_FACTORY_ENTRY("cuArrayCreate_v2",
                            Diagnostics::CANNOT_CAPUTURE_AGUMENTS, ARG(1)))
ASSIGNABLE_FACTORY(DELETER_FACTORY_ENTRY("cuArrayDestroy", ARG(0)))
ENTRY_UNSUPPORTED("cuTexObjectGetResourceViewDesc", Diagnostics::NOTSUPPORTED)
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Image_image_data,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Image_image_wrapper_attach,
        ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY(
            "cuTexRefSetArray", ARG(0), true, "attach",
            CALL(MapNames::getDpctNamespace() + "image_data", ARG(1))))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Image_image_wrapper_base_set_addressing_mode,
    ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cuTexRefSetAddressMode",
                                                 ARG(0), true, "set", ARG(2))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Image_image_wrapper_base_set_filtering_mode,
    ASSIGNABLE_FACTORY(MEMBER_CALL_FACTORY_ENTRY("cuTexRefSetFilterMode",
                                                 ARG(0), true, "set", ARG(1))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Image_image_wrapper_base_set_channel_type,
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Image_image_wrapper_base_set_channel_num,
        MULTI_STMTS_FACTORY_ENTRY(
            "cuTexRefSetFormat",
            MEMBER_CALL(ARG(0), true, "set_channel_type", ARG(1)),
            MEMBER_CALL(ARG(0), true, "set_channel_num", ARG(2)))))
