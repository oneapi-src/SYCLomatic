//===---APINamesTexture.inc ---------------------------------------------===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
ENTRY_TEMPLATED("cudaCreateChannelDesc", "dpct::create_image_channel")
ENTRY_RENAMED("cudaCreateChannelDescHalf", "dpct::create_image_channel<" + MapNames::getClNamespace() + "::cl_half>")

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
	POINTER_CHECKER(0),
    MEMBER_CALL_FACTORY_ENTRY("cudaUnbindTexture", DEREF(0), true, "detach"),
    MEMBER_CALL_FACTORY_ENTRY("cudaUnbindTexture", ARG(0), false, "detach")))

ASSIGNABLE_FACTORY(DELETER_FACTORY_ENTRY("cudaDestroyTextureObject", ARG(0)))

ASSIGNABLE_FACTORY(
    ASSIGN_FACTORY_ENTRY("cudaGetTextureObjectResourceDesc", DEREF(0),
                         MEMBER_CALL(ARG(1), true, "get_data")))
ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY("cudaGetTextureObjectTextureDesc",
                                        DEREF(0),
                                        MEMBER_CALL(ARG(1), true, "get_info")))

ENTRY_TEXTURE("tex1D", 0x01, 1)
ENTRY_TEXTURE("tex2D", 0x02, 1, 2)
ENTRY_TEXTURE("tex3D", 0x03, 1, 2, 3)
ENTRY_TEXTURE("tex1Dfetch", 0x01, 1)
ENTRY_TEXTURE("tex1DLayered", 0xF1, 2, 1)
ENTRY_TEXTURE("tex2DLayered", 0xF2, 3, 1, 2)

ASSIGNABLE_FACTORY(ENTRY_BIND("cudaBindTexture", 1 /*start arg idx*/,
                              2 /*size in bytes*/))
ASSIGNABLE_FACTORY(ENTRY_BIND("cudaBindTexture2D", 1 /*start arg idx*/,
                              2 /*size in dimension x*/,
                              3 /*size in dimension y*/, 4 /*pitch*/))
ASSIGNABLE_FACTORY(ENTRY_BIND("cudaBindTextureToArray", 0 /*start arg idx*/))

ASSIGNABLE_FACTORY(ENTRY_REORDER("cudaCreateTextureObject",
                                 "dpct::create_image",
                                 0 /*ref of texture object*/,
                                 1 /*ref of data desc*/,
                                 2 /*ref of info desc*/))

ENTRY_UNSUPPORTED("cudaGetTextureObjectResourceViewDesc",
                  Diagnostics::NOTSUPPORTED)
