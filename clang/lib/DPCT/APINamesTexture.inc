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
ENTRY_RENAMED("cudaCreateChannelDesc", "dpct::create_image_channel")
ENTRY_RENAMED("cudaCreateChannelDescHalf", "dpct::create_image_channel<" + MapNames::getClNamespace() + "::cl_half>")
ENTRY_RENAMED_ISASSIGNED("cudaUnbindTexture", "dpct::detach_image")
ENTRY_RENAMED_ISASSIGNED("cudaDestroyTextureObject", "dpct::dpct_free")
ENTRY_RENAMED_ISASSIGNED("cudaGetTextureObjectResourceDesc", "dpct::get_image_data")
ENTRY_RENAMED_ISASSIGNED("cudaGetTextureObjectTextureDesc", "dpct::get_image_info")
ENTRY_TEXTURE("tex1D", "dpct::read_image")
ENTRY_TEXTURE("tex2D", "dpct::read_image")
ENTRY_TEXTURE("tex3D", "dpct::read_image")
ENTRY_TEXTURE("tex1Dfetch", "dpct::read_image")
ENTRY_REORDER_ISASSIGNED("cudaBindTexture", "dpct::attach_image",
              1 /*ref to tex object*/, 2 /*source pointer*/,
              3 /*ref to channel desc*/, 4 /*size in bytes*/)
ENTRY_REORDER_ISASSIGNED("cudaBindTextureToArray", "dpct::attach_image",
              0 /*ref of texture object*/, 1 /*ref of array*/)
ENTRY_REORDER_ISASSIGNED("cudaCreateTextureObject", "dpct::create_image",
              0 /*ref of texture object*/, 1 /*ref of data desc*/,
              2 /*ref of info desc*/)
ENTRY_UNSUPPORTED("cudaGetTextureObjectResourceViewDesc",
                  Diagnostics::NOTSUPPORTED)
