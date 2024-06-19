//===--------------- APINamesGraph.inc ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

CONDITIONAL_FACTORY_ENTRY(
    UseExtGraph, ENTRY_TEXTURE("tex1DLod", 0x11, 1),
    UNSUPPORT_FACTORY_ENTRY("tex1DLod", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("tex1DLod"),
                            ARG("--use-experimental-features=bindless_images")))