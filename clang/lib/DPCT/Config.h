//===--------------- Config.h ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_CONFIG_H
#define DPCT_CONFIG_H

#include "clang/Basic/Version.inc"

#define STRINGIFY_(NUM) #NUM
#define STRINGIFY(NUM) STRINGIFY_(NUM)

#define TOOL_NAME "dpct"
#define DPCT_VERSION_MAJOR STRINGIFY(CLANG_VERSION_MAJOR)
#define DPCT_VERSION_MINOR STRINGIFY(CLANG_VERSION_MINOR)
#define DPCT_VERSION_PATCH STRINGIFY(CLANG_VERSION_PATCHLEVEL)

#endif
