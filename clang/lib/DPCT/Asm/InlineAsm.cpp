//===---------------------------- InlineAsm.cpp -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InlineAsm.h"

using namespace clang::dpct;

InlineAsmType::~InlineAsmType() = default;
InlineAsmDecl::~InlineAsmDecl() = default;
// PTXNamedDecl::~PTXNamedDecl() = default;
InlineAsmStmt::~InlineAsmStmt() = default;
