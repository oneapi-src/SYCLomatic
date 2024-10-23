//===--------------- FFTAPIMigration.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_FFT_API_MIGRATION_H
#define DPCT_FFT_API_MIGRATION_H

#include "MapNames.h"
#include "TextModification.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"

namespace clang {
namespace dpct {
TextModification *processFunctionPointer(const UnaryOperator *UO);
} // namespace dpct
} // namespace clang

#endif // !DPCT_FFT_API_MIGRATION_H
