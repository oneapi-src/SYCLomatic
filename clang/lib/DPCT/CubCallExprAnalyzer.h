//===--------------- CubCallExprAnalyzer.h --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_CUB_REDUNDANT_CALL_ANALYZER_H
#define DPCT_CUB_REDUNDANT_CALL_ANALYZER_H

#include "clang/AST/Expr.h"

namespace clang {
namespace dpct {

/// Pseudo code:
/// loop_1 {
///   ...
///   tempstorage = nullptr;
///   ...
///   loop_j {
///     ...
///     loop_N {
///       func(tempstorage, ...);
///       tempstorage = ...
///     }
///   }
/// }
/// The callexpr is redundant if following two conditions are meet:
/// (1) No modified reference between tempstorage initialization and callexpr.
/// (2) No modified reference in loop_j or deeper loop.
/// The redundant callexpr can be remove safely.
class CubRedundantCallAnalyzer {
public:
  static bool isRedundantCallExpr(const CallExpr *C);
};

/// Analyze temp_storage and temp_storage_size argument to determine
/// whether these two argument and related decl or cudaMalloc can be
/// removed.
/// If the d_temp_storage and temp_storage_bytes only used in
/// Reduce/Min/Max/Sum and cudaMalloc, then we can remove related decl
/// and cudaMalloc*.
/// Pseudo code:
/// loop_1 {
///   ...
///   tempstorage = nullptr;
///   ...
///   loop_j {
///     ...
///     loop_N {
///       func(tempstorage, ...);
///       tempstorage = ...
///     }
///   }
/// }
/// The callexpr is redundant if following two conditions are meet:
/// (1) No modified reference between tempstorage initialization and callexpr.
/// (2) No modified reference in loop_j or deeper loop.
/// The redundant callexpr can be remove safely.
class CubRedundantTempStorageAnalyzer {
public:
  static void removeRedundantTempVar(const CallExpr *CE);
};

} // namespace dpct
} // namespace clang

#endif // DPCT_CUB_REDUNDANT_CALL_ANALYZER_H
