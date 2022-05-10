//===--------------- BLASAPIMigration.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BLASAPIMigration.h"

namespace clang {
namespace dpct {

BLASEnumExpr BLASEnumExpr::create(const Expr *E,
                                  BLASEnumExpr::BLASEnumType BET) {
  BLASEnumExpr BEE;
  BEE.E = E;
  BEE.BET = BET;
  if (auto CSCE = dyn_cast<CStyleCastExpr>(E)) {
    BEE.SubExpr = CSCE->getSubExpr();
  }
  return BEE;
}

} // namespace dpct
} // namespace clang
