//===---BLASAPIMigration.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "BLASAPIMigration.h"

namespace clang {
namespace dpct {

BLASEnumExpr BLASEnumExpr::create(const Expr *E, BLASEnumExpr::BLASEnumType BET) {
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
