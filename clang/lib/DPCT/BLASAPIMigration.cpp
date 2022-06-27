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

bool checkConstQualifierInDoublePointerType(
    const Expr *E, bool IsBaseValueNeedConst /* <T [DoesHereHaveConst] * *> */,
    bool IsFirstLevelPointerNeedConst /* <T * [DoesHereHaveConst] *> */) {
  const Expr *InputArg = E->IgnoreImpCasts();
  QualType InputArgPtrPtrType = InputArg->getType();
  QualType InputArgPtrType;
  if (InputArgPtrPtrType->isPointerType()) {
    InputArgPtrType = InputArgPtrPtrType->getPointeeType();
  } else if (InputArgPtrPtrType->isArrayType()) {
    const ArrayType *AT = dyn_cast<ArrayType>(InputArgPtrPtrType.getTypePtr());
    InputArgPtrType = AT->getElementType();
  } else {
    return false;
  }
  bool IsInputArgPtrConst = InputArgPtrType.isConstQualified();

  QualType InputArgBaseValueType;
  if (InputArgPtrType->isPointerType()) {
    InputArgBaseValueType = InputArgPtrType->getPointeeType();
  } else if (InputArgPtrType->isArrayType()) {
    const ArrayType *AT = dyn_cast<ArrayType>(InputArgPtrType.getTypePtr());
    InputArgBaseValueType = AT->getElementType();
  } else {
    return false;
  }
  bool IsInputArgBaseValueConst = InputArgBaseValueType.isConstQualified();

  if ((IsFirstLevelPointerNeedConst == IsInputArgPtrConst) &&
      (IsBaseValueNeedConst == IsInputArgBaseValueConst)) {
    return true;
  }
  return false;
}

} // namespace dpct
} // namespace clang
