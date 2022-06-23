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

std::string getPotentialConstTypeCast(const Expr *E,
                                      const std::string &ElemetTypeStr,
                                      bool ElementConst, bool PtrConst) {
  const Expr *InputArg = E->IgnoreImpCasts();
  QualType PtrPtrType = InputArg->getType();
  QualType PtrType;
  if (PtrPtrType->isPointerType()) {
    PtrType = PtrPtrType->getPointeeType();
  } else if (PtrPtrType->isArrayType()) {
    const ArrayType *AT = dyn_cast<ArrayType>(PtrPtrType.getTypePtr());
    PtrType = AT->getElementType();
  } else {
    return "";
  }
  bool IsPtrConst = PtrType.isConstQualified();

  QualType ElementType;
  if (PtrType->isPointerType()) {
    ElementType = PtrType->getPointeeType();
  } else if (PtrType->isArrayType()) {
    const ArrayType *AT = dyn_cast<ArrayType>(PtrType.getTypePtr());
    ElementType = AT->getElementType();
  } else {
    return "";
  }
  bool IsElementConst = ElementType.isConstQualified();

  if ((PtrConst == IsPtrConst) && (ElementConst == IsElementConst)) {
    return "";
  }

  return "(" + ElemetTypeStr + " " + (ElementConst ? "const *" : "*") +
         (PtrConst ? "const *" : "*") + ")";
}

} // namespace dpct
} // namespace clang
