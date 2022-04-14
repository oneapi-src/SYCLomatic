//===---BLASAPIMigration.h -------------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef C2S_BLAS_API_MIGRAION_H
#define C2S_BLAS_API_MIGRAION_H

#include "CallExprRewriter.h"

namespace clang {
namespace c2s {

class BLASEnumExpr {
  BLASEnumExpr() = default;

public:
  enum BLASEnumType { Trans, Uplo, Diag, Side };
  BLASEnumType BET;
  const Expr *E = nullptr;
  const Expr *SubExpr = nullptr;

  template <class StreamT> void print(StreamT &Stream) const {
    if (SubExpr) {
      Expr::EvalResult ER;
      bool Evaluated = false;
      int64_t Value = -1; 
      if (!SubExpr->isValueDependent() && SubExpr->EvaluateAsInt(ER, C2SGlobalInfo::getContext())) {
        Evaluated = true;
        Value = ER.Val.getInt().getExtValue();
      }
      switch (BET) {
      case BLASEnumType::Trans: {
        if (Evaluated) {
          if (Value == 0) {
            Stream << "oneapi::mkl::transpose::nontrans";
            break;
          } else if (Value == 1) {
            Stream << "oneapi::mkl::transpose::trans";
            break;
          } else if (Value == 2) {
            Stream << "oneapi::mkl::transpose::conjtrans";
            break;
          }
        }
        requestFeature(HelperFeatureEnum::BlasUtils_get_transpose, E);
        Stream << MapNames::getC2SNamespace() << "get_transpose(";
        clang::c2s::print(Stream, SubExpr);
        Stream << ")";
        break;
      }
      case BLASEnumType::Uplo: {
        if (Evaluated) {
          if (Value == 0) {
            Stream << "oneapi::mkl::uplo::lower";
            break;
          } else if (Value == 1) {
            Stream << "oneapi::mkl::uplo::upper";
            break;
          }
        }
        clang::c2s::print(Stream, SubExpr);
        Stream << " == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper";
        break;
      }
      case BLASEnumType::Diag: {
        if (Evaluated) {
          if (Value == 0) {
            Stream << "oneapi::mkl::diag::nonunit";
            break;
          } else if (Value == 1) {
            Stream << "oneapi::mkl::diag::unit";
            break;
          }
        }
        Stream << "(oneapi::mkl::diag)";
        clang::c2s::print(Stream, SubExpr);
        break;
      }
      case BLASEnumType::Side: {
        if (Evaluated) {
          if (Value == 0) {
            Stream << "oneapi::mkl::side::left";
            break;
          } else if (Value == 1) {
            Stream << "oneapi::mkl::side::right";
            break;
          }
        }
        Stream << "(oneapi::mkl::side)";
        clang::c2s::print(Stream, SubExpr);
        break;
      }
      }
    } else {
      clang::c2s::print(Stream, E);
    }
  }

  static BLASEnumExpr create(const Expr *E, BLASEnumType BET);
};

} // namespace c2s
} // namespace clang

#endif // !C2S_BLAS_API_MIGRAION_H
