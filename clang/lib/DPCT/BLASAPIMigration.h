//===--------------- BLASAPIMigration.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_BLAS_API_MIGRAION_H
#define DPCT_BLAS_API_MIGRAION_H

#include "CallExprRewriter.h"

namespace clang {
namespace dpct {

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
      if (!SubExpr->isValueDependent() &&
          SubExpr->EvaluateAsInt(ER, DpctGlobalInfo::getContext())) {
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
        Stream << MapNames::getDpctNamespace() << "get_transpose(";
        clang::dpct::print(Stream, SubExpr);
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
        clang::dpct::print(Stream, SubExpr);
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
        clang::dpct::print(Stream, SubExpr);
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
        clang::dpct::print(Stream, SubExpr);
        break;
      }
      }
    } else {
      clang::dpct::print(Stream, E);
    }
  }

  static BLASEnumExpr create(const Expr *E, BLASEnumType BET);
};

// Some batch APIs, the input data parameter has type "T const * const *",
// but in Intel(R) oneAPI Math Kernel Library (oneMKL), the parameter type
// should be "T const **".
// So we need figure out whether the input argument type is desired.
// This function will check the if the base value type and the first level
// pointer meet the requirement of const qualifier.
bool checkConstQualifierInDoublePointerType(
    const Expr *E, bool IsBaseValueNeedConst /* <T [DoesHereHaveConst] * *> */,
    bool IsFirstLevelPointerNeedConst /* <T * [DoesHereHaveConst] *> */);

} // namespace dpct
} // namespace clang

#endif // !DPCT_BLAS_API_MIGRAION_H
