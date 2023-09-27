//===--------------- CallExprRewriterCUSOLVER.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"

namespace clang {
namespace dpct {

bool isNull(const Expr *Expr) {
  const auto *E = Expr->IgnoreImpCasts();
  Expr::EvalResult ExprResult;
  if ((!E->isValueDependent()) &&
      (E->EvaluateAsInt(ExprResult, dpct::DpctGlobalInfo::getContext()))) {
    if (!ExprResult.Val.getInt().getExtValue())
      return true;
  }
  if (isa<CXXNullPtrLiteralExpr, GNUNullExpr>(E)) {
    return true;
  }
  return false;
}

class NullptrOrCallArgExpr {
public:
  NullptrOrCallArgExpr() = default;
  const Expr *E = nullptr;

  template <class StreamT> void print(StreamT &Stream) const {
    if (isNull(E)) {
      Stream << "nullptr";
      return;
    }
    clang::dpct::print(Stream, E);
  }
};

std::function<NullptrOrCallArgExpr(const CallExpr *)>
makeNullptrOrCallArgCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> NullptrOrCallArgExpr {
    NullptrOrCallArgExpr NOCAE;
    NOCAE.E = C->getArg(Idx);
    return NOCAE;
  };
}

#define NULLPTR_OR_ARG(X) makeNullptrOrCallArgCreator(X)

void CallExprRewriterFactoryBase::initRewriterMapCUSPARSE() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesCUSPARSE.inc"
      }));
}

} // namespace dpct
} // namespace clang
