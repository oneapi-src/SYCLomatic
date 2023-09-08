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

class NullptrOrCallArgExpr {
  NullptrOrCallArgExpr() = default;

public:
  const Expr *E = nullptr;

  template <class StreamT> void print(StreamT &Stream) const {
    const Expr *Arg = E->IgnoreImpCasts();
    Expr::EvalResult ArgResult;
    if ((!Arg->isValueDependent()) &&
        (Arg->EvaluateAsInt(ArgResult, dpct::DpctGlobalInfo::getContext()))) {
      int64_t Value = ArgResult.Val.getInt().getExtValue();
      if (!Value) {
        Stream << "nullptr";
        return;
      }
    }
    if (dyn_cast<GNUNullExpr>(E)) {
      Stream << "nullptr";
      return;
    }
    clang::dpct::print(Stream, E);
  }

  static NullptrOrCallArgExpr create(const Expr *E);
};

NullptrOrCallArgExpr NullptrOrCallArgExpr::create(const Expr *E) {
  NullptrOrCallArgExpr NOCAE;
  NOCAE.E = E;
  return NOCAE;
}

inline std::function<NullptrOrCallArgExpr(const CallExpr *)>
makeNullptrOrCallArgCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> NullptrOrCallArgExpr {
    return NullptrOrCallArgExpr::create(C->getArg(Idx));
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
