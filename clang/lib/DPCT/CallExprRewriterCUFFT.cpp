//===--------------- CallExprRewriterCUFFT.cpp ----------------------------===//
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

class FFTDirExpr {
  FFTDirExpr() = default;

public:
  const Expr *E = nullptr;

  // For CUFFT_FORWARD and CUFFT_INVERSE in cufftExec call, they will be
  // migrated to dpct::fft::fft_direction::forward/backward
  // In other locations, they will be migrated to integer literal.
  template <class StreamT> void print(StreamT &Stream) const {
    requestFeature(HelperFeatureEnum::device_ext);
    Expr::EvalResult ER;
    int64_t Value = 0;
    if (!E->isValueDependent() &&
        E->EvaluateAsInt(ER, DpctGlobalInfo::getContext())) {
      Value = ER.Val.getInt().getExtValue();
      if (Value == -1) {
        Stream << MapNames::getLibraryHelperNamespace() << "fft::fft_direction::forward";
        return;
      } else if (Value == 1) {
        Stream << MapNames::getLibraryHelperNamespace()
               << "fft::fft_direction::backward";
        return;
      }
    }
    clang::dpct::print(Stream, E);
    Stream << " == 1 ? " << MapNames::getLibraryHelperNamespace()
           << "fft::fft_direction::backward : " << MapNames::getLibraryHelperNamespace()
           << "fft::fft_direction::forward";
  }

  static FFTDirExpr create(const Expr *E);
};

FFTDirExpr FFTDirExpr::create(const Expr *E) {
  FFTDirExpr FDE;
  FDE.E = E;
  return FDE;
}

inline std::function<FFTDirExpr(const CallExpr *)>
makeFFTDirExprCallArgCreator(unsigned Idx) {
  return [=](const CallExpr *C) -> FFTDirExpr {
    return FFTDirExpr::create(C->getArg(Idx));
  };
}

void CallExprRewriterFactoryBase::initRewriterMapCUFFT() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesCUFFT.inc"
      }));
}

} // namespace dpct
} // namespace clang
