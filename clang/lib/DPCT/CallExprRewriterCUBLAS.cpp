//===--------------- CallExprRewriterCUBLAS.cpp ---------------------------===//
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

class BufferOrUSMPtrCallArgExpr {
  BufferOrUSMPtrCallArgExpr() = default;

public:
  const Expr *E = nullptr;
  std::string DataType;

  template <class StreamT> void print(StreamT &Stream) const {
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      Stream << MapNames::getDpctNamespace() << "rvalue_ref_to_lvalue_ref("
             << MapNames::getDpctNamespace() << "get_buffer<" << DataType
             << ">(";
      clang::dpct::print(Stream, E);
      Stream << "))";
    } else {
      if (DataType == "std::complex<float>" ||
          DataType == "std::complex<double>")
        Stream << "(" << DataType << "*)";
      clang::dpct::print(Stream, E);
    }
  }

  static BufferOrUSMPtrCallArgExpr create(const Expr *E, std::string DataType);
};

BufferOrUSMPtrCallArgExpr
BufferOrUSMPtrCallArgExpr::create(const Expr *E, std::string DataType) {
  BufferOrUSMPtrCallArgExpr BOUPCAE;
  BOUPCAE.E = E;
  BOUPCAE.DataType = DataType;
  return BOUPCAE;
}

inline std::function<BufferOrUSMPtrCallArgExpr(const CallExpr *)>
makeBufferOrUSMPtrCallArgCreator(unsigned Idx, std::string DataType) {
  return [=](const CallExpr *C) -> BufferOrUSMPtrCallArgExpr {
    return BufferOrUSMPtrCallArgExpr::create(C->getArg(Idx), DataType);
  };
}

#define BUFFER_OR_USM_PTR(Idx, T) makeBufferOrUSMPtrCallArgCreator(Idx, T)

void CallExprRewriterFactoryBase::initRewriterMapCUBLAS() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesCUBLAS.inc"
      }));
}

} // namespace dpct
} // namespace clang
