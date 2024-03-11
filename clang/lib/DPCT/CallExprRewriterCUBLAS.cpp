//===--------------- CallExprRewriterCUBLAS.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"
#include <string>

namespace clang {
namespace dpct {

template <class ArgT> class BufferOrUSMPtrCallArgPrinter {
  ArgT Arg;
  std::string DataType;

public:
  BufferOrUSMPtrCallArgPrinter(ArgT &&Arg, std::string DataType)
      : Arg(std::forward<ArgT>(Arg)), DataType(DataType) {}
  template <class StreamT> void print(StreamT &Stream) const {
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      Stream << MapNames::getDpctNamespace() << "rvalue_ref_to_lvalue_ref("
             << MapNames::getDpctNamespace() << "get_buffer<" << DataType
             << ">(";
      clang::dpct::print(Stream, Arg);
      Stream << "))";
    } else {
      if (DataType == "std::complex<float>" ||
          DataType == "std::complex<double>")
        Stream << "(" << DataType << "*)";
      if constexpr (std::is_same_v<ArgT, const Expr *>)
        clang::dpct::print(Stream, Arg->IgnoreCasts());
      else
        clang::dpct::print(Stream, Arg);
    }
  }
};

template <class ArgT>
std::function<BufferOrUSMPtrCallArgPrinter<ArgT>(const CallExpr *)>
makeBufferOrUSMPtrCallArgCreator(std::function<ArgT(const CallExpr *)> Arg,
                                 std::string DataType) {
  return PrinterCreator<BufferOrUSMPtrCallArgPrinter<ArgT>,
                        std::function<ArgT(const CallExpr *)>,
                        std::function<std::string(const CallExpr *)>>(
      Arg, [=](const CallExpr *) { return DataType; });
}

#define BUFFER_OR_USM_PTR(Arg, T) makeBufferOrUSMPtrCallArgCreator(Arg, T)

void CallExprRewriterFactoryBase::initRewriterMapCUBLAS() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesCUBLAS.inc"
      }));
}

} // namespace dpct
} // namespace clang
