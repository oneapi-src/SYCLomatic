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
      Stream << MapNames::getLibraryHelperNamespace() << "rvalue_ref_to_lvalue_ref("
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

class ScalarInputValuePrinter {
  const Expr *Arg;
  const Expr *Handle;
  std::string DataType;

public:
  ScalarInputValuePrinter(const Expr *&&Arg, const Expr *&&Handle,
                          std::string DataType)
      : Arg(std::forward<const Expr *>(Arg)),
        Handle(std::forward<const Expr *>(Handle)), DataType(DataType) {}
  template <class StreamT> void print(StreamT &Stream) const {
    const auto *UO = dyn_cast_or_null<UnaryOperator>(Arg->IgnoreImpCasts());
    const auto *COCE = dyn_cast<CXXOperatorCallExpr>(Arg->IgnoreImpCasts());
    if ((UO && UO->getOpcode() == UO_AddrOf && UO->getSubExpr()) ||
        (COCE && COCE->getOperator() == OO_Amp && COCE->getArg(0))) {
      const Expr *Sub = UO ? UO->getSubExpr() : COCE->getArg(0);
      if (DataType == "std::complex<float>" ||
          DataType == "std::complex<double>") {
        Stream << DataType << "(";
        clang::dpct::print(Stream, Sub);
        Stream << ".x(), ";
        clang::dpct::print(Stream, Sub);
        Stream << ".y())";
      } else {
        clang::dpct::print(Stream, Sub);
      }
    } else {
      Stream << MapNames::getLibraryHelperNamespace() << "get_value(";
      clang::dpct::print(Stream, Arg);
      Stream << ", ";
      if (needExtraParensInMemberExpr(Handle)) {
        Stream << "(";
        clang::dpct::print(Stream, Handle);
        Stream << ")->get_queue())";
      } else {
        clang::dpct::print(Stream, Handle);
        Stream << "->get_queue())";
      }
    }
  }
};

std::function<ScalarInputValuePrinter(const CallExpr *)>
makeScalarInputValueCreator(
    std::function<const Expr *(const CallExpr *)> Arg,
    std::function<const Expr *(const CallExpr *)> Handle,
    std::string DataType) {
  return PrinterCreator<ScalarInputValuePrinter,
                        std::function<const Expr *(const CallExpr *)>,
                        std::function<const Expr *(const CallExpr *)>,
                        std::function<std::string(const CallExpr *)>>(
      Arg, Handle, [=](const CallExpr *) { return DataType; });
}

#define BUFFER_OR_USM_PTR(Arg, T) makeBufferOrUSMPtrCallArgCreator(Arg, T)
#define SCALAR_INPUT(Arg, T) makeScalarInputValueCreator(Arg, ARG(0), T)

void CallExprRewriterFactoryBase::initRewriterMapCUBLAS() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesCUBLAS.inc"
      }));
}

} // namespace dpct
} // namespace clang
