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

template <class VirtualPtrT> class BLASBufferDeclPrinter {
  VirtualPtrT VirtualPtr;
  unsigned int Idx;
  std::string Type;

public:
  BLASBufferDeclPrinter(VirtualPtrT &&VirtualPtr, unsigned int &&Idx,
                        llvm::StringRef Type)
      : VirtualPtr(std::forward<VirtualPtrT>(VirtualPtr)), Idx(Idx),
        Type(Type.str()) {}
  template <class StreamT> void print(StreamT &Stream) const {
    Stream << "auto Arg_" << std::to_string(Idx)
           << "_buf_ct = " << MapNames::getDpctNamespace() << "get_buffer<"
           << Type << ">(";
    dpct::print(Stream, VirtualPtr);
    Stream << ")";
  }
};

template <class VirtualPtrT>
inline std::function<BLASBufferDeclPrinter<VirtualPtrT>(const CallExpr *)>
makeBLASBufferDeclCreator(
    std::function<VirtualPtrT(const CallExpr *)> VirtualPtr, unsigned int Idx,
    std::string Type) {
  return PrinterCreator<BLASBufferDeclPrinter<VirtualPtrT>,
                        std::function<VirtualPtrT(const CallExpr *)>,
                        unsigned int, std::string>(
      std::move(VirtualPtr), std::move(Idx), Type);
}

template <class... StmtPrinters>
inline std::shared_ptr<CallExprRewriterFactoryBase>
createBLASMultiStmtsRewriterFactory(
    const std::string &SourceName,
    std::function<StmtPrinters(const CallExpr *)> &&...Creators) {
  return std::make_shared<AssignableRewriterFactory>(
      std::make_shared<CallExprRewriterFactory<
          PrinterRewriter<MultiStmtsPrinter<StmtPrinters...>>,
          std::function<StmtPrinters(const CallExpr *)>...>>(SourceName,
                                                             Creators...),
      true, true, true, false);
}

#define BLAS_MULTI_STMTS_FACTORY_ENTRY(FuncName, ...)                          \
  {FuncName, createBLASMultiStmtsRewriterFactory(FuncName, __VA_ARGS__)},
#define BLAS_BUFFER_DECL(VP, IDX, TYPE) makeBLASBufferDeclCreator(VP, IDX, TYPE)

void CallExprRewriterFactoryBase::initRewriterMapCUBLAS() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesCUBLAS.inc"
      }));
}

} // namespace dpct
} // namespace clang
