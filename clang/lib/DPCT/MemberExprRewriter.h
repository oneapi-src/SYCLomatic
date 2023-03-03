//===--------------- MemberExprRewriter.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef DPCT_MEMBEREXPR_H
#define DPCT_MEMBEREXPR_H

#include "CallExprRewriter.h"

namespace clang {
namespace dpct {

class MemberExprBaseRewriter {
protected:
  const MemberExpr *ME;

protected:
  MemberExprBaseRewriter(const MemberExpr *ME) : ME(ME) {}
public:
  virtual ~MemberExprBaseRewriter() {}
  virtual std::optional<std::string> rewrite() = 0;

  // Emits a warning/error/note and/or comment depending on MsgID. For details
  // see Diagnostics.inc, Diagnostics.h and Diagnostics.cpp
  template <typename IDTy, typename... Ts>
  inline void report(IDTy MsgID, bool UseTextBegin, Ts &&...Vals) {
    TransformSetTy TS;
    auto SL = ME->getBeginLoc();
    DiagnosticsUtils::report<IDTy, Ts...>(
        SL, MsgID, &TS, UseTextBegin, std::forward<Ts>(Vals)...);
    for (auto &T : TS)
      DpctGlobalInfo::getInstance().addReplacement(
          T->getReplacement(DpctGlobalInfo::getContext()));
  }
};

template <class Printer>
class MemberExprPrinterRewriter: Printer,  public MemberExprBaseRewriter {
public:
  template<class... ArgsT>
  MemberExprPrinterRewriter(const MemberExpr *ME, ArgsT &&...Args):
    Printer(std::forward<ArgsT>(Args)...), MemberExprBaseRewriter(ME) {}

  std::optional<std::string> rewrite() override {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Printer::print(OS);
    return OS.str();
  }
};

template <class BaseNameT, class MemberNameT>
class MemberExprFieldRewriter
    : public MemberExprPrinterRewriter<MemberExprPrinter<BaseNameT, MemberNameT>> {
public:
    MemberExprFieldRewriter(
      const MemberExpr *ME,
      const std::function<BaseNameT(const MemberExpr *)> &BaseNameCreator,
      const std::function<bool(const MemberExpr *)> &IsArrowCreator,
      const std::function<MemberNameT(const MemberExpr *)> &MemberExprCreator):
        MemberExprPrinterRewriter<MemberExprPrinter<BaseNameT, MemberNameT>>(ME,
          BaseNameCreator(ME), IsArrowCreator(ME), MemberExprCreator(ME)) {}
};

class MemberExprRewriterFactoryBase {
  public:
  virtual std::shared_ptr<MemberExprBaseRewriter> create(const MemberExpr *ME) const = 0;
  virtual ~MemberExprRewriterFactoryBase() {}

  static std::unique_ptr<std::unordered_map<
    std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>>
    MemberExprRewriterMap;

  static void initMemberExprRewriterMap();

  RulePriority Priority = RulePriority::Fallback;
};


template <class RewriterTy, class... TAs>
class MemberExprRewriterFactory : public MemberExprRewriterFactoryBase {
  std::tuple<TAs...> Initializer;

private:
  template <size_t... Idx>
  inline std::shared_ptr<MemberExprBaseRewriter>
  createRewriter(const MemberExpr *ME, std::index_sequence<Idx...>) const {
    return std::shared_ptr<RewriterTy>(new RewriterTy(ME, std::get<Idx>(Initializer)...));
  }

public:
  MemberExprRewriterFactory(TAs... TemplateArgs)
      : Initializer(std::forward<TAs>(TemplateArgs)...) {}

  std::shared_ptr<MemberExprBaseRewriter> create(const MemberExpr *ME) const override {
    return createRewriter(ME, std::index_sequence_for<TAs...>());
  }

};

template <class... MsgArgs>
class UnsupportExprRewriter : public MemberExprBaseRewriter {
  template <class T>
  std::string getMsgArg(const std::function<T(const MemberExpr *)> &Func,
                        const MemberExpr *M) {
    return getMsgArg(Func(M), M);
  }
  template <class T>
  static std::string getMsgArg(const T &InputArg, const MemberExpr *) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    print(OS, InputArg);
    return OS.str();
  }

public:
  UnsupportExprRewriter(const MemberExpr *ME, Diagnostics MsgID, const MsgArgs &...Args)
      : MemberExprBaseRewriter(ME) {
    report(MsgID, false, getMsgArg(Args, ME)...);
  }

  std::optional<std::string> rewrite() override { return std::nullopt; }

  friend UnsupportFunctionRewriterFactory<MsgArgs...>;
};

} // namespace dpct
} // namespace clang

#endif
