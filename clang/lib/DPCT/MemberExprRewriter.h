//===--------------- MemberExprRewriter.h ---------------------------------===//
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
namespace member_expr {
class CallExprRewriter {
protected:
  const MemberExpr *Call;
  StringRef SourceCalleeName;

protected:
  CallExprRewriter(const MemberExpr *Call, StringRef SourceCalleeName)
      : Call(Call), SourceCalleeName(SourceCalleeName) {}
  bool NoRewrite = false;

public:
  ArgumentAnalysis Analyzer;
  virtual ~CallExprRewriter() {}

  virtual std::optional<std::string> rewrite() = 0;
  template <typename IDTy, typename... Ts>
  inline void report(IDTy MsgID, bool UseTextBegin, Ts &&...Vals) {
    TransformSetTy TS;
    auto SL = Call->getBeginLoc();
    DiagnosticsUtils::report<IDTy, Ts...>(SL, MsgID, &TS, UseTextBegin,
                                          std::forward<Ts>(Vals)...);
    for (auto &T : TS)
      DpctGlobalInfo::getInstance().addReplacement(
          T->getReplacement(DpctGlobalInfo::getContext()));
  }

  bool isNoRewrite() { return NoRewrite; }

  bool getBlockLevelFormatFlag() const { return BlockLevelFormatFlag; }

protected:
  bool BlockLevelFormatFlag = false;
  std::vector<std::string> getMigratedArgs();
  std::string getMigratedArg(unsigned Index);
  std::string getMigratedArgWithExtraParens(unsigned Index);

  StringRef getSourceCalleeName() { return SourceCalleeName; }
};

class MemberExprRewriterFactoryBase {
public:
  virtual std::shared_ptr<CallExprRewriter>
  create(const MemberExpr *) const = 0;
  virtual ~MemberExprRewriterFactoryBase() {}

  static std::unique_ptr<std::unordered_map<
      std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>>
      MemberExprRewriterMap;
  static void initMemberExprRewriterMap();
  RulePriority Priority = RulePriority::Fallback;
};

template <class RewriterTy, class... Args>
class CallExprRewriterFactory : public MemberExprRewriterFactoryBase {
  std::tuple<std::string, Args...> Initializer;

private:
  template <size_t... Idx>
  inline std::shared_ptr<CallExprRewriter>
  createRewriter(const MemberExpr *Call, std::index_sequence<Idx...>) const {
    return std::shared_ptr<RewriterTy>(
        new RewriterTy(Call, std::get<Idx>(Initializer)...));
  }

public:
  CallExprRewriterFactory(StringRef SourceCalleeName, Args... Arguments)
      : Initializer(SourceCalleeName.str(), std::forward<Args>(Arguments)...) {}
  std::shared_ptr<CallExprRewriter>
  create(const MemberExpr *Call) const override {
    if (!Call)
      return std::shared_ptr<CallExprRewriter>();
    return createRewriter(Call,
                          std::index_sequence_for<std::string, Args...>());
  }
};

} // namespace member_expr
} // namespace dpct
} // namespace clang

#endif
