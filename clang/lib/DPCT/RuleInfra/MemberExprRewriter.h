//===--------------- MemberExprRewriter.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_MEMBEREXPR_H
#define DPCT_MEMBEREXPR_H

#include "RuleInfra/CallExprRewriter.h"

namespace clang {
namespace dpct {
class MERewriter {
protected:
  const MemberExpr *ME;
  StringRef SourceCalleeName;

protected:
  MERewriter(const MemberExpr *ME, StringRef SourceCalleeName)
      : ME(ME), SourceCalleeName(SourceCalleeName) {}
  bool NoRewrite = false;

public:
  ArgumentAnalysis Analyzer;
  virtual ~MERewriter() {}

  virtual std::optional<std::string> rewrite() = 0;
  template <typename IDTy, typename... Ts>
  inline void report(IDTy MsgID, bool UseTextBegin, Ts &&...Vals) {
    TransformSetTy TS;
    auto SL = ME->getBeginLoc();
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
  virtual std::shared_ptr<MERewriter> create(const MemberExpr *) const = 0;
  virtual ~MemberExprRewriterFactoryBase() {}

  static std::unique_ptr<std::unordered_map<
      std::string, std::shared_ptr<MemberExprRewriterFactoryBase>>>
      MemberExprRewriterMap;
  static void initMemberExprRewriterMap();
  RulePriority Priority = RulePriority::Fallback;
};

template <class RewriterTy, class... Args>
class MERewriterFactory : public MemberExprRewriterFactoryBase {
  std::tuple<std::string, Args...> Initializer;

private:
  template <size_t... Idx>
  inline std::shared_ptr<MERewriter>
  createRewriter(const MemberExpr *ME, std::index_sequence<Idx...>) const {
    return std::shared_ptr<RewriterTy>(
        new RewriterTy(ME, std::get<Idx>(Initializer)...));
  }

public:
  MERewriterFactory(StringRef SourceCalleeName, Args... Arguments)
      : Initializer(SourceCalleeName.str(), std::forward<Args>(Arguments)...) {}
  std::shared_ptr<MERewriter> create(const MemberExpr *ME) const override {
    if (!ME)
      return std::shared_ptr<MERewriter>();
    return createRewriter(ME, std::index_sequence_for<std::string, Args...>());
  }
};
} // namespace dpct
} // namespace clang

#endif
