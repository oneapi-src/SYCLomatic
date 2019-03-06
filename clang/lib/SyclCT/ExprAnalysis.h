//===--- ExprAnalysis.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_EXPR_ANALYSIS_H
#define SYCLCT_EXPR_ANALYSIS_H

#include "TextModification.h"
#include "Utility.h"

namespace clang {
namespace syclct {

class StringReplacement {
public:
  template <class... Args>
  StringReplacement(std::string &Src, size_t Off, size_t Len, Args &&... Txt...)
      : SourceStr(Src), Offset(Off), Length(Len),
        Text(std::forward<Args>(Txt)...) {}
  template <class... Args>
  StringReplacement(std::string &Src,
                    const std::pair<size_t, size_t> &OffAndLen, Args &&... Txt)
      : StringReplacement(Src, OffAndLen.first, OffAndLen.second,
                          std::forward<Args>(Txt)...) {}

  inline std::string &replaceString() {
    return SourceStr.replace(Offset, Length, Text);
  }

  inline const std::string &getReplacedText() { return Text; }

private:
  // SourceStr is the string which need replaced.
  // Offset is the position where replacement happen.
  // Length is the replaced substring length
  // Text is replace text.
  std::string &SourceStr;
  size_t Offset;
  size_t Length;
  const std::string Text;
};

class ExprAnalysis {
public:
  ExprAnalysis() : E(nullptr) {}

  inline void analysis(const Expr *Expression) {
    setExpr(Expression);
    analysisExpr(E);
  }

  inline bool hasReplacement() { return !Replacements.empty(); }
  inline const std::string &getReplacedString() {
    replaceString();
    return ExprString;
  }
  inline TextModification *getRepalcement() {
    return hasReplacement() ? new ReplaceStmt(E, getReplacedString()) : nullptr;
  }

protected:
  inline void addReplacement(const Expr *Expression, std::string &&Text) {
    auto OffAndLen = getOffsetAndLength(Expression);
    // At the same offset, only one replacement exist.
    Replacements.insert(std::pair<size_t, std::shared_ptr<StringReplacement>>(
        OffAndLen.first, std::make_shared<StringReplacement>(
                             ExprString, OffAndLen, std::move(Text))));
  }

  virtual void analysisExpr(const Expr *Expression) {
    switch (Expression->getStmtClass()) {
#define ANALYSIS_EXPR(EXPR)                                                    \
  case Stmt::EXPR##Class:                                                      \
    return analysis##EXPR(dyn_cast<EXPR>(Expression));
      ANALYSIS_EXPR(ImplicitCastExpr)
      ANALYSIS_EXPR(BinaryOperator)
#undef ANALYSIS_EXPR
    default:
      return;
    }
  }

private:
  inline void analysisImplicitCastExpr(const ImplicitCastExpr *ICE) {
    if (ICE)
      return analysisExpr(ICE->getSubExpr());
  }

  inline void analysisBinaryOperator(const BinaryOperator *BO) {
    if (BO) {
      analysisExpr(BO->getLHS());
      analysisExpr(BO->getRHS());
    }
  }

  std::pair<size_t, size_t> getOffsetAndLength(const Expr *TE);

  void replaceString() {
    auto RItr = Replacements.rbegin();
    while (RItr != Replacements.rend())
      RItr++->second->replaceString();
    Replacements.clear();
  }

  void setExpr(const Expr *Expression);

  // E is analysis target expression, while ExprString is the source text of E.
  // Replacements contains all the replacements happened in E.
  const Expr *E;
  unsigned ExprBeginOffset;
  std::string ExprString;
  std::map<size_t, std::shared_ptr<StringReplacement>> Replacements;
};

class TemplateArgumentInfo;

class ArraySizeExprAnalysis : public ExprAnalysis {
public:
  using Base = ExprAnalysis;
  ArraySizeExprAnalysis() : TemplateList(nullptr) {}

  void setTemplateArgsList(
      const std::vector<TemplateArgumentInfo> &TemplateArgsList) {
    TemplateList = &TemplateArgsList;
  }

protected:
  virtual void analysisExpr(const Expr *Expression) override {
    switch (Expression->getStmtClass()) {
    case Stmt::DeclRefExprClass:
      return analysisDeclRefExpr(dyn_cast<DeclRefExpr>(Expression));
    default:
      return Base::analysisExpr(Expression);
    }
  }

private:
  void analysisDeclRefExpr(const DeclRefExpr *Expression);
  const std::vector<TemplateArgumentInfo> *TemplateList;
};
} // namespace syclct
} // namespace clang

#endif // !SYCLCT_EXPR_ANALYSIS_H
