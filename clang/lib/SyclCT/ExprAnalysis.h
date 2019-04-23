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

#include "clang/AST/ExprCXX.h"

namespace clang {
namespace syclct {

// Store replacement info applied on a string
class StringReplacement {
public:
  StringReplacement(std::string &Src, size_t Off, size_t Len, std::string &&Txt)
      : SourceStr(Src), Offset(Off), Length(Len), Text(std::move(Txt)) {}

  inline void replaceString() { SourceStr.replace(Offset, Length, Text); }

  inline const std::string &getReplacedText() { return Text; }

private:
  // SourceStr is the string which need replaced.
  // Offset is the position where replacement happen.
  // Length is the replaced substring length
  // Text is replace text.
  std::string &SourceStr;
  size_t Offset;
  size_t Length;
  std::string Text;
};

// Store a expr source string which may need replaced and its replacements
class StringReplacements {
public:
  StringReplacements(const ASTContext &Context)
      : Context(Context), SourceBegin(0), ShiftLength(0) {}
  StringReplacements(const Expr *E, const ASTContext &Context)
      : StringReplacements(Context) {
    init(E);
  }

  void init(const Expr *E);
  inline void init() { return init(nullptr); }

  inline bool hasReplacements() { return !ReplMap.empty(); }
  inline const std::string &getReplacedString() {
    replaceString();
    return SourceStr;
  }
  // Replace a sub expr
  inline void addReplacement(const Expr *E, std::string &&Text) {
    return addReplacement(E->getBeginLoc(), E->getEndLoc(), std::move(Text));
  }
  // Replace a token with its begin location
  inline void addReplacement(SourceLocation SL, std::string &&Text) {
    auto LocInfo = getOffsetAndLength(SL);
    return addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
  }
  // Replace string between begin location and end location
  inline void addReplacement(SourceLocation Begin, SourceLocation End,
                             std::string &&Text) {
    auto LocInfo = getOffsetAndLength(Begin, End);
    return addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
  }
  // Replace string with relative offset to the stored string and length
  inline void addReplacement(size_t Offset, size_t Length, std::string &&Text) {
    auto Result = ReplMap.insert(std::make_pair(
        Offset, std::make_shared<StringReplacement>(SourceStr, Offset, Length,
                                                    std::move(Text))));
    if (Result.second)
      ShiftLength += Result.first->second->getReplacedText().length() - Length;
  }
  // Replace total string
  inline void addReplacement(std::string &&Text) {
    return addReplacement(SourceLocation(), std::move(Text));
  }

private:
  StringReplacements(const StringReplacements &) = delete;
  StringReplacements(StringReplacements &&) = delete;
  StringReplacements &operator=(StringReplacements) = delete;

  std::pair<size_t, size_t> getOffsetAndLength(SourceLocation Begin,
                                               SourceLocation End);
  std::pair<size_t, size_t> getOffsetAndLength(SourceLocation SL);

  void replaceString();

  const ASTContext &Context;
  unsigned SourceBegin;
  unsigned ShiftLength;
  std::string SourceStr;
  std::map<size_t, std::shared_ptr<StringReplacement>> ReplMap;
};

// Analysis expression and generate its migrated string
class ExprAnalysis {
public:
  ExprAnalysis() : ExprAnalysis(nullptr) {}
  explicit ExprAnalysis(const Expr *Expression);

  // Start ananlysis the expression passed in when inited.
  inline void analysis() {
    if (E)
      analysisExpression(E);
  }
  // Start analysis the argument expression
  inline void analysis(const Expr *Expression) {
    initExpression(Expression);
    analysis();
  }

  inline bool hasReplacement() { return ReplSet.hasReplacements(); }
  inline const std::string &getReplacedString() {
    return ReplSet.getReplacedString();
  }
  inline TextModification *getReplacement() {
    return hasReplacement() ? new ReplaceStmt(E, getReplacedString()) : nullptr;
  }

protected:
  // Prepare for analysis.
  inline void initExpression(const Expr *Expression) {
    E = Expression;
    ReplSet.init(Expression);
  }

  template <class... Args> inline void addReplacement(Args... Arguments...) {
    ReplSet.addReplacement(std::forward<Args>(Arguments)...);
  }

  // Analysis the expression, jump to corresponding anlysis function according
  // to its class
  virtual void analysisExpression(const Stmt *Expression);

  inline void analysisExpr(const CastExpr *ICE) {
    return analysisExpression(ICE->getSubExpr());
  }

  inline void analysisExpr(const MaterializeTemporaryExpr *MTE) {
    return analysisExpression(MTE->getTemporary());
  }

  inline void analysisExpr(const BinaryOperator *BO) {
    analysisExpression(BO->getLHS());
    analysisExpression(BO->getRHS());
  }

  inline void analysisExpr(const ParenExpr *PE) {
    analysisExpression(PE->getSubExpr());
  }

  void analysisExpr(const CXXConstructExpr *Ctor);
  void analysisExpr(const MemberExpr *ME);
  void analysisExpr(const UnaryExprOrTypeTraitExpr *UETT);
  void analysisExpr(const CallExpr *CE);

  // Doing nothing when it doesn't need analysis
  inline void analysisExpr(const Stmt *S) {}

  const ASTContext &Context;

private:
  // E is analysis target expression, while ExprString is the source text of E.
  // Replacements contains all the replacements happened in E.
  const Expr *E;
  StringReplacements ReplSet;
};

class TemplateArgumentInfo;

// Analysis expressions which represent size of an array.
class ArraySizeExprAnalysis : public ExprAnalysis {
public:
  using Base = ExprAnalysis;
  ArraySizeExprAnalysis(const Expr *Expression,
                        const std::vector<TemplateArgumentInfo> *TemplateList)
      : Base(Expression),
        TemplateList(TemplateList ? *TemplateList : NullList) {}

protected:
  virtual void analysisExpression(const Stmt *Expression) override;

private:
  // Generate replacements when template dependent variable is used.
  void analysisExpr(const DeclRefExpr *Expression);
  const std::vector<TemplateArgumentInfo> &TemplateList;

  const static std::vector<TemplateArgumentInfo> NullList;
};

// Analysis expression used as argument.
class ArgumentAnalysis : public ExprAnalysis {
public:
  using Base = ExprAnalysis;
  ArgumentAnalysis() {}
  // Special init is needed for argument expression.
  ArgumentAnalysis(const Expr *Arg) : Base(nullptr) { initArgumentExpr(Arg); }

  inline void analysis() { Base::analysis(); }
  // Special init is needed for argument expression.
  void analysis(const Expr *Expression) {
    initArgumentExpr(Expression);
    analysis();
  }

private:
  static const std::string &getDefaultArgument(const Expr *E);

  // Ignore the constructor when it's argument expression, it is copy/move
  // constructor and no migration for it.Start analysis its argument.
  // Replace total string when it is default argument expression.
  void initArgumentExpr(const Expr *Expression) {
    if (!Expression)
      initExpression(Expression);
    if (auto Ctor = dyn_cast<CXXConstructExpr>(Expression))
      Expression = Ctor->getArg(0);
    initExpression(Expression);
    if (auto DAE = dyn_cast<CXXDefaultArgExpr>(Expression))
      addReplacement(getDefaultArgument(DAE->getExpr()));
  }

  using DefaultArgMapTy = std::map<const Expr *, std::string>;
  static DefaultArgMapTy DefaultArgMap;
};

class VarInfo;
// Analysis CUDA kernel call arguments, get out the passed in pointer variables.
class KernelArgumentAnalysis : public ArgumentAnalysis {
public:
  using MapTy = std::map<const VarDecl *, std::shared_ptr<VarInfo>>;
  KernelArgumentAnalysis(MapTy &DeclMap) : DeclMap(DeclMap) {}

protected:
  void analysisExpression(const Stmt *Arg) override;

private:
  inline void analysisExpr(const DeclRefExpr *Arg);
  MapTy &DeclMap;
};
} // namespace syclct
} // namespace clang

#endif // !SYCLCT_EXPR_ANALYSIS_H
