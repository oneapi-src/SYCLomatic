//===--- ExprAnalysis.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_EXPR_ANALYSIS_H
#define DPCT_EXPR_ANALYSIS_H

#include "TextModification.h"

#include "Debug.h"

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

namespace clang {
namespace dpct {

/// Store replacement info applied on a string
class StringReplacement {
public:
  StringReplacement(std::string &Src, size_t Off, size_t Len, std::string Txt)
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

class TemplateArgumentInfo;

/// Store replacement dependent on template args
class TemplateDependentReplacement {
  std::string &SourceStr;
  size_t Offset;
  size_t Length;
  unsigned TemplateIndex;

public:
  TemplateDependentReplacement(std::string &SrcStr, size_t Offset,
                               size_t Length, unsigned TemplateIndex)
      : SourceStr(SrcStr), Offset(Offset), Length(Length),
        TemplateIndex(TemplateIndex) {}

  inline std::shared_ptr<TemplateDependentReplacement>
  alterSource(std::string &SrcStr) {
    return std::make_shared<TemplateDependentReplacement>(
        SrcStr, Offset, Length, TemplateIndex);
  }
  void replace(const std::vector<TemplateArgumentInfo> &TemplateList);
};

/// Store a string which actual text dependent on template args
class TemplateDependentStringInfo {
  std::string SourceStr;
  std::vector<std::shared_ptr<TemplateDependentReplacement>> TDRs;

public:
  TemplateDependentStringInfo() = default;
  TemplateDependentStringInfo(std::string &&SrcStr)
      : SourceStr(std::move(SrcStr)) {}
  TemplateDependentStringInfo(
      const std::string &SourceStr,
      const std::map<size_t, std::shared_ptr<TemplateDependentReplacement>>
          &InTDRs);

  std::string
  getReplacedString(const std::vector<TemplateArgumentInfo> &TemplateList);
};

/// Store a expr source string which may need replaced and its replacements
class StringReplacements {
public:
  StringReplacements() : ShiftLength(0) {}
  inline void init(std::string &&SrcStr) {
    SourceStr = std::move(SrcStr);
    ReplMap.clear();
  }
  inline void reset() { ReplMap.clear(); }

  // Add a template dependent replacement
  inline void addTemplateDependentReplacement(size_t Offset, size_t Length,
                                              unsigned TemplateIndex) {
    Offset += ShiftLength;
    TDRs.insert(
        std::make_pair(Offset, std::make_shared<TemplateDependentReplacement>(
                                   SourceStr, Offset, Length, TemplateIndex)));
  }
  // Add a string replacement
  void addStringReplacement(size_t Offset, size_t Length, std::string Text) {
    auto Result = ReplMap.insert(std::make_pair(
        Offset,
        std::make_shared<StringReplacement>(SourceStr, Offset, Length, Text)));
    if (Result.second)
      ShiftLength += Result.first->second->getReplacedText().length() - Length;
  }

  // Generate replacement text info which dependent on template args.
  std::shared_ptr<TemplateDependentStringInfo>
  getTemplateDependentStringInfo() {
    replaceString();
    return std::make_shared<TemplateDependentStringInfo>(SourceStr, TDRs);
  }
  inline bool hasReplacements() { return !ReplMap.empty(); }
  inline const std::string &getReplacedString() {
    replaceString();
    return SourceStr;
  }

private:
  StringReplacements(const StringReplacements &) = delete;
  StringReplacements(StringReplacements &&) = delete;
  StringReplacements &operator=(StringReplacements) = delete;

  void replaceString();

  unsigned ShiftLength;
  std::string SourceStr;
  std::map<size_t, std::shared_ptr<StringReplacement>> ReplMap;
  std::map<size_t, std::shared_ptr<TemplateDependentReplacement>> TDRs;
};

/// Analyze expression and generate its migrated string
class ExprAnalysis {
public:
  ExprAnalysis() : ExprAnalysis(nullptr) {}
  explicit ExprAnalysis(const Expr *Expression);

  // Start ananlysis the expression passed in when inited.
  inline void analyze() {
    if (E)
      dispatch(E);
  }
  // Start analyze the argument expression
  inline void analyze(const Expr *Expression) {
    initExpression(Expression);
    analyze();
  }

  inline bool hasReplacement() { return ReplSet.hasReplacements(); }
  inline const std::string &getReplacedString() {
    return ReplSet.getReplacedString();
  }
  inline std::shared_ptr<TemplateDependentStringInfo>
  getTemplateDependentStringInfo() {
    return ReplSet.getTemplateDependentStringInfo();
  }
  inline TextModification *getReplacement() {
    return hasReplacement() ? new ReplaceStmt(E, getReplacedString()) : nullptr;
  }

  inline void clearReplacement() { ReplSet.reset(); }

private:
  SourceLocation getExprLocation(SourceLocation Loc);
  size_t getOffset(SourceLocation Loc) {
    return SM.getFileOffset(Loc) - SrcBegin;
  }

protected:
  void analyzeArgument(const Expr *E) {
    switch (E->getStmtClass()) {
    case Stmt::CXXConstructExprClass:
      return dispatch(static_cast<const CXXConstructExpr *>(E)->getArg(0));
    default:
      dispatch(E);
    }
  }

  // Prepare for analyze.
  void initExpression(const Expr *Expression);

  std::pair<size_t, size_t> getOffsetAndLength(SourceLocation Begin,
                                               SourceLocation End);
  std::pair<size_t, size_t> getOffsetAndLength(SourceLocation SL);

  // Replace a sub expr
  template <class TextData>
  inline void addReplacement(const Expr *E, TextData Text) {
    addReplacement(E->getBeginLoc(), E->getEndLoc(), std::move(Text));
  }
  // Replace a token with its begin location
  template <class TextData>
  inline void addReplacement(SourceLocation SL, TextData Text) {
    auto LocInfo = getOffsetAndLength(SL);
    addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
  }
  // Replace string with relative offset to the stored string and length
  inline void addReplacement(SourceLocation Begin, size_t Length,
                             std::string Text) {
    addReplacement(getOffset(getExprLocation(Begin)), Length, std::move(Text));
  }
  // Replace string between begin location and end location
  template <class TextData>
  inline void addReplacement(SourceLocation Begin, SourceLocation End,
                             TextData Text) {
    auto LocInfo = getOffsetAndLength(Begin, End);
    addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
  }
  // Replace total string
  template <class TextData> inline void addReplacement(TextData Text) {
    addReplacement(SourceLocation(), std::move(Text));
  }
  // Replace string with relative offset to the stored string and length
  inline void addReplacement(size_t Offset, size_t Length, std::string Text) {
    ReplSet.addStringReplacement(Offset, Length, std::move(Text));
  }

  inline void addReplacement(size_t Offset, size_t Length,
                             unsigned TemplateIndex) {
    ReplSet.addTemplateDependentReplacement(Offset, Length, TemplateIndex);
  }

  // Analyze the expression, jump to corresponding anlysis function according
  // to its class
  // Precondition: Expression != nullptr
  virtual void dispatch(const Stmt *Expression);

  inline void analyzeExpr(const CastExpr *ICE) {
    return dispatch(ICE->getSubExpr());
  }

  inline void analyzeExpr(const MaterializeTemporaryExpr *MTE) {
    return dispatch(MTE->getTemporary());
  }

  inline void analyzeExpr(const PseudoObjectExpr *POE) {
    dispatch(POE->getResultExpr());
  }

  inline void analyzeExpr(const BinaryOperator *BO) {
    dispatch(BO->getLHS());
    dispatch(BO->getRHS());
  }

  inline void analyzeExpr(const ConditionalOperator *CO) {
    dispatch(CO->getCond());
    dispatch(CO->getLHS());
    dispatch(CO->getRHS());
  }

  inline void analyzeExpr(const DeclRefExpr *DRE);

  inline void analyzeExpr(const ParenExpr *PE) { dispatch(PE->getSubExpr()); }

  void analyzeExpr(const CXXConstructExpr *Ctor);
  void analyzeExpr(const MemberExpr *ME);
  void analyzeExpr(const UnaryExprOrTypeTraitExpr *UETT);
  void analyzeExpr(const CStyleCastExpr *Cast);
  void analyzeExpr(const CallExpr *CE);

  inline void analyzeType(const TypeSourceInfo *TSI) {
    analyzeType(TSI->getTypeLoc());
  }
  void analyzeType(const TypeLoc &TL);

  // Doing nothing when it doesn't need analyze
  inline void analyzeExpr(const Stmt *S) {}

  inline const Expr *getTargetExpr() { return E; }

  const ASTContext &Context;
  const SourceManager &SM;

  std::string RefString;

private:
  // E is analyze target expression, while ExprString is the source text of E.
  // Replacements contains all the replacements happened in E.
  const Expr *E;
  size_t SrcBegin;
  size_t SrcLength;
  StringReplacements ReplSet;
};

/// Analyze expression used as argument.
class ArgumentAnalysis : public ExprAnalysis {
public:
  using Base = ExprAnalysis;
  ArgumentAnalysis() {}
  // Special init is needed for argument expression.
  ArgumentAnalysis(const Expr *Arg) : Base(nullptr) { initArgumentExpr(Arg); }

  inline void analyze() { Base::analyze(); }
  // Special init is needed for argument expression.
  void analyze(const Expr *Expression) {
    initArgumentExpr(Expression);
    analyze();
  }

private:
  static const std::string &getDefaultArgument(const Expr *E);

  // Ignore the constructor when it's argument expression, it is copy/move
  // constructor and no migration for it.Start analyze its argument.
  // Replace total string when it is default argument expression.
  void initArgumentExpr(const Expr *Expression) {
    if (!Expression)
      initExpression(Expression);
    if (auto Ctor = dyn_cast<CXXConstructExpr>(Expression))
      Expression = Ctor->getArg(0);
    initExpression(Expression);
    if (auto DAE = dyn_cast<CXXDefaultArgExpr>(Expression))
      addReplacement(std::string(getDefaultArgument(DAE->getExpr())));
  }

  using DefaultArgMapTy = std::map<const Expr *, std::string>;
  static DefaultArgMapTy DefaultArgMap;
};

class KernelArgumentAnalysis : public ArgumentAnalysis {
public:
  bool isRedeclareRequired;
  bool isPointer;

  void analyze(const Expr *Expression) {
    isPointer = Expression->getType()->isPointerType();
    isRedeclareRequired = false;
    ArgumentAnalysis::analyze(Expression);
  }

protected:
  void dispatch(const Stmt *Arg) override;

private:
  inline void analyzeExpr(const DeclRefExpr *Arg);
  inline void analyzeExpr(const MemberExpr *Arg);
  inline void analyzeExpr(const UnaryOperator *Arg);
};

class KernelConfigAnalysis : public ArgumentAnalysis {
private:
  bool DoReverse = false;
  bool Reversed = false;
  bool DirectRef = false;

  void analyzeExpr(const CXXConstructExpr *Ctor);

  std::vector<std::string> getCtorArgs(const CXXConstructExpr *Ctor);
  inline std::string getCtorArg(ArgumentAnalysis &KCA, const Expr *Arg) {
    KCA.analyze(Arg);
    return KCA.getReplacedString();
  }

protected:
  void dispatch(const Stmt *Expression) override;

public:
  void analyze(const Expr *E, bool ReverseIfNeed = false) {
    DoReverse = ReverseIfNeed;
    ArgumentAnalysis::analyze(E);
    if (getTargetExpr()->IgnoreImplicit()->getStmtClass() ==
            Stmt::DeclRefExprClass ||
        getTargetExpr()->IgnoreImpCasts()->getStmtClass() ==
            Stmt::MemberExprClass) {
      DirectRef = true;
    }
  }

  inline bool reversed() { return Reversed; }
  inline bool isDirectRef() { return DirectRef; }
};

} // namespace dpct
} // namespace clang

#endif // !DPCT_EXPR_analyze_H
