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

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

namespace clang {
namespace syclct {

// Store replacement info applied on a string
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

// Store replacement dependent on template args
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

// Store a string which actual text dependent on template args
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

// Store a expr source string which may need replaced and its replacements
class StringReplacements {
public:
  StringReplacements() : ShiftLength(0) {}
  inline void init(std::string &&SrcStr) {
    SourceStr = std::move(SrcStr);
    ReplMap.clear();
  }

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
  inline std::shared_ptr<TemplateDependentStringInfo>
  getTemplateDependentStringInfo() {
    return ReplSet.getTemplateDependentStringInfo();
  }
  inline TextModification *getReplacement() {
    return hasReplacement() ? new ReplaceStmt(E, getReplacedString()) : nullptr;
  }

private:
  SourceLocation getExprLocation(SourceLocation Loc);
  size_t getOffset(SourceLocation Loc) {
    return SM.getFileOffset(Loc) - SrcBegin;
  }

protected:
  // Prepare for analysis.
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
  inline void analysisExpr(const DeclRefExpr *DRE) {
    if (auto TemplateDecl = dyn_cast<NonTypeTemplateParmDecl>(DRE->getDecl()))
      addReplacement(DRE, TemplateDecl->getIndex());
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
  const SourceManager &SM;

private:
  // E is analysis target expression, while ExprString is the source text of E.
  // Replacements contains all the replacements happened in E.
  const Expr *E;
  size_t SrcBegin;
  size_t SrcLength;
  StringReplacements ReplSet;
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
      addReplacement(std::string(getDefaultArgument(DAE->getExpr())));
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
