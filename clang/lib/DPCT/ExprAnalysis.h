//===--- ExprAnalysis.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
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

#include <assert.h>

namespace clang {
namespace dpct {

/// Store replacement info applied on a string
class StringReplacement {
public:
  StringReplacement(std::string &Src, size_t Off, size_t Len, std::string Txt)
      : SourceStr(Src), Offset(Off), Length(Len), Text(std::move(Txt)) {}

  inline void replaceString() {
    if (Offset <= SourceStr.length())
      SourceStr.replace(Offset, Length, Text);
#ifdef DPCT_DEBUG_BUILD
    else {
      llvm::errs() << "Encounter wrong replacement in ExprAnalysis:\n\""
                   << SourceStr << "\": " << Offset << ": +" << Length << ": \""
                   << Text << "\"\n";
      assert(0 && "ExprAnalysis try to apply an illegal replacement!");
    }
#endif
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
  TemplateDependentReplacement(const TemplateDependentReplacement &rhs)
      : TemplateDependentReplacement(rhs.SourceStr, rhs.Offset, rhs.Length,
                                     rhs.TemplateIndex) {}

  inline std::shared_ptr<TemplateDependentReplacement>
  alterSource(std::string &SrcStr) {
    return std::make_shared<TemplateDependentReplacement>(
        SrcStr, Offset, Length, TemplateIndex);
  }
  inline size_t getOffset() const { return Offset; }
  inline size_t getLength() const { return Length; }
  const TemplateArgumentInfo &
  getTargetArgument(const std::vector<TemplateArgumentInfo> &TemplateList);
  void replace(const std::vector<TemplateArgumentInfo> &TemplateList);
  inline void shift(int Shift) { Offset += Shift; }
};

/// Store a string which actual text dependent on template args
class TemplateDependentStringInfo {
  std::string SourceStr;
  std::vector<std::shared_ptr<TemplateDependentReplacement>> TDRs;
  bool IsDependOnWrittenArgument = false;

public:
  TemplateDependentStringInfo() = default;
  TemplateDependentStringInfo(std::string &&SrcStr)
      : SourceStr(std::move(SrcStr)) {}
  TemplateDependentStringInfo(
      const std::string &SourceStr,
      const std::map<size_t, std::shared_ptr<TemplateDependentReplacement>>
          &InTDRs);

  inline const std::string &getSourceString() const { return SourceStr; }

  /// Get the result when given template arguments are applied.
  /// e.g.: X<T> with template dependent replacement {2, 1, 0}, argument is int,
  /// the result will be X<int>.
  /// e.g.: X<T> with template dependent replacements {2, 1, 0}, argument is
  /// Y<T1> with template dependent replacement {2, 2, 0}, the result will be
  /// X<Y<T1>> with template dependent replacement {4, 2, 0}.
  std::shared_ptr<TemplateDependentStringInfo>
  applyTemplateArguments(const std::vector<TemplateArgumentInfo> &TemplateList);

  bool isDependOnWritten() const { return IsDependOnWrittenArgument; }
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
    TDRs.insert(
        std::make_pair(Offset, std::make_shared<TemplateDependentReplacement>(
                                   SourceStr, Offset, Length, TemplateIndex)));
  }
  // Add a string replacement
  void addStringReplacement(size_t Offset, size_t Length, std::string Text) {
    auto Result = ReplMap.insert(std::make_pair(
        Offset,
        std::make_shared<StringReplacement>(SourceStr, Offset, Length, Text)));
    if (Result.second) {
      auto Shift = Result.first->second->getReplacedText().length() - Length;
      ShiftLength += Shift;
      auto TDRItr = TDRs.upper_bound(Result.first->first);
      while (TDRItr != TDRs.end()) {
        TDRItr->second->shift(Shift);
        ++TDRItr;
      }
    }
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
  inline std::string getRewritePrefix() {
    return RewritePrefix;
  }

  inline std::string getRewritePostfix() {
    return RewritePostfix;
  }

  static std::string ref(const Expr *Expression) {
    ExprAnalysis EA(Expression);
    return EA.getReplacedString();
  }
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
  inline void analyze(const TypeLoc &TL) {
    initSourceRange(TL.getSourceRange());
    analyzeType(TL);
  }
  inline void analyze(const TemplateArgumentLoc &TAL) {
    initSourceRange(TAL.getSourceRange());
    analyzeTemplateArgument(TAL);
  }

  inline bool hasReplacement() { return ReplSet.hasReplacements(); }
  inline const std::string &getReplacedString() {
    return ReplSet.getReplacedString();
  }
  inline std::shared_ptr<TemplateDependentStringInfo>
  getTemplateDependentStringInfo() {
    return ReplSet.getTemplateDependentStringInfo();
  }
  // This function is not re-enterable, if caller need to check if it returns
  // nullptr, caller need to use temp variable to save the return value, then
  // check. Don't call twice for same Replacement.
  inline TextModification *getReplacement() {
    return hasReplacement() ? new ReplaceStmt(E, getReplacedString()) : nullptr;
  }

  inline void clearReplacement() { ReplSet.reset(); }

  SourceLocation getExprBeginSrcLoc() { return ExprBeginLoc; }
  SourceLocation getExprEndSrcLoc() { return ExprEndLoc; }

  // Replace a sub expr
  template <class TextData>
  inline void addReplacement(const Expr *E, TextData Text) {
    auto LocInfo = getOffsetAndLength(E);
    addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
  }

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

  template<class T> void analyzeTemplateSpecializationType(const T &TL) {
    for (size_t i = 0; i < TL.getNumArgs(); ++i)
      analyzeTemplateArgument(TL.getArgLoc(i));
  }

  // Prepare for analyze.
  void initExpression(const Expr *Expression);
  void initSourceRange(const SourceRange &Range);

  std::pair<size_t, size_t> getOffsetAndLength(SourceLocation Begin,
                                               SourceLocation End);
  // Advanced version to handle macros
  std::pair<size_t, size_t> getOffsetAndLength(SourceLocation Begin,
                                               SourceLocation End,
                                               const Expr *Parent);
  std::pair<size_t, size_t> getOffsetAndLength(SourceLocation SL);
  std::pair<size_t, size_t> getOffsetAndLength(const Expr *);

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
  // Replace string between begin location and end location.
  // Pass parent expr to calculate the correct location of macros
  template <class TextData>
  inline void addReplacement(SourceLocation Begin, SourceLocation End,
                             const Expr *P, TextData Text) {
    if (!P)
      return addReplacement(Begin, End, std::move(Text));
    auto LocInfo = getOffsetAndLength(Begin, End, P);
    if (LocInfo.first > 0 && LocInfo.first < SrcLength) {
      addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
    }
  }

  // Replace total string
  template <class TextData> inline void addReplacement(TextData Text) {
    addReplacement(SrcBegin, SrcLength, std::move(Text));
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
    return dispatch(MTE->getSubExpr());
  }

  inline void analyzeExpr(const PseudoObjectExpr *POE) {
    dispatch(POE->getResultExpr());
  }

  inline void analyzeExpr(const BinaryOperator *BO) {
    dispatch(BO->getLHS());
    dispatch(BO->getRHS());
  }

  inline void analyzeExpr(const UnaryOperator *UO) {
    dispatch(UO->getSubExpr());
  }

  inline void analyzeExpr(const ConditionalOperator *CO) {
    dispatch(CO->getCond());
    dispatch(CO->getLHS());
    dispatch(CO->getRHS());
  }

  inline void analyzeExpr(const DeclRefExpr *DRE);

  inline void analyzeExpr(const ParenExpr *PE) { dispatch(PE->getSubExpr()); }

  inline void analyzeExpr(const ArraySubscriptExpr *ASE) {
    dispatch(ASE->getBase());
    dispatch(ASE->getIdx());
  }


  void analyzeExpr(const CXXConstructExpr *Ctor);
  void analyzeExpr(const MemberExpr *ME);
  void analyzeExpr(const UnaryExprOrTypeTraitExpr *UETT);
  void analyzeExpr(const CStyleCastExpr *Cast);
  void analyzeExpr(const CallExpr *CE);
  void analyzeExpr(const CXXNamedCastExpr *NCE);

  inline void analyzeType(const TypeSourceInfo *TSI,
                          const Expr *CSCE = nullptr) {
    analyzeType(TSI->getTypeLoc(), CSCE);
  }
  void analyzeType(TypeLoc TL, const Expr *E = nullptr);

  // Doing nothing when it doesn't need analyze
  inline void analyzeExpr(const Stmt *S) {}

  void analyzeTemplateArgument(const TemplateArgumentLoc &TAL);

  inline const Expr *getTargetExpr() { return E; }

  ASTContext &Context;
  const SourceManager &SM;

  std::string RefString;

  bool IsInMacroDefine = false;
private:
  // E is analyze target expression, while ExprString is the source text of E.
  // Replacements contains all the replacements happened in E.
  const Expr *E;
  SourceLocation ExprBeginLoc;
  SourceLocation ExprEndLoc;
  size_t SrcBegin;
  size_t SrcLength;
  FileID FileId;
  StringReplacements ReplSet;
  std::string RewritePrefix;
  std::string RewritePostfix;
};

/// Analyze expression used as argument.
class ArgumentAnalysis : public ExprAnalysis {
public:
  using Base = ExprAnalysis;
  ArgumentAnalysis() {}
  ArgumentAnalysis(bool IsInMacroDefine) { this->IsInMacroDefine = IsInMacroDefine; }
  // Special init is needed for argument expression.
  ArgumentAnalysis(const Expr *Arg, bool IsInMacroDefine)
      : Base(nullptr) {
    this->IsInMacroDefine = IsInMacroDefine;
    initArgumentExpr(Arg);
  }

  inline void analyze() { Base::analyze(); }
  // Special init is needed for argument expression.
  void analyze(const Expr *Expression) {
    initArgumentExpr(Expression);
    analyze();
  }

protected:
  // Ignore the constructor when it's argument expression, it is copy/move
  // constructor and no migration for it.Start analyze its argument.
  // Replace total string when it is default argument expression.
  void initArgumentExpr(const Expr *Expression) {
    if (!Expression)
      initExpression(Expression);
    if (auto Ctor = dyn_cast<CXXConstructExpr>(Expression)) {
      if (Ctor->getConstructor()->isCopyConstructor() || Ctor->isElidable())
        Expression = Ctor->getArg(0);
    }
    initExpression(Expression);
    if (auto DAE = dyn_cast<CXXDefaultArgExpr>(Expression))
      addReplacement(std::string(getDefaultArgument(DAE->getExpr())));
  }

private:
  static const std::string &getDefaultArgument(const Expr *E);

  using DefaultArgMapTy = std::map<const Expr *, std::string>;
  static DefaultArgMapTy DefaultArgMap;
};

class KernelArgumentAnalysis : public ArgumentAnalysis {
public:
  bool IsRedeclareRequired;
  bool IsPointer;
  bool IsDefinedOnDevice = false;
  bool IsKernelParamPtr = false;

  KernelArgumentAnalysis(bool IsInMacroDefine)
      : ArgumentAnalysis(IsInMacroDefine) {}
  void analyze(const Expr *Expression) {
    IsPointer = Expression->getType()->isPointerType();
    IsKernelParamPtr = IsPointer;
    IsRedeclareRequired = false;
    ArgumentAnalysis::analyze(Expression);
  }

protected:
  void dispatch(const Stmt *Arg) override;

private:
  inline void analyzeExpr(const DeclRefExpr *Arg);
  inline void analyzeExpr(const MemberExpr *Arg);
  inline void analyzeExpr(const CallExpr *Arg) {
    IsRedeclareRequired = true;
    ExprAnalysis::analyzeExpr(Arg);
  }
  inline void analyzeExpr(const ArraySubscriptExpr *Arg) {
    IsRedeclareRequired = true;
    ExprAnalysis::analyzeExpr(Arg);
  }
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
  KernelConfigAnalysis(bool IsInMacroDefine)
      : ArgumentAnalysis(IsInMacroDefine) {}
  void analyze(const Expr *E, bool ReverseIfNeed = false) {
    if (IsInMacroDefine && SM.isMacroArgExpansion(E->getBeginLoc())) {
      Reversed = false;
      DirectRef = true;
      initArgumentExpr(E);
      return;
    }
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
