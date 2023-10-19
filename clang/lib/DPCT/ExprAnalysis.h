//===--------------- ExprAnalysis.h ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_EXPR_ANALYSIS_H
#define DPCT_EXPR_ANALYSIS_H

#include "Statics.h"
#include "TextModification.h"

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
  bool ContainsTemplateDependentMacro = false;
  std::set<HelperFeatureEnum> HelperFeatureSet;

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
  std::set<HelperFeatureEnum> getHelperFeatureSet() { return HelperFeatureSet; }
  void setHelperFeatureSet(std::set<HelperFeatureEnum> Set) {
    HelperFeatureSet = Set;
  }
  bool containsTemplateDependentMacro() const { return ContainsTemplateDependentMacro; }
};

/// Store an expr source string which may need replaced and its replacements
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
  inline std::string getRewritePrefix() { return RewritePrefix; }

  inline std::string getRewritePostfix() { return RewritePostfix; }

  static std::string ref(const Expr *Expression) {
    ExprAnalysis EA(Expression);
    return EA.getReplacedString();
  }
  ExprAnalysis() : ExprAnalysis(nullptr) {}
  explicit ExprAnalysis(const Expr *Expression);

  // Start analysis of the expression passed in when init-ed.
  inline void analyze() {
    if (E)
      dispatch(E);
  }
  // Analyze the argument expression
  inline void analyze(const Expr *Expression) {
    initExpression(Expression);
    analyze();
  }
  inline void analyze(const TypeLoc &TL) {
    initSourceRange(TL.getSourceRange());
    analyzeType(TL);
  }

  inline void analyze(const TypeLoc &TL, const NestedNameSpecifierLoc &NNSL) {
    auto SourceRange = getDefinitionRange(NNSL.getBeginLoc(),
                                          TL.getSourceRange().getEnd());
    initSourceRange(SourceRange);
    analyzeType(TL, nullptr, nullptr, &NNSL);
  }

  inline void analyze(const TypeLoc &TL, const DependentNameTypeLoc &DNTL) {
    auto SourceRange = getDefinitionRange(DNTL.getQualifierLoc().getBeginLoc(),
                                          TL.getSourceRange().getEnd());
    initSourceRange(SourceRange);
    analyzeType(TL, nullptr, &DNTL);
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
    auto Res = ReplSet.getTemplateDependentStringInfo();
    Res->setHelperFeatureSet(HelperFeatureSet);
    return Res;
  }
  // This function is not re-enterable, if caller need to check if it returns
  // nullptr, caller need to use temp variable to save the return value, then
  // check. Don't call twice for same Replacement.
  inline TextModification *getReplacement() {
    bool hasRepl = hasReplacement();
    std::string Repl = getReplacedString();
    if (E) {
      auto Range = getDefinitionRange(E->getBeginLoc(), E->getEndLoc());
      if (!isSameLocation(Range.getBegin(), Range.getEnd())) {
        return hasRepl ? new ReplaceStmt(E, true, Repl) : nullptr;
      }
    }
    return hasRepl ? new ReplaceText(SrcBeginLoc, SrcLength, std::move(Repl))
                   : nullptr;
  }

  inline void clearReplacement() { ReplSet.reset(); }

  inline SourceLocation getExprBeginSrcLoc() { return ExprBeginLoc; }
  inline SourceLocation getExprEndSrcLoc() { return ExprEndLoc; }
  inline size_t getExprLength() { return SrcLength; }

  // When NewRepl has larger range than existing Repl, remove Repl from
  // SubExprRepl and vice versa
  inline void addExtReplacement(std::shared_ptr<ExtReplacement> NewRepl) {
    for (auto Repl = SubExprRepl.begin(); Repl != SubExprRepl.end();) {
      if ((*Repl)->getFilePath() != NewRepl->getFilePath()) {
        Repl++;
        continue;
      } else if ((*Repl)->getOffset() <= NewRepl->getOffset() &&
                 (*Repl)->getOffset() + (*Repl)->getLength() >=
                     NewRepl->getOffset() + NewRepl->getLength()) {
        return;
      } else if ((*Repl)->getOffset() >= NewRepl->getOffset() &&
                 (*Repl)->getOffset() + (*Repl)->getLength() <=
                     NewRepl->getOffset() + NewRepl->getLength()) {
        SubExprRepl.erase(Repl);
        if (Repl == SubExprRepl.end()) {
          break;
        }
      } else {
        Repl++;
      }
    }
    SubExprRepl.push_back(NewRepl);
  }

  // Replace a sub expr
  inline void addReplacement(const Expr *E, std::string Text) {
    auto SpellingLocInfo = getSpellingOffsetAndLength(E);
    if (SM.getDecomposedLoc(SpellingLocInfo.first).first != FileId ||
        SM.getDecomposedLoc(SpellingLocInfo.first).second < SrcBegin ||
        SM.getDecomposedLoc(SpellingLocInfo.first).second +
                SpellingLocInfo.second >
            SrcBegin + SrcLength) {
      // If the spelling location is not in the parent range, add ExtReplacement
      addExtReplacement(std::make_shared<ExtReplacement>(
          SM, SpellingLocInfo.first, SpellingLocInfo.second, Text, nullptr));
    } else if (SM.getDecomposedLoc(SpellingLocInfo.first).first == FileId &&
               SM.getDecomposedLoc(SpellingLocInfo.first).second == SrcBegin &&
               SM.getDecomposedLoc(SpellingLocInfo.first).second +
                       SpellingLocInfo.second ==
                   SrcBegin + SrcLength) {
      // If the spelling location is the same as the parent range, add both
      addExtReplacement(std::make_shared<ExtReplacement>(
          SM, SpellingLocInfo.first, SpellingLocInfo.second, Text, nullptr));
      auto LocInfo = getOffsetAndLength(E);
      addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
    } else {
      // If the spelling location is inside the parent range, add string
      // replacement. The String replacement will be added to ExtReplacement
      // other where.
      auto LocInfo = getOffsetAndLength(E);
      addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
    }
  }

  // Replace a sub expr
  inline void addReplacement(const Expr *E, int length, std::string Text) {
    auto SpellingLocInfo = getSpellingOffsetAndLength(E);
    if (SM.getDecomposedLoc(SpellingLocInfo.first).first != FileId ||
        SM.getDecomposedLoc(SpellingLocInfo.first).second < SrcBegin ||
        SM.getDecomposedLoc(SpellingLocInfo.first).second +
                SpellingLocInfo.second >
            SrcBegin + SrcLength) {
      // If the spelling location is not in the parent range, add ExtReplacement
      addExtReplacement(std::make_shared<ExtReplacement>(
          SM, SpellingLocInfo.first, length, Text, nullptr));
    } else if (SM.getDecomposedLoc(SpellingLocInfo.first).first == FileId &&
               SM.getDecomposedLoc(SpellingLocInfo.first).second == SrcBegin &&
               SM.getDecomposedLoc(SpellingLocInfo.first).second +
                       SpellingLocInfo.second ==
                   SrcBegin + SrcLength) {
      // If the spelling location is the same as the parent range, add both
      addExtReplacement(std::make_shared<ExtReplacement>(
          SM, SpellingLocInfo.first, length, Text, nullptr));
      auto LocInfo = getOffsetAndLength(E);
      addReplacement(LocInfo.first, length, std::move(Text));
    } else {
      // If the spelling location is inside the parent range, add string
      // replacement. The String replacement will be added to ExtReplacement
      // other where.
      auto LocInfo = getOffsetAndLength(E);
      addReplacement(LocInfo.first, length, std::move(Text));
    }
  }

  /// Replace the text in source range \p SR with \p Text.
  /// \p ParentExpr is the parent expression contains \p SR which is used to
  /// calculate the correct replacement location.
  inline void addReplacement(SourceRange SR, const Expr *ParentExpr,
                             std::string Text) {
    auto ResultRange = getDefinitionRange(SR.getBegin(), SR.getEnd());
    auto LastTokenLength = Lexer::MeasureTokenLength(ResultRange.getEnd(), SM,
                                                     Context.getLangOpts());
    auto SpellingLocInfo = std::pair<SourceLocation, size_t>(
        ResultRange.getBegin(),
        SM.getCharacterData(ResultRange.getEnd()) -
            SM.getCharacterData(ResultRange.getBegin()) + LastTokenLength);

    if (SM.getDecomposedLoc(SpellingLocInfo.first).first != FileId ||
        SM.getDecomposedLoc(SpellingLocInfo.first).second < SrcBegin ||
        SM.getDecomposedLoc(SpellingLocInfo.first).second +
                SpellingLocInfo.second >
            SrcBegin + SrcLength) {
      // If the spelling location is not in the parent range, add ExtReplacement
      addExtReplacement(std::make_shared<ExtReplacement>(
          SM, SpellingLocInfo.first, SpellingLocInfo.second, Text, nullptr));
    } else if (SM.getDecomposedLoc(SpellingLocInfo.first).first == FileId &&
               SM.getDecomposedLoc(SpellingLocInfo.first).second == SrcBegin &&
               SM.getDecomposedLoc(SpellingLocInfo.first).second +
                       SpellingLocInfo.second ==
                   SrcBegin + SrcLength) {
      // If the spelling location is the same as the parent range, add both
      addExtReplacement(std::make_shared<ExtReplacement>(
          SM, SpellingLocInfo.first, SpellingLocInfo.second, Text, nullptr));
      auto LocInfo = getOffsetAndLength(SR.getBegin(), SR.getEnd(), ParentExpr);
      addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
    } else {
      // If the spelling location is inside the parent range, add string
      // replacement. The String replacement will be added to ExtReplacement
      // other where.
      auto LocInfo = getOffsetAndLength(SR.getBegin(), SR.getEnd(), ParentExpr);
      addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
    }
  }

  void applyAllSubExprRepl();
  inline std::vector<std::shared_ptr<ExtReplacement>> &getSubExprRepl() {
    return SubExprRepl;
  };
  // Replace a sub template arg
  inline void addReplacement(const Expr *E, unsigned TemplateIndex) {
    auto LocInfo = getOffsetAndLength(E);
    addReplacement(LocInfo.first, LocInfo.second, std::move(TemplateIndex));
  }
  std::set<HelperFeatureEnum> getHelperFeatureSet() { return HelperFeatureSet; }

  virtual ~ExprAnalysis() = default;
  SourceLocation CallSpellingBegin;
  SourceLocation CallSpellingEnd;
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

  template <class T> void analyzeTemplateSpecializationType(const T &TL) {
    for (size_t i = 0; i < TL.getNumArgs(); ++i) {
      analyzeTemplateArgument(TL.getArgLoc(i));
    }
  }

  // Prepare for analyze.
  void initExpression(const Expr *Expression);
  void initSourceRange(const SourceRange &Range);

  std::pair<SourceLocation, size_t>
  getSpellingOffsetAndLength(SourceLocation Begin, SourceLocation End);
  std::pair<SourceLocation, size_t>
  getSpellingOffsetAndLength(SourceLocation SL);
  std::pair<SourceLocation, size_t> getSpellingOffsetAndLength(const Expr *);

  std::pair<size_t, size_t> getOffsetAndLength(SourceLocation Begin,
                                               SourceLocation End);
  // Advanced version to handle macros
  std::pair<size_t, size_t> getOffsetAndLength(SourceLocation Begin,
                                               SourceLocation End,
                                               const Expr *Parent);
  std::pair<size_t, size_t> getOffsetAndLength(SourceLocation SL);
  std::pair<size_t, size_t> getOffsetAndLength(const Expr *, SourceLocation *Loc = nullptr);

  // Replace a token with its begin location
  inline void addReplacement(SourceLocation SL, std::string Text) {
    auto SpellingLocInfo = getSpellingOffsetAndLength(SL);
    if (SM.getDecomposedLoc(SpellingLocInfo.first).first != FileId ||
        SM.getDecomposedLoc(SpellingLocInfo.first).second < SrcBegin ||
        SM.getDecomposedLoc(SpellingLocInfo.first).second +
                SpellingLocInfo.second >
            SrcBegin + SrcLength) {
      // If the spelling location is not in the parent range, add ExtReplacement
      addExtReplacement(std::make_shared<ExtReplacement>(
          SM, SpellingLocInfo.first, SpellingLocInfo.second, Text, nullptr));
    } else if (SM.getDecomposedLoc(SpellingLocInfo.first).first == FileId &&
               SM.getDecomposedLoc(SpellingLocInfo.first).second == SrcBegin &&
               SM.getDecomposedLoc(SpellingLocInfo.first).second +
                       SpellingLocInfo.second ==
                   SrcBegin + SrcLength) {
      // If the spelling location is the same as the parent range, add both
      addExtReplacement(std::make_shared<ExtReplacement>(
          SM, SpellingLocInfo.first, SpellingLocInfo.second, Text, nullptr));
      auto LocInfo = getOffsetAndLength(SL);
      addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
    } else {
      // If the spelling location is inside the parent range, add string
      // replacement. The String replacement will be added to ExtReplacement
      // other where.
      auto LocInfo = getOffsetAndLength(SL);
      addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
    }
  }

  inline void addReplacement(SourceLocation SL, unsigned TemplateIndex) {
    auto LocInfo = getOffsetAndLength(SL);
    addReplacement(LocInfo.first, LocInfo.second, std::move(TemplateIndex));
  }

  // Replace string with relative offset to the stored string and length
  inline void addReplacement(SourceLocation Begin, size_t Length,
                             std::string Text) {
    addReplacement(getOffset(getExprLocation(Begin)), Length, std::move(Text));
  }

  // Replace string between begin location and end location
  inline void addReplacement(SourceLocation Begin, SourceLocation End,
                             std::string Text) {
    auto SpellingLocInfo = getSpellingOffsetAndLength(Begin, End);
    if (SM.getDecomposedLoc(SpellingLocInfo.first).first != FileId ||
        SM.getDecomposedLoc(SpellingLocInfo.first).second < SrcBegin ||
        SM.getDecomposedLoc(SpellingLocInfo.first).second +
                SpellingLocInfo.second >
            SrcBegin + SrcLength) {
      // If the spelling location is not in the parent range, add ExtReplacement
      addExtReplacement(std::make_shared<ExtReplacement>(
          SM, SpellingLocInfo.first, SpellingLocInfo.second, Text, nullptr));
    } else if (SM.getDecomposedLoc(SpellingLocInfo.first).first == FileId &&
               SM.getDecomposedLoc(SpellingLocInfo.first).second == SrcBegin &&
               SM.getDecomposedLoc(SpellingLocInfo.first).second +
                       SpellingLocInfo.second ==
                   SrcBegin + SrcLength) {
      // If the spelling location is the same as the parent range, add both
      addExtReplacement(std::make_shared<ExtReplacement>(
          SM, SpellingLocInfo.first, SpellingLocInfo.second, Text, nullptr));
      auto LocInfo = getOffsetAndLength(Begin, End);
      addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
    } else {
      // If the spelling location is inside the parent range, add string
      // replacement. The String replacement will be added to ExtReplacement
      // other where.
      // addExtReplacement(std::make_shared<ExtReplacement>(
      //  SM, SpellingLocInfo.first, SpellingLocInfo.second, Text, nullptr));
      auto LocInfo = getOffsetAndLength(Begin, End);
      addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
    }
  }

  inline void addReplacement(SourceLocation Begin, SourceLocation End,
                             unsigned TemplateIndex) {
    auto LocInfo = getOffsetAndLength(Begin, End);
    addReplacement(LocInfo.first, LocInfo.second, std::move(TemplateIndex));
  }

  // Replace string between begin location and end location.
  // Pass parent expr to calculate the correct location of macros
  inline void addReplacement(SourceLocation Begin, SourceLocation End,
                             const Expr *P, std::string Text) {
    if (!P)
      return addReplacement(Begin, End, std::move(Text));
    auto LocInfo = getOffsetAndLength(Begin, End, P);
    if (LocInfo.first + LocInfo.second < SrcLength) {
      auto SpellingLocInfo = getSpellingOffsetAndLength(Begin, End);
      if (SM.getDecomposedLoc(SpellingLocInfo.first).first != FileId ||
          SM.getDecomposedLoc(SpellingLocInfo.first).second < SrcBegin ||
          SM.getDecomposedLoc(SpellingLocInfo.first).second +
                  SpellingLocInfo.second >
              SrcBegin + SrcLength) {
        // If the spelling location is not in the parent range, add
        // ExtReplacement
        addExtReplacement(std::make_shared<ExtReplacement>(
            SM, SpellingLocInfo.first, SpellingLocInfo.second, Text, nullptr));
      } else if (SM.getDecomposedLoc(SpellingLocInfo.first).first == FileId &&
                 SM.getDecomposedLoc(SpellingLocInfo.first).second ==
                     SrcBegin &&
                 SM.getDecomposedLoc(SpellingLocInfo.first).second +
                         SpellingLocInfo.second ==
                     SrcBegin + SrcLength) {
        // If the spelling location is the same as the parent range, add both
        addExtReplacement(std::make_shared<ExtReplacement>(
            SM, SpellingLocInfo.first, SpellingLocInfo.second, Text, nullptr));
        addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
      } else {
        // If the spelling location is inside the parent range, add string
        // replacement. The String replacement will be added to ExtReplacement
        // other where.
        // addExtReplacement(std::make_shared<ExtReplacement>(
        //  SM, SpellingLocInfo.first, SpellingLocInfo.second, Text, nullptr));
        addReplacement(LocInfo.first, LocInfo.second, std::move(Text));
      }
    }
  }

  inline void addReplacement(SourceLocation Begin, SourceLocation End,
                             const Expr *P, unsigned TemplateIndex) {
    if (!P)
      return addReplacement(Begin, End, std::move(TemplateIndex));
    auto LocInfo = getOffsetAndLength(Begin, End, P);
    if (LocInfo.first > 0 && LocInfo.first < SrcLength) {
      addReplacement(LocInfo.first, LocInfo.second, std::move(TemplateIndex));
    }
  }

  // Replace total string
  inline void addReplacement(std::string Text) {
    addReplacement(0, SrcLength, std::move(Text));
  }

  inline void addReplacement(unsigned TemplateIndex) {
    addReplacement(0, SrcLength, std::move(TemplateIndex));
  }

  // Replace string with relative offset to the stored string and length
  inline void addReplacement(size_t Offset, size_t Length, std::string Text) {
    ReplSet.addStringReplacement(Offset, Length, std::move(Text));
  }

  inline void addReplacement(size_t Offset, size_t Length,
                             unsigned TemplateIndex) {
    ReplSet.addTemplateDependentReplacement(Offset, Length, TemplateIndex);
  }

  // Analyze the expression, jump to corresponding analysis function according
  // to its class
  // Precondition: Expression != nullptr
  virtual void dispatch(const Stmt *Expression);

  inline void analyzeExpr(const CastExpr *ICE) {
    return dispatch(ICE->getSubExpr());
  }

  inline void analyzeExpr(const MaterializeTemporaryExpr *MTE) {
    return dispatch(MTE->getSubExpr());
  }

  inline void analyzeExpr(const UnresolvedLookupExpr *ULE);

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
  inline void analyzeExpr(const ExprWithCleanups *EWC) {
    dispatch(EWC->getSubExpr());
  }

  void analyzeExpr(const CXXConstructExpr *Ctor);
  void analyzeExpr(const CXXTemporaryObjectExpr *Temp);
  void analyzeExpr(const CXXUnresolvedConstructExpr *Ctor);
  void analyzeExpr(const MemberExpr *ME);
  void analyzeExpr(const UnaryExprOrTypeTraitExpr *UETT);
  void analyzeExpr(const ExplicitCastExpr *Cast);
  void analyzeExpr(const CallExpr *CE);
  void analyzeExpr(const CXXMemberCallExpr *CMCE);
  void analyzeExpr(const CXXBindTemporaryExpr *CBTE);
  void analyzeExpr(const CompoundStmt *CS);
  void analyzeExpr(const ReturnStmt *RS);
  void analyzeExpr(const LambdaExpr *LE);
  void analyzeExpr(const IfStmt *IS);
  void analyzeExpr(const DeclStmt *DS);
  void analyzeExpr(const ConstantExpr *CE);
  void analyzeExpr(const InitListExpr *ILE);

  void removeCUDADeviceAttr(const LambdaExpr *LE);

  inline void analyzeType(const TypeSourceInfo *TSI,
                          const Expr *CSCE = nullptr) {
    analyzeType(TSI->getTypeLoc(), CSCE);
  }
  void analyzeType(TypeLoc TL, const Expr *E = nullptr,
                   const DependentNameTypeLoc *DNTL = nullptr,
                   const NestedNameSpecifierLoc *NNSL = nullptr);
  void analyzeDecltypeType(DecltypeTypeLoc TL);

  // Doing nothing when it doesn't need analyze
  inline void analyzeExpr(const Stmt *S) {}

  void analyzeTemplateArgument(const TemplateArgumentLoc &TAL);

  inline const Expr *getTargetExpr() { return E; }

  ASTContext &Context;
  const SourceManager &SM;

  std::string RefString;
  std::vector<std::shared_ptr<ExtReplacement>> SubExprRepl;
  bool IsInMacroDefine = false;

  bool BlockLevelFormatFlag = false;

private:
  // E is analyzed target expression, while ExprString is the source text of E.
  // Replacements contains all the replacements happened in E.
  const Expr *E;
  SourceLocation ExprBeginLoc;
  SourceLocation ExprEndLoc;
  SourceLocation SrcBeginLoc;
  size_t SrcBegin;
  size_t SrcLength;
  FileID FileId;
  StringReplacements ReplSet;
  std::string RewritePrefix;
  std::string RewritePostfix;
  std::set<HelperFeatureEnum> HelperFeatureSet;
};

// Analyze pointer allocated by cudaMallocManaged.
class ManagedPointerAnalysis : public ExprAnalysis {
  enum UseKind { NoUse = 0, Reference, Literal, Address };
  const CallExpr *Call;
  const Expr *FirstArg;
  const Expr *SecondArg;
  const VarDecl *Pointer;
  // class member pointer
  const FieldDecl *CPointer;
  const CompoundStmt *PointerScope;
  const CXXRecordDecl *CPointerScope;
  bool Assigned = false;
  bool Transfered = false;
  bool ReAssigned = false;
  bool Trackable = false;
  bool NeedDerefOp = true;
  std::vector<std::pair<std::pair<SourceLocation, SourceLocation>, std::string>>
      Repl;
  UseKind UK;
  std::string PointerTempType;
  std::string PointerCastType;
  std::string PointerName;

  void initAnalysisScope();
  void buildCallExprRepl();
  void dispatch(const Stmt *Expression) override;
  void dispatch(const Decl *Decleration);
  void addRepl();
  bool isInCudaPath(const Decl *Decleration);
  void analyzeExpr(const DeclRefExpr *DRE);
  void analyzeExpr(const MemberExpr *ME);
  void analyzeExpr(const CXXMemberCallExpr *CCE);
  void analyzeExpr(const CallExpr *CE);
  void analyzeExpr(const UnaryOperator *UO);
  void analyzeExpr(const BinaryOperator *BO);
  void analyzeExpr(const ArraySubscriptExpr *ASE);
  void analyzeExpr(const CXXRecordDecl *CRD);
  void analyzeExpr(const DeclStmt *DS);
  void analyzeExpr(const Stmt *S);

public:
  ManagedPointerAnalysis(const CallExpr *C, bool IsAssigned);
  ~ManagedPointerAnalysis() {}
  void RecursiveAnalyze();
};
/// Analyze expression used as argument.
class ArgumentAnalysis : public ExprAnalysis {
public:
  using Base = ExprAnalysis;
  ArgumentAnalysis() {}
  ArgumentAnalysis(bool IsInMacroDefine) {
    this->IsInMacroDefine = IsInMacroDefine;
  }
  // Special init is needed for argument expression.
  ArgumentAnalysis(const Expr *Arg, bool IsInMacroDefine = false) : Base(nullptr) {
    this->IsInMacroDefine = IsInMacroDefine;
    initArgumentExpr(Arg);
  }

  inline void analyze() { Base::analyze(); }

  // Special init is needed for argument expression.
  void analyze(const Expr *Expression) {
    initArgumentExpr(Expression);
    auto ExprBeginBeforeAnalyze = getExprBeginSrcLoc();
    analyze();
    int ReplLength = getExprLength();
    if (ReplLength > 0) {
      addExtReplacement(std::make_shared<ExtReplacement>(
          SM, ExprBeginBeforeAnalyze, getExprLength(), getReplacedString(),
          nullptr));
    }
  }

  inline void setCallSpelling(const Expr *E) {
    auto LocInfo = getSpellingOffsetAndLength(E);
    CallSpellingBegin = LocInfo.first;
    CallSpellingEnd = CallSpellingBegin.getLocWithOffset(LocInfo.second);
  }

  inline void setCallSpelling(SourceLocation Begin, SourceLocation End) {
    CallSpellingBegin = Begin;
    CallSpellingEnd = End;
  }

  std::string getRewriteString();

  std::pair<SourceLocation, SourceLocation> getLocInCallSpelling(const Expr *E);

protected:
  // Ignore the constructor when it's argument expression, it is copy/move
  // constructor and no migration for it. Start analyzing its argument.
  // Replace total string when it is default argument expression.
  void initArgumentExpr(const Expr *Expression) {
    if (!Expression)
      initExpression(Expression);
    if (auto Ctor = dyn_cast<CXXConstructExpr>(Expression)) {
      if (Ctor->getParenOrBraceRange().isInvalid() && Ctor->getNumArgs() == 1)
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
  bool IsRedeclareRequired = false;
  bool IsPointer = false;
  bool TryGetBuffer = false;
  bool IsDoublePointer = false;

  KernelArgumentAnalysis(bool IsInMacroDefine)
      : ArgumentAnalysis(IsInMacroDefine) {}
  void analyze(const Expr *Expression);

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
  inline void analyzeExpr(const CXXTemporaryObjectExpr *Temp);
  inline void analyzeExpr(const CXXDependentScopeMemberExpr *Arg);
  inline void analyzeExpr(const MaterializeTemporaryExpr *MTE);
  inline void analyzeExpr(const LambdaExpr *LE);

  bool isNullPtr(const Expr *);

  bool IsDefinedOnDevice = false;
  bool IsAddrOf = false;
};

class KernelConfigAnalysis : public ArgumentAnalysis {
private:
  bool DoReverse = false;
  bool Reversed = false;
  bool DirectRef = false;
  bool IsDim3Config = false;
  unsigned int ArgIndex = 0;
  bool NeedEmitWGSizeWarning = true;
  unsigned int SizeOfHighestDimension = 0;

  void analyzeExpr(const CXXConstructExpr *Ctor);
  void analyzeExpr(const CXXUnresolvedConstructExpr *Ctor);
  void analyzeExpr(const CXXTemporaryObjectExpr *Ctor);
  void analyzeExpr(const ExplicitCastExpr *Cast);
  void analyzeExpr(const DeclRefExpr *DRE);

  template <class T, class ArgIter>
  void handleDim3Ctor(const T *, SourceRange Parens, ArgIter ArgBegin,
                      ArgIter ArgEnd);
  template <class T, class ArgIter>
  void handleDim3Args(const T *Ctor, ArgIter ArgBegin, ArgIter ArgEnd);

protected:
  void dispatch(const Stmt *Expression) override;

public:
  KernelConfigAnalysis(bool IsInMacroDefine)
      : ArgumentAnalysis(IsInMacroDefine) {}
  void analyze(const Expr *E, unsigned int Idx, bool ReverseIfNeed = false);

  inline bool reversed() { return Reversed; }
  inline bool isDirectRef() { return DirectRef; }
  inline bool isNeedEmitWGSizeWarning() { return NeedEmitWGSizeWarning; }
  unsigned int getSizeOfHighestDimension() { return SizeOfHighestDimension; }
  unsigned int Dim = 3;
  bool IsTryToUseOneDimension = false;
  SmallVector<std::string, 3> Dim3Args;
};

class FunctorAnalysis : public ArgumentAnalysis {
private:
  void analyzeExpr(const CXXTemporaryObjectExpr* CTOE);
  void analyzeExpr(const DeclRefExpr* DRE);
  void analyzeExpr(const CXXConstructExpr *CCE);
  void analyzeExpr(const CXXFunctionalCastExpr *CFCE);
  void addConstQuailfier(const CXXRecordDecl *CRD);
  unsigned PlaceholderCount = 0;
protected:
  void dispatch(const Stmt *S) override;
public:
  void analyze(const Expr *E);
  FunctorAnalysis() : ArgumentAnalysis() {}
};

/// Analyzes the side effects of an expression while doing basic expression
/// analysis
class SideEffectsAnalysis : public ArgumentAnalysis {
public:
  explicit SideEffectsAnalysis(const Expr *E) : ArgumentAnalysis(E) {}
  inline bool hasSideEffects() { return HasSideEffects; }

protected:
  void dispatch(const Stmt *Expression) override;

private:
  using Base = ExprAnalysis;

  inline void analyzeExpr(const BinaryOperator *BO) {
    switch (BO->getOpcode()) {
    // Some binary operators have side effects
    case BO_Assign:
    case BO_MulAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_ShlAssign:
    case BO_ShrAssign:
    case BO_AndAssign:
    case BO_XorAssign:
    case BO_OrAssign:
      HasSideEffects = true;
      break;
    default:
      break;
    }
    Base::analyzeExpr(BO);
  }

private:
  bool HasSideEffects = false;
};

class IndexAnalysis : public ExprAnalysis {
public:
  explicit IndexAnalysis(const Expr *E) : ExprAnalysis() { dispatch(E); }
  inline bool isDifferenceBetweenThreadIdxXAndIndexConstant() {
    return isStrictlyMonotonic() && !IsThreadIdxXUnderNonAdditiveOp;
  }
  inline bool isStrictlyMonotonic() {
    return HasThreadIdxX && !ContainUnknownNode;
  }

protected:
  void dispatch(const Stmt *Expression) override;

private:
  using Base = ExprAnalysis;
  void analyzeExpr(const UnaryOperator *UO);
  void analyzeExpr(const BinaryOperator *BO);
  void analyzeExpr(const ImplicitCastExpr *ICE);
  void analyzeExpr(const DeclRefExpr *DRE);
  void analyzeExpr(const PseudoObjectExpr *POE);

private:
  bool ContainUnknownNode = false;
  bool HasThreadIdxX = false;
  bool IsThreadIdxXUnderNonAdditiveOp = false;
  std::stack<bool> ContainNonAdditiveOp;
};

} // namespace dpct
} // namespace clang

#endif // !DPCT_EXPR_analyze_H
