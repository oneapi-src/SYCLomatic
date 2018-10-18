//===--- TextModification.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_TEXT_MODIFICATION_H
#define SYCLCT_TEXT_MODIFICATION_H

#include "MapNames.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"

#include <string>

namespace clang {
namespace syclct {

class TextModification;
using TransformSetTy = std::vector<std::unique_ptr<TextModification>>;
enum InsertPosition {
  InsertPositionLeft = 0,
  InsertPositionRight,
};

/// Extend Replacemnt to contain more meta info of Replament inserted by
/// AST Rule. Further Analysis Pass like Merge Pass can happen based
/// on these meta info of Replament.
///  eg. Replament happen at same position may be merged to avoid conflict.
class ExtReplacement : public tooling::Replacement {
public:
  /// Creates an invalid (not applicable) replacement.
  ExtReplacement() : Replacement(){};

  /// Creates a replacement of the range [Offset, Offset+Length) in
  /// FilePath with ReplacementText.
  ///
  /// \param FilePath A source file accessible via a SourceManager.
  /// \param Offset The byte offset of the start of the range in the file.
  /// \param Length The length of the range in bytes.
  ExtReplacement(StringRef FilePath, unsigned Offset, unsigned Length,
                 StringRef ReplacementText)
      : Replacement(FilePath, Offset, Length, ReplacementText) {}

  /// Creates a Replacement of the range [Start, Start+Length) with
  /// ReplacementText.
  ExtReplacement(const SourceManager &Sources, SourceLocation Start,
                 unsigned Length, StringRef ReplacementText)
      : Replacement(Sources, Start, Length, ReplacementText) {}

  /// Creates a Replacement of the given range with ReplacementText.
  ExtReplacement(const SourceManager &Sources, const CharSourceRange &Range,
                 StringRef ReplacementText,
                 const LangOptions &LangOpts = LangOptions())
      : Replacement(Sources, Range, ReplacementText, LangOpts) {}

  /// Creates a Replacement of the node with ReplacementText.
  template <typename Node>
  ExtReplacement(const SourceManager &Sources, const Node &NodeToReplace,
                 StringRef ReplacementText,
                 const LangOptions &LangOpts = LangOptions())
      : Replacement(Sources, NodeToReplace, ReplacementText, LangOpts) {}
  void setInsertPosition(int IP) { InsertPosition = IP; }
  unsigned int getInsertPosition() const { return InsertPosition; }
  bool getMerged() const { return Merged; }
  void setMerged(bool M) { Merged = M; }

private:
  unsigned int InsertPosition = InsertPositionLeft;
  bool Merged = false;
};

/// Base class for compatibility tool-related source code modifications.
class TextModification {
public:
  virtual ~TextModification() {}
  /// Generate actual Replacement from this TextModification object.
  virtual ExtReplacement getReplacement(const ASTContext &Context) const = 0;
};
///  Insert string in given position.
class InsertText : public TextModification {
  SourceLocation Begin;
  std::string T;

public:
  InsertText(SourceLocation Loc, std::string &&S) : Begin(Loc), T(S) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

/// For macros and typedefs source location is unreliable (begin and end of the
/// source range point to the same character. Replacing by token is a simple
/// workaround.
class ReplaceToken : public TextModification {
  SourceLocation Begin;
  std::string T;

public:
  ReplaceToken(SourceLocation Loc, std::string &&S) : Begin(Loc), T(S) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

/// Replace a statement (w/o semicolon) with a specified string.
class ReplaceStmt : public TextModification {
  const Stmt *TheStmt;
  std::string ReplacementString;

public:
  ReplaceStmt(const Stmt *E, std::string &&S)
      : TheStmt(E), ReplacementString(S) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

/// Replace C-style cast with constructor call for a given type.
class ReplaceCCast : public TextModification {
  const CStyleCastExpr *Cast;
  std::string TypeName;

public:
  ReplaceCCast(const CStyleCastExpr *Cast, std::string &&TypeName)
      : Cast(Cast), TypeName(TypeName) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

/// Remove an attribute from a declaration.
class RemoveAttr : public TextModification {
  const Attr *TheAttr;

public:
  RemoveAttr(const Attr *A) : TheAttr(A) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

// Replace type in var. declaration.
class ReplaceTypeInVarDecl : public TextModification {
  const VarDecl *D;
  std::string T;

public:
  ReplaceTypeInVarDecl(const VarDecl *D, std::string &&T) : D(D), T(T) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

// Replace type in var. declaration.
class InsertNameSpaceInVarDecl : public TextModification {
  const VarDecl *D;
  std::string T;
  unsigned int InsertPosition;

public:
  InsertNameSpaceInVarDecl(const VarDecl *D, std::string &&T,
                           unsigned int InsertPosition = 0)
      : D(D), T(T), InsertPosition(InsertPosition) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

// Replace type in var. declaration.
class InsertNameSpaceInCastExpr : public TextModification {
  const CStyleCastExpr *D;
  std::string T;

public:
  InsertNameSpaceInCastExpr(const CStyleCastExpr *D, std::string &&T)
      : D(D), T(T) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

// Replace type in var. declaration.
class RemoveVarDecl : public TextModification {
  const VarDecl *D;
  std::string T;

public:
  RemoveVarDecl(const VarDecl *D, std::string &&T) : D(D), T(T) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

// Replace return type in function declaration.
class ReplaceReturnType : public TextModification {
  const FunctionDecl *FD;
  std::string T;

public:
  ReplaceReturnType(const FunctionDecl *FD, std::string &&T) : FD(FD), T(T) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

// Rename field in expression.
class RenameFieldInMemberExpr : public TextModification {
  const MemberExpr *ME;
  std::string T;
  unsigned PositionOfDot;

public:
  RenameFieldInMemberExpr(const MemberExpr *ME, std::string &&T,
                          unsigned PositionOfDot = 0)
      : ME(ME), T(T), PositionOfDot(PositionOfDot) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

class InsertAfterStmt : public TextModification {
  const Stmt *S;
  std::string T;

public:
  InsertAfterStmt(const Stmt *S, std::string &&T) : S(S), T(T) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

// Insert '/*  */' C style multi line comment
class InsertComment : public TextModification {
  // The comment will be inserted at this position
  SourceLocation SL;
  std::string Text;

public:
  InsertComment(SourceLocation SL, std::string Text) : SL(SL), Text(Text) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

/// Replace CallExpr with another call.
// TODO: return values are not handled.
// TODO: we probably need more genric class, which would take the list of
// strings and expressions and compose them to a single srting, also doing look
// up for already modified expressions and use their new spelling when needed.
class ReplaceCallExpr : public TextModification {
  // Call to replace.
  const CallExpr *C;
  // New function name.
  std::string Name;
  // New function params.
  std::vector<const Expr *> Args;
  // New function type.
  std::vector<std::string> Types;

public:
  ReplaceCallExpr(const CallExpr *Call, std::string &&NewName,
                  std::vector<const Expr *> &&NewArgs)
      : C(Call), Name(NewName), Args(NewArgs) {}

  ReplaceCallExpr(const CallExpr *Call, std::string &&NewName,
                  std::vector<const Expr *> &&NewArgs,
                  std::vector<std::string> NewTypes)
      : C(Call), Name(NewName), Args(NewArgs), Types(NewTypes) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

class InsertArgument : public TextModification {
  const FunctionDecl *FD;
  // Argument string without comma.
  std::string ArgName;
  bool Lazy = false;

public:
  InsertArgument(const FunctionDecl *FD, std::string &&ArgName)
      : FD(FD), ArgName(ArgName) {}
  InsertArgument(const FunctionDecl *FD, std::string &&ArgName, bool Lazy)
      : FD(FD), ArgName(ArgName), Lazy(Lazy) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

class ReplaceInclude : public TextModification {
  CharSourceRange Range;
  std::string T;

public:
  ReplaceInclude(CharSourceRange Range, std::string &&T) : Range(Range), T(T) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

class ReplaceKernelCallExpr : public TextModification {
  const CUDAKernelCallExpr *KCall;

  std::pair<const Expr *, const Expr *> getExecutionConfig() const;
  static std::string getDim3Translation(const Expr *E,
                                        const ASTContext &Context,
                                        unsigned int EffectDims);
  static unsigned int getDimsNum(const Expr *);

  static std::string
  buildTemplateArgList(const llvm::ArrayRef<TemplateArgument> &Args,
                       const ASTContext &Context) {
    std::string ArgsList;
    llvm::raw_string_ostream OStream(ArgsList);
    PrintingPolicy Policy(Context.getLangOpts());
    bool NotBegin = false;
    for (auto Arg : Args) {
      if (NotBegin)
        OStream << ", ";
      else
        NotBegin = true;
      Arg.print(Policy, OStream);
    }
    return OStream.str();
  }

  static std::string getTemplateArgs(const SourceLocation &L,
                                     const SourceLocation &R,
                                     const SourceManager &SM) {
    auto CharBegin = SM.getCharacterData(L);
    auto CharEnd = SM.getCharacterData(R);
    while (*CharBegin++ != '<')
      ;
    while (*CharEnd != '>')
      CharEnd--;
    return std::string(CharBegin, CharEnd);
  }

public:
  ReplaceKernelCallExpr(const CUDAKernelCallExpr *KCall) : KCall(KCall) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

/// A class that filters out Replacements that modify text inside a deleted code
/// block.
class ReplacementFilter {
  struct Interval {
    size_t Offset;
    size_t Length;
    bool operator<(const Interval &Other) const {
      return Offset < Other.Offset;
    }
  };

  using IntervalSet = std::vector<Interval>;

  const std::vector<ExtReplacement> &ReplSet;
  std::map<std::string, IntervalSet> FileMap;

private:
  bool containsInterval(const IntervalSet &IS, const Interval &I) const;
  bool isDeletedReplacement(const ExtReplacement &R) const;
  size_t findFirstNotDeletedReplacement(size_t Start) const;

  class iterator {
    const ReplacementFilter &RF;
    size_t Idx;

  public:
    iterator(const ReplacementFilter &RF, size_t Idx) : RF(RF), Idx(Idx) {}
    const ExtReplacement &operator*() const { return RF.ReplSet[Idx]; }
    iterator &operator++() {
      Idx = RF.findFirstNotDeletedReplacement(Idx + 1);
      return *this;
    }
    bool operator==(const iterator &Other) const {
      assert(&RF == &Other.RF && "Mismatching iterators");
      return Idx == Other.Idx;
    }
    bool operator!=(const iterator &Other) const { return !operator==(Other); }
  };

public:
  ReplacementFilter(const std::vector<ExtReplacement> &RS);

  iterator begin() {
    return iterator(*this, findFirstNotDeletedReplacement(0));
  }
  iterator end() { return iterator(*this, -1); }
};

class InsertBeforeStmt : public TextModification {
  const Stmt *S;
  std::string T;

public:
  InsertBeforeStmt(const Stmt *S, std::string &&T) : S(S), T(T) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

class RemoveArg : public TextModification {
  const CallExpr *CE;
  const unsigned N;

public:
  RemoveArg(const CallExpr *CE, const unsigned N) : CE(CE), N(N) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

class InsertBeforeCtrInitList : public TextModification {
  const CXXConstructorDecl *CDecl;
  std::string T;

public:
  InsertBeforeCtrInitList(const CXXConstructorDecl *S, std::string &&T)
      : CDecl(S), T(T) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

class InsertClassName : public TextModification {
  const CXXRecordDecl *CD;
  static unsigned Count;

public:
  InsertClassName(const CXXRecordDecl *C) : CD(C) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
};

} // namespace syclct
} // namespace clang

#endif // SYCLCT_TEXT_MODIFICATION_H
