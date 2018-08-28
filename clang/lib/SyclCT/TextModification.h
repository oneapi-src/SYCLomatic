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

#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"
#include <string>

namespace clang {
namespace syclct {

class TextModification;
using TransformSetTy = std::vector<std::unique_ptr<TextModification>>;

/// Base class for compatibility tool-related source code modifications.
class TextModification {
public:
  virtual ~TextModification() {}
  /// Generate actual Replacement from this TextModification object.
  virtual tooling::Replacement
  getReplacement(const ASTContext &Context) const = 0;
};

/// For macros and typedefs source location is unreliable (begin and end of the
/// source range point to the same character. Replacing by token is a simple
/// workaround.
class ReplaceToken : public TextModification {
  SourceLocation Begin;
  std::string T;

public:
  ReplaceToken(SourceLocation Loc, std::string &&S) : Begin(Loc), T(S) {}
  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

/// Replace a statement (w/o semicolon) with a specified string.
class ReplaceStmt : public TextModification {
  const Stmt *TheStmt;
  std::string ReplacementString;

public:
  ReplaceStmt(const Stmt *E, std::string &&S)
      : TheStmt(E), ReplacementString(S) {}

  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

/// Replace C-style cast with constructor call for a given type.
class ReplaceCCast : public TextModification {
  const CStyleCastExpr *Cast;
  std::string TypeName;

public:
  ReplaceCCast(const CStyleCastExpr *Cast, std::string &&TypeName)
      : Cast(Cast), TypeName(TypeName) {}
  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

/// Remove an attribute from a declaration.
class RemoveAttr : public TextModification {
  const Attr *TheAttr;

public:
  RemoveAttr(const Attr *A) : TheAttr(A) {}
  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

// Replace type in var. declaration.
class ReplaceTypeInVarDecl : public TextModification {
  const VarDecl *D;
  std::string T;

public:
  ReplaceTypeInVarDecl(const VarDecl *D, std::string &&T) : D(D), T(T) {}
  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

// Replace return type in function declaration.
class ReplaceReturnType : public TextModification {
  const FunctionDecl *FD;
  std::string T;

public:
  ReplaceReturnType(const FunctionDecl *FD, std::string &&T) : FD(FD), T(T) {}
  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

// Rename field in expression.
class RenameFieldInMemberExpr : public TextModification {
  const MemberExpr *ME;
  std::string T;

public:
  RenameFieldInMemberExpr(const MemberExpr *ME, std::string &&T)
      : ME(ME), T(T) {}

  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

class InsertAfterStmt : public TextModification {
  const Stmt *S;
  std::string T;

public:
  InsertAfterStmt(const Stmt *S, std::string &&T) : S(S), T(T) {}

  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

// Insert '/*  */' C style multi line comment
class InsertComment : public TextModification {
  // The comment will be inserted at this position
  SourceLocation SL;
  std::string Text;

public:
  InsertComment(SourceLocation SL, std::string Text) : SL(SL), Text(Text) {}

  tooling::Replacement getReplacement(const ASTContext &Context) const override;
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

  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

class InsertArgument : public TextModification {
  const FunctionDecl *FD;
  // Argument string without comma.
  std::string ArgName;

public:
  InsertArgument(const FunctionDecl *FD, std::string &&ArgName)
      : FD(FD), ArgName(ArgName) {}

  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

class ReplaceInclude : public TextModification {
  CharSourceRange Range;
  std::string T;

public:
  ReplaceInclude(CharSourceRange Range, std::string &&T) : Range(Range), T(T) {}
  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

class ReplaceKernelCallExpr : public TextModification {
  const CUDAKernelCallExpr *KCall;

public:
  ReplaceKernelCallExpr(const CUDAKernelCallExpr *KCall) : KCall(KCall) {}
  tooling::Replacement getReplacement(const ASTContext &Context) const override;
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

  const std::vector<tooling::Replacement> &ReplSet;
  std::map<std::string, IntervalSet> FileMap;

private:
  bool containsInterval(const IntervalSet &IS, const Interval &I) const;
  bool isDeletedReplacement(const tooling::Replacement &R) const;
  size_t findFirstNotDeletedReplacement(size_t Start) const;

  class iterator {
    const ReplacementFilter &RF;
    size_t Idx;

  public:
    iterator(const ReplacementFilter &RF, size_t Idx) : RF(RF), Idx(Idx) {}
    const tooling::Replacement &operator*() const { return RF.ReplSet[Idx]; }
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
  ReplacementFilter(const std::vector<tooling::Replacement> &RS);

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

  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

class RemoveArg : public TextModification {
  const CallExpr *CE;
  const unsigned N;

public:
  RemoveArg(const CallExpr *CE, const unsigned N) : CE(CE), N(N) {}

  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

class InsertBeforeCtrInitList : public TextModification {
  const CXXConstructorDecl *CDecl;
  std::string T;

public:
  InsertBeforeCtrInitList(const CXXConstructorDecl *S, std::string &&T)
      : CDecl(S), T(T) {}

  tooling::Replacement getReplacement(const ASTContext &Context) const override;
};

} // namespace syclct
} // namespace clang

#endif // SYCLCT_TEXT_MODIFICATION_H
