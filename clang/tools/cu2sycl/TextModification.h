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

#ifndef CU2SYCL_TEXT_MODIFICATION_H
#define CU2SYCL_TEXT_MODIFICATION_H

#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"

namespace clang {
namespace cu2sycl {

class TextModification;
using TransformSetTy = std::vector<std::unique_ptr<TextModification>>;

/// Base class for translator-related source code modifications.
class TextModification {
public:
  virtual ~TextModification() {}

  /// Generate actual Replacement from this TextModification object.
  virtual tooling::Replacement
  getReplacement(const SourceManager &SM) const = 0;
};

/// Replace a statement (w/o semicolon) with a specified string.
class ReplaceStmt : public TextModification {
  const Stmt *TheStmt;
  std::string ReplacementString;

public:
  ReplaceStmt(const Stmt *E, std::string &&S)
      : TheStmt(E), ReplacementString(S) {}

  tooling::Replacement getReplacement(const SourceManager &SM) const override;
};

/// Remove an attribute from a declaration.
class RemoveAttr : public TextModification {
  const Attr *TheAttr;

public:
  RemoveAttr(const Attr *A) : TheAttr(A) {}
  tooling::Replacement getReplacement(const SourceManager &SM) const override;
};

// Replace type in var. declaration.
class ReplaceTypeInVarDecl : public TextModification {
  const VarDecl *D;
  std::string T;

public:
  ReplaceTypeInVarDecl(const VarDecl *D, std::string &&T) : D(D), T(T) {}
  tooling::Replacement getReplacement(const SourceManager &SM) const override;
};

// Replace return type in function declaration.
class ReplaceReturnType : public TextModification {
  const FunctionDecl *FD;
  std::string T;

public:
  ReplaceReturnType(const FunctionDecl *FD, std::string &&T) : FD(FD), T(T) {}
  tooling::Replacement getReplacement(const SourceManager &SM) const override;
};

// Rename field in expression.
class RenameFieldInMemberExpr : public TextModification {
  const MemberExpr *ME;
  std::string T;

public:
  RenameFieldInMemberExpr(const MemberExpr *ME, std::string &&T)
      : ME(ME), T(T) {}

  tooling::Replacement getReplacement(const SourceManager &SM) const override;
};

class InsertAfterStmt : public TextModification {
  const Stmt *S;
  std::string T;

public:
  InsertAfterStmt(const Stmt *S, std::string &&T) : S(S), T(T) {}

  tooling::Replacement getReplacement(const SourceManager &SM) const override;
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
  InsertBeforeStmt(const Stmt *S, std::string &&T)
      : S(S), T(T) {}

  tooling::Replacement getReplacement(const SourceManager &SM) const override;
};

class RemoveArg : public TextModification {
  const CallExpr *CE;
  const unsigned N;

public:
  RemoveArg(const CallExpr *CE, const unsigned N)
      : CE(CE), N(N) {}

  tooling::Replacement getReplacement(const SourceManager &SM) const override;
};

} // namespace cu2sycl
} // namespace clang

#endif // CU2SYCL_TEXT_MODIFICATION_H
