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

/// Replace an expression (w/o semicolon) with a specified string.
class ReplaceExpr : public TextModification {
  const Expr *TheExpr;
  std::string ReplacementString;

public:
  ReplaceExpr(const Expr *E, std::string &&S)
      : TheExpr(E), ReplacementString(S) {}

  tooling::Replacement getReplacement(const SourceManager &SM) const override;
};

/// Remove an attribute from a declaration.
class RemoveAttr : public TextModification {
  const Attr *TheAttr;

public:
  RemoveAttr(const Attr *A) : TheAttr(A) {}
  tooling::Replacement getReplacement(const SourceManager &SM) const override;
};

} // namespace cu2sycl
} // namespace clang

#endif // CU2SYCL_TEXT_MODIFICATION_H
