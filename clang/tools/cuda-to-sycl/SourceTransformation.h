//===--- SourceTransformation.h -------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef CU2SYCL_SOURCETRANSFORMATION_H
#define CU2SYCL_SOURCETRANSFORMATION_H

#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"

namespace clang {
namespace cu2sycl {

class SourceTransformation;
using TransformSetTy = std::vector<std::unique_ptr<SourceTransformation>>;

class SourceTransformation {
protected:
  static unsigned getLength(const SourceLocation &Begin,
                            const SourceLocation &End,
                            const SourceManager &SM) {
    return SM.getCharacterData(End) - SM.getCharacterData(Begin) + 1;
  }

  static unsigned getLength(const SourceRange &SR, const SourceManager &SM) {
    return getLength(SR.getBegin(), SR.getEnd(), SM);
  }

public:
  virtual ~SourceTransformation() {}
  virtual tooling::Replacement
  getReplacement(const SourceManager &SM) const = 0;
  virtual const char *getTransformationID() const = 0;
};

class CudaBlockDim : public SourceTransformation {
  const MemberExpr &ME;
  unsigned Dimension;

  static const char ID;

public:
  CudaBlockDim(const MemberExpr &ME, char D) : ME(ME), Dimension(D) {}

  const MemberExpr *getMemberExpr() const { return &ME; }
  unsigned getDimension() const { return Dimension; }
  tooling::Replacement getReplacement(const SourceManager &SM) const override;

  const char *getTransformationID() const override { return &ID; }
  static bool classof(const SourceTransformation *ST) {
    return ST->getTransformationID() == &ID;
  }
};

class CudaThreadIdx : public SourceTransformation {
  const MemberExpr &ME;
  unsigned Dimension;

  static const char ID;

public:
  CudaThreadIdx(const MemberExpr &ME, char D) : ME(ME), Dimension(D) {}

  const MemberExpr *getMemberExpr() const { return &ME; }
  unsigned getDimension() const { return Dimension; }
  tooling::Replacement getReplacement(const SourceManager &SM) const override;

  const char *getTransformationID() const override { return &ID; }
  static bool classof(const SourceTransformation *ST) {
    return ST->getTransformationID() == &ID;
  }
};

class SyclItemLinearID : public SourceTransformation {
  const SourceLocation Begin;
  const SourceLocation End;

  static const char ID;

public:
  SyclItemLinearID(SourceLocation B, SourceLocation E) : Begin(B), End(E) {}
  tooling::Replacement getReplacement(const SourceManager &SM) const override;

  const char *getTransformationID() const override { return &ID; }
  static bool classof(const SourceTransformation *ST) {
    return ST->getTransformationID() == &ID;
  }
};

} // namespace cu2sycl
} // namespace clang

#endif // CU2SYCL_SOURCETRANSFORMATION_H
