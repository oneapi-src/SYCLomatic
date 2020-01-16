//===--- ExternalReplacement.h --------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef __EXTERNAL_REPLACEMENT_H__
#define __EXTERNAL_REPLACEMENT_H__

#include <map>

namespace llvm {
class StringRef;
}

namespace clang {
namespace tooling {
class RefactoringTool;
class Replacements;
}
} // namespace clang

int mergeExternalReps(std::string SrcFileName,
                      clang::tooling::Replacements &Replaces);
#endif
