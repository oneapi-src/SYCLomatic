//===--- ValidateArguments.h ---------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef SYCLCT_VALIDATE_ARGUMENTS_H
#define SYCLCT_VALIDATE_ARGUMENTS_H

#include <string>
#include <vector>

namespace llvm {
template <typename T> class SmallVectorImpl;
}

bool makeCanonicalOrSetDefaults(std::string &InRoot, std::string &OutRoot,
                                const std::vector<std::string> SourceFiles);

// Make sure files passed to SYCL Compatibility Tool are under the input root directory
// and have an extension.
bool validatePaths(const std::string &InRoot,
                   const std::vector<std::string> &SourceFiles);
#endif // SYCLCT_VALIDATE_ARGUMENTS_H
