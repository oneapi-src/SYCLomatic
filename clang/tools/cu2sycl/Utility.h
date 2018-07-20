//===--- Utility.h -------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef CU2SYCL_UTILITY_H
#define CU2SYCL_UTILITY_H

#include <string>

namespace llvm {
template <typename T> class SmallVectorImpl;
}

bool makeCanonical(llvm::SmallVectorImpl<char> &Path);
bool makeCanonical(std::string &Path);
bool isCanonical(const std::string &Path);

#endif // CU2SYCL_UTILITY_H
