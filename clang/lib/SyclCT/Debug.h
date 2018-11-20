//===--- Debug.h ---------------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef SYCLCT_DEBUG_H
#define SYCLCT_DEBUG_H

#ifndef NDEBUG // Debug build

#include "llvm/Support/Debug.h"

// General debug information with TYPE = "syclct"
#define SYCLCT_DEBUG(X) DEBUG_WITH_TYPE("syclct", X)

// End of Debug Build

#else // Release build

#define SYCLCT_DEBUG(X)                                                        \
  do {                                                                         \
  } while (false)

#endif // End of Release build

#endif // SYCLCT_DEBUG_H
