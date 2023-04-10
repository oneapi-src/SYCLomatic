//===--------------- CrashRecovery.h -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_CRASHRECOVERY_H
#define DPCT_CRASHRECOVERY_H

#include "llvm/ADT/STLFunctionalExtras.h"

#include <string>

namespace clang {
namespace dpct {

bool runWithCrashGuard(llvm::function_ref<void()>, std::string);
void initCrashRecovery();

} // namespace dpct
} // namespace clang

extern int FatalErrorCnt;
extern int FatalErrorASTCnt;
extern bool CurFileMeetErr;

#endif // DPCT_CRASHRECOVERY_H
