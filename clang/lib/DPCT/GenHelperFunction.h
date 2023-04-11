//===--------------- GenHelperFunction.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_GEN_HELPER_FUNCTION_H
#define DPCT_GEN_HELPER_FUNCTION_H

#include <string>

namespace clang {
namespace dpct {
void genHelperFunction(const std::string &OutRoot);
}
} // namespace clang

#endif // DPCT_GEN_HELPER_FUNCTION_H
