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
extern const std::string DpctAllContentStr;
extern const std::string AtomicAllContentStr;
extern const std::string BlasUtilsAllContentStr;
extern const std::string DnnlUtilsAllContentStr;
extern const std::string DeviceAllContentStr;
extern const std::string DplUtilsAllContentStr;
extern const std::string ImageAllContentStr;
extern const std::string KernelAllContentStr;
extern const std::string MathAllContentStr;
extern const std::string MemoryAllContentStr;
extern const std::string UtilAllContentStr;
extern const std::string RngUtilsAllContentStr;
extern const std::string LibCommonUtilsAllContentStr;
extern const std::string CclUtilsAllContentStr;
extern const std::string SparseUtilsAllContentStr;
extern const std::string FftUtilsAllContentStr;
extern const std::string LapackUtilsAllContentStr;
extern const std::string DplExtrasAlgorithmAllContentStr;
extern const std::string DplExtrasFunctionalAllContentStr;
extern const std::string DplExtrasIteratorsAllContentStr;
extern const std::string DplExtrasMemoryAllContentStr;
extern const std::string DplExtrasNumericAllContentStr;
extern const std::string DplExtrasVectorAllContentStr;
extern const std::string DplExtrasDpcppExtensionsAllContentStr;
void replaceEndOfLine(std::string &StrNeedProcess);
void genHelperFunction(const std::string &OutRoot);
}
} // namespace clang

#endif // DPCT_GEN_HELPER_FUNCTION_H
