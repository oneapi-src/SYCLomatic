//===--------------- RecommendLibraries.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef RECOMMEND_LIBRARIES_H
#define RECOMMEND_LIBRARIES_H

#include "AnalysisInfo.h"
#include "FileGenerator/GenFiles.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <unordered_map>

namespace clang {
namespace dpct {
enum class LibType { DPCPP, oneDPL, oneMath, oneCCL, oneDNNL, ISHMEM };

void CollectDepLib(ReplTy &MainSrcFilesRepls);
void PrintRecommendLibs(llvm::raw_ostream &OStream);
} // namespace dpct
} // namespace clang
#endif // RECOMMEND_LIBRARIES_H
