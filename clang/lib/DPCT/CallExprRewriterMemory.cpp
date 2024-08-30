//===--------------- CallExprRewriterMemory.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"

namespace clang {
namespace dpct {

/// Get helper function name with namespace which has 'dpct_' in dpct helper
/// functions and w/o in syclcompat.
/// If has "_async" suffix, the name in dpct helper function will have 'async_'
/// prefix and remove the suffix.
/// If `ExperimentalInSYCLCompat` is true, will add `experimental` namespace
/// in syclcompat.
std::string getMemoryHelperFunctionName(StringRef RawName,
                                        bool ExperimentalInSYCLCompat = false) {
  const static std::string AsyncSuffix = "_async";
  const static std::string AsyncPrefix = "async_";

  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << MapNames::getDpctNamespace();
  if (!DpctGlobalInfo::useSYCLCompat()) {
    if (RawName.ends_with(AsyncSuffix)) {
      RawName = RawName.drop_back(AsyncSuffix.length());
      OS << AsyncPrefix;
    }
    OS << "dpct_";
  } else if (ExperimentalInSYCLCompat) {
    OS << "experimental::";
  }
  OS << RawName;
  return Result;
}

std::string MemoryMigrationRule::getMemoryHelperFunctionName(
    StringRef Name, bool ExperimentalInSYCLCompat) {
  return dpct::getMemoryHelperFunctionName(Name, ExperimentalInSYCLCompat);
}

// clang-format off
void CallExprRewriterFactoryBase::initRewriterMapMemory() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#include "APINamesMemory.inc"
      }));
      // clang-format on
}

} // namespace dpct
} // namespace clang
