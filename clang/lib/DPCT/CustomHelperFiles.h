//===--------------- CustomHelperFiles.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_CUSTOM_HELPER_FILES_H
#define DPCT_CUSTOM_HELPER_FILES_H

#include <string>

namespace clang {
namespace dpct {

std::string getDpctVersionStr();

enum class HelperFeatureEnum : unsigned int {
  device_ext,
  no_feature_helper,
};

void requestFeature(HelperFeatureEnum Feature);
void requestHelperFeatureForEnumNames(const std::string Name);
void requestHelperFeatureForTypeNames(const std::string Name);

} // namespace dpct
} // namespace clang

#endif // DPCT_CUSTOM_HELPER_FILES_H
