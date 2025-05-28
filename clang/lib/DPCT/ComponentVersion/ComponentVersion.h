//===--------------- ComponentVersion.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_COMPONENT_VERSION
#define DPCT_COMPONENT_VERSION
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/YAMLTraits.h"
#include <string>
#include <vector>

namespace clang {
namespace dpct {
enum  ComponentType {
  DPCPP,
  oneDPL,
  oneMKL,
  oneCCL,
  oneDNNL,
  ISHMEM
};

struct CompStatus {
  std::string Feature;
  std::string SupportedVersion;
  std::string ReplacementText;
  ComponentType CompType;
  bool IsOpenSource;
  bool IsInNextOneAPIVersion;
  std::string Link;
  std::string Description;
};

void importStatus(std::vector<clang::tooling::UnifiedPath> &RuleFiles);

void emitCompStatusWarning(std::shared_ptr<clang::dpct::CompStatus> Stats,
                         std::stringstream &StrStream);
} // namespace dpct
} // namespace clang

#endif // DPCT_COMPONENT_VERSION