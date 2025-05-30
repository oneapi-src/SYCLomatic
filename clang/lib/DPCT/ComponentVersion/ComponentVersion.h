//===--------------- ComponentVersion.h
//----------------------------------------------===//
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
enum ComponentType { DPCPP, oneDPL, oneMKL, oneCCL, oneDNNL, ISHMEM };

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

const std::string CuroneAPIVersion = "2025.1";
class ComponentInfo {
public:
  ComponentInfo(const std::string &Name,
                const std::string &Version = "oneAPI " + CuroneAPIVersion)
      : ComponentName(Name), ComponentVersion(Version) {}
      ComponentInfo() : ComponentName(""), ComponentVersion("oneAPI " + CuroneAPIVersion) {}
  std::string ComponentName;
  std::string ComponentVersion;
  std::vector<std::string> ComponentDes;
};

void importStatus(std::vector<clang::tooling::UnifiedPath> &RuleFiles);


void collectNewVerInfo(ComponentInfo &Info,
                       const std::shared_ptr<clang::dpct::CompStatus> &Status);
} // namespace dpct
} // namespace clang

#endif // DPCT_COMPONENT_VERSION