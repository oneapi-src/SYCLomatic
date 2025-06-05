//===--------------- ComponentVersion.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ComponentVersion.h"
#include "AnalysisInfo.h"
#include <iostream>
#include <sstream>

using namespace llvm;

template <>
struct llvm::yaml::SequenceTraits<std::vector<clang::dpct::CompStatus>> {
  static size_t size(IO &io, std::vector<clang::dpct::CompStatus> &seq) {
    return seq.size();
  }

  static clang::dpct::CompStatus &
  element(IO &io, std::vector<clang::dpct::CompStatus> &seq, size_t index) {
    if (index >= seq.size())
      seq.resize(index + 1);
    return seq[index];
  }
};

template <>
struct llvm::yaml::MappingTraits<std::shared_ptr<clang::dpct::CompStatus>> {
  static void mapping(IO &io,
                      std::shared_ptr<clang::dpct::CompStatus> &Status) {
    Status = std::make_shared<clang::dpct::CompStatus>();
    io.mapRequired("Feature", Status->Feature);
    io.mapRequired("SupportedVersion", Status->SupportedVersion);
    io.mapRequired("ComponentType", Status->CompType);
    io.mapOptional("Link", Status->Link);
    io.mapRequired("ReplacementText", Status->ReplacementText);
    io.mapOptional("IsOpenSource", Status->IsOpenSource);
    io.mapRequired("IsInNextOneAPIVersion", Status->IsInNextOneAPIVersion);
    io.mapOptional("Description", Status->Description);
  }
};
template <> struct llvm::yaml::ScalarEnumerationTraits<ComponentType> {
  static void enumeration(llvm::yaml::IO &Io, ComponentType &Value) {
    Io.enumCase(Value, "Compiler", ComponentType::DPCPP);
    Io.enumCase(Value, "DPC++", ComponentType::DPCPP);
    Io.enumCase(Value, "DPCPP", ComponentType::DPCPP);
    Io.enumCase(Value, "oneDPL", ComponentType::oneDPL);
    Io.enumCase(Value, "oneMKL", ComponentType::oneMKL);
    Io.enumCase(Value, "oneCCL", ComponentType::oneCCL);
    Io.enumCase(Value, "oneDNNL", ComponentType::oneDNNL);
    Io.enumCase(Value, "ISHMEM", ComponentType::ISHMEM);
  }
};

namespace clang {
namespace dpct {

void displayOverallComponentInfo(
    const std::unordered_map<ComponentType, ComponentInfo>
        &SupportedComponentInfo) {
  std::stringstream ComponentOverallLog;
  ComponentOverallLog << "Compatible components:\n";
  for (const auto &Component : SupportedComponentInfo) {
    ComponentOverallLog << "  - " << Component.second.ComponentName << ": "
                        << Component.second.ComponentVersion << ".\n";
  }
  std::cout << ComponentOverallLog.str();
}

void displayComponentDetailsInfo(
    const std::unordered_map<ComponentType, ComponentInfo>
        &SupportedComponentInfo) {
  std::stringstream ComponentDetailLog;
  ComponentDetailLog << "\nCompatible Component details:\n";
  for (const auto &Component : SupportedComponentInfo) {
    for (const auto &Des : Component.second.ComponentDes) {
      ComponentDetailLog << "  - " << Des << "\n";
    }
  }
  std::cout << ComponentDetailLog.str();
}

void displaySupportedComponents(bool isPrintOverall) {
  auto CompsStatus = DpctGlobalInfo::getCompatibleCompsStatus();
  std::unordered_map<ComponentType, ComponentInfo> Components =
      dpct::DpctGlobalInfo::getSupportedComponentInfo();
  if (isPrintOverall) {
    for (auto &CompStatus : CompsStatus) {
      updateComInfoByCompStatus(Components[CompStatus->CompType],
                                     CompStatus);
    }
    displayOverallComponentInfo(Components);
  }
  displayComponentDetailsInfo(Components);
}
std::string
generateDescription(const ComponentInfo &Info,
                    const std::shared_ptr<clang::dpct::CompStatus> &Status) {
  std::ostringstream oss;
  oss << "The feature " << Status->Feature;
  if (Status->IsOpenSource) {
    oss << " is supported in " << Info.ComponentName << " open source "
        << Status->SupportedVersion << ". ";
  } else if (Status->IsInNextOneAPIVersion) {
    oss << " is supported in the next oneAPI version. ";
  }
  if (!Status->Link.empty()) {
    oss << "For more details, please refer to the provided link: "
        << Status->Link << ". \n";
  }
  return oss.str();
}

std::string getContentAfterSpace(const std::string &str) {
  size_t pos = str.find(' ');
  if (pos != std::string::npos) {
    return str.substr(pos + 1);
  }
  return str;
}
void updateComInfoByCompStatus(
    ComponentInfo &Info,
    const std::shared_ptr<clang::dpct::CompStatus> &Status) {
  std::string Description = generateDescription(Info, Status);
  Info.ComponentDes.push_back(Description);
  if (Status->IsOpenSource && !Info.ComponentVersion.empty() &&
      !Status->SupportedVersion.empty()) {
    try {
      float OrgDate = std::stof(getContentAfterSpace(Info.ComponentVersion));
      float NewDate = std::stof(getContentAfterSpace(Status->SupportedVersion));
      if (OrgDate < NewDate ||
          Info.ComponentVersion.find("oneAPI") != std::string::npos) {
        Info.ComponentVersion = ">= " + Status->SupportedVersion;
      }
    } catch (...) {
      std::cerr << "DPCT Exit: Error in component_version.yaml. The "
                   "'SupportedVersion' field must be a valid number. Please "
                   "check and correct it.\n";
      dpctExit(MigrationError);
    }
  }
}

void parseSupportComponentStatus(
    std::vector<clang::tooling::UnifiedPath> &RuleFiles,
    bool IsPrintComponentOpt) {
  auto file = llvm::MemoryBuffer::getFile(RuleFiles[0].getCanonicalPath());
  if (!file) {
    llvm::errs() << "Error: failed to read " << RuleFiles[0].getCanonicalPath()
                 << ": " << file.getError().message() << "\n";
    return;
  }
  std::vector<std::shared_ptr<clang::dpct::CompStatus>> CompsStatus;
  yaml::Input yin(file.get()->getBuffer());
  yin >> CompsStatus;
  DpctGlobalInfo::setCompatibleCompsStatus(CompsStatus);
}

} // namespace dpct
} // namespace clang
