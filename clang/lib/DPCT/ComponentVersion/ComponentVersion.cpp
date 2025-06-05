
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
    io.mapRequired("ReplacementText", Status->ReplacementText);
    io.mapRequired("ComponentType", Status->CompType);
    io.mapRequired("SupportedVersion", Status->SupportedVersion);
    io.mapOptional("IsOpenSource", Status->IsOpenSource);
    io.mapRequired("IsInNextOneAPIVersion", Status->IsInNextOneAPIVersion);
    io.mapOptional("Link", Status->Link);
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

void DisplayOverallComponentInfo(
    const std::unordered_map<ComponentType, ComponentInfo>
        &SupportedComponentInfo) {
  std::stringstream ComponentOverallLog;
  ComponentOverallLog << "Supported components:\n";
  for (const auto &Component : SupportedComponentInfo) {
    ComponentOverallLog << "  - " << Component.second.ComponentName << ": "
                        << Component.second.ComponentVersion << ".\n";
  }
  std::cout << ComponentOverallLog.str();
}

void DisplayComponentDetailsInfo(
    const std::unordered_map<ComponentType, ComponentInfo>
        &SupportedComponentInfo) {
  std::stringstream ComponentDetailLog;
  ComponentDetailLog << "\nSupport Component details:\n";
  std::cout << "Size " << SupportedComponentInfo.size() << "\n";
  for (const auto &Component : SupportedComponentInfo) {
    for (const auto &Des : Component.second.ComponentDes) {
      ComponentDetailLog << "  - " << Des << "\n";
    }
  }
  std::cout << ComponentDetailLog.str() << "\n";
}

void showSupportedComponents(bool isPrintOverall) {
  auto CompsStatus = DpctGlobalInfo::getSupportedCompsStatus();
  std::unordered_map<ComponentType, ComponentInfo> Components =
      dpct::DpctGlobalInfo::getSupportedComponentInfo();
  if (isPrintOverall) {
    for (auto &CompStatus : CompsStatus) {
      CollectNewVerInfo(Components[CompStatus->CompType], CompStatus);
    }
    DisplayOverallComponentInfo(Components);
  }
  DisplayComponentDetailsInfo(Components);
}

void CollectNewVerInfo(ComponentInfo &Info,
                       const std::shared_ptr<clang::dpct::CompStatus> &Status) {
  std::cout << "HHHHHH \n";
  std::string Description = "";
  Description += "The feature " + Status->Feature;
  if (Status->IsOpenSource) {
    Description += " is supported in " + Info.ComponentName + " open source " +
                   Status->SupportedVersion + ". ";
    if (!Info.ComponentVersion.empty()) {
      if (Info.ComponentVersion.find("oneAPI") == std::string::npos) {
        long orgDate = stoi(Info.ComponentVersion);
        long newDate = stoi(Status->SupportedVersion);
        if (orgDate < newDate)
          Info.ComponentVersion = Status->SupportedVersion;
      } else {
        Info.ComponentVersion = Status->SupportedVersion;
      }
    }
  } else {
    if (Status->IsInNextOneAPIVersion) {
      Description += " is supported in the next oneAPI version. ";
    }
  }
  if (!Status->Link.empty()) {
    Description +=
        "For more details, please refer to the provided link: " + Status->Link +
        ". \n";
  }
  std::cout << "Info name " << Info.ComponentName << "\n";
  Info.ComponentDes.push_back(Description);
}

void ParseSupportComponentStatus(
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
  DpctGlobalInfo::setSupportedCompsStatus(CompsStatus);
}

} // namespace dpct
} // namespace clang