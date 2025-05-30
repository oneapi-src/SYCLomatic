
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
void DisplayComponentInfo(const ComponentInfo &Info) {
  std::cout << "  - " << Info.ComponentName << ": " << Info.ComponentVersion
            << ".\n";
}
void SupportedComponents(const  std::unordered_map<ComponentType, ComponentInfo>  &Components) {

  std::cout << "Supported components:\n";
  std::stringstream ss;
  for (const auto &Component : Components) {
    DisplayComponentInfo(Component.second);
    for (const auto& Des : Component.second.ComponentDes) {
      ss << "  - " << Des << "\n";
    }
  }
  if (!ss.str().empty()) {
    std::cout << "\nDetails:\n";
    std::cout << ss.str();
  }

}


void printWarning(std::stringstream &ss,
                  const std::shared_ptr<clang::dpct::CompStatus> &Status,
                  std::string &ComponentType) {
  ss << ComponentType;
  if (Status->IsOpenSource) {
    ss << Status->SupportedVersion << "\n";
  } else {
  }
}

void collectNewVerInfo(ComponentInfo &Info,
                       const std::shared_ptr<clang::dpct::CompStatus> &Status) {
  std::string Description = "";
  Description += "The feature " + Status->Feature;
  if (Status->IsOpenSource) {
    Description += " is supported in " + Info.ComponentName + " " + Status->SupportedVersion + ". ";
    Info.ComponentVersion = Status->SupportedVersion + " (open source). ";
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
  // std::cout <<Description << "\n";
  Info.ComponentDes.push_back(Description);
}

void importStatus(std::vector<clang::tooling::UnifiedPath> &RuleFiles) {
  auto file = llvm::MemoryBuffer::getFile(RuleFiles[0].getCanonicalPath());
  if (!file) {
    llvm::errs() << "Error: failed to read " << RuleFiles[0].getCanonicalPath()
                 << ": " << file.getError().message() << "\n";
    return;
  }
  std::vector<std::shared_ptr<clang::dpct::CompStatus>> NewCompsStatus;
  yaml::Input yin(file.get()->getBuffer());
  yin >> NewCompsStatus;
  std::stringstream ss;
  DpctGlobalInfo::setSupportedCompsStatus(NewCompsStatus);
  if (NewCompsStatus.empty()) {
    return;
  }

  std::unordered_map<ComponentType, ComponentInfo> Components = {
      {ComponentType::DPCPP, ComponentInfo("DPCPP")},
      {ComponentType::oneDPL, ComponentInfo("oneDPL")},
      {ComponentType::oneMKL, ComponentInfo("oneMKL")},
      {ComponentType::oneDNNL, ComponentInfo("oneDNNL")},
      {ComponentType::oneCCL, ComponentInfo("oneCCL")},
      {ComponentType::ISHMEM, ComponentInfo("ISHMEM")}};
  DpctGlobalInfo::setSupportedComponentInfo(Components);

  for (auto &CompStatus : NewCompsStatus) {
    collectNewVerInfo(Components[CompStatus->CompType], CompStatus);
  }
  SupportedComponents(Components);

}

} // namespace dpct
} // namespace clang