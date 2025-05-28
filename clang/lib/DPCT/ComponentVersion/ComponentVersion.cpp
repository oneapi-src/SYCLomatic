
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
  static void mapping(IO &io, std::shared_ptr<clang::dpct::CompStatus> &Status) {
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

const std::string oneAPIVersion = "2025.1";
class ComponentInfo {
public:
  ComponentInfo(const std::string &Name,
                const std::string &Version = "oneAPI " + oneAPIVersion)
      : ComponentName(Name), ComponentVersion(Version) {}

  std::string ComponentName;
  std::string ComponentVersion;
};

void DisplayComponentInfo(const ComponentInfo &Info) {
  std::cout << "  - " << Info.ComponentName << ": " << Info.ComponentVersion << ".\n";
}
void SupportedComponents() {
  std::vector<ComponentInfo> Components = {
      {"DPC++/C++"},
      {"oneDPL"},
      {"oneMKL"},
      {"oneDNNL"},
      {"oneCCL"},
      {"ISHMEM"}};

  std::cout << "Supported components:\n";
  for (const auto &Component : Components) {
    DisplayComponentInfo(Component);
  }
}

 void emitCompStatusWarning(std::shared_ptr<clang::dpct::CompStatus> Status,
                             std::stringstream &ss) {
  ss << "The feature " << Status->Feature << " is supported in the "
     << Status->SupportedVersion << " daily build. ";
  if (Status->IsOpenSource) {
    ss << "This feature is open-source. ";
  }
  if (!Status->Link.empty()) {
    ss << "For more details, please refer to the provided link: " << Status->Link
       << ". ";
  }
  if (Status->IsInNextOneAPIVersion) {
    ss << "This feature will be included in the next oneAPI version. ";
  }
  if (!Status->Description.empty()) {
    ss << "Description: " << Status->Description << ".";
  }
  ss << "\n";
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
  SupportedComponents();
  if (NewCompsStatus.empty()) {
    return;
  }
  for (auto &CompStatus : NewCompsStatus) {
    // if ()
    switch (CompStatus->CompType) {
    case ComponentType::DPCPP:
      ss << "DPC++/C++: ";
      break;
    case ComponentType::oneDPL:
      ss << "oneDPL: ";
      break;
    case ComponentType::oneMKL:
      ss << "oneMKL: ";
      break;
    case ComponentType::oneCCL:   
      ss << "oneCCL: ";
      break;
    case ComponentType::oneDNNL:
      ss << "oneDNNL: ";
      break;
    case ComponentType::ISHMEM:
      ss << "ISHMEM: ";
      break;
  }
}

} // namespace dpct
} // namespace clang