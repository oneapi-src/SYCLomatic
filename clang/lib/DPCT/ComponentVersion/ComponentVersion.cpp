
#include "ComponentVersion.h"
#include "AnalysisInfo.h"
#include <iostream>
#include <sstream>

using namespace llvm;

template <>
struct llvm::yaml::SequenceTraits<std::vector<clang::dpct::CmpStats>> {
  static size_t size(IO &io, std::vector<clang::dpct::CmpStats> &seq) {
    return seq.size();
  }

  static clang::dpct::CmpStats &
  element(IO &io, std::vector<clang::dpct::CmpStats> &seq, size_t index) {
    if (index >= seq.size())
      seq.resize(index + 1);
    return seq[index];
  }
};

template <>
struct llvm::yaml::MappingTraits<std::shared_ptr<clang::dpct::CmpStats>> {
  static void mapping(IO &io, std::shared_ptr<clang::dpct::CmpStats> &Stats) {
    Stats = std::make_shared<clang::dpct::CmpStats>();
    io.mapRequired("Feature", Stats->Feature);
    io.mapRequired("ReplacementText", Stats->ReplacementText);
    io.mapRequired("TestComponent", Stats->TestComponent);
    io.mapRequired("SupportedVersion", Stats->SupportedVersion);
    io.mapOptional("IsOpenSource", Stats->IsOpenSource);
    io.mapRequired("IsInNextOneAPIVersion", Stats->IsInNextOneAPIVersion);
    io.mapOptional("Link", Stats->Link);
    io.mapOptional("Description", Stats->Description);
  }
};


namespace clang {
namespace dpct {

void emitCmpStatsWarning(std::shared_ptr<clang::dpct::CmpStats> Stats,
                         std::stringstream &ss) {
  ss << "The feature " << Stats->Feature << " is supported in the "
     << Stats->SupportedVersion << " daily build. ";
  if (Stats->IsOpenSource) {
    ss << "This feature is open-source. ";
  }
  if (!Stats->Link.empty()) {
    ss << "For more details, please refer to the provided link: " << Stats->Link
       << ". ";
  }
  if (Stats->IsInNextOneAPIVersion) {
    ss << "This feature will be included in the next oneAPI version. ";
  }
  if (!Stats->Description.empty()) {
    ss << "Description: " << Stats->Description << ".";
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

  std::vector<std::shared_ptr<clang::dpct::CmpStats>> features;
  yaml::Input yin(file.get()->getBuffer());
  yin >> features;
  std::stringstream ss;
  DpctGlobalInfo::setVerifiedCmpStats(features);
  for (auto &feature : features) {
    emitCmpStatsWarning(feature, ss);
  }
}

} // namespace dpct
} // namespace clang