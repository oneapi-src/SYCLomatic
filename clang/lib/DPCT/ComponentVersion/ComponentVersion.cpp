
#include "ComponentVersion.h"
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
    io.mapRequired("TestComponent", Stats->TestComponent);
    io.mapRequired("SupportedVersion", Stats->SupportedVersion);
    io.mapOptional("IsOpenSource", Stats->IsOpenSource);
    io.mapRequired("IsInNextOneAPIVersion", Stats->IsInNextOneAPIVersion);
    io.mapOptional("Link", Stats->Link);
    io.mapOptional("Description", Stats->Description);
  }
};

template <class T>
struct llvm::yaml::SequenceTraits<std::vector<std::shared_ptr<T>>> {
  static size_t size(llvm::yaml::IO &Io, std::vector<std::shared_ptr<T>> &Seq) {
    return Seq.size();
  }
  static std::shared_ptr<T> &element(IO &, std::vector<std::shared_ptr<T>> &Seq,
                                     size_t Index) {
    if (Index >= Seq.size())
      Seq.resize(Index + 1);
    return Seq[Index];
  }
};
namespace clang {
namespace dpct {
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

  for (auto &feature : features) {
    ss << "The feature " << feature->Feature << " is supported in the "
       << feature->SupportedVersion << " daily build. ";
    if (feature->IsOpenSource) {
      ss << "This feature is open-source. ";
    }
    if (!feature->Link.empty()) {
      ss << "For more details, please refer to the provided link: "
         << feature->Link << ". ";
    }
    if (feature->IsInNextOneAPIVersion) {
      ss << "This feature will be included in the next oneAPI version. ";
    }
    if (!feature->Description.empty()) {

      ss << "Description: " << feature->Description << ".\n";
    }
  }
  std::cout << ss.str() << "\n";
}

} // namespace dpct
} // namespace clang