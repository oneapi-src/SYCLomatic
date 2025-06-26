#ifndef RECOMMEND_LIBRARIES_H
#define RECOMMEND_LIBRARIES_H

#include "AnalysisInfo.h"
#include "FileGenerator/GenFiles.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <unordered_map>

namespace clang {
namespace dpct {

enum class ComponentType { DPCPP, oneDPL, oneMath, oneCCL, oneDNNL, ISHMEM };

struct DependencyStatus {
  std::string Feature;
  std::string SupportedVersion;
  std::string ReplacementText;
  ComponentType CompType;
  std::string Description;

  DependencyStatus() = default;
  DependencyStatus(std::unordered_map<ComponentType, DependencyStatus> &Table,
                   std::string Feature, std::string SupportedVersion,
                   ComponentType CT, std::string ReplacementText,
                   std::string Description)
      : Feature(Feature), SupportedVersion(SupportedVersion),
        ReplacementText(ReplacementText), CompType(CT),
        Description(Description) {
    Table[CT] = *this;
  }
};
extern std::unordered_map<clang::dpct::ComponentType,
                          clang::dpct::DependencyStatus>
    DepStatus;
extern std::vector<clang::dpct::DependencyStatus> DepStatusVec;

void CollectDepsResult(ReplTy &MainSrcFilesRepls);

std::string ComponentTypeToString(ComponentType Type);

inline void ShowDepsResult(llvm::raw_ostream &OStream) {
  if (DepStatusVec.empty())
    return;
  if (DpctGlobalInfo::isAnalysisModeEnabled())
    OStream << llvm::raw_ostream::Colors::BLUE;

  OStream << "Recommand Library Dependencies of SYCL Project:\n";

  if (DpctGlobalInfo::isAnalysisModeEnabled())
    OStream << llvm::raw_ostream::Colors::RESET;
  for (auto &Status : DepStatusVec) {
    OStream << "  - The " + Status.Feature + " is supported in " +
                   ComponentTypeToString(Status.CompType) + " and after " +
                   Status.SupportedVersion + ". " + Status.Description + "\n";
  }
}
} // namespace dpct
} // namespace clang
#endif // RECOMMEND_LIBRARIES_H
