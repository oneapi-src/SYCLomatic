

#include <string>
#include <unordered_map>
#include "FileGenerator/GenFiles.h"
#include "Statics.h"
#include "Run.h"

namespace clang {
namespace dpct {

std::unordered_map<clang::dpct::ComponentType, clang::dpct::DependencyStatus>
  DepStatus;
std::vector<clang::dpct::DependencyStatus> DepStatusVec;

#define RECOMMENDLIBRARY(NAME, Feature, VERSION, COMPTYPE, REPLACEMENT,        \
                         MSG)                                    \
  DependencyStatus DepStatus_##NAME(DepStatus, Feature, VERSION, COMPTYPE,     \
                                    REPLACEMENT, MSG);
#include "RecommandLibrariesVersion.inc"

void CollectDepsResult(ReplTy &MainSrcFilesRepls) {
  for (auto Entry : DepStatus) {
    auto &Status = Entry.second;
    [&]() {
      for (auto Repl : MainSrcFilesRepls) {
        for (auto Item : Repl.second) {
          llvm::outs() << "XXXXXXXXXX " << Item.getReplacementText().str() <<"\n";
          llvm::outs() << "XXXXXXXXXX2222 " << Status.ReplacementText <<"\n";
          if (Item.getReplacementText().str().find(Status.ReplacementText) !=
              std::string::npos) {
            DepStatusVec.push_back(Status);
            return;
          }
        }
      }
    }();
  }
}
std::string ComponentTypeToString(ComponentType version) {
  switch (version) {
  case ComponentType::DPCPP:
    return "oneAPI DPC++ compiler Open Source Version";
  case ComponentType::oneDPL:
    return "oneAPI DPC++ Library Open Source Version";
  case ComponentType::oneMath:
    return "oneAPI Math Library Open Source Version";
  case ComponentType::oneCCL:
    return "oneAPI Collective Communications Library Open Source Version";
  case ComponentType::oneDNNL:
    return "oneAPI Deep Neural Network Library Open Source Version";
  case ComponentType::ISHMEM:
    return "oneAPI SHMEM Library Open Source Version";
  default:
    return "Unknown Component Type  ";
  }
}


} // namespace dpct
} // namespace clang