//===--- IncrementalMigrationUtility.cpp ---------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#include "Config.h"
#include "Error.h"
#include "ExternalReplacement.h"
#include "IncrementalMigrationUtility.h"

namespace clang {
namespace c2s {

bool isOnlyContainDigit(const std::string &Str) {
  for (const auto &C : Str) {
    if (!std::isdigit(C))
      return false;
  }
  return true;
}

/// The \p VersionStr style must be major.minor.patch
bool convertToIntVersion(std::string VersionStr, unsigned int &Result) {
  // get Major version
  size_t FirstDotLoc = VersionStr.find('.');
  if (FirstDotLoc == std::string::npos)
    return false;
  std::string MajorStr = VersionStr.substr(0, FirstDotLoc);
  if (MajorStr.empty() || !isOnlyContainDigit(MajorStr))
    return false;
  int Major = std::stoi(MajorStr);

  // get Minor version
  ++FirstDotLoc;
  size_t SecondDotLoc = VersionStr.find('.', FirstDotLoc);
  if (SecondDotLoc == std::string::npos || FirstDotLoc > VersionStr.size())
    return false;
  std::string MinorStr =
      VersionStr.substr(FirstDotLoc, SecondDotLoc - FirstDotLoc);
  if (MinorStr.empty() || !isOnlyContainDigit(MinorStr))
    return false;
  int Minor = std::stoi(MinorStr);

  // get Patch version
  ++SecondDotLoc;
  if (SecondDotLoc > VersionStr.size())
    return false;
  std::string PatchStr = VersionStr.substr(SecondDotLoc);
  int Patch = 0;
  if (!PatchStr.empty() && isOnlyContainDigit(PatchStr))
    Patch = std::stoi(PatchStr);

  Result = Major * 100 + Minor * 10 + Patch;
  return true;
}

/// The \p VersionInYaml style must be major.minor.patch
/// Return VCR_CMP_FAILED if meets error
/// Return VCR_VERSION_SAME if \p VersionInYaml is same as current version
/// Return VCR_CURRENT_IS_OLDER if \p VersionInYaml is later than current
/// version Return VCR_CURRENT_IS_NEWER if \p VersionInYaml is earlier than
/// current version
VersionCmpResult compareToolVersion(std::string VersionInYaml) {
  unsigned int PreviousVersion;
  if (convertToIntVersion(VersionInYaml, PreviousVersion)) {
    unsigned int CurrentVersion = std::stoi(C2S_VERSION_MAJOR) * 100 +
                                  std::stoi(C2S_VERSION_MINOR) * 10 +
                                  std::stoi(C2S_VERSION_PATCH);
    if (PreviousVersion > CurrentVersion)
      return VersionCmpResult::VCR_CURRENT_IS_OLDER;
    if (PreviousVersion < CurrentVersion)
      return VersionCmpResult::VCR_CURRENT_IS_NEWER;
    else
      return VersionCmpResult::VCR_VERSION_SAME;
  } else {
    return VersionCmpResult::VCR_CMP_FAILED;
  }
}

/// This function only check whether all option in \p CurrentOpts shows up
/// in \p PreviousOpts, as \p CurrentOpts already contains all the options
/// need to be checked.
///
/// return 0, pass
/// return -1, some option's value is different
/// return -2, cannot found some option from yaml, pasring error
int checkC2SOptionSet(
    const std::map<std::string, clang::tooling::OptionInfo> &CurrentOpts,
    const std::map<std::string, clang::tooling::OptionInfo> &PreviousOpts) {
  for (const auto &CurrentOpt : CurrentOpts) {
    if (PreviousOpts.count(CurrentOpt.first)) {
      if (CurrentOpt.first == OPTION_RuleFile) {
        if (PreviousOpts.at(OPTION_RuleFile).ValueVec.size() !=
            CurrentOpt.second.ValueVec.size())
          return -1;
        for (size_t Idx = 0; Idx < CurrentOpt.second.ValueVec.size(); Idx++) {
          if (PreviousOpts.at(OPTION_RuleFile).ValueVec[Idx] !=
              CurrentOpt.second.ValueVec[Idx])
            return -1;
        }
      }
#ifdef _WIN32
      else if (CurrentOpt.first == OPTION_VcxprojFile) {
        if (!PreviousOpts.count(OPTION_CompilationsDir)) {
          return -2;
        }
        if ((PreviousOpts.at(OPTION_VcxprojFile).Specified &&
             !CurrentOpts.at(OPTION_VcxprojFile).Specified) ||
            (!PreviousOpts.at(OPTION_VcxprojFile).Specified &&
             CurrentOpts.at(OPTION_VcxprojFile).Specified)) {
          if (PreviousOpts.at(OPTION_CompilationsDir).Value !=
              CurrentOpts.at(OPTION_CompilationsDir).Value) {
            return -1;
          }
        } else {
          if (PreviousOpts.at(CurrentOpt.first).Value !=
              CurrentOpt.second.Value) {
            return -1;
          }
        }
      } else {
        if (PreviousOpts.at(CurrentOpt.first).Value !=
            CurrentOpt.second.Value) {
          return -1;
        }
      }
#else
      else if (PreviousOpts.at(CurrentOpt.first).Value != CurrentOpt.second.Value) {
        return -1;
      }
#endif
    } else {
      if (CurrentOpt.first == OPTION_NoUseGenericSpace) {
        if (CurrentOpt.second.Value == "true")
          return -1;
      } else {
        return -2;
      }
    }
  }
  return 0;
}

/// return true, print successfully
/// return false, print failed due to parsing error, \p Msg is invalid
bool printOptions(
    const std::map<std::string, clang::tooling::OptionInfo> &OptsMap,
                  std::string &Msg) {
  std::vector<std::string> Opts;
  for (const auto &Item : OptsMap) {
    const std::string Key = Item.first;
    const std::string Value = Item.second.Value;
    const std::vector<std::string> ValueVec = Item.second.ValueVec;
    const bool Specified = Item.second.Specified;

    if (Key == clang::c2s::OPTION_AsyncHandler) {
      if ("true" == Value)
        Opts.emplace_back("--always-use-async-handler");
    }
    if (Key == clang::c2s::OPTION_NDRangeDim && Specified) {
      if (std::to_string(static_cast<unsigned int>(
              AssumedNDRangeDimEnum::ARE_Dim1)) == Value)
        Opts.emplace_back("--assume-nd-range-dim=1");
      else if (std::to_string(static_cast<unsigned int>(
                   AssumedNDRangeDimEnum::ARE_Dim3)) == Value)
        Opts.emplace_back("--assume-nd-range-dim=3");
    }
    if (Key == clang::c2s::OPTION_CommentsEnabled) {
      if ("true" == Value)
        Opts.emplace_back("--comments");
    }
    if (Key == clang::c2s::OPTION_CustomHelperFileName && Specified) {
      Opts.emplace_back("--custom-helper-name=" + Value);
    }
    if (Key == clang::c2s::OPTION_CtadEnabled) {
      if ("true" == Value)
        Opts.emplace_back("--enable-ctad");
    }
    if (Key == clang::c2s::OPTION_ExplicitClNamespace) {
      if ("true" == Value)
        Opts.emplace_back("--no-cl-namespace-inline");
    }
    if (Key == clang::c2s::OPTION_ExtensionFlag && Specified) {
      std::string MaxValueStr = std::to_string(static_cast<unsigned>(-1));
      if (Value.empty() || Value.length() > MaxValueStr.length() ||
          !isOnlyContainDigit(Value)) {
        return false;
      }
      if ((Value.length() == MaxValueStr.length()) && (Value > MaxValueStr)) {
        return false;
      }
      unsigned int UValue = std::stoul(Value);
      if (UValue < static_cast<unsigned>(-1)) {
        if (!(UValue &
              static_cast<unsigned>(DPCPPExtensions::Ext_EnqueueBarrier)))
          Opts.emplace_back("--no-dpcpp-extensions=enqueued_barriers");
      }
    }
    if (Key == clang::c2s::OPTION_NoDRYPattern) {
      if ("true" == Value)
        Opts.emplace_back("--no-dry-pattern");
    }
    if (Key == clang::c2s::OPTION_NoUseGenericSpace) {
      if ("true" == Value)
        Opts.emplace_back("--no-use-generic-space");
    }
    if (Key == clang::c2s::OPTION_CompilationsDir && Specified) {
      Opts.emplace_back("-p=\"" + Value + "\"");
    }
#ifdef _WIN32
    if (Key == clang::c2s::OPTION_VcxprojFile && Specified) {
      Opts.emplace_back("--vcxprojfile=\"" + Value + "\"");
    }
#endif
    if (Key == clang::c2s::OPTION_ProcessAll) {
      if ("true" == Value)
        Opts.emplace_back("--process-all");
    }
    if (Key == clang::c2s::OPTION_SyclNamedLambda) {
      if ("true" == Value)
        Opts.emplace_back("--sycl-named-lambda");
    }
    if (Key == clang::c2s::OPTION_ExperimentalFlag && Specified) {
      std::string MaxValueStr = std::to_string(static_cast<unsigned>(-1));
      if (Value.empty() || Value.length() > MaxValueStr.length() ||
          !isOnlyContainDigit(Value)) {
        return false;
      }
      if ((Value.length() == MaxValueStr.length()) && (Value > MaxValueStr)) {
        return false;
      }
      unsigned int UValue = std::stoul(Value);
      std::string Str = "";
      if (UValue &
          (1 << static_cast<unsigned>(ExperimentalFeatures::Exp_FreeQueries)))
        Str = Str + "free-function-queries,";
      if (UValue & (1 << static_cast<unsigned>(
                        ExperimentalFeatures::Exp_NdRangeBarrier)))
        Str = Str + "nd_range_barrier,";
      if (UValue & (1 << static_cast<unsigned>(
                        ExperimentalFeatures::Exp_GroupSharedMemory)))
        Str = Str + "local-memory-kernel-scope-allocation,";

      if (!Str.empty()) {
        Str = "--use-experimental-features=" + Str;
        Opts.emplace_back(Str.substr(0, Str.size() - 1));
      }
    }
    if (Key == clang::c2s::OPTION_ExplicitNamespace && Specified) {
      std::string MaxValueStr = std::to_string(static_cast<unsigned>(-1));
      if (Value.empty() || Value.length() > MaxValueStr.length() ||
          !isOnlyContainDigit(Value)) {
        return false;
      }
      if ((Value.length() == MaxValueStr.length()) && (Value > MaxValueStr)) {
        return false;
      }
      unsigned int UValue = std::stoul(Value);
      std::vector<std::string> Values;
      if (UValue & (1 << static_cast<unsigned>(ExplicitNamespace::EN_None))) {
        Values.emplace_back("none");
      }
      if (UValue & (1 << static_cast<unsigned>(ExplicitNamespace::EN_CL))) {
        Values.emplace_back("cl");
      }
      if (UValue & (1 << static_cast<unsigned>(ExplicitNamespace::EN_SYCL))) {
        Values.emplace_back("sycl");
      }
      if (UValue &
          (1 << static_cast<unsigned>(ExplicitNamespace::EN_SYCL_Math))) {
        Values.emplace_back("sycl-math");
      }
      if (UValue & (1 << static_cast<unsigned>(ExplicitNamespace::EN_C2S))) {
        Values.emplace_back("c2s");
      }
      if (UValue & (1 << static_cast<unsigned>(ExplicitNamespace::EN_DPCT))) {
        Values.emplace_back("dpct");
      }

      std::string Str;
      if (!Values.empty())
        Str += "--use-explicit-namespace=";
      for (auto &I : Values)
        Str = Str + I + ",";
      Opts.emplace_back(Str.substr(0, Str.size() - 1));
    }
    if (Key == clang::c2s::OPTION_UsmLevel && Specified) {
      if (std::to_string(static_cast<unsigned int>(UsmLevel::UL_Restricted)) ==
          Value)
        Opts.emplace_back("--usm-level=restricted");
      else if (std::to_string(static_cast<unsigned int>(UsmLevel::UL_None)) ==
               Value)
        Opts.emplace_back("--usm-level=none");
    }
    if (Key == clang::c2s::OPTION_OptimizeMigration) {
      if ("true" == Value)
        Opts.emplace_back("--optimize-migration");
    }
    if (Key == clang::c2s::OPTION_RuleFile && Specified) {
      for (const auto &Item : ValueVec)
        Opts.emplace_back("--rule-file=\"" + Item + "\"");
    }
  }

  Msg = "";
  for (auto &I : Opts)
    Msg = Msg + I + " ";
  Msg = Msg.substr(0, Msg.size() - 1);
  return true;
}


// return ture: c2s do migration continuely
// return false: c2s should exit
bool canContinueMigration(std::string &Msg) {
  auto PreTU = std::make_shared<clang::tooling::TranslationUnitReplacements>();
  // Try to load the MainSourceFiles.yaml file
  SmallString<128> YamlFilePath(C2SGlobalInfo::getOutRoot());
  llvm::sys::path::append(YamlFilePath, "MainSourceFiles.yaml");

  if (!llvm::sys::fs::exists(YamlFilePath))
    return true;
  if (loadFromYaml(std::move(YamlFilePath), *PreTU, true) != 0) {
    llvm::errs() << getLoadYamlFailWarning(YamlFilePath.str().str());
    return true;
  }

  // check version
  auto VerCompRes = compareToolVersion(PreTU->C2SVersion);
  if (VerCompRes == VersionCmpResult::VCR_CMP_FAILED) {
    llvm::errs() << getLoadYamlFailWarning(YamlFilePath.str().str());
    return true;
  } 
  if (VerCompRes == VersionCmpResult::VCR_CURRENT_IS_NEWER ||
             VerCompRes == VersionCmpResult::VCR_CURRENT_IS_OLDER) {
    llvm::errs() << getCheckVersionFailWarning();
    return true;
  } 

  // check option set
  int Res =
      checkC2SOptionSet(C2SGlobalInfo::getCurrentOptMap(), PreTU->OptionMap);
  if (Res == -2) {
    llvm::errs() << getLoadYamlFailWarning(YamlFilePath.str().str());
    return true;
  }

  if (Res == -1) {
    Msg = "";
    bool Ret = printOptions(PreTU->OptionMap, Msg);
    if (!Ret) {
      // parsing error, skip yaml
      llvm::errs() << getLoadYamlFailWarning(YamlFilePath.str().str());
      return true;
    }
    return false;
  }

  C2SGlobalInfo::setMainSourceYamlTUR(PreTU);
  return true;
}

}
}
