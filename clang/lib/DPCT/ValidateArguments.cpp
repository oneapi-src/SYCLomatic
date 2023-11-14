//===--------------- ValidateArguments.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ValidateArguments.h"
#include "Error.h"
#include "Statics.h"
#include "Utility.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <cassert>

using namespace llvm;
using namespace std;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

// Set OutRoot to the current working directory.
static bool getDefaultOutRoot(std::string &OutRootPar) {
  SmallString<256> OutRoot;
  if (fs::current_path(OutRoot) != std::error_code()) {
    llvm::errs() << "Could not get current path.\n";
    return false;
  }
  OutRoot.append("/dpct_output");
  if (fs::is_directory(OutRoot)) {
    std::error_code EC;
    fs::directory_iterator Iter(OutRoot, EC);
    if ((bool)EC) {
      llvm::errs() << "Could not access output directory.\n";
      return false;
    }
    fs::directory_iterator End;
    if (Iter != End) {
      llvm::errs() << "dpct_output directory is not empty. Please use option"
                      " \"--out-root\" to set output directory.\n";
      return false;
    } else {
      clang::dpct::PrintMsg(
          "The directory \"dpct_output\" is used as \"out-root\"\n");
    }
  } else {
    std::error_code EC = fs::create_directory(OutRoot, false);
    if ((bool)EC) {
      llvm::errs() << "Could not create dpct_output directory.\n";
      return false;
    }
    clang::dpct::PrintMsg(
        "The directory \"dpct_output\" is used as \"out-root\"\n");
  }
  OutRootPar.assign(begin(OutRoot), end(OutRoot));
  return true;
}

// If input source files exist in the command line,
// set InRoot to the directory of the first input source file.
// If input source file does not exist,
// set InRoot to ".".
static bool getDefaultInRoot(std::string &InRootPar,
                             const vector<string> &SourceFiles) {
  if (SourceFiles.size() == 0) {
    InRootPar = ".";
    return true;
  }

  SmallString<256> InRoot = StringRef(SourceFiles.front());
  // Remove the last component from path.
  path::remove_filename(InRoot);
  if (!makeCanonical(InRoot))
    return false;

  InRootPar.assign(begin(InRoot), end(InRoot));
  return true;
}

bool makeInRootCanonicalOrSetDefaults(
    string &InRoot, const std::vector<std::string> SourceFiles) {
  if (InRoot.empty()) {
    if (!getDefaultInRoot(InRoot, SourceFiles))
      return false;
  } else if (!makeCanonical(InRoot)) {
    return false;
  }
  if (fs::get_file_type(InRoot) != fs::file_type::directory_file) {
    llvm::errs() << "Error: '" << InRoot << "' is not a directory.\n";
    return false;
  }

  SmallString<512> InRootAbs;
  std::error_code EC = dpct::real_path(InRoot, InRootAbs, true);
  if ((bool)EC) {
    clang::dpct::ShowStatus(MigrationErrorInvalidInRootPath);
    dpctExit(MigrationErrorInvalidInRootPath);
  }
  InRoot = InRootAbs.str().str();
  return true;
}

bool makeOutRootCanonicalOrSetDefaults(string &OutRoot) {
  if (OutRoot.empty()) {
    if (!getDefaultOutRoot(OutRoot))
      return false;
  }

  if (!makeCanonical(OutRoot)) {
    return false;
  }

  llvm::SmallString<512> AbsOutRootNative(OutRoot);
  llvm::sys::path::native(AbsOutRootNative);
  OutRoot = std::string(AbsOutRootNative.str());

  return true;
}

bool makeAnalysisScopeCanonicalOrSetDefaults(string &AnalysisScope,
                                             const string &InRoot) {
  assert(isCanonical(InRoot) && "InRoot must be a canonical path.");
  if (AnalysisScope.empty()) {
    // AnalysisScope defaults to the value of InRoot
    AnalysisScope = InRoot;
    return true;
  }
  if (!makeCanonical(AnalysisScope)) {
    return false;
  }
  SmallString<512> AnalysisScopeAbs;
  std::error_code EC = dpct::real_path(AnalysisScope, AnalysisScopeAbs, true);
  if ((bool)EC) {
    return false;
  }
  AnalysisScope = AnalysisScopeAbs.str().str();
  return true;
}

// Make sure all files have an extension and are under InRoot.
int validatePaths(const std::string &InRoot,
                  const std::vector<std::string> &SourceFiles) {
  assert(isCanonical(InRoot) && "InRoot must be a canonical path.");
  int Ok = 0;
  for (const auto &FilePath : SourceFiles) {
    auto AbsPath = FilePath;
    if (!makeCanonical(AbsPath)) {
      Ok = -1;
      continue;
    }

    if (!isChildPath(InRoot, AbsPath)) {
      Ok = -1;
      llvm::errs() << "Error: File '" << AbsPath
                   << "' is not under the specified input root directory '"
                   << InRoot << "'\n";
    }

    if (!path::has_extension(AbsPath)) {
      Ok = -2;
      llvm::errs() << "Error: File '" << AbsPath
                   << "' does not have an extension.\n";
    }
  }

  return Ok;
}

int checkSDKPathOrIncludePath(const std::string &Path, std::string &RealPath) {
  if (Path.empty()) {
    return 1;
  }
  SmallString<512> AbsPath;
  auto EC = dpct::real_path(Path, AbsPath, true);
  if ((bool)EC) {
    return -1;
  }

#if defined(_WIN32)
  RealPath = AbsPath.str().lower();
  if (RealPath.size() >= 3 && RealPath.substr(0, 3) == "unc") {
    RealPath = "\\" + RealPath.substr(3);
  }
#elif defined(__linux__)
  RealPath = AbsPath.c_str();
#else
#error Only support windows and Linux.
#endif
  return 0;
}

bool checkReportArgs(ReportTypeEnum &RType, ReportFormatEnum &RFormat,
                     std::string &RFile, bool &ROnly, bool &GenReport,
                     std::string &DVerbose) {
  bool Success = true;
  if (ROnly || !RFile.empty() || !DVerbose.empty() ||
      RType != ReportTypeEnum::RTE_NotSetType ||
      RFormat != ReportFormatEnum::RFE_NotSetFormat) {
    GenReport = true;
    // check user provided value and give default value if required.
    if (RType == ReportTypeEnum::RTE_NotSetType) {
      RType = ReportTypeEnum::RTE_Stats;
    }
    // check the report format value
    if (RFormat == ReportFormatEnum::RFE_NotSetFormat) {
      RFormat = ReportFormatEnum::RFE_CSV;
    } else if (!(RFormat == ReportFormatEnum::RFE_CSV ||
                 RFormat == ReportFormatEnum::RFE_Formatted)) {
      llvm::errs() << "error value provided in option: --report-format, use "
                      "[csv|formatted].\n\n";
      Success = false;
    }
    // check the report file value.
    if (RFile.empty()) {
      RFile = "stdout";
    }
#ifdef DPCT_DEBUG_BUILD
    // check the report diags content value.
    if (DVerbose.empty()) {
      clang::dpct::VerboseLevel = clang::dpct::VL_VerboseLow;
    } else if (DVerbose == "pass") {
      clang::dpct::VerboseLevel = clang::dpct::VL_VerboseLow;
    } else if (DVerbose == "transformation") {
      clang::dpct::VerboseLevel = clang::dpct::VL_VerboseHigh;
    } else {
      Success = false;
      llvm::errs()
          << "error value provided in option: -report-diags-content, use "
             "[pass|transformation].\n\n";
    }
#endif
  }

  return Success;
}
