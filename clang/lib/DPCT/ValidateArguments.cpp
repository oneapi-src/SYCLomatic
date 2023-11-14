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
static bool getDefaultOutRoot(clang::tooling::DpctPath &OutRootPar) {
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
  OutRootPar.setPath(OutRoot.str().str());
  return true;
}

// If input source files exist in the command line,
// set InRoot to the directory of the first input source file.
// If input source file does not exist,
// set InRoot to ".".
static bool getDefaultInRoot(clang::tooling::DpctPath &InRootPar,
                             const vector<string> &SourceFiles) {
  if (SourceFiles.size() == 0) {
    InRootPar.setPath(".");
    return true;
  }

  clang::tooling::DpctPath InRoot = SourceFiles.front();
  // Remove the last component from path.
  SmallString<512> InRootStr(InRoot.getCanonicalPath());
  path::remove_filename(InRootStr);
  InRootPar.setPath(InRootStr.str().str());
  if (InRootPar.getCanonicalPath().empty())
    return false;
  return true;
}

bool makeInRootCanonicalOrSetDefaults(
    clang::tooling::DpctPath &InRoot,
    const std::vector<std::string> SourceFiles) {
  if (InRoot.getPath().empty()) {
    if (!getDefaultInRoot(InRoot, SourceFiles))
      return false;
  } else if (InRoot.getCanonicalPath().empty()) {
    clang::dpct::ShowStatus(MigrationErrorInvalidInRootPath);
    dpctExit(MigrationErrorInvalidInRootPath);
  }
  if (fs::get_file_type(InRoot.getCanonicalPath()) !=
      fs::file_type::directory_file) {
    llvm::errs() << "Error: '" << InRoot.getCanonicalPath()
                 << "' is not a directory.\n";
    return false;
  }
  return true;
}

bool makeOutRootCanonicalOrSetDefaults(clang::tooling::DpctPath &OutRoot) {
  if (OutRoot.getPath().empty()) {
    if (!getDefaultOutRoot(OutRoot))
      return false;
  }
  return true;
}

bool makeAnalysisScopeCanonicalOrSetDefaults(
    clang::tooling::DpctPath &AnalysisScope,
    const clang::tooling::DpctPath &InRoot) {
  if (AnalysisScope.getPath().empty()) {
    // AnalysisScope defaults to the value of InRoot
    AnalysisScope = InRoot;
    return true;
  }
  if (AnalysisScope.getCanonicalPath().empty()) {
    return false;
  }
  return true;
}

// Make sure all files have an extension and are under InRoot.
int validatePaths(const clang::tooling::DpctPath &InRoot,
                  const std::vector<std::string> &SourceFiles) {
  int Ok = 0;
  for (const auto &FilePath : SourceFiles) {
    clang::tooling::DpctPath CanonicalPath(FilePath);
    if (CanonicalPath.getCanonicalPath().empty()) {
      Ok = -1;
      continue;
    }

    if (!isChildPath(InRoot, CanonicalPath)) {
      Ok = -1;
      llvm::errs() << "Error: File '" << CanonicalPath.getCanonicalPath()
                   << "' is not under the specified input root directory '"
                   << InRoot.getCanonicalPath() << "'\n";
    }

    if (!path::has_extension(CanonicalPath.getCanonicalPath())) {
      Ok = -2;
      llvm::errs() << "Error: File '" << CanonicalPath.getCanonicalPath()
                   << "' does not have an extension.\n";
    }
  }

  return Ok;
}

int checkSDKPathOrIncludePath(clang::tooling::DpctPath &Path) {
  if (Path.getPath().empty())
    return 1;
  if (Path.getCanonicalPath().empty())
    return -1;
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
