//===--------------- Error.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Error.h"

#include "AnalysisInfo.h"
#include "Diagnostics.h"
#include "Statics.h"

namespace clang {
namespace dpct {

void ShowStatus(int Status, std::string Message) {

  std::string StatusString;
  switch (Status) {
  case MigrationSucceeded:
    StatusString = "Migration process completed";
    break;
  case MigrationNoCodeChangeHappen:
    StatusString = "Migration not necessary; no CUDA code detected";
    break;
  case MigrationSkipped:
    StatusString = "Some migration rules were skipped";
    break;
  case MigrationError:
    StatusString = "An error has occurred during migration";
    break;
  case MigrationSaveOutFail:
    StatusString =
        "Error: Unable to save the output to the specified directory";
    break;
  case MigrationErrorInvalidCudaIncludePath:
    StatusString = "Error: Path for CUDA header files specified by "
                   "--cuda-include-path is invalid.";
    break;
  case MigrationErrorDetectedCudaVersionUnsupported:
    StatusString =
        "Error: The auto detect CUDA header files version is " + Message +
        " and not supported yet. "
        "The latest supported version is " +
        dpct::DpctGlobalInfo::getCudaVersion() +
        ". You can specify CUDA header files by option --cuda-include-path or "
        "use --force-migration to do the migration.";
    break;
  case MigrationErrorCudaVersionUnsupported:
    StatusString = "Error: The version of CUDA header files specified by "
                   "--cuda-include-path is" +
                   Message +
                   " and not supported yet. The latest supported "
                   "version is " +
                   dpct::DpctGlobalInfo::getCudaVersion() +
                   ". You can use --force-migration to "
                   "do the migration.";
    break;
  case MigrationErrorCannotDetectCudaPath:
    StatusString = "Error: Could not detect path to CUDA header files. Use "
                   "--cuda-include-path "
                   "to specify the correct path to the header files.";
    break;
  case MigrationErrorInvalidInRootOrOutRoot:
    StatusString = "Error: The path for --in-root or --out-root is not valid";
    break;
  case MigrationErrorInvalidInRootPath:
    StatusString = "Error: The path for --in-root is not valid";
    break;
  case MigrationErrorInvalidReportArgs:
    StatusString =
        "Error: The value(s) provided for report option(s) is incorrect.";
    break;
  case MigrationErrorInvalidWarningID:
    StatusString = "Error: Invalid warning ID or range; "
                   "valid warning IDs range from " +
                   std::to_string(DiagnosticsMessage::MinID) + " to " +
                   std::to_string(DiagnosticsMessage::MaxID);
    break;
  case MigrationOptionParsingError:
    StatusString = "Option parsing error,"
                   " run 'dpct --help' to see supported options and values";
    break;
  case MigrationErrorPathTooLong:
#if defined(_WIN32)
    StatusString = "Error: Path is too long; should be less than _MAX_PATH (" +
                   std::to_string(_MAX_PATH) + ")";
#else
    StatusString = "Error: Path is too long; should be less than PATH_MAX (" +
                   std::to_string(PATH_MAX) + ")";
#endif
    break;
  case MigrationErrorFileParseError:
    StatusString = "Error: Cannot parse input file(s)";
    break;
  case MigrationErrorCannotFindDatabase:
    StatusString = "Error: Cannot find compilation database";
    break;
  case MigrationErrorCannotParseDatabase:
    StatusString = "Error: Cannot parse compilation database";
    break;
  case MigrationErrorNoExplicitInRoot:
    StatusString =
        "Error: The option --process-all requires that the --in-root be "
        "specified explicitly. Use the --in-root option to specify the "
        "directory to be migrated.";
    break;
  case MigrationErrorSpecialCharacter:
    StatusString = "Error: Prefix contains special characters;"
                   " only alphabetical characters, digits and underscore "
                   "character are allowed";
    break;
  case MigrationErrorPrefixTooLong:
    StatusString =
        "Error: Prefix is too long; should be less than 128 characters";
    break;
  case MigrationErrorNoFileTypeAvail:
    StatusString = "Error: File Type not available for input file";
    break;
  case MigrationErrorInputDirContainCTTool:
    StatusString =
        "Error: Input folder specified by --in-root or --analysis-scope-path "
        "option is the parent of, or the same folder as, the installation "
        "directory of the dpct";
    break;
  case MigrationErrorRunFromSDKFolder:
    StatusString = "Error: Input folder specified by --in-root or "
                   "--analysis-scope-path option is in the CUDA_PATH folder";
    break;
  case MigrationErrorInputDirContainSDKFolder:
    StatusString = "Error: Input folder specified by --in-root or "
                   "--analysis-scope-path is the parent of, or the same folder "
                   "as, the CUDA_PATH folder";
    break;
  case MigrationErrorCannotAccessDirInDatabase:
    StatusString = "Error: Cannot access directory \"" + Message +
                   "\" from the compilation database, check if the directory "
                   "exists and can be accessed by the tool.";
    break;
  case MigrationErrorInconsistentFileInDatabase:
    StatusString = "Error: The file name(s) in the \"command\" and \"file\" "
                   "fields of the compilation database are inconsistent:\n" +
                   Message;
    break;
  case MigrationErrorInvalidExplicitNamespace:
    StatusString =
        "Error: The input for option --use-explicit-namespace is not valid. "
        "Run 'dpct --help' to see supported options and values.";
    break;
  case MigrationErrorDifferentOptSet:
    StatusString =
        "Error: Incremental migration requires the same option sets used "
        "across different dpct invocations. Specify --no-incremental-migration "
        "to disable incremental migration or use the same option set as in "
        "previous migration: \"" +
        Message +
        "\". See "
        "https://www.intel.com/content/www/us/en/docs/dpcpp-compatibility-tool/"
        "developer-guide-reference/current/overview.html for more details.";
    break;
  case MigrationErrorInvalidRuleFilePath:
    StatusString = "Error: The path for --rule-file is not valid";
    break;
  case MigrationErrorCannotParseRuleFile:
    StatusString = "Error: Cannot parse rule file";
    break;
  case MigrationErrorInvalidAnalysisScope:
    StatusString = "Error: The path for --analysis-scope-path is not the same "
                   "as or a parent directory of --in-root";
    break;
  case MigrationErrorNoAPIMapping:
    StatusString = "Error: The API mapping query for this API is not available "
                   "yet. You may get the API mapping by migrating sample code "
                   "from this CUDA API to the SYCL API with the tool.";
    break;
  case MigrationErrorAPIMappingWrongCUDAHeader:
    StatusString =
        "Error: Can not find '" + Message +
        "' in current CUDA header file: " + DpctGlobalInfo::getCudaPath() +
        ". Please check the API name or use a different CUDA header file with "
        "option \"--cuda-include-path\".";
    break;
  case MigrationErrorAPIMappingNoCUDAHeader:
    StatusString =
        "Error: Cannot find '" + Message +
        "' in current CUDA header file: " + DpctGlobalInfo::getCudaPath() +
        ". Please specify the header file for '" + Message +
        "' with option \"--extra-arg\".";
    break;
  case InterceptBuildError:
    StatusString = "Error: Call to intercept-build failed";
    break;
  default:
    DpctLog() << "Unknown error\n";
    dpctExit(-1);
  }

  if (Status != 0) {
    DpctLog() << "dpct exited with code: " << Status << " (" << StatusString
              << ")\n";
  }

  llvm::dbgs() << getDpctLogStr() << "\n";
  return;
}

std::string getLoadYamlFailWarning(std::string YamlPath) {
  return "Warning: Failed to load " + YamlPath +
         ". Migration continues with incremental migration disabled. See "
         "https://www.intel.com/content/www/us/en/docs/"
         "dpcpp-compatibility-tool/developer-guide-reference/current/"
         "overview.html for more "
         "details.\n";
}
std::string getCheckVersionFailWarning() {
  return "Warning: Incremental migration requires the same version of dpct. "
         "Migration continues with incremental migration disabled. See "
         "https://www.intel.com/content/www/us/en/docs/"
         "dpcpp-compatibility-tool/developer-guide-reference/current/"
         "overview.html for more "
         "details.\n";
}

} // namespace dpct
} // namespace clang
