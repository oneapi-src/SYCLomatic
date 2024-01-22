//===--------------- ValidateArguments.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_VALIDATE_ARGUMENTS_H
#define DPCT_VALIDATE_ARGUMENTS_H

#include "clang/Tooling/Tooling.h"

#include <string>
#include <vector>

#if defined(_WIN32)
#define MAX_PATH_LEN _MAX_PATH
#define MAX_NAME_LEN _MAX_FNAME
#else
#define MAX_PATH_LEN PATH_MAX
#define MAX_NAME_LEN NAME_MAX
#endif

/// The enum that sets the Unified Shared Memory (USM) level to use in
/// source code generation.
/// none:       uses helper functions from DPCT header files for memory
///             management migration
/// restricted: uses USM API for memory management migration
enum class UsmLevel { UL_None, UL_Restricted };
/// OutputVerbosityLevel defines various verbosity levels for dpct reports
enum class OutputVerbosityLevel {
  OVL_Silent,
  OVL_Normal,
  OVL_Detailed,
  OVL_Diagnostics
};
enum class DPCTFormatStyle { FS_LLVM, FS_Google, FS_Custom };
enum class ReportFormatEnum { RFE_NotSetFormat, RFE_CSV, RFE_Formatted };
enum class HelperFilesCustomizationLevel {
  HFCL_None,
  HFCL_File,
  HFCL_All,
  HFCL_API
};
enum class ReportTypeEnum {
  RTE_NotSetType,
  RTE_APIs,
  RTE_Stats,
  RTE_All,
  RTE_Diags
};
enum class AssumedNDRangeDimEnum : unsigned int { ARE_Dim1 = 1, ARE_Dim3 = 3 };
enum class ExplicitNamespace : unsigned int {
  EN_None = 0,
  // 1 means EN_CL and it has already been removed
  EN_SYCL = 2,
  EN_SYCL_Math = 3,
  EN_DPCT = 4
};
enum class DPCPPExtensionsDefaultEnabled : unsigned int {
  ExtDE_EnqueueBarrier = 0,
  ExtDE_DeviceInfo,
  ExtDE_BFloat16,
  ExtDE_PeerAccess,
  ExtDE_DPCPPExtensionsDefaultEnabledEnumSize
};
enum class DPCPPExtensionsDefaultDisabled : unsigned int {
  ExtDD_CCXXStandardLibrary = 0,
  ExtDD_IntelDeviceMath,
  ExtDD_DPCPPExtensionsDefaultDisabledEnumSize
};
enum class ExperimentalFeatures : unsigned int {
  Exp_NdRangeBarrier = 0, // Using nd_range_barrier.
  Exp_FreeQueries,        // Using free queries functions, like this_nd_item,
                          // this_group, this_subgroup.
  Exp_GroupSharedMemory,
  Exp_LogicalGroup,
  Exp_UserDefineReductions,
  Exp_MaskedSubGroupFunction,
  Exp_DPLExperimentalAPI,
  Exp_OccupancyCalculation,
  Exp_Matrix,
  Exp_BFloat16Math,
  Exp_BindlessImages,
  Exp_ExperimentalFeaturesEnumSize
};
enum class HelperFuncPreference : unsigned int { NoQueueDevice = 0 };
enum class SYCLFileExtensionEnum { DP_CPP, SYCL_CPP, CPP };

bool makeInRootCanonicalOrSetDefaults(
    clang::tooling::UnifiedPath &InRoot, const std::vector<std::string> SourceFiles);
bool makeOutRootCanonicalOrSetDefaults(clang::tooling::UnifiedPath &OutRoot);
bool makeAnalysisScopeCanonicalOrSetDefaults(clang::tooling::UnifiedPath &AnalysisScope,
                                             const clang::tooling::UnifiedPath &InRoot);

/// Make sure files passed to tool are under the
/// input root directory and have an extension.
/// return value:
/// 0: success (InRoot and SourceFiles are valid)
/// -1: fail for InRoot not valid or there is file SourceFiles not in InRoot
/// -2: fail for there is file in SourceFiles without extension
int validatePaths(const clang::tooling::UnifiedPath &InRoot,
                  const std::vector<std::string> &SourceFiles);

/// Make sure cmake script path is valide file path or directory path.
/// return value:
/// 0: success (InRoot and CmakeScriptPaths are valid)
/// -1: fail for InRoot not valid
/// -2: fail for there is file or directory not existing in CmakeScriptPaths
/// -3: fail for there is directory not in InRoot directory in CmakeScriptPaths
/// -4: fail for there is file not in InRoot directory in CmakeScriptPaths
/// -5: fail for there is file that is is not a cmake script file in CmakeScriptPaths
int validateCmakeScriptPaths(const clang::tooling::UnifiedPath &InRoot,
                             const std::vector<std::string> &CmakeScriptPaths);

bool checkReportArgs(ReportTypeEnum &RType, ReportFormatEnum &RFormat,
                     std::string &RFile, bool &ROnly, bool &GenReport,
                     std::string &DVerbose);

/// Return value:
///  0: Path is valid
///  1: Path is empty, option SDK include path is not used
/// -1: Path is invalid
int checkSDKPathOrIncludePath(clang::tooling::UnifiedPath &Path);

#endif // DPCT_VALIDATE_ARGUMENTS_H
