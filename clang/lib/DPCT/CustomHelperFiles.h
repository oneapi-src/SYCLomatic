//===--------------- CustomHelperFiles.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_CUSTOM_HELPER_FILES_H
#define DPCT_CUSTOM_HELPER_FILES_H

#include "clang/AST/Decl.h"
#include "clang/Tooling/Core/Replacement.h"

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace clang {
namespace dpct {

enum class HelperFileEnum : unsigned int {
#define HELPERFILE(PATH, UNIQUE_ENUM) UNIQUE_ENUM,
#define HELPER_FEATURE_MAP_TO_APINAME(File, FeatureName, APIName)
#include "../../runtime/dpct-rt/include/HelperFileAndFeatureNames.inc"
#undef HELPER_FEATURE_MAP_TO_APINAME
#undef HELPERFILE
  Unknown,
  HelperFileEnumTypeSize,
};

enum class HelperFeatureDependencyKind : unsigned int {
  HFDK_Both = 0,
  HFDK_UsmNone,
  HFDK_UsmRestricted
};

enum class HelperFeatureEnum : unsigned int {
  no_feature_helper,
#define DPCT_FEATURE_ENUM
#undef DPCT_FEATURE_ENUM_FEATURE_PAIR_MAP
#include "clang/DPCT/HelperFeatureEnum.inc"
#undef DPCT_FEATURE_ENUM
};

using HelperFeatureIDTy = std::pair<HelperFileEnum, std::string>;

struct HelperFunc {
  std::string Namespace = ""; // the namespace of this helper function feature
  int PositionIdx = -1;       // the position of this helper function feature
  bool IsCalled = false;      // has this feature be called
  std::set<std::string> CallerSrcFiles = {}; // files have called this feature
  std::vector<std::pair<HelperFeatureIDTy, HelperFeatureDependencyKind>>
      Dependency = {};         // some features which this feature depends on
  std::string Code = "";       // the code of this feature
  std::string USMCode = "";    // the code of this feature
  std::string NonUSMCode = ""; // the code of this feature
  HelperFeatureIDTy ParentFeature = {
      HelperFileEnum::Unknown,
      ""}; // If this feature is a sub-feature, this field saved its
           // parent feature. If this feature is not a sub-feature,
           // this field saved {Unknown, ""}.
};

void requestFeature(HelperFeatureEnum Feature, const std::string &UsedFile);
void requestFeature(HelperFeatureEnum Feature, clang::SourceLocation SL);
void requestFeature(HelperFeatureEnum Feature, const clang::Stmt *Stmt);
void requestFeature(HelperFeatureEnum Feature, const clang::Decl *Decl);

std::string getCopyrightHeader(const clang::dpct::HelperFileEnum File);
std::pair<std::string, std::string>
getHeaderGuardPair(const clang::dpct::HelperFileEnum File);
std::string
getHelperFileContent(const clang::dpct::HelperFileEnum File,
                     std::vector<clang::dpct::HelperFunc> ContentVec);
std::string getDpctVersionStr();
void emitDpctVersionWarningIfNeed(const std::string &VersionFromYaml);
void generateHelperFunctions();

void requestHelperFeatureForEnumNames(const std::string Name,
                                      const std::string File);
void requestHelperFeatureForEnumNames(const std::string Name,
                                      clang::SourceLocation File);
void requestHelperFeatureForEnumNames(const std::string Name,
                                      const clang::Stmt *File);
void requestHelperFeatureForEnumNames(const std::string Name,
                                      const clang::Decl *File);

void requestHelperFeatureForTypeNames(const std::string Name,
                                      const std::string File);
void requestHelperFeatureForTypeNames(const std::string Name,
                                      clang::SourceLocation File);
void requestHelperFeatureForTypeNames(const std::string Name,
                                      const clang::Stmt *File);
void requestHelperFeatureForTypeNames(const std::string Name,
                                      const clang::Decl *File);
std::string getCustomMainHelperFileName();

void updateHelperNameContentMap(
    const clang::tooling::TranslationUnitReplacements &TUR);
void updateTUR(clang::tooling::TranslationUnitReplacements &TUR);

void replaceEndOfLine(std::string &StrNeedProcess);

extern std::map<std::pair<clang::dpct::HelperFileEnum, std::string>,
                clang::dpct::HelperFunc>
    HelperNameContentMap;
extern std::unordered_map<clang::dpct::HelperFileEnum, std::string>
    HelperFileNameMap;
extern std::unordered_map<std::string, clang::dpct::HelperFileEnum>
    HelperFileIDMap;
extern const std::unordered_map<clang::dpct::HelperFileEnum, std::string>
    HelperFileHeaderGuardMacroMap;

extern const std::string DpctAllContentStr;
extern const std::string AtomicAllContentStr;
extern const std::string BlasUtilsAllContentStr;
extern const std::string DnnlUtilsAllContentStr;
extern const std::string DeviceAllContentStr;
extern const std::string DplUtilsAllContentStr;
extern const std::string ImageAllContentStr;
extern const std::string KernelAllContentStr;
extern const std::string MemoryAllContentStr;
extern const std::string UtilAllContentStr;
extern const std::string RngUtilsAllContentStr;
extern const std::string LibCommonUtilsAllContentStr;
extern const std::string CclUtilsAllContentStr;
extern const std::string SparseUtilsAllContentStr;
extern const std::string FftUtilsAllContentStr;
extern const std::string LapackUtilsAllContentStr;
extern const std::string DplExtrasAlgorithmAllContentStr;
extern const std::string DplExtrasFunctionalAllContentStr;
extern const std::string DplExtrasIteratorsAllContentStr;
extern const std::string DplExtrasMemoryAllContentStr;
extern const std::string DplExtrasNumericAllContentStr;
extern const std::string DplExtrasVectorAllContentStr;
extern const std::string DplExtrasDpcppExtensionsAllContentStr;

extern const std::map<clang::dpct::HelperFeatureIDTy, std::string>
    FeatureNameToAPINameMap;
extern const std::unordered_map<clang::dpct::HelperFeatureEnum,
                                clang::dpct::HelperFeatureIDTy>
    HelperFeatureEnumPairMap;

extern const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
    PropToGetFeatureMap;
extern const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
    PropToSetFeatureMap;
extern const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
    SamplingInfoToSetFeatureMap;
extern const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
    SamplingInfoToGetFeatureMap;
extern const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
    ImageWrapperBaseToSetFeatureMap;
extern const std::unordered_map<std::string, clang::dpct::HelperFeatureEnum>
    ImageWrapperBaseToGetFeatureMap;

} // namespace dpct
} // namespace clang

#endif // DPCT_CUSTOM_HELPER_FILES_H
