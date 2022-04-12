//===--- CustomHelperFiles.h -----------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef C2S_CUSTOM_HELPER_FILES_H
#define C2S_CUSTOM_HELPER_FILES_H

#include "clang/AST/Decl.h"
#include "clang/Tooling/Core/Replacement.h"

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace clang {
namespace c2s {

enum class HelperFileEnum : unsigned int {
#define HELPERFILE(PATH, UNIQUE_ENUM) UNIQUE_ENUM,
#define HELPER_FEATURE_MAP_TO_APINAME(File, FeatureName, APIName)
#include "../../runtime/c2s-rt/include/HelperFileAndFeatureNames.inc"
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
#define C2S_FEATURE_ENUM
#undef C2S_FEATURE_ENUM_FEATURE_PAIR_MAP
#include "clang/C2S/HelperFeatureEnum.inc"
#undef C2S_FEATURE_ENUM
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

std::string getCopyrightHeader(const clang::c2s::HelperFileEnum File);
std::pair<std::string, std::string>
getHeaderGuardPair(const clang::c2s::HelperFileEnum File);
std::string
getHelperFileContent(const clang::c2s::HelperFileEnum File,
                     std::vector<clang::c2s::HelperFunc> ContentVec);
std::string getC2SVersionStr();
void emitC2SVersionWarningIfNeed(const std::string &VersionFromYaml);
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

extern std::map<std::pair<clang::c2s::HelperFileEnum, std::string>,
                clang::c2s::HelperFunc>
    HelperNameContentMap;
extern std::unordered_map<clang::c2s::HelperFileEnum, std::string>
    HelperFileNameMap;
extern std::unordered_map<std::string, clang::c2s::HelperFileEnum>
    HelperFileIDMap;
extern const std::unordered_map<clang::c2s::HelperFileEnum, std::string>
    HelperFileHeaderGuardMacroMap;

extern const std::string C2SAllContentStr;
extern const std::string AtomicAllContentStr;
extern const std::string BlasUtilsAllContentStr;
extern const std::string DeviceAllContentStr;
extern const std::string DplUtilsAllContentStr;
extern const std::string ImageAllContentStr;
extern const std::string KernelAllContentStr;
extern const std::string MemoryAllContentStr;
extern const std::string UtilAllContentStr;
extern const std::string RngUtilsAllContentStr;
extern const std::string LibCommonUtilsAllContentStr;
extern const std::string DplExtrasAlgorithmAllContentStr;
extern const std::string DplExtrasFunctionalAllContentStr;
extern const std::string DplExtrasIteratorsAllContentStr;
extern const std::string DplExtrasMemoryAllContentStr;
extern const std::string DplExtrasNumericAllContentStr;
extern const std::string DplExtrasVectorAllContentStr;
extern const std::string DplExtrasDpcppExtensionsAllContentStr;

extern const std::map<clang::c2s::HelperFeatureIDTy, std::string>
    FeatureNameToAPINameMap;
extern const std::unordered_map<clang::c2s::HelperFeatureEnum,
                                clang::c2s::HelperFeatureIDTy>
    HelperFeatureEnumPairMap;

extern const std::unordered_map<std::string, clang::c2s::HelperFeatureEnum>
    PropToGetFeatureMap;
extern const std::unordered_map<std::string, clang::c2s::HelperFeatureEnum>
    PropToSetFeatureMap;
extern const std::unordered_map<std::string, clang::c2s::HelperFeatureEnum>
    SamplingInfoToSetFeatureMap;
extern const std::unordered_map<std::string, clang::c2s::HelperFeatureEnum>
    SamplingInfoToGetFeatureMap;
extern const std::unordered_map<std::string, clang::c2s::HelperFeatureEnum>
    ImageWrapperBaseToSetFeatureMap;
extern const std::unordered_map<std::string, clang::c2s::HelperFeatureEnum>
    ImageWrapperBaseToGetFeatureMap;

} // namespace c2s
} // namespace clang

#endif // C2S_CUSTOM_HELPER_FILES_H
