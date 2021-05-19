//===--- CustomHelperFiles.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_CUSTOM_HELPER_FILES_H
#define DPCT_CUSTOM_HELPER_FILES_H

#include "clang/AST/Decl.h"
#include "clang/Tooling/Core/Replacement.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace clang {
namespace dpct {

enum class HelperFileEnum : unsigned int {
#define HELPERFILE(PATH, UNIQUE_ENUM) UNIQUE_ENUM,
#include "../../runtime/dpct-rt/include/HelperFileNames.inc"
#undef HELPERFILE
  HelperFileEnumTypeSize,
};

using HelperFeatureIDTy = std::pair<HelperFileEnum, std::string>;

struct HelperFunc {
  std::string Namespace; // the namespace of this helper function feature
  int PositionIdx = -1;  // the position of this helper function feature
  bool IsCalled = false; // has this feature be called
  std::set<std::string> CallerSrcFiles; // files have called this feature
  std::vector<HelperFeatureIDTy>
      Dependency;   // some features which this feature depends on
  std::string Code; // the code of this feature
  std::string USMCode; // the code of this feature
  std::string NonUSMCode; // the code of this feature
};

void requestFeature(clang::dpct::HelperFileEnum FileID,
                    std::string HelperFunctionName,
                    const std::string &UsedFile);
void requestFeature(clang::dpct::HelperFileEnum FileID,
                    std::string HelperFunctionName, clang::SourceLocation SL);
void requestFeature(clang::dpct::HelperFileEnum FileID,
                    std::string HelperFunctionName, const clang::Stmt *Stmt);
void requestFeature(clang::dpct::HelperFileEnum FileID,
                    std::string HelperFunctionName, const clang::Decl *Decl);

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
} // namespace dpct
} // namespace clang

#endif // DPCT_CUSTOM_HELPER_FILES_H