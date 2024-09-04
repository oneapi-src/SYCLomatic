//===--------------- PatternRewriter.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_PATTERN_REWRITER_H
#define DPCT_PATTERN_REWRITER_H

#include <map>
#include <string>

#include "Rules.h"

std::string
applyPatternRewriter(const MetaRuleObject::PatternRewriter &PP,
                     const std::string &Input, std::string FileName = "",
                     std::string FrontPart = "",
                     const clang::tooling::UnifiedPath OutRoot = "");

bool fixLineEndings(const std::string &Input, std::string &Output);

enum SourceFileType { SFT_CAndCXXSource, SFT_CMakeScript };
void setFileTypeProcessed(enum SourceFileType FileType);

extern std::set<std::string> MainSrcFilesHasCudaSyntex;
extern bool LANG_Cplusplus_20_Used;
#endif // DPCT_PATTERN_REWRITER_H
