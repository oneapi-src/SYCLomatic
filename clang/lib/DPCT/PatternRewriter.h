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

std::string applyPatternRewriter(const MetaRuleObject::PatternRewriter &PP,
                                 const std::string &Input);

bool fixLineEndings(const std::string &Input, std::string &Output);

std::string convertCmakeCommandsToLower(const std::string &InputString);

#endif // DPCT_PATTERN_REWRITER_H