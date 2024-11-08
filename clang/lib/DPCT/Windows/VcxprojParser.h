//===--------------- VcxprojParser.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_VCXPROJPARSER_H
#define DPCT_VCXPROJPARSER_H

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

/// Parses \p VcxprojFile file to generate compilation database
/// "compile_commands.json" in the building directory \p BuildPath.
///
/// \param BuildDir user's building path.
/// \param VcxprojFile vcxproj file path.
void vcxprojParser(std::string &BuildPath, std::string &VcxprojFile);

#endif // DPCT_VCXPROJPARSER_H
