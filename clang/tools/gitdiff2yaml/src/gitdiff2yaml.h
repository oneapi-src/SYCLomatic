//===---------------------- gitdiff2yaml.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GITDIFF2YAML_H__
#define __GITDIFF2YAML_H__

#include <string>
#include <vector>

const std::string LineEnd = "\n";

struct Replacement {
  std::string NewFilePath;
  std::string OldFilePath;
  unsigned Offset = 0;
  unsigned Length = 0;
  std::string ReplacementText;
};

struct HunkContext {
  unsigned OldCurrentLine = 0;
  bool InHunk = false;
  bool FastForward = false;
  std::string CurrentNewFilePath;
  std::string CurrentOldFilePath;
};

std::string execGitCommand(const std::string &CMD);
std::vector<Replacement> parseDiff(const std::string &diffOutput,
                                   const std::string &RepoRoot);
void printYaml(std::ostream &stream, const std::vector<Replacement> &Repls);

#endif // __GITDIFF2YAML_H__
