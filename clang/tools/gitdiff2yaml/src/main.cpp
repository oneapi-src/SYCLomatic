//===------------------------- main.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Usage:
// $ cd /path/to/your/git/repo
// $ gitdiff2yaml -c <old_commit_id> -o <output_file>
// This will output the clang replacements in YAML format.
// Limitation:
// (1) The workspace and the staging area should be clean before running
// this tool.
// (2) The line ending in the file should be '\n'.
//===----------------------------------------------------------------------===//

#include "gitdiff2yaml.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static constexpr auto Description = R"--(

This gitdiff2yaml tool converts the output of `git diff` into a YAML file
containing a list of replacements.
)--";

static constexpr auto Examples = R"--(
EXAMPLES:

Output to the terminal:

  gitdiff2yaml -c <old_commit_id>

Output to a file:

  gitdiff2yaml -c <old_commit_id> -o Changes.yaml

)--";

static llvm::cl::OptionCategory &getG2YCategory() {
  static llvm::cl::OptionCategory G2YCategory("gitdiff2yaml tool options");
  return G2YCategory;
}

int main(int argc, char *argv[]) {
  llvm::cl::opt<std::string> OutputFilename(
      "o", llvm::cl::desc("[optional] Specify output filename"),
      llvm::cl::value_desc("filename"), llvm::cl::cat(getG2YCategory()),
      llvm::cl::Optional);

  llvm::cl::opt<std::string> OldCommitID(
      "c", llvm::cl::desc("[required] Specify the old commit ID"),
      llvm::cl::value_desc("commit_id"), llvm::cl::cat(getG2YCategory()),
      llvm::cl::Required);

  llvm::cl::extrahelp MoreHelp(Examples);

  llvm::cl::HideUnrelatedOptions(getG2YCategory());

  llvm::cl::ParseCommandLineOptions(argc, argv, Description);

  std::string RepoRoot = execGitCommand("git rev-parse --show-toplevel");
  RepoRoot = RepoRoot.substr(0, RepoRoot.size() - 1); // Remove the last '\n'

  std::string NewCommitID = execGitCommand("git log -1 --format=\"%H\"");
  std::string DiffOutput =
      execGitCommand("git diff --diff-algorithm=minimal --unified=0 " +
                     OldCommitID.getValue());

  execGitCommand("git reset --hard " + OldCommitID.getValue());
  std::vector<Replacement> Repls = parseDiff(DiffOutput, RepoRoot);

  // Erase emtpy replacements
  Repls.erase(std::remove_if(Repls.begin(), Repls.end(),
                             [](Replacement x) {
                               return (x.NewFilePath == "" &&
                                       x.OldFilePath == "" && x.Offset == 0 &&
                                       x.Length == 0 &&
                                       x.ReplacementText == "");
                             }),
              Repls.end());

  // Erase unrelated replacements
  Repls.erase(std::remove_if(
                  Repls.begin(), Repls.end(),
                  [](Replacement x) {
                    if (x.NewFilePath == "/dev/null")
                      return false;
                    llvm::StringRef PathRef = x.NewFilePath;
                    std::string Ext =
                        llvm::sys::path::extension(PathRef).substr(1).lower();
                    if (Ext == "cu" || Ext == "cuh" || Ext == "cpp" ||
                        Ext == "hpp" || Ext == "cxx" || Ext == "hxx" ||
                        Ext == "cc" || Ext == "hh" || Ext == "c" ||
                        Ext == "h") {
                      return false;
                    }
                    return true;
                  }),
              Repls.end());

  if (!OutputFilename.empty()) {
    std::ofstream OutFile(OutputFilename.getValue());
    if (!OutFile.is_open()) {
      std::cerr << "Failed to open output file: " << OutputFilename.getValue()
                << std::endl;
      return 1;
    }
    printYaml(OutFile, Repls);
    OutFile.close();
  } else {
    printYaml(std::cout, Repls);
  }

  execGitCommand("git reset --hard " + NewCommitID);

  return 0;
}
