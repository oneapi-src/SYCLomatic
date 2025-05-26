//===--- gitdiff2yaml.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Usage:
// $ g++ gitdiff2yaml.cpp -o gitdiff2yaml
// $ cd /path/to/your/git/repo
// $ ./gitdiff2yaml <old_commit_id>
// This will output the clang replacements in YAML format.
// Limitation:
// (1) The workspace and the staging area should be clean before running
// this tool.
// (2) The line ending in the file should be '\n'.
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string execGitCommand(const std::string &CMD) {
  std::array<char, 128> Buffer;
  std::unique_ptr<FILE, int (*)(FILE *)> Pipe(popen(CMD.c_str(), "r"), pclose);
  if (!Pipe) {
    throw std::runtime_error("popen() failed!");
  }

  std::string Result;
  while (fgets(Buffer.data(), Buffer.size(), Pipe.get()) != nullptr) {
    Result += Buffer.data();
  }
  return Result;
}

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

bool startsWith(const std::string &Str, const std::string &Prefix) {
  return Str.size() >= Prefix.size() &&
         Str.compare(0, Prefix.size(), Prefix) == 0;
}

bool parseHunkHeader(const std::string &Line, HunkContext &HC) {
  const std::string HunkHeaderPrefix = "@@ -";
  if (!startsWith(Line, HunkHeaderPrefix))
    return false;

  // E.g.,
  // @@ -0,0 +1,3 @@
  //        ^
  //        |-- OldEnd
  size_t OldEnd = Line.find(' ', HunkHeaderPrefix.size());
  std::string OldPart =
      Line.substr(HunkHeaderPrefix.size(), OldEnd - HunkHeaderPrefix.size());
  size_t Comma = OldPart.find(',');
  if (Comma == std::string::npos) {
    throw std::runtime_error("Invalid hunk header format: " + Line);
  }
  HC.OldCurrentLine = std::stoi(OldPart.substr(0, Comma));
  return true;
}

// vec[0] is -1, which is a placeholder.
// vec[i] /*i is the line number (1-based)*/ is the offset at the line
// beginning.
// Assumption: the line ending in the file is '\n'.
std::vector<unsigned> calculateOldOffset(const std::string &OldFileContent) {
  std::vector<unsigned> Ret;
  Ret.push_back(-1); // Placeholder for Ret[0].
  std::istringstream OldFileStream(OldFileContent);
  std::string Line;
  unsigned Offset = 0;

  while (std::getline(OldFileStream, Line)) {
    Ret.push_back(Offset);
    Offset += Line.size() +
              1; // std::getline does not include the newline character. And we
                 // assume the line ending is '\n'. So add 1 here.
  }

  return Ret;
}

// 1. Assume the line ending in the file is '\n'.
// 2. The pair (---, +++) may occurs multiple times in one hunk, so we use a
// variable to save the delete (-) operation. The continuous delete operations
// are treated as one operation.
// 3. After the delete operation, if the next line is one or more '+'
// operations, we make them as a replace-replacement. If the next line is a
// context line, the delete operation is a delete-replacement. Then clear the
// variable.
// 4. If we meet insertions ('+') when the variable is empty, we treat it as an
// insert-replacement.
void processHunkBody(const std::string &Line, HunkContext &Ctx,
                     std::vector<Replacement> &Repls,
                     const std::vector<unsigned> &CurrentOldFileOffset) {
  static std::optional<
      std::pair<unsigned /*Delele start line number*/, unsigned /*length*/>>
      DeleteInfo;
  static std::optional<
      std::pair<unsigned /*Add start line number*/, std::string>>
      AddInfo;

  auto addRepl = [&]() {
    Replacement R;
    if (DeleteInfo.has_value() && AddInfo.has_value()) {
      // replace-replacement
      R.OldFilePath = Ctx.CurrentOldFilePath;
      R.NewFilePath = Ctx.CurrentNewFilePath;
      R.Length = DeleteInfo->second;
      R.Offset = CurrentOldFileOffset[DeleteInfo->first];
      R.ReplacementText = AddInfo->second;
      DeleteInfo.reset();
      AddInfo.reset();
    } else if (DeleteInfo.has_value()) {
      // delete-replacement
      R.OldFilePath = Ctx.CurrentOldFilePath;
      R.NewFilePath = Ctx.CurrentNewFilePath;
      R.Length = DeleteInfo->second;
      R.Offset = CurrentOldFileOffset[DeleteInfo->first];
      R.ReplacementText = "";
      DeleteInfo.reset();
    } else if (AddInfo.has_value()) {
      // insert-replacement
      R.OldFilePath = Ctx.CurrentOldFilePath;
      R.NewFilePath = Ctx.CurrentNewFilePath;
      R.Length = AddInfo->second.length();
      R.Offset = CurrentOldFileOffset[AddInfo->first];
      R.ReplacementText = AddInfo->second;
      AddInfo.reset();
    }
    Repls.push_back(R);
  };

  // Hunk end
  if (Line.empty()) {
    addRepl();
    Ctx.InHunk = false;
    return;
  }

  switch (Line[0]) {
  case ' ': {
    addRepl();
    Ctx.OldCurrentLine++;
    break;
  }
  case '-': {
    if (!DeleteInfo.has_value()) {
      auto Item = std::pair<unsigned, unsigned>(
          Ctx.OldCurrentLine,
          Line.length()); // +1 for the newline character, -1 for the
                          // '-' in the line beginng
      DeleteInfo = Item;
    } else {
      DeleteInfo->second +=
          (Line.length()); // +1 for the newline character, -1 for the
                           // '-' in the line beginng
    }
    Ctx.OldCurrentLine++;
    break;
  }
  case '+': {
    if (!AddInfo.has_value()) {
      auto Item = std::pair<unsigned, std::string>(Ctx.OldCurrentLine,
                                                   Line.substr(1) + '\n');
      AddInfo = Item;
    } else {
      AddInfo->second += (Line.substr(1) + '\n');
    }
    break;
  }
  }
}

std::vector<Replacement> parseDiff(const std::string &diffOutput,
                                   const std::string &RepoRoot) {
  std::vector<Replacement> replacements;
  std::istringstream iss(diffOutput);
  std::string line;

  HunkContext HC;
  std::vector<unsigned> CurrentOldFileOffset;

  while (std::getline(iss, line)) {
    if (startsWith(line, "diff --git")) {
      HC.FastForward = false;
      continue;
    }
    if (HC.FastForward)
      continue;

    if (startsWith(line, "---")) {
      HC.CurrentOldFilePath =
          line.substr(4) == "/dev/null" ? "/dev/null" : line.substr(6);
      std::ifstream FileStream(RepoRoot + "/" + HC.CurrentOldFilePath);
      if (!FileStream.is_open()) {
        throw std::runtime_error("Failed to open file: " + RepoRoot + "/" +
                                 HC.CurrentOldFilePath);
      }
      std::stringstream Buffer;
      Buffer << FileStream.rdbuf();
      CurrentOldFileOffset = calculateOldOffset(Buffer.str());
      continue;
    }
    if (startsWith(line, "+++")) {
      HC.CurrentNewFilePath =
          line.substr(4) == "/dev/null" ? "/dev/null" : line.substr(6);
      if (HC.CurrentOldFilePath == "/dev/null" ||
          HC.CurrentNewFilePath == "/dev/null") {
        HC.FastForward = true;
        Replacement R;
        R.OldFilePath = HC.CurrentOldFilePath;
        R.NewFilePath = HC.CurrentNewFilePath;
        replacements.emplace_back(R);
      }
      continue;
    }

    if (parseHunkHeader(line, HC)) {
      // Hunk start
      HC.InHunk = true;
      continue;
    }

    if (HC.InHunk) {
      processHunkBody(line, HC, replacements, CurrentOldFileOffset);
      continue;
    }
  }

  return replacements;
}

void printYaml(const std::vector<Replacement> &Repls) {
  std::cout << "---" << std::endl;
  std::cout << "Replacements:" << std::endl;
  for (const auto &R : Repls) {
    std::cout << "  - FilePath:       " << "'" << R.OldFilePath << "'"
              << std::endl;
    std::cout << "    Offset:         " << R.Offset << std::endl;
    std::cout << "    Length:         " << R.Length << std::endl;
    std::cout << "    ReplacementText:" << "\"" << R.ReplacementText << "\""
              << std::endl;
    std::cout << "  - NewFilePath:    " << "'" << R.NewFilePath << "'"
              << std::endl;
  }
  std::cout << "..." << std::endl;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: gitdiff2yaml <old_commit_id>" << std::endl;
    return 1;
  }
  std::string OldCommitID = argv[1];

  std::string RepoRoot = execGitCommand("git rev-parse --show-toplevel");
  RepoRoot = RepoRoot.substr(0, RepoRoot.size() - 1); // Remove the last '\n'

  std::string NewCommitID = execGitCommand("git log -1 --format=\"%H\"");
  std::string DiffOutput = execGitCommand("git diff " + OldCommitID);

  execGitCommand("git reset --hard " + OldCommitID);
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

  printYaml(Repls);
  execGitCommand("git reset --hard " + NewCommitID);

  return 0;
}
