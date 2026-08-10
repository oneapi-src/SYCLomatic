//===--- gitdiff2yaml.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gitdiff2yaml.h"

#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_os_ostream.h"

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
bool startsWith(const std::string &Str, const std::string &Prefix) {
  return Str.size() >= Prefix.size() &&
         Str.compare(0, Prefix.size(), Prefix) == 0;
}

struct HunkContext {
  unsigned OldCurrentLine = 0;
  bool InHunk = false;
  bool FastForward = false;
  std::string CurrentNewFilePath;
  std::string CurrentOldFilePath;
};

bool parseHunkHeader(const std::string &Line, HunkContext &HC) {
  const std::string HunkHeaderPrefix = "@@ -";
  if (!startsWith(Line, HunkHeaderPrefix))
    return false;

  // E.g.,
  // @@ -0,0 +1,3 @@
  //        ^
  //        |-- OldEnd
  // E.g.,
  // @@ -24 +24 @@
  //       ^
  //       |-- OldEnd
  size_t OldEnd = Line.find(' ', HunkHeaderPrefix.size());
  std::string OldPart =
      Line.substr(HunkHeaderPrefix.size(), OldEnd - HunkHeaderPrefix.size());
  size_t Comma = OldPart.find(',');
  if (Comma == std::string::npos) {
    HC.OldCurrentLine = std::stoi(OldPart);
  } else {
    HC.OldCurrentLine = std::stoi(OldPart.substr(0, Comma));
  }
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

struct ModifyHunk {
  ModifyHunk() = default;
  ModifyHunk(std::string FilePath, unsigned Offset, unsigned Length,
             std::string ReplacementText)
      : FilePath(FilePath), Offset(Offset), Length(Length),
        ReplacementText(ReplacementText) {};
  std::string FilePath;
  unsigned Offset = 0;
  unsigned Length = 0;
  std::string ReplacementText;
};

struct AddHunk {
  AddHunk() = default;
  AddHunk(std::string NewFilePath) : NewFilePath(NewFilePath) {};
  std::string NewFilePath;
};

struct DeleteHunk {
  DeleteHunk() = default;
  DeleteHunk(std::string OldFilePath) : OldFilePath(OldFilePath) {};
  std::string OldFilePath;
};

struct MoveHunk {
  MoveHunk() = default;
  MoveHunk(std::string FilePath, unsigned Offset, unsigned Length,
           std::string ReplacementText, std::string NewFilePath)
      : FilePath(FilePath), Offset(Offset), Length(Length),
        ReplacementText(ReplacementText), NewFilePath(NewFilePath) {};
  std::string FilePath;
  unsigned Offset = 0;
  unsigned Length = 0;
  std::string ReplacementText;
  std::string NewFilePath;
};

struct GitDiffChanges {
  std::vector<ModifyHunk> ModifyHunks;
  std::vector<AddHunk> AddHunks;
  std::vector<DeleteHunk> DeleteHunks;
  std::vector<MoveHunk> MoveHunks;
};
} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(ModifyHunk)
LLVM_YAML_IS_SEQUENCE_VECTOR(AddHunk)
LLVM_YAML_IS_SEQUENCE_VECTOR(DeleteHunk)
LLVM_YAML_IS_SEQUENCE_VECTOR(MoveHunk)
namespace llvm {
namespace yaml {
template <> struct MappingTraits<ModifyHunk> {
  struct NormalizedModifyFileHunk {
    NormalizedModifyFileHunk(const IO &io) : Offset(0), Length(0) {}
    NormalizedModifyFileHunk(const IO &io, ModifyHunk &H)
        : FilePath(H.FilePath), Offset(H.Offset), Length(H.Length),
          ReplacementText(H.ReplacementText) {}
    ModifyHunk denormalize(const IO &) {
      ModifyHunk H(FilePath, Offset, Length, ReplacementText);
      return H;
    }
    std::string FilePath;
    unsigned int Offset;
    unsigned int Length;
    std::string ReplacementText;
  };

  static void mapping(IO &Io, ModifyHunk &H) {
    MappingNormalization<NormalizedModifyFileHunk, ModifyHunk> Keys(Io, H);
    Io.mapRequired("FilePath", Keys->FilePath);
    Io.mapRequired("Offset", Keys->Offset);
    Io.mapRequired("Length", Keys->Length);
    Io.mapRequired("ReplacementText", Keys->ReplacementText);
  }
};

template <> struct MappingTraits<AddHunk> {
  struct NormalizedAddFileHunk {
    NormalizedAddFileHunk(const IO &io) {}
    NormalizedAddFileHunk(const IO &io, AddHunk &H)
        : NewFilePath(H.NewFilePath) {}
    AddHunk denormalize(const IO &io) {
      AddHunk H(NewFilePath);
      return H;
    }
    std::string NewFilePath;
  };
  static void mapping(IO &Io, AddHunk &H) {
    MappingNormalization<NormalizedAddFileHunk, AddHunk> Keys(Io, H);
    Io.mapRequired("NewFilePath", Keys->NewFilePath);
  }
};

template <> struct MappingTraits<DeleteHunk> {
  struct NormalizedDeleteFileHunk {
    NormalizedDeleteFileHunk(const IO &io) {}
    NormalizedDeleteFileHunk(const IO &io, DeleteHunk &H)
        : OldFilePath(H.OldFilePath) {}
    DeleteHunk denormalize(const IO &io) {
      DeleteHunk H(OldFilePath);
      return H;
    }
    std::string OldFilePath;
  };
  static void mapping(IO &Io, DeleteHunk &H) {
    MappingNormalization<NormalizedDeleteFileHunk, DeleteHunk> Keys(Io, H);
    Io.mapRequired("OldFilePath", Keys->OldFilePath);
  }
};

template <> struct MappingTraits<MoveHunk> {
  struct NormalizedMoveFileHunk {
    NormalizedMoveFileHunk(const IO &io) : Offset(0), Length(0) {}
    NormalizedMoveFileHunk(const IO &io, MoveHunk &H)
        : FilePath(H.FilePath), Offset(H.Offset), Length(H.Length),
          ReplacementText(H.ReplacementText), NewFilePath(H.NewFilePath) {}
    MoveHunk denormalize(const IO &io) {
      MoveHunk H(FilePath, Offset, Length, ReplacementText, NewFilePath);
      return H;
    }
    std::string FilePath;
    unsigned int Offset;
    unsigned int Length;
    std::string ReplacementText;
    std::string NewFilePath;
  };
  static void mapping(IO &Io, MoveHunk &H) {
    MappingNormalization<NormalizedMoveFileHunk, MoveHunk> Keys(Io, H);
    Io.mapRequired("FilePath", Keys->FilePath);
    Io.mapRequired("Offset", Keys->Offset);
    Io.mapRequired("Length", Keys->Length);
    Io.mapRequired("ReplacementText", Keys->ReplacementText);
    Io.mapRequired("NewFilePath", Keys->NewFilePath);
  }
};

template <> struct MappingTraits<GitDiffChanges> {
  static void mapping(IO &Io, GitDiffChanges &GDC) {
    Io.mapOptional("ModifyFileHunks", GDC.ModifyHunks);
    Io.mapOptional("AddFileHunks", GDC.AddHunks);
    Io.mapOptional("DeleteFileHunks", GDC.DeleteHunks);
    Io.mapOptional("MoveFileHunks", GDC.MoveHunks);
  }
};
} // namespace yaml
} // namespace llvm

void printYaml(std::ostream &stream, const std::vector<Replacement> &Repls) {
  GitDiffChanges GDC;

  for (const auto &R : Repls) {
    if (R.OldFilePath == "/dev/null" && R.NewFilePath != "/dev/null") {
      // Add replacement
      AddHunk AH(R.NewFilePath);
      GDC.AddHunks.push_back(AH);
      continue;
    }
    if (R.OldFilePath != "/dev/null" && R.NewFilePath == "/dev/null") {
      // Delete replacement
      DeleteHunk DH(R.OldFilePath);
      GDC.DeleteHunks.push_back(DH);
      continue;
    }
    if (R.OldFilePath == R.NewFilePath && R.OldFilePath != "/dev/null") {
      // Modify replacement
      ModifyHunk MH(R.OldFilePath, R.Offset, R.Length, R.ReplacementText);
      GDC.ModifyHunks.push_back(MH);
      continue;
    }
    if (R.OldFilePath != R.NewFilePath) {
      // Move replacement
      MoveHunk MH(R.OldFilePath, R.Offset, R.Length, R.ReplacementText,
                  R.NewFilePath);
      GDC.MoveHunks.push_back(MH);
      continue;
    }
    throw std::runtime_error("Invalid replacement: " + R.OldFilePath + " -> " +
                             R.NewFilePath);
  }

  llvm::raw_os_ostream raw_os(stream);
  llvm::yaml::Output yout(raw_os);
  yout << GDC;
}

std::vector<Replacement> parseDiff(const std::string &diffOutput,
                                   const std::string &RepoRoot) {
  std::vector<Replacement> replacements;
  std::istringstream iss(diffOutput);
  std::string line;

  HunkContext HC;
  std::vector<unsigned> CurrentOldFileOffset;

  std::optional<
      std::pair<unsigned /*Delele start line number*/, unsigned /*length*/>>
      DeleteInfo;
  std::optional<std::pair<unsigned /*Add start line number*/, std::string>>
      AddInfo;

  auto addRepl = [&]() {
    Replacement R;
    if (DeleteInfo.has_value() && AddInfo.has_value()) {
      // replace-replacement
      R.OldFilePath = HC.CurrentOldFilePath;
      R.NewFilePath = HC.CurrentNewFilePath;
      R.Length = DeleteInfo->second;
      R.Offset = CurrentOldFileOffset[DeleteInfo->first];
      R.ReplacementText = AddInfo->second;
      DeleteInfo.reset();
      AddInfo.reset();
    } else if (DeleteInfo.has_value()) {
      // delete-replacement
      R.OldFilePath = HC.CurrentOldFilePath;
      R.NewFilePath = HC.CurrentNewFilePath;
      R.Length = DeleteInfo->second;
      R.Offset = CurrentOldFileOffset[DeleteInfo->first];
      R.ReplacementText = "";
      DeleteInfo.reset();
    } else if (AddInfo.has_value()) {
      // insert-replacement
      R.OldFilePath = HC.CurrentOldFilePath;
      R.NewFilePath = HC.CurrentNewFilePath;
      R.Length = 0;
      R.Offset = CurrentOldFileOffset[AddInfo->first];
      R.ReplacementText = AddInfo->second;
      AddInfo.reset();
    }
    replacements.push_back(R);
  };

  // Don't use std::getline as condition of the while loop, because it will
  // return false if the last line only containing EOF.
  while (iss.good()) {
    std::getline(iss, line);
    if (startsWith(line, "diff --git")) {
      if (HC.InHunk)
        addRepl();
      HC.InHunk = false;
      HC.FastForward = false;
      continue;
    }
    if (HC.FastForward)
      continue;

    if (startsWith(line, "---")) {
      if (HC.InHunk)
        addRepl();
      HC.InHunk = false;
      HC.CurrentOldFilePath =
          line.substr(4) == "/dev/null" ? "/dev/null" : line.substr(6);
      if (HC.CurrentOldFilePath != "/dev/null") {
        std::ifstream FileStream(RepoRoot + "/" + HC.CurrentOldFilePath);
        if (!FileStream.is_open()) {
          throw std::runtime_error("Failed to open file: " + RepoRoot + "/" +
                                   HC.CurrentOldFilePath);
        }
        std::stringstream Buffer;
        Buffer << FileStream.rdbuf();
        CurrentOldFileOffset = calculateOldOffset(Buffer.str());
      }
      continue;
    }
    if (startsWith(line, "+++")) {
      if (HC.InHunk)
        addRepl();
      HC.InHunk = false;
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
      if (HC.InHunk)
        addRepl();
      // Hunk start
      HC.InHunk = true;
      continue;
    }

    // parse hunk body
    // 1. Assume the line ending in the file is '\n'.
    // 2. The pair (---, +++) should occur only once in one hunk, since
    // --unified=0
    // 3. We use a variable to save the delete (-) operation. The continuous
    // delete operations are treated as one operation.
    // 4. After the delete operation, if the next line is one or more '+'
    // operations, we make them as a replace-replacement. If the next line is a
    // context line, the delete operation is a delete-replacement. Then clear
    // the variable.
    // 5. If we meet insertions ('+') when the variable is empty, we treat it as
    // an insert-replacement.
    if (HC.InHunk) {
      switch (line[0]) {
      case '-': {
        if (!DeleteInfo.has_value()) {
          auto Item = std::pair<unsigned, unsigned>(
              HC.OldCurrentLine,
              line.length()); // +1 for the newline character, -1 for the
                              // '-' in the line beginng
          DeleteInfo = Item;
        } else {
          DeleteInfo->second +=
              (line.length()); // +1 for the newline character, -1 for the
                               // '-' in the line beginng
        }
        HC.OldCurrentLine++;
        break;
      }
      case '+': {
        if (!AddInfo.has_value()) {
          auto Item = std::pair<unsigned, std::string>(
              HC.OldCurrentLine, line.substr(1) + LineEnd);
          AddInfo = Item;
        } else {
          AddInfo->second += (line.substr(1) + LineEnd);
        }
        break;
      }
      }
      continue;
    }
  }
  if (HC.InHunk)
    addRepl();

  return replacements;
}

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
