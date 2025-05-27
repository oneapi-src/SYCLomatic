//===--------------------------- Hunk.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_HUNK_H
#define CLANG_DPCT_HUNK_H

#include "clang/Tooling/Core/Replacement.h"

namespace clang {
namespace dpct {
class Hunk {
public:
  enum HunkType : unsigned { ModifyFile = 0, AddFile, DeleteFile, MoveFile };

private:
  HunkType HT;

public:
  Hunk(HunkType HT) : HT(HT) {}
  HunkType getHunkType() const { return HT; }
  virtual ~Hunk() = default;
};

class ModifyFileHunk : public Hunk, public clang::tooling::Replacement {
public:
  ModifyFileHunk(const std::string &FilePath, unsigned Offset, unsigned Length,
                 const std::string &ReplacementText)
      : Hunk(ModifyFile),
        Replacement(FilePath, Offset, Length, ReplacementText) {}
};

class AddFileHunk : public Hunk {
  std::string NewFilePath;

public:
  AddFileHunk(std::string NewFilePath)
      : Hunk(AddFile), NewFilePath(std::move(NewFilePath)) {}
};

class DeleteFileHunk : public Hunk {
  std::string OldFilePath;

public:
  DeleteFileHunk(std::string OldFilePath)
      : Hunk(DeleteFile), OldFilePath(std::move(OldFilePath)) {}
};

class MoveFileHunk : public Hunk, public clang::tooling::Replacement {
  std::string NewFilePath;

public:
  MoveFileHunk(const std::string &FilePath, unsigned Offset, unsigned Length,
               const std::string &ReplacementText,
               const std::string &NewFilePath)
      : Hunk(MoveFile), Replacement(FilePath, Offset, Length, ReplacementText),
        NewFilePath(std::move(NewFilePath)) {}
};
} // namespace dpct
} // namespace clang

#endif // CLANG_DPCT_HUNK_H
