//===----------------------- ReMigration.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_REMIGRATION_H__
#define __DPCT_REMIGRATION_H__

#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/ReplacementsYaml.h"

namespace clang {
namespace dpct {
class Hunk {
public:
  enum HunkType : unsigned {
    ModifyFile = 0,
    AddFile,
    DeleteFile,
    MoveFile,
    Unspecified
  };

private:
  HunkType HT = Unspecified;

public:
  Hunk(HunkType HT) : HT(HT) {}
  HunkType getHunkType() const { return HT; }
  virtual ~Hunk() = default;
};

class ModifyFileHunk : public Hunk, public tooling::Replacement {
public:
  ModifyFileHunk() : Hunk(ModifyFile), Replacement() {}
  ModifyFileHunk(const std::string &FilePath, unsigned Offset, unsigned Length,
                 const std::string &ReplacementText)
      : Hunk(ModifyFile),
        Replacement(FilePath, Offset, Length, ReplacementText) {}
  ModifyFileHunk(const Replacement &R) : Hunk(ModifyFile), Replacement(R) {}
};

class AddFileHunk : public Hunk {
  std::string NewFilePath;

public:
  AddFileHunk() : Hunk(AddFile) {}
  AddFileHunk(std::string NewFilePath);
  const std::string &getNewFilePath() const { return NewFilePath; }
};

class DeleteFileHunk : public Hunk {
  std::string OldFilePath;

public:
  DeleteFileHunk() : Hunk(DeleteFile) {}
  DeleteFileHunk(std::string OldFilePath);
  const std::string &getOldFilePath() const { return OldFilePath; }
};

class MoveFileHunk : public Hunk, public tooling::Replacement {
  std::string NewFilePath;

public:
  MoveFileHunk() : Hunk(MoveFile), Replacement() {}
  MoveFileHunk(const std::string &FilePath, unsigned Offset, unsigned Length,
               const std::string &ReplacementText,
               const std::string &NewFilePath)
      : Hunk(MoveFile), Replacement(FilePath, Offset, Length, ReplacementText),
        NewFilePath(std::move(NewFilePath)) {}
  MoveFileHunk(const Replacement &R, const std::string &NewFilePath)
      : Hunk(MoveFile), Replacement(R), NewFilePath(NewFilePath) {}
  std::string getNewFilePath() const { return NewFilePath; }
};

struct GitDiffChanges {
  std::vector<ModifyFileHunk> ModifyFileHunks;
  std::vector<AddFileHunk> AddFileHunks;
  std::vector<DeleteFileHunk> DeleteFileHunks;
  std::vector<MoveFileHunk> MoveFileHunks;
};
GitDiffChanges &getUpstreamChanges();
GitDiffChanges &getUserChanges();
} // namespace dpct
} // namespace clang

LLVM_YAML_IS_SEQUENCE_VECTOR(clang::dpct::ModifyFileHunk)
LLVM_YAML_IS_SEQUENCE_VECTOR(clang::dpct::AddFileHunk)
LLVM_YAML_IS_SEQUENCE_VECTOR(clang::dpct::DeleteFileHunk)
LLVM_YAML_IS_SEQUENCE_VECTOR(clang::dpct::MoveFileHunk)
LLVM_YAML_DECLARE_ENUM_TRAITS(clang::dpct::Hunk::HunkType)
namespace llvm {
namespace yaml {
inline void ScalarEnumerationTraits<clang::dpct::Hunk::HunkType>::enumeration(
    IO &io, clang::dpct::Hunk::HunkType &value) {
  io.enumCase(value, "ModifyFile", clang::dpct::Hunk::HunkType::ModifyFile);
  io.enumCase(value, "AddFile", clang::dpct::Hunk::HunkType::AddFile);
  io.enumCase(value, "DeleteFile", clang::dpct::Hunk::HunkType::DeleteFile);
  io.enumCase(value, "MoveFile", clang::dpct::Hunk::HunkType::MoveFile);
  io.enumCase(value, "Unspecified", clang::dpct::Hunk::HunkType::Unspecified);
}

template <> struct MappingTraits<clang::dpct::ModifyFileHunk> {
  struct NormalizedModifyFileHunk {
    NormalizedModifyFileHunk(const IO &io) : Offset(0), Length(0) {}
    NormalizedModifyFileHunk(const IO &io, clang::dpct::ModifyFileHunk &H)
        : FilePath(H.getFilePath()), Offset(H.getOffset()),
          Length(H.getLength()), ReplacementText(H.getReplacementText()) {}

    clang::dpct::ModifyFileHunk denormalize(const IO &) {
      clang::dpct::ModifyFileHunk H(FilePath, Offset, Length, ReplacementText);
      return H;
    }

    std::string FilePath;
    unsigned int Offset;
    unsigned int Length;
    std::string ReplacementText;
  };

  static void mapping(IO &Io, clang::dpct::ModifyFileHunk &H) {
    MappingNormalization<NormalizedModifyFileHunk, clang::dpct::ModifyFileHunk>
        Keys(Io, H);
    Io.mapRequired("FilePath", Keys->FilePath);
    Io.mapRequired("Offset", Keys->Offset);
    Io.mapRequired("Length", Keys->Length);
    Io.mapRequired("ReplacementText", Keys->ReplacementText);
  }
};

template <> struct MappingTraits<clang::dpct::AddFileHunk> {
  struct NormalizedAddFileHunk {
    NormalizedAddFileHunk(const IO &io) {}
    NormalizedAddFileHunk(const IO &io, clang::dpct::AddFileHunk &H)
        : NewFilePath(H.getNewFilePath()) {}

    clang::dpct::AddFileHunk denormalize(const IO &io) {
      clang::dpct::AddFileHunk H(NewFilePath);
      return H;
    }

    std::string NewFilePath;
  };
  static void mapping(IO &Io, clang::dpct::AddFileHunk &H) {
    MappingNormalization<NormalizedAddFileHunk, clang::dpct::AddFileHunk> Keys(
        Io, H);
    Io.mapRequired("NewFilePath", Keys->NewFilePath);
  }
};

template <> struct MappingTraits<clang::dpct::DeleteFileHunk> {
  struct NormalizedDeleteFileHunk {
    NormalizedDeleteFileHunk(const IO &io) {}
    NormalizedDeleteFileHunk(const IO &io, clang::dpct::DeleteFileHunk &H)
        : OldFilePath(H.getOldFilePath()) {}

    clang::dpct::DeleteFileHunk denormalize(const IO &io) {
      clang::dpct::DeleteFileHunk H(OldFilePath);
      return H;
    }

    std::string OldFilePath;
  };
  static void mapping(IO &Io, clang::dpct::DeleteFileHunk &H) {
    MappingNormalization<NormalizedDeleteFileHunk, clang::dpct::DeleteFileHunk>
        Keys(Io, H);
    Io.mapRequired("OldFilePath", Keys->OldFilePath);
  }
};

template <> struct MappingTraits<clang::dpct::MoveFileHunk> {
  struct NormalizedMoveFileHunk {
    NormalizedMoveFileHunk(const IO &io) : Offset(0), Length(0) {}
    NormalizedMoveFileHunk(const IO &io, clang::dpct::MoveFileHunk &H)
        : FilePath(H.getFilePath()), Offset(H.getOffset()),
          Length(H.getLength()), ReplacementText(H.getReplacementText()),
          NewFilePath(H.getNewFilePath()) {}

    clang::dpct::MoveFileHunk denormalize(const IO &io) {
      clang::dpct::MoveFileHunk H(FilePath, Offset, Length, ReplacementText,
                                  NewFilePath);
      return H;
    }

    std::string FilePath;
    unsigned int Offset;
    unsigned int Length;
    std::string ReplacementText;
    std::string NewFilePath;
  };

  static void mapping(IO &Io, clang::dpct::MoveFileHunk &H) {
    MappingNormalization<NormalizedMoveFileHunk, clang::dpct::MoveFileHunk>
        Keys(Io, H);
    Io.mapRequired("FilePath", Keys->FilePath);
    Io.mapRequired("Offset", Keys->Offset);
    Io.mapRequired("Length", Keys->Length);
    Io.mapRequired("ReplacementText", Keys->ReplacementText);
    Io.mapRequired("NewFilePath", Keys->NewFilePath);
  }
};

template <> struct MappingTraits<clang::dpct::GitDiffChanges> {
  static void mapping(IO &Io, clang::dpct::GitDiffChanges &GDC) {
    Io.mapOptional("ModifyFileHunks", GDC.ModifyFileHunks);
    Io.mapOptional("AddFileHunks", GDC.AddFileHunks);
    Io.mapOptional("DeleteFileHunks", GDC.DeleteFileHunks);
    Io.mapOptional("MoveFileHunks", GDC.MoveFileHunks);
  }
};
} // namespace yaml
} // namespace llvm

#endif // __DPCT_REMIGRATION_H__
