//===----------------------- UnifiedPath.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Core/UnifiedPath.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace tooling {
void UnifiedPath::makeCanonical(const std::string &CWD) {
  if (_Path.empty()) {
    return;
  }
  if (!_CanonicalPath.empty()) {
    return;
  }
  auto Iter = CanonicalPathCache.find(_Path);
  if (Iter != CanonicalPathCache.end()) {
    _CanonicalPath = Iter->second;
    return;
  }

  if (_Path == "<stdin>") {
    _CanonicalPath = "<stdin>";
    CanonicalPathCache.insert(std::pair(_Path, _CanonicalPath));
    return;
  }

  llvm::SmallString<512> Path(_Path);
  llvm::sys::fs::expand_tilde(Path, Path);
  if (!llvm::sys::path::is_absolute(Path)) {
    llvm::SmallString<512> TempPath;
    if (CWD == ".") {
      llvm::sys::fs::current_path(TempPath);
    } else {
      UnifiedPath UnifiedCWD(CWD);
      TempPath = UnifiedCWD.getCanonicalPath();
    }
    llvm::sys::path::append(TempPath, llvm::sys::path::Style::native, Path);
    Path = TempPath;
  }

  llvm::sys::path::remove_dots(Path, /* remove_dot_dot= */ true);
  llvm::sys::path::native(Path);

  llvm::SmallString<512> RealPath;
  // We need make sure the input `Path` for llvm::sys::fs::real_path is
  // exsiting, or else the behavior of real_path() is unexpected. 
  if (llvm::sys::fs::exists(Path)) {
    llvm::sys::fs::real_path(Path, RealPath, true);
  } else {
    llvm::SmallString<512> Suffix;
    if (llvm::sys::path::has_filename(Path)) {
      Suffix = llvm::sys::path::filename(Path).str();
      llvm::sys::path::remove_filename(Path);
    }
    while (!llvm::sys::fs::exists(Path)) {
      if (!llvm::sys::path::has_parent_path(Path)) {
        assert(0 && "no real directory found");
        return;
      }
      llvm::sys::path::reverse_iterator RI = llvm::sys::path::rbegin(llvm::StringRef(Path));
      llvm::SmallString<512> SuffixTemp(*RI);
      llvm::sys::path::append(SuffixTemp, llvm::sys::path::Style::native, Suffix);
      Suffix = SuffixTemp;
      Path = llvm::SmallString<512>(llvm::sys::path::parent_path(Path).str());
    }
    llvm::sys::fs::real_path(Path, RealPath, true);
    llvm::sys::path::append(RealPath, llvm::sys::path::Style::native, Suffix);
  }
#if defined(_WIN32)
  _CanonicalPath = RealPath.str().lower();
  if (_CanonicalPath.size() >= 3 && _CanonicalPath.substr(0, 3) == "unc") {
    _CanonicalPath = "\\" + _CanonicalPath.substr(3);
  }
#else
  _CanonicalPath = RealPath.str();
#endif
  CanonicalPathCache.insert(std::pair(_Path, _CanonicalPath));
}

void UnifiedPath::makeAbsolute(const std::string &CWD) {
  if (_Path.empty()) {
    return;
  }
  llvm::SmallString<512> Path(_Path);
  llvm::sys::fs::expand_tilde(Path, Path);
  if (!llvm::sys::path::is_absolute(Path)) {
    llvm::SmallString<512> TempPath;
    if (CWD == ".") {
      llvm::sys::fs::current_path(TempPath);
    } else {
      UnifiedPath UnifiedCWD(CWD);
      TempPath = UnifiedCWD.getCanonicalPath();
    }
    llvm::sys::path::append(TempPath, llvm::sys::path::Style::native, Path);
    Path = TempPath;
  }

  llvm::sys::path::remove_dots(Path, /* remove_dot_dot= */ true);
  llvm::sys::path::native(Path);
  _AbsolutePath = Path.str();
}
std::unordered_map<std::string, std::string> UnifiedPath::CanonicalPathCache;
bool operator==(const clang::tooling::UnifiedPath &LHS,
                const clang::tooling::UnifiedPath &RHS) {
  return LHS.getCanonicalPath() == RHS.getCanonicalPath();
}
bool operator!=(const clang::tooling::UnifiedPath &LHS,
                const clang::tooling::UnifiedPath &RHS) {
  return LHS.getCanonicalPath() != RHS.getCanonicalPath();
}
bool operator<(const clang::tooling::UnifiedPath &LHS,
               const clang::tooling::UnifiedPath &RHS) {
  return LHS.getCanonicalPath() < RHS.getCanonicalPath();
}
} // namespace tooling
} // namespace clang
namespace llvm {
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const clang::tooling::UnifiedPath &RHS) {
  return OS << RHS.getCanonicalPath();
}
} // namespace llvm
