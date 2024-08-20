//===-------------------- UnifiedPath.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_CORE_UNIFIEDPATH_H
#define LLVM_CLANG_TOOLING_CORE_UNIFIEDPATH_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <string>

namespace clang {
namespace tooling {

class UnifiedPath {
public:
  UnifiedPath() = default;
  UnifiedPath(const std::string &Path, const std::string &CWD = ".") {
    setPath(Path, CWD);
  }
  UnifiedPath(const llvm::StringRef Path, const std::string &CWD = ".") {
    setPath(Path.str(), CWD);
  }
  UnifiedPath(const llvm::Twine &Path, const std::string &CWD = ".") {
    setPath(Path.str(), CWD);
  }
  UnifiedPath(const llvm::SmallVectorImpl<char> &Path,
              const std::string &CWD = ".") {
    setPath(std::string(Path.data(), Path.size()), CWD);
  }
  UnifiedPath(const char *Path, const std::string &CWD = ".") {
    setPath(Path, CWD);
  }
  bool equalsTo(const std::string &RHS) {
    return this->equalsTo(UnifiedPath(RHS));
  }
  bool equalsTo(const llvm::StringRef RHS) {
    return this->equalsTo(UnifiedPath(RHS));
  }
  bool equalsTo(const llvm::Twine &RHS) {
    return this->equalsTo(UnifiedPath(RHS));
  }
  bool equalsTo(const llvm::SmallVectorImpl<char> &RHS) {
    return this->equalsTo(UnifiedPath(RHS));
  }
  bool equalsTo(UnifiedPath RHS) {
    return getCanonicalPath() == RHS.getCanonicalPath();
  }
  llvm::StringRef getCanonicalPath() const noexcept { return _CanonicalPath; }
  llvm::StringRef getPath() const noexcept { return _Path; }
  llvm::StringRef getAbsolutePath() const noexcept { return _AbsolutePath; }
  void setPath(const std::string &NewPath, const std::string &CWD = ".") {
    _Path = NewPath;
    _AbsolutePath.clear();
    _CanonicalPath.clear();
    makeCanonical(CWD);
    makeAbsolute(CWD);
  }

private:
  void makeCanonical(const std::string &CWD = ".");
  void makeAbsolute(const std::string &CWD = ".");
  std::string _Path;
  std::string _CanonicalPath;
  std::string _AbsolutePath;
  static std::unordered_map<std::string, std::string> CanonicalPathCache;
};
bool operator==(const clang::tooling::UnifiedPath &LHS,
                const clang::tooling::UnifiedPath &RHS);
bool operator!=(const clang::tooling::UnifiedPath &LHS,
                const clang::tooling::UnifiedPath &RHS);
bool operator<(const clang::tooling::UnifiedPath &LHS,
               const clang::tooling::UnifiedPath &RHS);
} // namespace tooling
} // namespace clang
template <> struct std::hash<clang::tooling::UnifiedPath> {
  std::size_t operator()(const clang::tooling::UnifiedPath &DP) const noexcept {
    return std::hash<std::string>{}(DP.getCanonicalPath().str());
  }
};
namespace llvm {
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const clang::tooling::UnifiedPath &RHS);
} // namespace llvm

#endif // LLVM_CLANG_TOOLING_CORE_UNIFIEDPATH_H
