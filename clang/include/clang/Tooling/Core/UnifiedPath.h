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
  UnifiedPath(const std::string &Path) : _Path(Path) { makeCanonical(); }
  UnifiedPath(const llvm::StringRef Path) : _Path(Path.str()) { makeCanonical(); }
  UnifiedPath(const llvm::Twine &Path) : _Path(Path.str()) { makeCanonical(); }
  UnifiedPath(const llvm::SmallVectorImpl<char> &Path) {
    _Path = std::string(Path.data(), Path.size());
    makeCanonical();
  }
  UnifiedPath(const char *Path) {
    _Path = std::string(Path);
    makeCanonical();
  }
  bool equalsTo(const std::string &RHS) {
    return this->equalsTo(UnifiedPath(RHS));
  }
  bool equalsTo(const llvm::StringRef RHS) { return this->equalsTo(UnifiedPath(RHS)); }
  bool equalsTo(const llvm::Twine &RHS) { return this->equalsTo(UnifiedPath(RHS)); }
  bool equalsTo(const llvm::SmallVectorImpl<char> &RHS) {
    return this->equalsTo(UnifiedPath(RHS));
  }
  bool equalsTo(UnifiedPath RHS) {
    return getCanonicalPath() == RHS.getCanonicalPath();
  }
  llvm::StringRef getCanonicalPath() const noexcept { return _CanonicalPath; }
  llvm::StringRef getPath() const noexcept { return _Path; }
  void setPath(const std::string &NewPath) {
    _Path = NewPath;
    _CanonicalPath.clear();
    makeCanonical();
  }

private:
  void makeCanonical();
  std::string _Path;
  std::string _CanonicalPath;
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
