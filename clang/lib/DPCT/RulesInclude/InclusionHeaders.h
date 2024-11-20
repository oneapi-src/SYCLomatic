//===--------------- InclusionHeaders.h------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_INCLUSIONHEADERS_H
#define DPCT_INCLUSIONHEADERS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace dpct {

enum HeaderType {
#define HEADER(Name, Spelling) HT_##Name,
#include "RulesInclude/HeaderTypes.inc"
  NUM_HEADERS,
  HT_NULL = -1
};

enum class RuleGroupKind : uint8_t {
  RK_Common = 0,
  RK_Sparse,
  RK_BLas,
  RK_Solver,
  RK_Rng,
  RK_FFT,
  RK_DNN,
  RK_NCCL,
  RK_Libcu,
  RK_Thrust,
  RK_CUB,
  RK_WMMA,
  NUM
};

enum class LibraryDependencies : uint8_t {
#define LIBRARY(LIBNAME, ...) LD_##LIBNAME,
#include "MigrationReport/Libraries.inc"
  NUMS
};

struct DpctInclusionInfo {
  enum InclusionFlag {
    IF_MarkInserted,
    IF_Replace,
    IF_Remove,
    IF_DoNothing
  };
  unsigned ProcessFlag : 2;
  unsigned MustAngled : 1;
  RuleGroupKind RuleGroup;
  llvm::SmallVector<HeaderType, 2> Headers;
};

class RuleGroups {
  using FlagsType = uint64_t;

  template <class... Args>
  static constexpr FlagsType flag(RuleGroupKind First, Args... Rest) noexcept {
    return flag(First) | flag(Rest...);
  }
  static constexpr FlagsType flag(RuleGroupKind K) noexcept {
    return 1 << static_cast<uint8_t>(K);
  }

  FlagsType Flags = flag(RuleGroupKind::RK_Common);

public:
  void enableRuleGroup(RuleGroupKind K) noexcept { Flags |= flag(K); }
  bool isEnabled(RuleGroupKind K) const noexcept { return Flags & flag(K); }
  bool isDependsOn(LibraryDependencies LD) const noexcept {
    static struct LibraryMasks {
      FlagsType BitMasks[static_cast<uint8_t>(LibraryDependencies::NUMS)];
      LibraryMasks() noexcept {
#define LIBRARY(LIBNAME, LIBDESC, ...)                                         \
  BitMasks[static_cast<uint8_t>(LibraryDependencies::LD_##LIBNAME)] =          \
      flag(__VA_ARGS__);
#include "MigrationReport/Libraries.inc"
      }
    } Masks;
    return Flags & Masks.BitMasks[static_cast<uint8_t>(LD)];
  }
  bool isMKLEnabled() const noexcept {
    return isDependsOn(LibraryDependencies::LD_MKL);
  }
};

class DpctInclusionHeadersMap {
  struct DpctInclusionHeadersMapInitializer {
    DpctInclusionHeadersMapInitializer();
  };
  static DpctInclusionHeadersMapInitializer Initializer;

public:
  enum MatchMode { Mode_FullMatch, Mode_Startwith };

public:
  static const DpctInclusionInfo *findHeaderInfo(llvm::StringRef IncludeFile);
  template <class... Args>
  static void registInclusionHeaderEntry(llvm::StringRef Filename,
                                         MatchMode Mode, RuleGroupKind Group,
                                         DpctInclusionInfo::InclusionFlag Flag,
                                         bool MustAngled, Args... Headers);
};

} // namespace dpct
} // namespace clang

#endif