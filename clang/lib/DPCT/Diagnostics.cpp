//===--------------- Diagnostics.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Diagnostics.h"
#include "clang/Basic/DiagnosticIDs.h"

namespace clang {
namespace dpct {
std::set<int> WarningIDs;
namespace DiagnosticsUtils {
unsigned int UniqueID = 0;
std::unordered_map<std::string, std::unordered_set<std::string>>
    ReportedWarningInfo::ReportedWarning;
bool checkDuplicated(const std::string &FileAndLine,
                     const std::string &WarningIDAndMsg) {
  if (ReportedWarningInfo::getInfo().count(FileAndLine) == 0) {
    ReportedWarningInfo::getInfo()[FileAndLine].insert(WarningIDAndMsg);
  } else {
    if (ReportedWarningInfo::getInfo()[FileAndLine].count(WarningIDAndMsg) ==
        0) {
      ReportedWarningInfo::getInfo()[FileAndLine].insert(WarningIDAndMsg);
    } else {
      return true;
    }
  }
  return false;
}
} // namespace DiagnosticsUtils

std::unordered_map<int, DiagnosticsMessage> DiagnosticIDTable;
std::unordered_map<int, DiagnosticsMessage> CommentIDTable;

#define HIGH_LEVEL EffortLevel::EL_High
#define MEDIUM_LEVEL EffortLevel::EL_Medium
#define LOW_LEVEL EffortLevel::EL_Low

#define DEF_WARNING(NAME, ID, LEVEL, MSG)                                      \
  DiagnosticsMessage wg_##NAME(DiagnosticIDTable, ID,                          \
                               clang::DiagnosticIDs::Warning, LEVEL, MSG);

#define DEF_COMMENT(NAME, ID, LEVEL, MSG)                                      \
  DiagnosticsMessage cg_##NAME(CommentIDTable, ID, clang::DiagnosticIDs::Note, \
                               LEVEL, MSG);

#include "Diagnostics.inc"

std::unordered_set<int> APIQueryNeedReportWarningIDSet = {
    // More IDs may need to be added, like: 1007, 1028, 1030, 1031, 1037,
    // 1051, 1053, 1067, 1069, 1076, 1082, 1090, 1107.
    1008, // API_NOT_MIGRATED_SYCL_UNDEF
    1009, // ERROR_HANDLING_API_REPLACED_BY_DUMMY
    1014, // STREAM_FLAG_PRIORITY_NOT_SUPPORTED
    1023, // MASK_UNSUPPORTED
    1029, // DEVICE_LIMIT_NOT_SUPPORTED
    1086, // ACTIVE_MASK
};

std::unordered_map<int, DiagnosticsMessage> MsgIDTable;
#define DEF_COMMENT(NAME, ID, MSG)                                             \
  DiagnosticsMessage cg_##NAME(MsgIDTable, ID, clang::DiagnosticIDs::Note,     \
                               EffortLevel::EL_Low, MSG);
#include "DiagnosticsBuildScript.inc"
#undef DEF_COMMENT

#define DEF_COMMENT(NAME, ID, MSG)                                             \
  DiagnosticsMessage cg_##NAME(MsgIDTable, ID, clang::DiagnosticIDs::Note,     \
                               EffortLevel::EL_Low, MSG);
#include "DiagnosticsCMakeScriptMigration.inc"
#undef DEF_COMMENT

void reportInvalidWarningID(const std::string &Str) {
  DpctLog() << "Invalid warning ID or range: " << Str << "\n";
  ShowStatus(MigrationErrorInvalidWarningID);
  dpctExit(MigrationErrorInvalidWarningID);
}

void initWarningIDs() {
  for (const auto &ID : SuppressWarnings) {
    auto Cur = ID.c_str();
    auto ParseNumber = [&]() {
      size_t Value = 0;
      char CurCh = *Cur;
      while (CurCh) {
        int Digit = *Cur - '0';
        if (Digit < 0 || Digit > 9)
          break;
        Value = Value * 10 + Digit;
        CurCh = *++Cur;
      }
      if (Value < DiagnosticsMessage::MinID || Value > DiagnosticsMessage::MaxID)
        reportInvalidWarningID(ID);
      return Value;
    };
    auto Begin = ParseNumber();

    if (*Cur == '\0') {
      WarningIDs.insert(Begin);
      continue;
    } else if (*Cur == '-') {
      ++Cur;
      auto End = ParseNumber();
      if (*Cur == '\0' && Begin < End) {
        for (size_t I = Begin; I < End; ++I)
          WarningIDs.insert(I);
        continue;
      }
    }
    reportInvalidWarningID(ID);
  }
}
} // namespace dpct
} // namespace clang
