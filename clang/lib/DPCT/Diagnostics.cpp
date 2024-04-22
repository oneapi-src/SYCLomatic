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
    // More IDs may need to be added, like: 1007, 1008, 1028, 1030, 1031, 1037,
    // 1051, 1053, 1067, 1069, 1076, 1082, 1090, 1107.
    1009, // TRNA_WARNING_ERROR_HANDLING_API_COMMENTED
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
  // Separate string into list by comma
  if (SuppressWarnings != "") {
    auto WarningStrs = split(SuppressWarnings, ',');
    for (const auto &Str : WarningStrs) {
      auto Range = split(Str, '-');
      if (Range.size() == 1) {
        // Invalid number format: 100e
        if (!containOnlyDigits(Str))
          reportInvalidWarningID(Str);
        size_t ID = std::stoi(Str);
        // Invalid warning ID, not in range: 999 or 1025
        if (ID < DiagnosticsMessage::MinID || ID > DiagnosticsMessage::MaxID)
          reportInvalidWarningID(Str);
        WarningIDs.insert(std::stoi(Str));
      } else if (Range.size() == 2) {
        // Invalid hyphen-separated range: -1000 or 1000-
        if (startsWith(Str, '-') || endsWith(Str, '-'))
          reportInvalidWarningID(Str);
        // Invalid number format for begin: 100e
        if (!containOnlyDigits(Range[0]))
          reportInvalidWarningID(Range[0]);
        // Invalid number format for end: 100e
        if (!containOnlyDigits(Range[1]))
          reportInvalidWarningID(Range[1]);
        size_t RangeBegin = std::stoi(Range[0]);
        size_t RangeEnd = std::stoi(Range[1]);
        // Invalid warning ID for begin, not in range: 999 or 1025
        if (RangeBegin < DiagnosticsMessage::MinID ||
            RangeBegin > DiagnosticsMessage::MaxID)
          reportInvalidWarningID(Range[0]);
        // Invalid warning ID for end, not in range: 999 or 1025
        if (RangeEnd < DiagnosticsMessage::MinID ||
            RangeEnd > DiagnosticsMessage::MaxID)
          reportInvalidWarningID(Range[1]);
        // Invalid range (begin > end): 1011-1010
        if (RangeBegin > RangeEnd)
          reportInvalidWarningID(Str);
        for (auto I = RangeBegin; I <= RangeEnd; ++I)
          WarningIDs.insert(I);
      } else {
        // Invalid hyphen-separated range: 1000-1024-1
        reportInvalidWarningID(Str);
      }
    }
  }
}
} // namespace dpct
} // namespace clang
