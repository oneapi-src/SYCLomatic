//===--- Diagnostics.h ---------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_AST_DIAGNOSTICS_H
#define DPCT_AST_DIAGNOSTICS_H

#include "AnalysisInfo.h"
#include "Debug.h"
#include "SaveNewFiles.h"
#include "TextModification.h"

#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/FormatVariadic.h"

#include <assert.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

extern llvm::cl::opt<std::string> SuppressWarnings;
extern llvm::cl::opt<std::string> OutputFile;
extern llvm::cl::opt<OutputVerbosityLev> OutputVerbosity;
extern bool SuppressWarningsAllFlag;

namespace clang {
namespace dpct {

struct DiagnosticsMessage;

extern std::unordered_map<int, DiagnosticsMessage> DiagnosticIDTable;
extern std::unordered_map<int, DiagnosticsMessage> CommentIDTable;

static std::set<int> WarningIDs;

struct DiagnosticsMessage {
  int ID;
  int Category;
  const char *Msg;
  DiagnosticsMessage() = default;
  DiagnosticsMessage(std::unordered_map<int, DiagnosticsMessage> &Table, int ID,
                     int Category, const char *Msg)
      : ID(ID), Category(Category), Msg(Msg) {
    assert(Table.find(ID) == Table.end() && "[DPCT Internal error] Two "
                                            "messages with the same ID "
                                            "are being registered");
    Table[ID] = *this;
  }
};

#define DEF_NOTE(NAME, ID, MSG) NAME = ID,
#define DEF_ERROR(NAME, ID, MSG) NAME = ID,
#define DEF_WARNING(NAME, ID, MSG) NAME = ID,
#define DEF_COMMENT(NAME, ID, MSG)
enum class Diagnostics {
#include "Diagnostics.inc"
#undef DEF_NOTE
#undef DEF_ERROR
#undef DEF_WARNING
#undef DEF_COMMENT
  END
};

#define DEF_NOTE(NAME, ID, MSG)
#define DEF_ERROR(NAME, ID, MSG)
#define DEF_WARNING(NAME, ID, MSG)
#define DEF_COMMENT(NAME, ID, MSG) NAME = ID,
enum class Comments {
#include "Diagnostics.inc"
#undef DEF_NOTE
#undef DEF_ERROR
#undef DEF_WARNING
#undef DEF_COMMENT
  END
};

#define DEF_NOTE(NAME, ID, MSG)
#define DEF_ERROR(NAME, ID, MSG)
#define DEF_WARNING(NAME, ID, MSG) NAME = ID,
#define DEF_COMMENT(NAME, ID, MSG)
enum class Warnings {
  BEGIN = 1000,
#include "Diagnostics.inc"
#undef DEF_NOTE
#undef DEF_ERROR
#undef DEF_WARNING
#undef DEF_COMMENT
  END
};

namespace DiagnosticsUtils {

extern unsigned int UniqueID;

template <typename... Ts> static void applyReport(DiagnosticBuilder &B) {}

template <typename FTy, typename... Ts>
static void applyReport(DiagnosticBuilder &B, const FTy &F,
                        const Ts &... Rest) {
  B << F;
  applyReport<Ts...>(B, Rest...);
}

static inline std::string getMessagePrefix(int ID) {
  return "DPCT" + std::to_string(ID) + ":" + std::to_string(UniqueID) + ": ";
}

template <typename... Ts>
void reportWarning(SourceLocation SL, const DiagnosticsMessage &Msg,
                   const CompilerInstance &CI, Ts &&... Vals) {

  DiagnosticsEngine &DiagEngine = CI.getDiagnostics();

  std::string Message = getMessagePrefix(Msg.ID) + Msg.Msg;

  if (!OutputFile.empty()) {
    //  Redirects warning message to output file if the option "-output-file" is
    //  set
    const SourceManager &SM = CI.getSourceManager();
    int LineNum = SM.getSpellingLineNumber(SL);
    const std::pair<FileID, unsigned> DecomposedLocation =
        SM.getDecomposedLoc(SL);

    FileID FID = DecomposedLocation.first;
    unsigned *LineCache =
        SM.getSLocEntry(FID).getFile().getContentCache()->SourceLineCache;
    const char *Buffer = SM.getBuffer(FID)->getBufferStart();
    std::string LineOriCode(Buffer + LineCache[LineNum - 1],
                            Buffer + LineCache[LineNum]);

    const SourceLocation FileLoc = SM.getFileLoc(SL);
    std::string File = FileLoc.printToString(SM);
    Message = File + " warning: " + Message + "\n" + LineOriCode;
    DpctTerm() << Message;
  }

  if (OutputVerbosity != silent) {
    unsigned ID = DiagEngine.getDiagnosticIDs()->getCustomDiagID(
        (DiagnosticIDs::Level)Msg.Category, Message);
    auto B = DiagEngine.Report(SL, ID);
    applyReport<Ts...>(B, Vals...);
  }
}

// Get the starting location of a line, going through lines ending with
// backslashes
static inline SourceLocation getStartOfLine(SourceLocation Loc,
                                            const SourceManager &SM,
                                            const LangOptions &LangOpts) {
  auto LocInfo = SM.getDecomposedLoc(SM.getExpansionLoc(Loc));
  auto Buffer = SM.getBufferData(LocInfo.first);
  auto NLPos = Buffer.find_last_of('\n', LocInfo.second);
  auto Skip = 1;
  while (NLPos != StringRef::npos) {
    if (Buffer[NLPos - 1] == '\r') {
      --NLPos;
      Skip = 2;
    }

    if (Buffer[NLPos - 1] == '\\')
      NLPos = Buffer.find_last_of('\n', NLPos - 1);
    else
      break;
  }
  if (NLPos == StringRef::npos) {
    NLPos = 0;
  } else {
    NLPos += Skip;
  }
  return SM.getExpansionLoc(Loc).getLocWithOffset(NLPos - LocInfo.second);
}

template <typename... Ts>
TextModification *
insertCommentPrevLine(SourceLocation SL, const DiagnosticsMessage &Msg,
                      const CompilerInstance &CI, Ts &&... Vals) {

  auto StartLoc = getStartOfLine(SL, CI.getSourceManager(), LangOptions());
  auto Formatted = llvm::formatv(Msg.Msg, std::forward<Ts>(Vals)...);
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << getMessagePrefix(Msg.ID);
  OS << Formatted;
  return new InsertComment(StartLoc, OS.str());
}

class ReportedWarningInfo {
public:
  static std::unordered_map<std::string, std::unordered_set<std::string>> &
  getInfo() {
    return ReportedWarning;
  }

private:
  static std::unordered_map<std::string, std::unordered_set<std::string>>
      ReportedWarning;
};

// Emits a warning/error/note and/or comment depending on MsgID. For details
template <typename IDTy, typename... Ts>
void report(SourceLocation SL, IDTy MsgID, const CompilerInstance &CI,
            TransformSetTy *TS, Ts &&... Vals) {
  auto &SM = clang::dpct::DpctGlobalInfo::getSourceManager();
  std::string FileAndLine = clang::dpct::buildString(
      SM.getBufferName(SL), ":", SM.getPresumedLineNumber(SL));
  std::string WarningIDAndMsg = clang::dpct::buildString(
      std::to_string(static_cast<int>(MsgID)), ":", std::forward<Ts>(Vals)...);
  if (ReportedWarningInfo::getInfo().count(FileAndLine) == 0) {
    ReportedWarningInfo::getInfo()[FileAndLine].insert(WarningIDAndMsg);
  } else if (ReportedWarningInfo::getInfo()[FileAndLine].count(
                 WarningIDAndMsg) != 0) {
    return;
  }

  if (!SuppressWarningsAllFlag) {
    // Only report warnings that are not suppressed
    if (WarningIDs.find((int)MsgID) == WarningIDs.end() &&
        DiagnosticIDTable.find((int)MsgID) != DiagnosticIDTable.end()) {
      reportWarning(SL, DiagnosticIDTable[(int)MsgID], CI,
                    std::forward<Ts>(Vals)...);
    }
  }
  if (TS && CommentIDTable.find((int)MsgID) != CommentIDTable.end()) {
    TS->emplace_back(insertCommentPrevLine(SL, CommentIDTable[(int)MsgID], CI,
                                           std::forward<Ts>(Vals)...));
  }
  UniqueID++;
}
} // namespace DiagnosticsUtils
} // namespace dpct
} // namespace clang
#endif // DPCT_AST_DIAGNOSTICS_H
