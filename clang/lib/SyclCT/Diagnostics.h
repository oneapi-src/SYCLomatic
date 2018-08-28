//===--- Diagnostics.h ---------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_AST_DIAGNOSTICS_H
#define SYCLCT_AST_DIAGNOSTICS_H

#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/FormatVariadic.h"

#include "TextModification.h"

#include <assert.h>
#include <unordered_map>

namespace clang {
namespace syclct {

struct DiagnosticsMessage;

extern std::unordered_map<int, DiagnosticsMessage> DiagnosticIDTable;
extern std::unordered_map<int, DiagnosticsMessage> CommentIDTable;

struct DiagnosticsMessage {
  int ID;
  int Category;
  const char *Msg;
  DiagnosticsMessage() = default;
  DiagnosticsMessage(std::unordered_map<int, DiagnosticsMessage> &Table, int ID,
                     int Category, const char *Msg)
      : ID(ID), Category(Category), Msg(Msg) {
    assert(Table.find(ID) == Table.end() && "[SYCLCT Internal error] Two "
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

namespace DiagnosticsUtils {

template <typename... Ts> static void applyReport(DiagnosticBuilder &B) {}

template <typename FTy, typename... Ts>
static void applyReport(DiagnosticBuilder &B, const FTy &F,
                        const Ts &... Rest) {
  B << F;
  applyReport<Ts...>(B, Rest...);
}

template <typename... Ts>
void reportWarning(SourceLocation SL, const DiagnosticsMessage &Msg,
                   const CompilerInstance &CI, Ts &&... Vals) {
  DiagnosticsEngine &DiagEngine = CI.getDiagnostics();
  unsigned ID = DiagEngine.getDiagnosticIDs()->getCustomDiagID(
      (DiagnosticIDs::Level)Msg.Category, Msg.Msg);
  auto B = DiagEngine.Report(SL, ID);
  applyReport<Ts...>(B, Vals...);
}

static inline SourceLocation getStartOfLine(SourceLocation Loc,
                                               const SourceManager &SM,
                                               const LangOptions &LangOpts) {
  auto LocInfo = SM.getDecomposedLoc(Loc);
  auto Buffer = SM.getBufferData(LocInfo.first);
  auto NLPos = Buffer.find_last_of('\n', LocInfo.second);
  if (NLPos == StringRef::npos) {
    NLPos = 0;
  } else {
    NLPos++;
  }
  return Loc.getLocWithOffset(NLPos - LocInfo.second);
}

template <typename... Ts>
TextModification *insertCommentPrevLine(SourceLocation SL,
                                       const DiagnosticsMessage &Msg,
                           const CompilerInstance &CI, Ts &&... Vals) {

  auto StartLoc = getStartOfLine(SL, CI.getSourceManager(), LangOptions());
  auto Formatted = llvm::formatv(Msg.Msg, std::forward<Ts>(Vals)...);
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << Formatted;
  return new InsertComment(StartLoc, OS.str());
}

// Emits a warning/error/note and/or comment depending on MsgID. For details
template <typename IDTy, typename... Ts>
void report(SourceLocation SL, IDTy MsgID, const CompilerInstance &CI,
                         TransformSetTy *TS,
            Ts &&... Vals) {
  if (DiagnosticIDTable.find((int)MsgID) != DiagnosticIDTable.end())
    reportWarning(SL, DiagnosticIDTable[(int)MsgID], CI,
                  std::forward<Ts>(Vals)...);
  if (TS && CommentIDTable.find((int)MsgID) != CommentIDTable.end())
    TS->emplace_back(insertCommentPrevLine(SL, CommentIDTable[(int)MsgID], CI,
                          std::forward<Ts>(Vals)...));
}
} // namespace DiagnosticsUtils
} // namespace syclct
} // namespace clang
#endif // SYCLCT_AST_DIAGNOSTICS_H
