//===--------------- Diagnostics.h ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_AST_DIAGNOSTICS_H
#define DPCT_AST_DIAGNOSTICS_H

#include "AnalysisInfo.h"
#include "SaveNewFiles.h"
#include "Statics.h"
#include "TextModification.h"

#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/FormatVariadic.h"

#include <assert.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

extern llvm::cl::opt<std::string> SuppressWarnings;
extern llvm::cl::opt<std::string> OutputFile;
extern llvm::cl::opt<OutputVerbosityLevel> OutputVerbosity;
extern bool SuppressWarningsAllFlag;

namespace clang {
namespace dpct {

struct DiagnosticsMessage;

extern std::unordered_map<int, DiagnosticsMessage> DiagnosticIDTable;
extern std::unordered_map<int, DiagnosticsMessage> CommentIDTable;
extern std::unordered_map<int, DiagnosticsMessage> MsgIDTable;

extern std::set<int> WarningIDs;

struct DiagnosticsMessage {
  int ID;
  int Category;
  const char *Msg;
#define DEF_NOTE(NAME, ID, MSG)
#define DEF_ERROR(NAME, ID, MSG)
#define DEF_WARNING(NAME, ID, MSG) ID,
#define DEF_COMMENT(NAME, ID, MSG)
  constexpr static size_t MinID = std::min({
#include "Diagnostics.inc"
  });
  constexpr static size_t MaxID = std::max({
#include "Diagnostics.inc"
  });
#undef DEF_NOTE
#undef DEF_ERROR
#undef DEF_WARNING
#undef DEF_COMMENT
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
};

#define DEF_NOTE(NAME, ID, MSG)
#define DEF_ERROR(NAME, ID, MSG)
#define DEF_WARNING(NAME, ID, MSG) NAME = ID,
#define DEF_COMMENT(NAME, ID, MSG)
enum class Warnings {
#include "Diagnostics.inc"
#undef DEF_NOTE
#undef DEF_ERROR
#undef DEF_WARNING
#undef DEF_COMMENT
};

#define DEF_COMMENT(NAME, ID, MSG) NAME = ID,
enum class MakefileMsgs {
#include "DiagnosticsBuildScript.inc"
#undef DEF_COMMENT
};

namespace DiagnosticsUtils {

extern unsigned int UniqueID;

template <typename... Ts> static void applyReport(DiagnosticBuilder &B) {}

template <typename FTy, typename... Ts>
static void applyReport(DiagnosticBuilder &B, const FTy &F, const Ts &...Rest) {
  B << F;
  applyReport<Ts...>(B, Rest...);
}

static inline std::string getMessagePrefix(int ID) {
  return "DPCT" + std::to_string(ID) + ":" + std::to_string(UniqueID) + ": ";
}

template <typename... Ts>
void reportWarning(SourceLocation SL, const DiagnosticsMessage &Msg,
                   DiagnosticsEngine &Engine, Ts &&...Vals) {
  std::string Message = getMessagePrefix(Msg.ID) + Msg.Msg;
  if (OutputVerbosity != OutputVerbosityLevel::OVL_Silent) {
    unsigned ID = Engine.getDiagnosticIDs()->getCustomDiagID(
        (DiagnosticIDs::Level)Msg.Category, Message);
    auto B = Engine.Report(SL, ID);
    applyReport<Ts...>(B, Vals...);
  }
}

// Get the starting location of a line, going through lines ending with
// backslashes
static inline SourceLocation getStartOfLine(SourceLocation Loc,
                                            const SourceManager &SM,
                                            const LangOptions &LangOpts,
                                            bool UseTextBegin = false) {
  auto LocInfo = SM.getDecomposedLoc(SM.getExpansionLoc(Loc));
  auto Buffer = SM.getBufferData(LocInfo.first);
  auto NLPos = Buffer.find_last_of('\n', LocInfo.second);
  auto Skip = 1;
  while (NLPos && (NLPos != StringRef::npos)) {
    if (Buffer[NLPos - 1] == '\r') {
      --NLPos;
      Skip = 2;
    }

    if (NLPos && (Buffer[NLPos - 1] == '\\'))
      NLPos = Buffer.find_last_of('\n', NLPos - 1);
    else
      break;
  }
  if (NLPos == StringRef::npos) {
    NLPos = 0;
  } else {
    NLPos += Skip;
  }
  auto LineBegin =
      SM.getExpansionLoc(Loc).getLocWithOffset(NLPos - LocInfo.second);

  if (!UseTextBegin) {
    return LineBegin;
  }

  while (isspace(Buffer[NLPos])) {
    if (Buffer[NLPos] == '\n' || Buffer[NLPos] == '\r') {
      break;
    }
    NLPos++;
  }
  auto TextBegin =
      SM.getExpansionLoc(Loc).getLocWithOffset(NLPos - LocInfo.second);
  return TextBegin;
}

bool checkDuplicated(const std::string &FileAndLine,
                     const std::string &WarningIDAndMsg);

template <typename... Ts>
TextModification *insertCommentPrevLine(SourceLocation SL,
                                        const DiagnosticsMessage &Msg,
                                        const SourceManager &SM,
                                        bool UseTextBegin, Ts &&...Vals) {
  auto StartLoc =
      getStartOfLine(SL, SM, LangOptions(), UseTextBegin);
  auto Formatted = llvm::formatv(Msg.Msg, std::forward<Ts>(Vals)...);
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << getMessagePrefix(Msg.ID);
  OS << Formatted;
  return new InsertComment(StartLoc, OS.str(), UseTextBegin);
}

// This function is only used to get warning text for regular expression
// matching. For normal warning emitting, please do not use this interface.
template <typename IDTy, typename... Ts>
std::string getWarningTextWithOutPrefix(IDTy MsgID, Ts &&...Vals) {
  std::string Text;
  if (CommentIDTable.find((int)MsgID) != CommentIDTable.end()) {
    DiagnosticsMessage Msg = CommentIDTable[(int)MsgID];
    auto Formatted = llvm::formatv(Msg.Msg, std::forward<Ts>(Vals)...);
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    OS << Formatted;
    Text = OS.str();
  }
  return Text;
}

template <typename IDTy, typename... Ts>
std::string getWarningText(IDTy MsgID, Ts &&...Vals) {
  std::string Text;
  if (CommentIDTable.find((int)MsgID) != CommentIDTable.end()) {
    DiagnosticsMessage Msg = CommentIDTable[(int)MsgID];
    auto Formatted = llvm::formatv(Msg.Msg, std::forward<Ts>(Vals)...);
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    OS << getMessagePrefix(Msg.ID);
    OS << Formatted;
    Text = OS.str();
  }
  return Text;
}

/// This function is used to get text for the generated makefile,
/// and should only be called by function genMakefile()
template <typename IDTy, typename... Ts>
std::string getMsgText(IDTy MsgID, Ts &&...Vals) {
  std::string Text;
  if (MsgIDTable.find((int)MsgID) != MsgIDTable.end()) {
    DiagnosticsMessage Msg = MsgIDTable[(int)MsgID];
    auto Formatted = llvm::formatv(Msg.Msg, std::forward<Ts>(Vals)...);
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    OS << getMessagePrefix(Msg.ID);
    OS << Formatted;
    Text = OS.str();
    UniqueID++;
  }
  return Text;
}

/// If this function is used to get text to inline warning into replacement,
/// then this function should only be called when the return value of report()
/// is true.
template <typename IDTy, typename... Ts>
std::string getWarningTextAndUpdateUniqueID(IDTy MsgID, Ts &&...Vals) {
  std::string Text = getWarningText(MsgID, std::forward<Ts>(Vals)...);
  UniqueID++;
  return Text;
}

template <typename IDTy, typename... Ts>
std::string getCommentToInsert(SourceLocation StartLoc, SourceManager &SM,
                               IDTy MsgID, bool UseTextBegin, Ts &&...Vals) {
  std::string OrigIndent = getIndent(StartLoc, SM).str();
  std::string Comment;
  if (UseTextBegin)
    Comment = (llvm::Twine("/*") + getNL() + OrigIndent +
               getWarningText(MsgID, Vals...) + getNL() + OrigIndent + "*/" +
               getNL() + OrigIndent)
                  .str();
  else
    Comment =
        (OrigIndent + llvm::Twine("/*") + getNL() + OrigIndent +
         getWarningText(MsgID, Vals...) + getNL() + OrigIndent + "*/" + getNL())
            .str();

  return Comment;
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

template <typename IDTy, typename... Ts>
bool report(const std::string &FileAbsPath, unsigned int Offset, IDTy MsgID,
            bool IsInsertWarningIntoCode, bool UseTextBegin, Ts &&...Vals);

// Emits a warning/error/note and/or comment depending on MsgID. For details
template <typename IDTy, typename... Ts>
inline bool report(SourceLocation SL, IDTy MsgID,
            TransformSetTy *TS, bool UseTextBegin, Ts &&... Vals) {
  if (DpctGlobalInfo::isQueryAPIMapping())
    return true;
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  if (SL.isMacroID() && !SM.isMacroArgExpansion(SL)) {
    auto ItMatch = dpct::DpctGlobalInfo::getMacroTokenToMacroDefineLoc().find(
        getHashStrFromLoc(SM.getImmediateSpellingLoc(SL)));
    if (ItMatch !=
        dpct::DpctGlobalInfo::getMacroTokenToMacroDefineLoc().end()) {
      if (ItMatch->second->IsInAnalysisScope) {
        return DiagnosticsUtils::report<IDTy, Ts...>(
            ItMatch->second->FilePath, ItMatch->second->Offset, MsgID, true,
            UseTextBegin, std::forward<Ts>(Vals)...);
      }
    }
  }

  auto FileName = DpctGlobalInfo::getLocInfo(SL).first;

  // Do not emit diagnostic message for source location outside --in-root
  if (!DpctGlobalInfo::isInRoot(FileName))
    return false;
  std::string FileAndLine = clang::dpct::buildString(
      FileName, ":", SM.getPresumedLineNumber(SL));
  std::string WarningIDAndMsg = clang::dpct::buildString(
      std::to_string(static_cast<int>(MsgID)), ":", Vals...);

  if (checkDuplicated(FileAndLine, WarningIDAndMsg))
    return false;

  if (!SuppressWarningsAllFlag) {
    // Only report warnings that are not suppressed
    if (WarningIDs.find((int)MsgID) == WarningIDs.end() &&
        DiagnosticIDTable.find((int)MsgID) != DiagnosticIDTable.end()) {
      reportWarning(SL, DiagnosticIDTable[(int)MsgID], SM.getDiagnostics(),
                    Vals...);
    }
  }
  if (TS && CommentIDTable.find((int)MsgID) != CommentIDTable.end()) {
    TS->emplace_back(insertCommentPrevLine(SL, CommentIDTable[(int)MsgID], SM,
                                           UseTextBegin, Vals...));
  }
  UniqueID++;
  return true;
}

class SourceManagerForWarning {
public:
  static SourceManagerForWarning &getInstance() {
    static SourceManagerForWarning instance;
    return instance;
  }
  SourceManagerForWarning(SourceManagerForWarning const &) = delete;
  void operator=(SourceManagerForWarning const &) = delete;
  static SourceManager &getSM() { return *(getInstance().SM); }

private:
  SourceManagerForWarning() {
    DiagOpts = new DiagnosticOptions();
    DiagOpts->ShowColors = DpctGlobalInfo::getInstance().getColorOption();
    DiagnosticPrinter = new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);
    Diagnostics = new DiagnosticsEngine(
        IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
        DiagnosticPrinter, false);

    FSO.WorkingDir = ".";
    FM = new FileManager(FSO, nullptr);
    SM = new SourceManager(*Diagnostics, *FM, false);
    DiagnosticPrinter->BeginSourceFile(DefaultLangOptions, nullptr);
  }

  LangOptions DefaultLangOptions;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  TextDiagnosticPrinter *DiagnosticPrinter;
  DiagnosticsEngine *Diagnostics;
  FileSystemOptions FSO;
  FileManager *FM;
  SourceManager *SM;
};

// Emits a warning/error/note and/or comment depending on MsgID. For details
template <typename IDTy, typename... Ts>
bool report(const std::string &FileAbsPath, unsigned int Offset, IDTy MsgID,
            bool IsInsertWarningIntoCode, bool UseTextBegin, Ts &&...Vals) {
  // Do not emit diagnostic message for source location outside --in-root
  if (!DpctGlobalInfo::isInRoot(FileAbsPath))
    return false;
  std::shared_ptr<DpctFileInfo> Fileinfo =
      dpct::DpctGlobalInfo::getInstance().insertFile(FileAbsPath);

  SmallString<4096> NativeFormPath(FileAbsPath);
  // Convert path to the native form.
  // E.g, on Windows all '/' are converted to '\'.
  llvm::sys::path::native(NativeFormPath);

  std::string FileAndLine = clang::dpct::buildString(
      NativeFormPath, ":", Fileinfo->getLineNumber(Offset));
  std::string WarningIDAndMsg = clang::dpct::buildString(
      std::to_string(static_cast<int>(MsgID)), ":", Vals...);

  if (checkDuplicated(FileAndLine, WarningIDAndMsg))
    return false;

  SourceManager &SM = SourceManagerForWarning::getSM();

  llvm::Expected<FileEntryRef> Result =
      SM.getFileManager().getFileRef(NativeFormPath);

  auto E = Result.takeError();
  if (E) {
    return false;
  }

  FileID FID = SM.getOrCreateFileID(Result.get(), SrcMgr::C_User);

  unsigned int LineNum = Fileinfo->getLineNumber(Offset);
  unsigned int ColNum = Offset - Fileinfo->getLineInfo(LineNum).Offset + 1;
  SourceLocation SL = SM.translateLineCol(FID, LineNum, ColNum);

  if (!SuppressWarningsAllFlag) {
    // Only report warnings that are not suppressed
    if (WarningIDs.find((int)MsgID) == WarningIDs.end() &&
        DiagnosticIDTable.find((int)MsgID) != DiagnosticIDTable.end()) {
      reportWarning(SL, DiagnosticIDTable[(int)MsgID], SM.getDiagnostics(),
                    Vals...);
    }
  }

  if (IsInsertWarningIntoCode) {
    auto StartLoc = getStartOfLine(SL, SM, LangOptions(), UseTextBegin);
    std::shared_ptr<ExtReplacement> R = std::make_shared<ExtReplacement>(
        NativeFormPath.str().str(), SM.getDecomposedLoc(StartLoc).second, 0,
        getCommentToInsert(StartLoc, SM, MsgID, UseTextBegin,
                           std::forward<Ts>(Vals)...),
        nullptr);
    if (UseTextBegin)
      R->setInsertPosition(InsertPosition::IP_Right);
    DpctGlobalInfo::getInstance().addReplacement(R);
    UniqueID++;
  }

  return true;
}

} // namespace DiagnosticsUtils
} // namespace dpct
} // namespace clang
#endif // DPCT_AST_DIAGNOSTICS_H
