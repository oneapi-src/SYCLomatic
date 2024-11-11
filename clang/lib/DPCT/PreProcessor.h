//===--------------------------- PreProcessor.h ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_PRE_PROCESSOR_H
#define DPCT_PRE_PROCESSOR_H

#include "AnalysisInfo.h"
#include "TextModification.h"
#include "Utility.h"
#include "clang/Basic/SourceLocation.h"
#include <unordered_set>

namespace clang {
namespace dpct {

/// Migration rules at the pre-processing stages, e.g. macro rewriting and
/// including directives rewriting.
class IncludesCallbacks : public PPCallbacks {
  TransformSetTy &TransformSet;
  SourceManager &SM;
  RuleGroups &Groups;

  std::unordered_set<std::string> SeenFiles;
  bool IsFileInCmd = true;

public:
  IncludesCallbacks(TransformSetTy &TransformSet, SourceManager &SM,
                    RuleGroups &G)
      : TransformSet(TransformSet), SM(SM), Groups(G) {}
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *SuggestedModule,
                          bool ModuleImported,
                          SrcMgr::CharacteristicKind FileType) override;
  /// Hook called whenever a macro definition is seen.
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override;
  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override;
  void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
             const MacroDefinition &MD) override;
  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override;
  // TODO: implement one of this for each source language.
  bool ReplaceCuMacro(const Token &MacroNameTok, MacroInfo *MI = nullptr);
  void ReplaceCuMacro(SourceRange ConditionRange, IfType IT,
                      SourceLocation IfLoc, SourceLocation ElifLoc);
  void Defined(const Token &MacroNameTok, const MacroDefinition &MD,
               SourceRange Range) override;
  void Endif(SourceLocation Loc, SourceLocation IfLoc) override;
  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID = FileID()) override;
  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind ConditionValue) override;
  void Else(SourceLocation Loc, SourceLocation IfLoc) override;
  void Elif(SourceLocation Loc, SourceRange ConditionRange,
            ConditionValueKind ConditionValue, SourceLocation IfLoc) override;
  bool ShouldEnter(StringRef FileName, bool IsAngled) override;
  bool isInAnalysisScope(SourceLocation Loc);
  // Find the "#" before a preprocessing directive, return -1 if have some false
  int findPoundSign(SourceLocation DirectiveStart);
  void insertCudaArchRepl(std::shared_ptr<clang::dpct::ExtReplacement> Repl);

private:
  /// e.g. "__launch_bounds(32, 32)  void foo()"
  /// Result is "void foo()"
  std::shared_ptr<TextModification>
  removeMacroInvocationAndTrailingSpaces(SourceRange Range);
};

} // namespace dpct
} // namespace clang




#endif // DPCT_PRE_PROCESSOR_H