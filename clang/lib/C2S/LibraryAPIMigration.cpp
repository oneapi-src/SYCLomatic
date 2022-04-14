//===--- LibraryAPIMigration.cpp -------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "LibraryAPIMigration.h"
#include "AnalysisInfo.h"
#include "Diagnostics.h"
#include "MapNames.h"
#include "Utility.h"

#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LangOptions.h"

namespace clang {
namespace c2s {

void initVars(const CallExpr *CE, const VarDecl *VD, const BinaryOperator *BO,
              LibraryMigrationFlags &Flags,
              LibraryMigrationStrings &ReplaceStrs,
              LibraryMigrationLocations &Locations) {
  auto &SM = C2SGlobalInfo::getSourceManager();
  SourceLocation FuncPtrDeclEnd;
  if (Flags.IsFunctionPointer || Flags.IsFunctionPointerAssignment) {
    if (Flags.IsFunctionPointer) {
      Locations.FuncPtrDeclBegin = SM.getExpansionLoc(VD->getBeginLoc());
      FuncPtrDeclEnd = SM.getExpansionLoc(VD->getEndLoc());
    } else if (Flags.IsFunctionPointerAssignment) {
      Locations.FuncPtrDeclBegin =
          SM.getExpansionLoc(BO->getRHS()->getBeginLoc());
      FuncPtrDeclEnd = SM.getExpansionLoc(BO->getRHS()->getEndLoc());
    }
    FuncPtrDeclEnd = FuncPtrDeclEnd.getLocWithOffset(Lexer::MeasureTokenLength(
        FuncPtrDeclEnd, SM, C2SGlobalInfo::getContext().getLangOpts()));
    Locations.FuncPtrDeclLen =
        SM.getDecomposedLoc(FuncPtrDeclEnd).second -
        SM.getDecomposedLoc(Locations.FuncPtrDeclBegin).second;
    ReplaceStrs.IndentStr = getIndent(Locations.FuncPtrDeclBegin, SM).str();

    if (Flags.IsFunctionPointerAssignment)
      return;

    TypeLoc TL = VD->getTypeSourceInfo()->getTypeLoc();
    QualType QT = VD->getType();

    // For typedef/using case like:
    // typedef cufftResult (*Func_t)(cufftHandle, cufftDoubleComplex *, double
    // *); Func_t ptr2cufftExec = &cufftExecZ2D; We need the "cufftHandle" in
    // the typedef be migrated, so here just return. And the SkipGeneration of
    // the FFTDescriptorTypeInfo whose SourceLocation ID is 0 will be set.
    // Otherwise, an assert will be hitten.
    if (QT->getAs<TypedefType>())
      return;

    if (QT->isPointerType()) {
      QT = QT->getPointeeType();
      TL = TL.getAs<PointerTypeLoc>().getPointeeLoc();
    } else {
      Locations.FuncPtrDeclHandleTypeBegin = Locations.FuncPtrDeclBegin;
      return;
    }

    if (QT->getAs<ParenType>()) {
      QT = QT->getAs<ParenType>()->getInnerType();
      TL = TL.getAs<ParenTypeLoc>().getInnerLoc();
    } else {
      Locations.FuncPtrDeclHandleTypeBegin = Locations.FuncPtrDeclBegin;
      return;
    }

    if (QT->getAs<FunctionProtoType>()) {
      TL = TL.getAs<FunctionProtoTypeLoc>()
               .getParam(0)
               ->getTypeSourceInfo()
               ->getTypeLoc();
      Locations.FuncPtrDeclHandleTypeBegin =
          SM.getExpansionLoc(TL.getBeginLoc());
    } else {
      Locations.FuncPtrDeclHandleTypeBegin = Locations.FuncPtrDeclBegin;
      return;
    }
    return;
  }

  Locations.FuncNameBegin = CE->getBeginLoc();
  Locations.FuncCallEnd = CE->getEndLoc();

  Locations.OutOfMacroInsertLoc = SM.getExpansionLoc(CE->getBeginLoc());

  // TODO: For case like:
  //  #define CHECK_STATUS(x) fun(c)
  //  CHECK_STATUS(anAPICall());
  // Below code can distinguish this kind of function like macro, need refine to
  // cover more cases.
  Flags.IsMacroArg = SM.isMacroArgExpansion(CE->getBeginLoc()) &&
                     SM.isMacroArgExpansion(CE->getEndLoc());

  // Offset 1 is the length of the last token ")"
  Locations.FuncCallEnd =
      SM.getExpansionLoc(Locations.FuncCallEnd).getLocWithOffset(1);
  auto SR =
      getScopeInsertRange(CE, Locations.FuncNameBegin, Locations.FuncCallEnd);
  Locations.PrefixInsertLoc = SR.getBegin();
  Locations.SuffixInsertLoc = SR.getEnd();

  Flags.CanAvoidUsingLambda = false;
  Flags.NeedUseLambda = isConditionOfFlowControl(CE, Flags.OriginStmtType,
                                                 Flags.CanAvoidUsingLambda,
                                                 Locations.OuterInsertLoc);
  bool IsInReturnStmt = isInReturnStmt(CE, Locations.OuterInsertLoc);
  Flags.CanAvoidBrace = false;
  const CompoundStmt *CS = findImmediateBlock(CE);
  if (CS && (CS->size() == 1)) {
    const Stmt *S = *(CS->child_begin());
    if (CE == S || dyn_cast<ReturnStmt>(S))
      Flags.CanAvoidBrace = true;
  }

  if (Flags.NeedUseLambda || Flags.IsMacroArg || IsInReturnStmt) {
    Flags.NeedUseLambda = true;
    SourceRange SR = getFunctionRange(CE);
    Locations.PrefixInsertLoc = SR.getBegin();
    Locations.SuffixInsertLoc = SR.getEnd();
    if (IsInReturnStmt) {
      Flags.OriginStmtType = "return";
      Flags.CanAvoidUsingLambda = true;
    }
  }

  ReplaceStrs.IndentStr = getIndent(Locations.PrefixInsertLoc, SM).str();

  // This length should be used only when NeedUseLambda is true.
  // If NeedUseLambda is false, Len may longer than the function call length,
  // because in this case, PrefixInsertLoc and SuffixInsertLoc are the begin
  // location of the whole statement and the location after the semi of the
  // statement.
  Locations.Len = SM.getDecomposedLoc(Locations.SuffixInsertLoc).second -
                  SM.getDecomposedLoc(Locations.PrefixInsertLoc).second;
}

// Need be called in asttraversal and add the location info to FileInfo
void replacementLocation(const LibraryMigrationLocations Locations,
                         const LibraryMigrationFlags Flags,
                         unsigned int &ReplaceOffset, unsigned int &ReplaceLen,
                         std::pair<unsigned int, unsigned int> &InsertOffsets,
                         std::string &FilePath) {
  if (Flags.IsFunctionPointer || Flags.IsFunctionPointerAssignment) {
    ReplaceOffset =
        C2SGlobalInfo::getLocInfo(Locations.FuncPtrDeclBegin).second;
    ReplaceLen = Locations.FuncPtrDeclLen;
    return;
  }

  SourceRange InsertLocations;
  SourceLocation ReplaceLocation;
  if (Flags.NeedUseLambda) {
    if ((Flags.MoveOutOfMacro && Flags.IsMacroArg) ||
        (Flags.CanAvoidUsingLambda && !Flags.IsMacroArg)) {
      if (Flags.MoveOutOfMacro && Flags.IsMacroArg) {
        InsertLocations = SourceRange(Locations.OutOfMacroInsertLoc,
                                      Locations.OutOfMacroInsertLoc);
      } else {
        InsertLocations =
            SourceRange(Locations.OuterInsertLoc, Locations.OuterInsertLoc);
      }
    } else {
      InsertLocations =
          SourceRange(Locations.PrefixInsertLoc, Locations.SuffixInsertLoc);
    }
    ReplaceLocation = Locations.PrefixInsertLoc;
    ReplaceLen = Locations.Len;
  } else {
    InsertLocations =
        SourceRange(Locations.PrefixInsertLoc, Locations.SuffixInsertLoc);

    auto &SM = C2SGlobalInfo::getSourceManager();
    ReplaceLocation = Locations.FuncNameBegin;
    ReplaceLen = SM.getDecomposedLoc(Locations.FuncCallEnd).second -
                 SM.getDecomposedLoc(Locations.FuncNameBegin).second;
  }

  // Assumption: these locations are in the same file
  InsertOffsets.first =
      C2SGlobalInfo::getLocInfo(InsertLocations.getBegin()).second;
  InsertOffsets.second =
      C2SGlobalInfo::getLocInfo(InsertLocations.getEnd()).second;
  ReplaceOffset = C2SGlobalInfo::getLocInfo(ReplaceLocation).second;
  FilePath = C2SGlobalInfo::getLocInfo(ReplaceLocation).first;
}

// Need be called in the buildInfo()
void replacementText(
    LibraryMigrationFlags Flags, const std::string PrePrefixStmt,
    const std::vector<std::string> PrefixStmts,
    const std::vector<std::string> SuffixStmts, std::string CallExprRepl,
    const std::string IndentStr, const std::string FilePath,
    const unsigned int ReplaceOffset, const unsigned int ReplaceLen,
    const std::pair<unsigned int, unsigned int> InsertOffsets) {
  LibraryAPIStmts OutPrefixStmts;
  LibraryAPIStmts OutSuffixStmts;
  std::string OutRepl;

  if (Flags.IsFunctionPointer || Flags.IsFunctionPointerAssignment) {
    OutPrefixStmts << PrefixStmts;
    OutRepl = CallExprRepl + ";";
    OutSuffixStmts << SuffixStmts;

    std::string Text = OutPrefixStmts.getAsString(IndentStr, false) + OutRepl +
                       OutSuffixStmts.getAsString(IndentStr, true);

    auto TextRepl = std::make_shared<ExtReplacement>(FilePath, ReplaceOffset,
                                                     ReplaceLen, Text, nullptr);

    TextRepl->setBlockLevelFormatFlag();
    C2SGlobalInfo::getInstance().addReplacement(TextRepl);
    return;
  }

  if (Flags.NeedUseLambda) {
    if (Flags.IsPrefixEmpty && Flags.IsSuffixEmpty) {
      // If there is one API call in the migrted code, it is unnecessary to
      // use a lambda expression
      Flags.NeedUseLambda = false;
    }
  }

  if (Flags.NeedUseLambda) {
    if ((Flags.MoveOutOfMacro && Flags.IsMacroArg) ||
        (Flags.CanAvoidUsingLambda && !Flags.IsMacroArg)) {
      std::string InsertString;
      if (C2SGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
          !Flags.CanAvoidBrace) {
        OutPrefixStmts << "{" << PrefixStmts;

        OutSuffixStmts << SuffixStmts << "}";
      } else {
        OutPrefixStmts << PrefixStmts;
        OutSuffixStmts << SuffixStmts;
      }
      CallExprRepl =
          CallExprRepl + ";"; // Note: Insert case, need use this Repl also.

      if (Flags.MoveOutOfMacro && Flags.IsMacroArg) {
        DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                                 Diagnostics::CODE_LOGIC_CHANGED, true, true,
                                 "function-like macro");
      } else {
        DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                                 Diagnostics::CODE_LOGIC_CHANGED, true, true,
                                 Flags.OriginStmtType == "if"
                                     ? "an " + Flags.OriginStmtType
                                     : "a " + Flags.OriginStmtType);
      }
      OutRepl = "0";
    } else {
      if (Flags.IsAssigned) {
        DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                                 Diagnostics::NOERROR_RETURN_LAMBDA, true,
                                 false);
        OutPrefixStmts << "[&](){" << PrefixStmts;
        OutSuffixStmts << SuffixStmts << "return 0;"
                       << "}()";
      } else {
        OutPrefixStmts << "[&](){" << PrefixStmts;
        OutSuffixStmts << SuffixStmts << "}()";
      }
      OutRepl = CallExprRepl + ";";
    }
  } else {
    if (C2SGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
        !Flags.CanAvoidBrace) {
      if (!Flags.IsPrefixEmpty || !Flags.IsSuffixEmpty) {
        OutPrefixStmts << PrePrefixStmt << "{" << PrefixStmts;
        OutSuffixStmts << SuffixStmts << "}";
      }
    } else {
      OutPrefixStmts << PrePrefixStmt << PrefixStmts;
      OutSuffixStmts << SuffixStmts;
    }
    if (Flags.IsAssigned) {
      OutRepl = "(" + CallExprRepl + ", 0)";
      DiagnosticsUtils::report(FilePath, InsertOffsets.first,
                               Diagnostics::NOERROR_RETURN_COMMA_OP, true,
                               true);
    } else {
      OutRepl = CallExprRepl;
    }
  }

  if (InsertOffsets.first == InsertOffsets.second) {
    auto InsertRepl = std::make_shared<ExtReplacement>(
        FilePath, InsertOffsets.first, 0,
        OutPrefixStmts.getAsString(IndentStr, false) + CallExprRepl +
            OutSuffixStmts.getAsString(IndentStr, true) + getNL() + IndentStr,
        nullptr);
    InsertRepl->setBlockLevelFormatFlag(true);
    C2SGlobalInfo::getInstance().addReplacement(InsertRepl);

    C2SGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(FilePath, ReplaceOffset, ReplaceLen,
                                         OutRepl, nullptr));
  } else {
    auto InsertBeforeRepl = std::make_shared<ExtReplacement>(
        FilePath, InsertOffsets.first, 0,
        OutPrefixStmts.getAsString(IndentStr, false), nullptr);
    auto ReplaceRepl = std::make_shared<ExtReplacement>(
        FilePath, ReplaceOffset, ReplaceLen, OutRepl, nullptr);
    auto InsertAfterRepl = std::make_shared<ExtReplacement>(
        FilePath, InsertOffsets.second, 0,
        OutSuffixStmts.getAsString(IndentStr, true), nullptr);

    InsertBeforeRepl->setBlockLevelFormatFlag(true);
    ReplaceRepl->setBlockLevelFormatFlag(true);
    InsertAfterRepl->setBlockLevelFormatFlag(true);

    C2SGlobalInfo::getInstance().addReplacement(InsertBeforeRepl);
    C2SGlobalInfo::getInstance().addReplacement(ReplaceRepl);
    C2SGlobalInfo::getInstance().addReplacement(InsertAfterRepl);
  }
}

} // namespace c2s
} // namespace clang
