//===--- LibraryAPIMigration.h ---------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_LIBRARY_API_MIGRAION_H
#define DPCT_LIBRARY_API_MIGRAION_H

#include "ExprAnalysis.h"
#include "MapNames.h"

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInstance.h"

#include <string>
#include <vector>

namespace clang {
namespace dpct {

enum class FFTDirectionType : int {
  Uninitialized = 0,
  Unknown = 1,
  Forward = 2,
  Backward = 3
};
enum class FFTPlacementType : int {
  Uninitialized = 0,
  Unknown = 1,
  Inplace = 2,
  Outofplace = 3
};
enum class FFTTypeEnum : int {
  C2R = 0,
  R2C = 1,
  C2C = 2,
  Z2D = 3,
  D2Z = 4,
  Z2Z = 5,
  Unknown = 6
};

struct FFTSetStreamAPIInfo {
  std::vector<std::pair<unsigned int, std::string>> Streams;
  std::string getLatestStream(unsigned int Offset) {
    std::string Result = "";
    std::sort(Streams.begin(), Streams.end(),
              [](std::pair<unsigned int, std::string> LHS,
                 std::pair<unsigned int, std::string> RHS) {
                return LHS.first < RHS.first;
              });
    for (const auto &Item : Streams) {
      if (Offset < Item.first)
        break;
      Result = Item.second;
    }
    return Result;
  }
};

struct FFTExecAPIInfo;

struct LibraryAPIStmts {
  LibraryAPIStmts() {}

  LibraryAPIStmts &operator<<(const LibraryAPIStmts &InputStmts) {
    S.insert(S.end(), InputStmts.S.begin(), InputStmts.S.end());
    return *this;
  }
  LibraryAPIStmts &operator<<(const std::vector<std::string> &InputStmts) {
    S.insert(S.end(), InputStmts.begin(), InputStmts.end());
    return *this;
  }
  LibraryAPIStmts &operator<<(const std::string &InputStmt) {
    if (InputStmt.empty())
      return *this;

    S.push_back(InputStmt);
    return *this;
  }

  std::string getAsString(std::string IndentStr, bool IsNLAtBegin) {
    std::ostringstream OS;
    for (const auto &Stmt : S) {
      if (IsNLAtBegin)
        OS << getNL() << IndentStr << Stmt; // For suffix string
      else
        OS << Stmt << getNL() << IndentStr; // For prefix string
    }
    return OS.str();
  }
  std::vector<std::string> S;
};

struct FFTHandleInfo {
  FFTDirectionType Direction = FFTDirectionType::Uninitialized;
  FFTPlacementType Placement = FFTPlacementType::Uninitialized;
  // Below 5 members do not cover the case that one handle is resued
  // in different plan APIs. If the related plan API is "many", the flag
  // will be true. The checking of C2C/Z2Z will be done in Exec API
  // migration.
  // Since handle resuing is not covered, these 5 members are just rewrited if
  // they are updated multi times.
  bool MayNeedReset = false;
  std::string InputDistance;
  std::string OutputDistance;
  std::string InembedStr;
  std::string OnembedStr;

  void updateDirectionFromExec(FFTDirectionType NewDirection) {
    if (Direction == FFTDirectionType::Uninitialized) {
      Direction = NewDirection;
      return;
    }
    if (Direction == NewDirection) {
      return;
    }

    // different directions and Direction is initialized
    Direction = FFTDirectionType::Unknown;
    return;
  }
  void updatePlacementFromExec(FFTPlacementType NewPlacement) {
    if (Placement == FFTPlacementType::Uninitialized) {
      Placement = NewPlacement;
      return;
    }
    if (Placement == NewPlacement) {
      return;
    }

    // different placements and Placement is initialized
    Placement = FFTPlacementType::Unknown;
    return;
  }
  void updateResetInfo(bool F, std::string ID, std::string OD, std::string IE,
                       std::string OE) {
    MayNeedReset = F;
    InputDistance = ID;
    OutputDistance = OD;
    InembedStr = IE;
    OnembedStr = OE;
  }
};

struct LibraryMigrationFlags {
  bool NeedUseLambda = false;
  bool CanAvoidUsingLambda = false;
  bool IsMacroArg = false;
  bool CanAvoidBrace = false;
  bool IsAssigned = false;
  bool MoveOutOfMacro = false;
  std::string OriginStmtType;
  bool IsPrefixEmpty = false;
  bool IsSuffixEmpty = false;
  bool IsPrePrefixEmpty = false;
  bool IsFunctionPointer = false;
  bool IsFunctionPointerAssignment = false;
};
struct LibraryMigrationLocations {
  SourceLocation PrefixInsertLoc;
  SourceLocation SuffixInsertLoc;
  SourceLocation OuterInsertLoc;
  SourceLocation FuncNameBegin;
  SourceLocation FuncCallEnd;
  SourceLocation OutOfMacroInsertLoc;
  unsigned int Len = 0;
  SourceLocation FuncPtrDeclBegin;
  SourceLocation FuncPtrDeclHandleTypeBegin;
  unsigned int FuncPtrDeclLen = 0;
};
struct LibraryMigrationStrings {
  std::string PrePrefixInsertStr;
  std::string PrefixInsertStr;
  std::string Repl;
  std::string SuffixInsertStr;
  std::string IndentStr;
};

struct FFTPlanAPIInfo {
  FFTPlanAPIInfo() {}
  void buildInfo();

  // Input info by Plan API
  std::string PrecAndDomainStr;
  FFTTypeEnum FFTType = FFTTypeEnum::Unknown; // C2R,R2C,C2C,D2Z,Z2D,Z2Z
  std::vector<std::string> ArgsList;
  std::vector<std::string> ArgsListAddRequiredParen;
  std::string IndentStr;
  std::string FuncName;
  LibraryMigrationFlags Flags;
  std::int64_t Rank = -1;
  std::string DescrMemberCallPrefix;
  std::string DescStr;
  bool NeedBatchFor1D = true;
  std::string HandleDeclFileAndOffset;

  // Input info by Exec API
  FFTPlacementType PlacementFromExec = FFTPlacementType::Uninitialized;
  FFTDirectionType DirectionFromExec = FFTDirectionType::Uninitialized;

  // Input info by setstream API
  std::string StreamStr;

  // Generated info
  std::string PrePrefixStmt;
  std::vector<std::string> PrefixStmts;
  std::vector<std::string> SuffixStmts;
  std::string CallExprRepl;
  std::string FilePath;
  std::pair<unsigned int, unsigned int> InsertOffsets;
  unsigned int ReplaceOffset = 0;
  unsigned int ReplaceLen = 0;
  std::string UnsupportedArg;

  void updateManyCommitCallExpr();
  void update1D2D3DCommitCallExpr(std::vector<int> DimIdxs);
  std::vector<std::string>
  update1D2D3DCommitPrefix(std::vector<std::string> Dims);
  void updateCommitCallExpr(std::vector<std::string> Dims);
  void addInfo(std::string PrecAndDomainStr, FFTTypeEnum FFTType,
               std::vector<std::string> ArgsList,
               std::vector<std::string> ArgsListAddRequiredParen,
               std::string IndentStr, std::string FuncName,
               LibraryMigrationFlags Flags, std::int64_t Rank,
               std::string DescrMemberCallPrefix, std::string DescStr);
  void setValueFor1DBatched();
  std::vector<std::string>
  setValueForBasicManyBatched(std::vector<std::string> Dims,
                              std::vector<std::string> DimsWithoutParen);
  void linkInfo();
};

class FFTFunctionCallBuilder {
public:
  FFTFunctionCallBuilder(const clang::CallExpr *TheCallExpr,
                         std::string IndentStr, std::string FuncName,
                         StringRef FuncPtrName,
                         LibraryMigrationLocations Locations,
                         LibraryMigrationFlags Flags)
      : TheCallExpr(TheCallExpr), IndentStr(IndentStr), FuncName(FuncName),
        FuncPtrName(FuncPtrName), Locations(Locations), Flags(Flags) {

    if (Flags.IsFunctionPointer || Flags.IsFunctionPointerAssignment) {
      // Currently, only support function pointer of exec API
      ArgsList.emplace_back("desc");
      ArgsList.emplace_back("in_data");
      ArgsList.emplace_back("out_data");
      if (FuncName[9] == FuncName[11])
        ArgsList.emplace_back("dir");
    } else {
      for (size_t i = 0; i < TheCallExpr->getNumArgs(); ++i) {
        ArgsList.emplace_back(ExprAnalysis::ref(TheCallExpr->getArg(i)));
        ArgsListAddRequiredParen.emplace_back(
            ExprAnalysis::ref(TheCallExpr->getArg(i)));
      }

      auto I = MapNames::FFTPlanAPINeedParenIdxMap.find(FuncName);
      if (I == MapNames::FFTPlanAPINeedParenIdxMap.end())
        return;

      std::vector<unsigned int> NeedParenIdxs = I->second;
      for (const auto &Idx : NeedParenIdxs) {
        const Expr *E = TheCallExpr->getArg(Idx);
        if (!E)
          continue;
        if (!dyn_cast<DeclRefExpr>(E->IgnoreImplicit()) &&
            !dyn_cast<ParenExpr>(E->IgnoreImplicit()) &&
            !dyn_cast<IntegerLiteral>(E->IgnoreImplicit()) &&
            !dyn_cast<FloatingLiteral>(E->IgnoreImplicit()) &&
            !dyn_cast<FixedPointLiteral>(E->IgnoreImplicit())) {
          ArgsListAddRequiredParen[Idx] =
              "(" + ArgsListAddRequiredParen[Idx] + ")";
        }
      }
    }
  }

  std::string getPrefixString();
  std::string getSuffixString();
  std::string getCallExprReplString();
  std::string getPrePrefixString();
  bool moveDeclOutOfBracesIfNeeds(const LibraryMigrationFlags Flags,
                                  SourceLocation &TypeBegin, int &TypeLength);
  void updateFFTPlanAPIInfo(FFTPlanAPIInfo &FPAInfo,
                            LibraryMigrationFlags &Flags);
  void updateExecCallExpr(std::string FFTHandleInfoKey);
  void updateExecCallExpr();
  void updateFFTExecAPIInfo(FFTExecAPIInfo &FEAInfo);
  void updateFFTHandleInfoFromPlan(std::string FFTHandleInfoKey);

private:
  void updateBufferArgs(unsigned int Idx, const std::string &TypeStr,
                        std::string PointerName = "");
  bool isInplace(const Expr *Ptr1, const Expr *Ptr2);
  void addDescriptorTypeInfo(std::string PrecAndDomain);
  std::string getDescrMemberCallPrefix();
  std::string getDescr();
  FFTTypeEnum getFFTType(unsigned int PrecDomainIdx);
  std::string getPrecAndDomainStr(unsigned int PrecDomainIdx);
  void assembleExecCallExpr();

  const clang::CallExpr *TheCallExpr;
  std::string PrePrefixStmt;
  std::vector<std::string> PrefixStmts;
  std::vector<std::string> SuffixStmts;
  std::vector<std::string> ArgsList;
  std::vector<std::string> ArgsListAddRequiredParen;
  std::string IndentStr;
  std::string FuncName;
  std::string FuncPtrName;
  std::string CallExprRepl;
  LibraryMigrationLocations Locations;
  LibraryMigrationFlags Flags;
  int64_t Dir = 0;
};

void initVars(const CallExpr *CE, const VarDecl *VD, const BinaryOperator *BO,
              LibraryMigrationFlags &Flags,
              LibraryMigrationStrings &ReplaceStrs,
              LibraryMigrationLocations &Locations);
void replacementLocation(const LibraryMigrationLocations Locations,
                         const LibraryMigrationFlags Flags,
                         unsigned int &ReplaceOffset, unsigned int &ReplaceLen,
                         std::pair<unsigned int, unsigned int> &InsertOffsets,
                         std::string &FilePath);
void replacementText(LibraryMigrationFlags Flags,
                     const std::string PrePrefixStmt,
                     const std::vector<std::string> PrefixStmts,
                     const std::vector<std::string> SuffixStmts,
                     std::string CallExprRepl, const std::string IndentStr,
                     const std::string FilePath,
                     const unsigned int ReplaceOffset,
                     const unsigned int ReplaceLen,
                     const std::pair<unsigned int, unsigned int> InsertOffsets);

struct FFTDescriptorTypeInfo {
  FFTDescriptorTypeInfo(unsigned int Length) : Length(Length) {}
  void buildInfo(std::string FilePath, unsigned int Offset);

  unsigned int Length;
  std::string PrecAndDom;
  bool IsValid = true;
  // E.g., if cufftExec API is declared as a function pointer, then all
  // declaration will be rewrite to a lambda, the paramter type can be deduced
  // from function name. So this type replacement and warning can be skipped.
  bool SkipGeneration = false;
};

struct FFTExecAPIInfo {

  FFTExecAPIInfo() {}
  void buildInfo();
  void linkInfo();
  void addInfo(std::string IndentStrInput, LibraryMigrationFlags FlagsInput,
               std::string PrePrefixStmtInput,
               std::vector<std::string> PrefixStmtsInput,
               std::vector<std::string> SuffixStmtsInput,
               std::string CallExprReplInput, bool IsComplexDomainInput,
               std::string DescStrInput, std::int64_t DirInput);
  void updateResetAndCommitStmts();

  // Input info by Exec API
  LibraryMigrationFlags Flags;
  std::string PrePrefixStmt;
  std::vector<std::string> PrefixStmts;
  std::vector<std::string> SuffixStmts;
  std::string CallExprRepl;
  std::string IndentStr;
  std::string FilePath;
  std::pair<unsigned int, unsigned int> InsertOffsets;
  unsigned int ReplaceOffset;
  unsigned int ReplaceLen;
  bool IsComplexDomain = false;
  std::string DescStr;
  std::int64_t Dir;
  std::string HandleDeclFileAndOffset;
  int QueueIndex = -1;
  unsigned int CompoundStmtBeginOffset = 0;
  unsigned int PlanHandleDeclBeginOffset = 0;
  unsigned int ExecAPIBeginOffset = 0;
  std::string DefiniteStream = "";

  // Input info by Plan API (from handle info)
  std::string InputDistance;
  std::string OutputDistance;
  std::string InembedStr;
  std::string OnembedStr;

  // Input info by setstream API
  std::string StreamStr;

  // generate info
  // code is like:
  // if (inembed != null && onembed != null) {
  // desc->set_value(FWD_DISTANCE, ...);
  // desc->set_value(BWD_DISTANCE, ...);
  // desc->commit(q);
  // }
  // TODO: Here maybe need a warning like:
  // inembed/onembed/input_dis/output_dis may be not accessable here, you may
  // need manully fix the code.
  std::vector<std::string> ResetAndCommitStmts; // These stmts should be added
                                                // at the begin of PrefixStmts
  bool NeedReset =
      false; // MayNeedReset(from handle info) && IsComplexDomain
             // && !IsFunctionPointer && !IsFunctionPointerAssignment
             // && (handle->dir == unknown || handle->dir == uninitialized)
};

bool isPreviousStmtRelatedSetStream(const CallExpr *ExecCall, int Index,
                                    std::string &StreamStr);

} // namespace dpct
} // namespace clang

#endif // !DPCT_LIBRARY_API_MIGRAION_H
