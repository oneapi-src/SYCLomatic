//===--------------- ExtReplacements.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExtReplacements.h"

#include "AnalysisInfo.h"
#include "Statics.h"

#include <assert.h>
#include <regex>

namespace clang {
namespace dpct {
ExtReplacements::ExtReplacements(std::string FilePath) : FilePath(FilePath) {}

bool ExtReplacements::isInvalid(std::shared_ptr<ExtReplacement> Repl,
                                std::shared_ptr<DpctFileInfo> FileInfo) {
  if (!Repl)
    return true;
  if (Repl->getFilePath().empty() || Repl->getFilePath() != FilePath)
    return true;
  if (Repl->getLength() == 0 && Repl->getReplacementText().empty())
    return true;
  auto EndOffset = Repl->getOffset() + Repl->getLength();
  if (EndOffset < Repl->getOffset() || EndOffset > FileInfo->getFileSize()) {
#ifdef DPCT_DEBUG_BUILD
    llvm::errs() << "Abandon Illegal Replacement:\n"
                 << Repl->toString() << "\n";
    assert(0 && "Find illegal replacement!");
#endif
    return true;
  }
  return isReplRedundant(Repl, FileInfo);
}
bool ExtReplacements::isReplRedundant(std::shared_ptr<ExtReplacement> Repl,
                                      std::shared_ptr<DpctFileInfo> FileInfo) {
  std::string &FileContent = FileInfo->getFileContent();
  if (FileContent.empty())
    return true;
  size_t Len = Repl->getLength();
  size_t Offset = Repl->getOffset();
  auto RepText = Repl->getReplacementText();
  if (Len != RepText.size())
    return false;
  return FileContent.substr(Offset, Len) == RepText;
}
/// Do merge for Short replacement and Longer replacement.
///
/// Return the merged replacement.
/// Prerequisite: Shorter replacement's length should be not more than Longer
/// replacement's.
std::shared_ptr<ExtReplacement> ExtReplacements::mergeComparedAtSameOffset(
    std::shared_ptr<ExtReplacement> Shorter,
    std::shared_ptr<ExtReplacement> Longer) {
  if (Shorter->getLength() == Longer->getLength()) {
    if (Shorter->getLength() && Shorter->equal(Longer)) {
      // Fully equal replacements which are not insert, just reserve one.
      Longer->mergeConstantInfo(Shorter);
      return Longer;
    } else if (Shorter->getLength()) {
      // Both Shorter and Longer are replace, just reserve the longer replace.
      if (Shorter->getReplacementText().str().length() <
          Longer->getReplacementText().str().length()) {
        Longer->mergeConstantInfo(Shorter);
        return Longer;
      } else {
        Shorter->mergeConstantInfo(Longer);
        return Shorter;
      }
    }
    // Both Shorter and Longer are insert, do merge.
    // inset replacement could be "namespace::", "(type cast)",  ")"  "(".
    return (Longer->getInsertPosition() <= Shorter->getInsertPosition())
               ? mergeReplacement(Longer, Shorter)
               : mergeReplacement(Shorter, Longer);
  } else if (!Shorter->getLength()) {
    // Shorter is insert replacement, Longer is code replacement, do merge.
    return mergeReplacement(Shorter, Longer);
  } else {
    // Shorter is a sub replacement of Longer, just reserve Longer.
    Longer->mergeConstantInfo(Shorter);
    return Longer;
  }
}

void ExtReplacements::removeCommentsInSrcCode(StringRef SrcCode,
                                              std::string &Result,
                                              bool &BlockComment) {
  StringRef Uncommented(Result);
  size_t Pos = 0, PrevPos = 0;
  bool FindResult /*current loop is process finding result*/ = false,
                  LineComment = false;
  while (Pos != std::string::npos) {
    if (BlockComment) {
      // current in block comments
      if (FindResult) {
        // have find "*/".
        BlockComment = false;
        FindResult = false;
        PrevPos = Pos += 2;
      } else {
        // haven't find "*/", to find it.
        Pos = SrcCode.find("*/", Pos);
        FindResult = true;
      }
      // block comment finished.
    } else if (FindResult) {
      // encount '/', check the next character.
      ++Pos;
      FindResult = false;
      if (SrcCode[Pos] == '/') {
        // encount "//", line comment.
        Result.append(SrcCode.data(), PrevPos, Pos - PrevPos - 1);
        LineComment = true;
        break;
      } else if (SrcCode[Pos] == '*') {
        // encount "/*", block comment.
        Result.append(SrcCode.data(), PrevPos, Pos - PrevPos - 1);
        BlockComment = true;
      }
      // else nothing to do.
    } else {
      // find next '/'
      Pos = SrcCode.find('/', Pos);
      FindResult = true;
    }
  }
  if (LineComment || BlockComment) {
    Result += getNL();
  } else
    Result.append(SrcCode.data() + PrevPos, SrcCode.end());
}

size_t ExtReplacements::findCR(StringRef Line) {
  auto Pos = Line.rfind('\n');
  if (Pos && Pos != std::string::npos) {
    if (Line[Pos - 1] == '\r')
      return --Pos;
  }
  return Pos;
}

bool ExtReplacements::isEndWithSlash(unsigned LineNumber,
                                     std::shared_ptr<DpctFileInfo> FileInfo) {
  if (!LineNumber)
    return false;
  auto Line = FileInfo->getLineString(LineNumber);
  auto CRPos = findCR(Line);
  if (!CRPos || CRPos == std::string::npos)
    return false;
  return Line[--CRPos] == '\\';
}

std::shared_ptr<ExtReplacement> ExtReplacements::buildOriginCodeReplacement(
    const SourceLineRange &LineRange, std::shared_ptr<DpctFileInfo> FileInfo) {
  if (!LineRange.SrcBeginLine)
    return std::shared_ptr<ExtReplacement>();
  std::string Text = "/* DPCT_ORIG ";
  bool BlockComment = false;
  for (unsigned Line = LineRange.SrcBeginLine; Line <= LineRange.SrcEndLine;
       ++Line)
    removeCommentsInSrcCode(FileInfo->getLineString(Line), Text, BlockComment);

  std::string Suffix = std::string(
      isEndWithSlash(LineRange.SrcBeginLine - 1, FileInfo) ? "*/ \\" : "*/");
  Text.insert(findCR(Text), Suffix);
  auto R = std ::make_shared<ExtReplacement>(FilePath, LineRange.SrcBeginOffset,
                                             0, std::move(Text), nullptr);
  R->setInsertPosition(IP_AlwaysLeft);
  return R;
}

// This function scans all the code replacements to collect LOC migrated to
// SYCL, LOC migrated to helper functions. While the old solution only
// scans replacements generated by AST matcher rules that are defined in the
// file MigrationRules.inc, and it ignores the code replacements generated by
// CallExprRewriter module, so the old migration status report is inaccurate.
void ExtReplacements ::getLOCStaticFromCodeRepls(
    std::shared_ptr<DpctFileInfo> FileInfo) {

  SourceLineRange ReplLineRange;
  for (auto &R : ReplMap) {
    auto &Repl = R.second;
    if (Repl->getLength()) {
      std::string FileName = Repl->getFilePath().str();
      FileInfo->setLineRange(ReplLineRange, Repl);
      std::string Key = FileName + ":" +
                        std::to_string(ReplLineRange.SrcBeginLine) + ":" +
                        std::to_string(ReplLineRange.SrcEndLine);
      if (DuplicateFilter.find(Key) == end(DuplicateFilter)) {
        DuplicateFilter.insert(Key);
        unsigned int Lines =
            ReplLineRange.SrcEndLine - ReplLineRange.SrcBeginLine + 1;
        if (Repl->getReplacementText().find(std::string("dpct::")) !=
            StringRef::npos) {
          LOCStaticsMap[FileName][0] += Lines;
        } else {
          LOCStaticsMap[FileName][1] += Lines;
        }
      }
    }

    if (DpctGlobalInfo::isDPCTNamespaceTempEnabled()) {
      // When option "--report-type=stats" or option " --report-type=all" is
      // specified to get the migration status report and dpct namespace is
      // enabled temporarily to get LOC migrated to helper functions,
      // dpct namespace should be removed in the replacements generated.
      std::string ReplText = Repl->getReplacementText().str();
      auto Pos = ReplText.find("dpct::");
      if (Pos != std::string::npos) {
        ReplText.erase(Pos, strlen("dpct::"));
        Repl->setReplacementText(ReplText);
      }
    }
  }
}

void ExtReplacements::buildOriginCodeReplacements(
    std::shared_ptr<DpctFileInfo> FileInfo) {
  SourceLineRange LineRange, ReplLineRange;
  for (auto &R : ReplMap) {
    auto &Repl = R.second;
    if (Repl->getLength()) {
      FileInfo->setLineRange(ReplLineRange, Repl);
      if (LineRange.SrcEndLine < ReplLineRange.SrcBeginLine) {
        addReplacement(buildOriginCodeReplacement(LineRange, FileInfo));
        LineRange = ReplLineRange;
      } else
        LineRange.SrcEndLine =
            std::max(LineRange.SrcEndLine, ReplLineRange.SrcEndLine);
    }
  }
  if (LineRange.SrcBeginLine)
    addReplacement(buildOriginCodeReplacement(LineRange, FileInfo));
}

std::vector<std::shared_ptr<ExtReplacement>>
ExtReplacements::mergeReplsAtSameOffset() {
  std::vector<std::shared_ptr<ExtReplacement>> ReplsList;
  std::shared_ptr<ExtReplacement> Insert, InsertLeft, InsertRight, Replace;
  unsigned Offset = ReplMap.begin()->first;
  for (auto &R : ReplMap) {
    if (R.first != Offset) {
      Offset = R.first;
      ReplsList.emplace_back(mergeAtSameOffset(
          mergeAtSameOffset(InsertLeft, mergeAtSameOffset(Insert, InsertRight)),
          Replace));
      InsertLeft.reset();
      InsertRight.reset();
      Insert.reset();
      Replace.reset();
    }
    auto &Repl = R.second;
    if (Repl->getLength()) {
      Replace = mergeAtSameOffset(Replace, Repl);
    } else if (Repl->getInsertPosition() == InsertPosition::IP_AlwaysLeft) {
      InsertLeft = mergeAtSameOffset(InsertLeft, Repl);
    } else if (Repl->getInsertPosition() == InsertPosition::IP_Right) {
      InsertRight = mergeAtSameOffset(InsertRight, Repl);
    } else {
      Insert = mergeAtSameOffset(Insert, Repl);
    }
  }
  if (Insert || Replace) {
    ReplsList.emplace_back(mergeAtSameOffset(
        mergeAtSameOffset(InsertLeft, mergeAtSameOffset(Insert, InsertRight)),
        Replace));
  }
  return ReplsList;
}

std::shared_ptr<ExtReplacement> ExtReplacements::filterOverlappedReplacement(
    std::shared_ptr<ExtReplacement> Repl, unsigned &PrevEnd) {
  auto ReplEnd = Repl->getOffset() + Repl->getLength();
  if (PrevEnd > ReplEnd)
    return std::shared_ptr<ExtReplacement>();
  if (PrevEnd == ReplEnd && Repl->getLength())
    return std::shared_ptr<ExtReplacement>();
  if ((Repl->getOffset() < PrevEnd) && !Repl->getReplacementText().empty()) {
    llvm::dbgs() << "Replacement Conflict.\nAbandon replacement: "
                 << Repl->toString() << "\n";
    return std::shared_ptr<ExtReplacement>();
  }

  PrevEnd = ReplEnd;
  return Repl;
}

void ExtReplacements::markAsAlive(std::shared_ptr<ExtReplacement> Repl) {
  ReplMap.insert(std::make_pair(Repl->getOffset(), Repl));
  if (auto PairID = Repl->getPairID()) {
    if (auto &R = PairReplsMap[PairID]) {
      if (R->Status == PairReplsStatus::Dead) {
        R->Status = PairReplsStatus::Alive;
        ReplMap.insert(std::make_pair(R->Repl->getOffset(), R->Repl));
      }
    } else
      R = std::make_shared<PairReplsStatus>(Repl, PairReplsStatus::Alive);
  }
}

bool ExtReplacements::isDuplicated(std::shared_ptr<ExtReplacement> Repl,
                                   ReplIterator Begin, ReplIterator End) {
  while (Begin != End) {
    if (*(Begin->second) == *Repl)
      return true;
    ++Begin;
  }
  return false;
}

void ExtReplacements::addReplacement(std::shared_ptr<ExtReplacement> Repl) {
  auto const &FileInfo = DpctGlobalInfo::getInstance().insertFile(FilePath);
  if (isInvalid(Repl, FileInfo))
    return;
  if (Repl->getLength()) {
    if (Repl->IsSYCLHeaderNeeded())
      FileInfo->insertHeader(HT_SYCL);
    // If Repl is not insert replacement, insert it.
    ReplMap.insert(std::make_pair(Repl->getOffset(), Repl));
    // If Repl is insert replacement, check whether it is alive or dead.
  } else if (checkLiveness(Repl)) {
    if (Repl->IsSYCLHeaderNeeded())
      FileInfo->insertHeader(HT_SYCL);
    markAsAlive(Repl);
  } else {
    markAsDead(Repl);
  }
}

void ExtReplacements::emplaceIntoReplSet(tooling::Replacements &ReplSet) {
  std::vector<std::shared_ptr<clang::dpct::ExtReplacement>> ReplsList =
      mergeReplsAtSameOffset();
  unsigned PrevEnd = 0;
  for (auto &R : ReplsList) {
    if (auto Repl = filterOverlappedReplacement(R, PrevEnd)) {
      if (auto Err = ReplSet.add(*Repl)) {
        llvm::dbgs() << Err << "\n";
      }
    }
  }
}
void ExtReplacements::processCudaArchMacro() {
  // process __CUDA_ARCH__ macro
  auto &CudaArchPPInfosMap =
      DpctGlobalInfo::getInstance().getCudaArchPPInfoMap()[FilePath];
  auto &CudaArchDefinedMap =
      DpctGlobalInfo::getInstance().getCudaArchDefinedMap()[FilePath];
  auto &ReplSet = DpctGlobalInfo::getInstance().getCudaArchMacroReplSet();
  // process __CUDA_ARCH__ macro of directive condition in generated host code:
  // if __CUDA_ARCH__ > 800      -->  if !DPCT_COMPATIBILITY_TEMP
  // if defined(__CUDA_ARCH__)   -->  if !defined(DPCT_COMPATIBILITY_TEMP)
  // if !defined(__CUDA_ARCH__)  -->  if defined(DPCT_COMPATIBILITY_TEMP)
  auto processIfMacro = [&](std::shared_ptr<ExtReplacement> Repl,
                            DirectiveInfo DI) {
    if (CudaArchDefinedMap.count((*Repl).getOffset())) {
      unsigned int ExclamationOffset =
          CudaArchDefinedMap[(*Repl).getOffset()] - DI.ConditionLoc - 1;
      if (ExclamationOffset <= (DI.Condition.length() - 1) &&
          DI.Condition[ExclamationOffset] == '!') {
        addReplacement(std::make_shared<ExtReplacement>(
            FilePath, CudaArchDefinedMap[(*Repl).getOffset()] - 1, 1, "",
            nullptr));
      } else {
        addReplacement(std::make_shared<ExtReplacement>(
            FilePath, CudaArchDefinedMap[(*Repl).getOffset()], 0, "!",
            nullptr));
      }
    } else {
      (*Repl).setReplacementText("!DPCT_COMPATIBILITY_TEMP");
    }
  };
  for (auto Repl = ReplSet.begin(); Repl != ReplSet.end();) {
    if ((*Repl)->getFilePath() != FilePath) {
      Repl++;
      continue;
    }
    unsigned CudaArchOffset = (*Repl)->getOffset();
    bool DirectiveReserved = true;
    for (auto Iterator = CudaArchPPInfosMap.begin();
         Iterator != CudaArchPPInfosMap.end(); Iterator++) {
      auto Info = Iterator->second;
      if (!Info.isInHDFunc)
        continue;
      unsigned Pos_a = 0, Len_a = 0, Pos_b = 0, Len_b = 0,
               Round = DpctGlobalInfo::getRunRound();
      if (CudaArchOffset >= Info.IfInfo.ConditionLoc &&
          CudaArchOffset <=
              Info.IfInfo.ConditionLoc + Info.IfInfo.Condition.length()) {
        if (Info.ElInfo.size() == 0) {
          if (Info.ElseInfo.DirectiveLoc == 0) {
            //  Remove unnecessary condition branch, as code is absolutely dead
            //  or active Origin Code:
            //  ...
            //  #ifdef __CUDA_ARCH__ / #if defined(__CUDA_ARCH__) / #if
            //  __CUDA_ARCH__ / #ifndef __CUDA_ARCH__ / #if
            //  !defined(__CUDA_ARCH__)
            //    host_code/device code;
            //  #endif
            //  ...
            //
            //  After Migration:
            //  Round = 0 for device code, final migration code:
            //    ...
            //    empty/device code;
            //    ...
            //  Round = 1 for host code, final migration code:
            //    ...
            //    host_code/empty;
            //    ...
            if ((Info.DT == IfType::IT_Ifdef && Round == 1) ||
                (Info.DT == IfType::IT_Ifndef && Round == 0) ||
                (Info.DT == IfType::IT_If && Round == 1 &&
                 (Info.IfInfo.Condition == "defined(__CUDA_ARCH__)" ||
                  Info.IfInfo.Condition == "__CUDA_ARCH__")) ||
                (Info.DT == IfType::IT_If && Round == 0 &&
                 Info.IfInfo.Condition == "!defined(__CUDA_ARCH__)")) {
              Pos_a = Info.IfInfo.NumberSignLoc;
              if (Pos_a != UINT_MAX) {
                Len_a =
                    Info.EndInfo.DirectiveLoc - Pos_a + 5 /*length of endif*/;
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_a, Len_a, "", nullptr));
                DirectiveReserved = false;
              }
            } else if ((Info.DT == IfType::IT_Ifdef && Round == 0) ||
                       (Info.DT == IfType::IT_Ifndef && Round == 1) ||
                       (Info.DT == IfType::IT_If && Round == 0 &&
                        (Info.IfInfo.Condition == "defined(__CUDA_ARCH__)" ||
                         Info.IfInfo.Condition == "__CUDA_ARCH__")) ||
                       (Info.DT == IfType::IT_If && Round == 1 &&
                        Info.IfInfo.Condition == "!defined(__CUDA_ARCH__)")) {
              Pos_a = Info.IfInfo.NumberSignLoc;
              Pos_b = Info.EndInfo.NumberSignLoc;
              if (Pos_a != UINT_MAX && Pos_b != UINT_MAX) {
                Len_a = Info.IfInfo.ConditionLoc +
                        Info.IfInfo.Condition.length() - Pos_a;
                Len_b =
                    Info.EndInfo.DirectiveLoc - Pos_b + 5 /*length of endif*/;
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_a, Len_a, "", nullptr));
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_b, Len_b, "", nullptr));
                DirectiveReserved = false;
              }
            }
          } else {
            //  Remove conditional branch, as code is absolutely dead or active
            //  Origin Code:
            //  ...
            //  #ifdef __CUDA_ARCH__ / #if defined(__CUDA_ARCH__) / #if
            //  __CUDA_ARCH__ / #ifndef __CUDA_ARCH / #if
            //  !defined(__CUDA_ARCH__)
            //    host_code/device_code;
            //  #else
            //    device_code/host_code;
            //  #endif
            //  ...
            //
            //  After Migration:
            //  Round = 0 for device code, final migration code:
            //    ...
            //    device_code;
            //    ...
            //  Round = 1 for host code, final migration code:
            //    ...
            //    host_code;
            //    ...
            if ((Info.DT == IfType::IT_Ifdef && Round == 1) ||
                (Info.DT == IfType::IT_Ifndef && Round == 0) ||
                (Info.DT == IfType::IT_If && Round == 1 &&
                 (Info.IfInfo.Condition == "defined(__CUDA_ARCH__)" ||
                  Info.IfInfo.Condition == "__CUDA_ARCH__")) ||
                (Info.DT == IfType::IT_If && Round == 0 &&
                 Info.IfInfo.Condition == "!defined(__CUDA_ARCH__)")) {
              Pos_a = Info.IfInfo.NumberSignLoc;
              Pos_b = Info.EndInfo.NumberSignLoc;
              if (Pos_a != UINT_MAX && Pos_b != UINT_MAX) {
                Len_a =
                    Info.ElseInfo.DirectiveLoc - Pos_a + 4 /*length of else*/;
                Len_b =
                    Info.EndInfo.DirectiveLoc - Pos_b + 5 /*length of endif*/;
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_a, Len_a, "", nullptr));
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_b, Len_b, "", nullptr));
                DirectiveReserved = false;
              }
            } else if ((Info.DT == IfType::IT_Ifdef && Round == 0) ||
                       (Info.DT == IfType::IT_Ifndef && Round == 1) ||
                       (Info.DT == IfType::IT_If && Round == 0 &&
                        (Info.IfInfo.Condition == "defined(__CUDA_ARCH__)" ||
                         Info.IfInfo.Condition == "__CUDA_ARCH__")) ||
                       (Info.DT == IfType::IT_If && Round == 1 &&
                        Info.IfInfo.Condition == "!defined(__CUDA_ARCH__)")) {
              Pos_a = Info.IfInfo.NumberSignLoc;
              Pos_b = Info.ElseInfo.NumberSignLoc;
              if (Pos_a != UINT_MAX && Pos_b != UINT_MAX) {
                Len_a = Info.IfInfo.ConditionLoc +
                        Info.IfInfo.Condition.length() - Pos_a;
                Len_b =
                    Info.EndInfo.DirectiveLoc - Pos_b + 5 /*length of endif*/;
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_a, Len_a, "", nullptr));
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_b, Len_b, "", nullptr));
                DirectiveReserved = false;
              }
            }
          }
        }
        //  if directive in which __CUDA_ARCH__ inside was reserved, then we
        //  need process this directive for generated host code:
        //  ifndef__CUDA_ARCH__ --> ifdef DPCT_COMPATIBILITY_TEMP
        //  ifdef __CUDA_ARCH__ --> ifndef DPCT_COMPATIBILITY_TEMP
        if (DirectiveReserved && Round == 1) {
          if (Info.DT == IfType::IT_Ifdef) {
            Pos_a = Info.IfInfo.DirectiveLoc;
            Len_a = 5 /*length of ifdef*/;
            addReplacement(std::make_shared<ExtReplacement>(
                FilePath, Pos_a, Len_a, "ifndef", nullptr));
          } else if (Info.DT == IfType::IT_Ifndef) {
            Pos_a = Info.IfInfo.DirectiveLoc;
            Len_a = 6 /*length of ifndef*/;
            addReplacement(std::make_shared<ExtReplacement>(
                FilePath, Pos_a, Len_a, "ifdef", nullptr));
          } else if (Info.DT == IfType::IT_If) {
            processIfMacro(*Repl, Info.IfInfo);
          }
        }
        break;
      } else {
        //  Info.ElInfo.size() == 0
        if (Round == 0)
          continue;
        for (auto &ElifInfoPair : Info.ElInfo) {
          auto &ElifInfo = ElifInfoPair.second;
          if (CudaArchOffset >= ElifInfo.ConditionLoc &&
              CudaArchOffset <=
                  ElifInfo.ConditionLoc + ElifInfo.Condition.length()) {
            processIfMacro(*Repl, ElifInfo);
            break;
          }
        }
      }
    }
    if (DirectiveReserved) {
      addReplacement(*Repl);
      Repl = ReplSet.erase(Repl);
    } else {
      Repl++;
    }
  }
}
void ExtReplacements::buildCudaArchHostFunc(
    std::shared_ptr<DpctFileInfo> FileInfo) {
  std::vector<std::shared_ptr<ExtReplacement>> ReplsList =
      mergeReplsAtSameOffset();
  std::vector<std::shared_ptr<ExtReplacement>> ProcessedReplList;
  unsigned PrevEnd = 0;
  for (auto &R : ReplsList) {
    if (auto Repl = filterOverlappedReplacement(R, PrevEnd)) {
      ProcessedReplList.emplace_back(Repl);
    }
  }
  static int id = 0;
  using PostfixMapTy = std::unordered_map<std::string, std::string>;
  static PostfixMapTy PostfixMap;
  std::vector<std::shared_ptr<ExtReplacement>> ExtraRepl;
  auto &HDFDIMap = DpctGlobalInfo::getInstance().getHostDeviceFuncDefInfoMap();
  auto &HDFDeclIMap =
      DpctGlobalInfo::getInstance().getHostDeviceFuncDeclInfoMap();
  auto &HDFCIMap = DpctGlobalInfo::getInstance().getHostDeviceFuncCallInfoMap();
  // process call
  for (auto &Call : HDFCIMap) {
    if (!PostfixMap.count(Call.first)) {
      PostfixMap[Call.first] = "_host_ct" + std::to_string(id++);
    }
    if (Call.second.first != FilePath || !HDFDIMap.count(Call.first))
      continue;
    unsigned Offset = Call.second.second;
    auto R = std::make_shared<ExtReplacement>(FilePath, Offset, 0,
                                              PostfixMap[Call.first], nullptr);
    ExtraRepl.emplace_back(R);
  }
  auto GenerateHostCode = [&ProcessedReplList, &ExtraRepl, &FileInfo](
                              HostDeviceFuncInfo &Info, PostfixMapTy &PMap,
                              std::string FuncName) {
    unsigned int Pos, Len;
    std::string OriginText = Info.FuncContentCache;
    StringRef SR(OriginText);
    RewriteBuffer RB;
    RB.Initialize(SR.begin(), SR.end());

    for (auto &R : ProcessedReplList) {
      unsigned ROffset = R->getOffset();
      if (ROffset >= Info.FuncStartOffset && ROffset <= Info.FuncEndOffset) {
        Pos = ROffset - Info.FuncStartOffset;
        Len = R->getLength();
        RB.ReplaceText(Pos, Len, R->getReplacementText());
      }
    }
    Pos = Info.FuncNameOffset - Info.FuncStartOffset;
    Len = 0;
    RB.ReplaceText(Pos, Len, PostfixMap[FuncName]);
    std::string DefResult;
    llvm::raw_string_ostream DefStream(DefResult);
    RB.write(DefStream);
    std::string NewFuncBody = DefStream.str();
    auto R = std::make_shared<ExtReplacement>(FileInfo->getFilePath(),
                                              Info.FuncEndOffset + 1, 0,
                                              getNL() + NewFuncBody, nullptr);
    ExtraRepl.emplace_back(R);
  };
  // process def
  for (auto &HDFDInfo : HDFDIMap) {
    if (!HDFCIMap.count(HDFDInfo.first) || HDFDInfo.second.first != FilePath)
      continue;
    GenerateHostCode(HDFDInfo.second.second, PostfixMap, HDFDInfo.first);
  }
  // process decl
  for (auto &Decl : HDFDeclIMap) {
    if (Decl.second.first != FilePath || !HDFCIMap.count(Decl.first) ||
        !HDFDIMap.count(Decl.first))
      continue;
    GenerateHostCode(Decl.second.second, PostfixMap, Decl.first);
  }
  for (auto &R : ExtraRepl) {
    auto &FileReplCache = DpctGlobalInfo::getFileReplCache();
    FileReplCache[R->getFilePath().str()]->addReplacement(R);
  }
  return;
}

void ExtReplacements::postProcess() {
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(FilePath);
  for (auto &R : ReplMap) {

    // D: device, used for pretty code
    // Q: queue, used for pretty code
    // V: vector size, used for rand API migration
    // R: range dim, used for built-in variables (threadIdx.x, ...) migration
    // C: cub group dim, used for cub API migration
    // F: free queries function migration, such as this_nd_item, this_group,
    // this_sub_group.
    // R: group dim size, used for cg::thread_block migration
    StringRef OriginalText = R.second->getReplacementText();
    std::regex RE("\\{\\{NEEDREPLACE[DQVRFCG][0-9]*\\}\\}");
    std::match_results<StringRef::const_iterator> Result;
    std::string NewText;
    auto Begin = OriginalText.begin(), End = OriginalText.end();
    while (std::regex_search(Begin, End, Result, RE)) {
      NewText.append(Result.prefix().first, Result.prefix().length());
      NewText += DpctGlobalInfo::getStringForRegexReplacement(
          StringRef(Result[0].first, Result[0].length()));
      Begin = Result.suffix().first;
    }
    if (NewText.size()) {
      NewText.append(Begin, End);
      auto &Old = R.second;
      auto New =
          std::make_shared<ExtReplacement>(Old->getFilePath(), Old->getOffset(),
                                           Old->getLength(), NewText, nullptr);
      New->setBlockLevelFormatFlag(Old->getBlockLevelFormatFlag());
      New->setInsertPosition(
          static_cast<dpct::InsertPosition>(Old->getInsertPosition()));
      Old = std::move(New);
    }
  }

  if (DpctGlobalInfo::isKeepOriginCode())
    buildOriginCodeReplacements(FileInfo);

  processCudaArchMacro();

  if (DpctGlobalInfo::getRunRound() == 1) {
    buildCudaArchHostFunc(FileInfo);
  }
  getLOCStaticFromCodeRepls(FileInfo);
  return;
}
} // namespace dpct
} // namespace clang
