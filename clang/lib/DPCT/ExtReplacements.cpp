//===--- ExtReplacements.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "ExtReplacements.h"

#include "AnalysisInfo.h"
#include "Debug.h"

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
    // TODO: Currently not make sure "keep the longer, remove shorter" is
    // correct, need to do in the future.
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
  R->setInsertPosition(InsertPositionAlwaysLeft);
  return R;
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
    } else if (Repl->getInsertPosition() ==
               InsertPosition::InsertPositionAlwaysLeft) {
      InsertLeft = mergeAtSameOffset(InsertLeft, Repl);
    } else if (Repl->getInsertPosition() ==
               InsertPosition::InsertPositionRight) {
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
      FileInfo->insertHeader(SYCL);
    // If Repl is not insert replacement, insert it.
    ReplMap.insert(std::make_pair(Repl->getOffset(), Repl));
    // If Repl is insert replacement, check whether it is alive or dead.
  } else if (checkLiveness(Repl)) {
    if (Repl->IsSYCLHeaderNeeded())
      FileInfo->insertHeader(SYCL);
    markAsAlive(Repl);
  } else {
    markAsDead(Repl);
  }
}

bool ExtReplacements::getStrReplacingPlaceholder(HelperFuncType HFT, int Index,
                                                 std::string &Text) {
  if (HFT != HelperFuncType::DefaultQueue &&
      HFT != HelperFuncType::CurrentDevice) {
    return false;
  }

  auto HelperFuncReplInfoIter =
      DpctGlobalInfo::getHelperFuncReplInfoMap().find(Index);

  if (DpctGlobalInfo::getDeviceChangedFlag() ||
      !DpctGlobalInfo::getUsingDRYPattern()) {
    if (HFT == HelperFuncType::DefaultQueue) {
      requestFeature(HelperFileEnum::Device, "get_default_queue", FilePath);
      Text = MapNames::getDpctNamespace() + "get_default_queue()";
    } else if (HFT == HelperFuncType::CurrentDevice) {
      requestFeature(HelperFileEnum::Device, "get_current_device", FilePath);
      Text = MapNames::getDpctNamespace() + "get_current_device()";
    }
    return true;
  }

  std::string CounterKey =
      HelperFuncReplInfoIter->second.DeclLocFile + ":" +
      std::to_string(HelperFuncReplInfoIter->second.DeclLocOffset);

  auto TempVariableDeclCounterIter =
      DpctGlobalInfo::getTempVariableDeclCounterMap().find(CounterKey);
  if (TempVariableDeclCounterIter ==
      DpctGlobalInfo::getTempVariableDeclCounterMap().end()) {
    return false;
  }

  // All cases of replacing placeholders:
  // dev_count  queue_count  dev_decl            queue_decl
  // 0          1            /                   get_default_queue
  // 1          0            get_current_device  /
  // 1          1            get_current_device  get_default_queue
  // 2          1            dev_ct1             get_default_queue
  // 1          2            dev_ct1             q_ct1
  // >=2        >=2          dev_ct1             q_ct1
  if (HFT == HelperFuncType::DefaultQueue) {
    if (!HelperFuncReplInfoIter->second.IsLocationValid) {
      requestFeature(HelperFileEnum::Device, "get_default_queue", FilePath);
      Text = MapNames::getDpctNamespace() + "get_default_queue()";
      return true;
    } else if (TempVariableDeclCounterIter->second.DefaultQueueCounter <= 1) {
      requestFeature(HelperFileEnum::Device, "get_default_queue", FilePath);
      Text = MapNames::getDpctNamespace() + "get_default_queue()";
      return true;
    } else {
      Text = "q_ct1";
      return true;
    }
  } else if (HFT == HelperFuncType::CurrentDevice) {
    if (!HelperFuncReplInfoIter->second.IsLocationValid) {
      requestFeature(HelperFileEnum::Device, "get_current_device", FilePath);
      Text = MapNames::getDpctNamespace() + "get_current_device()";
      return true;
    } else if (TempVariableDeclCounterIter->second.CurrentDeviceCounter <= 1 &&
               TempVariableDeclCounterIter->second.DefaultQueueCounter <= 1) {
      requestFeature(HelperFileEnum::Device, "get_current_device", FilePath);
      Text = MapNames::getDpctNamespace() + "get_current_device()";
      return true;
    } else {
      Text = "dev_ct1";
      return true;
    }
  }
  return false;
}

std::string ExtReplacements::processV() {
  std::string Res;
  if (DpctGlobalInfo::getDeviceRNGReturnNumSet().size() == 1) {
    Res = std::to_string(*DpctGlobalInfo::getDeviceRNGReturnNumSet().begin());
  } else {
    Res = "dpct_placeholder/*Fix the vec_size manually*/";
  }
  return Res;
}
std::string ExtReplacements::processR(unsigned int Index) {
  std::string Res = "2";
  if (auto DFI = DpctGlobalInfo::getCudaBuiltinXDFI(Index)) {
    auto Ptr = MemVarMap::getHeadWithoutPathCompression(&(DFI->getVarMap()));
    if (Ptr && Ptr->Dim == 1) {
      Res = "0";
    }
  }
  return Res;
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
            if ((Info.DT == IfType::Ifdef && Round == 1) ||
                (Info.DT == IfType::Ifndef && Round == 0) ||
                (Info.DT == IfType::If && Round == 1 &&
                 (Info.IfInfo.Condition == "defined(__CUDA_ARCH__)" ||
                  Info.IfInfo.Condition == "__CUDA_ARCH__")) ||
                (Info.DT == IfType::If && Round == 0 &&
                 Info.IfInfo.Condition == "!defined(__CUDA_ARCH__)")) {
              Pos_a = Info.IfInfo.NumberSignLoc;
              if (Pos_a != UINT_MAX) {
                Len_a =
                    Info.EndInfo.DirectiveLoc - Pos_a + 5 /*length of endif*/;
                addReplacement(std::make_shared<ExtReplacement>(
                    FilePath, Pos_a, Len_a, "", nullptr));
                DirectiveReserved = false;
              }
            } else if ((Info.DT == IfType::Ifdef && Round == 0) ||
                       (Info.DT == IfType::Ifndef && Round == 1) ||
                       (Info.DT == IfType::If && Round == 0 &&
                        (Info.IfInfo.Condition == "defined(__CUDA_ARCH__)" ||
                         Info.IfInfo.Condition == "__CUDA_ARCH__")) ||
                       (Info.DT == IfType::If && Round == 1 &&
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
            if ((Info.DT == IfType::Ifdef && Round == 1) ||
                (Info.DT == IfType::Ifndef && Round == 0) ||
                (Info.DT == IfType::If && Round == 1 &&
                 (Info.IfInfo.Condition == "defined(__CUDA_ARCH__)" ||
                  Info.IfInfo.Condition == "__CUDA_ARCH__")) ||
                (Info.DT == IfType::If && Round == 0 &&
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
            } else if ((Info.DT == IfType::Ifdef && Round == 0) ||
                       (Info.DT == IfType::Ifndef && Round == 1) ||
                       (Info.DT == IfType::If && Round == 0 &&
                        (Info.IfInfo.Condition == "defined(__CUDA_ARCH__)" ||
                         Info.IfInfo.Condition == "__CUDA_ARCH__")) ||
                       (Info.DT == IfType::If && Round == 1 &&
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
          if (Info.DT == IfType::Ifdef) {
            Pos_a = Info.IfInfo.DirectiveLoc;
            Len_a = 5 /*length of ifdef*/;
            addReplacement(std::make_shared<ExtReplacement>(
                FilePath, Pos_a, Len_a, "ifndef", nullptr));
          } else if (Info.DT == IfType::Ifndef) {
            Pos_a = Info.IfInfo.DirectiveLoc;
            Len_a = 6 /*length of ifndef*/;
            addReplacement(std::make_shared<ExtReplacement>(
                FilePath, Pos_a, Len_a, "ifdef", nullptr));
          } else if (Info.DT == IfType::If) {
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
    std::string OriginReplText = R.second->getReplacementText().str();
    std::string NewReplText;

    // D: deivce, used for pretty code
    // Q: queue, used for pretty code
    // V: vector size, used for rand API migration
    // R: range dim, used for built-in variables(threadIdx.x,...) migration
    std::regex RE("\\{\\{NEEDREPLACE[DQVR][1-9][0-9]*\\}\\}");
    std::smatch MRes;
    std::string MatchedSuffix;
    bool Matched = false;
    while (std::regex_search(OriginReplText, MRes, RE)) {
      Matched = true;
      std::string MatchedStr = MRes.str();
      NewReplText = NewReplText + std::string(MRes.prefix());

      if (MatchedStr.substr(13, 1) == "V" || MatchedStr.substr(13, 1) == "R") {
        if (MatchedStr.substr(13, 1) == "V") {
          NewReplText = NewReplText + processV();
        } else {
          unsigned int Index =
              std::stoi(MatchedStr.substr(14, MatchedStr.size() - 14));
          NewReplText = NewReplText + processR(Index);
        }
        MatchedSuffix = std::string(MRes.suffix());
        OriginReplText = MatchedSuffix;
      } else if (MatchedStr.substr(13, 1) == "Q" ||
                 MatchedStr.substr(13, 1) == "D") {
        // get the index from the placeholder string
        int Index = std::stoi(MatchedStr.substr(14, MatchedStr.size() - 14));
        // get the HelperFuncType from the placeholder string
        HelperFuncType HFT = HelperFuncType::InitValue;
        if (MatchedStr.substr(13, 1) == "Q")
          HFT = HelperFuncType::DefaultQueue;
        else if (MatchedStr.substr(13, 1) == "D")
          HFT = HelperFuncType::CurrentDevice;

        auto HelperFuncReplInfoIter =
            DpctGlobalInfo::getHelperFuncReplInfoMap().find(Index);
        if (HelperFuncReplInfoIter ==
            DpctGlobalInfo::getHelperFuncReplInfoMap().end()) {
          // Cannot found HelperFuncReplInfo in the map, migrate it to default
          // queue
          NewReplText = NewReplText + MapNames::getDpctNamespace() +
                        "get_default_queue()";
          requestFeature(HelperFileEnum::Device, "get_default_queue", FilePath);
        } else {
          std::string Text;
          if (getStrReplacingPlaceholder(HFT, Index, Text)) {
            NewReplText = NewReplText + Text;
          } else {
            NewReplText = NewReplText + MatchedStr;
          }
        }
        MatchedSuffix = std::string(MRes.suffix());
        OriginReplText = MatchedSuffix;
      }
    }

    if (Matched) {
      NewReplText = NewReplText + MatchedSuffix;
      auto NewRepl = std::make_shared<ExtReplacement>(
          FilePath, R.second->getOffset(), R.second->getLength(), NewReplText,
          nullptr);
      NewRepl->setBlockLevelFormatFlag(R.second->getBlockLevelFormatFlag());
      NewRepl->setInsertPosition(
          (dpct::InsertPosition)R.second->getInsertPosition());
      R.second = NewRepl;
    }
  }
  if (DpctGlobalInfo::isKeepOriginCode())
    buildOriginCodeReplacements(FileInfo);

  processCudaArchMacro();

  if (DpctGlobalInfo::getRunRound() == 1) {
    buildCudaArchHostFunc(FileInfo);
  }
  return;
}
} // namespace dpct
} // namespace clang
