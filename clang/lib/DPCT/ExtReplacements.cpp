//===--- ExtReplacements.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
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
ExtReplacements::ExtReplacements(DpctFileInfo *FileInfo)
    : FilePath(FileInfo->getFilePath()), FileInfo(FileInfo) {}

bool ExtReplacements::isInvalid(std::shared_ptr<ExtReplacement> Repl) {
  if (!Repl)
    return true;
  if (Repl->getFilePath() != FilePath)
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
  return false;
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
      return Longer;
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
    return Longer;
  }
}

void ExtReplacements::removeCommentsInSrcCode(const std::string &SrcCode,
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
        Result.append(SrcCode, PrevPos, Pos - PrevPos - 1);
        LineComment = true;
        break;
      } else if (SrcCode[Pos] == '*') {
        // encount "/*", block comment.
        Result.append(SrcCode, PrevPos, Pos - PrevPos - 1);
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
    Result.append(SrcCode.begin() + PrevPos, SrcCode.end());
}

size_t ExtReplacements::findCR(const std::string &Line) {
  auto Pos = Line.rfind('\n');
  if (Pos && Pos != std::string::npos) {
    if (Line[Pos - 1] == '\r')
      return --Pos;
  }
  return Pos;
}

bool ExtReplacements::isEndWithSlash(unsigned LineNumber) {
  if (!LineNumber)
    return false;
  auto &Line = FileInfo->getLineString(LineNumber);
  auto CRPos = findCR(Line);
  if (!CRPos || CRPos == std::string::npos)
    return false;
  return Line[--CRPos] == '\\';
}

std::shared_ptr<ExtReplacement>
ExtReplacements::buildOriginCodeReplacement(const SourceLineRange &LineRange) {
  if (!LineRange.SrcBeginLine)
    return std::shared_ptr<ExtReplacement>();
  std::string Text = "/* DPCT_ORIG ";
  bool BlockComment = false;
  for (unsigned Line = LineRange.SrcBeginLine; Line <= LineRange.SrcEndLine;
       ++Line)
    removeCommentsInSrcCode(FileInfo->getLineString(Line), Text, BlockComment);

  std::string Suffix =
      std::string(isEndWithSlash(LineRange.SrcBeginLine - 1) ? "*/ \\" : "*/");
  Text.insert(findCR(Text), Suffix);
  auto R = std ::make_shared<ExtReplacement>(FilePath, LineRange.SrcBeginOffset,
                                             0, std::move(Text), nullptr);
  R->setInsertPosition(InsertPositionAlwaysLeft);
  return R;
}

void ExtReplacements::buildOriginCodeReplacements() {
  SourceLineRange LineRange, ReplLineRange;
  for (auto &R : ReplMap) {
    auto &Repl = R.second;
    if (Repl->getLength()) {
      FileInfo->setLineRange(ReplLineRange, Repl);
      if (LineRange.SrcEndLine < ReplLineRange.SrcBeginLine) {
        addReplacement(buildOriginCodeReplacement(LineRange));
        LineRange = ReplLineRange;
      } else
        LineRange.SrcEndLine =
            std::max(LineRange.SrcEndLine, ReplLineRange.SrcEndLine);
    }
  }
  if (LineRange.SrcBeginLine)
    addReplacement(buildOriginCodeReplacement(LineRange));
}

std::vector<std::shared_ptr<ExtReplacement>>
ExtReplacements::mergeReplsAtSameOffset() {
  std::vector<std::shared_ptr<ExtReplacement>> ReplsList;
  std::shared_ptr<ExtReplacement> Insert, Replace;
  unsigned Offset = ReplMap.begin()->first;
  for (auto &R : ReplMap) {
    if (R.first != Offset) {
      Offset = R.first;
      ReplsList.emplace_back(mergeAtSameOffset(Insert, Replace));
      Insert.reset();
      Replace.reset();
    }
    auto &Repl = R.second;
    if (Repl->getLength()) {
      Replace = mergeAtSameOffset(Replace, Repl);
    } else {
      Insert = mergeAtSameOffset(Insert, Repl);
    }
  }
  if (Insert || Replace) {
    ReplsList.emplace_back(mergeAtSameOffset(Insert, Replace));
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
  if (isInvalid(Repl))
    return;
  if (Repl->getLength())
    // If Repl is not insert replacement, insert it.
    ReplMap.insert(std::make_pair(Repl->getOffset(), Repl));
  // If Repl is insert replacement, check whether it is alive or dead.
  else if (checkLiveness(Repl))
    markAsAlive(Repl);
  else
    markAsDead(Repl);
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
      Text = "dpct::get_default_queue()";
    } else if (HFT == HelperFuncType::CurrentDevice) {
      Text = "dpct::get_current_device()";
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
      Text = "dpct::get_default_queue()";
      return true;
    } else if (TempVariableDeclCounterIter->second.DefaultQueueCounter <= 1) {
      Text = "dpct::get_default_queue()";
      return true;
    } else {
      Text = "q_ct1";
      return true;
    }
  } else if (HFT == HelperFuncType::CurrentDevice) {
    if (!HelperFuncReplInfoIter->second.IsLocationValid) {
      Text = "dpct::get_current_device()";
      return true;
    } else if (TempVariableDeclCounterIter->second.CurrentDeviceCounter <= 1 &&
               TempVariableDeclCounterIter->second.DefaultQueueCounter <= 1) {
      Text = "dpct::get_current_device()";
      return true;
    } else {
      Text = "dev_ct1";
      return true;
    }
  }
  return false;
}

void ExtReplacements::emplaceIntoReplSet(tooling::Replacements &ReplSet) {
  for (auto &R : ReplMap) {
    std::string OriginReplText = R.second->getReplacementText().str();
    std::string NewReplText;

    std::regex RE("\\{\\{NEEDREPLACE[DQ][1-9][0-9]*\\}\\}");
    std::smatch MRes;
    if (std::regex_search(OriginReplText, MRes, RE)) {
      std::string MatchedStr = MRes.str();
      NewReplText = NewReplText + std::string(MRes.prefix());

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
        // Not found HelperFuncReplInfo in the map, it means this place is migrated,
        // So only the first time will have HelperFuncReplInfo.
        // In this case, just remove whole replacement.
        R.second = std::make_shared<ExtReplacement>(
            FilePath, R.second->getOffset(), 0, "", nullptr);
      } else {
        std::string Text;
        if (getStrReplacingPlaceholder(HFT, Index, Text)) {
          NewReplText = NewReplText + Text;
        } else {
          NewReplText = NewReplText + MatchedStr;
        }

        NewReplText = NewReplText + std::string(MRes.suffix());

        // Using "NewReplText" to generate a new ExtReplacement, then replace
        // the old one in the ReplMap
        R.second = std::make_shared<ExtReplacement>(
            FilePath, R.second->getOffset(), R.second->getLength(), NewReplText,
            nullptr);
      }
    }
  }

  if (DpctGlobalInfo::isKeepOriginCode())
    buildOriginCodeReplacements();

  std::vector<std::shared_ptr<ExtReplacement>> ReplsList =
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
} // namespace dpct
} // namespace clang
