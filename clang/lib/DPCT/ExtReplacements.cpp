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
ExtReplacements::ExtReplacements(std::string NewFilePath) {
#if defined(_WIN32)
  std::transform(NewFilePath.begin(), NewFilePath.end(), NewFilePath.begin(),
                 [](unsigned char c) { return std::tolower(c); });
#endif
  FilePath = NewFilePath;
}

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
void ExtReplacements::getLOCStaticFromCodeRepls(
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
  if (Insert || Replace || InsertLeft || InsertRight) {
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
    // G: group dim size, used for cg::thread_block migration
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

  getLOCStaticFromCodeRepls(FileInfo);
  return;
}
} // namespace dpct
} // namespace clang
