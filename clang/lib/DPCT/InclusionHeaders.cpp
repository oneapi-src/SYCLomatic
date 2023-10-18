//===--------------- InclusionHeaders.cpp----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InclusionHeaders.h"

#include "ASTTraversal.h"

namespace clang {
namespace dpct {

namespace {

// A class that uses the RAII idiom to selectively update the locations of the
// last inclusion directives.
class LastInclusionLocationUpdater {
public:
  LastInclusionLocationUpdater(std::shared_ptr<DpctFileInfo> FileInfo,
                               SourceLocation Loc)
      : File(std::move(FileInfo)), Loc(Loc), UpdateNeeded(true) {}
  ~LastInclusionLocationUpdater() {
    if (UpdateNeeded)
      File->setLastIncludeOffset(DpctGlobalInfo::getLocInfo(Loc).second);
  }
  void give_up() { UpdateNeeded = false; }

private:
  std::shared_ptr<DpctFileInfo> File;
  SourceLocation Loc;
  bool UpdateNeeded;
};

std::string applyUserDefinedHeader(const std::string &FileName) {
  // Apply user-defined rule if needed
  auto It = MapNames::HeaderRuleMap.find(FileName);
  if (It != MapNames::HeaderRuleMap.end() &&
      It->second.Priority == RulePriority::Takeover) {
    auto &Rule = It->second;
    std::string ReplHeaderStr = Rule.Prefix;
    llvm::raw_string_ostream OS(ReplHeaderStr);
    auto PrintHeader = [&](const std::string &Header) {
      OS << "#include ";
      {
        PairedPrinter Paired(OS, "\"", "\"",
                             Header.front() != '<' && Header.front() != '"');
        OS << Header;
      }
      static const StringRef NL = getNL();
      OS << NL;
    };
    for (auto &Header : Rule.Includes) {
      PrintHeader(Header);
    }
    PrintHeader(Rule.Out);
    OS << Rule.Postfix;
    return ReplHeaderStr;
  }
  return "";
}

void insertHeaders(std::shared_ptr<DpctFileInfo> File,
                   const SmallVector<HeaderType, 2> Headers) {
  for (auto Header : Headers)
    File->insertHeader(Header);
}
void setHeadersAsInserted(std::shared_ptr<DpctFileInfo> File,
                          const SmallVector<HeaderType, 2> Headers) {
  for (auto Header : Headers)
    File->setHeaderInserted(Header);
}

// For included file start with following string, inclusion will always be
// removed even in folder "/usr/include".
SmallVector<std::string, 8> AlwaysRemovedSDKFilePrefix = {
    "cuda", "cusolver", "cublas", "cusparse", "curand"};

bool isAlwaysRemoved(StringRef Path) {
  for (auto &S : AlwaysRemovedSDKFilePrefix)
    if (Path.startswith(S))
      return true;
  return false;
}

struct InclusionStartwithEntry {
  std::string Prefix;
  DpctInclusionInfo Info;

  InclusionStartwithEntry(StringRef Prefix) : Prefix(Prefix.str()) {}
};

llvm::StringMap<DpctInclusionInfo> InclusionFullMatchMap;
SmallVector<InclusionStartwithEntry> InclusionStartWithMap;

const DpctInclusionInfo *findInFullMatcheMode(StringRef Filename) {
  auto Iter = InclusionFullMatchMap.find(Filename);
  if (Iter != InclusionFullMatchMap.end())
    return &Iter->second;
  return nullptr;
}

const DpctInclusionInfo *findInStartwithMode(StringRef Filename) {
  for (auto &Entry : InclusionStartWithMap) {
    if (Filename.startswith(Entry.Prefix))
      return &Entry.Info;
  }
  return nullptr;
}

} // namespace

void IncludesCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, OptionalFileEntryRef File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {

  // If the header file included cannot be found, just return.
  if (!File) {
    return;
  }

  auto &Global = DpctGlobalInfo::getInstance();
  auto LocInfo = Global.getLocInfo(HashLoc);
  auto FileInfo = Global.insertFile(LocInfo.first);

  // Record the locations of the first and last inclusion directives in a file
  FileInfo->setFirstIncludeOffset(LocInfo.second);
  LastInclusionLocationUpdater Updater(FileInfo, FilenameRange.getEnd());

  std::string IncludedFile;
  if (auto OptionalAbs = Global.getAbsolutePath(File->getFileEntry()))
    IncludedFile = OptionalAbs.value();

  if (Global.isExcluded(IncludedFile))
    return;

  auto GenReplacement =
      [&, ReplaceRange =
              CharSourceRange(SourceRange(HashLoc, FilenameRange.getEnd()),
                              /*IsTokenRange=*/false)](
          std::string ReplacedStr, bool RemoveTrailingSpaces = false) {
        return new ReplaceInclude(ReplaceRange, std::move(ReplacedStr),
                                  RemoveTrailingSpaces);
      };
  auto EmplaceReplacement = [&](std::string ReplacedStr,
                                bool RemoveTrailingSpaces = false) {
    TransformSet.emplace_back(
        GenReplacement(std::move(ReplacedStr), RemoveTrailingSpaces));
  };
  auto RemoveInslusion = [&]() {
    EmplaceReplacement("", true);
    Updater.give_up();
  };

  if (Global.isInAnalysisScope(IncludedFile)) {
    auto FilePathWithoutSymlinks =
        Global.removeSymlinks(SM.getFileManager(), IncludedFile);
    IncludeFileMap[FilePathWithoutSymlinks] = false;
    Global.getIncludingFileSet().insert(FilePathWithoutSymlinks);

    // The "IncludedFile" is included by the "IncludingFile".
    // If "IncludedFile" is not under the AnalysisScope folder, do not record
    // the including relationship information.
    Global.recordIncludingRelationship(LocInfo.first, IncludedFile);

    SmallString<512> NewFileName = FileName;
    rewriteFileName(NewFileName, IncludedFile);
    if (NewFileName != FileName) {
      const auto Extension = path::extension(FileName);
      auto ReplacedStr = buildString("#include \"", NewFileName, "\"");
      if (Extension == ".cu" || Extension == ".cuh") {
        // For CUDA files, it will always change name.
        EmplaceReplacement(std::move(ReplacedStr));
      } else {
        // For other CppSource file type, it may change name or not, which
        // determined by whether it has CUDA syntax, so just record the
        // replacement in the IncludeMapSet.
        IncludeMapSet[IncludedFile].emplace_back(
            GenReplacement(std::move(ReplacedStr)));
      }
    }
    return;
  }

  if (!Global.isInAnalysisScope(LocInfo.first) &&
      !Global.getSourceManager().isWrittenInMainFile(HashLoc))
    return;


  // Apply user-defined rule if needed
  if (auto ReplacedStr = applyUserDefinedHeader(FileName.str());
      !ReplacedStr.empty()) {
    EmplaceReplacement(std::move(ReplacedStr));
    return;
  }

  do {
    auto InfoPtr = DpctInclusionHeadersMap::findHeaderInfo(FileName);
    if (!InfoPtr)
      break;
    auto &Info = *InfoPtr;

    if (Info.MustAngled && !IsAngled)
      break;

    Groups.enableRuleGroup(Info.RuleGroup);

    switch (Info.ProcessFlag) {
    case DpctInclusionInfo::IF_Replace:
      insertHeaders(FileInfo, Info.Headers);
      LLVM_FALLTHROUGH;
    case DpctInclusionInfo::IF_Remove:
      RemoveInslusion();
      break;
    case DpctInclusionInfo::IF_MarkInserted:
      setHeadersAsInserted(FileInfo, Info.Headers);
    case DpctInclusionInfo::IF_DoNothing:
    default:
      break;
    }
    return;
  } while (false);

  auto &CudaPath = Global.getCudaPath();
  // TODO: implement one of this for each source language.
  // Remove all includes from the SDK.
  if (isChildOrSamePath(CudaPath, SearchPath.str()) ||
      SearchPath.startswith("/usr/local/cuda")) {
    // If CudaPath is in /usr/include,
    // for all the include files without starting with specified string, keep it
    if (!StringRef(CudaPath).startswith("/usr/include") ||
        isAlwaysRemoved(FileName)) {
      RemoveInslusion();
    }
  }
}

const DpctInclusionInfo *
DpctInclusionHeadersMap::findHeaderInfo(llvm::StringRef IncludeFile) {
  if (auto Info = findInFullMatcheMode(IncludeFile))
    return Info;
  if (auto Info = findInStartwithMode(IncludeFile))
    return Info;
  return nullptr;
}

template <class... Args>
void DpctInclusionHeadersMap::registInclusionHeaderEntry(
    StringRef Filename, MatchMode Mode, RuleGroupKind Group,
    DpctInclusionInfo::InclusionFlag Flag, bool MustAngled, Args... Headers) {
  DpctInclusionInfo *Info;
  switch (Mode) {
  case MatchMode::Mode_FullMatch:
    Info = &InclusionFullMatchMap[Filename];
    break;
  case MatchMode::Mode_Startwith:
    Info = &InclusionStartWithMap.emplace_back(Filename).Info;
    break;
  default:
    return;
  }
  Info->RuleGroup = Group;
  Info->ProcessFlag = Flag;
  Info->MustAngled = MustAngled;
  Info->Headers.assign({ Headers... });
  return;
}

DpctInclusionHeadersMap::DpctInclusionHeadersMapInitializer::
    DpctInclusionHeadersMapInitializer() {
#define REGIST_INCLUSION(FILE, MODE, GROUP, FLAG, ...)                         \
  registInclusionHeaderEntry(FILE, DpctInclusionHeadersMap::Mode_##MODE,       \
                             RuleGroupKind::RK_##GROUP,                        \
                             DpctInclusionInfo::IF_##FLAG, __VA_ARGS__);
#include "InclusionHeaders.inc"
}

DpctInclusionHeadersMap::DpctInclusionHeadersMapInitializer
    DpctInclusionHeadersMap::Initializer;

} // namespace dpct
} // namespace clang