//===--- CustomHelperFiles.cpp ---------------------------*- C++ -*---===//
//
// Copyright (C) 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "CustomHelperFiles.h"

#include "AnalysisInfo.h"
#include "ASTTraversal.h"
#include "Config.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

#include <fstream>

namespace clang {
namespace dpct {

void requestFeature(clang::dpct::HelperFileEnum FileID,
                           std::string HelperFunctionName,
                           const std::string &UsedFile) {
  auto Key = std::make_pair(FileID, HelperFunctionName);
  auto Iter = MapNames::HelperNameContentMap.find(Key);
  if (Iter != MapNames::HelperNameContentMap.end()) {
    Iter->second.IsCalled = true;
    Iter->second.CallerSrcFiles.insert(UsedFile);
  }
}
void requestFeature(clang::dpct::HelperFileEnum FileID,
                           std::string HelperFunctionName, SourceLocation SL) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto ExpansionLoc = SM.getExpansionLoc(SL);

  std::string UsedFile = "";
  if (ExpansionLoc.isValid())
    UsedFile = dpct::DpctGlobalInfo::getLocInfo(ExpansionLoc).first;
  requestFeature(FileID, HelperFunctionName, UsedFile);
}
void requestFeature(clang::dpct::HelperFileEnum FileID,
                           std::string HelperFunctionName, const Stmt *Stmt) {
  if (!Stmt)
    return;
  requestFeature(FileID, HelperFunctionName, Stmt->getBeginLoc());
}
void requestFeature(clang::dpct::HelperFileEnum FileID,
                           std::string HelperFunctionName, const Decl *Decl) {
  if (!Decl)
    return;
  requestFeature(FileID, HelperFunctionName, Decl->getBeginLoc());
}

std::string getCopyrightHeader(const clang::dpct::HelperFileEnum File) {
  std::string CopyrightHeader =
      MapNames::HelperNameContentMap.at(std::make_pair(File, "License")).Code;
  replaceEndOfLine(CopyrightHeader);
  return CopyrightHeader;
}

std::pair<std::string, std::string>
getHeaderGuardPair(const clang::dpct::HelperFileEnum File) {
  std::string MacroName =
      MapNames::HelperFileHeaderGuardMacroMap.find(File)->second;
  std::pair<std::string, std::string> Pair;
  Pair.first =
      "#ifndef " + MacroName + getNL() + "#define " + MacroName + getNL();
  Pair.second = "#endif // " + MacroName;
  return Pair;
}

void addDependencyIncludeDirectives(
    const clang::dpct::HelperFileEnum FileID,
    std::vector<clang::dpct::HelperFunc> &ContentVec) {

  auto isDplFile = [](clang::dpct::HelperFileEnum FileID) -> bool {
    if (FileID == clang::dpct::HelperFileEnum::DplExtrasAlgorithm ||
        FileID == clang::dpct::HelperFileEnum::DplExtrasFunctional ||
        FileID == clang::dpct::HelperFileEnum::DplExtrasIterators ||
        FileID == clang::dpct::HelperFileEnum::DplExtrasMemory ||
        FileID == clang::dpct::HelperFileEnum::DplExtrasNumeric ||
        FileID == clang::dpct::HelperFileEnum::DplExtrasVector) {
      return true;
    }
    return false;
  };

  bool IsCurrentFileInDpExtra = isDplFile(FileID);

  auto Iter = MapNames::HelperNameContentMap.find(
      std::make_pair(FileID, "local_include_dependency"));
  if (Iter == MapNames::HelperNameContentMap.end())
    return;

  auto Content = Iter->second;

  std::set<clang::dpct::HelperFileEnum> FileDependency;
  for (const auto &Item : ContentVec) {
    for (const auto &Pair : Item.Dependency) {
      FileDependency.insert(Pair.first);
    }
  }
  std::string Directives;
  for (const auto &Item : FileDependency) {
    if (Item == FileID)
      continue;
    if (IsCurrentFileInDpExtra) {
      if (isDplFile(Item))
        Directives = Directives + "#include \"" +
                     MapNames::HelperFileNameMap.at(Item) + "\"" + getNL();
      else
        Directives = Directives + "#include \"../" +
                     MapNames::HelperFileNameMap.at(Item) + "\"" + getNL();
    } else {
      Directives = Directives + "#include \"" +
                   MapNames::HelperFileNameMap.at(Item) + "\"" + getNL();
    }
  }
  Content.Code = Directives;
  ContentVec.push_back(Content);
}

std::string
getHelperFileContent(const clang::dpct::HelperFileEnum File,
               std::vector<clang::dpct::HelperFunc> ContentVec) {
  if (ContentVec.empty())
    return "";

  std::string ContentStr;

  ContentStr = ContentStr + getCopyrightHeader(File) + getNL();
  ContentStr = ContentStr + getHeaderGuardPair(File).first + getNL();

  if (File != clang::dpct::HelperFileEnum::Dpct &&
      File != clang::dpct::HelperFileEnum::DplUtils) {
    // For Dpct and DplUtils, the include directives are determined
    // by other files.
    addDependencyIncludeDirectives(File, ContentVec);
  }

  auto CompareAsc = [](clang::dpct::HelperFunc A, clang::dpct::HelperFunc B) {
    return A.PositionIdx < B.PositionIdx;
  };
  std::sort(ContentVec.begin(), ContentVec.end(), CompareAsc);

  std::string CurrentNamespace;
  for (const auto &Item : ContentVec) {
    if (Item.Namespace.empty()) {
      // no namespace
      if (CurrentNamespace == "dpct") {
        ContentStr = ContentStr + "} // namespace dpct" + getNL() + getNL();
      } else if (CurrentNamespace == "dpct::detail") {
        ContentStr = ContentStr + "} // namespace detail" + getNL() + getNL();
        ContentStr = ContentStr + "} // namespace dpct" + getNL() + getNL();
      } else if (CurrentNamespace == "dpct::internal") {
        ContentStr =
            ContentStr + "} // namespace internal" + getNL() + getNL();
        ContentStr = ContentStr + "} // namespace dpct" + getNL() + getNL();
      }
      CurrentNamespace = "";
      std::string Code = Item.Code;
      replaceEndOfLine(Code);
      ContentStr = ContentStr + Code + getNL();
    } else if (Item.Namespace == "dpct") {
      // dpct namespace
      if (CurrentNamespace.empty()) {
        ContentStr = ContentStr + "namespace dpct {" + getNL() + getNL();
      } else if (CurrentNamespace == "dpct::detail") {
        ContentStr = ContentStr + "} // namespace detail" + getNL() + getNL();
      } else if (CurrentNamespace == "dpct::internal") {
        ContentStr = ContentStr + "} // namespace internal" + getNL() + getNL();
      }
      CurrentNamespace = "dpct";
      std::string Code = Item.Code;
      replaceEndOfLine(Code);
      ContentStr = ContentStr + Code + getNL();
    } else if (Item.Namespace == "dpct::detail") {
      // dpct::detail namespace
      if (CurrentNamespace.empty()) {
        ContentStr = ContentStr + "namespace dpct {" + getNL() + getNL();
        ContentStr = ContentStr + "namespace detail {" + getNL() + getNL();
      } else if (CurrentNamespace == "dpct") {
        ContentStr = ContentStr + "namespace detail {" + getNL() + getNL();
      }
      CurrentNamespace = "dpct::detail";
      std::string Code = Item.Code;
      replaceEndOfLine(Code);
      ContentStr = ContentStr + Code + getNL();
    } else if (Item.Namespace == "dpct::internal") {
      // dpct::internal namespace
      if (CurrentNamespace.empty()) {
        ContentStr = ContentStr + "namespace dpct {" + getNL() + getNL();
        ContentStr = ContentStr + "namespace internal {" + getNL() + getNL();
      } else if (CurrentNamespace == "dpct") {
        ContentStr = ContentStr + "namespace internal {" + getNL() + getNL();
      }
      CurrentNamespace = "dpct::internal";
      std::string Code = Item.Code;
      replaceEndOfLine(Code);
      ContentStr = ContentStr + Code + getNL();
    }
  }

  if (CurrentNamespace == "dpct") {
    ContentStr = ContentStr + "} // namespace dpct" + getNL() + getNL();
  } else if (CurrentNamespace == "dpct::detail") {
    ContentStr = ContentStr + "} // namespace detail" + getNL() + getNL();
    ContentStr = ContentStr + "} // namespace dpct" + getNL() + getNL();
  } else if (CurrentNamespace == "dpct::internal") {
    ContentStr = ContentStr + "} // namespace internal" + getNL() + getNL();
    ContentStr = ContentStr + "} // namespace dpct" + getNL() + getNL();
  }

  ContentStr = ContentStr + getHeaderGuardPair(File).second + getNL();
  return ContentStr;
}

std::string getDpctVersionStr() {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << DPCT_VERSION_MAJOR << "." << DPCT_VERSION_MINOR << "." <<
         DPCT_VERSION_PATCH;
  return OS.str();
}

void emitDpctVersionWarningIfNeed(const std::string &VersionFromYaml) {
  // If yaml file does not exist, this function will not be called.
  std::string CurrentToolVersion;
  llvm::raw_string_ostream OS(CurrentToolVersion);
  OS << DPCT_VERSION_MAJOR << "." << DPCT_VERSION_MINOR << "."
     << DPCT_VERSION_PATCH;
  OS.flush();

  if (VersionFromYaml.empty()) {
    // This is an increamental migration, and the previous migration used 2021
    // gold update1
    clang::dpct::PrintMsg(
        "NOTE: This is an incremental migration. Previous version of the tool "
        "used: 2021.2.0, current version: " +
        CurrentToolVersion + "." + getNL());
  } else if (VersionFromYaml != CurrentToolVersion) {
    // This is an increamental migration, and the previous migration used gold
    // version
    clang::dpct::PrintMsg(
        "NOTE: This is an incremental migration. Previous version of the tool "
        "used: " +
        VersionFromYaml + ", current version: " + CurrentToolVersion + "." +
        getNL());
  }
  // No previous migration, or previous migration using the same tool version:
  // no warning emitted.
}

void generateAllHelperFiles() {
  std::string ToPath = clang::dpct::DpctGlobalInfo::getOutRoot() + "/include";
  if (!llvm::sys::fs::is_directory(ToPath))
    llvm::sys::fs::create_directory(Twine(ToPath));
  ToPath = ToPath + "/" + getCustomMainHelperFileName();
  if (!llvm::sys::fs::is_directory(ToPath))
    llvm::sys::fs::create_directory(Twine(ToPath));
  if (!llvm::sys::fs::is_directory(Twine(ToPath + "/dpl_extras")))
    llvm::sys::fs::create_directory(Twine(ToPath + "/dpl_extras"));

#define GENERATE_ALL_FILE_CONTENT(FILE_NAME)                                   \
  {                                                                            \
    std::ofstream FILE_NAME##File(                                             \
        ToPath + "/" +                                                         \
            MapNames::HelperFileNameMap.at(                                    \
                clang::dpct::HelperFileEnum::FILE_NAME),                       \
        std::ios::binary);                                                     \
    std::string Code = MapNames::FILE_NAME##AllContentStr;                     \
    replaceEndOfLine(Code);                                               \
    FILE_NAME##File << Code;                                                   \
    FILE_NAME##File.flush();                                                   \
  }
#define GENERATE_DPL_EXTRAS_All_FILE_CONTENT(FILE_NAME)                        \
  {                                                                            \
    std::ofstream FILE_NAME##File(                                             \
        ToPath + "/dpl_extras/" +                                              \
            MapNames::HelperFileNameMap.at(                                    \
                clang::dpct::HelperFileEnum::FILE_NAME),                       \
        std::ios::binary);                                                     \
    std::string Code = MapNames::FILE_NAME##AllContentStr;                     \
    replaceEndOfLine(Code);                                               \
    FILE_NAME##File << Code;                                                   \
    FILE_NAME##File.flush();                                                   \
  }
  GENERATE_ALL_FILE_CONTENT(Atomic)
  GENERATE_ALL_FILE_CONTENT(BlasUtils)
  GENERATE_ALL_FILE_CONTENT(Device)
  GENERATE_ALL_FILE_CONTENT(Dpct)
  GENERATE_ALL_FILE_CONTENT(DplUtils)
  GENERATE_ALL_FILE_CONTENT(Image)
  GENERATE_ALL_FILE_CONTENT(Kernel)
  GENERATE_ALL_FILE_CONTENT(Memory)
  GENERATE_ALL_FILE_CONTENT(Util)
  GENERATE_DPL_EXTRAS_All_FILE_CONTENT(DplExtrasAlgorithm)
  GENERATE_DPL_EXTRAS_All_FILE_CONTENT(DplExtrasFunctional)
  GENERATE_DPL_EXTRAS_All_FILE_CONTENT(DplExtrasIterators)
  GENERATE_DPL_EXTRAS_All_FILE_CONTENT(DplExtrasMemory)
  GENERATE_DPL_EXTRAS_All_FILE_CONTENT(DplExtrasNumeric)
  GENERATE_DPL_EXTRAS_All_FILE_CONTENT(DplExtrasVector)
#undef GENERATE_ALL_FILE_CONTENT
#undef GENERATE_DPL_EXTRAS_All_FILE_CONTENT
}

void generateHelperFunctions() {
  auto getUsedAPINum = []() -> size_t {
    size_t Res = 0;
    for (const auto &Item : MapNames::HelperNameContentMap) {
      if (Item.second.IsCalled)
        Res++;
    }
    return Res;
  };

  // dpct.hpp is always exsit, so request its non_local_include_dependency
  // feature
  requestFeature(dpct::HelperFileEnum::Dpct, "non_local_include_dependency",
                 "");
  // 1. add dependent APIs
  size_t UsedAPINum = getUsedAPINum();
  do {
    UsedAPINum = getUsedAPINum();
    std::set<std::pair<std::pair<clang::dpct::HelperFileEnum, std::string>,
                          std::set<std::string>>>
        NeedInsert;
    for (const auto &Item : MapNames::HelperNameContentMap) {
      if (Item.second.IsCalled) {
        for (const auto &DepItem : Item.second.Dependency) {
          NeedInsert.insert(std::make_pair(DepItem, Item.second.CallerSrcFiles));
        }
      }
    }
    for (const auto &Item : NeedInsert) {
      auto Iter = MapNames::HelperNameContentMap.find(Item.first);
      if (Iter != MapNames::HelperNameContentMap.end()) {
        Iter->second.IsCalled = true;
        Iter->second.CallerSrcFiles.insert(Item.second.begin(),
                                           Item.second.end());
      }
    }
  } while (getUsedAPINum() > UsedAPINum);

  // 2. build info of necessary headers to out-root
  if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
      HelperFilesCustomizationLevel::none)
    return;
  else if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
           HelperFilesCustomizationLevel::all) {
    generateAllHelperFiles();
    return;
  }

  std::vector<clang::dpct::HelperFunc> AtomicFileContent;
  std::vector<clang::dpct::HelperFunc> BlasUtilsFileContent;
  std::vector<clang::dpct::HelperFunc> DeviceFileContent;
  std::vector<clang::dpct::HelperFunc> DpctFileContent;
  std::vector<clang::dpct::HelperFunc> DplUtilsFileContent;
  std::vector<clang::dpct::HelperFunc> ImageFileContent;
  std::vector<clang::dpct::HelperFunc> KernelFileContent;
  std::vector<clang::dpct::HelperFunc> MemoryFileContent;
  std::vector<clang::dpct::HelperFunc> UtilFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasAlgorithmFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasFunctionalFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasIteratorsFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasMemoryFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasNumericFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasVectorFileContent;

  std::vector<bool> FileUsedFlagVec(
      (unsigned int)clang::dpct::HelperFileEnum::HelperFileEnumTypeSize, false);
  if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
      HelperFilesCustomizationLevel::file) {
    // E.g., user code uses API2.
    // HelperFileA: API1(depends on API3), API2
    // HelperFileB: API3
    // In step1, only API2 is enabled. But current config is file, so API1 and
    // API2 are both printed, then we also need print API3.
    // But API1 and API3 are not set "IsCalled" flag, just insert elements into
    // content vector.
    auto getUsedFileNum = [&]() -> size_t {
      size_t Res = 0;
      for (const auto &Item : FileUsedFlagVec) {
        if (Item)
          Res++;
      }
      return Res;
    };

    for (const auto &Item : MapNames::HelperNameContentMap)
      if (Item.second.IsCalled)
        FileUsedFlagVec[size_t(Item.first.first)] = true;
    size_t UsedFileNum = getUsedFileNum();
    do {
      UsedFileNum = getUsedFileNum();
      for (unsigned int FileID = 0;
           FileID < (unsigned int)dpct::HelperFileEnum::HelperFileEnumTypeSize;
           ++FileID) {
        if (!FileUsedFlagVec[FileID])
          continue;
        for (const auto &Item : MapNames::HelperNameContentMap) {
          if (Item.first.first == (dpct::HelperFileEnum)FileID) {
            for (const auto &Dep : Item.second.Dependency) {
              FileUsedFlagVec[(unsigned int)Dep.first] = true;
            }
          }
        }
      }
    } while (getUsedFileNum() > UsedFileNum);
  }

#define UPDATE_FILE(FILENAME)                                                  \
  case clang::dpct::HelperFileEnum::FILENAME:                                  \
    FILENAME##FileContent.push_back(Item.second);                              \
    break;

  for (const auto &Item : MapNames::HelperNameContentMap) {
    if (Item.first.second == "local_include_dependency") {
      // local_include_dependency for dpct and dpl_utils is inserted in step3
      // local_include_dependency for others are inserted in getHelperFileContent()
      continue;
    } else if (Item.first.second == "non_local_include_dependency") {
      // non_local_include_dependency for dpct is inserted here
      // non_local_include_dependency for others is inserted in step3
      if (Item.first.first == clang::dpct::HelperFileEnum::Dpct) {
        DpctFileContent.push_back(Item.second);
      }
      continue;
    } else if (clang::dpct::DpctGlobalInfo::
                   getHelperFilesCustomizationLevel() ==
               HelperFilesCustomizationLevel::file) {
      if (!FileUsedFlagVec[size_t(Item.first.first)])
        continue;
      if (Item.first.second == "License")
        continue;
    }

    switch (Item.first.first) {
      UPDATE_FILE(Atomic)
      UPDATE_FILE(BlasUtils)
      UPDATE_FILE(Device)
      UPDATE_FILE(Dpct)
      UPDATE_FILE(DplUtils)
      UPDATE_FILE(Image)
      UPDATE_FILE(Kernel)
      UPDATE_FILE(Memory)
      UPDATE_FILE(Util)
      UPDATE_FILE(DplExtrasAlgorithm)
      UPDATE_FILE(DplExtrasFunctional)
      UPDATE_FILE(DplExtrasIterators)
      UPDATE_FILE(DplExtrasMemory)
      UPDATE_FILE(DplExtrasNumeric)
      UPDATE_FILE(DplExtrasVector)
    default:
      assert(0 && "unknown helper file ID");
    }
  }
#undef UPDATE_FILE

  // 3. prepare folder and insert non_local_include_dependency/local_include_dependency
  std::string ToPath = clang::dpct::DpctGlobalInfo::getOutRoot() + "/include";
  if (!llvm::sys::fs::is_directory(ToPath))
    llvm::sys::fs::create_directory(Twine(ToPath));
  ToPath = ToPath + "/" + getCustomMainHelperFileName();
  if (!llvm::sys::fs::is_directory(ToPath))
    llvm::sys::fs::create_directory(Twine(ToPath));
  if (!DplExtrasAlgorithmFileContent.empty() ||
      !DplExtrasFunctionalFileContent.empty() ||
      !DplExtrasIteratorsFileContent.empty() ||
      !DplExtrasMemoryFileContent.empty() ||
      !DplExtrasNumericFileContent.empty() ||
      !DplExtrasVectorFileContent.empty()) {
    if (!llvm::sys::fs::is_directory(Twine(ToPath + "/dpl_extras")))
      llvm::sys::fs::create_directory(Twine(ToPath + "/dpl_extras"));

    std::string IDDStr;

#define ADD_INCLUDE_DIRECTIVE_FOR_DPL(FILENAME)                                \
  if (!FILENAME##FileContent.empty()) {                                        \
    FILENAME##FileContent.push_back(                                           \
        MapNames::HelperNameContentMap.at(std::make_pair(                      \
            clang::dpct::HelperFileEnum::FILENAME, "non_local_include_dependency")));     \
    IDDStr = IDDStr + "#include \"dpl_extras/" +                               \
             MapNames::HelperFileNameMap.at(                                   \
                 clang::dpct::HelperFileEnum::FILENAME) +                      \
             "\"" + getNL();                                                   \
  }
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasAlgorithm)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasFunctional)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasIterators)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasMemory)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasNumeric)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasVector)
#undef ADD_INCLUDE_DIRECTIVE_FOR_DPL

    auto Item = MapNames::HelperNameContentMap.at(
        std::make_pair(clang::dpct::HelperFileEnum::DplUtils,
                       "local_include_dependency"));
    Item.Code = IDDStr;
    DplUtilsFileContent.push_back(Item);
  }

  if (!DplUtilsFileContent.empty() ||
      MapNames::HelperNameContentMap
          .at(std::make_pair(clang::dpct::HelperFileEnum::DplUtils,
                             "non_local_include_dependency"))
          .IsCalled) {
    DplUtilsFileContent.push_back(MapNames::HelperNameContentMap.at(
        std::make_pair(clang::dpct::HelperFileEnum::DplUtils,
                       "non_local_include_dependency")));
  }

  DpctFileContent.push_back(
      MapNames::HelperNameContentMap.at(std::make_pair(
          clang::dpct::HelperFileEnum::Dpct, "non_local_include_dependency")));
  std::string IDDStr;

#define ADD_INCLUDE_DIRECTIVE(FILENAME)                                        \
  if (!FILENAME##FileContent.empty()) {                                        \
    FILENAME##FileContent.push_back(                                           \
        MapNames::HelperNameContentMap.at(std::make_pair(                      \
            clang::dpct::HelperFileEnum::FILENAME, "non_local_include_dependency")));     \
    IDDStr = IDDStr + "#include \"" +                                          \
             MapNames::HelperFileNameMap.at(                                   \
                 clang::dpct::HelperFileEnum::FILENAME) +                      \
             "\"" + getNL();                                                   \
  }
  ADD_INCLUDE_DIRECTIVE(Atomic)
  ADD_INCLUDE_DIRECTIVE(BlasUtils)
  ADD_INCLUDE_DIRECTIVE(Device)
  // Do not include dpl_utils in dpct.hpp, since there is a bug in dpl_extras files.
  // All those functions are without the "inline" specifier, so there will be a multi
  // definition issue.
  // ADD_INCLUDE_DIRECTIVE(DplUtils)
  ADD_INCLUDE_DIRECTIVE(Image)
  ADD_INCLUDE_DIRECTIVE(Kernel)
  ADD_INCLUDE_DIRECTIVE(Memory)
  ADD_INCLUDE_DIRECTIVE(Util)
#undef ADD_INCLUDE_DIRECTIVE

  auto Item = MapNames::HelperNameContentMap.at(std::make_pair(
      clang::dpct::HelperFileEnum::Dpct, "local_include_dependency"));
  Item.Code = IDDStr;
  DpctFileContent.push_back(Item);

  // 4. generate headers to out-root
#define GENERATE_FILE(FILE_NAME)                                               \
  if (!FILE_NAME##FileContent.empty()) {                                       \
    std::string FILE_NAME##FileContentStr = getHelperFileContent(              \
        clang::dpct::HelperFileEnum::FILE_NAME, FILE_NAME##FileContent);       \
    std::ofstream FILE_NAME##File(                                             \
        ToPath + "/" +                                                         \
            MapNames::HelperFileNameMap.at(                                    \
                clang::dpct::HelperFileEnum::FILE_NAME),                       \
        std::ios::binary);                                                     \
    FILE_NAME##File << FILE_NAME##FileContentStr;                              \
    FILE_NAME##File.flush();                                                   \
  }
#define GENERATE_DPL_EXTRAS_FILE(FILE_NAME)                                    \
  if (!FILE_NAME##FileContent.empty()) {                                       \
    std::string FILE_NAME##FileContentStr = getHelperFileContent(              \
        clang::dpct::HelperFileEnum::FILE_NAME, FILE_NAME##FileContent);       \
    std::ofstream FILE_NAME##File(                                             \
        ToPath + "/dpl_extras/" +                                              \
            MapNames::HelperFileNameMap.at(                                    \
                clang::dpct::HelperFileEnum::FILE_NAME),                       \
        std::ios::binary);                                                     \
    FILE_NAME##File << FILE_NAME##FileContentStr;                              \
    FILE_NAME##File.flush();                                                   \
  }
  GENERATE_FILE(Atomic)
  GENERATE_FILE(BlasUtils)
  GENERATE_FILE(Device)
  GENERATE_FILE(Dpct)
  GENERATE_FILE(DplUtils)
  GENERATE_FILE(Image)
  GENERATE_FILE(Kernel)
  GENERATE_FILE(Memory)
  GENERATE_FILE(Util)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasAlgorithm)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasFunctional)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasIterators)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasMemory)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasNumeric)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasVector)
#undef GENERATE_FILE
#undef GENERATE_DPL_EXTRAS_FILE
}

#define ADD_HELPER_FEATURE_FOR_ENUM_NAMES(TYPE)                                \
  void requestHelperFeatureForEnumNames(const std::string Name, TYPE File) {   \
    auto HelperFeatureIter =                                                   \
        clang::dpct::EnumConstantRule::EnumNamesHelperFeaturesMap.find(Name);  \
    if (HelperFeatureIter !=                                                   \
        clang::dpct::EnumConstantRule::EnumNamesHelperFeaturesMap.end()) {     \
      requestFeature(                             \
          HelperFeatureIter->second.first, HelperFeatureIter->second.second,   \
          File);                                                               \
    }                                                                          \
  }
#define ADD_HELPER_FEATURE_FOR_TYPE_NAMES(TYPE)                                \
  void requestHelperFeatureForTypeNames(const std::string Name, TYPE File) {   \
    auto HelperFeatureIter = MapNames::TypeNamesHelperFeaturesMap.find(Name);  \
    if (HelperFeatureIter != MapNames::TypeNamesHelperFeaturesMap.end()) {     \
      requestFeature(                             \
          HelperFeatureIter->second.first, HelperFeatureIter->second.second,   \
          File);                                                               \
    }                                                                          \
  }
ADD_HELPER_FEATURE_FOR_ENUM_NAMES(const std::string)
ADD_HELPER_FEATURE_FOR_ENUM_NAMES(SourceLocation)
ADD_HELPER_FEATURE_FOR_ENUM_NAMES(const Stmt *)
ADD_HELPER_FEATURE_FOR_ENUM_NAMES(const Decl *)
ADD_HELPER_FEATURE_FOR_TYPE_NAMES(const std::string)
ADD_HELPER_FEATURE_FOR_TYPE_NAMES(SourceLocation)
ADD_HELPER_FEATURE_FOR_TYPE_NAMES(const Stmt *)
ADD_HELPER_FEATURE_FOR_TYPE_NAMES(const Decl *)
#undef ADD_HELPER_FEATURE_FOR_ENUM_NAMES
#undef ADD_HELPER_FEATURE_FOR_TYPE_NAMES

std::string getCustomMainHelperFileName() {
  return dpct::DpctGlobalInfo::getCustomHelperFileName();
}

  // update MapNames::HelperNameContentMap from TUR
void updateHelperNameContentMap(
    const clang::tooling::TranslationUnitReplacements &TUR) {
#define UPDATE_MAP_INFO(FILEID)                                                \
  for (auto &Entry : TUR.FILEID##HelperFuncMap) {                              \
    std::pair<HelperFileEnum, std::string> Key(HelperFileEnum::FILEID,         \
                                               Entry.first);                   \
    MapNames::HelperNameContentMap[Key].IsCalled =                             \
        MapNames::HelperNameContentMap[Key].IsCalled || Entry.second.IsCalled; \
    for (auto &CallerFileName : Entry.second.CallerSrcFiles) {                 \
      MapNames::HelperNameContentMap[Key].CallerSrcFiles.insert(               \
          CallerFileName);                                                     \
    }                                                                          \
  }
  UPDATE_MAP_INFO(Atomic)
  UPDATE_MAP_INFO(BlasUtils)
  UPDATE_MAP_INFO(Device)
  UPDATE_MAP_INFO(Dpct)
  UPDATE_MAP_INFO(DplUtils)
  UPDATE_MAP_INFO(Image)
  UPDATE_MAP_INFO(Kernel)
  UPDATE_MAP_INFO(Memory)
  UPDATE_MAP_INFO(Util)
  UPDATE_MAP_INFO(DplExtrasAlgorithm)
  UPDATE_MAP_INFO(DplExtrasFunctional)
  UPDATE_MAP_INFO(DplExtrasIterators)
  UPDATE_MAP_INFO(DplExtrasMemory)
  UPDATE_MAP_INFO(DplExtrasNumeric)
  UPDATE_MAP_INFO(DplExtrasVector)
#undef UPDATE_MAP_INFO
}

  // update TUR from MapNames::HelperNameContentMap
void updateTUR(
    clang::tooling::TranslationUnitReplacements &TUR) {
#define UPDATE_TUR_INFO(FILEID)                                                \
  case HelperFileEnum::FILEID:                                                 \
    TUR.FILEID##HelperFuncMap[Entry.first.second].IsCalled =                   \
        Entry.second.IsCalled;                                                 \
    TUR.FILEID##HelperFuncMap[Entry.first.second].CallerSrcFiles.clear();      \
    for (auto CallerFileName : Entry.second.CallerSrcFiles) {                  \
      TUR.FILEID##HelperFuncMap[Entry.first.second].CallerSrcFiles.push_back(  \
          CallerFileName);                                                     \
    }                                                                          \
    break;

  for (auto Entry : MapNames::HelperNameContentMap) {
    switch (Entry.first.first) {
      UPDATE_TUR_INFO(Atomic)
      UPDATE_TUR_INFO(BlasUtils)
      UPDATE_TUR_INFO(Device)
      UPDATE_TUR_INFO(Dpct)
      UPDATE_TUR_INFO(DplUtils)
      UPDATE_TUR_INFO(Image)
      UPDATE_TUR_INFO(Kernel)
      UPDATE_TUR_INFO(Memory)
      UPDATE_TUR_INFO(Util)
      UPDATE_TUR_INFO(DplExtrasAlgorithm)
      UPDATE_TUR_INFO(DplExtrasFunctional)
      UPDATE_TUR_INFO(DplExtrasIterators)
      UPDATE_TUR_INFO(DplExtrasMemory)
      UPDATE_TUR_INFO(DplExtrasNumeric)
      UPDATE_TUR_INFO(DplExtrasVector)
    default:
      dpct_unreachable("unknown helper file ID");
    }
  }
#undef UPDATE_TUR_INFO
}

void replaceAllOccurredStrsInStr(std::string &StrNeedProcess,
                                 const std::string &Pattern,
                                 const std::string &Repl) {
  if (StrNeedProcess.empty() || Pattern.empty()) {
    return;
  }

  size_t PatternLen = Pattern.size();
  size_t ReplLen = Repl.size();
  size_t Offset = 0;
  Offset = StrNeedProcess.find(Pattern, Offset);

  while (Offset != std::string::npos) {
    StrNeedProcess.replace(Offset, PatternLen, Repl);
    Offset = Offset + ReplLen;
    Offset = StrNeedProcess.find(Pattern, Offset);
  }
}

void replaceEndOfLine(std::string &StrNeedProcess) {
#ifdef _WIN64
  replaceAllOccurredStrsInStr(StrNeedProcess, "\n", "\r\n");
#endif
}

} // namespace dpct
} // namespace clang