//===--------------- CustomHelperFiles.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CustomHelperFiles.h"

#include "ASTTraversal.h"
#include "DNNAPIMigration.h"
#include "AnalysisInfo.h"
#include "LIBCUAPIMigration.h"
#include "Config.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

#include <fstream>

namespace clang {
namespace dpct {

void requestFeature(HelperFeatureEnum Feature, const std::string &UsedFile) {
  if (Feature == HelperFeatureEnum::no_feature_helper) {
    return;
  }
  if (!HelperFeatureEnumPairMap.count(Feature)) {
#ifdef DPCT_DEBUG_BUILD
    std::cout << "Unknown feature enum:" << (unsigned int)Feature << std::endl;
    assert(0 && "Unknown requested feature.\n");
#endif
  }
  auto Key = HelperFeatureEnumPairMap.at(Feature);
  auto Iter = HelperNameContentMap.find(Key);
  if (Iter != HelperNameContentMap.end()) {
    Iter->second.IsCalled = true;
    Iter->second.CallerSrcFiles.insert(UsedFile);
  } else {
#ifdef DPCT_DEBUG_BUILD
    std::cout << "Unknown feature: File:" << (unsigned int)Key.first
              << ", Feature:" << Key.second << std::endl;
    assert(0 && "Unknown requested feature.\n");
#endif
  }
}
void requestFeature(HelperFeatureEnum Feature, SourceLocation SL) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager();
  auto ExpansionLoc = SM.getExpansionLoc(SL);

  std::string UsedFile = "";
  if (ExpansionLoc.isValid())
    UsedFile = dpct::DpctGlobalInfo::getLocInfo(ExpansionLoc).first;
  requestFeature(Feature, UsedFile);
}
void requestFeature(HelperFeatureEnum Feature, const Stmt *Stmt) {
  if (!Stmt)
    return;
  requestFeature(Feature, Stmt->getBeginLoc());
}
void requestFeature(HelperFeatureEnum Feature, const Decl *Decl) {
  if (!Decl)
    return;
  requestFeature(Feature, Decl->getBeginLoc());
}

std::string getCopyrightHeader(const clang::dpct::HelperFileEnum File) {
  std::string CopyrightHeader =
      HelperNameContentMap.at(std::make_pair(File, "License")).Code;
  if (File == HelperFileEnum::Dpct) {
    std::string Prefix = "//==----";
    std::string Suffix = "-*- C++ -*----------------==//";
    std::string FileName = " " + getCustomMainHelperFileName() + ".hpp ";
    const size_t ColumnLimit = 80;
    size_t NumOfDashes = 0;
    if (Prefix.size() + Suffix.size() + FileName.size() <= ColumnLimit) {
      NumOfDashes =
          ColumnLimit - Prefix.size() - Suffix.size() - FileName.size();
    }

    CopyrightHeader = Prefix + FileName + std::string(NumOfDashes, '-') +
                      Suffix + "\n" + CopyrightHeader;
  }
  replaceEndOfLine(CopyrightHeader);
  return CopyrightHeader;
}

std::pair<std::string, std::string>
getHeaderGuardPair(const clang::dpct::HelperFileEnum File) {
  std::string MacroName = "";
  if (File == HelperFileEnum::Dpct && getCustomMainHelperFileName() != "dpct") {
    MacroName = getCustomMainHelperFileName();
    for (size_t Idx = 0; Idx < MacroName.size(); ++Idx)
      MacroName[Idx] = llvm::toUpper(MacroName[Idx]);
    MacroName = "__" + MacroName + "_HPP__";
  } else {
    MacroName = HelperFileHeaderGuardMacroMap.find(File)->second;
  }
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
        FileID == clang::dpct::HelperFileEnum::DplExtrasVector ||
        FileID == clang::dpct::HelperFileEnum::DplExtrasDpcppExtensions) {
      return true;
    }
    return false;
  };

  bool IsCurrentFileInDplExtra = isDplFile(FileID);

  auto Iter = HelperNameContentMap.find(
      std::make_pair(FileID, "local_include_dependency"));
  if (Iter == HelperNameContentMap.end())
    return;

  auto Content = Iter->second;

  std::set<clang::dpct::HelperFileEnum> FileDependency;
  for (const auto &Item : ContentVec) {
    for (const auto &Pair : Item.Dependency) {
      if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
          HelperFilesCustomizationLevel::HFCL_API) {
        if (Pair.second == HelperFeatureDependencyKind::HFDK_UsmNone) {
          if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
            FileDependency.insert(Pair.first.first);
        } else if (Pair.second ==
                   HelperFeatureDependencyKind::HFDK_UsmRestricted) {
          if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted)
            FileDependency.insert(Pair.first.first);
        } else {
          FileDependency.insert(Pair.first.first);
        }
      } else {
        FileDependency.insert(Pair.first.first);
      }
    }
  }
  std::string Directives;
  for (const auto &Item : FileDependency) {
    if (Item == FileID)
      continue;
    if (IsCurrentFileInDplExtra) {
      if (isDplFile(Item))
        Directives = Directives + "#include \"" + HelperFileNameMap.at(Item) +
                     "\"" + getNL();
      else
        Directives = Directives + "#include \"../" +
                     HelperFileNameMap.at(Item) + "\"" + getNL();
    } else {
      Directives = Directives + "#include \"" + HelperFileNameMap.at(Item) +
                   "\"" + getNL();
    }
  }
  Content.Code = Directives;
  ContentVec.push_back(Content);
}

std::string getCode(const HelperFunc &Item) {
  if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
      HelperFilesCustomizationLevel::HFCL_File) {
    return Item.Code;
  } else {
    // API level
    if (dpct::DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted) {
      if (!Item.USMCode.empty())
        return Item.USMCode;
    } else {
      if (!Item.NonUSMCode.empty())
        return Item.NonUSMCode;
    }
    return Item.Code;
  }
}

/// This class maintains a namespace state variable "PreviousNamespace".
/// If the input namespace is different from "PreviousNamespace", this class
/// can generate open/close statement(s) of namespace to meet the requirement.
/// E.g.,
/// PreviousNamespace is A::B::C, input namespace is A::D,
/// then the generated code will be:
/// \code
/// } // namespace C
/// } // namespace B
/// namespace D {
/// \endcode
class NamespaceGenerator {
public:
  std::string genCodeForNamespace(const std::string &CurrentNamespaceStr) {
    auto CurrentNamespace = splitNamespace(CurrentNamespaceStr);
    std::vector<std::string> CurrentNamespaceRemovedCommon;
    std::vector<std::string> PreviousNamespaceRemovedCommon;
    removeCommonNamespace(CurrentNamespace, CurrentNamespaceRemovedCommon,
                          PreviousNamespaceRemovedCommon);

    std::string Result;
    for (auto Iter = PreviousNamespaceRemovedCommon.rbegin();
         Iter != PreviousNamespaceRemovedCommon.rend(); Iter++) {
      Result = Result + "} // namespace " + *Iter + getNL() + getNL();
    }
    for (auto Iter = CurrentNamespaceRemovedCommon.begin();
         Iter != CurrentNamespaceRemovedCommon.end(); Iter++) {
      Result = Result + "namespace " + *Iter + " {" + getNL() + getNL();
    }
    PreviousNamespace = CurrentNamespace;
    return Result;
  }

private:
  bool findStr(const std::string &Str,
               const std::string::size_type &StartPosition,
               std::string::size_type &ResultPosition) {
    ResultPosition = Str.find("::", StartPosition);
    if (ResultPosition == std::string::npos)
      return false;
    else
      return true;
  }
  std::vector<std::string> splitNamespace(const std::string &Namespace) {
    std::vector<std::string> Splited;
    std::string::size_type StartPosition = 0;
    std::string::size_type ResultPosition = std::string::npos;
    while (findStr(Namespace, StartPosition, ResultPosition)) {
      Splited.push_back(
          Namespace.substr(StartPosition, ResultPosition - StartPosition));
      StartPosition = ResultPosition + std::strlen("::");
    }
    if (StartPosition < Namespace.size()) {
      Splited.push_back(Namespace.substr(StartPosition));
    }
    return Splited;
  }
  void removeCommonNamespace(
      const std::vector<std::string> &CurrentNamespace,
      std::vector<std::string> &CurrentNamespaceRemovedCommon,
      std::vector<std::string> &PreviousNamespaceRemovedCommon) {
    CurrentNamespaceRemovedCommon.clear();
    PreviousNamespaceRemovedCommon.clear();
    size_t Index = 0;
    while (true) {
      if (Index >= CurrentNamespace.size() ||
          Index >= PreviousNamespace.size()) {
        break;
      }
      if (CurrentNamespace[Index] != PreviousNamespace[Index]) {
        break;
      }
      ++Index;
    }
    auto CurrentNamespaceIter = CurrentNamespace.begin();
    auto PreviousNamespaceIter = PreviousNamespace.begin();
    std::advance(CurrentNamespaceIter, Index);
    std::advance(PreviousNamespaceIter, Index);
    CurrentNamespaceRemovedCommon.insert(CurrentNamespaceRemovedCommon.end(),
                                         CurrentNamespaceIter,
                                         CurrentNamespace.end());
    PreviousNamespaceRemovedCommon.insert(PreviousNamespaceRemovedCommon.end(),
                                          PreviousNamespaceIter,
                                          PreviousNamespace.end());
  }

  std::vector<std::string> PreviousNamespace;
};

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

  auto CompareAsc = [](const clang::dpct::HelperFunc &A,
                       const clang::dpct::HelperFunc &B) {
    return A.PositionIdx < B.PositionIdx;
  };
  std::sort(ContentVec.begin(), ContentVec.end(), CompareAsc);

  NamespaceGenerator NSG;
  for (const auto &Item : ContentVec) {
    ContentStr = ContentStr + NSG.genCodeForNamespace(Item.Namespace);
    std::string Code = getCode(Item);
    replaceEndOfLine(Code);
    ContentStr = ContentStr + Code + getNL();
  }

  ContentStr = ContentStr + NSG.genCodeForNamespace("");

  ContentStr = ContentStr + getHeaderGuardPair(File).second + getNL();
  return ContentStr;
}

std::string getDpctVersionStr() {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  OS << DPCT_VERSION_MAJOR << "." << DPCT_VERSION_MINOR << "."
     << DPCT_VERSION_PATCH;
  return OS.str();
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
            HelperFileNameMap.at(clang::dpct::HelperFileEnum::FILE_NAME),      \
        std::ios::binary);                                                     \
    std::string Code = FILE_NAME##AllContentStr;                               \
    replaceEndOfLine(Code);                                                    \
    FILE_NAME##File << Code;                                                   \
    FILE_NAME##File.flush();                                                   \
  }
#define GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(FILE_NAME)                        \
  {                                                                            \
    std::ofstream FILE_NAME##File(                                             \
        ToPath + "/dpl_extras/" +                                              \
            HelperFileNameMap.at(clang::dpct::HelperFileEnum::FILE_NAME),      \
        std::ios::binary);                                                     \
    std::string Code = FILE_NAME##AllContentStr;                               \
    replaceEndOfLine(Code);                                                    \
    FILE_NAME##File << Code;                                                   \
    FILE_NAME##File.flush();                                                   \
  }
  GENERATE_ALL_FILE_CONTENT(Atomic)
  GENERATE_ALL_FILE_CONTENT(BlasUtils)
  GENERATE_ALL_FILE_CONTENT(Device)
  GENERATE_ALL_FILE_CONTENT(Dpct)
  GENERATE_ALL_FILE_CONTENT(DplUtils)
  GENERATE_ALL_FILE_CONTENT(DnnlUtils)
  GENERATE_ALL_FILE_CONTENT(Image)
  GENERATE_ALL_FILE_CONTENT(Kernel)
  GENERATE_ALL_FILE_CONTENT(Memory)
  GENERATE_ALL_FILE_CONTENT(Util)
  GENERATE_ALL_FILE_CONTENT(RngUtils)
  GENERATE_ALL_FILE_CONTENT(LibCommonUtils)
  GENERATE_ALL_FILE_CONTENT(CclUtils)
  GENERATE_ALL_FILE_CONTENT(SparseUtils)
  GENERATE_ALL_FILE_CONTENT(FftUtils)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasAlgorithm)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasFunctional)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasIterators)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasMemory)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasNumeric)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasVector)
  GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT(DplExtrasDpcppExtensions)
#undef GENERATE_ALL_FILE_CONTENT
#undef GENERATE_DPL_EXTRAS_ALL_FILE_CONTENT
}

void generateHelperFunctions() {
  auto getUsedAPINum = []() -> size_t {
    size_t Res = 0;
    for (const auto &Item : HelperNameContentMap) {
      if (Item.second.IsCalled)
        Res++;
    }
    return Res;
  };

  // dpct.hpp is always exist, so request its non_local_include_dependency
  // feature
  requestFeature(dpct::HelperFeatureEnum::Dpct_non_local_include_dependency,
                 "");
  // 1. add dependent APIs
  size_t UsedAPINum = getUsedAPINum();
  do {
    UsedAPINum = getUsedAPINum();
    std::set<std::pair<HelperFeatureIDTy, std::set<std::string>>> NeedInsert;
    for (const auto &Item : HelperNameContentMap) {
      if (Item.second.IsCalled) {
        for (const auto &DepItem : Item.second.Dependency) {
          if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
              HelperFilesCustomizationLevel::HFCL_API) {
            if (DepItem.second == HelperFeatureDependencyKind::HFDK_UsmNone) {
              if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
                NeedInsert.insert(
                    std::make_pair(DepItem.first, Item.second.CallerSrcFiles));
            } else if (DepItem.second ==
                       HelperFeatureDependencyKind::HFDK_UsmRestricted) {
              if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted)
                NeedInsert.insert(
                    std::make_pair(DepItem.first, Item.second.CallerSrcFiles));
            } else {
              NeedInsert.insert(
                  std::make_pair(DepItem.first, Item.second.CallerSrcFiles));
            }
          } else {
            NeedInsert.insert(
                std::make_pair(DepItem.first, Item.second.CallerSrcFiles));
          }
        }
      }
    }
    for (const auto &Item : NeedInsert) {
      auto Iter = HelperNameContentMap.find(Item.first);
      if (Iter != HelperNameContentMap.end()) {
        Iter->second.IsCalled = true;
        Iter->second.CallerSrcFiles.insert(Item.second.begin(),
                                           Item.second.end());
      } else {
#ifdef DPCT_DEBUG_BUILD
        std::cout << "Unknown dependency: File:"
                  << (unsigned int)Item.first.first
                  << ", Feature:" << Item.first.second << std::endl;
        assert(0 && "Unknown dependency feature.\n");
#endif
      }
    }
  } while (getUsedAPINum() > UsedAPINum);

  // 2. build info of necessary headers to out-root
  if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
      HelperFilesCustomizationLevel::HFCL_None)
    return;
  else if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
           HelperFilesCustomizationLevel::HFCL_All) {
    generateAllHelperFiles();
    return;
  }

  std::vector<clang::dpct::HelperFunc> AtomicFileContent;
  std::vector<clang::dpct::HelperFunc> BlasUtilsFileContent;
  std::vector<clang::dpct::HelperFunc> DnnlUtilsFileContent;
  std::vector<clang::dpct::HelperFunc> DeviceFileContent;
  std::vector<clang::dpct::HelperFunc> DpctFileContent;
  std::vector<clang::dpct::HelperFunc> DplUtilsFileContent;
  std::vector<clang::dpct::HelperFunc> ImageFileContent;
  std::vector<clang::dpct::HelperFunc> KernelFileContent;
  std::vector<clang::dpct::HelperFunc> MemoryFileContent;
  std::vector<clang::dpct::HelperFunc> UtilFileContent;
  std::vector<clang::dpct::HelperFunc> RngUtilsFileContent;
  std::vector<clang::dpct::HelperFunc> LibCommonUtilsFileContent;
  std::vector<clang::dpct::HelperFunc> CclUtilsFileContent;
  std::vector<clang::dpct::HelperFunc> SparseUtilsFileContent;
  std::vector<clang::dpct::HelperFunc> FftUtilsFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasAlgorithmFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasFunctionalFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasIteratorsFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasMemoryFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasNumericFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasVectorFileContent;
  std::vector<clang::dpct::HelperFunc> DplExtrasDpcppExtensionsFileContent;

  std::vector<bool> FileUsedFlagVec(
      (unsigned int)clang::dpct::HelperFileEnum::HelperFileEnumTypeSize, false);
  if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==
      HelperFilesCustomizationLevel::HFCL_File) {
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

    for (const auto &Item : HelperNameContentMap)
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
        for (const auto &Item : HelperNameContentMap) {
          if (Item.first.first == (dpct::HelperFileEnum)FileID) {
            for (const auto &Dep : Item.second.Dependency) {
              if (clang::dpct::DpctGlobalInfo::
                      getHelperFilesCustomizationLevel() ==
                  HelperFilesCustomizationLevel::HFCL_API) {
                if (Dep.second == HelperFeatureDependencyKind::HFDK_UsmNone) {
                  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
                    FileUsedFlagVec[(unsigned int)Dep.first.first] = true;
                } else if (Dep.second ==
                           HelperFeatureDependencyKind::HFDK_UsmRestricted) {
                  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted)
                    FileUsedFlagVec[(unsigned int)Dep.first.first] = true;
                } else {
                  FileUsedFlagVec[(unsigned int)Dep.first.first] = true;
                }
              } else {
                FileUsedFlagVec[(unsigned int)Dep.first.first] = true;
              }
            }
          }
        }
      }
    } while (getUsedFileNum() > UsedFileNum);
  }

#define UPDATE_FILE(FILENAME)                                                  \
  case clang::dpct::HelperFileEnum::FILENAME:                                  \
    if (clang::dpct::DpctGlobalInfo::getHelperFilesCustomizationLevel() ==     \
        HelperFilesCustomizationLevel::HFCL_File) {                            \
      FILENAME##FileContent.push_back(Item.second);                            \
    } else if (clang::dpct::DpctGlobalInfo::                                   \
                   getHelperFilesCustomizationLevel() ==                       \
               HelperFilesCustomizationLevel::HFCL_API) {                      \
      if (Item.second.IsCalled)                                                \
        FILENAME##FileContent.push_back(Item.second);                          \
    }                                                                          \
    break;

  for (const auto &Item : HelperNameContentMap) {
    if (Item.first.second == "local_include_dependency") {
      // local_include_dependency for dpct and dpl_utils is inserted in step3
      // local_include_dependency for others is inserted in
      // getHelperFileContent()
      continue;
    } else if (Item.first.second == "non_local_include_dependency") {
      // non_local_include_dependency for dpct is inserted here
      // non_local_include_dependency for others is inserted in step3
      if (Item.first.first == clang::dpct::HelperFileEnum::Dpct) {
        DpctFileContent.push_back(Item.second);
      }
      continue;
    } else if (Item.first.second == "License") {
      continue;
    } else if (clang::dpct::DpctGlobalInfo::
                   getHelperFilesCustomizationLevel() ==
               HelperFilesCustomizationLevel::HFCL_File) {
      if (!FileUsedFlagVec[size_t(Item.first.first)])
        continue;
    }

    switch (Item.first.first) {
      UPDATE_FILE(Atomic)
      UPDATE_FILE(BlasUtils)
      UPDATE_FILE(DnnlUtils)
      UPDATE_FILE(Device)
      UPDATE_FILE(Dpct)
      UPDATE_FILE(DplUtils)
      UPDATE_FILE(Image)
      UPDATE_FILE(Kernel)
      UPDATE_FILE(Memory)
      UPDATE_FILE(Util)
      UPDATE_FILE(RngUtils)
      UPDATE_FILE(LibCommonUtils)
      UPDATE_FILE(CclUtils)
      UPDATE_FILE(SparseUtils)
      UPDATE_FILE(FftUtils)
      UPDATE_FILE(DplExtrasAlgorithm)
      UPDATE_FILE(DplExtrasFunctional)
      UPDATE_FILE(DplExtrasIterators)
      UPDATE_FILE(DplExtrasMemory)
      UPDATE_FILE(DplExtrasNumeric)
      UPDATE_FILE(DplExtrasVector)
      UPDATE_FILE(DplExtrasDpcppExtensions)
    default:
      assert(0 && "unknown helper file ID");
    }
  }
#undef UPDATE_FILE

  // 3. prepare folder and insert
  // non_local_include_dependency/local_include_dependency
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
      !DplExtrasVectorFileContent.empty() ||
      !DplExtrasDpcppExtensionsFileContent.empty()) {
    if (!llvm::sys::fs::is_directory(Twine(ToPath + "/dpl_extras")))
      llvm::sys::fs::create_directory(Twine(ToPath + "/dpl_extras"));

    std::string IDDStr;

    // There is an extra function replaceEndOfLine() to convert "\n" to
    // platform specific EOL for "#include ..." statement. Generally speaking,
    // for new added "#include ..." statement, developer should use "\n" instead
    // of getNL().
#define ADD_INCLUDE_DIRECTIVE_FOR_DPL(FILENAME)                                \
  if (!FILENAME##FileContent.empty()) {                                        \
    FILENAME##FileContent.push_back(HelperNameContentMap.at(                   \
        std::make_pair(clang::dpct::HelperFileEnum::FILENAME,                  \
                       "non_local_include_dependency")));                      \
    IDDStr = IDDStr + "#include \"dpl_extras/" +                               \
             HelperFileNameMap.at(clang::dpct::HelperFileEnum::FILENAME) +     \
             "\"\n";                                                           \
  }
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasAlgorithm)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasFunctional)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasIterators)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasMemory)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasNumeric)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasVector)
    ADD_INCLUDE_DIRECTIVE_FOR_DPL(DplExtrasDpcppExtensions)
#undef ADD_INCLUDE_DIRECTIVE_FOR_DPL

    auto Item = HelperNameContentMap.at(std::make_pair(
        clang::dpct::HelperFileEnum::DplUtils, "local_include_dependency"));
    Item.Code = IDDStr;
    DplUtilsFileContent.push_back(Item);
  }

  if (!DplUtilsFileContent.empty() ||
      HelperNameContentMap
          .at(std::make_pair(clang::dpct::HelperFileEnum::DplUtils,
                             "non_local_include_dependency"))
          .IsCalled) {
    DplUtilsFileContent.push_back(HelperNameContentMap.at(
        std::make_pair(clang::dpct::HelperFileEnum::DplUtils,
                       "non_local_include_dependency")));
  }

  std::string IDDStr;

#define ADD_INCLUDE_DIRECTIVE(FILENAME)                                        \
  if (!FILENAME##FileContent.empty()) {                                        \
    FILENAME##FileContent.push_back(HelperNameContentMap.at(                   \
        std::make_pair(clang::dpct::HelperFileEnum::FILENAME,                  \
                       "non_local_include_dependency")));                      \
    IDDStr = IDDStr + "#include \"" +                                          \
             HelperFileNameMap.at(clang::dpct::HelperFileEnum::FILENAME) +     \
             "\"\n";                                                           \
  }
  ADD_INCLUDE_DIRECTIVE(Atomic)
  ADD_INCLUDE_DIRECTIVE(BlasUtils)
  ADD_INCLUDE_DIRECTIVE(DnnlUtils)
  ADD_INCLUDE_DIRECTIVE(Device)
  // Do not include dpl_utils in dpct.hpp, since there is a bug in dpl_extras
  // files. All those functions are without the "inline" specifier, so there
  // will be a multi definition issue. ADD_INCLUDE_DIRECTIVE(DplUtils)
  ADD_INCLUDE_DIRECTIVE(Image)
  ADD_INCLUDE_DIRECTIVE(Kernel)
  ADD_INCLUDE_DIRECTIVE(Memory)
  ADD_INCLUDE_DIRECTIVE(Util)
  ADD_INCLUDE_DIRECTIVE(RngUtils)
  ADD_INCLUDE_DIRECTIVE(LibCommonUtils)
  ADD_INCLUDE_DIRECTIVE(CclUtils)
  ADD_INCLUDE_DIRECTIVE(SparseUtils)
  ADD_INCLUDE_DIRECTIVE(FftUtils)
#undef ADD_INCLUDE_DIRECTIVE

  auto Item = HelperNameContentMap.at(std::make_pair(
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
            HelperFileNameMap.at(clang::dpct::HelperFileEnum::FILE_NAME),      \
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
            HelperFileNameMap.at(clang::dpct::HelperFileEnum::FILE_NAME),      \
        std::ios::binary);                                                     \
    FILE_NAME##File << FILE_NAME##FileContentStr;                              \
    FILE_NAME##File.flush();                                                   \
  }
  GENERATE_FILE(Atomic)
  GENERATE_FILE(BlasUtils)
  GENERATE_FILE(DnnlUtils)
  GENERATE_FILE(Device)
  GENERATE_FILE(Dpct)
  GENERATE_FILE(DplUtils)
  GENERATE_FILE(Image)
  GENERATE_FILE(Kernel)
  GENERATE_FILE(Memory)
  GENERATE_FILE(Util)
  GENERATE_FILE(RngUtils)
  GENERATE_FILE(LibCommonUtils)
  GENERATE_FILE(CclUtils)
  GENERATE_FILE(SparseUtils)
  GENERATE_FILE(FftUtils)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasAlgorithm)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasFunctional)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasIterators)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasMemory)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasNumeric)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasVector)
  GENERATE_DPL_EXTRAS_FILE(DplExtrasDpcppExtensions)
#undef GENERATE_FILE
#undef GENERATE_DPL_EXTRAS_FILE
}

#define ADD_HELPER_FEATURE_FOR_ENUM_NAMES(TYPE)                                \
  void requestHelperFeatureForEnumNames(const std::string Name, TYPE File) {   \
    auto HelperFeatureIter =                                                   \
        clang::dpct::EnumConstantRule::EnumNamesMap.find(Name);                \
    if (HelperFeatureIter !=                                                   \
        clang::dpct::EnumConstantRule::EnumNamesMap.end()) {                   \
      requestFeature(HelperFeatureIter->second->RequestFeature, File);         \
      return;                                                                  \
    }                                                                          \
    auto CuDNNHelperFeatureIter =                                              \
        clang::dpct::CuDNNTypeRule::CuDNNEnumNamesHelperFeaturesMap.find(Name);\
    if (CuDNNHelperFeatureIter !=                                              \
        clang::dpct::CuDNNTypeRule::CuDNNEnumNamesHelperFeaturesMap.end()) {   \
      requestFeature(CuDNNHelperFeatureIter->second, File);                    \
    }                                                                          \
  }
#define ADD_HELPER_FEATURE_FOR_TYPE_NAMES(TYPE)                                \
  void requestHelperFeatureForTypeNames(const std::string Name, TYPE File) {   \
    auto HelperFeatureIter = MapNames::TypeNamesMap.find(Name);                \
    if (HelperFeatureIter != MapNames::TypeNamesMap.end()) {                   \
      requestFeature(HelperFeatureIter->second->RequestFeature, File);         \
      return;                                                                  \
    }                                                                          \
    auto CuDNNHelperFeatureIter = MapNames::CuDNNTypeNamesMap.find(Name);      \
    if (CuDNNHelperFeatureIter != MapNames::CuDNNTypeNamesMap.end()) {         \
      requestFeature(CuDNNHelperFeatureIter->second->RequestFeature, File);    \
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

void processFeatureMap(
    const std::map<std::string, clang::tooling::HelperFuncForYaml> &FeatureMap,
    HelperFileEnum CurrentFileID,
    std::map<std::string, std::set<std::string>> &APINameCallerSrcFilesMap) {
  for (const auto &FeatureFromYaml : FeatureMap) {
    HelperFeatureIDTy FeatureKey(CurrentFileID, FeatureFromYaml.first);
    auto FeatureIter = HelperNameContentMap.find(FeatureKey);
    if (FeatureIter != HelperNameContentMap.end()) {
      FeatureIter->second.IsCalled =
          FeatureIter->second.IsCalled || FeatureFromYaml.second.IsCalled;
      for (auto &CallerFileName : FeatureFromYaml.second.CallerSrcFiles) {
        FeatureIter->second.CallerSrcFiles.insert(CallerFileName);
      }

      // Process sub-features
      if (!FeatureFromYaml.second.SubFeatureMap.empty()) {
        processFeatureMap(FeatureFromYaml.second.SubFeatureMap, CurrentFileID,
                          APINameCallerSrcFilesMap);
      }
    }
  }
}

// Update HelperNameContentMap from TUR
void updateHelperNameContentMap(
    const clang::tooling::TranslationUnitReplacements &TUR) {
  std::map<std::string, std::set<std::string>> APINameCallerSrcFilesMap;

  for (const auto &FileFeatureMap : TUR.FeatureMap) {
    auto Iter = HelperFileIDMap.find(FileFeatureMap.first);
    if (Iter != HelperFileIDMap.end()) {
      auto CurrentFileID = Iter->second;
      processFeatureMap(FileFeatureMap.second, CurrentFileID,
                        APINameCallerSrcFilesMap);
    } else if (FileFeatureMap.first == (TUR.MainHelperFileName + ".hpp")) {
      processFeatureMap(FileFeatureMap.second, HelperFileEnum::Dpct,
                        APINameCallerSrcFilesMap);
    }
  }
}

// Update TUR from HelperNameContentMap
void updateTUR(clang::tooling::TranslationUnitReplacements &TUR) {
  auto updateAPIName = [](HelperFeatureIDTy Feature,
                          clang::tooling::HelperFuncForYaml &HFFY) {
    if (Feature.second == "License" ||
        Feature.second == "non_local_include_dependency" ||
        Feature.second == "local_include_dependency") {
      HFFY.APIName = "";
      return;
    }

    // If this feature can be found in the map, then save the API name (from
    // the map) into yaml file; otherwise save the feature name into yaml
    // file
    auto Iter = FeatureNameToAPINameMap.find(Feature);
    if (Iter != FeatureNameToAPINameMap.end()) {
      HFFY.APIName = Iter->second;
    } else {
      HFFY.APIName = Feature.second;
    }
  };

  for (const auto &Entry : HelperNameContentMap) {
    if (Entry.second.IsCalled) {
      std::string FileName = HelperFileNameMap[Entry.first.first];
      if (Entry.second.ParentFeature.first == HelperFileEnum::Unknown &&
          Entry.second.ParentFeature.second.empty()) {
        // This is not a sub-feature
        TUR.FeatureMap[FileName][Entry.first.second].IsCalled =
            Entry.second.IsCalled;
        TUR.FeatureMap[FileName][Entry.first.second].CallerSrcFiles.clear();

        for (const auto &CallerFileName : Entry.second.CallerSrcFiles) {
          TUR.FeatureMap[FileName][Entry.first.second].CallerSrcFiles.push_back(
              CallerFileName);
        }

        updateAPIName(Entry.first,
                      TUR.FeatureMap[FileName][Entry.first.second]);
      } else {
        // This is a sub-feature
        std::string ParentFeatureName = Entry.second.ParentFeature.second;
        TUR.FeatureMap[FileName][ParentFeatureName]
            .SubFeatureMap[Entry.first.second]
            .IsCalled = Entry.second.IsCalled;
        TUR.FeatureMap[FileName][ParentFeatureName]
            .SubFeatureMap[Entry.first.second]
            .CallerSrcFiles.clear();

        for (const auto &CallerFileName : Entry.second.CallerSrcFiles) {
          TUR.FeatureMap[FileName][ParentFeatureName]
              .SubFeatureMap[Entry.first.second]
              .CallerSrcFiles.push_back(CallerFileName);
        }

        updateAPIName(Entry.first, TUR.FeatureMap[FileName][ParentFeatureName]
                                       .SubFeatureMap[Entry.first.second]);
      }
    }
  }
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

std::map<HelperFeatureIDTy, clang::dpct::HelperFunc> HelperNameContentMap {
#define DPCT_CONTENT_BEGIN(File, Name, Namespace, Idx)                         \
  {                                                                            \
    {clang::dpct::HelperFileEnum::File, Name}, {Namespace, Idx, false, {},
#define DPCT_DEPENDENCY(...) {__VA_ARGS__},
#define DPCT_PARENT_FEATURE(ParentFeatureFile, ParentFeatureName)              \
  , { clang::dpct::HelperFileEnum::ParentFeatureFile, ParentFeatureName }
#define DPCT_CONTENT_END                                                       \
  }                                                                            \
  }                                                                            \
  ,
#include "clang/DPCT/atomic.inc"
#include "clang/DPCT/blas_utils.inc"
#include "clang/DPCT/ccl_utils.inc"
#include "clang/DPCT/dnnl_utils.inc"
#include "clang/DPCT/device.inc"
#include "clang/DPCT/dpct.inc"
#include "clang/DPCT/dpl_extras/algorithm.inc"
#include "clang/DPCT/dpl_extras/dpcpp_extensions.inc"
#include "clang/DPCT/dpl_extras/functional.inc"
#include "clang/DPCT/dpl_extras/iterators.inc"
#include "clang/DPCT/dpl_extras/memory.inc"
#include "clang/DPCT/dpl_extras/numeric.inc"
#include "clang/DPCT/dpl_extras/vector.inc"
#include "clang/DPCT/dpl_utils.inc"
#include "clang/DPCT/image.inc"
#include "clang/DPCT/kernel.inc"
#include "clang/DPCT/lib_common_utils.inc"
#include "clang/DPCT/sparse_utils.inc"
#include "clang/DPCT/fft_utils.inc"
#include "clang/DPCT/memory.inc"
#include "clang/DPCT/rng_utils.inc"
#include "clang/DPCT/util.inc"
#undef DPCT_CONTENT_BEGIN
#undef DPCT_DEPENDENCY
#undef DPCT_CONTENT_END
};

std::unordered_map<clang::dpct::HelperFileEnum, std::string> HelperFileNameMap{
    {clang::dpct::HelperFileEnum::Dpct, "dpct.hpp"},
    {clang::dpct::HelperFileEnum::Atomic, "atomic.hpp"},
    {clang::dpct::HelperFileEnum::BlasUtils, "blas_utils.hpp"},
    {clang::dpct::HelperFileEnum::DnnlUtils, "dnnl_utils.hpp"},
    {clang::dpct::HelperFileEnum::Device, "device.hpp"},
    {clang::dpct::HelperFileEnum::DplUtils, "dpl_utils.hpp"},
    {clang::dpct::HelperFileEnum::Image, "image.hpp"},
    {clang::dpct::HelperFileEnum::Kernel, "kernel.hpp"},
    {clang::dpct::HelperFileEnum::Memory, "memory.hpp"},
    {clang::dpct::HelperFileEnum::Util, "util.hpp"},
    {clang::dpct::HelperFileEnum::RngUtils, "rng_utils.hpp"},
    {clang::dpct::HelperFileEnum::LibCommonUtils, "lib_common_utils.hpp"},
    {clang::dpct::HelperFileEnum::CclUtils, "ccl_utils.hpp"},
    {clang::dpct::HelperFileEnum::SparseUtils, "sparse_utils.hpp"},
    {clang::dpct::HelperFileEnum::FftUtils, "fft_utils.hpp"},
    {clang::dpct::HelperFileEnum::DplExtrasAlgorithm, "algorithm.h"},
    {clang::dpct::HelperFileEnum::DplExtrasFunctional, "functional.h"},
    {clang::dpct::HelperFileEnum::DplExtrasIterators, "iterators.h"},
    {clang::dpct::HelperFileEnum::DplExtrasMemory, "memory.h"},
    {clang::dpct::HelperFileEnum::DplExtrasNumeric, "numeric.h"},
    {clang::dpct::HelperFileEnum::DplExtrasVector, "vector.h"},
    {clang::dpct::HelperFileEnum::DplExtrasDpcppExtensions,
     "dpcpp_extensions.h"}};

std::unordered_map<std::string, clang::dpct::HelperFileEnum> HelperFileIDMap{
    {"dpct.hpp", clang::dpct::HelperFileEnum::Dpct},
    {"atomic.hpp", clang::dpct::HelperFileEnum::Atomic},
    {"blas_utils.hpp", clang::dpct::HelperFileEnum::BlasUtils},
    {"dnnl_utils.hpp", clang::dpct::HelperFileEnum::DnnlUtils},
    {"device.hpp", clang::dpct::HelperFileEnum::Device},
    {"dpl_utils.hpp", clang::dpct::HelperFileEnum::DplUtils},
    {"image.hpp", clang::dpct::HelperFileEnum::Image},
    {"kernel.hpp", clang::dpct::HelperFileEnum::Kernel},
    {"memory.hpp", clang::dpct::HelperFileEnum::Memory},
    {"util.hpp", clang::dpct::HelperFileEnum::Util},
    {"rng_utils.hpp", clang::dpct::HelperFileEnum::RngUtils},
    {"lib_common_utils.hpp", clang::dpct::HelperFileEnum::LibCommonUtils},
    {"ccl_utils.hpp", clang::dpct::HelperFileEnum::CclUtils},
    {"sparse_utils.hpp", clang::dpct::HelperFileEnum::SparseUtils},
    {"fft_utils.hpp", clang::dpct::HelperFileEnum::FftUtils},
    {"algorithm.h", clang::dpct::HelperFileEnum::DplExtrasAlgorithm},
    {"functional.h", clang::dpct::HelperFileEnum::DplExtrasFunctional},
    {"iterators.h", clang::dpct::HelperFileEnum::DplExtrasIterators},
    {"memory.h", clang::dpct::HelperFileEnum::DplExtrasMemory},
    {"numeric.h", clang::dpct::HelperFileEnum::DplExtrasNumeric},
    {"vector.h", clang::dpct::HelperFileEnum::DplExtrasVector},
    {"dpcpp_extensions.h",
     clang::dpct::HelperFileEnum::DplExtrasDpcppExtensions}};

const std::unordered_map<clang::dpct::HelperFileEnum, std::string>
    HelperFileHeaderGuardMacroMap{
        {clang::dpct::HelperFileEnum::Dpct, "__DPCT_HPP__"},
        {clang::dpct::HelperFileEnum::Atomic, "__DPCT_ATOMIC_HPP__"},
        {clang::dpct::HelperFileEnum::BlasUtils, "__DPCT_BLAS_UTILS_HPP__"},
        {clang::dpct::HelperFileEnum::DnnlUtils, "__DPCT_DNNL_UTILS_HPP__"},
        {clang::dpct::HelperFileEnum::Device, "__DPCT_DEVICE_HPP__"},
        {clang::dpct::HelperFileEnum::DplUtils, "__DPCT_DPL_UTILS_HPP__"},
        {clang::dpct::HelperFileEnum::Image, "__DPCT_IMAGE_HPP__"},
        {clang::dpct::HelperFileEnum::Kernel, "__DPCT_KERNEL_HPP__"},
        {clang::dpct::HelperFileEnum::Memory, "__DPCT_MEMORY_HPP__"},
        {clang::dpct::HelperFileEnum::Util, "__DPCT_UTIL_HPP__"},
        {clang::dpct::HelperFileEnum::RngUtils, "__DPCT_RNG_UTILS_HPP__"},
        {clang::dpct::HelperFileEnum::LibCommonUtils,
         "__DPCT_LIB_COMMON_UTILS_HPP__"},
        {clang::dpct::HelperFileEnum::CclUtils, "__DPCT_CCL_UTILS_HPP__"},
        {clang::dpct::HelperFileEnum::SparseUtils, "__DPCT_SPARSE_UTILS_HPP__"},
        {clang::dpct::HelperFileEnum::FftUtils, "__DPCT_FFT_UTILS_HPP__"},
        {clang::dpct::HelperFileEnum::DplExtrasAlgorithm,
         "__DPCT_ALGORITHM_H__"},
        {clang::dpct::HelperFileEnum::DplExtrasFunctional,
         "__DPCT_FUNCTIONAL_H__"},
        {clang::dpct::HelperFileEnum::DplExtrasIterators,
         "__DPCT_ITERATORS_H__"},
        {clang::dpct::HelperFileEnum::DplExtrasMemory, "__DPCT_MEMORY_H__"},
        {clang::dpct::HelperFileEnum::DplExtrasNumeric, "__DPCT_NUMERIC_H__"},
        {clang::dpct::HelperFileEnum::DplExtrasVector, "__DPCT_VECTOR_H__"},
        {clang::dpct::HelperFileEnum::DplExtrasDpcppExtensions,
         "__DPCT_DPCPP_EXTENSIONS_H__"}};

const std::unordered_map<clang::dpct::HelperFeatureEnum,
                         clang::dpct::HelperFeatureIDTy>
    HelperFeatureEnumPairMap{
#define DPCT_FEATURE_ENUM_FEATURE_PAIR_MAP
#undef DPCT_FEATURE_ENUM
#include "clang/DPCT/HelperFeatureEnum.inc"
#undef DPCT_FEATURE_ENUM_FEATURE_PAIR_MAP
    };

const std::string DpctAllContentStr =
#include "clang/DPCT/dpct.all.inc"
    ;
const std::string AtomicAllContentStr =
#include "clang/DPCT/atomic.all.inc"
    ;
const std::string BlasUtilsAllContentStr =
#include "clang/DPCT/blas_utils.all.inc"
    ;
const std::string DnnlUtilsAllContentStr =
#include "clang/DPCT/dnnl_utils.all.inc"
    ;
const std::string DeviceAllContentStr =
#include "clang/DPCT/device.all.inc"
    ;
const std::string DplUtilsAllContentStr =
#include "clang/DPCT/dpl_utils.all.inc"
    ;
const std::string ImageAllContentStr =
#include "clang/DPCT/image.all.inc"
    ;
const std::string KernelAllContentStr =
#include "clang/DPCT/kernel.all.inc"
    ;
const std::string MemoryAllContentStr =
#include "clang/DPCT/memory.all.inc"
    ;
const std::string UtilAllContentStr =
#include "clang/DPCT/util.all.inc"
    ;
const std::string RngUtilsAllContentStr =
#include "clang/DPCT/rng_utils.all.inc"
    ;
const std::string LibCommonUtilsAllContentStr =
#include "clang/DPCT/lib_common_utils.all.inc"
    ;
const std::string CclUtilsAllContentStr =
#include "clang/DPCT/ccl_utils.all.inc"
    ;
const std::string SparseUtilsAllContentStr =
#include "clang/DPCT/sparse_utils.all.inc"
    ;
const std::string FftUtilsAllContentStr =
#include "clang/DPCT/fft_utils.all.inc"
    ;
const std::string DplExtrasAlgorithmAllContentStr =
#include "clang/DPCT/dpl_extras/algorithm.all.inc"
    ;
const std::string DplExtrasFunctionalAllContentStr =
#include "clang/DPCT/dpl_extras/functional.all.inc"
    ;
const std::string DplExtrasIteratorsAllContentStr =
#include "clang/DPCT/dpl_extras/iterators.all.inc"
    ;
const std::string DplExtrasMemoryAllContentStr =
#include "clang/DPCT/dpl_extras/memory.all.inc"
    ;
const std::string DplExtrasNumericAllContentStr =
#include "clang/DPCT/dpl_extras/numeric.all.inc"
    ;
const std::string DplExtrasVectorAllContentStr =
#include "clang/DPCT/dpl_extras/vector.all.inc"
    ;
const std::string DplExtrasDpcppExtensionsAllContentStr =
#include "clang/DPCT/dpl_extras/dpcpp_extensions.all.inc"
    ;

const std::map<std::pair<clang::dpct::HelperFileEnum, std::string>, std::string>
    FeatureNameToAPINameMap = {
#define HELPERFILE(PATH, UNIQUE_ENUM)
#define HELPER_FEATURE_MAP_TO_APINAME(File, FeatureName, APIName)              \
  {{clang::dpct::HelperFileEnum::File, FeatureName}, APIName},
#include "../../runtime/dpct-rt/include/HelperFileAndFeatureNames.inc"
#undef HELPER_FEATURE_MAP_TO_APINAME
#undef HELPERFILE
    };

const std::unordered_map<std::string, HelperFeatureEnum> PropToGetFeatureMap = {
    {"clockRate",
     HelperFeatureEnum::Device_device_info_get_max_clock_frequency},
    {"major", HelperFeatureEnum::Device_device_info_get_major_version},
    {"minor", HelperFeatureEnum::Device_device_info_get_minor_version},
    {"integrated", HelperFeatureEnum::Device_device_info_get_integrated},
    {"warpSize", HelperFeatureEnum::Device_device_info_get_max_sub_group_size},
    {"multiProcessorCount",
     HelperFeatureEnum::Device_device_info_get_max_compute_units},
    {"maxThreadsPerBlock",
     HelperFeatureEnum::Device_device_info_get_max_work_group_size},
    {"maxThreadsPerMultiProcessor",
     HelperFeatureEnum::Device_device_info_get_max_work_items_per_compute_unit},
    {"name", HelperFeatureEnum::Device_device_info_get_name},
    {"totalGlobalMem",
     HelperFeatureEnum::Device_device_info_get_global_mem_size},
    {"sharedMemPerMultiprocessor",
     HelperFeatureEnum::Device_device_info_get_local_mem_size},
    {"sharedMemPerBlock",
     HelperFeatureEnum::Device_device_info_get_local_mem_size},
    {"maxGridSize",
     HelperFeatureEnum::Device_device_info_get_max_nd_range_size},
    {"maxThreadsDim",
     HelperFeatureEnum::Device_device_info_get_max_work_item_sizes},
};

const std::unordered_map<std::string, HelperFeatureEnum> PropToSetFeatureMap = {
    {"clockRate",
     HelperFeatureEnum::Device_device_info_set_max_clock_frequency},
    {"major", HelperFeatureEnum::Device_device_info_set_major_version},
    {"minor", HelperFeatureEnum::Device_device_info_set_minor_version},
    {"integrated", HelperFeatureEnum::Device_device_info_set_integrated},
    {"warpSize", HelperFeatureEnum::Device_device_info_set_max_sub_group_size},
    {"multiProcessorCount",
     HelperFeatureEnum::Device_device_info_set_max_compute_units},
    {"maxThreadsPerBlock",
     HelperFeatureEnum::Device_device_info_set_max_work_group_size},
    {"maxThreadsPerMultiProcessor",
     HelperFeatureEnum::Device_device_info_set_max_work_items_per_compute_unit},
    {"name", HelperFeatureEnum::Device_device_info_set_name},
    {"totalGlobalMem",
     HelperFeatureEnum::Device_device_info_set_global_mem_size},
    {"sharedMemPerBlock",
     HelperFeatureEnum::Device_device_info_set_local_mem_size},
    {"maxGridSize",
     HelperFeatureEnum::Device_device_info_set_max_nd_range_size},
    {"maxThreadsDim",
     HelperFeatureEnum::Device_device_info_set_max_work_item_sizes},
};

const std::unordered_map<std::string, HelperFeatureEnum>
    SamplingInfoToSetFeatureMap = {
        {"coordinate_normalization_mode",
         HelperFeatureEnum::
             Image_sampling_info_set_coordinate_normalization_mode}};
const std::unordered_map<std::string, HelperFeatureEnum>
    SamplingInfoToGetFeatureMap = {
        {"addressing_mode",
         HelperFeatureEnum::Image_sampling_info_get_addressing_mode},
        {"filtering_mode",
         HelperFeatureEnum::Image_sampling_info_get_filtering_mode}};
const std::unordered_map<std::string, HelperFeatureEnum>
    ImageWrapperBaseToSetFeatureMap = {
        {"sampling_info",
         HelperFeatureEnum::Image_image_wrapper_base_set_sampling_info},
        {"data", HelperFeatureEnum::Image_image_wrapper_base_set_data},
        {"channel", HelperFeatureEnum::Image_image_wrapper_base_set_channel},
        {"channel_data_type",
         HelperFeatureEnum::Image_image_wrapper_base_set_channel_data_type},
        {"channel_size",
         HelperFeatureEnum::Image_image_wrapper_base_set_channel_size},
        {"coordinate_normalization_mode",
         HelperFeatureEnum::
             Image_image_wrapper_base_set_coordinate_normalization_mode},
        {"channel_num",
         HelperFeatureEnum::Image_image_wrapper_base_set_channel_num},
        {"channel_type",
         HelperFeatureEnum::Image_image_wrapper_base_set_channel_type}};
const std::unordered_map<std::string, HelperFeatureEnum>
    ImageWrapperBaseToGetFeatureMap = {
        {"sampling_info",
         HelperFeatureEnum::Image_image_wrapper_base_get_sampling_info},
        {"data", HelperFeatureEnum::Image_image_wrapper_base_get_data},
        {"channel", HelperFeatureEnum::Image_image_wrapper_base_get_channel},
        {"channel_data_type",
         HelperFeatureEnum::Image_image_wrapper_base_get_channel_data_type},
        {"channel_size",
         HelperFeatureEnum::Image_image_wrapper_base_get_channel_size},
        {"addressing_mode",
         HelperFeatureEnum::Image_image_wrapper_base_get_addressing_mode},
        {"filtering_mode",
         HelperFeatureEnum::Image_image_wrapper_base_get_filtering_mode},
        {"coordinate_normalization_mode",
         HelperFeatureEnum::
             Image_image_wrapper_base_get_coordinate_normalization_mode},
        {"channel_num",
         HelperFeatureEnum::Image_image_wrapper_base_get_channel_num},
        {"channel_type",
         HelperFeatureEnum::Image_image_wrapper_base_get_channel_type},
        {"sampler", HelperFeatureEnum::Image_image_wrapper_base_get_sampler},
    };

} // namespace dpct
} // namespace clang