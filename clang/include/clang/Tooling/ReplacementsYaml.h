//===-- ReplacementsYaml.h -- Serialiazation for Replacements ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the structure of a YAML document for serializing
/// replacements.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REPLACEMENTSYAML_H
#define LLVM_CLANG_TOOLING_REPLACEMENTSYAML_H

#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/Support/YAMLTraits.h"
#include <string>

LLVM_YAML_IS_SEQUENCE_VECTOR(clang::tooling::Replacement)
#ifdef SYCLomatic_CUSTOMIZATION
// Keep here only for backward compatibility - begin
LLVM_YAML_IS_STRING_MAP(clang::tooling::HelperFuncForYaml)
using FeatureOfFileMapTy = std::map<std::string/*feature name*/, clang::tooling::HelperFuncForYaml>;
LLVM_YAML_IS_STRING_MAP(FeatureOfFileMapTy)
// Keep here only for backward compatibility - end
LLVM_YAML_IS_SEQUENCE_VECTOR(clang::tooling::CompilationInfo)
LLVM_YAML_IS_STRING_MAP(std::vector<clang::tooling::CompilationInfo>)
LLVM_YAML_IS_SEQUENCE_VECTOR(clang::tooling::MainSourceFileInfo)
LLVM_YAML_IS_STRING_MAP(std::vector<clang::tooling::MainSourceFileInfo>)
LLVM_YAML_IS_STRING_MAP(clang::tooling::OptionInfo)
#endif // SYCLomatic_CUSTOMIZATION

namespace llvm {
namespace yaml {

/// Specialized MappingTraits to describe how a Replacement is
/// (de)serialized.
template <> struct MappingTraits<clang::tooling::Replacement> {
  /// Helper to (de)serialize a Replacement since we don't have direct
  /// access to its data members.
  struct NormalizedReplacement {
#ifdef SYCLomatic_CUSTOMIZATION
    NormalizedReplacement(const IO &)
        : FilePath(""), Offset(0), Length(0), ReplacementText(""),
          ConstantFlag(""), ConstantOffset(0), InitStr(""), NewHostVarName(""),
          BlockLevelFormatFlag(false) {
    }

    NormalizedReplacement(const IO &, const clang::tooling::Replacement &R)
        : FilePath(R.getFilePath()), Offset(R.getOffset()),
          Length(R.getLength()), ReplacementText(R.getReplacementText()),
          ConstantOffset(R.getConstantOffset()), InitStr(R.getInitStr()),
          NewHostVarName(R.getNewHostVarName()),
          BlockLevelFormatFlag(R.getBlockLevelFormatFlag()) {
      clang::dpct::ConstantFlagType Flag = R.getConstantFlag();
      if (Flag == clang::dpct::ConstantFlagType::HostDevice) {
        ConstantFlag = "HostDeviceConstant";
      } else if (Flag == clang::dpct::ConstantFlagType::Device) {
        ConstantFlag = "DeviceConstant";
      } else if (Flag == clang::dpct::ConstantFlagType::Host) {
        ConstantFlag = "HostConstant";
      } else {
        ConstantFlag = "";
      }
    }

    clang::tooling::Replacement denormalize(const IO &) {
      auto R = clang::tooling::Replacement(FilePath, Offset, Length,
                                           ReplacementText);
      if (ConstantFlag == "HostDeviceConstant") {
        R.setConstantFlag(clang::dpct::ConstantFlagType::HostDevice);
      } else if (ConstantFlag == "DeviceConstant") {
        R.setConstantFlag(clang::dpct::ConstantFlagType::Device);
      } else if (ConstantFlag == "HostConstant") {
        R.setConstantFlag(clang::dpct::ConstantFlagType::Host);
      } else {
        R.setConstantFlag(clang::dpct::ConstantFlagType::Default);
      }
      R.setConstantOffset(ConstantOffset);
      R.setInitStr(InitStr);
      R.setNewHostVarName(NewHostVarName);
      R.setBlockLevelFormatFlag(BlockLevelFormatFlag);
      return R;
    }
#else
    NormalizedReplacement(const IO &) : Offset(0), Length(0) {}

    NormalizedReplacement(const IO &, const clang::tooling::Replacement &R)
        : FilePath(R.getFilePath()), Offset(R.getOffset()),
          Length(R.getLength()), ReplacementText(R.getReplacementText()) {}

    clang::tooling::Replacement denormalize(const IO &) {
      return clang::tooling::Replacement(FilePath, Offset, Length,
                                         ReplacementText);
    }
#endif // SYCLomatic_CUSTOMIZATION

    std::string FilePath;
    unsigned int Offset;
    unsigned int Length;
    std::string ReplacementText;
#ifdef SYCLomatic_CUSTOMIZATION
    std::string ConstantFlag = "";
    unsigned int ConstantOffset = 0;
    std::string InitStr = "";
    std::string NewHostVarName = "";
    bool BlockLevelFormatFlag = false;
#endif // SYCLomatic_CUSTOMIZATION
  };

  static void mapping(IO &Io, clang::tooling::Replacement &R) {
    MappingNormalization<NormalizedReplacement, clang::tooling::Replacement>
    Keys(Io, R);
    Io.mapRequired("FilePath", Keys->FilePath);
    Io.mapRequired("Offset", Keys->Offset);
    Io.mapRequired("Length", Keys->Length);
    Io.mapRequired("ReplacementText", Keys->ReplacementText);
#ifdef SYCLomatic_CUSTOMIZATION
    Io.mapOptional("ConstantFlag", Keys->ConstantFlag);
    Io.mapOptional("ConstantOffset", Keys->ConstantOffset);
    Io.mapOptional("InitStr", Keys->InitStr);
    Io.mapOptional("NewHostVarName", Keys->NewHostVarName);
    Io.mapOptional("BlockLevelFormatFlag", Keys->BlockLevelFormatFlag);
#endif // SYCLomatic_CUSTOMIZATION
  }
};

#ifdef SYCLomatic_CUSTOMIZATION
template <> struct MappingTraits<clang::tooling::MainSourceFileInfo> {
  struct NormalizedMainSourceFilesDigest {

    NormalizedMainSourceFilesDigest(const IO &)
        : MainSourceFile(""), Digest(""), HasCUDASyntax(false) {}

    NormalizedMainSourceFilesDigest(const IO &,
                                    clang::tooling::MainSourceFileInfo &R)
        : MainSourceFile(R.MainSourceFile), Digest(R.Digest),
          HasCUDASyntax(R.HasCUDASyntax) {}

    clang::tooling::MainSourceFileInfo denormalize(const IO &) {
      return clang::tooling::MainSourceFileInfo(MainSourceFile, Digest,
                                                HasCUDASyntax);
    }

    std::string MainSourceFile = "";
    std::string Digest = "";
    bool HasCUDASyntax = false;
  };

  static void mapping(IO &Io, clang::tooling::MainSourceFileInfo &R) {
    MappingNormalization<NormalizedMainSourceFilesDigest,
                         clang::tooling::MainSourceFileInfo>
        Keys(Io, R);
    Io.mapOptional("MainSourceFile", Keys->MainSourceFile);
    Io.mapOptional("Digest", Keys->Digest);
    printf("#########Keys->HasCUDASyntax [%d]\n", Keys->HasCUDASyntax);
    Io.mapOptional("HasCUDASyntax", Keys->HasCUDASyntax);
  }
};

template <> struct MappingTraits<clang::tooling::CompilationInfo> {
  struct NormalizedCompileCmds {

    NormalizedCompileCmds(const IO &)
        : MigratedFileName(""), CompileOptions(""), Compiler(""){}

    NormalizedCompileCmds(const IO &, clang::tooling::CompilationInfo &CmpInfo)
        : MigratedFileName(CmpInfo.MigratedFileName),
          CompileOptions(CmpInfo.CompileOptions),
          Compiler(CmpInfo.Compiler) {}

    clang::tooling::CompilationInfo denormalize(const IO &) {
      clang::tooling::CompilationInfo CmpInfo;
      CmpInfo.MigratedFileName = MigratedFileName;
      CmpInfo.CompileOptions = CompileOptions;
      CmpInfo.Compiler = Compiler;
      return CmpInfo;
    }

    std::string MigratedFileName;
    std::string CompileOptions;
    std::string Compiler;
  };

  static void mapping(IO &Io, clang::tooling::CompilationInfo &CmpInfo) {
    MappingNormalization<NormalizedCompileCmds, clang::tooling::CompilationInfo>
        Keys(Io, CmpInfo);
    Io.mapOptional("MigratedFileName", Keys->MigratedFileName);
    Io.mapOptional("CompileOptions", Keys->CompileOptions);
    Io.mapOptional("Compiler", Keys->Compiler);
  }
};

template <> struct MappingTraits<clang::tooling::OptionInfo> {
  struct NormalizedOptionInfo {
    NormalizedOptionInfo(const IO &) : Value(""), Specified(true) {}
    NormalizedOptionInfo(const IO &, clang::tooling::OptionInfo &OptInfo)
        : Value(OptInfo.Value), ValueVec(OptInfo.ValueVec),
          Specified(OptInfo.Specified) {}

    clang::tooling::OptionInfo denormalize(const IO &) {
      clang::tooling::OptionInfo OptInfo;
      OptInfo.Value = Value;
      OptInfo.ValueVec = ValueVec;
      OptInfo.Specified = Specified;
      return OptInfo;
    }

    std::string Value;
    std::vector<std::string> ValueVec;
    bool Specified;
  };

  static void mapping(IO &Io, clang::tooling::OptionInfo &OptInfo) {
    MappingNormalization<NormalizedOptionInfo, clang::tooling::OptionInfo> Keys(
        Io, OptInfo);
    Io.mapOptional("Value", Keys->Value);
    Io.mapOptional("ValueVec", Keys->ValueVec);
    Io.mapOptional("Specified", Keys->Specified);
  }
};

// Keep here only for backward compatibility - begin
template <> struct MappingTraits<clang::tooling::HelperFuncForYaml> {
  struct NormalizedHelperFuncForYaml {
    NormalizedHelperFuncForYaml(const IO &)
        : IsCalled(false), CallerSrcFiles({}), SubFeatureMap({}) {}

    NormalizedHelperFuncForYaml(const IO &,
                               const clang::tooling::HelperFuncForYaml &HFFY)
        : IsCalled(HFFY.IsCalled), CallerSrcFiles(HFFY.CallerSrcFiles),
          SubFeatureMap(HFFY.SubFeatureMap) {}

    clang::tooling::HelperFuncForYaml denormalize(const IO &) {
      clang::tooling::HelperFuncForYaml HFFY;
      HFFY.IsCalled = IsCalled;
      HFFY.CallerSrcFiles = CallerSrcFiles;
      HFFY.SubFeatureMap = SubFeatureMap;
      return HFFY;
    }

    bool IsCalled = false;
    std::vector<std::string> CallerSrcFiles;
    std::map<std::string, clang::tooling::HelperFuncForYaml> SubFeatureMap;
  };

  static void mapping(IO &Io, clang::tooling::HelperFuncForYaml &HFFY) {
    MappingNormalization<NormalizedHelperFuncForYaml,
                         clang::tooling::HelperFuncForYaml>
        Keys(Io, HFFY);
    Io.mapOptional("IsCalled", Keys->IsCalled);
    Io.mapOptional("CallerSrcFiles", Keys->CallerSrcFiles);
    Io.mapOptional("SubFeatureMap", Keys->SubFeatureMap);
  }
};
// Keep here only for backward compatibility - end
#endif // SYCLomatic_CUSTOMIZATION


/// Specialized MappingTraits to describe how a
/// TranslationUnitReplacements is (de)serialized.
template <> struct MappingTraits<clang::tooling::TranslationUnitReplacements> {
  static void mapping(IO &Io,
                      clang::tooling::TranslationUnitReplacements &Doc) {
    Io.mapRequired("MainSourceFile", Doc.MainSourceFile);
    Io.mapRequired("Replacements", Doc.Replacements);
#ifdef SYCLomatic_CUSTOMIZATION
    Io.mapOptional("MainSourceFilesDigest", Doc.MainSourceFilesDigest);
    Io.mapOptional("DpctVersion", Doc.DpctVersion);
    Io.mapOptional("MainHelperFileName", Doc.MainHelperFileName);
    Io.mapOptional("USMLevel", Doc.USMLevel);
    // Keep here only for backward compatibility - begin
    Io.mapOptional("FeatureMap", Doc.FeatureMap);
    // Keep here only for backward compatibility - end
    Io.mapOptional("CompileTargets", Doc.CompileTargets);
    Io.mapOptional("OptionMap", Doc.OptionMap);
#endif
  }
};
} // end namespace yaml
} // end namespace llvm

#endif
