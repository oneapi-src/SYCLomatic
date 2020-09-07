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

#include "clang/Tooling/Refactoring.h"
#include "llvm/Support/YAMLTraits.h"
#include <string>

LLVM_YAML_IS_SEQUENCE_VECTOR(clang::tooling::Replacement)

namespace llvm {
namespace yaml {

/// Specialized MappingTraits to describe how a Replacement is
/// (de)serialized.
template <> struct MappingTraits<clang::tooling::Replacement> {
  /// Helper to (de)serialize a Replacement since we don't have direct
  /// access to its data members.
  struct NormalizedReplacement {
#ifdef INTEL_CUSTOMIZATION
    NormalizedReplacement(const IO &)
        : FilePath(""), Offset(0), Length(0), ReplacementText(""),
          ConstantFlag(""), ConstantOffset(0), InitStr(""), NewHostVarName("") {
    }

    NormalizedReplacement(const IO &, const clang::tooling::Replacement &R)
        : FilePath(R.getFilePath()), Offset(R.getOffset()),
          Length(R.getLength()), ReplacementText(R.getReplacementText()),
          ConstantOffset(R.getConstantOffset()), InitStr(R.getInitStr()),
          NewHostVarName(R.getNewHostVarName()) {
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
      return R;
    }
#else
    NormalizedReplacement(const IO &)
        : FilePath(""), Offset(0), Length(0), ReplacementText("") {}

    NormalizedReplacement(const IO &, const clang::tooling::Replacement &R)
        : FilePath(R.getFilePath()), Offset(R.getOffset()),
          Length(R.getLength()), ReplacementText(R.getReplacementText()) {
      size_t lineBreakPos = ReplacementText.find('\n');
      while (lineBreakPos != std::string::npos) {
        ReplacementText.replace(lineBreakPos, 1, "\n\n");
        lineBreakPos = ReplacementText.find('\n', lineBreakPos + 2);
      }
    }

    clang::tooling::Replacement denormalize(const IO &) {
      return clang::tooling::Replacement(FilePath, Offset, Length,
                                         ReplacementText);
    }
#endif

    std::string FilePath;
    unsigned int Offset;
    unsigned int Length;
    std::string ReplacementText;
#ifdef INTEL_CUSTOMIZATION
    std::string ConstantFlag = "";
    unsigned int ConstantOffset = 0;
    std::string InitStr = "";
    std::string NewHostVarName = "";
#endif
  };

  static void mapping(IO &Io, clang::tooling::Replacement &R) {
    MappingNormalization<NormalizedReplacement, clang::tooling::Replacement>
    Keys(Io, R);
    Io.mapRequired("FilePath", Keys->FilePath);
    Io.mapRequired("Offset", Keys->Offset);
    Io.mapRequired("Length", Keys->Length);
    Io.mapRequired("ReplacementText", Keys->ReplacementText);
#ifdef INTEL_CUSTOMIZATION
    Io.mapRequired("ConstantFlag", Keys->ConstantFlag);
    Io.mapRequired("ConstantOffset", Keys->ConstantOffset);
    Io.mapRequired("InitStr", Keys->InitStr);
    Io.mapRequired("NewHostVarName", Keys->NewHostVarName);
#endif
  }
};

/// Specialized MappingTraits to describe how a
/// TranslationUnitReplacements is (de)serialized.
template <> struct MappingTraits<clang::tooling::TranslationUnitReplacements> {
  static void mapping(IO &Io,
                      clang::tooling::TranslationUnitReplacements &Doc) {
    Io.mapRequired("MainSourceFile", Doc.MainSourceFile);
    Io.mapRequired("Replacements", Doc.Replacements);
  }
};
} // end namespace yaml
} // end namespace llvm

#endif
