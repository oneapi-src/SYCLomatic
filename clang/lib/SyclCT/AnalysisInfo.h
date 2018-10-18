//===--- AnalysisInfo.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef CU2SYCL_ANALYSIS_INFO_H
#define CU2SYCL_ANALYSIS_INFO_H

#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"
#include <map>
#include <string>

namespace clang {
namespace syclct {

class KernelInfo;
class ShareVarInfo;
// {kernel-name,  kernel-info}
using KernelInfoMap = std::map<std::string, KernelInfo>;
// {share-name, share-info}
using SMVInfoMap = std::map<std::string, ShareVarInfo>;

class ShareVarInfo {
public:
  ShareVarInfo() {}
  ShareVarInfo(std::string SVT, std::string SVN, bool IsArray, std::string Size,
               bool IsExtern)
      : SharedVarType(SVT), SharedVarName(SVN), IsArray(IsArray), Size(Size),
        IsExtern(IsExtern) {}

public:
  /// interface used to set sharememsize from "<<<x,x,sharememsize,x>>>"
  void setKernelSMVSize(std::string Size) { KernelShareMemSize = Size; }
  ///
  std::string &getType() { return SharedVarType; }
  std::string &getName() { return SharedVarName; }
  std::string &getSize() { return Size; }
  bool isArray() { return IsArray; }
  bool isExtern() { return IsExtern; }
  /// declcare __shared__ variable as sycl's accessor.
  std::string getAccessorDeclare() {
    std::string S;
    if (IsExtern) {
      S = KernelShareMemSize;
    } else {
      S = Size;
    }
    std::string Temp;
    Temp = "cl::sycl::accessor<" + SharedVarType +
           ", 1, cl::sycl::access::mode::read_write, "
           "cl::sycl::access::target::local> " +
           SharedVarName + "(cl::sycl::range<1>(" + S + "), cgh);";
    return Temp;
  }
  /// pass sycl's accessor for shared memory variable to kernel function.
  std::string getAsFuncArgs() {
    return "(" + SharedVarType + "*)" + SharedVarName + ".get_pointer()";
  }
  /// declare shared memory variable in kernel function.
  std::string getAsFuncArgDeclare() {
    return SharedVarType + " " + SharedVarName + "[]";
  }

private:
  std::string SharedVarType;
  std::string SharedVarName;
  bool IsArray;
  std::string Size;
  bool IsExtern;
  std::string KernelShareMemSize;
};

/// Record kernel relative info for multi rules co-operate when translate
class KernelInfo {
public:
  KernelInfo() {}
  KernelInfo(std::string KernelName) : KernelName(KernelName) {}

public:
  /// SMV: Shared Mem Variable
  bool insertSMVInfo(std::string SVT, std::string SVN, bool IsArray,
                     std::string Size, bool IsExtern) {
    auto It = ShareVarMap.find(SVT);
    if (It != ShareVarMap.end()) {
      return false;
    }
    ShareVarInfo SVI(SVT, SVN, IsArray, Size, IsExtern);
    ShareVarMap[SVN] = SVI;
    return true;
  }
  bool hasSMVDefined() { return ShareVarMap.size() > 0; }
  uint getNumSMVDefined() { return (uint)ShareVarMap.size(); }
  std::string declareLocalAcc(const char *NL, StringRef Indent) {
    std::string Var;
    for (auto KV : ShareVarMap) {
      Var += Indent;
      Var += KV.second.getAccessorDeclare();
      Var += NL;
    }
    return Var;
  }
  std::string declareSMVAsArgs() {
    std::string Var;
    int i = 0;
    for (auto KV : ShareVarMap) {
      if (i > 0)
        Var += ", ";
      Var += KV.second.getAsFuncArgDeclare();
      i++;
    }

    return Var;
  }
  std::string passSMVAsArgs() {
    std::string Var;
    int i = 0;
    for (auto KV : ShareVarMap) {
      if (i > 0)
        Var += ", ";
      Var += KV.second.getAsFuncArgs();
      i++;
    }

    return Var;
  }
  void setKernelSMVSize(std::string Size) {
    if (ShareVarMap.size() == 0) {
      assert(0);
    }
    for (SMVInfoMap::iterator it = ShareVarMap.begin(); it != ShareVarMap.end();
         ++it) {
      ShareVarInfo &SVI = it->second;
      if (SVI.isExtern()) {
        /// set the 1st one matched.
        SVI.setKernelSMVSize(Size);
        return;
      }
    }
    assert(0);
    return;
  }
  SMVInfoMap &getSMVInfoMap() { return ShareVarMap; }
  void appendKernelArgs(std::string NewArgs) { KernelNewArgs += NewArgs; }
  std::string &getKernelArgs() { return KernelNewArgs; }

private:
  SMVInfoMap ShareVarMap;
  std::string KernelNewArgs;
  std::string KernelName;
};

class KernelTransAssist {
  static KernelInfoMap KernelNameInfoMap;

public:
  /// check if <KN,XX> exist
  static bool hasKernelInfo(const std::string KN) {
    auto Search = KernelNameInfoMap.find(KN);
    if (Search == KernelNameInfoMap.end()) {
      return false;
    } else {
      return true;
    }
  }

  /// insert kernelinfo into KernelNmaeMap, return its kernelinfo.
  /// before call this function, make sure <KN,KI> doesn't exist in
  /// KernelInfoMap
  static bool insertKernel(const std::string KN, KernelInfo &KI) {
    KernelNameInfoMap[KN] = KI;
    return true;
  }

  /// before call this function, make sure a <KN,KI> exist in KernelInfoMap
  static KernelInfo &getKernelInfo(std::string KN) {
    auto Search = KernelNameInfoMap.find(KN);
    if (Search == KernelNameInfoMap.end()) {
      // TODO report exception error here.
    }
    return Search->second;
  }
};

} // namespace cu2sycl
} // namespace clang

#endif // CU2SYCL_ANALYSIS_INFO_H
