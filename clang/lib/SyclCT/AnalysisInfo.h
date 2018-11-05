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
class VarInfo;
// {kernel-name,  kernel-info}
using KernelInfoMap = std::map<std::string, KernelInfo>;
// {share-name, share-info}
using SMVInfoMap = std::map<std::string, VarInfo>;

class VarInfo {
public:
  VarInfo() {}
  // This constructor is used to insert share variable.
  VarInfo(std::string SVT, std::string SVN, bool IsArray, std::string Size,
          bool IsExtern)
      : VarType(SVT), VarName(SVN), IsArray(IsArray), Size(Size),
        IsExtern(IsExtern), IsShareVar(true) {}

  // This Constructor is used to insert const variable
  VarInfo(std::string CVT, std::string CVN, bool IsArray, std::string Size,
          std::string HashIDForConstantMem)
      : VarType(CVT), VarName(CVN), IsArray(IsArray), Size(Size),
        IsExtern(false), IsShareVar(false),
        HashIDForConstantMem(HashIDForConstantMem) {}

public:
  /// interface used to set memsize from "<<<x,x,memsize,x>>>"
  void setKernelSMVSize(std::string Size) { KernelMemSize = Size; }
  ///
  std::string &getType() { return VarType; }
  std::string &getName() { return VarName; }
  std::string &getSize() { return Size; }
  bool isArray() { return IsArray; }
  bool isExtern() { return IsExtern; }

  /// declcare __shared__ variable as sycl's accessor.
  std::string getAccessorDeclare() {
    std::string S;
    if (IsExtern) {
      S = KernelMemSize;
    } else {
      S = Size;
    }
    std::string Temp;

    if (IsShareVar) {
      // declcare __shared__ variable as sycl's accessor.
      Temp = "cl::sycl::accessor<" + VarType +
             ", 1, cl::sycl::access::mode::read_write, "
             "cl::sycl::access::target::local> " +
             VarName + "(cl::sycl::range<1>(" + S + "), cgh);";
    } else {
      // declcare __constant__ variable as sycl's accessor.
      std::string AccVarName = "const_acc_" + HashIDForConstantMem;
      if (IsArray) {
        Temp = "cl::sycl::buffer<cl::sycl::cl_" + VarType + ", 1> const_buf(&" +
               VarName + "[0], cl::sycl::range<1>(" + S + "));\n";
      } else {
        Temp = "cl::sycl::buffer<cl::sycl::cl_" + VarType + ", 1> const_buf(&" +
               VarName + ", cl::sycl::range<1>(" + S + "));\n";
      }

      Temp = Temp + "        auto  " + AccVarName +
             " = const_buf.get_access<cl::sycl::" +
             "access::mode::read, "
             "cl::sycl::access::target::constant_buffer>(cgh);";
    }

    return Temp;
  }
  /// pass sycl's accessor for shared memory variable to kernel function.
  std::string getAsFuncArgs() {
    if (IsShareVar) {
      return "(" + VarType + "*)" + VarName + ".get_pointer()";
    } else {
      std::string AccVarName = "const_acc_" + HashIDForConstantMem;
      return AccVarName;
    }
  }
  /// declare shared or constant memory variable in kernel function.
  std::string getAsFuncArgDeclare() {
    if (IsShareVar) {
      // declare shared memory variable in kernel function.
      return VarType + " " + VarName + "[]";
    } else {
      // declare constant memory variable in kernel function.
      return "cl::sycl::accessor<" + VarType +
             ", 1, cl::sycl::access::mode::read, "
             "cl::sycl::access::target::constant_buffer>  const_acc";
    }
  }

private:
  std::string VarType;
  std::string VarName;
  bool IsArray;
  bool IsShareVar; // true: Share Mem Var   false: Constant Mem Var
  std::string Size;
  bool IsExtern;
  std::string KernelMemSize;
  std::string HashIDForConstantMem;
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
    auto It = VarMap.find(SVT);
    if (It != VarMap.end()) {
      return false;
    }
    VarInfo SVI(SVT, SVN, IsArray, Size, IsExtern);
    VarMap[SVN] = SVI;
    return true;
  }

  /// CMV: Constant Mem Variable
  bool insertCMVInfo(std::string CVT, std::string CVN, bool IsArray,
                     std::string Size, std::string HashIDForConstantMem) {
    auto It = VarMap.find(CVT);
    if (It != VarMap.end()) {
      return false;
    }
    VarInfo CVI(CVT, CVN, IsArray, Size, HashIDForConstantMem);
    VarMap[CVN] = CVI;
    return true;
  }

  bool hasSMVDefined() { return VarMap.size() > 0; }

  uint getNumSMVDefined() { return (uint)VarMap.size(); }

  std::string declareLocalAcc(const char *NL, StringRef Indent) {
    std::string Var;
    for (auto KV : VarMap) {
      Var += Indent;
      Var += KV.second.getAccessorDeclare();
      Var += NL;
    }
    return Var;
  }
  std::string declareSMVAsArgs() {
    std::string Var;
    int i = 0;
    for (auto KV : VarMap) {
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
    for (auto KV : VarMap) {
      if (i > 0)
        Var += ", ";
      Var += KV.second.getAsFuncArgs();
      i++;
    }

    return Var;
  }
  void setKernelSMVSize(std::string Size) {
    if (VarMap.size() == 0) {
      assert(0);
    }
    for (SMVInfoMap::iterator it = VarMap.begin(); it != VarMap.end(); ++it) {
      VarInfo &SVI = it->second;
      if (SVI.isExtern()) {
        /// set the 1st one matched.
        SVI.setKernelSMVSize(Size);
        return;
      }
    }
    assert(0);
    return;
  }
  SMVInfoMap &getSMVInfoMap() { return VarMap; }
  void appendKernelArgs(std::string NewArgs) { KernelNewArgs += NewArgs; }
  std::string &getKernelArgs() { return KernelNewArgs; }

private:
  SMVInfoMap VarMap;
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

} // namespace syclct
} // namespace clang

#endif // CU2SYCL_ANALYSIS_INFO_H
