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
// {var name, var-info}
using VarInfoMap = std::map<std::string, VarInfo>;

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
  void setKernelMVSize(std::string Size) { KernelMemSize = Size; }
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
      std::string AccVarName = HashIDForConstantMem;

      std::string BufferOffsetVar = "buffer_and_offset_" + AccVarName;
      std::string BufferVar = "buffer_" + AccVarName;
      Temp = "auto " + BufferOffsetVar + " = syclct::get_buffer_and_offset(" +
             VarName + ".get_ptr());\n";
      Temp = Temp + "        auto " + BufferVar + " = " + BufferOffsetVar +
             ".first.reinterpret<" + VarType + ">(cl::sycl::range<1>(" + S +
             "));\n" + "        auto " + AccVarName + "= " + BufferVar +
             ".get_access<cl::sycl::access::mode::read,  "
             "cl::sycl::access::target::constant_buffer>(cgh);";
    }

    return Temp;
  }
  /// pass sycl's accessor for shared memory variable to kernel function.
  std::string getAsFuncArgs() {
    if (IsShareVar) {
      return "(" + VarType + "*)" + VarName + ".get_pointer()";
    } else {
      std::string AccVarName = HashIDForConstantMem;
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
  std::string Size;
  bool IsExtern;
  bool IsShareVar; // true: Share Mem Var   false: Constant Mem Var
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
    auto It = SharedVarMap.find(SVT);
    if (It != SharedVarMap.end()) {
      return false;
    }
    VarInfo SVI(SVT, SVN, IsArray, Size, IsExtern);
    SharedVarMap[SVN] = SVI;
    return true;
  }

  /// CMV: Constant Mem Variable
  bool insertCMVInfo(std::string CVT, std::string CVN, bool IsArray,
                     std::string Size, std::string HashIDForConstantMem) {

    RecordSort.push_back(CVN);

    auto It = ConstantVarMap.find(CVT);
    if (It != ConstantVarMap.end()) {
      return false;
    }
    VarInfo CVI(CVT, CVN, IsArray, Size, HashIDForConstantMem);
    ConstantVarMap[CVN] = CVI;
    return true;
  }

  bool hasSMVDefined() { return SharedVarMap.size() > 0; }
  bool hasCMVDefined() { return ConstantVarMap.size() > 0; }

  uint getNumSMVDefined() { return (uint)SharedVarMap.size(); }

  std::string declareLocalAcc(const char *NL, StringRef Indent) {
    std::string Var;
    for (auto KV : SharedVarMap) {
      Var += Indent;
      Var += KV.second.getAccessorDeclare();
      Var += NL;
    }
    return Var;
  }

  std::string declareConstantAcc(const char *NL, StringRef Indent) {
    std::string Var;
    for (auto KV : ConstantVarMap) {
      Var += Indent;
      Var += KV.second.getAccessorDeclare();
      Var += NL;
    }
    return Var;
  }

  std::string declareSMVAsArgs() {
    std::string Var;
    int i = 0;
    for (auto KV : SharedVarMap) {
      if (i > 0)
        Var += ", ";
      Var += KV.second.getAsFuncArgDeclare();
      i++;
    }
    return Var;
  }

  std::string declareCMVAsArgs() {
    std::string Var;
    int i = 0;
    for (auto KV : ConstantVarMap) {
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
    for (auto KV : SharedVarMap) {
      if (i > 0)
        Var += ", ";
      Var += KV.second.getAsFuncArgs();
      i++;
    }
    return Var;
  }

  std::string passCMVAsArgs() {
    std::string Var;
    int i = 0;

    for (std::vector<std::string>::iterator it = RecordSort.begin();
         it < RecordSort.end(); it++) {
      if (i > 0)
        Var += ", ";
      Var += ConstantVarMap[*it].getAsFuncArgs();
      i++;
    }
    return Var;
  }

  void setKernelSMVSize(std::string Size) {
    if (SharedVarMap.size() == 0) {
      assert(0);
    }
    for (VarInfoMap::iterator it = SharedVarMap.begin();
         it != SharedVarMap.end(); ++it) {
      VarInfo &SVI = it->second;
      if (SVI.isExtern()) {
        /// set the 1st one matched.
        SVI.setKernelMVSize(Size);
        return;
      }
    }
    assert(0);
    return;
  }

  void setKernelCMVSize(std::string Size) {
    if (ConstantVarMap.size() == 0) {
      assert(0);
    }
    for (VarInfoMap::iterator it = ConstantVarMap.begin();
         it != ConstantVarMap.end(); ++it) {
      VarInfo &CVI = it->second;
      if (CVI.isExtern()) {
        /// set the 1st one matched.
        CVI.setKernelMVSize(Size);
        return;
      }
    }
    assert(0);
    return;
  }

  VarInfoMap &getSMVInfoMap() { return SharedVarMap; }
  VarInfoMap &getCMVInfoMap() { return ConstantVarMap; }

  void appendKernelArgs(std::string NewArgs) { KernelNewArgs += NewArgs; }
  std::string &getKernelArgs() { return KernelNewArgs; }

private:
  VarInfoMap SharedVarMap;
  VarInfoMap ConstantVarMap;
  std::vector<std::string> RecordSort;
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
