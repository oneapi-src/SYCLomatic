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
  enum class VarAttrKind : unsigned { Shared = 0, Constant, Device };

  VarInfo(std::string SVN, std::string SVT, bool IsArray,
          std ::vector<std::string> &Size, bool IsExtern, bool IsTemplateType,
          unsigned TemplateIndex)
      : VarType(SVT), VarName(SVN), IsArray(IsArray),
        ArraySize(std::move(Size)), IsExtern(IsExtern),
        VarAttr(VarAttrKind::Shared), IsTemplateType(IsTemplateType),
        TemplateIndex(TemplateIndex) {}

  // This Constructor is used to insert const variable
  VarInfo(std::string CVN, std::string CVT, bool IsArray, std::string Size,
          std::string HashIDForConstantMem)
      : VarType(CVT), VarName(CVN), IsArray(IsArray), IsExtern(false),
        VarAttr(VarAttrKind::Constant),
        HashIDForConstantMem(HashIDForConstantMem), IsTemplateType(false) {
    ArraySize.push_back(Size);
  }

  // This Constructor is used to insert device variable
  VarInfo(std::string CVN, std::string CVT, bool IsArray, std::string Size)
      : VarType(CVT), VarName(CVN), IsArray(IsArray), IsExtern(false),
        VarAttr(VarAttrKind::Device) {
    ArraySize.push_back(Size);
  }

public:
  /// interface used to set memsize from "<<<x,x,memsize,x>>>"
  void setKernelMVSize(std::string Size) {
    if (isExtern()) {
      ArraySize.clear();
      ArraySize.push_back(Size);
    }
  }
  /// interface used to set sharememtype while it's template type
  void setKernelSMVType(std::string Type) { VarType = Type; }
  ///
  std::string &getType() { return VarType; }
  std::string &getName() { return VarName; }
  unsigned getTemplateIndex() { return TemplateIndex; }
  bool isArray() { return IsArray; }
  bool isExtern() { return IsExtern; }
  bool isTemplateType() { return IsTemplateType; }
  /// declcare __shared__ variable as sycl's accessor.
  std::string getAccessorDeclare() {
    std::string Range =
        "cl::sycl::range<" + std::to_string(ArraySize.size()) + ">(";
    for (auto &Size : ArraySize)
      Range += Size + ", ";
    Range.replace(Range.size() - 2, 2, ")");
    std::string Temp;

    switch (VarAttr) {
    case VarAttrKind::Shared: {
      // declcare __shared__ variable as sycl's accessor.
      Temp = "cl::sycl::accessor<" + VarType + ", " +
             std::to_string(ArraySize.size()) +
             ", cl::sycl::access::mode::read_write, "
             "cl::sycl::access::target::local> " +
             VarName + "(" + Range + ", cgh);";
      break;
    }
    case VarAttrKind::Constant: {
      // declcare __constant__ variable as sycl's accessor.
      std::string AccVarName = HashIDForConstantMem;

      std::string BufferOffsetVar = "buffer_and_offset_" + AccVarName;
      std::string BufferVar = "buffer_" + AccVarName;
      Temp = "auto " + BufferOffsetVar + " = syclct::get_buffer_and_offset(" +
             VarName + ".get_ptr());\n";
      Temp = Temp + "        auto " + BufferVar + " = " + BufferOffsetVar +
             ".first.reinterpret<" + VarType + ">(" + Range + ");\n" +
             "        auto " + AccVarName + "= " + BufferVar +
             ".get_access<cl::sycl::access::mode::read,  "
             "cl::sycl::access::target::constant_buffer>(cgh);";
      break;
    }
    case VarAttrKind::Device: {
      // declcare __device__ variable as sycl's accessor.
      const std::string AccVarName = "device_acc_" + VarName;
      const std::string BufferOffsetVar = "device_buffer_and_offset_" + VarName;
      const std::string BufferVar = "device_buffer_" + VarName;
      Temp = "auto " + BufferOffsetVar + " = syclct::get_buffer_and_offset(" +
             VarName + ".get_ptr());\n";
      Temp += "        auto " + BufferVar + " = " + BufferOffsetVar +
              ".first.reinterpret<" + VarType + ">(" + Range + ");\n" +
              "        auto " + AccVarName + " = " + BufferVar +
              ".get_access<cl::sycl::access::mode::read_write>(cgh);";
      break;
    }
    }

    return Temp;
  }
  /// pass sycl's accessor for shared memory variable to kernel function.
  std::string getAsFuncArgs() {
    std::string Temp;
    switch (VarAttr) {
    case VarAttrKind::Shared: {
      Temp = VarName;
      break;
    }
    case VarAttrKind::Constant: {
      Temp = HashIDForConstantMem;
      break;
    }
    case VarAttrKind::Device: {
      Temp = "device_acc_" + VarName;
      break;
    }
    }
    return Temp;
  }
  /// declare shared or constant memory variable in kernel function.
  std::string getAsFuncArgDeclare() {
    std::string Temp;
    switch (VarAttr) {
    case VarAttrKind::Shared: {
      // declare shared memory variable in kernel function.
      Temp = VarType + " " + VarName + "[]";
      break;
    }
    case VarAttrKind::Constant: {
      // declare constant memory variable in kernel function.
      Temp = "cl::sycl::accessor<" + VarType +
             ", 1, cl::sycl::access::mode::read, "
             "cl::sycl::access::target::constant_buffer>  const_acc";
      break;
    }
    case VarAttrKind::Device: {
      // declare device memory variable in kernel function.
      Temp = "cl::sycl::accessor<" + VarType +
             ", 1, cl::sycl::access::mode::read_write, "
             "cl::sycl::access::target::global_buffer> " +
             VarName;
      break;
    }
    }
    return Temp;
  }

private:
  std::string VarType;
  std::string VarName;
  bool IsArray;
  std::vector<std::string> ArraySize;
  bool IsExtern;
  const VarAttrKind VarAttr;
  std::string HashIDForConstantMem;
  std::string KernelMemSize;
  bool IsTemplateType;
  unsigned TemplateIndex;
};

/// Record kernel relative info for multi rules co-operate when translate
class KernelInfo {
public:
  KernelInfo() {}
  KernelInfo(std::string KernelName) : KernelName(KernelName) {}

public:
  template <class... Args>
  bool insertCMVarInfo(const std::string &VarName, Args... Arguments) {
    RecordSort.push_back(VarName);
    return insertVarInfo(getCMVInfoMap(), VarName, Arguments...);
  }
  template <class... Args>
  bool insertVarInfo(VarInfoMap &VarMap, const std::string &VarName,
                     Args... Arguments) {
    auto It = VarMap.find(VarName);
    if (It != VarMap.end()) {
      return false;
    }
    VarMap.insert({VarName, VarInfo(VarName, Arguments...)});
    return true;
  }

  bool hasSMVDefined() { return SharedVarMap.size() > 0; }
  bool hasCMVDefined() { return ConstantVarMap.size() > 0; }
  bool hasDMVDefined() { return DeviceVarMap.size() > 0; }

  unsigned getNumSMVDefined() { return (unsigned)SharedVarMap.size(); }

  std::string declareLocalAcc(const char *NL, StringRef Indent) {
    std::string Var;
    for (auto &KV : SharedVarMap) {
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

  std::string declareDeviceAcc(const char *NL, StringRef Indent) {
    std::string Var;
    for (auto KV : DeviceVarMap) {
      Var += Indent;
      Var += KV.second.getAccessorDeclare();
      Var += NL;
    }
    return Var;
  }

  std::string declareSMVAsArgs() {
    std::string Var;
    int i = 0;
    for (auto &KV : SharedVarMap) {
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
    for (auto &KV : ConstantVarMap) {
      if (i > 0)
        Var += ", ";
      Var += KV.second.getAsFuncArgDeclare();
      i++;
    }
    return Var;
  }

  std::string declareDMVAsArgs() {
    std::string Var;
    int i = 0;
    for (auto KV : DeviceVarMap) {
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
    for (auto &KV : SharedVarMap) {
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
      Var += ConstantVarMap.at(*it).getAsFuncArgs();
      i++;
    }
    return Var;
  }

  std::string passDMVAsArgs() {
    std::string Var;
    int i = 0;
    for (auto KV : DeviceVarMap) {
      if (i > 0)
        Var += ", ";
      Var += KV.second.getAsFuncArgs();
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
  VarInfoMap &getDMVInfoMap() { return DeviceVarMap; }

  void appendKernelArgs(std::string NewArgs) { KernelNewArgs += NewArgs; }
  std::string &getKernelArgs() { return KernelNewArgs; }
  void setTemplateArgs(std::vector<std::string> &Args) {
    for (auto &KV : SharedVarMap) {
      if (KV.second.isTemplateType())
        KV.second.setKernelSMVType(Args[KV.second.getTemplateIndex()]);
    }
  }

private:
  VarInfoMap SharedVarMap;
  VarInfoMap ConstantVarMap;
  VarInfoMap DeviceVarMap;
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
