//===--- CallExprRewriter.cpp -----------------------------*- C++ -*---===//
//
// Copyright (C) 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "AnalysisInfo.h"
#include "MapNames.h"

namespace clang {
namespace dpct {

std::string CallExprRewriter::getMigratedArg(unsigned Idx) {
  Analyzer.analyze(Call->getArg(Idx));
  return Analyzer.getReplacedString();
}

std::vector<std::string> CallExprRewriter::getMigratedArgs() {
  std::vector<std::string> ArgList;
  for (unsigned i = 0; i < Call->getNumArgs(); ++i)
    ArgList.emplace_back(getMigratedArg(i));
  return ArgList;
}

Optional<std::string> FuncCallExprRewriter::rewrite() {
  RewriteArgList = getMigratedArgs();
  return buildRewriteString();
}

Optional<std::string> FuncCallExprRewriter::buildRewriteString() {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << TargetCalleeName << "(";
  for (auto &Arg : RewriteArgList)
    OS << Arg << ", ";
  OS.flush();
  return RewriteArgList.empty() ? Result.append(")")
                                : Result.replace(Result.length() - 2, 2, ")");
}

Optional<std::string> MathCallExprRewriter::rewrite() {
  RewriteArgList = getMigratedArgs();
  setTargetCalleeName(SourceCalleeName);
  return buildRewriteString();
}

void MathCallExprRewriter::reportUnsupportedRoundingMode() {
  if (SourceCalleeName.endswith("_rd") || SourceCalleeName.endswith("_rn") ||
      SourceCalleeName.endswith("_ru") || SourceCalleeName.endswith("_rz")) {
    report(Diagnostics::ROUNDING_MODE_UNSUPPORTED);
  }
}

Optional<std::string> MathFuncNameRewriter::rewrite() {
  reportUnsupportedRoundingMode();
  RewriteArgList = getMigratedArgs();
  setTargetCalleeName(getNewFuncName());
  return buildRewriteString();
}

/// Policies to migrate math functions:
/// 1) Functions with the "std" namespace are treated as host functions;
/// 2) Functions with __device__ attribute but without __host__
///    attribute are treated as device functions;
/// 3) Functions whose calling functions are augmented with __device__
///    or __global__ attributes are treated as device functions;
/// 4) Other functions are treated as host functions.
///    eg. "__host__ __device__ fabs()" falls in 4) if fabs is not called in device or kernel
std::string MathFuncNameRewriter::getNewFuncName() {
  auto FD = Call->getDirectCallee();
  std::string NewFuncName;
  if (!FD) {
    NewFuncName = SourceCalleeName;
  } else {
    NewFuncName = TargetCalleeName;
    std::string NamespaceStr;
    auto DRE = dyn_cast<DeclRefExpr>(Call->getCallee()->IgnoreImpCasts());
    if (DRE) {
      auto Qualifier = DRE->getQualifier();
      if (Qualifier) {
        auto Namespace = Qualifier->getAsNamespace();
        if (Namespace)
          NamespaceStr = Namespace->getName();
      }
    }

    // For device functions
    if (NamespaceStr != "std" &&
        ((FD->hasAttr<CUDADeviceAttr>() && !FD->hasAttr<CUDAHostAttr>()) ||
         callingFuncHasDeviceAttr(Call))) {
      if (SourceCalleeName == "abs") {
        // further check the type of the args.
        if (!Call->getArg(0)->getType()->isIntegerType()) {
          NewFuncName = "cl::sycl::fabs";
        }
      }

      if (SourceCalleeName == "min" || SourceCalleeName == "max") {
        LangOptions LO;
        std::string FT = Call->getType().getAsString(PrintingPolicy(LO));
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          std::string ArgT =
              Call->getArg(i)->getType().getAsString(PrintingPolicy(LO));
          auto ArgExpr = Call->getArg(i)->getStmtClass();
          if (ArgT != FT || ArgExpr == Stmt::BinaryOperatorClass) {
            RewriteArgList[i] = "(" + FT + ")(" + RewriteArgList[i] + ")";
          }
        }
      }

      if (std::find(SingleFuctions.begin(), SingleFuctions.end(),
                    SourceCalleeName) != SingleFuctions.end()) {
        LangOptions LO;
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          if (SourceCalleeName == "ldexpf" && i == 1)
            continue;
          std::string ArgT =
              Call->getArg(i)->IgnoreImplicit()->getType().getAsString(
                  PrintingPolicy(LO));
          std::string ArgExpr = Call->getArg(i)->getStmtClassName();
          if (ArgT != "float" && ArgT != "double") {
            RewriteArgList[i] = "(float)(" + RewriteArgList[i] + ")";
          }
        }
      } else if (std::find(DoubleFuctions.begin(), DoubleFuctions.end(),
                           SourceCalleeName) != DoubleFuctions.end()) {
        LangOptions LO;
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          if (SourceCalleeName == "ldexp" && i == 1)
            continue;
          std::string ArgT =
              Call->getArg(i)->IgnoreImplicit()->getType().getAsString(
                  PrintingPolicy(LO));
          std::string ArgExpr = Call->getArg(i)->getStmtClassName();
          if (ArgT != "double" && ArgT != "float") {
            RewriteArgList[i] = "(double)(" + RewriteArgList[i] + ")";
          }
        }
      }
    }
    // For host functions
    else {
      // Insert "#include <cmath>" to migrated code
      DpctGlobalInfo::getInstance().insertHeader(Call->getBeginLoc(), Math);
      NewFuncName = SourceCalleeName;
      if (SourceCalleeName == "abs" || SourceCalleeName == "max" ||
          SourceCalleeName == "min") {
        auto *BT =
            dyn_cast<BuiltinType>(Call->getArg(0)->IgnoreImpCasts()->getType());
        if (BT) {
          auto K = BT->getKind();
          if (K == BuiltinType::Float) {
            NewFuncName = "f" + SourceCalleeName.str();
            NewFuncName += "f";
          } else if (K == BuiltinType::Double) {
            NewFuncName = "f" + SourceCalleeName.str();
          } else if (K == BuiltinType::LongDouble) {
            NewFuncName = "f" + SourceCalleeName.str();
            NewFuncName += "l";
          }
        }
      }
      if (NamespaceStr != "")
        NewFuncName = NamespaceStr + "::" + NewFuncName;
    }
  }
  return NewFuncName;
}

Optional<std::string> MathUnsupportedRewriter::rewrite() {
  report(Diagnostics::NOTSUPPORTED, SourceCalleeName);
  return Base::rewrite();
}

Optional<std::string> MathTypeCastRewriter::rewrite() {
  auto FD = Call->getDirectCallee();
  if (!FD || !FD->hasAttr<CUDADeviceAttr>())
    return Base::rewrite();

  using SSMap = std::map<std::string, std::string>;
  static SSMap RoundingModeMap{{"", "automatic"},
                               {"rd", "rtn"},
                               {"rn", "rte"},
                               {"ru", "rtp"},
                               {"rz", "rtz"}};
  const StringRef &FuncName = SourceCalleeName;
  std::string ReplStr;
  llvm::raw_string_ostream OS(ReplStr);

  auto MigratedArg0 = getMigratedArg(0);
  if (FuncName == "__float22half2_rn") {
    OS << MigratedArg0
       << ".convert<cl::sycl::half, cl::sycl::rounding_mode::rte>()";
  } else if (FuncName == "__float2half2_rn") {
    OS << "cl::sycl::float2{" << MigratedArg0 << "," << MigratedArg0
       << "}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>()";
  } else if (FuncName == "__floats2half2_rn") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << "cl::sycl::float2{" << MigratedArg0 << "," << MigratedArg1
       << "}.convert<cl::sycl::half, cl::sycl::rounding_mode::rte>()";
  } else if (FuncName == "__half22float2") {
    OS << MigratedArg0
       << ".convert<float, cl::sycl::rounding_mode::automatic>()";
  } else if (FuncName == "__half2half2") {
    OS << "cl::sycl::half2{" << MigratedArg0 << "," << MigratedArg0 << "}";
  } else if (FuncName == "__halves2half2") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << "cl::sycl::half2{" << MigratedArg0 << "," << MigratedArg1 << "}";
  } else if (FuncName == "__high2float") {
    OS << MigratedArg0 << ".get_value(0)";
  } else if (FuncName == "__high2half") {
    OS << MigratedArg0 << ".get_value(0)";
  } else if (FuncName == "__high2half2") {
    OS << "cl::sycl::half2{" << MigratedArg0 << ".get_value(0), "
       << MigratedArg0 << ".get_value(0)}";
  } else if (FuncName == "__highs2half2") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << "cl::sycl::half2{" << MigratedArg0 << ".get_value(0), "
       << MigratedArg1 << ".get_value(0)}";
  } else if (FuncName == "__low2float") {
    OS << MigratedArg0 << ".get_value(1)";
  } else if (FuncName == "__low2half") {
    OS << MigratedArg0 << ".get_value(1)";
  } else if (FuncName == "__low2half2") {
    OS << "cl::sycl::half2{" << MigratedArg0 << ".get_value(1), "
       << MigratedArg0 << ".get_value(1)}";
  } else if (FuncName == "__lowhigh2highlow") {
    OS << "cl::sycl::half2{" << MigratedArg0 << ".get_value(1), "
       << MigratedArg0 << ".get_value(0)}";
  } else if (FuncName == "__lows2half2") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << "cl::sycl::half2{" << MigratedArg0 << ".get_value(1), "
       << MigratedArg1 << ".get_value(1)}";
  } else {
    //__half2short_rd and __half2float
    static SSMap TypeMap{{"ll", "long long"},
                         {"ull", "unsigned long long"},
                         {"ushort", "unsigned short"},
                         {"uint", "unsigned int"},
                         {"half", "cl::sycl::half"}};
    std::string RoundingMode;
    if (FuncName[FuncName.size() - 3] == '_')
      RoundingMode = FuncName.substr(FuncName.size() - 2);
    auto FN = FuncName.substr(2, FuncName.find('_', 2) - 2);
    auto Types = split(FN, '2');
    assert(Types.size() == 2);
    MapNames::replaceName(TypeMap, Types[0]);
    MapNames::replaceName(TypeMap, Types[1]);
    OS << "cl::sycl::vec<" << Types[0] << ", 1>{" << MigratedArg0
       << "}.convert<" << Types[1]
       << ", cl::sycl::rounding_mode::" << RoundingModeMap[RoundingMode]
       << ">().get_value(0)";
  }
  OS.flush();
  return ReplStr;
}

bool isArgMigratedToAccessor(const CallExpr *Call, unsigned Index) {
  if (auto DRE = dyn_cast<DeclRefExpr>(Call->getArg(Index)->IgnoreImpCasts())) {
    if (!DRE->getDecl()->hasAttrs())
      return false;
    for (auto A : DRE->getDecl()->getAttrs()) {
      auto K = A->getKind();
      if (K == attr::CUDAConstant || K == attr::CUDADevice ||
          K == attr::CUDAShared)
        return true;
    }
  }
  return false;
}

std::string getTypecastName(const CallExpr *Call) {
  auto Arg0TypeName = Call->getArg(0)->getType().getAsString();
  auto Arg1TypeName = Call->getArg(1)->getType().getAsString();
  auto RetTypeName = Call->getType().getAsString();
  bool B0 = isArgMigratedToAccessor(Call, 0);
  bool B1 = isArgMigratedToAccessor(Call, 1);
  if (B0 && !B1)
    return Arg1TypeName;
  if (!B0 && B1)
    return Arg0TypeName;
  if (B0 && B1)
    return RetTypeName;
  return {};
}

Optional<std::string> MathSimulatedRewriter::rewrite() {
  report(Diagnostics::MATH_EMULATION, SourceCalleeName, TargetCalleeName);
  auto FD = Call->getDirectCallee();
  if (!FD || !FD->hasAttr<CUDADeviceAttr>())
    return Base::rewrite();

  const std::string FuncName = SourceCalleeName;
  std::string ReplStr;
  llvm::raw_string_ostream OS(ReplStr);
  auto MigratedArg0 = getMigratedArg(0);

  if (FuncName == "frexp" || FuncName == "frexpf") {
    std::string ArgT =
        Call->getArg(0)->IgnoreImplicit()->getType().getAsString(
            PrintingPolicy(LangOptions()));
    std::string ArgExpr = Call->getArg(0)->getStmtClassName();
    if (ArgT == "int") {
      if (FuncName == "frexpf")
        MigratedArg0 = "(float)(" + MigratedArg0 + ")";
      else
        MigratedArg0 = "(double)(" + MigratedArg0 + ")";
    }
    auto MigratedArg1 = getMigratedArg(1);
    OS << "cl::sycl::frexp(" << MigratedArg0 << ", cl::sycl::make_ptr<int, "
       << "cl::sycl::access::address_space::global_space>(" << MigratedArg1
       << "))";
  } else if (FuncName == "modf" || FuncName == "modff") {
    std::string ArgT =
        Call->getArg(0)->IgnoreImplicit()->getType().getAsString(
            PrintingPolicy(LangOptions()));
    std::string ArgExpr = Call->getArg(0)->getStmtClassName();
    if (ArgT == "int") {
      if (FuncName == "modff")
        MigratedArg0 = "(float)(" + MigratedArg0 + ")";
      else
        MigratedArg0 = "(double)(" + MigratedArg0 + ")";
    }
    auto MigratedArg1 = getMigratedArg(1);
    OS << "cl::sycl::modf(" << MigratedArg0;
    if (FuncName == "modf")
      OS << ", cl::sycl::make_ptr<double, "
            "cl::sycl::access::address_space::global_space>(";
    else
      OS << ", cl::sycl::make_ptr<float, "
            "cl::sycl::access::address_space::global_space>(";
    OS << MigratedArg1 << "))";
  } else if (FuncName == "nan" || FuncName == "nanf") {
    OS << "cl::sycl::nan(0u)";
  } else if (FuncName == "sincos" || FuncName == "sincosf" ||
             FuncName == "__sincosf") {
    std::string ArgT =
        Call->getArg(0)->IgnoreImplicit()->getType().getAsString(
            PrintingPolicy(LangOptions()));
    std::string ArgExpr = Call->getArg(0)->getStmtClassName();
    if (ArgT == "int") {
      if (FuncName == "sincosf" || FuncName == "__sincosf")
        MigratedArg0 = "(float)(" + MigratedArg0 + ")";
      else
        MigratedArg0 = "(double)(" + MigratedArg0 + ")";
    }
    auto MigratedArg1 = getMigratedArg(1);
    auto MigratedArg2 = getMigratedArg(2);
    if (MigratedArg1[0] == '&')
      OS << MigratedArg1.substr(1);
    else
      OS << "*(" + MigratedArg1 + ")";
    OS << " = cl::sycl::sincos(" << MigratedArg0;
    if (FuncName == "sincos")
      OS << ", cl::sycl::make_ptr<double, "
            "cl::sycl::access::address_space::global_space>(";
    else
      OS << ", cl::sycl::make_ptr<float, "
            "cl::sycl::access::address_space::global_space>(";
    OS << MigratedArg2 << "))";
  } else if (FuncName == "sincospi" || FuncName == "sincospif") {
    auto MigratedArg1 = getMigratedArg(1);
    auto MigratedArg2 = getMigratedArg(2);
    if (MigratedArg1[0] == '&')
      OS << MigratedArg1.substr(1);
    else
      OS << "*(" + MigratedArg1 + ")";
    OS << " = cl::sycl::sincos(" << MigratedArg0;
    if (FuncName == "sincospi")
      OS << " * DPCT_PI";
    else
      OS << " * DPCT_PI_F";

    if (FuncName == "sincospi")
      OS << ", cl::sycl::make_ptr<double, "
            "cl::sycl::access::address_space::global_space>(";
    else
      OS << ", cl::sycl::make_ptr<float, "
            "cl::sycl::access::address_space::global_space>(";
    OS << MigratedArg2 << "))";
  } else if (FuncName == "remquo" || FuncName == "remquof") {
    {
      std::string ArgT =
          Call->getArg(0)->IgnoreImplicit()->getType().getAsString(
              PrintingPolicy(LangOptions()));
      std::string ArgExpr = Call->getArg(0)->getStmtClassName();
      if (ArgT == "int") {
        if (FuncName == "remquof")
          MigratedArg0 = "(float)(" + MigratedArg0 + ")";
        else
          MigratedArg0 = "(double)(" + MigratedArg0 + ")";
      }
    }
    auto MigratedArg1 = getMigratedArg(1);
    {
      std::string ArgT =
          Call->getArg(1)->IgnoreImplicit()->getType().getAsString(
              PrintingPolicy(LangOptions()));
      std::string ArgExpr = Call->getArg(1)->getStmtClassName();
      if (ArgT == "int") {
        if (FuncName == "remquof")
          MigratedArg1 = "(float)(" + MigratedArg1 + ")";
        else
          MigratedArg1 = "(double)(" + MigratedArg1 + ")";
      }
    }
    auto MigratedArg2 = getMigratedArg(2);
    OS << "cl::sycl::remquo(" << MigratedArg0 << ", " << MigratedArg1
       << ", cl::sycl::make_ptr<int, "
          "cl::sycl::access::address_space::global_space>("
       << MigratedArg2 << "))";
  } else if (FuncName == "nearbyint" || FuncName == "nearbyintf") {
    OS << "cl::sycl::floor(" << MigratedArg0 << " + 0.5)";
  } else if (FuncName == "rhypot" || FuncName == "rhypotf") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << "1 / cl::sycl::hypot(" << MigratedArg0 << ", " << MigratedArg1 << ")";
  }
  OS.flush();
  return ReplStr;
}

Optional<std::string> MathBinaryOperatorRewriter::rewrite() {
  reportUnsupportedRoundingMode();
  if (SourceCalleeName == "__hneg" || SourceCalleeName == "__hneg2") {
    setLHS("");
    setRHS(getMigratedArg(0));
  } else {
    setLHS(getMigratedArg(0));
    setRHS(getMigratedArg(1));
  }
  return buildRewriteString();
}

Optional<std::string> WarpFunctionRewriter::rewrite() {
  if (SourceCalleeName == "__activemask" || SourceCalleeName == "__ballot" ||
      SourceCalleeName == "__ballot_sync") {
    report(Diagnostics::NOTSUPPORTED, SourceCalleeName);
    RewriteArgList = getMigratedArgs();
    setTargetCalleeName(SourceCalleeName);
  } else {
    if (SourceCalleeName == "__all" || SourceCalleeName == "__any") {
      RewriteArgList.emplace_back(getMigratedArg(0));
    } else if (SourceCalleeName == "__all_sync" ||
               SourceCalleeName == "__any_sync") {
      reportNoMaskWarning();
      RewriteArgList.emplace_back(getMigratedArg(1));
    } else if (SourceCalleeName.endswith("_sync")) {
      reportNoMaskWarning();
      RewriteArgList.emplace_back(getMigratedArg(1));
      RewriteArgList.emplace_back(getMigratedArg(2));
    } else {
      RewriteArgList.emplace_back(getMigratedArg(0));
      RewriteArgList.emplace_back(getMigratedArg(1));
    }
    setTargetCalleeName(buildString(
        DpctGlobalInfo::getItemName(), ".get_sub_group().",
        MapNames::findReplacedName(WarpFunctionsMap, SourceCalleeName)));
  }
  return buildRewriteString();
}

Optional<std::string> ReorderFunctionRewriter::rewrite() {
  for (auto ArgIdx : RewriterArgsIdx) {
    if (ArgIdx < Call->getNumArgs())
      appendRewriteArg(getMigratedArg(ArgIdx));
  }
  return buildRewriteString();
}

void TexFunctionRewriter::setTextureInfo() {
  const Expr *Obj = nullptr;
  std::string DataTy;
  int Dimension = 0, Idx = 0;
  auto &Global = DpctGlobalInfo::getInstance();
  if (Call->getArg(0)->getType()->isPointerType()) {
    DataTy = Global.getUnqualifiedTypeName(
        Call->getArg(0)->getType()->getPointeeType());
    Obj = Call->getArg(1);
    Dimension = Call->getNumArgs() - 2;
    Idx = 1;
  } else {
    DataTy =
        Global.getUnqualifiedTypeName(Call->getType().getUnqualifiedType());
    Obj = Call->getArg(0);
    Dimension = Call->getNumArgs() - 1;
    Idx = 0;
  }

  if (auto FD = DpctGlobalInfo::findAncestor<FunctionDecl>(Call)) {
    if (auto ObjInfo =
            DeviceFunctionDecl::LinkRedecls(FD)
                ->addCallee(Call)
                ->addTextureObjectArg(
                    Idx, dyn_cast<DeclRefExpr>(Obj->IgnoreImpCasts()))) {
      ObjInfo->setType(std::move(DataTy), Dimension);
    }
  }
}

#define REWRITER_FACTORY_ENTRY(FuncName, RewriterTy, ...)                      \
  {FuncName, std::make_shared<RewriterTy>(FuncName, __VA_ARGS__)},
#define FUNC_NAME_FACTORY_ENTRY(FuncName, RewriterName)                        \
  REWRITER_FACTORY_ENTRY(FuncName, FuncCallExprRewriterFactory, RewriterName)
#define MATH_FUNCNAME_FACTORY_ENTRY(FuncName, RewriterName)                    \
  REWRITER_FACTORY_ENTRY(FuncName, MathFuncNameRewriterFactory, RewriterName)
#define MATH_SIMULATED_FUNC_FACTORY_ENTRY(FuncName, RewriterName)              \
  REWRITER_FACTORY_ENTRY(FuncName, MathSimulatedRewriterFactory, RewriterName)
#define MATH_TYPECAST_FACTORY_ENTRY(FuncName)                                  \
  REWRITER_FACTORY_ENTRY(FuncName, MathTypeCastRewriterFactory, FuncName)
#define MATH_BO_FACTORY_ENTRY(FuncName, OpKind)                                \
  REWRITER_FACTORY_ENTRY(FuncName, MathBinaryOperatorRewriterFactory, OpKind)
#define MATH_UNSUPPORTED_FUNC_FACTORY_ENTRY(FuncName)                          \
  REWRITER_FACTORY_ENTRY(FuncName, MathUnsupportedRewriterFactory, FuncName)
#define WARP_FUNC_FACTORY_ENTRY(FuncName, RewriterName)                        \
  REWRITER_FACTORY_ENTRY(FuncName, WarpFunctionRewriterFactory, RewriterName)
#define REORDER_FUNC_FACTORY_ENTRY(FuncName, RewriterName, ...)                \
  REWRITER_FACTORY_ENTRY(FuncName, ReorderFunctionRewriterFactory,             \
                         RewriterName, std::vector<unsigned>{__VA_ARGS__})
#define TEX_FUNCTION_FACTORY_ENTRY(FuncName, RewriterName)                     \
  REWRITER_FACTORY_ENTRY(FuncName, TexFunctionRewriterFactory, RewriterName)
#define UNSUPPORTED_FACTORY_ENTRY(FuncName, MsgID)                             \
  REWRITER_FACTORY_ENTRY(FuncName, UnsupportFunctionRewriterFactory, MsgID)

const std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>
    CallExprRewriterFactoryBase::RewriterMap = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)                            \
  MATH_FUNCNAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME)                     \
  MATH_FUNCNAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME)                     \
  MATH_FUNCNAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)                           \
  MATH_SIMULATED_FUNC_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND) MATH_BO_FACTORY_ENTRY(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME) MATH_TYPECAST_FACTORY_ENTRY(APINAME)
#define ENTRY_UNSUPPORTED(APINAME) MATH_UNSUPPORTED_FUNC_FACTORY_ENTRY(APINAME)
#include "APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED

#define ENTRY_WARP(SOURCEAPINAME, TARGETAPINAME)                               \
  WARP_FUNC_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#include "APINamesWarp.inc"
#undef ENTRY_WARP

#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)                            \
  FUNC_NAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_TEXTURE(SOURCEAPINAME, TARGETAPINAME)                            \
  TEX_FUNCTION_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_UNSUPPORTED(SOURCEAPINAME, MSGID)                                \
  UNSUPPORTED_FACTORY_ENTRY(SOURCEAPINAME, MSGID)
#define ENTRY_REORDER(SOURCEAPINAME, TARGETAPINAME, ...)                       \
  REORDER_FUNC_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME, __VA_ARGS__)
#include "APINamesTexture.inc"
#undef ENTRY_RENAMED
#undef ENTRY_TEXTURE
#undef ENTRY_UNSUPPORTED
#undef UNSUPPORTED_FACTORY_ENTRY
};

const std::vector<std::string> MathFuncNameRewriter::SingleFuctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME)
#define ENTRY_UNSUPPORTED(APINAME)
#include "APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
};

const std::vector<std::string> MathFuncNameRewriter::DoubleFuctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME)
#define ENTRY_UNSUPPORTED(APINAME)
#include "APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
};

} // namespace dpct
} // namespace clang
