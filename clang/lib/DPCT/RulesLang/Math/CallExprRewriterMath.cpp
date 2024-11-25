//===--------------- CallExprRewriterMath.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterMath.h"
#include "RulesLang/MapNamesLang.h"

namespace clang {
namespace dpct {

std::optional<std::string> MathFuncNameRewriter::rewrite() {
  // If the function is not a target math function, do not migrate it
  if (!isTargetMathFunction(Call->getDirectCallee())) {
    // No actions needed here, just return an empty string
    return {};
  }

  reportUnsupportedRoundingMode();
  RewriteArgList = getMigratedArgs();
  auto NewFuncName = getNewFuncName();

  if (NewFuncName.empty() || NewFuncName == SourceCalleeName)
    return {};

  setTargetCalleeName(NewFuncName);
  return buildRewriteString();
}

/// Policies to migrate math functions:
/// 1) Functions with the "std" namespace are treated as host functions;
/// 2) Functions with __device__ attribute but without __host__
///    attribute are treated as device functions;
/// 3) Functions whose calling functions are augmented with __device__
///    or __global__ attributes are treated as device functions;
/// 4) Other functions are treated as host functions.
///    eg. "__host__ __device__ fabs()" falls in 5) if fabs is not called in
///    device or kernel
std::string MathFuncNameRewriter::getNewFuncName() {
  auto FD = Call->getDirectCallee();
  std::string NewFuncName;
  if (!FD) {
    NewFuncName = SourceCalleeName.str();
  } else {
    NewFuncName = TargetCalleeName;
    std::string NamespaceStr;
    auto DRE = dyn_cast<DeclRefExpr>(Call->getCallee()->IgnoreImpCasts());
    if (DRE) {
      auto Qualifier = DRE->getQualifier();
      if (Qualifier) {
        auto Namespace = Qualifier->getAsNamespace();
        if (Namespace)
          NamespaceStr = Namespace->getName().str();
      }
    }

    if (dpct::DpctGlobalInfo::isInAnalysisScope(FD->getBeginLoc())) {
      return "";
    }

    auto ContextFD = getImmediateOuterFuncDecl(Call);
    if (NamespaceStr == "std" && ContextFD &&
        !ContextFD->hasAttr<CUDADeviceAttr>() &&
        !ContextFD->hasAttr<CUDAGlobalAttr>()) {
      return "";
    }
    // For device functions
    else if ((FD->hasAttr<CUDADeviceAttr>() && !FD->hasAttr<CUDAHostAttr>()) ||
             (ContextFD && (ContextFD->hasAttr<CUDADeviceAttr>() ||
                            ContextFD->hasAttr<CUDAGlobalAttr>()))) {
      if (SourceCalleeName == "abs") {
        // further check the type of the args.
        if (!Call->getArg(0)->getType()->isIntegerType()) {
          NewFuncName = MapNames::getClNamespace(false, true) + "fabs";
        }
      }

      if (SourceCalleeName == "__clz" || SourceCalleeName == "__clzll") {
        LangOptions LO;
        auto Arg = Call->getArg(0);
        std::string ArgT =
            Arg->IgnoreImplicit()->getType().getAsString(PrintingPolicy(LO));
        auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
        auto IL = dyn_cast<IntegerLiteral>(Arg->IgnoreCasts());
        if (SourceCalleeName == "__clz") {
          if (ArgT != "int") {
            if (DRE || IL)
              RewriteArgList[0] = "(int)" + RewriteArgList[0];
            else
              RewriteArgList[0] = "(int)(" + RewriteArgList[0] + ")";
          }
        } else {
          if (ArgT != "long long") {
            if (DRE || IL)
              RewriteArgList[0] = "(long long)" + RewriteArgList[0];
            else
              RewriteArgList[0] = "(long long)(" + RewriteArgList[0] + ")";
          }
        }
      } else if (SourceCalleeName == "__mul24" || SourceCalleeName == "mul24" ||
                 SourceCalleeName == "__umul24" ||
                 SourceCalleeName == "umul24" ||
                 SourceCalleeName == "__mulhi") {
        std::string ParamType = "int";
        if (SourceCalleeName == "__umul24" || SourceCalleeName == "umul24")
          ParamType = "unsigned int";
        LangOptions LO;
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          auto Arg = Call->getArg(i)->IgnoreImpCasts();
          std::string ArgT =
              Arg->IgnoreImplicit()->getType().getAsString(PrintingPolicy(LO));
          auto ArgExpr = Arg->getStmtClass();
          if (ArgExpr == Stmt::PseudoObjectExprClass) {
            // The type of (blockDim/blockIdx/threadIdx/gridDim).(x/y/z) is
            // unsigned int but it is migrated to size_t (unsigned long in
            // typical 64-bit systems). However, sycl::mul24 only takes 32-bit
            // integers, so it is necessary to convert the migrated type to
            // int or unsigned int.
            if (isContainTargetSpecialExpr(Arg))
              RewriteArgList[i] = "(" + ParamType + ")" + RewriteArgList[i];
          } else {
            auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
            auto IL = dyn_cast<IntegerLiteral>(Arg->IgnoreCasts());
            auto FL = dyn_cast<FloatingLiteral>(Arg->IgnoreCasts());
            if (ArgT != ParamType) {
              if (DRE || IL || FL)
                RewriteArgList[i] = "(" + ParamType + ")" + RewriteArgList[i];
              else
                RewriteArgList[i] =
                    "(" + ParamType + ")(" + RewriteArgList[i] + ")";
            }
          }
        }
      } else if (MapNamesLang::MathTypeCastingMap.count(
                     SourceCalleeName.str())) {
        auto TypePair =
            MapNamesLang::MathTypeCastingMap[SourceCalleeName.str()];
        bool NeedFromType = false;
        if (auto ICE = dyn_cast_or_null<ImplicitCastExpr>(Call->getArg(0))) {
          if (ICE->getCastKind() != CastKind::CK_LValueToRValue)
            NeedFromType = true;
        }
        if (NeedFromType)
          NewFuncName =
              NewFuncName + "<" + TypePair.first + ", " + TypePair.second + ">";
        else
          NewFuncName = NewFuncName + "<" + TypePair.first + ">";
      }

      if (std::find(SingleFunctions.begin(), SingleFunctions.end(),
                    SourceCalleeName) != SingleFunctions.end()) {
        LangOptions LO;
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          auto Arg = Call->getArg(i);
          std::string ArgT =
              Arg->IgnoreImplicit()->getType().getCanonicalType().getAsString(
                  PrintingPolicy(LO));
          std::string ArgExpr = Arg->getStmtClassName();
          auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
          auto IL = dyn_cast<IntegerLiteral>(Arg->IgnoreCasts());
          std::string ParamType = "float";
          auto PVD = FD->getParamDecl(i);
          if (PVD) {
            ParamType = PVD->getType()
                            .getCanonicalType()
                            .getUnqualifiedType()
                            .getAsString();
          }
          // Since isnan is overloaded for both float and double, so there is no
          // need to add type conversions for isnan.
          if (ArgT != ParamType && SourceCalleeName != "isnan") {
            if (DRE || IL)
              RewriteArgList[i] = "(" + ParamType + ")" + RewriteArgList[i];
            else
              RewriteArgList[i] =
                  "(" + ParamType + ")(" + RewriteArgList[i] + ")";
          } else if (ParamType == "int" || ParamType == "unsigned int") {
            if (DRE || IL)
              RewriteArgList[i] = "(float)" + RewriteArgList[i];
            else
              RewriteArgList[i] = "(float)(" + RewriteArgList[i] + ")";
          }
        }
      } else if (std::find(DoubleFunctions.begin(), DoubleFunctions.end(),
                           SourceCalleeName) != DoubleFunctions.end()) {
        LangOptions LO;
        for (unsigned i = 0; i < Call->getNumArgs(); i++) {
          auto Arg = Call->getArg(i);
          std::string ArgT =
              Arg->IgnoreImplicit()->getType().getCanonicalType().getAsString(
                  PrintingPolicy(LO));
          std::string ArgExpr = Arg->getStmtClassName();
          auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
          auto IL = dyn_cast<IntegerLiteral>(Arg->IgnoreCasts());
          std::string ParamType = "double";
          auto PVD = FD->getParamDecl(i);
          if (PVD) {
            ParamType = PVD->getType()
                            .getCanonicalType()
                            .getUnqualifiedType()
                            .getAsString();
          }
          if (ArgT != ParamType) {
            if (DRE || IL)
              RewriteArgList[i] = "(" + ParamType + ")" + RewriteArgList[i];
            else
              RewriteArgList[i] =
                  "(" + ParamType + ")(" + RewriteArgList[i] + ")";
          } else if (ParamType == "int" || ParamType == "unsigned int") {
            if (DRE || IL)
              RewriteArgList[i] = "(double)" + RewriteArgList[i];
            else
              RewriteArgList[i] = "(double)(" + RewriteArgList[i] + ")";
          }
        }
      }
    }
    // For host functions
    else {
      // The vector type constructors (e.g. make_double3) are available in
      // the host, but should not need to include cmath nor be migrated to
      // SourceCalleeName.
      if (!SourceCalleeName.starts_with("make_")) {
        // Insert "#include <cmath>" to migrated code
        if (DpctGlobalInfo::getContext().getLangOpts().CUDA)
          DpctGlobalInfo::getInstance().insertHeader(Call->getBeginLoc(),
                                                     HT_Math);
        NewFuncName = SourceCalleeName.str();
      }

      if (SourceCalleeName == "abs") {
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

      if (!NamespaceStr.empty())
        NewFuncName = NamespaceStr + "::" + NewFuncName;
    }
  }
  return NewFuncName;
}

std::optional<std::string> MathCallExprRewriter::rewrite() {
  RewriteArgList = getMigratedArgs();
  setTargetCalleeName(SourceCalleeName.str());
  return buildRewriteString();
}

void MathCallExprRewriter::reportUnsupportedRoundingMode() {
  if (SourceCalleeName.ends_with("_rd") || SourceCalleeName.ends_with("_rn") ||
      SourceCalleeName.ends_with("_ru") || SourceCalleeName.ends_with("_rz")) {
    report(Diagnostics::ROUNDING_MODE_UNSUPPORTED, false);
  }
}

std::optional<std::string> MathUnsupportedRewriter::rewrite() {
  report(Diagnostics::API_NOT_MIGRATED, false,
         MapNames::ITFName.at(SourceCalleeName.str()));
  return Base::rewrite();
}

std::optional<std::string> MathTypeCastRewriter::rewrite() {
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

  auto MigratedArg0 = getMigratedArgWithExtraParens(0);
  if (FuncName == "__float22half2_rn") {
    OS << MigratedArg0
       << ".convert<" + MapNames::getClNamespace() + "half, " +
              MapNames::getClNamespace() + "rounding_mode::rte>()";
  } else if (FuncName == "__float2half2_rn") {
    OS << MapNames::getClNamespace() + "float2{" << MigratedArg0 << ","
       << MigratedArg0
       << "}.convert<" + MapNames::getClNamespace() + "half, " +
              MapNames::getClNamespace() + "rounding_mode::rte>()";
  } else if (FuncName == "__floats2half2_rn") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace() + "float2{" << MigratedArg0 << ","
       << MigratedArg1
       << "}.convert<" + MapNames::getClNamespace() + "half, " +
              MapNames::getClNamespace() + "rounding_mode::rte>()";
  } else if (FuncName == "__half22float2") {
    OS << MigratedArg0
       << ".convert<float, " + MapNames::getClNamespace() +
              "rounding_mode::automatic>()";
  } else if (FuncName == "__half2half2") {
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << ","
       << MigratedArg0 << "}";
  } else if (FuncName == "__halves2half2") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << ","
       << MigratedArg1 << "}";
  } else if (FuncName == "__high2half") {
    OS << MigratedArg0 << "[0]";
  } else if (FuncName == "__high2half2") {
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << "[0], "
       << MigratedArg0 << "[0]}";
  } else if (FuncName == "__highs2half2") {
    auto MigratedArg1 = getMigratedArgWithExtraParens(1);
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << "[0], "
       << MigratedArg1 << "[0]}";
  } else if (FuncName == "__low2half") {
    OS << MigratedArg0 << "[1]";
  } else if (FuncName == "__low2half2") {
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << "[1], "
       << MigratedArg0 << "[1]}";
  } else if (FuncName == "__lowhigh2highlow") {
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << "[1], "
       << MigratedArg0 << "[0]}";
  } else if (FuncName == "__lows2half2") {
    auto MigratedArg1 = getMigratedArgWithExtraParens(1);
    OS << MapNames::getClNamespace() + "half2{" << MigratedArg0 << "[1], "
       << MigratedArg1 << "[1]}";
  } else {
    //__half2short_rd and __half2float
    static SSMap TypeMap{{"ll", "long long"},
                         {"ull", "unsigned long long"},
                         {"ushort", "unsigned short"},
                         {"uint", "unsigned int"},
                         {"half", MapNames::getClNamespace() + "half"}};
    std::string RoundingMode;
    if (FuncName[FuncName.size() - 3] == '_')
      RoundingMode = FuncName.substr(FuncName.size() - 2).str();
    auto FN = FuncName.substr(2, FuncName.find('_', 2) - 2).str();
    auto Types = split(FN, '2');
    assert(Types.size() == 2);
    MapNames::replaceName(TypeMap, Types[0]);
    MapNames::replaceName(TypeMap, Types[1]);
    OS << MapNames::getClNamespace() + "vec<" << Types[0] << ", 1>{"
       << MigratedArg0 << "}.convert<" << Types[1]
       << ", " + MapNames::getClNamespace() + "rounding_mode::"
       << RoundingModeMap[RoundingMode] << ">()[0]";
  }
  OS.flush();
  return ReplStr;
}

std::optional<std::string> MathSimulatedRewriter::rewrite() {
  std::string NamespaceStr;
  auto DRE = dyn_cast<DeclRefExpr>(Call->getCallee()->IgnoreImpCasts());
  if (DRE) {
    auto Qualifier = DRE->getQualifier();
    if (Qualifier) {
      auto Namespace = Qualifier->getAsNamespace();
      if (Namespace)
        NamespaceStr = Namespace->getName().str();
    }
  }

  Analyzer.setCallSpelling(Call);

  auto FD = Call->getDirectCallee();
  if (!FD)
    return Base::rewrite();

  if (dpct::DpctGlobalInfo::isInAnalysisScope(FD->getBeginLoc())) {
    return {};
  }

  auto ContextFD = getImmediateOuterFuncDecl(Call);
  if (NamespaceStr == "std" && ContextFD &&
      !ContextFD->hasAttr<CUDADeviceAttr>() &&
      !ContextFD->hasAttr<CUDAGlobalAttr>()) {
    return {};
  }

  if (!FD->hasAttr<CUDADeviceAttr>() && ContextFD &&
      !ContextFD->hasAttr<CUDADeviceAttr>() &&
      !ContextFD->hasAttr<CUDAGlobalAttr>())
    return Base::rewrite();

  // Do not need to report warnings for pow, funnelshift, or drcp migrations
  if (SourceCalleeName != "pow" && SourceCalleeName != "powf" &&
      SourceCalleeName != "__powf" && SourceCalleeName != "__drcp_rd" &&
      SourceCalleeName != "__drcp_rn" && SourceCalleeName != "__drcp_ru" &&
      SourceCalleeName != "__drcp_rz")
    report(Diagnostics::MATH_EMULATION, false,
           MapNames::ITFName.at(SourceCalleeName.str()), TargetCalleeName);

  const std::string FuncName = SourceCalleeName.str();
  std::string ReplStr;
  llvm::raw_string_ostream OS(ReplStr);
  auto MigratedArg0 = getMigratedArg(0);

  if (FuncName == "frexp" || FuncName == "frexpf") {
    auto Arg = Call->getArg(0);
    std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
        PrintingPolicy(LangOptions()));
    std::string ArgExpr = Arg->getStmtClassName();
    auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
    if (ArgT == "int") {
      if (FuncName == "frexpf") {
        if (DRE)
          MigratedArg0 = "(float)" + MigratedArg0;
        else
          MigratedArg0 = "(float)(" + MigratedArg0 + ")";
      } else {
        if (DRE)
          MigratedArg0 = "(double)" + MigratedArg0;
        else
          MigratedArg0 = "(double)(" + MigratedArg0 + ")";
      }
    }
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace(false, true) + "frexp(" << MigratedArg0
       << ", " + MapNames::getClNamespace() + "address_space_cast<"
       << MapNames::getClNamespace() + "access::address_space::generic_space, "
       << MapNames::getClNamespace() + "access::decorated::yes>("
       << MigratedArg1 << "))";
  } else if (FuncName == "modf" || FuncName == "modff") {
    clang::QualType ParamType = Call->getArg(0)->getType().getCanonicalType();
    ParamType.removeLocalFastQualifiers(clang::Qualifiers::CVRMask);
    clang::QualType Arg0Type =
        Call->getArg(0)->IgnoreImpCasts()->getType().getCanonicalType();
    Arg0Type.removeLocalFastQualifiers(clang::Qualifiers::CVRMask);

    auto DRE = dyn_cast<DeclRefExpr>(Call->getArg(0)->IgnoreCasts());
    if (Arg0Type.getAsString() != ParamType.getAsString()) {
      if (DRE)
        MigratedArg0 = "(" + ParamType.getAsString() + ")" + MigratedArg0;
      else
        MigratedArg0 =
            "(" + ParamType.getAsString() + ")(" + MigratedArg0 + ")";
    }
    auto MigratedArg1 = getMigratedArg(1);
    OS << MapNames::getClNamespace(false, true) + "modf(" << MigratedArg0;

    OS << ", " + MapNames::getClNamespace() + "address_space_cast<"
       << MapNames::getClNamespace() + "access::address_space::generic_space, "
       << MapNames::getClNamespace() + "access::decorated::yes>("
       << MigratedArg1 << "))";
  } else if (FuncName == "nan" || FuncName == "nanf") {
    OS << MapNames::getClNamespace(false, true) + "nan(0u)";
  } else if (FuncName == "remquo" || FuncName == "remquof") {
    {
      auto Arg = Call->getArg(0);
      std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
          PrintingPolicy(LangOptions()));
      std::string ArgExpr = Arg->getStmtClassName();
      auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
      if (ArgT == "int") {
        if (FuncName == "remquof") {
          if (DRE)
            MigratedArg0 = "(float)" + MigratedArg0;
          else
            MigratedArg0 = "(float)(" + MigratedArg0 + ")";
        } else {
          if (DRE)
            MigratedArg0 = "(double)" + MigratedArg0;
          else
            MigratedArg0 = "(double)(" + MigratedArg0 + ")";
        }
      }
    }
    auto MigratedArg1 = getMigratedArg(1);
    {
      auto Arg = Call->getArg(1);
      std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
          PrintingPolicy(LangOptions()));
      std::string ArgExpr = Arg->getStmtClassName();
      auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
      if (ArgT == "int") {
        if (FuncName == "remquof") {
          if (DRE)
            MigratedArg1 = "(float)" + MigratedArg1;
          else
            MigratedArg1 = "(float)(" + MigratedArg1 + ")";
        } else {
          if (DRE)
            MigratedArg1 = "(double)" + MigratedArg1;
          else
            MigratedArg1 = "(double)(" + MigratedArg1 + ")";
        }
      }
    }
    auto MigratedArg2 = getMigratedArg(2);
    OS << MapNames::getClNamespace(false, true) + "remquo(" << MigratedArg0
       << ", " << MigratedArg1
       << ", " + MapNames::getClNamespace() + "address_space_cast<"
       << MapNames::getClNamespace() + "access::address_space::generic_space, "
       << MapNames::getClNamespace() + "access::decorated::yes>("
       << MigratedArg2 << "))";
  } else if (FuncName == "nearbyint" || FuncName == "nearbyintf") {
    OS << MapNames::getClNamespace(false, true) + "floor(" << MigratedArg0
       << " + 0.5)";
  } else if (FuncName == "rhypot" || FuncName == "rhypotf") {
    auto MigratedArg1 = getMigratedArg(1);
    OS << "1 / " + MapNames::getClNamespace(false, true) + "hypot("
       << MigratedArg0 << ", " << MigratedArg1 << ")";
  } else if (SourceCalleeName == "pow" || SourceCalleeName == "powf" ||
             SourceCalleeName == "__powf") {
    RewriteArgList = getMigratedArgs();
    if (Call->getNumArgs() != 2) {
      TargetCalleeName = SourceCalleeName.str();
      return buildRewriteString();
    }
    auto &Context = dpct::DpctGlobalInfo::getContext();
    auto PP = Context.getPrintingPolicy();
    PP.PrintCanonicalTypes = 1;
    auto Arg0 = Call->getArg(0);
    auto Arg1 = Call->getArg(1);

    std::string T0, T1;
    if (auto CXXFCE = dyn_cast<CXXFunctionalCastExpr>(Call->getArg(0))) {
      T0 = CXXFCE->getType().getAsString(PP);
    } else {
      T0 = Arg0->IgnoreCasts()->getType().getAsString(PP);
    }

    if (auto CXXFCE = dyn_cast<CXXFunctionalCastExpr>(Call->getArg(1))) {
      T1 = CXXFCE->getType().getAsString(PP);
    } else {
      T1 = Arg1->IgnoreCasts()->getType().getAsString(PP);
    }

    auto IL1 = dyn_cast<IntegerLiteral>(Arg1->IgnoreCasts());
    auto FL1 = dyn_cast<FloatingLiteral>(Arg1->IgnoreCasts());

    // For integer literal 2 or floating literal 2.0/2.0f, expand pow to
    // multiply expression:
    // pow(x, 2) ==> x * x, if x is an expression that has no side effects.
    // pow(x, 2.0) ==> x * x, if x is an expression that has no side effects.
    // pow(x, 2.0f) ==> x * x, if x is an expression that has no side effects.
    bool IsExponentTwo = false;
    if (IL1) {
      if (IL1->getValue().getZExtValue() == 2)
        IsExponentTwo = true;
    } else if (FL1) {
      if (!FL1->getBeginLoc().isMacroID() && !FL1->getEndLoc().isMacroID()) {
        llvm::APFloat FL1Value = FL1->getValue();
        if (FL1Value.compare(llvm::APFloat(FL1Value.getSemantics(), "2.0")) ==
            llvm::APFloat::cmpEqual)
          IsExponentTwo = true;
      }
    }
    SideEffectsAnalysis SEA(Arg0);
    SEA.setCallSpelling(Call);
    SEA.analyze();
    bool Arg0HasSideEffects = SEA.hasSideEffects();
    if (!Arg0HasSideEffects && IsExponentTwo) {
      auto Arg0Str = SEA.getRewriteString();
      if (!needExtraParens(Arg0))
        return Arg0Str + " * " + Arg0Str;
      else
        return "(" + Arg0Str + ") * (" + Arg0Str + ")";
    }
    return buildRewriteString();
  } else if (FuncName == "erfcxf") {
    OS << MapNames::getClNamespace(false, true) << "exp(" << MigratedArg0 << "*"
       << MigratedArg0 << ")*" << TargetCalleeName << "(" << MigratedArg0
       << ")";
  } else if (FuncName == "scalbln" || FuncName == "scalblnf" ||
             FuncName == "scalbn" || FuncName == "scalbnf") {
    OS << MigratedArg0 << "*(2<<" << getMigratedArg(1) << ")";
  } else if (FuncName == "__double2hiint") {
    requestFeature(HelperFeatureEnum::device_ext);
    OS << MapNames::getDpctNamespace() << "cast_double_to_int(" << MigratedArg0
       << ")";
  } else if (FuncName == "__double2loint") {
    requestFeature(HelperFeatureEnum::device_ext);
    OS << MapNames::getDpctNamespace() << "cast_double_to_int(" << MigratedArg0
       << ", false)";
  } else if (FuncName == "__hiloint2double") {
    requestFeature(HelperFeatureEnum::device_ext);
    OS << MapNames::getDpctNamespace() << "cast_ints_to_double(" << MigratedArg0
       << ", " << getMigratedArg(1) << ")";
  } else if (FuncName == "__sad" || FuncName == "__usad") {
    OS << TargetCalleeName << "(" << MigratedArg0 << ", " << getMigratedArg(1)
       << ")"
       << "+" << getMigratedArg(2);
  } else if (FuncName == "__drcp_rd" || FuncName == "__drcp_rn" ||
             FuncName == "__drcp_ru" || FuncName == "__drcp_rz") {
    auto Arg0 = Call->getArg(0);
    auto T0 = Arg0->IgnoreCasts()->getType();
    auto DRE0 = dyn_cast<DeclRefExpr>(Arg0->IgnoreCasts());
    report(Diagnostics::ROUNDING_MODE_UNSUPPORTED, false);
    if (T0->isSpecificBuiltinType(BuiltinType::Double)) {
      if (DRE0)
        OS << "(1.0/" << MigratedArg0 << ")";
      else
        OS << "(1.0/(" << MigratedArg0 << "))";
    } else if (T0->isSpecificBuiltinType(BuiltinType::Float)) {
      OS << TargetCalleeName;
      if (DRE0)
        OS << "((float)" << MigratedArg0 << ")";
      else
        OS << "((float)(" << MigratedArg0 << "))";
    } else {
      OS << TargetCalleeName;
      OS << "(" << MigratedArg0 << ")";
    }
  }

  OS.flush();
  return ReplStr;
}

std::optional<std::string> MathBinaryOperatorRewriter::rewrite() {
  reportUnsupportedRoundingMode();
  if (SourceCalleeName == "__hneg" || SourceCalleeName == "__hneg2") {
    setLHS("");
    setRHS(getMigratedArgWithExtraParens(0));
  } else {
    setLHS(getMigratedArgWithExtraParens(0));
    setRHS(getMigratedArgWithExtraParens(1));
  }
  return buildRewriteString();
}

void CallExprRewriterFactoryBase::initRewriterMapMath() {
  RewriterMap->merge(
      createBfloat16PrecisionConversionAndDataMovementRewriterMap());
  RewriterMap->merge(createCXXAPIRoutinesRewriterMap());
  RewriterMap->merge(createDoublePrecisionIntrinsicsRewriterMap());
  RewriterMap->merge(createDoublePrecisionMathematicalFunctionsRewriterMap());
  RewriterMap->merge(createHalf2ArithmeticFunctionsRewriterMap());
  RewriterMap->merge(createHalf2ComparisonFunctionsRewriterMap());
  RewriterMap->merge(createHalf2MathFunctionsRewriterMap());
  RewriterMap->merge(createHalfArithmeticFunctionsRewriterMap());
  RewriterMap->merge(createHalfComparisonFunctionsRewriterMap());
  RewriterMap->merge(createHalfMathFunctionsRewriterMap());
  RewriterMap->merge(createHalfPrecisionConversionAndDataMovementRewriterMap());
  RewriterMap->merge(createIntegerIntrinsicsRewriterMap());
  RewriterMap->merge(createIntegerMathematicalFunctionsRewriterMap());
  RewriterMap->merge(createOverloadRewriterMap());
  RewriterMap->merge(createSIMDIntrinsicsRewriterMap());
  RewriterMap->merge(createSinglePrecisionIntrinsicsRewriterMap());
  RewriterMap->merge(createSinglePrecisionMathematicalFunctionsRewriterMap());
  RewriterMap->merge(createSTDFunctionsRewriterMap());
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)                            \
  MATH_FUNCNAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME)                 \
  NO_REWRITE_FUNCNAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME)                     \
  MATH_FUNCNAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME)                     \
  MATH_FUNCNAME_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)                           \
  MATH_SIMULATED_FUNC_FACTORY_ENTRY(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND) MATH_BO_FACTORY_ENTRY(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME) MATH_TYPECAST_FACTORY_ENTRY(APINAME)
#define ENTRY_UNSUPPORTED(APINAME) MATH_UNSUPPORTED_FUNC_FACTORY_ENTRY(APINAME)
#define ENTRY_REWRITE(APINAME)
#include "RulesLang/APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_NO_REWRITE
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
#undef ENTRY_REWRITE
      }));
}

const std::vector<std::string> MathFuncNameRewriter::SingleFunctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME)
#define ENTRY_UNSUPPORTED(APINAME)
#define ENTRY_REWRITE(APINAME)
#include "RulesLang/APINamesMath.inc"
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_NO_REWRITE
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
#undef ENTRY_REWRITE
};

const std::vector<std::string> MathFuncNameRewriter::DoubleFunctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME)
#define ENTRY_UNSUPPORTED(APINAME)
#define ENTRY_REWRITE(APINAME)
#include "RulesLang/APINamesMath.inc"
#undef ENTRY_RENAMED_NO_REWRITE
#undef ENTRY_RENAMED
#undef ENTRY_RENAMED_SINGLE
#undef ENTRY_RENAMED_DOUBLE
#undef ENTRY_EMULATED
#undef ENTRY_OPERATOR
#undef ENTRY_TYPECAST
#undef ENTRY_UNSUPPORTED
#undef ENTRY_REWRITE
};

} // namespace dpct
} // namespace clang