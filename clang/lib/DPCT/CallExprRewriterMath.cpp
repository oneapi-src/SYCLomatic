//===--------------- CallExprRewriterMath.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"

namespace clang {
namespace dpct {

using MathFuncNameRewriterFactory =
    CallExprRewriterFactory<MathFuncNameRewriter, std::string>;
using MathUnsupportedRewriterFactory =
    CallExprRewriterFactory<MathUnsupportedRewriter, std::string>;
using MathSimulatedRewriterFactory =
    CallExprRewriterFactory<MathSimulatedRewriter, std::string>;
using MathTypeCastRewriterFactory =
    CallExprRewriterFactory<MathTypeCastRewriter, std::string>;
using MathBinaryOperatorRewriterFactory =
    CallExprRewriterFactory<MathBinaryOperatorRewriter, BinaryOperatorKind>;
using WarpFunctionRewriterFactory =
    CallExprRewriterFactory<WarpFunctionRewriter, std::string>;
using NoRewriteFuncNameRewriterFactory =
    CallExprRewriterFactory<NoRewriteFuncNameRewriter, std::string>;

/// Base class for rewriting math function calls
class MathCallExprRewriter : public FuncCallExprRewriter {
public:
  virtual std::optional<std::string> rewrite() override;

protected:
  MathCallExprRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : FuncCallExprRewriter(Call, SourceCalleeName, TargetCalleeName) {}

  void reportUnsupportedRoundingMode();
};

/// The rewriter for warning on unsupported math functions
class MathUnsupportedRewriter : public MathCallExprRewriter {
protected:
  using Base = MathCallExprRewriter;
  MathUnsupportedRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                          StringRef TargetCalleeName)
      : Base(Call, SourceCalleeName, TargetCalleeName) {}

  virtual std::optional<std::string> rewrite() override;

  friend MathUnsupportedRewriterFactory;
};

/// The rewriter for replacing math function calls with type casting expressions
class MathTypeCastRewriter : public MathCallExprRewriter {
protected:
  using Base = MathCallExprRewriter;
  MathTypeCastRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : Base(Call, SourceCalleeName, TargetCalleeName) {}

  virtual std::optional<std::string> rewrite() override;

  friend MathTypeCastRewriterFactory;
};

/// The rewriter for replacing math function calls with emulations
class MathSimulatedRewriter : public MathCallExprRewriter {
protected:
  using Base = MathCallExprRewriter;
  MathSimulatedRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                        StringRef TargetCalleeName)
      : Base(Call, SourceCalleeName, TargetCalleeName) {}

  virtual std::optional<std::string> rewrite() override;

  friend MathSimulatedRewriterFactory;
};

/// The rewriter for replacing math function calls with binary operator
/// expressions
class MathBinaryOperatorRewriter : public MathCallExprRewriter {
  std::string LHS, RHS;
  BinaryOperatorKind Op;

protected:
  MathBinaryOperatorRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                             BinaryOperatorKind Op)
      : MathCallExprRewriter(Call, SourceCalleeName, ""), Op(Op) {}

public:
  virtual ~MathBinaryOperatorRewriter() {}

  virtual std::optional<std::string> rewrite() override;

protected:
  void setLHS(std::string L) { LHS = L; }
  void setRHS(std::string R) { RHS = R; }

  // Build string which is used to replace original expression.
  inline std::optional<std::string> buildRewriteString() {
    if (LHS == "")
      return buildString(BinaryOperator::getOpcodeStr(Op), RHS);
    return buildString(LHS, " ", BinaryOperator::getOpcodeStr(Op), " ", RHS);
  }

  friend MathBinaryOperatorRewriterFactory;
};

/// The rewriter for renaming math function calls
class MathFuncNameRewriter : public MathCallExprRewriter {
protected:
  MathFuncNameRewriter(const CallExpr *Call, StringRef SourceCalleeName,
                       StringRef TargetCalleeName)
      : MathCallExprRewriter(Call, SourceCalleeName, TargetCalleeName) {}

public:
  virtual std::optional<std::string> rewrite() override;

protected:
  std::string getNewFuncName();
  static const std::vector<std::string> SingleFuctions;
  static const std::vector<std::string> DoubleFuctions;
  friend MathFuncNameRewriterFactory;
};

// Judge if a function is declared in user code or not
static inline bool isTargetMathFunction(const FunctionDecl *FD) {
  if (!FD)
    return false;
  auto FilePath = DpctGlobalInfo::getLocInfo(FD).first;
  if (isChildOrSamePath(DpctGlobalInfo::getAnalysisScope(), FilePath))
    return false;
  return true;
}

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
      } else if (MapNames::MathTypeCastingMap.count(SourceCalleeName.str())) {
        auto TypePair = MapNames::MathTypeCastingMap[SourceCalleeName.str()];
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

      if (std::find(SingleFuctions.begin(), SingleFuctions.end(),
                    SourceCalleeName) != SingleFuctions.end()) {
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
      } else if (std::find(DoubleFuctions.begin(), DoubleFuctions.end(),
                           SourceCalleeName) != DoubleFuctions.end()) {
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
      if (!SourceCalleeName.startswith("make_")) {
	// Insert "#include <cmath>" to migrated code
        DpctGlobalInfo::getInstance().insertHeader(Call->getBeginLoc(), HT_Math);
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
  if (SourceCalleeName.endswith("_rd") || SourceCalleeName.endswith("_rn") ||
      SourceCalleeName.endswith("_ru") || SourceCalleeName.endswith("_rz")) {
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
  if (SourceCalleeName != "pow"                     && 
      SourceCalleeName != "powf"                    &&
      SourceCalleeName != "__powf"                  && 
      SourceCalleeName != "__drcp_rd"               &&
      SourceCalleeName != "__drcp_rn"               &&
      SourceCalleeName != "__drcp_ru"               &&
      SourceCalleeName != "__drcp_rz")
    report(Diagnostics::MATH_EMULATION, false,
           MapNames::ITFName.at(SourceCalleeName.str()), TargetCalleeName);

  const std::string FuncName = SourceCalleeName.str();
  std::string ReplStr;
  llvm::raw_string_ostream OS(ReplStr);
  auto MigratedArg0 = getMigratedArg(0);

  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*Call);
  bool IsInReturnStmt = false;
  if (Parents.size())
    if (auto ParentStmt = getParentStmt(Call))
      if (ParentStmt->getStmtClass() == Stmt::StmtClass::ReturnStmtClass)
          IsInReturnStmt = true;

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
       << MapNames::getClNamespace() +
              "access::address_space::" + getAddressSpace(Call, 1)
       << ", " << MapNames::getClNamespace() + "access::decorated::yes"
       << ", "
       << "int"
       << ">(" << MigratedArg1 << "))";
  } else if (FuncName == "modf" || FuncName == "modff") {
    auto Arg = Call->getArg(0);
    std::string ArgT = Arg->IgnoreImplicit()->getType().getAsString(
        PrintingPolicy(LangOptions()));
    std::string ArgExpr = Arg->getStmtClassName();
    auto DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreCasts());
    if (ArgT == "int") {
      if (FuncName == "modff") {
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
    OS << MapNames::getClNamespace(false, true) + "modf(" << MigratedArg0;
    if (FuncName == "modf")
      OS << ", " + MapNames::getClNamespace() + "address_space_cast<"
         << MapNames::getClNamespace() +
                "access::address_space::" + getAddressSpace(Call, 1)
         << ", " << MapNames::getClNamespace() + "access::decorated::yes"
         << ", "
         << "double"
         << ">(";
    else
      OS << ", " + MapNames::getClNamespace() + "address_space_cast<"
         << MapNames::getClNamespace() +
                "access::address_space::" + getAddressSpace(Call, 1)
         << ", " << MapNames::getClNamespace() + "access::decorated::yes"
         << ", "
         << "float"
         << ">(";

    OS << MigratedArg1 << "))";
  } else if (FuncName == "nan" || FuncName == "nanf") {
    OS << MapNames::getClNamespace(false, true) + "nan(0u)";
  } else if (FuncName == "sincospi" || FuncName == "sincospif") {
    std::string Buf;
    llvm::raw_string_ostream RSO(Buf);

    auto MigratedArg1 = getMigratedArg(1);
    auto MigratedArg2 = getMigratedArg(2);
    if (MigratedArg1[0] == '&')
      RSO << MigratedArg1.substr(1);
    else
      RSO << "*(" + MigratedArg1 + ")";
    RSO << " = " + MapNames::getClNamespace(false, true) + "sincos("
       << MigratedArg0;
    if (FuncName == "sincospi") {
      RSO << " * DPCT_PI";
      requestFeature(HelperFeatureEnum::device_ext);
    } else {
      RSO << " * DPCT_PI_F";
      requestFeature(HelperFeatureEnum::device_ext);
    }

    if (FuncName == "sincospi")
      RSO << ", " + MapNames::getClNamespace() + "address_space_cast<"
          << MapNames::getClNamespace() +
                 "access::address_space::" + getAddressSpace(Call, 2)
          << ", " << MapNames::getClNamespace() + "access::decorated::yes"
          << ", "
          << "double"
          << ">(";
    else
      RSO << ", " + MapNames::getClNamespace() + "address_space_cast<"
          << MapNames::getClNamespace() +
                 "access::address_space::" + getAddressSpace(Call, 2)
          << ", " << MapNames::getClNamespace() + "access::decorated::yes"
          << ", "
          << "float"
          << ">(";

    RSO << MigratedArg2 << "))";
    if(IsInReturnStmt) {
      OS << "[&](){ " << Buf << ";"<< " }()";
      BlockLevelFormatFlag = true;
    } else {
      OS << Buf;
    }
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
       << MapNames::getClNamespace() +
              "access::address_space::" + getAddressSpace(Call, 2)
       << ", " << MapNames::getClNamespace() + "access::decorated::yes"
       << ", "
       << "int"
       << ">(" << MigratedArg2 << "))";
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
  } else if (FuncName == "erfcx" || FuncName == "erfcxf") {
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
  } else if (FuncName == "__drcp_rd" ||
             FuncName == "__drcp_rn" ||
             FuncName == "__drcp_ru" ||
             FuncName == "__drcp_rz") {
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

#define REWRITER_FACTORY_ENTRY(FuncName, RewriterFactory, ...)                 \
  {FuncName, std::make_shared<RewriterFactory>(FuncName, __VA_ARGS__)},
#define FUNC_NAME_FACTORY_ENTRY(FuncName, RewriterName)                        \
  REWRITER_FACTORY_ENTRY(FuncName, FuncCallExprRewriterFactory, RewriterName)
#define MATH_FUNCNAME_FACTORY_ENTRY(FuncName, RewriterName)                    \
  REWRITER_FACTORY_ENTRY(FuncName, MathFuncNameRewriterFactory, RewriterName)
#define NO_REWRITE_FUNCNAME_FACTORY_ENTRY(FuncName, NewName)                   \
  REWRITER_FACTORY_ENTRY(FuncName, NoRewriteFuncNameRewriterFactory,           \
                         NewName)
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
#define UNSUPPORTED_FACTORY_ENTRY(FuncName, MsgID)                             \
  REWRITER_FACTORY_ENTRY(FuncName, UnsupportFunctionRewriterFactory<>, MsgID)

namespace math {
bool useStdLibdevice() {
  return DpctGlobalInfo::useCAndCXXStandardLibrariesExt();
}

bool useMathLibdevice() {
  return DpctGlobalInfo::useIntelDeviceMath();
}

bool useExtBFloat16Math() { return DpctGlobalInfo::useExtBFloat16Math(); }

auto IsPerf = [](const CallExpr *C) -> bool {
  return DpctGlobalInfo::isOptimizeMigration();
};

inline auto UseIntelDeviceMath = [](const CallExpr *C) -> bool {
  return DpctGlobalInfo::useIntelDeviceMath();
};

inline auto UseBFloat16 = [](const CallExpr *C) -> bool {
  return DpctGlobalInfo::useBFloat16();
};

auto IsPureHost = [](const CallExpr *C) -> bool {
  const FunctionDecl *FD = C->getDirectCallee();
  if (!FD)
    return false;
  if (!(FD->hasAttr<CUDADeviceAttr>()))
    return true;

  SourceLocation DeclLoc =
      dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
          FD->getLocation());
  std::string DeclLocFilePath = dpct::DpctGlobalInfo::getLocInfo(DeclLoc).first;
  makeCanonical(DeclLocFilePath);

  if (FD->getAttr<CUDADeviceAttr>()->isImplicit() &&
      FD->isConstexprSpecified() &&
      !isChildPath(dpct::DpctGlobalInfo::getCudaPath(), DeclLocFilePath)) {
    return true;
  }
  return false;
};
auto IsPureDevice = makeCheckAnd(
    HasDirectCallee(),
    makeCheckAnd(IsDirectCalleeHasAttribute<CUDADeviceAttr>(),
                 makeCheckNot(IsDirectCalleeHasAttribute<CUDAHostAttr>())));

auto IsDirectCallerPureDevice = [](const CallExpr *C) -> bool {
  auto ContextFD = getImmediateOuterFuncDecl(C);
  while (auto LE = getImmediateOuterLambdaExpr(ContextFD)) {
    ContextFD = getImmediateOuterFuncDecl(LE);
  }
  if (!ContextFD)
    return false;
  if (ContextFD->getAttr<CUDADeviceAttr>() &&
      !ContextFD->getAttr<CUDAHostAttr>()) {
    return true;
  }
  return false;
};
auto IsUnresolvedLookupExpr = [](const CallExpr *C) -> bool {
  return dyn_cast_or_null<UnresolvedLookupExpr>(C->getCallee());
};
auto UsingDpctMinMax = [](const CallExpr *C) -> bool {
  if (IsUnresolvedLookupExpr(C))
    return true;
  if (C->getBeginLoc().isMacroID() || C->getEndLoc().isMacroID())
    return true;
  QualType Arg0T = C->getArg(0)->IgnoreImpCasts()->getType();
  QualType Arg1T = C->getArg(1)->IgnoreImpCasts()->getType();
  Arg0T.removeLocalFastQualifiers(Qualifiers::CVRMask);
  Arg1T.removeLocalFastQualifiers(Qualifiers::CVRMask);
  return Arg0T != Arg1T;
};
}

inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createMathAPIRewriterDeviceImpl(
    const std::string &Name, std::function<bool(const CallExpr *)> PerfPred,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        DevicePerf,
    std::array<
        std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>, 4>
        DeviceNodes) {
  if (DeviceNodes[0].second) {
    // DEVICE_NORMAL: SYCL API or helper function (impl by SYCL API)
    return createConditionalFactory(makeCheckAnd(math::IsPerf, PerfPred),
                                    std::move(DevicePerf),
                                    std::move(DeviceNodes[0]));
  }
  if (DeviceNodes[1].second) {
    // MATH_LIBDEVICE: sycl::ext::intel::math API
    if (math::useMathLibdevice()) {
      return createConditionalFactory(makeCheckAnd(math::IsPerf, PerfPred),
                                      std::move(DevicePerf),
                                      std::move(DeviceNodes[1]));
    }
  }
  if (DeviceNodes[2].second) {
    // DEVICE_STD: std API
    if (math::useStdLibdevice()) {
      return createConditionalFactory(makeCheckAnd(math::IsPerf, PerfPred),
                                      std::move(DevicePerf),
                                      std::move(DeviceNodes[2]));
    }
  }
  if (DeviceNodes[3].second) {
    // DEVICE_EMU: emulation
    return std::move(DeviceNodes[3]);
  }
  // report unsupport
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      {DeviceNodes[0].first,
       std::make_shared<UnsupportFunctionRewriterFactory<std::string>>(
           DeviceNodes[0].first, Diagnostics::API_NOT_MIGRATED,
           DeviceNodes[0].first)});
}

inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createMathAPIRewriterDeviceImpl(
    const std::string &Name,
    std::array<
        std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>, 4>
        DeviceNodes) {
  if (DeviceNodes[0].second) {
    // DEVICE_NORMAL: SYCL API or helper function (impl by SYCL API)
    return std::move(DeviceNodes[0]);
  }
  if (DeviceNodes[1].second) {
    // MATH_LIBDEVICE: sycl::ext::intel::math API
    if (math::useMathLibdevice()) {
      if (auto *CRF = dynamic_cast<MathSpecificElseEmuRewriterFactory *>(
              DeviceNodes[1].second.get())) {
        // Make the last DeviceNodes as the back up migration rule.
        CRF->setElse(std::move(DeviceNodes[3].second));
      }
      return std::move(DeviceNodes[1]);
    }
  }
  if (DeviceNodes[2].second) {
    // DEVICE_STD: std API
    if (math::useStdLibdevice()) {
      return std::move(DeviceNodes[2]);
    }
  }
  if (DeviceNodes[3].second) {
    // DEVICE_EMU: emulation
    return std::move(DeviceNodes[3]);
  }
  // report unsupport
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      {DeviceNodes[0].first,
       std::make_shared<UnsupportFunctionRewriterFactory<std::string>>(
           DeviceNodes[0].first, Diagnostics::API_NOT_MIGRATED,
           DeviceNodes[0].first)});
}

template <class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createMathAPIRewriterDevice(
    const std::string &Name, std::function<bool(const CallExpr *)> PerfPred,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&DevicePerf,
    T,
    const std::array<
        std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>, 4>
        &DeviceNodes) {
  return createConditionalFactory(
      math::IsPureDevice,
      createConditionalFactory(
          math::IsDefinedInCUDA(),
          std::move(createMathAPIRewriterDeviceImpl(Name, PerfPred, DevicePerf,
                                                    DeviceNodes)),
          {Name,
           std::make_shared<NoRewriteFuncNameRewriterFactory>(Name, Name)}),
      createConditionalFactory(
          math::IsUnresolvedLookupExpr,
          createConditionalFactory(
              math::IsDirectCallerPureDevice,
              std::move(createMathAPIRewriterDeviceImpl(
                  Name, PerfPred, DevicePerf, DeviceNodes)),
              {Name,
               std::make_shared<NoRewriteFuncNameRewriterFactory>(Name, Name)}),
          createConditionalFactory(
              math::IsDefinedInCUDA(),
              std::move(createMathAPIRewriterDeviceImpl(
                  Name, PerfPred, DevicePerf, DeviceNodes)),
              {Name, std::make_shared<NoRewriteFuncNameRewriterFactory>(
                         Name, Name)})));
}

inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createMathAPIRewriterDevice(
    const std::string &Name,
    const std::array<
        std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>, 4>
        &DeviceNodes) {
  return createConditionalFactory(
      math::IsPureDevice,
      createConditionalFactory(
          math::IsDefinedInCUDA(),
          std::move(createMathAPIRewriterDeviceImpl(Name, DeviceNodes)),
          {Name,
           std::make_shared<NoRewriteFuncNameRewriterFactory>(Name, Name)}),
      createConditionalFactory(
          math::IsUnresolvedLookupExpr,
          createConditionalFactory(
              math::IsDirectCallerPureDevice,
              std::move(createMathAPIRewriterDeviceImpl(Name, DeviceNodes)),
              {Name,
               std::make_shared<NoRewriteFuncNameRewriterFactory>(Name, Name)}),
          createConditionalFactory(
              math::IsDefinedInCUDA(),
              std::move(
                  createMathAPIRewriterDeviceImpl(Name, DeviceNodes)),
              {Name, std::make_shared<NoRewriteFuncNameRewriterFactory>(
                         Name, Name)})));
}

template <class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createMathAPIRewriterExperimentalBfloat16(
    const std::string &Name,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Rewriter1,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&Rewriter2,
    T) {
  if (DpctGlobalInfo::useBFloat16()) {
    if (math::useExtBFloat16Math() && Rewriter1.second)
      return createConditionalFactory(
          math::IsDefinedInCUDA(), std::move(Rewriter1),
          {Name,
           std::make_shared<NoRewriteFuncNameRewriterFactory>(Name, Name)});
    if (Rewriter2.second)
      return createConditionalFactory(
          math::IsDefinedInCUDA(), std::move(Rewriter2),
          {Name,
           std::make_shared<NoRewriteFuncNameRewriterFactory>(Name, Name)});
  }
  // report unsupport
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
      {Name, std::make_shared<UnsupportFunctionRewriterFactory<std::string>>(
                 Name, Diagnostics::API_NOT_MIGRATED, Name)});
}

template <class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createMathAPIRewriterHost(
    const std::string &Name,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&HostNormal,
    T) {
  return createConditionalFactory(
      math::IsDefinedInCUDA(), std::move(HostNormal),
      {Name, std::make_shared<NoRewriteFuncNameRewriterFactory>(Name, Name)});
}

template <class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createMathAPIRewriterHost(
    const std::string &Name, std::function<bool(const CallExpr *)> PerfPred,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&HostPerf,
    T,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        &&HostNormal,
    T) {
  return createConditionalFactory(
      math::IsDefinedInCUDA(),
      createConditionalFactory(makeCheckAnd(math::IsPerf, PerfPred),
                               std::move(HostPerf), std::move(HostNormal)),
      {Name, std::make_shared<NoRewriteFuncNameRewriterFactory>(Name, Name)});
}

#define EMPTY_FACTORY_ENTRY(NAME)                                              \
  std::make_pair(NAME, std::shared_ptr<CallExprRewriterFactoryBase>(nullptr)),

#define MATH_API_REWRITER_DEVICE_WITH_PERF(NAME, PERF_PRED, DEVICE_PERF, ...)  \
  createMathAPIRewriterDevice(NAME, PERF_PRED, DEVICE_PERF 0, __VA_ARGS__),
#define MATH_API_REWRITER_DEVICE(NAME, ...)                                    \
  createMathAPIRewriterDevice(NAME, __VA_ARGS__),
#define MATH_API_REWRITER_DEVICE_OVERLOAD(CONDITION, DEVICE_REWRITER_1,        \
                                          DEVICE_REWRITER_2)                   \
  createConditionalFactory(CONDITION, DEVICE_REWRITER_1 DEVICE_REWRITER_2 0),
#define MATH_API_REWRITER_EXPERIMENTAL_BFLOAT16(NAME, REWRITER_1, REWRITER_2)  \
  createMathAPIRewriterExperimentalBfloat16(NAME, REWRITER_1 REWRITER_2 0),
#define MATH_API_SPECIFIC_ELSE_EMU(CONDITION, DEVICE_REWRITER)                 \
  createMathSpecificElseEmuRewriterFactory(CONDITION, DEVICE_REWRITER 0),

#define MATH_API_REWRITER_HOST_WITH_PERF(NAME, PERF_PRED, HOST_PERF,           \
                                         HOST_NORMAL)                          \
  createMathAPIRewriterHost(NAME, PERF_PRED, HOST_PERF 0, HOST_NORMAL 0),
#define MATH_API_REWRITER_HOST(NAME, HOST_NORMAL)                              \
  createMathAPIRewriterHost(NAME, HOST_NORMAL 0),

#define MATH_API_REWRITER_HOST_DEVICE(HOST_REWRITER, DEVICE_REWRITER)          \
  createConditionalFactory(math::IsPureHost, HOST_REWRITER DEVICE_REWRITER 0),

template <typename T>
std::array<std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>,
           4>
makeMathAPIDeviceNodes(
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        DeviceNormal,
    T,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        MathLibDevice,
    T,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        DeviceStd,
    T,
    std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
        DeviceEmu,
    T) {
  return std::array<
      std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>, 4> {
    { DeviceNormal, MathLibDevice, DeviceStd, DeviceEmu }
  };
}

#define MATH_API_DEVICE_NODES(DEVICE_NORMAL, MATH_LIBDEVICE, DEVICE_STD,       \
                              DEVICE_EMU)                                      \
  makeMathAPIDeviceNodes(DEVICE_NORMAL 0, MATH_LIBDEVICE 0, DEVICE_STD 0,      \
                         DEVICE_EMU 0)

void CallExprRewriterFactoryBase::initRewriterMapMath() {
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
#include "APINamesMath.inc"
#include "APINamesMathRewrite.inc"
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

const std::vector<std::string> MathFuncNameRewriter::SingleFuctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME)
#define ENTRY_UNSUPPORTED(APINAME)
#define ENTRY_REWRITE(APINAME)
#include "APINamesMath.inc"
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

const std::vector<std::string> MathFuncNameRewriter::DoubleFuctions = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME)
#define ENTRY_UNSUPPORTED(APINAME)
#define ENTRY_REWRITE(APINAME)
#include "APINamesMath.inc"
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