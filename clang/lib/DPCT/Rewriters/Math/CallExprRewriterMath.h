//===--------------- CallExprRewriterMath.h -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_REWRITERS_MATH_CALL_EXPR_REWRITER_MATH_H
#define DPCT_REWRITERS_MATH_CALL_EXPR_REWRITER_MATH_H

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
  static const std::vector<std::string> SingleFunctions;
  static const std::vector<std::string> DoubleFunctions;
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
inline bool useStdLibdevice() {
  return DpctGlobalInfo::useCAndCXXStandardLibrariesExt();
}

inline bool useMathLibdevice() { return DpctGlobalInfo::useIntelDeviceMath(); }

inline bool useExtBFloat16Math() {
  return DpctGlobalInfo::useExtBFloat16Math();
}

inline auto IsPerf = [](const CallExpr *C) -> bool {
  return DpctGlobalInfo::isOptimizeMigration();
};

inline auto UseIntelDeviceMath = [](const CallExpr *C) -> bool {
  return DpctGlobalInfo::useIntelDeviceMath();
};

inline auto UseBFloat16 = [](const CallExpr *C) -> bool {
  return DpctGlobalInfo::useBFloat16();
};

inline auto IsPureHost = [](const CallExpr *C) -> bool {
  const FunctionDecl *FD = C->getDirectCallee();
  if (!FD)
    return false;
  if (!(FD->hasAttr<CUDADeviceAttr>()))
    return true;

  SourceLocation DeclLoc =
      dpct::DpctGlobalInfo::getSourceManager().getExpansionLoc(
          FD->getLocation());
  clang::tooling::UnifiedPath DeclLocFilePath =
      dpct::DpctGlobalInfo::getLocInfo(DeclLoc).first;

  if (FD->getAttr<CUDADeviceAttr>()->isImplicit() &&
      FD->isConstexprSpecified() &&
      !isChildPath(dpct::DpctGlobalInfo::getCudaPath(), DeclLocFilePath)) {
    return true;
  }
  return false;
};
inline auto IsPureDevice = makeCheckAnd(
    HasDirectCallee(),
    makeCheckAnd(IsDirectCalleeHasAttribute<CUDADeviceAttr>(),
                 makeCheckNot(IsDirectCalleeHasAttribute<CUDAHostAttr>())));

inline auto IsDirectCallerPureDevice = [](const CallExpr *C) -> bool {
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
inline auto IsUnresolvedLookupExpr = [](const CallExpr *C) -> bool {
  return dyn_cast_or_null<UnresolvedLookupExpr>(C->getCallee());
};
inline auto UsingDpctMinMax = [](const CallExpr *C) -> bool {
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
} // namespace math

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

typedef std::unordered_map<std::string,
                           std::shared_ptr<CallExprRewriterFactoryBase>>
    RewriterMap;

RewriterMap createBfloat16PrecisionConversionAndDataMovementRewriterMap();
RewriterMap createCXXAPIRoutinesRewriterMap();
RewriterMap createDoublePrecisionIntrinsicsRewriterMap();
RewriterMap createDoublePrecisionMathematicalFunctionsRewriterMap();
RewriterMap createHalf2ArithmeticFunctionsRewriterMap();
RewriterMap createHalf2ComparisonFunctionsRewriterMap();
RewriterMap createHalf2MathFunctionsRewriterMap();
RewriterMap createHalfArithmeticFunctionsRewriterMap();
RewriterMap createHalfComparisonFunctionsRewriterMap();
RewriterMap createHalfMathFunctionsRewriterMap();
RewriterMap createHalfPrecisionConversionAndDataMovementRewriterMap();
RewriterMap createIntegerIntrinsicsRewriterMap();
RewriterMap createIntegerMathematicalFunctionsRewriterMap();
RewriterMap createOverloadRewriterMap();
RewriterMap createSIMDIntrinsicsRewriterMap();
RewriterMap createSinglePrecisionIntrinsicsRewriterMap();
RewriterMap createSinglePrecisionMathematicalFunctionsRewriterMap();
RewriterMap createSTDFunctionsRewriterMap();

} // namespace dpct
} // namespace clang

#endif // DPCT_REWRITERS_MATH_CALL_EXPR_REWRITER_MATH_H
