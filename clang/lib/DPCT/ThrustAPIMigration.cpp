//===--------------- ThrustAPIMigration.cpp--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThrustAPIMigration.h"
#include "ASTTraversal.h"
#include "ExprAnalysis.h"
#include "TextModification.h"

namespace clang {
namespace dpct {

using namespace clang;
using namespace clang::dpct;
using namespace clang::ast_matchers;

void ThrustAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  // API register
  auto functionName = [&]() { return hasAnyName("on"); };
  MF.addMatcher(callExpr(callee(functionDecl(anyOf(
                             hasDeclContext(namespaceDecl(hasName("thrust"))),
                             functionName()))))
                    .bind("thrustFuncCall"),
                this);

  MF.addMatcher(
      unresolvedLookupExpr(hasAnyDeclaration(namedDecl(hasDeclContext(
                               namespaceDecl(hasName("thrust"))))),
                           hasParent(callExpr().bind("thrustApiCallExpr")))
          .bind("unresolvedThrustAPILookupExpr"),
      this);
}

void ThrustAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  ExprAnalysis EA;
  if (const auto ULExpr = getAssistNodeAsType<UnresolvedLookupExpr>(
          Result, "unresolvedThrustAPILookupExpr")) {
    if (const auto CE =
            getAssistNodeAsType<CallExpr>(Result, "thrustApiCallExpr"))
      thrustFuncMigration(Result, CE, ULExpr);
  } else if (const CallExpr *CE =
                 getNodeAsType<CallExpr>(Result, "thrustFuncCall")) {
    thrustFuncMigration(Result, CE);
  } else {
    return;
  }
}

void ThrustAPIRule::thrustFuncMigration(const MatchFinder::MatchResult &Result,
                                        const CallExpr *CE,
                                        const UnresolvedLookupExpr *ULExpr) {

  auto &SM = DpctGlobalInfo::getSourceManager();

  // handle the regular call expr
  std::string ThrustFuncName;
  if (ULExpr) {
    std::string Namespace;
    if (auto NNS = ULExpr->getQualifier()) {
      if (auto NS = NNS->getAsNamespace()) {
        Namespace = NS->getNameAsString();
      }
    }
    if (!Namespace.empty() && Namespace == "thrust")
      ThrustFuncName = ULExpr->getName().getAsString();
  } else {
    ThrustFuncName = CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  }
  // Process API: "thrust::cuda::par(thrust_allocator).on(stream)"
  const CXXMemberCallExpr *CMCE = dyn_cast_or_null<CXXMemberCallExpr>(CE);
  if (CMCE) {
    auto PP = DpctGlobalInfo::getContext().getPrintingPolicy();
    PP.PrintCanonicalTypes = true;
    auto BaseType = CMCE->getObjectType().getUnqualifiedType().getAsString(PP);
    StringRef BaseTypeRef(BaseType);
    if (BaseTypeRef.startswith("thrust::cuda_cub::execute_on_stream_base<") &&
        ThrustFuncName == "on") {
      std::string ReplStr = "oneapi::dpl::execution::make_device_policy(" +
                            getDrefName(CMCE->getArg(0)) + ")";
      emplaceTransformation(new ReplaceStmt(CMCE, std::move(ReplStr)));
      return;
    }
  }

  std::string ThrustFuncNameWithNamespace = "thrust::" + ThrustFuncName;

  auto ReplInfo =
      MapNames::ThrustFuncNamesMap.find(ThrustFuncNameWithNamespace);

  // For the API migration defined in APINamesThrust.inc
  if (ReplInfo == MapNames::ThrustFuncNamesMap.end()) {
    dpct::ExprAnalysis EA;
    EA.analyze(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  const unsigned NumArgs = CE->getNumArgs();
  auto QT = CE->getArg(0)->getType();
  LangOptions LO;
  std::string ArgT = QT.getAsString(PrintingPolicy(LO));

  // For the API migration defined in APINamesMapThrust.inc
  auto HelperFeatureIter = MapNames::ThrustFuncNamesHelperFeaturesMap.find(
      ThrustFuncNameWithNamespace);
  if (HelperFeatureIter != MapNames::ThrustFuncNamesHelperFeaturesMap.end()) {
    requestFeature(HelperFeatureIter->second, CE);
  }

  auto NewName = ReplInfo->second.ReplName;

  bool hasExecutionPolicy =
      ArgT.find("execution_policy_base") != std::string::npos;
  bool PolicyProcessed = false;

  if (ThrustFuncName == "sort") {
    auto ExprLoc = SM.getExpansionLoc(CE->getBeginLoc());
    if (SortULExpr.count(ExprLoc) != 0)
      return;
    else if (ULExpr) {
      SortULExpr.insert(ExprLoc);
    }
    if (NumArgs == 4) {
      hasExecutionPolicy = true;
    } else if (NumArgs == 3) {
      std::string FirstArgType = CE->getArg(0)->getType().getAsString();
      std::string SecondArgType = CE->getArg(1)->getType().getAsString();
      if (FirstArgType != SecondArgType)
        hasExecutionPolicy = true;
    }
  }
  // To migrate "thrust::cuda::par.on" that appears in CE' first arg to
  // "oneapi::dpl::execution::make_device_policy".
  const CallExpr *Call = nullptr;
  if (hasExecutionPolicy) {
    if (const auto *ICE = dyn_cast<ImplicitCastExpr>(CE->getArg(0))) {
      if (const auto *MT =
              dyn_cast<MaterializeTemporaryExpr>(ICE->getSubExpr())) {
        if (auto SubICE = dyn_cast<ImplicitCastExpr>(MT->getSubExpr())) {
          Call = dyn_cast<CXXMemberCallExpr>(SubICE->getSubExpr());
        }
      }
    } else if (const auto *SubCE = dyn_cast<CallExpr>(CE->getArg(0))) {
      Call = SubCE;
    } else {
      Call = dyn_cast<CXXMemberCallExpr>(CE->getArg(0));
    }
  }

  if (Call) {
    auto StreamArg = Call->getArg(0);
    std::ostringstream OS;
    if (const auto *ME = dyn_cast<MemberExpr>(Call->getCallee())) {
      auto BaseName =
          DpctGlobalInfo::getUnqualifiedTypeName(ME->getBase()->getType());
      if (BaseName == "thrust::cuda_cub::par_t") {
        OS << "oneapi::dpl::execution::make_device_policy(";
        printDerefOp(OS, StreamArg);
        OS << ")";
        emplaceTransformation(new ReplaceStmt(Call, OS.str()));
        PolicyProcessed = true;
      }
    }
  }

  // All the thrust APIs (such as thrust::copy, thrust::fill,
  // thrust::count, thrust::equal) called in device function , should be
  // migrated to oneapi::dpl APIs without a policy on the SYCL side
  if (auto FD = DpctGlobalInfo::getParentFunction(CE)) {
    if (FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) {
      if (ThrustFuncName == "sort") {
        report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
               "thrust::" + ThrustFuncName);
        return;
      } else if (hasExecutionPolicy) {
        emplaceTransformation(removeArg(CE, 0, *Result.SourceManager));
      }
    }
  }

  if (ThrustFuncName == "binary_search" &&
      (NumArgs <= 4 || (NumArgs == 5 && hasExecutionPolicy))) {
    // Currently, we do not support migration of 4 of the 8 overloaded versions
    // of thrust::binary_search.  The ones we do not support are the ones
    // searching for a single value instead of a vector of values
    //
    // Supported parameter profiles:
    // 1. (policy, firstIt, lastIt, valueFirstIt, valueLastIt, resultIt)
    // 2. (firstIt, lastIt, valueFirstIt, valueLastIt, resultIt)
    // 3. (policy, firstIt, lastIt, valueFirstIt, valueLastIt, resultIt, comp)
    // 4. (firstIt, lastIt, valueFirstIt, valueLastIt, resultIt, comp)
    //
    // Not supported parameter profiles:
    // 1. (policy, firstIt, lastIt, value)
    // 2. (firstIt, lastIt, value)
    // 3. (policy, firstIt, lastIt, value, comp)
    // 4. (firstIt, lastIt, value, comp)
    //
    // The logic in the above if condition filters out the ones
    // currently not supported and issues a warning
    report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
           "thrust::" + ThrustFuncName);
    return;
  } 

  if (ThrustFuncName == "sort") {
    
    // Rule of thrust::sort migration
    // #. thrust API
    //   dpcpp API
    // 1. thurst::sort(policy, h_vec.begin(), h_vec.end())
    //   std::sort(oneapi::dpl::exection::par_unseq, h_vec.begin(), h_vec.end())
    //
    // 2. thrust::sort(h_vec.begin(), h_vec.end())
    //   std::sort(h_vec.begin(), h_vec.end())
    //
    // 3. thrust::sort(policy, d_vec.begin(), d_vec.end())
    //   oneapi::dpl::sort(make_device_policy(queue), d_vec.begin(),
    //   d_vec.end())
    //
    // 4. thrust::sort(d_vec.begin(), d_vec.end())
    //   oneapi::dpl::sort(make_device_policy(queue), d_vec.begin(),
    //   d_vec.end())
    //
    // When thrust::sort inside template function and is a UnresolvedLookupExpr,
    // we will map to oneapi::dpl::sort

    auto IteratorArg = CE->getArg(1);
    auto IteratorType = IteratorArg->getType().getAsString();
    if (IteratorType.find("device_ptr") == std::string::npos) {
      if (hasExecutionPolicy) {
        emplaceTransformation(new ReplaceStmt(
            CE->getArg(0), "oneapi::dpl::execution::par_unseq"));
      }
      emplaceTransformation(new ReplaceCalleeName(CE, std::move("std::sort")));
      return;
    } else {
      if (PolicyProcessed) {
        emplaceTransformation(new ReplaceCalleeName(CE, std::move(NewName)));
        return;
      } else if (hasExecutionPolicy)
        emplaceTransformation(removeArg(CE, 0, *Result.SourceManager));
    }
  } else if (hasExecutionPolicy) {
    emplaceTransformation(new ReplaceCalleeName(CE, std::move(NewName)));
    return;
  }

  if (ULExpr) {
    if (hasExecutionPolicy && ThrustFuncName == "sort") {
      emplaceTransformation(removeArg(CE, 0, *Result.SourceManager));
    }
    auto BeginLoc = ULExpr->getBeginLoc();
    auto EndLoc = ULExpr->hasExplicitTemplateArgs()
                      ? ULExpr->getLAngleLoc().getLocWithOffset(-1)
                      : ULExpr->getEndLoc();
    emplaceTransformation(
        new ReplaceToken(BeginLoc, EndLoc, std::move(NewName)));
  } else {
    emplaceTransformation(new ReplaceCalleeName(CE, std::move(NewName)));
  }
  
  if (CE->getNumArgs() <= 0)
    return;
  auto ExtraParam = ReplInfo->second.ExtraParam;
  if (!ExtraParam.empty()) {
    // This is a temporary fix until, the Intel(R) oneAPI DPC++ Compiler and
    // Intel(R) oneAPI DPC++ Library support creating a SYCL execution policy
    // without creating a unique one for every use
    if (ExtraParam == "oneapi::dpl::execution::sycl") {
      // If no policy is specified and raw pointers are used
      // a host execution policy must be specified to match the thrust
      // behavior
      if (CE->getArg(0)->getType()->isPointerType()) {
        ExtraParam = "oneapi::dpl::execution::seq";
      } else {
        if (isPlaceholderIdxDuplicated(CE))
          return;
        ExtraParam = makeDevicePolicy(CE);
      }
    }
    emplaceTransformation(
        new InsertBeforeStmt(CE->getArg(0), ExtraParam + ", "));
  }
}

void ThrustTypeRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  // TYPE register
  auto ThrustTypeHasNames = [&]() {
    return hasAnyName("thrust::greater_equal", "thrust::less_equal",
                      "thrust::logical_and", "thrust::bit_and",
                      "thrust::bit_or", "thrust::minimum", "thrust::bit_xor");
  };
  MF.addMatcher(typeLoc(loc(hasCanonicalType(qualType(
                            hasDeclaration(namedDecl(ThrustTypeHasNames()))))))
                    .bind("thrustTypeLoc"),
                this);

  // CTOR register
  auto hasAnyThrustRecord = []() {
    return cxxRecordDecl(hasName("complex"),
                         hasDeclContext(namespaceDecl(hasName("thrust"))));
  };

  MF.addMatcher(
      cxxConstructExpr(hasType(hasAnyThrustRecord())).bind("thrustCtorExpr"),
      this);

  auto hasFunctionalActor = []() {
    return hasType(qualType(hasDeclaration(
        cxxRecordDecl(hasName("thrust::detail::functional::actor")))));
  };

  MF.addMatcher(
      cxxConstructExpr(anyOf(hasFunctionalActor(),
                             hasType(qualType(hasDeclaration(
                                 typedefNameDecl(hasFunctionalActor()))))))
          .bind("thrustCtorPlaceHolder"),
      this);

  // Var register
  auto hasPolicyName = [&]() { return hasAnyName("seq", "host", "device"); };

  MF.addMatcher(declRefExpr(to(varDecl(hasPolicyName()).bind("varDecl")))
                    .bind("declRefExpr"),
                this);
}

void ThrustTypeRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  ExprAnalysis EA;
  if (auto TL = getNodeAsType<TypeLoc>(Result, "thrustTypeLoc")) {
    EA.analyze(*TL);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  } else if (const CXXConstructExpr *CE =
                 getNodeAsType<CXXConstructExpr>(Result, "thrustCtorExpr")) {
    thrustCtorMigration(CE);
  } else if (const CXXConstructExpr *CE = getNodeAsType<CXXConstructExpr>(
                 Result, "thrustCtorPlaceHolder")) {
    // handle constructor expressions with placeholders (_1, _2, etc)
    replacePlaceHolderExpr(CE);
  } else if (auto DRE = getNodeAsType<DeclRefExpr>(Result, "declRefExpr")) {
    auto VD = getAssistNodeAsType<VarDecl>(Result, "varDecl", false);
    if (DRE->hasQualifier()) {

      auto ND = DRE->getQualifierLoc()
                           .getNestedNameSpecifier()
                           ->getAsNamespace();

      if (!ND||ND->getName() != "thrust")
        return;

      const std::string ThrustVarName = ND->getNameAsString() + "::" + VD->getName().str();

      std::string Replacement =
          MapNames::findReplacedName(MapNames::TypeNamesMap, ThrustVarName);
      insertHeaderForTypeRule(ThrustVarName, DRE->getBeginLoc());
      requestHelperFeatureForTypeNames(ThrustVarName, DRE);
      if (Replacement == "oneapi::dpl::execution::dpcpp_default")
        Replacement = makeDevicePolicy(DRE);

      if (!Replacement.empty()) {
        emplaceTransformation(new ReplaceToken(
            DRE->getBeginLoc(), DRE->getEndLoc(), std::move(Replacement)));
      }
    }
  } else {
    return;
  }
}

void ThrustTypeRule::replacePlaceHolderExpr(const CXXConstructExpr *CE) {
  unsigned PlaceholderCount = 0;

  auto placeholderStr = [](unsigned Num) {
    return std::string("_") + std::to_string(Num);
  };

  // Walk the expression and replace all placeholder occurrences
  std::function<void(const Stmt *)> walk = [&](const Stmt *S) {
    if (auto DRE = dyn_cast<DeclRefExpr>(S)) {
      auto DREStr = getStmtSpelling(DRE);
      auto TypeStr = DRE->getType().getAsString();
      std::string PlaceHolderTypeStr =
          "const thrust::detail::functional::placeholder<";
      if (TypeStr.find(PlaceHolderTypeStr) == 0) {
        unsigned PlaceholderNum =
            (TypeStr[PlaceHolderTypeStr.length()] - '0') + 1;
        if (PlaceholderNum > PlaceholderCount)
          PlaceholderCount = PlaceholderNum;
        emplaceTransformation(
            new ReplaceStmt(DRE, placeholderStr(PlaceholderNum)));
      }
      return;
    }
    for (auto SI : S->children())
      walk(SI);
  };
  walk(CE);

  if (PlaceholderCount == 0)
    // No placeholders were found, so no replacement is necessary
    return;

  // Construct the lambda wrapper and insert around placeholder expression
  std::string LambdaPrefix = "[=](";
  for (unsigned i = 1; i <= PlaceholderCount; ++i) {
    if (i > 1)
      LambdaPrefix += ",";
    LambdaPrefix += "auto _" + std::to_string(i);
  }
  LambdaPrefix += "){return ";
  emplaceTransformation(new InsertBeforeStmt(CE, std::move(LambdaPrefix)));
  std::string LambdaPostfix = ";}";
  emplaceTransformation(new InsertAfterStmt(CE, std::move(LambdaPostfix)));
}

void ThrustTypeRule::thrustCtorMigration(const CXXConstructExpr *CE) {
  // handle constructor expressions for thrust::complex
  std::string ExprStr = getStmtSpelling(CE);
  if (ExprStr.substr(0, 8) != "thrust::") {
    return;
  }
  auto P = ExprStr.find('<');
  if (P != std::string::npos) {
    auto ReplInfo = MapNames::ThrustFuncNamesMap.find(ExprStr);
    if (ReplInfo == MapNames::ThrustFuncNamesMap.end()) {
      return;
    }
    std::string ReplName = ReplInfo->second.ReplName;
    if (ReplName == "std::complex") {
      DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), HT_Complex);
    }
    emplaceTransformation(
        new ReplaceText(CE->getBeginLoc(), P, std::move(ReplName)));
  }
}

} // namespace dpct
} // namespace clang
