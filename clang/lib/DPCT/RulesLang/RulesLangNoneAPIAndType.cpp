//===--------------- RulesLangNoneAPIAndType.cpp--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "RuleInfra/CallExprRewriterCommon.h"
#include "RuleInfra/ExprAnalysis.h"
#include "RuleInfra/MigrationStatistics.h"
#include "RulesLang.h"
#include "TextModification.h"
#include "Utility.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Basic/Cuda.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/MacroArgs.h"
#include "llvm/ADT/StringSet.h"

#include <regex>
#include <string>
#include <utility>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::dpct;
using namespace clang::tooling;

extern clang::tooling::UnifiedPath
    DpctInstallPath; // Installation directory for this tool
extern DpctOption<opt, bool> ProcessAll;
extern DpctOption<opt, bool> AsyncHandler;

namespace clang {
namespace dpct {

void LinkageSpecDeclRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(linkageSpecDecl().bind("LinkageSpecDecl"), this);
}

void LinkageSpecDeclRule::runRule(const MatchFinder::MatchResult &Result) {
  const LinkageSpecDecl *LSD =
      getNodeAsType<LinkageSpecDecl>(Result, "LinkageSpecDecl");
  if (!LSD)
    return;
  if (LSD->getLanguage() != clang::LinkageSpecLanguageIDs::C)
    return;
  if (!LSD->hasBraces())
    return;

  SourceLocation Begin =
      DpctGlobalInfo::getSourceManager().getExpansionLoc(LSD->getExternLoc());
  SourceLocation End =
      DpctGlobalInfo::getSourceManager().getExpansionLoc(LSD->getRBraceLoc());
  auto BeginLocInfo = DpctGlobalInfo::getLocInfo(Begin);
  auto EndLocInfo = DpctGlobalInfo::getLocInfo(End);
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(BeginLocInfo.first);

  FileInfo->getExternCRanges().push_back(
      std::make_pair(BeginLocInfo.second, EndLocInfo.second));
}

void MemVarRefMigrationRule::registerMatcher(MatchFinder &MF) {
  auto DeclMatcher =
      varDecl(anyOf(hasAttr(attr::CUDAConstant), hasAttr(attr::CUDADevice),
                    hasAttr(attr::CUDAShared), hasAttr(attr::HIPManaged)),
              unless(hasAnyName("threadIdx", "blockDim", "blockIdx", "gridDim",
                                "warpSize")));
  MF.addMatcher(
      declRefExpr(anyOf(hasParent(implicitCastExpr(
                                      unless(hasParent(arraySubscriptExpr())))
                                      .bind("impl")),
                        anything()),
                  to(DeclMatcher.bind("decl")),
                  hasAncestor(functionDecl().bind("func")))
          .bind("used"),
      this);
}

void MemVarRefMigrationRule::runRule(const MatchFinder::MatchResult &Result) {
  auto getRHSOfTheNonConstAssignedVar =
      [](const DeclRefExpr *DRE) -> const Expr * {
    auto isExpectedRHS = [](const Expr *E, DynTypedNode Current,
                            QualType QT) -> bool {
      return (E == Current.get<Expr>()) && QT->isPointerType() &&
             !QT->getPointeeType().isConstQualified();
    };

    auto &Context = DpctGlobalInfo::getContext();
    DynTypedNode Current = DynTypedNode::create(*DRE);
    DynTypedNodeList Parents = Context.getParents(Current);
    while (!Parents.empty()) {
      const BinaryOperator *BO = Parents[0].get<BinaryOperator>();
      const VarDecl *VD = Parents[0].get<VarDecl>();
      if (BO) {
        if (BO->isAssignmentOp() &&
            isExpectedRHS(BO->getRHS(), Current, BO->getLHS()->getType()))
          return BO->getRHS();
      } else if (VD) {
        if (VD->hasInit() &&
            isExpectedRHS(VD->getInit(), Current, VD->getType()))
          return VD->getInit();
        return nullptr;
      }
      Current = Parents[0];
      Parents = Context.getParents(Current);
    }
    return nullptr;
  };

  auto MemVarRef = getNodeAsType<DeclRefExpr>(Result, "used");
  auto Func = getAssistNodeAsType<FunctionDecl>(Result, "func");
  auto Decl = getAssistNodeAsType<VarDecl>(Result, "decl");
  DpctGlobalInfo &Global = DpctGlobalInfo::getInstance();
  if (MemVarRef && Func && Decl) {
    if (isCubVar(Decl)) {
      return;
    }
    auto GetReplRange =
        [&](const Stmt *ReplaceNode) -> std::pair<SourceLocation, unsigned> {
      auto Range = getDefinitionRange(ReplaceNode->getBeginLoc(),
                                      ReplaceNode->getEndLoc());
      auto &SM = DpctGlobalInfo::getSourceManager();
      auto Begin = Range.getBegin();
      auto End = Range.getEnd();
      auto Length = Lexer::MeasureTokenLength(
          End, SM, dpct::DpctGlobalInfo::getContext().getLangOpts());
      Length +=
          SM.getDecomposedLoc(End).second - SM.getDecomposedLoc(Begin).second;
      return std::make_pair(Begin, Length);
    };
    const auto *Parent = getParentStmt(MemVarRef);
    bool HasTypeCasted = false;
    auto Info = Global.findMemVarInfo(Decl);

    if (Info && Info->isUseDeviceGlobal()) {
      auto VarType = Info->getType();
      if (VarType->isArray()) {
        if (const auto *const ICE =
                dyn_cast_or_null<ImplicitCastExpr>(Parent)) {
          if (ICE->getCastKind() == CK_ArrayToPointerDecay) {
            if (!dyn_cast_or_null<ArraySubscriptExpr>(getParentStmt(ICE))) {
              emplaceTransformation(new InsertAfterStmt(MemVarRef, ".get()"));
            }
          }
        }
      } else {
        emplaceTransformation(new InsertAfterStmt(MemVarRef, ".get()"));
      }
      return;
    }
    // 1. Handle assigning a 2 or more dimensions array pointer to a variable.
    if (const auto *const ICE = dyn_cast_or_null<ImplicitCastExpr>(Parent)) {
      if (const auto *arrType = MemVarRef->getType()->getAsArrayTypeUnsafe()) {
        if (ICE->getCastKind() == CK_ArrayToPointerDecay &&
            arrType->getElementType()->isArrayType() &&
            isAssignOperator(getParentStmt(Parent))) {
          auto Range = GetReplRange(MemVarRef);
          emplaceTransformation(
              new ReplaceText(Range.first, Range.second,
                              buildString("(", ICE->getType(), ")",
                                          Decl->getName(), ".get_ptr()")));
          HasTypeCasted = true;
        }
      }
    }
    // 2. Handle address-of operation.
    else if (const UnaryOperator *UO =
                 dyn_cast_or_null<UnaryOperator>(Parent)) {
      if (!Decl->hasAttr<CUDASharedAttr>() && UO->getOpcode() == UO_AddrOf) {
        CtTypeInfo TypeAnalysis(Decl, false);
        auto Range = GetReplRange(UO);
        if (TypeAnalysis.getDimension() >= 2) {
          // Dim >= 2
          emplaceTransformation(new ReplaceText(
              Range.first, Range.second,
              buildString("reinterpret_cast<", UO->getType(), ">(",
                          Decl->getName(), ".get_ptr())")));
          HasTypeCasted = true;
        } else if (TypeAnalysis.getDimension() == 1) {
          // Dim == 1
          emplaceTransformation(
              new ReplaceText(Range.first, Range.second,
                              buildString("reinterpret_cast<", UO->getType(),
                                          ">(&", Decl->getName(), ")")));
          HasTypeCasted = true;
        } else {
          // Dim == 0
          if (Decl->hasAttr<CUDAConstantAttr>() &&
              (MemVarRef->getType()->getTypeClass() !=
               Type::TypeClass::Elaborated)) {
            const Expr *RHS = getRHSOfTheNonConstAssignedVar(MemVarRef);
            if (RHS) {
              auto Range = GetReplRange(RHS);
              emplaceTransformation(new ReplaceText(
                  Range.first, Range.second,
                  buildString("const_cast<", RHS->getType(), ">(",
                              ExprAnalysis::ref(RHS), ")")));
              HasTypeCasted = true;
            }
          }
        }
      }
    }
    if (!HasTypeCasted && Decl->hasAttr<CUDAConstantAttr>() &&
        (MemVarRef->getType()->getTypeClass() ==
         Type::TypeClass::ConstantArray)) {
      const Expr *RHS = getRHSOfTheNonConstAssignedVar(MemVarRef);
      if (RHS) {
        auto Range = GetReplRange(RHS);
        emplaceTransformation(
            new ReplaceText(Range.first, Range.second,
                            buildString("const_cast<", RHS->getType(), ">(",
                                        ExprAnalysis::ref(RHS), ")")));
      }
    }
    auto VD = dyn_cast<VarDecl>(MemVarRef->getDecl());
    if (Func->isImplicit() ||
        Func->getTemplateSpecializationKind() == TSK_ImplicitInstantiation)
      return;
    if (VD == nullptr)
      return;
    auto Var = Global.findMemVarInfo(VD);
    if (Func->hasAttr<CUDAGlobalAttr>() || Func->hasAttr<CUDADeviceAttr>()) {
      if (DpctGlobalInfo::useGroupLocalMemory() &&
          VD->hasAttr<CUDASharedAttr>() && VD->getStorageClass() != SC_Extern) {
        if (!Var)
          return;
        if (auto B = dyn_cast_or_null<CompoundStmt>(Func->getBody())) {
          if (B->body_empty())
            return;
          emplaceTransformation(new InsertBeforeStmt(
              B->body_front(), Var->getDeclarationReplacement(VD)));
          return;
        }
      }
    } else {
      if (Var && !VD->getType()->isArrayType() &&
          VD->hasAttr<HIPManagedAttr>()) {
        emplaceTransformation(new InsertAfterStmt(MemVarRef, "[0]"));
      }
    }
  }
}

void ConstantMemVarMigrationRule::registerMatcher(MatchFinder &MF) {
  auto DeclMatcher =
      varDecl(hasAttr(attr::CUDAConstant),
              unless(hasAnyName("threadIdx", "blockDim", "blockIdx", "gridDim",
                                "warpSize")));
  MF.addMatcher(DeclMatcher.bind("var"), this);
  MF.addMatcher(varDecl(hasParent(translationUnitDecl())).bind("hostGlobalVar"),
                this);
}
// When using the --optimize-migration option, if the runtime symbol API does
// not utilize this constant variable, the C++ 'const' qualifier can be applied
// for migration. This approach eliminates the need for a helper class
// constant_memory and further processing of this variable's references.
void ConstantMemVarMigrationRule::runRule(
    const MatchFinder::MatchResult &Result) {
  std::string CanonicalType;
  if (auto MemVar = getAssistNodeAsType<VarDecl>(Result, "var")) {
    if (isCubVar(MemVar)) {
      return;
    }

    CanonicalType = MemVar->getType().getCanonicalType().getAsString();
    if (CanonicalType.find("block_tile_memory") != std::string::npos) {
      emplaceTransformation(new ReplaceVarDecl(MemVar, ""));
      return;
    }
    if (auto VTD = DpctGlobalInfo::findParent<VarTemplateDecl>(MemVar)) {
      report(VTD->getBeginLoc(), Diagnostics::TEMPLATE_VAR, false,
             MemVar->getName());
    }
    auto Info = MemVarInfo::buildMemVarInfo(MemVar);
    if (!Info)
      return;
    if (Info->isUseDeviceGlobal()) {
      Info->migrateToDeviceGlobal(MemVar);
      return;
    }

    Info->setIgnoreFlag(true);
    if (currentIsDevice(MemVar, Info))
      return;

    Info->setIgnoreFlag(false);
    if (!Info->isShared() && Info->getType()->getDimension() > 3 &&
        DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      report(MemVar->getBeginLoc(), Diagnostics::EXCEED_MAX_DIMENSION, true);
    }
    emplaceTransformation(ReplaceVarDecl::getVarDeclReplacement(
        MemVar, Info->getDeclarationReplacement(MemVar)));
    return;
  }

  if (auto VD = getNodeAsType<VarDecl>(Result, "hostGlobalVar")) {
    auto VarName = VD->getNameAsString();
    bool IsHost =
        !(VD->hasAttr<CUDAConstantAttr>() || VD->hasAttr<CUDADeviceAttr>() ||
          VD->hasAttr<CUDASharedAttr>() || VD->hasAttr<HIPManagedAttr>());
    if (IsHost) {
      dpct::DpctGlobalInfo::getGlobalVarNameSet().insert(VarName);

      if (currentIsHost(VD, VarName))
        return;
    }
  }
}

void ConstantMemVarMigrationRule::previousHCurrentD(const VarDecl *VD,
                                                    tooling::Replacement &R) {
  // 1. emit DPCT1055 warning
  // 2. add a new variable for host
  // 3. insert dpct::constant_memory and add the info from that replacement
  //     into current replacement.
  // 4. remove the replacement of removing "__constant__". In yaml case, clang
  //    replacement merging mechanism will occur error due to overlapping.
  //    The reason of setting offset as 0 is to avoid doing merge.
  //    4.1 About removing the replacement of removing "__constant__",
  //        e.g., the code is: static __constant__ a;
  //        the repl of removing "__constant__" and repl of replacing
  //        "static __constant__ a;" to "static dpct::constant_memory<float, 0>
  //        a;" are overlapped. And this merging is not in dpct but in clang's
  //        file (Replacement.cpp), clang's merging mechanism will occur error
  //        due to overlapping.
  //    4.2 About setting the offset equals to 0,
  //        if we keep the original offset, in clang's merging, a new merged
  //        replacement will be saved, and it will not contain the additional
  //        info we added. So we need to avoid this merge.
  // 5. remove previous DPCT1056 warning (will be handled in
  // removeHostConstantWarning)

  auto &SM = DpctGlobalInfo::getSourceManager();

  std::string HostVariableName = VD->getNameAsString() + "_host_ct1";
  report(VD->getBeginLoc(), Diagnostics::HOST_DEVICE_CONSTANT, false,
         VD->getNameAsString(), HostVariableName);

  std::string InitStr =
      VD->hasInit() ? ExprAnalysis::ref(VD->getInit()) : std::string("");
  std::string NewDecl =
      DpctGlobalInfo::getReplacedTypeName(VD->getType()) + " " +
      HostVariableName +
      (InitStr.empty() ? InitStr : std::string(" = " + InitStr)) + ";" +
      getNL() + getIndent(SM.getExpansionLoc(VD->getBeginLoc()), SM).str();
  if (VD->getStorageClass() == SC_Static)
    NewDecl = "static " + NewDecl;
  emplaceTransformation(new InsertText(SM.getExpansionLoc(VD->getBeginLoc()),
                                       std::move(NewDecl)));

  if (auto DeviceRepl = ReplaceVarDecl::getVarDeclReplacement(
          VD, MemVarInfo::buildMemVarInfo(VD)->getDeclarationReplacement(VD))) {
    DeviceRepl->setConstantFlag(dpct::ConstantFlagType::HostDevice);
    DeviceRepl->setConstantOffset(R.getConstantOffset());
    DeviceRepl->setInitStr(InitStr);
    DeviceRepl->setNewHostVarName(HostVariableName);
    emplaceTransformation(DeviceRepl);
  }

  R = tooling::Replacement(R.getFilePath(), 0, 0, "");
}

void ConstantMemVarMigrationRule::previousDCurrentH(const VarDecl *VD,
                                                    tooling::Replacement &R) {
  // 1. change DeviceConstant to HostDeviceConstant
  // 2. emit DPCT1055 warning (warning info is from previous device case)
  // 3. add a new variable for host (decl info is from previous device case)

  auto &SM = DpctGlobalInfo::getSourceManager();
  R.setConstantFlag(dpct::ConstantFlagType::HostDevice);

  std::string HostVariableName = R.getNewHostVarName();
  std::string InitStr = R.getInitStr();
  std::string NewDecl =
      DpctGlobalInfo::getReplacedTypeName(VD->getType()) + " " +
      HostVariableName +
      (InitStr.empty() ? InitStr : std::string(" = " + InitStr)) + ";" +
      getNL() + getIndent(SM.getExpansionLoc(VD->getBeginLoc()), SM).str();
  if (VD->getStorageClass() == SC_Static)
    NewDecl = "static " + NewDecl;

  SourceLocation SL = SM.getComposedLoc(
      SM.getDecomposedLoc(SM.getExpansionLoc(VD->getBeginLoc())).first,
      R.getConstantOffset());
  SL = DiagnosticsUtils::getStartOfLine(
      SL, SM, DpctGlobalInfo::getContext().getLangOpts(), false);
  report(SL, Diagnostics::HOST_DEVICE_CONSTANT, false, VD->getNameAsString(),
         HostVariableName);

  emplaceTransformation(new InsertText(SL, std::move(NewDecl)));
}

void ConstantMemVarMigrationRule::removeHostConstantWarning(Replacement &R) {
  std::string ReplStr = R.getReplacementText().str();

  // warning text of Diagnostics::HOST_CONSTANT
  std::string Warning = "The use of [_a-zA-Z][_a-zA-Z0-9]+ in device "
                        "code was not detected. If this variable is also used "
                        "in device code, you need to rewrite the code.";
  std::string Pattern =
      "/\\*\\s+DPCT" +
      std::to_string(static_cast<int>(Diagnostics::HOST_CONSTANT)) +
      ":[0-9]+: " + Warning + "\\s+\\*/" + getNL();
  std::regex RE(Pattern);
  std::smatch MRes;
  std::string Result;
  std::regex_replace(std::back_inserter(Result), ReplStr.begin(), ReplStr.end(),
                     RE, "");
  R.setReplacementText(Result);
}

bool ConstantMemVarMigrationRule::currentIsDevice(
    const VarDecl *MemVar, std::shared_ptr<MemVarInfo> Info) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto BeginLoc = SM.getExpansionLoc(MemVar->getBeginLoc());
  auto OffsetOfLineBegin = getOffsetOfLineBegin(BeginLoc, SM);
  auto BeginLocInfo = DpctGlobalInfo::getLocInfo(BeginLoc);
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(BeginLocInfo.first);
  auto &S = FileInfo->getConstantMacroTMSet();
  auto &Map = DpctGlobalInfo::getConstantReplProcessedFlagMap();
  for (auto &TM : S) {
    if (TM == nullptr)
      continue;
    if (!Map[TM]) {
      TransformSet->emplace_back(TM);
      Map[TM] = true;
    }
    if ((TM->getConstantFlag() == dpct::ConstantFlagType::Device ||
         TM->getConstantFlag() ==
             dpct::ConstantFlagType::HostDeviceInOnePass) &&
        TM->getLineBeginOffset() == OffsetOfLineBegin) {
      TM->setIgnoreTM(true);
      // current __constant__ variable used in device, using
      // OffsetOfLineBegin link the R(reomving __constant__) and
      // R(dcpt::constant_memery):
      // 1. check previous processed replacements, if found, do not check
      // info from yaml
      if (!FileInfo->getReplsSYCL())
        return false;
      auto &M = FileInfo->getReplsSYCL()->getReplMap();
      bool RemoveWarning = false;
      for (auto &R : M) {
        if ((R.second->getConstantFlag() == dpct::ConstantFlagType::Host ||
             R.second->getConstantFlag() ==
                 dpct::ConstantFlagType::HostDeviceInOnePass) &&
            R.second->getConstantOffset() == TM->getConstantOffset()) {
          // using flag and the offset of __constant__ to link
          // R(dcpt::constant_memery)  and R(reomving __constant__) from
          // previous execution, previous is host, current is device:
          previousHCurrentD(MemVar, *(R.second));
          dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(
              MemVar->getNameAsString());
          RemoveWarning = true;
          break;
        } else if ((R.second->getConstantFlag() ==
                        dpct::ConstantFlagType::Device ||
                    R.second->getConstantFlag() ==
                        dpct::ConstantFlagType::HostDevice) &&
                   R.second->getConstantOffset() == TM->getConstantOffset()) {
          TM->setIgnoreTM(true);
          return true;
        }
      }
      if (RemoveWarning) {
        for (auto &R : M) {
          if (R.second->getConstantOffset() == TM->getConstantOffset()) {
            removeHostConstantWarning(*(R.second));
            TM->setIgnoreTM(true);
            return true;
          }
        }
        TM->setIgnoreTM(true);
        return true;
      }

      // 2. if no info found, check info from yaml
      if (FileInfo->PreviousTUReplFromYAML) {
        auto &ReplsFromYAML = FileInfo->getReplacements();
        for (auto &R : ReplsFromYAML) {
          if (R.getConstantFlag() == dpct::ConstantFlagType::Host &&
              R.getConstantOffset() == TM->getConstantOffset()) {
            // using flag and the offset of __constant__ to link
            // R(dcpt::constant_memery) and R(reomving __constant__) from
            // previous execution previous is host, current is device:
            previousHCurrentD(MemVar, R);
            dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(
                MemVar->getNameAsString());
            RemoveWarning = true;
            break;
          } else if ((R.getConstantFlag() == dpct::ConstantFlagType::Device ||
                      R.getConstantFlag() ==
                          dpct::ConstantFlagType::HostDevice) &&
                     R.getConstantOffset() == TM->getConstantOffset()) {
            TM->setIgnoreTM(true);
            return true;
          }
        }
        if (RemoveWarning) {
          for (auto &R : ReplsFromYAML) {
            if (R.getConstantOffset() == TM->getConstantOffset()) {
              removeHostConstantWarning(R);
              TM->setIgnoreTM(true);
              return true;
            }
          }
          TM->setIgnoreTM(true);
          return true;
        }
      }
      if (Info->getType()->getDimension() > 3 &&
          DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
        report(MemVar->getBeginLoc(), Diagnostics::EXCEED_MAX_DIMENSION, true);
      }
      // Code here means this is the first migration, need save info to
      // replacement
      Info->setIgnoreFlag(false);
      TM->setIgnoreTM(true);
      auto ReplaceStr = Info->getDeclarationReplacement(MemVar);
      auto SourceFileType = GetSourceFileType(Info->getFilePath());
      if ((SourceFileType == SPT_CudaHeader ||
           SourceFileType == SPT_CppHeader) &&
          !Info->isStatic()) {
        ReplaceStr = "inline " + ReplaceStr;
      }
      auto RVD =
          ReplaceVarDecl::getVarDeclReplacement(MemVar, std::move(ReplaceStr));
      if (!RVD)
        return true;
      RVD->setConstantFlag(TM->getConstantFlag());
      RVD->setConstantOffset(TM->getConstantOffset());
      RVD->setInitStr(MemVar->hasInit() ? ExprAnalysis::ref(MemVar->getInit())
                                        : std::string(""));
      RVD->setNewHostVarName(MemVar->getNameAsString() + "_host_ct1");
      emplaceTransformation(RVD);
      return true;
    }
  }
  return false;
}

bool ConstantMemVarMigrationRule::currentIsHost(const VarDecl *VD,
                                                std::string VarName) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto BeginLoc = SM.getExpansionLoc(VD->getBeginLoc());
  auto OffsetOfLineBegin = getOffsetOfLineBegin(BeginLoc, SM);
  auto BeginLocInfo = DpctGlobalInfo::getLocInfo(BeginLoc);
  auto FileInfo = DpctGlobalInfo::getInstance().insertFile(BeginLocInfo.first);
  if (!FileInfo)
    return false;
  auto &S = FileInfo->getConstantMacroTMSet();
  auto &Map = DpctGlobalInfo::getConstantReplProcessedFlagMap();
  for (auto &TM : S) {
    if (TM == nullptr)
      continue;

    if (!Map[TM]) {
      TransformSet->emplace_back(TM);
      Map[TM] = true;
    }
    if ((TM->getConstantFlag() == dpct::ConstantFlagType::Host ||
         TM->getConstantFlag() ==
             dpct::ConstantFlagType::HostDeviceInOnePass) &&
        TM->getLineBeginOffset() == OffsetOfLineBegin) {
      // current __constant__ variable used in host, using OffsetOfLineBegin
      // link the R(reomving __constant__) and here

      // 1. check previous processed replacements, if found, do not check
      // info from yaml

      if (!FileInfo->getReplsSYCL())
        return false;
      auto &M = FileInfo->getReplsSYCL()->getReplMap();
      for (auto &R : M) {
        if ((R.second->getConstantFlag() == dpct::ConstantFlagType::Device ||
             R.second->getConstantFlag() ==
                 dpct::ConstantFlagType::HostDeviceInOnePass) &&
            R.second->getConstantOffset() == TM->getConstantOffset()) {
          if (R.second->getConstantFlag() ==
              dpct::ConstantFlagType::HostDeviceInOnePass) {
            if (R.second->getNewHostVarName().empty()) {
              continue;
            }
          }
          // using flag and the offset of __constant__ to link previous
          // execution of previous is device, current is host:
          previousDCurrentH(VD, *(R.second));
          dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(VarName);
          TM->setIgnoreTM(true);
          return true;
        } else if ((R.second->getConstantFlag() ==
                        dpct::ConstantFlagType::Host ||
                    R.second->getConstantFlag() ==
                        dpct::ConstantFlagType::HostDevice) &&
                   R.second->getConstantOffset() == TM->getConstantOffset()) {
          if (R.second->getConstantFlag() == dpct::ConstantFlagType::HostDevice)
            dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(VarName);
          TM->setIgnoreTM(true);
          return true;
        }
      }

      // 2. if no info found, check info from yaml
      if (FileInfo->PreviousTUReplFromYAML) {
        auto &ReplsFromYAML = FileInfo->getReplacements();
        for (auto &R : ReplsFromYAML) {
          if (R.getConstantFlag() == dpct::ConstantFlagType::Device &&
              R.getConstantOffset() == TM->getConstantOffset()) {
            // using flag and the offset of __constant__ to link here and
            // R(reomving __constant__) from previous execution, previous is
            // device, current is host.
            previousDCurrentH(VD, R);
            dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(VarName);
            TM->setIgnoreTM(true);
            return true;
          } else if ((R.getConstantFlag() == dpct::ConstantFlagType::Host ||
                      R.getConstantFlag() ==
                          dpct::ConstantFlagType::HostDevice) &&
                     R.getConstantOffset() == TM->getConstantOffset()) {
            if (R.getConstantFlag() == dpct::ConstantFlagType::HostDevice)
              dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(VarName);
            TM->setIgnoreTM(true);
            return true;
          }
        }
      }

      // Code here means this is the first migration, only emit a warning
      // Add the constant offset in the replacement
      // The constant offset will be used in previousHCurrentD to distinguish
      // unnecessary warnings.
      if (TM->getConstantFlag() == dpct::ConstantFlagType::Host) {
        dpct::DpctGlobalInfo::removeVarNameInGlobalVarNameSet(VarName);
        if (report(VD->getBeginLoc(), Diagnostics::HOST_CONSTANT, false,
                   VD->getNameAsString())) {
          TransformSet->back()->setConstantOffset(TM->getConstantOffset());
        }
      }
    }
  }
  return false;
}

void MemVarMigrationRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto DeclMatcherWithoutConstant =
      varDecl(anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAShared),
                    hasAttr(attr::HIPManaged)),
              unless(hasAnyName("threadIdx", "blockDim", "blockIdx", "gridDim",
                                "warpSize")));
  MF.addMatcher(DeclMatcherWithoutConstant.bind("var"), this);
}

void MemVarMigrationRule::processTypeDeclaredLocal(
    const VarDecl *MemVar, std::shared_ptr<MemVarInfo> Info) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto DS = Info->getDeclStmtOfVarType();
  if (!DS)
    return;
  // this token is ';'
  auto InsertSL = getDefinitionRange(DS->getBeginLoc(), DS->getEndLoc())
                      .getEnd()
                      .getLocWithOffset(1);
  auto GenDeclStmt = [=, &SM](StringRef TypeName) -> std::string {
    bool IsReference = !Info->getType()->getDimension();
    std::string Ret;
    llvm::raw_string_ostream OS(Ret);
    OS << getNL(DS->getEndLoc().isMacroID()) << getIndent(InsertSL, SM);
    OS << TypeName << ' ';
    if (IsReference)
      OS << '&';
    else
      OS << '*';
    OS << Info->getName();
    OS << " = ";
    if (IsReference)
      OS << '*';
    // add typecast for the __shared__ variable, since after migration the
    // __shared__ variable type will be uint8_t*
    OS << '(' << TypeName << " *)";
    OS << Info->getNameAppendSuffix() << ';';
    return OS.str();
  };
  if (Info->isAnonymousType()) {
    // keep the origin type declaration, only remove variable name
    //  }  a_variable  ,  b_variable ;
    //   |                |
    // begin             end
    // ReplaceToken replacing [begin, end]
    auto BR = Info->getDeclOfVarType()->getBraceRange();
    auto DRange = getDefinitionRange(BR.getBegin(), BR.getEnd());
    auto BeginWithOffset =
        DRange.getEnd().getLocWithOffset(1); // this token is }
    SourceLocation End =
        getDefinitionRange(MemVar->getBeginLoc(), MemVar->getEndLoc()).getEnd();
    emplaceTransformation(new ReplaceToken(BeginWithOffset, End, ""));

    std::string NewTypeName = Info->getLocalTypeName();

    // add a typename
    emplaceTransformation(new InsertText(DRange.getBegin(), " " + NewTypeName));

    // add typecast for the __shared__ variable, since after migration the
    // __shared__ variable type will be uint8_t*
    emplaceTransformation(new InsertText(InsertSL, GenDeclStmt(NewTypeName)));
  } else if (DS) {
    // remove var decl
    emplaceTransformation(ReplaceVarDecl::getVarDeclReplacement(
        MemVar, Info->getDeclarationReplacement(MemVar)));

    Info->setLocalTypeName(Info->getType()->getBaseName());
    emplaceTransformation(
        new InsertText(InsertSL, GenDeclStmt(Info->getType()->getBaseName())));
  }
}

void MemVarMigrationRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto MemVar = getAssistNodeAsType<VarDecl>(Result, "var")) {
    if (isCubVar(MemVar) || MemVar->hasAttr<CUDAConstantAttr>()) {
      return;
    }
    std::string CanonicalType =
        MemVar->getType().getCanonicalType().getAsString();
    if (CanonicalType.find("block_tile_memory") != std::string::npos) {
      emplaceTransformation(new ReplaceVarDecl(MemVar, ""));
      return;
    }
    auto Info = MemVarInfo::buildMemVarInfo(MemVar);
    if (!Info)
      return;
    if (Info->isUseDeviceGlobal()) {
      Info->migrateToDeviceGlobal(MemVar);
      return;
    }

    if (auto VTD = DpctGlobalInfo::findParent<VarTemplateDecl>(MemVar)) {
      report(VTD->getBeginLoc(), Diagnostics::TEMPLATE_VAR, false,
             MemVar->getName());
    }
    if (Info->isTypeDeclaredLocal()) {
      processTypeDeclaredLocal(MemVar, Info);
    } else {
      if (!Info->isShared() && Info->getType()->getDimension() > 3 &&
          DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
        report(MemVar->getBeginLoc(), Diagnostics::EXCEED_MAX_DIMENSION, true);
      }
      emplaceTransformation(ReplaceVarDecl::getVarDeclReplacement(
          MemVar, Info->getDeclarationReplacement(MemVar)));
    }
    return;
  }
}

/// __constant__/__shared__/__device__ var information collection
void MemVarAnalysisRule::registerMatcher(MatchFinder &MF) {
  auto DeclMatcher =
      varDecl(anyOf(hasAttr(attr::CUDAConstant), hasAttr(attr::CUDADevice),
                    hasAttr(attr::CUDAShared), hasAttr(attr::HIPManaged)),
              unless(hasAnyName("threadIdx", "blockDim", "blockIdx", "gridDim",
                                "warpSize")));
  MF.addMatcher(DeclMatcher.bind("var"), this);
  MF.addMatcher(
      declRefExpr(anyOf(hasParent(implicitCastExpr(
                                      unless(hasParent(arraySubscriptExpr())))
                                      .bind("impl")),
                        anything()),
                  to(DeclMatcher.bind("decl")),
                  hasAncestor(functionDecl().bind("func")))
          .bind("used"),
      this);
}

void MemVarAnalysisRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto MemVar = getAssistNodeAsType<VarDecl>(Result, "var")) {
    if (isCubVar(MemVar)) {
      return;
    }
    std::string CanonicalType =
        MemVar->getType().getCanonicalType().getAsString();
    if (CanonicalType.find("block_tile_memory") != std::string::npos) {
      return;
    }
    auto FD = DpctGlobalInfo::getParentFunction(MemVar);
    if (FD && DpctGlobalInfo::useGroupLocalMemory() &&
        !DpctGlobalInfo::useFreeQueries() &&
        MemVar->hasAttr<CUDASharedAttr>()) {
      if (auto DFI = DeviceFunctionDecl::LinkRedecls(FD)) {
        DFI->setItem();
      }
    }
    auto Info = MemVarInfo::buildMemVarInfo(MemVar);
    if (Info && Info->isTypeDeclaredLocal() && !Info->isAnonymousType()) {
      if (Info->getDeclStmtOfVarType()) {
        Info->setLocalTypeName(Info->getType()->getBaseName());
      }
    }
    return;
  }
  auto MemVarRef = getNodeAsType<DeclRefExpr>(Result, "used");
  auto Func = getAssistNodeAsType<FunctionDecl>(Result, "func");
  auto Decl = getAssistNodeAsType<VarDecl>(Result, "decl");
  DpctGlobalInfo &Global = DpctGlobalInfo::getInstance();
  if (MemVarRef && Func && Decl) {
    if (isCubVar(Decl)) {
      return;
    }
    auto VD = dyn_cast<VarDecl>(MemVarRef->getDecl());
    if (Func->isImplicit() ||
        Func->getTemplateSpecializationKind() == TSK_ImplicitInstantiation)
      return;
    if (VD == nullptr)
      return;

    auto Var = Global.findMemVarInfo(VD);
    if (Func->hasAttr<CUDAGlobalAttr>() || Func->hasAttr<CUDADeviceAttr>()) {
      if (!(DpctGlobalInfo::useGroupLocalMemory() &&
            VD->hasAttr<CUDASharedAttr>() &&
            VD->getStorageClass() != SC_Extern)) {
        if (Var) {
          if (auto DFI = DeviceFunctionDecl::LinkRedecls(Func))
            DFI->addVar(Var);
        }
      }
    }
  }
}

void ErrorHandlingIfStmtRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      // Match if-statement that does not have else and has a condition of
      // either an operator!= or a variable of type enum.
      ifStmt(unless(hasElse(anything())),
             hasCondition(
                 anyOf(binaryOperator(hasOperatorName("!=")).bind("op!="),
                       ignoringImpCasts(
                           declRefExpr(hasType(hasCanonicalType(enumType())))
                               .bind("var")))))
          .bind("errIf"),
      this);
  MF.addMatcher(
      // Match if-statement that does not have else and has a condition of
      // operator==.
      ifStmt(unless(hasElse(anything())),
             hasCondition(binaryOperator(hasOperatorName("==")).bind("op==")))
          .bind("errIfSpecial"),
      this);
}

static bool isVarRef(const Expr *E) {
  if (auto D = dyn_cast<DeclRefExpr>(E))
    return isa<VarDecl>(D->getDecl());
  else
    return false;
}

static std::string getVarType(const Expr *E) {
  return E->getType().getCanonicalType().getUnqualifiedType().getAsString();
}

static bool isCudaFailureCheck(const BinaryOperator *Op, bool IsEq = false) {
  auto Lhs = Op->getLHS()->IgnoreImplicit();
  auto Rhs = Op->getRHS()->IgnoreImplicit();

  const Expr *Literal = nullptr;

  if (isVarRef(Lhs) && (getVarType(Lhs) == "enum cudaError" ||
                        getVarType(Lhs) == "enum cudaError_enum")) {
    Literal = Rhs;
  } else if (isVarRef(Rhs) && (getVarType(Rhs) == "enum cudaError" ||
                               getVarType(Rhs) == "enum cudaError_enum")) {
    Literal = Lhs;
  } else
    return false;

  if (auto IntLit = dyn_cast<IntegerLiteral>(Literal)) {
    if (IsEq ^ (IntLit->getValue() != 0))
      return false;
  } else if (auto D = dyn_cast<DeclRefExpr>(Literal)) {
    auto EnumDecl = dyn_cast<EnumConstantDecl>(D->getDecl());
    if (!EnumDecl)
      return false;
    if (IsEq ^ (EnumDecl->getInitVal() != 0))
      return false;
  } else {
    // The expression is neither an int literal nor an enum value.
    return false;
  }

  return true;
}

static bool isCudaFailureCheck(const DeclRefExpr *E) {
  return isVarRef(E) && (getVarType(E) == "enum cudaError" ||
                         getVarType(E) == "enum cudaError_enum");
}

void ErrorHandlingIfStmtRule::runRule(const MatchFinder::MatchResult &Result) {
  static std::vector<std::string> NameList = {"errIf", "errIfSpecial"};
  const IfStmt *If = getNodeAsType<IfStmt>(Result, "errIf");
  if (!If)
    if (!(If = getNodeAsType<IfStmt>(Result, "errIfSpecial")))
      return;
  auto EmitNotRemoved = [&](SourceLocation SL, const Stmt *R) {
    report(SL, Diagnostics::STMT_NOT_REMOVED, false);
  };
  auto isErrorHandlingSafeToRemove = [&](const Stmt *S) {
    if (const auto *CE = dyn_cast<CallExpr>(S)) {
      if (!CE->getDirectCallee()) {
        EmitNotRemoved(S->getSourceRange().getBegin(), S);
        return false;
      }
      auto Name = CE->getDirectCallee()->getNameAsString();
      static const llvm::StringSet<> SafeCallList = {
          "printf", "puts", "exit", "cudaDeviceReset", "fprintf"};
      if (SafeCallList.find(Name) == SafeCallList.end()) {
        EmitNotRemoved(S->getSourceRange().getBegin(), S);
        return false;
      }
      return true;
    }
    EmitNotRemoved(S->getSourceRange().getBegin(), S);
    return false;
  };

  auto isErrorHandling = [&](const Stmt *Block) {
    if (!isa<CompoundStmt>(Block))
      return isErrorHandlingSafeToRemove(Block);
    const CompoundStmt *CS = cast<CompoundStmt>(Block);
    for (const auto *S : CS->children()) {
      if (auto *E = dyn_cast_or_null<Expr>(S)) {
        if (!isErrorHandlingSafeToRemove(E->IgnoreImplicit())) {
          return false;
        }
      }
    }
    return true;
  };

  if (![&] {
        bool IsIfstmtSpecialCase = false;
        SourceLocation Ip;
        if (auto Op = getNodeAsType<BinaryOperator>(Result, "op!=")) {
          if (!isCudaFailureCheck(Op))
            return false;
        } else if (auto Op = getNodeAsType<BinaryOperator>(Result, "op==")) {
          if (!isCudaFailureCheck(Op, true))
            return false;
          IsIfstmtSpecialCase = true;
          Ip = Op->getBeginLoc();

        } else {
          auto CondVar = getNodeAsType<DeclRefExpr>(Result, "var");
          if (!isCudaFailureCheck(CondVar))
            return false;
        }
        // We know that it's error checking condition, check the body
        if (!isErrorHandling(If->getThen())) {
          if (IsIfstmtSpecialCase) {
            report(Ip, Diagnostics::IFSTMT_SPECIAL_CASE, false);
          } else {
            report(If->getSourceRange().getBegin(),
                   Diagnostics::IFSTMT_NOT_REMOVED, false);
          }
          return false;
        }
        return true;
      }()) {

    return;
  }

  emplaceTransformation(new ReplaceStmt(If, ""));

  // if the last token right after the ifstmt is ";"
  // remove the token
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto EndLoc = Lexer::getLocForEndOfToken(
      SM.getSpellingLoc(If->getEndLoc()), 0, SM, Result.Context->getLangOpts());
  Token Tok;
  Lexer::getRawToken(EndLoc, Tok, SM, Result.Context->getLangOpts(), true);
  if (Tok.getKind() == tok::semi) {
    emplaceTransformation(new ReplaceText(EndLoc, 1, ""));
  }
}

void ZeroLengthArrayRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(typeLoc(loc(constantArrayType())).bind("ConstantArrayType"),
                this);
}
void ZeroLengthArrayRule::runRule(const MatchFinder::MatchResult &Result) {
  auto TL = getNodeAsType<TypeLoc>(Result, "ConstantArrayType");
  if (!TL)
    return;
  const ConstantArrayType *CAT =
      dyn_cast_or_null<ConstantArrayType>(TL->getTypePtr());
  if (!CAT)
    return;

  // Check the array length
  if (!(CAT->getSize().isZero()))
    return;

  const clang::FieldDecl *MemberVariable =
      DpctGlobalInfo::findAncestor<clang::FieldDecl>(TL);
  if (MemberVariable) {
    report(TL->getBeginLoc(), Diagnostics::ZERO_LENGTH_ARRAY, false);
  } else {
    const clang::FunctionDecl *FD = DpctGlobalInfo::getParentFunction(TL);
    if (FD) {
      // Check if the array is in device code
      if (!(FD->getAttr<CUDADeviceAttr>()) && !(FD->getAttr<CUDAGlobalAttr>()))
        return;
    }
  }

  // Check if the array is a shared variable
  const VarDecl *VD = DpctGlobalInfo::findAncestor<VarDecl>(TL);
  if (VD && VD->getAttr<CUDASharedAttr>())
    return;

  report(TL->getBeginLoc(), Diagnostics::ZERO_LENGTH_ARRAY, false);
}

void GuessIndentWidthRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      functionDecl(allOf(hasParent(translationUnitDecl()),
                         hasBody(compoundStmt(unless(anyOf(
                             statementCountIs(0), statementCountIs(1)))))))
          .bind("FunctionDecl"),
      this);
  MF.addMatcher(
      cxxMethodDecl(hasParent(cxxRecordDecl(hasParent(translationUnitDecl()))))
          .bind("CXXMethodDecl"),
      this);
  MF.addMatcher(
      fieldDecl(hasParent(cxxRecordDecl(hasParent(translationUnitDecl()))))
          .bind("FieldDecl"),
      this);
}

void GuessIndentWidthRule::runRule(const MatchFinder::MatchResult &Result) {
  if (DpctGlobalInfo::getGuessIndentWidthMatcherFlag())
    return;
  SourceManager &SM = DpctGlobalInfo::getSourceManager();
  // Case 1:
  // TranslationUnitDecl
  // `-FunctionDecl
  //   `-CompoundStmt
  //     |-Stmt_1
  //     |-Stmt_2
  //     ...
  //     |-Stmt_n-1
  //     `-Stmt_n
  // The stmt in the compound stmt should >= 2, then we use the indent of the
  // first stmt as IndentWidth.
  auto FD = getNodeAsType<FunctionDecl>(Result, "FunctionDecl");
  if (FD) {
    CompoundStmt *CS = nullptr;
    Stmt *S = nullptr;
    if ((CS = dyn_cast<CompoundStmt>(FD->getBody())) &&
        (!CS->children().empty()) && (S = *(CS->children().begin()))) {
      DpctGlobalInfo::setIndentWidth(
          getIndent(SM.getExpansionLoc(S->getBeginLoc()), SM).size());
      DpctGlobalInfo::setGuessIndentWidthMatcherFlag(true);
      return;
    }
  }

  // Case 2:
  // TranslationUnitDecl
  // `-CXXRecordDecl
  //   |-CXXRecordDecl
  //   `-CXXMethodDecl
  // Use the indent of the CXXMethodDecl as the IndentWidth.
  auto CMD = getNodeAsType<CXXMethodDecl>(Result, "CXXMethodDecl");
  if (CMD) {
    DpctGlobalInfo::setIndentWidth(
        getIndent(SM.getExpansionLoc(CMD->getBeginLoc()), SM).size());
    DpctGlobalInfo::setGuessIndentWidthMatcherFlag(true);
    return;
  }

  // Case 3:
  // TranslationUnitDecl
  // `-CXXRecordDecl
  //   |-CXXRecordDecl
  //   `-FieldDecl
  // Use the indent of the FieldDecl as the IndentWidth.
  auto FieldD = getNodeAsType<FieldDecl>(Result, "FieldDecl");
  if (FieldD) {
    DpctGlobalInfo::setIndentWidth(
        getIndent(SM.getExpansionLoc(FieldD->getBeginLoc()), SM).size());
    DpctGlobalInfo::setGuessIndentWidthMatcherFlag(true);
    return;
  }
}

void NamespaceRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(usingDirectiveDecl().bind("usingDirective"), this);
  MF.addMatcher(namespaceAliasDecl().bind("namespaceAlias"), this);
  MF.addMatcher(usingDecl().bind("using"), this);
}

void NamespaceRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto UDD =
          getAssistNodeAsType<UsingDirectiveDecl>(Result, "usingDirective")) {
    std::string Namespace = UDD->getNominatedNamespace()->getNameAsString();
    if (Namespace == "cooperative_groups" || Namespace == "placeholders" ||
        Namespace == "nvcuda")
      emplaceTransformation(new ReplaceDecl(UDD, ""));
  } else if (auto NAD = getAssistNodeAsType<NamespaceAliasDecl>(
                 Result, "namespaceAlias")) {
    std::string Namespace = NAD->getNamespace()->getNameAsString();
    if (Namespace == "cooperative_groups" || Namespace == "placeholders")
      emplaceTransformation(new ReplaceDecl(NAD, ""));
  } else if (auto UD = getAssistNodeAsType<UsingDecl>(Result, "using")) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    SourceLocation Beg, End;
    unsigned int Len, Toklen;
    if (UD->getBeginLoc().isMacroID()) {
      // For scenario like "#define USING_1(FUNC) using std::FUNC; int a = 1;".
      // The macro include other statement or decl, we keep the origin code.
      if (auto CS = DpctGlobalInfo::findAncestor<CompoundStmt>(UD)) {
        const DeclStmt *DS =
            DpctGlobalInfo::getContext().getParents(*UD)[0].get<DeclStmt>();
        if (!DS) {
          return;
        }
        for (auto child : CS->children()) {
          if (child == DS) {
            continue;
          } else if (child->getBeginLoc().isMacroID() &&
                     SM.getExpansionLoc(child->getBeginLoc()) ==
                         SM.getExpansionLoc(UD->getBeginLoc())) {
            return;
          }
        }
      } else if (auto TS =
                     DpctGlobalInfo::findAncestor<TranslationUnitDecl>(UD)) {
        for (const auto &child : TS->decls()) {
          if (child == UD) {
            continue;
          } else if (auto USD = dyn_cast<UsingShadowDecl>(child)) {
            // To process implicit UsingShadowDecl node generated by UsingDecl
            // in global scope
            if (USD->getIntroducer() == UD) {
              continue;
            }
          } else if (child->getBeginLoc().isMacroID() &&
                     SM.getExpansionLoc(child->getBeginLoc()) ==
                         SM.getExpansionLoc(UD->getBeginLoc())) {
            return;
          }
        }
      } else {
        return;
      }
      auto Range = SM.getExpansionRange(UD->getBeginLoc());
      Beg = Range.getBegin();
      End = Range.getEnd();
    } else {
      Beg = UD->getBeginLoc();
      End = UD->getEndLoc();
    }
    Toklen = Lexer::MeasureTokenLength(
        End, SM, DpctGlobalInfo::getContext().getLangOpts());
    Len = SM.getFileOffset(End) - SM.getFileOffset(Beg) + Toklen;

    bool IsAllTargetsInCUDA = true;
    for (const auto &child : UD->getDeclContext()->decls()) {
      if (child == UD) {
        continue;
      } else if (const clang::UsingShadowDecl *USD =
                     dyn_cast<UsingShadowDecl>(child)) {
        if (USD->getIntroducer() == UD) {
          if (const auto *FD = dyn_cast<FunctionDecl>(USD->getTargetDecl())) {
            if (!isFromCUDA(FD)) {
              IsAllTargetsInCUDA = false;
              break;
            }
          } else if (const auto *FTD =
                         dyn_cast<FunctionTemplateDecl>(USD->getTargetDecl())) {
            if (!isFromCUDA(FTD)) {
              IsAllTargetsInCUDA = false;
              break;
            }
          } else {
            IsAllTargetsInCUDA = false;
            break;
          }
        }
      }
    }

    if (IsAllTargetsInCUDA) {
      auto NextTok = Lexer::findNextToken(
          End, SM, DpctGlobalInfo::getContext().getLangOpts());
      if (NextTok.has_value() && NextTok.value().is(tok::semi)) {
        Len = SM.getFileOffset(NextTok.value().getLocation()) -
              SM.getFileOffset(Beg) + 1;
      }
      emplaceTransformation(new ReplaceText(Beg, Len, ""));
    }
  }
}

void RemoveBaseClassRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(cxxRecordDecl(isDirectlyDerivedFrom(hasAnyName(
                                  "unary_function", "binary_function")))
                    .bind("derivedFrom"),
                this);
}

void RemoveBaseClassRule::runRule(const MatchFinder::MatchResult &Result) {
  auto SM = Result.SourceManager;
  auto LOpts = Result.Context->getLangOpts();
  auto findColon = [&](SourceRange SR) {
    Token Tok;
    auto E = SR.getEnd();
    SourceLocation Loc = SR.getBegin();
    Lexer::getRawToken(Loc, Tok, *SM, LOpts, true);
    bool ColonFound = false;
    while (Loc <= E) {
      if (Tok.is(tok::TokenKind::colon)) {
        ColonFound = true;
        break;
      }
      Tok = Lexer::findNextToken(Tok.getLocation(), *SM, LOpts).value();
      Loc = Tok.getLocation();
    }
    if (ColonFound)
      return Loc;
    else
      return SourceLocation();
  };

  auto getBaseDecl = [](QualType QT) {
    const Type *T = QT.getTypePtr();
    const NamedDecl *ND = nullptr;
    if (const auto *E = dyn_cast<ElaboratedType>(T)) {
      T = E->desugar().getTypePtr();
      if (const auto *TT = dyn_cast<TemplateSpecializationType>(T))
        ND = TT->getTemplateName().getAsTemplateDecl();
    } else
      ND = T->getAsCXXRecordDecl();
    return ND;
  };

  if (auto D = getNodeAsType<CXXRecordDecl>(Result, "derivedFrom")) {
    if (D->getNumBases() != 1)
      return;
    auto SR = SourceRange(D->getInnerLocStart(), D->getBraceRange().getBegin());
    auto ColonLoc = findColon(SR);
    if (ColonLoc.isValid()) {
      auto QT = D->bases().begin()->getType();
      const NamedDecl *BaseDecl = getBaseDecl(QT);
      if (BaseDecl) {
        auto BaseName = BaseDecl->getDeclName().getAsString();
        auto ThrustName = "thrust::" + BaseName;
        auto StdName = "std::" + BaseName;
        report(ColonLoc, Diagnostics::DEPRECATED_BASE_CLASS, false, ThrustName,
               StdName);
        auto Len = SM->getFileOffset(D->getBraceRange().getBegin()) -
                   SM->getFileOffset(ColonLoc);
        emplaceTransformation(new ReplaceText(ColonLoc, Len, ""));
      }
    }
  }
}

// The EDG frontend can allow code like below:
//
//     template <class T1, class T2> struct AAAAA {
//       template <class T3> void foo(T3 x);
//     };
//     template <typename T4, typename T5>
//     template <typename T6>
//     void AAAAA<T4, T5>::foo<T6>(T6 x) {}
//
// But clang/gcc emits error.
// We suppress the error in Sema and record the source range and remove
// the "invalid" code in this rule.
void CompatWithClangRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      cxxMethodDecl(hasParent(functionTemplateDecl())).bind("TemplateMethod"),
      this);
}

void CompatWithClangRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto CMD = getNodeAsType<CXXMethodDecl>(Result, "TemplateMethod")) {
    auto SR = CMD->getDuplicatedExplicitlySpecifiedTemplateArgumentsRange();
    if (SR.isValid()) {
      auto DefinitionSR = getDefinitionRange(SR.getBegin(), SR.getEnd());
      auto Begin = DefinitionSR.getBegin();
      auto End =
          DefinitionSR.getEnd().getLocWithOffset(Lexer::MeasureTokenLength(
              DefinitionSR.getEnd(), DpctGlobalInfo::getSourceManager(),
              DpctGlobalInfo::getContext().getLangOpts()));
      auto Length = End.getRawEncoding() - Begin.getRawEncoding();
      emplaceTransformation(new ReplaceText(Begin, Length, ""));
    }
  }
}

void IterationSpaceBuiltinRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      memberExpr(hasObjectExpression(opaqueValueExpr(hasSourceExpression(
                     declRefExpr(to(varDecl(hasAnyName("threadIdx", "blockDim",
                                                       "blockIdx", "gridDim"))
                                        .bind("varDecl")))
                         .bind("declRefExpr")))),
                 hasAncestor(functionDecl().bind("func")))
          .bind("memberExpr"),
      this);
  MF.addMatcher(declRefExpr(to(varDecl(hasAnyName("warpSize")).bind("varDecl")))
                    .bind("declRefExpr"),
                this);

  MF.addMatcher(declRefExpr(to(varDecl(hasAnyName("threadIdx", "blockDim",
                                                  "blockIdx", "gridDim"))),
                            hasAncestor(functionDecl().bind("funcDecl")))
                    .bind("declRefExprUnTempFunc"),
                this);
}

bool IterationSpaceBuiltinRule::renameBuiltinName(const DeclRefExpr *DRE,
                                                  std::string &NewName) {
  auto BuiltinName = DRE->getDecl()->getName();
  if (BuiltinName == "threadIdx")
    NewName = DpctGlobalInfo::getItem(DRE) + ".get_local_id(";
  else if (BuiltinName == "blockDim")
    NewName = DpctGlobalInfo::getItem(DRE) + ".get_local_range(";
  else if (BuiltinName == "blockIdx")
    NewName = DpctGlobalInfo::getItem(DRE) + ".get_group(";
  else if (BuiltinName == "gridDim")
    NewName = DpctGlobalInfo::getItem(DRE) + ".get_group_range(";
  else if (BuiltinName == "warpSize")
    NewName = DpctGlobalInfo::getSubGroup(DRE) + ".get_local_range().get(0)";
  else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected field name: " << BuiltinName;
    return false;
  }

  return true;
}
void IterationSpaceBuiltinRule::runRule(
    const MatchFinder::MatchResult &Result) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  if (const DeclRefExpr *DRE =
          getNodeAsType<DeclRefExpr>(Result, "declRefExprUnTempFunc")) {
    // Take the case of instantiated template function for example:
    // template <typename IndexType = int> __device__ void thread_id() {
    //  auto tidx_template = static_cast<IndexType>(threadIdx.x);
    //}
    // On Linux platform, .x(MemberExpr, __cuda_builtin_threadIdx_t) in
    // static_cast statement is not available in AST, while 'threadIdx' is
    // available, so dpct migrates it by 'threadIdx' matcher to identify the
    // SourceLocation of 'threadIdx', then look forward 2 tokens to check
    // whether .x appears.
    auto FD = getAssistNodeAsType<FunctionDecl>(Result, "funcDecl");
    if (!FD)
      return;
    const auto Begin = SM.getSpellingLoc(DRE->getBeginLoc());
    auto End = SM.getSpellingLoc(DRE->getEndLoc());
    End = End.getLocWithOffset(
        Lexer::MeasureTokenLength(End, *Result.SourceManager, LangOptions()));

    const auto Type = DRE->getDecl()
                          ->getType()
                          .getCanonicalType()
                          .getUnqualifiedType()
                          .getAsString();

    if (Type.find("__cuda_builtin") == std::string::npos)
      return;

    const auto Tok2Ptr = Lexer::findNextToken(End, SM, LangOptions());
    if (!Tok2Ptr.has_value())
      return;

    const auto Tok2 = Tok2Ptr.value();
    if (Tok2.getKind() == tok::raw_identifier) {
      std::string TypeStr = Tok2.getRawIdentifier().str();
      const char *StartPos = SM.getCharacterData(Begin);
      const char *EndPos = SM.getCharacterData(Tok2.getEndLoc());
      const auto TyLen = EndPos - StartPos;

      if (TyLen <= 0)
        return;

      std::string Replacement;
      if (!renameBuiltinName(DRE, Replacement))
        return;

      const auto FieldName = Tok2.getRawIdentifier().str();
      unsigned Dimension;
      auto DFI = DeviceFunctionDecl::LinkRedecls(FD);
      if (!DFI)
        return;

      if (FieldName == "x") {
        DpctGlobalInfo::getInstance().insertBuiltinVarInfo(Begin, TyLen,
                                                           Replacement, DFI);
        DpctGlobalInfo::updateSpellingLocDFIMaps(DRE->getBeginLoc(), DFI);
        return;
      } else if (FieldName == "y") {
        Dimension = 1;
        DFI->getVarMap().Dim = 3;
      } else if (FieldName == "z") {
        Dimension = 0;
        DFI->getVarMap().Dim = 3;
      } else
        return;

      Replacement += std::to_string(Dimension);
      Replacement += ")";

      emplaceTransformation(
          new ReplaceText(Begin, TyLen, std::move(Replacement)));
    }
    return;
  }

  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "memberExpr");
  const VarDecl *VD = getAssistNodeAsType<VarDecl>(Result, "varDecl");
  const DeclRefExpr *DRE = getNodeAsType<DeclRefExpr>(Result, "declRefExpr");
  std::shared_ptr<DeviceFunctionInfo> DFI = nullptr;
  if (!VD || !DRE) {
    return;
  }
  bool IsME = false;
  if (ME) {
    auto FD = getAssistNodeAsType<FunctionDecl>(Result, "func");
    if (!FD)
      return;
    DFI = DeviceFunctionDecl::LinkRedecls(FD);
    if (!DFI)
      return;
    IsME = true;
  } else {
    std::string InFile = dpct::DpctGlobalInfo::getSourceManager()
                             .getFilename(VD->getBeginLoc())
                             .str();

    if (!isChildOrSamePath(DpctInstallPath, InFile)) {
      return;
    }
  }

  std::string Replacement;
  StringRef BuiltinName = VD->getName();
  if (!renameBuiltinName(DRE, Replacement))
    return;

  if (IsME) {
    ValueDecl *Field = ME->getMemberDecl();
    StringRef FieldName = Field->getName();
    unsigned Dimension;
    if (FieldName == "__fetch_builtin_x") {
      auto Range = getDefinitionRange(ME->getBeginLoc(), ME->getEndLoc());
      SourceLocation Begin = Range.getBegin();
      SourceLocation End = Range.getEnd();

      End = End.getLocWithOffset(Lexer::MeasureTokenLength(
          End, SM, DpctGlobalInfo::getContext().getLangOpts()));

      unsigned int Len =
          SM.getDecomposedLoc(End).second - SM.getDecomposedLoc(Begin).second;
      DpctGlobalInfo::getInstance().insertBuiltinVarInfo(Begin, Len,
                                                         Replacement, DFI);
      DpctGlobalInfo::updateSpellingLocDFIMaps(ME->getBeginLoc(), DFI);
      return;
    } else if (FieldName == "__fetch_builtin_y") {
      Dimension = 1;
      DFI->getVarMap().Dim = 3;
    } else if (FieldName == "__fetch_builtin_z") {
      Dimension = 0;
      DFI->getVarMap().Dim = 3;
    } else {
      llvm::dbgs() << "[" << getName()
                   << "] Unexpected field name: " << FieldName;
      return;
    }

    Replacement += std::to_string(Dimension);
    Replacement += ")";
  }
  if (IsME) {
    emplaceTransformation(new ReplaceStmt(ME, std::move(Replacement)));
  } else {
    auto isDefaultParmWarpSize = [=](const FunctionDecl *&FD,
                                     const ParmVarDecl *&PVD) -> bool {
      if (BuiltinName != "warpSize")
        return false;
      PVD = DpctGlobalInfo::findAncestor<ParmVarDecl>(DRE);
      if (!PVD || !PVD->hasDefaultArg())
        return false;
      FD = dyn_cast_or_null<FunctionDecl>(PVD->getParentFunctionOrMethod());
      if (!FD)
        return false;
      if (FD->hasAttr<CUDADeviceAttr>())
        return true;
      return false;
    };

    const ParmVarDecl *PVD = nullptr;
    const FunctionDecl *FD = nullptr;
    if (isDefaultParmWarpSize(FD, PVD)) {
      SourceManager &SM = DpctGlobalInfo::getSourceManager();
      bool IsConstQualified = PVD->getType().isConstQualified();
      emplaceTransformation(new ReplaceStmt(DRE, "0"));
      unsigned int Idx = PVD->getFunctionScopeIndex();

      for (const auto FDIter : FD->redecls()) {
        if (IsConstQualified) {
          SourceRange SR;
          const ParmVarDecl *CurrentPVD = FDIter->getParamDecl(Idx);
          if (getTypeRange(CurrentPVD, SR)) {
            auto Length =
                SM.getFileOffset(SR.getEnd()) - SM.getFileOffset(SR.getBegin());
            QualType NewType = CurrentPVD->getType();
            NewType.removeLocalConst();
            std::string NewTypeStr =
                DpctGlobalInfo::getReplacedTypeName(NewType);
            emplaceTransformation(
                new ReplaceText(SR.getBegin(), Length, std::move(NewTypeStr)));
          }
        }

        const Stmt *Body = FD->getBody();
        if (!Body)
          continue;
        if (const CompoundStmt *BodyCS = dyn_cast<CompoundStmt>(Body)) {
          if (BodyCS->child_begin() != BodyCS->child_end()) {
            SourceLocation InsertLoc =
                SM.getExpansionLoc((*(BodyCS->child_begin()))->getBeginLoc());
            std::string IndentStr = getIndent(InsertLoc, SM).str();
            std::string Text =
                "if (!" + PVD->getName().str() + ") " + PVD->getName().str() +
                " = " + DpctGlobalInfo::getSubGroup(BodyCS) +
                ".get_local_range().get(0);" + getNL() + IndentStr;
            emplaceTransformation(new InsertText(InsertLoc, std::move(Text)));
          }
        }
      }
    } else
      emplaceTransformation(new ReplaceStmt(DRE, std::move(Replacement)));
  }
}

} // namespace dpct
} // namespace clang
