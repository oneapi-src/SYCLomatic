//===--------------- RulesLang.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RulesLang.h"
#include "AnalysisInfo.h"
#include "RuleInfra/ASTmatcherCommon.h"
#include "RuleInfra/CallExprRewriter.h"
#include "RuleInfra/CallExprRewriterCommon.h"
#include "RuleInfra/ExprAnalysis.h"
#include "RuleInfra/MemberExprRewriter.h"
#include "RuleInfra/MigrationStatistics.h"
#include "RulesLang/BarrierFenceSpaceAnalyzer.h"
#include "RulesLang/GroupFunctionAnalyzer.h"
#include "RulesLang/MapNamesLang.h"
#include "RulesMathLib/MapNamesRandom.h"
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
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::dpct;
using namespace clang::tooling;

extern clang::tooling::UnifiedPath DpctInstallPath; // Installation directory for this tool
extern DpctOption<opt, bool> ProcessAll;
extern DpctOption<opt, bool> AsyncHandler;

namespace clang{
namespace dpct{

static const CXXConstructorDecl *getIfConstructorDecl(const Decl *ND) {
  if (const auto *Tmpl = dyn_cast<FunctionTemplateDecl>(ND))
    ND = Tmpl->getTemplatedDecl();
  return dyn_cast<CXXConstructorDecl>(ND);
}

static internal::Matcher<NamedDecl> vectorTypeName() {
  std::vector<std::string> TypeNames(MapNamesLang::SupportedVectorTypes.begin(),
                                     MapNamesLang::SupportedVectorTypes.end());
  return internal::Matcher<NamedDecl>(
      new internal::HasNameMatcher(std::move(TypeNames)));
}

void ErrorHandlingHostAPIRule::registerMatcher(MatchFinder &MF) {
  auto isMigratedHostAPI = [&]() {
    return allOf(
        anyOf(returns(asString("cudaError_t")),
              returns(asString("cublasStatus_t")),
              returns(asString("nvgraphStatus_t")),
              returns(asString("cusparseStatus_t")),
              returns(asString("cusolverStatus_t")),
              returns(asString("cufftResult_t")),
              returns(asString("curandStatus_t")),
              returns(asString("ncclResult_t"))),
        // cudaGetLastError returns cudaError_t but won't fail in the call
        unless(hasName("cudaGetLastError")),
        anyOf(unless(hasAttr(attr::CUDADevice)), hasAttr(attr::CUDAHost)));
  };

  // Match host API call in the condition session of flow control
  MF.addMatcher(
      functionDecl(
          allOf(
              unless(hasDescendant(functionDecl())),
              unless(
                  anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal))),
              anyOf(
                  hasDescendant(ifStmt(hasCondition(expr(hasDescendant(
                      callExpr(callee(functionDecl(isMigratedHostAPI())))))))),
                  hasDescendant(doStmt(hasCondition(expr(hasDescendant(
                      callExpr(callee(functionDecl(isMigratedHostAPI())))))))),
                  hasDescendant(whileStmt(hasCondition(expr(hasDescendant(
                      callExpr(callee(functionDecl(isMigratedHostAPI())))))))),
                  hasDescendant(switchStmt(hasCondition(expr(hasDescendant(
                      callExpr(callee(functionDecl(isMigratedHostAPI())))))))),
                  hasDescendant(
                      forStmt(hasCondition(expr(hasDescendant(callExpr(
                          callee(functionDecl(isMigratedHostAPI())))))))))))
          .bind("inConditionHostAPI"),
      this);

  // Match host API call whose return value used inside flow control or return
  MF.addMatcher(
      functionDecl(
          allOf(unless(hasDescendant(functionDecl())),
                unless(anyOf(hasAttr(attr::CUDADevice),
                             hasAttr(attr::CUDAGlobal))),
                hasDescendant(callExpr(allOf(
                    callee(functionDecl(isMigratedHostAPI())),
                    anyOf(hasAncestor(binaryOperator(allOf(
                              hasLHS(declRefExpr()), isAssignmentOperator()))),
                          hasAncestor(varDecl())),
                    anyOf(hasAncestor(ifStmt()), hasAncestor(doStmt()),
                          hasAncestor(switchStmt()), hasAncestor(whileStmt()),
                          hasAncestor(callExpr()), hasAncestor(forStmt())))))))
          .bind("inLoopHostAPI"),
      this);

  MF.addMatcher(
      functionDecl(allOf(unless(hasDescendant(functionDecl())),
                         unless(anyOf(hasAttr(attr::CUDADevice),
                                      hasAttr(attr::CUDAGlobal))),
                         hasDescendant(callExpr(
                             allOf(callee(functionDecl(isMigratedHostAPI())),
                                   hasAncestor(returnStmt()))))))
          .bind("inReturnHostAPI"),
      this);

  // Match host API call whose return value captured and used
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(isMigratedHostAPI())),
                     anyOf(hasAncestor(binaryOperator(
                               allOf(hasLHS(declRefExpr().bind("targetLHS")),
                                     isAssignmentOperator()))),
                           hasAncestor(varDecl().bind("targetVarDecl"))),
                     unless(hasDescendant(functionDecl())),
                     hasAncestor(
                         functionDecl(unless(anyOf(hasAttr(attr::CUDADevice),
                                                   hasAttr(attr::CUDAGlobal))))
                             .bind("savedHostAPI"))))
          .bind("referencedHostAPI"),
      this);
}

void ErrorHandlingHostAPIRule::runRule(const MatchFinder::MatchResult &Result) {
  // if host API call in the condition session of flow control
  // or host API call whose return value used inside flow control or return
  // then add try catch.
  auto FD = getNodeAsType<FunctionDecl>(Result, "inConditionHostAPI");
  if (!FD) {
    FD = getNodeAsType<FunctionDecl>(Result, "inLoopHostAPI");
  }
  if (!FD) {
    FD = getNodeAsType<FunctionDecl>(Result, "inReturnHostAPI");
  }
  if (FD) {
    insertTryCatch(FD);
    return;
  }

  // Check if the return value is saved in a variable,
  // if yes, get the varDecl as the target varDecl TD.
  FD = getAssistNodeAsType<FunctionDecl>(Result, "savedHostAPI");
  if (!FD)
    return;
  auto TVD = getAssistNodeAsType<VarDecl>(Result, "targetVarDecl");
  auto TLHS = getAssistNodeAsType<DeclRefExpr>(Result, "targetLHS");
  const ValueDecl *TD = nullptr;
  if (TVD || TLHS) {
    TD = TVD ? TVD : TLHS->getDecl();
  }

  if (!TD)
    return;

  // Get the location of the API call to make sure the variable is referenced
  // AFTER the API call.
  auto CE = getAssistNodeAsType<CallExpr>(Result, "referencedHostAPI");

  if (!CE)
    return;

  // For each reference of TD, check if the location is after CE,
  // if yes, add try catch.
  std::vector<const DeclRefExpr *> Refs;
  getVarReferencedInFD(FD->getBody(), TD, Refs);
  SourceManager &SM = DpctGlobalInfo::getSourceManager();
  auto CallLoc = SM.getExpansionLoc(CE->getBeginLoc());
  for (auto It = Refs.begin(); It != Refs.end(); ++It) {
    auto RefLoc = SM.getExpansionLoc((*It)->getBeginLoc());
    if (SM.getCharacterData(RefLoc) - SM.getCharacterData(CallLoc) > 0) {
      insertTryCatch(FD);
      return;
    }
  }
}

void ErrorHandlingHostAPIRule::insertTryCatch(const FunctionDecl *FD) {
  SourceManager &SM = DpctGlobalInfo::getSourceManager();
  bool IsLambda = false;
  bool IsInMacro = false;
  if (auto CMD = dyn_cast<CXXMethodDecl>(FD)) {
    if (CMD->getParent()->isLambda()) {
      IsLambda = true;
    }
  }

  auto BodyRange = getDefinitionRange(FD->getBody()->getBeginLoc(),
                                      FD->getBody()->getEndLoc());
  auto It = dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().find(
      getCombinedStrFromLoc(BodyRange.getEnd()));
  if (It != dpct::DpctGlobalInfo::getExpansionRangeToMacroRecord().end()) {
    IsInMacro = true;
  }

  std::string IndentStr = getIndent(FD->getBeginLoc(), SM).str();
  std::string InnerIndentStr = IndentStr + "  ";

  std::string NewLine = getNL();
  if(IsInMacro)
    NewLine = "\\" + NewLine;

  if (IsLambda) {
    if (auto CSM = dyn_cast<CompoundStmt>(FD->getBody())) {
      // IndentStr = getIndent((*(CSM->body_begin()))->getBeginLoc(), SM).str();
      std::string TryStr = "try{ " + std::string(getNL()) + IndentStr;
      emplaceTransformation(
          new InsertBeforeStmt(*(CSM->body_begin()), std::move(TryStr)));
    }
  } else if (const CXXConstructorDecl *CDecl = getIfConstructorDecl(FD)) {
    emplaceTransformation(new InsertBeforeCtrInitList(CDecl, " try "));
  } else {
    emplaceTransformation(new InsertBeforeStmt(FD->getBody(), " try "));
  }

  std::string ReplaceStr =
      NewLine + IndentStr +
      std::string("catch (" + MapNames::getClNamespace(true) +
                  "exception const &exc) {") +
      NewLine + InnerIndentStr +
      std::string("std::cerr << exc.what() << \"Exception caught at file:\" << "
                  "__FILE__ << "
                  "\", line:\" << __LINE__ << std::endl;") +
      NewLine + InnerIndentStr + std::string("std::exit(1);") + NewLine +
      IndentStr + "}";
  if (IsLambda) {
    ReplaceStr += NewLine + IndentStr + "}";
  }
  emplaceTransformation(
      new InsertAfterStmt(FD->getBody(), std::move(ReplaceStr)));
}

void MiscAPIRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName("cudaOccupancyMaxActiveBlocksPerMultiprocessor",
                      "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                      "cudaOccupancyMaxPotentialBlockSize",
                      "cuGetExportTable");
  };

  MF.addMatcher(
      callExpr(callee(functionDecl(functionName()))).bind("FunctionCall"),
      this);
}
void MiscAPIRule::runRule(const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  ExprAnalysis EA(CE);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}

// Rule for types migration in var declarations and field declarations
void TypeInDeclRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      typeLoc(
          loc(qualType(hasDeclaration(namedDecl(hasAnyName(
              "dim3", "cudaError", "curandStatus", "cublasStatus", "CUstream",
              "CUstream_st", "thrust::complex", "thrust::device_vector",
              "thrust::device_ptr", "thrust::device_reference",
              "thrust::host_vector", "cublasHandle_t", "CUevent_st", "__half",
              "half", "__half2", "half2", "cudaMemoryAdvise", "cudaError_enum",
              "cudaDeviceProp", "cudaStreamCaptureStatus",
              "cudaGraphExecUpdateResult", "cudaPitchedPtr",
              "thrust::counting_iterator", "thrust::transform_iterator",
              "thrust::permutation_iterator", "thrust::iterator_difference",
              "cusolverDnHandle_t", "cusolverDnParams_t", "gesvdjInfo_t",
              "syevjInfo_t", "thrust::device_malloc_allocator",
              "thrust::divides", "thrust::tuple", "thrust::maximum",
              "thrust::multiplies", "thrust::plus", "cudaDataType_t",
              "cudaError_t", "CUresult", "CUdevice", "cudaEvent_t",
              "cublasStatus_t", "cuComplex", "cuFloatComplex",
              "cuDoubleComplex", "CUevent", "cublasFillMode_t",
              "cublasDiagType_t", "cublasSideMode_t", "cublasOperation_t",
              "cusolverStatus_t", "cusolverEigType_t", "cusolverEigMode_t",
              "curandStatus_t", "cudaStream_t", "cusparseStatus_t",
              "cusparseDiagType_t", "cusparseFillMode_t", "cusparseIndexBase_t",
              "cusparseMatrixType_t", "cusparseAlgMode_t",
              "cusparseOperation_t", "cusparseMatDescr_t", "cusparseHandle_t",
              "CUcontext", "cublasPointerMode_t", "cusparsePointerMode_t",
              "cublasGemmAlgo_t", "cusparseSolveAnalysisInfo_t", "cudaDataType",
              "cublasDataType_t", "curandState_t", "curandState",
              "curandStateXORWOW_t", "curandStateXORWOW",
              "curandStatePhilox4_32_10_t", "curandStatePhilox4_32_10",
              "curandStateMRG32k3a_t", "curandStateMRG32k3a", "thrust::minus",
              "thrust::negate", "thrust::logical_or", "thrust::equal_to",
              "thrust::less", "cudaSharedMemConfig", "curandGenerator_t",
              "curandRngType_t", "curandOrdering_t", "cufftHandle", "cufftReal",
              "cufftDoubleReal", "cufftComplex", "cufftDoubleComplex",
              "cufftResult_t", "cufftResult", "cufftType_t", "cufftType",
              "thrust::pair", "CUdeviceptr", "cudaDeviceAttr", "CUmodule",
              "CUjit_option", "CUfunction", "cudaMemcpyKind", "cudaComputeMode",
              "__nv_bfloat16", "cooperative_groups::__v1::thread_group",
              "cooperative_groups::__v1::thread_block", "libraryPropertyType_t",
              "libraryPropertyType", "cudaDataType_t", "cudaDataType",
              "cublasComputeType_t", "cublasAtomicsMode_t", "cublasMath_t",
              "CUmem_advise_enum", "CUmem_advise", "CUmemorytype",
              "CUmemorytype_enum", "thrust::tuple_element",
              "thrust::tuple_size", "thrust::zip_iterator",
              "cudaPointerAttributes", "CUpointer_attribute",
              "cusolverEigRange_t", "cudaUUID_t", "cusolverDnFunction_t",
              "cusolverAlgMode_t", "cusparseIndexType_t", "cusparseFormat_t",
              "cusparseDnMatDescr_t", "cusparseOrder_t", "cusparseDnVecDescr_t",
              "cusparseConstDnVecDescr_t", "cusparseSpMatDescr_t",
              "cusparseSpMMAlg_t", "cusparseSpMVAlg_t", "cusparseSpGEMMDescr_t",
              "cusparseSpSVDescr_t", "cusparseSpGEMMAlg_t", "CUuuid",
              "cusparseSpSVAlg_t", "cudaFuncAttributes",
              "cudaLaunchAttributeValue", "cusparseSpSMDescr_t",
              "cusparseConstSpMatDescr_t", "cusparseSpSMAlg_t",
              "cusparseConstDnMatDescr_t", "cudaMemcpy3DParms", "CUDA_MEMCPY3D",
              "cudaMemcpy3DPeerParms", "CUDA_MEMCPY3D_PEER", "CUDA_MEMCPY2D",
              "CUDA_ARRAY_DESCRIPTOR", "CUDA_ARRAY3D_DESCRIPTOR",
              "cublasLtHandle_t", "cublasLtMatmulDesc_t", "cublasLtOrder_t",
              "cublasLtPointerMode_t", "cublasLtMatrixLayout_t",
              "cublasLtMatrixLayoutAttribute_t", "CUjitInputType",
              "cublasLtMatmulDescAttributes_t", "cublasLtMatmulAlgo_t",
              "cublasLtEpilogue_t", "cublasLtMatmulPreference_t",
              "cublasLtMatmulHeuristicResult_t", "CUjit_target",
              "cublasLtMatrixTransformDesc_t", "cudaGraphicsMapFlags",
              "cudaGraphicsRegisterFlags", "cudaExternalMemoryHandleType",
              "CUstreamCallback"))))))
          .bind("cudaTypeDef"),
      this);

  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(hasAnyName(
                  "cooperative_groups::__v1::coalesced_group",
                  "cooperative_groups::__v1::grid_group",
                  "cooperative_groups::__v1::thread_block_tile", "cudaGraph_t",
                  "cudaGraphExec_t", "cudaGraphNode_t", "cudaGraphicsResource",
                  "cudaGraphicsResource_t", "CUgraphicsResource",
                  "cudaExternalMemory_t", "cudaExternalMemoryHandleDesc",
                  "cudaExternalMemoryMipmappedArrayDesc",
                  "cudaExternalMemoryBufferDesc"))))))
          .bind("cudaTypeDefEA"),
      this);
  MF.addMatcher(varDecl(hasType(classTemplateSpecializationDecl(
                            hasAnyTemplateArgument(refersToType(hasDeclaration(
                                namedDecl(hasName("use_default"))))))))
                    .bind("useDefaultVarDeclInTemplateArg"),
                this);
  MF.addMatcher(declRefExpr(to(varDecl(hasType(qualType(hasDeclaration(
                                namedDecl(hasAnyName("CUcontext"))))))))
                    .bind("driver_ctx"),
                this);
}

template <typename T>
bool getLocation(const Type *TypePtr, SourceLocation &SL) {
  auto TType = TypePtr->getAs<T>();
  if (TType) {
    auto TypeDecl = TType->getDecl();
    if (TypeDecl) {
      SL = TypeDecl->getLocation();
      return true;
    } else {
      return false;
    }
  }
  return false;
}

bool getTypeDeclLocation(const Type *TypePtr, SourceLocation &SL) {
  if (getLocation<EnumType>(TypePtr, SL)) {
    return true;
  } else if (getLocation<TypedefType>(TypePtr, SL)) {
    return true;
  } else if (getLocation<RecordType>(TypePtr, SL)) {
    return true;
  }
  return false;
}

bool getTemplateTypeReplacement(std::string TypeStr, std::string &Replacement,
                                unsigned &Len) {
  auto P1 = TypeStr.find('<');
  if (P1 != std::string::npos) {
    auto P2 = Replacement.find('<');
    if (P2 != std::string::npos) {
      Replacement = Replacement.substr(0, P2);
    }
    Len = P1;
    return true;
  }
  return false;
}

bool isAuto(const char *StrChar, unsigned Len) {
  return std::string(StrChar, Len) == "auto";
}

void insertComplexHeader(SourceLocation SL, std::string &Replacement) {
  if (SL.isValid() && Replacement.substr(0, 12) == "std::complex") {
    DpctGlobalInfo::getInstance().insertHeader(SL, HT_Complex);
  }
}

bool TypeInDeclRule::replaceTemplateSpecialization(
    SourceManager *SM, LangOptions &LOpts, SourceLocation BeginLoc,
    const TemplateSpecializationTypeLoc TSL) {

  for (unsigned i = 0; i < TSL.getNumArgs(); ++i) {
    auto ArgLoc = TSL.getArgLoc(i);
    if (ArgLoc.getArgument().getKind() != TemplateArgument::Type)
      continue;
    auto TSI = ArgLoc.getTypeSourceInfo();
    if (!TSI)
      continue;
    auto UTL = TSI->getTypeLoc().getUnqualifiedLoc();

    if (UTL.getTypeLocClass() == clang::TypeLoc::Elaborated) {
      auto ETC = UTL.getAs<ElaboratedTypeLoc>();

      auto ETBeginLoc = ETC.getQualifierLoc().getBeginLoc();
      auto ETEndLoc = ETC.getQualifierLoc().getEndLoc();

      if (ETBeginLoc.isInvalid() || ETEndLoc.isInvalid())
        continue;

      const char *Start = SM->getCharacterData(ETBeginLoc);
      const char *End = SM->getCharacterData(ETEndLoc);
      auto TyLen = End - Start;
      if (TyLen <= 0)
        return false;

      std::string RealTypeNameStr(Start, TyLen);

      auto Pos = RealTypeNameStr.find('<');
      if (Pos != std::string::npos) {
        RealTypeNameStr = RealTypeNameStr.substr(0, Pos);
        TyLen = Pos;
      }

      requestHelperFeatureForTypeNames(RealTypeNameStr);
      std::string Replacement =
          MapNames::findReplacedName(MapNames::TypeNamesMap, RealTypeNameStr);
      insertHeaderForTypeRule(RealTypeNameStr, ETBeginLoc);

      if (!Replacement.empty()) {
        SrcAPIStaticsMap[RealTypeNameStr]++;
        emplaceTransformation(
            new ReplaceText(ETBeginLoc, TyLen, std::move(Replacement)));
      }
    }
  }

  Token Tok;
  Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
  if (!Tok.isAnyIdentifier()) {
    return false;
  }

  auto TypeNameStr = Tok.getRawIdentifier().str();
  // skip to the next identifier after keyword "typename" or "const"
  if (TypeNameStr == "typename" || TypeNameStr == "const") {
    Tok = Lexer::findNextToken(BeginLoc, *SM, LOpts).value();
    BeginLoc = Tok.getLocation();
  }
  auto LAngleLoc = TSL.getLAngleLoc();

  const char *Start = SM->getCharacterData(BeginLoc);
  const char *End = SM->getCharacterData(LAngleLoc);
  auto TyLen = End - Start;
  if (TyLen <= 0)
    return false;

  std::string RealTypeNameStr(Start, TyLen);
  const auto StartPos = RealTypeNameStr.find_last_not_of(" ");
  // Remove spaces between type name and template arg, like
  // "thrust::device_ptr[Spaces]<int> tmp".
  if (StartPos != std::string::npos)
    RealTypeNameStr = RealTypeNameStr.substr(0, StartPos + 1);

  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None &&
      RealTypeNameStr.find("device_malloc_allocator") != std::string::npos) {
    report(BeginLoc, Diagnostics::KNOWN_UNSUPPORTED_TYPE, false,
            RealTypeNameStr);
    return true;
  }

  requestHelperFeatureForTypeNames(RealTypeNameStr);
  std::string Replacement =
      MapNames::findReplacedName(MapNames::TypeNamesMap, RealTypeNameStr);
  insertHeaderForTypeRule(RealTypeNameStr, BeginLoc);
  if (!Replacement.empty()) {
    insertComplexHeader(BeginLoc, Replacement);
    emplaceTransformation(
        new ReplaceText(BeginLoc, TyLen, std::move(Replacement)));
    return true;
  }
  return false;
}

// There's no AST matcher for dealing with DependentNameTypeLocs so
// it is handled 'manually'
bool TypeInDeclRule::replaceDependentNameTypeLoc(SourceManager *SM,
                                                 LangOptions &LOpts,
                                                 const TypeLoc *TL) {
  auto D = DpctGlobalInfo::findAncestor<Decl>(TL);
  TypeSourceInfo *TSI = nullptr;
  if (auto TD = dyn_cast<TypedefDecl>(D))
    TSI = TD->getTypeSourceInfo();
  else if (auto VD = dyn_cast<VarDecl>(D))
    TSI = VD->getTypeSourceInfo();
  else if (auto FD = dyn_cast<FieldDecl>(D))
    TSI = FD->getTypeSourceInfo();
  else if (auto TAD = dyn_cast<TypeAliasDecl>(D))
    TSI = TAD->getTypeSourceInfo();
  else
    return false;

  auto TTL = TSI->getTypeLoc();
  auto SR = TTL.getSourceRange();
  auto DTL = TTL.getAs<DependentNameTypeLoc>();
  if (!DTL)
    return false;

  auto NNSL = DTL.getQualifierLoc();
  auto NNTL = NNSL.getTypeLoc();

  auto BeginLoc = SR.getBegin();
  if (NNTL.getTypeLocClass() == clang::TypeLoc::TemplateSpecialization &&
      NNTL.getBeginLoc() == TL->getBeginLoc()) {
    auto TSL = NNTL.getUnqualifiedLoc().getAs<TemplateSpecializationTypeLoc>();
    if (replaceTemplateSpecialization(SM, LOpts, BeginLoc, TSL)) {
      // Check if "::type" needs replacement (only needed for
      // thrust::iterator_difference)
      Token Tok;
      Lexer::getRawToken(SR.getEnd(), Tok, *SM, LOpts, true);
      auto TypeNameStr =
          Tok.isAnyIdentifier() ? Tok.getRawIdentifier().str() : "";
      Lexer::getRawToken(TSL.getBeginLoc(), Tok, *SM, LOpts, true);
      auto TemplateNameStr =
          Tok.isAnyIdentifier() ? Tok.getRawIdentifier().str() : "";
      if (TypeNameStr == "type" && TemplateNameStr == "iterator_difference") {
        emplaceTransformation(
            new ReplaceText(SR.getEnd(), 4, std::string("difference_type")));
      }
      return true;
    }
  }
  return false;
}

// Make the necessary replacements for thrust::transform_iterator.
// The mapping requires swapping of the two template parameters, i.e.
//   thrust::transform_iterator<Functor, Iterator> ->
//     oenapi::dpl::transform_iterator<Iterator, Functor>
// This is a special transformation, because it requires the template
// parameters to be processed as part of the top level processing of
// the transform_iterator itself.  Simply processing the TypeLocs
// representing the template arguments when they are matched would
// result in wrong replacements being produced.
//
// For example:
//   thrust::transform_iterator<F, thrust::transform_iterator<F,I>>
// Should produce:
//   oneapi::dpl::transform_iterator<oneapi::dpl::transform_iterator<I,F>, F>
//
// The processing is therefore done by recursively walking all the
// TypeLocs that can be reached from the template arguments, and
// marking them as processed, so they won't be processed again, when
// their TypeLocs are matched by the matcher
bool TypeInDeclRule::replaceTransformIterator(SourceManager *SM,
                                              LangOptions &LOpts,
                                              const TypeLoc *TL) {

  // Local helper functions

  auto getFileLoc = [&](SourceLocation Loc) -> SourceLocation {
    // The EndLoc of some TypeLoc objects are Extension Locs, even
    // when the BeginLoc is a regular FileLoc.  This seems to happen
    // when the last typearg in a template specialization
    // is itself a template type.  For example.
    // SomeType<T1, AnotherType<T2>>.  The EndLoc for the TypeLoc for
    // AnotherType<T2> is an extension Loc.
    return SM->getFileLoc(Loc);
  };

  // Get the string from the source between [B, E].  The E location
  // is extended to the end of the token.  Special handling of the
  // '>' token is required in case it's followed by another '>'
  // For example: T<F, I<X>>
  // Without the special case condition, '>>' is considered one token
  auto getStr = [&](SourceLocation B, SourceLocation E) {
    B = getFileLoc(B);
    E = getFileLoc(E);
    if (*(SM->getCharacterData(E)) == '>')
      E = E.getLocWithOffset(1);
    else
      E = Lexer::getLocForEndOfToken(E, 0, *SM, LOpts);
    return std::string(SM->getCharacterData(B), SM->getCharacterData(E));
  };

  // Strip the 'typename' keyword when used in front of template types
  // This is necessary when looking up the typename string in the TypeNamesMap
  auto stripTypename = [](std::string &Str) {
    if (Str.substr(0, 8) == "typename") {
      Str = Str.substr(8);
      Str.erase(Str.begin(), std::find_if(Str.begin(), Str.end(), [](char ch) {
                  return !std::isspace(ch);
                }));
      return true;
    }
    return false;
  };

  // Get the Typename without potential template arguments.
  // For example:
  //   thrust::transform_iterator<F, I>
  //     -> thrust::transform_iterator
  auto getBaseTypeName = [&](const TypeLoc *TL) -> std::string {
    std::string Name = getStr(TL->getBeginLoc(), TL->getEndLoc());
    auto LAnglePos = Name.find("<");
    if (LAnglePos != std::string::npos)
      return Name.substr(0, LAnglePos);
    else
      return Name;
  };
  // Get the mapped typename, if one exists.  If not return the input
  auto mapName = [&](std::string Name) -> std::string {
    std::string NameToMap = Name;
    bool Stripped = stripTypename(NameToMap);
    std::string Replacement =
        MapNames::findReplacedName(MapNames::TypeNamesMap, NameToMap);
    insertHeaderForTypeRule(NameToMap, TL->getBeginLoc());
    requestHelperFeatureForTypeNames(NameToMap);
    if (Replacement.empty())
      return Name;
    else if (Stripped)
      return std::string("typename ") + Replacement;
    else
      return Replacement;
  };

  // Returns whether a TypeLoc has a template specialization, if
  // so the specialization is returned as well
  auto hasTemplateSpecialization =
      [&](const TypeLoc *TL, TemplateSpecializationTypeLoc &TSTL) -> bool {
    if (TL->getTypeLocClass() == clang::TypeLoc::TemplateSpecialization) {
      TSTL = TL->getAs<TemplateSpecializationTypeLoc>();
      return true;
    }
    if (TL->getTypeLocClass() == clang::TypeLoc::Elaborated) {
      auto ETL = TL->getAs<ElaboratedTypeLoc>();
      auto NTL = ETL.getNamedTypeLoc();
      if (NTL.getTypeLocClass() == clang::TypeLoc::TemplateSpecialization) {
        TSTL = NTL.getUnqualifiedLoc().getAs<TemplateSpecializationTypeLoc>();
        return true;
      } else
        return false;
    }
    return false;
  };

  // Returns whether a TypeLoc represents the thrust::transform_iterator
  // type with exactly 2 template arguments
  auto isTransformIterator = [&](const TypeLoc *TL) -> bool {
    TemplateSpecializationTypeLoc TSTL;
    if (hasTemplateSpecialization(TL, TSTL)) {
      std::string TypeStr = getStr(TSTL.getBeginLoc(), TSTL.getBeginLoc());
      if (TypeStr == "transform_iterator" && TSTL.getNumArgs() == 2) {
        return true;
      }
    }
    return false;
  };

  // Returns the full string replacement for a TypeLoc.  If necessary
  // template arguments are recursively walked to get potential replacement
  // for those as well.
  std::function<std::string(const TypeLoc *)> getNewTypeStr =
      [&](const TypeLoc *TL) -> std::string {
    std::string BaseTypeStr = getBaseTypeName(TL);
    std::string NewBaseTypeStr = mapName(BaseTypeStr);
    TemplateSpecializationTypeLoc TSTL;
    bool hasTSTL = hasTemplateSpecialization(TL, TSTL);
    // Mark this TL as having been processed
    ProcessedTypeLocs.insert(*TL);
    if (!hasTSTL) {
      // Not a template specialization, so recursion can terminate
      return NewBaseTypeStr;
    }
    // Mark the TSTL TypeLoc as having been processed
    ProcessedTypeLocs.insert(TSTL);
    if (isTransformIterator(TL) &&
        TSTL.getArgLoc(0).getArgument().getKind() == TemplateArgument::Type &&
        TSTL.getArgLoc(1).getArgument().getKind() == TemplateArgument::Type) {
      // Two template arguments must be swapped
      auto TSI1 = TSTL.getArgLoc(0).getTypeSourceInfo();
      auto TSI2 = TSTL.getArgLoc(1).getTypeSourceInfo();
      if (TSI1 && TSI2) {
        auto Arg1 = TSI1->getTypeLoc();
        auto Arg2 = TSI2->getTypeLoc();
        std::string Arg1Str = getNewTypeStr(&Arg1);
        std::string Arg2Str = getNewTypeStr(&Arg2);
        return NewBaseTypeStr + "<" + Arg2Str + ", " + Arg1Str + ">";
      }
    }
    // Recurse down through the template arguments
    std::string NewTypeStr = NewBaseTypeStr + "<";
    for (unsigned i = 0; i < TSTL.getNumArgs(); ++i) {
      std::string ArgStr;
      if (TSTL.getArgLoc(i).getArgument().getKind() == TemplateArgument::Type) {
        if (auto TSI = TSTL.getArgLoc(i).getTypeSourceInfo()) {
          auto ArgLoc = TSI->getTypeLoc();
          ArgStr = getNewTypeStr(&ArgLoc);
        }
      } else {
        ExprAnalysis EA;
        EA.analyze(TSTL.getArgLoc(i));
        ArgStr = EA.getReplacedString();
      }
      if (i != 0)
        NewTypeStr += ", ";
      NewTypeStr += ArgStr;
    }
    NewTypeStr += ">";
    return NewTypeStr;
  };

  // Main function:
  // Perform the complete replacement for the input TypeLoc.
  // TypeLocs that are being processed during the walk are inserted
  // into the ProcessedTypeLocs set, to prevent further processing
  // by the main matcher function
  if (!isTransformIterator(TL)) {
    return false;
  }
  std::string NewTypeStr = getNewTypeStr(TL);
  emplaceTransformation(new ReplaceToken(getFileLoc(TL->getBeginLoc()),
                                         getFileLoc(TL->getEndLoc()),
                                         std::move(NewTypeStr)));
  return true;
}

void TypeInDeclRule::processCudaStreamType(const DeclaratorDecl *DD) {
  auto SD = getAllDecls(DD);

  auto replaceInitParam = [&](const clang::Expr *replExpr) {
    if (!replExpr)
      return;

    if (auto type = DpctGlobalInfo::getUnqualifiedTypeName(replExpr->getType());
        !(type == "CUstream" || type == "cudaStream_t"))
      return;

    if (isDefaultStream(replExpr)) {
      int Index = getPlaceholderIdx(replExpr);
      if (Index == 0) {
        Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      }
      buildTempVariableMap(Index, replExpr, HelperFuncType::HFT_DefaultQueue);
      std::string Repl = "{{NEEDREPLACEZ" + std::to_string(Index) + "}}";
      emplaceTransformation(new ReplaceStmt(replExpr, Repl));
    }
  };

  for (auto It = SD.begin(); It != SD.end(); ++It) {
    const clang::Expr *replExpr = nullptr;
    if (const auto VD = dyn_cast<clang::VarDecl>(*It))
      replExpr = VD->getInit();
    else if (const auto FD = dyn_cast<clang::FieldDecl>(*It))
      replExpr = FD->getInClassInitializer();

    if (!replExpr)
      continue;

    if (const auto VarInitExpr = dyn_cast<InitListExpr>(replExpr)) {
      auto arrayReplEXpr = VarInitExpr->inits();
      for (auto replExpr : arrayReplEXpr) {
        replaceInitParam(replExpr);
      }
      return;
    }

    replaceInitParam(replExpr);
  }
}

void TypeInDeclRule::runRule(const MatchFinder::MatchResult &Result) {
  SourceManager *SM = Result.SourceManager;
  auto LOpts = Result.Context->getLangOpts();
  if (auto TL = getNodeAsType<TypeLoc>(Result, "cudaTypeDefEA")) {
    ExprAnalysis EA;
    EA.analyze(*TL);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
  if (auto DRE = getNodeAsType<DeclRefExpr>(Result, "driver_ctx")) {
    if (auto BO = DpctGlobalInfo::findAncestor<BinaryOperator>(DRE)) {
      if (BO->getOpcode() == BO_EQ) {
        const clang::Expr *AnotherSide = nullptr;
        if (BO->getRHS()->IgnoreImplicitAsWritten() == DRE) {
          AnotherSide = BO->getLHS();
        } else if (BO->getLHS()->IgnoreImplicitAsWritten() == DRE) {
          AnotherSide = BO->getRHS();
        }
        if (AnotherSide) {
          auto E = AnotherSide->IgnoreImplicitAsWritten();
          if (isa<CXXNullPtrLiteralExpr>(E) || isa<GNUNullExpr>(E)) {
            emplaceTransformation(new ReplaceStmt(E, "-1"));
          }
        }
      }
    }
    return;
  }
  if (const auto *TL = getNodeAsType<TypeLoc>(Result, "cudaTypeDef")) {
    if (const auto *TypePtr = TL->getTypePtr()) {
      if (isTypeInAnalysisScope(TypePtr)) {
        if (const auto *const ET = dyn_cast<ElaboratedType>(TypePtr))
          TypePtr = ET->getNamedType().getTypePtr();

        // The definition of the type is in current files for analysis and
        // neither they are typedefed. We donot want to migrate such types.
        if (TypePtr->getTypeClass() != clang::Type::Typedef)
          return;

        // When a CUDA type is redefined in the files under analysis we
        // want to migrate them.
        const auto *TT = dyn_cast<TypedefType>(TypePtr);
        if (!isRedeclInCUDAHeader(TT))
          return;
      }
      if (const auto *RecDeclRepr =
              TL->getType().getCanonicalType()->getAsRecordDecl()) {
        // Skip types whose names are matching with CUDA types and defined in
        // includes outside of in-root
        if (!DpctGlobalInfo::isInCudaPath(RecDeclRepr->getBeginLoc()))
          return;
      }
    }

    // if TL is the T in
    // template<typename T> void foo(T a);
    if (TL->getType()->getTypeClass() == clang::Type::SubstTemplateTypeParm ||
        TL->getBeginLoc().isInvalid()) {
      return;
    }

    if(isCapturedByLambda(TL))
      return;

    auto TypeStr =
        DpctGlobalInfo::getTypeName(TL->getType().getUnqualifiedType());

    if (auto FD = DpctGlobalInfo::getParentFunction(TL)) {
      if (FD->isImplicit())
        return;
    }

    if (ProcessedTypeLocs.find(*TL) != ProcessedTypeLocs.end())
      return;

    // Try to migrate cudaSuccess to sycl::info::event_command_status if it is
    // used in cases like "cudaSuccess == cudaEventQuery()".
    if (EventAPICallRule::getEventQueryTraversal().startFromTypeLoc(*TL))
      return;

    // when the following code is not in AnalysisScope
    // #define MACRO_SHOULD_NOT_BE_MIGRATED (MatchedType)3
    // Even if MACRO_SHOULD_NOT_BE_MIGRATED used in AnalysisScope, DPCT should not
    // migrate MatchedType.
    if (!DpctGlobalInfo::isInAnalysisScope(SM->getSpellingLoc(TL->getBeginLoc())) &&
        isPartOfMacroDef(SM->getSpellingLoc(TL->getBeginLoc()),
                         SM->getSpellingLoc(TL->getEndLoc()))) {
      return;
    }

    auto Range = getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc());
    auto BeginLoc = Range.getBegin();
    auto EndLoc = Range.getEnd();

    // WA for concatinated macro token
    if (SM->isWrittenInScratchSpace(SM->getSpellingLoc(TL->getBeginLoc()))) {
      BeginLoc = SM->getExpansionRange(TL->getBeginLoc()).getBegin();
      EndLoc = SM->getExpansionRange(TL->getBeginLoc()).getEnd();
    }

    std::string CanonicalTypeStr = DpctGlobalInfo::getUnqualifiedTypeName(
        TL->getType().getCanonicalType());

    if (CanonicalTypeStr == "cudaStreamCaptureStatus") {
      if (!DpctGlobalInfo::useExtGraph()) {
        report(TL->getBeginLoc(), Diagnostics::TRY_EXPERIMENTAL_FEATURE, false,
               "cudaStreamCaptureStatus", "--use-experimental-features=graph");
      }
    }

    if (CanonicalTypeStr == "cudaGraphExecUpdateResult") {
      report(TL->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             CanonicalTypeStr);
      return;
    }

    if (CanonicalTypeStr == "cudaGraphicsRegisterFlags" ||
        CanonicalTypeStr == "cudaGraphicsMapFlags") {
      if (!DpctGlobalInfo::useExtBindlessImages()) {
        report(TL->getBeginLoc(), Diagnostics::TRY_EXPERIMENTAL_FEATURE, false,
               CanonicalTypeStr,
               "--use-experimental-features=bindless_images");
      }
    }

    if (CanonicalTypeStr == "cooperative_groups::__v1::thread_group" ||
        CanonicalTypeStr == "cooperative_groups::__v1::thread_block") {
      if (auto ETL = TL->getUnqualifiedLoc().getAs<ElaboratedTypeLoc>()) {
        SourceLocation Begin = ETL.getBeginLoc();
        SourceLocation End = ETL.getEndLoc();
        if (Begin.isMacroID())
          Begin = SM->getSpellingLoc(Begin);
        if (End.isMacroID())
          End = SM->getSpellingLoc(End);
        End = End.getLocWithOffset(Lexer::MeasureTokenLength(
            End, *SM, DpctGlobalInfo::getContext().getLangOpts()));
        const auto *FD = DpctGlobalInfo::getParentFunction(TL);
        if (!FD)
          return;
        auto DFI = DeviceFunctionDecl::LinkRedecls(FD);
        if (!DFI)
          return;
        auto Index = DpctGlobalInfo::getCudaKernelDimDFIIndexThenInc();
        DpctGlobalInfo::insertCudaKernelDimDFIMap(Index, DFI);

        std::string GroupType = "";
        if (DpctGlobalInfo::useLogicalGroup())
          GroupType = MapNames::getDpctNamespace() +
                       "experimental::group_base" + "<{{NEEDREPLACEG" +
                       std::to_string(Index) + "}}>";
        if (CanonicalTypeStr == "cooperative_groups::__v1::thread_block") {
          if (ETL.getBeginLoc().isMacroID())
            GroupType = "auto";
          else
            GroupType = MapNames::getClNamespace() + "group" +
                         "<{{NEEDREPLACEG" + std::to_string(Index) + "}}>";
        }
        if (!GroupType.empty())
          emplaceTransformation(new ReplaceText(
              Begin, End.getRawEncoding() - Begin.getRawEncoding(),
              std::move(GroupType)));
        return;
      }
    }

    if (replaceDependentNameTypeLoc(SM, LOpts, TL)) {
      return;
    }

    if (replaceTransformIterator(SM, LOpts, TL)) {
      return;
    }

    if (TL->getTypeLocClass() == clang::TypeLoc::Elaborated) {
      auto ETC = TL->getAs<ElaboratedTypeLoc>();
      auto NTL = ETC.getNamedTypeLoc();

      if (const auto *RD = NTL.getType().getCanonicalType()->getAsRecordDecl())
        if (DpctGlobalInfo::isInCudaPath(RD->getBeginLoc()) &&
            (TypeStr == "cudaMemcpy3DParms" || TypeStr == "CUDA_MEMCPY3D" ||
             TypeStr == "CUDA_MEMCPY2D" || TypeStr == "cudaMemcpy3DPeerParms" ||
             TypeStr == "CUDA_MEMCPY3D_PEER"))
          if (const auto *VarD = DpctGlobalInfo::findAncestor<VarDecl>(TL))
            if (const auto *Init = VarD->getInit())
              if (const auto *VarInitExpr = dyn_cast<InitListExpr>(Init))
                emplaceTransformation(new ReplaceStmt(VarInitExpr, "{}"));

      if (NTL.getTypeLocClass() == clang::TypeLoc::TemplateSpecialization) {
        auto TSL =
            NTL.getUnqualifiedLoc().getAs<TemplateSpecializationTypeLoc>();

        if (replaceTemplateSpecialization(SM, LOpts, BeginLoc, TSL)) {
          return;
        }
      } else if (NTL.getTypeLocClass() == clang::TypeLoc::Record) {
        if (TypeStr.find("nv_bfloat16") != std::string::npos &&
            !DpctGlobalInfo::useBFloat16()) {
          return;
        }

        auto TSL = NTL.getUnqualifiedLoc().getAs<RecordTypeLoc>();

        const std::string TyName =
            dpct::DpctGlobalInfo::getTypeName(TSL.getType());
        std::string Replacement =
            MapNames::findReplacedName(MapNames::TypeNamesMap, TyName);
        requestHelperFeatureForTypeNames(TyName);
        insertHeaderForTypeRule(TyName, TL->getBeginLoc());

        if (!Replacement.empty()) {
          SrcAPIStaticsMap[TyName]++;
          emplaceTransformation(new ReplaceToken(BeginLoc, TSL.getEndLoc(),
                                                 std::move(Replacement)));
          return;
        }
      }
    } else if (TL->getTypeLocClass() == clang::TypeLoc::Qualified) {
      // To process the case like "typename
      // thrust::device_vector<int>::iterator itr;".
      auto ETL = TL->getUnqualifiedLoc().getAs<ElaboratedTypeLoc>();
      if (ETL) {
        auto NTL = ETL.getNamedTypeLoc();
        if (NTL.getTypeLocClass() == clang::TypeLoc::TemplateSpecialization) {
          auto TSL =
              NTL.getUnqualifiedLoc().getAs<TemplateSpecializationTypeLoc>();
          if (replaceTemplateSpecialization(SM, LOpts, BeginLoc, TSL)) {
            return;
          }
        }
      }
    } else if (TL->getTypeLocClass() ==
               clang::TypeLoc::TemplateSpecialization) {
      // To process cases like "tuple_element<0, TupleTy>" in
      // "typename thrust::tuple_element<0, TupleTy>::type"
      auto TSL = TL->getAs<TemplateSpecializationTypeLoc>();
      auto Parents = Result.Context->getParents(TSL);
      if (!Parents.empty()) {
        const auto *NNSL = Parents[0].get<NestedNameSpecifierLoc>();

        // To migrate "type" in case like "typename
        // thrust::iterator_difference<int
        // *>::type Var".
        if (NNSL && getNestedNameSpecifierString(*NNSL).find(
                        "thrust::iterator_difference") != std::string::npos) {
          auto Parents2 = Result.Context->getParents(*NNSL);
          if (!Parents2.empty()) {
            const auto *NNSL2 = Parents2[0].get<TypeLoc>();
            if (NNSL2) {
              Token Tok;
              Lexer::getRawToken(TSL.getBeginLoc(), Tok, *SM, LOpts, true);
              emplaceTransformation(new ReplaceText(
                  NNSL2->getEndLoc(), 4, std::string("difference_type")));
            }
          }
        }

        if (const auto *NNSL = Parents[0].get<NestedNameSpecifierLoc>()) {
          if (replaceTemplateSpecialization(SM, LOpts, NNSL->getBeginLoc(),
                                            TSL)) {
            return;
          }
        }
      }
    }

    std::string Str =
        MapNames::findReplacedName(MapNames::TypeNamesMap, TypeStr);
    insertHeaderForTypeRule(TypeStr, BeginLoc);
    requestHelperFeatureForTypeNames(TypeStr);
    if (Str.empty()) {
      auto Itr = MapNamesRandom::DeviceRandomGeneratorTypeMap.find(TypeStr);
      if (Itr != MapNamesRandom::DeviceRandomGeneratorTypeMap.end()) {
        if (TypeStr == "curandState_t" || TypeStr == "curandState" ||
            TypeStr == "curandStateXORWOW_t" ||
            TypeStr == "curandStateXORWOW") {
          report(BeginLoc, Diagnostics::DIFFERENT_GENERATOR, false);
        }
        Str = Itr->second;
      }
    } else if (llvm::StringRef(TypeStr).starts_with("cublas")) {
      // In most cases, we do not need to insert blas_utils.hpp manually since
      // the cublas_v2.h will be migrated. However, when the include directive
      // of cublas_v2.h is not in the in-root, the migrated code cannot be
      // built successfully.
      DpctGlobalInfo::getInstance().insertHeader(
          TL->getBeginLoc(), HeaderType::HT_DPCT_BLAS_Utils);
    }

    // Add '#include <complex>' directive to the file only once
    if (TypeStr == "cuComplex" || TypeStr == "cuDoubleComplex" ||
        TypeStr == "cuFloatComplex") {
      DpctGlobalInfo::getInstance().insertHeader(BeginLoc, HT_Complex);
    }
    // Add '#include <dpct/lib_common_utils.hpp>' directive to the file only
    // once
    if (TypeStr == "libraryPropertyType" ||
        TypeStr == "libraryPropertyType_t" || TypeStr == "cudaDataType_t" ||
        TypeStr == "cudaDataType" || TypeStr == "cublasComputeType_t") {
      DpctGlobalInfo::getInstance().insertHeader(BeginLoc,
                                                 HT_DPCT_COMMON_Utils);
    }

    const DeclaratorDecl *DD = nullptr;
    const VarDecl *VarD = DpctGlobalInfo::findAncestor<VarDecl>(TL);
    const FieldDecl *FieldD = DpctGlobalInfo::findAncestor<FieldDecl>(TL);
    const FunctionDecl *FD = DpctGlobalInfo::findAncestor<FunctionDecl>(TL);
    if (FD &&
        (FD->hasAttr<CUDADeviceAttr>() || FD->hasAttr<CUDAGlobalAttr>())) {
      if (DpctGlobalInfo::getUnqualifiedTypeName(TL->getType()) == "cublasHandle_t") {
        report(BeginLoc, Diagnostics::HANDLE_IN_DEVICE, false);
        return;
      }
    }
    if (VarD) {
      DD = VarD;
    } else if (FieldD) {
      DD = FieldD;
    } else if (FD) {
      DD = FD;
    }

    if (DD) {
      if (TL->getType().getCanonicalType()->isPointerType()) {
        const auto *PtrTy =
            TL->getType().getCanonicalType()->getAs<PointerType>();
        if (!PtrTy)
          return;
        if (PtrTy->getPointeeType()->isRecordType()) {
          const auto *RecordTy = PtrTy->getPointeeType()->getAs<RecordType>();
          if (!RecordTy)
            return;
          const auto *RD = RecordTy->getAsRecordDecl();
          if (!RD)
            return;
          if (RD->getName() == "CUstream_st" &&
              DpctGlobalInfo::isInCudaPath(RD->getBeginLoc()))
            processCudaStreamType(DD);
        }
      }
    }

    if (!Str.empty()) {
      SrcAPIStaticsMap[TypeStr]++;

      auto Len = Lexer::MeasureTokenLength(
          EndLoc, *SM, DpctGlobalInfo::getContext().getLangOpts());
      Len += SM->getDecomposedLoc(EndLoc).second -
             SM->getDecomposedLoc(BeginLoc).second;
      emplaceTransformation(new ReplaceText(BeginLoc, Len, std::move(Str)));
      return;
    }
  }

  if (auto VD =
          getNodeAsType<VarDecl>(Result, "useDefaultVarDeclInTemplateArg")) {
    auto TL = VD->getTypeSourceInfo()->getTypeLoc();

    auto TSTL = TL.getAsAdjusted<TemplateSpecializationTypeLoc>();
    if (!TSTL)
      return;
    auto TST = TSTL.getType()->getAsAdjusted<TemplateSpecializationType>();
    if (!TST || TST->template_arguments().empty())
      return;

    if (!DpctGlobalInfo::getTypeName(TST->template_arguments()[0].getAsType())
             .compare("thrust::use_default")) {
      auto ArgBeginLoc = TSTL.getArgLoc(0).getSourceRange().getBegin();
      auto ArgEndLoc = TSTL.getArgLoc(0).getSourceRange().getEnd();
      emplaceTransformation(new ReplaceToken(ArgBeginLoc, ArgEndLoc, ""));
    }
  }
}


// Rule for types replacements in var. declarations.
void VectorTypeNamespaceRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(
                            anyOf(namedDecl(vectorTypeName()),
                                  typedefDecl(vectorTypeName()))))))
                    .bind("vectorTypeTL"),
                this);

  MF.addMatcher(
      cxxRecordDecl(isDirectlyDerivedFrom(hasAnyName(SUPPORTEDVECTORTYPENAMES)))
          .bind("inheritanceType"),
      this);

  auto Vec3Types = [&]() {
    return hasAnyName("char3", "uchar3", "short3", "ushort3", "int3", "uint3",
                      "long3", "ulong3", "float3", "double3", "longlong3",
                      "ulonglong3");
  };

  MF.addMatcher(stmt(sizeOfExpr(hasArgumentOfType(hasCanonicalType(
                         hasDeclaration(namedDecl(Vec3Types()))))))
                    .bind("SizeofVector3Warn"),
                this);

  MF.addMatcher(
      declRefExpr(
          hasParent(implicitCastExpr(hasParent(cxxReinterpretCastExpr(hasType(
              pointsTo(namedDecl(Vec3Types()).bind("nameVec3Name"))))))))
          .bind("declRefExpr3Warn"),
      this);

  MF.addMatcher(cxxRecordDecl(isDirectlyDerivedFrom(hasAnyName(
                                  "char1", "uchar1", "short1", "ushort1",
                                  "int1", "uint1", "long1", "ulong1", "float1",
                                  "longlong1", "ulonglong1", "double1", "__half_raw")))
                    .bind("inherit"),
                this);
  // Matcher for __half_raw implicitly convert to half.
  MF.addMatcher(
      declRefExpr(allOf(unless(hasParent(memberExpr())),
                        unless(hasParent(unaryOperator(hasOperatorName("&")))),
                        to(varDecl(hasType(qualType(hasDeclaration(
                                       namedDecl(hasAnyName("__half_raw"))))))),
                        hasParent(implicitCastExpr())))
          .bind("halfRawExpr"),
      this);

  auto HasLongLongVecTypeArg = [&]() {
    return hasAnyTemplateArgument(refersToType(hasDeclaration(
        namedDecl(hasAnyName("longlong1", "longlong2", "longlong3", "longlong4",
                             "ulonglong1", "ulonglong2", "ulonglong3",
                             "ulonglong4"))
            .bind("vectorDecl"))));
  };
  MF.addMatcher(classTemplateSpecializationDecl(HasLongLongVecTypeArg())
                    .bind("vectorTypeInTemplateArg"),
                this);
  MF.addMatcher(
      functionDecl(HasLongLongVecTypeArg()).bind("vectorTypeInTemplateArg"),
      this);
}

void VectorTypeNamespaceRule::runRule(const MatchFinder::MatchResult &Result) {
  SourceManager *SM = Result.SourceManager;
  if (auto TL = getNodeAsType<TypeLoc>(Result, "vectorTypeTL")) {
    if (TL->getBeginLoc().isInvalid())
      return;

    // To skip user-defined type (found in in-root and from third party includes
    // outside of in-root)
    if (const auto *ND = getNamedDecl(TL->getTypePtr())) {
      auto Loc = ND->getBeginLoc();
      if (DpctGlobalInfo::isInAnalysisScope(Loc))
        return;
    }

    if (const auto *RecDeclRepr =
            TL->getType().getCanonicalType()->getAsRecordDecl()) {
      // Skip types whose names are matching with CUDA types and defined in
      // includes outside of in-root
      if (!DpctGlobalInfo::isInCudaPath(RecDeclRepr->getBeginLoc()))
        return;
    }

    auto BeginLoc =
        getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc()).getBegin();

    bool IsInScratchspace = false;
    // WA for concatinated macro token
    if (SM->isWrittenInScratchSpace(SM->getSpellingLoc(TL->getBeginLoc()))) {
      BeginLoc = SM->getExpansionLoc(TL->getBeginLoc());
      IsInScratchspace = true;
    }

    const FieldDecl *FD = DpctGlobalInfo::findAncestor<FieldDecl>(TL);
    if (auto D = dyn_cast_or_null<CXXRecordDecl>(getParentDecl(FD))) {
      // To process cases like "union struct_union {float2 data;};".
      auto Type = FD->getType();
      if (D && D->isUnion() && !Type->isPointerType() && !Type->isArrayType()) {
        // To add a default member initializer list "{}" to the
        // vector variant member of the union, because a union contains a
        // non-static data member with a non-trivial default constructor, the
        // default constructor of the union will be deleted by default.
        auto Loc = FD->getEndLoc().getLocWithOffset(Lexer::MeasureTokenLength(
            FD->getEndLoc(), *SM, Result.Context->getLangOpts()));
        emplaceTransformation(new InsertText(Loc, "{}"));
      }
    }
    bool NeedRemoveVolatile = true;
    Token Tok;
    auto LOpts = Result.Context->getLangOpts();
    Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
    if (Tok.isAnyIdentifier()) {
      const std::string TypeStr = Tok.getRawIdentifier().str();
      if (TypeStr.find("nv_bfloat16") != std::string::npos &&
          !DpctGlobalInfo::useBFloat16()) {
        return;
      }
      std::string Str =
          MapNames::findReplacedName(MapNames::TypeNamesMap, TypeStr);
      insertHeaderForTypeRule(TypeStr, BeginLoc);
      requestHelperFeatureForTypeNames(TypeStr);
      if (!Str.empty()) {
        SrcAPIStaticsMap[TypeStr]++;
        emplaceTransformation(new ReplaceToken(BeginLoc, std::move(Str)));
      }
      if (TypeStr.back() == '1') {
        NeedRemoveVolatile = false;
      }
    }

    if (IsInScratchspace) {
      std::string TypeStr = TL->getType().getUnqualifiedType().getAsString();
      auto Begin = SM->getImmediateExpansionRange(TL->getBeginLoc()).getBegin();
      auto End = SM->getImmediateExpansionRange(TL->getEndLoc()).getEnd();
      if (TypeStr.back() == '1') {
        // Make (Begin, End) be the range of "##1"
        Begin = SM->getSpellingLoc(Begin);
        End = SM->getSpellingLoc(End);
        Begin = Begin.getLocWithOffset(Lexer::MeasureTokenLength(
            Begin, *SM, DpctGlobalInfo::getContext().getLangOpts()));
        End = End.getLocWithOffset(Lexer::MeasureTokenLength(
            End, *SM, DpctGlobalInfo::getContext().getLangOpts()));
        auto Length = SM->getFileOffset(End) - SM->getFileOffset(Begin);
        return emplaceTransformation(new ReplaceText(Begin, Length, ""));
      } else {
        // Make Begin be the begin of "MACROARG##1"
        Begin = SM->getSpellingLoc(Begin);
        return emplaceTransformation(
            new InsertText(Begin, MapNames::getClNamespace()));
      }
    }

    // check whether the vector has volatile qualifier, if so, remove the
    // qualifier and emit a warning.
    if (!NeedRemoveVolatile)
      return;

    const ValueDecl *VD = DpctGlobalInfo::findAncestor<ValueDecl>(TL);
    if (!VD)
      return;

    bool isPointerToVolatile = false;
    if (const auto PT = dyn_cast<PointerType>(VD->getType())) {
      isPointerToVolatile = PT->getPointeeType().isVolatileQualified();
    }

    if (isPointerToVolatile || VD->getType().isVolatileQualified()) {
      SourceLocation Loc = SM->getExpansionLoc(VD->getBeginLoc());
      report(Loc, Diagnostics::VOLATILE_VECTOR_ACCESS, false);

      // remove the volatile qualifier and trailing spaces
      Token Tok;
      // Get the range of variable declaration
      SourceRange SpellingRange =
          getDefinitionRange(VD->getBeginLoc(), VD->getEndLoc());
      Lexer::getRawToken(SpellingRange.getBegin(), Tok, *SM,
                         DpctGlobalInfo::getContext().getLangOpts(), true);
      unsigned int EndLocOffset =
          SM->getDecomposedExpansionLoc(SpellingRange.getEnd()).second;
      // Look for volatile in the above found range of declaration
      while (SM->getDecomposedExpansionLoc(SM->getSpellingLoc(Tok.getEndLoc()))
                     .second <= EndLocOffset &&
             !Tok.is(tok::TokenKind::eof)) {
        SourceLocation TokBegLoc = SM->getSpellingLoc(Tok.getLocation());
        SourceLocation TokEndLoc = SM->getSpellingLoc(Tok.getEndLoc());

        if (Tok.is(tok::TokenKind::raw_identifier) &&
            Tok.getRawIdentifier().str() == "volatile") {
          emplaceTransformation(
              new ReplaceText(TokBegLoc,
                              getLenIncludingTrailingSpaces(
                                  SourceRange(TokBegLoc, TokEndLoc), *SM),
                              ""));
          break;
        }
        Lexer::getRawToken(Tok.getEndLoc(), Tok, *SM,
                           DpctGlobalInfo::getContext().getLangOpts(), true);
      }
    }
  }

  if (auto RD = getNodeAsType<CXXRecordDecl>(Result, "inherit")) {
    report(RD->getBeginLoc(), Diagnostics::VECTYPE_INHERITATED, false);
  }

  if (const auto *UETT =
          getNodeAsType<UnaryExprOrTypeTraitExpr>(Result, "SizeofVector3Warn")) {

    // Ignore shared variables.
    // .e.g: __shared__ int a[sizeof(float3)], b[sizeof(float3)], ...;
    if (const auto *V = DpctGlobalInfo::findAncestor<VarDecl>(UETT)) {
      if (V->hasAttr<CUDASharedAttr>())
        return;
    }
    std::string argTypeName = DpctGlobalInfo::getTypeName(UETT->getTypeOfArgument());
    std::string argCanTypeName = DpctGlobalInfo::getTypeName(UETT->getTypeOfArgument().getCanonicalType());
    if (argTypeName != argCanTypeName)
      argTypeName += " (aka " + argCanTypeName + ")";

    report(
        UETT, Diagnostics::SIZEOF_WARNING, true, argTypeName,
        "Check that the allocated memory size in the migrated code is correct");
  }
  // Runrule for __half_raw implicitly convert to half.
  if (auto DRE = getNodeAsType<DeclRefExpr>(Result, "halfRawExpr")) {
    if (const auto *RT =
            DRE->getType().getCanonicalType()->getAs<RecordType>()) {
      if (isUserDefinedDecl(RT->getDecl()))
        return;
    }
    ExprAnalysis EA;
    std::string Replacement;
    llvm::raw_string_ostream OS(Replacement);
    OS << MapNames::getClNamespace() + "bit_cast<" +
              MapNames::getClNamespace() + "half>(";
    EA.analyze(DRE);
    OS << EA.getReplacedString();
    OS << ")";
    OS.flush();
    emplaceTransformation(new ReplaceStmt(DRE, Replacement));
    return;
  }

  if (const auto *DRE =
          getNodeAsType<DeclRefExpr>(Result, "declRefExpr3Warn")) {

    if (const auto *NameDecl =
            getNodeAsType<NamedDecl>(Result, "nameVec3Name")) {
      report(DRE, Diagnostics::SIZEOF_WARNING, true,
             NameDecl->getNameAsString(), "You may need to adjust the code");
    }
  }

  if (const auto *D = getNodeAsType<Decl>(Result, "vectorTypeInTemplateArg")) {
    if (const auto *VD = getAssistNodeAsType<NamedDecl>(Result, "vectorDecl")) {
      auto TypeStr = VD->getNameAsString();
      report(D->getBeginLoc(), Diagnostics::VEC_IN_TEMPLATE_ARG, false, TypeStr,
             MapNames::findReplacedName(MapNames::TypeNamesMap, TypeStr));
    }
  }
}


void VectorTypeMemberAccessRule::registerMatcher(MatchFinder &MF) {
  auto memberAccess = [&]() {
    return hasObjectExpression(
        hasType(qualType(anyOf(hasCanonicalType(recordType(hasDeclaration(
                                   cxxRecordDecl(vectorTypeName())))),
                               hasDeclaration(namedDecl(vectorTypeName()))))));
  };

  // int2.x => int2.x()
  MF.addMatcher(
      memberExpr(allOf(memberAccess(), unless(hasParent(binaryOperator(allOf(
                                           hasLHS(memberExpr(memberAccess())),
                                           isAssignmentOperator()))))))
          .bind("VecMemberExpr"),
      this);

  // class A : int2{ void foo(){x = 3;}}
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(
                               pointsTo(cxxRecordDecl(vectorTypeName())))))
                    .bind("DerivedVecMemberExpr"),
                this);

  // int2.x += xxx => int2.x() += xxx
  MF.addMatcher(
      binaryOperator(allOf(hasLHS(memberExpr(memberAccess())
                                      .bind("VecMemberExprAssignmentLHS")),
                           isAssignmentOperator()))
          .bind("VecMemberExprAssignment"),
      this);

  // int2 *a; a->x = 1;
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(pointerType(pointee(qualType(
              anyOf(recordType(hasDeclaration(cxxRecordDecl(vectorTypeName()))),
                    hasDeclaration(namedDecl(vectorTypeName())))))))))
          .bind("VecMemberExprArrow"),
      this);

  // No inner filter is available for decltypeType(). Thus, this matcher will
  // match all decltypeType. Detail control flow for different types is in
  // runRule().
  MF.addMatcher(typeLoc(loc(decltypeType())).bind("TypeLoc"), this);
}

void VectorTypeMemberAccessRule::renameMemberField(const MemberExpr *ME) {
  ExprAnalysis EA(ME);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}

void VectorTypeMemberAccessRule::runRule(
    const MatchFinder::MatchResult &Result) {
  if (const MemberExpr *ME =
          getNodeAsType<MemberExpr>(Result, "VecMemberExpr")) {
    auto Parents = Result.Context->getParents(*ME);
    if (Parents.size() == 0) {
      return;
    }
    renameMemberField(ME);
  }

  if (auto ME = getNodeAsType<MemberExpr>(Result, "DerivedVecMemberExpr")) {
    renameMemberField(ME);
  }

  if (auto ME =
          getNodeAsType<MemberExpr>(Result, "VecMemberExprAssignmentLHS")) {
    renameMemberField(ME);
  }

  if (auto ME = getNodeAsType<MemberExpr>(Result, "VecMemberExprArrow")) {
    renameMemberField(ME);
  }

  if (auto *TL = getNodeAsType<DecltypeTypeLoc>(Result, "TypeLoc")) {
    ExprAnalysis EA;
    EA.analyze(*TL);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

} //namespace clang
} //namespace dpct

namespace clang {
namespace ast_matchers {

AST_MATCHER(FunctionDecl, overloadedVectorOperator) {
  if (!DpctGlobalInfo::isInAnalysisScope(Node.getBeginLoc()))
    return false;
  if (Node.isTemplateInstantiation())
    return false; // Template operator function need not add namespace.

  switch (Node.getOverloadedOperator()) {
  default: {
    return false;
  }
#define OVERLOADED_OPERATOR_MULTI(...)
#define OVERLOADED_OPERATOR(Name, ...)                                         \
  case OO_##Name: {                                                            \
    break;                                                                     \
  }
#include "clang/Basic/OperatorKinds.def"
#undef OVERLOADED_OPERATOR
#undef OVERLOADED_OPERATOR_MULTI
  }

  // Check parameter is vector type
  auto SupportedParamType = [&](const ParmVarDecl *PD) {
    if (!PD)
      return false;
    const IdentifierInfo *IDInfo =
        PD->getOriginalType().getBaseTypeIdentifier();
    if (!IDInfo)
      return false;

    const std::string TypeName = IDInfo->getName().str();
    if (MapNamesLang::SupportedVectorTypes.find(TypeName) !=
        MapNamesLang::SupportedVectorTypes.end()) {
      if (const auto *ND = getNamedDecl(PD->getType().getTypePtr())) {
        auto Loc = ND->getBeginLoc();
        if (DpctGlobalInfo::isInAnalysisScope(Loc))
          return false;
      }
      return true;
    }
    return false;
  };

  // As long as one parameter is vector type
  for (unsigned i = 0, End = Node.getNumParams(); i != End; ++i) {
    if (SupportedParamType(Node.getParamDecl(i))) {
      return true;
    }
  }

  return false;
}

} // namespace ast_matchers
} // namespace clang

namespace clang {
namespace dpct {

void VectorTypeOperatorRule::registerMatcher(MatchFinder &MF) {
  auto vectorTypeOverLoadedOperator = [&]() {
    return functionDecl(overloadedVectorOperator(),
                        unless(hasAncestor(cxxRecordDecl())));
  };

  // Matches user overloaded operator declaration
  MF.addMatcher(vectorTypeOverLoadedOperator().bind("overloadedOperatorDecl"),
                this);

  // Matches call of user overloaded operator
  MF.addMatcher(cxxOperatorCallExpr(callee(vectorTypeOverLoadedOperator()),
                                    hasAncestor(vectorTypeOverLoadedOperator()))
                    .bind("callOverloadedOperatorInOverloadedOperator"),
                this);

  MF.addMatcher(
      cxxOperatorCallExpr(callee(vectorTypeOverLoadedOperator()),
                          unless(hasAncestor(vectorTypeOverLoadedOperator())))
          .bind("callOverloadedOperatorNotInOverloadedOperator"),
      this);
}

const char VectorTypeOperatorRule::NamespaceName[] =
    "dpct_operator_overloading";

void VectorTypeOperatorRule::MigrateOverloadedOperatorDecl(
    const MatchFinder::MatchResult &Result, const FunctionDecl *FD) {
  if (!FD)
    return;

  // Helper function to get the scope of function declaration
  // Eg.:
  //
  //    void test();
  //   ^            ^
  //   |            |
  // Begin         End
  //
  //    void test() {}
  //   ^              ^
  //   |              |
  // Begin           End
  auto GetFunctionSourceRange = [&](const SourceManager &SM,
                                    const SourceLocation &StartLoc,
                                    SourceLocation EndLoc) {
    const std::pair<FileID, unsigned> StartLocInfo =
        SM.getDecomposedExpansionLoc(StartLoc);
    // Set the EndLoc to the end of token. If the first token after the EndLoc
    // is ';', then set after the ';'.
    EndLoc = Lexer::getLocForEndOfToken(EndLoc, 0, SM,
                                        Result.Context->getLangOpts());
    const auto Tok = Lexer::findNextToken(EndLoc, SM, LangOptions()).value();
    if (Tok.is(tok::TokenKind::semi)) {
      EndLoc = Tok.getEndLoc();
    }
    const std::pair<FileID, unsigned> EndLocInfo =
        SM.getDecomposedExpansionLoc(EndLoc);
    assert(StartLocInfo.first == EndLocInfo.first);

    return SourceRange(
        SM.getComposedLoc(StartLocInfo.first, StartLocInfo.second),
        SM.getComposedLoc(EndLocInfo.first, EndLocInfo.second));
  };

  // Add namespace to user overloaded operator declaration
  // double2& operator+=(double2& lhs, const double2& rhs)
  // =>
  // namespace dpct_operator_overloading {
  //
  // double2& operator+=(double2& lhs, const double2& rhs)
  //
  // }
  const auto &SM = *Result.SourceManager;
  const std::string NL = getNL(FD->getBeginLoc(), SM);

  std::ostringstream Prologue;
  // clang-format off
  Prologue << "namespace " << NamespaceName << " {" << NL
           << NL;
  // clang-format on

  std::ostringstream Epilogue;
  // clang-format off
  Epilogue << NL
           << "}  // namespace " << NamespaceName << NL
           << NL;
  // clang-format on
  SourceRange SR;
  auto P = getParentDecl(FD);
  // Deal with functions as well as function templates
  if (auto FTD = dyn_cast<FunctionTemplateDecl>(P)) {
    SR = GetFunctionSourceRange(SM, FTD->getBeginLoc(), FTD->getEndLoc());
  } else {
    SR = GetFunctionSourceRange(SM, FD->getBeginLoc(), FD->getEndLoc());
  }
  report(SR.getBegin(), Diagnostics::TRNA_WARNING_OVERLOADED_API_FOUND, false);
  emplaceTransformation(new InsertText(SR.getBegin(), Prologue.str()));
  emplaceTransformation(new InsertText(SR.getEnd(), Epilogue.str()));
}

void VectorTypeOperatorRule::MigrateOverloadedOperatorCall(
    const MatchFinder::MatchResult &Result, const CXXOperatorCallExpr *CE,
    bool InOverloadedOperator) {
  if (!CE)
    return;
  if (!InOverloadedOperator &&
      (DpctGlobalInfo::findAncestor<FunctionTemplateDecl>(CE) ||
       DpctGlobalInfo::findAncestor<ClassTemplateDecl>(CE))) {
    return;
  }
  // Explicitly call user overloaded operator
  //
  // For non-assignment operator:
  // a == b
  // =>
  // dpct_operator_overloading::operator==(a, b)
  //
  // For assignment operator:
  // a += b
  // =>
  // dpct_operator_overloading::operator+=(a, b)
  if (!clang::getOperatorSpelling(CE->getOperator()))
    return;
  const std::string OperatorName =
      std::string(clang::getOperatorSpelling(CE->getOperator()));
  std::ostringstream FuncCall;

  FuncCall << NamespaceName << "::operator" << OperatorName;

  std::string OperatorReplacement = (CE->getNumArgs() == 1)
                                        ? /* Unary operator */ ""
                                        : /* Binary operator */ ",";
  emplaceTransformation(
      new ReplaceToken(CE->getOperatorLoc(), std::move(OperatorReplacement)));
  insertAroundStmt(CE, FuncCall.str() + "(", ")");
}

void VectorTypeOperatorRule::runRule(const MatchFinder::MatchResult &Result) {
  // Add namespace to user overloaded operator declaration
  MigrateOverloadedOperatorDecl(
      Result, getNodeAsType<FunctionDecl>(Result, "overloadedOperatorDecl"));

  // Explicitly call user overloaded operator
  MigrateOverloadedOperatorCall(
      Result,
      getNodeAsType<CXXOperatorCallExpr>(
          Result, "callOverloadedOperatorInOverloadedOperator"),
      true);

  MigrateOverloadedOperatorCall(
      Result,
      getNodeAsType<CXXOperatorCallExpr>(
          Result, "callOverloadedOperatorNotInOverloadedOperator"),
      false);
}


void DeviceInfoVarRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(anyOf(
              hasType(qualType(hasCanonicalType(recordType(
                  hasDeclaration(cxxRecordDecl(hasAnyName(
                    "cudaDeviceProp", "cudaPointerAttributes"))))))),
              hasType(
                  pointsTo(qualType(hasCanonicalType(recordType(hasDeclaration(
                    cxxRecordDecl(hasAnyName(
                      "cudaDeviceProp", "cudaPointerAttributes")))))))))))
          .bind("FieldVar"),
      this);
}

void DeviceInfoVarRule::runRule(const MatchFinder::MatchResult &Result) {
  const MemberExpr *ME = getNodeAsType<MemberExpr>(Result, "FieldVar");
  if (!ME)
    return;
  auto MemberName = ME->getMemberNameInfo().getAsString();

  auto BaseType = ME->getBase()->getType();
  if (BaseType->isPointerType()) {
    BaseType = BaseType->getPointeeType();
  }
  std::string MemberExprName =
                      DpctGlobalInfo::getTypeName(BaseType.getCanonicalType())
                        + "." + MemberName;
  if (MemberExprRewriterFactoryBase::MemberExprRewriterMap->find(MemberExprName)
        != MemberExprRewriterFactoryBase::MemberExprRewriterMap->end()) {
      ExprAnalysis EA;
      EA.analyze(ME);
      emplaceTransformation(EA.getReplacement());
      EA.applyAllSubExprRepl();
      return;
  }

  // not functionally compatible properties
  if (MemberName == "deviceOverlap" || MemberName == "concurrentKernels") {
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, "true");
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), "true"));
    return;
  } else if (MemberName == "canMapHostMemory" ||
             MemberName == "kernelExecTimeoutEnabled") {
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, "false");
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), "false"));
    return;
  } else if (MemberName == "pciDomainID" || MemberName == "pciBusID") {
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, "-1");
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), "-1"));
    return;
  } else if (MemberName == "memPitch") {
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, "INT_MAX");
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), "INT_MAX"));
    return;
  } else if (MemberName == "textureAlignment") {
    requestFeature(HelperFeatureEnum::device_ext);
    std::string Repl =
        MapNames::getDpctNamespace() + "get_current_device().get_info<" +
        MapNames::getClNamespace() + "info::device::mem_base_addr_align>()";
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, Repl);
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), std::move(Repl)));
    return;
  } else if (MemberName == "l2CacheSize") {
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, "global_mem_cache_size");
  } else if (MemberName == "ECCEnabled") {
    requestFeature(HelperFeatureEnum::device_ext);
    std::string Repl = MapNames::getDpctNamespace() +
                       "get_current_device().get_info<" +
                       MapNames::getClNamespace() +
                       "info::device::error_correction_support>()";
    emplaceTransformation(
        new ReplaceToken(ME->getBeginLoc(), ME->getEndLoc(), std::move(Repl)));
    return;
  } else if (MemberName == "regsPerBlock") {
    report(ME->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
           MemberName, "get_max_register_size_per_work_group");
  }

  if (MemberName == "sharedMemPerBlock" ||
      MemberName == "sharedMemPerMultiprocessor" ||
      MemberName == "sharedMemPerBlockOptin") {
    report(ME->getBeginLoc(), Diagnostics::LOCAL_MEM_SIZE, false, MemberName);
  } else if (MemberName == "maxGridSize") {
    report(ME->getBeginLoc(), Diagnostics::MAX_GRID_SIZE, false);
  }

  if (!DpctGlobalInfo::useDeviceInfo() &&
      (MemberName == "pciDeviceID" || MemberName == "uuid")) {
    report(ME->getBeginLoc(), Diagnostics::UNMIGRATED_DEVICE_PROP, false,
           MemberName);
    return;
  }

  auto Search = PropNamesMap.find(MemberName);
  if (Search == PropNamesMap.end()) {
    return;
  }
  
  // migrate to get_XXX() eg. "b=a.minor" to "b=a.get_minor_version()"
  auto Parents = Result.Context->getParents(*ME);
  if (Parents.size() < 1)
    return;
  if ((Search->second.compare(0, 13, "major_version") == 0) ||
      (Search->second.compare(0, 13, "minor_version") == 0)) {
    report(ME->getBeginLoc(), Comments::VERSION_COMMENT, false);
  }
  if (Search->second.compare(0, 10, "integrated") == 0) {
    report(ME->getBeginLoc(), Comments::NOT_SUPPORT_API_INTEGRATEDORNOT, false);
  }
  std::string TmplArg = "";
  if (MemberName == "maxGridSize" ||
      MemberName == "maxThreadsDim") {
    // Similar code in ExprAnalysis.cpp
    TmplArg = "<int *>";
  }
  if (auto *BO = Parents[0].get<clang::BinaryOperator>()) {
    // migrate to set_XXX() eg. "a.minor = 1" to "a.set_minor_version(1)"
    if (BO->getOpcode() == clang::BO_Assign) {
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, "set_" + Search->second));
      emplaceTransformation(new ReplaceText(BO->getOperatorLoc(), 1, "("));
      emplaceTransformation(new InsertAfterStmt(BO, ")"));
      return;
    }
  } else if (auto *OCE = Parents[0].get<clang::CXXOperatorCallExpr>()) {
    // migrate to set_XXX() for types with an overloaded = operator
    if (OCE->getOperator() == clang::OverloadedOperatorKind::OO_Equal) {
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, "set_" + Search->second));
      emplaceTransformation(new ReplaceText(OCE->getOperatorLoc(), 1, "("));
      emplaceTransformation(new InsertAfterStmt(OCE, ")"));
      return;
    }
  }
  emplaceTransformation(new RenameFieldInMemberExpr(
      ME, "get_" + Search->second + TmplArg + "()"));
  return ;
}


// Rule for Enums constants.
void EnumConstantRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(
          to(enumConstantDecl(anyOf(
              hasType(enumDecl(hasAnyName(
                  "cudaComputeMode", "cudaMemcpyKind", "cudaMemoryAdvise",
                  "cudaStreamCaptureStatus", "cudaDeviceAttr",
                  "libraryPropertyType_t", "cudaDataType_t",
                  "CUmem_advise_enum", "cufftType_t",
                  "cufftType", "cudaMemoryType", "CUctx_flags_enum",
                  "CUpointer_attribute_enum", "CUmemorytype_enum",
                  "cudaGraphicsMapFlags", "cudaGraphicsRegisterFlags"))),
              matchesName("CUDNN_.*"), matchesName("CUSOLVER_.*")))))
          .bind("EnumConstant"),
      this);
}

void EnumConstantRule::handleComputeMode(std::string EnumName,
                                         const DeclRefExpr *E) {
  report(E->getBeginLoc(), Diagnostics::COMPUTE_MODE, false);
  auto P = getParentStmt(E);
  if (auto ICE = dyn_cast<ImplicitCastExpr>(P)) {
    P = getParentStmt(ICE);
    if (auto BO = dyn_cast<BinaryOperator>(P)) {
      auto LHS = BO->getLHS()->IgnoreImpCasts();
      auto RHS = BO->getRHS()->IgnoreImpCasts();
      const MemberExpr *ME = nullptr;
      if (auto MEL = dyn_cast<MemberExpr>(LHS))
        ME = MEL;
      else if (auto MER = dyn_cast<MemberExpr>(RHS))
        ME = MER;
      if (ME) {
        auto MD = ME->getMemberDecl();
        auto BaseTy = DpctGlobalInfo::getUnqualifiedTypeName(
            ME->getBase()->getType().getCanonicalType(),
            DpctGlobalInfo::getContext());
        if (MD->getNameAsString() == "computeMode" &&
            BaseTy == "cudaDeviceProp") {
          if (EnumName == "cudaComputeModeDefault") {
            if (BO->getOpcodeStr() == "==")
              emplaceTransformation(new ReplaceStmt(P, "true"));
            else if (BO->getOpcodeStr() == "!=")
              emplaceTransformation(new ReplaceStmt(P, "false"));
          } else {
            if (BO->getOpcodeStr() == "==")
              emplaceTransformation(new ReplaceStmt(P, "false"));
            else if (BO->getOpcodeStr() == "!=")
              emplaceTransformation(new ReplaceStmt(P, "true"));
          }
          return;
        }
      }
    }
  }
  // default => 1
  // others  => 0
  if (EnumName == "cudaComputeModeDefault") {
    emplaceTransformation(new ReplaceStmt(E, "1"));
    return;
  } else if (EnumName == "cudaComputeModeExclusive" ||
             EnumName == "cudaComputeModeProhibited" ||
             EnumName == "cudaComputeModeExclusiveProcess") {
    emplaceTransformation(new ReplaceStmt(E, "0"));
    return;
  }
}

void EnumConstantRule::runRule(const MatchFinder::MatchResult &Result) {
  const DeclRefExpr *E = getNodeAsType<DeclRefExpr>(Result, "EnumConstant");
  if (!E)
    return;
  std::string EnumName = E->getNameInfo().getName().getAsString();
  if (EnumName == "cudaStreamCaptureStatusInvalidated" ||
      EnumName == "cudaExternalMemoryHandleTypeOpaqueWin32Kmt" ||
      EnumName == "cudaExternalMemoryHandleTypeD3D12Heap" ||
      EnumName == "cudaExternalMemoryHandleTypeD3D11Resource" ||
      EnumName == "cudaExternalMemoryHandleTypeD3D11ResourceKmt" ||
      EnumName == "cudaExternalMemoryHandleTypeNvSciBuf") {
    report(E->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, EnumName);
    return;
  } else if (EnumName == "cudaComputeModeDefault" ||
             EnumName == "cudaComputeModeExclusive" ||
             EnumName == "cudaComputeModeProhibited" ||
             EnumName == "cudaComputeModeExclusiveProcess") {
    handleComputeMode(EnumName, E);
    return;
  } else if ((EnumName == "cudaStreamCaptureStatusActive" ||
              EnumName == "cudaStreamCaptureStatusNone") &&
             !DpctGlobalInfo::useExtGraph()) {
    report(E->getBeginLoc(), Diagnostics::TRY_EXPERIMENTAL_FEATURE, false,
           EnumName, "--use-experimental-features=graph");
    return;
  } else if (!DpctGlobalInfo::useExtBindlessImages() &&
             (EnumName == "cudaGraphicsRegisterFlagsNone" ||
              EnumName == "cudaGraphicsRegisterFlagsReadOnly" ||
              EnumName == "cudaGraphicsRegisterFlagsWriteDiscard" ||
              EnumName == "cudaGraphicsRegisterFlagsSurfaceLoadStore" ||
              EnumName == "cudaGraphicsRegisterFlagsTextureGather" ||
              EnumName == "cudaGraphicsMapFlagsNone" ||
              EnumName == "cudaGraphicsMapFlagsReadOnly" ||
              EnumName == "cudaGraphicsMapFlagsWriteDiscard" ||
              EnumName == "cudaExternalMemoryHandleTypeOpaqueFd" ||
              EnumName == "cudaExternalMemoryHandleTypeOpaqueWin32" ||
              EnumName == "cudaExternalMemoryHandleTypeD3D12Resource")) {
    report(E->getBeginLoc(), Diagnostics::TRY_EXPERIMENTAL_FEATURE, false,
           EnumName, "--use-experimental-features=bindless_images");
    return;
  } else if (auto ET = dyn_cast<EnumType>(E->getType())) {
    if (auto ETD = ET->getDecl()) {
      auto EnumTypeName = ETD->getName().str();
      if (EnumTypeName == "cudaMemoryAdvise" ||
          EnumTypeName == "CUmem_advise_enum") {
        report(E->getBeginLoc(), Diagnostics::DEFAULT_MEM_ADVICE, false,
               " and was set to 0");
      } else if (EnumTypeName == "cudaDeviceAttr") {
        auto &Context = DpctGlobalInfo::getContext();
        auto Parent = Context.getParents(*E)[0];
        if (auto PCE = Parent.get<CallExpr>()) {
          if (auto DC = PCE->getDirectCallee()) {
            if (DC->getNameAsString() == "cudaDeviceGetAttribute")
              return;
          }
        }
        if (auto EC = dyn_cast<EnumConstantDecl>(E->getDecl())) {
          std::string Repl = toString(EC->getInitVal(), 10);
          emplaceTransformation(new ReplaceStmt(E, Repl));
          return;
        }
      }
    }
  }

  auto Search = MapNames::EnumNamesMap.find(EnumName);
  if (Search == MapNames::EnumNamesMap.end()) {
    return;
  }
  if (auto ET = dyn_cast<EnumType>(E->getType())) {
    if (auto ETD = ET->getDecl()) {
      if (ETD->getName().str() == "libraryPropertyType_t" ||
          ETD->getName().str() == "cudaDataType_t" ||
          ETD->getName().str() == "cublasComputeType_t") {
        DpctGlobalInfo::getInstance().insertHeader(
            DpctGlobalInfo::getSourceManager().getExpansionLoc(
                E->getBeginLoc()),
            HT_DPCT_COMMON_Utils);
      }
    }
  }
  emplaceTransformation(new ReplaceStmt(E, Search->second->NewName));
  requestHelperFeatureForEnumNames(EnumName);
}


void ErrorConstantsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(
                              hasDeclContext(enumDecl(anyOf(
                                hasName("cudaError"), hasName("cufftResult_t"),
                                hasName("cudaError_enum"),
                                hasName("cudaSharedMemConfig")))))))
                    .bind("ErrorConstants"),
                this);
}

void ErrorConstantsRule::runRule(const MatchFinder::MatchResult &Result) {
  const DeclRefExpr *DE = getNodeAsType<DeclRefExpr>(Result, "ErrorConstants");
  if (!DE)
    return;
  if (EventAPICallRule::getEventQueryTraversal().startFromEnumRef(DE))
    return;
  auto *EC = cast<EnumConstantDecl>(DE->getDecl());
  std::string Repl = toString(EC->getInitVal(), 10);

  if (EC->getDeclName().getAsString() == "cudaErrorNotReady") {
    emplaceTransformation(new ReplaceStmt(DE, "1"));
    return;
  }

  emplaceTransformation(new ReplaceStmt(DE, Repl));
}

// Rule for CU_JIT enums.
void CU_JITEnumsRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(declRefExpr(to(enumConstantDecl(
                                anyOf(matchesName("(CU_JIT_*)"),
                                      matchesName("(CU_TARGET_COMPUTE_*)")))))
                    .bind("CU_JITConstants"),
                this);
}

void CU_JITEnumsRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *DE =
          getNodeAsType<DeclRefExpr>(Result, "CU_JITConstants")) {
    emplaceTransformation(new ReplaceStmt(DE, "0"));

    report(DE->getBeginLoc(),
           Diagnostics::HOSTALLOCMACRO_NO_MEANING,
           true, DE->getDecl()->getNameAsString());
  }
}


/// The function returns the migrated arguments of the scalar parameters.
/// In the original code, the type of this parameter is pointer.
/// (1) If original type is float/double and argument is like "&alpha",
///     this function will return "alpha".
/// (2) If original type is float2/double2 and argument is like "&alpha",
///     this function will return
///     "std::complex<float/double>(alpha.x(), alpha.x())".
/// (3) If original argument is like "alpha", this function will return
///     "dpct::get_value(alpha, q)".
/// \p Expr is used to distinguish case(1,2) and case(3)
/// \p ExprStr and \p QueueStr are used for case(3)
/// \p ValueType is used for case(2)
std::string getValueStr(const Expr *Expr, std::string ExprStr,
                        std::string QueueStr, std::string ValueType = "") {
  if (auto UO = dyn_cast_or_null<UnaryOperator>(Expr->IgnoreImpCasts())) {
    if (UO->getOpcode() == UO_AddrOf && UO->getSubExpr()) {
      ExprAnalysis EA;
      std::string NewStr = EA.ref(UO->getSubExpr());
      if (ValueType == "std::complex<float>" ||
          ValueType == "std::complex<double>")
        return ValueType + "(" + NewStr + ".x(), " + NewStr + ".y())";
      else
        return NewStr;
    }
  } else if (auto COCE =
                 dyn_cast<CXXOperatorCallExpr>(Expr->IgnoreImpCasts())) {
    if (COCE->getOperator() == OO_Amp && COCE->getArg(0)) {
      ExprAnalysis EA;
      std::string NewStr = EA.ref(COCE->getArg(0));
      if (ValueType == "std::complex<float>" ||
          ValueType == "std::complex<double>")
        return ValueType + "(" + NewStr + ".x(), " + NewStr + ".y())";
      else
        return NewStr;
    }
  }
  requestFeature(HelperFeatureEnum::device_ext);
  return MapNames::getLibraryHelperNamespace() + "get_value(" + ExprStr + ", " +
         QueueStr + ")";
}



void FunctionCallRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName(
        "cudaGetDeviceCount", "cudaGetDeviceProperties",
        "cudaGetDeviceProperties_v2", "cudaDeviceReset", "cudaSetDevice",
        "cudaDeviceGetAttribute", "cudaDeviceGetP2PAttribute",
        "cudaDeviceGetPCIBusId", "cudaGetDevice", "cudaDeviceSetLimit",
        "cudaGetLastError", "cudaPeekAtLastError", "cudaDeviceSynchronize",
        "cudaThreadSynchronize", "cudnnGetErrorString", "cudaGetErrorString",
        "cudaGetErrorName", "cudaDeviceSetCacheConfig",
        "cudaDeviceGetCacheConfig", "clock",
        "cudaOccupancyMaxPotentialBlockSize", "cudaThreadSetLimit",
        "cudaFuncSetCacheConfig", "cudaThreadExit", "cudaDeviceGetLimit",
        "cudaDeviceSetSharedMemConfig", "cudaIpcCloseMemHandle",
        "cudaIpcGetEventHandle", "cudaIpcGetMemHandle",
        "cudaIpcOpenEventHandle", "cudaIpcOpenMemHandle", "cudaSetDeviceFlags",
        "cudaDeviceCanAccessPeer", "cudaDeviceDisablePeerAccess",
        "cudaDeviceEnablePeerAccess", "cudaDriverGetVersion",
        "cuDeviceCanAccessPeer", "cudaFuncSetAttribute",
        "cudaRuntimeGetVersion", "clock64", "__nanosleep",
        "cudaFuncSetSharedMemConfig", "cuFuncSetCacheConfig",
        "cudaPointerGetAttributes", "cuPointerGetAttributes",
        "cuCtxSetCacheConfig", "cuCtxSetLimit", "cudaCtxResetPersistingL2Cache",
        "cuCtxResetPersistingL2Cache", "cudaStreamSetAttribute",
        "cudaStreamGetAttribute", "cudaProfilerStart", "cudaProfilerStop",
        "__trap", "cuCtxEnablePeerAccess");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(functionName())), parentStmt()))
          .bind("FunctionCall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(functionName())),
                               unless(parentStmt())))
                    .bind("FunctionCallUsed"),
                this);
}

std::string FunctionCallRule::findValueofAttrVar(const Expr *AttrArg,
                                                 const CallExpr *CE) {
  std::string AttributeName;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &CT = DpctGlobalInfo::getContext();
  int MinDistance = INT_MAX;
  int RecognizedMinDistance = INT_MAX;
  if (!AttrArg || !CE)
    return "";
  auto DRE = dyn_cast<DeclRefExpr>(AttrArg->IgnoreImpCasts());
  if (!DRE)
    return "";
  auto Decl = dyn_cast<VarDecl>(DRE->getDecl());
  if (!Decl || CT.getParents(*Decl)[0].get<TranslationUnitDecl>())
    return "";
  int DRELocOffset = SM.getFileOffset(SM.getExpansionLoc(DRE->getBeginLoc()));

  if (Decl->hasInit()) {
    // get the attribute name from definition
    if (auto Init = dyn_cast<DeclRefExpr>(Decl->getInit())) {
      SourceLocation InitLoc = SM.getExpansionLoc(Init->getLocation());
      MinDistance = DRELocOffset - SM.getFileOffset(InitLoc);
      RecognizedMinDistance = MinDistance;
      AttributeName = Init->getNameInfo().getName().getAsString();
    }
  }
  std::string AttrVarName = DRE->getNameInfo().getName().getAsString();
  auto AttrVarScope = findImmediateBlock(Decl);
  if (!AttrVarScope)
    return "";

  // we need to track the reference of attr var in its scope
  auto AttrVarMatcher =
      findAll(declRefExpr(to(varDecl(hasName(AttrVarName)))).bind("AttrVar"));
  auto MatchResult = ast_matchers::match(AttrVarMatcher, *AttrVarScope,
                                         DpctGlobalInfo::getContext());

  for (auto &SubResult : MatchResult) {
    const DeclRefExpr *AugDRE = SubResult.getNodeAs<DeclRefExpr>("AttrVar");
    if (!AugDRE)
      break;
    SourceLocation AugLoc = SM.getExpansionLoc(AugDRE->getBeginLoc());
    int CurrentDistance = DRELocOffset - SM.getFileOffset(AugLoc);
    // we need to skip no effect reference
    if (CurrentDistance <= 0 || !isModifiedRef(AugDRE)) {
      continue;
    }
    MinDistance = MinDistance > CurrentDistance ? CurrentDistance : MinDistance;

    auto BO = CT.getParents(*AugDRE)[0].get<BinaryOperator>();
    if (BO && BO->getOpcode() == BinaryOperatorKind::BO_Assign) {
      auto Condition = [&](const clang::DynTypedNode &Node) -> bool {
        if (Node.get<IfStmt>() || Node.get<WhileStmt>() ||
            Node.get<ForStmt>() || Node.get<DoStmt>() || Node.get<CaseStmt>() ||
            Node.get<SwitchStmt>() || Node.get<CompoundStmt>()) {
          return true;
        }
        return false;
      };
      auto BOCS = DpctGlobalInfo::findAncestor<CompoundStmt>(BO, Condition);
      auto CECS = DpctGlobalInfo::findAncestor<CompoundStmt>(CE, Condition);
      if (!(BOCS && CECS && BOCS == CECS))
        continue;
      if (auto RHS = dyn_cast<DeclRefExpr>(BO->getRHS())) {
        RecognizedMinDistance = CurrentDistance < RecognizedMinDistance
                                    ? CurrentDistance
                                    : RecognizedMinDistance;
        AttributeName = RHS->getNameInfo().getName().getAsString();
      }
    }
  }
  // if there is a non-recognized reference closer than recognized reference,
  // then we need to clear current attribute name
  if (RecognizedMinDistance > MinDistance)
    AttributeName.clear();
  return AttributeName;
}

void FunctionCallRule::runRule(const MatchFinder::MatchResult &Result) {
  if (!CallExprRewriterFactoryBase::RewriterMap)
    return;

  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "FunctionCallUsed")))
      return;
    IsAssigned = true;
  }

  if (!CE->getDirectCallee())
    return;

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  auto Iter = CallExprRewriterFactoryBase::RewriterMap->find(FuncName);
  if (Iter != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  std::string Prefix, Suffix;
  if (IsAssigned) {
    Prefix = MapNames::getCheckErrorMacroName() + "(";
    Suffix = ")";
  }

  if (FuncName == "cudaGetDeviceCount") {
    if (IsAssigned) {
      requestFeature(HelperFeatureEnum::device_ext);
    }
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(
        new InsertBeforeStmt(CE, Prefix + ResultVarName + " = "));
    emplaceTransformation(new ReplaceStmt(CE, MapNames::getDpctNamespace() +
                                                  "device_count()" + Suffix));
    requestFeature(HelperFeatureEnum::device_ext);
  } else if (FuncName == "cudaGetDeviceProperties" ||
             FuncName == "cudaGetDeviceProperties_v2") {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  } else if (FuncName == "cudaDriverGetVersion" ||
             FuncName == "cudaRuntimeGetVersion") {
    if (IsAssigned) {
      requestFeature(HelperFeatureEnum::device_ext);
    }
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(
        new InsertBeforeStmt(CE, Prefix + ResultVarName + " = "));

    std::string ReplStr = MapNames::getDpctNamespace() + "get_major_version(";
    if (DpctGlobalInfo::useNoQueueDevice()) {
      ReplStr += DpctGlobalInfo::getGlobalDeviceName();
      ReplStr += ")";
    } else {
      ReplStr += MapNames::getDpctNamespace();
      ReplStr += "get_current_device())";
    }
    emplaceTransformation(new ReplaceStmt(CE, ReplStr + Suffix));
    report(CE->getBeginLoc(), Warnings::TYPE_MISMATCH, false);
    requestFeature(HelperFeatureEnum::device_ext);
  } else if (FuncName == "cudaDeviceReset" || FuncName == "cudaThreadExit") {
    if (IsAssigned) {
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (isPlaceholderIdxDuplicated(CE))
      return;
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::HFT_CurrentDevice);
    emplaceTransformation(new ReplaceStmt(CE, Prefix + "{{NEEDREPLACED" +
                                                  std::to_string(Index) +
                                                  "}}.reset()" + Suffix));
    requestFeature(HelperFeatureEnum::device_ext);
  } else if (FuncName == "cudaSetDevice") {
    if (DpctGlobalInfo::useNoQueueDevice()) {
      emplaceTransformation(new ReplaceStmt(CE, "0"));
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             "cudaSetDevice",
             "it is redundant if it is migrated with option "
             "--helper-function-preference=no-queue-device "
             "which declares a global SYCL device and queue.");
    } else {
      DpctGlobalInfo::setDeviceChangedFlag(true);
      report(CE->getBeginLoc(), Diagnostics::DEVICE_ID_DIFFERENT, false,
             getStmtSpelling(CE->getArg(0)));
      emplaceTransformation(new ReplaceStmt(
          CE->getCallee(),
          Prefix + MapNames::getDpctNamespace() + "select_device"));
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (IsAssigned)
      emplaceTransformation(new InsertAfterStmt(CE, ")"));
  } else if (FuncName == "cudaDeviceGetAttribute") {
    std::string ResultVarName = getDrefName(CE->getArg(0));
    auto AttrArg = CE->getArg(1);
    std::string AttributeName;
    if (auto DRE = dyn_cast<DeclRefExpr>(AttrArg)) {
      AttributeName = DRE->getNameInfo().getName().getAsString();
    } else {
      AttributeName = findValueofAttrVar(AttrArg, CE);
      if (AttributeName.empty()) {
        report(CE->getBeginLoc(), Diagnostics::UNPROCESSED_DEVICE_ATTRIBUTE,
               false);
        return;
      }
    }
    std::string ReplStr{ResultVarName};
    auto StmtStrArg2 = getStmtSpelling(CE->getArg(2));

    if (AttributeName == "cudaDevAttrConcurrentManagedAccess" &&
        DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
      std::string ReplStr = getDrefName(CE->getArg(0));
      ReplStr += " = false";
      if (IsAssigned)
        ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
      emplaceTransformation(new ReplaceStmt(CE, ReplStr));
      return;
    }

    if (AttributeName == "cudaDevAttrComputeMode") {
      report(CE->getBeginLoc(), Diagnostics::COMPUTE_MODE, false);
      ReplStr += " = 1";
    } else if (AttributeName == "cudaDevAttrTextureAlignment" &&
               DpctGlobalInfo::useSYCLCompat()) {
      ReplStr += " = " + MapNames::getDpctNamespace() + "get_device(";
      ReplStr += StmtStrArg2;
      ReplStr += ").get_mem_base_addr_align() / 8";
      requestFeature(HelperFeatureEnum::device_ext);
    } else {
      auto Search = MapNames::EnumNamesMap.find(AttributeName);
      if (Search == MapNames::EnumNamesMap.end()) {
        return;
      }
      requestHelperFeatureForEnumNames(AttributeName);

      if (AttributeName == "cudaDevAttrMaxSharedMemoryPerBlockOptin") {
        report(CE->getBeginLoc(), Diagnostics::LOCAL_MEM_SIZE, false,
               AttributeName);
      }

      ReplStr += " = " + MapNames::getDpctNamespace() + "get_device(";
      ReplStr += StmtStrArg2;
      ReplStr += ").";
      ReplStr += Search->second->NewName;
      ReplStr += "()";
      requestFeature(HelperFeatureEnum::device_ext);

      if (AttributeName == "cudaDevAttrTextureAlignment") {
        report(CE->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
               AttributeName, Search->second->NewName);
      }
    }
    if (IsAssigned)
      ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaDeviceGetP2PAttribute") {
    std::string ResultVarName = getDrefName(CE->getArg(0));
    emplaceTransformation(new ReplaceStmt(CE, ResultVarName + " = 0"));
    report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
           "cudaDeviceGetP2PAttribute");
  } else if (FuncName == "cudaDeviceGetPCIBusId") {
    report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
           "cudaDeviceGetPCIBusId");
  } else if (FuncName == "cudaGetDevice") {
    std::string ReplStr = getDrefName(CE->getArg(0)) + " = ";
    if (DpctGlobalInfo::useNoQueueDevice()) {
      ReplStr += "0";
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             "cudaGetDevice",
             "it is redundant if it is migrated with option "
             "--helper-function-preference=no-queue-device "
             "which declares a global SYCL device and queue.");
    } else {
      ReplStr += MapNames::getDpctNamespace() + "get_current_device_id()";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (IsAssigned)
      ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaDeviceSynchronize" ||
             FuncName == "cudaThreadSynchronize") {
    if (isPlaceholderIdxDuplicated(CE))
      return;
    std::string ReplStr;
    if (DpctGlobalInfo::useNoQueueDevice()) {
      ReplStr = DpctGlobalInfo::getGlobalQueueName() + ".wait_and_throw()";
    } else {
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_CurrentDevice);
      ReplStr = "{{NEEDREPLACED" + std::to_string(Index) +
                "}}.queues_wait_and_throw()";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (IsAssigned) {
      ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));

  } else if (FuncName == "cudaGetLastError" ||
             FuncName == "cudaPeekAtLastError" ||
             FuncName == "cudaGetErrorString" ||
             FuncName == "cudaGetErrorName") {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  } else if (FuncName == "clock" || FuncName == "clock64" ||
             FuncName == "__nanosleep") {
    if (CE->getDirectCallee()->hasAttr<CUDAGlobalAttr>() ||
        CE->getDirectCallee()->hasAttr<CUDADeviceAttr>()) {
      report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED_SYCL_UNDEF, false,
             FuncName);
    }
    // Add '#include <time.h>' directive to the file only once
    auto Loc = CE->getBeginLoc();
    DpctGlobalInfo::getInstance().insertHeader(Loc, HT_Time);
  } else if (FuncName == "cudaDeviceSetLimit" ||
             FuncName == "cudaThreadSetLimit" ||
             FuncName == "cudaDeviceSetCacheConfig" ||
             FuncName == "cudaDeviceGetCacheConfig" ||
             FuncName == "cuCtxSetCacheConfig" || FuncName == "cuCtxSetLimit" ||
             FuncName == "cudaCtxResetPersistingL2Cache" ||
             FuncName == "cuCtxResetPersistingL2Cache") {
    if (auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
        Msg != MapNames::RemovedAPIWarningMessage.end()) {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceStmt(CE, "0"));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceStmt(CE, ""));
      }
    }
  } else if(FuncName == "cudaStreamSetAttribute" ||
             FuncName == "cudaStreamGetAttribute" ){
    std::string ArgStr = getStmtSpelling(CE->getArg(1));
    if (ArgStr == "cudaStreamAttributeAccessPolicyWindow") {
      if (IsAssigned) {
        report(
            CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
            MapNames::ITFName.at(FuncName),
            "SYCL currently does not support setting cache config on devices.");
        emplaceTransformation(new ReplaceStmt(CE, "0"));
      } else {
        report(
            CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
            MapNames::ITFName.at(FuncName),
            "SYCL currently does not support setting cache config on devices.");
        emplaceTransformation(new ReplaceStmt(CE, ""));
      }
    } else if (ArgStr == "cudaLaunchAttributeIgnore") {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName),
               "this functionality is redundant in SYCL.");
        emplaceTransformation(new ReplaceStmt(CE, "0"));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName),
               "this functionality is redundant in SYCL.");
        emplaceTransformation(new ReplaceStmt(CE, ""));
      }
    } else {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName),
               "SYCL currently does not support corresponding setting.");
        emplaceTransformation(new ReplaceStmt(CE, "0"));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName),
               "SYCL currently does not support corresponding setting.");
        emplaceTransformation(new ReplaceStmt(CE, ""));
      }
    }
  } else if(FuncName == "cudaFuncSetAttribute"){
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName),
             "SYCL currently does not support corresponding setting.");
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName),
             "SYCL currently does not support corresponding setting.");
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
  }else if (FuncName == "cudaOccupancyMaxPotentialBlockSize") {
    report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
           MapNames::ITFName.at(FuncName));
  } else if (FuncName == "cudaDeviceGetLimit") {
    ExprAnalysis EA;
    EA.analyze(CE->getArg(0));
    auto Arg0Str = EA.getReplacedString();
    std::string ReplStr{"*"};
    ReplStr += Arg0Str;
    ReplStr += " = 0";
    if (IsAssigned) {
      ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    report(CE->getBeginLoc(), Diagnostics::DEVICE_LIMIT_NOT_SUPPORTED, false);
  } else if (FuncName == "cudaDeviceSetSharedMemConfig" ||
             FuncName == "cudaFuncSetSharedMemConfig" ||
             FuncName == "cudaFuncSetCacheConfig" ||
             FuncName == "cuFuncSetCacheConfig") {
    std::string Msg = "SYCL currently does not support configuring shared "
                      "memory on devices.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
  } else if (FuncName == "cudaSetDeviceFlags") {
    std::string Msg =
        "SYCL currently does not support setting flags for devices.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg);
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
  } else if (FuncName == "cudaIpcGetEventHandle" ||
             FuncName == "cudaIpcOpenEventHandle" ||
             FuncName == "cudaIpcGetMemHandle" ||
             FuncName == "cudaIpcOpenMemHandle" ||
             FuncName == "cudaIpcCloseMemHandle") {
    report(CE->getBeginLoc(), Diagnostics::IPC_NOT_SUPPORTED, false);
  } else if (FuncName == "__trap") {
    if (DpctGlobalInfo::useAssert()) {
      emplaceTransformation(new ReplaceStmt(CE, "assert(0)"));
    } else {
      report(
          CE->getBeginLoc(), Diagnostics::NOT_SUPPORTED_PARAMETER, false,
          FuncName,
          "assert extension is disabled. You can migrate the code with assert "
          "extension by not specifying --no-dpcpp-extensions=assert");
    }
  } else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected function name: " << FuncName;
    return;
  }
}


EventAPICallRule *EventAPICallRule::CurrentRule = nullptr;
void EventAPICallRule::registerMatcher(MatchFinder &MF) {
  auto eventAPIName = [&]() {
    return hasAnyName(
        "cudaEventCreate", "cudaEventCreateWithFlags", "cudaEventDestroy",
        "cudaEventRecord", "cudaEventElapsedTime", "cudaEventSynchronize",
                      "cudaEventQuery", "cuEventCreate", "cuEventRecord",
        "cuEventSynchronize", "cuEventQuery", "cuEventElapsedTime",
        "cuEventDestroy_v2");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(eventAPIName())), parentStmt()))
          .bind("eventAPICall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(eventAPIName())),
                               unless(parentStmt())))
                    .bind("eventAPICallUsed"),
                this);
  MF.addMatcher(declRefExpr(to(enumConstantDecl(hasType(
                                enumDecl(hasName("CUevent_flags_enum"))))))
                    .bind("eventEnum"),
                this);
}

bool isEqualOperator(const Stmt *S) {
  if (!S)
    return false;
  if (auto BO = dyn_cast<BinaryOperator>(S))
    return BO->getOpcode() == BO_EQ || BO->getOpcode() == BO_NE;

  if (auto COCE = dyn_cast<CXXOperatorCallExpr>(S))
    return COCE->getOperator() == OO_EqualEqual ||
           COCE->getOperator() == OO_ExclaimEqual;

  return false;
}

const Expr *getLhs(const Stmt *);
const Expr *getRhs(const Stmt *);
const VarDecl *getAssignTargetDecl(const Stmt *E) {
  if (isAssignOperator(E))
    if (auto L = getLhs(E))
      if (auto DRE = dyn_cast<DeclRefExpr>(L->IgnoreImpCasts()))
        return dyn_cast<VarDecl>(DRE->getDecl());

  return nullptr;
}

const VarDecl *EventQueryTraversal::getAssignTarget(const CallExpr *Call) {
  auto ParentMap = Context.getParents(*Call);
  if (ParentMap.size() == 0)
    return nullptr;

  auto &Parent = ParentMap[0];
  if (auto VD = Parent.get<VarDecl>()) {
    return VD;
  }
  if (auto BO = Parent.get<BinaryOperator>())
    return getAssignTargetDecl(BO);

  if (auto COE = Parent.get<CXXOperatorCallExpr>())
    return getAssignTargetDecl(COE);

  return nullptr;
}

bool EventQueryTraversal::isEventQuery(const CallExpr *Call) {
  if (!Call)
    return false;
  if (auto Callee = Call->getDirectCallee())
    if (Callee->getName() == "cudaEventQuery" ||
        Callee->getName() == "cuEventQuery")
      return QueryCallUsed = true;
  return false;
}

std::string EventQueryTraversal::getReplacedEnumValue(const DeclRefExpr *DRE) {
  if (!DRE)
    return std::string();
  if (auto ECD = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
    auto Name = ECD->getName();
    if (Name == "cudaSuccess") {
      return MapNames::getClNamespace() +
             "info::event_command_status::complete";
    }
  }
  return std::string();
}

TextModification *
EventQueryTraversal::buildCallReplacement(const CallExpr *Call) {
  static std::string MemberName = "get_info<" + MapNames::getClNamespace() +
                                  "info::event::command_execution_status>";
  std::string ReplStr;
  MemberCallPrinter<const Expr *, StringRef, false> Printer(Call->getArg(0),
                                                            true, MemberName);
  llvm::raw_string_ostream OS(ReplStr);
  Printer.print(OS);
  return new ReplaceStmt(Call, std::move(OS.str()));
}

bool EventQueryTraversal::checkVarDecl(const VarDecl *VD,
                                       const FunctionDecl *TargetFD) {
  if (!VD || !TargetFD)
    return false;
  if (dyn_cast<FunctionDecl>(VD->getDeclContext()) == TargetFD &&
      VD->getKind() == Decl::Var) {
    auto DS = DpctGlobalInfo::findParent<DeclStmt>(VD);
    return DS && DS->isSingleDecl();
  }
  return false;
}

bool EventQueryTraversal::traverseFunction(const FunctionDecl *FD,
                                           const VarDecl *VD) {
  if (!checkVarDecl(VD, FD))
    return Rule->VarDeclCache[VD] = false;
  ResultTy Result;
  auto Ret = traverseStmt(FD->getBody(), VD, Result) && QueryCallUsed;

  for (const auto &R : Result) {
    Rule->ExprCache[R.first] = Ret;
    if (Ret)
      Rule->emplaceTransformation(R.second);
  }
  return Rule->VarDeclCache[VD] = Ret;
}

bool EventQueryTraversal::traverseAssignRhs(const Expr *Rhs, ResultTy &Result) {
  if (!Rhs)
    return true;
  auto Call = dyn_cast<CallExpr>(Rhs->IgnoreImpCasts());
  if (!isEventQuery(Call))
    return false;

  Result.emplace_back(Call, buildCallReplacement(Call));
  return true;
}

bool EventQueryTraversal::traverseEqualStmt(const Stmt *S, const VarDecl *VD,
                                            ResultTy &Result) {
  const Expr *L = getLhs(S), *R = getRhs(S);
  do {
    if (!L || !R)
      break;
    const DeclRefExpr *LRef = dyn_cast<DeclRefExpr>(L->IgnoreImpCasts()),
                      *RRef = dyn_cast<DeclRefExpr>(R->IgnoreImpCasts());
    if (!LRef || !RRef)
      break;

    const DeclRefExpr *TargetExpr = nullptr;
    if (LRef->getDecl() == VD)
      TargetExpr = RRef;
    else if (RRef->getDecl() == VD)
      TargetExpr = LRef;

    auto Replaced = getReplacedEnumValue(TargetExpr);
    if (Replaced.empty())
      break;
    Result.emplace_back(TargetExpr, new ReplaceStmt(TargetExpr, Replaced));
    return true;
  } while (false);
  for (auto Child : S->children())
    if (!traverseStmt(Child, VD, Result))
      return false;
  return true;
}

bool EventQueryTraversal::traverseStmt(const Stmt *S, const VarDecl *VD,
                                       ResultTy &Result) {
  if (!S)
    return true;
  switch (S->getStmtClass()) {
  case Stmt::DeclStmtClass: {
    auto DS = static_cast<const DeclStmt *>(S);
    if (DS->isSingleDecl() && VD == DS->getSingleDecl()) {
      Result.emplace_back(
          S, new ReplaceTypeInDecl(VD, MapNames::getClNamespace() +
                                           "info::event_command_status"));
      return traverseAssignRhs(VD->getInit(), Result);
    }
    for (auto D : DS->decls())
      if (auto VDecl = dyn_cast<VarDecl>(D))
        if (!traverseStmt(VDecl->getInit(), VD, Result))
          return false;
    break;
  }
  case Stmt::DeclRefExprClass:
    if (auto D =
            dyn_cast<VarDecl>(static_cast<const DeclRefExpr *>(S)->getDecl()))
      return D != VD;
    break;
  case Stmt::BinaryOperatorClass:
  case Stmt::CXXOperatorCallExprClass:
    if (getAssignTargetDecl(S) == VD)
      return traverseAssignRhs(getRhs(S), Result);
    if (isEqualOperator(S))
      return traverseEqualStmt(S, VD, Result);
    LLVM_FALLTHROUGH;
  default:
    for (auto Child : S->children())
      if (!traverseStmt(Child, VD, Result))
        return false;
    break;
  }
  return true;
}

bool EventQueryTraversal::startFromStmt(
    const Stmt *S, const std::function<const VarDecl *()> &VDGetter) {
  if (!Rule)
    return false;
  auto ExprIter = Rule->ExprCache.find(S);
  if (ExprIter != Rule->ExprCache.end())
    return ExprIter->second;

  const VarDecl *VD = VDGetter();
  if (!VD)
    return Rule->ExprCache[S] = false;
  auto VarDeclIter = Rule->VarDeclCache.find(VD);
  if (VarDeclIter != Rule->VarDeclCache.end())
    return Rule->ExprCache[S] = VarDeclIter->second;

  return traverseFunction(DpctGlobalInfo::findAncestor<FunctionDecl>(S), VD);
}

// Handle case like "cudaSuccess == cudaEventQuery()" or "cudaSuccess !=
// cudaEventQeury()".
void EventQueryTraversal::handleDirectEqualStmt(const DeclRefExpr *DRE,
                                                const CallExpr *Call) {
  if (!isEventQuery(Call))
    return;
  auto DREReplaceStr = getReplacedEnumValue(DRE);
  if (DREReplaceStr.empty())
    return;
  Rule->emplaceTransformation(new ReplaceStmt(DRE, DREReplaceStr));
  Rule->emplaceTransformation(buildCallReplacement(Call));
  Rule->ExprCache[DRE] = Rule->ExprCache[Call] = true;
  return;
}

bool EventQueryTraversal::startFromQuery(const CallExpr *Call) {
  return startFromStmt(Call, [&]() -> const VarDecl * {
    if (isEventQuery(Call))
      return getAssignTarget(Call);
    return nullptr;
  });
}

bool EventQueryTraversal::startFromEnumRef(const DeclRefExpr *DRE) {
  if (getReplacedEnumValue(DRE).empty())
    return false;

  return startFromStmt(DRE, [&]() -> const VarDecl * {
    auto ImpCast = DpctGlobalInfo::findParent<ImplicitCastExpr>(DRE);
    if (!ImpCast)
      return nullptr;
    auto S = DpctGlobalInfo::findParent<Stmt>(ImpCast);
    if (!isEqualOperator(S))
      return nullptr;
    const Expr *TargetExpr = nullptr, *L = getLhs(S), *R = getRhs(S);
    if (L == ImpCast)
      TargetExpr = R;
    else if (R == ImpCast)
      TargetExpr = L;

    if (!TargetExpr)
      return nullptr;
    if (auto TargetDRE = dyn_cast<DeclRefExpr>(TargetExpr->IgnoreImpCasts()))
      return dyn_cast<VarDecl>(TargetDRE->getDecl());
    else if (auto Call = dyn_cast<CallExpr>(TargetExpr->IgnoreImpCasts()))
      handleDirectEqualStmt(DRE, Call);
    return nullptr;
  });
}

bool EventQueryTraversal::startFromTypeLoc(TypeLoc TL) {
  if (DpctGlobalInfo::getUnqualifiedTypeName(QualType(TL.getTypePtr(), 0)) ==
      "cudaError_t")
    if (auto DS = DpctGlobalInfo::findAncestor<DeclStmt>(&TL))
      return startFromStmt(DS, [&]() -> const VarDecl * {
        if (DS->isSingleDecl())
          if (auto VD = dyn_cast<VarDecl>(DS->getSingleDecl()))
            return (VD->getTypeSourceInfo()->getTypeLoc() == TL) ? VD : nullptr;
        return nullptr;
      });
  return false;
}

EventQueryTraversal EventAPICallRule::getEventQueryTraversal() {
  return EventQueryTraversal(CurrentRule);
}

bool EventAPICallRule::isEventElapsedTimeFollowed(const CallExpr *Expr) {
  bool IsMeasureTime = false;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto CELoc = SM.getExpansionLoc(Expr->getBeginLoc()).getRawEncoding();
  auto FD = getImmediateOuterFuncDecl(Expr);
  if (!FD)
    return false;
  auto FuncBody = FD->getBody();
  for (auto It = FuncBody->child_begin(); It != FuncBody->child_end(); ++It) {
    auto Loc = SM.getExpansionLoc(It->getBeginLoc()).getRawEncoding();
    if (Loc < CELoc)
      continue;

    const CallExpr *Call = nullptr;
    findEventAPI(*It, Call, "cudaEventElapsedTime");
    if (!Call) {
      findEventAPI(*It, Call, "cuEventElapsedTime");
    }

    if (Call) {
      // To check the argment of "cudaEventQuery" is same as the second argument
      // of "cudaEventElapsedTime", in the code pieces:
      // ...
      // unsigned long int counter = 0;
      // while (cudaEventQuery(stop) == cudaErrorNotReady) {
      //  counter++;
      // }
      // cudaEventElapsedTime(&gpu_time, start, stop);
      // ...
      auto Arg2 = getStmtSpelling(Call->getArg(2));
      auto Arg0 = getStmtSpelling(Expr->getArg(0));
      if (Arg2 == Arg0)
        IsMeasureTime = true;
    }
  }
  return IsMeasureTime;
}

void EventAPICallRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto *DRE = getNodeAsType<DeclRefExpr>(Result, "eventEnum")) {
    if (auto *EC = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
      std::string EName = EC->getName().str();
      report(DRE->getBeginLoc(), Diagnostics::UNSUPPORTED_FEATURE_IN_SYCL,
             false, EName, "is", "event");
      emplaceTransformation(new ReplaceStmt(DRE, "0"));
    }
    return;
  }

  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "eventAPICall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "eventAPICallUsed")))
      return;
    IsAssigned = true;
  }

  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(FuncName);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  if (FuncName == "cudaEventQuery" || FuncName == "cuEventQuery") {
    if (getEventQueryTraversal().startFromQuery(CE))
      return;

    // Pattern-based solution for migration of time measurement code is enabled
    // only when option '--enable-profiling' is disabled.
    if (!isEventElapsedTimeFollowed(CE) &&
        !DpctGlobalInfo::getEnablepProfilingFlag()) {
      auto FD = getImmediateOuterFuncDecl(CE);
      if (!FD)
        return;
      auto FuncBody = FD->getBody();

      if (!FuncBody)
        return;
      reset();
      TimeElapsedCE = CE;
      if (FuncName == "cudaEventQuery") {
        updateAsyncRange(FuncBody, "cudaEventCreate");
      } else {
        updateAsyncRange(FuncBody, "cuEventCreate");
      }
      if (RecordBegin && RecordEnd) {
        processAsyncJob(FuncBody);

        if (!IsKernelInLoopStmt) {
          DpctGlobalInfo::getInstance().updateTimeStubTypeInfo(
              RecordBegin->getBeginLoc(), TimeElapsedCE->getEndLoc());
        }
      }
    }

    if (DpctGlobalInfo::useSYCLCompat()) {
      report(CE->getBeginLoc(), Diagnostics::UNSUPPORT_SYCLCOMPAT, false,
             FuncName);
      return;
    }
    std::string ReplStr = MapNames::getDpctNamespace() + "sycl_event_query";
    emplaceTransformation(new ReplaceCalleeName(CE, std::move(ReplStr)));
  } else if (FuncName == "cudaEventRecord" || FuncName == "cuEventRecord") {
    handleEventRecord(CE, Result, IsAssigned);
  } else if (FuncName == "cudaEventElapsedTime" ||
             FuncName == "cuEventElapsedTime") {
    // Reset from last migration on time measurement.
    // Do NOT delete me.
    reset();
    TimeElapsedCE = CE;
    handleEventElapsedTime(IsAssigned);
  } else if (FuncName == "cudaEventSynchronize" ||
             FuncName == "cuEventSynchronize") {
    if(DpctGlobalInfo::getEnablepProfilingFlag()) {
      // Option '--enable-profiling' is enabled
      std::string ReplStr;
      ExprAnalysis EA(CE->getArg(0));
      ReplStr = EA.getReplacedString();
      if (dyn_cast<CStyleCastExpr>(CE->getArg(0)->IgnoreImplicitAsWritten())) {
        ReplStr = "(" + ReplStr + ")";
      }
      ReplStr += "->wait_and_throw()";
      if (IsAssigned) {
        ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
        requestFeature(HelperFeatureEnum::device_ext);
      }
      emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    } else {
      // Option '--enable-profiling' is not enabled
      bool NeedReport = false;
      std::string ReplStr;
      ExprAnalysis EA(CE->getArg(0));
      ReplStr = EA.getReplacedString();
      if (dyn_cast<CStyleCastExpr>(CE->getArg(0)->IgnoreImplicitAsWritten())) {
        ReplStr = "(" + ReplStr + ")";
      }
      ReplStr += "->wait_and_throw()";
      if (IsAssigned) {
        ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
        NeedReport = true;
      }

      auto &Context = dpct::DpctGlobalInfo::getContext();
      const auto &TM = ReplaceStmt(CE, ReplStr);
      const auto R = TM.getReplacement(Context);
      DpctGlobalInfo::getInstance().insertEventSyncTypeInfo(R, NeedReport,
                                                            IsAssigned);
    }
  } else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected function name: " << FuncName;
    return;
  }
}

// Gets the declared size of the array referred in E, if E is either
// ConstantArrayType or VariableArrayType; otherwise return an empty
// string.
// For example:
// int maxSize = 100;
// int a[10];
// int b[maxSize];
//
// E1(const Expr *)-> a[2]
// E2(const Expr *)-> b[3]
// getArraySize(E1) => 10
// getArraySize(E2) => maxSize
std::string getArrayDeclSize(const Expr *E) {
  const ArrayType *AT = nullptr;
  if (auto ME = dyn_cast<MemberExpr>(E))
    AT = ME->getMemberDecl()->getType()->getAsArrayTypeUnsafe();
  else if (auto DRE = dyn_cast<DeclRefExpr>(E))
    AT = DRE->getDecl()->getType()->getAsArrayTypeUnsafe();

  if (!AT)
    return {};

  if (auto CAT = dyn_cast<ConstantArrayType>(AT))
    return std::to_string(*CAT->getSize().getRawData());
  if (auto VAT = dyn_cast<VariableArrayType>(AT))
    return ExprAnalysis::ref(VAT->getSizeExpr());
  return {};
}

// Returns true if E is array type for a MemberExpr or DeclRefExpr; returns
// false if E is pointer type.
// Requires: E is the base of ArraySubscriptExpr
bool isArrayType(const Expr *E) {
  if (auto ME = dyn_cast<MemberExpr>(E)) {
    auto AT = ME->getMemberDecl()->getType()->getAsArrayTypeUnsafe();
    return AT ? true : false;
  }
  if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
    auto AT = DRE->getDecl()->getType()->getAsArrayTypeUnsafe();
    return AT ? true : false;
  }
  return false;
}

// Get the time point helper variable name for an event. The helper variable is
// declared right after its corresponding event variable.
std::string getTimePointNameForEvent(const Expr *E, bool IsDecl) {
  std::string TimePointName;
  E = E->IgnoreImpCasts();
  if (auto UO = dyn_cast<UnaryOperator>(E))
    return getTimePointNameForEvent(UO->getSubExpr(), IsDecl);
  if (auto ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    auto Base = ASE->getBase()->IgnoreImpCasts();
    if (isArrayType(Base))
      return getTimePointNameForEvent(Base, IsDecl) + "[" +
             (IsDecl ? getArrayDeclSize(Base)
                     : ExprAnalysis::ref(ASE->getIdx())) +
             "]";
    return getTimePointNameForEvent(Base, IsDecl) + "_" +
           ExprAnalysis::ref(ASE->getIdx());
  }
  if (auto ME = dyn_cast<MemberExpr>(E)) {
    auto Base = ME->getBase()->IgnoreImpCasts();
    return ((IsDecl || ME->isImplicitAccess())
                ? ""
                : ExprAnalysis::ref(Base) + (ME->isArrow() ? "->" : ".")) +
           ME->getMemberDecl()->getNameAsString() + getCTFixedSuffix();
  }
  if (auto DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getDecl()->getNameAsString() + getCTFixedSuffix();
  return TimePointName;
}

// Get the (potentially inner) decl of E for common Expr types, including
// UnaryOperator, ArraySubscriptExpr, MemberExpr and DeclRefExpr; otherwise
// returns nullptr;
const ValueDecl *getDecl(const Expr *E) {
  E = E->IgnoreImpCasts();
  if (auto UO = dyn_cast<UnaryOperator>(E))
    return getDecl(UO->getSubExpr());
  if (auto ASE = dyn_cast<ArraySubscriptExpr>(E))
    return getDecl(ASE->getBase()->IgnoreImpCasts());
  if (auto ME = dyn_cast<MemberExpr>(E))
    return ME->getMemberDecl();
  if (auto DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getDecl();
  if (auto *CastExpr = dyn_cast<CStyleCastExpr>(E))
    return getDecl(CastExpr->getSubExpr());
  return nullptr;
}

void EventAPICallRule::findEventAPI(const Stmt *Node, const CallExpr *&Call,
                                    const std::string EventAPIName) {
  if (!Node)
    return;

  if (auto CE = dyn_cast<CallExpr>(Node)) {
    if (CE->getDirectCallee()) {
      if (CE->getDirectCallee()->getNameAsString() == EventAPIName) {
        Call = CE;
        return;
      }
    }
  }
  for (auto It = Node->child_begin(); It != Node->child_end(); ++It) {
    findEventAPI(*It, Call, EventAPIName);
  }
}

void EventAPICallRule::handleEventRecordWithProfilingEnabled(
    const CallExpr *CE, const MatchFinder::MatchResult &Result,
    bool IsAssigned) {
  auto StreamArg = CE->getArg(CE->getNumArgs() - 1);
  auto EventArg = CE->getArg(0);
  ExprAnalysis StreamEA(StreamArg);
  ExprAnalysis Arg0EA(EventArg);
  auto StreamName = StreamEA.getReplacedString();
  auto ArgName = Arg0EA.getReplacedString();
  bool IsDefaultStream = isDefaultStream(StreamArg);
  auto IndentLoc = CE->getBeginLoc();
  auto &SM = DpctGlobalInfo::getSourceManager();

  if (needExtraParens(EventArg)) {
    ArgName = "(" + ArgName + ")";
  }

  if (needExtraParensInMemberExpr(StreamArg)) {
    StreamName = "(" + StreamName + ")";
  }

  if (IndentLoc.isMacroID())
    IndentLoc = SM.getExpansionLoc(IndentLoc);

  if (IsAssigned) {

    std::string StmtStr;
    if (IsDefaultStream) {
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
      std::string Str;
      if (!DpctGlobalInfo::useEnqueueBarrier()) {
        // ext_oneapi_submit_barrier is specified in the value of option
        // --no-dpcpp-extensions.
        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {

          Str = MapNames::getDpctNamespace() +
                "get_current_device().queues_wait_and_throw();";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          std::string SubStr = "{{NEEDREPLACEQ" + std::to_string(Index) +
                               "}}.single_task([=](){});";
          SubStr = "*" + ArgName + " = " + SubStr;
          Str += SubStr;

          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += MapNames::getDpctNamespace() +
                 "get_current_device().queues_wait_and_throw();";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += "return 0;";

          Str = "[&](){" + Str + "}()";
          emplaceTransformation(new ReplaceStmt(CE, std::move(Str)));
          return;
        }
        Str = "{{NEEDREPLACEQ" + std::to_string(Index) +
              "}}.single_task([=](){})";

      } else {
        if (DpctGlobalInfo::useSYCLCompat()) {
          report(CE->getBeginLoc(), Diagnostics::UNSUPPORT_SYCLCOMPAT, false,
                 "cudaEventRecord");
          return;
        }
        std::string ReplaceStr;
        ReplaceStr = MapNames::getDpctNamespace() + "sync_barrier";
        emplaceTransformation(new ReplaceCalleeName(CE, std::move(ReplaceStr)));
        emplaceTransformation(new InsertBeforeStmt(CE, MapNames::getCheckErrorMacroName() + "("));
        emplaceTransformation(new InsertAfterStmt(CE, ")"));
        report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_ZERO, false);
        return;
      }
      StmtStr = "*" + ArgName + " = " + Str;
    } else {
      std::string Str;
      if (!DpctGlobalInfo::useEnqueueBarrier()) {
        // ext_oneapi_submit_barrier is specified in the value of option
        // --no-dpcpp-extensions.

        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {

          Str = MapNames::getDpctNamespace() +
                "get_current_device().queues_wait_and_throw();";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += StreamName + "->" + "single_task([=](){});";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += MapNames::getDpctNamespace() +
                 "get_current_device().queues_wait_and_throw(); return 0;";

          Str = "[&](){" + Str + "}()";
          emplaceTransformation(new ReplaceStmt(CE, std::move(Str)));
          return;
        }
        Str = StreamName + "->" + "single_task([=](){})";

      } else {
        Str = StreamName + "->" + "ext_oneapi_submit_barrier()";
      }
      StmtStr = "*" + ArgName + " = " + Str;
    }
    StmtStr = MapNames::getCheckErrorMacroName() + "(" + StmtStr + ")";

    emplaceTransformation(new ReplaceStmt(CE, std::move(StmtStr)));

    report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_ZERO, false);

  } else {
    std::string ReplStr;
    if (IsDefaultStream) {
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
      std::string Str;
      if (!DpctGlobalInfo::useEnqueueBarrier()) {
        // ext_oneapi_submit_barrier is specified in the value of option
        // --no-dpcpp-extensions.

        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {

          Str = MapNames::getDpctNamespace() +
                "get_current_device().queues_wait_and_throw();";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += "*" + ArgName + " = {{NEEDREPLACEQ" + std::to_string(Index) +
                 "}}.single_task([=](){});";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += MapNames::getDpctNamespace() +
                 "get_current_device().queues_wait_and_throw()";

        } else {
          Str = "*" + ArgName + " = {{NEEDREPLACEQ" + std::to_string(Index) +
                "}}.single_task([=](){})";
        }

      } else {
        if (DpctGlobalInfo::useSYCLCompat()) {
          report(CE->getBeginLoc(), Diagnostics::UNSUPPORT_SYCLCOMPAT, false,
                 "cudaEventRecord");
          return;
        }
        std::string ReplaceStr;
        ReplaceStr = MapNames::getDpctNamespace() + "sync_barrier";
        emplaceTransformation(new ReplaceCalleeName(CE, std::move(ReplaceStr)));
        return;
      }
      ReplStr += Str;
    } else {

      std::string Str;
      if (!DpctGlobalInfo::useEnqueueBarrier()) {
        // ext_oneapi_submit_barrier is specified in the value of option
        // --no-dpcpp-extensions.

        if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {

          Str = MapNames::getDpctNamespace() +
                "get_current_device().queues_wait_and_throw();";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();

          Str += "*" + ArgName + " = " + StreamName + "->single_task([=](){});";
          Str += getNL();
          Str += getIndent(IndentLoc, SM).str();
          Str += MapNames::getDpctNamespace() +
                 "get_current_device().queues_wait_and_throw()";

        } else {
          Str = "*" + ArgName + " = " + StreamName + "->single_task([=](){})";
        }

      } else {
        if (DpctGlobalInfo::useSYCLCompat()) {
          report(CE->getBeginLoc(), Diagnostics::UNSUPPORT_SYCLCOMPAT, false,
                 "cudaEventRecord");
          return;
        }
        std::string ReplaceStr;
        ReplaceStr = MapNames::getDpctNamespace() + "sync_barrier";
        emplaceTransformation(new ReplaceCalleeName(CE, std::move(ReplaceStr)));
        return;
      }
      ReplStr += Str;
    }

    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  }
}

void EventAPICallRule::handleEventRecordWithProfilingDisabled(
    const CallExpr *CE, const MatchFinder::MatchResult &Result,
    bool IsAssigned) {

  // Insert the helper variable right after the event variables
  static std::set<std::pair<const Decl *, std::string>> DeclDupFilter;
  auto &SM = DpctGlobalInfo::getSourceManager();

  const ValueDecl *MD = nullptr;
  if ((MD = getDecl(CE->getArg(0))) == nullptr)
    return;

  bool IsParmVarDecl = isa<ParmVarDecl>(MD);

  if (!IsParmVarDecl)
    report(CE->getBeginLoc(), Diagnostics::TIME_MEASUREMENT_FOUND, false);

  DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), HT_Chrono);

  std::string InsertStr;
  if (isInMacroDefinition(MD->getBeginLoc(), MD->getEndLoc())) {
    InsertStr += "\\";
  }
  InsertStr += getNL();
  InsertStr += getIndent(MD->getBeginLoc(), SM).str();
  InsertStr += "std::chrono::time_point<std::chrono::steady_clock> ";
  InsertStr += getTimePointNameForEvent(CE->getArg(0), true);
  InsertStr += ";";
  auto Pair = std::make_pair(MD, InsertStr);
  if (DeclDupFilter.find(Pair) == DeclDupFilter.end()) {
    DeclDupFilter.insert(Pair);
    if (!IsParmVarDecl)
      emplaceTransformation(new InsertAfterDecl(MD, std::move(InsertStr)));
  }

  std::ostringstream Repl;
  // Replace event recording with std::chrono timing
  if (!IsParmVarDecl) {
    Repl << getTimePointNameForEvent(CE->getArg(0), false)
        << " = std::chrono::steady_clock::now()";
  }

  const std::string Name =
      CE->getCalleeDecl()->getAsFunction()->getNameAsString();

  auto StreamArg = CE->getArg(CE->getNumArgs() - 1);
  auto StreamName = ExprAnalysis::ref(StreamArg);
  auto EventArg = CE->getArg(0);
  auto EventName = ExprAnalysis::ref(EventArg);
  bool IsDefaultStream = isDefaultStream(StreamArg);
  auto IndentLoc = CE->getBeginLoc();
  auto &Context = dpct::DpctGlobalInfo::getContext();

  if (needExtraParens(EventArg)) {
    EventName = "(" + EventName + ")";
  }

  if (needExtraParensInMemberExpr(StreamArg)) {
    StreamName = "(" + StreamName + ")";
  }

  if (IsAssigned) {
    if (!DpctGlobalInfo::useEnqueueBarrier()) {
      // ext_oneapi_submit_barrier is specified in the value of option
      // --no-dpcpp-extensions.
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      std::string StmtStr;

      if (IsDefaultStream) {
        if (isPlaceholderIdxDuplicated(CE))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);

        std::string Str = "{{NEEDREPLACEQ" + std::to_string(Index) +
                          "}}.ext_oneapi_submit_barrier()";
        StmtStr = "*" + EventName + " = " + Str;
      } else {
        std::string Str = StreamName + "->" + "ext_oneapi_submit_barrier()";
        StmtStr = "*" + EventName + " = " + Str;
      }
      StmtStr = MapNames::getCheckErrorMacroName() + "(" + StmtStr + ")";

      auto ReplWithSubmitBarrier =
          ReplaceStmt(CE, StmtStr).getReplacement(Context);
      auto ReplWithoutSubmitBarrier =
          ReplaceStmt(CE, "0").getReplacement(Context);
      DpctGlobalInfo::getInstance().insertTimeStubTypeInfo(
          ReplWithSubmitBarrier, ReplWithoutSubmitBarrier);
    }
    if (!IsParmVarDecl)
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_ZERO, false);

    auto OuterStmt = findNearestNonExprNonDeclAncestorStmt(CE);

    if (!IsParmVarDecl)
      Repl << "; ";

    if (IndentLoc.isMacroID())
      IndentLoc = SM.getExpansionLoc(IndentLoc);

    if (!IsParmVarDecl)
      Repl << getNL() << getIndent(IndentLoc, SM).str();

    auto TM = new InsertText(SM.getExpansionLoc(OuterStmt->getBeginLoc()),
                             std::move(Repl.str()));
    TM->setInsertPosition(IP_Right);
    emplaceTransformation(TM);
  } else {
    if (!DpctGlobalInfo::useEnqueueBarrier()) {
      // ext_oneapi_submit_barrier is specified in the value of option
      // --no-dpcpp-extensions.
      auto TM = new ReplaceStmt(CE, std::move(Repl.str()));
      TM->setInsertPosition(IP_Right);
      emplaceTransformation(TM);
    } else {
      std::string StrWithoutSubmitBarrier = Repl.str();
      auto ReplWithoutSB =
          ReplaceStmt(CE, StrWithoutSubmitBarrier).getReplacement(Context);
      std::string ReplStr;
      if (!IsParmVarDecl)
        ReplStr += ";";
      if (isInMacroDefinition(MD->getBeginLoc(), MD->getEndLoc())) {
        ReplStr += "\\";
      }
      if (IsDefaultStream) {
        if (isPlaceholderIdxDuplicated(CE))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
        std::string Str = "*" + EventName + " = {{NEEDREPLACEQ" +
                          std::to_string(Index) +
                          "}}.ext_oneapi_submit_barrier()";
        if (!IsParmVarDecl)
          ReplStr += getNL();
        ReplStr += getIndent(IndentLoc, SM).str();
        ReplStr += Str;
      } else {
        std::string Str = "*" + EventName + " = " + StreamName +
                          "->ext_oneapi_submit_barrier()";
        if (!IsParmVarDecl)
          ReplStr += getNL();
        ReplStr += getIndent(IndentLoc, SM).str();
        ReplStr += Str;
      }
      Repl << ReplStr;
      auto ReplWithSB = ReplaceStmt(CE, Repl.str()).getReplacement(Context);
      DpctGlobalInfo::getInstance().insertTimeStubTypeInfo(ReplWithSB,
                                                           ReplWithoutSB);
    }
  }
}

void EventAPICallRule::handleEventRecord(const CallExpr *CE,
                                         const MatchFinder::MatchResult &Result,
                                         bool IsAssigned) {
  if (DpctGlobalInfo::getEnablepProfilingFlag()) {
    // Option '--enable-profiling' is enabled
    handleEventRecordWithProfilingEnabled(CE, Result, IsAssigned);
  } else {
    // Option '--enable-profiling' is disabled
    handleEventRecordWithProfilingDisabled(CE, Result, IsAssigned);
  }
}

void EventAPICallRule::handleEventElapsedTime(bool IsAssigned) {
  if(DpctGlobalInfo::getEnablepProfilingFlag()) {
    // Option '--enable-profiling' is enabled
    auto StmtStrArg0 = getStmtSpelling(TimeElapsedCE->getArg(0));
    auto StmtStrArg1 = getStmtSpelling(TimeElapsedCE->getArg(1));
    auto StmtStrArg2 = getStmtSpelling(TimeElapsedCE->getArg(2));

    std::ostringstream Repl;
    std::string Assginee = "*(" + StmtStrArg0 + ")";
    if (auto UO = dyn_cast<UnaryOperator>(TimeElapsedCE->getArg(0))) {
      if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf)
        Assginee = getStmtSpelling(UO->getSubExpr());
    }

    auto StartTimeStr = StmtStrArg1 + "->get_profiling_info<"
                            "sycl::info::event_profiling::command_start>()";
    auto StopTimeStr =  StmtStrArg2 + "->get_profiling_info<"
                            "sycl::info::event_profiling::command_end>()";

    Repl << Assginee << " = ("
        << StopTimeStr << " - " << StartTimeStr << ") / 1000000.0f";
    if (IsAssigned) {
      std::ostringstream Temp;
      Temp << MapNames::getCheckErrorMacroName() + "(" << Repl.str() << ")";
      Repl = std::move(Temp);
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(TimeElapsedCE, std::move(Repl.str())));
  } else {
    // Option '--enable-profiling' is not enabled
    auto StmtStrArg0 = getStmtSpelling(TimeElapsedCE->getArg(0));
    auto StmtStrArg1 = getTimePointNameForEvent(TimeElapsedCE->getArg(1), false);
    auto StmtStrArg2 = getTimePointNameForEvent(TimeElapsedCE->getArg(2), false);
    std::ostringstream Repl;
    std::string Assginee = "*(" + StmtStrArg0 + ")";
    if (auto UO = dyn_cast<UnaryOperator>(TimeElapsedCE->getArg(0))) {
      if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf)
        Assginee = getStmtSpelling(UO->getSubExpr());
    }
    Repl << Assginee << " = std::chrono::duration<float, std::milli>("
        << StmtStrArg2 << " - " << StmtStrArg1 << ").count()";
    if (IsAssigned) {
      std::ostringstream Temp;
      Temp << MapNames::getCheckErrorMacroName() + "((" << Repl.str() << "))";
      Repl = std::move(Temp);
      requestFeature(HelperFeatureEnum::device_ext);
    }
    const std::string Name =
        TimeElapsedCE->getCalleeDecl()->getAsFunction()->getNameAsString();
    emplaceTransformation(new ReplaceStmt(TimeElapsedCE, std::move(Repl.str())));
    handleTimeMeasurement();
  }
}

bool EventAPICallRule::IsEventArgArraySubscriptExpr(const Expr *E) {
  E = E->IgnoreImpCasts();
  if (auto UO = dyn_cast<UnaryOperator>(E))
    return IsEventArgArraySubscriptExpr(UO->getSubExpr());
  if (auto PE = dyn_cast<ParenExpr>(E))
    return IsEventArgArraySubscriptExpr(PE->getSubExpr());
  if (dyn_cast<ArraySubscriptExpr>(E))
    return true;
  return false;
}

const Expr *EventAPICallRule::findNextRecordedEvent(const Stmt *Node,
                                                    unsigned KCallLoc) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  for (auto Iter = Node->child_begin(); Iter != Node->child_end(); ++Iter) {

    const CallExpr *Call = nullptr;
    findEventAPI(*Iter, Call, "cudaEventRecord");

    if (!Call)
      findEventAPI(*Iter, Call, "cuEventRecord");

    if (Call) {
      if (SM.getExpansionLoc(Call->getBeginLoc()).getRawEncoding() > KCallLoc)
        return Call->getArg(0);
    }
  }
  return nullptr;
}

//  The following is a typical code piece, in which three
//  locations are used to help migrate:
//
//  cudaEventRecord(start);                 // <<== RecordBeginLoc
//  ...
//  mem_calls();
//  kernel_calls();
//  mem_calls();
//  ...
//  cudaEventRecord(stop);                  // <<== RecordEndLoc
//  ...
//  sync_calls();
//  ...
//  cudaEventElapsedTime(&et, start, stop); // <<== TimeElapsedLoc
//
//  or
//
//  cudaEventCreate(&stop);               // <<== RecordBeginLoc
//  ...
//  async_mem_calls
//  kernel_calls
//  async_mem_calls
//  ...
//  cudaEventRecord(stop);
//    while (cudaEventQuery(stop) == cudaErrorNotReady) { // <<== RecordEndLoc
//                                                             /TimeElapsedLoc
//        ...
//    }
//  processAsyncJob is used to process all sync calls between
//  RecordEndLoc and RecordEndLoc, and RecordEndLoc and TimeElapsedLoc.
void EventAPICallRule::processAsyncJob(const Stmt *Node) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  RecordBeginLoc =
      SM.getExpansionLoc(RecordBegin->getBeginLoc()).getRawEncoding();
  RecordEndLoc = SM.getExpansionLoc(RecordEnd->getBeginLoc()).getRawEncoding();
  TimeElapsedLoc =
      SM.getExpansionLoc(TimeElapsedCE->getBeginLoc()).getRawEncoding();

  // Handle the kernel calls and async memory operations between start and stop
  handleTargetCalls(Node);

  if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
    for (const auto &NewEventName : Events2Wait) {
      std::ostringstream SyncStmt;
      SyncStmt
          << NewEventName << getNL()
          << getIndent(SM.getExpansionLoc(RecordEnd->getBeginLoc()), SM).str();
      auto TM = new InsertText(SM.getExpansionLoc(RecordEnd->getBeginLoc()),
                               SyncStmt.str());
      TM->setInsertPosition(IP_AlwaysLeft);
      emplaceTransformation(TM);
    }
  } else {
    for (const auto &T : Queues2Wait) {
      std::ostringstream SyncStmt;
      SyncStmt
          << std::get<0>(T) << getNL()
          << getIndent(SM.getExpansionLoc(RecordEnd->getBeginLoc()), SM).str();
      auto TM = new InsertBeforeStmt(RecordEnd, SyncStmt.str(), 0 /*PairID*/,
                                     true /*DoMacroExpansion*/);
      TM->setInsertPosition(IP_AlwaysLeft);
      emplaceTransformation(TM);
    }
  }
}

void EventAPICallRule::findThreadSyncLocation(const Stmt *Node) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  const CallExpr *Call = nullptr;
  findEventAPI(Node, Call, "cudaThreadSynchronize");

  if (Call) {
    ThreadSyncLoc = SM.getExpansionLoc(Call->getBeginLoc()).getRawEncoding();
  }
}

void EventAPICallRule::updateAsyncRangRecursive(
    const Stmt *Node, const CallExpr *AsyncCE, const std::string EventAPIName) {
  if (!Node)
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto CELoc = SM.getExpansionLoc(AsyncCE->getBeginLoc()).getRawEncoding();
  for (auto Iter = Node->child_begin(); Iter != Node->child_end(); ++Iter) {
    if (*Iter == nullptr)
      continue;
    if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None)
      findThreadSyncLocation(*Iter);

    if (SM.getExpansionLoc(Iter->getBeginLoc()).getRawEncoding() > CELoc) {
      return;
    }

    if (EventAPIName == "cudaEventRecord" || EventAPIName == "cuEventRecord") {
      const CallExpr *Call = nullptr;
      findEventAPI(*Iter, Call, EventAPIName);

      if (Call) {
        // Find the last call of Event Record on start and stop before
        // calculate the time elapsed
        auto Arg0 = getStmtSpelling(Call->getArg(0));
        if (Arg0 == getStmtSpelling(AsyncCE->getArg(1))) {
          RecordBegin = findNearestNonExprNonDeclAncestorStmt(Call);
        } else if (Arg0 == getStmtSpelling(AsyncCE->getArg(2))) {
          RecordEnd = findNearestNonExprNonDeclAncestorStmt(Call);
        }
      }

    } else if (EventAPIName == "cudaEventCreate" ||
               EventAPIName == "cuEventCreate") {

      const CallExpr *Call = nullptr;
      findEventAPI(*Iter, Call, EventAPIName);

      if (Call) {
        std::string Arg0;
        if (auto UO = dyn_cast<UnaryOperator>(Call->getArg(0))) {
          if (UO->getOpcode() == UnaryOperatorKind::UO_AddrOf) {
            Arg0 = getStmtSpelling(UO->getSubExpr());
          }
        }
        if (Arg0.empty())
          Arg0 = getStmtSpelling(Call->getArg(0));

        if (Arg0 == getStmtSpelling(AsyncCE->getArg(0)))
          RecordBegin = findNearestNonExprNonDeclAncestorStmt(Call);
      }

      // To update RecordEnd
      Call = nullptr;
      findEventAPI(*Iter, Call, "cudaEventRecord");
      if (!Call)
        findEventAPI(*Iter, Call, "cuEventRecord");

      if (Call) {
        auto Arg0 = getStmtSpelling(Call->getArg(0));
        if (Arg0 == getStmtSpelling(AsyncCE->getArg(0))) {
          RecordEnd = findNearestNonExprNonDeclAncestorStmt(Call);
        }
      }
    }

    // Recursively update range in deeper code structures
    updateAsyncRangRecursive(*Iter, AsyncCE, EventAPIName);
  }
}

//  The following is a typical code piece, in which three
//  locations are used to help migrate:
//
//  cudaEventRecord(start);                 // <<== RecordBeginLoc
//  ...
//  mem_calls();
//  kernel_calls();
//  mem_calls();
//  ...
//  cudaEventRecord(stop);                  // <<== RecordEndLoc
//  ...
//  sync_calls();
//  ...
//  cudaEventElapsedTime(&et, start, stop); // <<== TimeElapsedLoc
//
//  or
//
//  cudaEventCreate(&stop);               // <<== RecordBeginLoc
//  ...
//  async_mem_calls
//  kernel_calls
//  async_mem_calls
//  ...
//  cudaEventRecord(stop);
//    while (cudaEventQuery(stop) == cudaErrorNotReady) { // <<== RecordEndLoc
//                                                             /TimeElapsedLoc
//        ...
//    }
// \p FuncBody is the body of the function which calls function pointed by
// TimeElapsedCE \p EventAPIName is the EventAPI name (.i.e cudaEventRecord or
// cudaEventCreate) to help to locate RecordEndLoc.
void EventAPICallRule::updateAsyncRange(const Stmt *FuncBody,
                                        const std::string EventAPIName) {
  auto EventArg = TimeElapsedCE->getArg(0);
  if (IsEventArgArraySubscriptExpr(EventArg)) {
    // If the event arg is a ArraySubscriptExpr, if not async range is not
    // identified, mark all kernels in the current function to wait.
    updateAsyncRangRecursive(FuncBody, TimeElapsedCE, EventAPIName);
    if (!RecordEnd) {
      IsKernelSync = true;
      RecordBegin = *FuncBody->child_begin();
      RecordEnd = TimeElapsedCE;
    }
  } else {
    updateAsyncRangRecursive(FuncBody, TimeElapsedCE, EventAPIName);
  }
}

//  The following is a typical piece of time-measurement code, in which three
//  locations are used to help migrate:
//
//  cudaEventRecord(start);                 // <<== RecordBeginLoc
//  ...
//  mem_calls();
//  kernel_calls();
//  mem_calls();
//  ...
//  cudaEventRecord(stop);                  // <<== RecordEndLoc
//  ...
//  sync_calls();
//  ...
//  cudaEventElapsedTime(&et, start, stop); // <<== TimeElapsedLoc
void EventAPICallRule::handleTimeMeasurement() {

  auto FD = getImmediateOuterFuncDecl(TimeElapsedCE);
  if (!FD)
    return;

  const Stmt *FuncBody = nullptr;
  if (FD->isTemplateInstantiation()) {
    auto FTD = FD->getPrimaryTemplate();
    if (!FTD)
      return;
    FuncBody = FTD->getTemplatedDecl()->getBody();
  } else {
    FuncBody = FD->getBody();
  }

  if (!FuncBody)
    return;

  updateAsyncRange(FuncBody, "cudaEventRecord");
  updateAsyncRange(FuncBody, "cuEventRecord");

  if (!RecordBegin || !RecordEnd) {
    return;
  }

  // To store the range of code where time measurement takes place.
  processAsyncJob(FuncBody);

  if (!IsKernelInLoopStmt) {
    DpctGlobalInfo::getInstance().updateTimeStubTypeInfo(
        RecordBegin->getBeginLoc(), TimeElapsedCE->getEndLoc());
  }
}

// To get the redundant parent ParenExpr for \p Call to handle case like
// "(cudaEventSynchronize(stop))".
const clang::Stmt *
EventAPICallRule::getRedundantParenExpr(const CallExpr *Call) {
  auto &Context = dpct::DpctGlobalInfo::getContext();
  auto Parents = Context.getParents(*Call);
  if (Parents.size()) {
    auto &Parent = Parents[0];
    if (auto PE = Parent.get<ParenExpr>()) {
      if (auto ParentStmt = getParentStmt(PE)) {
        auto ParentStmtClass = ParentStmt->getStmtClass();
        bool Ret = ParentStmtClass == Stmt::StmtClass::IfStmtClass ||
                   ParentStmtClass == Stmt::StmtClass::WhileStmtClass ||
                   ParentStmtClass == Stmt::StmtClass::DoStmtClass ||
                   ParentStmtClass == Stmt::StmtClass::CallExprClass ||
                   ParentStmtClass == Stmt::StmtClass::ImplicitCastExprClass ||
                   ParentStmtClass == Stmt::StmtClass::BinaryOperatorClass ||
                   ParentStmtClass == Stmt::StmtClass::ForStmtClass;
        if (!Ret) {
          return PE;
        }
      }
    }
  }
  return nullptr;
}

//  cudaEventRecord(start);                 // <<== RecordBeginLoc
//  ...
//  mem_calls();
//  kernel_calls();
//  mem_calls();
//  ...
//  cudaEventRecord(stop);                  // <<== RecordEndLoc
//  ...
//  sync_calls();
//  ...
//  cudaEventElapsedTime(&et, start, stop); // <<== TimeElapsedLoc
void EventAPICallRule::handleTargetCalls(const Stmt *Node, const Stmt *Last) {
  if (!Node)
    return;
  auto &SM = DpctGlobalInfo::getSourceManager();

  for (auto It = Node->child_begin(); It != Node->child_end(); ++It) {
    if (*It == nullptr)
      continue;
    auto Loc = SM.getExpansionLoc(It->getBeginLoc()).getRawEncoding();

    // Skip statements before RecordBeginLoc or after TimeElapsedLoc
    if (Loc > RecordBeginLoc && Loc <= TimeElapsedLoc) {

      // Handle cudaEventSynchronize between RecordEndLoc and TimeElapsedLoc
      if (Loc > RecordEndLoc && Loc < TimeElapsedLoc) {

        const CallExpr *Call = nullptr;
        std::string OriginalAPIName = "";
        findEventAPI(*It, Call, "cudaEventSynchronize");
        if (Call) {
          OriginalAPIName = "cudaEventSynchronize";
        } else {
          findEventAPI(*It, Call, "cuEventSynchronize");
          if (Call)
            OriginalAPIName = "cuEventSynchronize";
        }

        if (Call) {
          if (const clang::Stmt *S = getRedundantParenExpr(Call)) {
            // To remove statement like "(cudaEventSynchronize(stop));"
            emplaceTransformation(new ReplaceStmt(S, false, true, ""));
          }

          const auto &TM = ReplaceStmt(Call, "");
          auto &Context = dpct::DpctGlobalInfo::getContext();
          auto R = TM.getReplacement(Context);
          DpctGlobalInfo::getInstance().updateEventSyncTypeInfo(R);
        }
      }

      // Now handle all statements between RecordBeginLoc and RecordEndLoc
      switch (It->getStmtClass()) {
      case Stmt::CallExprClass: {
        handleOrdinaryCalls(dyn_cast<CallExpr>(*It));
        break;
      }
      case Stmt::CUDAKernelCallExprClass: {

        if (Last && (Last->getStmtClass() == Stmt::DoStmtClass ||
                     Last->getStmtClass() == Stmt::WhileStmtClass ||
                     Last->getStmtClass() == Stmt::ForStmtClass)) {
          IsKernelInLoopStmt = true;
        }

        auto FD = getImmediateOuterFuncDecl(Node);
        if (FD)
          handleKernelCalls(FD->getBody(), dyn_cast<CUDAKernelCallExpr>(*It));
        break;
      }
      case Stmt::ExprWithCleanupsClass: {
        auto ExprS = dyn_cast<ExprWithCleanups>(*It);
        auto *SubExpr = ExprS->getSubExpr();
        if (auto *KCall = dyn_cast<CUDAKernelCallExpr>(SubExpr)) {

          if (Last && (Last->getStmtClass() == Stmt::DoStmtClass ||
                       Last->getStmtClass() == Stmt::WhileStmtClass ||
                       Last->getStmtClass() == Stmt::ForStmtClass)) {
            IsKernelInLoopStmt = true;
          }

          auto FD = getImmediateOuterFuncDecl(Node);
          if (FD)
            handleKernelCalls(FD->getBody(), KCall);
        }
        break;
      }

      default:
        break;
      }
    }

    handleTargetCalls(*It, Node);
  }
}

void EventAPICallRule::handleKernelCalls(const Stmt *Node,
                                         const CUDAKernelCallExpr *KCall) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto KCallLoc = SM.getExpansionLoc(KCall->getBeginLoc()).getRawEncoding();
  auto K = DpctGlobalInfo::getInstance().insertKernelCallExpr(KCall);
  auto EventExpr = findNextRecordedEvent(Node, KCallLoc);
  if (!EventExpr && TimeElapsedCE->getNumArgs() == 3)
    EventExpr = TimeElapsedCE->getArg(2);

  auto ArgName = ExprAnalysis::ref(EventExpr);
  // Skip statements before RecordBeginLoc or after RecordEndLoc
  if (KCallLoc < RecordBeginLoc || KCallLoc > RecordEndLoc)
    return;

  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_None) {
    bool NeedWait = false;
    // In usm none mode, if cudaThreadSynchronize apears after kernel call,
    // kernel wait is not needed.
    NeedWait = ThreadSyncLoc > KCallLoc;

    if (KCallLoc > RecordBeginLoc && !NeedWait) {
      if (IsKernelSync) {
        K->setEvent(ArgName);
        K->setSync();
      } else {
        Queues2Wait.emplace_back(MapNames::getDpctNamespace() +
                                     "get_current_device()."
                                     "queues_wait_and_throw();",
                                 nullptr);
        requestFeature(HelperFeatureEnum::device_ext);
      }
    }
  }

  if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
    if (KCallLoc > RecordBeginLoc) {
      if (!IsKernelInLoopStmt && !IsKernelSync) {
        K->setEvent(ArgName);
        Events2Wait.push_back(ArgName + "->wait();");
      } else if (IsKernelSync) {
        K->setEvent(ArgName);
        K->setSync();
        // Events2Wait.push_back("(" + ArgName + ")" + ".wait();");
      } else {
        std::string WaitQueue = MapNames::getDpctNamespace() +
                                "get_current_device()."
                                "queues_wait_and_throw();";
        Events2Wait.push_back(WaitQueue);
        requestFeature(HelperFeatureEnum::device_ext);
      }
    }
  }
}

void EventAPICallRule::handleOrdinaryCalls(const CallExpr *Call) {
  auto Callee = Call->getDirectCallee();
  if (!Callee)
    return;
  auto CalleeName = Callee->getName();
  if (CalleeName.starts_with("cudaMemcpy") && CalleeName.ends_with("Async")) {
    auto StreamArg = Call->getArg(Call->getNumArgs() - 1);
    bool IsDefaultStream = isDefaultStream(StreamArg);
    bool NeedStreamWait = false;

    if (StreamArg->IgnoreImpCasts()->getStmtClass() ==
        Stmt::ArraySubscriptExprClass)
      NeedStreamWait = true;

    if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
      // std::string EventName = getTempNameForExpr(TimeElapsedCE->getArg(2));
      std::string EventName;
      if (TimeElapsedCE->getNumArgs() == 3) {
        EventName = getTempNameForExpr(TimeElapsedCE->getArg(2));
      } else {
        EventName = getTempNameForExpr(TimeElapsedCE->getArg(0));
      }
      std::string QueueName =
          IsDefaultStream ? "q_ct1_" : getTempNameForExpr(StreamArg);
      std::string NewEventName =
          EventName + QueueName + std::to_string(++QueueCounter[QueueName]);
      Events2Wait.push_back(NewEventName + ".wait();");
      auto &SM = DpctGlobalInfo::getSourceManager();
      std::ostringstream SyncStmt;
      SyncStmt << MapNames::getClNamespace() << "event " << NewEventName << ";"
               << getNL()
               << getIndent(SM.getExpansionLoc(RecordBegin->getBeginLoc()), SM)
                      .str();
      emplaceTransformation(new InsertText(
          SM.getExpansionLoc(RecordBegin->getBeginLoc()), SyncStmt.str()));

      auto TM = new InsertBeforeStmt(Call, NewEventName + " = ");
      TM->setInsertPosition(IP_Right);
      emplaceTransformation(TM);
    } else {
      std::tuple<bool, std::string, const CallExpr *> T;
      if (IsDefaultStream && !DefaultQueueAdded) {
        DefaultQueueAdded = true;
        if (isPlaceholderIdxDuplicated(Call))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        auto &SM = DpctGlobalInfo::getSourceManager();
        std::ostringstream SyncStmt;
        SyncStmt << "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.wait();"
                 << getNL()
                 << getIndent(SM.getExpansionLoc(RecordEnd->getBeginLoc()), SM)
                        .str();
        buildTempVariableMap(Index, Call, HelperFuncType::HFT_DefaultQueue);

        emplaceTransformation(new InsertText(
            SM.getExpansionLoc(RecordEnd->getBeginLoc()), SyncStmt.str()));
      } else if (!IsDefaultStream) {
        if (NeedStreamWait) {
          Queues2Wait.emplace_back(MapNames::getDpctNamespace() +
                                       "get_current_device()."
                                       "queues_wait_and_throw();",
                                   nullptr);
          requestFeature(HelperFeatureEnum::device_ext);
        } else {
          auto ArgName = getStmtSpelling(StreamArg);
          Queues2Wait.emplace_back(ArgName + "->wait();", nullptr);
        }
      }
    }
  }
}


void ProfilingEnableOnDemandRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(callExpr(allOf(callee(functionDecl(hasAnyName(
                                   "cudaEventElapsedTime", "cudaEventRecord"))),
                               parentStmt()))
                    .bind("cudaEventElapsedTimeCall"),
                this);
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(hasName("cudaEventElapsedTime"))),
                     unless(parentStmt())))
          .bind("cudaEventElapsedTimeUsed"),
      this);
}

// When cudaEventElapsedTimeCall() is called in the source code, event profiling
// opton "--enable-profiling" is enabled to measure the execution time of a
// specific kernel or command in SYCL device.
void ProfilingEnableOnDemandRule::runRule(
    const MatchFinder::MatchResult &Result) {

  if (DpctGlobalInfo::getEnablepProfilingFlag())
    return;

  const CallExpr *CE =
      getNodeAsType<CallExpr>(Result, "cudaEventElapsedTimeCall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "cudaEventElapsedTimeUsed")))
      return;
  }

  if (!CE->getDirectCallee())
    return;

  DpctGlobalInfo::setEnablepProfilingFlag(true);
}


void StreamAPICallRule::registerMatcher(MatchFinder &MF) {
  auto streamFunctionName = [&]() {
    return hasAnyName(
        "cudaStreamCreate", "cudaStreamCreateWithFlags",
        "cudaStreamCreateWithPriority", "cudaStreamDestroy",
        "cudaStreamSynchronize", "cudaStreamGetPriority", "cudaStreamGetFlags",
        "cudaDeviceGetStreamPriorityRange", "cudaStreamAttachMemAsync",
        "cudaStreamBeginCapture", "cudaStreamEndCapture",
        "cudaStreamIsCapturing", "cudaStreamQuery", "cudaStreamWaitEvent",
        "cudaStreamAddCallback", "cuStreamCreate", "cuStreamSynchronize",
        "cuStreamWaitEvent", "cuStreamDestroy_v2", "cuStreamAttachMemAsync",
        "cuStreamAddCallback", "cuStreamQuery");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(streamFunctionName())), parentStmt()))
          .bind("streamAPICall"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(streamFunctionName())),
                               unless(parentStmt())))
                    .bind("streamAPICallUsed"),
                this);
}

std::string getNewQueue(int Index) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  printPartialArguments(OS << "{{NEEDREPLACED" << std::to_string(Index)
                           << "}}.create_queue(",
                        AsyncHandler ? 1 : 0, "true")
      << ")";
  return OS.str();
}

void StreamAPICallRule::runRule(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "streamAPICall");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "streamAPICallUsed")))
      return;
    IsAssigned = true;
  }

  if (!CE->getDirectCallee())
    return;
  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();

  if (!CallExprRewriterFactoryBase::RewriterMap)
    return;
  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(FuncName);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
  if (FuncName == "cudaStreamIsCapturing") {
    if (!DpctGlobalInfo::useExtGraph()) {
      report(CE->getBeginLoc(), Diagnostics::TRY_EXPERIMENTAL_FEATURE, false,
             "cudaStreamIsCapturing", "--use-experimental-features=graph");
      return;
    }
    std::string ReplStr;
    std::string StreamName;
    auto StmtStr0 = getStmtSpelling(CE->getArg(1));
    std::ostringstream OS;
    printDerefOp(OS, CE->getArg(1));
    ReplStr = OS.str() + " = ";
    if (isDefaultStream(CE->getArg(0))) {
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
      StreamName = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.";
      ReplStr += StreamName + "ext_oneapi_get_state()";
    } else {
      auto StreamArg = CE->getArg(0);
      StreamName = getStmtSpelling(StreamArg);
      if (needExtraParensInMemberExpr(StreamArg)) {
        StreamName = "(" + StreamName + ")";
      }
      ReplStr += StreamName + "->" + "ext_oneapi_get_state()";
    }
    if (IsAssigned) {
      ReplStr = MapNames::getCheckErrorMacroName() + "((" + ReplStr + "))";
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
    return;
  }

  if (FuncName == "cudaStreamCreate" || FuncName == "cuStreamCreate" ||
      FuncName == "cudaStreamCreateWithFlags" ||
      FuncName == "cudaStreamCreateWithPriority") {
    std::string ReplStr;
    auto StmtStr0 = getStmtSpelling(CE->getArg(0));
    // TODO: simplify expression
    if (StmtStr0[0] == '&')
      ReplStr = StmtStr0.substr(1);
    else
      ReplStr = "*(" + StmtStr0 + ")";

    if (DpctGlobalInfo::useNoQueueDevice()) {
      // Now the UsmLevel must not be UL_None here.
      ReplStr += " = new " + MapNames::getClNamespace() + "queue(" +
                DpctGlobalInfo::getGlobalDeviceName() + ", " +
                MapNames::getClNamespace() + "property_list{" +
                MapNames::getClNamespace() + "property::queue::in_order()";
      if (DpctGlobalInfo::getEnablepProfilingFlag()) {
        ReplStr += ", " + MapNames::getClNamespace() +
                   "property::queue::enable_profiling()";
      }
      ReplStr += "})";
    } else {
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_CurrentDevice);
      ReplStr += " = " + getNewQueue(Index);
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (IsAssigned) {
      ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
    }
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
    if (FuncName == "cudaStreamCreateWithFlags" ||
        FuncName == "cudaStreamCreateWithPriority") {
      report(CE->getBeginLoc(), Diagnostics::QUEUE_CREATED_IGNORING_OPTIONS,
             false);
    }
  } else if (FuncName == "cudaStreamDestroy") {
    auto StmtStr0 = getStmtSpelling(CE->getArg(0));
    if (isPlaceholderIdxDuplicated(CE))
      return;
    int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
    buildTempVariableMap(Index, CE, HelperFuncType::HFT_CurrentDevice);
    auto ReplStr = "{{NEEDREPLACED" + std::to_string(Index) +
                   "}}.destroy_queue(" + StmtStr0 + ")";
    requestFeature(HelperFeatureEnum::device_ext);
    if (IsAssigned) {
      ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
    }
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaStreamSynchronize" ||
             FuncName == "cuStreamSynchronize") {
    auto StmtStr = getStmtSpelling(CE->getArg(0));
    std::string ReplStr;
    if (StmtStr == "0" || StmtStr == "cudaStreamDefault" ||
        StmtStr == "cudaStreamPerThread" || StmtStr == "cudaStreamLegacy") {
      if (isPlaceholderIdxDuplicated(CE))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
      ReplStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.";
    } else {
      ReplStr = StmtStr + "->";
    }
    ReplStr += "wait()";
    const std::string Name =
        CE->getCalleeDecl()->getAsFunction()->getNameAsString();
    if (IsAssigned) {
      ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaStreamGetFlags" ||
             FuncName == "cudaStreamGetPriority") {
    report(CE->getBeginLoc(), Diagnostics::UNSUPPORTED_FEATURE_IN_SYCL, false,
           "flag and priority options", "are", "queues");
    auto StmtStr1 = getStmtSpelling(CE->getArg(1));
    std::string ReplStr{"*("};
    ReplStr += StmtStr1;
    ReplStr += ") = 0";
    if (IsAssigned) {
      ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    const std::string Name =
        CE->getCalleeDecl()->getAsFunction()->getNameAsString();
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
  } else if (FuncName == "cudaStreamAttachMemAsync" ||
             FuncName == "cudaStreamQuery" || FuncName == "cuStreamQuery" ||
             FuncName == "cudaDeviceGetStreamPriorityRange") {
    // if extension feature sycl_ext_oneapi_queue_empty is used, member
    // functions "ext_oneapi_empty" in SYCL queue is used to map
    // cudaStreamQuery or cuStreamQuery.
    if ((FuncName == "cudaStreamQuery" || FuncName == "cuStreamQuery") &&
        DpctGlobalInfo::useQueueEmpty()) {
      auto StreamArg = CE->getArg(0);
      bool IsDefaultStream = isDefaultStream(StreamArg);
      std::string StreamName;
      std::string ReplStr;
      if (IsDefaultStream) {
        if (isPlaceholderIdxDuplicated(CE))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);
        StreamName = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.";
        ReplStr = StreamName + "ext_oneapi_empty()";
      } else {
        StreamName = getStmtSpelling(StreamArg);
        if (needExtraParensInMemberExpr(StreamArg)) {
          StreamName = "(" + StreamName + ")";
        }
        ReplStr = StreamName + "->" + "ext_oneapi_empty()";
      }
      if (IsAssigned) {
        ReplStr = MapNames::getCheckErrorMacroName() + "((" + ReplStr + "))";
      }
      emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
      return;
    }

    if (auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
        Msg != MapNames::RemovedAPIWarningMessage.end()) {
      if (IsAssigned) {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceStmt(CE, "0"));
      } else {
        report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
               MapNames::ITFName.at(FuncName), Msg->second);
        emplaceTransformation(new ReplaceStmt(CE, ""));
      }
    }
  } else if (FuncName == "cudaStreamWaitEvent" ||
             FuncName == "cuStreamWaitEvent") {
    std::string ReplStr;
    ExprAnalysis EA(CE->getArg(1));
    std::string StmtStr1 = EA.getReplacedString();
    if (!DpctGlobalInfo::useEnqueueBarrier()) {
      // ext_oneapi_submit_barrier is specified in the value of option
      // --no-dpcpp-extensions.
      ReplStr = StmtStr1 + "->wait()";
    } else {
      StmtStr1 = "*" + StmtStr1;
      auto StreamArg = CE->getArg(0);
      bool IsDefaultStream = isDefaultStream(StreamArg);
      std::string StmtStr0;
      if (IsDefaultStream) {
        if (isPlaceholderIdxDuplicated(CE))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, CE, HelperFuncType::HFT_DefaultQueue);

        StmtStr0 = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.";
      } else {
        StmtStr0 = getStmtSpelling(CE->getArg(0)) + "->";
      }
      ReplStr = StmtStr0 + "ext_oneapi_submit_barrier({" +
                StmtStr1 + "})";
    }
    if (IsAssigned) {
      ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "cudaStreamAddCallback") {
    auto StmtStr0 = getStmtSpelling(CE->getArg(0));
    auto StmtStr1 = getStmtSpelling(CE->getArg(1));
    auto StmtStr2 = getStmtSpelling(CE->getArg(2));
    std::string ReplStr{"std::async([&]() { "};
    ReplStr += StmtStr0;
    ReplStr += "->wait(); ";
    ReplStr += StmtStr1;
    ReplStr += "(";
    ReplStr += StmtStr0;
    ReplStr += ", 0, ";
    ReplStr += StmtStr2;
    ReplStr += "); ";
    ReplStr += "})";
    if (IsAssigned) {
      ReplStr = MapNames::getCheckErrorMacroName() + "(" + ReplStr + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, ReplStr));
    DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), HT_Future);
  } else {
    llvm::dbgs() << "[" << getName()
                 << "] Unexpected function name: " << FuncName;
    return;
  }
}

void KernelCallRefRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(declRefExpr(allOf(to(functionDecl(hasAttr(attr::CUDAGlobal))),
                                  unless(hasAncestor(cudaKernelCallExpr()))))
                    .bind("kernelRef"),
                this);
  MF.addMatcher(unresolvedLookupExpr(unless(hasAncestor(cudaKernelCallExpr())))
                    .bind("unresolvedRef"),
                this);
}

std::string KernelCallRefRule::getTypeRepl(const Expr *E) {
  std::string TypeRef;
  if (auto BO = DpctGlobalInfo::findParent<BinaryOperator, Expr,
                                           ImplicitCastExpr, ParenExpr>(E)) {
    TypeRef = "decltype(" + ExprAnalysis::ref(BO->getLHS()) + ")";
  } else if (auto VD =
                 DpctGlobalInfo::findParent<VarDecl, Expr, ImplicitCastExpr,
                                            ParenExpr>(E)) {
    TypeRef = "decltype(" + VD->getNameAsString() + ")";
  } else if (auto RS =
                 DpctGlobalInfo::findParent<ReturnStmt, Expr, ImplicitCastExpr,
                                            ParenExpr>(E)) {
    auto FD = DpctGlobalInfo::findAncestor<FunctionDecl>(RS);
    TypeRef =
        getStmtSpelling(FD->getReturnTypeSourceRange(), FD->getSourceRange());
  } else if (auto CE =
                 DpctGlobalInfo::findParent<CallExpr, Expr, ImplicitCastExpr,
                                            ParenExpr>(E)) {
    size_t N = 0;
    for (auto Arg : CE->arguments()) {
      if (Arg == E) {
        break;
      }
      N++;
    }
    TypeRef = "typename " + MapNames::getDpctNamespace() +
              "nth_argument_type<decltype(" +
              ExprAnalysis::ref(CE->getCallee()) + "), " + std::to_string(N) +
              ">::type";
  }
  if (!TypeRef.empty()) {
    return "<" + TypeRef + ">";
  }
  return TypeRef;
}

template <typename T>
void KernelCallRefRule::insertWrapperPostfix(const T *Node,
                                             std::string &&TypeRepl,
                                             bool isInsertWrapperRegister) {
  auto NLoc = DpctGlobalInfo::getSourceManager().getSpellingLoc(
      Node->getNameInfo().getBeginLoc());
  emplaceTransformation(new InsertText(
      NLoc.getLocWithOffset(Node->getNameInfo().getAsString().length()),
      "_wrapper"));

  if (!isInsertWrapperRegister) {
    return;
  }
  const Expr *E = Node;
  if (auto UO = DpctGlobalInfo::findParent<UnaryOperator, T, ImplicitCastExpr,
                                           ParenExpr>(Node)) {
    if (UO->getOpcode() == UO_AddrOf) {
      E = UO;
    }
  } else if (auto COC = DpctGlobalInfo::findParent<CXXOperatorCallExpr, T,
                                                   ImplicitCastExpr, ParenExpr>(
                 Node)) {
    if (COC->getOperator() == clang::OO_Amp) {
      E = COC;
    }
  }
  emplaceTransformation(new InsertBeforeStmt(
      E, MapNames::getDpctNamespace() + "wrapper_register" + TypeRepl + "("));
  emplaceTransformation(new InsertAfterStmt(E, ").get()"));
}

void KernelCallRefRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto DRE = getAssistNodeAsType<DeclRefExpr>(Result, "kernelRef")) {
    if (auto ParentCE = DpctGlobalInfo::findAncestor<CallExpr>(DRE)) {
      if (auto Callee = ParentCE->getDirectCallee()) {
        if (dpct::DpctGlobalInfo::isInCudaPath(Callee->getBeginLoc())) {
          return;
        }
      }
    }
    const FunctionDecl *FD = dyn_cast<FunctionDecl>(DRE->getDecl());
    bool IsTemplateRelated = false;
    size_t TemplateParamNum = 0;
    if (FD) {
      if (FD->getTemplatedKind() !=
          FunctionDecl::TemplatedKind::TK_NonTemplate) {
        IsTemplateRelated = true;
      }
      if (auto FTD = FD->getPrimaryTemplate()) {
        if (auto TP = FTD->getTemplateParameters())
          TemplateParamNum = TP->size();
      }
      if (auto DFI = DeviceFunctionDecl::LinkRedecls(FD)) {
        DFI->collectInfoForWrapper(FD);
      }
    }
    if (auto *OuterFD = DpctGlobalInfo::findAncestor<FunctionDecl>(DRE)) {
      if ((OuterFD->getTemplatedKind() ==
           FunctionDecl::TemplatedKind::TK_NonTemplate) ||
          (OuterFD->getTemplatedKind() ==
           FunctionDecl::TemplatedKind::TK_FunctionTemplate)) {
        std::string TypeRepl;
        if (DpctGlobalInfo::isCVersionCUDALaunchUsed()) {
          if ((IsTemplateRelated &&
               (!DRE->hasExplicitTemplateArgs() ||
                (DRE->getNumTemplateArgs() <= TemplateParamNum))) ||
              DRE->hadMultipleCandidates()) {
            TypeRepl = getTypeRepl(DRE);
          }
        }
        insertWrapperPostfix<DeclRefExpr>(
            DRE, std::move(TypeRepl),
            DpctGlobalInfo::isCVersionCUDALaunchUsed());
      }
    }
  }
  if (auto ULE =
          getAssistNodeAsType<UnresolvedLookupExpr>(Result, "unresolvedRef")) {
    if (!DpctGlobalInfo::isCVersionCUDALaunchUsed()) {
      return;
    }
    bool KernelRefFound = false;
    for (auto *D : ULE->decls()) {
      const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
      if (!FD) {
        if (const FunctionTemplateDecl *FTD =
                dyn_cast<FunctionTemplateDecl>(D)) {
          FD = FTD->getTemplatedDecl();
        }
      }
      if (FD && FD->hasAttr<CUDAGlobalAttr>()) {
        KernelRefFound = true;
        break;
      }
    }
    if (!KernelRefFound) {
      return;
    }
    if (auto ParentCE = dpct::DpctGlobalInfo::findAncestor<CallExpr>(ULE)) {
      if (auto Callee = ParentCE->getDirectCallee()) {
        if (dpct::DpctGlobalInfo::isInCudaPath(Callee->getBeginLoc())) {
          return;
        }
      } else if (auto PULE =
                     dyn_cast<UnresolvedLookupExpr>(ParentCE->getCallee())) {
        for (auto *D : PULE->decls()) {
          if (dpct::DpctGlobalInfo::isInCudaPath(D->getBeginLoc())) {
            return;
          }
        }
      }
    }
    insertWrapperPostfix<UnresolvedLookupExpr>(ULE, getTypeRepl(ULE), true);
  }
}

// kernel call information collection
void KernelCallRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      cudaKernelCallExpr(hasAncestor(functionDecl().bind("callContext")))
          .bind("kernelCall"),
      this);

  auto launchAPIName = [&]() {
    return hasAnyName("cudaLaunchKernel", "cudaLaunchCooperativeKernel");
  };
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(launchAPIName())), parentStmt()))
          .bind("launch"),
      this);
  MF.addMatcher(callExpr(allOf(callee(functionDecl(launchAPIName())),
                               unless(parentStmt())))
                    .bind("launchUsed"),
                this);
}

void KernelCallRule::instrumentKernelLogsForCodePin(const CUDAKernelCallExpr *KCall,
                                    SourceLocation &EpilogLocation) {
  const auto &SM = DpctGlobalInfo::getSourceManager();
  auto KCallSpellingRange = getTheLastCompleteImmediateRange(
      KCall->getBeginLoc(), KCall->getEndLoc());
if (CodePinInstrumentation.find(KCallSpellingRange.first) !=
      CodePinInstrumentation.end()) 
      return ;
  llvm::SmallString<512> RelativePath;

  std::string StreamStr = "0";
  int Index = getPlaceholderIdx(KCall);
  if (Index == 0) {
    Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
  }
  std::string QueueStr = "&{{NEEDREPLACEQ" + std::to_string(Index) + "}}";
  if (auto *Config = KCall->getConfig()) {
    if (Config->getNumArgs() > 3) {
      auto StreamStrSpell = getStmtSpelling(Config->getArg(3));
      if (!StreamStrSpell.empty()) {
        StreamStr = StreamStrSpell;
      } else if (!isDefaultStream(Config->getArg(3))) {
        QueueStr = StreamStrSpell;
      }
    }
  }
  std::string KernelName =
      KCall->getCalleeDecl()->getAsFunction()->getNameAsString();

  auto InstrumentKernel = [&](std::string StreamStr, HeaderType HT,
                              dpct::ReplacementType CodePinType) {
    std::string CodePinKernelArgsString = "(\"";
    CodePinKernelArgsString += KernelName + ":" +
                               llvm::sys::path::convert_to_slash(
                                   KCallSpellingRange.first.printToString(SM)) +
                               "\", ";
    CodePinKernelArgsString += StreamStr;

    buildTempVariableMap(Index, KCall, HelperFuncType::HFT_DefaultQueue);

    for (auto *Arg : KCall->arguments()) {
      if (const auto *DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts())) {
        if (DRE->isLValue()) {
          CodePinKernelArgsString += ", ";
          std::string VarNameStr =
              "\"" + DRE->getNameInfo().getAsString() + "\"";
          CodePinKernelArgsString += VarNameStr + ", ";
          CodePinKernelArgsString += getStmtSpelling(Arg);
        }
      }
    }
    CodePinKernelArgsString += ");" + std::string(getNL());
    emplaceTransformation(new InsertText(
        KCallSpellingRange.first,
        "dpctexp::codepin::gen_prolog_API_CP" + CodePinKernelArgsString, 0,
        CodePinType));

    emplaceTransformation(new InsertText(
        EpilogLocation,
        "dpctexp::codepin::gen_epilog_API_CP" + CodePinKernelArgsString, 0,
        CodePinType));

    CodePinInstrumentation.insert(KCallSpellingRange.first);
    DpctGlobalInfo::getInstance().insertHeader(KCall->getBeginLoc(), HT,
                                               CodePinType);
  };
  InstrumentKernel(StreamStr, HT_DPCT_CodePin_CUDA, RT_CUDAWithCodePin);
  InstrumentKernel(QueueStr, HT_DPCT_CodePin_SYCL, RT_ForSYCLMigration);
}

void KernelCallRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto KCall =
          getAssistNodeAsType<CUDAKernelCallExpr>(Result, "kernelCall")) {
    auto FD = getAssistNodeAsType<FunctionDecl>(Result, "callContext");
    if (!FD)
      return;
    if (FD->hasAttr<CUDAGlobalAttr>()) {
      report(KCall->getBeginLoc(), Diagnostics::NOT_SUPPORT_DYN_PARALLEL,
             false);
      return;
    }
    bool IsDirectCall = true;
    if (!KCall->getDirectCallee() &&
        !dyn_cast<UnresolvedLookupExpr>(KCall->getCallee())) {
      IsDirectCall = false;
      std::string ReplStr;
      llvm::raw_string_ostream OS(ReplStr);
      OS << MapNames::getDpctNamespace() + "kernel_launcher::launch("
         << ExprAnalysis::ref(KCall->getCallee());
      if (const CallExpr *Configs = KCall->getConfig()) {
        size_t ConfigArgsNum = Configs->getNumArgs();
        for (size_t i = 0; i < ConfigArgsNum; i++) {
          if (auto Config = Configs->getArg(i)) {
            if (i == 0 || i == 1) {
              OS << ", " << ExprAnalysis::ref(Config);
            } else if (i == 2 || i == 3) {
              if (Config->isDefaultArgument()) {
                OS << ", 0";
              } else {
                OS << ", " << ExprAnalysis::ref(Config);
              }
            }
          }
        }
        size_t ArgsNum = KCall->getNumArgs();
        for (size_t i = 0; i < ArgsNum; i++) {
          if (auto Arg = KCall->getArg(i)) {
            if (!Arg->isDefaultArgument()) {
              OS << ", " << ExprAnalysis::ref(Arg);
            } else {
              break;
            }
          }
        }
        OS << ")";
        emplaceTransformation(new ReplaceStmt(KCall, OS.str()));
      }
    }

    const auto &SM = (*Result.Context).getSourceManager();

    if (SM.isMacroArgExpansion(KCall->getCallee()->getBeginLoc())) {
      // Report warning message
      report(KCall->getBeginLoc(), Diagnostics::KERNEL_CALLEE_MACRO_ARG, false);
    }
    if (IsDirectCall) {
      // Remove KCall in the original location
      auto KCallSpellingRange = getTheLastCompleteImmediateRange(
          KCall->getBeginLoc(), KCall->getEndLoc());
      auto KCallLen = SM.getCharacterData(KCallSpellingRange.second) -
                      SM.getCharacterData(KCallSpellingRange.first) +
                      Lexer::MeasureTokenLength(KCallSpellingRange.second, SM,
                                                Result.Context->getLangOpts());
      emplaceTransformation(
          new ReplaceText(KCallSpellingRange.first, KCallLen, ""));
    }
    auto EpilogLocation = findAndRemoveTrailingSemicolon(
        KCall, Result, IsDirectCall ? true : false);
    if (DpctGlobalInfo::isCodePinEnabled()) {
      instrumentKernelLogsForCodePin(KCall, EpilogLocation);
    }
    bool Flag = true;
    unsigned int IndentLen = calculateIndentWidth(
        KCall, SM.getExpansionLoc(KCall->getBeginLoc()), Flag);
    if (Flag)
      DpctGlobalInfo::insertKCIndentWidth(IndentLen);

    for (const Expr *Arg : KCall->arguments()) {
      if (!isDeviceCopyable(Arg->getType(), this)) {
        report(KCall->getBeginLoc(),
               Diagnostics::NOT_DEVICE_COPYABLE_ADD_SPECIALIZATION, true,
               DpctGlobalInfo::getOriginalTypeName(Arg->getType()));
      }
    }

    // Add kernel call to map,
    // will do code generation in Global.buildReplacements();
    if (IsDirectCall && !FD->isTemplateInstantiation()) {
      DpctGlobalInfo::getInstance().insertKernelCallExpr(KCall);
    }
    const CallExpr *Config = KCall->getConfig();
    if (IsDirectCall && Config) {
      if (Config->getNumArgs() > 2) {
        const Expr *SharedMemSize = Config->getArg(2);
        if (containSizeOfType(SharedMemSize)) {
          auto KCallInfo =
              DpctGlobalInfo::getInstance().insertKernelCallExpr(KCall);
          KCallInfo->setEmitSizeofWarningFlag(true);
        } else {
          const Expr *ExprContainSizeofType = nullptr;
          if (checkIfContainSizeofTypeRecursively(SharedMemSize,
                                                  ExprContainSizeofType)) {
            if (ExprContainSizeofType) {
              report(ExprContainSizeofType->getBeginLoc(),
                     Diagnostics::SIZEOF_WARNING, false, "local memory",
                     "Check that the allocated memory size in the migrated "
                     "code is correct");
            }
          }
        }
      }
    }

    // Filter out compiler generated methods
    if (const CXXMethodDecl *CXXMDecl = dyn_cast<CXXMethodDecl>(FD)) {
      if (!CXXMDecl->isUserProvided()) {
        return;
      }
    }

    auto BodySLoc = FD->getBody()->getSourceRange().getBegin().getRawEncoding();
    if (Insertions.find(BodySLoc) != Insertions.end())
      return;

    Insertions.insert(BodySLoc);
  } else {
    bool IsAssigned = false;
    const auto *LaunchKernelCall = getNodeAsType<CallExpr>(Result, "launch");
    if (!LaunchKernelCall) {
      LaunchKernelCall = getNodeAsType<CallExpr>(Result, "launchUsed");
      IsAssigned = true;
    }
    if (!LaunchKernelCall)
      return;
    const Expr *CalleeDRE = LaunchKernelCall->getArg(0);
    bool IsFuncTypeErased = true;
    auto QT = CalleeDRE->getType();

    if (QT->isPointerType()) {
      QT = QT->getPointeeType();
    }
    if (QT->isFunctionType()) {
      IsFuncTypeErased = false;
    }

    if (!getAddressedRef(CalleeDRE)) {
      if (IsFuncTypeErased) {
        DpctGlobalInfo::setCVersionCUDALaunchUsed();
      }
      std::string ReplStr;
      llvm::raw_string_ostream OS(ReplStr);
      if (IsAssigned) {
        OS << MapNames::getCheckErrorMacroName() << "(";
      }
      OS << MapNames::getDpctNamespace() << "kernel_launcher::launch(";
      size_t ArgsNum = LaunchKernelCall->getNumArgs();
      for (size_t i = 0; i < ArgsNum; i++) {
        if (auto Arg = LaunchKernelCall->getArg(i)) {
          if (i == 0) {
            if (auto E = getAddressedRef(CalleeDRE, false, nullptr)) {
              OS << ExprAnalysis::ref(E);
            } else {
              OS << ExprAnalysis::ref(Arg);
            }
          } else {
            OS << ", " << ExprAnalysis::ref(Arg);
          }
        }
      }
      OS << ")";
      if (IsAssigned) {
        OS << ")";
      }
      emplaceTransformation(new ReplaceStmt(LaunchKernelCall, OS.str()));
      return;
    }

    if (!IsAssigned)
      findAndRemoveTrailingSemicolon(LaunchKernelCall, Result);
    if (DpctGlobalInfo::getInstance().buildLaunchKernelInfo(LaunchKernelCall,
                                                            IsAssigned)) {
      emplaceTransformation(new ReplaceStmt(LaunchKernelCall, true, false, ""));
    }
  }
}

// Find and remove the semicolon after the kernel call
SourceLocation KernelCallRule::findAndRemoveTrailingSemicolon(
    const CallExpr *KCall, const ast_matchers::MatchFinder::MatchResult &Result,
    bool Remove) {
  const auto &SM = (*Result.Context).getSourceManager();
  auto KELoc =
      getTheLastCompleteImmediateRange(KCall->getBeginLoc(), KCall->getEndLoc())
          .second;
  auto Tok = Lexer::findNextToken(KELoc, SM, LangOptions()).value();
  if (Tok.is(tok::TokenKind::semi)) {
    if (Remove) {
      emplaceTransformation(new ReplaceToken(Tok.getLocation(), ""));
    }
    return Lexer::findNextToken(Tok.getLocation(), SM, LangOptions())
        .value()
        .getLocation();
  }
  return Tok.getLocation();
}

bool isRecursiveDeviceFuncDecl(const FunctionDecl* FD) {
  // Build call graph for FunctionDecl and look for cycles in call graph.
  // Emit the warning message when the recursive call exists in kernel function.
  if (!FD) return false;
  CallGraph CG;
  CG.addToCallGraph(const_cast<FunctionDecl *>(FD));
  bool FDIsRecursive = false;
  for (llvm::scc_iterator<CallGraph *> SCCI = llvm::scc_begin(&CG),
                              SCCE = llvm::scc_end(&CG);
                              SCCI != SCCE; ++SCCI) {
    if (SCCI.hasCycle()) FDIsRecursive = true;
  }
  return FDIsRecursive;
}

bool isRecursiveDeviceCallExpr(const CallExpr* CE) {
  if (isRecursiveDeviceFuncDecl(CE->getDirectCallee()))
    return true;
  return false;
}

static void checkCallGroupFunctionInControlFlow(FunctionDecl *FD) {
  GroupFunctionCallInControlFlowAnalyzer A(DpctGlobalInfo::getContext());
  (void)A.checkCallGroupFunctionInControlFlow(FD);
}

// __device__ function call information collection
void DeviceFunctionDeclRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto DeviceFunctionMatcher =
      functionDecl(anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))
          .bind("funcDecl");

  MF.addMatcher(callExpr(hasAncestor(DeviceFunctionMatcher)).bind("callExpr"),
                this);
  MF.addMatcher(
      cxxConstructExpr(hasAncestor(DeviceFunctionMatcher)).bind("CtorExpr"),
      this);

  MF.addMatcher(DeviceFunctionMatcher, this);

  MF.addMatcher(callExpr(hasAncestor(DeviceFunctionMatcher),
                         callee(functionDecl(hasName("printf"))))
                    .bind("PrintfExpr"),
                this);

  MF.addMatcher(varDecl(hasAncestor(DeviceFunctionMatcher)).bind("varGrid"),
                this);

  MF.addMatcher(
      functionDecl(anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))
          .bind("deviceFuncDecl"),
      this);

  MF.addMatcher(
      cxxNewExpr(hasAncestor(DeviceFunctionMatcher)).bind("CxxNewExpr"), this);
  MF.addMatcher(
      cxxDeleteExpr(hasAncestor(DeviceFunctionMatcher)).bind("CxxDeleteExpr"),
      this);
  MF.addMatcher(callExpr(hasAncestor(DeviceFunctionMatcher),
                         callee(functionDecl(hasAnyName(
                             "malloc", "free", "delete", "__builtin_alloca"))))
                    .bind("MemoryManipulation"),
                this);

  MF.addMatcher(typeLoc(hasAncestor(DeviceFunctionMatcher),
                        loc(qualType(hasDeclaration(namedDecl(hasAnyName(
                            "__half", "half", "__half2", "half2"))))))
                    .bind("fp16"),
                this);

  MF.addMatcher(
      typeLoc(hasAncestor(DeviceFunctionMatcher), loc(asString("double")))
          .bind("fp64"),
      this);
}

void DeviceFunctionDeclRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {

  if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "deviceFuncDecl")) {
    if (FD->isTemplateInstantiation())
      return;

    // We need skip lambda in host code, but cannot skip lambda in device code.
    if (const FunctionDecl *OuterMostFD = findTheOuterMostFunctionDecl(FD);
        OuterMostFD && (!OuterMostFD->hasAttr<CUDADeviceAttr>() &&
                        !OuterMostFD->hasAttr<CUDAGlobalAttr>()))
      return;

    const auto &FTL = FD->getFunctionTypeLoc();
    if (!FTL)
      return;

    auto BeginLoc = FD->getBeginLoc();
    if (auto DFT = FD->getDescribedFunctionTemplate())
      BeginLoc = DFT->getBeginLoc();
    auto EndLoc = FTL.getRParenLoc();

    auto BeginLocInfo = DpctGlobalInfo::getLocInfo(BeginLoc);
    auto EndLocInfo = DpctGlobalInfo::getLocInfo(EndLoc);
    auto FileInfo =
        DpctGlobalInfo::getInstance().insertFile(BeginLocInfo.first);
    auto &Map = FileInfo->getFuncDeclRangeMap();
    auto Name = FD->getNameAsString();
    auto Iter = Map.find(Name);
    if (Iter == Map.end()) {
      std::vector<std::pair<unsigned int, unsigned int>> Vec;
      Vec.push_back(std::make_pair(BeginLocInfo.second, EndLocInfo.second));
      Map[Name] = Vec;
    } else {
      Iter->second.push_back(
          std::make_pair(BeginLocInfo.second, EndLocInfo.second));
    }
  }

  std::shared_ptr<DeviceFunctionInfo> FuncInfo;
  auto FD = getAssistNodeAsType<FunctionDecl>(Result, "funcDecl");
  if (!FD || (FD->hasAttr<CUDADeviceAttr>() && FD->hasAttr<CUDAHostAttr>() &&
              DpctGlobalInfo::getRunRound() == 1))
    return;

  // We need skip lambda in host code, but cannot skip lambda in device code.
  if (const FunctionDecl *OuterMostFD = findTheOuterMostFunctionDecl(FD);
      OuterMostFD && (!OuterMostFD->hasAttr<CUDADeviceAttr>() &&
                      !OuterMostFD->hasAttr<CUDAGlobalAttr>()))
    return;

  if (FD->isVariadic()) {
    report(FD->getBeginLoc(), Warnings::DEVICE_VARIADIC_FUNCTION, false);
  }

  if (FD->isVirtualAsWritten()) {
    report(FD->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION,
                              false, "Virtual functions");
  }

  if(isRecursiveDeviceFuncDecl(FD))
    report(FD->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION,
                            false, "Recursive functions");

  FuncInfo = DeviceFunctionDecl::LinkRedecls(FD);
  if (!FuncInfo)
    return;

  if (FD->isOverloadedOperator()) {
    FuncInfo->setOverloadedOperatorKind(FD->getOverloadedOperator());
  }

  if (FD->doesThisDeclarationHaveABody()) {
    size_t ParamCounter = 0;
    for (auto &Param : FD->parameters()) {
      FuncInfo->setParameterReferencedStatus(ParamCounter,
                                             Param->isReferenced());
      ParamCounter++;
    }
    auto &SM = DpctGlobalInfo::getSourceManager();
    auto &CTX = DpctGlobalInfo::getContext();
    size_t LocalVariableSize = 0;
    for(auto D : FD->decls()){
      if(auto VD = dyn_cast_or_null<VarDecl>(D)) {
        if(VD->hasAttr<CUDASharedAttr>() || !VD->isLocalVarDecl() || isCubVar(VD)){
          continue;
        }
        auto Size = CTX.getTypeSizeInCharsIfKnown(VD->getType());
        if(Size.has_value()) {
          LocalVariableSize += Size.value().getQuantity();
        }
      }
    }
    // For Xe-LP architecture, if the sub-group size is 32, then each work-item
    // can use 128 * 32 Byte / 32 = 128 Byte registers.
    if(LocalVariableSize > 128) {
      report(SM.getExpansionLoc(FD->getBeginLoc()), Warnings::REGISTER_USAGE,
             false, FD->getDeclName().getAsString());
    }
  }
  if (isLambda(FD) && !FuncInfo->isLambda()) {
    FuncInfo->setLambda();
  }
  if (FD->hasAttr<CUDAGlobalAttr>()) {
    FuncInfo->setKernel();
  }
  if (FD->isInlined()) {
    FuncInfo->setInlined();
  }
  if (auto CE = getAssistNodeAsType<CallExpr>(Result, "callExpr")) {
    if (auto COCE = dyn_cast<CXXOperatorCallExpr>(CE)) {
      if ((COCE->getOperator() != OverloadedOperatorKind::OO_None) &&
          (COCE->getOperator() != OverloadedOperatorKind::OO_Call)) {
        return;
      }
    }
    if (CE->getDirectCallee()) {
      if (CE->getDirectCallee()->isVirtualAsWritten())
        report(CE->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION,
                        false, "Virtual functions");
    }

    if (isRecursiveDeviceCallExpr(CE))
      report(CE->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION,
                                false, "Recursive functions");
    auto CallInfo = FuncInfo->addCallee(CE);
    checkCallGroupFunctionInControlFlow(const_cast<FunctionDecl *>(FD));
    if (CallInfo->hasSideEffects())
      report(CE->getBeginLoc(), Diagnostics::CALL_GROUP_FUNC_IN_COND, false);
  } else if (CE = getAssistNodeAsType<CallExpr>(Result, "PrintfExpr")) {
    if (FD->hasAttr<CUDAHostAttr>()) {
      report(CE->getBeginLoc(), Warnings::PRINTF_FUNC_NOT_SUPPORT, false);
      return;
    }
    std::string ReplacedStmt;
    llvm::raw_string_ostream OS(ReplacedStmt);
    if (DpctGlobalInfo::useExpNonStandardSYCLBuiltins()) {
      OS << MapNames::getClNamespace() << "ext::oneapi::experimental::printf";
      std::vector<std::string> ArgsSpelling;
      for (unsigned I = 0, E = CE->getNumArgs(); I != E; ++I) {
        ExprAnalysis ArgEA;
        ArgEA.analyze(CE->getArg(I));
        ArgsSpelling.push_back(ArgEA.getReplacedString());
      }
      OS << '(' << llvm::join(ArgsSpelling, ", ") << ')';
    } else {
      OS << DpctGlobalInfo::getStreamName() << " << ";
      CE->getArg(0)->printPretty(OS, nullptr,
                                 Result.Context->getPrintingPolicy());
      if (CE->getNumArgs() > 1 ||
          CE->getArg(0)->IgnoreImplicitAsWritten()->getStmtClass() !=
              Stmt::StringLiteralClass)
        report(CE->getBeginLoc(), Warnings::PRINTF_FUNC_MIGRATION_WARNING,
               false);
      FuncInfo->setStream();
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(OS.str())));
  } else if (auto Ctor =
                 getAssistNodeAsType<CXXConstructExpr>(Result, "CtorExpr")) {
    FuncInfo->addCallee(Ctor);
  }

  if (auto CXX = getAssistNodeAsType<CXXNewExpr>(Result, "CxxNewExpr")) {
    report(CXX->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION,
           false, "The usage of dynamic memory allocation and deallocation APIs");
  }

  if (auto CXX = getAssistNodeAsType<CXXDeleteExpr>(Result, "CxxDeleteExpr")) {
    report(CXX->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION,
           false, "The usage of dynamic memory allocation and deallocation APIs");
  }

  if (auto CE = getAssistNodeAsType<CallExpr>(Result, "MemoryManipulation")) {
    report(CE->getBeginLoc(), Warnings::DEVICE_UNSUPPORTED_CALL_FUNCTION, false,
           "The usage of dynamic memory allocation and deallocation APIs");
  }

  if (auto Var = getAssistNodeAsType<VarDecl>(Result, "varGrid")) {
    if (!Var->getInit())
      return;
    if (auto CE =
            dyn_cast<CallExpr>(Var->getInit()->IgnoreUnlessSpelledInSource())) {
      if (CE->getType().getCanonicalType().getAsString() !=
          "class cooperative_groups::__v1::grid_group")
        return;
      if (!DpctGlobalInfo::useNdRangeBarrier()) {
        return;
      }

      FuncInfo->setSync();
      auto Begin = Var->getBeginLoc();
      auto End = Var->getEndLoc();
      const auto &SM = *Result.SourceManager;

      End = End.getLocWithOffset(Lexer::MeasureTokenLength(
          End, SM, dpct::DpctGlobalInfo::getContext().getLangOpts()));

      Token Tok;
      Tok = Lexer::findNextToken(
                End, SM, dpct::DpctGlobalInfo::getContext().getLangOpts())
                .value();
      End = Tok.getLocation();

      auto Length = SM.getFileOffset(End) - SM.getFileOffset(Begin);

      // Remove statement "cg::grid_group grid = cg::this_grid();"
      emplaceTransformation(new ReplaceText(Begin, Length, ""));
    }
  }
  if (getAssistNodeAsType<TypeLoc>(Result, "fp64")) {
    FuncInfo->setBF64();
  }
  if (getAssistNodeAsType<TypeLoc>(Result, "fp16")) {
    FuncInfo->setBF16();
  }
}

std::string MemoryMigrationRule::getTypeStrRemovedAddrOf(const Expr *E,
                                                         bool isCOCE) {
  QualType QT;
  if (isCOCE) {
    auto COCE = dyn_cast<CXXOperatorCallExpr>(E);
    if (!COCE) {
      return "";
    }
    QT = COCE->getArg(0)->getType();
  } else {
    auto UO = dyn_cast<UnaryOperator>(E);
    if (!UO) {
      return "";
    }
    QT = UO->getSubExpr()->getType();
  }
  std::string ReplType = DpctGlobalInfo::getReplacedTypeName(QT);
  return ReplType;
}

/// Get the assigned part of the malloc function call.
/// \param [in] E The expression needs to be analyzed.
/// \param [in] Arg0Str The original string of the first arg of the malloc.
/// e.g.:
/// origin code:
///   int2 const * d_data;
///   cudaMalloc((void **)&d_data, sizeof(int2));
/// This function will return a string "d_data = (sycl::int2 const *)"
/// In this example, \param E is "&d_data", \param Arg0Str is "(void **)&d_data"
std::string MemoryMigrationRule::getAssignedStr(const Expr *E,
                                                const std::string &Arg0Str) {
  std::ostringstream Repl;
  std::string Type;
  printDerefOp(Repl, E, &Type);
  Repl << " = (" << Type << ")";

  return Repl.str();
}

const ArraySubscriptExpr *
MemoryMigrationRule::getArraySubscriptExpr(const Expr *E) {
  if (const auto MTE = dyn_cast<MaterializeTemporaryExpr>(E)) {
    if (auto TE = MTE->getSubExpr()) {
      if (auto UO = dyn_cast<UnaryOperator>(TE)) {
        if (auto Arg = dyn_cast<ArraySubscriptExpr>(UO->getSubExpr())) {
          return Arg;
        }
      }
    }
  }
  return nullptr;
}

const Expr *MemoryMigrationRule::getUnaryOperatorExpr(const Expr *E) {
  if (const auto MTE = dyn_cast<MaterializeTemporaryExpr>(E)) {
    if (auto TE = MTE->getSubExpr()) {
      if (auto UO = dyn_cast<UnaryOperator>(TE)) {
        return UO->getSubExpr();
      }
    }
  }
  return nullptr;
}

llvm::raw_ostream &printMemcpy3DParmsName(llvm::raw_ostream &OS,
                                          StringRef BaseName,
                                          StringRef MemberName) {
  return OS << BaseName << "_" << MemberName << getCTFixedSuffix();
}

std::pair<std::string, std::string>
MemoryMigrationRule::getMemAPIVarNameAndArrayOffset(const Expr *E) {
  std::string VarName;
  auto Sub = E->IgnoreImplicitAsWritten();
  if (auto MTE = dyn_cast<MaterializeTemporaryExpr>(Sub)) {
    Sub = MTE->getSubExpr()->IgnoreImplicitAsWritten();
  }
  if (auto UO = dyn_cast<UnaryOperator>(Sub)) {
    if (UO->getOpcode() == UO_AddrOf) {
      Sub = UO->getSubExpr()->IgnoreImplicitAsWritten();
    }
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(Sub)) {
    if (COCE->getOperator() == OO_Amp) {
      Sub = COCE->getArg(0);
    }
  }
  std::string ArrayOffset;
  if (auto ASE = dyn_cast<ArraySubscriptExpr>(Sub)) {
    Sub = ASE->getBase()->IgnoreImplicitAsWritten();
    auto Idx = ASE->getIdx();
    Expr::EvalResult ER;
    ArrayOffset = ExprAnalysis::ref(Idx);
    if (!Idx->isValueDependent() &&
        Idx->EvaluateAsInt(ER, DpctGlobalInfo::getContext())) {
      if (ER.Val.getInt().getZExtValue() == 0) {
        ArrayOffset.clear();
      }
    }
  }
  if (auto DRE = dyn_cast<DeclRefExpr>(Sub)) {
    if (auto VI = DpctGlobalInfo::getInstance().findMemVarInfo(
            dyn_cast<VarDecl>(DRE->getDecl()))) {
      VarName = VI->getName();
    }
  } else if (auto SL = dyn_cast<StringLiteral>(Sub)) {
    VarName = SL->getString();
  }
  return {VarName, ArrayOffset};
}

void MemoryMigrationRule::replaceMemAPIArg(const Expr *E,
                                           const std::string &StreamStr,
                                           std::string OffsetFromBaseStr) {
  std::string VarName, ArrayOffset;
  std::tie(VarName, ArrayOffset) = getMemAPIVarNameAndArrayOffset(E);
  if (VarName.empty())
    return;

  std::string Replaced;
  llvm::raw_string_ostream OS(Replaced);

  auto PrintVarName = [&](llvm::raw_ostream &Out) {
    Out << VarName << ".get_ptr(";
    if (!StreamStr.empty()) {
      requestFeature(HelperFeatureEnum::device_ext);
      Out << "*" << StreamStr;
    } else {
      requestFeature(HelperFeatureEnum::device_ext);
    }
    Out << ")";
    if (!ArrayOffset.empty())
      Out << " + " << ArrayOffset;
  };

  if (OffsetFromBaseStr.empty()) {
    PrintVarName(OS);
  } else {
    OS << "(char *)(";
    PrintVarName(OS);
    OS << ") + " << OffsetFromBaseStr;
  }
  emplaceTransformation(
      new ReplaceToken(E->getBeginLoc(), E->getEndLoc(), std::move(OS.str())));
}

bool MemoryMigrationRule::canUseTemplateStyleMigration(
    const Expr *AllocatedExpr, const Expr *SizeExpr, std::string &ReplType,
    std::string &ReplSize) {
  const Expr *AE = nullptr;
  if (auto CSCE = dyn_cast<CStyleCastExpr>(AllocatedExpr)) {
    AE = CSCE->getSubExpr()->IgnoreImplicitAsWritten();
  } else {
    AE = AllocatedExpr;
  }

  QualType DerefQT = AE->getType();
  if (DerefQT->isPointerType()) {
    DerefQT = DerefQT->getPointeeType();
    if (DerefQT->isPointerType()) {
      DerefQT = DerefQT->getPointeeType();
    } else {
      return false;
    }
  } else {
    return false;
  }

  std::string TypeStr = DpctGlobalInfo::getReplacedTypeName(DerefQT);
  // ReplType will be used as the template argument in memory API.
  ReplType = getFinalCastTypeNameStr(TypeStr);

  auto BO = dyn_cast<BinaryOperator>(SizeExpr);
  if (BO && BO->getOpcode() == BinaryOperatorKind::BO_Mul) {
    std::string Repl;
    if (!isContainMacro(BO->getLHS()) &&
        isSameSizeofTypeWithTypeStr(BO->getLHS(), TypeStr)) {
      // case 1: sizeof(b) * a
      ArgumentAnalysis AA;
      AA.setCallSpelling(BO);
      AA.analyze(BO->getRHS());
      Repl = AA.getRewritePrefix() + AA.getRewriteString() +
             AA.getRewritePostfix();
    } else if (!isContainMacro(BO->getRHS()) &&
               isSameSizeofTypeWithTypeStr(BO->getRHS(), TypeStr)) {
      // case 2: a * sizeof(b)
      ArgumentAnalysis AA;
      AA.setCallSpelling(BO);
      AA.analyze(BO->getLHS());
      Repl = AA.getRewritePrefix() + AA.getRewriteString() +
             AA.getRewritePostfix();
    } else {
      return false;
    }

    SourceLocation RemoveBegin, RemoveEnd;
    SourceRange RemoveRange = getStmtExpansionSourceRange(BO);
    RemoveBegin = RemoveRange.getBegin();
    RemoveEnd = RemoveRange.getEnd();
    RemoveEnd = RemoveEnd.getLocWithOffset(
        Lexer::MeasureTokenLength(RemoveEnd, DpctGlobalInfo::getSourceManager(),
                                  DpctGlobalInfo::getContext().getLangOpts()));
    emplaceTransformation(replaceText(RemoveBegin, RemoveEnd, std::move(Repl),
                                      DpctGlobalInfo::getSourceManager()));
    return true;
  } else {
    // case 3: sizeof(b)
    if (!isContainMacro(SizeExpr) &&
        isSameSizeofTypeWithTypeStr(SizeExpr, TypeStr)) {
      SourceLocation RemoveBegin, RemoveEnd;
      SourceRange RemoveRange = getStmtExpansionSourceRange(SizeExpr);
      RemoveBegin = RemoveRange.getBegin();
      RemoveEnd = RemoveRange.getEnd();
      RemoveEnd = RemoveEnd.getLocWithOffset(Lexer::MeasureTokenLength(
          RemoveEnd, DpctGlobalInfo::getSourceManager(),
          DpctGlobalInfo::getContext().getLangOpts()));
      emplaceTransformation(replaceText(RemoveBegin, RemoveEnd, "1",
                                        DpctGlobalInfo::getSourceManager()));

      return true;
    }
  }

  return false;
}

void MemoryMigrationRule::instrumentAddressToSizeRecordForCodePin(
    const CallExpr *C, int PtrArgLoc, int AllocMemSizeLoc) {
  auto &SM = dpct::DpctGlobalInfo::getSourceManager(); 
  if (DpctGlobalInfo::isCodePinEnabled()) {
    SourceLocation CallEnd = C->getEndLoc();
    if(SM.isMacroArgExpansion(C->getEndLoc())){
      CallEnd = SM.getExpansionRange(C->getEndLoc()).getEnd();
    } else {
      CallEnd = getDefinitionRange(C->getBeginLoc(), C->getEndLoc()).getEnd();
    }

    auto PtrSizeLoc = Lexer::findLocationAfterToken(
        CallEnd, tok::semi, SM, DpctGlobalInfo::getContext().getLangOpts(),
        false);

    emplaceTransformation(new InsertText(
        PtrSizeLoc,
        std::string(getNL()) + "dpctexp::codepin::get_ptr_size_map()[" +
            getDrefName(C->getArg(PtrArgLoc)) + "] = " +
            std::string(Lexer::getSourceText(
                CharSourceRange::getTokenRange(
                    C->getArg(AllocMemSizeLoc)->getSourceRange()),
                DpctGlobalInfo::getSourceManager(), LangOptions())) +
            ";",
        0, RT_CUDAWithCodePin));
    emplaceTransformation(new InsertText(
        PtrSizeLoc,
        std::string(getNL()) + "dpctexp::codepin::get_ptr_size_map()[" +
            getDrefName(C->getArg(PtrArgLoc)) +
            "] = " + ExprAnalysis::ref(C->getArg(AllocMemSizeLoc)) + ";",
        0, RT_ForSYCLMigration));
    DpctGlobalInfo::getInstance().insertHeader(
        C->getBeginLoc(), HT_DPCT_CodePin_CUDA, RT_CUDAWithCodePin);
    DpctGlobalInfo::getInstance().insertHeader(
        C->getBeginLoc(), HT_DPCT_CodePin_SYCL, RT_ForSYCLMigration);
  }
  return;
}

/// Transform cudaMallocxxx() to xxx = mallocxxx();
void MemoryMigrationRule::mallocMigrationWithTransformation(
    SourceManager &SM, const CallExpr *C, const std::string &CallName,
    std::string &&ReplaceName, const std::string &PaddingArgs,
    bool NeedTypeCast, size_t AllocatedArgIndex, size_t SizeArgIndex) {
  std::string ReplSize, ReplType;
  if (DpctGlobalInfo::getUsmLevel() == UsmLevel::UL_Restricted &&
      CallName != "cudaMallocArray" && CallName != "cudaMalloc3DArray" &&
      CallName != "cublasAlloc" &&
      canUseTemplateStyleMigration(C->getArg(AllocatedArgIndex),
                                   C->getArg(SizeArgIndex), ReplType,
                                   ReplSize)) {
    auto TM = new InsertBeforeStmt(
        C, getTransformedMallocPrefixStr(C->getArg(AllocatedArgIndex),
                                         NeedTypeCast, true));
    TM->setInsertPosition(IP_Right);
    emplaceTransformation(TM);

    emplaceTransformation(
        new ReplaceCalleeName(C, ReplaceName + "<" + ReplType + ">"));
  } else {
    auto TM = new InsertBeforeStmt(
        C, getTransformedMallocPrefixStr(C->getArg(AllocatedArgIndex),
                                         NeedTypeCast));
    TM->setInsertPosition(IP_Right);
    emplaceTransformation(TM);

    emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceName)));
  }
  emplaceTransformation(removeArg(C, AllocatedArgIndex, SM));
  if (!PaddingArgs.empty())
    emplaceTransformation(
        new InsertText(C->getRParenLoc(), ", " + PaddingArgs));
}

/// e.g., for int *a and cudaMalloc(&a, size), print "a = ".
/// If \p DerefType is not null, assign a string "int *".
void printDerefOp(std::ostream &OS, const Expr *E, std::string *DerefType) {
  E = E->IgnoreImplicitAsWritten();
  bool NeedDerefOp = true;
  if (auto UO = dyn_cast<UnaryOperator>(E)) {
    if (UO->getOpcode() == clang::UO_AddrOf) {
      E = UO->getSubExpr()->IgnoreImplicitAsWritten();
      NeedDerefOp = false;
    }
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (COCE->getOperator() == clang::OO_Amp && COCE->getNumArgs() == 1) {
      E = COCE->getArg(0)->IgnoreImplicitAsWritten();
      NeedDerefOp = false;
    }
  }
  E = E->IgnoreParens();

  std::unique_ptr<ParensPrinter<std::ostream>> PP;
  if (NeedDerefOp) {
    OS << "*";
    switch (E->getStmtClass()) {
    case Stmt::DeclRefExprClass:
    case Stmt::MemberExprClass:
    case Stmt::ParenExprClass:
    case Stmt::CallExprClass:
    case Stmt::IntegerLiteralClass:
      break;
    default:
      PP = std::make_unique<ParensPrinter<std::ostream>>(OS);
      break;
    }
  }
  ExprAnalysis EA(E);
  EA.analyze();
  OS << EA.getReplacedString();

  if (DerefType) {
    QualType DerefQT;
    if (auto ArraySub = dyn_cast<ArraySubscriptExpr>(E)) {
      QualType BaseType = ArraySub->getBase()->getType();
      if (BaseType->isArrayType()) {
        if (auto Array = BaseType->getAsArrayTypeUnsafe()) {
          DerefQT = Array->getElementType();
        }
      } else if (BaseType->isPointerType()) {
        DerefQT = BaseType->getPointeeType();
      }
    }
    if (DerefQT.isNull()) {
      DerefQT = E->getType();
    }
    if (NeedDerefOp)
      DerefQT = DerefQT->getPointeeType();
    *DerefType = DpctGlobalInfo::getReplacedTypeName(DerefQT);
  }
}

/// e.g., for int *a and cudaMalloc(&a, size), return "a = (int *)".
/// If \p NeedTypeCast is false, return "a = ";
/// If \p TemplateStyle is true, \p NeedTypeCast will be specified as false
/// always
std::string MemoryMigrationRule::getTransformedMallocPrefixStr(
    const Expr *MallocOutArg, bool NeedTypeCast, bool TemplateStyle) {
  if (TemplateStyle)
    NeedTypeCast = false;
  std::ostringstream OS;
  std::string CastTypeName;
  MallocOutArg = MallocOutArg->IgnoreImplicitAsWritten();
  if (auto CSCE = dyn_cast<CStyleCastExpr>(MallocOutArg)) {
    MallocOutArg = CSCE->getSubExpr()->IgnoreImplicitAsWritten();
    if (!TemplateStyle)
      NeedTypeCast = true;
  }
  printDerefOp(OS, MallocOutArg, NeedTypeCast ? &CastTypeName : nullptr);

  OS << " = ";
  if (!CastTypeName.empty())
    OS << "(" << getFinalCastTypeNameStr(CastTypeName) << ")";

  return OS.str();
}

/// Common migration for cudaMallocArray and cudaMalloc3DArray.
void MemoryMigrationRule::mallocArrayMigration(const CallExpr *C,
                                               const std::string &Name,
                                               const std::string &Flag,
                                               SourceManager &SM) {

  requestFeature(HelperFeatureEnum::device_ext);
  mallocMigrationWithTransformation(
      SM, C, Name, "new " + MapNames::getDpctNamespace() + "image_matrix", "",
      false);

  emplaceTransformation(new ReplaceStmt(C->getArg(C->getNumArgs() - 1), Flag));

  std::ostringstream OS;
  printDerefOp(OS, C->getArg(1));
  emplaceTransformation(new ReplaceStmt(C->getArg(1), OS.str()));
}

void MemoryMigrationRule::mallocMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  if (isPlaceholderIdxDuplicated(C))
    return;
  int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
  if (Name == "cudaMalloc" || Name == "cuMemAlloc_v2") {
    if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
      // Leverage CallExprRewritter to migrate the USM version
      ExprAnalysis EA(C);
      auto LocInfo = DpctGlobalInfo::getLocInfo(C->getBeginLoc());
      auto Info = std::make_shared<PriorityReplInfo>();
      if (auto TM = EA.getReplacement())
        Info->Repls.push_back(TM->getReplacement(DpctGlobalInfo::getContext()));
      Info->Repls.insert(Info->Repls.end(), EA.getSubExprRepl().begin(),
                         EA.getSubExprRepl().end());
      DpctGlobalInfo::addPriorityReplInfo(
          LocInfo.first.getCanonicalPath().str() + std::to_string(LocInfo.second), Info);
    } else {
      DpctGlobalInfo::getInstance().insertCudaMalloc(C);
      auto LocInfo = DpctGlobalInfo::getLocInfo(C->getBeginLoc());
      auto Action = []() {
        requestFeature(HelperFeatureEnum::device_ext);
      };
      auto Info = std::make_shared<PriorityReplInfo>();
      auto &Context = DpctGlobalInfo::getContext();
      auto &SM = *Result.SourceManager;
      Info->RelatedAction.emplace_back(Action);
      if (auto TM = removeArg(C, 0, SM))
        Info->Repls.push_back(TM->getReplacement(Context));
      ExprAnalysis EA(C);
      if (auto TM = EA.getReplacement())
        Info->Repls.push_back(TM->getReplacement(DpctGlobalInfo::getContext()));
      Info->Repls.insert(Info->Repls.end(), EA.getSubExprRepl().begin(),
                         EA.getSubExprRepl().end());

      DpctGlobalInfo::addPriorityReplInfo(
          LocInfo.first.getCanonicalPath().str() + std::to_string(LocInfo.second), Info);
    }
    // Insert header here since PriorityReplInfo delay the replacement addation
    // to the post process. At that time, the MainFile is invalid.
    DpctGlobalInfo::getInstance().insertHeader(C->getBeginLoc(),
                                               HeaderType::HT_SYCL);
    instrumentAddressToSizeRecordForCodePin(C,0,1);
  } else if (Name == "cudaHostAlloc" || Name == "cudaMallocHost" ||
             Name == "cuMemHostAlloc" || Name == "cuMemAllocHost_v2" ||
             Name == "cuMemAllocPitch_v2" || Name == "cudaMallocPitch" ||
             Name == "cudaMallocMipmappedArray") {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  } else if (Name == "cudaMallocManaged" || Name == "cuMemAllocManaged") {
    if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
      // Leverage CallExprRewriter to migrate the USM version
      ExprAnalysis EA(C);
      auto LocInfo = DpctGlobalInfo::getLocInfo(C->getBeginLoc());
      auto Info = std::make_shared<PriorityReplInfo>();
      if (auto TM = EA.getReplacement())
        Info->Repls.push_back(TM->getReplacement(DpctGlobalInfo::getContext()));
      Info->Repls.insert(Info->Repls.end(), EA.getSubExprRepl().begin(),
                         EA.getSubExprRepl().end());
      DpctGlobalInfo::addPriorityReplInfo(
          LocInfo.first.getCanonicalPath().str() + std::to_string(LocInfo.second), Info);
      // Insert header here since PriorityReplInfo delay the replacement
      // addation to the post process. At that time, the MainFile is invalid.
      DpctGlobalInfo::getInstance().insertHeader(C->getBeginLoc(),
                                                 HeaderType::HT_SYCL);
    } else {
      ManagedPointerAnalysis MPA(C, IsAssigned);
      MPA.RecursiveAnalyze();
      MPA.applyAllSubExprRepl();
    }
  } else if (Name == "cublasAlloc") {
    // TODO: migrate functions when they are in template
    // TODO: migrate functions when they are in macro body
    auto ArgRange0 = getStmtExpansionSourceRange(C->getArg(0));
    auto ArgEnd0 = ArgRange0.getEnd().getLocWithOffset(
        Lexer::MeasureTokenLength(ArgRange0.getEnd(), *(Result.SourceManager),
                                  Result.Context->getLangOpts()));
    auto ArgRange1 = getStmtExpansionSourceRange(C->getArg(1));
    emplaceTransformation(
        replaceText(ArgEnd0, ArgRange1.getBegin(), "*", *Result.SourceManager));
    insertAroundStmt(C->getArg(0), "(", ")");
    insertAroundStmt(C->getArg(1), "(", ")");
    DpctGlobalInfo::getInstance().insertCublasAlloc(C);
    emplaceTransformation(removeArg(C, 2, *Result.SourceManager));
    if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
      buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
      if (IsAssigned)
        emplaceTransformation(new InsertBeforeStmt(C, MapNames::getCheckErrorMacroName() + "("));
      mallocMigrationWithTransformation(
          *Result.SourceManager, C, Name,
          MapNames::getClNamespace() + "malloc_device",
          "{{NEEDREPLACEQ" + std::to_string(Index) + "}}", true, 2);
      if (IsAssigned) {
        emplaceTransformation(new InsertAfterStmt(C, ")"));
        requestFeature(HelperFeatureEnum::device_ext);
      }
    } else {
      ExprAnalysis EA(C->getArg(2));
      EA.analyze();
      std::ostringstream OS;
      std::string Type;
      if (IsAssigned)
        OS << MapNames::getCheckErrorMacroName() + "(";
      printDerefOp(OS, C->getArg(2)->IgnoreCasts()->IgnoreParens(), &Type);
      if (Type != "NULL TYPE" && Type != "void *")
        OS << " = (" << Type << ")";
      else
        OS << " = ";

      emplaceTransformation(new InsertBeforeStmt(C, OS.str()));
      emplaceTransformation(new ReplaceCalleeName(
          C, MapNames::getDpctNamespace() + "dpct_malloc"));
      requestFeature(HelperFeatureEnum::device_ext);
      if (IsAssigned) {
        emplaceTransformation(new InsertAfterStmt(C, ")"));
        requestFeature(HelperFeatureEnum::device_ext);
      }
    }
  } else if (Name == "cudaMalloc3D") {
    std::ostringstream OS;
    std::string Type;
    if (IsAssigned)
      OS << MapNames::getCheckErrorMacroName() + "(";
    printDerefOp(OS, C->getArg(0)->IgnoreCasts()->IgnoreParens(), &Type);
    if (Name != "cudaMalloc3D" && Type != "NULL TYPE" && Type != "void *")
      OS << " = (" << Type << ")";
    else
      OS << " = ";

    requestFeature(HelperFeatureEnum::device_ext);
    emplaceTransformation(new InsertBeforeStmt(C, OS.str()));
    emplaceTransformation(
        new ReplaceCalleeName(C, MemoryMigrationRule::getMemoryHelperFunctionName("malloc")));
    emplaceTransformation(removeArg(C, 0, *Result.SourceManager));
    std::ostringstream OS2;
    printDerefOp(OS2, C->getArg(1));
    if (IsAssigned) {
      emplaceTransformation(new InsertAfterStmt(C, ")"));
      requestFeature(HelperFeatureEnum::device_ext);
    }
  } else if (Name == "cudaMalloc3DArray") {
    Expr::EvalResult ER;
    std::string ImageType = "image_type::standard";
    if (!C->getArg(3)->isValueDependent() &&
        C->getArg(3)->EvaluateAsInt(ER, *Result.Context)) {
      int64_t Value = ER.Val.getInt().getExtValue();
      const auto &ImageTypePair = MapNamesLang::ArrayFlagMap.find(Value);
      if (ImageTypePair != MapNamesLang::ArrayFlagMap.end())
        ImageType = "image_type::" + ImageTypePair->second;
    }
    if (DpctGlobalInfo::useExtBindlessImages()) {
      std::string Replacement;
      llvm::raw_string_ostream OS(Replacement);
      DerefExpr(C->getArg(0), C).print(OS);
      OS << " = new " << MapNames::getDpctNamespace()
         << "experimental::image_mem_wrapper(";
      DerefExpr(C->getArg(1), C).print(OS);
      OS << ", " << ExprAnalysis::ref(C->getArg(2)) << ", "
         << MapNames::getClNamespace()
         << "ext::oneapi::experimental::" << ImageType << ")";
      return emplaceTransformation(new ReplaceStmt(C, Replacement));
    }
    mallocArrayMigration(C, Name, MapNames::getDpctNamespace() + ImageType,
                         *Result.SourceManager);
  } else if (Name == "cudaMallocArray") {
    if (DpctGlobalInfo::useExtBindlessImages()) {
      std::string Replacement;
      llvm::raw_string_ostream OS(Replacement);
      DerefExpr(C->getArg(0), C).print(OS);
      OS << " = new " << MapNames::getDpctNamespace()
         << "experimental::image_mem_wrapper(";
      DerefExpr(C->getArg(1), C).print(OS);
      OS << ", " << ExprAnalysis::ref(C->getArg(2));
      if (!C->getArg(3)->isDefaultArgument())
        OS << ", " << ExprAnalysis::ref(C->getArg(3));
      OS << ")";
      return emplaceTransformation(new ReplaceStmt(C, Replacement));
    }
    mallocArrayMigration(C, Name,
                         MapNames::getDpctNamespace() + "image_type::standard",
                         *Result.SourceManager);
    static std::string SizeClassName =
        DpctGlobalInfo::getCtadClass(MapNames::getClNamespace() + "range", 2);
    if (C->getArg(3)->isDefaultArgument())
      aggregateArgsToCtor(C, SizeClassName, 2, 2, ", 0", *Result.SourceManager);
    else
      aggregateArgsToCtor(C, SizeClassName, 2, 3, "", *Result.SourceManager);
  }
}

void MemoryMigrationRule::memcpyMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  for (unsigned I = 0, E = C->getNumArgs(); I != E; ++I) {
    if (isa<PackExpansionExpr>(C->getArg(I))) {
      return;
    }
  }

  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  if (!CallExprRewriterFactoryBase::RewriterMap)
    return;
  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(Name);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  std::string ReplaceStr;
  // Detect if there is Async in the func name and crop the async substr
  std::string NameRef = Name;
  bool IsAsync = false;
  // Whether in experimental namespace in syclcompat.
  bool IsExperimentalInSYCLCompat = false;
  size_t AsyncLoc = NameRef.find("Async");
  if (AsyncLoc != std::string::npos) {
    IsAsync = true;
    report(DpctGlobalInfo::getSourceManager().getExpansionLoc(C->getBeginLoc()),
           Diagnostics::ASYNC_MEMCPY_WARNING, true, Name);
    NameRef = NameRef.substr(0, AsyncLoc);
  }
  if (!NameRef.compare("cudaMemcpy2D")) {
    handleDirection(C, 6);
    handleAsync(C, 7, Result);
  } else if (NameRef.rfind("cudaMemcpy3D", 0) == 0 ||
             NameRef.rfind("cuMemcpy3D", 0) == 0 ||
             NameRef.rfind("cuMemcpy2D", 0) == 0) {
    handleAsync(C, 1, Result);
    std::string Replacement;
    llvm::raw_string_ostream OS(Replacement);
    DerefExpr(C->getArg(0), C).print(OS);
    emplaceTransformation(new ReplaceStmt(C->getArg(0), Replacement));
    IsExperimentalInSYCLCompat = true;
  } else if (!NameRef.compare("cudaMemcpy") ||
             NameRef.rfind("cuMemcpyDtoH", 0) == 0) {
    if (!NameRef.compare("cudaMemcpy")) {
      handleDirection(C, 3);
    }
    std::string AsyncQueue;
    bool NeedTypeCast = false;

    size_t StreamIndex = NameRef.compare("cudaMemcpy") ? 3 : 4;
    if (StreamIndex < C->getNumArgs()) {
      auto StreamArg = C->getArg(StreamIndex);
      // Is the stream argument a default stream handle we recognize?
      // Note: the value for the default stream argument in
      // cudaMemcpyAsync is 0, aka the default stream
      if (StreamArg->isDefaultArgument() || isDefaultStream(StreamArg)) {
        AsyncQueue = "";
      }
      // Are we casting from an integer?
      else if (auto Cast = dyn_cast<CastExpr>(StreamArg);
               Cast && Cast->getCastKind() != clang::CK_LValueToRValue &&
               Cast->getSubExpr()->getType()->isIntegerType()) {
        requestFeature(HelperFeatureEnum::device_ext);
        AsyncQueue = MapNames::getDpctNamespace() + "int_as_queue_ptr(" +
                     ExprAnalysis::ref(Cast->getSubExpr()) + ")";
      } else {
        // If we are implicitly casting from something other than
        // an int (e.g. a user defined class), we need to explicitly
        // insert that cast in the migration to use member access (->).
        if (auto ICE = dyn_cast<ImplicitCastExpr>(StreamArg))
          NeedTypeCast = ICE->getCastKind() != clang::CK_LValueToRValue;
        AsyncQueue = ExprAnalysis::ref(StreamArg);
      }
    }

    replaceMemAPIArg(C->getArg(0), AsyncQueue);
    replaceMemAPIArg(C->getArg(1), AsyncQueue);
    if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
      // Since the range of removeArg is larger than the range of
      // handleDirection, the handle direction replacement will be removed.
      emplaceTransformation(removeArg(C, 3, *Result.SourceManager));
      if (IsAsync) {
        emplaceTransformation(removeArg(C, 4, *Result.SourceManager));
      } else {
        if (NameRef.compare("cudaMemcpy") || !canOmitMemcpyWait(C)) {
          // wait is needed when FuncName is not cudaMemcpy or
          // cudaMemcpy really needs wait
          emplaceTransformation(new InsertAfterStmt(C, ".wait()"));
        }
      }
      if (AsyncQueue.empty()) {
        if (isPlaceholderIdxDuplicated(C))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
        ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memcpy";
      } else {
        if (NeedTypeCast)
          AsyncQueue = buildString("((sycl::queue *)(", AsyncQueue, "))");

        ReplaceStr = AsyncQueue + "->memcpy";
      }
    } else {
      if (!NameRef.compare("cudaMemcpy")) {
        handleAsync(C, 4, Result);
      } else {
        emplaceTransformation(new InsertAfterStmt(
            C->getArg(2), ", " + MapNames::getDpctNamespace() + "automatic"));
        handleAsync(C, 3, Result);
      }
    }
  } else if (!NameRef.compare("cudaMemcpyPeer") ||
             !NameRef.compare("cuMemcpyPeer")) {
    handleAsync(C, 5, Result);
  }

  if (ReplaceStr.empty()) {
    if (IsAsync) {
      ReplaceStr = MemoryMigrationRule::getMemoryHelperFunctionName(
          "memcpy_async", IsExperimentalInSYCLCompat);
      requestFeature(HelperFeatureEnum::device_ext);
    } else {
      ReplaceStr = MemoryMigrationRule::getMemoryHelperFunctionName(
          "memcpy", IsExperimentalInSYCLCompat);
      requestFeature(HelperFeatureEnum::device_ext);
    }
  }

  if (ULExpr) {
    auto BeginLoc = ULExpr->getBeginLoc();
    auto EndLoc = ULExpr->hasExplicitTemplateArgs()
                 ? ULExpr->getLAngleLoc().getLocWithOffset(-1)
                 : ULExpr->getEndLoc();
    emplaceTransformation(new ReplaceToken(BeginLoc, EndLoc, std::move(ReplaceStr)));
  } else {
    emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
  }
}

void MemoryMigrationRule::arrayMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  if (DpctGlobalInfo::useSYCLCompat()) {
    ExprAnalysis EA;
    if (ULExpr)
      EA.analyze(ULExpr);
    else
      EA.analyze(C);
    emplaceTransformation(EA.getReplacement());
    return;
  }
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  auto& SM = *Result.SourceManager;
  std::string ReplaceStr;
  StringRef NameRef(Name);
  auto EndPos = C->getNumArgs() - 1;
  bool IsAsync = NameRef.ends_with("Async");
  if (NameRef == "cuMemcpyAtoH_v2" || NameRef == "cuMemcpyHtoA_v2" ||
      NameRef == "cuMemcpyAtoHAsync_v2" || NameRef == "cuMemcpyHtoAAsync_v2" ||
      NameRef == "cuMemcpyAtoD_v2" || NameRef == "cuMemcpyDtoA_v2" ||
      NameRef == "cuMemcpyAtoA_v2" || NameRef == "cudaGetMipmappedArrayLevel") {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  if (IsAsync) {
    NameRef = NameRef.drop_back(5 /* len of "Async" */);
    ReplaceStr =
        DpctGlobalInfo::useExtBindlessImages()
            ? MapNames::getDpctNamespace() + "experimental::async_dpct_memcpy"
            : MapNames::getDpctNamespace() + "async_dpct_memcpy";

    auto StreamExpr = C->getArg(EndPos);
    std::string Str;
    if (isDefaultStream(StreamExpr)) {
      emplaceTransformation(removeArg(C, EndPos, SM));
      emplaceTransformation(removeArg(C, --EndPos, SM));
    } else {
      auto Begin = getArgEndLocation(C, EndPos - 2, SM),
           End = getArgEndLocation(C, EndPos, SM);
      llvm::raw_string_ostream OS(Str);
      if (!DpctGlobalInfo::useExtBindlessImages())
        OS << ", " << MapNames::getDpctNamespace() << "automatic";
      OS << ", ";
      DerefExpr(StreamExpr, C).print(OS);
      emplaceTransformation(replaceText(Begin, End, std::move(Str), SM));
    }
    requestFeature(HelperFeatureEnum::device_ext);
  } else {
    ReplaceStr =
        DpctGlobalInfo::useExtBindlessImages()
            ? MapNames::getDpctNamespace() + "experimental::dpct_memcpy"
            : MapNames::getDpctNamespace() + "dpct_memcpy";
    emplaceTransformation(removeArg(C, EndPos, SM));
    requestFeature(HelperFeatureEnum::device_ext);
  }

  if (NameRef == "cudaMemcpy2DArrayToArray") {
    if (DpctGlobalInfo::useExtBindlessImages()) {
      emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
    } else {
      insertToPitchedData(C, 0);
      aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
      insertToPitchedData(C, 3);
      aggregate3DVectorClassCtor(C, "id", 4, "0", SM);
      aggregate3DVectorClassCtor(C, "range", 6, "1", SM);
    }
  } else if (NameRef == "cudaMemcpy2DFromArray") {
    if (DpctGlobalInfo::useExtBindlessImages()) {
      emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
    } else {
      aggregatePitchedData(C, 0, 1, SM);
      insertZeroOffset(C, 2);
      insertToPitchedData(C, 2);
      aggregate3DVectorClassCtor(C, "id", 3, "0", SM);
      aggregate3DVectorClassCtor(C, "range", 5, "1", SM);
    }
  } else if (NameRef == "cudaMemcpy2DToArray") {
    if (DpctGlobalInfo::useExtBindlessImages()) {
      emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
    } else {
      insertToPitchedData(C, 0);
      aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
      aggregatePitchedData(C, 3, 4, SM);
      insertZeroOffset(C, 5);
      aggregate3DVectorClassCtor(C, "range", 5, "1", SM);
    }
  } else if (NameRef == "cudaMemcpyArrayToArray") {
    if (DpctGlobalInfo::useExtBindlessImages()) {
      emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
    } else {
      insertToPitchedData(C, 0);
      aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
      insertToPitchedData(C, 3);
      aggregate3DVectorClassCtor(C, "id", 4, "0", SM);
      aggregate3DVectorClassCtor(C, "range", 6, "1", SM, 1);
    }
  } else if (NameRef == "cudaMemcpyFromArray") {
    if (DpctGlobalInfo::useExtBindlessImages()) {
      emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
    } else {
      aggregatePitchedData(C, 0, 4, SM, true);
      insertZeroOffset(C, 1);
      insertToPitchedData(C, 1);
      aggregate3DVectorClassCtor(C, "id", 2, "0", SM);
      aggregate3DVectorClassCtor(C, "range", 4, "1", SM, 1);
    }
  } else if (NameRef == "cudaMemcpyToArray") {
    if (DpctGlobalInfo::useExtBindlessImages()) {
      emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
    } else {
      insertToPitchedData(C, 0);
      aggregate3DVectorClassCtor(C, "id", 1, "0", SM);
      aggregatePitchedData(C, 3, 4, SM, true);
      insertZeroOffset(C, 4);
      aggregate3DVectorClassCtor(C, "range", 4, "1", SM, 1);
    }
  }

  if (ULExpr) {
    auto BeginLoc = ULExpr->getBeginLoc();
    auto EndLoc = ULExpr->hasExplicitTemplateArgs()
                      ? ULExpr->getLAngleLoc().getLocWithOffset(-1)
                      : ULExpr->getEndLoc();
    emplaceTransformation(new ReplaceToken(BeginLoc, EndLoc, std::move(ReplaceStr)));
  } else {
    emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
  }
}

void MemoryMigrationRule::memcpySymbolMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string DirectionName;
  // Currently, if memory API occurs in a template, we will migrate the API call
  // under the undeclared decl AST node and the explicit specialization AST
  // node. The API call in explicit specialization is same as without template.
  // But if the API has non-specified default parameters and it is in an
  // undeclared decl, these default parameters will not be counted into the
  // number of call arguments. So, we need check the argument number before get
  // it.
  if (C->getNumArgs() >= 5 && !C->getArg(4)->isDefaultArgument()) {
    const Expr *Direction = C->getArg(4);
    const DeclRefExpr *DD = dyn_cast_or_null<DeclRefExpr>(Direction);
    if (DD && isa<EnumConstantDecl>(DD->getDecl())) {
      DirectionName = DD->getNameInfo().getName().getAsString();
      auto Search = MapNames::EnumNamesMap.find(DirectionName);
      if (Search == MapNames::EnumNamesMap.end())
        return;
      requestHelperFeatureForEnumNames(DirectionName);
      Direction = nullptr;
      DirectionName = Search->second->NewName;
    }
  }

  DpctGlobalInfo &Global = DpctGlobalInfo::getInstance();
  auto MallocInfo = Global.findCudaMalloc(C->getArg(1));
  auto VD = CudaMallocInfo::getDecl(C->getArg(0));
  if (MallocInfo && VD) {
    if (auto Var = Global.findMemVarInfo(VD)) {
      requestFeature(HelperFeatureEnum::device_ext);
      emplaceTransformation(new ReplaceStmt(
          C, Var->getName() + ".assign(" +
                 MallocInfo->getAssignArgs(Var->getType()->getBaseName()) +
                 ")"));
      return;
    }
  }

  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  std::string ReplaceStr;
  std::string StreamStr;
  if (isPlaceholderIdxDuplicated(C))
    return;
  int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
  if (Name == "cudaMemcpyToSymbol" || Name == "cudaMemcpyFromSymbol") {
    if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
      buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
      ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memcpy";
    } else {
      requestFeature(HelperFeatureEnum::device_ext);
      ReplaceStr = MapNames::getDpctNamespace() + "dpct_memcpy";
    }
  } else {
    if (C->getNumArgs() == 6 && !C->getArg(5)->isDefaultArgument()) {
      if (!isDefaultStream(C->getArg(5))) {
        StreamStr = ExprAnalysis::ref(C->getArg(5));
      }
    }
    if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
      if (StreamStr.empty()) {
        buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
        ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memcpy";
      } else {
        ReplaceStr = StreamStr + "->memcpy";
      }
    } else {
      requestFeature(HelperFeatureEnum::device_ext);
      ReplaceStr = MapNames::getDpctNamespace() + "async_dpct_memcpy";
    }
  }

  if (ULExpr) {
    auto BeginLoc = ULExpr->getBeginLoc();
    auto EndLoc = ULExpr->hasExplicitTemplateArgs()
                 ? ULExpr->getLAngleLoc().getLocWithOffset(-1)
                 : ULExpr->getEndLoc();
    emplaceTransformation(new ReplaceToken(BeginLoc, EndLoc, std::move(ReplaceStr)));
  } else {
    emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
  }

  ExprAnalysis EA;
  std::string OffsetFromBaseStr;
  if (C->getNumArgs() >= 4 && !C->getArg(3)->isDefaultArgument()) {
    EA.analyze(C->getArg(3));
    OffsetFromBaseStr = EA.getReplacedString();
  } else {
    OffsetFromBaseStr = "0";
  }

  if ((Name == "cudaMemcpyToSymbol" || Name == "cudaMemcpyToSymbolAsync") &&
      OffsetFromBaseStr != "0") {
    replaceMemAPIArg(C->getArg(0), StreamStr, OffsetFromBaseStr);
  } else {
    replaceMemAPIArg(C->getArg(0), StreamStr);
  }

  if ((Name == "cudaMemcpyFromSymbol" || Name == "cudaMemcpyFromSymbolAsync") &&
      OffsetFromBaseStr != "0") {
    replaceMemAPIArg(C->getArg(1), StreamStr, OffsetFromBaseStr);
  } else {
    replaceMemAPIArg(C->getArg(1), StreamStr);
  }

  // Remove C->getArg(3)
  if (C->getNumArgs() >= 4 && !C->getArg(3)->isDefaultArgument()) {
    if (auto TM = removeArg(C, 3, *Result.SourceManager))
      emplaceTransformation(TM);
  }

  if (C->getNumArgs() >= 5 && !C->getArg(4)->isDefaultArgument()) {
    emplaceTransformation(
        new ReplaceStmt(C->getArg(4), std::move(DirectionName)));
  }

  // Async
  if (Name == "cudaMemcpyToSymbolAsync" ||
      Name == "cudaMemcpyFromSymbolAsync") {
    if (C->getNumArgs() == 6 && !C->getArg(4)->isDefaultArgument()) {
      if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
        if (auto TM = removeArg(C, 4, *Result.SourceManager))
          emplaceTransformation(TM);
        if (!C->getArg(5)->isDefaultArgument()) {
          if (auto TM = removeArg(C, 5, *Result.SourceManager))
            emplaceTransformation(TM);
        }
      } else {
        handleAsync(C, 5, Result);
      }
    } else if (C->getNumArgs() == 5 && !C->getArg(4)->isDefaultArgument()) {
      if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
        if (auto TM = removeArg(C, 4, *Result.SourceManager))
          emplaceTransformation(TM);
      }
    }
  } else {
    if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
      if (C->getNumArgs() == 5 && !C->getArg(4)->isDefaultArgument()) {
        if (auto TM = removeArg(C, 4, *Result.SourceManager))
          emplaceTransformation(TM);
      }
      if (!canOmitMemcpyWait(C)) {
        emplaceTransformation(new InsertAfterStmt(C, ".wait()"));
      }
    }
  }
}

void MemoryMigrationRule::freeMigration(const MatchFinder::MatchResult &Result,
                                        const CallExpr *C,
                                        const UnresolvedLookupExpr *ULExpr,
                                        bool IsAssigned) {

  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }
  if (isPlaceholderIdxDuplicated(C))
    return;

  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(Name);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
  int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
  if (Name == "cudaFreeHost" || Name == "cuMemFreeHost") {
    if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
      CheckCanUseCLibraryMallocOrFree Checker(0, true);
      ExprAnalysis EA;
      EA.analyze(C->getArg(0));
      std::ostringstream Repl;
      if(Checker(C)) {
        Repl << "free(" << EA.getReplacedString() << ")";
      } else {
        buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
        Repl << MapNames::getClNamespace() + "free(" << EA.getReplacedString()
           << ", {{NEEDREPLACEQ" + std::to_string(Index) + "}})";
      }
      emplaceTransformation(new ReplaceStmt(C, std::move(Repl.str())));
    } else {
      emplaceTransformation(new ReplaceCalleeName(C, "free"));
    }
  } else if (Name == "cudaFreeArray") {
    ExprAnalysis EA(C->getArg(0));
    EA.analyze();
    emplaceTransformation(
        new ReplaceStmt(C, "delete " + EA.getReplacedString()));
  }
}

void MemoryMigrationRule::memsetMigration(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(Name);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  std::string ReplaceStr;
  StringRef NameRef(Name);
  bool IsAsync = NameRef.ends_with("Async");
  if (IsAsync) {
    NameRef = NameRef.drop_back(5 /* len of "Async" */);
    ReplaceStr = MemoryMigrationRule::getMemoryHelperFunctionName("memset_async");
    requestFeature(HelperFeatureEnum::device_ext);
  } else {
    ReplaceStr = MemoryMigrationRule::getMemoryHelperFunctionName("memset");
    requestFeature(HelperFeatureEnum::device_ext);
  }

  if (NameRef == "cudaMemset2D") {
    handleAsync(C, 5, Result);
  } else if (NameRef == "cudaMemset3D") {
    handleAsync(C, 3, Result);
  } else if (NameRef == "cudaMemset") {
    std::string AsyncQueue;
    bool NeedTypeCast = false;
    if (C->getNumArgs() > 3 && !C->getArg(3)->isDefaultArgument()) {
      if (auto ICE = dyn_cast<ImplicitCastExpr>(C->getArg(3)))
        NeedTypeCast = ICE->getCastKind() != clang::CK_LValueToRValue;

      if (!isDefaultStream(C->getArg(3)))
        AsyncQueue = ExprAnalysis::ref(C->getArg(3));
    }
    replaceMemAPIArg(C->getArg(0), AsyncQueue);
    if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
      if (IsAsync) {
        emplaceTransformation(removeArg(C, 3, *Result.SourceManager));
      } else {
        emplaceTransformation(new InsertAfterStmt(C, ".wait()"));
      }
      if (AsyncQueue.empty()) {
        if (isPlaceholderIdxDuplicated(C))
          return;
        int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
        buildTempVariableMap(Index, C, HelperFuncType::HFT_DefaultQueue);
        ReplaceStr = "{{NEEDREPLACEQ" + std::to_string(Index) + "}}.memset";
      } else {
        if (NeedTypeCast)
          AsyncQueue = buildString("((sycl::queue *)(", AsyncQueue, "))");

        ReplaceStr = AsyncQueue + "->memset";
      }
    } else {
      handleAsync(C, 3, Result);
    }
  }

  emplaceTransformation(new ReplaceCalleeName(C, std::move(ReplaceStr)));
}

void MemoryMigrationRule::getSymbolSizeMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  // Here only handle ordinary variable name reference, for accessing the
  // size of something residing on the device directly from host side should
  // not be possible.
  std::string Replacement;
  ExprAnalysis EA;
  EA.analyze(C->getArg(0));
  auto StmtStrArg0 = EA.getReplacedString();
  EA.analyze(C->getArg(1));
  auto StmtStrArg1 = EA.getReplacedString();

  requestFeature(HelperFeatureEnum::device_ext);
  Replacement = getDrefName(C->getArg(0)) + " = " + StmtStrArg1 + ".get_size()";
  emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
}

void MemoryMigrationRule::prefetchMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string FuncName;
  if (ULExpr) {
    FuncName = ULExpr->getName().getAsString();
  } else {
    FuncName = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(FuncName);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end() &&
      DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
    const SourceManager *SM = Result.SourceManager;
    std::string Replacement;
    ExprAnalysis EA;
    EA.analyze(C->getArg(0));
    auto StmtStrArg0 = EA.getReplacedString();
    EA.analyze(C->getArg(1));
    auto StmtStrArg1 = EA.getReplacedString();
    EA.analyze(C->getArg(2));
    auto StmtStrArg2 = EA.getReplacedString();
    std::string StmtStrArg3;
    if (C->getNumArgs() == 4 && !C->getArg(3)->isDefaultArgument()) {
      if (!isDefaultStream(C->getArg(3)))
        StmtStrArg3 = ExprAnalysis::ref(C->getArg(3));
    } else {
      StmtStrArg3 = "0";
    }

    // In clang "define NULL __null"
    if (StmtStrArg3 == "0" || StmtStrArg3 == "") {
      const auto Prefix = MapNames::getDpctNamespace() +
                          (StmtStrArg2 == "cudaCpuDeviceId"
                               ? "cpu_device()"
                               : "get_device(" + StmtStrArg2 + ")");
      requestFeature(HelperFeatureEnum::device_ext);
      Replacement = Prefix + "." +
                    DpctGlobalInfo::getDefaultQueueMemFuncName() + "()" +
                    (DpctGlobalInfo::useSYCLCompat() ? "->" : ".") +
                    "prefetch(" + StmtStrArg0 + "," + StmtStrArg1 + ")";
    } else {
      if (SM->getCharacterData(C->getArg(3)->getBeginLoc()) -
              SM->getCharacterData(C->getArg(3)->getEndLoc()) ==
          0) {
        Replacement =
            StmtStrArg3 + "->prefetch(" + StmtStrArg0 + "," + StmtStrArg1 + ")";
      } else {
        Replacement = "(" + StmtStrArg3 + ")->prefetch(" + StmtStrArg0 + "," +
                      StmtStrArg1 + ")";
      }
    }
    emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
  } else {
    report(C->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, FuncName);
  }
}

void MemoryMigrationRule::miscMigration(const MatchFinder::MatchResult &Result,
                                        const CallExpr *C,
                                        const UnresolvedLookupExpr *ULExpr,
                                        bool IsAssigned) {
  std::string Name;
  if (ULExpr) {
    Name = ULExpr->getName().getAsString();
  } else {
    Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  }

  if (Name == "cudaHostGetDevicePointer" ||
      Name == "cuMemHostGetDevicePointer_v2") {
    if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
      ExprAnalysis EA(C);
      auto LocInfo = DpctGlobalInfo::getLocInfo(C->getBeginLoc());
      auto Info = std::make_shared<PriorityReplInfo>();
      if (auto TM = EA.getReplacement())
        Info->Repls.push_back(TM->getReplacement(DpctGlobalInfo::getContext()));
      Info->Repls.insert(Info->Repls.end(), EA.getSubExprRepl().begin(),
                         EA.getSubExprRepl().end());
      DpctGlobalInfo::addPriorityReplInfo(
          LocInfo.first.getCanonicalPath().str() + std::to_string(LocInfo.second), Info);
      // Insert header here since PriorityReplInfo delay the replacement
      // addation to the post process. At that time, the MainFile is invalid.
      DpctGlobalInfo::getInstance().insertHeader(C->getBeginLoc(),
                                                 HeaderType::HT_SYCL);
    } else {
      report(C->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             MapNames::ITFName.at(Name));
    }
  } else if (Name == "make_cudaExtent" || Name == "make_cudaPos") {
    std::string CtorName;
    llvm::raw_string_ostream OS(CtorName);
    DpctGlobalInfo::printCtadClass(
        OS,
        buildString(MapNames::getClNamespace(),
                    (Name == "make_cudaPos") ? "id" : "range"),
        3);
    emplaceTransformation(new ReplaceCalleeName(C, std::move(OS.str())));
  } else if (Name == "cudaGetChannelDesc") {
    std::ostringstream OS;
    printDerefOp(OS, C->getArg(0));
    OS << " = " << ExprAnalysis::ref(C->getArg(1)) << "->get_channel()";
    emplaceTransformation(new ReplaceStmt(C, OS.str()));
    requestFeature(HelperFeatureEnum::device_ext);
  } else if (Name == "cuMemGetInfo_v2" || Name == "cudaMemGetInfo") {
    if (DpctGlobalInfo::useDeviceInfo()) {
      std::ostringstream OS;
      if (IsAssigned)
        OS << MapNames::getCheckErrorMacroName() + "(";
      OS << MapNames::getDpctNamespace() + "get_current_device().get_memory_info";
      OS << "(";
      printDerefOp(OS, C->getArg(0));
      OS << ", ";
      printDerefOp(OS, C->getArg(1));
      OS << ")";

      emplaceTransformation(new ReplaceStmt(C, OS.str()));
      if (IsAssigned) {
        OS << ")";
      }
      emplaceTransformation(new ReplaceStmt(C, OS.str()));
      requestFeature(HelperFeatureEnum::device_ext);
      report(C->getBeginLoc(), Diagnostics::EXTENSION_DEVICE_INFO, false,
             Name == "cuMemGetInfo_v2" ? "cuMemGetInfo" : Name);
    } else {
      auto &SM = DpctGlobalInfo::getSourceManager();
      std::ostringstream OS;
      if (IsAssigned)
        OS << MapNames::getCheckErrorMacroName() + "(";

      auto SecondArg = C->getArg(1);
      printDerefOp(OS, SecondArg);
      OS << " = " << MapNames::getDpctNamespace()
         << "get_current_device().get_device_info()"
            ".get_global_mem_size()";
      requestFeature(HelperFeatureEnum::device_ext);
      if (IsAssigned) {
        OS << ")";
      }
      SourceLocation CallBegin(C->getBeginLoc());
      SourceLocation CallEnd(C->getEndLoc());

      bool IsMacroArg =
          SM.isMacroArgExpansion(CallBegin) && SM.isMacroArgExpansion(CallEnd);

      if (CallBegin.isMacroID() && IsMacroArg) {
        CallBegin = SM.getImmediateSpellingLoc(CallBegin);
        CallBegin = SM.getExpansionLoc(CallBegin);
      } else if (CallBegin.isMacroID()) {
        CallBegin = SM.getExpansionLoc(CallBegin);
      }

      if (CallEnd.isMacroID() && IsMacroArg) {
        CallEnd = SM.getImmediateSpellingLoc(CallEnd);
        CallEnd = SM.getExpansionLoc(CallEnd);
      } else if (CallEnd.isMacroID()) {
        CallEnd = SM.getExpansionLoc(CallEnd);
      }
      CallEnd = CallEnd.getLocWithOffset(1);

      emplaceTransformation(replaceText(CallBegin, CallEnd, OS.str(), SM));
      report(C->getBeginLoc(), Diagnostics::UNSUPPORT_FREE_MEMORY_SIZE, false);
    }
  } else {
    auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(Name);
    if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
      ExprAnalysis EA(C);
      emplaceTransformation(EA.getReplacement());
      EA.applyAllSubExprRepl();
      return;
    }
  }
}

void MemoryMigrationRule::cudaArrayGetInfo(
    const MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  std::string IndentStr =
      getIndent(C->getBeginLoc(), *Result.SourceManager).str();
  if (IsAssigned)
    IndentStr += "  ";
  std::ostringstream OS;
  std::string Arg3Str = ExprAnalysis::ref(C->getArg(3));
  printDerefOp(OS, C->getArg(0));
  OS << " = " << Arg3Str << "->get_channel();" << getNL() << IndentStr;
  printDerefOp(OS, C->getArg(1));
  OS << " = " << Arg3Str << "->get_range();" << getNL() << IndentStr;
  printDerefOp(OS, C->getArg(2));
  OS << " = 0";
  emplaceTransformation(new ReplaceStmt(C, OS.str()));
  requestFeature(HelperFeatureEnum::device_ext);
}

void MemoryMigrationRule::cudaMemAdvise(const MatchFinder::MatchResult &Result,
                                        const CallExpr *C,
                                        const UnresolvedLookupExpr *ULExpr,
                                        bool IsAssigned) {
  auto FuncName = C->getCalleeDecl()->getAsFunction()->getNameAsString();
  // Do nothing if USM is disabled
  if (DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_None) {
    report(C->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, FuncName);
    return;
  }

  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(FuncName);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end() &&
      DpctGlobalInfo::getUsmLevel() ==  UsmLevel::UL_Restricted) {
    ExprAnalysis EA(C);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  auto Arg2Expr = C->getArg(2);
  if (auto NamedCaster = dyn_cast<ExplicitCastExpr>(Arg2Expr)) {
    if (NamedCaster->getTypeAsWritten()->isIntegerType()) {
      Arg2Expr = NamedCaster->getSubExpr();
    } else if (DpctGlobalInfo::getUnqualifiedTypeName(
                   NamedCaster->getTypeAsWritten()) == "cudaMemoryAdvise" &&
               NamedCaster->getSubExpr()->getType()->isIntegerType()) {
      Arg2Expr = NamedCaster->getSubExpr();
    }
  }
  auto Arg0Str = ExprAnalysis::ref(C->getArg(0));
  auto Arg1Str = ExprAnalysis::ref(C->getArg(1));
  auto Arg3Str = ExprAnalysis::ref(C->getArg(3));

  std::string Arg2Str;
  if (Arg2Expr->getStmtClass() == Stmt::IntegerLiteralClass) {
    Arg2Str = "0";
  } else {
    Arg2Str = ExprAnalysis::ref(Arg2Expr);
  }

  if (Arg2Str == "0") {
    report(C->getBeginLoc(), Diagnostics::DEFAULT_MEM_ADVICE, false,
           " and was set to 0");
  } else {
    report(C->getBeginLoc(), Diagnostics::DEFAULT_MEM_ADVICE, false, "");
  }

  std::ostringstream OS;
  if (getStmtSpelling(C->getArg(3)) == "cudaCpuDeviceId") {
    OS << MapNames::getDpctNamespace() + "cpu_device()." +
              DpctGlobalInfo::getDefaultQueueMemFuncName() + "()";
    OS << (DpctGlobalInfo::useSYCLCompat() ? "->" : ".") << "mem_advise("
       << Arg0Str << ", " << Arg1Str << ", " << Arg2Str << ")";
    emplaceTransformation(new ReplaceStmt(C, OS.str()));
    requestFeature(HelperFeatureEnum::device_ext);
    return;
  }
  OS << MapNames::getDpctNamespace() + "get_device(" << Arg3Str
     << ")." + DpctGlobalInfo::getDefaultQueueMemFuncName() + "()";
  OS << (DpctGlobalInfo::useSYCLCompat() ? "->" : ".") << "mem_advise("
     << Arg0Str << ", " << Arg1Str << ", " << Arg2Str << ")";
  emplaceTransformation(new ReplaceStmt(C, OS.str()));
  requestFeature(HelperFeatureEnum::device_ext);
}

// Memory migration rules live here.
void MemoryMigrationRule::registerMatcher(MatchFinder &MF) {
  auto memoryAPI = [&]() {
    return hasAnyName(
        "cudaMalloc", "cudaMemcpy", "cudaMemcpyAsync", "cudaMemcpyToSymbol",
        "cudaMemcpyToSymbolAsync", "cudaMemcpyFromSymbol",
        "cudaMemcpyFromSymbolAsync", "cudaFree", "cudaMemset",
        "cudaMemsetAsync", "cublasFree", "cublasAlloc", "cudaGetSymbolAddress",
        "cudaFreeHost", "cudaHostAlloc", "cudaHostGetDevicePointer",
        "cudaHostRegister", "cudaHostUnregister", "cudaMallocHost",
        "cudaMallocManaged", "cudaGetSymbolSize", "cudaMemPrefetchAsync",
        "cudaMalloc3D", "cudaMallocPitch", "cudaMemset2D", "cudaMemset3D",
        "cudaMemset2DAsync", "cudaMemset3DAsync", "cudaMemcpy2D",
        "cudaMemcpy3D", "cudaMemcpy2DAsync", "cudaMemcpy3DAsync",
        "cudaMemcpy3DPeer", "cudaMemcpy3DPeerAsync", "cudaMemcpy2DArrayToArray",
        "cudaMemcpy2DToArray", "cudaMemcpy2DToArrayAsync",
        "cudaMemcpy2DFromArray", "cudaMemcpy2DFromArrayAsync",
        "cudaMemcpyArrayToArray", "cudaMemcpyToArray", "cudaMemcpyToArrayAsync",
        "cudaMemcpyFromArray", "cudaMemcpyFromArrayAsync", "cudaMallocArray",
        "cudaMalloc3DArray", "cudaFreeArray", "cudaArrayGetInfo",
        "cudaHostGetFlags", "cudaMemAdvise", "cuMemAdvise",
        "cudaGetChannelDesc", "cuMemHostAlloc", "cuMemFreeHost",
        "cuMemGetInfo_v2", "cuMemAlloc_v2", "cuMemcpyHtoD_v2",
        "cuMemcpyDtoH_v2", "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2",
        "cuMemcpy2D_v2", "cuMemcpy2DAsync_v2", "cuMemcpy3D_v2",
        "cuMemcpy3DAsync_v2", "cuMemcpy3DPeer", "cuMemcpy3DPeerAsync",
        "cudaMemGetInfo", "cuMemAllocManaged", "cuMemAllocHost_v2",
        "cuMemHostGetDevicePointer_v2", "cuMemcpyDtoDAsync_v2",
        "cuMemcpyDtoD_v2", "cuMemAllocPitch_v2", "cuMemPrefetchAsync",
        "cuMemFree_v2", "cuDeviceTotalMem_v2", "cuMemHostGetFlags",
        "cuMemHostRegister_v2", "cuMemHostUnregister", "cuMemcpy",
        "cuMemcpyAsync", "cuMemcpyHtoA_v2", "cuMemcpyAtoH_v2",
        "cuMemcpyHtoAAsync_v2", "cuMemcpyAtoHAsync_v2", "cuMemcpyDtoA_v2",
        "cuMemcpyAtoD_v2", "cuMemcpyAtoA_v2", "cuMemsetD16_v2",
        "cuMemsetD16Async", "cuMemsetD2D16_v2", "cuMemsetD2D16Async",
        "cuMemsetD2D32_v2", "cuMemsetD2D32Async", "cuMemsetD2D8_v2",
        "cuMemsetD2D8Async", "cuMemsetD32_v2", "cuMemsetD32Async",
        "cuMemsetD8_v2", "cuMemsetD8Async", "cudaMallocMipmappedArray",
        "cudaGetMipmappedArrayLevel", "cudaFreeMipmappedArray",
        "cudaMemcpyPeer", "cudaMemcpyPeerAsync", "cuMemcpyPeer",
        "cuMemcpyPeerAsync");
  };

  MF.addMatcher(callExpr(allOf(callee(functionDecl(memoryAPI())), parentStmt()))
                    .bind("call"),
                this);

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(memoryAPI())), unless(parentStmt())))
          .bind("callUsed"),
      this);

  MF.addMatcher(
      unresolvedLookupExpr(
          hasAnyDeclaration(namedDecl(memoryAPI())),
          hasParent(callExpr(unless(parentStmt())).bind("callExprUsed")))
          .bind("unresolvedCallUsed"),
      this);

  MF.addMatcher(
      unresolvedLookupExpr(hasAnyDeclaration(namedDecl(memoryAPI())),
                           hasParent(callExpr(parentStmt()).bind("callExpr")))
          .bind("unresolvedCall"),
      this);
}

void MemoryMigrationRule::runRule(const MatchFinder::MatchResult &Result) {
  auto MigrateCallExpr = [&](const CallExpr *C, const bool IsAssigned,
                             const UnresolvedLookupExpr *ULExpr = NULL) {
    if (!C)
      return;

    std::string Name;
    if (ULExpr && C) {
      Name = ULExpr->getName().getAsString();
    } else {
      Name = C->getCalleeDecl()->getAsFunction()->getNameAsString();
    }
    if (MigrationDispatcher.find(Name) == MigrationDispatcher.end())
      return;

    // If there is a malloc function call in a template function, and the
    // template function is implicitly instantiated with two types. Then there
    // will be three FunctionDecl nodes in the AST. We should do replacement on
    // the FunctionDecl node which is not implicitly instantiated.
    auto &Context = dpct::DpctGlobalInfo::getContext();
    auto Parents = Context.getParents(*C);
    while (Parents.size() == 1) {
      auto *Parent = Parents[0].get<FunctionDecl>();
      if (Parent) {
        if (Parent->getTemplateSpecializationKind() ==
                TSK_ExplicitSpecialization ||
            Parent->getTemplateSpecializationKind() == TSK_Undeclared)
          break;
        else
          return;
      } else {
        Parents = Context.getParents(Parents[0]);
      }
    }

    MigrationDispatcher.at(Name)(Result, C, ULExpr, IsAssigned);
    // if API is removed, then no need to add (*, 0)
    // There are some cases where (*, 0) has already been added.
    // If the API is processed with rewriter in APINamesMemory.inc,
    // need to exclude the API from additional processing.
    if (IsAssigned && Name.compare("cudaHostRegister") &&
        Name.compare("cudaHostUnregister") && Name.compare("cudaMemAdvise") &&
        Name.compare("cudaArrayGetInfo") && Name.compare("cudaMalloc") &&
        Name.compare("cudaMallocPitch") && Name.compare("cudaMalloc3D") &&
        Name.compare("cublasAlloc") && Name.compare("cuMemGetInfo_v2") &&
        Name.compare("cudaHostAlloc") && Name.compare("cudaMallocHost") &&
        Name.compare("cuMemHostAlloc") && Name.compare("cudaMemGetInfo") &&
        Name.compare("cudaMallocManaged") &&
        Name.compare("cuMemAllocManaged") &&
        Name.compare("cuMemAllocHost_v2") &&
        Name.compare("cudaHostGetDevicePointer") &&
        Name.compare("cuMemHostGetDevicePointer_v2") &&
        Name.compare("cuMemcpyDtoDAsync_v2") &&
        Name.compare("cuMemcpyDtoD_v2") && Name.compare("cuMemAdvise") &&
        Name.compare("cuMemPrefetchAsync") &&
        Name.compare("cuMemcpyHtoDAsync_v2") &&
        Name.compare("cuMemcpyDtoD_v2") &&
        Name.compare("cuMemHostUnregister") &&
        Name.compare("cuMemHostRegister_v2") &&
        Name.compare("cudaHostGetFlags") && Name.compare("cuMemHostGetFlags") &&
        Name.compare("cuMemcpy") && Name.compare("cuMemcpyAsync") &&
        Name.compare("cuMemAllocPitch_v2") && Name.compare("cuMemAlloc_v2") &&
        Name.compare("cudaMallocMipmappedArray") &&
        Name.compare("cudaGetMipmappedArrayLevel") &&
        Name.compare("cudaFreeMipmappedArray") && Name.compare("cudaMemcpy") &&
        Name.compare("cudaFree") && Name.compare("cublasFree")) {
      requestFeature(HelperFeatureEnum::device_ext);
      insertAroundStmt(C, MapNames::getCheckErrorMacroName() + "(", ")");
    } else if (IsAssigned && !Name.compare("cudaMemAdvise") &&
               DpctGlobalInfo::getUsmLevel() != UsmLevel::UL_None) {
      requestFeature(HelperFeatureEnum::device_ext);
      insertAroundStmt(C, MapNames::getCheckErrorMacroName() + "(", ")");
    } else if (IsAssigned && !Name.compare("cudaArrayGetInfo")) {
      requestFeature(HelperFeatureEnum::device_ext);
      std::string IndentStr =
          getIndent(C->getBeginLoc(), *Result.SourceManager).str();
      IndentStr += "  ";
      std::string PreStr{MapNames::getCheckErrorMacroName() + "([&](){"};
      PreStr += getNL();
      PreStr += IndentStr;
      std::string PostStr{";"};
      PostStr += getNL();
      PostStr += IndentStr;
      PostStr += "}())";
      insertAroundStmt(C, std::move(PreStr), std::move(PostStr));
    }
  };

  MigrateCallExpr(getAssistNodeAsType<CallExpr>(Result, "call"),
                  /* IsAssigned */ false);
  MigrateCallExpr(getAssistNodeAsType<CallExpr>(Result, "callUsed"),
                  /* IsAssigned */ true);
  MigrateCallExpr(
      getAssistNodeAsType<CallExpr>(Result, "callExprUsed"),
      /* IsAssigned */ true,
      getAssistNodeAsType<UnresolvedLookupExpr>(Result, "unresolvedCallUsed"));

  MigrateCallExpr(
      getAssistNodeAsType<CallExpr>(Result, "callExpr"),
      /* IsAssigned */ false,
      getAssistNodeAsType<UnresolvedLookupExpr>(Result, "unresolvedCall"));
}

void MemoryMigrationRule::getSymbolAddressMigration(
    const ast_matchers::MatchFinder::MatchResult &Result, const CallExpr *C,
    const UnresolvedLookupExpr *ULExpr, bool IsAssigned) {
  // Here only handle ordinary variable name reference, for accessing the
  // address of something residing on the device directly from host side should
  // not be possible.
  std::string Replacement;
  ExprAnalysis EA;
  EA.analyze(C->getArg(0));
  auto StmtStrArg0 = EA.getReplacedString();
  EA.analyze(C->getArg(1));
  auto StmtStrArg1 = EA.getReplacedString();
  Replacement = "*(" + StmtStrArg0 + ")" + " = " + StmtStrArg1 + ".get_ptr()";
  requestFeature(HelperFeatureEnum::device_ext);
  emplaceTransformation(new ReplaceStmt(C, std::move(Replacement)));
}

MemoryMigrationRule::MemoryMigrationRule() {
  std::map<
      std::string,
      std::function<void(MemoryMigrationRule *,
                         const ast_matchers::MatchFinder::MatchResult &,
                         const CallExpr *, const UnresolvedLookupExpr *, bool)>>
      Dispatcher{
          {"cudaMalloc", &MemoryMigrationRule::mallocMigration},
          {"cuMemAlloc_v2", &MemoryMigrationRule::mallocMigration},
          {"cudaHostAlloc", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocHost", &MemoryMigrationRule::mallocMigration},
          {"cuMemAllocHost_v2", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocManaged", &MemoryMigrationRule::mallocMigration},
          {"cuMemAllocManaged", &MemoryMigrationRule::mallocMigration},
          {"cublasAlloc", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocPitch", &MemoryMigrationRule::mallocMigration},
          {"cudaMalloc3D", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocArray", &MemoryMigrationRule::mallocMigration},
          {"cudaMallocMipmappedArray", &MemoryMigrationRule::mallocMigration},
          {"cudaMalloc3DArray", &MemoryMigrationRule::mallocMigration},
          {"cudaMemcpy", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyHtoD_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyDtoH_v2", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpyAsync", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyDtoHAsync_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyHtoDAsync_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyDtoDAsync_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyDtoD_v2", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpyToSymbol", &MemoryMigrationRule::memcpySymbolMigration},
          {"cudaMemcpyToSymbolAsync",
           &MemoryMigrationRule::memcpySymbolMigration},
          {"cudaMemcpyFromSymbol", &MemoryMigrationRule::memcpySymbolMigration},
          {"cudaMemcpyFromSymbolAsync",
           &MemoryMigrationRule::memcpySymbolMigration},
          {"cudaMemcpy2D", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpy2D_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpy2DAsync_v2", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy3D", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy3DPeer", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy3DPeerAsync", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpy3D_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpy3DAsync_v2", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpy3DPeer", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpy3DPeerAsync", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy2DAsync", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpy3DAsync", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpyPeer", &MemoryMigrationRule::memcpyMigration},
          {"cudaMemcpyPeerAsync", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyPeer", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyPeerAsync", &MemoryMigrationRule::memcpyMigration},
          {"cudaGetMipmappedArrayLevel", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DArrayToArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DFromArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DFromArrayAsync", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DToArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpy2DToArrayAsync", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyArrayToArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyToArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyToArrayAsync", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyFromArray", &MemoryMigrationRule::arrayMigration},
          {"cudaMemcpyFromArrayAsync", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyAtoH_v2", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyHtoA_v2", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyAtoHAsync_v2", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyHtoAAsync_v2", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyAtoD_v2", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyDtoA_v2", &MemoryMigrationRule::arrayMigration},
          {"cuMemcpyAtoA_v2", &MemoryMigrationRule::arrayMigration},
          {"cudaFree", &MemoryMigrationRule::freeMigration},
          {"cuMemFree_v2", &MemoryMigrationRule::freeMigration},
          {"cudaFreeArray", &MemoryMigrationRule::freeMigration},
          {"cudaFreeMipmappedArray", &MemoryMigrationRule::freeMigration},
          {"cudaFreeHost", &MemoryMigrationRule::freeMigration},
          {"cuMemFreeHost", &MemoryMigrationRule::freeMigration},
          {"cublasFree", &MemoryMigrationRule::freeMigration},
          {"cudaMemset", &MemoryMigrationRule::memsetMigration},
          {"cudaMemsetAsync", &MemoryMigrationRule::memsetMigration},
          {"cudaMemset2D", &MemoryMigrationRule::memsetMigration},
          {"cudaMemset2DAsync", &MemoryMigrationRule::memsetMigration},
          {"cudaMemset3D", &MemoryMigrationRule::memsetMigration},
          {"cudaMemset3DAsync", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD16_v2", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD16Async", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD2D16_v2", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD2D16Async", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD2D32_v2", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD2D32Async", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD2D8_v2", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD2D8Async", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD32_v2", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD32Async", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD8_v2", &MemoryMigrationRule::memsetMigration},
          {"cuMemsetD8Async", &MemoryMigrationRule::memsetMigration},
          {"cudaGetSymbolAddress",
           &MemoryMigrationRule::getSymbolAddressMigration},
          {"cudaGetSymbolSize", &MemoryMigrationRule::getSymbolSizeMigration},
          {"cudaHostGetDevicePointer", &MemoryMigrationRule::miscMigration},
          {"cuMemHostGetDevicePointer_v2", &MemoryMigrationRule::miscMigration},
          {"cudaHostRegister", &MemoryMigrationRule::miscMigration},
          {"cudaHostUnregister", &MemoryMigrationRule::miscMigration},
          {"cuMemHostRegister_v2", &MemoryMigrationRule::miscMigration},
          {"cuMemHostUnregister", &MemoryMigrationRule::miscMigration},
          {"cuMemHostGetFlags", &MemoryMigrationRule::miscMigration},
          {"cudaMemPrefetchAsync", &MemoryMigrationRule::prefetchMigration},
          {"cuMemPrefetchAsync", &MemoryMigrationRule::prefetchMigration},
          {"cudaArrayGetInfo", &MemoryMigrationRule::cudaArrayGetInfo},
          {"cudaHostGetFlags", &MemoryMigrationRule::miscMigration},
          {"cudaMemAdvise", &MemoryMigrationRule::cudaMemAdvise},
          {"cuMemAdvise", &MemoryMigrationRule::cudaMemAdvise},
          {"cudaGetChannelDesc", &MemoryMigrationRule::miscMigration},
          {"cuMemHostAlloc", &MemoryMigrationRule::mallocMigration},
          {"cuMemAllocPitch_v2", &MemoryMigrationRule::mallocMigration},
          {"cuMemGetInfo_v2", &MemoryMigrationRule::miscMigration},
          {"cudaMemGetInfo", &MemoryMigrationRule::miscMigration},
          {"cuDeviceTotalMem_v2", &MemoryMigrationRule::miscMigration},
          {"cuMemcpy", &MemoryMigrationRule::memcpyMigration},
          {"cuMemcpyAsync", &MemoryMigrationRule::memcpyMigration}};

  for (auto &P : Dispatcher)
    MigrationDispatcher[P.first] =
        std::bind(P.second, this, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4);
}

/// Convert a raw pointer argument and a pitch argument to a dpct::pitched_data
/// constructor. If \p ExcludeSizeArg is true, the argument represents the
/// pitch size will not be included in the constructor.
/// e.g. (...data, pitch, ...) => (...dpct::pitched_data(data, pitch, pitch, 1),
/// ...).
/// If \p ExcludeSizeArg is true, e.g. (...data, ..., pitch, ...) =>
/// (...dpct::pitched_data(data, pitch, pitch, 1), ..., pitch, ...)
void MemoryMigrationRule::aggregatePitchedData(const CallExpr *C,
                                               size_t DataArgIndex,
                                               size_t SizeArgIndex,
                                               SourceManager &SM,
                                               bool ExcludeSizeArg) {
  if (C->getNumArgs() <= DataArgIndex || C->getNumArgs() <= SizeArgIndex)
    return;
  size_t EndArgIndex = SizeArgIndex;
  std::string PaddingArgs, SizeArg;
  llvm::raw_string_ostream PaddingOS(PaddingArgs);
  ArgumentAnalysis A(C->getArg(SizeArgIndex), false);
  A.analyze();
  SizeArg = A.getReplacedString();
  if (ExcludeSizeArg) {
    PaddingOS << ", " << SizeArg;
    EndArgIndex = DataArgIndex;
  }
  PaddingOS << ", " << SizeArg << ", 1";
  aggregateArgsToCtor(C, MapNames::getDpctNamespace() + "pitched_data",
                      DataArgIndex, EndArgIndex, PaddingOS.str(), SM);
  requestFeature(HelperFeatureEnum::device_ext);
}

/// Convert several arguments to a constructor of class \p ClassName.
/// e.g. (...width, height, ...) => (...sycl::range<3>(width, height, 1), ...)
void MemoryMigrationRule::aggregateArgsToCtor(
    const CallExpr *C, const std::string &ClassName, size_t StartArgIndex,
    size_t EndArgIndex, const std::string &PaddingArgs, SourceManager &SM) {
  auto EndLoc = getStmtExpansionSourceRange(C->getArg(EndArgIndex)).getEnd();
  EndLoc = EndLoc.getLocWithOffset(Lexer::MeasureTokenLength(
      EndLoc, SM, DpctGlobalInfo::getContext().getLangOpts()));
  insertAroundRange(
      getStmtExpansionSourceRange(C->getArg(StartArgIndex)).getBegin(), EndLoc,
      ClassName + "(", PaddingArgs + ")");
}

/// Convert several arguments to a 3D vector constructor, like id<3> or
/// range<3>.
/// e.g. (...width, height, ...) => (...sycl::range<3>(width, height, 1), ...)
void MemoryMigrationRule::aggregate3DVectorClassCtor(
    const CallExpr *C, StringRef ClassName, size_t StartArgIndex,
    StringRef DefaultValue, SourceManager &SM, size_t ArgsNum) {
  if (C->getNumArgs() <= StartArgIndex + ArgsNum - 1)
    return;
  std::string Class, Padding;
  llvm::raw_string_ostream ClassOS(Class), PaddingOS(Padding);
  ClassOS << MapNames::getClNamespace();
  DpctGlobalInfo::printCtadClass(ClassOS, ClassName, 3);
  for (size_t i = 0; i < 3 - ArgsNum; ++i) {
    PaddingOS << ", " << DefaultValue;
  }
  aggregateArgsToCtor(C, ClassOS.str(), StartArgIndex,
                      StartArgIndex + ArgsNum - 1, PaddingOS.str(), SM);
}

void MemoryMigrationRule::handleDirection(const CallExpr *C, unsigned i) {
  if (DpctGlobalInfo::useSYCLCompat()) {
    emplaceTransformation(removeArg(C, i, DpctGlobalInfo::getSourceManager()));
    return;
  }
  if (C->getNumArgs() > i && !C->getArg(i)->isDefaultArgument()) {
    if (auto DRE = dyn_cast<DeclRefExpr>(C->getArg(i))) {
      if (auto Enum = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
        auto &ReplaceDirection = MapNames::findReplacedName(
            MapNames::EnumNamesMap, Enum->getName().str());
        if (!ReplaceDirection.empty()) {
          emplaceTransformation(new ReplaceStmt(DRE, ReplaceDirection));
          requestHelperFeatureForEnumNames(Enum->getName().str());
        }
      }
    }
  }
}

void MemoryMigrationRule::handleAsync(const CallExpr *C, unsigned i,
                                      const MatchFinder::MatchResult &Result) {
  if (C->getNumArgs() > i && !C->getArg(i)->isDefaultArgument()) {
    auto StreamExpr = C->getArg(i)->IgnoreImplicitAsWritten();
    if (isDefaultStream(StreamExpr)) {
      emplaceTransformation(removeArg(C, i, *Result.SourceManager));
      return;
    }
    emplaceTransformation(new InsertBeforeStmt(StreamExpr, "*"));
    if (!isa<DeclRefExpr>(StreamExpr)) {
      insertAroundStmt(StreamExpr, "(", ")");
    }
  }
}


const Expr *getRhs(const Stmt *);
TextModification *
ReplaceMemberAssignAsSetMethod(SourceLocation EndLoc, const MemberExpr *ME,
                               StringRef MethodName, StringRef ReplacedArg,
                               StringRef ExtraArg, StringRef ExtraFeild) {
  return new ReplaceToken(
      ME->getMemberLoc(), EndLoc,
      buildString(ExtraFeild + "set", MethodName.empty() ? "" : "_", MethodName,
                  "(", ExtraArg, ExtraArg.empty() ? "" : ", ", ReplacedArg,
                  ")"));
}

TextModification *
ReplaceMemberAssignAsSetMethod(const Expr *E, const MemberExpr *ME,
                               StringRef MethodName, StringRef ReplacedArg,
                               StringRef ExtraArg, StringRef ExtraFeild) {
  if (ReplacedArg.empty()) {
    if (auto RHS = getRhs(E)) {
      return ReplaceMemberAssignAsSetMethod(
          getStmtExpansionSourceRange(E).getEnd(), ME, MethodName,
          ExprAnalysis::ref(RHS), ExtraArg, ExtraFeild);
    }
  }
  return ReplaceMemberAssignAsSetMethod(getStmtExpansionSourceRange(E).getEnd(),
                                        ME, MethodName, ReplacedArg, ExtraArg);
}

void MemoryDataTypeRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      memberExpr(hasObjectExpression(declRefExpr(hasType(namedDecl(hasAnyName(
                     "CUDA_ARRAY_DESCRIPTOR", "CUDA_ARRAY3D_DESCRIPTOR"))))))
          .bind("arrayMember"),
      this);
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(declRefExpr(hasType(namedDecl(hasAnyName(
              "cudaMemcpy3DParms", "CUDA_MEMCPY3D", "cudaMemcpy3DPeerParms",
              "CUDA_MEMCPY3D_PEER", "CUDA_MEMCPY2D"))))))
          .bind("parmsMember"),
      this);
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(recordDecl(hasAnyName(
                               "cudaExtent", "cudaPos", "cudaPitchedPtr")))))
                    .bind("otherMember"),
                this);
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName("make_cudaExtent", "make_cudaPos",
                                              "make_cudaPitchedPtr"))))
          .bind("makeData"),
      this);
}

void MemoryDataTypeRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto ME = getNodeAsType<MemberExpr>(Result, "arrayMember")) {
    const auto *BO = DpctGlobalInfo::findParent<BinaryOperator>(ME);
    if (BO && BO->getOpcode() != BO_Assign) {
      BO = nullptr;
    }
    if (isRemove(ME->getMemberDecl()->getName().str())) {
      if (BO)
        return emplaceTransformation(new ReplaceStmt(BO, ""));
      return emplaceTransformation(new ReplaceStmt(ME, ""));
    }
    const auto &Replace = MapNames::findReplacedName(
        ArrayDescMemberNames, ME->getMemberDecl()->getName().str());
    if (!Replace.empty())
      emplaceTransformation(new ReplaceToken(
          ME->getMemberLoc(), ME->getEndLoc(), std::string(Replace)));
  } else if (auto CE = getNodeAsType<CallExpr>(Result, "makeData")) {
    if (auto FD = CE->getDirectCallee()) {
      auto Name = FD->getName();
      std::string ReplaceName;
      if (Name == "make_cudaExtent" || Name == "make_cudaPos" ||
          Name == "make_cudaPitchedPtr") {
        ExprAnalysis EA(CE);
        emplaceTransformation(EA.getReplacement());
        EA.applyAllSubExprRepl();
      } else {
        DpctDiags() << "Unexpected function name [" << Name
                    << "] in MemoryDataTypeRule";
      }
    }
  } else if (auto M = getNodeAsType<MemberExpr>(Result, "otherMember")) {
    auto BaseName =
        DpctGlobalInfo::getUnqualifiedTypeName(M->getBase()->getType());
    auto MemberName = M->getMemberDecl()->getName();
    if (BaseName == "cudaPos") {
      auto &Replace = MapNames::findReplacedName(
          MapNamesLang::Dim3MemberNamesMap, MemberName.str());
      if (!Replace.empty())
        emplaceTransformation(new ReplaceToken(
            M->getOperatorLoc(), M->getEndLoc(), std::string(Replace)));
    } else if (BaseName == "cudaExtent") {
      auto &Replace =
          MapNames::findReplacedName(ExtentMemberNames, MemberName.str());
      if (!Replace.empty())
        emplaceTransformation(new ReplaceToken(
            M->getOperatorLoc(), M->getEndLoc(), std::string(Replace)));
    } else if (BaseName == "cudaPitchedPtr") {
      auto &Replace =
          MapNames::findReplacedName(PitchMemberNames, MemberName.str());
      if (Replace.empty())
        return;
      static const std::unordered_map<std::string, HelperFeatureEnum>
          PitchMemberNameToSetFeatureMap = {
              {"pitch", HelperFeatureEnum::device_ext},
              {"ptr", HelperFeatureEnum::device_ext},
              {"xsize", HelperFeatureEnum::device_ext},
              {"ysize", HelperFeatureEnum::device_ext}};
      static const std::unordered_map<std::string, HelperFeatureEnum>
          PitchMemberNameToGetFeatureMap = {
              {"pitch", HelperFeatureEnum::device_ext},
              {"ptr", HelperFeatureEnum::device_ext},
              {"xsize", HelperFeatureEnum::device_ext},
              {"ysize", HelperFeatureEnum::device_ext}};
      if (auto BO = DpctGlobalInfo::findParent<BinaryOperator>(M)) {
        if (BO->getOpcode() == BO_Assign) {
          requestFeature(PitchMemberNameToSetFeatureMap.at(MemberName.str()));
          emplaceTransformation(ReplaceMemberAssignAsSetMethod(BO, M, Replace));
          return;
        }
      }
      emplaceTransformation(new ReplaceToken(
          M->getMemberLoc(), buildString("get_", Replace, "()")));
      requestFeature(PitchMemberNameToGetFeatureMap.at(MemberName.str()));
    }
  } else if (auto M = getNodeAsType<MemberExpr>(Result, "parmsMember")) {
    auto MemberName = M->getMemberDecl()->getName();
    const auto *BO = DpctGlobalInfo::findParent<BinaryOperator>(M);
    if (BO && BO->getOpcode() != BO_Assign) {
      BO = nullptr;
    }
    if (isRemove(MemberName.str())) {
      if (BO)
        return emplaceTransformation(new ReplaceStmt(BO, ""));
      return emplaceTransformation(new ReplaceStmt(M, ""));
    }
    auto Replace =
        MapNames::findReplacedName(DirectReplMemberNames, MemberName.str());
    if (DpctGlobalInfo::useExtBindlessImages() &&
        Replace.find("image") != std::string::npos)
      Replace += "_bindless";
    // TODO: Need remove these code when sycl compat updated.
    if (DpctGlobalInfo::useSYCLCompat()) {
      if (MemberName == "WidthInBytes")
        Replace = "size[0]";
      else if (MemberName == "dstXInBytes")
        Replace = "to.pos[0]";
      else if (MemberName == "srcXInBytes")
        Replace = "from.pos[0]";
    }
    if (MemberName.contains("Device") && M->getType().getAsString() != "int") {
      // The field srcDevice/dstDevice has different meaning in different struct
      // type.
      Replace.clear();
    }
    if (!Replace.empty())
      return emplaceTransformation(new ReplaceToken(
          M->getMemberLoc(), M->getEndLoc(), std::string(Replace)));
    Replace =
        MapNames::findReplacedName(GetSetReplMemberNames, MemberName.str());
    const std::string ExtraFeild =
        MemberName.starts_with("src") ? "from.pitched." : "to.pitched.";
    if (BO) {
      return emplaceTransformation(
          ReplaceMemberAssignAsSetMethod(BO, M, Replace, "", "", ExtraFeild));
    }
    emplaceTransformation(new ReplaceToken(
        M->getMemberLoc(), buildString(ExtraFeild + "get_", Replace, "()")));
  }
}


void UnnamedTypesRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(typedefDecl(hasDescendant(loc(recordType(hasDeclaration(
                    cxxRecordDecl(unless(anyOf(has(cxxRecordDecl(isImplicit())),
                                               isImplicit())),
                                  hasDefinition())
                        .bind("unnamedType")))))),
                this);
}

void UnnamedTypesRule::runRule(const MatchFinder::MatchResult &Result) {
  auto D = getNodeAsType<CXXRecordDecl>(Result, "unnamedType");
  if (D && D->getName().empty()) {
    if (DpctGlobalInfo::isCodePinEnabled()) {
      emplaceTransformation(new InsertClassName(D, RT_CUDAWithCodePin));
    }
    emplaceTransformation(new InsertClassName(D, RT_ForSYCLMigration));
  }
}


void TypeMmberRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(typedefDecl().bind("TypeDef"), this);
}

std::optional<SourceLocation>
TypeMmberRule::findTokenEndBeforeColonColon(SourceLocation TokStart,
                                            const SourceManager &SM) {
  bool Invalid = false;
  const char *OriginTokPtr = SM.getCharacterData(TokStart, &Invalid);
  const char *TokPtr = OriginTokPtr;

  if (Invalid)
    return std::nullopt;

  bool FoundColonColon = false;
  // Find coloncolon
  while (TokPtr && (TokPtr - 1)) {
    if (*TokPtr == ':' && *(TokPtr - 1) == ':') {
      TokPtr = TokPtr - 2;
      FoundColonColon = true;
      break;
    }
    --TokPtr;
  }

  if (!FoundColonColon)
    return std::nullopt;

  // Find previous non-space char
  while (TokPtr) {
    if (*TokPtr != ' ') {
      break;
    }
    --TokPtr;
  }

  if (!TokPtr)
    return std::nullopt;

  unsigned int Length = OriginTokPtr - TokPtr - 1;
  return TokStart.getLocWithOffset(-Length);
}

void TypeMmberRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const TypedefDecl *TD =
          getAssistNodeAsType<TypedefDecl>(Result, "TypeDef")) {
    const FunctionDecl *FD = DpctGlobalInfo::findAncestor<FunctionDecl>(TD);
    const ClassTemplateSpecializationDecl *CTSD =
        DpctGlobalInfo::findAncestor<ClassTemplateSpecializationDecl>(TD);
    if ((FD && FD->isTemplateInstantiation()) || CTSD) {
      if (const ElaboratedType *ET =
              dyn_cast<ElaboratedType>(TD->getUnderlyingType().getTypePtr())) {
        if (const TypedefType *TT = dyn_cast<TypedefType>(ET->desugar())) {
          std::string TypeStr =
              DpctGlobalInfo::getOriginalTypeName(QualType(TT, 0));
          StringRef TypeStrRef(TypeStr);
          if (TypeStrRef.starts_with("thrust::detail::cons<") &&
              TypeStrRef.ends_with("::head_type")) {
            const auto &SM = DpctGlobalInfo::getSourceManager();
            const auto &LangOpts = DpctGlobalInfo::getContext().getLangOpts();
            auto DefinitionSR = getDefinitionRange(
                TD->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                TD->getTypeSourceInfo()->getTypeLoc().getEndLoc());
            SourceLocation BeginLoc = DefinitionSR.getBegin();
            SourceLocation EndLoc = DefinitionSR.getEnd();
            auto EndLocBeforeColonColonOpt =
                findTokenEndBeforeColonColon(EndLoc, SM);
            if (!EndLocBeforeColonColonOpt)
              return;
            SourceLocation EndLocBeforeColonColon =
                EndLocBeforeColonColonOpt.value();
            StringRef OriginTypeStr = Lexer::getSourceText(
                Lexer::getAsCharRange(
                    SourceRange(BeginLoc, EndLocBeforeColonColon), SM,
                    LangOpts),
                SM, LangOpts);
            std::string Repl =
                "typename std::tuple_element_t<0, " + OriginTypeStr.str() + ">";
            EndLoc = EndLoc.getLocWithOffset(
                Lexer::MeasureTokenLength(EndLoc, SM, LangOpts));
            emplaceTransformation(
                new ReplaceText(BeginLoc, EndLoc, std::move(Repl)));
          }
        }
      }
    }
  }
}


void CMemoryAPIRule::registerMatcher(MatchFinder &MF) {
  auto cMemoryAPI = [&]() { return hasAnyName("calloc", "realloc", "malloc"); };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(cMemoryAPI())),
                     hasParent(implicitCastExpr().bind("implicitCast")))),
      this);
}

void CMemoryAPIRule::runRule(const MatchFinder::MatchResult &Result) {
  auto ICE = getNodeAsType<ImplicitCastExpr>(Result, "implicitCast");
  if (!ICE)
    return;
  auto Repl = new InsertText(
      ICE->getBeginLoc(),
      "(" + DpctGlobalInfo::getReplacedTypeName(ICE->getType()) + ")");
  Repl->setSYCLHeaderNeeded(false);
  emplaceTransformation(Repl);
}

void MathFunctionsRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> MathFunctionsCallExpr = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME) SOURCEAPINAME,
#define ENTRY_OPERATOR(APINAME, OPKIND) APINAME,
#define ENTRY_TYPECAST(APINAME) APINAME,
#define ENTRY_UNSUPPORTED(APINAME) APINAME,
#define ENTRY_REWRITE(APINAME) APINAME,
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

  std::vector<std::string> MathFunctionsUnresolvedLookupExpr = {
#define ENTRY_RENAMED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_NO_REWRITE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_SINGLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_RENAMED_DOUBLE(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_EMULATED(SOURCEAPINAME, TARGETAPINAME)
#define ENTRY_OPERATOR(APINAME, OPKIND)
#define ENTRY_TYPECAST(APINAME)
#define ENTRY_UNSUPPORTED(APINAME)
#define ENTRY_REWRITE(APINAME) APINAME,
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

  MF.addMatcher(
      callExpr(callee(functionDecl(
                   internal::Matcher<NamedDecl>(
                       new internal::HasNameMatcher(MathFunctionsCallExpr)),
                   anyOf(unless(hasDeclContext(namespaceDecl(anything()))),
                         hasDeclContext(namespaceDecl(hasName("std")))))))
          .bind("math"),
      this);

  MF.addMatcher(
      callExpr(callee(unresolvedLookupExpr(hasAnyDeclaration(namedDecl(
                   internal::Matcher<NamedDecl>(new internal::HasNameMatcher(
                       MathFunctionsUnresolvedLookupExpr)))))))
          .bind("unresolved"),
      this);
}

void MathFunctionsRule::runRule(const MatchFinder::MatchResult &Result) {
   const CallExpr *CE = getAssistNodeAsType<CallExpr>(Result, "math");
   if (!CE)
     CE = getNodeAsType<CallExpr>(Result, "unresolved");
   if (!CE)
     return;

  ExprAnalysis EA(CE);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();

  auto FD = CE->getDirectCallee();
  // For CUDA file, nvcc can include math header files implicitly.
  // So we need add the cmath header file if the API is not from SDK
  // header.
  bool NeedInsertCmath = DpctGlobalInfo::getContext().getLangOpts().CUDA;
  if (FD) {
    std::string Name = FD->getNameInfo().getName().getAsString();
    if (Name == "__brev" || Name == "__brevll") {
      requestFeature(HelperFeatureEnum::device_ext);
    } else if (Name == "__byte_perm") {
      requestFeature(HelperFeatureEnum::device_ext);
    } else if (Name == "__ffs" || Name == "__ffsll") {
      requestFeature(HelperFeatureEnum::device_ext);
    }
    NeedInsertCmath = NeedInsertCmath && !math::IsDefinedInCUDA()(CE);
  }
  if (NeedInsertCmath) {
    DpctGlobalInfo::getInstance().insertHeader(CE->getBeginLoc(), HT_Math);
  }
}


void WarpFunctionsRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> WarpFunctions = {"__reduce_add_sync",
                                            "__reduce_min_sync",
                                            "__reduce_and_sync",
                                            "__reduce_or_sync",
                                            "__reduce_xor_sync",
                                            "__reduce_max_sync",
                                            "__shfl_up_sync",
                                            "__shfl_down_sync",
                                            "__shfl_sync",
                                            "__shfl_up",
                                            "__shfl_down",
                                            "__shfl",
                                            "__shfl_xor",
                                            "__shfl_xor_sync",
                                            "__all",
                                            "__all_sync",
                                            "__any",
                                            "__any_sync",
                                            "__ballot",
                                            "__ballot_sync",
                                            "__match_any_sync",
                                            "__match_all_sync",
                                            "__activemask"};

  MF.addMatcher(callExpr(callee(functionDecl(internal::Matcher<NamedDecl>(
                             new internal::HasNameMatcher(WarpFunctions)))),
                         hasAncestor(functionDecl().bind("ancestor")))
                    .bind("warp"),
                this);
}

void WarpFunctionsRule::runRule(const MatchFinder::MatchResult &Result) {
  auto CE = getNodeAsType<CallExpr>(Result, "warp");
  if (!CE)
    return;

  if (auto *CalleeDecl = CE->getDirectCallee()) {
    if (isUserDefinedDecl(CalleeDecl)) {
      return;
    }
  }

  ExprAnalysis EA(CE);
  emplaceTransformation(EA.getReplacement());
  EA.applyAllSubExprRepl();
}

void SyncThreadsRule::registerMatcher(MatchFinder &MF) {
  auto SyncAPI = [&]() {
    return hasAnyName("__syncthreads", "__threadfence_block", "__threadfence",
                      "__threadfence_system", "__syncthreads_and",
                      "__syncthreads_or", "__syncthreads_count", "__syncwarp",
                      "__barrier_sync");
  };
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(SyncAPI())), parentStmt(),
                     hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                                    hasAttr(attr::CUDAGlobal)))
                                     .bind("FuncDecl"))))
          .bind("SyncFuncCall"),
      this);
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(SyncAPI())), unless(parentStmt()),
                     hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                                    hasAttr(attr::CUDAGlobal)))
                                     .bind("FuncDeclUsed"))))
          .bind("SyncFuncCallUsed"),
      this);
}

void SyncThreadsRule::runRule(const MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "SyncFuncCall");
  const FunctionDecl *FD =
      getAssistNodeAsType<FunctionDecl>(Result, "FuncDecl");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "SyncFuncCallUsed")))
      return;
    FD = getAssistNodeAsType<FunctionDecl>(Result, "FuncDeclUsed");
    IsAssigned = true;
  }
  if (!FD)
    return;

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  if (FuncName == "__syncthreads" || FuncName == "__barrier_sync") {
    DpctGlobalInfo::registerNDItemUser(CE);
    const FunctionDecl *FD = nullptr;
    if (FD = getAssistNodeAsType<FunctionDecl>(Result, "FuncDecl")) {
      GroupFunctionCallInControlFlowAnalyzer A(DpctGlobalInfo::getContext());
      A.checkCallGroupFunctionInControlFlow(const_cast<FunctionDecl *>(FD));
      auto FnInfo = DeviceFunctionDecl::LinkRedecls(FD);
      if (!FnInfo)
        return;
      auto CallInfo = FnInfo->addCallee(CE);
      if (CallInfo->hasSideEffects())
        report(CE->getBeginLoc(), Diagnostics::CALL_GROUP_FUNC_IN_COND, false);
    }
  } else if (FuncName == "__threadfence_block") {
    std::string CLNS = MapNames::getClNamespace();
    std::string ReplStr = CLNS + "atomic_fence(" + CLNS +
                          "memory_order::acq_rel, " + CLNS +
                          "memory_scope::work_group" + ")";
    report(CE->getBeginLoc(), Diagnostics::MEMORY_ORDER_PERFORMANCE_TUNNING,
           true);
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "__threadfence") {
    std::string CLNS = MapNames::getClNamespace();
    std::string ReplStr = CLNS + "atomic_fence(" + CLNS +
                          "memory_order::acq_rel, " + CLNS +
                          "memory_scope::device" + ")";
    report(CE->getBeginLoc(), Diagnostics::MEMORY_ORDER_PERFORMANCE_TUNNING,
           true);
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "__threadfence_system") {
    std::string CLNS = MapNames::getClNamespace();
    std::string ReplStr = CLNS + "atomic_fence(" + CLNS +
                          "memory_order::acq_rel, " + CLNS +
                          "memory_scope::system" + ")";
    report(CE->getBeginLoc(), Diagnostics::MEMORY_ORDER_PERFORMANCE_TUNNING,
           true);
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "__syncthreads_and" ||
             FuncName == "__syncthreads_or" ||
             FuncName == "__syncthreads_count") {
    std::string ReplStr;
    if (IsAssigned) {
      ReplStr = "(";
      ReplStr += DpctGlobalInfo::getItem(CE) + ".barrier(), ";
    } else {
      ReplStr += DpctGlobalInfo::getItem(CE) + ".barrier();" + getNL();
      ReplStr += getIndent(CE->getBeginLoc(), *Result.SourceManager).str();
    }
    if (FuncName == "__syncthreads_and") {
      ReplStr += MapNames::getClNamespace() + "all_of_group(";
    } else if (FuncName == "__syncthreads_or") {
      ReplStr += MapNames::getClNamespace() + "any_of_group(";
    } else {
      ReplStr += MapNames::getClNamespace() + "reduce_over_group(";
    }
    ReplStr += DpctGlobalInfo::getGroup(CE) + ", ";
    if (FuncName == "__syncthreads_count") {
      ReplStr += ExprAnalysis::ref(CE->getArg(0)) + " == 0 ? 0 : 1, " +
                 MapNames::getClNamespace() + "ext::oneapi::plus<>()";
    } else {
      ReplStr += ExprAnalysis::ref(CE->getArg(0));
    }

    ReplStr += ")";
    if (IsAssigned)
      ReplStr += ")";
    report(CE->getBeginLoc(), Diagnostics::BARRIER_PERFORMANCE_TUNNING, true,
           "nd_item");
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  } else if (FuncName == "__syncwarp") {
    std::string ReplStr;
    ReplStr = MapNames::getClNamespace() + "group_barrier(" +
              DpctGlobalInfo::getSubGroup(CE) + ")";
    emplaceTransformation(new ReplaceStmt(CE, std::move(ReplStr)));
  }
}


void SyncThreadsMigrationRule::registerMatcher(MatchFinder &MF) {
  auto SyncAPI = [&]() {
    return hasAnyName("__syncthreads", "__barrier_sync");
  };
  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(SyncAPI())), parentStmt(),
                     hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                                    hasAttr(attr::CUDAGlobal)))
                                     .bind("FuncDecl"))))
          .bind("SyncFuncCall"),
      this);
}

void SyncThreadsMigrationRule::runRule(const MatchFinder::MatchResult &Result) {
  static std::map<std::string, bool> LocationResultMapForTemplate;
  auto emplaceReplacement = [&](BarrierFenceSpaceAnalyzerResult Res,
                                const CallExpr *CE) {
    std::string Replacement;
    if (Res.CanUseLocalBarrier) {
      if (Res.MayDependOn1DKernel) {
        report(CE->getBeginLoc(), Diagnostics::ONE_DIMENSION_KERNEL_BARRIER,
               true, Res.GlobalFunctionName);
      }
      Replacement = DpctGlobalInfo::getItem(CE) + ".barrier(" +
                    MapNames::getClNamespace() +
                    "access::fence_space::local_space)";
    } else if (Res.CanUseLocalBarrierWithCondition) {
      report(CE->getBeginLoc(), Diagnostics::ONE_DIMENSION_KERNEL_BARRIER, true,
             Res.GlobalFunctionName);
      Replacement =
          "(" + Res.Condition + ") ? " + DpctGlobalInfo::getItem(CE) +
          ".barrier(" + MapNames::getClNamespace() +
          "access::fence_space::local_space) : " + DpctGlobalInfo::getItem(CE) +
          ".barrier()";
    } else {
      report(CE->getBeginLoc(), Diagnostics::BARRIER_PERFORMANCE_TUNNING, true,
             "nd_item");
      Replacement = DpctGlobalInfo::getItem(CE) + ".barrier()";
    }
    emplaceTransformation(new ReplaceStmt(CE, std::move(Replacement)));
  };

  const CallExpr *CE = getAssistNodeAsType<CallExpr>(Result, "SyncFuncCall");
  const FunctionDecl *FD =
      getAssistNodeAsType<FunctionDecl>(Result, "FuncDecl");
  if (!CE || !FD)
    return;

  std::string FuncName =
      CE->getDirectCallee()->getNameInfo().getName().getAsString();
  if (FuncName == "__syncthreads" || FuncName == "__barrier_sync") {
    BarrierFenceSpaceAnalyzer A;
    const FunctionTemplateDecl *FTD = FD->getDescribedFunctionTemplate();
    if (FTD) {
      if (FTD->specializations().empty()) {
        emplaceReplacement(A.analyze(CE), CE);
      }
    } else {
      if (FD->getTemplateSpecializationKind() ==
          TemplateSpecializationKind::TSK_Undeclared) {
        emplaceReplacement(A.analyze(CE), CE);
      } else {
        auto CurRes = A.analyze(CE, true);
        std::string LocHash = getHashStrFromLoc(CE->getBeginLoc());
        auto Iter = LocationResultMapForTemplate.find(LocHash);
        if (Iter != LocationResultMapForTemplate.end()) {
          if (Iter->second != CurRes.CanUseLocalBarrier) {
            report(CE->getBeginLoc(),
                   Diagnostics::CANNOT_UNIFY_FUNCTION_CALL_IN_MACRO_OR_TEMPLATE,
                   false, FuncName);
          }
        } else {
          LocationResultMapForTemplate[LocHash] = CurRes.CanUseLocalBarrier;
          emplaceReplacement(CurRes, CE);
        }
      }
    }
  }
}


void KernelFunctionInfoRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(
      callExpr(callee(functionDecl(hasAnyName("cudaFuncGetAttributes"))))
          .bind("call"),
      this);
  MF.addMatcher(callExpr(callee(functionDecl(hasAnyName("cuFuncGetAttribute"))))
                    .bind("callFuncGetAttribute"),
                this);
  MF.addMatcher(
      memberExpr(anyOf(has(implicitCastExpr(hasType(pointsTo(
                           recordDecl(hasName("cudaFuncAttributes")))))),
                       hasObjectExpression(
                           hasType(recordDecl(hasName("cudaFuncAttributes"))))))
          .bind("member"),
      this);

  MF.addMatcher(callExpr(callee(functionDecl(hasName("cuFuncSetAttribute"))))
                    .bind("cuFuncSetAttribute"),
                this);
}

void KernelFunctionInfoRule::runRule(const MatchFinder::MatchResult &Result) {
  if (auto C = getNodeAsType<CallExpr>(Result, "call")) {
    if (isAssigned(C)) {
      emplaceTransformation(new ReplaceToken(
          C->getBeginLoc(), MapNames::getCheckErrorMacroName() + "(" + MapNames::getDpctNamespace() +
                                "get_kernel_function_info"));
      emplaceTransformation(new InsertAfterStmt(C, ")"));
    } else {
      emplaceTransformation(
          new ReplaceToken(C->getBeginLoc(), MapNames::getDpctNamespace() +
                                                 "get_kernel_function_info"));
    }
    requestFeature(HelperFeatureEnum::device_ext);
    auto FuncArg = C->getArg(1);
    emplaceTransformation(new InsertBeforeStmt(FuncArg, "(const void *)"));
  } else if (auto C = getNodeAsType<CallExpr>(Result, "callFuncGetAttribute")) {
    ExprAnalysis EA;
    EA.analyze(C);
    const auto *AttrArg = C->getArg(1);
    if (auto DRE = dyn_cast<DeclRefExpr>(AttrArg)) {
      if (auto AttrEnumConst = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
        std::string EnumName = AttrEnumConst->getName().str();
        std::string MemberName, Desc;
        if (EnumName == "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES") {
          MemberName = "shared_size_bytes";
          Desc = "statically allocated shared memory";
        } else if (EnumName == "CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES") {
          MemberName = "const_size_bytes";
          Desc = "memory size of user-defined constants";
        } else if (EnumName == "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES") {
          MemberName = "local_size_bytes";
          Desc = "local memory";
        } else if (EnumName == "CU_FUNC_ATTRIBUTE_NUM_REGS") {
          MemberName = "num_regs";
          Desc = "required number of registers";
        }
        if (!MemberName.empty() and !Desc.empty()) {
          report(C->getBeginLoc(), Diagnostics::UNSUPPORTED_KERNEL_ATTRIBUTE, false, Desc, MemberName);
        }
      }
    }
    emplaceTransformation(EA.getReplacement());
  } else if (auto M = getNodeAsType<MemberExpr>(Result, "member")) {
    auto MemberName = M->getMemberNameInfo();
    auto NameMap = AttributesNamesMap.find(MemberName.getAsString());
    if (NameMap != AttributesNamesMap.end())
      emplaceTransformation(new ReplaceToken(MemberName.getBeginLoc(),
                                             std::string(NameMap->second)));

  } else if (auto *CallNode =
                 getNodeAsType<CallExpr>(Result, "cuFuncSetAttribute")) {
    std::string FuncName =
        CallNode->getDirectCallee()->getNameInfo().getName().getAsString();
    if (auto Msg = MapNames::RemovedAPIWarningMessage.find(FuncName);
        Msg != MapNames::RemovedAPIWarningMessage.end()) {
      std::string CallReplacement{""};
      if (isAssigned(CallNode)) {
        CallReplacement = "0";
      }
      report(CallNode->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             MapNames::ITFName.at(FuncName), Msg->second);
      emplaceTransformation(new ReplaceStmt(CallNode, CallReplacement));
    }
  }
}


std::vector<std::vector<std::string>>
RecognizeAPINameRule::splitAPIName(std::vector<std::string> &AllAPINames) {
  std::vector<std::vector<std::string>> Result;
  std::vector<std::string> FuncNames, FuncNamesHasNS, FuncNamespaces;
  size_t ScopeResolutionOpSize = 2; // The length of string("::")
  for (auto &APIName : AllAPINames) {
    size_t ScopeResolutionOpPos = APIName.rfind("::");
    // 1. FunctionName
    if (ScopeResolutionOpPos == std::string::npos) {
      FuncNames.emplace_back(APIName);
    } else {
      // 2. NameSpace::FunctionName
      if (std::find(FuncNamespaces.begin(), FuncNamespaces.end(),
                    APIName.substr(0, ScopeResolutionOpPos)) ==
          FuncNamespaces.end()) {
        FuncNamespaces.emplace_back(APIName.substr(0, ScopeResolutionOpPos));
      }
      FuncNamesHasNS.emplace_back(
          APIName.substr(ScopeResolutionOpPos + ScopeResolutionOpSize));
    }
  }
  return {FuncNames, FuncNamesHasNS, FuncNamespaces};
}

void RecognizeAPINameRule::registerMatcher(MatchFinder &MF) {
  std::vector<std::string> AllAPINames = MigrationStatistics::GetAllAPINames();
  // AllAPIComponent[0] : FuncNames
  // AllAPIComponent[1] : FuncNamesHasNS
  // AllAPIComponent[2] : FuncNamespaces
  std::vector<std::vector<std::string>> AllAPIComponent =
      splitAPIName(AllAPINames);
  if (!AllAPIComponent[0].empty()) {
    MF.addMatcher(
        callExpr(
            allOf(callee(functionDecl(internal::Matcher<NamedDecl>(
                      new internal::HasNameMatcher(AllAPIComponent[0])))),
                  unless(hasAncestor(cudaKernelCallExpr())),
                  unless(callee(hasDeclContext(namedDecl(hasName("std")))))))
            .bind("APINamesUsed"),
        this);
  }

  if (!AllAPIComponent[1].empty() && !AllAPIComponent[2].empty()) {
    MF.addMatcher(
        callExpr(
            callee(functionDecl(allOf(
                namedDecl(internal::Matcher<NamedDecl>(
                    new internal::HasNameMatcher(AllAPIComponent[1]))),
                hasAncestor(
                    namespaceDecl(namedDecl(internal::Matcher<NamedDecl>(
                        new internal::HasNameMatcher(AllAPIComponent[2])))))))))
            .bind("APINamesHasNSUsed"),
        this);
  }
}

const std::string
RecognizeAPINameRule::getFunctionSignature(const FunctionDecl *Func,
                                           std::string ObjName) {
  std::string Buf;
  llvm::raw_string_ostream OS(Buf);
  OS << Func->getReturnType().getAsString() << " " << ObjName
     << Func->getQualifiedNameAsString() << "(";

  for (unsigned int Index = 0; Index < Func->getNumParams(); Index++) {
    if (Index > 0) {
      OS << ",";
    }
    OS << QualType::getAsString(Func->parameters()[Index]->getType().split(),
                                PrintingPolicy{{}})
       << " " << Func->parameters()[Index]->getQualifiedNameAsString();
  }
  OS << ")";
  return OS.str();
}

void RecognizeAPINameRule::processFuncCall(const CallExpr *CE) {
  const NamedDecl *ND;
  std::string Namespace = "";
  std::string ObjName = "";
  std::string APIName = "";
  if (dyn_cast<CXXOperatorCallExpr>(CE))
    return;
  if (auto MD = dyn_cast<CXXMemberCallExpr>(CE)) {
    QualType ObjType = MD->getImplicitObjectArgument()
                           ->IgnoreImpCasts()
                           ->getType()
                           .getCanonicalType();
    ND = getNamedDecl(ObjType.getTypePtr());
    if (!ND)
      return;
    ObjName = ND->getNameAsString();
  } else {
    // Match the static member function call, like: A a; a.staticCall();
    if (auto ME = dyn_cast<MemberExpr>(CE->getCallee()->IgnoreImpCasts())) {
      auto ObjType = ME->getBase()->getType().getCanonicalType();
      ND = getNamedDecl(ObjType.getTypePtr());
      if (!ND)
        return;
      ObjName = ND->getNameAsString();
    // Match the static call, like: A::staticCall();
    } else if (auto RT = dyn_cast<RecordDecl>(
                   CE->getCalleeDecl()->getDeclContext())) {
      ObjName = RT->getNameAsString();
      ND = dyn_cast<NamedDecl>(RT);
    } else {
      ND = dyn_cast<NamedDecl>(CE->getCalleeDecl());
    }
  }

  if (!dpct::DpctGlobalInfo::isInCudaPath(ND->getLocation()) &&
      !isChildOrSamePath(DpctInstallPath,
                         dpct::DpctGlobalInfo::getLocInfo(ND).first)) {
    if (ND->getIdentifier() && !ND->getName().starts_with("cudnn") &&
        !ND->getName().starts_with("nccl"))
      return;
  }

  recordRecognizedAPI(CE);
  auto *NSD = dyn_cast<NamespaceDecl>(ND->getDeclContext());
  Namespace = getNameSpace(NSD);
  APIName = CE->getCalleeDecl()->getAsFunction()->getNameAsString();
  if (!ObjName.empty())
    APIName = ObjName + "::" + APIName;
  if (!Namespace.empty())
    APIName = Namespace + "::" + APIName;
  SrcAPIStaticsMap[getFunctionSignature(CE->getCalleeDecl()->getAsFunction(),
                                        "")]++;

  if (!MigrationStatistics::IsMigrated(APIName)) {
    const SourceManager &SM = DpctGlobalInfo::getSourceManager();
    const SourceLocation FileLoc = SM.getFileLoc(CE->getBeginLoc());

    std::string SLStr = FileLoc.printToString(SM);

    std::size_t PosCol = SLStr.rfind(':');
    std::size_t PosRow = SLStr.rfind(':', PosCol - 1);
    std::string FileName = SLStr.substr(0, PosRow);
    LOCStaticsMap[FileName][2]++;

    auto Iter = MapNames::ITFName.find(APIName.c_str());
    if (Iter != MapNames::ITFName.end())
      report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             Iter->second);
    else
      report(CE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, APIName);
  }
}

void RecognizeAPINameRule::runRule(const MatchFinder::MatchResult &Result) {
  const CallExpr *CE = nullptr;
  if ((CE = getNodeAsType<CallExpr>(Result, "APINamesUsed")) ||
      (CE = getNodeAsType<CallExpr>(Result, "APINamesHasNSUsed")))
    processFuncCall(CE);
}


void RecognizeTypeRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto TypeTable = MigrationStatistics::GetTypeTable();
  std::vector<std::string> UnsupportedType;
  std::vector<std::string> UnsupportedPointerType;
  std::vector<std::string> AllTypes;
  for (auto &Type : TypeTable) {
    if (!Type.second) {
      if (Type.first.find("*") != std::string::npos) {
        UnsupportedPointerType.push_back(
            Type.first.substr(0, Type.first.length() - 1));
      } else {
        UnsupportedType.push_back(Type.first);
      }
    }
    if (DpctGlobalInfo::isAnalysisModeEnabled()) {
      AllTypes.push_back(Type.first);
    }
  }
  MF.addMatcher(
      typeLoc(
          anyOf(loc(qualType(
                    hasDeclaration(namedDecl(internal::Matcher<NamedDecl>(
                        new internal::HasNameMatcher(UnsupportedType)))))),
                loc(pointerType(pointee(qualType(hasDeclaration(namedDecl(
                    internal::Matcher<NamedDecl>(new internal::HasNameMatcher(
                        UnsupportedPointerType))))))))))
          .bind("unsupportedtypeloc"),
      this);

  if (DpctGlobalInfo::isAnalysisModeEnabled()) {
    MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(
                              namedDecl(internal::Matcher<NamedDecl>(
                                  new internal::HasNameMatcher(AllTypes)))))))
                      .bind("alltypeloc"),
                  this);
  }
}

void RecognizeTypeRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const TypeLoc* TL = getNodeAsType<TypeLoc>(Result, "unsupportedtypeloc")) {
    auto& Context = DpctGlobalInfo::getContext();
    QualType QTy = TL->getType();
    if (QTy.isCanonical())
      return;
    std::string TypeName =
      DpctGlobalInfo::getTypeName(QTy.getUnqualifiedType(), Context);
    // process pointer type
    if (!QTy->isTypedefNameType() && QTy->isPointerType()) {
      std::string PointeeTy = DpctGlobalInfo::getTypeName(
        QTy->getPointeeType().getUnqualifiedType(), Context);
      report(TL->getBeginLoc(), Diagnostics::KNOWN_UNSUPPORTED_TYPE, false,
        PointeeTy + " *");
      return;
    }
    report(TL->getBeginLoc(), Diagnostics::KNOWN_UNSUPPORTED_TYPE, false,
      TypeName);
    return;
  }
  if (const TypeLoc *TL = getNodeAsType<TypeLoc>(Result, "alltypeloc")) {
    recordRecognizedType(*TL);
  }
}

void CXXNewExprRule::registerMatcher(MatchFinder &MF) {
  MF.addMatcher(cxxNewExpr().bind("newExpr"), this);
}

void CXXNewExprRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto CNE = getAssistNodeAsType<CXXNewExpr>(Result, "newExpr")) {
    // E.g., new cudaEvent_t *()
    Token Tok;
    auto LOpts = Result.Context->getLangOpts();
    SourceManager *SM = Result.SourceManager;
    auto BeginLoc =
        CNE->getAllocatedTypeSourceInfo()->getTypeLoc().getBeginLoc();
    Lexer::getRawToken(BeginLoc, Tok, *SM, LOpts, true);
    if (Tok.isAnyIdentifier()) {
      std::string Str = MapNames::findReplacedName(
          MapNames::TypeNamesMap, Tok.getRawIdentifier().str());
      insertHeaderForTypeRule(Tok.getRawIdentifier().str(), BeginLoc);
      requestHelperFeatureForTypeNames(Tok.getRawIdentifier().str());

      SourceManager &SM = DpctGlobalInfo::getSourceManager();
      BeginLoc = SM.getExpansionLoc(BeginLoc);
      if (!Str.empty()) {
        emplaceTransformation(new ReplaceToken(BeginLoc, std::move(Str)));
        return;
      }
    }

    // E.g., #define NEW_STREAM new cudaStream_t
    //      stream = NEW_STREAM;
    auto TypeName = CNE->getAllocatedType().getAsString();
    auto ReplName = std::string(
        MapNames::findReplacedName(MapNames::TypeNamesMap, TypeName));
    insertHeaderForTypeRule(TypeName, BeginLoc);
    requestHelperFeatureForTypeNames(TypeName);

    if (!ReplName.empty()) {
      auto BeginLoc =
          CNE->getAllocatedTypeSourceInfo()->getTypeLoc().getBeginLoc();
      emplaceTransformation(new ReplaceToken(BeginLoc, std::move(ReplName)));
    }
  }
}

void VirtualMemRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto virtualmemoryAPI = [&]() {
    return hasAnyName("cuMemCreate", "cuMemAddressReserve", "cuMemMap",
                      "cuMemUnmap", "cuMemAddressFree", "cuMemRelease",
                      "cuMemSetAccess", "cuMemGetAllocationGranularity",
                      "cuMemGetAllocationPropertiesFromHandle");
  };
  auto virtualmemoryType = [&]() {
    return hasAnyName("CUmemAllocationProp", "CUmemGenericAllocationHandle",
                      "CUmemAccessDesc", "CUmemLocationType",
                      "CUmemAllocationType", "CUmemAllocationGranularity_flags",
                      "CUmemAccess_flags");
  };
  auto virtualmemoryEnum = [&]() {
    return hasAnyName(
        "CU_MEM_ALLOCATION_TYPE_PINNED", "CU_MEM_ALLOCATION_TYPE_INVALID",
        "CU_MEM_ALLOCATION_TYPE_MAX", "CU_MEM_LOCATION_TYPE_DEVICE",
        "CU_MEM_LOCATION_TYPE_INVALID", "CU_MEM_LOCATION_TYPE_MAX",
        "CU_MEM_ACCESS_FLAGS_PROT_NONE", "CU_MEM_ACCESS_FLAGS_PROT_READ",
        "CU_MEM_ACCESS_FLAGS_PROT_READWRITE",
        "CU_MEM_ALLOC_GRANULARITY_RECOMMENDED",
        "CU_MEM_ALLOC_GRANULARITY_MINIMUM");
  };
  MF.addMatcher(
      callExpr(callee(functionDecl(virtualmemoryAPI()))).bind("vmCall"), this);
  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(virtualmemoryType())))))
          .bind("vmType"),
      this);
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(virtualmemoryEnum()))).bind("vmEnum"),
      this);
}

void VirtualMemRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "vmCall")) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
  if (auto TL = getNodeAsType<TypeLoc>(Result, "vmType")) {
    auto TypeStr =
        DpctGlobalInfo::getTypeName(TL->getType().getUnqualifiedType());
    if (!DpctGlobalInfo::useExpVirtualMemory()) {
      report(TL->getBeginLoc(), Diagnostics::TRY_EXPERIMENTAL_FEATURE, false,
             TypeStr, "--use-experimental-features=virtual_memory");
      return;
    }
    if (!DpctGlobalInfo::isInAnalysisScope(
            SM.getSpellingLoc(TL->getBeginLoc()))) {
      return;
    }
    auto Range = getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc());
    auto BeginLoc = Range.getBegin();
    auto EndLoc = Range.getEnd();

    if (SM.isWrittenInScratchSpace(SM.getSpellingLoc(TL->getBeginLoc()))) {
      BeginLoc = SM.getExpansionRange(TL->getBeginLoc()).getBegin();
      EndLoc = SM.getExpansionRange(TL->getBeginLoc()).getEnd();
    }
    std::string Str =
        MapNames::findReplacedName(MapNames::TypeNamesMap, TypeStr);
    if (!Str.empty()) {
      auto Len = Lexer::MeasureTokenLength(
          EndLoc, SM, DpctGlobalInfo::getContext().getLangOpts());
      Len += SM.getDecomposedLoc(EndLoc).second -
             SM.getDecomposedLoc(BeginLoc).second;
      emplaceTransformation(new ReplaceText(BeginLoc, Len, std::move(Str)));
      return;
    }
  }
  if (auto *E = getNodeAsType<DeclRefExpr>(Result, "vmEnum")) {
    std::string EnumName = E->getNameInfo().getName().getAsString();
    if (!DpctGlobalInfo::useExpVirtualMemory()) {
      report(E->getBeginLoc(), Diagnostics::TRY_EXPERIMENTAL_FEATURE, false,
             EnumName, "--use-experimental-features=virtual_memory");
      return;
    }
    auto Search = MapNames::EnumNamesMap.find(EnumName);
    if (Search == MapNames::EnumNamesMap.end()) {
      report(E->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false, EnumName);
      return;
    }
    emplaceTransformation(new ReplaceStmt(E, Search->second->NewName));
  }
}


void DriverModuleAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto DriverModuleAPI = [&]() {
    return hasAnyName("cuModuleLoad", "cuModuleLoadData", "cuModuleLoadDataEx",
                      "cuModuleUnload", "cuModuleGetFunction", "cuLaunchKernel",
                      "cuModuleGetTexRef");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(DriverModuleAPI())), parentStmt()))
          .bind("call"),
      this);

  MF.addMatcher(callExpr(allOf(callee(functionDecl(DriverModuleAPI())),
                               unless(parentStmt())))
                    .bind("callUsed"),
                this);
}

void DriverModuleAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "callUsed"))) {
      return;
    }
  }

  std::string APIName = "";
  if (auto DC = CE->getDirectCallee()) {
    APIName = DC->getNameAsString();
  } else {
    return;
  }

  if (APIName == "cuModuleLoad") {
    report(CE->getBeginLoc(), Diagnostics::MODULE_LOAD, false,
           getStmtSpelling(CE->getArg(1)));
  } else if (APIName == "cuModuleLoadData" || APIName == "cuModuleLoadDataEx") {
    report(CE->getBeginLoc(), Diagnostics::MODULE_LOAD_DATA, false,
           getStmtSpelling(CE->getArg(1)));
  }

  if (isAssigned(CE) &&
      (APIName == "cuModuleLoad" || APIName == "cuModuleLoadData" ||
       APIName == "cuModuleLoadDataEx" || APIName == "cuModuleGetFunction")) {
    requestFeature(HelperFeatureEnum::device_ext);
    insertAroundStmt(CE, MapNames::getCheckErrorMacroName() + "(", ")");
  }

  ExprAnalysis EA;
  EA.analyze(CE);
  emplaceTransformation(EA.getReplacement());
}


void DriverDeviceAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {

  auto DriverDeviceAPI = [&]() {
    return hasAnyName(
        "cuDeviceGet", "cuDeviceComputeCapability", "cuDriverGetVersion",
        "cuDeviceGetCount", "cuDeviceGetAttribute", "cuDeviceGetName",
        "cuDeviceGetUuid", "cuDeviceGetUuid_v2", "cuGetErrorString");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(DriverDeviceAPI())), parentStmt()))
          .bind("call"),
      this);

  MF.addMatcher(callExpr(allOf(callee(functionDecl(DriverDeviceAPI())),
                               unless(parentStmt())))
                    .bind("callUsed"),
                this);
}

void DriverDeviceAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  std::string APIName;
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "callUsed"))) {
      return;
    }
    IsAssigned = true;
  }
  if (auto DC = CE->getDirectCallee()) {
    APIName = DC->getNameAsString();
  } else {
    return;
  }
  std::ostringstream OS;


  if (APIName == "cuDeviceGet") {
    if (IsAssigned)
      OS << MapNames::getCheckErrorMacroName() + "(";
    auto FirArg = CE->getArg(0)->IgnoreImplicitAsWritten();
    auto SecArg = CE->getArg(1)->IgnoreImplicitAsWritten();

    ExprAnalysis SecEA(SecArg);
    SecEA.analyze();
    std::string Rep;
    printDerefOp(OS, FirArg);
    OS << " = " << SecEA.getReplacedString();
    if (IsAssigned) {
      OS << ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDeviceGetName") {
    if (IsAssigned)
      OS << MapNames::getCheckErrorMacroName() + "(";
    auto FirArg = CE->getArg(0)->IgnoreImplicitAsWritten();
    auto SecArg = CE->getArg(1)->IgnoreImplicitAsWritten();
    auto ThrArg = CE->getArg(2)->IgnoreImplicitAsWritten();
    ExprAnalysis FirEA(FirArg);
    ExprAnalysis SecEA(SecArg);
    ExprAnalysis ThrEA(ThrArg);
    FirEA.analyze();
    SecEA.analyze();
    ThrEA.analyze();
    OS << "memcpy(" << FirEA.getReplacedString()
       << ", " + MapNames::getDpctNamespace() + "get_device("
       << ThrEA.getReplacedString() << ").get_info<"
       << MapNames::getClNamespace() << "info::device::name>().c_str(), "
       << SecEA.getReplacedString() << ")";
    requestFeature(HelperFeatureEnum::device_ext);
    if (IsAssigned) {
      OS << ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDeviceComputeCapability") {
    auto &SM = DpctGlobalInfo::getSourceManager();
    std::string Indent =
        getIndent(SM.getExpansionLoc(CE->getBeginLoc()), SM).str();
    if (IsAssigned)
      OS << "[&](){" << getNL();
    auto FirArg = CE->getArg(0)->IgnoreImplicitAsWritten();
    auto SecArg = CE->getArg(1)->IgnoreImplicitAsWritten();
    auto ThrArg = CE->getArg(2)->IgnoreImplicitAsWritten();
    std::string device_str;
    if (DpctGlobalInfo::useNoQueueDevice()) {
      device_str = DpctGlobalInfo::getGlobalDeviceName();
    } else {
      std::string ThrRep;
      ExprAnalysis EA(ThrArg);
      EA.analyze();
      ThrRep = EA.getReplacedString();
      device_str = MapNames::getDpctNamespace() + "get_device(" + ThrRep + ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
    if (IsAssigned) {
      OS << Indent << "  ";
      printDerefOp(OS, FirArg);
      OS << " = " << MapNames::getDpctNamespace() << "get_major_version("
         << device_str << ");" << getNL();
      OS << Indent << "  ";
      printDerefOp(OS, SecArg);
      OS << " = " << MapNames::getDpctNamespace() << "get_minor_version("
         << device_str << ");" << getNL();
      OS << Indent << "  "
         << "return 0;" << getNL();
    } else {
      printDerefOp(OS, FirArg);
      OS << " = " << MapNames::getDpctNamespace() << "get_major_version("
         << device_str << ");" << getNL() << Indent;
      printDerefOp(OS, SecArg);
      OS << " = " << MapNames::getDpctNamespace() << "get_minor_version("
         << device_str << ")";
    }
    if (IsAssigned) {
      OS << Indent << "}()";
      report(CE->getBeginLoc(), Diagnostics::NOERROR_RETURN_LAMBDA, false);
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDeviceGetCount") {
    if (IsAssigned)
      OS << MapNames::getCheckErrorMacroName() + "(";
    auto Arg = CE->getArg(0)->IgnoreImplicitAsWritten();
    printDerefOp(OS, Arg);
    OS << " = " << MapNames::getDpctNamespace() + "device_count()";
    requestFeature(HelperFeatureEnum::device_ext);
    if (IsAssigned) {
      OS << ")";
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDeviceGetUuid" || APIName == "cuDeviceGetUuid_v2") {
    if (!DpctGlobalInfo::useDeviceInfo()) {
      report(CE->getBeginLoc(), Diagnostics::UNMIGRATED_DEVICE_PROP, false,
             APIName);
      return;
    }
    if (IsAssigned)
      OS << MapNames::getCheckErrorMacroName() + "(";
    auto Arg = CE->getArg(0)->IgnoreImplicitAsWritten();
    printDerefOp(OS, Arg);
    ExprAnalysis SecEA(CE->getArg(1));
    OS << " = "
       << MapNames::getDpctNamespace() + "get_device(" +
              SecEA.getReplacedString() + ")" + ".get_device_info()" +
              ".get_uuid()";
    requestFeature(HelperFeatureEnum::device_ext);
    if (IsAssigned) {
      OS << ")";
    }
    emplaceTransformation(new ReplaceStmt(CE, OS.str()));
  } else if (APIName == "cuDeviceGetAttribute") {
    auto SecArg = CE->getArg(1);
    if (auto DRE = dyn_cast<DeclRefExpr>(SecArg)) {
      auto AttributeName = DRE->getNameInfo().getAsString();
      if (AttributeName ==
              "CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED" &&
          !DpctGlobalInfo::useExpVirtualMemory()) {
        report(CE->getBeginLoc(), Diagnostics::TRY_EXPERIMENTAL_FEATURE, false,
               AttributeName, "--use-experimental-features=virtual_mem");
        return;
      }
      auto Search = MapNames::EnumNamesMap.find(AttributeName);
      if (Search == MapNames::EnumNamesMap.end()) {
        report(CE->getBeginLoc(), Diagnostics::NOT_SUPPORTED_PARAMETER, false,
               APIName,
               "parameter " + getStmtSpelling(SecArg) + " is unsupported");
        return;
      }
      if (AttributeName == "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY" ||
          AttributeName == "CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT" ||
          AttributeName == "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK") {
        report(CE->getBeginLoc(), Diagnostics::UNCOMPATIBLE_DEVICE_PROP, false,
          AttributeName, Search->second->NewName);
      }
    } else {
      report(CE->getBeginLoc(), Diagnostics::UNPROCESSED_DEVICE_ATTRIBUTE,
            false);
      return;
    }
  }
  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(APIName);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }
}


void DriverContextAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto contextAPI = [&]() {
    return hasAnyName(
        "cuInit", "cuCtxCreate_v2", "cuCtxCreate_v3", "cuCtxCreate_v4",
        "cuCtxSetCurrent", "cuCtxGetCurrent", "cuCtxSynchronize",
        "cuCtxDestroy_v2", "cuDevicePrimaryCtxRetain",
        "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease",
        "cuCtxGetDevice", "cuCtxGetApiVersion", "cuCtxGetLimit",
        "cuCtxPushCurrent_v2", "cuCtxPopCurrent_v2");
  };

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(contextAPI())), parentStmt()))
          .bind("call"),
      this);

  MF.addMatcher(
      callExpr(allOf(callee(functionDecl(contextAPI())), unless(parentStmt())))
          .bind("callUsed"),
      this);
}

void DriverContextAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  bool IsAssigned = false;
  std::string APIName;
  std::ostringstream OS;
  auto &SM = DpctGlobalInfo::getSourceManager();
  const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call");
  if (!CE) {
    if (!(CE = getNodeAsType<CallExpr>(Result, "callUsed")))
      return;
    IsAssigned = true;
  }

  if (auto DC = CE->getDirectCallee()) {
    APIName = DC->getNameAsString();
  } else {
    return;
  }

  if (!CallExprRewriterFactoryBase::RewriterMap)
    return;

  auto Itr = CallExprRewriterFactoryBase::RewriterMap->find(APIName);
  if (Itr != CallExprRewriterFactoryBase::RewriterMap->end()) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  if (IsAssigned) {
    OS << MapNames::getCheckErrorMacroName() + "(";
  }
  if (APIName == "cuInit") {
    std::string Msg = "this functionality is redundant in SYCL.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             APIName, Msg);
      emplaceTransformation(new ReplaceStmt(CE, "0"));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false, APIName,
             Msg);
      emplaceTransformation(new ReplaceStmt(CE, ""));
    }
    return;
  } else if (APIName == "cuCtxDestroy_v2" ||
             APIName == "cuDevicePrimaryCtxRelease_v2" ||
             APIName == "cuDevicePrimaryCtxRelease") {
    SourceLocation CallBegin(CE->getBeginLoc());
    SourceLocation CallEnd(CE->getEndLoc());

    bool IsMacroArg =
        SM.isMacroArgExpansion(CallBegin) && SM.isMacroArgExpansion(CallEnd);

    if (CallBegin.isMacroID() && IsMacroArg) {
      CallBegin = SM.getImmediateSpellingLoc(CallBegin);
      CallBegin = SM.getExpansionLoc(CallBegin);
    } else if (CallBegin.isMacroID()) {
      CallBegin = SM.getExpansionLoc(CallBegin);
    }

    if (CallEnd.isMacroID() && IsMacroArg) {
      CallEnd = SM.getImmediateSpellingLoc(CallEnd);
      CallEnd = SM.getExpansionLoc(CallEnd);
    } else if (CallEnd.isMacroID()) {
      CallEnd = SM.getExpansionLoc(CallEnd);
    }
    CallEnd = CallEnd.getLocWithOffset(1);

    std::string Msg = "this functionality is redundant in SYCL.";
    if (IsAssigned) {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
             APIName, Msg);
      emplaceTransformation(replaceText(CallBegin, CallEnd, "0", SM));
    } else {
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false, APIName,
             Msg);
      CallEnd = CallEnd.getLocWithOffset(1);
      emplaceTransformation(replaceText(CallBegin, CallEnd, "", SM));
    }
    return;
  } else if (APIName == "cuCtxSetCurrent") {
    if (DpctGlobalInfo::useNoQueueDevice()) {
      OS << "0";
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             "cuCtxSetCurrent",
             "it is redundant if it is migrated with option "
             "--helper-function-preference=no-queue-device "
             "which declares a global SYCL device and queue.");
    } else {
      auto Arg = CE->getArg(0)->IgnoreImplicitAsWritten();
      ExprAnalysis EA(Arg);
      EA.analyze();
      OS << MapNames::getDpctNamespace() + "select_device("
         << EA.getReplacedString() << ")";
      requestFeature(HelperFeatureEnum::device_ext);
    }
  } else if (APIName == "cuCtxGetCurrent") {
    auto Arg = CE->getArg(0)->IgnoreImplicitAsWritten();
    printDerefOp(OS, Arg);
    OS << " = ";
    if (DpctGlobalInfo::useNoQueueDevice()) {
      OS << "0";
      report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
             "cuCtxGetCurrent",
             "it is redundant if it is migrated with option "
             "--helper-function-preference=no-queue-device "
             "which declares a global SYCL device and queue.");
    } else {
      OS << MapNames::getDpctNamespace() + "get_current_device_id()";
      requestFeature(HelperFeatureEnum::device_ext);
    }
  } else if (APIName == "cuCtxSynchronize") {
    OS << MapNames::getDpctNamespace() +
              "get_current_device().queues_wait_and_throw()";
    requestFeature(HelperFeatureEnum::device_ext);
  } else if (APIName == "cuCtxGetLimit") {
    auto SecArg = CE->getArg(1);
    if (auto DRE = dyn_cast<DeclRefExpr>(SecArg)) {
      std::string AttributeName = DRE->getNameInfo().getAsString();
      auto Search = MapNames::EnumNamesMap.find(AttributeName);
      if (Search != MapNames::EnumNamesMap.end()) {
        printDerefOp(OS, CE->getArg(0));
        OS << " = " << Search->second->NewName;
      } else if (auto Msg = MapNames::RemovedAPIWarningMessage.find(APIName);
                 Msg != MapNames::RemovedAPIWarningMessage.end()) {
        if (IsAssigned) {
          report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED_0, false,
                 MapNames::ITFName.at(APIName), Msg->second);
          emplaceTransformation(new ReplaceStmt(CE, "0"));
        } else {
          report(CE->getBeginLoc(), Diagnostics::FUNC_CALL_REMOVED, false,
                 MapNames::ITFName.at(APIName), Msg->second);
          emplaceTransformation(new ReplaceStmt(CE, ""));
        }
        return;
      }
    }
  }
  if (IsAssigned) {
    OS << ")";
    requestFeature(HelperFeatureEnum::device_ext);
  }
  emplaceTransformation(new ReplaceStmt(CE, OS.str()));
}

// In host device function, macro __CUDA_ARCH__ is used to differentiate
// different code blocks. And migration of the two code blocks will result in
// different requirement on parameter of the function signature, device code
// block requires sycl::nd_item. Current design does two round parses and
// generates an extra host version function. The first-round parse dpct defines
// macro __CUDA_ARCH__ and the second-round parse dpct undefines macro
// __CUDA_ARCH__ to generate replacement for different code blocks.
// Implementation steps as follow:
//   1. Match all host device function's declaration and caller.
//   2. Check if macro CUDA_ARCH used to differentiate different
//   code blocks and called in host side, then record relative information.
//   3. Record host device function call expression.
//   4. All these information will be used in post-process stage and generate
//   final replacements.
// Condition to trigger the rule:
//   1. The function has host and device attribute.
//   2. The function uses macro CUDA_ARCH used to differentiate different code
//   blocks.
//   3. The function has been called in host side.
// Example code:
// __host__ __device__ int foo() {
//    #ifdef __CUDA_ARCH__
//      return threadIdx.x;
//    #else
//      return -1;
//    #endif
// }
//
// __global__ void kernel() {
//   foo();
// }
//
// int main() {
//   foo();
// }
//
// After migration:
// int foo(sycl::nd_item<3> item) {
//   return item.get_local_id(2);
// }
//
// int foo_host_ct1() {
//   return -1;
// }
//
// void kernel(sycl::nd_item<3> item) {
//   foo(item);
// }
//
// int main() {
//   foo_host_ct1();
// }
void CudaArchMacroRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto HostDeviceFunctionMatcher =
      functionDecl(allOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAHost),
                         unless(cxxMethodDecl())));
  MF.addMatcher(callExpr(callee(HostDeviceFunctionMatcher)).bind("callExpr"),
                this);
  MF.addMatcher(HostDeviceFunctionMatcher.bind("funcDecl"), this);
}
void CudaArchMacroRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &Global = DpctGlobalInfo::getInstance();
  auto &CT = DpctGlobalInfo::getContext();
  DpctNameGenerator DNG;
  const FunctionDecl *FD =
      getAssistNodeAsType<FunctionDecl>(Result, "funcDecl");
  auto &HDFIMap = Global.getHostDeviceFuncInfoMap();
  HostDeviceFuncLocInfo HDFLI;
  // process __host__ __device__ function definition except overloaded operator
  if (FD && !FD->isOverloadedOperator() &&
      FD->getTemplateSpecializationKind() ==
          TemplateSpecializationKind::TSK_Undeclared) {
    auto NameInfo = FD->getNameInfo();
    // TODO: add support for macro
    if (NameInfo.getBeginLoc().isMacroID())
      return;
    auto BeginLoc = SM.getExpansionLoc(FD->getBeginLoc());
    if (FD->isTemplated()) {
      auto P = CT.getParents(*FD);
      if (!P.size())
        return;
      const FunctionTemplateDecl *FTD = P[0].get<FunctionTemplateDecl>();
      if (FTD)
        BeginLoc = SM.getExpansionLoc(FTD->getBeginLoc());
    }
    auto EndLoc = SM.getExpansionLoc(FD->getEndLoc());
    auto Beg = Global.getLocInfo(BeginLoc);
    auto End = Global.getLocInfo(EndLoc);
    auto T = Lexer::findNextToken(EndLoc, SM, LangOptions());
    if (T.has_value() && T.value().is(tok::TokenKind::semi)) {
      End = Global.getLocInfo(T.value().getLocation());
    }
    auto FileInfo = DpctGlobalInfo::getInstance().insertFile(Beg.first);
    std::string &FileContent = FileInfo->getFileContent();
    auto NameLocInfo = Global.getLocInfo(NameInfo.getBeginLoc());
    std::string ManglingName = DNG.getName(FD);
    Global.getMainSourceFileMap()[NameLocInfo.first].push_back(
        Global.getMainFile()->getFilePath());
    HDFLI.FuncStartOffset = Beg.second;
    HDFLI.FuncEndOffset = End.second;
    HDFLI.FuncNameOffset = NameLocInfo.second + NameInfo.getAsString().length();
    HDFLI.FuncContentCache =
        FileContent.substr(Beg.second, End.second - Beg.second + 1);
    HDFLI.FilePath = NameLocInfo.first;
    if (!FD->isThisDeclarationADefinition()) {
      HDFLI.Type = HDFuncInfoType::HDFI_Decl;
      HDFIMap[ManglingName].LocInfos.insert(
          {HDFLI.FilePath.getCanonicalPath().str() + "Decl" + std::to_string(HDFLI.FuncEndOffset),
           HDFLI});
      return;
    }
    HDFLI.Type = HDFuncInfoType::HDFI_Def;
    bool NeedInsert = false;
    for (auto &Info : Global.getCudaArchPPInfoMap()[FileInfo->getFilePath()]) {
      if ((Info.first > Beg.second) && (Info.first < End.second) &&
          (!Info.second.ElInfo.empty() ||
           (Info.second.IfInfo.DirectiveLoc &&
            (Info.second.DT != IfType::IT_Unknow)))) {
        Info.second.isInHDFunc = true;
        NeedInsert = true;
      }
    }
    if (NeedInsert) {
      HDFIMap[ManglingName].isDefInserted = true;
      HDFIMap[ManglingName].LocInfos.insert(
          {HDFLI.FilePath.getCanonicalPath().str() + "Def" + std::to_string(HDFLI.FuncEndOffset),
           HDFLI});
    }
  } // address __host__ __device__ function call
  else if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "callExpr")) {
    // TODO: add support for macro
    if (CE->getBeginLoc().isMacroID())
      return;
    if (auto *PF = DpctGlobalInfo::getParentFunction(CE)) {
      if ((PF->hasAttr<CUDADeviceAttr>() && !PF->hasAttr<CUDAHostAttr>()) ||
          PF->hasAttr<CUDAGlobalAttr>()) {
        return;
      } else if (PF->hasAttr<CUDADeviceAttr>() && PF->hasAttr<CUDAHostAttr>()) {
        HDFLI.CalledByHostDeviceFunction = true;
      }
    }
    const FunctionDecl *DC = CE->getDirectCallee();
    if (DC) {
      unsigned int Offset = DC->getNameAsString().length();
      std::string ManglingName(DNG.getName(DC));
      if (DC->isTemplateInstantiation()) {
        if (auto DFT = DC->getPrimaryTemplate()) {
          const FunctionDecl *TFD = DFT->getTemplatedDecl();
          if (TFD)
            ManglingName = DNG.getName(TFD);
        }
      }
      auto LocInfo = Global.getLocInfo(CE->getBeginLoc());
      Global.getMainSourceFileMap()[LocInfo.first].push_back(
          Global.getMainFile()->getFilePath());
      HDFLI.Type = HDFuncInfoType::HDFI_Call;
      HDFLI.FilePath = LocInfo.first;
      HDFLI.FuncEndOffset = LocInfo.second + Offset;
      HDFIMap[ManglingName].LocInfos.insert(
          {HDFLI.FilePath.getCanonicalPath().str() + "Call" +
               std::to_string(HDFLI.FuncEndOffset),
           HDFLI});
    }
  }
}

void ComplexAPIRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto ComplexAPI = [&]() {
    return hasAnyName("make_cuDoubleComplex", "cuCreal", "cuCrealf", "cuCimag",
                      "cuCimagf", "cuCadd", "cuCsub", "cuCmul", "cuCdiv",
                      "cuCabs", "cuConj", "make_cuFloatComplex", "cuCaddf",
                      "cuCsubf", "cuCmulf", "cuCdivf", "cuCabsf", "cuConjf",
                      "make_cuComplex", "__saturatef", "cuComplexDoubleToFloat",
                      "cuComplexFloatToDouble", "cuCfma", "cuCfmaf");
  };

  MF.addMatcher(callExpr(callee(functionDecl(ComplexAPI()))).bind("call"),
                this);
}

void ComplexAPIRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "call")) {
    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}


void TemplateSpecializationTypeLocRule::registerMatcher(
    ast_matchers::MatchFinder &MF) {
  auto TargetTypeName = [&]() {
    return hasAnyName("thrust::not_equal_to", "thrust::constant_iterator",
                      "thrust::system::cuda::experimental::pinned_allocator",
                      "thrust::random::default_random_engine",
                      "thrust::random::uniform_real_distribution",
                      "thrust::random::normal_distribution",
                      "thrust::random::linear_congruential_engine",
                      "thrust::random::uniform_int_distribution");
  };

  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(TargetTypeName())))))
          .bind("loc"),
      this);

  MF.addMatcher(declRefExpr().bind("declRefExpr"), this);
}

void TemplateSpecializationTypeLocRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {

  const DeclRefExpr *DRE = getNodeAsType<DeclRefExpr>(Result, "declRefExpr");
  if (DRE) {
    std::string TypeName = DpctGlobalInfo::getTypeName(DRE->getType());
    std::string Name = DRE->getNameInfo().getName().getAsString();
    if (TypeName.find("thrust::random::linear_congruential_engine") !=
            std::string::npos &&
        Name == "max") {
      emplaceTransformation(
          new ReplaceStmt(DRE, "oneapi::dpl::default_engine::max()"));
    }
  }

  if (auto TL = getNodeAsType<TypeLoc>(Result, "loc")) {
    ExprAnalysis EA;
    EA.analyze(*TL);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}


void CudaStreamCastRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
     castExpr(hasType(qualType(hasCanonicalType(
        qualType(pointsTo(namedDecl(hasName("CUstream_st"))))))))
     .bind("cast"),
     this);
}

void CudaStreamCastRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto CE = getNodeAsType<CastExpr>(Result, "cast")) {
    if (CE->getCastKind() == clang::CK_LValueToRValue ||
        CE->getCastKind() == clang::CK_NoOp)
      return;

    if (isDefaultStream(CE->getSubExpr())) {
      if (isPlaceholderIdxDuplicated(CE->getSubExpr()))
        return;
      int Index = DpctGlobalInfo::getHelperFuncReplInfoIndexThenInc();
      buildTempVariableMap(Index, CE->getSubExpr(),
                           HelperFuncType::HFT_DefaultQueue);
      emplaceTransformation(
          new ReplaceStmt(CE, "{{NEEDREPLACEZ" + std::to_string(Index) + "}}"));
    } else if (CE->getSubExpr()->getType()->isIntegerType()) {
      requestFeature(HelperFeatureEnum::device_ext);
      emplaceTransformation(new ReplaceStmt(
          CE, MapNames::getDpctNamespace() + "int_as_queue_ptr(" +
                  ExprAnalysis::ref(CE->getSubExpr()) + ")"));
    }
  }
}


void CudaExtentRule::registerMatcher(ast_matchers::MatchFinder &MF) {

  // 1. Match any cudaExtent TypeLoc.
  MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(
                            namedDecl(hasAnyName("cudaExtent", "cudaPos"))))))
                    .bind("loc"),
                this);

  // 2. Match cudaExtent default ctor.
  //    cudaExtent()    - CXXTemporaryObjectExpr, handled by (1) and (2).
  //    cudaExtent a    - VarDecl, handled by (1) and (2)
  MF.addMatcher(
      cxxConstructExpr(hasType(namedDecl(hasAnyName("cudaExtent", "cudaPos"))))
          .bind("defaultCtor"),
      this);

  // 3. Match field declaration, which doesn't has an in-class initializer.
  //    The in-class initializer case will handled by other matchers.
  MF.addMatcher(
      fieldDecl(hasType(namedDecl(hasAnyName("cudaExtent", "cudaPos"))),
                unless(hasInClassInitializer(anything())))
          .bind("fieldDeclHasNoInit"),
      this);

  // 4. Match c++ initializer_list, which has cudaExtent type.
  MF.addMatcher(
      initListExpr(hasType(namedDecl(hasAnyName("cudaExtent", "cudaPos"))))
          .bind("initListExpr"),
      this);
}

void CudaExtentRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {

  // cudaExtent -> sycl::range<3>
  if (const TypeLoc *TL = getAssistNodeAsType<TypeLoc>(Result, "loc")) {
    ExprAnalysis EA;
    EA.analyze(*TL);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
    return;
  }

  // cudaExtent a;  -> sycl::range<3> a{0, 0, 0};
  // cudaExtent()   -> sycl::range<3>{0, 0, 0};
  // struct Foo { cudaExtent e; Foo() : e() {} }; -> struct Foo { sycl::range<3> e; Foo() : e{0, 0, 0} {} };
  if (const CXXConstructExpr *Ctor =
          getNodeAsType<CXXConstructExpr>(Result, "defaultCtor")) {

    // Ignore implicit move/copy ctor
    if (Ctor->getNumArgs() != 0)
      return;
    CharSourceRange CSR;
    SourceRange SR = Ctor->getParenOrBraceRange();
    auto &SM = DpctGlobalInfo::getSourceManager();
    std::string Replacement = "{0, 0, 0}";

    if (SR.isInvalid()) {
      auto CtorLoc = Ctor->getLocation().isMacroID()
                         ? SM.getSpellingLoc(Ctor->getLocation())
                         : Ctor->getLocation();
      auto CtorEndLoc = Lexer::getLocForEndOfToken(
          CtorLoc, 0, SM, DpctGlobalInfo::getContext().getLangOpts());
      CSR = CharSourceRange(SourceRange(CtorEndLoc, CtorEndLoc), false);
      DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            SM, CSR, Replacement, nullptr));
    } else {
      auto CtorEndLoc = Lexer::getLocForEndOfToken(
          SR.getEnd(), 0, SM, DpctGlobalInfo::getContext().getLangOpts());
      CharSourceRange CSR(SourceRange(SR.getBegin(), CtorEndLoc), false);
      DpctGlobalInfo::getInstance().addReplacement(
          std::make_shared<ExtReplacement>(
              SM, CSR, Replacement, nullptr));
    }
    return;
  }

  // struct Foo { cudaExtent a; }; -> struct Foo { syc::range<3> a{0, 0, 0}; };
  if (const FieldDecl *FD =
          getNodeAsType<FieldDecl>(Result, "fieldDeclHasNoInit")) {
    auto &SM = DpctGlobalInfo::getSourceManager();
    auto IdentBeginLoc = FD->getEndLoc().isMacroID()
                             ? SM.getSpellingLoc(FD->getEndLoc())
                             : FD->getEndLoc();
    auto IdentEndLoc = Lexer::getLocForEndOfToken(
        IdentBeginLoc, 0, SM, DpctGlobalInfo::getContext().getLangOpts());
    CharSourceRange CSR =
        CharSourceRange(SourceRange(IdentEndLoc, IdentEndLoc), false);
    std::string Replacement = "{0, 0, 0}";
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            SM, CSR, Replacement, nullptr));
    return;
  }

  // cudaExtent a{};          -> sycl::range<3> a{0, 0, 0};
  // cudaExtent b{1};         -> sycl::range<3> b{1, 0, 0};
  // cudaExtent c{1, 1};      -> sycl::range<3> c{1, 1, 0};
  // cudaExtent d{1, 1, 1};   -> sycl::range<3> d{1, 1, 1};
  // cudaExtent({1, 1, 1});   -> sycl::range<3>({1, 1, 1});
  if (const InitListExpr *Init =
          getNodeAsType<InitListExpr>(Result, "initListExpr")) {
    if (Init->getLBraceLoc() == Init->getRBraceLoc()) {
      return; // implicit init list in init list.
    }
    auto &SM = DpctGlobalInfo::getSourceManager();
    std::string Replacement;
    llvm::raw_string_ostream OS(Replacement);
    OS << "{";
    for (size_t I = 0; I < Init->getNumInits(); ++I) {
      const Expr *E = Init->getInit(I);
      if (isa<ImplicitValueInitExpr>(E)) {
        OS << "0";
      } else {
        ExprAnalysis EA;
        EA.analyze(E);
        OS << EA.getReplacedString();
      }
      if (I + 1 < Init->getNumInits())
        OS << ", ";
    }
    OS << "}";
    OS.flush();
    DpctGlobalInfo::getInstance().addReplacement(
        std::make_shared<ExtReplacement>(
            SM, Init, Replacement, nullptr));
    return;
  }
}


void CudaUuidRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(memberExpr(hasObjectExpression(hasType(namedDecl(hasAnyName(
                               "CUuuid_st", "cudaUUID_t", "CUuuid")))),
                           member(hasName("bytes")))
                    .bind("UUID_bytes"),
                this);
}

void CudaUuidRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto ME = Result.Nodes.getNodeAs<MemberExpr>("UUID_bytes")) {
    const auto SM = Result.SourceManager;
    const auto Begin = SM->getSpellingLoc(ME->getOperatorLoc());
    return emplaceTransformation(new ReplaceText(Begin, 6, ""));
  }
}


void TypeRemoveRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      binaryOperator(allOf(isAssignmentOperator(),
                           hasLHS(hasDescendant(memberExpr(hasType(namedDecl(
                               hasAnyName("cudaAccessPolicyWindow"))))))))
          .bind("AssignStmtRemove"),
      this);
}

void TypeRemoveRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto BO = getNodeAsType<BinaryOperator>(Result, "AssignStmtRemove"))
    emplaceTransformation(new ReplaceStmt(BO, ""));
  return;
}

void AssertRule::registerMatcher(MatchFinder &MF) {
  auto functionName = [&]() {
    return hasAnyName("__assert_fail", "__assertfail");
  };
  MF.addMatcher(
      callExpr(callee(functionDecl(functionName())),
               hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                              hasAttr(attr::CUDAGlobal)))))
          .bind("FunctionCall"),
      this);
}
void AssertRule::runRule(const MatchFinder::MatchResult &Result) {
  if (const CallExpr *CE = getNodeAsType<CallExpr>(Result, "FunctionCall")) {
    // The std assert is a macro, it will expand to __assert_fail.
    // But we should not touch the std assert.
    auto SpellingLoc =
        DpctGlobalInfo::getSourceManager().getSpellingLoc(CE->getBeginLoc());
    if (DpctGlobalInfo::isInAnalysisScope(SpellingLoc)) {
      ExprAnalysis EA(CE);
      emplaceTransformation(EA.getReplacement());
      EA.applyAllSubExprRepl();
    }
  }
}

} //namespace dpct
} // namespace clang
