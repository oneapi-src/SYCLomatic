//===--------------- RulesLangAtomic.cpp----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "RuleInfra/ASTmatcherCommon.h"
#include "RuleInfra/CallExprRewriter.h"
#include "RuleInfra/CallExprRewriterCommon.h"
#include "RuleInfra/ExprAnalysis.h"
#include "RuleInfra/MigrationStatistics.h"
#include "RulesLang.h"
#include "RulesLang/MapNamesLang.h"
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

#include <string>
#include <unordered_map>
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

void TextureMemberSetRule::registerMatcher(MatchFinder &MF) {
  auto ObjectType =
      hasObjectExpression(hasType(namedDecl(hasAnyName("cudaResourceDesc"))));
  // myres.res.array.array = a;
  auto AssignResArrayArray = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("array")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("array")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("ArrayMember"))))))))));
  // myres.resType = cudaResourceTypeArray;
  auto ArraySetCompound = binaryOperator(allOf(
      isAssignmentOperator(),
      hasLHS(memberExpr(allOf(ObjectType, member(hasName("resType"))))
                 .bind("ResTypeMemberExpr")),
      hasRHS(declRefExpr(
          hasDeclaration(enumConstantDecl(hasName("cudaResourceTypeArray"))))),
      hasParent(
          compoundStmt(has(AssignResArrayArray.bind("AssignResArrayArray"))))));
  MF.addMatcher(ArraySetCompound.bind("ArraySetCompound"), this);
  // myres.res.pitch2D.devPtr = p;
  auto AssignRes2DPtr = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("devPtr")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("pitch2D")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("PtrMember"))))))))));
  // myres.res.pitch2D.desc = desc42;
  auto AssignRes2DDesc = cxxOperatorCallExpr(
      allOf(isAssignmentOperator(),
            has(memberExpr(allOf(
                member(hasName("desc")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("pitch2D")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("DescMember"))))))))));
  // myres.res.pitch2D.width = sizeof(float4) * 32;
  auto AssignRes2DWidth = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("width")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("pitch2D")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("WidthMember"))))))))));
  // myres.res.pitch2D.height = 32;
  auto AssignRes2DHeight = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("height")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("pitch2D")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("HeightMember"))))))))));
  // myres.res.pitch2D.pitchInBytes = sizeof(float4) * 32;
  auto AssignRes2DPitchInBytes = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("pitchInBytes")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("pitch2D")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("PitchMember"))))))))));
  // myres.resType = cudaResourceTypePitch2D;
  auto Pitch2DSetCompound = binaryOperator(allOf(
      isAssignmentOperator(),
      hasLHS(memberExpr(allOf(ObjectType, member(hasName("resType"))))
                 .bind("ResTypeMemberExpr")),
      hasRHS(declRefExpr(hasDeclaration(
          enumConstantDecl(hasName("cudaResourceTypePitch2D"))))),
      hasParent(compoundStmt(allOf(
          has(AssignRes2DPtr.bind("AssignRes2DPtr")),
          has(AssignRes2DDesc.bind("AssignRes2DDesc")),
          has(AssignRes2DWidth.bind("AssignRes2DWidth")),
          has(AssignRes2DHeight.bind("AssignRes2DHeight")),
          has(AssignRes2DPitchInBytes.bind("AssignRes2DPitchInBytes")))))));
  MF.addMatcher(Pitch2DSetCompound.bind("Pitch2DSetCompound"), this);
  // myres.res.linear.devPtr = d_data21;
  auto AssignResLinearPtr = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("devPtr")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("linear")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("PtrMember"))))))))));
  // myres.res.linear.sizeInBytes = sizeof(float4) * 32;
  auto AssignResLinearSize = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(
                member(hasName("sizeInBytes")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("linear")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("SizeMember"))))))))));
  // myres.res.linear.desc = desc42;
  auto AssignResLinearDesc = cxxOperatorCallExpr(
      allOf(isAssignmentOperator(),
            has(memberExpr(allOf(
                member(hasName("desc")),
                hasObjectExpression(memberExpr(allOf(
                    member(hasName("linear")),
                    hasObjectExpression(
                        memberExpr(allOf(ObjectType, member(hasName("res"))))
                            .bind("DescMember"))))))))));
  // myres.resType = cudaResourceTypeLinear;
  auto LinearSetCompound = binaryOperator(
      allOf(isAssignmentOperator(),
            hasLHS(memberExpr(allOf(ObjectType, member(hasName("resType"))))
                       .bind("ResTypeMemberExpr")),
            hasRHS(declRefExpr(hasDeclaration(
                enumConstantDecl(hasName("cudaResourceTypeLinear"))))),
            hasParent(compoundStmt(
                allOf(has(AssignResLinearPtr.bind("AssignResLinearPtr")),
                      has(AssignResLinearSize.bind("AssignResLinearSize")),
                      has(AssignResLinearDesc.bind("AssignResLinearDesc")))))));
  MF.addMatcher(LinearSetCompound.bind("LinearSetCompound"), this);
}

void TextureMemberSetRule::removeRange(SourceRange R) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &LO = DpctGlobalInfo::getContext().getLangOpts();
  auto End = R.getEnd();
  End = End.getLocWithOffset(Lexer::MeasureTokenLength(End, SM, LO));
  End = End.getLocWithOffset(Lexer::MeasureTokenLength(End, SM, LO));
  emplaceTransformation(replaceText(R.getBegin(), End, "", SM));
}

void TextureMemberSetRule::runRule(const MatchFinder::MatchResult &Result) {
  auto &SM = DpctGlobalInfo::getSourceManager();
  auto &LO = DpctGlobalInfo::getContext().getLangOpts();
  if (auto BO = getNodeAsType<BinaryOperator>(Result, "Pitch2DSetCompound")) {
    auto AssignPtrExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignRes2DPtr");
    auto AssignWidthExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignRes2DWidth");
    auto AssignHeightExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignRes2DHeight");
    auto AssignDescExpr =
        getNodeAsType<CXXOperatorCallExpr>(Result, "AssignRes2DDesc");
    auto AssignPitchExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignRes2DPitchInBytes");
    auto ResTypeMemberExpr =
        getNodeAsType<MemberExpr>(Result, "ResTypeMemberExpr");
    auto PtrMemberExpr = getNodeAsType<MemberExpr>(Result, "PtrMember");
    auto WidthMemberExpr = getNodeAsType<MemberExpr>(Result, "WidthMember");
    auto HeightMemberExpr = getNodeAsType<MemberExpr>(Result, "HeightMember");
    auto PitchMemberExpr = getNodeAsType<MemberExpr>(Result, "PitchMember");
    auto DescMemberExpr = getNodeAsType<MemberExpr>(Result, "DescMember");

    if (!AssignPtrExpr || !AssignWidthExpr || !AssignHeightExpr ||
        !AssignDescExpr || !AssignPitchExpr || !ResTypeMemberExpr ||
        !PtrMemberExpr || !WidthMemberExpr || !HeightMemberExpr ||
        !PitchMemberExpr || !DescMemberExpr)
      return;

    // Compare the name of all resource obj
    std::string ResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(ResTypeMemberExpr->getBase())) {
      ResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string PtrResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(PtrMemberExpr->getBase())) {
      PtrResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string WidthResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(WidthMemberExpr->getBase())) {
      WidthResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string HeightResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(HeightMemberExpr->getBase())) {
      HeightResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string PitchResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(PitchMemberExpr->getBase())) {
      PitchResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string DescResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(DescMemberExpr->getBase())) {
      DescResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    if (ResName.compare(PtrResName) || ResName.compare(WidthResName) ||
        ResName.compare(HeightResName) || ResName.compare(PitchResName) ||
        ResName.compare(DescResName)) {
      // Won't do pretty code if the resource name is different
      return;
    }
    // Calculate insert location
    std::string MemberOpt = ResTypeMemberExpr->isArrow() ? "->" : ".";
    auto BORange = getStmtExpansionSourceRange(BO);
    auto AssignPtrRange = getStmtExpansionSourceRange(AssignPtrExpr);
    auto AssignWidthRange = getStmtExpansionSourceRange(AssignWidthExpr);
    auto AssignHeightRange = getStmtExpansionSourceRange(AssignHeightExpr);
    auto AssignPitchRange = getStmtExpansionSourceRange(AssignPitchExpr);
    auto AssignDescRange = getStmtExpansionSourceRange(AssignDescExpr);
    auto LastPos = BORange.getEnd();
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignPtrRange.getEnd()).second) {
      LastPos = AssignPtrRange.getEnd();
    }
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignWidthRange.getEnd()).second) {
      LastPos = AssignWidthRange.getEnd();
    }
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignHeightRange.getEnd()).second) {
      LastPos = AssignHeightRange.getEnd();
    }
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignPitchRange.getEnd()).second) {
      LastPos = AssignPitchRange.getEnd();
    }
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignDescRange.getEnd()).second) {
      LastPos = AssignDescRange.getEnd();
    }
    // Skip the last token
    LastPos =
        LastPos.getLocWithOffset(Lexer::MeasureTokenLength(LastPos, SM, LO));
    // Skip ";"
    LastPos =
        LastPos.getLocWithOffset(Lexer::MeasureTokenLength(LastPos, SM, LO));
    // Generate insert str
    ExprAnalysis EA;
    EA.analyze(AssignPtrExpr->getRHS());
    std::string AssignPtrRHS = EA.getReplacedString();
    EA.analyze(AssignWidthExpr->getRHS());
    std::string AssignWidthRHS = EA.getReplacedString();
    EA.analyze(AssignHeightExpr->getRHS());
    std::string AssignHeightRHS = EA.getReplacedString();
    EA.analyze(AssignPitchExpr->getRHS());
    std::string AssignPitchRHS = EA.getReplacedString();
    EA.analyze(AssignDescExpr->getArg(1));
    std::string AssignDescRHS = EA.getReplacedString();
    std::string IndentStr = getIndent(AssignPtrExpr->getBeginLoc(), SM).str();
    std::string InsertStr = getNL() + IndentStr + ResName + MemberOpt +
                            "set_data(" + AssignPtrRHS + ", " + AssignWidthRHS +
                            ", " + AssignHeightRHS + ", " + AssignPitchRHS +
                            ", " + AssignDescRHS + ");";
    requestFeature(HelperFeatureEnum::device_ext);
    // Remove all the assign expr
    removeRange(BORange);
    removeRange(AssignPtrRange);
    removeRange(AssignWidthRange);
    removeRange(AssignHeightRange);
    removeRange(AssignPitchRange);
    removeRange(AssignDescRange);
    emplaceTransformation(new InsertText(LastPos, std::move(InsertStr)));
  } else if (auto BO =
                 getNodeAsType<BinaryOperator>(Result, "LinearSetCompound")) {
    auto AssignPtrExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignResLinearPtr");
    auto AssignSizeExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignResLinearSize");
    auto AssignDescExpr =
        getNodeAsType<CXXOperatorCallExpr>(Result, "AssignResLinearDesc");
    auto ResTypeMemberExpr =
        getNodeAsType<MemberExpr>(Result, "ResTypeMemberExpr");
    auto PtrMemberExpr = getNodeAsType<MemberExpr>(Result, "PtrMember");
    auto SizeMemberExpr = getNodeAsType<MemberExpr>(Result, "SizeMember");
    auto DescMemberExpr = getNodeAsType<MemberExpr>(Result, "DescMember");

    if (!BO || !AssignPtrExpr || !AssignSizeExpr || !AssignDescExpr ||
        !ResTypeMemberExpr || !PtrMemberExpr || !SizeMemberExpr ||
        !DescMemberExpr)
      return;

    // Compare the name of all resource obj
    std::string ResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(ResTypeMemberExpr->getBase())) {
      ResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string PtrResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(PtrMemberExpr->getBase())) {
      PtrResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string SizeResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(SizeMemberExpr->getBase())) {
      SizeResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string DescResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(DescMemberExpr->getBase())) {
      DescResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    if (ResName.compare(PtrResName) || ResName.compare(SizeResName) ||
        ResName.compare(DescResName)) {
      // Won't do pretty code if the resource name is different
      return;
    }
    // Calculate insert location
    std::string MemberOpt = ResTypeMemberExpr->isArrow() ? "->" : ".";
    auto BORange = getStmtExpansionSourceRange(BO);
    auto AssignPtrRange = getStmtExpansionSourceRange(AssignPtrExpr);
    auto AssignSizeRange = getStmtExpansionSourceRange(AssignSizeExpr);
    auto AssignDescRange = getStmtExpansionSourceRange(AssignDescExpr);
    auto LastPos = BORange.getEnd();
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignPtrRange.getEnd()).second) {
      LastPos = AssignPtrRange.getEnd();
    }
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignSizeRange.getEnd()).second) {
      LastPos = AssignSizeRange.getEnd();
    }
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignDescRange.getEnd()).second) {
      LastPos = AssignDescRange.getEnd();
    }
    // Skip the last token
    LastPos =
        LastPos.getLocWithOffset(Lexer::MeasureTokenLength(LastPos, SM, LO));
    // Skip ";"
    LastPos =
        LastPos.getLocWithOffset(Lexer::MeasureTokenLength(LastPos, SM, LO));
    // Generate insert str
    ExprAnalysis EA;
    EA.analyze(AssignPtrExpr->getRHS());
    std::string AssignPtrRHS = EA.getReplacedString();
    EA.analyze(AssignSizeExpr->getRHS());
    std::string AssignSizeRHS = EA.getReplacedString();
    EA.analyze(AssignDescExpr->getArg(1));
    std::string AssignDescRHS = EA.getReplacedString();
    std::string IndentStr = getIndent(AssignPtrExpr->getBeginLoc(), SM).str();
    std::string InsertStr = getNL() + IndentStr + ResName + MemberOpt +
                            "set_data(" + AssignPtrRHS + ", " + AssignSizeRHS +
                            ", " + AssignDescRHS + ");";
    requestFeature(HelperFeatureEnum::device_ext);
    // Remove all the assign expr
    removeRange(BORange);
    removeRange(AssignPtrRange);
    removeRange(AssignSizeRange);
    removeRange(AssignDescRange);
    emplaceTransformation(new InsertText(LastPos, std::move(InsertStr)));
  } else if (auto BO =
                 getNodeAsType<BinaryOperator>(Result, "ArraySetCompound")) {
    auto AssignArrayExpr =
        getNodeAsType<BinaryOperator>(Result, "AssignResArrayArray");
    auto ResTypeMemberExpr =
        getNodeAsType<MemberExpr>(Result, "ResTypeMemberExpr");
    auto ArrayMemberExpr = getNodeAsType<MemberExpr>(Result, "ArrayMember");

    if (!BO || !AssignArrayExpr || !ResTypeMemberExpr || !ArrayMemberExpr)
      return;

    // Compare the name of all resource obj
    std::string ResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(ResTypeMemberExpr->getBase())) {
      ResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }
    std::string ArrayResName = "";
    if (auto DRE = dyn_cast<DeclRefExpr>(ArrayMemberExpr->getBase())) {
      ArrayResName = DRE->getDecl()->getNameAsString();
    } else {
      return;
    }

    if (ResName.compare(ArrayResName)) {
      // Won't do pretty code if the resource name is different
      return;
    }
    // Calculate insert location
    std::string MemberOpt = ResTypeMemberExpr->isArrow() ? "->" : ".";
    auto BORange = getStmtExpansionSourceRange(BO);
    auto AssignArrayRange = getStmtExpansionSourceRange(AssignArrayExpr);

    auto LastPos = BORange.getEnd();
    if (SM.getDecomposedLoc(LastPos).second <
        SM.getDecomposedLoc(AssignArrayRange.getEnd()).second) {
      LastPos = AssignArrayRange.getEnd();
    }

    // Skip the last token
    LastPos =
        LastPos.getLocWithOffset(Lexer::MeasureTokenLength(LastPos, SM, LO));
    // Skip ";"
    LastPos =
        LastPos.getLocWithOffset(Lexer::MeasureTokenLength(LastPos, SM, LO));
    // Generate insert str
    ExprAnalysis EA;
    EA.analyze(AssignArrayExpr->getRHS());
    std::string AssignArrayRHS = EA.getReplacedString();
    std::string IndentStr = getIndent(AssignArrayExpr->getBeginLoc(), SM).str();
    std::string InsertStr = getNL() + IndentStr + ResName + MemberOpt +
                            "set_data(" + AssignArrayRHS + ");";
    requestFeature(HelperFeatureEnum::device_ext);
    // Remove all the assign expr
    removeRange(BORange);
    removeRange(AssignArrayRange);

    emplaceTransformation(new InsertText(LastPos, std::move(InsertStr)));
  }
}

void TextureRule::registerMatcher(MatchFinder &MF) {
  auto DeclMatcher =
      varDecl(hasType(classTemplateSpecializationDecl(hasName("texture"))));

  auto DeclMatcherUTF = varDecl(hasType(classTemplateDecl(hasName("texture"))));
  MF.addMatcher(DeclMatcherUTF.bind("texDeclForUnspecializedTemplateFunc"),
                this);

  // Match texture object's declaration
  MF.addMatcher(DeclMatcher.bind("texDecl"), this);
  MF.addMatcher(
      declRefExpr(
          hasDeclaration(DeclMatcher.bind("texDecl")),
          anyOf(hasAncestor(functionDecl(anyOf(hasAttr(attr::CUDADevice),
                                               hasAttr(attr::CUDAGlobal)))
                                .bind("texFunc")),
                // Match the __global__/__device__ functions inside which
                // texture object is referenced
                anything()) // Make this matcher available whether it has
                            // ancestors as before
          )
          .bind("tex"),
      this);
  MF.addMatcher(typeLoc(loc(qualType(hasDeclaration(typedefDecl(hasAnyName(
                            "cudaTextureObject_t", "cudaSurfaceObject_t",
                            "CUsurfObject", "CUtexObject"))))))
                    .bind("texObj"),
                this);
  MF.addMatcher(
      memberExpr(
          hasObjectExpression(hasType(type(hasUnqualifiedDesugaredType(
              recordType(hasDeclaration(recordDecl(hasAnyName(
                  "cudaChannelFormatDesc", "cudaTextureDesc",
                  "cudaResourceDesc", "textureReference",
                  "CUDA_RESOURCE_DESC_st", "CUDA_TEXTURE_DESC_st")))))))))
          .bind("texMember"),
      this);
  MF.addMatcher(
      typeLoc(
          loc(qualType(hasDeclaration(namedDecl(hasAnyName(
              "cudaChannelFormatDesc", "cudaChannelFormatKind",
              "cudaTextureDesc", "cudaResourceDesc", "cudaResourceType",
              "CUDA_RESOURCE_DESC", "cudaTextureAddressMode",
              "cudaTextureFilterMode", "cudaArray", "cudaArray_t", "CUarray_st",
              "CUarray", "CUarray_format", "CUarray_format_enum",
              "CUresourcetype", "CUresourcetype_enum", "CUaddress_mode",
              "CUaddress_mode_enum", "CUfilter_mode", "CUfilter_mode_enum",
              "CUDA_TEXTURE_DESC", "CUtexref", "textureReference",
              "cudaMipmappedArray", "cudaMipmappedArray_t"))))))
          .bind("texType"),
      this);

  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(hasType(enumDecl(hasAnyName(
                      "cudaTextureAddressMode", "cudaTextureFilterMode",
                      "cudaChannelFormatKind", "cudaResourceType",
                      "CUarray_format_enum", "CUaddress_mode_enum",
                      "CUfilter_mode_enum"))))))

          .bind("texEnum"),
      this);

  std::vector<std::string> APINamesSelected = {
      "cudaCreateChannelDesc",
      "cudaCreateChannelDescHalf",
      "cudaCreateChannelDescHalf1",
      "cudaCreateChannelDescHalf2",
      "cudaCreateChannelDescHalf4",
      "cudaUnbindTexture",
      "cudaBindTextureToArray",
      "cudaBindTextureToMipmappedArray",
      "cudaBindTexture",
      "cudaBindTexture2D",
      "tex1D",
      "tex1DLod",
      "tex2D",
      "tex2DLod",
      "tex3D",
      "tex3DLod",
      "tex1Dfetch",
      "tex1DLayered",
      "tex2DLayered",
      "surf1Dwrite",
      "surf2Dwrite",
      "surf3Dwrite",
      "surf2DLayeredwrite",
      "surf1Dread",
      "surf2Dread",
      "surf3Dread",
      "cudaCreateTextureObject",
      "cudaDestroyTextureObject",
      "cudaGetTextureObjectResourceDesc",
      "cudaGetTextureObjectTextureDesc",
      "cudaGetTextureObjectResourceViewDesc",
      "cudaCreateSurfaceObject",
      "cudaDestroySurfaceObject",
      "cudaGetSurfaceObjectResourceDesc",
      "cuSurfObjectCreate",
      "cuSurfObjectGetResourceDesc",
      "cuSurfObjectDestroy",
      "cuArray3DCreate_v2",
      "cuArrayCreate_v2",
      "cuArrayDestroy",
      "cuTexObjectCreate",
      "cuTexObjectDestroy",
      "cuTexObjectGetTextureDesc",
      "cuTexObjectGetResourceDesc",
      "cuTexRefSetArray",
      "cuTexRefSetFormat",
      "cuTexRefSetAddressMode",
      "cuTexRefSetFilterMode",
      "cuTexRefSetFlags",
      "cuTexRefGetAddressMode",
      "cuTexRefGetFilterMode",
      "cuTexRefGetFlags",
      "cuTexRefSetAddress_v2",
      "cuTexRefSetAddress2D_v3",
  };

  auto hasAnyFuncName = [&]() {
    return internal::Matcher<NamedDecl>(
        new internal::HasNameMatcher(APINamesSelected));
  };

  MF.addMatcher(callExpr(callee(functionDecl(hasAnyFuncName()))).bind("call"),
                this);

  MF.addMatcher(
      unresolvedLookupExpr(hasAnyDeclaration(namedDecl(hasAnyFuncName())),
                           hasParent(callExpr().bind("callExpr")))
          .bind("unresolvedLookupExpr"),
      this);
}

bool TextureRule::removeExtraMemberAccess(const MemberExpr *ME) {
  if (auto ParentME = getParentMemberExpr(ME)) {
    emplaceTransformation(new ReplaceToken(ME->getMemberLoc(), ""));
    emplaceTransformation(new ReplaceToken(ParentME->getOperatorLoc(), ""));
    return true;
  }
  return false;
}

bool TextureRule::tryMerge(const MemberExpr *ME, const Expr *BO) {
  static std::unordered_map<std::string, std::vector<std::string>> MergeMap = {
      {"textureReference", {"addressMode", "filterMode", "normalized"}},
      {"cudaTextureDesc", {"addressMode", "filterMode", "normalizedCoords"}},
      {"CUDA_TEXTURE_DESC", {"addressMode", "filterMode", "flags"}},
  };

  auto Iter = MergeMap.find(
      DpctGlobalInfo::getUnqualifiedTypeName(ME->getBase()->getType()));
  if (Iter == MergeMap.end())
    return false;

  SettersMerger Merger(Iter->second, this);
  return Merger.tryMerge(BO);
}

void TextureRule::replaceTextureMember(const MemberExpr *ME,
                                       ASTContext &Context, SourceManager &SM) {
  auto AssignedBO = getParentAsAssignedBO(ME, Context);
  if (!DpctGlobalInfo::useExtBindlessImages() && tryMerge(ME, AssignedBO))
    return;

  auto Field = ME->getMemberNameInfo().getAsString();
  if (Field == "channelDesc") {
    if (removeExtraMemberAccess(ME))
      return;
  }
  auto IsMipmapMember =
      Field == "maxAnisotropy" || Field == "mipmapFilterMode" ||
      Field == "minMipmapLevelClamp" || Field == "maxMipmapLevelClamp";
  auto ReplField = MapNames::findReplacedName(TextureMemberNames, Field);
  if (ReplField.empty() ||
      (!DpctGlobalInfo::useExtBindlessImages() && IsMipmapMember)) {
    if (Field == "readMode")
      report(ME->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_NORM_READ_MODE,
             false);
    else
      report(ME->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
             DpctGlobalInfo::getOriginalTypeName(ME->getBase()->getType()) +
                 "::" + Field);
    if (AssignedBO) {
      emplaceTransformation(new ReplaceStmt(AssignedBO, ""));
    } else {
      emplaceTransformation(new ReplaceStmt(ME, "0"));
    }
    return;
  }

  if (AssignedBO) {
    StringRef MethodName;
    auto AssignedValue = getMemberAssignedValue(AssignedBO, Field, MethodName);
    if (MethodName.empty()) {
      requestFeature(HelperFeatureEnum::device_ext);
    } else {
      if (MapNamesLang::SamplingInfoToSetFeatureMap.count(MethodName.str())) {
        requestFeature(
            MapNamesLang::SamplingInfoToSetFeatureMap.at(MethodName.str()));
      }
      if (MapNamesLang::ImageWrapperBaseToSetFeatureMap.count(
              MethodName.str())) {
        requestFeature(
            MapNamesLang::ImageWrapperBaseToSetFeatureMap.at(MethodName.str()));
      }
    }
    emplaceTransformation(ReplaceMemberAssignAsSetMethod(
        AssignedBO, ME, MethodName, AssignedValue));
  } else {
    if (ReplField == "coordinate_normalization_mode") {
      emplaceTransformation(
          new RenameFieldInMemberExpr(ME, "is_coordinate_normalized()"));
      requestFeature(HelperFeatureEnum::device_ext);
    } else {
      emplaceTransformation(new RenameFieldInMemberExpr(
          ME, buildString("get_", ReplField, "()")));
      if (MapNamesLang::SamplingInfoToGetFeatureMap.count(ReplField)) {
        requestFeature(MapNamesLang::SamplingInfoToGetFeatureMap.at(ReplField));
      }
      if (MapNamesLang::ImageWrapperBaseToGetFeatureMap.count(ReplField)) {
        requestFeature(
            MapNamesLang::ImageWrapperBaseToGetFeatureMap.at(ReplField));
      }
    }
  }
}

const Expr *TextureRule::getParentAsAssignedBO(const Expr *E,
                                               ASTContext &Context) {
  auto Parents = Context.getParents(*E);
  if (Parents.size() > 0)
    return getAssignedBO(Parents[0].get<Expr>(), Context);
  return nullptr;
}

// Return the binary operator if E is the lhs of an assign expression, otherwise
// nullptr.
const Expr *TextureRule::getAssignedBO(const Expr *E, ASTContext &Context) {
  if (dyn_cast<MemberExpr>(E)) {
    // Continue finding parents when E is MemberExpr.
    return getParentAsAssignedBO(E, Context);
  } else if (auto ICE = dyn_cast<ImplicitCastExpr>(E)) {
    // Stop finding parents and return nullptr when E is ImplicitCastExpr,
    // except for ArrayToPointerDecay cast.
    if (ICE->getCastKind() == CK_ArrayToPointerDecay) {
      return getParentAsAssignedBO(E, Context);
    }
  } else if (auto ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    // Continue finding parents when E is ArraySubscriptExpr, and remove
    // subscript operator anyway for texture object's member.
    emplaceTransformation(new ReplaceToken(
        Lexer::getLocForEndOfToken(ASE->getLHS()->getEndLoc(), 0,
                                   Context.getSourceManager(),
                                   Context.getLangOpts()),
        ASE->getRBracketLoc(), ""));
    return getParentAsAssignedBO(E, Context);
  } else if (auto BO = dyn_cast<BinaryOperator>(E)) {
    // If E is BinaryOperator, return E only when it is assign expression,
    // otherwise return nullptr.
    auto Opcode = BO->getOpcode();
    if (Opcode == BO_Assign || Opcode == BO_OrAssign)
      return BO;
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (COCE->getOperator() == OO_Equal) {
      return COCE;
    }
  }
  return nullptr;
}

bool TextureRule::processTexVarDeclInDevice(const VarDecl *VD) {
  if (auto FD =
          dyn_cast_or_null<FunctionDecl>(VD->getParentFunctionOrMethod())) {
    if (FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) {
      auto Tex = DpctGlobalInfo::getInstance().insertTextureInfo(VD);

      auto DataType = Tex->getType()->getDataType();
      if (DataType.back() != '4' && !DpctGlobalInfo::useExtBindlessImages()) {
        report(VD->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_FORMAT, true);
      }

      ParameterStream PS;
      Tex->getFuncDecl(PS);
      emplaceTransformation(new ReplaceToken(VD->getBeginLoc(), VD->getEndLoc(),
                                             std::move(PS.Str)));
      return true;
    }
  }
  return false;
}

void TextureRule::runRule(const MatchFinder::MatchResult &Result) {

  if (getAssistNodeAsType<UnresolvedLookupExpr>(Result,
                                                "unresolvedLookupExpr")) {
    const CallExpr *CE = getAssistNodeAsType<CallExpr>(Result, "callExpr");
    ExprAnalysis A;
    A.analyze(CE);
    emplaceTransformation(A.getReplacement());
  }

  if (auto VD = getAssistNodeAsType<VarDecl>(
          Result, "texDeclForUnspecializedTemplateFunc")) {

    auto TST = VD->getType()->getAs<TemplateSpecializationType>();
    if (!TST)
      return;

    if (DpctGlobalInfo::useSYCLCompat()) {
      report(VD->getLocation(), Diagnostics::UNSUPPORT_SYCLCOMPAT, false,
             VD->getName());
      return;
    }

    std::string Name =
        TST->getTemplateName().getAsTemplateDecl()->getNameAsString();

    if (Name == "texture") {
      auto Args = TST->template_arguments();

      if (!isa<ParmVarDecl>(VD) || Args.size() != 3)
        return;

      if (getStmtSpelling(Args[2].getAsExpr()) == "cudaReadModeNormalizedFloat")
        report(VD->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_NORM_READ_MODE,
               true);

      processTexVarDeclInDevice(VD);
    }
  } else if (auto VD = getAssistNodeAsType<VarDecl>(Result, "texDecl")) {
    auto TST = VD->getType()->getAs<TemplateSpecializationType>();
    if (!TST)
      return;
    if (DpctGlobalInfo::useSYCLCompat()) {
      report(VD->getLocation(), Diagnostics::UNSUPPORT_SYCLCOMPAT, false,
             VD->getName());
      return;
    }
    auto Args = TST->template_arguments();

    if (Args.size() == 3) {
      if (getStmtSpelling(Args[2].getAsExpr()) == "cudaReadModeNormalizedFloat")
        report(VD->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_NORM_READ_MODE,
               true);
    }

    auto Tex = DpctGlobalInfo::getInstance().insertTextureInfo(VD);

    if (auto FD = getAssistNodeAsType<FunctionDecl>(Result, "texFunc")) {

      if (!isa<ParmVarDecl>(VD))
        if (auto DFI = DeviceFunctionDecl::LinkRedecls(FD))
          DFI->addTexture(Tex);
    }

    if (processTexVarDeclInDevice(VD))
      return;

    auto DataType = Tex->getType()->getDataType();
    if (DataType.back() != '4' && !DpctGlobalInfo::useExtBindlessImages()) {
      report(VD->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_FORMAT, true);
    }
    emplaceTransformation(new ReplaceVarDecl(VD, Tex->getHostDeclString()));

  } else if (auto ME = getNodeAsType<MemberExpr>(Result, "texMember")) {
    if (DpctGlobalInfo::useSYCLCompat())
      return;
    auto BaseTy = DpctGlobalInfo::getUnqualifiedTypeName(
        ME->getBase()->getType().getDesugaredType(*Result.Context),
        *Result.Context);
    auto MemberName = ME->getMemberNameInfo().getAsString();
    if (BaseTy == "cudaResourceDesc" || BaseTy == "CUDA_RESOURCE_DESC_st" ||
        BaseTy == "CUDA_RESOURCE_DESC") {
      if (MemberName == "res") {
        removeExtraMemberAccess(ME);
        replaceResourceDataExpr(getParentMemberExpr(ME), *Result.Context);
      } else if (MemberName == "resType") {
        if (auto BO = getParentAsAssignedBO(ME, *Result.Context)) {
          requestFeature(HelperFeatureEnum::device_ext);
          emplaceTransformation(
              ReplaceMemberAssignAsSetMethod(BO, ME, "data_type"));
        } else {
          requestFeature(HelperFeatureEnum::device_ext);
          emplaceTransformation(
              new RenameFieldInMemberExpr(ME, "get_data_type()"));
        }
      }
    } else if (BaseTy == "cudaChannelFormatDesc") {
      static std::map<std::string, std::string> MethodNameMap = {
          {"x", "channel_size"},
          {"y", "channel_size"},
          {"z", "channel_size"},
          {"w", "channel_size"},
          {"f", "channel_data_type"}};
      static const std::unordered_map<std::string, HelperFeatureEnum>
          MethodNameToGetFeatureMap = {{"x", HelperFeatureEnum::device_ext},
                                       {"y", HelperFeatureEnum::device_ext},
                                       {"z", HelperFeatureEnum::device_ext},
                                       {"w", HelperFeatureEnum::device_ext},
                                       {"f", HelperFeatureEnum::device_ext}};
      static const std::unordered_map<std::string, HelperFeatureEnum>
          MethodNameToSetFeatureMap = {{"x", HelperFeatureEnum::device_ext},
                                       {"y", HelperFeatureEnum::device_ext},
                                       {"z", HelperFeatureEnum::device_ext},
                                       {"w", HelperFeatureEnum::device_ext},
                                       {"f", HelperFeatureEnum::device_ext}};
      static std::map<std::string, std::string> ExtraArgMap = {
          {"x", "1"}, {"y", "2"}, {"z", "3"}, {"w", "4"}, {"f", ""}};
      std::string MemberName = ME->getMemberNameInfo().getAsString();
      if (auto BO = getParentAsAssignedBO(ME, *Result.Context)) {
        requestFeature(HelperFeatureEnum::device_ext);
        requestFeature(MethodNameToSetFeatureMap.at(MemberName));
        emplaceTransformation(ReplaceMemberAssignAsSetMethod(
            BO, ME, MethodNameMap[MemberName], "", ExtraArgMap[MemberName]));
      } else {
        requestFeature(HelperFeatureEnum::device_ext);
        requestFeature(MethodNameToGetFeatureMap.at(MemberName));
        emplaceTransformation(new RenameFieldInMemberExpr(
            ME, buildString("get_", MethodNameMap[MemberName], "()")));
      }
    } else {
      replaceTextureMember(ME, *Result.Context, *Result.SourceManager);
    }
  } else if (auto TL = getNodeAsType<TypeLoc>(Result, "texType")) {
    if (isCapturedByLambda(TL))
      return;
    if (DpctGlobalInfo::useSYCLCompat()) {
      report(TL->getBeginLoc(), Diagnostics::UNSUPPORT_SYCLCOMPAT, false,
             DpctGlobalInfo::getUnqualifiedTypeName(TL->getType(),
                                                    *Result.Context));
      return;
    }
    const std::string &ReplType = MapNames::findReplacedName(
        MapNames::TypeNamesMap,
        DpctGlobalInfo::getUnqualifiedTypeName(TL->getType(), *Result.Context));

    requestHelperFeatureForTypeNames(
        DpctGlobalInfo::getUnqualifiedTypeName(TL->getType(), *Result.Context));
    insertHeaderForTypeRule(
        DpctGlobalInfo::getUnqualifiedTypeName(TL->getType(), *Result.Context),
        TL->getBeginLoc());
    if (!ReplType.empty())
      emplaceTransformation(new ReplaceToken(TL->getBeginLoc(), TL->getEndLoc(),
                                             std::string(ReplType)));
  } else if (auto CE = getNodeAsType<CallExpr>(Result, "call")) {
    auto Name = CE->getDirectCallee()->getNameAsString();
    if (DpctGlobalInfo::useSYCLCompat()) {
      report(CE->getBeginLoc(), Diagnostics::UNSUPPORT_SYCLCOMPAT, false, Name);
      return;
    }
    if (Name == "cuTexRefSetFlags") {
      StringRef MethodName;
      auto Value = getTextureFlagsSetterInfo(CE->getArg(1), MethodName);
      requestFeature(HelperFeatureEnum::device_ext);
      std::shared_ptr<CallExprRewriter> Rewriter =
          std::make_shared<AssignableRewriter>(
              CE, std::make_shared<PrinterRewriter<MemberCallPrinter<
                      const Expr *, RenameWithSuffix, false, StringRef>>>(
                      CE, Name, CE->getArg(0), true,
                      RenameWithSuffix("set", MethodName), Value));
      std::optional<std::string> Result = Rewriter->rewrite();
      if (Result.has_value())
        emplaceTransformation(
            new ReplaceStmt(CE, true, std::move(Result).value()));
      return;
    }
    if (Name == "cudaCreateChannelDesc" &&
        !DpctGlobalInfo::useExtBindlessImages()) {
      auto Callee =
          dyn_cast<DeclRefExpr>(CE->getCallee()->IgnoreImplicitAsWritten());
      if (Callee) {
        auto TemArg = Callee->template_arguments();
        if (TemArg.size() != 0) {
          auto ChnType = TemArg[0]
                             .getArgument()
                             .getAsType()
                             .getCanonicalType()
                             .getAsString();
          if (ChnType.back() != '4' &&
              !DpctGlobalInfo::useExtBindlessImages()) {
            report(CE->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_FORMAT,
                   true);
          }
        } else if (getStmtSpelling(CE->getArg(0)) == "0" ||
                   getStmtSpelling(CE->getArg(1)) == "0" ||
                   getStmtSpelling(CE->getArg(2)) == "0" ||
                   getStmtSpelling(CE->getArg(3)) == "0") {
          report(CE->getBeginLoc(), Diagnostics::UNSUPPORTED_IMAGE_FORMAT,
                 true);
        }
      }
    }
    ExprAnalysis A;
    A.analyze(CE);
    emplaceTransformation(A.getReplacement());
    A.applyAllSubExprRepl();
  } else if (auto DRE = getNodeAsType<DeclRefExpr>(Result, "texEnum")) {
    if (auto ECD = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
      std::string EnumName = ECD->getName().str();
      requestHelperFeatureForEnumNames(EnumName);
      if (MapNames::replaceName(MapNames::EnumNamesMap, EnumName)) {
        emplaceTransformation(new ReplaceStmt(DRE, EnumName));
      } else {
        report(DRE->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
               EnumName);
      }
    }
  } else if (auto TL = getNodeAsType<TypeLoc>(Result, "texObj")) {
    if (DpctGlobalInfo::useSYCLCompat()) {
      report(TL->getBeginLoc(), Diagnostics::UNSUPPORT_SYCLCOMPAT, false,
             DpctGlobalInfo::getUnqualifiedTypeName(TL->getType(),
                                                    *Result.Context));
      return;
    }
    if (auto FD = DpctGlobalInfo::getParentFunction(TL)) {
      if ((FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) &&
          !DpctGlobalInfo::useExtBindlessImages()) {
        return;
      }
    }
    ExprAnalysis A;
    A.analyze(*TL);
    emplaceTransformation(A.getReplacement());
    A.applyAllSubExprRepl();
    requestFeature(HelperFeatureEnum::device_ext);
  }
}

void TextureRule::replaceResourceDataExpr(const MemberExpr *ME,
                                          ASTContext &Context) {
  if (!ME)
    return;
  auto TopMember = getParentMemberExpr(ME);
  if (!TopMember)
    return;

  removeExtraMemberAccess(ME);

  auto AssignedBO = getParentAsAssignedBO(TopMember, Context);
  auto FieldName =
      ResourceTypeNames[TopMember->getMemberNameInfo().getAsString()];
  if (FieldName.empty() ||
      !DpctGlobalInfo::useExtBindlessImages() &&
          TopMember->getMemberNameInfo().getAsString() == "mipmap") {
    report(ME->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
           DpctGlobalInfo::getOriginalTypeName(ME->getBase()->getType()) +
               "::" + ME->getMemberDecl()->getName().str());
  }

  if (FieldName == "channel") {
    if (removeExtraMemberAccess(TopMember))
      return;
  }

  if (AssignedBO) {
    requestFeature(HelperFeatureEnum::device_ext);
    emplaceTransformation(
        ReplaceMemberAssignAsSetMethod(AssignedBO, TopMember, FieldName));
  } else {
    auto MemberName = TopMember->getMemberDecl()->getName();
    if (MemberName == "array" || MemberName == "hArray") {
      emplaceTransformation(new InsertBeforeStmt(
          TopMember, "(" + MapNames::getDpctNamespace() + "image_matrix_p)"));
      requestFeature(HelperFeatureEnum::device_ext);
    }
    static const std::unordered_map<std::string, HelperFeatureEnum>
        ResourceTypeNameToGetFeature = {
            {"devPtr", HelperFeatureEnum::device_ext},
            {"desc", HelperFeatureEnum::device_ext},
            {"array", HelperFeatureEnum::device_ext},
            {"width", HelperFeatureEnum::device_ext},
            {"height", HelperFeatureEnum::device_ext},
            {"pitchInBytes", HelperFeatureEnum::device_ext},
            {"sizeInBytes", HelperFeatureEnum::device_ext},
            {"hArray", HelperFeatureEnum::device_ext},
            {"format", HelperFeatureEnum::device_ext},
            {"numChannels", HelperFeatureEnum::device_ext}};
    requestFeature(ResourceTypeNameToGetFeature.at(
        TopMember->getMemberNameInfo().getAsString()));
    emplaceTransformation(new RenameFieldInMemberExpr(
        TopMember, buildString("get_", FieldName, "()")));
    if (TopMember->getMemberNameInfo().getAsString() == "devPtr") {
      emplaceTransformation(new InsertBeforeStmt(ME, buildString("(char *)")));
    }
  }
}

bool isAssignOperator(const Stmt *S) {
  if (auto BO = dyn_cast<BinaryOperator>(S)) {
    return BO->getOpcode() == BO_Assign;
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(S)) {
    return COCE->getOperator() == OO_Equal;
  }
  return false;
}

const Expr *getLhs(const Stmt *S) {
  if (auto BO = dyn_cast<BinaryOperator>(S)) {
    return BO->getLHS();
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (COCE->getNumArgs() > 0) {
      return COCE->getArg(0);
    }
  }
  return nullptr;
}

const Expr *getRhs(const Stmt *S) {
  if (auto BO = dyn_cast<BinaryOperator>(S)) {
    return BO->getRHS();
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (COCE->getNumArgs() > 1) {
      return COCE->getArg(1);
    }
  }
  return nullptr;
}

void TextureRule::SettersMerger::traverseBinaryOperator(const Stmt *S) {
  do {
    if (S != Target && Result.empty())
      return;

    if (!isAssignOperator(S))
      break;

    const Expr *LHS = getLhs(S);
    if (!LHS)
      break;
    if (auto ASE = dyn_cast<ArraySubscriptExpr>(LHS)) {
      LHS = ASE->getBase()->IgnoreImpCasts();
    }
    if (const MemberExpr *ME = dyn_cast<MemberExpr>(LHS)) {
      auto Method = ME->getMemberDecl()->getName();
      if (auto DRE = dyn_cast<DeclRefExpr>(ME->getBase()->IgnoreImpCasts())) {
        if (Result.empty()) {
          D = DRE->getDecl();
          IsArrow = ME->isArrow();
        } else if (DRE->getDecl() != D) {
          break;
        }
        unsigned i = 0;
        for (const auto &Name : MethodNames) {
          if (Method == Name) {
            Result.emplace_back(i, S);
            return;
          }
          ++i;
        }
      }
    }
  } while (false);
  traverse(getLhs(S));
  traverse(getRhs(S));
}

void TextureRule::SettersMerger::traverse(const Stmt *S) {
  if (Stop || !S)
    return;

  switch (S->getStmtClass()) {
  case Stmt::BinaryOperatorClass:
  case Stmt::CXXOperatorCallExprClass:
    traverseBinaryOperator(S);
    break;
  case Stmt::DeclRefExprClass:
    if (static_cast<const DeclRefExpr *>(S)->getDecl() != D) {
      break;
    }
    LLVM_FALLTHROUGH;
  case Stmt::IfStmtClass:
  case Stmt::WhileStmtClass:
  case Stmt::DoStmtClass:
  case Stmt::SwitchStmtClass:
  case Stmt::ForStmtClass:
  case Stmt::CaseStmtClass:
    if (!Result.empty()) {
      Stop = true;
    }
    break;
  default:
    for (auto Child : S->children()) {
      traverse(Child);
    }
  }
}

StringRef getCoordinateNormalizationStr(bool IsNormalized) {
  if (IsNormalized) {
    static std::string NormalizedName =
        MapNames::getClNamespace() +
        "coordinate_normalization_mode::normalized";
    return NormalizedName;
  } else {
    static std::string UnnormalizedName =
        MapNames::getClNamespace() +
        "coordinate_normalization_mode::unnormalized";
    return UnnormalizedName;
  }
}

std::string TextureRule::getTextureFlagsSetterInfo(const Expr *Flags,
                                                   StringRef &SetterName) {
  SetterName = "";
  if (!Flags->isValueDependent()) {
    Expr::EvalResult Result;
    if (Flags->EvaluateAsInt(Result, DpctGlobalInfo::getContext())) {
      auto Val = Result.Val.getInt().getZExtValue();
      if (Val != 1 && Val != 3) {
        report(Flags, Diagnostics::TEX_FLAG_UNSUPPORT, false);
      }
      return getCoordinateNormalizationStr(Val & 0x02).str();
    }
  }
  SetterName = "coordinate_normalization_mode";
  report(Flags, Diagnostics::TEX_FLAG_UNSUPPORT, false);
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  printWithParens(OS, Flags);
  OS << " & 0x02";
  return OS.str();
}

std::string TextureRule::getMemberAssignedValue(const Stmt *AssignStmt,
                                                StringRef MemberName,
                                                StringRef &SetMethodName) {
  SetMethodName = "";
  if (auto RHS = getRhs(AssignStmt)) {
    RHS = RHS->IgnoreImpCasts();
    if (MemberName == "normalizedCoords" || MemberName == "normalized") {
      if (auto IL = dyn_cast<IntegerLiteral>(RHS)) {
        return getCoordinateNormalizationStr(IL->getValue().getZExtValue())
            .str();
      } else if (auto BL = dyn_cast<CXXBoolLiteralExpr>(RHS)) {
        return getCoordinateNormalizationStr(BL->getValue()).str();
      }
      SetMethodName = "coordinate_normalization_mode";
    } else if (MemberName == "flags") {
      return getTextureFlagsSetterInfo(RHS, SetMethodName);
    } else if (MemberName == "channelDesc") {
      SetMethodName = "channel";
    } else if (MemberName == "maxAnisotropy") {
      SetMethodName = "max_anisotropy";
    } else if (MemberName == "mipmapFilterMode") {
      SetMethodName = "mipmap_filtering";
    } else if (MemberName == "minMipmapLevelClamp") {
      SetMethodName = "min_mipmap_level_clamp";
    } else if (MemberName == "maxMipmapLevelClamp") {
      SetMethodName = "max_mipmap_level_clamp";
    }
    return ExprAnalysis::ref(RHS);
  } else {
    return std::string();
  }
}

bool TextureRule::SettersMerger::applyResult() {
  class ResultMapInserter {
    unsigned LastIndex = 0;
    std::vector<const Stmt *> LatestStmts;
    std::vector<const Stmt *> DuplicatedStmts;
    TextureRule *Rule;
    std::map<const Stmt *, bool> &ResultMap;

  public:
    ResultMapInserter(size_t MethodNum, TextureRule *TexRule)
        : LatestStmts(MethodNum, nullptr), Rule(TexRule),
          ResultMap(TexRule->ProcessedBO) {}
    ~ResultMapInserter() {
      for (auto S : DuplicatedStmts) {
        Rule->emplaceTransformation(new ReplaceStmt(S, ""));
        ResultMap[S] = true;
      }
      for (auto S : LatestStmts) {
        ResultMap[S] = false;
      }
    }
    ResultMapInserter(const ResultMapInserter &) = delete;
    ResultMapInserter &operator=(const ResultMapInserter &) = delete;
    void update(size_t Index, const Stmt *S) {
      auto &Latest = LatestStmts[Index];
      if (Latest)
        DuplicatedStmts.push_back(Latest);
      Latest = S;
      LastIndex = Index;
    }
    void success(std::string &Replaced) {
      Rule->emplaceTransformation(
          new ReplaceStmt(LatestStmts[LastIndex], true, std::move(Replaced)));
      DuplicatedStmts.insert(DuplicatedStmts.end(), LatestStmts.begin(),
                             LatestStmts.begin() + LastIndex);
      DuplicatedStmts.insert(DuplicatedStmts.end(),
                             LatestStmts.begin() + LastIndex + 1,
                             LatestStmts.end());
      LatestStmts.clear();
    }
  };

  ResultMapInserter Inserter(MethodNames.size(), Rule);
  std::vector<std::string> ArgsList(MethodNames.size());
  unsigned ActualArgs = 0;
  for (const auto &R : Result) {
    if (ArgsList[R.first].empty())
      ++ActualArgs;
    Inserter.update(R.first, R.second);
    static StringRef Dummy;
    ArgsList[R.first] =
        Rule->getMemberAssignedValue(R.second, MethodNames[R.first], Dummy);
  }
  if (ActualArgs != ArgsList.size()) {
    return false;
  }

  std::string ReplacedText;
  llvm::raw_string_ostream OS(ReplacedText);
  MemberCallPrinter<StringRef, StringRef, false, std::vector<std::string>>
      Printer(D->getName(), IsArrow, "set", std::move(ArgsList));
  Printer.print(OS);

  Inserter.success(OS.str());
  return true;
}

bool TextureRule::SettersMerger::tryMerge(const Stmt *BO) {
  auto Iter = ProcessedBO.find(BO);
  if (Iter != ProcessedBO.end())
    return Iter->second;

  Target = BO;
  auto CS = DpctGlobalInfo::findAncestor<CompoundStmt>(
      BO, [&](const DynTypedNode &Node) -> bool {
        if (Node.get<IfStmt>() || Node.get<WhileStmt>() ||
            Node.get<ForStmt>() || Node.get<DoStmt>() || Node.get<CaseStmt>() ||
            Node.get<SwitchStmt>() || Node.get<CompoundStmt>()) {
          return true;
        }
        return false;
      });

  if (!CS) {
    return ProcessedBO[BO] = false;
  }

  traverse(CS);
  if (applyResult()) {
    requestFeature(HelperFeatureEnum::device_ext);
    return true;
  } else {
    return false;
  }
}

} // namespace dpct
} // namespace clang
