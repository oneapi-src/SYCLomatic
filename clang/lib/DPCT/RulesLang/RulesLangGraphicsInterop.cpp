//===-------------------- RulesLangGraphicsInterop.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RuleInfra/ExprAnalysis.h"

#include "RuleInfra/MigrationStatistics.h"
#include "RulesLang.h"
#include "Utility.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include "clang/Basic/Cuda.h"

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

void GraphicsInteropRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  auto externalResourceMember = [&]() {
    return hasAnyName("cudaExternalMemoryHandleDesc",
                      "cudaExternalMemoryMipmappedArrayDesc",
                      "cudaExternalMemoryBufferDesc");
  };
  MF.addMatcher(
      memberExpr(hasObjectExpression(hasType(
                     type(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                         recordDecl(externalResourceMember()))))))))
          .bind("extResMember"),
      this);

  MF.addMatcher(declRefExpr(to(enumConstantDecl(hasType(enumDecl(
                                hasAnyName("cudaExternalMemoryHandleType"))))))
                    .bind("extResEnum"),
                this);

  auto graphicsInteropAPI = [&]() {
    return hasAnyName(
        "cudaGraphicsD3D11RegisterResource", "cudaGraphicsResourceSetMapFlags",
        "cudaGraphicsMapResources", "cuGraphicsMapResources",
        "cudaGraphicsResourceGetMappedPointer",
        "cuGraphicsResourceGetMappedPointer_v2",
        "cudaGraphicsResourceGetMappedMipmappedArray",
        "cudaGraphicsSubResourceGetMappedArray", "cudaGraphicsUnmapResources",
        "cuGraphicsUnmapResources", "cudaGraphicsUnregisterResource",
        "cuGraphicsUnregisterResource", "cudaImportExternalMemory",
        "cudaExternalMemoryGetMappedMipmappedArray",
        "cudaExternalMemoryGetMappedBuffer", "cudaDestroyExternalMemory");
  };
  MF.addMatcher(
      callExpr(callee(functionDecl(graphicsInteropAPI()))).bind("call"), this);
}

bool GraphicsInteropRule::removeExtraMemberAccess(const MemberExpr *ME) {
  if (auto ParentME = getParentMemberExpr(ME)) {
    emplaceTransformation(new ReplaceToken(ME->getMemberLoc(), ""));
    emplaceTransformation(new ReplaceToken(ParentME->getOperatorLoc(), ""));
    return true;
  }
  return false;
}

void GraphicsInteropRule::runRule(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto ME = getNodeAsType<MemberExpr>(Result, "extResMember")) {
    if (DpctGlobalInfo::useSYCLCompat())
      return;
    auto BaseTy = DpctGlobalInfo::getUnqualifiedTypeName(
        ME->getBase()->getType().getDesugaredType(*Result.Context),
        *Result.Context);
    auto MemberName = ME->getMemberNameInfo().getAsString();
    if (BaseTy == "cudaExternalMemoryHandleDesc") {
      if (MemberName == "handle") {
        removeExtraMemberAccess(ME);
        replaceExtMemHandleDataExpr(getParentMemberExpr(ME), *Result.Context);
      } else {
        auto FieldName =
            ExtMemHandleDescNames[ME->getMemberNameInfo().getAsString()];
        if (FieldName.empty() ||
            (FieldName == "flags" && !DpctGlobalInfo::useExtBindlessImages())) {
          report(ME->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
                 DpctGlobalInfo::getOriginalTypeName(ME->getBase()->getType()) +
                     "::" + ME->getMemberDecl()->getName().str());
          return;
        }

        requestFeature(HelperFeatureEnum::device_ext);
        if (auto BO = getParentAsAssignedBO(ME, *Result.Context)) {
          StringRef ReplacedArg = "";

          if (FieldName == "flags") {
            clang::Expr::EvalResult evalResult;
            if (auto BinOp = dyn_cast<BinaryOperator>(BO);
                BinOp &&
                BinOp->getRHS()->EvaluateAsInt(evalResult, *Result.Context)) {
              ReplacedArg = "0";
            }
          }
          emplaceTransformation(
              ReplaceMemberAssignAsSetMethod(BO, ME, FieldName, ReplacedArg));
        } else {
          emplaceTransformation(new RenameFieldInMemberExpr(
              ME, buildString("get_", FieldName, "()")));
        }
      }
    } else if (BaseTy == "cudaExternalMemoryMipmappedArrayDesc") {
      auto FieldName =
          ExtMemHandleDescNames[ME->getMemberNameInfo().getAsString()];
      if (FieldName.empty() || FieldName == "mem_offset" ||
          (FieldName == "flags" && !DpctGlobalInfo::useExtBindlessImages())) {
        report(ME->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
               DpctGlobalInfo::getOriginalTypeName(ME->getBase()->getType()) +
                   "::" + ME->getMemberDecl()->getName().str());
        return;
      }

      if (FieldName == "flags") {
        // change the FieldName from flags to img_type for img_desc
        FieldName = "image_type";
      }

      requestFeature(HelperFeatureEnum::device_ext);
      if (auto BO = getParentAsAssignedBO(ME, *Result.Context)) {
        StringRef ReplacedArg = "";

        if (FieldName == "image_type") {
          // hex values of cudaArray* flags maped against
          // SYCL image_type equivalents (if available)
          static std::map<int, std::string> image_types = {
              {0x00, MapNames::getExpNamespace() + "image_type::standard"},
              {0x01, MapNames::getExpNamespace() + "image_type::array"},
              {0x02, MapNames::getExpNamespace() + "image_type::mipmap"},
              {0x04, MapNames::getExpNamespace() + "image_type::cubemap"},
              {0x08, "cudaArrayTextureGather"},
              {0x40, "cudaArraySparse"},
              {0x80, "cudaArrayDeferredMapping"}};

          clang::Expr::EvalResult evalResult;
          if (auto BinOp = dyn_cast<BinaryOperator>(BO);
              BinOp &&
              BinOp->getRHS()->EvaluateAsInt(evalResult, *Result.Context)) {
            int flag = evalResult.Val.getInt().getSExtValue();
            ReplacedArg = image_types.at(flag);

            if (flag > 4) {
              report(ME->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
                     ReplacedArg);
            }
          }
        }
        emplaceTransformation(
            ReplaceMemberAssignAsSetMethod(BO, ME, FieldName, ReplacedArg));
      } else {
        emplaceTransformation(new RenameFieldInMemberExpr(
            ME, buildString("get_", FieldName, "()")));
      }
    } else if (BaseTy == "cudaExternalMemoryBufferDesc") {
      auto FieldName =
          ExtMemHandleDescNames[ME->getMemberNameInfo().getAsString()];
      if (FieldName.empty()) {
        report(ME->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
               DpctGlobalInfo::getOriginalTypeName(ME->getBase()->getType()) +
                   "::" + ME->getMemberDecl()->getName().str());
        return;
      }

      requestFeature(HelperFeatureEnum::device_ext);
      if (auto BO = getParentAsAssignedBO(ME, *Result.Context)) {
        emplaceTransformation(
            ReplaceMemberAssignAsSetMethod(BO, ME, FieldName));
      } else {
        emplaceTransformation(new RenameFieldInMemberExpr(
            ME, buildString("get_", FieldName, "()")));
      }
    }
  } else if (auto CE = getNodeAsType<CallExpr>(Result, "call")) {
    auto Name = CE->getDirectCallee()->getNameAsString();
    if (DpctGlobalInfo::useSYCLCompat()) {
      report(CE->getBeginLoc(), Diagnostics::UNSUPPORT_SYCLCOMPAT, false, Name);
      return;
    }

    ExprAnalysis EA(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  } else if (auto DRE = getNodeAsType<DeclRefExpr>(Result, "extResEnum")) {
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
  }
}

void GraphicsInteropRule::replaceExtMemHandleDataExpr(const MemberExpr *ME,
                                                      ASTContext &Context) {
  if (!ME)
    return;

  if (ME->getMemberNameInfo().getAsString() == "win32") {
    removeExtraMemberAccess(ME);

    ME = getParentMemberExpr(ME);
    if (!ME)
      return;
  }

  auto FieldName = ExtMemHandleDescNames[ME->getMemberNameInfo().getAsString()];
  if (FieldName.empty()) {
    report(ME->getBeginLoc(), Diagnostics::API_NOT_MIGRATED, false,
           DpctGlobalInfo::getOriginalTypeName(ME->getBase()->getType()) +
               "::" + ME->getMemberDecl()->getName().str());
  }

  requestFeature(HelperFeatureEnum::device_ext);
  auto AssignedBO = getParentAsAssignedBO(ME, Context);
  if (AssignedBO) {
    emplaceTransformation(
        ReplaceMemberAssignAsSetMethod(AssignedBO, ME, FieldName));
  } else {
    emplaceTransformation(
        new RenameFieldInMemberExpr(ME, buildString("get_", FieldName, "()")));
  }
}

const Expr *GraphicsInteropRule::getParentAsAssignedBO(const Expr *E,
                                                       ASTContext &Context) {
  auto Parents = Context.getParents(*E);
  if (Parents.size() > 0)
    return getAssignedBO(Parents[0].get<Expr>(), Context);
  return nullptr;
}

// Return the binary operator if E is the lhs of an assign expression, otherwise
// nullptr.
const Expr *GraphicsInteropRule::getAssignedBO(const Expr *E,
                                               ASTContext &Context) {
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
    if (BO->getOpcode() == BO_Assign)
      return BO;
  } else if (auto COCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (COCE->getOperator() == OO_Equal) {
      return COCE;
    }
  }
  return nullptr;
}

} // namespace dpct
} // namespace clang
